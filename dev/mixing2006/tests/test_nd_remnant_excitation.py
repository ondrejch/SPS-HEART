"""The excited baryonic remnant of the non-diffractive class (Round 5
follow-up card TASK-20260928-r5-gate5-followup-nd-remnant-excitation).

- The frozen split (intact 0.5, low-mass 0.5, excited 0.0) draws no
  excited remnant and consumes exactly two ``spectral_state`` uniforms
  per event (the Round 5 budget), so every Round 5 pin stands.
- Released weights draw excited remnants inside the pre-registered mass
  window; every event conserves ``(B, Q, S) = (2, 2, 0)`` and closes;
  the remnant's final state carries its line label; the kind census
  matches the weights; the charged multiplicity and the low-``x_F``
  proton fraction rise.
- The kind-pair reweighting of a sample generated at equal weights
  reproduces direct generation at the frozen split within errors.
- The multi-neutral class takes the same samplers.
- The sampler returns the window's lower edge without a draw when the
  other remnant closes the window.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import topology_runner as TR
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_elastic import BEAM_MOMENTUM, PROTON_MASS, SQRT_S
from na49_mixing.topology_mass_laws import PI0_MASS, MassLawParams
from na49_mixing.topology_non_diffractive import make_remnant_mass_sampler
from na49_mixing.topology_remnants import (
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    low_mass_records,
)

NESTED = Path(__file__).resolve().parents[1]
KIND_ORDER = (KIND_INTACT_PROTON, KIND_LOW_MASS_BARYONIC_RESONANCE, KIND_HIGH_MASS_BARYONIC_CONTINUUM)
FROZEN = {KIND_INTACT_PROTON: 0.5, KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5, KIND_HIGH_MASS_BARYONIC_CONTINUUM: 0.0}
EQUAL = {k: 1.0 / 3.0 for k in KIND_ORDER}
RELEASED = {KIND_INTACT_PROTON: 0.25, KIND_LOW_MASS_BARYONIC_RESONANCE: 0.25, KIND_HIGH_MASS_BARYONIC_CONTINUUM: 0.5}


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(NESTED / "config" / "v2_singlet" / "resonance_catalog.json")


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(NESTED / "config" / "v2_singlet" / "continuum_resonances.json"))


class _Counting:
    def __init__(self, source):
        self._source = source
        self.calls = 0

    def random(self):
        self.calls += 1
        return float(self._source.random())


def _params(weights):
    return TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, class_fractions=dict(TR.NOMINAL_PARAMS.class_fractions),
                                    remnant_weights=dict(weights), p_n2=0.0)


def _rngs(seed, count_spectral=False):
    labels = ("rng_topology_selection", "rng_spectral_state", "rng_exchange_kinematics", "rng_decay_channel", "rng_decay_kinematics")
    out = {k: np.random.default_rng(7_000_003 * seed + i) for i, k in enumerate(labels)}
    if count_spectral:
        out["rng_spectral_state"] = _Counting(out["rng_spectral_state"])
    return out


def _conserved(event):
    finals = list(event.final_state)
    bqs = tuple(sum(int(h.qn[j]) for h in finals) for j in range(3))
    e_total = math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2) + PROTON_MASS
    sums = [math.fsum(float(h.four_momentum[j]) for h in finals) for j in range(4)]
    closure = max(abs(sums[0] - e_total) / e_total, abs(sums[1]) / BEAM_MOMENTUM, abs(sums[2]) / BEAM_MOMENTUM, abs(sums[3] - BEAM_MOMENTUM) / BEAM_MOMENTUM)
    return bqs == (2, 2, 0) and closure <= 1e-9


def _xf_protons(event):
    beta = BEAM_MOMENTUM / (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2) + PROTON_MASS)
    gamma = (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2) + PROTON_MASS) / SQRT_S
    out = []
    for h in event.final_state:
        if h.pdg_id == 2212:
            e, px, py, pz = (float(x) for x in h.four_momentum)
            out.append(2.0 * gamma * (pz - beta * e) / SQRT_S)
    return out


def test_frozen_split_draws_no_excited_remnant_and_keeps_the_budget(catalog, fams):
    for seed in range(60):
        rngs = _rngs(seed, count_spectral=True)
        event = TR.run_topology("nd", _params(FROZEN), catalog, fams, **rngs)
        assert all(r.kind != KIND_HIGH_MASS_BARYONIC_CONTINUUM and r.pdg_id is not None for r in event.remnants)
        assert rngs["rng_spectral_state"].calls == 2


def test_released_weights_excite_remnants_inside_the_window(catalog, fams):
    n = 400
    census = {k: 0 for k in KIND_ORDER}
    n_ch_released = []
    xf_low = 0
    xf_all = 0
    for seed in range(n):
        event = TR.run_topology("nd", _params(RELEASED), catalog, fams, **_rngs(seed))
        assert _conserved(event)
        other = {0: event.remnants[1].mass, 1: event.remnants[0].mass}
        for i, remnant in enumerate(event.remnants):
            census[remnant.kind] += 1
            if remnant.kind == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
                assert remnant.pdg_id is None
                lo = PROTON_MASS + PI0_MASS
                heaviest = max([PROTON_MASS] + [float(r.pole_mass) for r in low_mass_records(catalog)])
                hi_projectile = SQRT_S - (heaviest + PI0_MASS) - PI0_MASS if i == 0 else SQRT_S - other[1] - PI0_MASS
                assert lo - 1e-9 <= remnant.mass <= hi_projectile + 1e-9
        labels = {h.line for h in event.final_state}
        assert labels <= {"projectile", "target", "central"}
        n_ch_released.append(sum(1 for h in event.final_state if int(h.qn[1]) != 0))
        xfs = _xf_protons(event)
        xf_all += len(xfs)
        xf_low += sum(1 for x in xfs if abs(x) < 0.4)
    total = 2 * n
    for kind, weight in RELEASED.items():
        expected = weight * total
        assert abs(census[kind] - expected) < 5.0 * math.sqrt(expected * (1 - weight)) + 1.0, (kind, census[kind], expected)
    n_ch_frozen = [sum(1 for h in TR.run_topology("nd", _params(FROZEN), catalog, fams, **_rngs(seed)).final_state if int(h.qn[1]) != 0) for seed in range(n)]
    assert np.mean(n_ch_released) > np.mean(n_ch_frozen) + 0.3
    assert xf_low / xf_all > 0.10


def test_kind_pair_reweighting_reproduces_direct_generation(catalog, fams):
    n = 800
    gen = []
    for seed in range(n):
        event = TR.run_topology("nd", _params(EQUAL), catalog, fams, **_rngs(seed))
        kinds = tuple(KIND_ORDER.index(r.kind) for r in event.remnants)
        gen.append((kinds, sum(1 for h in event.final_state if int(h.qn[1]) != 0)))
    target = [0.5, 0.5, 0.0]
    w = np.asarray([target[k1] * target[k2] / ((1 / 3) * (1 / 3)) for (k1, k2), _ in gen])
    n_ch = np.asarray([v for _, v in gen], dtype=float)
    wn = w / np.sum(w)
    reweighted = float(np.sum(wn * n_ch))
    se_re = float(math.sqrt(np.sum(wn ** 2 * (n_ch - reweighted) ** 2)))
    direct = np.asarray([sum(1 for h in TR.run_topology("nd", _params(FROZEN), catalog, fams, **_rngs(10_000 + seed)).final_state if int(h.qn[1]) != 0) for seed in range(n)], dtype=float)
    pull = (reweighted - float(np.mean(direct))) / math.sqrt(se_re ** 2 + np.var(direct, ddof=1) / n)
    assert abs(pull) < 4.0, (reweighted, float(np.mean(direct)), pull)


def test_multi_neutral_takes_the_released_weights(catalog, fams):
    params = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, class_fractions=dict(TR.NOMINAL_PARAMS.class_fractions),
                                      remnant_weights=dict(RELEASED), p_n2=0.5)
    excited = 0
    for seed in range(60):
        event = TR.run_topology("nd_multi", params, catalog, fams, **_rngs(seed), p_n2=0.5)
        assert _conserved(event)
        excited += sum(1 for r in event.remnants if r.kind == KIND_HIGH_MASS_BARYONIC_CONTINUUM)
    assert excited > 0


def test_sampler_window_and_closure():
    params = MassLawParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd)
    counter = _Counting(np.random.default_rng(3))
    sampler = make_remnant_mass_sampler(params, counter, PROTON_MASS)
    values = [sampler(PROTON_MASS, 5.0) for _ in range(50)]
    assert all(PROTON_MASS + PI0_MASS - 1e-9 <= v <= SQRT_S - PROTON_MASS - PI0_MASS + 1e-9 for v in values)
    assert max(values) > 5.0  # the family's 5 GeV support edge does not cap the excitation
    assert counter.calls > 0
    closed = make_remnant_mass_sampler(params, _Counting(np.random.default_rng(4)), SQRT_S - PROTON_MASS - PI0_MASS - 0.05)
    counter2 = _Counting(np.random.default_rng(5))
    closed = make_remnant_mass_sampler(params, counter2, SQRT_S - PROTON_MASS - PI0_MASS - 0.05)
    assert closed(PROTON_MASS, 5.0) == pytest.approx(PROTON_MASS + PI0_MASS)
    assert counter2.calls == 0


def test_event_records_the_excitation_windows(catalog, fams):
    """The record carries the two remnant windows (2026-09-29): the projectile
    window reserves the heaviest un-excited target kind plus a pion, the
    target window the drawn projectile mass, and every excited mass lies
    inside its window."""
    from na49_mixing.topology_non_diffractive import remnant_excitation_window, _heaviest_intact_or_low_mass
    heaviest = _heaviest_intact_or_low_mass(catalog)
    seen = 0
    for seed in range(200):
        event = TR.run_topology("nd", _params(RELEASED), catalog, fams, **_rngs(seed))
        assert event.remnant_windows[0] == remnant_excitation_window(heaviest + PI0_MASS)
        assert event.remnant_windows[1] == remnant_excitation_window(event.remnants[0].mass)
        for remnant, (lo, hi) in zip(event.remnants, event.remnant_windows):
            if remnant.kind == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
                seen += 1
                assert lo - 1e-9 <= remnant.mass <= hi + 1e-9
    assert seen > 50


def test_decoupled_nd_exponent_changes_only_the_nd_class(catalog, fams):
    """``epsilon_nd`` (2026-09-29, Block A) drives the non-diffractive mass
    laws; ``None`` reproduces the shared exponent bit for bit, and the
    diffractive classes ignore it."""
    shared = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, remnant_weights=dict(RELEASED))
    same = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, remnant_weights=dict(RELEASED), epsilon_nd=None)
    flat = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, remnant_weights=dict(RELEASED), epsilon_nd=-1.0)
    differs = 0
    for seed in range(40):
        a = TR.run_topology("nd", shared, catalog, fams, **_rngs(seed))
        b = TR.run_topology("nd", same, catalog, fams, **_rngs(seed))
        c = TR.run_topology("nd", flat, catalog, fams, **_rngs(seed))
        assert a.central_mass == b.central_mass and [r.mass for r in a.remnants] == [r.mass for r in b.remnants]
        if c.central_mass != a.central_mass or [r.mass for r in c.remnants] != [r.mass for r in a.remnants]:
            differs += 1
        # the diffractive classes keep the frozen split (their un-excited line
        # never draws the continuum kind) and ignore epsilon_nd
        shared_d = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, remnant_weights=dict(FROZEN))
        flat_d = TR.TopologyNominalParams(epsilon=0.0, b_sd=TR.NOMINAL_PARAMS.b_sd, remnant_weights=dict(FROZEN), epsilon_nd=-1.0)
        for cls in ("sd_projectile", "dd"):
            x = TR.run_topology(cls, shared_d, catalog, fams, **_rngs(seed))
            y = TR.run_topology(cls, flat_d, catalog, fams, **_rngs(seed))
            assert [h.four_momentum for h in x.final_state] == [h.four_momentum for h in y.final_state]
    assert differs > 20
