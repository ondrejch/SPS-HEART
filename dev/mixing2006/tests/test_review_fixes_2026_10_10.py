"""The fixes of the physics implementation review of 2026-10-10 (grok-4.7):

- M1: the leading-baryon split draws a root over the split channels on its
  branch, and the flavor ledger records that pool
  (``cascade_decay.draw_decay_with_pool``), so the reweight of a
  split-branch vertex is the ratio of the split-channel probabilities;
- M3: ``quantum_numbers`` resolves the antibaryon of a listed baryon where
  the bundled database lists only the baryon, so every antibaryon of the
  charge-conjugate towers decays as the conjugate of its baryon; the
  Delta(1700) multiplet carries B = 1;
- L1: the per-event slope normalization takes a negative slope as the
  scalar normalization does.

The draws, the towers-off decay tables and the positive-slope
normalization are unchanged."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.cascade_termination import run_cascade
from na49_mixing.conjugate_towers import charge_conjugate
from na49_mixing.flavor_weights import FlavorParams, ledger_log_weight, vertex_ledger_record
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.quantum_numbers import _NONSTRANGE_BARYONS, _STRANGE_BARYONS, quantum_numbers
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.topology_mass_laws import _log_t_window_norm_slopes, log_t_window_norm_vec

NESTED_ROOT = Path(__file__).resolve().parents[1]
R89 = json.loads((NESTED_ROOT / "config" / "runs" / "r89.json").read_text())["kernel"]
GEN = FlavorParams(gamma_s_had=0.284, r_bbar=1.75, assoc_norm=2.3)
MOVED = FlavorParams(gamma_s_had=0.8, r_bbar=1.75, assoc_norm=0.6)


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(KernelConfig(**R89), NESTED_ROOT)


def _root(mass, pz=1.5):
    e = math.sqrt(mass * mass + pz * pz)
    return C.CascadeState(momentum=(e, 0.0, 0.0, pz), mass=mass, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                          axis=(0.0, 0.0, 1.0), alignment_label=C.ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                          exchange_ancestry=("review_2026_10_10",), depth=0)


def _continuum(state, cat, fams, flavor):
    return [c for c in C.enumerate_channels(state, cat, fams, flavor=flavor) if c.kind == C.KIND_CONTINUUM]


def _same(a, b):
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


@pytest.mark.parametrize("mass", [3.5, 5.0])
def test_the_split_branch_draws_over_the_split_channels(kernel, mass):
    cat, fams = kernel
    state = _root(mass)
    family = C._family_by_id(fams, "baryonic_np")
    fraction = float(family.leading_split_fraction)
    continuum = _continuum(state, cat, fams, GEN)
    _family, subset = C._leading_split(state, continuum, cat, fams)
    assert 0 < len(subset) < len(continuum)
    branches = set()
    for seed in range(120):
        u = float(np.random.default_rng(seed).random())
        ch, daughters, pool = C.draw_decay_with_pool(state, continuum, cat, fams, np.random.default_rng(seed),
                                                     np.random.default_rng(seed + 500))
        ch0, daughters0 = C.draw_decay(state, continuum, cat, fams, np.random.default_rng(seed),
                                       np.random.default_rng(seed + 500))
        assert ch is ch0 and [d.momentum for d in daughters] == [d.momentum for d in daughters0]
        on_split = u < fraction
        branches.add(on_split)
        assert _same(pool, subset if on_split else continuum)
    assert branches == {True, False}


@pytest.mark.parametrize("mass", [3.5, 5.0])
def test_the_ledger_of_a_split_vertex_is_the_split_channel_ratio(kernel, mass):
    """At a split-branch vertex the ledger weight from the generation flavor
    to another point is the ratio of the drawn channel's probability within
    the split channels; on the other branch it is the ratio within every
    continuum channel.  The full-list record of a split-branch vertex
    misses the first ratio."""
    cat, fams = kernel
    state = _root(mass)
    pools = {}
    for name, flavor in (("gen", GEN), ("moved", MOVED)):
        continuum = _continuum(state, cat, fams, flavor)
        pools[name] = (continuum, C._leading_split(state, continuum, cat, fams)[1])

    def prob(name, which, daughters):
        pool = pools[name][which]
        return next(c.weight for c in pool if c.daughters == daughters) / math.fsum(c.weight for c in pool)

    seen, missed = set(), 0.0
    for seed in range(200):
        ch, _d, pool = C.draw_decay_with_pool(state, pools["gen"][0], cat, fams, np.random.default_rng(seed),
                                              np.random.default_rng(seed + 900))
        which = 1 if _same(pool, pools["gen"][1]) else 0
        seen.add(which)
        record = vertex_ledger_record(state.qn, pool, ch, cat, fams, GEN)
        got = ledger_log_weight([record], GEN, MOVED)
        want = math.log(prob("moved", which, ch.daughters) / prob("gen", which, ch.daughters))
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)
        if which == 1:
            full = vertex_ledger_record(state.qn, pools["gen"][0], ch, cat, fams, GEN)
            missed = max(missed, abs(ledger_log_weight([full], GEN, MOVED) - want))
    assert seen == {0, 1}
    assert missed > 1e-3


def test_the_cascade_ledger_records_the_drawn_pool(kernel):
    cat, fams = kernel
    state = _root(5.0)
    fraction = float(C._family_by_id(fams, "baryonic_np").leading_split_fraction)
    continuum = _continuum(state, cat, fams, GEN)
    subset = C._leading_split(state, continuum, cat, fams)[1]
    groups = {name: vertex_ledger_record(state.qn, pool, pool[0], cat, fams, GEN)[1]
              for name, pool in (("split", subset), ("every", continuum))}
    seen = set()
    for seed in range(40):
        ledger = []
        run_cascade(state, cat, fams, np.random.default_rng(seed), np.random.default_rng(seed + 300), flavor=GEN,
                    ledger=ledger)
        name = "split" if float(np.random.default_rng(seed).random()) < fraction else "every"
        seen.add(name)
        assert ledger[0][1] == pytest.approx(groups[name], rel=1e-12)
    assert seen == {"split", "every"}


def test_every_antibaryon_of_a_listed_baryon_resolves(kernel):
    cat, _fams = kernel
    listed = _NONSTRANGE_BARYONS | _STRANGE_BARYONS
    baryons = [int(r.pdg_id) for r in cat.records() if int(r.B) == 1]
    assert set(baryons) <= listed
    for pdg in baryons:
        assert quantum_numbers(-pdg) == -quantum_numbers(pdg)
        record = cat.get(-pdg)
        assert quantum_numbers(-pdg).as_tuple() == (int(record.B), int(record.Q), int(record.S))
    assert quantum_numbers(-12212).as_tuple() == (-1, -1, 0) and quantum_numbers(-3222).as_tuple() == (-1, -1, 1)
    for pdg, q in ((122224, 2), (122214, 1), (122114, 0), (121114, -1)):
        assert quantum_numbers(pdg).as_tuple() == (1, q, 0)
    # an id outside the listed baryons stays unknown
    for unknown in (-111, -9000221):
        with pytest.raises(KeyError):
            quantum_numbers(unknown)


@pytest.mark.parametrize("pdg", [-102112, -102212, -122112, -122212, -212114, -212214, -102134])
def test_an_antibaryon_decays_as_the_conjugate_of_its_baryon(kernel, pdg):
    cat, _fams = kernel
    decayer = ResonanceDecayer(cat, 0, 0, 0)
    record, baryon = cat.get(pdg), cat.get(-pdg)
    for factor in (1.0, 1.15):
        mass = min(float(baryon.pole_mass) * factor, float(baryon.mass_range[1]))
        objects, total = decayer._open_objects(record, mass)
        base, base_total = decayer._open_objects(baryon, mass)
        assert total == base_total
        assert ({tuple(sorted(p)): w for _c, p, w in objects}
                == {tuple(sorted(charge_conjugate(d, cat) for d in p)): w for _c, p, w in base})
    if pdg == -102112:
        # aN(1535)0 keeps both charge states of its aN*(1440) pi mode
        pairs = {tuple(sorted(p)) for _c, p, _w in decayer._open_objects(record, float(record.pole_mass))[0]}
        assert {(-12212, 211), (-12112, 111)} <= pairs


def test_the_per_event_slope_normalization_takes_a_negative_slope():
    lo = np.array([-2.0, -0.5, -3.0, -1.0, -4.0])
    hi = np.array([-0.1, -0.5, 0.0, -0.2, -3.9])
    for b in (-7.0, -0.3, 0.0, 2.5, 12.0):
        assert _log_t_window_norm_slopes(np.full(5, b), lo, hi) == pytest.approx(log_t_window_norm_vec(b, lo, hi),
                                                                                    rel=1e-13, abs=1e-15)
    slopes = np.array([-7.0, 3.0, -0.3, 12.0, -1.0])
    mixed = _log_t_window_norm_slopes(slopes, lo, hi)
    for i, b in enumerate(slopes):
        assert mixed[i] == pytest.approx(log_t_window_norm_vec(b, lo[i:i + 1], hi[i:i + 1])[0], rel=1e-13)
    # the closed form: log of the integral of exp(b t) over the window
    b, a, c = -2.0, -1.5, -0.25
    assert _log_t_window_norm_slopes(np.array([b]), np.array([a]), np.array([c]))[0] == pytest.approx(
        math.log((math.exp(b * c) - math.exp(b * a)) / b), rel=1e-13)
