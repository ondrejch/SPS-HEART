"""The isovector (charge) exchange class (TASK-20261004-r8-isovector-exchange-
class, plan task 47): the Clebsch-Gordan line states, the per-event draw, the
ledger records, the line remnants and root families, the Delta-like
families, and the runner option (off by default, vacuum events unchanged)."""

from __future__ import annotations

import collections
import dataclasses
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.exchange_ledger import (
    ALLOWED_EXCHANGES,
    ISOVECTOR_EXCHANGES,
    ROUND8_EXCHANGES,
    VACUUM,
    Exchange,
    LineState,
    NonSingletTransferError,
    apply_transfer,
    audit_event,
)
from na49_mixing.isovector_exchange import (
    ISOVECTOR_CONFIGURATIONS,
    PROTON_LINE,
    VACUUM_CONFIGURATION,
    ExchangeConfiguration,
    LineCharge,
    draw_exchange_configuration,
    ground_state_pdg,
    line_low_mass_records,
    line_root_family,
    line_share,
    select_line_remnant,
)
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.resonance_decay import clebsch_gordan_squared
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_remnants import KERNEL_BOUNDARY
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
from na49_mixing.topology_single_diffraction import _incoming_lines

NESTED_ROOT = Path(__file__).resolve().parents[1]
ALL = ("baryonic", "strange_baryonic", "strange_mesonic")
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
EXPECTED_CONFIGURATIONS = [
    (Fraction(1, 9), 1, (2, 1.5), (0, 1.5)), (Fraction(2, 9), 1, (2, 1.5), (0, 0.5)),
    (Fraction(4, 27), 0, (1, 1.5), (1, 1.5)), (Fraction(2, 27), 0, (1, 1.5), (1, 0.5)),
    (Fraction(2, 27), 0, (1, 0.5), (1, 1.5)), (Fraction(1, 27), 0, (1, 0.5), (1, 0.5)),
    (Fraction(1, 9), -1, (0, 1.5), (2, 1.5)), (Fraction(2, 9), -1, (0, 0.5), (2, 1.5)),
]
#: the configuration sequence of the first 24 draws at fraction 0.6 on the
#: pinned stream (configuration index, -1 for the vacuum; 2026-10-04)
DRAW_SEQUENCE = [2, -1, 7, 7, -1, 0, 1, 7, -1, 7, -1, 2, 5, -1, -1, 7, 0, 0, -1, 7, 0, -1, 2, -1]


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(dataclasses.replace(R74, isospin_completion=ALL, delta_lines=True), NESTED_ROOT)


def _rngs(i: int, base: int = 9_600_000_000):
    subs = derive_substreams(20261004, 17, base + i)
    return {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}


# ----------------------------------------------------------------------
# The line states
# ----------------------------------------------------------------------
@pytest.mark.parametrize("t", [-1, 0, 1])
def test_the_line_shares_are_the_clebsch_gordan_squares(t):
    for isospin in (0.5, 1.5):
        cg = clebsch_gordan_squared(isospin, 2 * t + 1, 0.5, 1, 1.0, 2 * t)
        assert float(line_share(t, isospin)) == pytest.approx(cg, abs=1e-12)
    assert line_share(t, 0.5) + line_share(t, 1.5) == 1


def test_the_configurations_and_their_probabilities():
    got = [(p, c.i3_transfer, (c.projectile.Q, c.projectile.isospin), (c.target.Q, c.target.isospin))
           for p, c in ISOVECTOR_CONFIGURATIONS]
    assert got == EXPECTED_CONFIGURATIONS
    assert sum(p for p, *_ in got) == 1
    # each I3 transfer carries one third; two thirds exchange charge
    for t in (-1, 0, 1):
        assert sum(p for p, c in ISOVECTOR_CONFIGURATIONS if c.i3_transfer == t) == Fraction(1, 3)
    # the I = 1/2 neutron line is twice as likely as the I = 1/2 proton
    # line (the pion-exchange ratio of pp -> nX to pp -> pX)
    for side in ("projectile", "target"):
        n = sum(p for p, c in ISOVECTOR_CONFIGURATIONS if getattr(c, side) == LineCharge(0, 0.5))
        pr = sum(p for p, c in ISOVECTOR_CONFIGURATIONS if getattr(c, side) == LineCharge(1, 0.5))
        assert n == 2 * pr
    # the mirror symmetry of the table
    mirror = collections.Counter((p, (c.target, c.projectile)) for p, c in ISOVECTOR_CONFIGURATIONS)
    assert mirror == collections.Counter((p, (c.projectile, c.target)) for p, c in ISOVECTOR_CONFIGURATIONS)


def test_the_configuration_records():
    for _p, c in ISOVECTOR_CONFIGURATIONS:
        ex_p, ex_t = c.exchanges()
        assert ex_p in ISOVECTOR_EXCHANGES and ex_t in ISOVECTOR_EXCHANGES
        assert ex_p.transfer == (0, c.i3_transfer, 0, 2 * c.i3_transfer)
        assert ex_t.transfer == tuple(-v for v in ex_p.transfer)
        lp, lt = c.line_states()
        assert (lp.Q, lp.I3x2, lt.Q, lt.I3x2) == (c.projectile.Q, 2 * c.projectile.Q - 1, c.target.Q, 2 * c.target.Q - 1)
        # the audit's inheritance rule holds with the per-line records
        report = audit_event(_incoming_lines(), (ex_p, ex_t), (lp, lt), None, (), allowed=ROUND8_EXCHANGES)
        assert report.violations == ()
    assert VACUUM_CONFIGURATION.exchanges() == (VACUUM, VACUUM)
    assert VACUUM_CONFIGURATION.is_vacuum and VACUUM_CONFIGURATION.projectile == PROTON_LINE


@pytest.mark.parametrize("kwargs", [
    {"name": "pomeron", "i3_transfer": 0, "projectile": PROTON_LINE, "target": PROTON_LINE},
    {"name": "vacuum", "i3_transfer": 1, "projectile": LineCharge(2, 1.5), "target": LineCharge(0, 0.5)},
    {"name": "vacuum", "i3_transfer": 0, "projectile": LineCharge(1, 1.5), "target": PROTON_LINE},
    {"name": "isovector", "i3_transfer": 1, "projectile": LineCharge(2, 1.5), "target": LineCharge(1, 0.5)},
    {"name": "isovector", "i3_transfer": 2, "projectile": LineCharge(2, 1.5), "target": LineCharge(0, 0.5)},
])
def test_a_configuration_validates(kwargs):
    with pytest.raises(ValueError):
        ExchangeConfiguration(**kwargs)


@pytest.mark.parametrize("q, iso", [(2, 0.5), (1, 1.0), (3, 1.5), (True, 0.5)])
def test_a_line_validates(q, iso):
    with pytest.raises(ValueError):
        LineCharge(q, iso)


# ----------------------------------------------------------------------
# The ledger
# ----------------------------------------------------------------------
def test_the_round5_contract_is_unchanged_and_round8_admits_the_isovector_records():
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})
    assert ISOVECTOR_EXCHANGES == frozenset(Exchange("isovector", (0, t, 0, 2 * t)) for t in (-1, 0, 1))
    assert ROUND8_EXCHANGES == ALLOWED_EXCHANGES | ISOVECTOR_EXCHANGES
    proton = LineState(B=1, Q=1, S=0, I3x2=1)
    charge_exchange = Exchange("isovector", (0, -1, 0, -2))
    with pytest.raises(NonSingletTransferError):
        apply_transfer(proton, charge_exchange)
    assert apply_transfer(proton, charge_exchange, allowed=ROUND8_EXCHANGES).qn == (1, 0, 0, -1)
    lines = (LineState(B=1, Q=0, S=0, I3x2=-1), LineState(B=1, Q=2, S=0, I3x2=3))
    pair = (charge_exchange, Exchange("isovector", (0, 1, 0, 2)))
    default = audit_event(_incoming_lines(), pair, lines, None, ())
    assert len(default.violations) == 2 and all(v.startswith("(a)") for v in default.violations)
    assert audit_event(_incoming_lines(), pair, lines, None, (), allowed=ROUND8_EXCHANGES).violations == ()


# ----------------------------------------------------------------------
# The per-event draw
# ----------------------------------------------------------------------
def test_the_draw_budget():
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    assert draw_exchange_configuration(0.0, rng) is VACUUM_CONFIGURATION
    assert rng.bit_generator.state == state  # zero draws at fraction 0
    for fraction in (0.3, 1.0):
        rng_a, rng_b = np.random.default_rng(6), np.random.default_rng(6)
        draw_exchange_configuration(fraction, rng_a)
        rng_b.random(); rng_b.random()
        assert rng_a.bit_generator.state == rng_b.bit_generator.state  # exactly two uniforms
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            draw_exchange_configuration(bad, np.random.default_rng(0))


def test_the_draw_frequencies():
    rng = np.random.default_rng(20261004)
    n = 27_000
    counts = collections.Counter(draw_exchange_configuration(0.6, rng) for _ in range(n))
    expected = {VACUUM_CONFIGURATION: 0.4 * n, **{c: 0.6 * float(p) * n for p, c in ISOVECTOR_CONFIGURATIONS}}
    assert set(counts) == set(expected)
    chi2 = sum((counts[c] - e) ** 2 / e for c, e in expected.items())
    assert chi2 < 26.1  # chi-square 8 dof, p = 0.001


def test_the_draw_sequence_is_pinned():
    rng = np.random.default_rng(np.random.SeedSequence(20261004))
    index = {c: k for k, (_p, c) in enumerate(ISOVECTOR_CONFIGURATIONS)}
    sequence = [index.get(draw_exchange_configuration(0.6, rng), -1) for _ in range(24)]
    assert sequence == DRAW_SEQUENCE


# ----------------------------------------------------------------------
# The line remnants and root families
# ----------------------------------------------------------------------
def test_the_ground_states_and_the_low_mass_sets(kernel):
    catalog, _fams = kernel
    expected = {(1, 0.5): (2212, 7), (0, 0.5): (2112, 7), (2, 1.5): (2224, 3), (1, 1.5): (2214, 3), (0, 1.5): (2114, 3)}
    for (q, iso), (ground, n_low) in expected.items():
        line = LineCharge(q, iso)
        assert ground_state_pdg(line) == ground
        records = line_low_mass_records(catalog, line)
        assert len(records) == n_low and records[0].pdg_id == ground
        for r in records:
            assert (int(r.B), int(r.Q), int(r.S), float(r.isospin)) == (1, q, 0, iso)
            assert r.mass_range[0] < KERNEL_BOUNDARY


def test_select_line_remnant(kernel):
    catalog, _fams = kernel
    neutron_line = LineCharge(0, 0.5)
    rng = np.random.default_rng(1)
    state = np.random.default_rng(1); state.random()
    sel = select_line_remnant({"intact_proton": 1.0}, catalog, rng, None, neutron_line)
    assert (sel.kind, sel.pdg_id, sel.qn) == ("intact_proton", 2112, (1, 0, 0, -1))
    assert sel.mass == catalog.get(2112).pole_mass
    assert rng.bit_generator.state == state.bit_generator.state  # one uniform
    delta = select_line_remnant({"intact_proton": 1.0}, catalog, np.random.default_rng(1), None, LineCharge(2, 1.5))
    assert (delta.pdg_id, delta.mass, delta.qn) == (2224, 1.232, (1, 2, 0, 3))
    seen = {select_line_remnant({"low_mass_baryonic_resonance": 1.0}, catalog, np.random.default_rng(k), None,
                                LineCharge(0, 1.5)).pdg_id for k in range(60)}
    assert seen == {2114, 202114, 122114}
    calls = []
    high = select_line_remnant({"high_mass_baryonic_continuum": 1.0}, catalog, np.random.default_rng(2),
                               lambda lo, hi: calls.append((lo, hi)) or 3.3, LineCharge(0, 1.5))
    assert (high.pdg_id, high.mass, high.qn, len(calls)) == (None, 3.3, (1, 0, 0, -1), 1)
    with pytest.raises(ValueError):
        select_line_remnant({"neutron_charge_exchange": 1.0}, catalog, np.random.default_rng(0), None, neutron_line)


def test_the_line_root_families(kernel):
    _catalog, fams = kernel
    expected = {(2, 1.5): "baryonic_np_q2", (1, 0.5): "baryonic_np", (1, 1.5): "baryonic_np_delta_q1",
                (0, 0.5): "baryonic_np_q0", (0, 1.5): "baryonic_np_delta_q0"}
    for (q, iso), family_id in expected.items():
        assert line_root_family(fams, LineCharge(q, iso)).family_id == family_id
    _c, without = load_kernel(dataclasses.replace(R74, isospin_completion=ALL), NESTED_ROOT)
    with pytest.raises(ValueError):
        line_root_family(without, LineCharge(1, 1.5))


def test_the_delta_line_families(kernel):
    catalog, fams = kernel
    added = [f for f in fams if f.family_id.startswith("baryonic_np_delta")]
    assert [(f.family_id, tuple(f.qn), f.isospin, f.active) for f in added] == [
        ("baryonic_np_delta_q1", (1, 1, 0), 1.5, False), ("baryonic_np_delta_q0", (1, 0, 0), 1.5, False)]
    assert fams[-2:] == added
    for f in added:
        assert (f.continuation_floor, f.continuation_mass_law, f.continuation_barrier_l) == (1.7, "density", 1)
    # as continuation daughters the I = 1/2 families stay first
    from na49_mixing.cascade_decay import _family_by_qn
    assert _family_by_qn(fams, (1, 1, 0)).family_id == "baryonic_np"
    assert _family_by_qn(fams, (1, 0, 0)).family_id == "baryonic_np_q0"


@pytest.mark.parametrize("kwargs", [{"delta_lines": True}, {"delta_lines": 1},
                                    {"delta_lines": True, "isospin_completion": ("strange_mesonic",)}])
def test_the_delta_lines_option_validates(kwargs):
    with pytest.raises(ValueError):
        KernelConfig(**kwargs)


# ----------------------------------------------------------------------
# The runner option
# ----------------------------------------------------------------------
def _params(cls, fraction):
    weights = ({"intact_proton": 0.3, "low_mass_baryonic_resonance": 0.3, "high_mass_baryonic_continuum": 0.4}
               if cls == "nd" else dict(NOMINAL_PARAMS.remnant_weights))
    return dataclasses.replace(NOMINAL_PARAMS, epsilon=0.63, epsilon_nd=0.23, remnant_weights=weights,
                               isovector_fraction=fraction)


def _content(event):
    return [(h.pdg_id, h.four_momentum, h.line) for h in event.final_state]


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd", "nd"])
def test_isovector_events_conserve_and_pass_the_round8_audit(kernel, cls):
    catalog, fams = kernel
    transfers = collections.Counter()
    for i in range(24):
        event = run_topology(cls, _params(cls, 1.0), catalog, fams, **_rngs(i))
        config = event.exchange_configuration
        assert config is not None and not config.is_vacuum
        transfers[config.i3_transfer] += 1
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
        assert tuple(event.lines) == config.line_states(event.lines[0].four_momentum, event.lines[1].four_momentum)
        report = audit_event(_incoming_lines(), config.exchanges(), event.lines, None, (), allowed=ROUND8_EXCHANGES)
        assert report.violations == ()
        # each line's hadrons carry the line's charge (no central charge)
        if cls in ("dd", "nd"):
            for label, line in (("projectile", config.projectile), ("target", config.target)):
                assert sum(int(h.qn[1]) for h in event.final_state if h.line == label) == line.Q
    assert len(transfers) == 3


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd", "nd"])
def test_vacuum_drawn_events_are_the_fraction_zero_events(kernel, cls):
    catalog, fams = kernel
    n_vacuum = 0
    for i in range(16):
        mixed = run_topology(cls, _params(cls, 0.5), catalog, fams, **_rngs(i))
        if mixed.exchange_configuration.is_vacuum:
            n_vacuum += 1
            plain = run_topology(cls, _params(cls, 0.0), catalog, fams, **_rngs(i))
            assert plain.exchange_configuration is None
            assert _content(mixed) == _content(plain) and mixed.lines == plain.lines
    assert n_vacuum > 0


def test_the_default_draws_nothing(kernel):
    catalog, fams = kernel
    for cls in ("sd_projectile", "dd", "nd"):
        rngs = _rngs(3)
        state = rngs["rng_topology_selection"].bit_generator.state
        event = run_topology(cls, _params(cls, 0.0), catalog, fams, **rngs)
        assert rngs["rng_topology_selection"].bit_generator.state == state
        assert event.exchange_configuration is None


def test_a_charge_exchanged_sd_remnant_is_a_neutron(kernel):
    catalog, fams = kernel
    found = 0
    for i in range(40):
        event = run_topology("sd_projectile", dataclasses.replace(_params("sd_projectile", 1.0),
                                                                  remnant_weights={"intact_proton": 1.0}),
                             catalog, fams, **_rngs(i, 9_500_000_000))
        config = event.exchange_configuration
        if config.target == LineCharge(0, 0.5):
            found += 1
            remnant = [h for h in event.final_state if h.line == "remnant"]
            assert [h.pdg_id for h in remnant] == [2112]
    assert found > 0


def test_the_runner_refusals(kernel):
    catalog, fams = kernel
    with pytest.raises(ValueError):
        run_topology("nd_multi", _params("nd", 0.3), catalog, fams, **_rngs(0))
    rngs = _rngs(0)
    rngs["rng_topology_selection"] = None
    with pytest.raises(ValueError):
        run_topology("dd", _params("dd", 0.3), catalog, fams, **rngs)
    # the elastic class keeps the vacuum exchange and draws nothing
    rngs = _rngs(1)
    state = rngs["rng_topology_selection"].bit_generator.state
    run_topology("elastic", _params("dd", 0.3), catalog, fams, **rngs)
    assert rngs["rng_topology_selection"].bit_generator.state == state


# ----------------------------------------------------------------------
# The isovector transfer slope (R76 preparation, 2026-10-04)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_the_isovector_slope_acts_on_isovector_events_only(kernel, cls):
    catalog, fams = kernel
    shared = _params(cls, 0.5)
    steep = dataclasses.replace(shared, b_isovector=12.0)
    n_iv = n_vac = 0
    for i in range(16):
        a = run_topology(cls, shared, catalog, fams, **_rngs(i, 9_400_000_000))
        b = run_topology(cls, steep, catalog, fams, **_rngs(i, 9_400_000_000))
        assert a.exchange_configuration == b.exchange_configuration
        if a.exchange_configuration.is_vacuum:
            n_vac += 1
            assert _content(a) == _content(b)
        else:
            n_iv += 1
            t_a = a.t if cls == "sd_projectile" else a.t1
            t_b = b.t if cls == "sd_projectile" else b.t1
            assert abs(t_b) <= abs(t_a) + 1e-12  # same uniform, steeper slope: smaller |t|
    assert n_iv > 0 and n_vac > 0


def test_the_default_slope_is_shared(kernel):
    catalog, fams = kernel
    for i in range(6):
        a = run_topology("sd_projectile", _params("sd_projectile", 1.0), catalog, fams, **_rngs(i))
        b = run_topology("sd_projectile", dataclasses.replace(_params("sd_projectile", 1.0), b_isovector=None), catalog, fams, **_rngs(i))
        assert _content(a) == _content(b)
    with pytest.raises(ValueError):
        run_topology("sd_projectile", dataclasses.replace(_params("sd_projectile", 1.0), b_isovector=-1.0), catalog, fams, **_rngs(0))


# ----------------------------------------------------------------------
# The I = 3/2 coupling ratio and the non-diffractive slope (2026-10-05)
# ----------------------------------------------------------------------
def test_the_coupling_ratio():
    from na49_mixing.isovector_exchange import isovector_configurations
    assert isovector_configurations(1.0) is ISOVECTOR_CONFIGURATIONS
    for r, ce, q0_half in ((0.25, 0.6, 8 / 9), (0.5, 0.6522, 0.8), (2.0, 0.6575, 0.5)):
        cs = isovector_configurations(r)
        assert sum(p for p, _c in cs) == pytest.approx(1.0, abs=1e-12)
        assert sum(p for p, c in cs if c.i3_transfer != 0) == pytest.approx(ce, abs=1e-4)
        q0 = [(p, c) for p, c in cs if c.i3_transfer == -1]
        assert sum(p for p, c in q0 if c.projectile.isospin == 0.5) / sum(p for p, _c in q0) == pytest.approx(q0_half)
        # the Q = 2 line stays I = 3/2 and the mirror symmetry holds
        assert all(c.projectile.isospin == 1.5 for _p, c in cs if c.projectile.Q == 2)
        assert collections.Counter((round(p, 15), (c.target, c.projectile)) for p, c in cs) == \
            collections.Counter((round(p, 15), (c.projectile, c.target)) for p, c in cs)
    with pytest.raises(ValueError):
        isovector_configurations(0.0)
    rng = np.random.default_rng(11)
    n = 20_000
    counts = collections.Counter(draw_exchange_configuration(1.0, rng, 0.25) for _ in range(n))
    for p, c in isovector_configurations(0.25):
        assert counts[c] / n == pytest.approx(p, abs=4 * (p * (1 - p) / n) ** 0.5 + 1e-3)


def test_the_unit_ratio_is_the_default(kernel):
    catalog, fams = kernel
    for i in range(6):
        a = run_topology("nd", _params("nd", 0.5), catalog, fams, **_rngs(i))
        b = run_topology("nd", dataclasses.replace(_params("nd", 0.5), isovector_delta_ratio=1.0), catalog, fams, **_rngs(i))
        assert _content(a) == _content(b) and a.exchange_configuration == b.exchange_configuration


def test_the_nd_slope_acts_on_the_nd_lines_only(kernel):
    catalog, fams = kernel
    for cls in ("sd_projectile", "dd"):
        for i in range(4):
            a = run_topology(cls, _params(cls, 0.0), catalog, fams, **_rngs(i))
            b = run_topology(cls, dataclasses.replace(_params(cls, 0.0), b_nd=1.5), catalog, fams, **_rngs(i))
            assert _content(a) == _content(b)
    flatter = 0
    for i in range(12):
        a = run_topology("nd", _params("nd", 0.0), catalog, fams, **_rngs(i))
        b = run_topology("nd", dataclasses.replace(_params("nd", 0.0), b_nd=1.5), catalog, fams, **_rngs(i))
        assert tuple(sum(int(h.qn[j]) for h in b.final_state) for j in range(3)) == (2, 2, 0)
        assert abs(b.t1) >= abs(a.t1) - 1e-12  # the same uniform, a flatter slope: larger |t|
        flatter += abs(b.t1) > abs(a.t1)
    assert flatter > 0
    with pytest.raises(ValueError):
        run_topology("nd", dataclasses.replace(_params("nd", 0.0), b_nd=0.0), catalog, fams, **_rngs(0))
