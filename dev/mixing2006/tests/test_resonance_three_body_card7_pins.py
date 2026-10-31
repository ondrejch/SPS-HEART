"""Card-7 three-body decay vertex: committed-behavior coverage pins.

Task ``TASK-20260913-r2-selected-three-body-decays`` (plan section 22 task
15; the three-body clause of plan section 10.2, line 692). Sibling of
``tests/test_resonance_three_body.py`` (the solver's card file, which pins
the kernel on hand-constructed parents and a hermetic omega stand-in): this
module pins the committed card-7 behavior of ``na49_mixing/resonance_decay.py``
(nested 449e9cc) on the committed catalog
``config/v2_singlet/resonance_catalog.json``.

Solver scope decision pinned here: the catalog is byte-unchanged by card 7,
and the approved set (omega(782), phi(1020), a1(1260), rho(770),
Delta(1232)) intersects the committed catalog in the omega(782) -> pi+ pi-
pi0 channel only. The ``is_three_body`` flag is the card-2 import flag and
the card-2 renormalized branching fractions are unchanged: every approved
parent's represented modes sum to exactly 1.0. phi(1020), a1(1260),
rho(770), and Delta(1232) carry no three-body channels: adding the approved
PDG branching fractions would over-normalize them. The committed catalog
also carries the card-2 import flags on eta(548), eta-prime(958),
rho3(1690), and Lambda(1520); the same kernel routes those through the same
three-body vertex, so the kernel-level pins below run on the committed
omega(782) channel and do not claim catalog-wide uniqueness.

Pins:

- ``test_omega_three_body_routing``: catalog-level approved-set intersection
  (omega(782) only; its flagged channel is (211, -211, 111) with the
  unchanged card-2 branching fraction 0.9831367794555275 and the two-body
  channel 0.016863220544472535; all approved parents sum to exactly 1.0)
  plus kernel-level routing: the flagged channel routes through the
  three-body vertex (3 pi daughters, all stable) and the unflagged channel
  through the two-body vertex (2 pi daughters), both occurring in a
  deterministic pole-mass sweep.
- ``test_omega_four_momentum_closure``: 1,000 committed omega(782) three-body
  vertices at the pole mass (at rest) conserve the parent four-momentum to
  1e-9 per component.
- ``test_omega_daughters_on_shell``: the same sweep; every daughter of every
  vertex is on shell to 1e-9 at its ``pdg_database`` mass.
- ``test_omega_bqs_conserved``: the same sweep; (B, Q, S) of the pi+ pi- pi0
  charge states sums to (0, 0, 0) at every vertex (Q = +1, -1, 0; B = S = 0
  for every daughter).
- ``test_omega_mbc_range``: the same sweep; the leading-pair pi+ pi-
  invariant mass M_bc lies in [m_pi+ + m_pi-, M - m_pi0] at every vertex.
- ``test_omega_three_body_threshold_hard_zero``: the width law returns the
  exact hard zero (0.0) at 0.41 GeV (below the committed 3pi threshold
  0.4141164) and a positive width at 0.42 GeV (above it); a hand-constructed
  three-body parent below the threshold raises ``NoOpenChannelError`` naming
  the parent, its pdg id, the exact mass, and the no-clamp semantics; the
  committed omega(782) below the three-body threshold decays only through
  the open two-body pi+ pi- channel (the three-body channel is ineligible,
  no error is raised).
- ``test_omega_determinism``: identical ``(master_seed, replica,
  event_index)`` produces identical vertices; a different event index
  produces a different vertex at the pole mass.
- ``test_omega_draw_budget``: the three-body vertex consumes exactly 4
  uniforms (one per ``random()``) on the ``decay_kinematics`` substream and
  1 on ``decay_channel``; the two-body vertex consumes 2 and 1. Pinned
  white-box through the decayer's substream generators via the
  ``bit_generator.state`` draw-budget precedent of
  ``tests/test_resonance_widths.py``.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from na49_mixing.hadron_catalog import HadronCatalog, HadronRecord, DecayChannel
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import quantum_numbers
from na49_mixing.resonance_decay import (
    DecayVertex,
    NoOpenChannelError,
    ResonanceDecayer,
)
from na49_mixing.resonance_widths import channel_width
from na49_mixing.rng_substreams import derive_substreams

CATALOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "v2_singlet"
    / "resonance_catalog.json"
)

MASTER_SEED = 20260913
OMEGA_PDG = 223
OMEGA_MASS = 0.78265
OMEGA_3PI_DAUGHTERS = (211, -211, 111)
OMEGA_PIPI_DAUGHTERS = (211, -211)
# The radiative revision of 2026-10-02 (catalog_radiative_revision.py):
# the legacy three-mode table normalized over pi+ pi- pi0, pi0 gamma and
# pi+ pi- (the Round 2 values 0.98314 / 0.01686 had folded pi0 gamma out).
OMEGA_3PI_BR = 0.892 / (0.892 + 0.0833 + 0.0153)
OMEGA_PIPI_BR = 0.0153 / (0.892 + 0.0833 + 0.0153)
OMEGA_PI0GAMMA_DAUGHTERS = (111, 22)
OMEGA_PI0GAMMA_BR = 0.0833 / (0.892 + 0.0833 + 0.0153)
OMEGA_TWO_BODY_DAUGHTERS = {OMEGA_PIPI_DAUGHTERS, OMEGA_PI0GAMMA_DAUGHTERS}
PHI_REVISED_SUM = math.fsum((0.501, 0.336, 0.049, 0.049, 0.049, 0.01306, 0.00133))
OMEGA_BQS = (0, 0, 0)

# The other approved-set parents: no three-body channels in the committed
# catalog; represented modes sum to exactly 1.0.
PHI_PDG = 333
A1_IDS = (20113, 20213, -20213)
RHO_IDS = (113, 213, -213)
DELTA_IDS = (1114, 2114, 2214, 2224, -1114, -2114, -2214, -2224)
NON_OMEGA_APPROVED = (PHI_PDG, *A1_IDS, *RHO_IDS, *DELTA_IDS)

MPI = float(get_mass(211))
MPI0 = float(get_mass(111))
OMEGA_3PI_THRESHOLD = math.fsum((MPI, MPI, MPI0))

# 0.41 GeV is below the committed 3pi threshold (0.4141164); 0.42 GeV is
# above it.
BELOW_THRESHOLD_MASS = 0.41
ABOVE_THRESHOLD_MASS = 0.42

N_CLOSURE = 1_000
CLOSURE_TOL = 1e-9
BR_EXACT_TOL = 1e-15
N_BELOW_THRESHOLD_EVENTS = 50
SWEEP_MAX_EVENTS = 10_000
# Deterministic channel tags at MASTER_SEED / replica 0 / pole mass
# (catalog-dependent; verified against the committed catalog).
THREE_BODY_EVENT = 1
TWO_BODY_EVENT = 31
RADIATIVE_EVENT = 0
HAND_BUILT_3PI_PDG = 87000324


@lru_cache(maxsize=1)
def _committed_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@lru_cache(maxsize=1)
def _omega_pole_sweep() -> tuple[
    HadronCatalog, tuple[DecayVertex, ...], tuple[DecayVertex, ...]
]:
    """Every vertex of the committed omega(782) at its pole mass at rest,
    event_index = i, one decayer per decay (the card-6 sweep convention),
    until N_CLOSURE three-body vertices are collected (both channel types
    occur in the window). Deterministic for the fixed seed and the
    committed catalog."""
    catalog = _committed_catalog()
    three_body: list[DecayVertex] = []
    two_body: list[DecayVertex] = []
    for event_index in range(SWEEP_MAX_EVENTS):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, event_index)
        produced = decayer.decay_resonance(OMEGA_PDG, (OMEGA_MASS, 0.0, 0.0, 0.0))
        assert len(produced) == 1  # the pions are stable: a single vertex
        vertex = produced[0]
        if vertex.channel.is_three_body:
            three_body.append(vertex)
        else:
            two_body.append(vertex)
        if len(three_body) >= N_CLOSURE:
            break
    assert len(three_body) == N_CLOSURE
    return catalog, tuple(three_body), tuple(two_body)


def _omega_record(catalog: HadronCatalog):
    return catalog.get(OMEGA_PDG)


def _omega_3pi_channel(record):
    flagged = [ch for ch in record.decay_channels if ch.is_three_body]
    assert len(flagged) == 1
    return flagged[0]


def test_omega_three_body_routing() -> None:
    catalog = _committed_catalog()
    # Catalog level: the approved set intersects the committed catalog in
    # the omega(782) three-body channel only.
    record_ids = {int(record.pdg_id) for record in catalog.records()}
    for pid in (OMEGA_PDG, *NON_OMEGA_APPROVED):
        assert pid in record_ids
    for record in catalog.records():
        if int(record.pdg_id) == OMEGA_PDG:
            continue
        if int(record.pdg_id) in NON_OMEGA_APPROVED:
            for ch in record.decay_channels:
                assert not ch.is_three_body, (
                    f"approved parent {record.pdg_id} {record.name} "
                    f"carries a three-body channel "
                    f"{tuple(ch.daughters)}"
                )
    # Omega(782): exactly one three-body channel, the card-2 flagged
    # (211, -211, 111), with the unchanged card-2 branching fractions.
    omega = _omega_record(catalog)
    channel = _omega_3pi_channel(omega)
    assert tuple(int(d) for d in channel.daughters) == OMEGA_3PI_DAUGHTERS
    assert channel.branching_fraction == OMEGA_3PI_BR
    assert channel.provenance is not None
    assert channel.provenance.transcribed_value == OMEGA_3PI_BR
    two_body = {tuple(int(d) for d in ch.daughters): ch for ch in omega.decay_channels if not ch.is_three_body}
    assert set(two_body) == OMEGA_TWO_BODY_DAUGHTERS
    assert two_body[OMEGA_PIPI_DAUGHTERS].branching_fraction == OMEGA_PIPI_BR
    assert two_body[OMEGA_PI0GAMMA_DAUGHTERS].branching_fraction == OMEGA_PI0GAMMA_BR
    assert two_body[OMEGA_PI0GAMMA_DAUGHTERS].is_radiative
    br_sum = math.fsum(ch.branching_fraction for ch in omega.decay_channels)
    assert abs(br_sum - 1.0) <= BR_EXACT_TOL
    for pid in NON_OMEGA_APPROVED:
        record = catalog.get(pid)
        br = math.fsum(
            ch.branching_fraction for ch in record.decay_channels
        )
        # phi(1020) carries the PDG Live table of the radiative revision
        # (2026-10-02), whose unlisted modes leave a documented 0.161%
        expected = PHI_REVISED_SUM if pid == 333 else 1.0
        assert abs(br - expected) <= BR_EXACT_TOL, (
            f"approved parent {pid} {record.name} sums to {br}"
        )
    # Kernel level: the committed omega routes the flagged channel through
    # the three-body vertex and the unflagged channel through the
    # two-body vertex; both occur in the deterministic sweep.
    _catalog, three_body, two_body_vertices = _omega_pole_sweep()
    assert len(three_body) == N_CLOSURE
    assert two_body_vertices  # both channel types occur
    for vertex in three_body:
        assert vertex.channel is channel
        assert len(vertex.daughters) == 3
        assert vertex.charge_combination == OMEGA_3PI_DAUGHTERS
        assert all(not d.is_resonance for d in vertex.daughters)
    for vertex in two_body_vertices:
        assert not vertex.channel.is_three_body
        assert len(vertex.daughters) == 2
        assert vertex.charge_combination in OMEGA_TWO_BODY_DAUGHTERS
        assert all(not d.is_resonance for d in vertex.daughters)
    assert {v.charge_combination for v in two_body_vertices} == OMEGA_TWO_BODY_DAUGHTERS


def test_omega_four_momentum_closure() -> None:
    _catalog, three_body, _two_body = _omega_pole_sweep()
    worst = 0.0
    for vertex in three_body:
        e, px, py, pz = vertex.parent_four_momentum
        total = (
            e - sum(d.four_momentum[0] for d in vertex.daughters),
            px - sum(d.four_momentum[1] for d in vertex.daughters),
            py - sum(d.four_momentum[2] for d in vertex.daughters),
            pz - sum(d.four_momentum[3] for d in vertex.daughters),
        )
        for residual in total:
            assert abs(residual) <= CLOSURE_TOL
        worst = max(worst, max(abs(r) for r in total))
    assert worst <= CLOSURE_TOL


def test_omega_daughters_on_shell() -> None:
    _catalog, three_body, _two_body = _omega_pole_sweep()
    worst = 0.0
    for vertex in three_body:
        for daughter in vertex.daughters:
            mass = float(get_mass(daughter.pdg_id))
            e, px, py, pz = daughter.four_momentum
            deviation = abs(e * e - px * px - py * py - pz * pz - mass * mass)
            assert deviation <= CLOSURE_TOL
            worst = max(worst, deviation)
    assert worst <= CLOSURE_TOL


def test_omega_bqs_conserved() -> None:
    _catalog, three_body, _two_body = _omega_pole_sweep()
    for vertex in three_body:
        total = [0, 0, 0]
        charges = []
        for daughter in vertex.daughters:
            qn = quantum_numbers(int(daughter.pdg_id))
            assert int(qn.B) == 0
            assert int(qn.S) == 0
            charges.append(int(qn.Q))
            for k, value in enumerate((qn.B, qn.Q, qn.S)):
                total[k] += int(value)
        assert tuple(total) == OMEGA_BQS
        assert tuple(charges) == (1, -1, 0)


def _m_bc_squared(vertex: DecayVertex) -> float:
    """Invariant mass squared of the leading pair (the two charged pions,
    the card-3 leading-pair convention)."""
    e = vertex.daughters[0].four_momentum[0] + vertex.daughters[1].four_momentum[0]
    px = (
        vertex.daughters[0].four_momentum[1]
        + vertex.daughters[1].four_momentum[1]
    )
    py = (
        vertex.daughters[0].four_momentum[2]
        + vertex.daughters[1].four_momentum[2]
    )
    pz = (
        vertex.daughters[0].four_momentum[3]
        + vertex.daughters[1].four_momentum[3]
    )
    return e * e - px * px - py * py - pz * pz


def test_omega_mbc_range() -> None:
    _catalog, three_body, _two_body = _omega_pole_sweep()
    lo = MPI + MPI
    hi = OMEGA_MASS - MPI0
    assert lo < hi
    for vertex in three_body:
        m_bc = math.sqrt(_m_bc_squared(vertex))
        assert lo <= m_bc <= hi


def _hand_built_three_body_parent() -> HadronCatalog:
    """A hand-constructed three-body-only parent whose mass window opens
    exactly at the 3pi threshold, mirroring the solver card file's fixture
    on a fresh placeholder pdg id (87000324)."""
    record = HadronRecord(
        pdg_id=HAND_BUILT_3PI_PDG,
        name="P3pi7",
        B=0,
        Q=0,
        S=0,
        isospin=0.0,
        isospin3=0,
        spin=1.0,
        parity="-",
        pole_mass=0.9,
        nominal_width=0.15,
        mass_range=(OMEGA_3PI_THRESHOLD, 1.0),
        decay_channels=(
            DecayChannel(
                parent_pdg=HAND_BUILT_3PI_PDG,
                daughters=OMEGA_3PI_DAUGHTERS,
                branching_fraction=1.0,
                orbital_angular_momentum=0,
                is_three_body=True,
                provenance=None,
            ),
        ),
        stable_for_decay=False,
        provenance=None,
    )
    return HadronCatalog([record])


def test_omega_three_body_threshold_hard_zero() -> None:
    catalog = _committed_catalog()
    omega = _omega_record(catalog)
    channel = _omega_3pi_channel(omega)
    masses = {pdg: float(get_mass(pdg)) for pdg in OMEGA_3PI_DAUGHTERS}
    threshold = math.fsum(masses.values())
    assert threshold == OMEGA_3PI_THRESHOLD
    assert BELOW_THRESHOLD_MASS < threshold < ABOVE_THRESHOLD_MASS
    # Width law: the exact hard zero below the threshold, positive above.
    assert channel_width(BELOW_THRESHOLD_MASS, channel, omega, masses) == 0.0
    assert channel_width(ABOVE_THRESHOLD_MASS, channel, omega, masses) > 0.0
    # Hand-constructed three-body parent below the threshold:
    # NoOpenChannelError, no clamp.
    decayer = ResonanceDecayer(
        _hand_built_three_body_parent(), MASTER_SEED, 0, 0
    )
    try:
        decayer.decay_resonance(
            HAND_BUILT_3PI_PDG, (BELOW_THRESHOLD_MASS, 0.0, 0.0, 0.0)
        )
    except NoOpenChannelError as exc:
        message = str(exc)
        assert "P3pi7" in message
        assert str(HAND_BUILT_3PI_PDG) in message
        assert repr(BELOW_THRESHOLD_MASS) in message
        assert "no clamp" in message
    else:
        raise AssertionError(
            "expected NoOpenChannelError below the three-body threshold"
        )
    # Committed omega(782) below the three-body threshold: the two-body pi+
    # pi- and pi0 gamma channels are open and the three-body channel is
    # ineligible, so the decay proceeds without error and no three-body
    # vertex is ever produced.
    for i in range(N_BELOW_THRESHOLD_EVENTS):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 7_000_000 + i)
        vertices = decayer.decay_resonance(
            OMEGA_PDG, (BELOW_THRESHOLD_MASS, 0.0, 0.0, 0.0)
        )
        assert vertices
        for vertex in vertices:
            assert not vertex.channel.is_three_body
            assert vertex.charge_combination in OMEGA_TWO_BODY_DAUGHTERS


def test_omega_determinism() -> None:
    catalog = _committed_catalog()

    def run(event_index: int) -> tuple[DecayVertex, ...]:
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, event_index)
        return tuple(
            decayer.decay_resonance(OMEGA_PDG, (OMEGA_MASS, 0.0, 0.0, 0.0))
        )

    first = run(4242)
    second = run(4242)
    assert len(first) == len(second) == 1
    va, vb = first[0], second[0]
    assert va.channel is vb.channel
    assert va.charge_combination == vb.charge_combination
    assert va.parent_four_momentum == vb.parent_four_momentum
    for da, db in zip(va.daughters, vb.daughters):
        assert da.pdg_id == db.pdg_id
        assert da.is_resonance == db.is_resonance
        for x, y in zip(da.four_momentum, db.four_momentum):
            assert abs(x - y) <= 1e-15
    other = run(4243)
    oc = other[0]
    differs = oc.charge_combination != va.charge_combination
    for da, db in zip(va.daughters, oc.daughters):
        if any(abs(x - y) > 1e-9 for x, y in zip(da.four_momentum, db.four_momentum)):
            differs = True
    assert differs


def test_omega_draw_budget() -> None:
    """The three-body vertex consumes exactly 4 uniforms (one per
    ``random()``) on the ``decay_kinematics`` substream and 1 on
    ``decay_channel``; the two-body vertex consumes 2 and 1. Pinned via the
    ``bit_generator.state`` draw-budget precedent (the state of the decayer's
    substream generators equals the state of a fresh generator advanced by
    exactly n draws)."""
    catalog = _committed_catalog()
    for event_index, expected_kin_draws in (
        (THREE_BODY_EVENT, 4),
        (TWO_BODY_EVENT, 2),
        (RADIATIVE_EVENT, 2),
    ):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, event_index)
        vertices = decayer.decay_resonance(
            OMEGA_PDG, (OMEGA_MASS, 0.0, 0.0, 0.0)
        )
        assert len(vertices) == 1
        assert vertices[0].channel.is_three_body == (expected_kin_draws == 4)
        subs = derive_substreams(MASTER_SEED, 0, event_index)
        reference = np.random.default_rng(
            subs.stream_for("decay_kinematics")
        )
        for _ in range(expected_kin_draws):
            reference.random()
        assert (
            decayer._rng_kinematics.bit_generator.state
            == reference.bit_generator.state
        )
        reference_channel = np.random.default_rng(
            subs.stream_for("decay_channel")
        )
        reference_channel.random()
        assert (
            decayer._rng_channel.bit_generator.state
            == reference_channel.bit_generator.state
        )
