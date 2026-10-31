"""Recursive two-body decay kernel tests (Round 2, card 6).

Task ``TASK-20260913-r2-recursive-two-body-decays`` (plan section 22
task 14; plan section 8.4 step 4).  Nine deterministic tests carry the
card's clauses on a hand-constructed two-generation catalog (a
Delta-like ``I = 3/2`` state to N pi, an N*-like ``I = 1/2`` state to
the Delta and to N pi; hand masses; no event generation in the sense of
the staged pipeline), and three statistical tests verify the card-5
charge weights, the spin-averaged angular kernel, and the daughter
momentum moment through the full kernel at 100,000 decays:

- ``test_bqs_conserved_per_vertex``: a deterministic sweep of at least
  1,000 vertices (fixed seed) conserves ``(B, Q, S)`` at every vertex
  (plan section 8.4, the conservation clause).
- ``test_four_momentum_closure``: every vertex's daughter four-momenta
  sum to the parent four-momentum within ``1e-9`` GeV (same sweep).
- ``test_mass_shell``: every daughter is on shell within ``1e-9``
  (the exact-kinematics clause; a resonance daughter's mass is the
  realized mass of its own vertex, a stable daughter's mass its pole
  mass).
- ``test_termination_on_stable_set``: the final particles are exactly
  members of the catalog stable set, no resonance is left queued
  (every resonance daughter appears as a later vertex), and the
  ``decay_source`` entry point returns the final stable particles with
  their feed-down lineage.
- ``test_determinism``: two decayers with the same ``(master_seed,
  replica, event)`` produce identical vertex lists (four-momenta to
  ``1e-15``, the same channel and combination sequence); a different
  ``event_index`` produces a different sequence.
- ``test_cycle_guard``: a hand-constructed cyclic ladder (masses
  descending along the chain, so the strict daughter-mass invariant
  alone would not catch it, with the back edge that closes the cycle in
  the channel graph) raises ``DecayCycleError`` naming the chain
  through the depth guard of 64 vertices per chain; a chain deeper
  than 64 raises the same guard.  The daughter-mass invariant is
  enforced by the mass-budget window and asserted at draw time; through
  the public surface it is unreachable by construction (the budget
  window enforces it), so the depth guard is the operative mechanism
  the test exercises.
- ``test_inaccessible_channel``: a parent whose channels are all above
  its realized mass raises ``NoOpenChannelError`` naming the parent and
  mass.
- ``test_width_normalization``: a parent whose open weight is zero at
  the realized mass (all widths zero: the channel closed at the pole)
  and a parent with a non-finite open width raise
  ``NoOpenChannelError`` naming the width normalization.
- ``test_closed_charge_state_between_thresholds``: Delta+ at a
  realized mass strictly between the two N-pi thresholds (``p pi0``
  open, ``n pi+`` kinematically closed) decays 200 of 200 events to
  ``(2212, 111)`` at the exact open-channel kinematics; the closed
  ``(2112, 211)`` class-map combination carries weight zero and is
  never sampled (plan section 8.4 samples one allowed open channel; a
  closed combination is ineligible, never sampled-then-failed).
- ``test_charge_weight_sampling`` (statistical, 100,000 decays of the
  hand-constructed Delta+ at a fixed mass): the empirical ``p pi0 : n
  pi+`` split matches ``2/3 : 1/3`` to 3-sigma (the card-5 weights
  through the full kernel).
- ``test_isotropic_kernel`` (statistical, same sweep): the empirical
  ``cos(theta)`` histogram is uniform in 10 bins to 3-sigma (the
  spin-averaged kernel, round 2 index item 9).
- ``test_daughter_momentum_moment`` (statistical, same sweep): the mean
  daughter momentum in the parent rest frame matches the analytic
  charge-weighted two-body ``q(M)`` to 3-sigma.

The hand-built ladder registers no vendored-table entries: its
placeholder PDG ids are opaque to the card-5 quantum-number resolution
(the class maps come back empty and the kernel falls back to the
channel branching shares, the documented card-6 fallback), and all
 ``(B, Q, S)`` bookkeeping runs on the record fields.

Hermetic: hand-constructed records only; no staged-pipeline event
generation, no external tables.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from na49_mixing.hadron_catalog import (  # noqa: E402
    DecayChannel,
    HadronCatalog,
    HadronRecord,
)
from na49_mixing.pdg_database import get_mass  # noqa: E402
from na49_mixing.quantum_numbers import quantum_numbers  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    MAX_CHAIN_DEPTH,
    DecayCycleError,
    DecayVertex,
    KinematicsError,
    NoOpenChannelError,
    ResonanceDecayer,
    SourceRecord,
    apply_isospin_derivation,
    class_pair_of,
    derive_charge_weights,
)
from na49_mixing.resonance_widths import (  # noqa: E402
    channel_width,
    decay_momentum,
)

MASTER_SEED = 20260913

# Vendored-table daughter masses (the card-2 catalog values).
MP = 0.93827231
MN = 0.93956541
MPI = 0.1395699
MPI0 = 0.1349766
M_DELTA = 1.232

DELTA_PLUS = 2214
DELTA_ZERO = 2114
NSTAR_PLUS = 12212
GAMMA = 22

N_STATISTICAL = 100_000
CLOSURE_TOL = 1e-9


# ----------------------------------------------------------------------
# hand-built records
# ----------------------------------------------------------------------
def _channel(parent_pdg, daughters, branching, L=0):
    return DecayChannel(
        parent_pdg=int(parent_pdg),
        daughters=tuple(int(d) for d in daughters),
        branching_fraction=float(branching),
        orbital_angular_momentum=int(L),
        is_three_body=False,
        provenance=None,
    )


_DAUGHTER_POLE = {
    2212: MP, -2212: MP,
    2112: MN, -2112: MN,
    211: MPI, -211: MPI, 111: MPI0,
    DELTA_PLUS: M_DELTA, DELTA_ZERO: M_DELTA,
    NSTAR_PLUS: 1.44,
}


def _record(pdg_id, name, B, Q, S, isospin, isospin3, channels,
            pole_mass, nominal_width, m_up):
    threshold = max(
        math.fsum(_DAUGHTER_POLE[int(d)] for d in ch.daughters)
        for ch in channels
    )
    return HadronRecord(
        pdg_id=int(pdg_id),
        name=name,
        B=int(B), Q=int(Q), S=int(S),
        isospin=float(isospin),
        isospin3=int(isospin3),
        spin=0.5,
        parity="+",
        pole_mass=float(pole_mass),
        nominal_width=float(nominal_width),
        mass_range=(float(threshold), float(m_up)),
        decay_channels=tuple(channels),
        stable_for_decay=False,
        provenance=None,
    )


@lru_cache(maxsize=1)
def _two_generation_catalog() -> HadronCatalog:
    """The hand-built two-generation catalog after the card-5
    completion: Delta+ and Delta0 to N pi (one representative channel
    each), N*+ to the Delta and to N pi (one representative channel per
    class)."""
    delta_plus = _record(
        DELTA_PLUS, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(DELTA_PLUS, (2212, 111), 1.0, L=1)],
        pole_mass=M_DELTA, nominal_width=0.117, m_up=M_DELTA + 5 * 0.117,
    )
    delta_zero = _record(
        DELTA_ZERO, "Delta0", 1, 0, 0, 1.5, -1,
        [_channel(DELTA_ZERO, (2112, 111), 1.0, L=1)],
        pole_mass=M_DELTA, nominal_width=0.117, m_up=M_DELTA + 5 * 0.117,
    )
    nstar = _record(
        NSTAR_PLUS, "N*+", 1, 1, 0, 0.5, 1,
        [_channel(NSTAR_PLUS, (DELTA_PLUS, 111), 0.6, L=0),
         _channel(NSTAR_PLUS, (2212, 111), 0.4, L=0)],
        pole_mass=1.44, nominal_width=0.30, m_up=1.44 + 5 * 0.30,
    )
    return apply_isospin_derivation(
        HadronCatalog([delta_plus, delta_zero, nstar])
    )


def _pole_mass_of(catalog: HadronCatalog, pdg_id: int) -> float:
    """Pole mass of a daughter: the catalog record's value, else the
    vendored PDG table mass."""
    try:
        return float(catalog.get(pdg_id).pole_mass)
    except KeyError:
        return float(get_mass(pdg_id))


def _bqs_of(catalog: HadronCatalog, pdg_id: int) -> tuple[int, int, int]:
    """``(B, Q, S)`` of a daughter, resolved independently of the
    kernel: the record fields for records, the exact PDG quantum
    numbers for stable states."""
    try:
        record = catalog.get(pdg_id)
    except KeyError:
        qn = quantum_numbers(pdg_id)
        return (int(qn.B), int(qn.Q), int(qn.S))
    return (int(record.B), int(record.Q), int(record.S))


# ----------------------------------------------------------------------
# the deterministic sweep (>= 1,000 vertices, fixed seeds)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _sweep() -> tuple[HadronCatalog, list[DecayVertex]]:
    """A deterministic sweep of N*+ decays at the fixed rest mass
    1.6 GeV until at least 1,000 vertices are collected (fixed master
    seed, one decayer per decay with ``event_index = i``)."""
    catalog = _two_generation_catalog()
    vertices: list[DecayVertex] = []
    for i in range(2000):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, i)
        vertices.extend(
            decayer.decay_resonance(NSTAR_PLUS, (1.6, 0.0, 0.0, 0.0))
        )
        if len(vertices) >= 1000:
            break
    assert len(vertices) >= 1000
    return catalog, vertices


def _realized_masses(
    catalog: HadronCatalog, vertices: list[DecayVertex]
) -> dict[tuple[int, tuple[float, float, float, float]], float]:
    """The realized mass of every resonance daughter, keyed by
    ``(pdg_id, four_momentum)``: its own vertex's parent mass."""
    return {
        (vertex.parent_pdg, vertex.parent_four_momentum): vertex.parent_mass
        for vertex in vertices
    }


def test_bqs_conserved_per_vertex():
    catalog, vertices = _sweep()
    assert len(vertices) >= 1000
    for vertex in vertices:
        parent_qn = _bqs_of(catalog, vertex.parent_pdg)
        total = [0, 0, 0]
        for daughter in vertex.daughters:
            qn = _bqs_of(catalog, daughter.pdg_id)
            for k in range(3):
                total[k] += qn[k]
        assert tuple(total) == parent_qn, (
            f"vertex {vertex.parent_pdg} -> {vertex.charge_combination}: "
            f"daughters {tuple(total)} != parent {parent_qn}"
        )


def test_four_momentum_closure():
    _catalog, vertices = _sweep()
    worst = 0.0
    for vertex in vertices:
        e, px, py, pz = vertex.parent_four_momentum
        total = [e, px, py, pz]
        for daughter in vertex.daughters:
            for k in range(4):
                total[k] -= daughter.four_momentum[k]
        worst = max(worst, max(abs(x) for x in total))
        assert max(abs(x) for x in total) <= CLOSURE_TOL
    assert worst <= CLOSURE_TOL


def test_mass_shell():
    catalog, vertices = _sweep()
    realized = _realized_masses(catalog, vertices)
    worst = 0.0
    for vertex in vertices:
        for daughter in vertex.daughters:
            if daughter.is_resonance:
                mass = realized[(daughter.pdg_id, daughter.four_momentum)]
            else:
                mass = _pole_mass_of(catalog, daughter.pdg_id)
            e, px, py, pz = daughter.four_momentum
            m2 = e * e - px * px - py * py - pz * pz
            deviation = abs(m2 - mass * mass)
            worst = max(worst, deviation)
            assert deviation <= CLOSURE_TOL, (
                f"daughter pdg {daughter.pdg_id}: p^2 - m^2 = {deviation!r}"
            )
    assert worst <= CLOSURE_TOL


def test_termination_on_stable_set():
    catalog, vertices = _sweep()
    stable = catalog.stable_states()
    n_final = 0
    for index, vertex in enumerate(vertices):
        for daughter in vertex.daughters:
            if daughter.is_resonance:
                # the resonance daughter is queued: its own vertex
                # appears later in queue order (parent first)
                later = [
                    other for other in vertices[index + 1:]
                    if other.parent_pdg == daughter.pdg_id
                    and other.parent_four_momentum
                    == daughter.four_momentum
                ]
                assert later, (
                    f"resonance daughter {daughter.pdg_id} of vertex "
                    f"{index} has no vertex"
                )
            else:
                assert daughter.pdg_id in stable
                n_final += 1
    assert n_final >= len(vertices)

    # the decay_source entry point: the final stable particles carry
    # their feed-down lineage
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 999_999)
    source_vertices, finals = decayer.decay_source(SourceRecord(
        pdg_id=NSTAR_PLUS,
        four_momentum=(1.6, 0.0, 0.0, 0.0),
        B=1, Q=1, S=0,
        mass=1.6,
        source_id=7,
    ))
    expected = [
        daughter
        for vertex in source_vertices
        for daughter in vertex.daughters
        if not daughter.is_resonance
    ]
    assert len(finals) == len(expected)
    for particle, daughter in zip(finals, expected):
        assert particle.pdg_id == daughter.pdg_id
        assert particle.pdg_id in stable
        assert particle.source_id == 7
        assert particle.parent_id is not None
        assert 0 <= particle.parent_id < len(source_vertices)
        assert particle.p4.as_dict() == dict(
            zip(("e", "px", "py", "pz"), daughter.four_momentum)
        )
        assert particle.channel_family == "resonance_decay"


def test_determinism():
    catalog = _two_generation_catalog()
    first = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4242
    ).decay_resonance(NSTAR_PLUS, (1.6, 0.0, 0.0, 0.0))
    second = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4242
    ).decay_resonance(NSTAR_PLUS, (1.6, 0.0, 0.0, 0.0))
    assert len(first) == len(second) >= 1
    for va, vb in zip(first, second):
        assert va.parent_pdg == vb.parent_pdg
        assert va.channel == vb.channel
        assert va.charge_combination == vb.charge_combination
        assert va.parent_four_momentum == vb.parent_four_momentum
        for da, db in zip(va.daughters, vb.daughters):
            assert da.pdg_id == db.pdg_id
            assert da.is_resonance == db.is_resonance
            for x, y in zip(da.four_momentum, db.four_momentum):
                assert abs(x - y) <= 1e-15

    other = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4243
    ).decay_resonance(NSTAR_PLUS, (1.6, 0.0, 0.0, 0.0))
    differs = len(other) != len(first)
    for va, vb in zip(first, other):
        if va.charge_combination != vb.charge_combination:
            differs = True
            break
        for da, db in zip(va.daughters, vb.daughters):
            if any(abs(x - y) > 1e-9
                   for x, y in zip(da.four_momentum, db.four_momentum)):
                differs = True
                break
        if differs:
            break
    assert differs, "a different event_index produced the same sequence"


# ----------------------------------------------------------------------
# the cycle / inaccessibility / width-normalization protections
# ----------------------------------------------------------------------
def _ladder_catalog(n_states: int, cyclic: bool) -> tuple[list[int], HadronCatalog]:
    """A hand-built descending ladder of placeholder states
    ``L_i -> L_{i+1} + gamma`` with poles descending by 1e-5 per level
    (so the strict daughter-mass invariant passes at every vertex) and
    a spectrally alive last level.  With ``cyclic`` the last state's
    channel table carries the back edge to the first state: the channel
    graph contains the cycle A -> ... -> A; the back edge is
    deliberately kinematically inconsistent (a descending ladder can
    never decay back up), which is why the depth guard, not the
    kinematics, is the protection that fires."""
    ids = [87000001 + i for i in range(n_states)]
    gamma = HadronRecord(
        pdg_id=GAMMA, name="gamma", B=0, Q=0, S=0,
        isospin=0.0, isospin3=0, spin=1.0, parity="+",
        pole_mass=0.0, nominal_width=0.0,
        mass_range=(0.0, 0.0), decay_channels=(),
        stable_for_decay=True, provenance=None,
    )
    records = [gamma]
    for i, pid in enumerate(ids):
        pole = 2.0 - 1e-5 * i
        channels: list[DecayChannel] = []
        if i < n_states - 1:
            channels.append(_channel(pid, (ids[i + 1], GAMMA), 1.0, L=0))
            low = 2.0 - 1e-5 * (i + 1)
        else:
            if cyclic:
                channels.append(
                    _channel(pid, (ids[0], GAMMA), 0.5, L=0))
            channels.append(_channel(pid, (111, GAMMA), 0.5, L=0))
            low = pole - 1e-5
        records.append(HadronRecord(
            pdg_id=pid, name=f"L{len(records):03d}",
            B=0, Q=0, S=0, isospin=0.0, isospin3=0,
            spin=0.0, parity="+",
            pole_mass=pole, nominal_width=1e-7,
            mass_range=(low, pole + 0.005),
            decay_channels=tuple(channels),
            stable_for_decay=False, provenance=None,
        ))
    return ids, HadronCatalog(records)


def test_cycle_guard():
    # (a) the cyclic catalog: the chain walks the descending ladder and
    # the depth guard fires with the chain named
    ids, catalog = _ladder_catalog(MAX_CHAIN_DEPTH + 1, cyclic=True)
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    with pytest.raises(DecayCycleError) as excinfo:
        decayer.decay_resonance(ids[0], (2.0, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert str(MAX_CHAIN_DEPTH) in message
    assert str(ids[0]) in message and str(ids[-1]) in message

    # (b) a chain deeper than 64: the same depth guard
    ids_deep, catalog_deep = _ladder_catalog(MAX_CHAIN_DEPTH + 2, cyclic=False)
    decayer_deep = ResonanceDecayer(catalog_deep, MASTER_SEED, 0, 0)
    with pytest.raises(DecayCycleError) as excinfo_deep:
        decayer_deep.decay_resonance(
            ids_deep[0], (2.0, 0.0, 0.0, 0.0))
    assert str(MAX_CHAIN_DEPTH) in str(excinfo_deep.value)
    assert str(ids_deep[0]) in str(excinfo_deep.value)


def test_inaccessible_channel():
    parent = _record(
        87000200, "Pacc", 1, 1, 0, 0.5, 1,
        [_channel(87000200, (2212, 211), 1.0, L=0)],
        pole_mass=2.0, nominal_width=0.2, m_up=2.9,
    )
    catalog = HadronCatalog([parent])
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    with pytest.raises(NoOpenChannelError) as excinfo:
        decayer.decay_resonance(87000200, (1.0, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert "87000200" in message and "Pacc" in message
    assert "1.0" in message


def test_width_normalization():
    # all widths zero at the realized mass: the channel is closed at
    # the pole (pole mass below the channel threshold), so the width
    # law gives zero at every mass although the realized mass is above
    # the threshold
    parent = _record(
        87000201, "Pw", 1, 1, 0, 0.5, 1,
        [_channel(87000201, (2212, 211), 1.0, L=0)],
        pole_mass=1.0, nominal_width=0.2, m_up=2.5,
    )
    catalog = HadronCatalog([parent])
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    with pytest.raises(NoOpenChannelError) as excinfo:
        decayer.decay_resonance(87000201, (2.0, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert "87000201" in message and "Pw" in message
    assert "width normalization" in message

    # a non-finite open width: the open weight is not finite
    parent_inf = HadronRecord(
        pdg_id=87000202, name="Pinf", B=1, Q=1, S=0,
        isospin=0.5, isospin3=1, spin=0.5, parity="+",
        pole_mass=2.0, nominal_width=float("inf"),
        mass_range=(1.07784221, 2.9),
        decay_channels=(_channel(87000202, (2212, 211), 1.0, L=0),),
        stable_for_decay=False, provenance=None,
    )
    catalog_inf = HadronCatalog([parent_inf])
    decayer_inf = ResonanceDecayer(catalog_inf, MASTER_SEED, 0, 0)
    with pytest.raises(NoOpenChannelError) as excinfo_inf:
        decayer_inf.decay_resonance(87000202, (2.5, 0.0, 0.0, 0.0))
    assert "width normalization" in str(excinfo_inf.value)


# ----------------------------------------------------------------------
# the statistical tests (100,000 decays of the Delta+ at a fixed mass)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _delta_sample():
    """100,000 decays of the hand-constructed Delta+ at its pole mass
    (at rest); per decay the root charge combination and the first
    daughter's lab (= parent rest frame, the parent is at rest)
    momentum components."""
    catalog = _two_generation_catalog()
    records = []
    for i in range(N_STATISTICAL):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, i)
        vertices = decayer.decay_resonance(
            DELTA_PLUS, (M_DELTA, 0.0, 0.0, 0.0))
        assert len(vertices) == 1
        vertex = vertices[0]
        px, py, pz = (
            vertex.daughters[0].four_momentum[k] for k in (1, 2, 3)
        )
        records.append((vertex.charge_combination, px, py, pz))
    return tuple(records)


def test_charge_weight_sampling():
    sample = _delta_sample()
    n = len(sample)
    assert n == N_STATISTICAL
    n_p_pi0 = sum(1 for combo, _, _, _ in sample
                  if combo == (2212, 111))
    share = n_p_pi0 / n
    expected = 2.0 / 3.0
    sigma = math.sqrt(expected * (1.0 - expected) / n)
    assert abs(share - expected) <= 3.0 * sigma, (
        f"p pi0 share {share!r} vs {expected!r} "
        f"(3 sigma = {3.0 * sigma!r})"
    )


def test_isotropic_kernel():
    sample = _delta_sample()
    n = len(sample)
    bins = [0] * 10
    for _, px, py, pz in sample:
        p = math.sqrt(px * px + py * py + pz * pz)
        cos_theta = pz / p
        index = min(int((cos_theta + 1.0) / 2.0 * 10), 9)
        bins[index] += 1
    expected = n / 10.0
    sigma = math.sqrt(expected * (1.0 - 0.1))
    for index, count in enumerate(bins):
        assert abs(count - expected) <= 3.0 * sigma, (
            f"cos(theta) bin {index}: {count} vs {expected!r} "
            f"(3 sigma = {3.0 * sigma!r})"
        )


def test_daughter_momentum_moment():
    catalog = _two_generation_catalog()
    sample = _delta_sample()
    n = len(sample)
    mean = math.fsum(
        math.sqrt(px * px + py * py + pz * pz)
        for _, px, py, pz in sample
    ) / n
    # the analytic charge-weighted two-body momentum at the fixed mass:
    # the realized combination shares are exactly the card-5 weights
    # (the class width factor cancels within the class)
    shares = {
        (2212, 111): 2.0 / 3.0,
        (2112, 211): 1.0 / 3.0,
    }
    q_values = {
        (2212, 111): decay_momentum(M_DELTA, MP, MPI0),
        (2112, 211): decay_momentum(M_DELTA, MN, MPI),
    }
    expected = math.fsum(
        shares[combo] * q_values[combo] for combo in shares
    )
    # the moment's variance: the between-combination term (the momentum
    # is deterministic within a combination at the fixed mass)
    variance = math.fsum(
        shares[combo] * (q_values[combo] - expected) ** 2
        for combo in shares
    )
    sigma = math.sqrt(variance / n)
    assert abs(mean - expected) <= 3.0 * sigma, (
        f"mean daughter momentum {mean!r} vs {expected!r} "
        f"(3 sigma = {3.0 * sigma!r})"
    )


# ----------------------------------------------------------------------
# the kinematically closed charge state (the open-object filter)
# ----------------------------------------------------------------------
def test_closed_charge_state_between_thresholds():
    """Delta+ at a realized mass strictly between the two N-pi
    thresholds: ``p pi0`` is open and ``n pi+`` is kinematically closed
    (zero channel width), while the card-5 class map still carries both
    charge combinations.  The kernel samples one allowed open channel,
    so every decay yields ``(2212, 111)`` at the exact open-channel
    kinematics and the closed ``(2112, 211)`` combination is never
    sampled (no ``KinematicsError``, no fallback DecayChannel binding a
    closed combination to another channel)."""
    catalog = _two_generation_catalog()
    record = catalog.get(DELTA_PLUS)
    mass = 1.07619211  # strictly between MP + MPI0 and MN + MPI
    assert MP + MPI0 < mass < MN + MPI

    p_pi0 = next(channel for channel in record.decay_channels
                 if tuple(int(d) for d in channel.daughters) == (2212, 111))
    n_pi = next(channel for channel in record.decay_channels
                if tuple(int(d) for d in channel.daughters) == (2112, 211))

    # the width law at the realized mass: the hard zero closes the
    # n pi+ combination while p pi0 stays open
    masses = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter not in masses:
                masses[daughter] = float(get_mass(daughter))
    assert channel_width(mass, p_pi0, record, masses) > 0.0
    assert channel_width(mass, n_pi, record, masses) == 0.0

    # the card-5 class map is mass-independent and still carries both
    # combinations (the fixture really exercises the closed-combination
    # window); the kernel must not emit the closed combination
    class_map = derive_charge_weights(record, class_pair_of(p_pi0), catalog)
    assert set(class_map) == {(2212, 111), (2112, 211)}
    assert class_map[(2212, 111)] > 0.0 and class_map[(2112, 211)] > 0.0

    # the open objects at the realized mass: exactly one entry, bound to
    # its own channel, carrying the whole class width
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    objects, open_total = decayer._open_objects(record, mass)
    assert len(objects) == 1
    channel, pair, weight = objects[0]
    assert (channel, pair) == (p_pi0, (2212, 111))
    expected = channel_width(mass, p_pi0, record, masses)
    assert abs(weight - expected) <= 1e-12
    assert abs(open_total - expected) <= 1e-12

    # through the kernel: 200 decays, every one a (2212, 111) vertex at
    # the exact open-channel kinematics (never a KinematicsError)
    q_expected = decay_momentum(mass, MP, MPI0)
    assert q_expected > 0.0
    for i in range(200):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, i)
        vertices = decayer.decay_resonance(DELTA_PLUS, (mass, 0.0, 0.0, 0.0))
        assert len(vertices) == 1
        vertex = vertices[0]
        assert vertex.charge_combination == (2212, 111)
        assert vertex.channel.daughters == (2212, 111)
        d0, d1 = vertex.daughters
        assert (d0.pdg_id, d1.pdg_id) == (2212, 111)
        assert not d0.is_resonance and not d1.is_resonance
        _, px, py, pz = d0.four_momentum
        assert abs(math.sqrt(px * px + py * py + pz * pz)
                   - q_expected) <= 1e-12
