"""Selected three-body decay vertex tests (Round 2, card 7).

Task ``TASK-20260913-r2-selected-three-body-decays`` (plan section 22
task 15; the three-body clause of plan section 10.2, line 692).  Six
deterministic tests carry the card's clauses on a hand-constructed
parent with a ``pi pi pi`` three-body channel (placeholder PDG id, the
pion daughters resolved from the vendored PDG table; no staged-pipeline
event generation), and two statistical tests verify the pre-registered
flat ``M_bc**2`` phase-space density and the charge balance of the
approved ``omega -> pi+ pi- pi0`` channel through the full kernel at
100,000 vertices:

- ``test_four_momentum_closure_3b``: 1,000 vertices (fixed seed) sum
  the three daughters to the parent four-momentum within ``1e-9`` GeV
  (the card's conservation assertion).
- ``test_mass_shell_3b``: all three daughters are on shell within
  ``1e-9`` (the exact-on-shell clause).
- ``test_bqs_conserved_3b``: ``(B, Q, S)`` is conserved at every
  vertex (the conservation clause on the channel's concrete charge
  states).
- ``test_mbc_range``: the sampled ``(b, c)`` invariant mass lies in
  ``[mb + mc, M - md]`` at every vertex (the leading pair is
  ``(daughters[0], daughters[1])``, the card-3 leading-pair
  convention).
- ``test_threshold_hard_zero``: a parent realized below its three-body
  threshold ``mb + mc + md`` raises ``NoOpenChannelError`` (the
  three-body channel is ineligible below threshold; the card-3 width
  law's hard zero through the open-object enumeration, no clamp, no
  rejection-with-retry).
- ``test_determinism_3b``: the same ``(master_seed, replica, event)``
  triple gives identical vertex lists; a different ``event_index``
  differs.
- ``test_mbc_flatness`` (statistical, 100,000 vertices): the ``M_bc**2``
  histogram is flat in 10 bins over the kinematic range to 3-sigma (the
  pre-registered flat density; uniform in ``M_bc**2``).
- ``test_charge_balance`` (statistical, 100,000 vertices on the
  approved ``omega -> pi+ pi- pi0`` channel, hand-built hermetic
  stand-in at the committed omega pole mass and width): the total
  charge per vertex is exactly 0 at every vertex (deterministic in
  fact; asserted in the sweep) and the composition is exactly one
  ``pi+``, one ``pi-``, one ``pi0``.

Hermetic: hand-constructed records and the vendored PDG table only; no
external tables, no staged-pipeline event generation.
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
    NoOpenChannelError,
    ResonanceDecayer,
    three_body_phase_space,
)

MASTER_SEED = 20260913

# Vendored-table daughter masses (the card-2 catalog values).
MPI = float(get_mass(211))
MPI0 = float(get_mass(111))

N_CLOSURE = 1_000
N_STATISTICAL = 100_000
CLOSURE_TOL = 1e-9

# The hand-built ``pi pi pi`` three-body parent (placeholder PDG id
# opaque to the card-5 quantum-number resolution, the card-6 fixture
# convention) and the hand-built hermetic omega stand-in for the
# approved charge-balance channel (the committed omega pole mass,
# nominal width, and m_up; a single represented channel).
P3PI_PDG = 87000323
OMEGA_PDG = 223
P3PI_MASS = 0.9
P3PI_WIDTH = 0.15
OMEGA_MASS = 0.78265
OMEGA_WIDTH = 0.00849

THREE_PI_THRESHOLD = math.fsum((MPI, MPI, MPI0))


# ----------------------------------------------------------------------
# hand-built records
# ----------------------------------------------------------------------
def _three_pi_channel(parent_pdg: int) -> DecayChannel:
    return DecayChannel(
        parent_pdg=int(parent_pdg),
        daughters=(211, -211, 111),
        branching_fraction=1.0,
        orbital_angular_momentum=0,
        is_three_body=True,
        provenance=None,
    )


@lru_cache(maxsize=1)
def _three_body_catalog() -> HadronCatalog:
    """The hand-built catalog: the ``pi pi pi`` three-body parent with
    one flagged channel, and the hermetic omega stand-in (one flagged
    ``pi+ pi- pi0`` channel) for the charge-balance sweep."""
    p3pi = HadronRecord(
        pdg_id=P3PI_PDG, name="P3pi", B=0, Q=0, S=0,
        isospin=0.0, isospin3=0, spin=1.0, parity="-",
        pole_mass=P3PI_MASS, nominal_width=P3PI_WIDTH,
        mass_range=(THREE_PI_THRESHOLD, P3PI_MASS + 5.0 * P3PI_WIDTH),
        decay_channels=(_three_pi_channel(P3PI_PDG),),
        stable_for_decay=False, provenance=None,
    )
    omega = HadronRecord(
        pdg_id=OMEGA_PDG, name="omega", B=0, Q=0, S=0,
        isospin=0.0, isospin3=0, spin=1.0, parity="-",
        pole_mass=OMEGA_MASS, nominal_width=OMEGA_WIDTH,
        mass_range=(THREE_PI_THRESHOLD, OMEGA_MASS + 5.0 * OMEGA_WIDTH),
        decay_channels=(_three_pi_channel(OMEGA_PDG),),
        stable_for_decay=False, provenance=None,
    )
    return HadronCatalog([p3pi, omega])


def _vertex_sweep(pdg_id: int, mass: float, count: int):
    """*count* single-vertex decays of *pdg_id* at rest at *mass*, one
    decayer per decay with ``event_index = i`` (the card-6 sweep
    convention), collected once and cached."""
    catalog = _three_body_catalog()
    vertices = []
    for i in range(count):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, i)
        produced = decayer.decay_resonance(pdg_id, (mass, 0.0, 0.0, 0.0))
        assert len(produced) == 1
        vertices.append(produced[0])
    return catalog, vertices


@lru_cache(maxsize=1)
def _closure_sweep():
    return _vertex_sweep(P3PI_PDG, P3PI_MASS, N_CLOSURE)


@lru_cache(maxsize=1)
def _flatness_sweep():
    _, vertices = _vertex_sweep(P3PI_PDG, P3PI_MASS, N_STATISTICAL)
    lo = (MPI + MPI) ** 2
    hi = (P3PI_MASS - MPI0) ** 2
    values = []
    for vertex in vertices:
        values.append(_m_bc_squared(vertex))
    return lo, hi, np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=1)
def _charge_sweep():
    return _vertex_sweep(OMEGA_PDG, OMEGA_MASS, N_STATISTICAL)


def _m_bc_squared(vertex) -> float:
    """The squared invariant mass of the leading pair
    ``(daughters[0], daughters[1])`` (the sampled ``M_bc**2``)."""
    e = vertex.daughters[0].four_momentum[0] \
        + vertex.daughters[1].four_momentum[0]
    px = vertex.daughters[0].four_momentum[1] \
        + vertex.daughters[1].four_momentum[1]
    py = vertex.daughters[0].four_momentum[2] \
        + vertex.daughters[1].four_momentum[2]
    pz = vertex.daughters[0].four_momentum[3] \
        + vertex.daughters[1].four_momentum[3]
    return e * e - px * px - py * py - pz * pz


# ----------------------------------------------------------------------
# deterministic tests
# ----------------------------------------------------------------------
def test_four_momentum_closure_3b():
    """1,000 vertices (fixed seed): the three daughter four-momenta sum
    to the parent four-momentum within 1e-9 GeV."""
    _catalog, vertices = _closure_sweep()
    assert len(vertices) == N_CLOSURE
    worst = 0.0
    for vertex in vertices:
        e, px, py, pz = vertex.parent_four_momentum
        total = (
            e - sum(d.four_momentum[0] for d in vertex.daughters),
            px - sum(d.four_momentum[1] for d in vertex.daughters),
            py - sum(d.four_momentum[2] for d in vertex.daughters),
            pz - sum(d.four_momentum[3] for d in vertex.daughters),
        )
        worst = max(worst, max(abs(x) for x in total))
        assert max(abs(x) for x in total) <= CLOSURE_TOL
    assert worst <= CLOSURE_TOL


def test_mass_shell_3b():
    """All three daughters on shell within 1e-9 at every vertex (the
    pion daughters at their vendored pole masses)."""
    _catalog, vertices = _closure_sweep()
    worst = 0.0
    for vertex in vertices:
        for daughter in vertex.daughters:
            mass = float(get_mass(daughter.pdg_id))
            e, px, py, pz = daughter.four_momentum
            deviation = abs(e * e - px * px - py * py - pz * pz
                            - mass * mass)
            worst = max(worst, deviation)
            assert deviation <= CLOSURE_TOL, (
                f"daughter pdg {daughter.pdg_id}: p^2 - m^2 = "
                f"{deviation!r}"
            )
    assert worst <= CLOSURE_TOL


def test_bqs_conserved_3b():
    """(B, Q, S) conserved at every vertex, resolved independently of
    the kernel through ``quantum_numbers``."""
    _catalog, vertices = _closure_sweep()
    parent_qn = (0, 0, 0)
    for vertex in vertices:
        total = [0, 0, 0]
        for daughter in vertex.daughters:
            qn = quantum_numbers(int(daughter.pdg_id))
            for k, value in enumerate((qn.B, qn.Q, qn.S)):
                total[k] += int(value)
        assert tuple(total) == parent_qn, (
            f"vertex {vertex.parent_pdg} -> "
            f"{vertex.charge_combination}: daughters {tuple(total)} != "
            f"parent {parent_qn}"
        )


def test_mbc_range():
    """The sampled (b, c) invariant mass lies in
    ``[mb + mc, M - md]`` at every vertex."""
    _catalog, vertices = _closure_sweep()
    lo = MPI + MPI
    hi = P3PI_MASS - MPI0
    assert lo < hi
    for vertex in vertices:
        m_bc = math.sqrt(_m_bc_squared(vertex))
        assert lo <= m_bc <= hi, (
            f"M_bc = {m_bc!r} outside [{lo!r}, {hi!r}]"
        )


def test_threshold_hard_zero():
    """A parent realized below its three-body threshold
    ``mb + mc + md`` raises ``NoOpenChannelError`` naming the parent
    and mass (the three-body channel is ineligible below threshold; no
    clamp, no rejection-with-retry)."""
    catalog = _three_body_catalog()
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    below = THREE_PI_THRESHOLD - 0.01
    assert below > 0.0
    with pytest.raises(NoOpenChannelError) as excinfo:
        decayer.decay_resonance(P3PI_PDG, (below, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert "87000323" in message and "P3pi" in message
    assert repr(float(below)) in message or str(below) in message


def test_determinism_3b():
    """The same ``(master_seed, replica, event)`` triple gives identical
    vertex lists; a different ``event_index`` differs."""
    catalog = _three_body_catalog()
    first = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4242
    ).decay_resonance(P3PI_PDG, (P3PI_MASS, 0.0, 0.0, 0.0))
    second = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4242
    ).decay_resonance(P3PI_PDG, (P3PI_MASS, 0.0, 0.0, 0.0))
    assert len(first) == len(second) == 1
    va, vb = first[0], second[0]
    assert va.channel is vb.channel
    assert va.charge_combination == vb.charge_combination
    assert va.parent_four_momentum == vb.parent_four_momentum
    assert len(va.daughters) == 3
    for da, db in zip(va.daughters, vb.daughters):
        assert da.pdg_id == db.pdg_id
        assert da.is_resonance == db.is_resonance
        for x, y in zip(da.four_momentum, db.four_momentum):
            assert abs(x - y) <= 1e-15

    other = ResonanceDecayer(
        catalog, MASTER_SEED, 0, 4243
    ).decay_resonance(P3PI_PDG, (P3PI_MASS, 0.0, 0.0, 0.0))
    assert len(other) == 1
    oc = other[0]
    differs = oc.charge_combination != va.charge_combination
    for da, db in zip(va.daughters, oc.daughters):
        if any(abs(x - y) > 1e-9
               for x, y in zip(da.four_momentum, db.four_momentum)):
            differs = True
    assert differs, "a different event_index produced the same vertex"


# ----------------------------------------------------------------------
# statistical tests (100,000 vertices, fixed seeds)
# ----------------------------------------------------------------------
def test_mbc_flatness():
    """The ``M_bc**2`` histogram is flat in 10 bins over the kinematic
    range to 3-sigma (the pre-registered phase-space-flat density)."""
    lo, hi, values = _flatness_sweep()
    n = values.size
    assert n == N_STATISTICAL
    counts, _ = np.histogram(values, bins=10, range=(lo, hi))
    expected = n / 10.0
    sigma = math.sqrt(expected * (1.0 - 0.1))
    for index, count in enumerate(counts):
        assert abs(int(count) - expected) <= 3.0 * sigma, (
            f"M_bc^2 bin {index}: {int(count)} vs {expected!r} "
            f"(3 sigma = {3.0 * sigma!r})"
        )


def test_charge_balance():
    """100,000 vertices of the approved ``omega -> pi+ pi- pi0``
    channel: the total charge is exactly 0 at every vertex and the
    composition is exactly one ``pi+``, one ``pi-``, one ``pi0``."""
    _catalog, vertices = _charge_sweep()
    assert len(vertices) == N_STATISTICAL
    for vertex in vertices:
        assert vertex.charge_combination == (211, -211, 111)
        composition = tuple(sorted(d.pdg_id for d in vertex.daughters))
        assert composition == (-211, 111, 211)
        charge = sum(int(quantum_numbers(int(d.pdg_id)).Q)
                     for d in vertex.daughters)
        assert charge == 0


# ----------------------------------------------------------------------
# the phase-space kernel's direct surface (the pre-registered draw
# budget and the closed-channel refusal)
# ----------------------------------------------------------------------
def test_three_body_phase_space_draw_budget_and_threshold():
    """``three_body_phase_space`` consumes exactly 4 uniforms per call
    (the pre-registered budget) and refuses a parent mass at or below
    the three-body threshold with ``ValueError`` (the hard zero below
    threshold is the caller's concern)."""

    class _CountingRng:
        def __init__(self):
            self.draws = 0

        def random(self):
            self.draws += 1
            return 0.5

    counting = _CountingRng()
    points = three_body_phase_space(
        P3PI_MASS, MPI, MPI, MPI0, counting  # type: ignore[arg-type]
    )
    assert counting.draws == 4
    assert len(points) == 3
    total = points[0] + points[1] + points[2]
    assert abs(total[0] - P3PI_MASS) <= CLOSURE_TOL
    assert abs(total[1]) <= CLOSURE_TOL
    assert abs(total[2]) <= CLOSURE_TOL
    assert abs(total[3]) <= CLOSURE_TOL

    with pytest.raises(ValueError):
        three_body_phase_space(
            THREE_PI_THRESHOLD, MPI, MPI, MPI0,
            np.random.default_rng(0),
        )
