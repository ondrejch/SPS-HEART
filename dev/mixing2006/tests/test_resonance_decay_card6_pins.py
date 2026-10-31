"""Card-6 recursive two-body decay kernel: the committed-behavior pins.

Task ``TASK-20260913-r2-recursive-two-body-decays`` (plan section 22
task 14; plan section 8.4 step 4).  This sibling of
``tests/test_resonance_decay.py`` pins the committed card-6 surface of
``na49_mixing/resonance_decay.py`` (nested ef915d4) on the scope items
the primary file leaves open:

- ``test_multi_generation_chain_in_flight``: the seven-step loop on the
  hand-constructed two-generation catalog with the root in flight
  (non-zero three-momentum): every vertex conserves the four-momentum
  to ``1e-9`` GeV, every daughter is on shell to ``1e-9``, the decay
  queue is parent-first BFS (every non-root vertex is the unique later
  vertex of one resonance daughter of an earlier vertex, and each
  resonance daughter is queued exactly once), the strict daughter-mass
  invariant holds on the realized masses, the chain terminates on the
  catalog stable set, and the final stable particles sum to the root
  four-momentum to ``1e-9`` (the global chain conservation the
  per-vertex closure implies).
- ``test_depth_guard_allows_max_chain_depth``: a hand-constructed
  descending ladder of exactly ``MAX_CHAIN_DEPTH`` states decays to
  completion with exactly ``MAX_CHAIN_DEPTH`` vertices (the guard
  bounds the chain from above; the primary file pins that 65 and 66
  raise ``DecayCycleError`` naming the chain).
- ``test_inaccessible_all_channels_closed``: a parent whose two
  channels are both above its realized mass (the channel is open at
  the pole) raises ``NoOpenChannelError`` naming the parent, the pdg,
  and the exact realized mass; the message carries the no-repair
  semantics (no clamp, no rejection-with-retry).
- ``test_width_zero_at_realized_threshold``: a parent open at the pole
  whose realized mass sits exactly at the channel threshold carries a
  zero open weight (the card-3 hard zero at the threshold, not a
  clamp) and raises ``NoOpenChannelError`` naming the width
  normalization.
- ``test_phi_uniform_kernel`` (statistical, 100,000 decays): the
  azimuth ``phi`` of the first daughter is uniform in 10 bins to
  3-sigma (the second uniform of the spin-averaged kernel, round 2
  index item 9; the primary file pins the ``cos(theta)`` histogram of
  the same sample).
- ``test_class_shares_follow_width_law`` (statistical, 30,000 decays of
  the N*+ at a fixed mass): the between-class open-weight shares of the
  root follow the card-3/4 partial-width ratios at the realized mass
  (not the branching-fraction ratios), and the within-class charge
  split of the Delta-pi class keeps the card-5 ``2/3 : 1/3`` shares
  through the kernel (the card-6 scope decision on the open-weight
  formula; the primary file pins the within-class split of the Delta+
  N-pi class only).
- ``test_feed_down_lineage``: every ``DecayDaughter`` carries its
  direct ``parent_pdg``; every final particle's ``parent_id`` is the
  exact index of its producing vertex (not just a bound); the full
  ancestry walk from every final particle reaches the source root
  through the vertex list; the final particles carry the source
  ``(B, Q, S)``.
- ``test_determinism_replica_and_seed``: two decayers with the same
  ``(master_seed, replica, event)`` produce identical vertex lists of
  the in-flight root (four-momenta and realized masses to ``1e-15``,
  the same channel and combination sequence); a different replica
  index or master seed produces a different sequence (the primary file
  pins the event-index variation at rest).
- ``test_source_record_entry_point``: the ``decay_source`` entry point
  accepts the card-6 ``SourceRecord`` and returns
  ``(list[DecayVertex], list[FinalParticle])`` identical to
  ``decay_resonance`` for the same triple; a quantum-number mismatch
  and a mass-invariant mismatch raise ``KinematicsError`` (checked,
  never repaired); a source that is already a stable state returns
  ``([], [])``.
- ``test_stable_record_terminates``: a daughter record flagged
  ``stable_for_decay`` is a member of ``catalog.stable_states()``, is
  not queued (a final daughter at its pole mass), and decays to an
  empty vertex list as the root: the loop stops only on the configured
  stable set.
- ``test_card5_isospin_surface_unchanged``: the card-5 public
  interface (``derive_charge_weights``, ``isospin_closure_report``,
  ``PerParentIsospinReport``, ``IsospinCompletionReport``) keeps its
  documented shape and values on the card-6 fixture; the completed
  catalog's per-parent reports carry no weight-sum or forbidden
  channel errors.  The card-5 test file
  (``tests/test_resonance_isospin.py``) is re-run unmodified by the
  card validation.
- ``test_card34_width_surface_unchanged``: the card-3/4 width surface
  (``draw_mass``, ``total_width``, ``decay_momentum``,
  ``channel_width``, the card-4 default barrier) keeps its documented
  shape, and the card-6 kernel consumes it as-is (the module-level
  imports bind to the same objects).  The card-3/4 test files
  (``tests/test_resonance_widths.py``, ``tests/test_barrier_factors.py``)
  are re-run unmodified by the card validation.

The hand-built fixtures are the committed file's (imported, not
duplicated), except the protection and configured-stable fixtures
built here on the same placeholder-PDG-id convention.  Hermetic: no
staged-pipeline event generation, no external tables.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import fields
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

import na49_mixing.resonance_decay as _resonance_decay_module  # noqa: E402
import na49_mixing.resonance_widths as _resonance_widths_module  # noqa: E402
from na49_mixing.hadron_catalog import (  # noqa: E402
    HadronCatalog,
    HadronRecord,
)
from na49_mixing.pdg_database import get_mass  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    MAX_CHAIN_DEPTH,
    DecayVertex,
    IsospinCompletionReport,
    KinematicsError,
    NoOpenChannelError,
    PerParentIsospinReport,
    ResonanceDecayer,
    SourceRecord,
    derive_charge_weights,
    isospin_closure_report,
)
from na49_mixing.resonance_widths import channel_width  # noqa: E402
from na49_mixing.source_event import FinalParticle  # noqa: E402

# The committed card-6 fixtures (imported, not duplicated).
from tests.test_resonance_decay import (  # noqa: E402
    CLOSURE_TOL,
    DELTA_PLUS,
    MASTER_SEED,
    M_DELTA,
    MPI,
    MP,
    NSTAR_PLUS,
    _bqs_of,
    _channel,
    _delta_sample,
    _ladder_catalog,
    _pole_mass_of,
    _record,
    _two_generation_catalog,
)

# A root four-momentum in flight (non-zero three-momentum, the legacy
# ``_decay_rest_frame`` boost convention exercised down the chain).
INFLIGHT_ROOT = (1.75, 0.4, 0.25, 0.15)
_E, _PX, _PY, _PZ = INFLIGHT_ROOT
INFLIGHT_MASS = math.sqrt(_E * _E - _PX * _PX - _PY * _PY - _PZ * _PZ)
N_CHAINS = 16

# The Delta-pi charge combinations of the card-5-completed N*+ record.
DELTA_PI_COMBOS = {(2214, 111), (2114, 211)}
N_PI_COMBOS = {(2212, 111), (2112, 211)}
N_CLASS = 30_000
CLASS_MASS = 1.6


# ----------------------------------------------------------------------
# 1. The seven-step loop: exact on-shell kinematics and termination on
#    a multi-generation chain with the root in flight
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _in_flight_chains() -> tuple[tuple[int, list], ...]:
    """A deterministic batch of in-flight N*+ decays (one fresh decayer
    per decay, ``event_index = i``), as ``(event_index, vertices)``."""
    catalog = _two_generation_catalog()
    chains: list[tuple[int, list]] = []
    for i in range(N_CHAINS):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, i)
        chains.append((i, decayer.decay_resonance(NSTAR_PLUS, INFLIGHT_ROOT)))
    return tuple(chains)


def test_multi_generation_chain_in_flight():
    catalog = _two_generation_catalog()
    stable = catalog.stable_states()
    n_multi = 0
    for _, vertices in _in_flight_chains():
        assert vertices
        root = vertices[0]
        assert root.parent_pdg == NSTAR_PLUS
        assert abs(root.parent_mass - INFLIGHT_MASS) <= 1e-12
        assert root.parent_four_momentum == INFLIGHT_ROOT

        # the realized mass of every resonance daughter (its own
        # vertex's parent mass), keyed by (pdg_id, four_momentum)
        realized = {
            (vertex.parent_pdg, vertex.parent_four_momentum):
            vertex.parent_mass
            for vertex in vertices
        }

        for index, vertex in enumerate(vertices):
            # the per-vertex four-momentum conservation clause (1e-9)
            for k in range(4):
                total = vertex.parent_four_momentum[k]
                for daughter in vertex.daughters:
                    total -= daughter.four_momentum[k]
                assert abs(total) <= CLOSURE_TOL, (
                    f"vertex {vertex.parent_pdg} component {k} misses by "
                    f"{total!r} GeV"
                )
            # the exact-kinematics clause: every daughter on shell
            for daughter in vertex.daughters:
                mass = (
                    realized[(daughter.pdg_id, daughter.four_momentum)]
                    if daughter.is_resonance
                    else _pole_mass_of(catalog, daughter.pdg_id)
                )
                e, px, py, pz = daughter.four_momentum
                deviation = abs(e * e - px * px - py * py - pz * pz
                                - mass * mass)
                assert deviation <= CLOSURE_TOL, (
                    f"daughter pdg {daughter.pdg_id} off shell by "
                    f"{deviation!r} GeV^2"
                )
            # the decay queue is parent-first BFS: a non-root vertex is
            # the later vertex of exactly one resonance daughter of an
            # earlier vertex (no double-queued vertex)
            if index > 0:
                earlier = [
                    (i2, d)
                    for i2, v2 in enumerate(vertices[:index])
                    for d in v2.daughters
                    if d.is_resonance
                    and d.pdg_id == vertex.parent_pdg
                    and d.four_momentum == vertex.parent_four_momentum
                ]
                assert len(earlier) == 1, (
                    f"vertex {index} (pdg {vertex.parent_pdg}) is the "
                    f"later vertex of {len(earlier)} resonance daughters"
                )

        # step 7 plus the strict daughter-mass invariant: every
        # resonance daughter is queued exactly once at a strictly lower
        # realized mass; every stable daughter is in the configured
        # stable set
        for index, vertex in enumerate(vertices):
            for daughter in vertex.daughters:
                if not daughter.is_resonance:
                    assert daughter.pdg_id in stable
                    continue
                later = [
                    other for other in vertices[index + 1:]
                    if other.parent_pdg == daughter.pdg_id
                    and other.parent_four_momentum
                    == daughter.four_momentum
                ]
                assert len(later) == 1, (
                    f"resonance daughter pdg {daughter.pdg_id} of "
                    f"vertex {index} has {len(later)} later vertices"
                )
                assert later[0].parent_mass < vertex.parent_mass, (
                    f"daughter pdg {daughter.pdg_id} realized mass "
                    f"{later[0].parent_mass!r} is not strictly below the "
                    f"parent realized mass {vertex.parent_mass!r}"
                )

        # the global chain conservation: the final stable particles sum
        # to the root four-momentum to 1e-9
        for k in range(4):
            total = INFLIGHT_ROOT[k]
            for vertex in vertices:
                for daughter in vertex.daughters:
                    if not daughter.is_resonance:
                        total -= daughter.four_momentum[k]
            assert abs(total) <= CLOSURE_TOL, (
                f"the final particles miss the root four-momentum "
                f"component {k} by {total!r} GeV"
            )

        if len(vertices) >= 2:
            n_multi += 1
    # the batch actually exercised the multi-generation chain
    # (N*+ -> Delta + pi -> N pi), not only the single-vertex N pi one
    assert n_multi > 0, "no multi-generation chain in the batch"


# ----------------------------------------------------------------------
# 2. The cycle protection: the depth guard allows exactly 64 vertices
#    (the primary file pins that 65 and 66 raise DecayCycleError)
# ----------------------------------------------------------------------
def test_depth_guard_allows_max_chain_depth():
    ids, catalog = _ladder_catalog(MAX_CHAIN_DEPTH, cyclic=False)
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    vertices = decayer.decay_resonance(ids[0], (2.0, 0.0, 0.0, 0.0))
    assert len(vertices) == MAX_CHAIN_DEPTH
    assert [v.parent_pdg for v in vertices] == list(ids)
    # the last vertex decays to the stable set (the chain terminates
    # at the guard boundary, no protection fired)
    for daughter in vertices[-1].daughters:
        assert not daughter.is_resonance


# ----------------------------------------------------------------------
# 3. The inaccessibility protection: every channel above the realized
#    mass (the primary file pins the single-channel case)
# ----------------------------------------------------------------------
def test_inaccessible_all_channels_closed():
    # two channels, both above the realized mass (the channel is open
    # at the pole: the zero open weight is the realized mass, not the
    # pole closure of the primary file's case)
    parent = _record(
        87000300, "Pacc2", 1, 1, 0, 0.5, 1,
        [_channel(87000300, (2212, 211), 0.5, L=0),
         _channel(87000300, (2112, 211), 0.5, L=0)],
        pole_mass=2.0, nominal_width=0.2, m_up=2.9,
    )
    catalog = HadronCatalog([parent])
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    mass = math.sqrt(1.05 * 1.05)
    with pytest.raises(NoOpenChannelError) as excinfo:
        decayer.decay_resonance(87000300, (1.05, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert "Pacc2" in message and "87000300" in message
    # the exact realized mass is named
    assert repr(mass) in message
    # the no-repair semantics of the legacy threshold discipline
    assert "no clamp" in message


# ----------------------------------------------------------------------
# 4. The width-normalization protection: zero open weight at the
#    realized threshold of an open-at-pole channel
# ----------------------------------------------------------------------
def test_width_zero_at_realized_threshold():
    # the channel is open at the pole (pole above the threshold); the
    # realized mass sits exactly at the threshold, where the card-3
    # width law carries the hard zero: the open weight is zero, the
    # total width is zero, and the protection names the width
    # normalization
    parent = _record(
        87000301, "Pthr", 1, 1, 0, 0.5, 1,
        [_channel(87000301, (2212, 211), 1.0, L=0)],
        pole_mass=1.3, nominal_width=0.2, m_up=1.6,
    )
    catalog = HadronCatalog([parent])
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    threshold = MP + MPI
    with pytest.raises(NoOpenChannelError) as excinfo:
        decayer.decay_resonance(87000301, (threshold, 0.0, 0.0, 0.0))
    message = str(excinfo.value)
    assert "Pthr" in message and "87000301" in message
    assert "width normalization" in message

    # the independent width-law check: the same channel carries a
    # positive width at the pole and exactly zero at the threshold
    # (the hard zero, not a clamp)
    record = catalog.get(87000301)
    masses = {2212: MP, 211: MPI}
    assert channel_width(1.3, record.decay_channels[0], record, masses) > 0.0
    assert channel_width(threshold, record.decay_channels[0], record,
                         masses) == 0.0


# ----------------------------------------------------------------------
# 5. The spin-averaged angular kernel: the azimuth uniformity (the
#    primary file pins the cos(theta) histogram of the same sample)
# ----------------------------------------------------------------------
def test_phi_uniform_kernel():
    sample = _delta_sample()
    n = len(sample)
    bins = [0] * 10
    for _, px, py, pz in sample:
        phi = math.atan2(py, px)
        index = min(int((phi + math.pi) / (2.0 * math.pi) * 10), 9)
        bins[index] += 1
    expected = n / 10.0
    sigma = math.sqrt(expected * (1.0 - 0.1))
    for index, count in enumerate(bins):
        assert abs(count - expected) <= 3.0 * sigma, (
            f"phi bin {index}: {count} vs {expected!r} "
            f"(3 sigma = {3.0 * sigma!r})"
        )


# ----------------------------------------------------------------------
# 6. The open-weight formula: between-class shares follow the width
#    law, and the within-class card-5 shares hold through the kernel
# ----------------------------------------------------------------------
def test_class_shares_follow_width_law():
    catalog = _two_generation_catalog()
    record = catalog.get(NSTAR_PLUS)
    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter in masses:
                continue
            try:
                masses[daughter] = float(catalog.get(daughter).pole_mass)
            except KeyError:
                masses[daughter] = float(get_mass(daughter))
    # the independent open-weight side: the card-3/4 partial widths at
    # the realized root mass (the between-class shares of the kernel's
    # open objects are exactly these ratios); the Delta pi class takes
    # the convolved width of its broad Delta daughter (FAB-10), the law
    # the kernel applies with its catalog
    gamma_delta = math.fsum(
        channel_width(CLASS_MASS, channel, record, masses, catalog=catalog)
        for channel in record.decay_channels
        if int(channel.daughters[0]) in (2114, 2214)
    )
    gamma_npi = math.fsum(
        channel_width(CLASS_MASS, channel, record, masses, catalog=catalog)
        for channel in record.decay_channels
        if int(channel.daughters[0]) in (2112, 2212)
    )
    expected_delta = gamma_delta / (gamma_delta + gamma_npi)
    # the width-law share discriminates from the branching-fraction
    # share (0.6 of the class budget): the test pins the former
    assert expected_delta != 0.6

    n_delta = 0
    n_npi = 0
    n_p_pi0 = 0
    n_delta0_pi = 0
    for i in range(N_CLASS):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 2_000_000 + i)
        vertices = decayer.decay_resonance(
            NSTAR_PLUS, (CLASS_MASS, 0.0, 0.0, 0.0))
        assert vertices[0].parent_pdg == NSTAR_PLUS
        combo = vertices[0].charge_combination
        if combo in DELTA_PI_COMBOS:
            n_delta += 1
            if combo == (2214, 111):
                n_p_pi0 += 1
            else:
                n_delta0_pi += 1
        else:
            assert combo in N_PI_COMBOS
            n_npi += 1
    total = n_delta + n_npi
    assert total == N_CLASS
    share = n_delta / total
    sigma = math.sqrt(expected_delta * (1.0 - expected_delta) / total)
    assert abs(share - expected_delta) <= 3.0 * sigma, (
        f"Delta-pi class share {share!r} vs the width-law share "
        f"{expected_delta!r} (3 sigma = {3.0 * sigma!r})"
    )

    # the within-class charge split keeps the card-5 shares (2/3 :
    # 1/3 for Delta+ pi0 : Delta0 pi+) through the full kernel
    assert n_delta > 0
    share_within = n_p_pi0 / n_delta
    sigma_within = math.sqrt((2.0 / 3.0) * (1.0 / 3.0) / n_delta)
    assert abs(share_within - 2.0 / 3.0) <= 3.0 * sigma_within, (
        f"within-class (2214, 111) share {share_within!r} vs 2/3 "
        f"(3 sigma = {3.0 * sigma_within!r})"
    )


# ----------------------------------------------------------------------
# 7. Feed-down lineage: the direct parent_pdg, the exact producing
#    vertex index, and the recoverable full chain
# ----------------------------------------------------------------------
def test_feed_down_lineage():
    catalog = _two_generation_catalog()
    stable = catalog.stable_states()
    vertices: list = []
    finals: list = []
    found = False
    for i in range(N_CHAINS):
        decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 900_000 + i)
        vertices, finals = decayer.decay_source(SourceRecord(
            pdg_id=NSTAR_PLUS, four_momentum=INFLIGHT_ROOT,
            B=1, Q=1, S=0, mass=INFLIGHT_MASS, source_id=42,
        ))
        if len(vertices) >= 2:
            found = True
            break
    assert found, "no multi-generation chain in the first events"

    # every DecayDaughter carries its direct parent_pdg
    for vertex in vertices:
        for daughter in vertex.daughters:
            assert daughter.parent_pdg == vertex.parent_pdg

    # the final particles: the exact index of the producing vertex,
    # the stable set, the source identity
    assert [f.particle_id for f in finals] == list(range(len(finals)))
    for final in finals:
        assert final.pdg_id in stable
        assert final.source_id == 42
        assert final.channel_family == "resonance_decay"
        assert final.parent_id is not None
        producer = vertices[final.parent_id]
        matches = [
            daughter for daughter in producer.daughters
            if not daughter.is_resonance
            and daughter.pdg_id == final.pdg_id
        ]
        assert len(matches) == 1, (
            f"final particle {final.pdg_id} at index "
            f"{final.particle_id} is not the unique stable daughter of "
            f"its producing vertex {final.parent_id}"
        )
        for x, y in zip(matches[0].four_momentum,
                        (final.p4.e, final.p4.px, final.p4.py, final.p4.pz)):
            assert abs(x - y) <= 1e-15

    # the full chain is recoverable from the vertex list: the ancestry
    # walk from every final particle reaches the source root
    for final in finals:
        index = final.parent_id
        while vertices[index].parent_pdg != NSTAR_PLUS:
            parent_pdg = vertices[index].parent_pdg
            parent_p4 = vertices[index].parent_four_momentum
            assert parent_pdg not in stable
            earlier = [
                (i2, d)
                for i2, v2 in enumerate(vertices[:index])
                for d in v2.daughters
                if d.is_resonance and d.pdg_id == parent_pdg
                and d.four_momentum == parent_p4
            ]
            assert len(earlier) == 1, (
                f"the resonance ancestor pdg {parent_pdg} of vertex "
                f"{index} has no producing vertex"
            )
            index = earlier[0][0]
        assert vertices[index].parent_pdg == NSTAR_PLUS

    # the final particles carry the source (B, Q, S)
    total = [0, 0, 0]
    for final in finals:
        for k, value in enumerate(_bqs_of(catalog, final.pdg_id)):
            total[k] += value
    assert tuple(total) == (1, 1, 0)


# ----------------------------------------------------------------------
# 8. Determinism: the replica and seed variation (the primary file
#    pins the event-index variation at rest)
# ----------------------------------------------------------------------
def test_determinism_replica_and_seed():
    catalog = _two_generation_catalog()

    def run(seed, replica, event):
        return ResonanceDecayer(catalog, seed, replica, event).decay_resonance(
            NSTAR_PLUS, INFLIGHT_ROOT)

    first = run(MASTER_SEED, 0, 555)
    second = run(MASTER_SEED, 0, 555)
    assert len(first) == len(second) >= 1
    for va, vb in zip(first, second):
        assert va.parent_pdg == vb.parent_pdg
        assert va.channel == vb.channel
        assert va.charge_combination == vb.charge_combination
        assert va.parent_four_momentum == vb.parent_four_momentum
        assert abs(va.parent_mass - vb.parent_mass) <= 1e-15
        for da, db in zip(va.daughters, vb.daughters):
            assert da.pdg_id == db.pdg_id
            assert da.is_resonance == db.is_resonance
            for x, y in zip(da.four_momentum, db.four_momentum):
                assert abs(x - y) <= 1e-15

    def differs(a, b) -> bool:
        if len(a) != len(b):
            return True
        for va, vb in zip(a, b):
            if va.charge_combination != vb.charge_combination:
                return True
            if abs(va.parent_mass - vb.parent_mass) > 1e-15:
                return True
            for da, db in zip(va.daughters, vb.daughters):
                if any(abs(x - y) > 1e-9
                       for x, y in zip(da.four_momentum, db.four_momentum)):
                    return True
        return False

    assert differs(first, run(MASTER_SEED, 1, 555)), (
        "a different replica_index produced the same sequence"
    )
    assert differs(first, run(MASTER_SEED + 1, 0, 555)), (
        "a different master_seed produced the same sequence"
    )


# ----------------------------------------------------------------------
# 9. The SourceRecord entry point: the return contract, the checked
#    (never repaired) clauses, and the stable-source short circuit
# ----------------------------------------------------------------------
def _stable_record_catalog() -> HadronCatalog:
    """A hand-built catalog with a daughter record flagged
    ``stable_for_decay`` (the configured stable set) and a root that
    decays to it plus pi0."""
    stub = HadronRecord(
        pdg_id=87000401, name="Stub", B=1, Q=1, S=0,
        isospin=0.5, isospin3=1, spin=0.5, parity="+",
        pole_mass=0.8, nominal_width=0.0,
        mass_range=(0.8, 0.8), decay_channels=(),
        stable_for_decay=True, provenance=None,
    )
    root = HadronRecord(
        pdg_id=87000400, name="Root", B=1, Q=1, S=0,
        isospin=0.5, isospin3=1, spin=0.5, parity="+",
        pole_mass=1.3, nominal_width=0.15,
        mass_range=(0.935, 1.7),
        decay_channels=(_channel(87000400, (87000401, 111), 1.0, L=0),),
        stable_for_decay=False, provenance=None,
    )
    return HadronCatalog([stub, root])


def test_source_record_entry_point():
    catalog = _two_generation_catalog()
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 666)
    expected = decayer.decay_resonance(NSTAR_PLUS, INFLIGHT_ROOT)
    decayer2 = ResonanceDecayer(catalog, MASTER_SEED, 0, 666)
    result = decayer2.decay_source(SourceRecord(
        pdg_id=NSTAR_PLUS, four_momentum=INFLIGHT_ROOT,
        B=1, Q=1, S=0, mass=INFLIGHT_MASS, source_id=7,
    ))
    assert isinstance(result, tuple) and len(result) == 2
    vertices, finals = result
    assert isinstance(vertices, list)
    assert all(isinstance(v, DecayVertex) for v in vertices)
    assert all(isinstance(f, FinalParticle) for f in finals)
    # identical to decay_resonance for the same triple
    assert len(vertices) == len(expected)
    for va, vb in zip(vertices, expected):
        assert va.parent_pdg == vb.parent_pdg
        assert va.charge_combination == vb.charge_combination
        assert abs(va.parent_mass - vb.parent_mass) <= 1e-15
        for da, db in zip(va.daughters, vb.daughters):
            assert da.pdg_id == db.pdg_id
            for x, y in zip(da.four_momentum, db.four_momentum):
                assert abs(x - y) <= 1e-15

    # the source quantum numbers are checked, never repaired
    with pytest.raises(KinematicsError) as excinfo:
        decayer2.decay_source(SourceRecord(
            pdg_id=NSTAR_PLUS, four_momentum=INFLIGHT_ROOT,
            B=1, Q=0, S=0, mass=INFLIGHT_MASS,
        ))
    assert "12212" in str(excinfo.value)

    # the fixed mass is checked against the four-momentum invariant
    with pytest.raises(KinematicsError) as excinfo2:
        decayer2.decay_source(SourceRecord(
            pdg_id=NSTAR_PLUS, four_momentum=INFLIGHT_ROOT,
            B=1, Q=1, S=0, mass=INFLIGHT_MASS + 0.1,
        ))
    assert "invariant mass" in str(excinfo2.value)

    # a source that is already a stable state returns ([], [])
    decayer3 = ResonanceDecayer(_stable_record_catalog(), MASTER_SEED, 0, 0)
    assert decayer3.decay_source(SourceRecord(
        pdg_id=87000401, four_momentum=(0.8, 0.0, 0.0, 0.0),
        B=1, Q=1, S=0, mass=0.8, source_id=5,
    )) == ([], [])


# ----------------------------------------------------------------------
# 10. Termination on the configured stable set: a stable_for_decay
#     record is a terminal state (the primary file pins the record-
#     absent stable set)
# ----------------------------------------------------------------------
def test_stable_record_terminates():
    catalog = _stable_record_catalog()
    assert 87000401 in catalog.stable_states()
    decayer = ResonanceDecayer(catalog, MASTER_SEED, 0, 0)
    vertices = decayer.decay_resonance(87000400, (1.4, 0.0, 0.0, 0.0))
    assert len(vertices) == 1
    assert vertices[0].charge_combination == (87000401, 111)
    stub_d, pi0_d = vertices[0].daughters
    # the configured-stable record is not queued: it is a final
    # daughter at its pole mass
    assert stub_d.pdg_id == 87000401
    assert not stub_d.is_resonance
    e, px, py, pz = stub_d.four_momentum
    assert abs(e * e - px * px - py * py - pz * pz - 0.8 ** 2) <= CLOSURE_TOL
    assert pi0_d.pdg_id == 111
    assert not pi0_d.is_resonance
    # a configured-stable record as the root has no decay
    assert decayer.decay_resonance(87000401, (0.8, 0.0, 0.0, 0.0)) == []


# ----------------------------------------------------------------------
# 11. The card-5 isospin surface is not weakened: the public interface
#     keeps its documented shape and values on the card-6 fixture
# ----------------------------------------------------------------------
def test_card5_isospin_surface_unchanged():
    assert {f.name for f in fields(PerParentIsospinReport)} == {
        "pdg_id", "missing_charge_partners", "weight_sum_errors",
        "forbidden_channels", "mirror_mismatches",
    }
    assert {f.name for f in fields(IsospinCompletionReport)} == {
        "per_parent", "errors", "warnings",
    }

    catalog = _two_generation_catalog()
    record = catalog.get(NSTAR_PLUS)
    delta_pi = derive_charge_weights(record, (2214, 111), catalog)
    assert set(delta_pi) == DELTA_PI_COMBOS
    assert abs(delta_pi[(2214, 111)] - 0.4) <= 1e-12
    assert abs(delta_pi[(2114, 211)] - 0.2) <= 1e-12
    # the N-pi class of the I = 1/2 parent: p pi0 : n pi+ = 1/3 : 2/3
    # (the card-5 Clebsch-Gordan shares of the I = 1/2 component, the
    # mirror of the Delta+ I = 3/2 split)
    n_pi = derive_charge_weights(record, (2212, 111), catalog)
    assert set(n_pi) == N_PI_COMBOS
    assert abs(n_pi[(2212, 111)] - 1.0 / 3.0 * 0.4) <= 1e-12
    assert abs(n_pi[(2112, 211)] - 2.0 / 3.0 * 0.4) <= 1e-12

    report = isospin_closure_report(catalog)
    assert isinstance(report, IsospinCompletionReport)
    assert len(report.per_parent) == len(catalog)
    for per_parent in report.per_parent:
        assert isinstance(per_parent, PerParentIsospinReport)
        # the card-5-completed classes are weight-closed: no
        # weight-sum or forbidden-channel errors at the channel level
        assert per_parent.weight_sum_errors == ()
        assert per_parent.forbidden_channels == ()


# ----------------------------------------------------------------------
# 12. The card-3/4 width surface is not weakened: the public interface
#     keeps its documented shape, and the kernel consumes it as-is
# ----------------------------------------------------------------------
def test_card34_width_surface_unchanged():
    # the card-6 kernel's module-level imports bind to the same
    # objects as the card-3/4 module (the kernel uses the card-3/4
    # width law, not a re-implementation)
    for name in ("channel_width", "decay_momentum", "draw_mass",
                 "total_width"):
        assert getattr(_resonance_decay_module, name) is \
            getattr(_resonance_widths_module, name)
    # the card-4 default barrier is the documented Blatt-Weisskopf
    # factor (the default swap of the width-law signatures)
    for function in (_resonance_widths_module.channel_width,
                     _resonance_widths_module.total_width,
                     _resonance_widths_module.draw_mass):
        defaults = function.__defaults__
        assert defaults is not None
        assert defaults[0] is _resonance_widths_module.BLATT_WEISSKOPF_BARRIER
    # the documented shape: the barrier values and the hard zero at
    # the threshold, with the hand-computed decay momentum elsewhere
    assert _resonance_widths_module.BLATT_WEISSKOPF_BARRIER(0, 0.7) == 1.0
    threshold = MP + MPI
    assert _resonance_widths_module.decay_momentum(
        threshold, MP, MPI) == 0.0
    q_hand = math.sqrt(
        (CLASS_MASS ** 2 - (MP + MPI) ** 2)
        * (CLASS_MASS ** 2 - (MP - MPI) ** 2)
    ) / (2.0 * CLASS_MASS)
    assert abs(_resonance_widths_module.decay_momentum(
        CLASS_MASS, MP, MPI) - q_hand) <= 1e-12
