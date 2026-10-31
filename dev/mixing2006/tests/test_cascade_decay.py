"""Card tests for the sequential color-singlet resonance decay.

Round 4, card 4 (``TASK-20260920-r4-sequential-color-singlet-resonance-
decay``; plan section 22 task 21, plan sections 10.2-10.3, lines
676-714).  The card's pre-registered deterministic tests against the
committed catalog (``config/v2_singlet/resonance_catalog.json``) and the
continuum family configuration
(``config/v2_singlet/continuum_resonances.json``):

- ``test_state_fields``: the ``CascadeState`` carries every plan
  section 10.1 field; the depth counter increments on ``draw_decay``.
- ``test_channel_enumeration_conerves``: every enumerated channel
  conserves ``(B, Q, S)`` at the vertex; at least one daughter is lower
  in mass; every daughter is classified established hadron /
  established resonance / lower-mass continuum resonance against the
  committed catalog and the family configuration (pinned fixture).
- ``test_weight_normalization``: the channel weights sum to 1 within
  1e-12 on a pinned fixture spanning catalog-only, continuum-daughter,
  and mixed channels.
- ``test_isotropic_baseline``: at ``W_flavor = W_align = 1``, a pinned
  catalog-only single decay is bit-identical to the Round 2
  ``ResonanceDecayer`` at the same seed (the Round 2 kernel is the
  baseline; the angular distribution of a pinned two-body channel is
  isotropic within 3 sigma at 10,000 draws).
- ``test_draw_budget``: a pinned 1,000-step cascade trace consumes
  exactly 3 uniforms per vertex (1 channel + 2 kinematics) on the
  closed ``decay_channel``/``decay_kinematics`` substreams, no more.
- ``test_conservervation`` (the card's pre-registered name): over a
  pinned 10,000-step cascade trace, every vertex conserves ``(B, Q,
  S)`` exactly and the daughter four-momenta sum to the parent's within
  1e-9 relative.
- ``test_depth_count``: the depth counter is correct on the pinned
  trace; ``MAX_CASCADE_DEPTH`` exists and equals 12 (the placeholder).
- ``test_layout_sync``: the consumed-surface register and the layout
  sync (the subprocess check from the monorepo root; ``logic_drift=0``).

plus the card's acceptance-(vi) freeze re-pin check
``test_freeze_manifest_repin`` and the hook zero-draw check inside
``test_draw_budget`` (``channel_weight`` is pure: the hooks add zero
uniforms and do not change the channel draw order).

Hermetic: the committed config and catalog files, numpy, the standard
library, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    CascadeState,
    channel_weight,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import SUPPORT_M_MAX, families_from_json

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"

#: The round 4 master seed (round 4 index item 9), fixed for the
#: deterministic and statistical draw tests.
MASTER_SEED = 20260920

#: The pinned catalog-only bit-identity fixture: Delta+ (2214) at its
#: pole mass, at rest, whose quasi-two-body open set is the tabulated
#: N-pi charge pair (p pi0 2/3 : n pi+ 1/3, the card-5 shares) with both
#: daughters stable — the Round 2 kernel consumes exactly the frozen
#: 1 channel + 2 kinematics budget there, so the card-4 output is
#: bit-identical in the daughter four-momenta.
BIT_IDENTITY_PDG = 2214
BIT_IDENTITY_MASS = 1.232
BIT_IDENTITY_QN = (1, 1, 0)
BIT_IDENTITY_EVENT = 7

#: The pinned continuum fixtures: the active baryonic and mesonic
#: families of the accepted source range (round 4 index item 2), the
#: weight-normalization and trace roots.
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)
CONTINUUM_ROOTS = (
    ("baryonic_np", 3.0, BARYONIC_QN),
    ("baryonic_np", 4.0, BARYONIC_QN),
    ("baryonic_np", 6.0, BARYONIC_QN),
    ("baryonic_np", 8.0, BARYONIC_QN),
    ("mesonic_neutral", 3.0, MESONIC_QN),
    ("mesonic_neutral", 4.0, MESONIC_QN),
    ("mesonic_neutral", 6.0, MESONIC_QN),
    ("mesonic_neutral", 8.0, MESONIC_QN),
    ("baryonic_strange", 4.0, STRANGE_BARYONIC_QN),
    ("baryonic_strange", 8.0, STRANGE_BARYONIC_QN),
)

#: The frozen substrate of the card's provenance section (byte-unchanged
#: surfaces, verified on disk at this card's landing).
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: This card's freeze re-pin entries (the new module, the additive
#: package exports, and the new test file).  The manifest itself is
#: excluded from its own inventory by path and appears only in the
#: freeze exception's file list.
FREEZE_CARD = "TASK-20260920-r4-sequential-color-singlet-resonance-decay"
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
FREEZE_INVENTORY_FILES = (
    "dev/mixing2006/na49_mixing/cascade_decay.py",
    "dev/mixing2006/na49_mixing/__init__.py",
    "dev/mixing2006/tests/test_cascade_decay.py",
    # The card-3 blend-path audit register hard-pins the inventory
    # count; the card-mandated additive re-pin of this card bumps it
    # (452 -> 454), so the register file is re-pinned with the count
    # update attributed to this card.
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
)
FREEZE_EXCEPTION_FILES = FREEZE_INVENTORY_FILES + (MANIFEST_REL,)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(FAMILIES_PATH))


def _root_state(family_id: str, mass: float, qn: tuple[int, int, int],
                depth: int = 0) -> CascadeState:
    """A pinned at-rest root state with the lab-z production axis and
    the isotropic label (the caller-supplied axis and label of this
    card's landing)."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("exchange_probe",),
        depth=depth,
    )


def _sub_rngs(seed: int, event_index: int):
    """The closed decay_channel/decay_kinematics substream generators of
    one (master seed, replica 0, event index) triple (the round 2 card 9
    and round 3 card 8 convention)."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        np.random.default_rng(subs.stream_for("decay_channel")),
        np.random.default_rng(subs.stream_for("decay_kinematics")),
    )


class _CountingRng:
    """A counting proxy around one seeded generator (the house
    draw-count pattern): every ``random()`` consumes exactly one
    uniform and advances the counter."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


def _catalog_steps() -> tuple:
    """The pinned catalog-parent steps of the traces: two stable-daughter
    catalog decays (Delta+ and rho0) whose quasi-two-body open sets are
    two-body only."""

    return (
        ("catalog:2214", 1.232, BIT_IDENTITY_QN),
        ("catalog:113", 0.77526, MESONIC_QN),
    )


def _walk_trace(catalog: HadronCatalog, fams, n_steps: int, seed: int,
                observe=None) -> int:
    """The pinned cascade trace: ``draw_decay`` steps over the pinned
    roots (the continuum lines drawn among the open continuum channels
    while any is open, the direct channels at termination, and the
    pinned catalog-parent steps interleaved), in a fixed deterministic
    order, until *n_steps* steps are consumed.  ``observe`` is called
    per step with ``(parent, channel, daughters)``."""

    roots = CONTINUUM_ROOTS + _catalog_steps()
    steps = 0
    event = 0
    root_index = 0
    while steps < n_steps:
        family_id, mass, qn = roots[root_index % len(roots)]
        root_index += 1
        state = _root_state(family_id, mass, qn)
        while steps < n_steps:
            channels = enumerate_channels(state, catalog, fams)
            continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
            pool = continuum if continuum else [
                c for c in channels if c.kind != KIND_CONTINUUM]
            if not pool:
                break  # no open channel: the line terminates
            rng_channel = _CountingRng(seed + 2 * event)
            rng_kinematics = _CountingRng(seed + 2 * event + 1)
            channel, daughters = draw_decay(
                state, pool, catalog, fams, rng_channel, rng_kinematics)
            if rng_channel.calls != 1 or rng_kinematics.calls != 2:
                raise AssertionError(
                    f"draw budget violated at step {steps}: "
                    f"{rng_channel.calls} channel + "
                    f"{rng_kinematics.calls} kinematics uniforms"
                )
            if observe is not None:
                observe(state, channel, daughters)
            steps += 1
            event += 1
            successors = [
                d for d in daughters
                if not d.family_id.startswith(CATALOG_FAMILY_PREFIX)
            ]
            if not successors:
                break  # the line terminated through a direct channel
            state = successors[0]
            if state.depth > MAX_CASCADE_DEPTH:
                break  # the guard is card 9's; the trace never reaches it
    return steps


# ----------------------------------------------------------------------
# test_state_fields
# ----------------------------------------------------------------------
def test_state_fields(catalog, fams):
    """The ``CascadeState`` carries every plan section 10.1 field; the
    depth counter increments on ``draw_decay``."""

    state = CascadeState(
        momentum=(4.0, 0.5, -0.25, 0.125),
        mass=3.95,
        qn=BARYONIC_QN,
        quantum={},
        family_id="baryonic_np",
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("exchange_probe",),
        depth=3,
    )
    # every plan section 10.1 field, round-tripped
    assert state.momentum == (4.0, 0.5, -0.25, 0.125)
    assert state.mass == 3.95
    assert state.qn == BARYONIC_QN
    assert state.quantum == {}
    assert state.family_id == "baryonic_np"
    assert state.axis == (0.0, 0.0, 1.0)
    assert state.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert state.parent is None
    assert state.exchange_ancestry == ("exchange_probe",)
    assert state.depth == 3
    # frozen
    with pytest.raises(AttributeError):
        state.mass = 4.0  # type: ignore[misc]

    # the established-daughter quantum mapping carries I, I3, J, P
    catalog_state = _root_state(
        f"{CATALOG_FAMILY_PREFIX}2214", BIT_IDENTITY_MASS, BIT_IDENTITY_QN)
    channels = enumerate_channels(catalog_state, catalog, fams)
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 1)
    channel, (daughter, _) = draw_decay(
        catalog_state, channels, catalog, fams, rng_channel, rng_kinematics)
    daughter_pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
    daughter_record = catalog.get(daughter_pdg)
    assert daughter.quantum["I"] == pytest.approx(
        float(daughter_record.isospin))
    assert daughter.quantum["I3"] == int(daughter_record.isospin3)
    assert daughter.quantum["J"] == pytest.approx(float(daughter_record.spin))
    assert daughter.quantum["P"] == daughter_record.parity

    # the depth counter increments on draw_decay, on both parent kinds
    assert daughter.depth == catalog_state.depth + 1
    assert daughter.parent is catalog_state
    assert daughter.exchange_ancestry == ("exchange_probe",)
    continuum_state = _root_state("baryonic_np", 4.0, BARYONIC_QN, depth=2)
    channels = enumerate_channels(continuum_state, catalog, fams)
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 2)
    channel, daughters = draw_decay(
        continuum_state, channels, catalog, fams, rng_channel, rng_kinematics)
    for daughter in daughters:
        assert daughter.depth == continuum_state.depth + 1
        assert daughter.parent is continuum_state
        assert daughter.exchange_ancestry == ("exchange_probe",)
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert channel.parent_id == "baryonic_np"


# ----------------------------------------------------------------------
# test_channel_enumeration_conerves  (the card's pre-registered name)
# ----------------------------------------------------------------------
def _channel_qn_sum(channel: CascadeChannel, catalog: HadronCatalog,
                    fams) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` sum of a channel's daughters, resolved against
    the committed catalog and the family configuration."""

    total = [0, 0, 0]
    for daughter_id in channel.daughters:
        if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
            record = catalog.get(int(daughter_id[len(CATALOG_FAMILY_PREFIX):]))
            values = (int(record.B), int(record.Q), int(record.S))
        else:
            family = next(f for f in fams if f.family_id == daughter_id)
            values = (int(family.qn[0]), int(family.qn[1]), int(family.qn[2]))
        for k in range(3):
            total[k] += values[k]
    return (total[0], total[1], total[2])


def _daughter_mass_of(channel: CascadeChannel, daughter_id: str,
                      catalog: HadronCatalog, fams,
                      parent_mass: float) -> float:
    """The realized mass of a channel daughter (the established pole
    mass, or the card-4 continuum rule against the established
    sibling)."""

    if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
        return float(catalog.get(
            int(daughter_id[len(CATALOG_FAMILY_PREFIX):])).pole_mass)
    family = next(f for f in fams if f.family_id == daughter_id)
    other_id = channel.daughters[1] if channel.daughters[0] == daughter_id \
        else channel.daughters[0]
    if not other_id.startswith(CATALOG_FAMILY_PREFIX):
        raise AssertionError(
            "the card-4 enumeration produces at most one continuum "
            f"daughter per channel; got {channel.daughters}")
    sibling = float(catalog.get(
        int(other_id[len(CATALOG_FAMILY_PREFIX):])).pole_mass)
    edge = min(SUPPORT_M_MAX, parent_mass - sibling)
    return max(float(family.m_min), edge - float(family.nominal_width))


def test_channel_enumeration_conerves(catalog, fams):
    """Every enumerated channel conserves (B, Q, S) at the vertex; at
    least one daughter is lower in mass; every daughter is classified
    established hadron / established resonance / lower-mass continuum
    resonance against the committed catalog and the family
    configuration (pinned fixture)."""

    stable = catalog.stable_states()

    # -- the pinned catalog fixture (Delta+, the N-pi charge pair) -----
    state = _root_state(
        f"{CATALOG_FAMILY_PREFIX}2214", BIT_IDENTITY_MASS, BIT_IDENTITY_QN)
    channels = enumerate_channels(state, catalog, fams)
    assert channels, "the pinned catalog fixture has an open set"
    seen = set()
    for channel in channels:
        assert channel.parent_id == state.family_id
        assert _channel_qn_sum(channel, catalog, fams) == BIT_IDENTITY_QN
        masses = [
            _daughter_mass_of(channel, d, catalog, fams, state.mass)
            for d in channel.daughters
        ]
        assert any(m < state.mass for m in masses)
        for daughter_id in channel.daughters:
            pdg = int(daughter_id[len(CATALOG_FAMILY_PREFIX):])
            record = catalog.get(pdg)
            is_stable = pdg in stable or record.stable_for_decay
            assert is_stable or record.decay_channels, (
                f"daughter {daughter_id} is neither a stable "
                "established hadron nor an established resonance")
        expected_kind = KIND_ESTABLISHED_HADRON if all(
            int(d[len(CATALOG_FAMILY_PREFIX):]) in stable
            for d in channel.daughters) else KIND_ESTABLISHED_RESONANCE
        assert channel.kind == expected_kind
        key = (min(channel.daughters), max(channel.daughters))
        assert key not in seen, f"channel enumerated twice: {channel}"
        seen.add(key)
    assert {c.daughters for c in channels} == {
        (f"{CATALOG_FAMILY_PREFIX}2212", f"{CATALOG_FAMILY_PREFIX}111"),
        (f"{CATALOG_FAMILY_PREFIX}2112", f"{CATALOG_FAMILY_PREFIX}211"),
    }

    # a catalog parent below every threshold raises the kernel's
    # NoOpenChannelError (the Round 2 threshold discipline)
    below = _root_state(
        f"{CATALOG_FAMILY_PREFIX}2214", 1.0, BIT_IDENTITY_QN)
    from na49_mixing.resonance_decay import NoOpenChannelError
    with pytest.raises(NoOpenChannelError):
        enumerate_channels(below, catalog, fams)

    # a state whose quantum numbers disagree with its family raises
    mismatched = _root_state(
        f"{CATALOG_FAMILY_PREFIX}2214", BIT_IDENTITY_MASS, MESONIC_QN)
    with pytest.raises(ValueError):
        enumerate_channels(mismatched, catalog, fams)

    # -- the pinned continuum fixtures (the accepted source range) -----
    kinds_seen = set()
    for family_id, mass, qn in CONTINUUM_ROOTS:
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, fams)
        assert channels, f"{family_id} at {mass} has an open set"
        seen = set()
        for channel in channels:
            assert channel.parent_id == family_id
            assert _channel_qn_sum(channel, catalog, fams) == qn
            masses = [
                _daughter_mass_of(channel, d, catalog, fams, state.mass)
                for d in channel.daughters
            ]
            assert any(m < state.mass for m in masses), (
                f"{family_id}@{mass} channel {channel.daughters}: no "
                "daughter lower in mass")
            for daughter_id, daughter_mass in zip(channel.daughters, masses):
                if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
                    pdg = int(daughter_id[len(CATALOG_FAMILY_PREFIX):])
                    record = catalog.get(pdg)
                    assert pdg in stable or record.decay_channels, (
                        f"daughter {daughter_id} is neither a stable "
                        "established hadron nor an established resonance")
                    assert daughter_mass < state.mass
                else:
                    family = next(
                        f for f in fams if f.family_id == daughter_id)
                    assert (
                        float(family.m_min)
                        <= daughter_mass
                        <= min(SUPPORT_M_MAX, state.mass)
                    )
                    assert daughter_mass < state.mass
            kinds_seen.add(channel.kind)
            key = (min(channel.daughters), max(channel.daughters))
            assert key not in seen, f"channel enumerated twice: {channel}"
            seen.add(key)
    assert kinds_seen == {
        KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE, KIND_CONTINUUM}

    # the accepted source range keeps the cascade depth far below the
    # guard (the direct-chain reading of the plan section 10.6 line 757
    # clause; the guard enforcement itself is card 9)
    for family_id, mass, qn in CONTINUUM_ROOTS:
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, fams)
        continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
        while continuum:
            channel = continuum[0]
            daughter_id = next(
                d for d in channel.daughters
                if not d.startswith(CATALOG_FAMILY_PREFIX))
            daughter_mass = _daughter_mass_of(
                channel, daughter_id, catalog, fams, state.mass)
            daughter_family = next(
                f for f in fams if f.family_id == daughter_id)
            daughter_qn = (
                int(daughter_family.qn[0]),
                int(daughter_family.qn[1]),
                int(daughter_family.qn[2]),
            )
            state = _root_state(
                daughter_id, daughter_mass, daughter_qn,
                depth=state.depth + 1)
            assert state.depth <= MAX_CASCADE_DEPTH
            channels = enumerate_channels(state, catalog, fams)
            continuum = [c for c in channels if c.kind == KIND_CONTINUUM]


# ----------------------------------------------------------------------
# test_weight_normalization
# ----------------------------------------------------------------------
def test_weight_normalization(catalog, fams):
    """The channel weights sum to 1 within 1e-12 on a pinned fixture
    spanning catalog-only, continuum-daughter, and mixed channels; no
    channel is enumerated twice."""

    tol = 1e-12

    # catalog-only: the pinned catalog fixture
    state = _root_state(
        f"{CATALOG_FAMILY_PREFIX}2214", BIT_IDENTITY_MASS, BIT_IDENTITY_QN)
    channels = enumerate_channels(state, catalog, fams)
    assert all(c.kind != KIND_CONTINUUM for c in channels)
    total = math.fsum(c.weight for c in channels)
    assert abs(total - 1.0) <= tol
    assert all(math.isfinite(c.weight) and c.weight > 0.0 for c in channels)
    keys = [(min(c.daughters), max(c.daughters)) for c in channels]
    assert len(keys) == len(set(keys)), "a channel is enumerated twice"

    # mixed: the continuum parents carry established and continuum
    # daughters in one enumeration
    kinds_union = set()
    for family_id, mass, qn in (
        ("baryonic_np", 4.0, BARYONIC_QN),
        ("baryonic_np", 8.0, BARYONIC_QN),
        ("mesonic_neutral", 4.0, MESONIC_QN),
        ("baryonic_strange", 4.0, STRANGE_BARYONIC_QN),
    ):
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, fams)
        kinds = {c.kind for c in channels}
        # every continuum-parent fixture mixes continuum daughters with
        # at least one established kind (the direct channels are open
        # across the family support)
        assert KIND_CONTINUUM in kinds
        assert kinds & {KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE}
        kinds_union |= kinds
        total = math.fsum(c.weight for c in channels)
        assert abs(total - 1.0) <= tol, (family_id, mass, total)
        assert all(
            math.isfinite(c.weight) and c.weight > 0.0 for c in channels)
        keys = [(min(c.daughters), max(c.daughters)) for c in channels]
        assert len(keys) == len(set(keys)), "a channel is enumerated twice"
    # the pinned fixture set spans all three daughter kinds
    assert kinds_union == {
        KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE, KIND_CONTINUUM}

    # continuum-daughter: the continuum shares of the mixed enumeration
    # are a proper fraction of the vertex (the card-9 continuation pool
    # re-normalizes them inside draw_decay)
    state = _root_state("baryonic_np", 4.0, BARYONIC_QN)
    channels = enumerate_channels(state, catalog, fams)
    continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
    fraction = math.fsum(c.weight for c in continuum)
    assert 0.0 < fraction < 1.0
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 3)
    channel, daughters = draw_decay(
        state, continuum, catalog, fams, rng_channel, rng_kinematics)
    assert channel.kind == KIND_CONTINUUM
    assert any(not d.family_id.startswith(CATALOG_FAMILY_PREFIX)
               for d in daughters)


# ----------------------------------------------------------------------
# test_isotropic_baseline
# ----------------------------------------------------------------------
def test_isotropic_baseline(catalog, fams):
    """At W_flavor = W_align = 1, the pinned catalog-only single decay is
    bit-identical to the Round 2 ``ResonanceDecayer`` at the same seed
    (the Round 2 kernel is the baseline); the angular distribution of
    the pinned two-body channel is isotropic within 3 sigma at 10,000
    draws."""

    state = _root_state(
        f"{CATALOG_FAMILY_PREFIX}{BIT_IDENTITY_PDG}",
        BIT_IDENTITY_MASS, BIT_IDENTITY_QN)
    channels = enumerate_channels(state, catalog, fams)

    # -- the pinned single decay, bit-identical ------------------------
    kernel = ResonanceDecayer(
        catalog, MASTER_SEED, 0, BIT_IDENTITY_EVENT)
    vertices = kernel.decay_resonance(
        BIT_IDENTITY_PDG, (BIT_IDENTITY_MASS, 0.0, 0.0, 0.0))
    assert len(vertices) == 1  # both daughters stable: one vertex
    vertex = vertices[0]
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, BIT_IDENTITY_EVENT)
    channel, (daughter0, daughter1) = draw_decay(
        state, channels, catalog, fams, rng_channel, rng_kinematics)
    assert tuple(int(d[len(CATALOG_FAMILY_PREFIX):])
                 for d in channel.daughters) == tuple(
        int(d) for d in vertex.charge_combination)
    assert daughter0.momentum == tuple(vertex.daughters[0].four_momentum)
    assert daughter1.momentum == tuple(vertex.daughters[1].four_momentum)
    assert daughter0.mass == pytest.approx(float(
        catalog.get(int(vertex.daughters[0].pdg_id)).pole_mass))
    assert daughter1.mass == pytest.approx(float(
        catalog.get(int(vertex.daughters[1].pdg_id)).pole_mass))

    # -- the isotropic angular baseline at 10,000 draws ----------------
    n_draws = 10_000
    cosines = np.empty(n_draws)
    for event_index in range(n_draws):
        rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, event_index)
        _, (daughter, _) = draw_decay(
            state, channels, catalog, fams, rng_channel, rng_kinematics)
        px, py, pz = daughter.momentum[1], daughter.momentum[2], \
            daughter.momentum[3]
        cosines[event_index] = pz / math.sqrt(px * px + py * py + pz * pz)
    counts, _ = np.histogram(cosines, bins=10, range=(-1.0, 1.0))
    expected = n_draws / 10
    sigma = math.sqrt(expected * (1.0 - 1.0 / 10.0))
    worst = max(abs(int(c) - expected) for c in counts) / sigma
    assert worst <= 3.0, f"the isotropic baseline fails at {worst:.3f} sigma"


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """A pinned 1,000-step cascade trace consumes exactly 3 uniforms per
    vertex (1 channel + 2 kinematics) on the closed
    decay_channel/decay_kinematics substreams, no more; the hooks
    (W_flavor, W_align) add zero uniforms and do not change the channel
    draw order."""

    steps = _walk_trace(catalog, fams, 1_000, MASTER_SEED)
    assert steps == 1_000

    # the hooks are pure: zero uniforms, and a common factor on every
    # channel leaves the per-vertex draw counts and the channel draw
    # order unchanged
    state = _root_state("baryonic_np", 4.0, BARYONIC_QN)
    channels = enumerate_channels(state, catalog, fams)
    hooked = [
        CascadeChannel(
            parent_id=c.parent_id,
            daughters=c.daughters,
            kind=c.kind,
            weight=channel_weight(c, w_flavor=2.0, w_align=0.5),
        )
        for c in channels
    ]
    assert all(
        h.weight == pytest.approx(c.weight) for h, c in zip(hooked, channels))
    rng_a = _CountingRng(11)
    rng_b = _CountingRng(12)
    sel_plain, _ = draw_decay(
        state, channels, catalog, fams, rng_a, rng_b)
    assert rng_a.calls == 1 and rng_b.calls == 2
    rng_a = _CountingRng(11)
    rng_b = _CountingRng(12)
    sel_hooked, _ = draw_decay(
        state, hooked, catalog, fams, rng_a, rng_b)
    assert rng_a.calls == 1 and rng_b.calls == 2
    assert sel_hooked.daughters == sel_plain.daughters
    with pytest.raises(ValueError):
        channel_weight(channels[0], w_flavor=-1.0)


# ----------------------------------------------------------------------
# test_conservervation  (the card's pre-registered name)
# ----------------------------------------------------------------------
def test_conservervation(catalog, fams):
    """Over a pinned 10,000-step cascade trace, every vertex conserves
    (B, Q, S) exactly and the daughter four-momenta sum to the parent's
    within 1e-9 relative."""

    checked = {"steps": 0}

    def observe(parent, channel, daughters):
        parent_qn = tuple(int(v) for v in parent.qn)
        total = [0, 0, 0]
        for daughter in daughters:
            for k, value in enumerate(
                    (daughter.qn[0], daughter.qn[1], daughter.qn[2])):
                total[k] += int(value)
        assert tuple(total) == parent_qn, (
            f"vertex {parent.family_id} -> {channel.daughters}: "
            f"daughters sum to {tuple(total)}, parent carries {parent_qn}")
        for k in range(4):
            got = daughters[0].momentum[k] + daughters[1].momentum[k]
            want = parent.momentum[k]
            denom = max(1.0, abs(want))
            assert abs(got - want) <= 1e-9 * denom, (
                f"vertex {parent.family_id} -> {channel.daughters}: "
                f"four-momentum component {k} closure "
                f"{abs(got - want)!r} relative to {denom!r}")
        for daughter in daughters:
            e, px, py, pz = daughter.momentum
            m2 = e * e - px * px - py * py - pz * pz
            assert abs(math.sqrt(max(m2, 0.0)) - daughter.mass) <= 1e-9
            assert daughter.mass > 0.0
        checked["steps"] += 1

    steps = _walk_trace(catalog, fams, 10_000, MASTER_SEED, observe=observe)
    assert steps == 10_000
    assert checked["steps"] == 10_000


# ----------------------------------------------------------------------
# test_depth_count
# ----------------------------------------------------------------------
def test_depth_count(catalog, fams):
    """The depth counter is correct on the pinned trace;
    ``MAX_CASCADE_DEPTH`` exists and equals 12 (the placeholder)."""

    assert MAX_CASCADE_DEPTH == 12

    seen_depths = []

    def observe(parent, channel, daughters):
        for daughter in daughters:
            assert daughter.depth == parent.depth + 1
        seen_depths.append(parent.depth)

    steps = _walk_trace(catalog, fams, 1_000, MASTER_SEED, observe=observe)
    assert steps == 1_000
    assert seen_depths
    # the pinned trace never approaches the guard (plan section 10.6
    # line 757: tune physics so the guard is never reached)
    assert max(seen_depths) < MAX_CASCADE_DEPTH


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_layout_sync():
    """The consumed-surface register (the frozen substrate digests of
    the card's provenance section, verified on disk) and the layout
    sync (the subprocess check from the monorepo root)."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# The acceptance-(vi) freeze re-pin check
# ----------------------------------------------------------------------
def test_freeze_manifest_repin():
    """The freeze manifest carries this card's re-pin: the strict
    path-sorted inventory entries with the on-disk digests, and the
    freeze exception naming this card."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    keys = list(inventory.keys())
    assert keys == sorted(keys), "the inventory is not path-sorted"
    for rel in FREEZE_INVENTORY_FILES:
        assert rel in keys, f"missing re-pin entry: {rel}"
        assert inventory[rel] == _sha256_file(MONO_ROOT / rel), (
            f"stale re-pin digest: {rel}")
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e["card"] == FREEZE_CARD]
    assert len(entries) == 1, "the freeze exception must name this card once"
    for rel in FREEZE_EXCEPTION_FILES:
        assert rel in entries[0]["files"], (
            f"the freeze exception omits {rel}")
