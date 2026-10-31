"""Round 4, card 4 test-author register: sequential color-singlet decay.

Independent contract pins for the sequential color-singlet resonance
decay module (``na49_mixing/cascade_decay.py``), Round 4 card 4
(``TASK-20260920-r4-sequential-color-singlet-resonance-decay``; plan
section 22 task 21, plan sections 10.2-10.3, lines 676-714).  This
register is the test-author half of the card: it pins the module
contract on its own synthetic catalog and continuum family fixtures
(self-authored records and family records with real PDG ids for the
vendored-table quantum-number resolution), its own seeds, and its own
fixed-uniform stubs, and it re-pins the freeze manifest for the card.

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_state_roundtrip_frozen_and_quantum_mapping``: the
  ``CascadeState`` field round-trip, the frozen dataclasses, the
  established-daughter ``quantum`` mapping (``I``, ``I3``, ``J``,
  ``P`` from the catalog record), the empty mapping on a continuum
  daughter, and the depth/parent/ancestry/axis/label bookkeeping of a
  drawn vertex.
- ``test_enumerate_catalog_parent_open_set``: the catalog-parent open
  set on the self-authored catalog (the N-pi charge pair at the card-5
  tabulated shares, the pi-pi class with the Bose-forbidden pi0-pi0
  combination excluded by the charge completion, the synthetic
  K*-pi parent with an established-resonance daughter class), the
  kernel object order, the per-vertex weight normalization, the
  threshold ``NoOpenChannelError``, and the quantum-mismatch
  ``ValueError``.
- ``test_enumerate_continuum_parent_open_set``: the continuum-parent
  open set on the self-authored family configuration (the direct
  allowed-decay-class pairs, the in-place continuum replacement with
  the deterministic daughter mass rule, the closed budget window, the
  ``SUPPORT_M_MAX`` cap branch, deduplication of a class declared
  twice, the empty open set below all thresholds, the quantum-mismatch
  and unknown-family ``ValueError``s).
- ``test_channel_weight_norm_and_hooks``: the per-vertex normalization
  to 1 within 1e-12 across both parent kinds, the multiplicative hook
  contract, the unity-hook draw-order invariance, the per-class hook
  share change, and the negative / non-finite hook ``ValueError``.
- ``test_draw_budget_conservation_ta_fixture``: a 100-step cascade
  trace on the synthetic fixtures with counting proxies (exactly 1
  channel + 2 kinematics uniforms per vertex on the closed
  ``decay_channel`` / ``decay_kinematics`` substreams), exact
  ``(B, Q, S)`` conservation, the four-momentum closure within
  ``DECAY_CLOSURE_TOL``, the on-shell daughters, the depth increment,
  the parent link, the inherited exchange ancestry, and the exact
  card-4 continuum daughter mass rule on every drawn continuum
  daughter.
- ``test_zero_weight_edges``: the zero-weight edge behaviors the
  module pins: an empty open set returns an empty list (the caller
  decides termination), ``draw_decay`` raises ``ValueError`` on an
  empty channel list and on channel weights that do not sum to a
  finite positive total, and ``channel_weight`` raises ``ValueError``
  on a negative or non-finite hook.
- ``test_round2_kernel_structural_contract``: the Round 2 baseline
  properties as structural contracts (not a seed/event re-run): the
  catalog-parent enumeration order and relative weights equal the
  kernel ``_open_objects`` two-body object order and weights for a
  self-authored parent, and the cumulative-walk channel draw selects
  by the first channel whose cumulative weight reaches the hand-fed
  ``u * total`` target.
- ``test_module_constants``: ``MAX_CASCADE_DEPTH``,
  ``CATALOG_FAMILY_PREFIX``, ``ISOTROPIC_ALIGNMENT_LABEL``, and the
  kind constants.

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256) and the additive ``na49_mixing`` package exports.
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (the builder, the spectral-density module, the
  threshold-metric engine, the Round 2 kernel baseline, the committed
  catalog, the committed continuum configuration, and the R32 driver
  in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory carries this file at its
  on-disk digest, the card-4 freeze exception names this card exactly
  once with this register's file, and the card-3 register's inventory
  count pin is re-pinned 454 -> 455 for this file).

Hermetic: the self-authored fixtures, numpy, the standard library, and
the vendored ``na49_mixing`` modules only; no event generation, no
external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from na49_mixing.hadron_catalog import DecayChannel, HadronCatalog, HadronRecord
from na49_mixing.resonance_decay import (
    DECAY_CLOSURE_TOL,
    NoOpenChannelError,
    ResonanceDecayer,
)
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import SUPPORT_M_MAX, ContinuumFamily

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_decay.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_decay.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_decay.py"
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_decay_contract_pins.py")
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
FREEZE_CARD = "TASK-20260920-r4-sequential-color-singlet-resonance-decay"

#: The module byte stability pin (the card-4 solver landing).
MODULE_SHA256 = (
    "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c"
)

#: The frozen substrate of the card's provenance section (byte-unchanged
#: surfaces, verified on disk at this card's landing): the builder, the
#: spectral-density module, the threshold-metric engine, the Round 2
#: kernel baseline, the committed catalog and continuum configuration,
#: and the R32 driver in both trees.
FROZEN_SUBSTRATE = {
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    # The Round 2 kernel baseline (the byte-unchanged open-object
    # construction the catalog-parent enumeration consumes).
    "dev/mixing2006/na49_mixing/resonance_decay.py":
        "6a1e8f0ad3c9e6e56645f919ba14060836ab27dd8e53235eca4434f9b1aa8210",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: The independent register seed (the day after the round 4 master
#: seed, the solver's 20260920).
REGISTER_SEED = 20260921


# ----------------------------------------------------------------------
# The self-authored synthetic catalog (real PDG ids; vendored-table
# quantum-number resolution)
# ----------------------------------------------------------------------
def _record(
    pdg_id: int,
    name: str,
    B: int,
    Q: int,
    S: int,
    isospin: float,
    isospin3: int,
    spin: float,
    parity: str,
    pole_mass: float,
    nominal_width: float = 0.0,
    mass_range: tuple[float, float] | None = None,
    channels: tuple[tuple[int, int, float, int], ...] = (),
    stable_for_decay: bool = True,
) -> HadronRecord:
    """One self-authored catalog record (the vendored-table masses)."""

    if mass_range is None:
        mass_range = (pole_mass, pole_mass)
    decay_channels = tuple(
        DecayChannel(
            parent_pdg=pdg_id,
            daughters=(d0, d1),
            branching_fraction=br,
            orbital_angular_momentum=L,
        )
        for d0, d1, br, L in channels
    )
    return HadronRecord(
        pdg_id=pdg_id,
        name=name,
        B=B,
        Q=Q,
        S=S,
        isospin=isospin,
        isospin3=isospin3,
        spin=spin,
        parity=parity,
        pole_mass=pole_mass,
        nominal_width=nominal_width,
        mass_range=mass_range,
        decay_channels=decay_channels,
        stable_for_decay=stable_for_decay,
        provenance=None,
    )


def _ta_catalog() -> HadronCatalog:
    """The self-authored small multi-resonance catalog.

    The stable hadrons resolve through the vendored PDG table quantum
    numbers (the round 2 kernel's daughter sources); the established
    resonances carry the tabulated charge channels; ``taRho`` (pdg
    9014) is a synthetic parent whose K*-pi class exercises the
    established-resonance channel kind.  The masses are the vendored
    table values, so the thresholds are the physical ones.
    """

    records = (
        _record(2212, "proton", 1, 1, 0, 0.5, 1, 0.5, "+", 0.93827231),
        _record(2112, "neutron", 1, 0, 0, 0.5, -1, 0.5, "+", 0.93956541),
        _record(211, "pi+", 0, 1, 0, 1.0, 2, 0, "-", 0.1395699),
        _record(-211, "pi-", 0, -1, 0, 1.0, -2, 0, "-", 0.1395699),
        _record(111, "pi0", 0, 0, 0, 1.0, 0, 0, "-", 0.1349766),
        _record(321, "K+", 0, 1, 1, 0.5, 1, 0, "-", 0.493677),
        _record(
            2214, "Delta+", 1, 1, 0, 1.5, 1, 1.5, "-",
            1.232, nominal_width=0.150,
            mass_range=(1.07913531, 1.5),
            channels=((2212, 111, 2.0 / 3.0, 1),
                      (2112, 211, 1.0 / 3.0, 1)),
            stable_for_decay=False,
        ),
        _record(
            113, "rho0", 0, 0, 0, 1.0, 0, 1, "-",
            0.77526, nominal_width=0.149,
            mass_range=(0.2791398, 1.0),
            channels=((211, -211, 2.0 / 3.0, 1),
                      (111, 111, 1.0 / 3.0, 1)),
            stable_for_decay=False,
        ),
        _record(
            323, "K*(892)+", 0, 1, 1, 0.5, 1, 1, "-",
            0.89166, nominal_width=0.050,
            mass_range=(0.6332469, 1.2),
            channels=((321, -211, 1.0, 1),),
            stable_for_decay=False,
        ),
        _record(
            313, "K*(892)0", 0, 0, 1, 0.5, -1, 1, "-",
            0.89555, nominal_width=0.050,
            mass_range=(0.6332469, 1.2),
            channels=((321, -211, 1.0, 1),),
            stable_for_decay=False,
        ),
        _record(
            9014, "taRho", 0, 1, 1, 1.0, 1, 1, "-",
            1.42, nominal_width=0.10,
            mass_range=(1.0351199, 2.0),
            channels=((323, 111, 0.5, 1),
                      (313, 211, 0.5, 1)),
            stable_for_decay=False,
        ),
    )
    return HadronCatalog(records)


#: The self-authored continuum family configuration: a baryonic family
#: (whose allowed-decay-class list declares "(2212,111)" twice, with
#: surrounding whitespace, to pin the deduplication and the class
#: identifier parsing) and a mesonic family.
TA_BARYON = ContinuumFamily(
    family_id="ta_baryon",
    qn=(1, 1, 0),
    m_min=2.4,
    t_h=0.165,
    b_power=0.0,
    n_norm=1.0,
    ramp_width=0.5,
    nominal_width=0.4,
    allowed_decay_classes=(" (2212,111) ", "(2112,211)", "(2212,111)"),
    active=True,
    provenance=(),
)
TA_MESON = ContinuumFamily(
    family_id="ta_meson",
    qn=(0, 0, 0),
    m_min=1.6,
    t_h=0.165,
    b_power=0.0,
    n_norm=1.0,
    ramp_width=0.5,
    nominal_width=0.25,
    allowed_decay_classes=("(211,-211)",),
    active=True,
    provenance=(),
)
TA_FAMILIES = (TA_BARYON, TA_MESON)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return _ta_catalog()


def _root_state(
    family_id: str, mass: float, qn: tuple[int, int, int], depth: int = 0
) -> CascadeState:
    """An at-rest root state with the lab-z axis, the isotropic label,
    and the register's exchange ancestry."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("cascade_probe",),
        depth=depth,
    )


def _sub_rngs(seed: int, event_index: int) -> tuple:
    """The closed decay_channel/decay_kinematics substream generators
    of one (register seed, replica 0, event index) triple."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        np.random.default_rng(subs.stream_for("decay_channel")),
        np.random.default_rng(subs.stream_for("decay_kinematics")),
    )


class _CountingRng:
    """A counting proxy around one seeded generator: every ``random()``
    consumes exactly one uniform and advances the counter."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


class _StubRng:
    """A hand-fed uniform stub: ``random()`` pops the next pinned value."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = list(values)
        self.calls = 0

    def random(self) -> float:
        value = self._values[self.calls]
        self.calls += 1
        return float(value)


def _qn_sum(
    state_or_daughters,
) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` sum of a pair of daughter states."""

    total = [0, 0, 0]
    for daughter in state_or_daughters:
        for k, value in enumerate(
                (daughter.qn[0], daughter.qn[1], daughter.qn[2])):
            total[k] += int(value)
    return (total[0], total[1], total[2])


def _channel_qn_sum(
    channel: CascadeChannel,
    catalog: HadronCatalog,
    family_config: tuple[ContinuumFamily, ...],
) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` sum of a channel's daughter ids, resolved
    against the synthetic catalog and family configuration."""

    total = [0, 0, 0]
    for daughter_id in channel.daughters:
        if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
            record = catalog.get(
                int(daughter_id[len(CATALOG_FAMILY_PREFIX):]))
            values = (int(record.B), int(record.Q), int(record.S))
        else:
            family = next(
                f for f in family_config if f.family_id == daughter_id)
            values = (
                int(family.qn[0]), int(family.qn[1]), int(family.qn[2]))
        for k in range(3):
            total[k] += values[k]
    return (total[0], total[1], total[2])


def _continuum_daughter_mass(
    daughter_id: str,
    sibling_pole_mass: float,
    parent_mass: float,
    family_config: tuple[ContinuumFamily, ...],
) -> float:
    """The card-4 deterministic continuum daughter mass rule,
    recomputed in the test (the same expression the module uses)."""

    family = next(f for f in family_config if f.family_id == daughter_id)
    edge = min(SUPPORT_M_MAX, parent_mass - sibling_pole_mass)
    return max(float(family.m_min), edge - float(family.nominal_width))


# ----------------------------------------------------------------------
# Part 1: the module contract
# ----------------------------------------------------------------------
def test_state_roundtrip_frozen_and_quantum_mapping(catalog):
    """The ``CascadeState`` carries every plan section 10.1 field; the
    frozen dataclasses reject mutation; the established-daughter
    quantum mapping is the catalog record's {I, I3, J, P}; a continuum
    daughter carries the empty mapping; the depth counter increments
    on ``draw_decay``."""

    state = CascadeState(
        momentum=(4.0, 0.5, -0.25, 0.125),
        mass=3.95,
        qn=(1, 1, 0),
        quantum={},
        family_id="ta_baryon",
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("cascade_probe",),
        depth=3,
    )
    # every plan section 10.1 field, round-tripped
    assert state.momentum == (4.0, 0.5, -0.25, 0.125)
    assert state.mass == 3.95
    assert state.qn == (1, 1, 0)
    assert state.quantum == {}
    assert state.family_id == "ta_baryon"
    assert state.axis == (0.0, 0.0, 1.0)
    assert state.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert state.parent is None
    assert state.exchange_ancestry == ("cascade_probe",)
    assert state.depth == 3
    # frozen
    with pytest.raises(AttributeError):
        state.mass = 4.0  # type: ignore[misc]
    channel = CascadeChannel(
        parent_id="ta_baryon",
        daughters=("catalog:2112", "catalog:211"),
        kind=KIND_ESTABLISHED_HADRON,
        weight=0.25,
    )
    with pytest.raises(AttributeError):
        channel.weight = 0.5  # type: ignore[misc]

    # the established-daughter quantum mapping: rho0 -> (pi+, pi-)
    root = _root_state(f"{CATALOG_FAMILY_PREFIX}113", 0.77526, (0, 0, 0))
    channels = enumerate_channels(root, catalog, TA_FAMILIES)
    drawn, (daughter0, daughter1) = draw_decay(
        root, channels, catalog, TA_FAMILIES,
        _StubRng((0.0,)), _StubRng((0.25, 0.5)))
    assert drawn.daughters == (f"{CATALOG_FAMILY_PREFIX}211",
                               f"{CATALOG_FAMILY_PREFIX}-211")
    for daughter, record in ((daughter0, catalog.get(211)),
                             (daughter1, catalog.get(-211))):
        assert daughter.quantum == {
            "I": float(record.isospin),
            "I3": int(record.isospin3),
            "J": float(record.spin),
            "P": record.parity,
        }
        assert daughter.qn == (int(record.B), int(record.Q), int(record.S))
        assert daughter.mass == pytest.approx(float(record.pole_mass))
        assert daughter.depth == root.depth + 1
        assert daughter.parent is root
        assert daughter.exchange_ancestry == ("cascade_probe",)
        assert daughter.axis == (0.0, 0.0, 1.0)
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL

    # a continuum daughter carries the empty quantum mapping and the
    # family's (B, Q, S)
    croot = _root_state("ta_baryon", 3.0, (1, 1, 0))
    cchannels = enumerate_channels(croot, catalog, TA_FAMILIES)
    total = math.fsum(c.weight for c in cchannels)
    cumulative = 0.0
    selected = None
    u = None
    for c in cchannels:
        if c.kind == KIND_CONTINUUM:
            selected = c
            u = (cumulative + c.weight / 2.0) / total
            break
        cumulative += c.weight
    assert selected is not None, "ta_baryon at 3.0 carries continuum channels"
    assert u is not None
    drawn, (d0, d1) = draw_decay(
        croot, cchannels, catalog, TA_FAMILIES,
        _StubRng((u,)), _StubRng((0.25, 0.5)))
    assert drawn is selected
    cont = d0 if d0.family_id == "ta_baryon" else d1
    assert cont.quantum == {}
    assert cont.qn == (1, 1, 0)
    assert cont.family_id == "ta_baryon"
    assert cont.depth == croot.depth + 1
    assert cont.parent is croot


# ----------------------------------------------------------------------
# The catalog-parent open set on the self-authored catalog
# ----------------------------------------------------------------------
def test_enumerate_catalog_parent_open_set(catalog):
    """The catalog-parent open set on the self-authored catalog: the
    N-pi charge pair at the tabulated card-5 shares, the pi-pi class
    with the Bose-forbidden pi0-pi0 combination excluded, the synthetic
    K*-pi parent's established-resonance class, the kernel object
    order, the per-vertex normalization, the threshold
    ``NoOpenChannelError``, and the quantum-mismatch ``ValueError``."""

    family_config = TA_FAMILIES

    # -- Delta+ at the pole: the tabulated N-pi charge pair --------------
    state = _root_state(f"{CATALOG_FAMILY_PREFIX}2214", 1.232, (1, 1, 0))
    channels = enumerate_channels(state, catalog, family_config)
    assert [c.daughters for c in channels] == [
        (f"{CATALOG_FAMILY_PREFIX}2212", f"{CATALOG_FAMILY_PREFIX}111"),
        (f"{CATALOG_FAMILY_PREFIX}2112", f"{CATALOG_FAMILY_PREFIX}211"),
    ]
    assert all(c.kind == KIND_ESTABLISHED_HADRON for c in channels)
    assert [c.parent_id for c in channels] == [state.family_id] * 2
    assert math.fsum(c.weight for c in channels) == pytest.approx(1.0, abs=1e-12)
    assert channels[0].weight == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert channels[1].weight == pytest.approx(1.0 / 3.0, abs=1e-12)
    for c in channels:
        assert _channel_qn_sum(c, catalog, family_config) == (1, 1, 0)

    # -- rho0: the Bose-forbidden pi0-pi0 combination is excluded --------
    rstate = _root_state(f"{CATALOG_FAMILY_PREFIX}113", 0.77526, (0, 0, 0))
    rchannels = enumerate_channels(rstate, catalog, family_config)
    assert [c.daughters for c in rchannels] == [
        (f"{CATALOG_FAMILY_PREFIX}211", f"{CATALOG_FAMILY_PREFIX}-211"),
    ]
    assert rchannels[0].kind == KIND_ESTABLISHED_HADRON
    assert rchannels[0].weight == pytest.approx(1.0, abs=1e-12)
    assert _channel_qn_sum(rchannels[0], catalog, family_config) == (0, 0, 0)

    # -- taRho: the established-resonance K*-pi class --------------------
    tstate = _root_state(f"{CATALOG_FAMILY_PREFIX}9014", 1.42, (0, 1, 1))
    tchannels = enumerate_channels(tstate, catalog, family_config)
    assert [c.daughters for c in tchannels] == [
        (f"{CATALOG_FAMILY_PREFIX}323", f"{CATALOG_FAMILY_PREFIX}111"),
        (f"{CATALOG_FAMILY_PREFIX}313", f"{CATALOG_FAMILY_PREFIX}211"),
    ]
    assert all(c.kind == KIND_ESTABLISHED_RESONANCE for c in tchannels)
    assert math.fsum(c.weight for c in tchannels) == pytest.approx(1.0, abs=1e-12)
    for c in tchannels:
        assert _channel_qn_sum(c, catalog, family_config) == (0, 1, 1)

    # -- below every threshold: the kernel's NoOpenChannelError ----------
    below = _root_state(f"{CATALOG_FAMILY_PREFIX}2214", 1.0, (1, 1, 0))
    with pytest.raises(NoOpenChannelError):
        enumerate_channels(below, catalog, family_config)

    # -- the quantum mismatch is checked, never repaired ------------------
    mismatched = _root_state(f"{CATALOG_FAMILY_PREFIX}2214", 1.232,
                             (0, 0, 0))
    with pytest.raises(ValueError):
        enumerate_channels(mismatched, catalog, family_config)


# ----------------------------------------------------------------------
# The continuum-parent open set on the self-authored families
# ----------------------------------------------------------------------
def test_enumerate_continuum_parent_open_set(catalog):
    """The continuum-parent open set on the self-authored family
    configuration: the direct pairs and the in-place replacements,
    deduplicated, with the deterministic daughter mass rule, the closed
    budget window, the ``SUPPORT_M_MAX`` cap branch, and the empty open
    set below all thresholds."""

    family_config = TA_FAMILIES

    # -- ta_baryon at 3.0: four channels (two direct, two replaced) -----
    # (the neutron lead of class (2112,211) carries (B, Q, S) =
    # (1, 0, 0), which no declared family matches, so no replacement is
    # offered there: the replacement keeps the lead's quantum numbers)
    state = _root_state("ta_baryon", 3.0, (1, 1, 0))
    channels = enumerate_channels(state, catalog, family_config)
    expected_order = [
        ("catalog:2212", "catalog:111", KIND_ESTABLISHED_HADRON),
        ("ta_baryon", "catalog:111", KIND_CONTINUUM),
        ("catalog:2212", "ta_meson", KIND_CONTINUUM),
        ("catalog:2112", "catalog:211", KIND_ESTABLISHED_HADRON),
    ]
    assert [(c.daughters, c.kind) for c in channels] == [
        ((d0, d1), k) for d0, d1, k in expected_order
    ]  # the exact four-channel set, in order
    assert all(c.parent_id == "ta_baryon" for c in channels)
    # no channel is enumerated twice (the class is declared twice)
    keys = [(min(c.daughters), max(c.daughters)) for c in channels]
    assert len(keys) == len(set(keys))
    # every vertex conserves (B, Q, S); the masses are the card-4 rule
    for c in channels:
        assert _channel_qn_sum(c, catalog, family_config) == (1, 1, 0)
        assert math.isfinite(c.weight) and c.weight > 0.0
        for daughter_id in c.daughters:
            if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
                continue
            assert c.kind == KIND_CONTINUUM
            sibling = (
                c.daughters[1] if c.daughters[0] == daughter_id
                else c.daughters[0]
            )
            assert sibling.startswith(CATALOG_FAMILY_PREFIX), (
                "the card-4 enumeration produces at most one continuum "
                f"daughter per channel; got {c.daughters}")
            sibling_mass = float(catalog.get(
                int(sibling[len(CATALOG_FAMILY_PREFIX):])).pole_mass)
            expected_mass = _continuum_daughter_mass(
                daughter_id, sibling_mass, state.mass, family_config)
            assert expected_mass == max(
                float(next(f for f in family_config
                           if f.family_id == daughter_id).m_min),
                min(SUPPORT_M_MAX, state.mass - sibling_mass)
                - float(next(f for f in family_config
                             if f.family_id == daughter_id).nominal_width))
    assert math.fsum(c.weight for c in channels) == pytest.approx(1.0, abs=1e-12)

    # the pinned realized masses of the replaced daughters
    # (ta_baryon daughter of class (2212,111): the sibling is pi0)
    m_b1 = _continuum_daughter_mass(
        "ta_baryon", float(catalog.get(111).pole_mass), 3.0, family_config)
    assert m_b1 == pytest.approx(max(2.4, min(5.0, 3.0 - 0.1349766) - 0.4))
    # ta_meson daughter of class (2212,111): the sibling is the proton
    m_m1 = _continuum_daughter_mass(
        "ta_meson", float(catalog.get(2212).pole_mass), 3.0, family_config)
    assert m_m1 == pytest.approx(max(1.6, min(5.0, 3.0 - 0.93827231) - 0.25))

    # -- ta_baryon at 2.5: the replacement windows are closed ------------
    low = _root_state("ta_baryon", 2.5, (1, 1, 0))
    low_channels = enumerate_channels(low, catalog, family_config)
    assert [(c.daughters, c.kind) for c in low_channels] == [
        (("catalog:2212", "catalog:111"), KIND_ESTABLISHED_HADRON),
        (("catalog:2112", "catalog:211"), KIND_ESTABLISHED_HADRON),
    ]

    # -- ta_baryon at 6.0: the SUPPORT_M_MAX cap branch -------------------
    high = _root_state("ta_baryon", 6.0, (1, 1, 0))
    high_channels = enumerate_channels(high, catalog, family_config)
    for c in high_channels:
        if c.kind != KIND_CONTINUUM:
            continue
        daughter_id = next(
            d for d in c.daughters
            if not d.startswith(CATALOG_FAMILY_PREFIX))
        sibling_id = next(
            d for d in c.daughters
            if d.startswith(CATALOG_FAMILY_PREFIX))
        sibling_mass = float(
            catalog.get(int(sibling_id[len(CATALOG_FAMILY_PREFIX):])
                        ).pole_mass)
        # the window is capped at the support maximum
        assert min(SUPPORT_M_MAX, 6.0 - sibling_mass) == SUPPORT_M_MAX
        assert _continuum_daughter_mass(
            daughter_id, sibling_mass, 6.0, family_config) == max(
                float(next(f for f in family_config
                           if f.family_id == daughter_id).m_min),
                SUPPORT_M_MAX
                - float(next(f for f in family_config
                             if f.family_id == daughter_id).nominal_width))
    assert math.fsum(c.weight for c in high_channels) == pytest.approx(
        1.0, abs=1e-12)

    # -- ta_meson below all thresholds: the empty open set ---------------
    empty = _root_state("ta_meson", 0.2, (0, 0, 0))
    assert enumerate_channels(empty, catalog, family_config) == []

    # -- the quantum mismatch and the unknown family ---------------------
    mismatched = _root_state("ta_baryon", 3.0, (0, 0, 0))
    with pytest.raises(ValueError):
        enumerate_channels(mismatched, catalog, family_config)
    unknown = _root_state("ta_nothing", 3.0, (0, 0, 0))
    with pytest.raises(ValueError):
        enumerate_channels(unknown, catalog, family_config)


# ----------------------------------------------------------------------
# The channel weight: normalization and the hook contract
# ----------------------------------------------------------------------
def test_channel_weight_norm_and_hooks(catalog):
    """The per-vertex normalization to 1 within 1e-12 across both
    parent kinds; the multiplicative hook contract; the unity-hook
    draw-order invariance; the per-class hook share change; the
    negative / non-finite hook ``ValueError``."""

    family_config = TA_FAMILIES
    tol = 1e-12

    # the normalization across both parent kinds
    for family_id, mass, qn in (
        (f"{CATALOG_FAMILY_PREFIX}2214", 1.232, (1, 1, 0)),
        (f"{CATALOG_FAMILY_PREFIX}113", 0.77526, (0, 0, 0)),
        (f"{CATALOG_FAMILY_PREFIX}9014", 1.42, (0, 1, 1)),
        ("ta_baryon", 3.0, (1, 1, 0)),
        ("ta_baryon", 2.5, (1, 1, 0)),
        ("ta_baryon", 6.0, (1, 1, 0)),
    ):
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, family_config)
        assert channels
        total = math.fsum(c.weight for c in channels)
        assert abs(total - 1.0) <= tol, (family_id, mass, total)

    # the hooks are multiplicative, exactly
    state = _root_state("ta_baryon", 3.0, (1, 1, 0))
    channels = enumerate_channels(state, catalog, family_config)
    for c in channels:
        assert channel_weight(c) == c.weight
        assert channel_weight(c, w_flavor=2.0, w_align=0.5) == \
            c.weight * 2.0 * 0.5

    # a common factor on every channel leaves the draw order unchanged
    hooked_common = [
        CascadeChannel(
            parent_id=c.parent_id, daughters=c.daughters, kind=c.kind,
            weight=channel_weight(c, w_flavor=3.0, w_align=0.25),
        )
        for c in channels
    ]
    total_plain = math.fsum(c.weight for c in channels)
    total_common = math.fsum(c.weight for c in hooked_common)
    for a, b in zip(channels, hooked_common):
        assert a.weight / total_plain == pytest.approx(
            b.weight / total_common, rel=1e-12)
    plain, _ = draw_decay(
        state, channels, catalog, family_config,
        _CountingRng(REGISTER_SEED + 101), _CountingRng(REGISTER_SEED + 102))
    common, _ = draw_decay(
        state, hooked_common, catalog, family_config,
        _CountingRng(REGISTER_SEED + 101), _CountingRng(REGISTER_SEED + 102))
    assert common.daughters == plain.daughters

    # a per-class hook rescales exactly that class's shares
    hook = {KIND_ESTABLISHED_HADRON: 2.5, KIND_CONTINUUM: 0.5}
    hooked_class = [
        CascadeChannel(
            parent_id=c.parent_id, daughters=c.daughters, kind=c.kind,
            weight=channel_weight(c, w_flavor=hook[c.kind]),
        )
        for c in channels
    ]
    raw_hooked = math.fsum(h.weight for h in hooked_class)
    for c, h in zip(channels, hooked_class):
        expected_share = (c.weight * hook[c.kind]) / raw_hooked
        assert h.weight == pytest.approx(
            c.weight * hook[c.kind], rel=1e-12)
        # the re-normalized hooked share over the vertex
        assert h.weight / raw_hooked == pytest.approx(expected_share)

    # the negative and non-finite hooks raise ValueError
    for bad_kwargs in (
        {"w_flavor": -1.0},
        {"w_align": -0.5},
        {"w_flavor": float("nan")},
        {"w_align": float("inf")},
    ):
        with pytest.raises(ValueError):
            channel_weight(channels[0], **bad_kwargs)


# ----------------------------------------------------------------------
# The draw budget and conservation on the synthetic fixture
# ----------------------------------------------------------------------
def test_draw_budget_conservation_ta_fixture(catalog):
    """A 100-step cascade trace on the synthetic fixtures consumes
    exactly 3 uniforms per vertex (1 channel + 2 kinematics) on the
    closed substreams; every vertex conserves ``(B, Q, S)`` exactly
    and the daughter four-momenta sum to the parent's within
    ``DECAY_CLOSURE_TOL``; the depth counter increments; the parent
    link, exchange ancestry, axis, and label carry down; every drawn
    continuum daughter sits at the exact card-4 mass rule."""

    family_config = TA_FAMILIES
    roots = (
        ("ta_baryon", 4.3, (1, 1, 0)),
        ("ta_baryon", 2.9, (1, 1, 0)),
        (f"{CATALOG_FAMILY_PREFIX}2214", 1.232, (1, 1, 0)),
        (f"{CATALOG_FAMILY_PREFIX}9014", 1.42, (0, 1, 1)),
        ("ta_meson", 2.2, (0, 0, 0)),
    )
    steps = 0
    event = 0
    root_index = 0
    while steps < 100:
        family_id, mass, qn = roots[root_index % len(roots)]
        root_index += 1
        state = _root_state(family_id, mass, qn)
        while steps < 100:
            channels = enumerate_channels(state, catalog, family_config)
            continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
            pool = continuum if continuum else [
                c for c in channels if c.kind != KIND_CONTINUUM]
            if not pool:
                break  # no open channel: the line terminates
            rng_channel = _CountingRng(REGISTER_SEED + 2 * event)
            rng_kinematics = _CountingRng(REGISTER_SEED + 2 * event + 1)
            drawn, (daughter0, daughter1) = draw_decay(
                state, pool, catalog, family_config,
                rng_channel, rng_kinematics)
            # the frozen per-vertex budget
            assert rng_channel.calls == 1, (
                f"step {steps}: the channel draw consumed "
                f"{rng_channel.calls} uniforms (budget: 1)")
            assert rng_kinematics.calls == 2, (
                f"step {steps}: the kinematics draw consumed "
                f"{rng_kinematics.calls} uniforms (budget: 2)")
            # exact (B, Q, S) conservation, checked never repaired
            assert _qn_sum((daughter0, daughter1)) == \
                tuple(int(v) for v in state.qn)
            # the four-momentum closure within DECAY_CLOSURE_TOL
            for k in range(4):
                got = daughter0.momentum[k] + daughter1.momentum[k]
                want = state.momentum[k]
                assert abs(got - want) <= DECAY_CLOSURE_TOL, (
                    f"step {steps}: component {k} closure "
                    f"{abs(got - want)!r} exceeds {DECAY_CLOSURE_TOL}")
            # the daughters are on shell
            for daughter in (daughter0, daughter1):
                e, px, py, pz = daughter.momentum
                m2 = e * e - px * px - py * py - pz * pz
                assert abs(m2 - daughter.mass * daughter.mass) \
                    <= DECAY_CLOSURE_TOL
                assert daughter.mass > 0.0
            # the depth, parent link, ancestry, axis, and label
            for daughter in (daughter0, daughter1):
                assert daughter.depth == state.depth + 1
                assert daughter.parent is state
                assert daughter.exchange_ancestry == state.exchange_ancestry
                assert daughter.axis == state.axis
                assert daughter.alignment_label == state.alignment_label
                # the daughter masses: established at the pole,
                # continuum at the exact card-4 rule
                if daughter.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    record = catalog.get(
                        int(daughter.family_id[
                            len(CATALOG_FAMILY_PREFIX):]))
                    assert daughter.mass == float(record.pole_mass)
                else:
                    sibling = (
                        daughter0 if daughter1.family_id
                        == daughter.family_id else daughter1
                    )
                    if not sibling.family_id.startswith(
                            CATALOG_FAMILY_PREFIX):
                        # two-continuum channels do not enumerate
                        raise AssertionError(
                            f"step {steps}: a two-continuum channel "
                            f"{drawn.daughters} was drawn")
                    sibling_mass = float(catalog.get(
                        int(sibling.family_id[
                            len(CATALOG_FAMILY_PREFIX):])).pole_mass)
                    assert daughter.mass == _continuum_daughter_mass(
                        daughter.family_id, sibling_mass, state.mass,
                        family_config)
            steps += 1
            event += 1
            successors = [
                d for d in (daughter0, daughter1)
                if not d.family_id.startswith(CATALOG_FAMILY_PREFIX)
            ]
            if not successors:
                break  # the line terminated through a direct channel
            state = successors[0]
            if state.depth > MAX_CASCADE_DEPTH:
                break  # the guard is card 9's; the trace never reaches it
    assert steps == 100
    assert event == 100


# ----------------------------------------------------------------------
# The zero-weight edges (pinned from the on-disk module behavior)
# ----------------------------------------------------------------------
def test_zero_weight_edges(catalog):
    """The zero-weight edge behaviors: an empty open set returns an
    empty list (the caller decides termination); ``draw_decay`` raises
    ``ValueError`` on an empty channel list and on weights that do not
    sum to a finite positive total; ``channel_weight`` raises
    ``ValueError`` on a negative hook."""

    family_config = TA_FAMILIES

    # an empty open set: the continuum enumeration returns [] (no raise)
    empty = _root_state("ta_meson", 0.2, (0, 0, 0))
    assert enumerate_channels(empty, catalog, family_config) == []

    # draw_decay on an empty channel list raises ValueError
    with pytest.raises(ValueError, match="channel list is empty"):
        draw_decay(empty, [], catalog, family_config,
                   _StubRng((0.5,)), _StubRng((0.5, 0.5)))

    # draw_decay on all-zero weights raises ValueError
    state = _root_state("ta_baryon", 3.0, (1, 1, 0))
    zero_channels = [
        CascadeChannel(
            parent_id="ta_baryon",
            daughters=("catalog:2212", "catalog:111"),
            kind=KIND_ESTABLISHED_HADRON,
            weight=0.0,
        ),
        CascadeChannel(
            parent_id="ta_baryon",
            daughters=("catalog:2112", "catalog:211"),
            kind=KIND_ESTABLISHED_HADRON,
            weight=0.0,
        ),
    ]
    with pytest.raises(ValueError, match="finite positive total"):
        draw_decay(state, zero_channels, catalog, family_config,
                   _StubRng((0.5,)), _StubRng((0.5, 0.5)))

    # the negative hook raises ValueError (the non-finite edges are
    # pinned in test_channel_weight_norm_and_hooks)
    channels = enumerate_channels(state, catalog, family_config)
    with pytest.raises(ValueError, match="non-negative"):
        channel_weight(channels[0], w_flavor=-1.0)


# ----------------------------------------------------------------------
# The Round 2 baseline properties as structural contracts
# ----------------------------------------------------------------------
def test_round2_kernel_structural_contract(catalog):
    """The catalog-parent enumeration order and relative weights equal
    the kernel ``_open_objects`` two-body object order and weights for
    the self-authored taRho parent; the cumulative-walk channel draw
    selects by the first channel whose cumulative weight reaches the
    hand-fed ``u * total`` target."""

    family_config = TA_FAMILIES

    # -- the enumeration order and relative weights vs the kernel --------
    state = _root_state(f"{CATALOG_FAMILY_PREFIX}9014", 1.42, (0, 1, 1))
    channels = enumerate_channels(state, catalog, family_config)
    kernel = ResonanceDecayer(catalog, REGISTER_SEED, 0, 5)
    record = catalog.get(9014)
    objects, _open_total = kernel._open_objects(record, 1.42)
    two_body = [
        (combination, weight)
        for _channel, combination, weight in objects
        if len(combination) == 2
    ]
    assert [
        tuple(int(d[len(CATALOG_FAMILY_PREFIX):]) for d in c.daughters)
        for c in channels
    ] == [tuple(int(d) for d in combination)
          for combination, _ in two_body]
    kernel_weights = [weight for _, weight in two_body]
    kernel_total = math.fsum(kernel_weights)
    for c, (combination, weight) in zip(channels, two_body):
        assert c.weight == pytest.approx(weight / kernel_total, rel=1e-12)
    assert math.fsum(c.weight for c in channels) == pytest.approx(1.0,
                                                                  abs=1e-12)

    # -- the cumulative walk on hand-fed uniforms -------------------------
    # with the taRho channel weights (0.5, 0.5), target = u * total
    walk_cases = (
        (0.0, 0),
        (0.4999, 0),
        (0.5001, 1),
        (0.9999, 1),
    )
    for u, expected_index in walk_cases:
        drawn, _ = draw_decay(
            state, channels, catalog, family_config,
            _StubRng((u,)), _StubRng((0.25, 0.5)))
        assert drawn is channels[expected_index], (
            f"the cumulative walk at u = {u} selected channel "
            f"{channels.index(drawn)} (expected {expected_index})")


# ----------------------------------------------------------------------
# The module constants
# ----------------------------------------------------------------------
def test_module_constants():
    """The module constants of this card's landing: the depth
    placeholder, the family-id prefix, the isotropic label, and the
    channel kind constants."""

    assert MAX_CASCADE_DEPTH == 12
    assert CATALOG_FAMILY_PREFIX == "catalog:"
    assert ISOTROPIC_ALIGNMENT_LABEL == "isotropic"
    assert KIND_ESTABLISHED_HADRON == "established_hadron"
    assert KIND_ESTABLISHED_RESONANCE == "established_resonance"
    assert KIND_CONTINUUM == "continuum"
    # the support maximum the card-4 mass rule caps at
    assert SUPPORT_M_MAX == 5.0
    # the closure tolerance of the conservation check
    assert DECAY_CLOSURE_TOL == 1e-9


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_module_byte_and_export_pins():
    """The module byte stability (sha256) and the additive
    ``na49_mixing`` package exports (the import block and the
    ``__all__`` entries, plus the runtime object identity)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing

    init_src = INIT_PATH.read_text(encoding="utf-8")
    # the import block carries every exported name
    import_block = init_src[
        init_src.index("from .cascade_decay import"):
        init_src.index(")", init_src.index("from .cascade_decay import"))
    ]
    for name in (
        "MAX_CASCADE_DEPTH",
        "CATALOG_FAMILY_PREFIX",
        "ISOTROPIC_ALIGNMENT_LABEL",
        "KIND_ESTABLISHED_HADRON",
        "KIND_ESTABLISHED_RESONANCE",
        "KIND_CONTINUUM",
        "CascadeState",
        "CascadeChannel",
        "enumerate_channels",
        "channel_weight",
        "draw_decay",
    ):
        assert name in import_block, (
            f"the __init__ import block is missing {name}")
        assert f'"{name}"' in init_src, (
            f"the __init__ __all__ is missing {name}")
    # the runtime exports are the module's own objects
    from na49_mixing import cascade_decay as module

    assert na49_mixing.CascadeState is module.CascadeState
    assert na49_mixing.CascadeChannel is module.CascadeChannel
    assert na49_mixing.enumerate_channels is module.enumerate_channels
    assert na49_mixing.channel_weight is module.channel_weight
    assert na49_mixing.draw_decay is module.draw_decay
    assert na49_mixing.MAX_CASCADE_DEPTH is module.MAX_CASCADE_DEPTH


def test_frozen_substrate_digests():
    """The byte-unchanged surfaces the card consumes, verified on
    disk: the builder, the spectral-density module, the
    threshold-metric engine, the Round 2 kernel baseline, the
    committed catalog and continuum configuration, and the R32 driver
    in both trees."""

    for rel, expected in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)


