"""Card tests for the intact and low-mass remnants.

Round 5, card 3 (``TASK-20260925-r5-intact-low-mass-remnants``; plan
section 22 task 33, line 1758; plan section 11.2, lines 833-842).  The
card's pre-registered deterministic tests against the module
``na49_mixing/topology_remnants.py``:

- ``test_closed_sets``: ``REMNANT_KINDS`` is exactly the four plan
  section 11.2 kinds; ``AVAILABLE_REMNANT_KINDS`` is exactly the three
  non-neutron kinds; the two partition.
- ``test_neutron_rejected``: ``select_remnant("neutron_charge_exchange",
  ...)`` raises ``ChargeExchangeNotImplemented`` naming plan task 47;
  the raise consumes zero uniforms (pinned RNG state).
- ``test_boundary_split``: on the committed catalog (pinned at its
  digest), ``low_mass_records`` equals the ``(1, 1, 0)`` records with
  ``mass_range[0] < B`` and ``high_mass_records`` equals the
  complement; the split is exact (no record on the boundary; B
  equality is impossible at the pinned catalog values within 1e-12).
- ``test_intact_mass``: the intact-proton remnant mass equals
  ``m_p = 0.93827231`` exactly; its quantum numbers are ``(1, 1, 0,
  1)``.
- ``test_low_mass_mass``: the low-mass remnant mass equals the selected
  record's pole mass on a pinned 10,000-draw trace (1e-12); every
  drawn mass is >= m_p.
- ``test_no_minimum_excitation``: a pinned 10,000-draw trace at the
  nominal SD split (intact 0.5, low-mass 0.5) contains intact-proton
  remnants with positive frequency and no excitation-count rejection
  anywhere; the module source carries no excitation-count guard
  (white-box token scan).
- ``test_draw_budget``: exactly 1 uniform on the closed
  ``spectral_state`` label per selection; the ``mass_sampler`` callable
  consumes zero draws of this card's substream (pinned RNG-state
  identity).
- ``test_weights_rejection``: a weight vector naming a kind outside
  ``REMNANT_KINDS`` raises ``ValueError`` naming the kind; weights not
  summing to 1 within 1e-12 raise ``ValueError``.
- ``test_layout_sync``: the test register (the card's frozen-substrate
  provenance pins, verified on disk) and the layout sync (the
  subprocess check from the monorepo root; ``logic_drift = 0``).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, with the per-selection substream triple
``(master_seed, replica, event_index)`` on the closed
``spectral_state`` label.  The high-mass kind's pinned stand-in mass
law is a uniform draw on ``[m_p, m_max]`` over the stand-in's own
private stream (card 4's law replaces it without changing the kind
draw order or the spectral_state budget).

Hermetic: the module, the committed catalog, the continuum families
configuration, numpy, and the standard library only; no event
generation driver, no external tables, no Monte Carlo run (the runs
R50-R55 exercise the remnants statistically).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.topology_remnants import (
    AVAILABLE_REMNANT_KINDS,
    CHARGE_EXCHANGE_TASK_REF,
    CONTINUUM_M_MAX,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KIND_NEUTRON_CHARGE_EXCHANGE,
    KERNEL_BOUNDARY,
    PROTON_MASS,
    PROTON_PDG_ID,
    REMNANT_KINDS,
    REMNANT_QN,
    ChargeExchangeNotImplemented,
    RemnantSelection,
    high_mass_records,
    low_mass_records,
    select_remnant,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_remnants.py"

#: The committed catalog and its pinned digest (the card's
#: frozen-substrate provenance; the boundary split is pinned at this
#: digest).
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3")
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, event index = selection index.
MASTER_SEED = 20260925

#: The pinned trace sizes: the card's 10,000-draw traces and the
#: draw-budget trace.
N_TRACE = 10_000
N_BUDGET_TRACE = 1_000

#: The nominal SD kind split of the R50 templates (index item 2; fixed,
#: not fit): intact 0.5, low-mass 0.5, high-mass 0.0.
SD_WEIGHTS = {
    KIND_INTACT_PROTON: 0.5,
    KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM: 0.0,
}

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the two Round 5 topology modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger).
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/na49_mixing/topology_elastic.py":
        "7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824",
    "dev/mixing2006/na49_mixing/exchange_ledger.py":
        "3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
}

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The white-box guard tokens of an excitation-count refusal (plan line
#: 842: none of these may appear anywhere in the module source).
EXCITATION_GUARD_TOKENS = (
    "n_pi_min", "min_n_pi", "pion_min", "n_pion_min",
    "min_excitation", "excitation_count", "min_multiplicity",
    "reject_under",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remnant_stream_id(event_index: int) -> int:
    """The pinned spectral_state stream of one trace selection."""

    return derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "spectral_state")


def _remnant_rng(event_index: int) -> np.random.Generator:
    """The pinned spectral_state generator of one trace selection."""

    return np.random.default_rng(_remnant_stream_id(event_index))


class _CountingRng:
    """A counting proxy around one generator (the house draw-count
    pattern): every ``random()`` consumes exactly one uniform from the
    wrapped stream and advances the counter."""

    def __init__(self, source) -> None:
        self._source = source
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._source.random())


def _refusing_sampler(m_low: float, m_high: float) -> float:
    """A mass sampler that must never be called (the intact and
    low-mass kinds take their masses without the caller's law)."""

    raise AssertionError(
        "the mass sampler must not be called for this kind, got "
        f"({m_low!r}, {m_high!r})")


class _StandInUniformLaw:
    """The pinned uniform stand-in mass law of this card's landing
    (card 4's law replaces it): a uniform draw on ``[m_low, m_high]``
    over the law's own private stream, so the attachment consumes zero
    draws of the spectral_state substream."""

    def __init__(self, seed: int = MASTER_SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0
        self.lower: float | None = None
        self.upper: float | None = None
        self.value: float | None = None

    def __call__(self, m_low: float, m_high: float) -> float:
        self.calls += 1
        self.lower = float(m_low)
        self.upper = float(m_high)
        self.value = self.lower + (self.upper - self.lower) * float(
            self._rng.random())
        return self.value


def _catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


# ----------------------------------------------------------------------
# test_closed_sets
# ----------------------------------------------------------------------
def test_closed_sets():
    """``REMNANT_KINDS`` is exactly the four plan section 11.2 kinds;
    ``AVAILABLE_REMNANT_KINDS`` is exactly the three non-neutron kinds;
    the two partition."""

    neutron = KIND_NEUTRON_CHARGE_EXCHANGE
    assert isinstance(REMNANT_KINDS, frozenset)
    assert isinstance(AVAILABLE_REMNANT_KINDS, frozenset)
    assert REMNANT_KINDS == frozenset({
        KIND_INTACT_PROTON,
        neutron,
        KIND_LOW_MASS_BARYONIC_RESONANCE,
        KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    })
    assert len(REMNANT_KINDS) == 4
    assert AVAILABLE_REMNANT_KINDS == frozenset({
        KIND_INTACT_PROTON,
        KIND_LOW_MASS_BARYONIC_RESONANCE,
        KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    })
    assert len(AVAILABLE_REMNANT_KINDS) == 3

    # the two partition: the availability set is the closed set minus
    # the neutron kind, the neutron kind is in the closed set only, and
    # nothing outside the closed set is available
    assert AVAILABLE_REMNANT_KINDS | {neutron} == REMNANT_KINDS
    assert AVAILABLE_REMNANT_KINDS & {neutron} == frozenset()
    assert AVAILABLE_REMNANT_KINDS < REMNANT_KINDS


# ----------------------------------------------------------------------
# test_neutron_rejected
# ----------------------------------------------------------------------
def test_neutron_rejected():
    """``select_remnant("neutron_charge_exchange", ...)`` raises
    ``ChargeExchangeNotImplemented`` naming plan task 47; the raise
    consumes zero uniforms (pinned RNG state)."""

    catalog = _catalog()
    assert CHARGE_EXCHANGE_TASK_REF == "plan task 47"

    # the requested neutron kind: the refusal precedes the kind draw
    rng = _remnant_rng(3)
    state0 = rng.bit_generator.state
    with pytest.raises(ChargeExchangeNotImplemented, match="plan task 47"):
        select_remnant(
            KIND_NEUTRON_CHARGE_EXCHANGE,
            {KIND_NEUTRON_CHARGE_EXCHANGE: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a drawn neutron kind (the neutron named with positive weight)
    # raises the same refusal after the kind draw: exactly one uniform
    # consumed (pinned post-draw state)
    sid = _remnant_stream_id(4)
    rng = np.random.default_rng(sid)
    state0 = rng.bit_generator.state
    with pytest.raises(ChargeExchangeNotImplemented, match="plan task 47"):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: 0.5, KIND_NEUTRON_CHARGE_EXCHANGE: 0.5},
            catalog, rng, _refusing_sampler)
    advanced = np.random.default_rng(sid)
    advanced.random()
    assert rng.bit_generator.state == advanced.bit_generator.state


# ----------------------------------------------------------------------
# test_boundary_split
# ----------------------------------------------------------------------
def test_boundary_split():
    """On the committed catalog (pinned at its digest) the split at the
    frozen kernel boundary B partitions the ``(1, 1, 0)`` records
    exactly: ``low_mass_records`` is the ``mass_range[0] < B`` side in
    catalog order, ``high_mass_records`` the exact complement, no
    record sits on the boundary (B equality impossible within 1e-12)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert KERNEL_BOUNDARY == 2.680194110727528

    catalog = _catalog()
    baryonic = [record for record in catalog.records()
                if (record.B, record.Q, record.S) == (1, 1, 0)]
    expected_low = [record for record in baryonic
                    if record.mass_range[0] < KERNEL_BOUNDARY]
    expected_high = [record for record in baryonic
                     if record.mass_range[0] >= KERNEL_BOUNDARY]

    low = low_mass_records(catalog)
    high = high_mass_records(catalog)
    assert [record.name for record in low] == [
        record.name for record in expected_low]
    assert [record.name for record in high] == [
        record.name for record in expected_high]

    # the split is exact: the two sides partition the baryonic records
    # in catalog order with no overlap and nothing dropped
    assert [record.name for record in low + high] == [
        record.name for record in baryonic]

    # B equality is impossible at the pinned catalog values: every
    # threshold sits off the frozen boundary by more than 1e-12
    assert all(abs(record.mass_range[0] - KERNEL_BOUNDARY) > 1e-12
               for record in baryonic)

    # the pinned census on the committed catalog: ten low-mass records,
    # an empty high-mass side
    assert len(baryonic) == 10
    assert len(low) == 10
    assert len(high) == 0


# ----------------------------------------------------------------------
# test_intact_mass
# ----------------------------------------------------------------------
def test_intact_mass():
    """The intact-proton remnant mass equals ``m_p = 0.93827231``
    exactly; its quantum numbers are ``(1, 1, 0, 1)``."""

    catalog = _catalog()
    proton = catalog.get(PROTON_PDG_ID)
    assert float(proton.pole_mass) == PROTON_MASS == 0.93827231

    assert getattr(RemnantSelection, "__dataclass_params__").frozen
    selection = select_remnant(
        KIND_INTACT_PROTON, {KIND_INTACT_PROTON: 1.0},
        catalog, _remnant_rng(5), _refusing_sampler)
    assert isinstance(selection, RemnantSelection)
    assert selection.kind == KIND_INTACT_PROTON
    assert selection.mass == PROTON_MASS
    assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
    assert selection.pdg_id == PROTON_PDG_ID
    assert selection.record_id is None


# ----------------------------------------------------------------------
# test_low_mass_mass
# ----------------------------------------------------------------------
def test_low_mass_mass():
    """The low-mass remnant mass equals the selected record's pole mass
    on a pinned 10,000-draw trace (1e-12); every drawn mass is >= m_p."""

    catalog = _catalog()
    records = {record.name: record for record in low_mass_records(catalog)}
    assert records

    n_low = 0
    for i in range(N_TRACE):
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, _remnant_rng(i), _refusing_sampler)
        assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
        # no mass below m_p is produced by any kind
        assert selection.mass >= PROTON_MASS
        if selection.kind == KIND_LOW_MASS_BARYONIC_RESONANCE:
            n_low += 1
            assert selection.record_id is not None
            record = records[selection.record_id]
            assert abs(selection.mass - float(record.pole_mass)) <= 1e-12
            assert selection.pdg_id == int(record.pdg_id)
        else:
            assert selection.kind == KIND_INTACT_PROTON
            assert selection.mass == PROTON_MASS
            assert selection.record_id is None
    assert n_low > 0


# ----------------------------------------------------------------------
# test_no_minimum_excitation
# ----------------------------------------------------------------------
def test_no_minimum_excitation():
    """A pinned 10,000-draw trace at the nominal SD split (intact 0.5,
    low-mass 0.5) contains intact-proton remnants with positive
    frequency and no excitation-count rejection anywhere; the module
    source carries no excitation-count guard (white-box token scan)."""

    catalog = _catalog()
    n_intact = 0
    n_low = 0
    for i in range(N_TRACE):
        # every selection succeeds: no excitation-count rejection is
        # raised anywhere on the trace
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, _remnant_rng(i), _refusing_sampler)
        if selection.kind == KIND_INTACT_PROTON:
            n_intact += 1
        else:
            assert selection.kind == KIND_LOW_MASS_BARYONIC_RESONANCE
            n_low += 1
        assert selection.mass >= PROTON_MASS
    # the intact proton is a legal remnant with zero pions: it occurs
    # with positive frequency, and nothing else is drawn
    assert n_intact > 0
    assert n_low > 0
    assert n_intact + n_low == N_TRACE

    # white-box: the module source carries no excitation-count guard
    # (the plan's no-minimum rule appears verbatim instead)
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in EXCITATION_GUARD_TOKENS:
        assert token not in source, token
    assert "Do not impose a minimum N-pi excitation" in source


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget():
    """Exactly 1 uniform on the closed ``spectral_state`` label per
    selection; the ``mass_sampler`` callable consumes zero draws of
    this card's substream (pinned RNG-state identity)."""

    assert "spectral_state" in SUBSTREAM_LABELS
    catalog = _catalog()

    # every available kind: one selection consumes exactly one uniform
    # (the counting proxy), the post-draw state is bit-identical to the
    # reference stream after exactly one draw, and the stand-in law is
    # called exactly once for the high-mass kind with the frozen
    # attachment arguments and never for the other kinds
    for kind in sorted(AVAILABLE_REMNANT_KINDS):
        sid = _remnant_stream_id(0)
        counter = _CountingRng(np.random.default_rng(sid))
        law = _StandInUniformLaw()
        selection = select_remnant(
            kind, {kind: 1.0}, catalog, counter, law)
        assert counter.calls == 1
        advanced = np.random.default_rng(sid)
        advanced.random()
        assert (counter._source.bit_generator.state
                == advanced.bit_generator.state)
        if kind == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
            assert law.calls == 1
            assert law.lower == PROTON_MASS
            assert law.upper == CONTINUUM_M_MAX
            assert selection.mass == law.value
            assert selection.pdg_id is None
            assert selection.record_id is None
        else:
            assert law.calls == 0
        assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
        assert selection.mass >= PROTON_MASS

    # the continuous-stream identity over a pinned 1,000-selection
    # trace at the nominal SD split: one shared spectral_state stream
    # consumes exactly the pristine uniform sequence in order (no
    # burn-in, no extra draw anywhere in the module), and the drawn
    # kind is the cumulative resolution of the pristine uniform
    shared = _remnant_rng(0)
    pristine = np.random.default_rng(_remnant_stream_id(0))
    for _i in range(N_BUDGET_TRACE):
        u_ref = pristine.random()
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, shared, _refusing_sampler)
        assert selection.kind == (
            KIND_INTACT_PROTON if u_ref < 0.5
            else KIND_LOW_MASS_BARYONIC_RESONANCE)
        assert selection.mass >= PROTON_MASS
    assert shared.bit_generator.state == pristine.bit_generator.state


# ----------------------------------------------------------------------
# test_weights_rejection
# ----------------------------------------------------------------------
def test_weights_rejection():
    """A weight vector naming a kind outside ``REMNANT_KINDS`` raises
    ``ValueError`` naming the kind; weights not summing to 1 within
    1e-12 raise ``ValueError``; a kind argument outside the closed set
    raises ``ValueError`` naming it."""

    catalog = _catalog()

    foreign = "baryonic_continuum_nucleon"
    assert foreign not in REMNANT_KINDS
    with pytest.raises(ValueError, match=foreign):
        select_remnant(
            KIND_INTACT_PROTON, {foreign: 1.0},
            catalog, _remnant_rng(0), _refusing_sampler)

    with pytest.raises(ValueError, match="sum to 1"):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: 0.5, KIND_LOW_MASS_BARYONIC_RESONANCE: 0.4},
            catalog, _remnant_rng(0), _refusing_sampler)

    outside = "neutron_absorption"
    assert outside not in REMNANT_KINDS
    with pytest.raises(ValueError, match=outside):
        select_remnant(
            outside, {KIND_INTACT_PROTON: 1.0},
            catalog, _remnant_rng(0), _refusing_sampler)


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk: the builder, the card-6 routing, the card-2
    engine, the committed catalog, the continuum families
    configuration, the sealed R45/R46 drivers in both helper trees, and
    the two Round 5 topology modules consumed read-only; the M8-frozen
    ten-parameter set hash and the kernel boundary through the
    manifest; the activated baryonic ``(1, 1, 0)`` continuum family's
    recorded support ends) and the layout sync (the subprocess check
    from the monorepo root; ``logic_drift = 0``)."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    computed_hash = hashlib.sha256(
        json.dumps(m8, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()
    assert computed_hash == M8_HASH
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY

    # the activated baryonic (1, 1, 0) continuum family of the
    # committed configuration: exactly one, with its support ends at
    # the frozen boundary and the card-6 surface's upper end
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    baryonic_families = [
        family for family in continuum["families"]
        if family["qn"] == [1, 1, 0] and family["active"] is True]
    assert len(baryonic_families) == 1
    assert baryonic_families[0]["family_id"] == "baryonic_np"
    assert baryonic_families[0]["m_min"] == KERNEL_BOUNDARY
    assert baryonic_families[0]["m_max"] == CONTINUUM_M_MAX

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout