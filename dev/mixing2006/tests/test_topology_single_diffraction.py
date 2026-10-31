"""Card tests for the projectile and target single-diffraction classes.

Round 5, card 5 (``TASK-20260925-r5-single-diffraction``; plan section
22 task 29, line 1754; plan section 11.1 items 2-3, lines 825-826).  The
card's pre-registered deterministic tests against the module
``na49_mixing/topology_single_diffraction.py``:

- ``test_conservation``: over a pinned 10,000-event trace the final
  state carries exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum
  closure is within ``1e-9`` relative; every event passes card 2's
  ``audit_event`` with zero violations (vacuum transfer, inheritance
  ``(1, 1, 0, 1)`` on the excitation line).
- ``test_remnant_legality``: on the pinned trace every remnant is an
  intact proton (mass == ``m_p`` exactly) or a committed-catalog
  ``(1, 1, 0)`` low-mass resonance (mass == the record pole mass,
  ``1e-12``); no neutron, no high-mass continuum (the nominal split).
- ``test_phase_space``: ``M^2`` in ``[m_p^2, (sqrt(s) - m_remnant)^2]``
  and ``t`` in ``[t_min(M, m_remnant), 0]`` exactly on the pinned trace;
  ``sd_kinematics`` at ``t = 0`` (the intact limit) returns the remnant
  four-vector equal to its beam four-vector within ``1e-12``.
- ``test_gap``: the gap size on a pinned event equals the hand-computed
  rapidity difference within ``1e-12``; the gap is positive on the
  pinned trace (the decay products do not close the gap at nominal
  parameters within ``1e-12``).
- ``test_mirror``: ``mirror_sd`` of a pinned projectile event is the
  exact beam-reverse (four-momenta mirrored in ``p_L``, identical
  quantum numbers, masses, and ``t``); the mirrored event passes the
  same conservation and audit checks; the target-side event generated
  directly at the same seed is the exact mirror of the projectile-side
  event.
- ``test_draw_budget``: a pinned 1,000-event trace consumes exactly the
  pre-registered counts on the closed labels -- 1 ``spectral_state`` and
  3 ``exchange_kinematics`` per event with pinned RNG-state identity,
  and the frozen Round 4 per-vertex budget on ``decay_channel`` /
  ``decay_kinematics`` (the pinned per-event vertex structure and the
  recorded totals at this landing; the ``topology_selection`` draw
  belongs to the runner, card 9).
- ``test_frozen_cascade``: the excitation decay is the frozen simplified
  cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic
  nominal width 0.3 GeV, W_flavor = 1.0); a white-box check that no
  module in this card references a re-fit of the alignment parameters
  (the R46 ``no_refit`` clause) or claims the separation of exchange
  recoil from decay transverse dynamics.
- ``test_layout_sync``: the test register (the card's frozen-substrate
  provenance pins, verified on disk) and the layout sync (the subprocess
  check from the monorepo root; ``logic_drift = 0``).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, one substream quadruple per event index on the
closed labels (``spectral_state``, ``exchange_kinematics``,
``decay_channel``, ``decay_kinematics``); the ``topology_selection``
label is never touched by the module (the runner's draw, card 9).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only), numpy,
and the standard library only; no event generation driver, no external
tables, no Monte Carlo run (the runs R50-R55 exercise the class
statistically).
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

from na49_mixing.exchange_ledger import (
    Exchange,
    LineState,
    audit_event,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
)
from na49_mixing.topology_mass_laws import sd_limits, t_range
from na49_mixing.topology_remnants import (
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KERNEL_BOUNDARY,
    REMNANT_QN,
    low_mass_records,
)
from na49_mixing.topology_single_diffraction import (
    NOMINAL_PARAMS,
    NOMINAL_SD_REMNANT_WEIGHTS,
    SD_CLASS_LABEL_PROJECTILE,
    SD_CLASS_LABEL_TARGET,
    SDSIDES,
    FinalHadron,
    SDEvent,
    _incoming_lines,
    gap_size,
    generate_sd,
    mirror_sd,
    sd_kinematics,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_single_diffraction.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quadruple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the card's 10,000-event conservation trace
#: and the 1,000-event draw-budget trace).
N_CONSERVATION = 10_000
N_BUDGET = 1_000

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the four Round 5 modules consumed read-only (card 1's
#: elastic reference, card 2's exchange ledger, card 3's remnants,
#: card 4's mass laws).
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
    "dev/mixing2006/na49_mixing/topology_remnants.py":
        "8c26ae419cbb0e7de3b36d95bb6e097c1854df7d2c69f8558b7f71bd4a0342ca",
    "dev/mixing2006/na49_mixing/topology_mass_laws.py":
        "6af0d8fbdf67204e316c42cae8e9962fb87251000b9e1364135045227224db0a",
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

#: The white-box tokens of a refit of the sealed R45 alignment point or
#: of a separation claim (the R46 clauses; none may appear anywhere in
#: the module source).
NO_REFIT_TOKENS = (
    "kappa_fit", "refit_kappa", "align_refit", "refit_alignment",
    "separation_claimed = True", "exchange_recoil_decay_separation",
)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


def _event_rngs(event_index: int):
    """The closed substream generators of one pinned event."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
    return tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("spectral_state", "exchange_kinematics",
                      "decay_channel", "decay_kinematics"))


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


def _incoming_sum() -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the pinned incoming pair."""

    lines = _incoming_lines()
    beam = lines[0].four_momentum
    target = lines[1].four_momentum
    if beam is None or target is None:
        raise ValueError("the pinned incoming lines carry no four-momenta")
    return (beam[0] + target[0], beam[1] + target[1],
            beam[2] + target[2], beam[3] + target[3])


def _final_sum(event: SDEvent) -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the event's final state."""

    return (
        math.fsum(h.four_momentum[0] for h in event.final_state),
        math.fsum(h.four_momentum[1] for h in event.final_state),
        math.fsum(h.four_momentum[2] for h in event.final_state),
        math.fsum(h.four_momentum[3] for h in event.final_state),
    )


def _final_qn(event: SDEvent) -> tuple[int, int, int]:
    """The exact ``(B, Q, S)`` sum of the event's final state."""

    return (
        sum(h.qn[0] for h in event.final_state),
        sum(h.qn[1] for h in event.final_state),
        sum(h.qn[2] for h in event.final_state),
    )


def _exchange_record(event: SDEvent) -> Exchange:
    """The card-2 ledger exchange record of the event's vacuum exchange."""

    name, transfer = event.exchange
    return Exchange(name=name,
                    transfer=(transfer[0], transfer[1], transfer[2],
                              transfer[3]))


def _excitation_parent(event: SDEvent) -> LineState:
    """The excitation line of the event (the outgoing line of the
    excited side)."""

    return event.lines[0] if event.side == "projectile" else event.lines[1]


def _decay_vertices(event: SDEvent):
    """The card-2 ledger decay vertex of the event: the excitation line
    as the parent and the excitation-side final-state hadrons as the
    daughters."""

    daughters = tuple(
        LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
        for h in event.final_state if h.line == "excitation")
    return ((_excitation_parent(event), daughters),)


def _audit(event: SDEvent):
    """Card 2's five-rule audit over the event's ledger records."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=(_exchange_record(event),),
        outgoing_lines=event.lines,
        central_source=None,
        decay_vertices=_decay_vertices(event),
    )


def _check_conservation(event: SDEvent) -> float:
    """The exact ``(B, Q, S)`` bookkeeping and the relative
    four-momentum closure of one event; returns the worst relative
    closure component."""

    assert _final_qn(event) == (2, 2, 0)
    total = _final_sum(event)
    reference = _incoming_sum()
    e_total = reference[0]
    p_beam = reference[3]
    return max(
        abs(total[0] - reference[0]) / e_total,
        abs(total[1]) / p_beam,
        abs(total[2]) / p_beam,
        abs(total[3] - reference[3]) / p_beam,
    )


def _check_mass_shell(event: SDEvent) -> float:
    """The worst relative mass-shell deviation of the final-state
    hadrons (the below-threshold excitation included)."""

    worst = 0.0
    for hadron in event.final_state:
        e, px, py, pz = hadron.four_momentum
        m2 = e * e - px * px - py * py - pz * pz
        # a massless final state (the photon of a radiative channel,
        # 2026-10-02) is on shell relative to its energy scale
        scale = max(hadron.mass * hadron.mass, 1e-9) if hadron.mass > 0.0 else max(float(hadron.four_momentum[0]) ** 2, 1e-9)
        worst = max(worst, abs(m2 - hadron.mass * hadron.mass) / scale)
    return worst


@pytest.fixture(scope="session")
def pinned_trace(catalog, fams):
    """The pinned 10,000-event projectile-side trace."""

    events = []
    for event_index in range(N_CONSERVATION):
        rngs = _event_rngs(event_index)
        events.append(generate_sd(
            "projectile", NOMINAL_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS,
            catalog, fams, *rngs))
    return events


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation(catalog, fams, pinned_trace):
    """Over the pinned 10,000-event trace the final state carries
    exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum closure is
    within ``1e-9`` relative; every event passes card 2's
    ``audit_event`` with zero violations (vacuum transfer, inheritance
    ``(1, 1, 0, 1)`` on the excitation line)."""

    assert len(pinned_trace) == N_CONSERVATION
    worst_closure = 0.0
    worst_shell = 0.0
    for event in pinned_trace:
        assert isinstance(event, SDEvent)
        assert event.class_label == SD_CLASS_LABEL_PROJECTILE
        worst_closure = max(worst_closure, _check_conservation(event))
        worst_shell = max(worst_shell, _check_mass_shell(event))
        report = _audit(event)
        assert not report.violations, report.violations
    assert worst_closure <= 1e-9, worst_closure
    assert worst_shell <= 1e-9, worst_shell


# ----------------------------------------------------------------------
# test_remnant_legality
# ----------------------------------------------------------------------
def test_remnant_legality(catalog, pinned_trace):
    """On the pinned trace every remnant is an intact proton (mass ==
    ``m_p`` exactly) or a committed-catalog ``(1, 1, 0)`` low-mass
    resonance (mass == the record pole mass, ``1e-12``); no neutron, no
    high-mass continuum (the nominal split)."""

    records = {record.name: record
               for record in low_mass_records(catalog)}
    n_intact = 0
    n_low = 0
    for event in pinned_trace:
        remnant = event.remnant
        assert remnant.qn == REMNANT_QN == (1, 1, 0, 1)
        if remnant.kind == KIND_INTACT_PROTON:
            n_intact += 1
            assert remnant.mass == PROTON_MASS
            assert remnant.pdg_id == 2212
            assert remnant.record_id is None
        else:
            assert remnant.kind == KIND_LOW_MASS_BARYONIC_RESONANCE
            n_low += 1
            assert remnant.record_id in records
            record = records[remnant.record_id]
            assert abs(remnant.mass - float(record.pole_mass)) <= 1e-12
            assert remnant.pdg_id == int(record.pdg_id)
    # the nominal split at the pinned trace: both kinds occur, nothing
    # else is ever drawn
    assert n_intact > 0 and n_low > 0
    assert n_intact + n_low == N_CONSERVATION
    neutron = "neutron_charge_exchange"
    high = "high_mass_baryonic_continuum"
    assert all(event.remnant.kind not in (neutron, high)
               for event in pinned_trace)


# ----------------------------------------------------------------------
# test_phase_space
# ----------------------------------------------------------------------
def test_phase_space(catalog, fams, pinned_trace):
    """``M^2`` in ``[m_p^2, (sqrt(s) - m_remnant)^2]`` and ``t`` in
    ``[t_min(M, m_remnant), 0]`` exactly on the pinned trace;
    ``sd_kinematics`` at ``t = 0`` (the intact limit) returns the
    remnant four-vector equal to its beam four-vector within
    ``1e-12``."""

    for event in pinned_trace:
        m2_min, m2_max = sd_limits(event.remnant.mass, SQRT_S)
        m2 = event.excitation_mass ** 2
        assert m2_min <= m2 <= m2_max, (event.excitation_mass, m2_min, m2_max)
        t_min = t_range(m2, event.remnant.mass, SQRT_S)[0]
        assert t_min <= event.t <= 0.0, (event.t, t_min)

    # the t = 0 limit at the intact configuration: the remnant
    # four-vector equals its beam four-vector within 1e-12
    excitation, remnant = sd_kinematics(
        PROTON_MASS, 0.0, PROTON_MASS, SQRT_S)
    beam = (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2),
            0.0, 0.0, BEAM_MOMENTUM)
    target = (PROTON_MASS, 0.0, 0.0, 0.0)
    assert max(abs(a - b) for a, b in zip(remnant, target)) <= 1e-12
    # and the excitation carries the beam four-vector there (the
    # elastic-forward configuration)
    assert max(abs(a - b) for a, b in zip(excitation, beam)) <= 1e-12

    # the solved pair closes the incoming sum within 1e-9 relative at a
    # generic point
    excitation, remnant = sd_kinematics(3.0, -1.0, PROTON_MASS, SQRT_S)
    total = tuple(excitation[i] + remnant[i] for i in range(4))
    reference = _incoming_sum()
    relative = max(
        abs(total[0] - reference[0]) / reference[0],
        abs(total[1]) / reference[3],
        abs(total[3] - reference[3]) / reference[3])
    assert relative <= 1e-9, relative


# ----------------------------------------------------------------------
# test_gap
# ----------------------------------------------------------------------
def test_gap(catalog, fams, pinned_trace):
    """The gap size on a pinned event equals the hand-computed rapidity
    difference within ``1e-12``; the gap is positive on the pinned
    trace (the decay products do not close the gap at nominal
    parameters within ``1e-12``)."""

    # the hand computation on the pinned event index 0
    event = pinned_trace[0]
    remnant_hadron = next(h for h in event.final_state if h.line == "remnant")
    y_remnant = 0.5 * math.log(
        (remnant_hadron.four_momentum[0] + remnant_hadron.four_momentum[3])
        / (remnant_hadron.four_momentum[0] - remnant_hadron.four_momentum[3]))
    hand = min(
        abs(0.5 * math.log(
            (h.four_momentum[0] + h.four_momentum[3])
            / (h.four_momentum[0] - h.four_momentum[3])) - y_remnant)
        for h in event.final_state if h.line == "excitation")
    assert abs(gap_size(event) - hand) <= 1e-12
    assert event.gap_size == hand

    # positive on the whole pinned trace
    for pinned in pinned_trace:
        assert pinned.gap_size > 1e-12, pinned.gap_size
        assert gap_size(pinned) == pinned.gap_size


# ----------------------------------------------------------------------
# test_mirror
# ----------------------------------------------------------------------
def test_mirror(catalog, fams, pinned_trace):
    """``mirror_sd`` of a pinned projectile event is the exact
    beam-reverse (four-momenta mirrored in ``p_L``, identical quantum
    numbers, masses, and ``t``); the mirrored event passes the same
    conservation and audit checks against the mirrored incoming sum;
    the target-side event generated directly at the same seed is the
    exact mirror of the projectile-side event."""

    pinned = pinned_trace[0]
    mirrored = mirror_sd(pinned)
    assert isinstance(mirrored, SDEvent)
    assert mirrored.side == "target"
    assert mirrored.class_label == SD_CLASS_LABEL_TARGET
    assert mirrored.remnant is pinned.remnant
    assert mirrored.excitation_mass == pinned.excitation_mass
    assert mirrored.t == pinned.t
    assert mirrored.exchange == pinned.exchange
    assert mirrored.gap_size == pinned.gap_size
    def _p4(line: LineState) -> tuple[float, float, float, float]:
        momentum = line.four_momentum
        if momentum is None:
            raise ValueError("the ledger line carries no four-momentum")
        return momentum

    for line_p, line_m in zip(pinned.lines, mirrored.lines):
        assert line_m.qn == line_p.qn
        assert _p4(line_m)[:3] == _p4(line_p)[:3]
        assert _p4(line_m)[3] == -_p4(line_p)[3]
    for h_p, h_m in zip(pinned.final_state, mirrored.final_state):
        assert h_m.qn == h_p.qn and h_m.mass == h_p.mass
        assert h_m.line == h_p.line
        assert h_m.four_momentum[:3] == h_p.four_momentum[:3]
        assert h_m.four_momentum[3] == -h_p.four_momentum[3]

    # the mirrored conservation and audit: the final state sums to the
    # mirrored incoming four-momentum and the ledger audit passes
    total = _final_sum(mirrored)
    reference = _incoming_sum()
    mirrored_reference = (reference[0], reference[1], reference[2],
                          -reference[3])
    relative = max(
        abs(total[k] - mirrored_reference[k]) / abs(mirrored_reference[k])
        for k in (0, 3))
    assert relative <= 1e-9, relative
    mirrored_audit = audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=(_exchange_record(mirrored),),
        outgoing_lines=mirrored.lines,
        central_source=None,
        decay_vertices=(),
    )
    assert not mirrored_audit.violations

    # the target class generated directly at the same seed: the same
    # remnant selection, mass, transfer, and gap (the shared substream
    # draws), the mirrored class label, and a passing ledger audit
    rngs = _event_rngs(0)
    target_event = generate_sd(
        "target", NOMINAL_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS,
        catalog, fams, *rngs)
    assert target_event.side == "target"
    assert target_event.class_label == SD_CLASS_LABEL_TARGET
    assert target_event.excitation_mass == pinned.excitation_mass
    assert target_event.t == pinned.t
    # the target class is the CM mirror of the projectile construction
    # (2026-10-02): the same rapidity interval, to the rounding of the
    # mirror's boost round trip
    assert target_event.gap_size == pytest.approx(pinned.gap_size, rel=1e-12)
    assert target_event.remnant.kind == pinned.remnant.kind
    assert not _audit(target_event).violations


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """A pinned 1,000-event trace consumes exactly the pre-registered
    counts on the closed labels: 1 ``spectral_state`` and 3
    ``exchange_kinematics`` per event (pinned RNG-state identity), and
    the frozen Round 4 per-vertex budget on ``decay_channel`` /
    ``decay_kinematics`` (the pinned per-event vertex structure and the
    recorded totals at this landing; the ``topology_selection`` draw
    belongs to the runner, card 9)."""

    for label in ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"):
        assert label in SUBSTREAM_LABELS
    assert "topology_selection" in SUBSTREAM_LABELS

    # one shared stream per label across the pinned 1,000-event trace,
    # wrapped in the counting proxies
    subs = derive_substreams(MASTER_SEED, 0, 0)
    counter_spectral = _CountingRng(np.random.default_rng(
        subs.stream_for("spectral_state")))
    counter_exchange = _CountingRng(np.random.default_rng(
        subs.stream_for("exchange_kinematics")))
    counter_channel = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_channel")))
    counter_kinematics = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_kinematics")))

    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    for _event_index in range(N_BUDGET):
        before_channel = counter_channel.calls
        before_kinematics = counter_kinematics.calls
        generate_sd("projectile", NOMINAL_PARAMS,
                    NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams,
                    counter_spectral, counter_exchange,
                    counter_channel, counter_kinematics)
        per_event_channel.append(counter_channel.calls - before_channel)
        per_event_kinematics.append(
            counter_kinematics.calls - before_kinematics)

    # the kinematic labels: exactly the pre-registered counts per event
    assert counter_spectral.calls == N_BUDGET
    assert counter_exchange.calls == 3 * N_BUDGET

    # the frozen Round 4 per-vertex budget: every event consumes at
    # least one vertex; the cascade's own vertices carry exactly one
    # channel uniform each, and the kinematics stream carries exactly
    # two uniforms per vertex plus the Round 2 kernel's own mass
    # realizations inside a catalog handoff (the kernel's documented
    # budget; card 9's handoff convention).  The pinned per-event
    # totals at this landing:
    # re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
    # re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
    # re-pinned 2026-10-02 for the N pi threshold (every excitation decays)
    assert sum(per_event_channel) == 5_217
    assert sum(per_event_kinematics) == 12_208
    # every event consumes a non-negative whole number of vertices; the
    # kinematics stream carries at least the two uniforms per cascade
    # vertex (a catalog handoff adds the Round 2 kernel's own mass
    # realizations on top; since the N pi threshold of 2026-10-02 every
    # excitation decays, so no event consumes zero)
    assert all(c >= 0 and k >= 0
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert sum(1 for c in per_event_channel if c == 0) == 0

    # pinned RNG-state identity: the post-trace states equal fresh
    # streams advanced by exactly the counted draws
    subs = derive_substreams(MASTER_SEED, 0, 0)
    reference_spectral = np.random.default_rng(
        subs.stream_for("spectral_state"))
    reference_exchange = np.random.default_rng(
        subs.stream_for("exchange_kinematics"))
    for _ in range(N_BUDGET):
        reference_spectral.random()
    for _ in range(3 * N_BUDGET):
        reference_exchange.random()
    assert (counter_spectral._source.bit_generator.state
            == reference_spectral.bit_generator.state)
    assert (counter_exchange._source.bit_generator.state
            == reference_exchange.bit_generator.state)


# ----------------------------------------------------------------------
# test_frozen_cascade
# ----------------------------------------------------------------------
def test_frozen_cascade(catalog, fams):
    """The excitation decay is the frozen simplified cascade (kappa_B =
    0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width
    0.3 GeV, W_flavor = 1.0); a white-box check that no module in this
    card references a re-fit of the alignment parameters (the R46
    ``no_refit`` clause) or claims the separation of exchange recoil
    from decay transverse dynamics."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports the sealed Round 4 machinery (the constant-kappa alignment
    # kernels, the decay form factor, the termination driver) and
    # introduces no alignment re-fit and no separation claim (the R46
    # clauses; the white-box token scan).  The sealed R45 point
    # (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic
    # nominal width 0.3 GeV) is the R45 driver's frozen configuration,
    # pinned through the R45 driver digest in FROZEN_SURFACES.
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    from na49_mixing.cascade_decay import channel_weight
    # the W_flavor hook default is exactly 1.0 (Round 6 owns the flavor
    # weights); the hook is multiplicative and consumes zero draws
    from na49_mixing.cascade_decay import CascadeChannel
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # the excited line decays through the baryonic_np continuum family
    # (the cascade line's family id)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'EXCITATION_FAMILY_ID: str = "baryonic_np"' in source
    assert "handoff_decay" in source and "draw_decay" in source
    for token in NO_REFIT_TOKENS:
        assert token not in source, token

    # one event's excitation decay conserves the excitation line's
    # quantum numbers through the ledger's local decay closure
    rngs = _event_rngs(0)
    event = generate_sd("projectile", NOMINAL_PARAMS,
                        NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *rngs)
    report = _audit(event)
    assert not report.violations


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk: the builder, the card-6 routing, the card-2
    engine, the committed catalog, the continuum families
    configuration, the sealed R45/R46 drivers in both helper trees, and
    the four Round 5 modules consumed read-only; the M8-frozen
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
    assert m8["regge.effective_transfer_slope"] == ELASTIC_TRANSFER_SLOPE

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
    assert baryonic_families[0]["m_max"] == 5.0

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# Closed-set guards (the card's side set and record surface)
# ----------------------------------------------------------------------
def test_closed_sets():
    """The closed ``SDSide`` set, the two class labels, the frozen
    record surfaces, and the nominal parameters."""

    assert isinstance(SDSIDES, frozenset)
    assert SDSIDES == frozenset({"projectile", "target"})
    assert SD_CLASS_LABEL_PROJECTILE == "sd_projectile"
    assert SD_CLASS_LABEL_TARGET == "sd_target"
    assert getattr(SDEvent, "__dataclass_params__").frozen
    assert tuple(SDEvent.__dataclass_fields__) == (
        "class_label", "side", "remnant", "excitation_mass", "t",
        "lines", "exchange", "final_state", "gap_size",
        "exchange_configuration", "excitation_m_min", "slope_factor", "tail", "ope")  # Round 8 2026-10-04, default None (the vacuum exchange);
    # tail: 2026-10-23 (R112), None without the tail component, True / False for a tail / coherence-region event;
    # ope: 2026-10-26, None without the one-pion-exchange component, True for a p + p -> n + X++ event;
    # excitation_m_min: 2026-10-06, None = the shared edge (line_thresholds off)
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    assert getattr(FinalHadron, "__dataclass_params__").frozen
    assert tuple(FinalHadron.__dataclass_fields__) == (
        "pdg_id", "four_momentum", "mass", "qn", "line")
    # the nominal parameters: the R50 fixed values (epsilon = 0, B_SD
    # the M8-frozen slope; the nominal remnant split)
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
    assert NOMINAL_SD_REMNANT_WEIGHTS == {
        KIND_INTACT_PROTON: 0.5,
        KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }
