"""Card tests for the double-diffraction class.

Round 5, card 6 (``TASK-20260925-r5-double-diffraction``; plan section
22 task 30, line 1755; plan section 11.1 item 4, line 827).  The card's
pre-registered deterministic tests against the module
``na49_mixing/topology_double_diffraction.py``:

- ``test_conservation``: over a pinned 10,000-event trace, the final
  state carries exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum
  closure is within ``1e-9`` relative; every event passes card 2's
  ``audit_event`` with zero violations (two vacuum transfers, both
  excitation lines ``(1, 1, 0, 1)``).
- ``test_joint_closure``: ``M_1 + M_2 <= sqrt(s)`` within ``1e-12`` on
  the pinned trace; ``M_i^2`` in ``[m_p^2, (sqrt(s) - m_p)^2]`` and
  ``t_i`` in ``[t_min, 0]`` exactly; the rejection cap of card 4 is the
  only exit on the pinned pathological configuration.
- ``test_gap``: the central gap size on a pinned event equals the
  hand-computed rapidity difference within ``1e-12``; the gap is
  positive on the pinned trace (the two systems' decay products do not
  close the central gap at nominal parameters within ``1e-12``).
- ``test_no_intact_remnant``: the event record carries no intact proton
  remnant and no remnant kind draw (white-box field scan); both lines
  are excited baryonic systems (plan line 827).
- ``test_draw_budget``: a pinned 1,000-event trace consumes exactly 6
  uniforms on the closed ``exchange_kinematics`` label per accepted
  event (plus the rejection attempts on the same stream) and the frozen
  Round 4 per-vertex budget on ``decay_channel`` / ``decay_kinematics``;
  zero ``spectral_state`` uniforms (pinned RNG-state identity; the
  ``topology_selection`` draw belongs to the runner, card 9).
- ``test_frozen_cascade``: both excitation decays are the frozen
  simplified cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0
  GeV^-2, nominal width 0.3 GeV, W_flavor = 1.0); a white-box check
  that no module in this card references a re-fit of the alignment
  parameters (the R46 ``no_refit`` clause) or claims the separation of
  exchange recoil from decay transverse dynamics.
- ``test_layout_sync``: the frozen-substrate provenance pins (verified
  on disk) and the layout sync (the subprocess check from the monorepo
  root; ``logic_drift = 0``).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, one substream triple per event index on the
closed labels (``exchange_kinematics``, ``decay_channel``,
``decay_kinematics``); the ``spectral_state`` and ``topology_selection``
labels are never touched by the module (no remnant kind draw; the class
draw belongs to the runner, card 9).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules, and the committed cards 1-5
modules (all consumed read-only), numpy, and the standard library only;
no event generation driver, no external tables, no Monte Carlo run (the
runs R50-R55 exercise the class statistically).
"""

from __future__ import annotations

import collections
import hashlib
import inspect
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
from na49_mixing.topology_mass_laws import (
    REJECTION_CAP,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    draw_joint_dd,
    t_range,
)
from na49_mixing.topology_remnants import KERNEL_BOUNDARY, REMNANT_QN
from na49_mixing.topology_double_diffraction import (
    DD_CLASS_LABEL,
    DD_LINE_LABELS,
    NOMINAL_PARAMS,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    DDEvent,
    FinalHadron,
    _incoming_lines,
    central_gap,
    dd_kinematics,
    generate_dd,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_double_diffraction.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream triple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the card's 10,000-event conservation trace
#: and the 1,000-event draw-budget trace).
N_CONSERVATION = 10_000
N_BUDGET = 1_000

#: The pinned draw-budget totals of the 1,000-event budget trace
#: (master seed 20260925, replica 0, one shared stream per label in
#: event order): the joint-law attempts per event (the pinned attempt
#: histogram; each accepted event consumes 6 ``exchange_kinematics``
#: uniforms plus 4 uniforms per rejected attempt on the same stream),
#: and the frozen Round 4 per-vertex decay budget totals on the
#: ``decay_channel`` / ``decay_kinematics`` streams at this landing.
# re-pinned 2026-10-02 for the kernel fixes (the CM-mirrored target diffraction, the N pi threshold, the radiative channels)
DD_ATTEMPT_HISTOGRAM = {1: 829, 2: 136, 3: 31, 4: 4}
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
# re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
DD_CHANNEL_TOTAL = 8_014
DD_KINEMATICS_TOTAL = 18_310

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the five committed Round 5 modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger, card 3's
#: remnants, card 4's mass laws, card 5's single diffraction).
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
    "dev/mixing2006/na49_mixing/topology_single_diffraction.py":
        "bbfb63d651bb4fa12a9a75b464b696c6a7be0c88eddc27bd267714a9479e205a",
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

#: The white-box tokens of a remnant path (the no-remnant contract,
#: plan line 827): none may appear anywhere in the module source (the
#: class has no remnant selection, no remnant kind draw, and no
#: spectral_state stream).
NO_REMNANT_TOKENS = (
    "select_remnant", "RemnantSelection", "REMNANT_KINDS",
    "rng_spectral_state", "KIND_INTACT_PROTON",
)

#: The exact marginal window of the joint law (both lines at the intact
#: proton mass; the card's ``[m_p^2, (sqrt(s) - m_p)^2]`` limits at the
#: recorded ``SQRT_S = 17.270``, the card-4 recorded values).
M2_MIN = 0.880354927712736
M2_MAX = 266.7253293403128


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
    """The closed substream generators of one pinned event (the
    double-diffraction triple; no ``spectral_state`` stream exists in
    the class's budget)."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
    return tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("exchange_kinematics", "decay_channel",
                      "decay_kinematics"))


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


def _final_sum(event: DDEvent) -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the event's final state."""

    return (
        math.fsum(h.four_momentum[0] for h in event.final_state),
        math.fsum(h.four_momentum[1] for h in event.final_state),
        math.fsum(h.four_momentum[2] for h in event.final_state),
        math.fsum(h.four_momentum[3] for h in event.final_state),
    )


def _final_qn(event: DDEvent) -> tuple[int, int, int]:
    """The exact ``(B, Q, S)`` sum of the event's final state."""

    return (
        sum(h.qn[0] for h in event.final_state),
        sum(h.qn[1] for h in event.final_state),
        sum(h.qn[2] for h in event.final_state),
    )


def _exchange_records(event: DDEvent) -> tuple[Exchange, Exchange]:
    """The card-2 ledger exchange records of the event's two vacuum
    exchanges (the test-side conversion of the pair-tuple convention)."""

    records = []
    for name, transfer in event.exchanges:
        records.append(Exchange(
            name=name,
            transfer=(transfer[0], transfer[1], transfer[2], transfer[3])))
    return (records[0], records[1])


def _decay_vertices(event: DDEvent):
    """The card-2 ledger decay vertices of the event: each excited
    line as the parent and its final-state hadrons as the daughters."""

    vertices = []
    for label, line in ((PROJECTILE_LINE_LABEL, event.lines[0]),
                        (TARGET_LINE_LABEL, event.lines[1])):
        daughters = tuple(
            LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
            for h in event.final_state if h.line == label)
        if not daughters:
            raise ValueError(
                f"the event carries no {label!r}-side final state")
        vertices.append((line, daughters))
    return tuple(vertices)


def _audit(event: DDEvent):
    """Card 2's five-rule audit over the event's ledger records."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=_exchange_records(event),
        outgoing_lines=event.lines,
        central_source=None,
        decay_vertices=_decay_vertices(event),
    )


def _check_conservation(event: DDEvent) -> float:
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


def _check_mass_shell(event: DDEvent) -> float:
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
    """The pinned 10,000-event double-diffraction trace."""

    events = []
    for event_index in range(N_CONSERVATION):
        rngs = _event_rngs(event_index)
        events.append(generate_dd(NOMINAL_PARAMS, catalog, fams, *rngs))
    return events


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation(catalog, fams, pinned_trace):
    """Over the pinned 10,000-event trace the final state carries
    exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum closure is
    within ``1e-9`` relative; every event passes card 2's
    ``audit_event`` with zero violations (two vacuum transfers, both
    excitation lines ``(1, 1, 0, 1)``)."""

    assert len(pinned_trace) == N_CONSERVATION
    worst_closure = 0.0
    worst_shell = 0.0
    for event in pinned_trace:
        assert isinstance(event, DDEvent)
        assert event.class_label == DD_CLASS_LABEL
        worst_closure = max(worst_closure, _check_conservation(event))
        worst_shell = max(worst_shell, _check_mass_shell(event))
        report = _audit(event)
        assert not report.violations, report.violations
    assert worst_closure <= 1e-9, worst_closure
    assert worst_shell <= 1e-9, worst_shell


# ----------------------------------------------------------------------
# test_joint_closure
# ----------------------------------------------------------------------
def test_joint_closure(catalog, fams, pinned_trace):
    """``M_1 + M_2 <= sqrt(s)`` within ``1e-12`` on the pinned trace;
    ``M_i^2`` in ``[m_p^2, (sqrt(s) - m_p)^2]`` and ``t_i`` in
    ``[t_min, 0]`` exactly; the rejection cap of card 4 is the only
    exit on the pinned pathological configuration."""

    for event in pinned_trace:
        assert event.m1 + event.m2 <= SQRT_S + 1e-12, (event.m1, event.m2)
        assert M2_MIN <= event.m1 * event.m1 <= M2_MAX, (
            event.m1, M2_MIN, M2_MAX)
        assert M2_MIN <= event.m2 * event.m2 <= M2_MAX, (event.m2,)
        t_min_1 = t_range(event.m1 * event.m1, PROTON_MASS, SQRT_S)[0]
        t_min_2 = t_range(event.m2 * event.m2, PROTON_MASS, SQRT_S)[0]
        assert t_min_1 <= event.t1 <= 0.0, (event.t1, t_min_1)
        assert t_min_2 <= event.t2 <= 0.0, (event.t2, t_min_2)

    # the exact two-body solve closes the incoming sum within 1e-9
    # relative at a generic point
    system1, system2 = dd_kinematics(2.0, -1.0, 3.0, -0.5, SQRT_S)
    total = tuple(system1[i] + system2[i] for i in range(4))
    reference = _incoming_sum()
    relative = max(
        abs(total[0] - reference[0]) / reference[0],
        abs(total[1]) / reference[3],
        abs(total[2]) / reference[3],
        abs(total[3] - reference[3]) / reference[3])
    assert relative <= 1e-9, relative

    # the exact forward configuration at the intact two-body edge: both
    # systems at the proton mass with t_1 = t_2 = 0 resolve at the
    # elastic-forward configuration (the forward edge of the intact
    # system is exactly 0): system 1 carries the beam four-vector and
    # system 2 the target at rest, within 1e-12
    system1, system2 = dd_kinematics(
        PROTON_MASS, 0.0, PROTON_MASS, 0.0, SQRT_S)
    beam = (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2),
            0.0, 0.0, BEAM_MOMENTUM)
    target = (PROTON_MASS, 0.0, 0.0, 0.0)
    assert max(abs(a - b) for a, b in zip(system1, beam)) <= 1e-12
    assert max(abs(a - b) for a, b in zip(system2, target)) <= 1e-12

    # the drawn-domain refusals: positive transfers, non-positive
    # masses, and a pair closed beyond the joint-closure tolerance are
    # refused (checked, never repaired)
    with pytest.raises(ValueError):
        dd_kinematics(2.0, 1.0, 3.0, -0.5, SQRT_S)
    with pytest.raises(ValueError):
        dd_kinematics(-2.0, -1.0, 3.0, -0.5, SQRT_S)
    with pytest.raises(ValueError):
        dd_kinematics(17.0, -1.0, 1.0, -0.5, SQRT_S)

    # the rejection cap of card 4 is the only exit on the pinned
    # pathological configuration: each marginal on the exact
    # individual upper-limit window (m2_min = (sqrt(s) - m_p)^2, the
    # card's m2_max = s), epsilon = 0, b_sd = 0 -- the individually
    # legal windows are jointly impossible (M_1 + M_2 >= 2 (sqrt_s -
    # m_p) > sqrt_s), so the cap fires at exactly REJECTION_CAP
    # attempts with exactly 4 uniforms per rejected attempt (two
    # marginals; the closure is checked before the transfers), with
    # pinned RNG-state identity.  generate_dd consumes the same joint
    # law (the white-box source check: the module's generate_dd calls
    # card 4's draw_joint_dd on the exact windows).
    pathological_min = (SQRT_S - PROTON_MASS) ** 2
    pathological_max = SQRT_S * SQRT_S
    rng = np.random.default_rng(7)
    state0 = rng.bit_generator.state
    with pytest.raises(PhaseSpaceRejectionExceeded) as excinfo:
        draw_joint_dd(pathological_min, pathological_max,
                      pathological_min, pathological_max,
                      SQRT_S, MassLawParams(0.0, 0.0), rng)
    message = str(excinfo.value)
    assert "1000 attempts" in message
    assert "epsilon=0.0" in message and "b_sd=0.0" in message
    reference = np.random.default_rng(7)
    for _ in range(4 * REJECTION_CAP):
        reference.random()
    assert rng.bit_generator.state == reference.bit_generator.state
    assert rng.bit_generator.state != state0
    assert REJECTION_CAP == 1000

    # the generate_dd-level joint-law call is card 4's: the white-box
    # source check names the joint draw and the exact windows
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "draw_joint_dd(" in source
    assert "sd_limits(PROTON_MASS, SQRT_S)" in source
    assert REJECTION_CAP == 1000


# ----------------------------------------------------------------------
# test_gap
# ----------------------------------------------------------------------
def test_gap(catalog, fams, pinned_trace):
    """The central gap size on a pinned event equals the hand-computed
    rapidity difference within ``1e-12``; the gap is positive on the
    pinned trace (the two systems' decay products do not close the
    central gap at nominal parameters within ``1e-12``)."""

    # the hand computation on the pinned event index 0: the interval
    # between the nearest projectile-side and target-side final-state
    # hadrons (the cross-side pair distance minimum)
    event = pinned_trace[0]

    def _rapidity(four_momentum):
        e, _px, _py, pz = four_momentum
        return 0.5 * math.log((e + pz) / (e - pz))

    proj_ys = [_rapidity(h.four_momentum) for h in event.final_state
               if h.line == PROJECTILE_LINE_LABEL]
    targ_ys = [_rapidity(h.four_momentum) for h in event.final_state
               if h.line == TARGET_LINE_LABEL]
    hand = min(abs(y_p - y_t) for y_p in proj_ys for y_t in targ_ys)
    assert abs(central_gap(event) - hand) <= 1e-12
    assert event.gap_size == hand

    # positive on the whole pinned trace (the two systems' decay
    # products do not close the central gap at nominal parameters)
    for pinned in pinned_trace:
        assert pinned.gap_size > 1e-12, pinned.gap_size
        assert central_gap(pinned) == pinned.gap_size


# ----------------------------------------------------------------------
# test_no_intact_remnant
# ----------------------------------------------------------------------
def test_no_intact_remnant(catalog, fams, pinned_trace):
    """The event record carries no intact proton remnant and no remnant
    kind draw (white-box field scan); both lines are excited baryonic
    systems (plan line 827)."""

    # the white-box field scan: the record's fields carry no remnant
    # and no kind draw
    fields = tuple(DDEvent.__dataclass_fields__)
    for field in fields:
        assert "remnant" not in field
        assert "kind" not in field

    # the white-box module scan: no remnant selection machinery and no
    # spectral_state stream anywhere in the module's code
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in NO_REMNANT_TOKENS:
        assert token not in source, token

    # the generator's signature carries no spectral_state stream (the
    # class draws zero spectral_state uniforms)
    parameters = tuple(inspect.signature(generate_dd).parameters)
    assert "rng_spectral_state" not in parameters
    assert "rng_topology_selection" not in parameters

    # both lines are excited baryonic systems: the ledger lines inherit
    # the proton quantum numbers after the vacuum transfers and each
    # side's final state sums to the baryonic line
    for event in pinned_trace:
        for line in event.lines:
            assert line.qn == REMNANT_QN == (1, 1, 0, 1)
        assert {h.line for h in event.final_state} == DD_LINE_LABELS


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """A pinned 1,000-event trace consumes exactly 6 uniforms on the
    closed ``exchange_kinematics`` label per accepted event (plus the
    rejection attempts on the same stream) and the frozen Round 4
    per-vertex budget on ``decay_channel`` / ``decay_kinematics``; zero
    ``spectral_state`` uniforms (pinned RNG-state identity; the
    ``topology_selection`` draw belongs to the runner, card 9)."""

    for label in ("exchange_kinematics", "decay_channel",
                  "decay_kinematics", "spectral_state",
                  "topology_selection"):
        assert label in SUBSTREAM_LABELS

    # one shared stream per label across the pinned 1,000-event trace,
    # wrapped in the counting proxies (the class draws no
    # spectral_state and no topology_selection uniform: the streams are
    # created here but never passed)
    subs = derive_substreams(MASTER_SEED, 0, 0)
    counter_exchange = _CountingRng(np.random.default_rng(
        subs.stream_for("exchange_kinematics")))
    counter_channel = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_channel")))
    counter_kinematics = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_kinematics")))
    counter_spectral = _CountingRng(np.random.default_rng(
        subs.stream_for("spectral_state")))
    counter_topology = _CountingRng(np.random.default_rng(
        subs.stream_for("topology_selection")))

    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    per_event_attempts: list[int] = []
    for _event_index in range(N_BUDGET):
        before_channel = counter_channel.calls
        before_kinematics = counter_kinematics.calls
        before_exchange = counter_exchange.calls
        generate_dd(NOMINAL_PARAMS, catalog, fams,
                    counter_exchange, counter_channel, counter_kinematics)
        per_event_channel.append(counter_channel.calls - before_channel)
        per_event_kinematics.append(
            counter_kinematics.calls - before_kinematics)
        used = counter_exchange.calls - before_exchange
        # six uniforms per accepted event plus the rejection attempts
        # on the same stream: 4 uniforms per rejected attempt (the two
        # M^2 draws), never fewer than six
        assert used >= 6 and (used - 6) % 4 == 0, used
        per_event_attempts.append(1 + (used - 6) // 4)

    # the exchange_kinematics label: exactly six uniforms per accepted
    # event plus 4 per rejected attempt (card 4's joint-law budget)
    assert counter_exchange.calls == 6 * N_BUDGET + 4 * sum(
        attempts - 1 for attempts in per_event_attempts)
    # the pinned attempt histogram at this landing (the joint law's
    # rejection on the shared stream, master seed 20260925, replica 0)
    assert dict(sorted(collections.Counter(per_event_attempts).items())) \
        == dict(sorted(DD_ATTEMPT_HISTOGRAM.items()))
    # zero spectral_state uniforms and zero topology_selection uniforms
    assert counter_spectral.calls == 0
    assert counter_topology.calls == 0

    # the frozen Round 4 per-vertex budget on the decay labels (the
    # pinned per-event totals at this landing; a catalog handoff adds
    # the Round 2 kernel's own draws on the same streams)
    assert sum(per_event_channel) == DD_CHANNEL_TOTAL
    assert sum(per_event_kinematics) == DD_KINEMATICS_TOTAL
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert sum(1 for c in per_event_channel if c == 0) == 0

    # pinned RNG-state identity: the post-trace states equal fresh
    # streams advanced by exactly the counted draws
    subs = derive_substreams(MASTER_SEED, 0, 0)
    reference_exchange = np.random.default_rng(
        subs.stream_for("exchange_kinematics"))
    reference_channel = np.random.default_rng(
        subs.stream_for("decay_channel"))
    reference_kinematics = np.random.default_rng(
        subs.stream_for("decay_kinematics"))
    for _ in range(counter_exchange.calls):
        reference_exchange.random()
    for _ in range(counter_channel.calls):
        reference_channel.random()
    for _ in range(counter_kinematics.calls):
        reference_kinematics.random()
    assert (counter_exchange._source.bit_generator.state
            == reference_exchange.bit_generator.state)
    assert (counter_channel._source.bit_generator.state
            == reference_channel.bit_generator.state)
    assert (counter_kinematics._source.bit_generator.state
            == reference_kinematics.bit_generator.state)

    # the interface draws no spectral_state and no topology_selection
    # stream (the class draw belongs to the runner, card 9): the
    # untouched counters were never passed to the generator
    parameter_names = tuple(inspect.signature(generate_dd).parameters)
    assert "rng_spectral_state" not in parameter_names
    assert "rng_topology_selection" not in parameter_names
    assert counter_spectral.calls == 0
    assert counter_topology.calls == 0


# ----------------------------------------------------------------------
# test_frozen_cascade
# ----------------------------------------------------------------------
def test_frozen_cascade(catalog, fams):
    """Both excitation decays are the frozen simplified cascade
    (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, nominal width
    0.3 GeV, W_flavor = 1.0); a white-box check that no module in this
    card references a re-fit of the alignment parameters (the R46
    ``no_refit`` clause) or claims the separation of exchange recoil
    from decay transverse dynamics."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports the sealed Round 4 machinery (the constant-kappa
    # alignment kernels, the decay form factor, the termination
    # driver) and introduces no alignment re-fit and no separation
    # claim (the R46 clauses; the white-box token scan).  The sealed
    # R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2,
    # baryonic nominal width 0.3 GeV) is the R45 driver's frozen
    # configuration, pinned through the R45 driver digest in
    # FROZEN_SURFACES.
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

    # both excited lines decay through the baryonic_np continuum family
    # (the cascade line's family id)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'EXCITATION_FAMILY_ID: str = "baryonic_np"' in source
    assert "handoff_decay" in source and "draw_decay" in source
    for token in NO_REFIT_TOKENS:
        assert token not in source, token

    # one event's both decays close locally through the ledger's
    # local decay closure
    rngs = _event_rngs(0)
    event = generate_dd(NOMINAL_PARAMS, catalog, fams, *rngs)
    report = _audit(event)
    assert not report.violations


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The frozen-substrate provenance pins (verified on disk: the
    builder, the card-6 routing, the card-2 engine, the committed
    catalog, the continuum families configuration, the sealed R45/R46
    drivers in both helper trees, and the five committed Round 5
    modules consumed read-only; the M8-frozen ten-parameter set hash
    and the kernel boundary through the manifest; the activated
    baryonic ``(1, 1, 0)`` continuum family's recorded support ends)
    and the layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

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
# Closed-set guards (the card's record surface and nominal parameters)
# ----------------------------------------------------------------------
def test_closed_sets():
    """The frozen record surface, the class label, the closed
    line-label set, and the nominal parameters."""

    assert DD_CLASS_LABEL == "dd"
    assert isinstance(DD_LINE_LABELS, frozenset)
    assert DD_LINE_LABELS == frozenset({"projectile", "target"})
    assert getattr(DDEvent, "__dataclass_params__").frozen
    assert tuple(DDEvent.__dataclass_fields__) == (
        "class_label", "m1", "m2", "t1", "t2", "lines", "exchanges",
        "final_state", "gap_size",
        "exchange_configuration", "line_m_min", "slope_factor")  # Round 8 2026-10-04, default None (the vacuum exchanges);
    # line_m_min: 2026-10-06, None = the shared edges (line_thresholds off)
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    assert getattr(FinalHadron, "__dataclass_params__").frozen
    assert tuple(FinalHadron.__dataclass_fields__) == (
        "pdg_id", "four_momentum", "mass", "qn", "line")
    # the nominal parameters: the R50 fixed values (epsilon = 0, B_SD
    # the M8-frozen slope)
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
