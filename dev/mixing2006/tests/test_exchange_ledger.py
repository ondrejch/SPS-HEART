"""Card tests for the explicit exchange quantum-number ledger.

Round 5, card 2 (``TASK-20260925-r5-exchange-quantum-number-ledger``;
plan section 22 task 35, line 1760; plan section 11.4, lines 871-881).
The card's pre-registered deterministic tests against the module
``na49_mixing/exchange_ledger.py``:

- ``test_vacuum_zero``: the vacuum exchange transfer is ``(0, 0, 0,
  0)`` exactly; ``apply_transfer`` on the proton line returns the
  proton line unchanged, bit-identical fields.
- ``test_inheritance``: a pinned transfer ``(0, -1, 0, -1)`` applied
  to the proton line yields ``(1, 0, 0, 0)`` (the neutron ``(B, Q,
  S)`` on the arithmetic fixture; the doubled-I3 bookkeeping follows
  the declared transfer), and the same transfer raises
  ``NonSingletTransferError`` on the Round 5 closed set (names the
  exchange).
- ``test_closed_set``: ``ALLOWED_EXCHANGES == {VACUUM}`` in Round 5;
  ``register_exchange`` raises on any exchange, naming it.
- ``test_decay_closure``: ``close_decay`` returns True on the pinned
  10,000-vertex fixture (the committed catalog channels of the ``(1,
  1, 0)`` and ``(0, 0, 0)`` parents, pinned at the committed catalog
  digest) and False on each planted violation (one daughter charge off
  by 1, one daughter strangeness off by 1, one daughter baryon number
  off by 1).
- ``test_central_neutrality``: a central source with ``(B, Q, S) =
  (0, 0, 0)`` passes rule (e); each of the three planted nonzero
  central quantum numbers produces exactly one named violation.
- ``test_audit_compliant``: a pinned compliant event (two proton
  lines, two vacuum exchanges, two outgoing excitations ``(1, 1, 0,
  1)``, a neutral central source, ten closed decay vertices) returns
  zero violations.
- ``test_audit_detection``: each of the five violation classes,
  planted one at a time on the compliant event, is detected with the
  named rule (100% detection on the pinned fixture).
- ``test_zero_draws``: the ledger machinery consumes zero random
  numbers (pinned RNG-state identity over a full audit trace; the
  structural no-RNG-parameter and no-numpy pins).
- ``test_layout_sync``: the test register (the card's frozen-substrate
  provenance pins, verified on disk) and the layout sync (the
  subprocess check from the monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, numpy, and the standard
library only; no event generation driver, no external tables, no Monte
Carlo run (the ledger consumes zero random numbers; runs R50-R55
exercise the ledger, and R54 asserts zero violations across at least
one million events).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.exchange_ledger import (
    ALLOWED_EXCHANGES,
    Exchange,
    LedgerAuditReport,
    LineState,
    NonSingletTransferError,
    VACUUM,
    ZERO_TRANSFER,
    apply_transfer,
    apply_transfer_qn,
    audit_event,
    close_decay,
    register_exchange,
)
from na49_mixing.hadron_catalog import HadronCatalog

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "exchange_ledger.py"

#: The committed catalog and its pinned digest (the card's
#: frozen-substrate provenance; the decay-closure fixture is pinned at
#: this digest).
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: RNG-state identity trace.
MASTER_SEED = 20260925

#: The pinned closure-fixture size (the card's 10,000-vertex fixture;
#: the 87 committed base vertices of the two parent classes cycle).
N_CLOSURE_VERTICES = 10_000

#: A pinned four-momentum fixture (the optional LineState field is
#: carried through apply_transfer unchanged; the ledger performs no
#: kinematics).
FOUR_MOMENTUM_FIXTURE = (2.0, 0.1, -0.2, 1.9)

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, and the sealed Round 4 R45/R46 drivers in both
#: helper trees.
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
#: separators=(",", ":"))``; the card's provenance pin) and the kernel
#: boundary B.
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
KERNEL_BOUNDARY = 2.680194110727528


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proton_line(four_momentum=None) -> LineState:
    """The proton line ``(B, Q, S, 2I3) = (1, 1, 0, 1)`` (the card's
    pre-registered instantiation)."""

    return LineState(1, 1, 0, 1, four_momentum=four_momentum)


def _compliant_event():
    """The pinned compliant event: two proton lines, two vacuum
    exchanges, two outgoing excitations ``(1, 1, 0, 1)`` (the intact
    inheritance of the zero transfer), a neutral central source, and
    ten closed decay vertices (the pinned two-body channels pi+ pi-,
    pi0 pi0, and K+ K- of the neutral central system, cycled 4 + 3 +
    3)."""

    incoming = (_proton_line(), _proton_line())
    exchanges = (VACUUM, VACUUM)
    outgoing = (LineState(1, 1, 0, 1), LineState(1, 1, 0, 1))
    central = LineState(0, 0, 0, 0)
    pi_pair = (LineState(0, 1, 0, 2), LineState(0, -1, 0, -2))
    pi0_pair = (LineState(0, 0, 0, 0), LineState(0, 0, 0, 0))
    k_pair = (LineState(0, 1, 1, 1), LineState(0, -1, -1, -1))
    closed = ((central, pi_pair), (central, pi0_pair), (central, k_pair))
    decay_vertices = tuple(closed[i % 3] for i in range(10))
    return incoming, exchanges, outgoing, central, decay_vertices


def _catalog_vertices():
    """The committed catalog decay vertices of the ``(1, 1, 0)`` and
    ``(0, 0, 0)`` parents, in committed record/channel order; each
    daughter's ``(B, Q, S, 2I3)`` comes from the committed catalog
    record (the doubled ``isospin3`` field)."""

    catalog = HadronCatalog.from_json(CATALOG_PATH)
    records = {record.pdg_id: record for record in catalog.records()}
    vertices = []
    for record in catalog.records():
        if (record.B, record.Q, record.S) not in ((1, 1, 0), (0, 0, 0)):
            continue
        parent = LineState(record.B, record.Q, record.S, record.isospin3)
        for channel in (record.decay_channels or ()):
            # the photon of a radiative channel (2026-10-02) is no record:
            # it carries (0, 0, 0, 0)
            daughters = tuple(
                LineState(0, 0, 0, 0) if int(pdg) == 22 else
                LineState(records[pdg].B, records[pdg].Q, records[pdg].S,
                          records[pdg].isospin3)
                for pdg in channel.daughters)
            vertices.append((parent, daughters))
    return vertices


def _planted_vertex(vertices, component: str):
    """The first pinned vertex (in committed order) with a nonzero
    *component* on some daughter, and that daughter's index: the
    deterministic host of the planted one-unit violation."""

    for parent, daughters in vertices:
        for index, daughter in enumerate(daughters):
            if getattr(daughter, component) != 0:
                return parent, daughters, index
    raise AssertionError(f"no daughter with nonzero {component}")


def _off_by_one(state: LineState, component: str) -> LineState:
    """The state with one component lowered by one (the planted
    violation)."""

    fields = {"B": state.B, "Q": state.Q, "S": state.S,
              "I3x2": state.I3x2}
    fields[component] -= 1
    return LineState(fields["B"], fields["Q"], fields["S"], fields["I3x2"],
                     four_momentum=state.four_momentum)


def _global_rng_snapshot():
    """A hashable snapshot of the numpy global generator state (the
    legacy ``RandomState`` state tuple; the MT state array is hashed
    through its bytes)."""

    return tuple(
        part.tobytes() if isinstance(part, np.ndarray) else part
        for part in np.random.get_state())


# ----------------------------------------------------------------------
# test_vacuum_zero
# ----------------------------------------------------------------------
def test_vacuum_zero():
    """The vacuum exchange transfer is ``(0, 0, 0, 0)`` exactly and
    ``apply_transfer`` on the proton line returns the proton line
    unchanged, bit-identical fields."""

    assert VACUUM.name == "vacuum"
    assert VACUUM.transfer == (0, 0, 0, 0)
    assert ZERO_TRANSFER == (0, 0, 0, 0)
    assert VACUUM in ALLOWED_EXCHANGES

    # the frozen records and their exact field sets
    assert getattr(LineState, "__dataclass_params__").frozen
    assert getattr(Exchange, "__dataclass_params__").frozen
    assert getattr(LedgerAuditReport, "__dataclass_params__").frozen
    assert [f.name for f in dataclasses.fields(LineState)] == [
        "B", "Q", "S", "I3x2", "four_momentum"]
    assert [f.name for f in dataclasses.fields(Exchange)] == [
        "name", "transfer"]
    assert [f.name for f in dataclasses.fields(LedgerAuditReport)] == [
        "violations"]

    # the record-convention tie to card 1 (consumed read-only): the
    # same doubled-I3 proton quadruple
    from na49_mixing.topology_elastic import PROTON_QN
    assert PROTON_QN == (1, 1, 0, 1)

    line = _proton_line(four_momentum=FOUR_MOMENTUM_FIXTURE)
    result = apply_transfer(line, VACUUM)
    assert result == line
    assert (result.B, result.Q, result.S, result.I3x2) == (1, 1, 0, 1)
    assert result.four_momentum == FOUR_MOMENTUM_FIXTURE
    assert result.qn == line.qn == (1, 1, 0, 1)


# ----------------------------------------------------------------------
# test_inheritance
# ----------------------------------------------------------------------
def test_inheritance():
    """A pinned transfer ``(0, -1, 0, -1)`` applied to the proton line
    yields ``(1, 0, 0, 0)`` (the neutron ``(B, Q, S)`` on the
    arithmetic fixture), and the same transfer raises
    ``NonSingletTransferError`` on the Round 5 closed set (names the
    exchange)."""

    proton = _proton_line()
    transfer = (0, -1, 0, -1)

    # the arithmetic fixture: the exact integer map (outgoing =
    # incoming after the declared transfer, no tolerance)
    assert apply_transfer_qn(proton.qn, transfer) == (1, 0, 0, 0)
    assert apply_transfer_qn((1, 1, 0, 1), (0, -1, 0, -1)) == (1, 0, 0, 0)
    # the vacuum transfer is the exact identity on the same map
    assert apply_transfer_qn(proton.qn, ZERO_TRANSFER) == (1, 1, 0, 1)

    # the same transfer on the Round 5 closed set: the exchange record
    # is outside {VACUUM}, so apply_transfer raises, naming the exchange
    exchange = Exchange("charge-exchange", transfer)
    with pytest.raises(NonSingletTransferError, match="charge-exchange"):
        apply_transfer(proton, exchange)


# ----------------------------------------------------------------------
# test_closed_set
# ----------------------------------------------------------------------
def test_closed_set():
    """``ALLOWED_EXCHANGES == {VACUUM}`` in Round 5; register_exchange
    raises on any exchange, naming it."""

    assert isinstance(ALLOWED_EXCHANGES, frozenset)
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})
    assert len(ALLOWED_EXCHANGES) == 1

    # register_exchange raises on any exchange, naming it (the set is
    # closed at this round; charge exchange is plan task 47 and the
    # secondary singlets are plan tasks 45-46, Round 8)
    for candidate in (
            Exchange("charge-exchange", (0, -1, 0, -1)),
            Exchange("secondary-singlet", (0, 0, 0, 0)),
            VACUUM,
    ):
        with pytest.raises(NonSingletTransferError,
                           match=re.escape(candidate.name)):
            register_exchange(candidate)

    # the closed set is unchanged after the failed registrations
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})


# ----------------------------------------------------------------------
# test_decay_closure
# ----------------------------------------------------------------------
def test_decay_closure():
    """``close_decay`` returns True on the pinned 10,000-vertex fixture
    (the committed catalog channels of the ``(1, 1, 0)`` and ``(0, 0,
    0)`` parents, pinned at the committed catalog digest) and False on
    each planted violation (one daughter charge off by 1, one daughter
    strangeness off by 1, one daughter baryon number off by 1)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    base = _catalog_vertices()
    # 87 + the 13 channels of the radiative revision (2026-10-02) - the seven of the
    # eta-prime at 377 that the orphan-record revision removed (2026-10-29)
    assert len(base) == 100 - 7
    n_baryonic = sum(1 for parent, _ in base if parent.B == 1)
    assert n_baryonic == 50
    assert len(base) - n_baryonic == 50 - 7

    # the pinned 10,000-vertex fixture: the committed channels cycle
    vertices = [base[i % len(base)] for i in range(N_CLOSURE_VERTICES)]
    for parent, daughters in vertices:
        assert close_decay(parent, daughters) is True

    # the three planted violations, each on a pinned committed vertex
    for component in ("Q", "S", "B"):
        parent, daughters, index = _planted_vertex(base, component)
        bad = (daughters[:index]
               + (_off_by_one(daughters[index], component),)
               + daughters[index + 1:])
        assert close_decay(parent, bad) is False
        # the unmutated vertex closes (the violation is the one unit)
        assert close_decay(parent, daughters) is True


# ----------------------------------------------------------------------
# test_central_neutrality
# ----------------------------------------------------------------------
def test_central_neutrality():
    """A central source with ``(B, Q, S) = (0, 0, 0)`` passes rule (e);
    each of the three planted nonzero central quantum numbers produces
    exactly one named violation."""

    incoming, exchanges, outgoing, _central, decay_vertices = (
        _compliant_event())

    report = audit_event(incoming, exchanges, outgoing,
                         LineState(0, 0, 0, 0), decay_vertices)
    assert report.violations == ()

    for name, bad_source in (
            ("B", LineState(1, 0, 0, 0)),
            ("Q", LineState(0, 1, 0, 0)),
            ("S", LineState(0, 0, 1, 0)),
    ):
        report = audit_event(incoming, exchanges, outgoing, bad_source,
                             decay_vertices)
        assert len(report.violations) == 1, (name, report.violations)
        violation = report.violations[0]
        assert violation.startswith("(e) central-source neutrality")
        assert f"carries {name} = 1" in violation


# ----------------------------------------------------------------------
# test_audit_compliant
# ----------------------------------------------------------------------
def test_audit_compliant():
    """The pinned compliant event (two proton lines, two vacuum
    exchanges, two outgoing excitations ``(1, 1, 0, 1)``, a neutral
    central source, ten closed decay vertices) returns zero
    violations."""

    incoming, exchanges, outgoing, central, decay_vertices = (
        _compliant_event())
    report = audit_event(incoming, exchanges, outgoing, central,
                         decay_vertices)
    assert isinstance(report, LedgerAuditReport)
    assert report.violations == ()


# ----------------------------------------------------------------------
# test_audit_detection
# ----------------------------------------------------------------------
def test_audit_detection():
    """Each of the five violation classes, planted one at a time on the
    compliant event, is detected with the named rule (100% detection on
    the pinned fixture; the exact violation tuples are pinned)."""

    incoming, exchanges, outgoing, central, decay_vertices = (
        _compliant_event())

    # (a) an exchange record outside the closed set, appended beyond
    # the two outgoing lines' pairing indices (rule (c) is untouched)
    planted = exchanges + (Exchange("charge-exchange", (0, -1, 0, -1)),)
    report = audit_event(incoming, planted, outgoing, central,
                         decay_vertices)
    assert report.violations == (
        "(a) closed-set membership: exchange 2 ('charge-exchange', "
        "transfer (0, -1, 0, -1)) is outside the closed allowed-exchange "
        "set {vacuum}",
    )

    # (b) a vacuum-named exchange with a nonzero transfer (also outside
    # the closed set; both rules name the record)
    planted = exchanges + (Exchange("vacuum", (0, 1, 0, 0)),)
    report = audit_event(incoming, planted, outgoing, central,
                         decay_vertices)
    assert report.violations == (
        "(a) closed-set membership: exchange 2 ('vacuum', transfer "
        "(0, 1, 0, 0)) is outside the closed allowed-exchange set "
        "{vacuum}",
        "(b) vacuum zero transfer: exchange 2 ('vacuum') carries the "
        "nonzero transfer (0, 1, 0, 0)",
    )

    # (c) an outgoing excitation that does not inherit the line quantum
    # numbers after the declared transfer
    bad_outgoing = (outgoing[0], LineState(1, 0, 0, 1))
    report = audit_event(incoming, exchanges, bad_outgoing, central,
                         decay_vertices)
    assert report.violations == (
        "(c) inheritance after the declared transfer: outgoing line 1 "
        "carries (B, Q, S, 2I3) (1, 0, 0, 1), expected (1, 1, 0, 1) "
        "(incoming line 1 with (1, 1, 0, 1) after the transfer "
        "(0, 0, 0, 0) of exchange 1 ('vacuum'))",
    )

    # (d) a decay vertex that does not close locally (vertex 9 planted:
    # pi+ pi+ instead of the pinned K+ K-)
    bad_vertices = list(decay_vertices)
    bad_vertices[9] = (central,
                       (LineState(0, 1, 0, 2), LineState(0, 1, 0, 2)))
    report = audit_event(incoming, exchanges, outgoing, central,
                         tuple(bad_vertices))
    assert report.violations == (
        "(d) local decay closure: decay vertex 9 daughters sum to "
        "(B, Q, S, 2I3) (0, 2, 0, 4), the parent carries (0, 0, 0, 0)",
    )

    # (e) a central source carrying baryon number
    report = audit_event(incoming, exchanges, outgoing,
                         LineState(1, 0, 0, 0), decay_vertices)
    assert report.violations == (
        "(e) central-source neutrality: the central source carries "
        "B = 1 (plan line 849: B = Q = S = 0)",
    )


# ----------------------------------------------------------------------
# test_zero_draws
# ----------------------------------------------------------------------
def test_zero_draws():
    """The ledger machinery consumes zero random numbers: no ledger
    callable accepts an RNG parameter, the module imports no random
    source, and the pinned RNG-state identity holds over a full audit
    trace (the fresh generator and the global bit generator are both
    untouched)."""

    # structural: no ledger callable accepts an RNG-shaped parameter
    for func in (apply_transfer, apply_transfer_qn, audit_event,
                 close_decay, register_exchange):
        signature = inspect.signature(func)
        for name, parameter in signature.parameters.items():
            assert parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD), (func, name)
            assert not re.search(r"rng|random|generator|seed|draw",
                                 name, re.IGNORECASE), (func, name)

    # structural: the module imports no random source at all (standard
    # library bookkeeping only)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "numpy" not in source
    assert "import random" not in source
    assert "np.random" not in source

    # the pinned RNG-state identity over a full audit trace
    rng = np.random.default_rng(MASTER_SEED)
    state0 = rng.bit_generator.state
    global0 = _global_rng_snapshot()

    incoming, exchanges, outgoing, central, decay_vertices = (
        _compliant_event())
    assert apply_transfer(_proton_line(), VACUUM).qn == (1, 1, 0, 1)
    assert apply_transfer_qn(_proton_line().qn, (0, -1, 0, -1)) == (1, 0, 0, 0)
    for parent, daughters in decay_vertices:
        assert close_decay(parent, daughters)
    assert audit_event(incoming, exchanges, outgoing, central,
                       decay_vertices).violations == ()
    for planted in (
            exchanges + (Exchange("charge-exchange", (0, -1, 0, -1)),),
            exchanges + (Exchange("vacuum", (0, 1, 0, 0)),),
    ):
        assert audit_event(incoming, planted, outgoing, central,
                           decay_vertices).violations != ()
    with pytest.raises(NonSingletTransferError):
        register_exchange(Exchange("charge-exchange", (0, -1, 0, -1)))

    assert rng.bit_generator.state == state0
    assert _global_rng_snapshot() == global0


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk: the builder, the card-6 routing, the card-2
    engine, the committed catalog, the continuum families, and the
    sealed R45/R46 drivers in both helper trees; the M8-frozen
    ten-parameter set hash and the kernel boundary through the
    manifest) and the layout sync (the subprocess check from the
    monorepo root; ``logic_drift = 0``)."""

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

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
