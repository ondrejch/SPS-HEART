"""The explicit exchange quantum-number ledger (Round 5, card 2).

``TASK-20260925-r5-exchange-quantum-number-ledger``; plan section 22
task 35 (line 1760); plan section 11.4 (lines 871-881).  The module
replaces unrestricted global quantum-number allocation with an
explicit exchange ledger (plan lines 873-879):

- each incoming hadronic line has a ledger state (:class:`LineState`);
- each exchange has a declared ``(B, Q, S, 2I3)`` transfer
  (:class:`Exchange`);
- the vacuum exchange transfers zero (:data:`VACUUM`, plan line 877);
- each outgoing excitation inherits the line quantum numbers after the
  declared transfer (:func:`apply_transfer`; rule (c) of
  :func:`audit_event`);
- every decay closes locally (:func:`close_decay`; rule (d)).

"This prevents a central source from acquiring charge, strangeness, or
baryon number without a physical color-singlet carrier" (plan line
881): rule (e) of :func:`audit_event` holds the central source at
``B = Q = S = 0`` (plan line 849).  The ledger is the color-singlet
audit mechanism of Gate 5 (plan section 5.5 item 7, line 323: "it
passes the color-singlet audit") and the assertion mechanism of run
R54 (plan lines 928-930: zero violations across at least one million
events), and it is the mechanism-side guarantee of plan section 24
line 1864 (no quantum-number allocation without explicit exchange
carriers).

The closed allowed-exchange set
-------------------------------

The closed allowed-exchange set of Round 5 is exactly
:data:`ALLOWED_EXCHANGES = {VACUUM}`; charge exchange (plan task 47)
and the secondary singlet exchanges (plan tasks 45-46) are Round 8 and
are not in the set.  :func:`register_exchange` is the Round 8
interface: in Round 5 the set is closed, so calling it raises
:class:`NonSingletTransferError` naming the exchange, and
:func:`apply_transfer` raises the same error for any exchange outside
the closed set.  A neutron remnant is therefore unreachable through
the ledger at this round's landing (plan line 838: "a neutron only
through explicit charge exchange").

The Round 8 isovector exchange (2026-10-04)
-------------------------------------------

``TASK-20261004-r8-isovector-exchange-class`` (plan task 47) declares the
isovector exchange records :data:`ISOVECTOR_EXCHANGES`: an I = 1
color-singlet exchange with the I3 transfer ``t`` = -1, 0 or +1 to a line,
the transfer ``(0, t, 0, 2t)``.  The two lines of an event take opposite
transfers.  :data:`ROUND8_EXCHANGES` is the vacuum exchange and the three
isovector records.  The Round 5 contract stays as it was: the closed set
:data:`ALLOWED_EXCHANGES` is ``{VACUUM}``, :func:`register_exchange` still
raises, and :func:`apply_transfer` and :func:`audit_event` check against
the Round 5 set unless the caller passes ``allowed=ROUND8_EXCHANGES``.

Conventions
-----------

All ledger quantities are exact integers, with the third isospin
component stored doubled (``I3x2 = 2 * I3``): the proton is ``(B, Q,
S, 2I3) = (1, 1, 0, 1)`` and the total p+p state is ``(2, 2, 0, 2)``
(the card's pre-registered instantiation).  The sign convention of the
declared transfer follows the card's pinned inheritance fixture: the
transfer is the quantum-number change imparted to the line, so the
outgoing excitation inherits the line quantum numbers *after* the
declared transfer, ``outgoing = incoming + transfer`` element-wise in
exact integer arithmetic.  The pinned fixture applies the transfer
``(0, -1, 0, -1)`` to the proton line ``(1, 1, 0, 1)`` and yields
``(1, 0, 0, 0)`` (the neutron ``(B, Q, S)`` on the arithmetic
fixture).  Every comparison and every sum carries no tolerance.

The audit (:func:`audit_event`) pairs the outgoing lines with the
incoming lines and the exchanges by index: outgoing line ``i``
inherits incoming line ``i mod n_in`` after the transfer of exchange
``i mod n_ex`` (with no exchange record in the event, the incoming
quantum numbers pass through unchanged).  The audit compares quantum
numbers only.  The optional four-momentum field exists for event-level
audits, is carried through :func:`apply_transfer` unchanged, and is
never compared: the ledger performs no kinematics.

Round 5 record consumption
--------------------------

The ledger consumes zero random numbers; it is pure bookkeeping over
the class records of cards 1, 5, 6, and 7, and it changes no draw
count and no draw order of the class generators.  Card 1's
``na49_mixing/topology_elastic.py`` record convention (the per-line
doubled-I3 ``(B, Q, S, 2I3)`` quantum numbers and the vacuum exchange
with the ``(0, 0, 0, 0)`` transfer) is consumed read-only: the ledger
records restate the same doubled-I3 convention with the quantum
numbers as named fields.

Hermetic: standard library only; no event generation driver, no
external tables, no Monte Carlo run (the acceptance is hermetic:
pinned fixtures, exact integer identities, draw budgets; the runs
R50-R55 exercise the ledger, and R54 asserts zero violations across at
least one million events).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ALLOWED_EXCHANGES",
    "ISOVECTOR_EXCHANGES",
    "ROUND8_EXCHANGES",
    "Exchange",
    "LedgerAuditReport",
    "LineState",
    "NonSingletTransferError",
    "QN_LABELS",
    "VACUUM",
    "ZERO_TRANSFER",
    "apply_transfer",
    "apply_transfer_qn",
    "audit_event",
    "close_decay",
    "register_exchange",
]


#: The doubled-I3 quantum-number labels, in the fixed record order.
QN_LABELS: tuple[str, str, str, str] = ("B", "Q", "S", "2I3")

#: The zero quantum-number transfer (plan line 877: the vacuum exchange
#: transfers zero).
ZERO_TRANSFER: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class LineState:
    """One hadronic line's ledger state: the doubled-I3 integer quantum
    numbers ``(B, Q, S, 2I3)`` and an optional four-momentum ``(E, px,
    py, pz)`` for event-level audits."""

    B: int
    Q: int
    S: int
    I3x2: int
    four_momentum: tuple[float, float, float, float] | None = None

    @property
    def qn(self) -> tuple[int, int, int, int]:
        """The quantum-number quadruple in record order."""
        return (self.B, self.Q, self.S, self.I3x2)

    def __post_init__(self) -> None:
        for label, value in zip(QN_LABELS, (self.B, self.Q, self.S,
                                            self.I3x2)):
            if not isinstance(value, int):
                raise ValueError(
                    f"LineState: the {label} component must be an exact "
                    f"integer, got {value!r}")


@dataclass(frozen=True)
class Exchange:
    """One exchange: its name from the closed allowed set and its
    declared ``(B, Q, S, 2I3)`` transfer, the quantum-number change
    imparted to the line (the vacuum exchange transfers zero, plan
    line 877)."""

    name: str
    transfer: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.transfer) != 4:
            raise ValueError(
                f"Exchange {self.name!r}: the transfer must be a "
                f"(B, Q, S, 2I3) quadruple, got {self.transfer!r}")
        for label, value in zip(QN_LABELS, self.transfer):
            if not isinstance(value, int):
                raise ValueError(
                    f"Exchange {self.name!r}: the transfer {label} "
                    f"component must be an exact integer, got {value!r}")


#: The vacuum exchange, the single color-singlet exchange with zero
#: quantum-number transfer (plan line 877).
VACUUM = Exchange("vacuum", ZERO_TRANSFER)

#: The closed allowed-exchange set of Round 5: exactly the vacuum
#: exchange.  Charge exchange (plan task 47) and the secondary singlet
#: exchanges (plan tasks 45-46) are Round 8 and are not in the set.
ALLOWED_EXCHANGES: frozenset[Exchange] = frozenset({VACUUM})

#: The Round 8 isovector exchange records (plan task 47, 2026-10-04): the
#: I = 1 exchange with the I3 transfer t = -1, 0, +1 to a line, the
#: transfer ``(0, t, 0, 2t)`` (``na49_mixing.isovector_exchange``).
ISOVECTOR_EXCHANGES: frozenset[Exchange] = frozenset(
    Exchange("isovector", (0, t, 0, 2 * t)) for t in (-1, 0, 1))

#: The Round 8 exchange set: the vacuum exchange and the isovector records.
ROUND8_EXCHANGES: frozenset[Exchange] = ALLOWED_EXCHANGES | ISOVECTOR_EXCHANGES


class NonSingletTransferError(ValueError):
    """Raised when an exchange transfer is outside the closed
    :data:`ALLOWED_EXCHANGES` set (names the offending exchange)."""


def _closed_set_names(allowed: frozenset[Exchange] = ALLOWED_EXCHANGES) -> str:
    """The allowed-exchange set's names, sorted for a deterministic
    message."""
    return ", ".join(sorted({exchange.name for exchange in allowed}))


def register_exchange(exchange: Exchange) -> None:
    """Round 8 interface: add an exchange to the closed set.

    In Round 5 the set is fixed at ``{VACUUM}``; calling this raises
    :class:`NonSingletTransferError` naming the exchange (charge
    exchange is plan task 47 and the secondary singlet exchanges are
    plan tasks 45-46, both Round 8).  The closed set itself never
    changes at this round.
    """

    if not isinstance(exchange, Exchange):
        raise ValueError(
            f"register_exchange: expected an Exchange, got {type(exchange)!r}")
    raise NonSingletTransferError(
        "register_exchange: the Round 5 allowed-exchange set is closed "
        f"at {{{_closed_set_names()}}}; registering the exchange "
        f"{exchange.name!r} with the declared transfer "
        f"{tuple(exchange.transfer)} is not available at this round "
        "(charge exchange is plan task 47 and the secondary singlet "
        "exchanges are plan tasks 45-46, Round 8; plan section 24 line "
        "1864: no quantum-number allocation without explicit exchange "
        "carriers)")


def apply_transfer_qn(
    qn: tuple[int, int, int, int],
    transfer: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """The exact integer transfer map on one quantum-number quadruple:
    ``outgoing = incoming + transfer`` element-wise (see the module
    docstring for the sign convention and the pinned fixture)."""

    if len(qn) != 4 or len(transfer) != 4:
        raise ValueError(
            f"apply_transfer_qn: expected (B, Q, S, 2I3) quadruples, "
            f"got qn {qn!r} and transfer {transfer!r}")
    return (qn[0] + transfer[0], qn[1] + transfer[1],
            qn[2] + transfer[2], qn[3] + transfer[3])


def apply_transfer(line: LineState, exchange: Exchange,
                   allowed: frozenset[Exchange] = ALLOWED_EXCHANGES) -> LineState:
    """The outgoing excitation of *line* under *exchange*: the line
    quantum numbers after the declared transfer (``outgoing =
    incoming + transfer``, exact integers; see the module docstring).

    Raises :class:`NonSingletTransferError` if the exchange is not in
    *allowed* (the Round 5 :data:`ALLOWED_EXCHANGES` by default;
    :data:`ROUND8_EXCHANGES` admits the isovector records), naming the
    exchange.  The optional
    four-momentum field is carried through unchanged (the ledger
    performs no kinematics).
    """

    if not isinstance(line, LineState):
        raise ValueError(
            f"apply_transfer: expected a LineState, got {type(line)!r}")
    if not isinstance(exchange, Exchange):
        raise ValueError(
            f"apply_transfer: expected an Exchange, got {type(exchange)!r}")
    if exchange not in allowed:
        raise NonSingletTransferError(
            f"apply_transfer: the exchange {exchange.name!r} with the "
            f"declared transfer {tuple(exchange.transfer)} is outside the "
            f"closed allowed-exchange set {{{_closed_set_names(allowed)}}} "
            "(plan section 24 line 1864: no quantum-number allocation "
            "without explicit exchange carriers)")
    b, q, s, i3x2 = apply_transfer_qn(line.qn, exchange.transfer)
    return LineState(B=b, Q=q, S=s, I3x2=i3x2, four_momentum=line.four_momentum)


def _sum_qn(states) -> tuple[int, int, int, int]:
    """The exact integer component-wise sum of the states' quantum
    numbers (the empty sum is ``(0, 0, 0, 0)``)."""

    total = [0, 0, 0, 0]
    for state in states:
        qn = state.qn
        for k in range(4):
            total[k] += qn[k]
    return (total[0], total[1], total[2], total[3])


def close_decay(parent: LineState, daughters: tuple[LineState, ...]) -> bool:
    """Local decay closure: the daughter ``(B, Q, S, 2I3)`` sum equals
    the parent exactly (exact integer arithmetic, no tolerance)."""

    if not isinstance(parent, LineState):
        raise ValueError(
            f"close_decay: expected a LineState parent, got {type(parent)!r}")
    return _sum_qn(daughters) == parent.qn


@dataclass(frozen=True)
class LedgerAuditReport:
    """The audit outcome: one entry per violated rule instance, empty
    on a compliant event; each entry names the rule and the record."""

    violations: tuple[str, ...]


def audit_event(
    incoming_lines: tuple[LineState, LineState],
    exchanges: tuple[Exchange, ...],
    outgoing_lines: tuple[LineState, ...],
    central_source: LineState | None,
    decay_vertices: tuple[tuple[LineState, tuple[LineState, ...]], ...],
    allowed: frozenset[Exchange] = ALLOWED_EXCHANGES,
) -> LedgerAuditReport:
    """The five-rule audit of one event record (plan section 11.4,
    lines 875-879, and plan line 849):

    (a) every exchange record is in the allowed-exchange set *allowed*
        (the Round 5 :data:`ALLOWED_EXCHANGES` by default;
        :data:`ROUND8_EXCHANGES` admits the isovector records);
    (b) the vacuum exchange transfers zero;
    (c) each outgoing excitation equals the line quantum numbers after
        the declared transfer (outgoing line ``i`` inherits incoming
        line ``i mod n_in`` after the transfer of exchange ``i mod
        n_ex``, exact integers);
    (d) every decay vertex closes locally (the daughter sum equals the
        parent exactly);
    (e) no central source carries nonzero B, Q, or S (``B = Q = S = 0``,
        plan line 849).

    Returns a :class:`LedgerAuditReport` whose violation list is empty
    exactly when the event satisfies the five rules; each violation
    names the rule and the record.  Exact integer arithmetic with no
    tolerance; no random number is consumed.
    """

    violations: list[str] = []
    n_in = len(incoming_lines)
    n_ex = len(exchanges)

    # (a) closed-set membership
    for idx, exchange in enumerate(exchanges):
        if exchange not in allowed:
            violations.append(
                f"(a) closed-set membership: exchange {idx} "
                f"({exchange.name!r}, transfer {tuple(exchange.transfer)}) "
                "is outside the closed allowed-exchange set "
                f"{{{_closed_set_names(allowed)}}}")

    # (b) the vacuum exchange transfers zero
    for idx, exchange in enumerate(exchanges):
        if (exchange.name == VACUUM.name
                and tuple(exchange.transfer) != ZERO_TRANSFER):
            violations.append(
                f"(b) vacuum zero transfer: exchange {idx} "
                f"({exchange.name!r}) carries the nonzero transfer "
                f"{tuple(exchange.transfer)}")

    # (c) inheritance after the declared transfer
    for idx, out_line in enumerate(outgoing_lines):
        if n_in == 0:
            violations.append(
                "(c) inheritance after the declared transfer: outgoing "
                f"line {idx} has no incoming line record to inherit from")
            continue
        in_line = incoming_lines[idx % n_in]
        if n_ex:
            exchange = exchanges[idx % n_ex]
            transfer = exchange.transfer
            origin = (f" after the transfer {tuple(transfer)} of exchange "
                      f"{idx % n_ex} ({exchange.name!r})")
        else:
            transfer = ZERO_TRANSFER
            origin = " (no exchange record in the event)"
        expected = apply_transfer_qn(in_line.qn, transfer)
        actual = out_line.qn
        if actual != expected:
            violations.append(
                "(c) inheritance after the declared transfer: outgoing "
                f"line {idx} carries (B, Q, S, 2I3) {actual}, expected "
                f"{expected} (incoming line {idx % n_in} with "
                f"{in_line.qn}{origin})")

    # (d) local decay closure
    for vidx, (parent, daughters) in enumerate(decay_vertices):
        if not close_decay(parent, daughters):
            violations.append(
                f"(d) local decay closure: decay vertex {vidx} daughters "
                f"sum to (B, Q, S, 2I3) {_sum_qn(daughters)}, the parent "
                f"carries {parent.qn}")

    # (e) central-source neutrality
    if central_source is not None:
        for label, value in zip(("B", "Q", "S"),
                                (central_source.B, central_source.Q,
                                 central_source.S)):
            if value != 0:
                violations.append(
                    f"(e) central-source neutrality: the central source "
                    f"carries {label} = {value} (plan line 849: "
                    "B = Q = S = 0)")

    return LedgerAuditReport(violations=tuple(violations))
