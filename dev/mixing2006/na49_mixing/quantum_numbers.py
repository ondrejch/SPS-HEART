"""Exact additive quantum numbers for excitation-source events.

The legacy analysis database stores charge only.  This module adds the integer
``(B, Q, S)`` bookkeeping required by the color-neutral excitation model
without changing legacy ``Particle`` behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pdg_database import get_charge


@dataclass(frozen=True)
class QuantumNumbers:
    """Integer baryon number, electric charge, and strangeness."""

    B: int = 0
    Q: int = 0
    S: int = 0

    def __add__(self, other: "QuantumNumbers") -> "QuantumNumbers":
        return QuantumNumbers(self.B + other.B, self.Q + other.Q, self.S + other.S)

    def __sub__(self, other: "QuantumNumbers") -> "QuantumNumbers":
        return QuantumNumbers(self.B - other.B, self.Q - other.Q, self.S - other.S)

    def __neg__(self) -> "QuantumNumbers":
        return QuantumNumbers(-self.B, -self.Q, -self.S)

    def as_tuple(self) -> tuple[int, int, int]:
        return self.B, self.Q, self.S

    def as_dict(self) -> dict[str, int]:
        return {"B": self.B, "Q": self.Q, "S": self.S}


# Baryons represented by the bundled particle database.  The sign of the PDG
# id distinguishes baryons and antibaryons for every listed state.
_NONSTRANGE_BARYONS = frozenset({
    2212, 2112,
    2224, 2214, 2114, 1114,
    202224, 202214, 202114, 201114,
    122224, 122214, 122114, 121114,
    12112, 12212, 2116, 2216, 22212, 22112,
    102112, 102212, 122112, 122212, 212114, 212214,
    9911201, 9912201, 9921201, 9922201, 9931201, 9932201,
    2219,
})
_STRANGE_BARYONS = frozenset({
    3122, 3222, 3212, 3112, 102134, 3224, 3214, 3114,
})
_POSITIVE_STRANGENESS_MESONS = frozenset({
    321, 311, 323, 313, 10323, 10313, 325, 315,
})


def quantum_numbers(pdg_id: int) -> QuantumNumbers:
    """Return exact ``(B,Q,S)`` for a supported PDG id.

    Unknown particles are rejected rather than silently assigned zero quantum
    numbers.  This makes an incomplete channel table fail during validation.
    The antibaryon of a listed baryon resolves even where the bundled
    database lists only the baryon.
    """
    pid = int(pdg_id)
    apid = abs(pid)
    charge = get_charge(pid, default=float("nan"))
    if charge != charge and pid < 0 and apid in _NONSTRANGE_BARYONS | _STRANGE_BARYONS:
        # the bundled database lists some antibaryons only through their
        # baryon (the charged anti-N*, anti-Sigma and anti-Delta of the
        # charge-conjugate towers); the antibaryon carries the negated
        # charge (review grok-4.7 M3, 2026-10-10)
        charge = -get_charge(apid, default=float("nan"))
    if charge != charge:
        raise KeyError(f"unknown PDG id {pid}")
    q = int(round(charge))
    if abs(charge - q) > 1e-12:
        raise ValueError(f"non-integral charge {charge!r} for PDG id {pid}")

    if apid in _NONSTRANGE_BARYONS:
        return QuantumNumbers(1 if pid > 0 else -1, q, 0)
    if apid in _STRANGE_BARYONS:
        sign = 1 if pid > 0 else -1
        return QuantumNumbers(sign, q, -sign)
    if apid in _POSITIVE_STRANGENESS_MESONS:
        return QuantumNumbers(0, q, 1 if pid > 0 else -1)
    # K0S/K0L are flavor mixtures and are not legal source-local strong
    # channel endpoints.  They are accepted for legacy weak-decay audits with
    # zero net strangeness.
    return QuantumNumbers(0, q, 0)


def sum_quantum_numbers(pdg_ids) -> QuantumNumbers:
    total = QuantumNumbers()
    for pdg_id in pdg_ids:
        total = total + quantum_numbers(int(pdg_id))
    return total
