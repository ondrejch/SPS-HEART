"""State-type labels and the per-state color-singlet check.

Round 1, card TASK-20260911-r1-allowed-state-type-validation (plan
section 6, tasks 7-8).  Every generated state carries a ``state_type``
label drawn from a closed five-value set:

- ``hadron``: a stable hadron from the vendored PDG mass table with
  integer ``(B, Q, S)`` (a decay daughter on a staged event, or a
  generated track on the legacy path).
- ``established_resonance``: a source state that is a named resonance
  of the established PDG table (non-zero decay width) rather than a
  continuum mass draw.
- ``continuum_resonance``: a source state whose mass comes from a
  continuum draw (the staged ``SourceMassSampler`` mass, recorded as
  ``metadata["sampled_mass"]``).
- ``exchange``: an exchange record of the color-neutral exchange
  channel (labeled by role, not by a PDG identity).
- ``nuclear_remnant``: a spectator nucleon of a p+A event, the remnant
  of the target nucleus in the independent-nucleon picture.

The set is closed.  A sixth label needs its own card; a ``state_type``
outside ``STATE_TYPES`` has probability zero of passing validation: it
is rejected at validation time, never repaired at generation time.

A *missing label* (``state_type is None``) is distinct from an *invalid
label*: a record serialized before the field existed parses back with
``state_type = None`` and does not fail validation.  An invalid label
raises :class:`UnrecognizedStateType`; a state whose ``(B, Q, S)`` is
not an integer triple raises :class:`NonSingletState`.

No physics: the mapping is a labeling rule over the existing singlet
ontology.  Every state already carries integer ``(B, Q, S)``; this
module makes the label explicit and asserts the integers per state.
"""

from __future__ import annotations

import math
import numbers

from .pdg_database import get_particle
from .quantum_numbers import quantum_numbers

__all__ = [
    "STATE_TYPES",
    "UnrecognizedStateType",
    "NonSingletState",
    "state_type_of",
    "validate_state_type",
    "is_color_singlet",
]


class UnrecognizedStateType(ValueError):
    """A ``state_type`` label outside the closed ``STATE_TYPES`` set."""


class NonSingletState(ValueError):
    """A state whose ``(B, Q, S)`` is not an integer color-singlet triple."""


#: The closed set of allowed state-type labels (plan section 6, task 7).
STATE_TYPES: frozenset = frozenset({
    "hadron",
    "established_resonance",
    "continuum_resonance",
    "exchange",
    "nuclear_remnant",
})

# Role vocabulary of the mapping.  ``final_particle`` is a decay daughter
# or generated track, ``source`` an excitation source, ``spectator`` the
# nuclear remnant of a p+A event, and ``exchange_record`` an entry of the
# color-neutral exchange channel.
_ROLE_FINAL_PARTICLE = "final_particle"
_ROLE_SOURCE = "source"
_ROLE_SPECTATOR = "spectator"
_ROLE_EXCHANGE_RECORD = "exchange_record"


def _is_integer_value(value: object) -> bool:
    """True when *value* is a finite number with an integer value."""
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, numbers.Real):
        as_float = float(value)
        return math.isfinite(as_float) and as_float == math.trunc(as_float)
    return False


def _require_table_hadron(pdg_id: int) -> None:
    """Raise :class:`UnrecognizedStateType` unless *pdg_id* is a hadron
    of the vendored PDG mass table with integer ``(B, Q, S)``."""
    pid = int(pdg_id)
    if get_particle(pid) is None:
        raise UnrecognizedStateType(
            f"pdg_id {pid} is not in the vendored PDG mass table"
        )
    try:
        quantum_numbers(pid)
    except (KeyError, ValueError) as exc:
        raise UnrecognizedStateType(
            f"pdg_id {pid} carries no integer (B, Q, S): {exc}"
        ) from exc


def _is_established_resonance(pdg_id: int) -> bool:
    """True when *pdg_id* names a resonance of the established PDG table."""
    entry = get_particle(int(pdg_id))
    return entry is not None and entry.width > 0.0


def state_type_of(pdg_id: int, role: str, metadata: dict | None = None) -> str:
    """Map a ``(pdg_id, role, metadata)`` triple onto ``STATE_TYPES``.

    ``role`` is one of ``"final_particle"``, ``"source"``,
    ``"spectator"``, ``"exchange_record"``.  A source counts as a
    continuum resonance when its metadata carries a continuum mass draw
    (``metadata["sampled_mass"]``), and as an established resonance when
    it carries a named PDG resonance id.  Any other combination raises
    :class:`UnrecognizedStateType`.
    """
    if metadata is None:
        clean_metadata: dict = {}
    elif isinstance(metadata, dict):
        clean_metadata = metadata
    else:
        raise UnrecognizedStateType(
            f"source metadata must be a dict or None, got {type(metadata).__name__}"
        )

    if role == _ROLE_FINAL_PARTICLE:
        _require_table_hadron(pdg_id)
        return "hadron"
    if role == _ROLE_SOURCE:
        if "sampled_mass" in clean_metadata:
            return "continuum_resonance"
        if _is_established_resonance(pdg_id):
            return "established_resonance"
        raise UnrecognizedStateType(
            f"source state_type undetermined: pdg_id {pdg_id!r} carries "
            "neither a continuum mass draw (metadata['sampled_mass']) nor "
            "an established resonance id"
        )
    if role == _ROLE_SPECTATOR:
        return "nuclear_remnant"
    if role == _ROLE_EXCHANGE_RECORD:
        return "exchange"
    raise UnrecognizedStateType(f"unknown state role {role!r}")


def validate_state_type(state_type: str) -> None:
    """Raise :class:`UnrecognizedStateType` for a label outside the set."""
    if state_type not in STATE_TYPES:
        raise UnrecognizedStateType(
            f"unrecognized state_type {state_type!r}; the allowed set is "
            f"{sorted(STATE_TYPES)}"
        )


def is_color_singlet(b: int, q: float, s: int) -> bool:
    """True iff ``(b, q, s)`` is an integer color-singlet triple.

    A color-singlet hadron carries integer baryon number, integer
    charge, and integer strangeness; a colored object would carry
    fractional baryon number or non-integer charge in the singlet
    decomposition used here.  The check asserts integer values only; it
    never changes the quantum numbers.
    """
    return (
        _is_integer_value(b)
        and _is_integer_value(q)
        and _is_integer_value(s)
    )
