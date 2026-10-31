"""Typed hadron/resonance catalog schema (Round 2, card 1).

Task ``TASK-20260913-r2-hadron-resonance-schema`` (plan section 22,
task 9; record fields of plan section 8.1).  The module is the single
typed source of truth for the Round 2 resonance network: the catalog
import (card 2) fills it, the width law (card 3) reads ``pole_mass``,
``nominal_width``, ``mass_range``, and ``orbital_angular_momentum``,
and the decay machinery (cards 5-7) reads ``B, Q, S``,
``decay_channels``, and ``branching_fraction``.

Per record, the schema adds the fields the vendored PDG table
(``pdg_database.py``) does not carry (isospin, isospin3, spin, parity,
``mass_range``, ``decay_channels``, ``branching_fraction``,
``orbital_angular_momentum``, ``provenance``), plus the catalog-level
``stable_states`` termination list.  Existing code is reused, not
reinvented: the vendored table (``pdg_database.py``) supplies the
reference masses, ``quantum_numbers.py`` the integer ``(B, Q, S)``
bookkeeping, and ``state_validation.py`` the color-singlet discipline
(``is_color_singlet``).

No physics values are transcribed here.  Masses, widths, and branching
fractions land with the catalog import (card 2) and carry a
:class:`ProvenanceTag` from that point on; ``None`` provenance is
allowed only before that import.

``HadronCatalog.validate`` is deterministic and report-based: it
returns a :class:`CatalogValidationReport` whose empty ``errors`` tuple
means valid.  It never repairs a record.  The rules are:

- ``pdg_id`` unique per record; every channel daughter is a catalog
  record ``pdg_id`` or a member of ``stable_states()``.
- every channel conserves ``(B, Q, S)`` exactly (plan section 2.5);
  the daughters are summed via ``quantum_numbers()`` and a violating
  channel is an error, never repaired.
- ``branching_fraction`` in ``[0, 1]`` per channel and a per-parent
  sum ``<= 1 + 1e-9``; a residual ``1 - sum`` is the unrepresented
  channel budget and is a warning until card 16 tightens it.
- ``mass_range[0]`` equals the minimum channel threshold within
  ``1e-12`` (the line shape is open wherever any channel is open; the
  card-1 maximum-threshold rule zeroed it below the highest threshold
  and left the Delta(1600), Delta(1700), and K1(1270) poles outside
  their own support, corrected 2026-09-28, FAB-07),
  ``mass_range[1] >= mass_range[0]``, ``nominal_width >= 0``.
- ``pole_mass`` agrees with the vendored table mass (the value
  ``get_mass`` returns when no process-local
  ``mass_width_overrides`` context is active) to ``1e-6`` GeV.
- ``stable_for_decay`` is ``True`` iff the record has no
  ``decay_channels``.
- ``isospin3`` satisfies the Gell-Mann-Nishijima relation
  ``Q = I3 + (B + S)/2`` with ``isospin3 = 2 * I3``.
- every record is a color-singlet hadron: a non-integer ``(B, Q, S)``
  triple fails validation (the ``NonSingletState`` discipline of
  ``state_validation.py``).

Spin and parity are recorded per record; the Round 2 decay is
spin-averaged (round 2 index, item 9), so they are bookkeeping in
this round, not yet consumed by an angular kernel.

The JSON round trip (``to_dict`` / ``to_json`` / ``from_json``) is the
serialization contract for the card-2 catalog file
(``config/v2_singlet/resonance_catalog.json``).

Hermetic: standard library plus the vendored ``na49_mixing`` modules
only; no event generation, no external tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .pdg_database import get_particle
from .quantum_numbers import QuantumNumbers, quantum_numbers
from .state_validation import is_color_singlet

__all__ = [
    "STABLE_STATES",
    "SCHEMA_VERSION",
    "THREE_BODY_LAWS",
    "DecayChannel",
    "ProvenanceTag",
    "HadronRecord",
    "HadronCatalog",
    "CatalogValidationReport",
]


#: Serialization schema version of the catalog JSON payload.
SCHEMA_VERSION = 1

#: The intermediate-mass laws of the three-body decay vertex
#: (``na49_mixing.resonance_decay.three_body_phase_space``): ``"flat_m2"``
#: draws the (b, c) invariant mass flat in ``M_bc**2`` (the Round 2 card 7
#: vertex); ``"phase_space"`` draws it with the flat three-body phase-space
#: density (2026-10-06, the code and physics review's M-3);
#: ``"phase_space_dalitz"`` is ``"phase_space"`` with the measured Dalitz
#: distributions of the omega, eta and eta' three-body decays
#: (:mod:`na49_mixing.dalitz_matrix_elements`, 2026-10-20).  The law is a
#: run attribute of a catalog object, never part of the catalog payload.
THREE_BODY_LAWS: tuple[str, ...] = ("flat_m2", "phase_space", "phase_space_dalitz")

# The cascade termination list (round 2 index, item 11): the vendored
# table's stable entries plus the K0S/K0L weak-decay endpoints.  Every
# other catalog hadron is decayable.  Channel daughters must be catalog
# records or members of this set.
STABLE_STATES: frozenset[int] = frozenset({
    2212, 2112, -2212, -2112,   # p, n, pbar, nbar
    211, -211, 111,             # pi+, pi-, pi0
    321, -321,                  # K+, K-
    310, 130,                   # K0S, K0L (weak-decay endpoints)
})

#: The photon id: a stable final state, never a catalog record.  A
#: channel with a photon daughter is a radiative channel
#: (:attr:`DecayChannel.is_radiative`; 2026-10-02, the radiative modes of
#: eta, eta-prime, omega and phi).
PHOTON_PDG: int = 22

#: The non-hadronic final states: stable daughters that are not catalog
#: records and stay outside the pre-registered hadron termination set
#: :data:`STABLE_STATES` (which every record set of Round 2 carries as
#: records).  :meth:`HadronCatalog.stable_states` and the daughter
#: coverage rule include them.
NON_HADRONIC_FINAL_STATES: frozenset[int] = frozenset({PHOTON_PDG})

# Deterministic validation tolerances (task card, validation rules).
MASS_TABLE_TOL = 1e-6       # GeV: pole_mass vs vendored table mass
MASS_RANGE_TOL = 1e-12      # GeV: mass_range[0] vs min channel threshold
BRANCHING_SUM_TOL = 1e-9    # per-parent branching sum tolerance above 1
RESIDUAL_WARN_TOL = 1e-12   # below this the unrepresented budget is float noise


@dataclass(frozen=True)
class ProvenanceTag:
    """Where a transcribed value came from (round 2 index, item 3).

    The tag follows the Round 1 data-provenance discipline: source
    document, section or table, field, accessed-on date, transcribed
    value, and transcriber.  Branching fractions and particle
    properties are external physics inputs, never fit parameters
    (plan section 8.2).
    """

    document: str                       # e.g. "PDG 2024 review, sect. 27"
    section_or_table: str
    field: str                          # "pole_mass", "branching_fraction"
    accessed_on: str                    # ISO date
    transcribed_value: float | str | None
    transcriber: str                    # e.g. "na49-res r2 card 2"


@dataclass(frozen=True)
class DecayChannel:
    """One decay channel of a catalog record.

    ``daughters`` are PDG ids and are order-independent.  The channel
    conserves ``(B, Q, S)`` exactly (plan section 2.5); ``validate``
    reports a violating channel as an error and never repairs it.
    ``orbital_angular_momentum`` is the L of the dominant partial wave
    (the total L for a three-body channel, card 7).
    """

    parent_pdg: int
    daughters: tuple[int, ...]
    branching_fraction: float           # in [0, 1]; sum over channels <= 1
    orbital_angular_momentum: int
    is_three_body: bool = False         # card 7 flag; L is the total L
    # The card-2 tag; None allowed only before the catalog import.
    provenance: ProvenanceTag | None = None

    @property
    def is_radiative(self) -> bool:
        """True for a channel with a photon daughter (2026-10-02: the
        radiative modes of eta, eta-prime, omega and phi, which had been
        folded into the hadronic channels).  An electromagnetic decay does
        not conserve isospin, so a radiative channel stays outside the
        card-5 charge completion and is its own open object in the kernel,
        as a three-body channel is."""
        return PHOTON_PDG in tuple(int(d) for d in self.daughters)


@dataclass(frozen=True)
class HadronRecord:
    """One color-singlet hadron or established resonance of the catalog.

    The field set is the plan section 8.1 record list verbatim;
    ``branching_fraction`` and ``orbital_angular_momentum`` live on
    :class:`DecayChannel`, the rest here.  ``mass_range`` is the
    ``(threshold, m_up)`` support of the line shape in GeV/c^2 (plan
    section 8.2); for a stable record (no channels) the threshold
    equality is not applicable and the pair only has to be ordered.
    """

    pdg_id: int
    name: str
    B: int
    Q: int
    S: int
    isospin: float                      # half-integer; 0.5 for the pion
    isospin3: int                       # 2*I3 in integer units
    spin: float                         # half-integer
    parity: str                         # "+" or "-"
    pole_mass: float                    # GeV/c^2
    nominal_width: float                # GeV, at the pole mass
    mass_range: tuple[float, float]     # (threshold, m_up), GeV/c^2
    decay_channels: tuple[DecayChannel, ...]
    stable_for_decay: bool              # True: terminal (never queued)
    provenance: ProvenanceTag | None


@dataclass(frozen=True)
class CatalogValidationReport:
    """Deterministic validation outcome; empty ``errors`` means valid."""

    record_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _record_label(record: HadronRecord) -> str:
    return f"{record.name} (pdg {record.pdg_id})"


def _channel_label(record: HadronRecord, channel: DecayChannel) -> str:
    daughters = ", ".join(str(int(d)) for d in channel.daughters)
    return f"channel {_record_label(record)} -> [{daughters}]"


def _provenance_to_json(tag: ProvenanceTag | None):
    if tag is None:
        return None
    return {
        "document": tag.document,
        "section_or_table": tag.section_or_table,
        "field": tag.field,
        "accessed_on": tag.accessed_on,
        "transcribed_value": tag.transcribed_value,
        "transcriber": tag.transcriber,
    }


def _provenance_from_json(payload):
    if payload is None:
        return None
    return ProvenanceTag(
        document=str(payload["document"]),
        section_or_table=str(payload["section_or_table"]),
        field=str(payload["field"]),
        accessed_on=str(payload["accessed_on"]),
        transcribed_value=payload["transcribed_value"],
        transcriber=str(payload["transcriber"]),
    )


def _decay_channel_to_json(channel: DecayChannel) -> dict:
    return {
        "parent_pdg": int(channel.parent_pdg),
        "daughters": [int(d) for d in channel.daughters],
        "branching_fraction": float(channel.branching_fraction),
        "orbital_angular_momentum": int(channel.orbital_angular_momentum),
        "is_three_body": bool(channel.is_three_body),
        "provenance": _provenance_to_json(channel.provenance),
    }


def _decay_channel_from_json(payload: dict) -> DecayChannel:
    return DecayChannel(
        parent_pdg=int(payload["parent_pdg"]),
        daughters=tuple(int(d) for d in payload["daughters"]),
        branching_fraction=float(payload["branching_fraction"]),
        orbital_angular_momentum=int(payload["orbital_angular_momentum"]),
        is_three_body=bool(payload.get("is_three_body", False)),
        provenance=_provenance_from_json(payload.get("provenance")),
    )


def _hadron_record_to_json(record: HadronRecord) -> dict:
    return {
        "pdg_id": int(record.pdg_id),
        "name": record.name,
        "B": int(record.B),
        "Q": int(record.Q),
        "S": int(record.S),
        "isospin": float(record.isospin),
        "isospin3": int(record.isospin3),
        "spin": float(record.spin),
        "parity": record.parity,
        "pole_mass": float(record.pole_mass),
        "nominal_width": float(record.nominal_width),
        "mass_range": [float(record.mass_range[0]), float(record.mass_range[1])],
        "decay_channels": [_decay_channel_to_json(c) for c in record.decay_channels],
        "stable_for_decay": bool(record.stable_for_decay),
        "provenance": _provenance_to_json(record.provenance),
    }


def _hadron_record_from_json(payload: dict) -> HadronRecord:
    lo, hi = payload["mass_range"]
    return HadronRecord(
        pdg_id=int(payload["pdg_id"]),
        name=str(payload["name"]),
        B=int(payload["B"]),
        Q=int(payload["Q"]),
        S=int(payload["S"]),
        isospin=float(payload["isospin"]),
        isospin3=int(payload["isospin3"]),
        spin=float(payload["spin"]),
        parity=str(payload["parity"]),
        pole_mass=float(payload["pole_mass"]),
        nominal_width=float(payload["nominal_width"]),
        mass_range=(float(lo), float(hi)),
        decay_channels=tuple(
            _decay_channel_from_json(c) for c in payload["decay_channels"]
        ),
        stable_for_decay=bool(payload["stable_for_decay"]),
        provenance=_provenance_from_json(payload.get("provenance")),
    )


class HadronCatalog:
    """An immutable, typed set of :class:`HadronRecord` objects.

    The catalog is the substrate shared by the width law and the decay
    machinery.  Construction is permissive; ``validate`` reports all
    rule violations without raising, so a broken catalog can be
    inspected instead of disappearing into an exception.
    """

    def __init__(self, records: Sequence[HadronRecord], *, three_body_law: str = "flat_m2") -> None:
        if three_body_law not in THREE_BODY_LAWS:
            raise ValueError(f"HadronCatalog: three_body_law must be one of {THREE_BODY_LAWS!r}, got {three_body_law!r}")
        # the run attribute of the three-body vertex (not serialized)
        self._three_body_law = str(three_body_law)
        self._records: tuple[HadronRecord, ...] = tuple(records)
        # First occurrence wins on duplicate ids; validate() reports the
        # duplicate as an error, get() stays deterministic.
        self._by_pdg: dict[int, HadronRecord] = {}
        for record in self._records:
            self._by_pdg.setdefault(int(record.pdg_id), record)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return the JSON-serializable catalog payload."""
        return {
            "schema_version": SCHEMA_VERSION,
            "records": [_hadron_record_to_json(r) for r in self._records],
        }

    def to_json(self, path: Path) -> None:
        """Write the catalog payload to *path* (UTF-8, indent 2)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: Path) -> "HadronCatalog":
        """Parse a catalog payload written by :meth:`to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported hadron_catalog schema_version {version!r} "
                f"in {path}; expected {SCHEMA_VERSION}"
            )
        return cls(
            [
                _hadron_record_from_json(r)
                for r in payload["records"]
            ]
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def three_body_law(self) -> str:
        """The intermediate-mass law of the three-body vertex
        (:data:`THREE_BODY_LAWS`; a run attribute, not in :meth:`to_dict`)."""
        return self._three_body_law

    def with_three_body_law(self, law: str) -> "HadronCatalog":
        """A catalog with the same records and the three-body vertex law *law*."""
        return type(self)(self._records, three_body_law=law)

    def records(self) -> tuple[HadronRecord, ...]:
        """Return the records in construction order."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def get(self, pdg_id: int) -> HadronRecord:
        """Return the record for *pdg_id*; KeyError when absent."""
        try:
            return self._by_pdg[int(pdg_id)]
        except KeyError:
            raise KeyError(f"no HadronRecord for pdg_id {pdg_id}") from None

    def _daughter_quantum_numbers(self, pdg_id: int) -> QuantumNumbers:
        """The ``(B, Q, S)`` of a channel daughter: the record's fields
        when the daughter is a record, else the exact PDG quantum numbers."""
        record = self._by_pdg.get(int(pdg_id))
        if record is not None:
            return QuantumNumbers(int(record.B), int(record.Q), int(record.S))
        return quantum_numbers(int(pdg_id))

    def stable_states(self) -> frozenset[int]:
        """The termination set: the pre-registered stable hadrons plus
        every record flagged ``stable_for_decay``."""
        return STABLE_STATES | NON_HADRONIC_FINAL_STATES | {
            int(r.pdg_id) for r in self._records if r.stable_for_decay
        }

    # ------------------------------------------------------------------
    # Kinematics helper
    # ------------------------------------------------------------------
    def _daughter_mass(self, pdg_id: int) -> float:
        """Pole mass of a channel daughter: the catalog record's
        ``pole_mass`` when the daughter is a record, else the vendored
        PDG table mass."""
        record = self._by_pdg.get(int(pdg_id))
        if record is not None:
            return float(record.pole_mass)
        entry = get_particle(int(pdg_id))
        if entry is not None:
            return float(entry.mass)
        raise KeyError(
            f"daughter pdg_id {pdg_id} is neither a catalog record nor "
            "in the vendored PDG table"
        )

    def _channel_threshold(self, channel: DecayChannel) -> float:
        """Sum of daughter pole masses (the channel threshold), GeV/c^2."""
        return math.fsum(self._daughter_mass(d) for d in channel.daughters)

    def threshold(self, parent_pdg: int, channel: DecayChannel) -> float:
        """Sum of daughter pole masses (the channel threshold)."""
        if int(channel.parent_pdg) != int(parent_pdg):
            raise ValueError(
                f"channel parent_pdg {channel.parent_pdg} does not match "
                f"the requested parent pdg_id {parent_pdg}"
            )
        return self._channel_threshold(channel)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> CatalogValidationReport:
        """Run every deterministic schema rule and report the outcome.

        The check never repairs a record and never raises for a rule
        violation; violations are collected as error strings (empty
        tuple == valid).  Warnings carry the unrepresented channel
        budget of plan section 8.2 until card 16 tightens it.
        """
        errors: list[str] = []
        warnings: list[str] = []
        seen: dict[int, str] = {}

        for record in self._records:
            label = _record_label(record)

            # -- pdg_id uniqueness ------------------------------------
            if int(record.pdg_id) in seen:
                errors.append(
                    f"duplicate pdg_id {record.pdg_id}: {label} duplicates "
                    f"{seen[int(record.pdg_id)]}"
                )
            else:
                seen[int(record.pdg_id)] = label

            # -- color-singlet audit (integer (B, Q, S)) ---------------
            if not is_color_singlet(record.B, record.Q, record.S):
                errors.append(
                    f"record {label}: (B, Q, S) = ({record.B}, {record.Q}, "
                    f"{record.S}) is not an integer color-singlet triple"
                )

            # -- Gell-Mann-Nishijima: Q = I3 + (B + S)/2 ---------------
            if 2 * record.Q != record.isospin3 + record.B + record.S:
                i3 = record.isospin3 / 2
                baryon_strangeness = (record.B + record.S) / 2
                errors.append(
                    f"record {label}: isospin3 inconsistent with "
                    f"Q = I3 + (B + S)/2: Q = {record.Q}, "
                    f"I3 = {i3}, (B + S)/2 = {baryon_strangeness}"
                )

            # -- stable_for_decay iff no decay_channels ----------------
            if record.stable_for_decay and record.decay_channels:
                errors.append(
                    f"record {label}: stable_for_decay is True but the "
                    f"record declares {len(record.decay_channels)} "
                    "decay channel(s)"
                )
            if not record.stable_for_decay and not record.decay_channels:
                errors.append(
                    f"record {label}: stable_for_decay is False but the "
                    "record declares no decay channels"
                )

            # -- mass_range shape --------------------------------------
            if len(record.mass_range) != 2:
                errors.append(
                    f"record {label}: mass_range must have exactly two "
                    f"entries, got {len(record.mass_range)}"
                )

            # -- per-channel rules -------------------------------------
            branching_sum = 0.0
            thresholds: list[float] = []
            for channel in record.decay_channels:
                ch_label = _channel_label(record, channel)
                thresholds.append(self._channel_threshold(channel))

                if int(channel.parent_pdg) != int(record.pdg_id):
                    errors.append(
                        f"{ch_label}: parent_pdg {channel.parent_pdg} does "
                        f"not match the record pdg_id {record.pdg_id}"
                    )

                if not 0.0 <= channel.branching_fraction <= 1.0:
                    errors.append(
                        f"{ch_label}: branching_fraction "
                        f"{channel.branching_fraction} outside [0, 1]"
                    )

                # daughter coverage: record or stable state
                unresolved = [
                    int(d)
                    for d in channel.daughters
                    if int(d) not in self._by_pdg
                    and int(d) not in STABLE_STATES
                    and int(d) not in NON_HADRONIC_FINAL_STATES
                ]
                for pdg_id in unresolved:
                    errors.append(
                        f"{ch_label}: daughter pdg {pdg_id} is neither a "
                        "catalog record nor a member of stable_states()"
                    )

                # (B, Q, S) conservation: a record daughter by its record
                # fields, a stable state by quantum_numbers() (the kernel's
                # _bqs_of sources; 2026-10-08, a record absent from the
                # vendored PDG table such as the isoscalar dipion's f0(500))
                if not unresolved:
                    try:
                        daughter_sum = sum(
                            (
                                self._daughter_quantum_numbers(int(d))
                                for d in channel.daughters
                            ),
                            QuantumNumbers(),
                        )
                    except (KeyError, ValueError) as exc:
                        errors.append(
                            f"{ch_label}: daughter quantum numbers "
                            f"unresolvable: {exc}"
                        )
                    else:
                        parent_qn = QuantumNumbers(
                            int(record.B), int(record.Q), int(record.S)
                        )
                        if daughter_sum != parent_qn:
                            errors.append(
                                f"{ch_label}: daughters "
                                f"{tuple(channel.daughters)} sum to "
                                f"(B, Q, S) = {daughter_sum.as_tuple()} but "
                                f"the parent carries (B, Q, S) = "
                                f"{parent_qn.as_tuple()}"
                            )

            branching_sum = math.fsum(
                c.branching_fraction for c in record.decay_channels
            )

            # -- per-parent branching sum ------------------------------
            if record.decay_channels:
                if branching_sum > 1.0 + BRANCHING_SUM_TOL:
                    errors.append(
                        f"record {label}: branching fractions sum to "
                        f"{branching_sum}, exceeding 1 by more than "
                        f"{BRANCHING_SUM_TOL}"
                    )
                elif 1.0 - branching_sum > RESIDUAL_WARN_TOL:
                    residual = 1.0 - branching_sum
                    warnings.append(
                        f"record {label}: unrepresented channel budget: "
                        f"branching fractions sum to {branching_sum} "
                        f"(residual {residual})"
                    )

            # -- mass_range consistency --------------------------------
            if record.decay_channels and len(record.mass_range) == 2:
                lowest = min(thresholds)
                if abs(record.mass_range[0] - lowest) > MASS_RANGE_TOL:
                    errors.append(
                        f"record {label}: mass_range[0] = "
                        f"{record.mass_range[0]} is below or above the "
                        f"lowest channel threshold {lowest} (tolerance "
                        f"{MASS_RANGE_TOL})"
                    )
            if len(record.mass_range) == 2 and record.mass_range[1] < record.mass_range[0]:
                errors.append(
                    f"record {label}: mass_range[1] = "
                    f"{record.mass_range[1]} is below mass_range[0] = "
                    f"{record.mass_range[0]}"
                )

            # -- width --------------------------------------------------
            if not record.nominal_width >= 0.0:
                errors.append(
                    f"record {label}: nominal_width {record.nominal_width} "
                    "is negative"
                )

            # -- pole_mass vs the vendored PDG table --------------------
            entry = get_particle(int(record.pdg_id))
            if entry is not None:
                deviation = abs(float(record.pole_mass) - float(entry.mass))
                if deviation > MASS_TABLE_TOL:
                    errors.append(
                        f"record {label}: pole_mass {record.pole_mass} "
                        f"disagrees with the vendored PDG table mass "
                        f"{entry.mass} by more than {MASS_TABLE_TOL}"
                    )

        return CatalogValidationReport(
            record_count=len(self._records),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )
