"""Event records for color-neutral hadronic excitation sources."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import numbers
from typing import Any, Iterable

from .excitation_kinematics import FourVector, sum_four_vectors
from .pdg_database import get_mass, get_particle
from .quantum_numbers import QuantumNumbers, quantum_numbers
from .state_validation import (
    NonSingletState,
    UnrecognizedStateType,
    is_color_singlet,
    state_type_of,
    validate_state_type,
)


# Metadata keys a ``SourceRecord`` may carry a PDG identity under, in
# precedence order.  ``pdg_id`` and ``resonance_pdg`` name a resonance
# directly; the staged p+A builder records the struck nucleon identity
# as ``target_pdg``.  ``state_type_of`` keeps ``sampled_mass``-first
# precedence, so a nucleon id next to a mass draw cannot change a stamp.
_SOURCE_PDG_METADATA_KEYS = ("pdg_id", "resonance_pdg", "target_pdg")


@dataclass
class SourceRecord:
    source_id: int
    role: str
    p4: FourVector
    quantum_numbers: QuantumNumbers
    exchange_class: str = "P"
    collision_index: int = 0
    parent_source_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Closed five-value state-type label (see na49_mixing.state_validation).
    # Stamped at construction time from the record's own metadata; ``None``
    # is a missing label on a legacy serialized record, not a failure.
    state_type: str | None = None

    def __post_init__(self) -> None:
        if self.state_type is None:
            self.state_type = self._stamp_state_type()

    def _stamp_state_type(self) -> str | None:
        """Stamp the state-type label at construction time.

        Every source built by the staged builders carries a continuum
        mass draw (``metadata["sampled_mass"]``) and stamps
        ``continuum_resonance``.  A source whose metadata instead
        carries a PDG identity (``metadata["pdg_id"]`` or
        ``metadata["resonance_pdg"]``) stamps ``established_resonance``
        when the id names a resonance of the established PDG table.  An
        undeterminable source keeps ``None`` (a missing label, not an
        invalid label); validation rejects invalid labels, never
        missing ones.
        """
        try:
            return state_type_of(self._metadata_pdg_id(), "source", self.metadata)
        except UnrecognizedStateType:
            return None

    def _metadata_pdg_id(self) -> int:
        """The PDG identity carried by the record metadata, 0 when absent.

        ``sampled_mass``-first precedence is ``state_type_of``'s rule;
        this method only resolves the identity.  A missing, boolean, or
        non-integer value is unresolvable and returns 0.
        """
        for key in _SOURCE_PDG_METADATA_KEYS:
            value = self.metadata.get(key)
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                pid = int(value)
                if pid != 0:
                    return pid
        return 0

    @property
    def mass(self) -> float:
        return self.p4.mass()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "role": self.role,
            "p4": self.p4.as_dict(),
            "mass": self.mass,
            "quantum_numbers": self.quantum_numbers.as_dict(),
            "exchange_class": self.exchange_class,
            "collision_index": self.collision_index,
            "parent_source_id": self.parent_source_id,
            "metadata": self.metadata,
            "state_type": self.state_type,
        }


@dataclass
class FinalParticle:
    particle_id: int
    pdg_id: int
    p4: FourVector
    source_id: int
    parent_id: int | None
    channel_family: str
    generation: int = 1
    status: str = "final"
    # Closed five-value state-type label (see na49_mixing.state_validation).
    # Stamped at construction time; ``None`` is a missing label on a legacy
    # serialized record, not a failure.
    state_type: str | None = None

    def __post_init__(self) -> None:
        if self.state_type is None:
            self.state_type = self._stamp_state_type()

    def _stamp_state_type(self) -> str | None:
        """Stamp the state-type label at construction time.

        Spectator nucleons of a p+A event (``channel_family ==
        "spectator"``) are the nuclear remnant and stamp
        ``nuclear_remnant``.  Every other final particle is a stable
        hadron of the vendored PDG mass table and stamps ``hadron``.  A
        particle whose PDG identity is not a table hadron keeps ``None``
        (a missing label, not an invalid label); validation rejects
        invalid labels, never missing ones.
        """
        role = "spectator" if self.channel_family == "spectator" else "final_particle"
        try:
            return state_type_of(self.pdg_id, role)
        except UnrecognizedStateType:
            return None

    @property
    def quantum_numbers(self) -> QuantumNumbers:
        return quantum_numbers(self.pdg_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "particle_id": self.particle_id,
            "pdg_id": self.pdg_id,
            "p4": self.p4.as_dict(),
            "mass": self.p4.mass(),
            "source_id": self.source_id,
            "parent_id": self.parent_id,
            "channel_family": self.channel_family,
            "generation": self.generation,
            "status": self.status,
            "quantum_numbers": self.quantum_numbers.as_dict(),
            "state_type": self.state_type,
        }


@dataclass
class ExcitationEvent:
    event_id: int
    stage: str
    incoming: list[FourVector]
    incoming_quantum_numbers: QuantumNumbers
    sources: list[SourceRecord]
    spectators: list[FinalParticle] = field(default_factory=list)
    final_particles: list[FinalParticle] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None

    def outgoing_four_vector(self) -> FourVector:
        records: Iterable[FourVector]
        if self.final_particles:
            records = [p.p4 for p in self.final_particles] + [p.p4 for p in self.spectators]
        else:
            records = [s.p4 for s in self.sources] + [p.p4 for p in self.spectators]
        return sum_four_vectors(records)

    def outgoing_quantum_numbers(self) -> QuantumNumbers:
        out = QuantumNumbers()
        if self.final_particles:
            for particle in self.final_particles:
                out = out + particle.quantum_numbers
        else:
            for source in self.sources:
                out = out + source.quantum_numbers
        for spectator in self.spectators:
            out = out + spectator.quantum_numbers
        return out

    def conservation_residuals(self) -> dict[str, Any]:
        incoming_p4 = sum_four_vectors(self.incoming)
        residual = incoming_p4 - self.outgoing_four_vector()
        dq = self.incoming_quantum_numbers - self.outgoing_quantum_numbers()
        scale = max(abs(incoming_p4.e), 1.0)
        return {
            "four_vector": residual.as_dict(),
            "relative_max": max(
                abs(residual.e), abs(residual.px), abs(residual.py), abs(residual.pz)
            ) / scale,
            "quantum_numbers": dq.as_dict(),
        }

    def validate(self, relative_tolerance: float = 1e-10) -> None:
        self._validate_state_types()
        residuals = self.conservation_residuals()
        if residuals["relative_max"] > relative_tolerance:
            raise ValueError(
                f"event {self.event_id} four-momentum residual "
                f"{residuals['relative_max']:.3e} exceeds {relative_tolerance:.3e}"
            )
        if tuple(residuals["quantum_numbers"].values()) != (0, 0, 0):
            raise ValueError(
                f"event {self.event_id} quantum-number residual "
                f"{residuals['quantum_numbers']}"
            )
        for source in self.sources:
            daughters = [p for p in self.final_particles if p.source_id == source.source_id]
            if not daughters:
                continue
            q = QuantumNumbers()
            p4 = FourVector(0.0, 0.0, 0.0, 0.0)
            for daughter in daughters:
                q = q + daughter.quantum_numbers
                p4 = p4 + daughter.p4
            if q != source.quantum_numbers:
                raise ValueError(
                    f"source {source.source_id} quantum-number residual "
                    f"{(source.quantum_numbers - q).as_dict()}"
                )
            dp = source.p4 - p4
            source_scale = max(abs(source.p4.e), 1.0)
            rel = max(abs(dp.e), abs(dp.px), abs(dp.py), abs(dp.pz)) / source_scale
            if rel > relative_tolerance:
                raise ValueError(
                    f"source {source.source_id} four-momentum residual {rel:.3e}"
                )
            threshold = sum(p.p4.mass() for p in daughters)
            if threshold > source.mass + 1e-10:
                raise ValueError(
                    f"source {source.source_id} decayed below threshold: "
                    f"{threshold} > {source.mass}"
                )
        self._validate_pdg_on_shell(relative_tolerance)

    def _validate_state_types(self) -> None:
        """Per-state singlet audit (plan section 6, tasks 7-8).

        Every final particle, spectator, and source must carry a
        ``state_type`` from the closed five-value set, and every state
        must be a color singlet: integer ``(B, Q, S)``.  A label outside
        the set raises :class:`UnrecognizedStateType`; a state with
        non-integer ``(B, Q, S)`` raises :class:`NonSingletState`.  A
        missing label (``state_type is None``, e.g. a record serialized
        before the field existed) is a missing label, not an invalid
        label, and passes here.  This check is additive: it runs before
        the conservation residuals and does not alter any existing
        four-momentum, mass-shell, or quantum-number check.
        """
        for particle in list(self.final_particles) + list(self.spectators):
            if particle.state_type is not None:
                validate_state_type(particle.state_type)
            qn = particle.quantum_numbers
            if not is_color_singlet(qn.B, qn.Q, qn.S):
                raise NonSingletState(
                    f"event {self.event_id} particle {particle.particle_id} "
                    f"PDG {particle.pdg_id} is not a color singlet: "
                    f"(B, Q, S) = {(qn.B, qn.Q, qn.S)}"
                )
        for source in self.sources:
            if source.state_type is not None:
                validate_state_type(source.state_type)
            qn = source.quantum_numbers
            if not is_color_singlet(qn.B, qn.Q, qn.S):
                raise NonSingletState(
                    f"event {self.event_id} source {source.source_id} "
                    f"role {source.role!r} is not a color singlet: "
                    f"(B, Q, S) = {(qn.B, qn.Q, qn.S)}"
                )

    def _validate_pdg_on_shell(self, relative_tolerance: float) -> None:
        """Require every final and spectator four-vector mass to match PDG."""
        scale_floor = 1.0
        for particle in list(self.final_particles) + list(self.spectators):
            if get_particle(particle.pdg_id) is None:
                continue
            pdg_mass = get_mass(particle.pdg_id)
            reconstructed = particle.p4.mass()
            scale = max(abs(pdg_mass), scale_floor)
            if abs(reconstructed - pdg_mass) > relative_tolerance * scale:
                raise ValueError(
                    f"particle {particle.particle_id} PDG {particle.pdg_id} "
                    f"off shell: reconstructed mass {reconstructed} vs "
                    f"PDG {pdg_mass}"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "stage": self.stage,
            "incoming": [p.as_dict() for p in self.incoming],
            "incoming_quantum_numbers": self.incoming_quantum_numbers.as_dict(),
            "sources": [s.as_dict() for s in self.sources],
            "spectators": [p.as_dict() for p in self.spectators],
            "final_particles": [p.as_dict() for p in self.final_particles],
            "metadata": self.metadata,
            "rejection_reason": self.rejection_reason,
            "conservation_residuals": self.conservation_residuals(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["event_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload
