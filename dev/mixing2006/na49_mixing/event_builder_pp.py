"""Exact p+p two-source event builder for staged excitation studies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .continuum_channels import ContinuumParameters, SourceDecayer, minimum_excited_mass
from .exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
    SourceMassSampler,
    transfer_slope,
)
from .excitation_kinematics import FourVector, scatter_two_body
from .pdg_database import get_mass
from .quantum_numbers import QuantumNumbers
from .reaction import Reaction
from .source_event import ExcitationEvent, SourceRecord


@dataclass(frozen=True)
class PPExcitationConfig:
    beam_momentum: float = 158.0
    production_mode: str = "simple"
    simple: SimpleExcitationParameters = SimpleExcitationParameters()
    regge: ReggeExcitationParameters = ReggeExcitationParameters()
    continuum: ContinuumParameters = ContinuumParameters()
    decay_sources: bool = True
    conservation_tolerance: float = 1e-10
    max_event_attempts: int = 100

    def validate(self) -> None:
        if self.production_mode not in {"simple", "regge"}:
            raise ValueError("production_mode must be 'simple' or 'regge'")
        if not (math.isfinite(self.beam_momentum) and self.beam_momentum > 0.0):
            raise ValueError("beam_momentum must be finite and positive")
        if not (0.0 < self.conservation_tolerance <= 1e-6):
            raise ValueError("conservation_tolerance must lie in (0, 1e-6]")
        if int(self.max_event_attempts) < 1:
            raise ValueError("max_event_attempts must be positive")
        self.simple.validate()
        self.regge.validate()
        self.continuum.validate()


class PPEventBuilder:
    """Generate two color-neutral ``(B,Q,S)=(1,1,0)`` sources and decay them."""

    def __init__(
        self,
        config: PPExcitationConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reaction = Reaction(config.beam_momentum, 2212, 2212)
        self.sqrt_s = self.reaction.sqrt_s()
        proton_qn = QuantumNumbers(1, 1, 0)
        minimum = minimum_excited_mass(proton_qn)
        parameters = config.simple if config.production_mode == "simple" else config.regge
        self.mass_sampler = SourceMassSampler(
            self.sqrt_s, minimum, minimum, config.production_mode, parameters, self.rng
        )
        self.decayer = SourceDecayer(config.continuum, self.rng)
        proton_mass = get_mass(2212)
        pin = math.sqrt(max(self.sqrt_s * self.sqrt_s / 4.0 - proton_mass * proton_mass, 0.0))
        ein = self.sqrt_s / 2.0
        self.incoming = [
            FourVector(ein, 0.0, 0.0, pin),
            FourVector(ein, 0.0, 0.0, -pin),
        ]

    def generate(self, event_id: int = 0, stage: str | None = None) -> ExcitationEvent:
        total = self.incoming[0] + self.incoming[1]
        parameters = (
            self.config.simple
            if self.config.production_mode == "simple"
            else self.config.regge
        )
        for attempt in range(1, int(self.config.max_event_attempts) + 1):
            m1, m2 = self.mass_sampler.sample()
            try:
                p1, p2, t = scatter_two_body(
                    total,
                    self.incoming[0],
                    m1,
                    m2,
                    transfer_slope(total.mass2(), self.config.production_mode, parameters),
                    self.rng,
                )
                sources = [
                    SourceRecord(
                        0, "projectile", p1, QuantumNumbers(1, 1, 0),
                        collision_index=1,
                        metadata={"sampled_mass": m1, "t": t},
                    ),
                    SourceRecord(
                        1, "target", p2, QuantumNumbers(1, 1, 0),
                        collision_index=1,
                        metadata={"sampled_mass": m2, "t": t},
                    ),
                ]
                event = ExcitationEvent(
                    event_id=int(event_id),
                    stage=stage or self.config.production_mode,
                    incoming=list(self.incoming),
                    incoming_quantum_numbers=QuantumNumbers(2, 2, 0),
                    sources=sources,
                    metadata={
                        "system": "pp",
                        "sqrt_s": self.sqrt_s,
                        "production_mode": self.config.production_mode,
                        "exchange_class": "P",
                        "generation_attempt": attempt,
                    },
                )
                if self.config.decay_sources:
                    particle_id = 0
                    for source in sources:
                        daughters = self.decayer.decay(source, particle_id)
                        event.final_particles.extend(daughters)
                        particle_id += len(daughters)
                event.validate(self.config.conservation_tolerance)
                return event
            except (ValueError, FloatingPointError):
                continue
        raise RuntimeError(
            f"failed to generate a valid p+p event after {self.config.max_event_attempts} attempts"
        )
