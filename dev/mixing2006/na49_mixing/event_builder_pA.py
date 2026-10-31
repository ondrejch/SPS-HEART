"""Sequential projectile plus multiple target-source p+A event builder.

Reference frames
----------------
The builder constructs every p+A event in the laboratory frame: the beam
proton carries ``(E_beam, 0, 0, p_beam)`` and every target nucleon sits
at rest. The p+p builder (``event_builder_pp``) emits its events in the
nucleon-nucleon center-of-mass frame. The NA49 p+A observables are
defined in the nucleon-nucleon CM frame of the beam proton on one target
nucleon at rest: Feynman x_F = 2 p_z^CM / sqrt(s_NN), the invariant
weight E^CM / p_T, and the CM rapidity. Every p+A event therefore records
its frame in the metadata (``frame = "lab"``, with the boost parameters
``nn_cm_beta``, ``nn_cm_gamma``, and ``sqrt_s_nn``), and consumers read
the tracks through :func:`nn_cm_p4`, :func:`nn_cm_xf`, or
:func:`event_in_nn_cm`. These helpers boost a lab-frame event and return
a CM-frame event (any p+p event) unchanged, so the p+p x_F stays bit for
bit what ``2 p_z / sqrt(s)`` gives. The transverse momentum is invariant
under the longitudinal boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .glauber import NucleusSpec, WoundedNucleonSampler
from .parallel import MIN_WOUND_RANGE_FRACTION
from .pdg_database import get_mass
from .quantum_numbers import QuantumNumbers
from .reaction import Reaction
from .source_event import ExcitationEvent, FinalParticle, SourceRecord


# --- Reference frames ---------------------------------------------------------

FRAME_LAB = "lab"
FRAME_NN_CM = "nn_cm"
PA_STAGE = "baryon_nucleus"


def nn_cm_boost(beam_momentum: float) -> tuple[float, float, float]:
    """The nucleon-nucleon CM boost of a proton beam on a nucleon at rest.

    Returns ``(beta, gamma, sqrt_s_nn)`` with ``beta = p_beam / (E_beam +
    m_p)`` and ``gamma = (E_beam + m_p) / sqrt(s_NN)``, the CM velocity
    and Lorentz factor along the beam axis. ``sqrt_s_nn`` is
    ``Reaction(beam_momentum, 2212, 2212).sqrt_s()``, the same sqrt(s) the
    p+p builder and the staged drivers divide by (17.2702 GeV at 158
    GeV/c), so ``gamma`` and ``sqrt_s_nn`` are consistent by construction:
    ``gamma * sqrt_s_nn`` is the total lab energy of the nucleon pair.
    The proton mass is the convention of the NA49 nucleon-nucleon frame.
    A neutron target carries its own mass, so a neutron-struck pair moves
    in this frame with ``p_z = -beta gamma (m_n - m_p)``, about -12 MeV/c
    at 158 GeV/c.
    """
    reaction = Reaction(float(beam_momentum), 2212, 2212)
    mass = get_mass(2212)
    p_beam = float(beam_momentum)
    e_total = math.sqrt(mass * mass + p_beam * p_beam) + mass
    sqrt_s_nn = reaction.sqrt_s()
    return p_beam / e_total, e_total / sqrt_s_nn, sqrt_s_nn


def lab_to_nn_cm(p4: FourVector, beta: float, gamma: float) -> FourVector:
    """Boost a lab-frame four-vector into the frame moving with ``beta``
    along +z: ``E' = gamma (E - beta p_z)``, ``p_z' = gamma (p_z - beta
    E)``; the transverse components are unchanged."""
    return FourVector(
        gamma * (p4.e - beta * p4.pz),
        p4.px,
        p4.py,
        gamma * (p4.pz - beta * p4.e),
    )


def event_frame(event) -> str:
    """The reference frame of an event's four-vectors.

    The p+A builder records ``metadata["frame"]``. An event without the
    record is a nucleon-nucleon CM event (the p+p builder emits CM events
    and records no frame); a p+A-stage event without the record predates
    the frame record and was built in the laboratory, so it is refused
    rather than silently read as CM.
    """
    metadata = getattr(event, "metadata", None) or {}
    frame = metadata.get("frame")
    if frame is None:
        if getattr(event, "stage", None) == PA_STAGE:
            raise ValueError(
                "p+A event without a frame record: the p+A builder works "
                "in the laboratory frame, so its x_F needs the "
                "nucleon-nucleon CM boost (regenerate the event with the "
                "frame-recording builder)")
        return FRAME_NN_CM
    if frame not in (FRAME_LAB, FRAME_NN_CM):
        raise ValueError(f"unknown event frame {frame!r}")
    return frame


def nn_cm_p4(event, p4: FourVector) -> FourVector:
    """The four-vector ``p4`` of a track of ``event`` in the NN CM frame.

    Lab-frame events are boosted with the event's recorded ``nn_cm_beta``
    and ``nn_cm_gamma``; CM-frame events return ``p4`` itself."""
    if event_frame(event) == FRAME_NN_CM:
        return p4
    metadata = event.metadata
    return lab_to_nn_cm(p4, float(metadata["nn_cm_beta"]),
                        float(metadata["nn_cm_gamma"]))


def nn_cm_xf(event, p4: FourVector, sqrt_s_nn: float) -> float:
    """Feynman x_F = 2 p_z^CM / sqrt(s_NN) of a track of ``event``.

    For a CM-frame event (p+p) this is exactly ``2.0 * float(p4.pz) /
    float(sqrt_s_nn)``, bit for bit."""
    return 2.0 * float(nn_cm_p4(event, p4).pz) / float(sqrt_s_nn)


def event_in_nn_cm(event: ExcitationEvent) -> ExcitationEvent:
    """The event with every four-vector in the NN CM frame.

    A CM-frame event is returned as the same object. A lab-frame event is
    copied with the incoming vectors, the sources, the spectators, and the
    final particles boosted, and ``metadata["frame"] = "nn_cm"``; the
    input event is not modified. Four-momentum conservation and the mass
    shells carry over (the boost is linear and preserves the invariants).
    """
    if event_frame(event) == FRAME_NN_CM:
        return event
    beta = float(event.metadata["nn_cm_beta"])
    gamma = float(event.metadata["nn_cm_gamma"])

    def boost(p4: FourVector) -> FourVector:
        return lab_to_nn_cm(p4, beta, gamma)

    metadata = dict(event.metadata)
    metadata["frame"] = FRAME_NN_CM
    return replace(
        event,
        incoming=[boost(p4) for p4 in event.incoming],
        sources=[replace(s, p4=boost(s.p4)) for s in event.sources],
        spectators=[replace(p, p4=boost(p.p4)) for p in event.spectators],
        final_particles=[replace(p, p4=boost(p.p4))
                         for p in event.final_particles],
        metadata=metadata,
    )


@dataclass(frozen=True)
class PAExcitationConfig:
    beam_momentum: float = 158.0
    production_mode: str = "regge"
    simple: SimpleExcitationParameters = SimpleExcitationParameters()
    regge: ReggeExcitationParameters = ReggeExcitationParameters()
    continuum: ContinuumParameters = ContinuumParameters()
    nucleus: NucleusSpec = field(default_factory=NucleusSpec)
    glauber_dr_fm: float = 0.05
    decay_sources: bool = True
    conservation_tolerance: float = 1e-10
    max_event_attempts: int = 100
    # Optional centrality slice on the wounded target-nucleon number K
    # (inclusive [lo, hi]). ``None`` keeps the minimum-bias draw. When set,
    # ``PAEventBuilder.generate`` samples ``(b, K)`` from the
    # interaction-weighted distribution restricted to the slice via
    # ``WoundedNucleonSampler.sample_event_in_range`` (the same
    # range-conditioned draw the chunked p+A worker uses for
    # ``GenerationConfig.target_wound_range``), so every event lands
    # inside the slice without an acceptance loop. The integer/range rules
    # mirror the ``GenerationConfig`` construction validation; the
    # non-zero restricted-probability rule is enforced at builder
    # construction, where the sampler that owns the closed-form slice
    # fraction exists.
    target_wound_range: tuple[int, int] | None = None

    def validate(self) -> None:
        if self.production_mode not in {"simple", "regge"}:
            raise ValueError("production_mode must be 'simple' or 'regge'")
        if not (math.isfinite(self.beam_momentum) and self.beam_momentum > 0.0):
            raise ValueError("beam_momentum must be finite and positive")
        if not (math.isfinite(self.glauber_dr_fm) and self.glauber_dr_fm > 0.0):
            raise ValueError("glauber_dr_fm must be finite and positive")
        if not (0.0 < self.conservation_tolerance <= 1e-6):
            raise ValueError("conservation_tolerance must lie in (0, 1e-6]")
        if int(self.max_event_attempts) < 1:
            raise ValueError("max_event_attempts must be positive")
        self._validate_target_wound_range()
        self.simple.validate()
        self.regge.validate()
        self.continuum.validate()

    def _validate_target_wound_range(self) -> None:
        """Construction rules for the optional centrality slice.

        Mirrors the ``GenerationConfig.target_wound_range`` validation:
        integer ``lo >= 1`` and ``hi >= lo`` (integer-valued floats
        accepted, booleans and fractional values rejected), ``lo <= A``
        (the truncated-at-A sampler assigns ``K > A`` zero probability;
        ``hi`` beyond the densest support is harmless), and the
        point-target rule ``K = 1`` inside every requested range. The
        restricted-probability check runs in :class:`PAEventBuilder`
        construction, where the sampler that owns the closed-form slice
        fraction exists.
        """
        rng = self.target_wound_range
        if rng is None:
            return
        if isinstance(rng, (str, bytes)) or not isinstance(rng, (tuple, list)):
            raise ValueError(
                "PAExcitationConfig target_wound_range must be a "
                f"(lo, hi) pair, got {rng!r}"
            )
        if len(rng) != 2:
            raise ValueError(
                "PAExcitationConfig target_wound_range must be a "
                f"(lo, hi) pair, got length {len(rng)}"
            )
        lo, hi = rng
        for name, value in (("lo", lo), ("hi", hi)):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"PAExcitationConfig target_wound_range {name} must be "
                    f"an integer, got {value!r}"
                )
            if isinstance(value, (int, np.integer)):
                continue
            if (isinstance(value, (float, np.floating))
                    and math.isfinite(float(value))
                    and int(value) == value):
                continue
            raise ValueError(
                f"PAExcitationConfig target_wound_range {name} must be an "
                f"integer, got {value!r}"
            )
        lo_i, hi_i = int(lo), int(hi)
        if lo_i < 1 or hi_i < lo_i:
            raise ValueError(
                "PAExcitationConfig target_wound_range must satisfy "
                f"1 <= lo <= hi, got ({lo_i}, {hi_i})"
            )
        cap_a = int(self.nucleus.A)
        if cap_a == 1 and not (lo_i <= 1 <= hi_i):
            raise ValueError(
                "PAExcitationConfig point target (A=1) always has K=1; "
                f"target_wound_range must include 1, got ({lo_i}, {hi_i})"
            )
        if lo_i > cap_a:
            raise ValueError(
                "PAExcitationConfig target_wound_range "
                f"({lo_i}, {hi_i}) is unreachable for nucleus "
                f"{self.nucleus.name} (A={cap_a}): lo > A, and the "
                "truncated-at-A sampler assigns K > A zero probability."
            )
        object.__setattr__(self, "target_wound_range", (lo_i, hi_i))


class PAEventBuilder:
    """Generate p+A events with one cumulatively excited projectile source.

    Generation-attempt bookkeeping (p+A acceptance audit)
    -----------------------------------------------------
    ``generate`` redraws the whole Glauber event (impact parameter and
    wounded-nucleon multiplicity) at every attempt, so a failed kinematic
    construction resamples the geometry as well as the source masses. The
    following instance-level counters make that resampling measurable and
    are reset by :meth:`reset_generation_counters` (call it between
    samples for deterministic audits):

    - ``generation_attempts``: total Glauber draws (attempts), i.e. the
      number of ``(b, nu)`` samples consumed.
    - ``generation_accepted``: accepted events.
    - ``generation_rejected``: events that exhausted
      ``max_event_attempts``; ``generate`` raises ``RuntimeError`` for
      each of them.
    - ``failure_reason_counts``: failed attempts counted per exception
      class name (``"ValueError"``, ``"RuntimeError"``, ...).
    - ``accepted_generation_attempts``: the 1-based attempt index of
      every accepted event (``1`` when the first attempt succeeds).
    - ``glauber_draws``: one ``(b_fm, wounds, accepted)`` tuple per draw
      in draw order; ``accepted`` is true only for the draw that produced
      an accepted event. The full draw list is the original Glauber
      distribution at the same geometry, sampled without kinematic
      rejection, with the same draw count as the generation itself.

    Every accepted event carries the per-event metadata
    ``generation_attempt`` (the 1-based attempt count) and
    ``generation_last_failure_reason`` (the last failed attempt's
    ``"<ExceptionClass>: <message>"`` when any attempt failed before
    acceptance, ``None`` otherwise), mirroring ``event_builder_pp``.

    Centrality slices (``PAExcitationConfig.target_wound_range``)
    ------------------------------------------------------------
    When ``target_wound_range`` is set, the draw inside ``generate`` is
    the range-conditioned
    ``WoundedNucleonSampler.sample_event_in_range(lo, hi)`` (the same
    measure the chunked p+A worker uses for
    ``GenerationConfig.target_wound_range``), so every draw lands inside
    the slice and the per-event cost is independent of the slice's
    minimum-bias rarity. The resampling loop, the counters, and the
    metadata are unchanged; with ``target_wound_range=None`` the stream
    is byte-identical to the pre-slice builder (pinned by regression
    test).
    """

    def __init__(
        self,
        config: PAExcitationConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.glauber = WoundedNucleonSampler(
            config.nucleus, self.rng, dr_fm=config.glauber_dr_fm
        )
        if config.target_wound_range is not None:
            # Non-zero restricted probability (the ``MIN_WOUND_RANGE_FRACTION
            # = 0.0`` semantics of the ``GenerationConfig`` rule): any slice
            # with strictly positive interaction-weighted probability is
            # generable at two uniform draws per event, so only slices whose
            # restricted probability vanishes (structurally, or through
            # double-precision underflow of the analytic fraction) are
            # refused.
            lo, hi = config.target_wound_range
            fraction = self.glauber.fraction_for_wound_range(lo, hi)
            if fraction <= MIN_WOUND_RANGE_FRACTION:
                raise ValueError(
                    "PAExcitationConfig target_wound_range "
                    f"({lo}, {hi}) is unreachable for nucleus "
                    f"{config.nucleus.name} (A={config.nucleus.A}): the "
                    f"interaction-weighted slice probability is "
                    f"{fraction:.3e}, identically zero within double "
                    "precision, so the conditioned sampler has "
                    "nothing to draw."
                )
        self.decayer = SourceDecayer(config.continuum, self.rng)
        # The NN CM boost of the lab-frame events (recorded per event; no
        # random number is consumed).
        self.nn_cm_beta, self.nn_cm_gamma, self.sqrt_s_nn = nn_cm_boost(
            config.beam_momentum)
        self.reset_generation_counters()

    def reset_generation_counters(self) -> None:
        """Reset the generation-attempt bookkeeping to its empty state."""
        self.generation_attempts = 0
        self.generation_accepted = 0
        self.generation_rejected = 0
        self.failure_reason_counts: dict[str, int] = {}
        self.accepted_generation_attempts: list[int] = []
        self.glauber_draws: list[tuple[float, int, bool]] = []

    def _target_identities(self, wounds: int) -> tuple[list[int], list[int]]:
        spec = self.config.nucleus
        hit_protons = int(self.rng.hypergeometric(spec.Z, spec.A - spec.Z, wounds))
        struck = [2212] * hit_protons + [2112] * (wounds - hit_protons)
        self.rng.shuffle(struck)
        spectators = [2212] * (spec.Z - hit_protons)
        spectators += [2112] * ((spec.A - spec.Z) - (wounds - hit_protons))
        self.rng.shuffle(spectators)
        return struck, spectators

    def _attempt(self, event_id: int, impact_parameter: float, wounds: int) -> ExcitationEvent:
        cfg = self.config
        struck, spectator_ids = self._target_identities(wounds)
        beam_mass = get_mass(2212)
        beam_energy = math.sqrt(beam_mass * beam_mass + cfg.beam_momentum ** 2)
        projectile = FourVector(beam_energy, 0.0, 0.0, cfg.beam_momentum)
        incoming = [projectile]
        target_inputs: list[FourVector] = []
        for pid in struck + spectator_ids:
            target = FourVector(get_mass(pid), 0.0, 0.0, 0.0)
            incoming.append(target)
            if len(target_inputs) < len(struck):
                target_inputs.append(target)

        parameters = cfg.simple if cfg.production_mode == "simple" else cfg.regge
        target_sources: list[SourceRecord] = []
        history: list[dict] = []
        projectile_qn = QuantumNumbers(1, 1, 0)
        for index, (target_pid, target_p4) in enumerate(zip(struck, target_inputs), start=1):
            target_qn = QuantumNumbers(1, 1 if target_pid == 2212 else 0, 0)
            pair = projectile + target_p4
            root_s = pair.mass()
            min_projectile = max(projectile.mass(), minimum_excited_mass(projectile_qn))
            min_target = minimum_excited_mass(target_qn)
            sampler = SourceMassSampler(
                root_s,
                min_projectile,
                min_target,
                cfg.production_mode,
                parameters,
                self.rng,
            )
            m_projectile, m_target = sampler.sample()
            projectile_out, target_out, t = scatter_two_body(
                pair,
                projectile,
                m_projectile,
                m_target,
                transfer_slope(pair.mass2(), cfg.production_mode, parameters),
                self.rng,
            )
            history.append({
                "collision_index": index,
                "incoming_projectile": projectile.as_dict(),
                "outgoing_projectile": projectile_out.as_dict(),
                "projectile_mass": m_projectile,
                "target_mass": m_target,
                "target_pdg": target_pid,
                "t": t,
                "sqrt_s": root_s,
            })
            target_sources.append(
                SourceRecord(
                    source_id=index,
                    role="target",
                    p4=target_out,
                    quantum_numbers=target_qn,
                    collision_index=index,
                    metadata={
                        "sampled_mass": m_target,
                        "target_pdg": target_pid,
                        "t": t,
                    },
                )
            )
            projectile = projectile_out

        projectile_source = SourceRecord(
            source_id=0,
            role="projectile",
            p4=projectile,
            quantum_numbers=projectile_qn,
            collision_index=wounds,
            metadata={
                "sampled_mass": projectile.mass(),
                "n_encounters": wounds,
                "history": history,
            },
        )
        sources = [projectile_source] + target_sources
        spectators: list[FinalParticle] = []
        for i, pid in enumerate(spectator_ids):
            spectators.append(
                FinalParticle(
                    particle_id=i,
                    pdg_id=pid,
                    p4=FourVector(get_mass(pid), 0.0, 0.0, 0.0),
                    source_id=-1,
                    parent_id=None,
                    channel_family="spectator",
                    generation=0,
                )
            )

        event = ExcitationEvent(
            event_id=int(event_id),
            stage=PA_STAGE,
            incoming=incoming,
            incoming_quantum_numbers=QuantumNumbers(
                cfg.nucleus.A + 1, cfg.nucleus.Z + 1, 0
            ),
            sources=sources,
            spectators=spectators,
            metadata={
                "system": f"p+{cfg.nucleus.name}",
                "production_mode": cfg.production_mode,
                "exchange_class": "P",
                "impact_parameter_fm": impact_parameter,
                "target_wounds": wounds,
                "struck_target_pdgs": struck,
                "nuclear_initial_state": "independent_nucleons_at_rest_no_binding",
                # Laboratory frame (target nucleons at rest); the observables
                # read the tracks through nn_cm_p4 / nn_cm_xf / event_in_nn_cm.
                "frame": FRAME_LAB,
                "nn_cm_beta": self.nn_cm_beta,
                "nn_cm_gamma": self.nn_cm_gamma,
                "sqrt_s_nn": self.sqrt_s_nn,
            },
        )
        if cfg.decay_sources:
            particle_id = len(spectators)
            for source in sources:
                daughters = self.decayer.decay(source, particle_id)
                event.final_particles.extend(daughters)
                particle_id += len(daughters)
        event.validate(cfg.conservation_tolerance)
        return event

    def generate(self, event_id: int = 0) -> ExcitationEvent:
        wound_range = self.config.target_wound_range
        last_error: Exception | None = None
        for attempt in range(1, int(self.config.max_event_attempts) + 1):
            if wound_range is not None:
                # Range-conditioned draw: b and K come from the
                # interaction-weighted measure restricted to the slice, so
                # every draw lands inside [lo, hi] by construction. The
                # resampling loop and the counters are unchanged.
                lo, hi = wound_range
                b, wounds = self.glauber.sample_event_in_range(lo, hi)
            else:
                b, wounds = self.glauber.sample_event()
            self.generation_attempts += 1
            try:
                event = self._attempt(event_id, b, wounds)
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                reason = type(exc).__name__
                self.failure_reason_counts[reason] = (
                    self.failure_reason_counts.get(reason, 0) + 1
                )
                self.glauber_draws.append((b, wounds, False))
                last_error = exc
                continue
            self.generation_accepted += 1
            self.accepted_generation_attempts.append(attempt)
            self.glauber_draws.append((b, wounds, True))
            event.metadata["generation_attempt"] = attempt
            event.metadata["generation_last_failure_reason"] = (
                None
                if last_error is None
                else f"{type(last_error).__name__}: {last_error}"
            )
            return event
        self.generation_rejected += 1
        raise RuntimeError(
            f"failed to generate a valid p+A event after "
            f"{self.config.max_event_attempts} attempts: {last_error}"
        )
