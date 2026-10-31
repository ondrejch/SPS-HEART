"""Resonance-only p+p event builder for run R23 (Round 2, card 13).

Task ``TASK-20260913-r2-run-r23-resonance-only-pp-ablation`` (plan
section 8 run R23, line 567; the Round 2 card set, card 13).  The
builder is the R23 ablation object: it reuses the legacy staged p+p
builder's source/exchange layer bit-identically (the M8 frozen
ten-parameter set on the two-source ``(B, Q, S) = (1, 1, 0)`` topology)
and replaces only the low-mass decay kernel: sources at or below the
M8-frozen ``continuum.match_mass`` boundary decay through the card-6/7
``ResonanceDecayer`` over the card-9 mixture model, sources above the
boundary keep the legacy ``SourceDecayer`` (H3) kernel.  The legacy
builders and the ``SourceDecayer`` are read here, never modified.

Source/exchange layer (bit-identical by construction)
----------------------------------------------------

The per-event generator is the legacy staged convention,
``numpy.random.default_rng(numpy.random.SeedSequence(seed,
spawn_key=(event_id,)))``, the same rule the R01 driver pins.  The
``SourceMassSampler`` and ``scatter_two_body`` draws are consumed from
that one shared generator in exactly the legacy order (the mass-pair
cell draw, the two in-cell mass uniforms, the Regge transfer draw, the
scattering direction draw), so the topology selection, the exchange
masses ``(m1, m2)``, the momentum transfer ``t``, and the realized
source four-momenta are bit-identical to ``PPEventBuilder`` at the same
``(seed, event_id)``.  The closed ``SUBSTREAM_LABELS`` entries
``topology_selection``, ``exchange_kinematics``, and ``spectral_state``
are the draw-attribution labels of these layers (plan section 6, task
5; the labels are attribution only and no label stream is separated
here, because the legacy builder draws them from one shared stream and
the bit-identity claim is about that stream).

Kernel switch (per realized source, in source order)
----------------------------------------------------

For each realized source, in source order 0 then 1:

- source mass ``>`` boundary: the legacy ``SourceDecayer`` decays the
  source on the shared generator, constructed once per event exactly as
  ``PPEventBuilder`` does.  The branch is stateless apart from the
  generator, so given the generator state the legacy builder would hand
  it, the decay is bit-identical to the legacy builder at the same
  seed.
- source mass ``<=`` boundary (inclusive): the catalog-driven kernel.
  The card-9 mixture model resolves the record and the card-6/7
  ``ResonanceDecayer`` decays the source:

  1. the open set is the catalog records with the source's
     ``(B, Q, S)`` whose truncated spectral window
     ``[mass_range[0], min(mass_range[1], M)]`` is non-empty (exactly
     ``mass_range[0] <= M``), in catalog record order; asserted
     non-empty (no fallback decay path, the card-9 discipline);
  2. one uniform over the open set draws the record (equal record
     weights; the card-5 isospin charge weights act inside each
     record).  The draw runs on the one-step spawned child
     ``SeedSequence(stream_id).spawn(1)[0]`` of the ``decay_channel``
     labeled stream id -- the card-9 scope decision #4 stream split --
     so the record selection never shares a uniform with the kernel's
     first vertex-channel draw;
  3. the record is realized at the source mass ``M``: the R23
     adaptation of the card-9 mass rule.  In the isolated-source grids
     of R21/R22 the card-3 spectral draw realized the record below the
     grid mass because the record itself became the source; in R23 the
     source mass is the exchange-layer draw and the whole source
     four-momentum must be decayed (the event's binned spectra are
     built from the final particles), so re-drawing the record below
     ``M`` would discard the mass-energy
     ``gamma * (M - m_record) * (1, beta)`` and break the event
     bookkeeping.  The record's channels are therefore evaluated at the
     source mass through the card-3/4 width law (no extra uniform is
     consumed on this path);
  4. the record decays through the card-6/7 kernel via
     ``decay_source`` with the source's own four-momentum (invariant
     ``M``), so the daughter four-momenta sum to the source four-momentum
     and the event four-momentum bookkeeping is preserved.  The kernel
     consumes the labeled ``decay_channel`` / ``decay_kinematics``
     stream ids re-derived from
     ``derive_substreams(master_seed, replica_index, event_id)`` with
     the Round 2 master seed; one decayer and one spawned mixture
     generator serve one event, consumed by the below-boundary sources
     in source order and across generation attempts (deterministic for
     the same code path; the shared stream advances across attempts
     exactly as the legacy retry does).
- a drawn record that is already in the stable set has no decay (plan
  section 8.4 step 7): the source contributes itself as the single
  final particle at its own four-momentum, with no producing vertex --
  the card-9 stable-identity semantics on the moving source.  Such a
  final particle is a direct source product, never a resonance
  feed-down.  (This is the R21/R22 scan convention carried into R23;
  the source mass is a continuum draw, so the identity final is not on
  its PDG shell.  The builder's conservation checks are the
  pre-registered ``(B, Q)`` event balance and the per-source
  ``(B, Q, S)`` balance; ``ExcitationEvent.validate``'s PDG on-shell
  pass is not run by this builder.)

Feed-down lineage
-----------------

Every final particle of the catalog-driven branch carries its direct
parent: ``FinalParticle.parent_id`` is the index of the producing
vertex in the returned vertex list (the full ancestry is recoverable
from the vertex list).  The legacy branch keeps the legacy convention
(``parent_id == source_id``).  Per-event and per-replica resonant
feed-down fractions follow the card-9 convention: a final particle of a
catalog-driven source is *resonant feed-down* when its producing
vertex's parent is a resonance other than the drawn root record.

Conservation
------------

Every p+p event conserves total ``B = 2`` and ``Q = 2`` exactly, and
every source's final set conserves the source's ``(B, Q, S)``: checked
per event, never repaired; a violation raises
:class:`ConservationViolationError` and stops the run.  The kernel's
own per-vertex ``(B, Q, S)``, four-momentum-closure, and on-shell
protections propagate unchanged (``NoOpenChannelError``,
``DecayCycleError``, ``KinematicsError`` stop the run; they are
``RuntimeError`` subclasses and are never caught by the retry, which
handles only the legacy ``ValueError`` triggers exactly as the legacy
builder does).

Boundary constant
-----------------

``KERNEL_BOUNDARY_M8`` records the M8-frozen
``continuum.match_mass = 2.680194110727528 GeV`` (the
``m8_frozen_pp_parameters`` block of
``config/v2_singlet/legacy_v1_manifest.json``).  The builder reads the
boundary from the config it is handed
(``config.continuum.match_mass``), which the R23 driver pins to the M8
frozen value through the manifest hash verification; the constant is
the documented cross-check of the pre-registered boundary.

Card-6 blend routing (R32; additive)
------------------------------------

Run R32 (Round 3, card 9,
``TASK-20260918-r3-run-r32-spectral-systematic-pp``) extends the
builder additively with a ``blend`` parameter: the legacy source and
exchange layers are untouched, and the per-source kernel switch at the
boundary is replaced by the card-6 blend routing
(:func:`na49_mixing.spectral_density.route_source_decay`) evaluated at
the frozen source mass:

- ``blend = None`` (the default) keeps the R23 kernel switch and the
  R23 behavior byte-stable: no blend state is built, no extra
  generator is constructed, and every draw sequence is the R23
  sequence (asserted by the card-13 pins and by the R32 regression
  pin, which regenerates the sealed R23 replica-0 stream's first 1,000
  event hashes bit for bit).
- ``blend = BlendSpec(boundary, ramp_width)`` routes each realized
  source exactly as the card-6 contract defines: ``M < B`` the catalog
  mixture (zero routing uniforms, so the catalog-path draw sequence is
  byte-identical to the no-blend convention at the same seed); ``B <=
  M < B + W`` one routing uniform on the spawned ``decay_channel``
  child consumed before any record draw, continuum exactly when the
  draw ``u < w(M)``; ``M >= B + W`` the continuum family mixture (zero
  routing uniforms).  The routing never consults the blend's
  boundary-matching strength factor: the factor fixes the strength of
  the blended density only (the card-6 double-counting identity) and
  the routed Monte Carlo never calls
  :func:`blended_density`/:func:`blended_normalization_report`.

The continuum route realizes the single activated continuum family of
the source ``(B, Q, S)`` (asserted exactly one at construction, the
R31 convention) at the frozen source four-momentum: the continuum
state is the state at the source mass (no family-mass redraw), one
allowed decay class is drawn uniformly over the family's
``allowed_decay_classes`` (one uniform on the spawned
``decay_channel`` child, after the routing uniform), and the root
split follows the kernel's own two-body convention in the source rest
frame (the isotropic direction from two uniforms, the daughter masses
in channel order, each resonance daughter from the card-3
``draw_mass`` inside its on-shell budget window with the partner at
its pole mass, each stable daughter at its pole mass, the Kaellen
decay momentum and the exact energy split), boosted to the lab frame
with the source velocity (the kernel's own boost convention).  Every
resonance daughter cascades through the Round 2 kernel
(``ResonanceDecayer.decay_resonance``) on the labeled substreams.  The
root split is recorded as vertex 0 of the source's vertex list with
``parent_pdg = 0`` (``CONTINUUM_ROOT_SENTINEL``): the continuum state
is not a catalog resonance, so the root-split daughters are direct
source products and only the cascade products are resonant feed-down
(the corrected lineage convention of Round 3 card 3).

The ``families`` argument carries the continuum family records
(:func:`na49_mixing.spectral_density.families_from_json`); it is
required exactly when ``blend`` is given and refused otherwise.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no event generation at scale, no external tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .continuum_channels import SourceDecayer, minimum_excited_mass
from .event_builder_pp import PPExcitationConfig
from .exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
    SourceMassSampler,
    transfer_slope,
)
from .excitation_kinematics import FourVector, scatter_two_body
from .hadron_catalog import DecayChannel, HadronCatalog, HadronRecord
from .pdg_database import get_particle
from .quantum_numbers import QuantumNumbers
from .reaction import Reaction
from . import resonance_decay as resonance_kernel
from .resonance_decay import ResonanceDecayer
from .resonance_widths import decay_momentum, draw_mass
from .rng_substreams import derive_substreams
from .source_event import ExcitationEvent, FinalParticle, SourceRecord
from .spectral_density import (
    BlendSpec,
    ContinuumFamily,
    _daughter_masses,
    active_families,
    route_source_decay,
)

__all__ = [
    "KERNEL_BOUNDARY_M8",
    "ROUND2_MASTER_SEED",
    "KERNEL_SUBLABELS",
    "KERNEL_LABEL",
    "LEGACY_LABEL",
    "CONTINUUM_LABEL",
    "CONTINUUM_ROOT_SENTINEL",
    "CONTINUUM_FAMILY",
    "STABLE_IDENTITY_FAMILY",
    "ConservationViolationError",
    "SourceKernelRecord",
    "ResonancePPResult",
    "ResonancePPBuilder",
]

#: The M8-frozen ``continuum.match_mass`` boundary (the
#: ``m8_frozen_pp_parameters`` block of
#: ``config/v2_singlet/legacy_v1_manifest.json``).  Sources at or below
#: the boundary decay through the catalog-driven kernel (inclusive);
#: sources above it keep the legacy ``SourceDecayer``.  The builder
#: reads the operative boundary from the handed config, which the R23
#: driver pins to this value through the manifest hash verification.
KERNEL_BOUNDARY_M8 = 2.680194110727528

#: The Round 2 master seed of the kernel/mixture layer (the card-9/10
#: seed convention; the legacy source layer carries its own per-replica
#: seed, the M8 rule ``20260913 + 40000 + k``).
ROUND2_MASTER_SEED = 20260913

#: The closed labels the catalog-driven branch binds (no extension of
#: ``SUBSTREAM_LABELS``; the card-6/7 kernel's own labels).
KERNEL_SUBLABELS = ("decay_channel", "decay_kinematics")

#: The kernel tags of the per-source diagnostics.
KERNEL_LABEL = "resonance"
LEGACY_LABEL = "legacy"

#: The ``channel_family`` of a stable-identity final particle (the
#: card-9 stable-identity semantics: the source contributes itself, a
#: direct source product, never a resonance feed-down).
STABLE_IDENTITY_FAMILY = "source_stable"

#: The kernel tag of a continuum-routed source (the R32 blend routing:
#: the source decays through the card-6 continuum family mixture).
CONTINUUM_LABEL = "continuum"

#: The root-vertex ``parent_pdg`` sentinel of a continuum-routed source
#: (the R31 convention): the continuum state is not a catalog
#: resonance, so the root-split daughters are direct source products
#: and only the cascade products are resonant feed-down.
CONTINUUM_ROOT_SENTINEL = 0

#: The ``channel_family`` of a continuum root-split final particle (a
#: direct source product of the continuum route, never a resonance
#: feed-down; the cascade finals keep the kernel's
#: ``"resonance_decay"`` family).
CONTINUUM_FAMILY = "continuum_decay"

_EVENT_SEED_NOTE = (
    "numpy.random.default_rng(numpy.random.SeedSequence(seed, "
    "spawn_key=(event_id,))); the legacy staged p+p convention (the R01 "
    "driver's rule); one shared generator serves the topology "
    "selection, exchange kinematics, and spectral-state layers and the "
    "legacy decay branch"
)


class ConservationViolationError(RuntimeError):
    """The pre-registered conservation check of the R23 builder fired.

    Every p+p event must conserve total ``(B, Q, S) = (2, 2, 0)`` and
    every source its own ``(B, Q, S)``; the check runs per event and is
    never repaired.  A subclass of :class:`RuntimeError` (so the legacy
    ``ValueError``/``FloatingPointError`` retry of :meth:
    `ResonancePPBuilder.generate` never catches it): a conservation
    violation is a bug that stops the run.
    """


@dataclass(frozen=True)
class SourceKernelRecord:
    """Per-source diagnostics of one realized source's decay.

    ``kernel`` is ``"resonance"`` (the catalog-driven card-6/7 branch),
    ``"legacy"`` (the ``SourceDecayer`` H3 branch), or ``"continuum"``
    (the R32 blend routing's continuum family branch).
    ``boundary_new_kernel`` is the pre-registered switch decision
    (``mass <= boundary``).  ``record_pdg``/``record_name`` name the
    drawn mixture record on the catalog-driven branch; ``n_open_records``
    is the open-set size at the source mass.  The R32 blend routing
    adds three optional diagnostics (``None`` on the R23 paths):
    ``route`` is ``'catalog'`` or ``'continuum'`` for a blended builder
    (the routing census), ``blend_weight`` the ramp weight ``w(M)`` at
    the frozen source mass, and ``family_id`` the continuum family of a
    continuum-routed source.
    """

    source_id: int
    role: str
    mass: float
    kernel: str
    boundary_new_kernel: bool
    record_pdg: int | None = None
    record_name: str | None = None
    n_open_records: int = 0
    route: str | None = None
    blend_weight: float | None = None
    family_id: str | None = None


@dataclass
class ResonancePPResult:
    """One generated p+p event with its R23 decay diagnostics.

    ``event`` is the standard ``ExcitationEvent`` (sources, final
    particles, metadata).  ``source_records`` carries one
    :class:`SourceKernelRecord` per source in source order.
    ``kernel_vertices`` carries, per source in source order, the
    card-6/7 vertex list of the catalog-driven branch (empty for the
    stable-identity path, ``None`` for the legacy branch), so the
    feed-down lineage is recoverable without re-decaying.
    """

    event: ExcitationEvent
    source_records: list[SourceKernelRecord] = field(default_factory=list)
    kernel_vertices: list = field(default_factory=list)


class ResonancePPBuilder:
    """Generate two color-neutral ``(B, Q, S) = (1, 1, 0)`` sources and
    decay them through the R23 kernel switch.

    The builder mirrors ``PPEventBuilder``'s construction and retry
    semantics exactly on the shared per-event generator; only the
    below-boundary decay branch is new.  One builder serves one event
    (the driver constructs one per ``(seed, event_id)``, as the legacy
    staged pipeline does).
    """

    def __init__(
        self,
        config: PPExcitationConfig,
        catalog: HadronCatalog,
        seed: int,
        event_id: int,
        master_seed: int = ROUND2_MASTER_SEED,
        replica_index: int = 0,
        blend: BlendSpec | None = None,
        families: list[ContinuumFamily] | None = None,
    ) -> None:
        config.validate()
        if not isinstance(catalog, HadronCatalog):
            raise TypeError("catalog must be a HadronCatalog")
        self.config = config
        self.catalog = catalog
        self.seed = int(seed)
        self.event_id = int(event_id)
        self.master_seed = int(master_seed)
        self.replica_index = int(replica_index)

        # -- the shared per-event generator (the legacy convention) -----
        self.rng = np.random.default_rng(
            np.random.SeedSequence(self.seed, spawn_key=(self.event_id,))
        )
        self.reaction = Reaction(config.beam_momentum, 2212, 2212)
        self.sqrt_s = self.reaction.sqrt_s()
        proton_qn = QuantumNumbers(1, 1, 0)
        minimum = minimum_excited_mass(proton_qn)
        parameters = (
            config.simple if config.production_mode == "simple" else config.regge
        )
        self.mass_sampler = SourceMassSampler(
            self.sqrt_s, minimum, minimum, config.production_mode, parameters, self.rng
        )
        # The legacy branch: the same construction the legacy builder
        # makes, on the same shared generator.
        self.decayer = SourceDecayer(config.continuum, self.rng)

        # -- the kernel switch boundary ----------------------------------
        self.kernel_boundary = float(config.continuum.match_mass)
        if not (math.isfinite(self.kernel_boundary) and self.kernel_boundary > 0.0):
            raise ValueError(
                f"the kernel boundary (continuum.match_mass) must be "
                f"finite and positive, got {self.kernel_boundary!r}"
            )

        # -- the catalog-driven branch (its own labeled substreams) ------
        self._source_qn = (1, 1, 0)
        subs = derive_substreams(self.master_seed, self.replica_index, self.event_id)
        self._substreams = subs
        # The card-9 scope decision #4 stream split: the mixture record
        # draw consumes the one-step spawned child of the decay_channel
        # labeled id; the kernel keeps the labeled ids themselves.
        self._rng_record = np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
        )
        self._resonance_decayer = ResonanceDecayer(
            catalog, self.master_seed, self.replica_index, self.event_id
        )
        # The (B, Q, S) records of the catalog in record order with their
        # open-set thresholds (the truncated-window rule reduces to
        # ``mass_range[0] <= M``).
        self._class_records: list[tuple[float, HadronRecord]] = []
        for record in catalog.records():
            if (
                int(record.B),
                int(record.Q),
                int(record.S),
            ) != self._source_qn:
                continue
            self._class_records.append((float(record.mass_range[0]), record))
        if not self._class_records:
            raise ValueError(
                "the catalog carries no record with the p+p source "
                f"quantum numbers {self._source_qn}"
            )
        self._stable = catalog.stable_states()

        # -- the card-6 blend routing (R32; additive; None keeps the
        #    R23 behavior byte-stable) -----------------------------------
        self.blend = blend
        if blend is None:
            if families is not None:
                raise TypeError(
                    "resonance_pp_builder: families is only valid "
                    "together with a BlendSpec blend (the R32 card-6 "
                    "routing); pass blend=None alone for the R23 "
                    "byte-stable behavior"
                )
        else:
            if not isinstance(blend, BlendSpec):
                raise TypeError(
                    "resonance_pp_builder: blend must be a "
                    f"na49_mixing.spectral_density.BlendSpec, got "
                    f"{type(blend).__name__}"
                )
            if families is None:
                raise TypeError(
                    "resonance_pp_builder: families is required when a "
                    "BlendSpec blend is given (the card-6 routing draws "
                    "the continuum family mixture of the source quantum "
                    "numbers)"
                )
            blend_families = list(families)
            if not blend_families:
                raise ValueError(
                    "resonance_pp_builder: the blend routing needs a "
                    "non-empty continuum family list"
                )
            for family in blend_families:
                if not isinstance(family, ContinuumFamily):
                    raise TypeError(
                        "resonance_pp_builder: families items must be "
                        f"ContinuumFamily records, got "
                        f"{type(family).__name__}"
                    )
            actives = active_families(blend_families, self._source_qn)
            if len(actives) != 1:
                raise ValueError(
                    "resonance_pp_builder: the blend routing expects "
                    "exactly one activated continuum family for the "
                    f"source quantum numbers {self._source_qn}, found "
                    f"{[f.family_id for f in actives]}"
                )
            self._blend_families = blend_families
            self._blend_family = actives[0]
            # The spawned decay_kinematics child: the continuum root
            # split's direction and daughter-mass draws (the routing
            # and class draws share the spawned decay_channel child
            # with the catalog branch's record draw).
            self._rng_kinematics = np.random.default_rng(
                np.random.SeedSequence(
                    subs.stream_for("decay_kinematics")
                ).spawn(1)[0]
            )
            # Pure-compute preparation of the allowed decay classes
            # (no draws): each class's daughter records and their
            # card-3 daughter-mass maps.
            self._blend_classes: list[tuple[int, int, HadronRecord, HadronRecord, dict, dict]] = []
            for cls in self._blend_family.allowed_decay_classes:
                d0, d1 = (int(v) for v in str(cls).strip("()").split(","))
                rec0 = self.catalog.get(d0)
                rec1 = self.catalog.get(d1)
                self._blend_classes.append((
                    d0, d1, rec0, rec1,
                    _daughter_masses(self.catalog, rec0),
                    _daughter_masses(self.catalog, rec1),
                ))
            if not self._blend_classes:
                raise ValueError(
                    "resonance_pp_builder: the activated continuum "
                    f"family {self._blend_family.family_id} carries no "
                    "allowed decay class; the continuum route has no "
                    "realization"
                )

        # -- the incoming pair (the legacy construction) ------------------
        proton_mass = self.catalog_pole_mass(2212)
        pin = math.sqrt(
            max(self.sqrt_s * self.sqrt_s / 4.0 - proton_mass * proton_mass, 0.0)
        )
        ein = self.sqrt_s / 2.0
        self.incoming = [
            FourVector(ein, 0.0, 0.0, pin),
            FourVector(ein, 0.0, 0.0, -pin),
        ]

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def catalog_pole_mass(self, pdg_id: int) -> float:
        """The pole mass of *pdg_id* (the catalog record's, else the
        vendored PDG table's)."""

        try:
            return float(self.catalog.get(int(pdg_id)).pole_mass)
        except KeyError:
            entry = get_particle(int(pdg_id))
            if entry is None:
                raise KeyError(
                    f"pdg_id {pdg_id} is neither a catalog record nor in "
                    "the vendored PDG table"
                ) from None
            return float(entry.mass)

    def open_records(self, mass: float) -> list[HadronRecord]:
        """The card-9 open set at *mass*: the records with the source's
        ``(B, Q, S)`` whose truncated spectral window
        ``[mass_range[0], min(mass_range[1], M)]`` is non-empty, in
        catalog record order.  The rule reduces to
        ``mass_range[0] <= M``."""

        out: list[HadronRecord] = []
        for threshold, record in self._class_records:
            if threshold <= float(mass):
                out.append(record)
        return out

    # ------------------------------------------------------------------
    # The kernel switch (public: the pre-registered dispatch surface)
    # ------------------------------------------------------------------
    def decay_source(
        self, source: SourceRecord, first_particle_id: int = 0
    ) -> tuple[list[FinalParticle], list | None, SourceKernelRecord]:
        """Decay one realized source through the R23 kernel switch, or
        through the card-6 blend routing when a ``BlendSpec`` blend was
        handed to the builder.

        The legacy branch decays on the shared generator (the state the
        legacy builder would hand it); the catalog-driven branch draws
        the mixture record and decays through the card-6/7 kernel on
        the labeled substreams.  With a blend, the per-source switch at
        the boundary is replaced by
        :func:`na49_mixing.spectral_density.route_source_decay`
        evaluated at the frozen source mass (the card-6 contract: the
        routing never consults the blend's boundary-matching strength
        factor).  Returns the final particles, the vertex list of the
        kernel-driven branch (the catalog-driven list, the continuum
        root-split + cascade list, empty or ``None`` on the other
        paths), and the per-source diagnostics record.
        """

        if self.blend is not None:
            return self._decay_blended(source, first_particle_id)
        mass = float(source.mass)
        new_kernel = mass <= self.kernel_boundary
        if not new_kernel:
            daughters = self.decayer.decay(source, first_particle_id)
            record = SourceKernelRecord(
                source_id=int(source.source_id),
                role=str(source.role),
                mass=mass,
                kernel=LEGACY_LABEL,
                boundary_new_kernel=False,
            )
            return daughters, None, record
        return self._decay_resonance(source, first_particle_id)

    def _decay_blended(
        self, source: SourceRecord, first_particle_id: int
    ) -> tuple[list[FinalParticle], list, SourceKernelRecord]:
        """The card-6 blend routing evaluated at the frozen source mass
        (the R32 extension).

        ``M < B``: the catalog route (zero routing uniforms; the
        catalog-path draw sequence is byte-identical to the no-blend
        convention at the same seed).  ``B <= M < B + W``: one routing
        uniform on the spawned ``decay_channel`` child before any
        record draw; continuum exactly when the draw ``u < w(M)``.
        ``M >= B + W``: the continuum route (zero routing uniforms).
        """

        mass = float(source.mass)
        blend = self.blend
        if not isinstance(blend, BlendSpec):  # defensive; decay_source guards
            raise RuntimeError(
                "resonance_pp_builder: _decay_blended reached without a "
                "BlendSpec blend (an internal dispatch violation)"
            )
        route = route_source_decay(
            self._source_qn,
            mass,
            blend,
            self.catalog,
            self._blend_families,
            self._rng_record,
        )
        weight = float(blend.weight(mass))
        if route == "catalog":
            finals, vertices, record = self._decay_resonance(
                source, first_particle_id, route=route, blend_weight=weight
            )
            return finals, vertices, record
        return self._decay_continuum(source, first_particle_id, weight)

    def _decay_continuum(
        self, source: SourceRecord, first_particle_id: int, blend_weight: float
    ) -> tuple[list[FinalParticle], list, SourceKernelRecord]:
        """The continuum route: the activated continuum family of the
        source ``(B, Q, S)`` realized at the frozen source four-momentum.

        The continuum state is the state at the source mass (the plan
        section 9.3 blend evaluated at the source mass; no family-mass
        redraw).  One allowed decay class is drawn uniformly over the
        family's ``allowed_decay_classes`` (one uniform on the spawned
        ``decay_channel`` child, after the routing uniform), and the
        root split follows the kernel's own two-body convention in the
        source rest frame: the isotropic direction from two uniforms on
        the spawned ``decay_kinematics`` child, the daughter masses in
        channel order (each resonance daughter from the card-3
        ``draw_mass`` inside its on-shell budget window with the
        partner at its pole mass, each stable daughter at its pole
        mass), the Kaellen decay momentum and the exact energy split.
        The rest-frame daughters are boosted to the lab frame with the
        source velocity (the kernel's own boost convention), and every
        resonance daughter cascades through the Round 2 kernel
        (``ResonanceDecayer.decay_resonance``) on the labeled
        substreams.

        The root split is vertex 0 of the returned list with
        ``parent_pdg = CONTINUUM_ROOT_SENTINEL`` (0): the root-split
        daughters are direct source products and only the cascade
        products are resonant feed-down.
        """

        mass = float(source.mass)
        source_p4 = source.p4
        energy = float(source_p4.e)
        beta = (
            float(source_p4.px) / energy,
            float(source_p4.py) / energy,
            float(source_p4.pz) / energy,
        )
        family = self._blend_family
        classes = self._blend_classes
        u_cls = float(self._rng_record.random())
        d0, d1, rec0, rec1, dmap0, dmap1 = classes[
            min(int(u_cls * len(classes)), len(classes) - 1)
        ]
        # -- the root split in the source rest frame ----------------------
        u_cos = float(self._rng_kinematics.random())
        u_phi = float(self._rng_kinematics.random())
        cos_theta = 2.0 * u_cos - 1.0
        phi = 2.0 * math.pi * u_phi
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        direction = (
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        )
        stable = self._stable
        res0 = d0 not in stable and not rec0.stable_for_decay
        res1 = d1 not in stable and not rec1.stable_for_decay
        m0_pole = float(rec0.pole_mass)
        m1_pole = float(rec1.pole_mass)
        if res0:
            lo0 = float(rec0.mass_range[0])
            hi0 = min(float(rec0.mass_range[1]), mass - m1_pole)
            drawn0 = draw_mass((lo0, hi0), rec0, dmap0, self._rng_kinematics)
            if drawn0 is None:
                raise RuntimeError(
                    "resonance_pp_builder: the continuum root daughter "
                    f"{rec0.name} (pdg {d0}) has an empty on-shell "
                    f"budget window {(lo0, hi0)!r} at source mass "
                    f"{mass!r} GeV; no fallback mass is implemented "
                    "(the run stops)"
                )
            mass0 = float(drawn0)
        else:
            mass0 = m0_pole
        if res1:
            lo1 = float(rec1.mass_range[0])
            hi1 = min(float(rec1.mass_range[1]), mass - mass0)
            drawn1 = draw_mass((lo1, hi1), rec1, dmap1, self._rng_kinematics)
            if drawn1 is None:
                raise RuntimeError(
                    "resonance_pp_builder: the continuum root daughter "
                    f"{rec1.name} (pdg {d1}) has an empty on-shell "
                    f"budget window {(lo1, hi1)!r} at source mass "
                    f"{mass!r} GeV; no fallback mass is implemented "
                    "(the run stops)"
                )
            mass1 = float(drawn1)
        else:
            mass1 = m1_pole
        q_star = float(decay_momentum(mass, mass0, mass1))
        energy0 = (mass * mass + mass0 * mass0 - mass1 * mass1) / (2.0 * mass)
        energy1 = (mass * mass - mass0 * mass0 + mass1 * mass1) / (2.0 * mass)
        rest0 = (
            energy0,
            q_star * direction[0],
            q_star * direction[1],
            q_star * direction[2],
        )
        rest1 = (
            energy1,
            -q_star * direction[0],
            -q_star * direction[1],
            -q_star * direction[2],
        )
        lab0 = resonance_kernel._boost_lab(rest0, beta)
        lab1 = resonance_kernel._boost_lab(rest1, beta)
        root_channel = DecayChannel(
            parent_pdg=CONTINUUM_ROOT_SENTINEL,
            daughters=(d0, d1),
            branching_fraction=1.0,
            orbital_angular_momentum=0,
        )
        root_vertex = resonance_kernel.DecayVertex(
            parent_pdg=CONTINUUM_ROOT_SENTINEL,
            parent_mass=mass,
            parent_four_momentum=(
                float(source_p4.e),
                float(source_p4.px),
                float(source_p4.py),
                float(source_p4.pz),
            ),
            channel=root_channel,
            charge_combination=(d0, d1),
            daughters=(
                resonance_kernel.DecayDaughter(
                    pdg_id=d0, four_momentum=lab0,
                    is_resonance=res0, parent_pdg=CONTINUUM_ROOT_SENTINEL,
                ),
                resonance_kernel.DecayDaughter(
                    pdg_id=d1, four_momentum=lab1,
                    is_resonance=res1, parent_pdg=CONTINUUM_ROOT_SENTINEL,
                ),
            ),
        )
        vertices: list = [root_vertex]
        finals: list[FinalParticle] = []

        def _emit(pdg_id: int, p4_tuple: tuple, is_resonance: bool) -> None:
            if is_resonance:
                cascade = self._resonance_decayer.decay_resonance(
                    int(pdg_id), p4_tuple
                )
                base = len(vertices)
                vertices.extend(cascade)
                for offset, vertex in enumerate(cascade):
                    for daughter in vertex.daughters:
                        if daughter.is_resonance:
                            continue
                        finals.append(FinalParticle(
                            particle_id=first_particle_id + len(finals),
                            pdg_id=int(daughter.pdg_id),
                            p4=FourVector(*daughter.four_momentum),
                            source_id=int(source.source_id),
                            parent_id=base + offset,
                            channel_family="resonance_decay",
                        ))
            else:
                finals.append(FinalParticle(
                    particle_id=first_particle_id + len(finals),
                    pdg_id=int(pdg_id),
                    p4=FourVector(*p4_tuple),
                    source_id=int(source.source_id),
                    parent_id=0,
                    channel_family=CONTINUUM_FAMILY,
                ))

        _emit(d0, lab0, res0)
        _emit(d1, lab1, res1)
        if not finals:
            raise RuntimeError(
                "resonance_pp_builder: the continuum decay of "
                f"{family.family_id} class ({d0}, {d1}) at source mass "
                f"{mass!r} GeV produced no final particles (the run "
                "stops)"
            )
        record = SourceKernelRecord(
            source_id=int(source.source_id),
            role=str(source.role),
            mass=mass,
            kernel=CONTINUUM_LABEL,
            boundary_new_kernel=mass <= self.kernel_boundary,
            route="continuum",
            blend_weight=float(blend_weight),
            family_id=str(family.family_id),
        )
        return finals, vertices, record

    def _decay_resonance(
        self,
        source: SourceRecord,
        first_particle_id: int,
        route: str | None = None,
        blend_weight: float | None = None,
    ) -> tuple[list[FinalParticle], list, SourceKernelRecord]:
        """The catalog-driven branch: the card-9 mixture over the open
        records at the source mass, decayed through the card-6/7 kernel
        at the source four-momentum.  The optional ``route`` and
        ``blend_weight`` stamps carry the card-6 routing diagnostics of
        a blended builder (``None`` on the R23 paths)."""

        mass = float(source.mass)
        records = self.open_records(mass)
        if not records:
            raise RuntimeError(
                "resonance_pp_builder: the open set at source quantum "
                f"numbers {self._source_qn} and mass {mass!r} GeV is "
                "empty; the pre-registered mixture model has no record "
                "to draw and no fallback decay path is implemented "
                "(the run stops)"
            )
        index = min(
            int(self._rng_record.random() * len(records)), len(records) - 1
        )
        mix_record = records[int(index)]
        diagnostics = SourceKernelRecord(
            source_id=int(source.source_id),
            role=str(source.role),
            mass=mass,
            kernel=KERNEL_LABEL,
            boundary_new_kernel=True,
            record_pdg=int(mix_record.pdg_id),
            record_name=str(mix_record.name),
            n_open_records=len(records),
            route=route,
            blend_weight=blend_weight,
        )
        if mix_record.stable_for_decay:
            # The card-9 stable-identity semantics on the moving source:
            # the source contributes itself as the single final particle
            # at its own four-momentum, with no producing vertex (plan
            # section 8.4 step 7; a direct source product, never a
            # resonance feed-down).
            finals = [
                FinalParticle(
                    particle_id=first_particle_id,
                    pdg_id=int(mix_record.pdg_id),
                    p4=source.p4,
                    source_id=int(source.source_id),
                    parent_id=None,
                    channel_family=STABLE_IDENTITY_FAMILY,
                )
            ]
            return finals, [], diagnostics
        vertices, finals = self._resonance_decayer.decay_source(
            resonance_kernel.SourceRecord(
                pdg_id=int(mix_record.pdg_id),
                four_momentum=(
                    float(source.p4.e),
                    float(source.p4.px),
                    float(source.p4.py),
                    float(source.p4.pz),
                ),
                B=int(self._source_qn[0]),
                Q=int(self._source_qn[1]),
                S=int(self._source_qn[2]),
                mass=mass,
                source_id=int(source.source_id),
            )
        )
        if not finals:
            raise RuntimeError(
                "resonance_pp_builder: the catalog-driven decay of "
                f"{mix_record.name} (pdg {int(mix_record.pdg_id)}) at "
                f"mass {mass!r} GeV produced no final particles (a "
                "non-stable record must decay; the run stops)"
            )
        for offset, final in enumerate(finals):
            final.particle_id = first_particle_id + offset
        return finals, vertices, diagnostics

    # ------------------------------------------------------------------
    # Event generation (the legacy structure, the switch inside)
    # ------------------------------------------------------------------
    def generate(self, stage: str | None = None) -> ResonancePPResult:
        """Generate one p+p event: the bit-identical source layer, the
        per-source kernel switch, and the pre-registered conservation
        checks."""

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
                    transfer_slope(
                        total.mass2(), self.config.production_mode, parameters
                    ),
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
                    event_id=self.event_id,
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
                        "builder": "resonance_pp",
                        "kernel_boundary_gev": self.kernel_boundary,
                        "kernel_master_seed": self.master_seed,
                        "kernel_replica_index": self.replica_index,
                    },
                )
                result = ResonancePPResult(event=event)
                if self.config.decay_sources:
                    particle_id = 0
                    for source in sources:
                        daughters, vertices, record = self.decay_source(
                            source, particle_id
                        )
                        event.final_particles.extend(daughters)
                        particle_id += len(daughters)
                        result.source_records.append(record)
                        result.kernel_vertices.append(vertices)
                self.check_event_conservation(event)
                return result
            except (ValueError, FloatingPointError):
                # The legacy retry discipline: the mass sampler's and the
                # legacy branch's ValueError triggers retry the event with
                # fresh draws from the shared generator.  The catalog-
                # driven kernel's protections and the conservation check
                # (ConservationViolationError) are RuntimeError subclasses
                # and are never caught (they stop the run).
                continue
        raise RuntimeError(
            "resonance_pp_builder: failed to generate a valid p+p event "
            f"after {self.config.max_event_attempts} attempts "
            f"(seed {self.seed}, event_id {self.event_id})"
        )

    # ------------------------------------------------------------------
    # Conservation (checked, never repaired)
    # ------------------------------------------------------------------
    def check_event_conservation(self, event: ExcitationEvent) -> None:
        """The pre-registered R23 conservation checks.

        Every final particle set of the event sums to the total
        ``(B, Q, S) = (2, 2, 0)`` of the p+p pair, and every source's
        final set sums to the source's ``(B, Q, S)``.  A violation
        raises :class:`ConservationViolationError` and stops the run
        (never repaired).
        """

        total_b = total_q = total_s = 0
        for particle in event.final_particles:
            qn = particle.quantum_numbers
            total_b += int(qn.B)
            total_q += int(qn.Q)
            total_s += int(qn.S)
        if (total_b, total_q, total_s) != (2, 2, 0):
            raise ConservationViolationError(
                "resonance_pp_builder: event "
                f"{event.event_id} final particles sum to "
                f"(B, Q, S) = {(total_b, total_q, total_s)} but the p+p "
                "pair carries (2, 2, 0); conservation is checked, never "
                "repaired (the run stops)"
            )
        for source in event.sources:
            source_b = source_q = source_s = 0
            for particle in event.final_particles:
                if int(particle.source_id) != int(source.source_id):
                    continue
                qn = particle.quantum_numbers
                source_b += int(qn.B)
                source_q += int(qn.Q)
                source_s += int(qn.S)
            expected = (
                int(source.quantum_numbers.B),
                int(source.quantum_numbers.Q),
                int(source.quantum_numbers.S),
            )
            if (source_b, source_q, source_s) != expected:
                raise ConservationViolationError(
                    "resonance_pp_builder: event "
                    f"{event.event_id} source {source.source_id} "
                    f"({source.role}) finals sum to "
                    f"(B, Q, S) = {(source_b, source_q, source_s)} but "
                    f"the source carries {expected}; conservation is "
                    "checked, never repaired (the run stops)"
                )
