"""The isotropic-versus-aligned ablation runner of the aligned
resonance cascade.

Round 4, card 10 (``TASK-20260920-r4-isotropic-aligned-ablation-runner``;
plan section 22 task 27, line 1749, plan section 10 run R43, lines
790-798).  The module packages the three decay variants of run R43 on
identical exchange events and common random numbers, as the package-side
machinery the R43 run card (card 14) drives:

- **v1 ``isotropic_direct``**: the single-step direct decay of the
  high-mass parent, the ablation floor of the "direct" family.  At a
  catalog handoff root the step is the byte-unchanged Round 2 kernel's
  own vertex (the card-9 handoff binding restricted to one vertex; the
  queued resonance daughters are the step's terminal output, the
  caller's event assembly).  At a continuum root the step is card 4's
  ``draw_decay`` over the direct (established) pool, the plan section
  10.6 direct-channel rule.  The line ends after the single step in
  both cases: no cascade chain, no alignment, no form factor.
- **v2 ``isotropic_sequential``**: card 4's sequential cascade with the
  alignment dispatch at ``kappa_B = kappa_M = 0`` (the bit-stable
  isotropic branches of cards 6-7) and the card-8 form factor at the
  nominal ``b_dec = 1.0 GeV^-2``.  At the isotropic point the dispatch
  is card 4's own ``2u - 1`` draw bit for bit and the cascade baseline
  is card 9's driver semantics (the off limit ``b_dec = 0`` reproduces
  the unattached card-9 ``run_cascade`` line bit for bit; the card test
  pins it).
- **v3 ``aligned_sequential``**: card 4's sequential cascade at the
  pre-registered nominal point ``kappa_B = kappa_M = 1.0`` (round 4
  index item 3) and ``b_dec = 1.0 GeV^-2``; the only difference from
  v2 is the attached weights and the dispatched cos(theta).

The common-random-numbers contract (plan line 798, "Use identical
exchange events and common random numbers") is carried by the stream
spawning: the exchange layer (topology, exchange kinematics, spectral
state) is generated once per ``(master_seed, event_index)`` through
:func:`exchange_substreams` -- the closed ``topology_selection`` /
``exchange_kinematics`` / ``spectral_state`` labels of
:mod:`na49_mixing.rng_substreams`, spawned outside the variant loop and
shared across all three variants -- and the per-variant decay streams
are child ``SeedSequence`` spawns keyed on the variant id (0, 1, 2) of
:func:`variant_stream`, so the decay uniforms of one variant never leak
into another and the per-vertex draw order (card 4's frozen budget,
1 channel + 2 kinematics) is preserved within each variant.

Support boundary (plan sections 5.2 line 275 and 5.3 line 288)
--------------------------------------------------------------

v1's single-step support differs from v2/v3's multi-vertex support
(the cascade chain): v1 is regenerated per the plan section 5.2
boundary rule and the boundary is recorded in the machine-readable
:data:`SUPPORT_BOUNDARY` block (the pre-registration shape card 14
consumes).  The support of v2 and v3 is identical -- the same channels,
the same parent masses, only the angular and form-factor weights
differ -- so v2/v3 share common random numbers without regeneration.

The toggle points
-----------------

Card 9's ``run_cascade`` driver carries no attachment points (its
vertex is card 4's ``draw_decay`` over the raw enumeration), so the
runner mirrors its routing -- the same hard depth guard
(:class:`~na49_mixing.cascade_termination.CascadeDepthGuardError` at
depth 12), the same catalog handoff predicate and the byte-unchanged
kernel delegation, the same continuum-continuation and direct-channel
pool routing, and the same no-channel refusal -- and wires the two
attachments at the pre-registered toggle points of the per-vertex
step:

1. **The pool weights** (the card-8 attachment): every pool channel's
   weight is card 4's hooked weight (:func:`~na49_mixing.cascade_decay.
   channel_weight` at the Round 4 hook defaults) scaled by the card-8
   W_dec factor at the channel's two-body momentum against the
   inherited axis, evaluated at the pinned reference emission
   configuration :data:`W_DEC_REFERENCE_COS_THETA` (see
   :func:`reference_direction`).  The attached pool is passed at the
   raw scale, so the vertex walk re-normalizes it exactly like card 4's
   hook design; at the off limit ``b_dec = 0`` the factor is exactly
   ``1.0`` and the attached weights are the unattached ones bit for
   bit.
2. **The cos(theta) draw** (the card-6/7 dispatch): the first
   kinematics quantile of the vertex is dispatched through
   :func:`~na49_mixing.cascade_alignment.draw_vertex_cos_theta` at
   ``(kappa_b, kappa_m)`` in place of card 4's isotropic draw, inside
   the frozen 1 + 2 per-vertex budget; the azimuth stays the unchanged
   Round 2 second draw.  The alignment labels of the daughters follow
   :func:`~na49_mixing.cascade_alignment.assign_alignment_label` with
   the governing kernel of the drawn vertex (the baryonic one at any
   vertex with one baryon-carrying daughter, the mesonic one at the
   symmetric mesonic vertex and at the baryon-antibaryon pair vertex of
   a mesonic parent, plan section 12.4), so the label bookkeeping and
   the drawn law agree at every vertex.

The attachments are pure: both consume zero random numbers, so the
per-vertex draw budget is card 4's frozen 1 channel + 2 kinematics for
every policy.  The axis inheritance along the line is card 9's driver
behavior (the parent's axis copied verbatim, card 4's daughter-state
construction); card 5's rest-frame transport is the run cards'
caller-side convention.

The exchange event
------------------

:func:`decay_variant` consumes the exchange event as a mapping with the
pinned keys :data:`EXCHANGE_EVENT_KEYS` (``topology``, ``momentum``,
``family_id``, ``mass``, ``qn``, ``axis``, and the optional
``ancestry``): the exchange layer's record of plan line 798, generated
once per event by the run card and shared across the variants.  The
mapping is validated and never mutated (checked, never repaired); the
root :class:`~na49_mixing.cascade_decay.CascadeState` is the pure
function :func:`exchange_root_state` of the mapping, identical for all
three variants.

Hermetic: standard library, numpy, the committed catalog and family
configuration, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned seeds, pinned exchange events, and draw counts; the
R43 generation of 800,000 exchange events is card 14's scope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .cascade_alignment import (
    BARYONIC_KAPPA_DEFAULT,
    MESONIC_KAPPA_DEFAULT,
    assign_alignment_label,
    draw_vertex_cos_theta,
)
from .cascade_axes import AXIS_NORM_TOL
from .cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    CascadeState,
    channel_weight,
    draw_decay,
    enumerate_channels,
    _daughter_identity,
    _daughter_masses,
    _daughter_qn,
    _quantum_of,
)
from .cascade_form_factor import DECAY_FORM_FACTOR_B_DEFAULT
from .cascade_form_factor import attach_to_channel_weight
from .cascade_termination import (
    CascadeDepthGuardError,
    handoff_decay,
    is_catalog_handoff,
)
from .hadron_catalog import HadronCatalog
from .quantum_numbers import quantum_numbers
from .resonance_decay import (
    DECAY_CLOSURE_TOL,
    SOURCE_MASS_TOL,
    KinematicsError,
    ResonanceDecayer,
    _boost_lab,
)
from .resonance_widths import decay_momentum
from .rng_substreams import RngSubstreams, derive_substreams
from .spectral_density import families_from_json

__all__ = [
    "ABLATION_VARIANTS",
    "ABLATION_POLICIES",
    "AblationPolicy",
    "EXCHANGE_EVENT_KEYS",
    "SUPPORT_BOUNDARY",
    "W_DEC_REFERENCE_COS_THETA",
    "W_DEC_REFERENCE_SIN_THETA",
    "exchange_root_state",
    "exchange_substreams",
    "variant_stream",
    "decay_variant",
    "reference_direction",
]


#: The closed ablation register of run R43 (plan section 10, lines
#: 792-796): the isotropic direct decay, the isotropic sequential
#: resonance decay, and the aligned sequential resonance cascade, in
#: the plan's order.  The register order fixes the variant ids
#: (0, 1, 2) of the per-variant decay streams.
ABLATION_VARIANTS: tuple[str, ...] = (
    "isotropic_direct",
    "isotropic_sequential",
    "aligned_sequential",
)

#: The variant id of each register entry (the key of the per-variant
#: decay-stream spawn).
VARIANT_IDS: dict[str, int] = {
    name: index for index, name in enumerate(ABLATION_VARIANTS)
}

#: The pinned keys of the exchange-event mapping (the exchange layer's
#: record of plan line 798; see the module docstring).  ``ancestry`` is
#: optional and defaults to ``(topology,)``.
EXCHANGE_EVENT_KEYS: tuple[str, ...] = (
    "topology",
    "momentum",
    "family_id",
    "mass",
    "qn",
    "axis",
)


@dataclass(frozen=True)
class AblationPolicy:
    """The per-variant decay policy of the ablation runner.

    ``variant`` is one of :data:`ABLATION_VARIANTS` and selects the
    decay machinery (v1 single-step direct, v2/v3 card 4's sequential
    cascade).  ``kappa_b`` and ``kappa_m`` are the alignment strengths
    of the card-6/7 dispatch (0.0 for the isotropic variants, the
    pre-registered nominal 1.0 for v3; the R40 scan grid caps at 4).
    ``b_dec`` is the card-8 form-factor slope in GeV^-2 (the nominal
    1.0 for v2/v3; the v1 single-step path never applies the form
    factor).  The values are the toggle surface of the per-attachment
    ablation: the machinery is keyed on the variant, the parameters on
    the attachments.  A variant outside :data:`ABLATION_VARIANTS`, a
    non-finite kappa, or a non-finite / negative ``b_dec`` raises
    ``ValueError`` at construction (checked, never repaired).
    """

    variant: str
    kappa_b: float
    kappa_m: float
    b_dec: float

    def __post_init__(self) -> None:
        variant = str(self.variant)
        if variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"ablation policy variant {self.variant!r} is outside "
                f"the closed register {ABLATION_VARIANTS}"
            )
        kappa_b = float(self.kappa_b)
        kappa_m = float(self.kappa_m)
        b_dec = float(self.b_dec)
        for name, value in (("kappa_b", kappa_b), ("kappa_m", kappa_m)):
            if not math.isfinite(value):
                raise ValueError(
                    f"ablation policy {name} = {value!r} must be finite "
                    "(effective spin alignment, plan section 10.4 line "
                    "730)"
                )
        if not (math.isfinite(b_dec) and b_dec >= 0.0):
            raise ValueError(
                f"ablation policy b_dec = {self.b_dec!r} must be finite "
                "and non-negative (a negative slope is an unphysical "
                "vertex form factor)"
            )
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "kappa_b", kappa_b)
        object.__setattr__(self, "kappa_m", kappa_m)
        object.__setattr__(self, "b_dec", b_dec)


#: The pre-registered policies of the three R43 variants (the register
#: keyed by the variant name).  v1 and v2 sit at the isotropic point
#: ``kappa_B = kappa_M = 0`` (the bit-stable isotropic branches of
#: cards 6-7); v3 sits at the pre-registered nominal point
#: ``(kappa_B, kappa_M, b_dec) = (1.0, 1.0, 1.0 GeV^-2)`` (round 4
#: index item 3, the R40 starting value, not a fit parameter).  v1's
#: ``b_dec`` records the nominal point; the v1 single-step path never
#: applies the form factor.
ABLATION_POLICIES: dict[str, AblationPolicy] = {
    "isotropic_direct": AblationPolicy(
        "isotropic_direct", 0.0, 0.0, DECAY_FORM_FACTOR_B_DEFAULT),
    "isotropic_sequential": AblationPolicy(
        "isotropic_sequential", 0.0, 0.0, DECAY_FORM_FACTOR_B_DEFAULT),
    "aligned_sequential": AblationPolicy(
        "aligned_sequential",
        BARYONIC_KAPPA_DEFAULT,
        MESONIC_KAPPA_DEFAULT,
        DECAY_FORM_FACTOR_B_DEFAULT),
}

#: The pinned reference emission cosine of the card-8 W_dec attachment:
#: the channel's two-body daughter momentum is evaluated at this polar
#: cosine against the inherited production axis (the pre-registered
#: oblique reference configuration of the card-8 caller-side
#: attachment, referenced to the axis the q_perp convention is defined
#: against).  The factor is monotone in the channel's two-body
#: momentum, independent of the azimuth (q_perp is azimuth-invariant),
#: and consumes zero random numbers.
W_DEC_REFERENCE_COS_THETA: float = 0.3

#: The pinned reference transverse fraction
#: ``sqrt(1 - W_DEC_REFERENCE_COS_THETA^2)`` (the sine of the reference
#: emission angle).
W_DEC_REFERENCE_SIN_THETA: float = math.sqrt(
    1.0 - W_DEC_REFERENCE_COS_THETA * W_DEC_REFERENCE_COS_THETA
)

#: The machine-readable support boundary of the ablation (plan
#: sections 5.2 line 275 and 5.3 line 288; the pre-registration block
#: shape the R43 run card consumes).  v1's single-step support differs
#: from v2/v3's multi-vertex support, so v1 is regenerated with the
#: boundary recorded here; the support of v2 and v3 is identical (the
#: same channels, the same parent masses, only the angular and
#: form-factor weights differ), so v2/v3 share common random numbers
#: without regeneration.
SUPPORT_BOUNDARY: dict = {
    "plan_line_275": (
        "Use common random numbers for parameter comparisons whenever "
        "support does not change. If support changes, regenerate and "
        "document the boundary."
    ),
    "plan_line_288": (
        "Reweight only when the changed parameter alters a probability "
        "density but not kinematic support. Regenerate when thresholds, "
        "allowed channels, or parent masses change."
    ),
    "variants": {
        "isotropic_direct": {
            "decay_steps": "single_step",
            "support": "single_step",
            "common_random_numbers": "regenerate",
            "boundary": (
                "v1's single-step support (the parent's direct channel "
                "set) differs from v2/v3's multi-vertex cascade "
                "support; v1 is regenerated per plan section 5.2 line "
                "275 and this block is the documented boundary"
            ),
        },
        "isotropic_sequential": {
            "decay_steps": "multi_vertex",
            "support": "multi_vertex",
            "common_random_numbers": "shared",
            "boundary": None,
        },
        "aligned_sequential": {
            "decay_steps": "multi_vertex",
            "support": "multi_vertex",
            "common_random_numbers": "shared",
            "boundary": (
                "the support is identical to isotropic_sequential (the "
                "same channels, the same parent masses; only the "
                "angular and form-factor weights differ), so no "
                "regeneration is required against v2"
            ),
        },
    },
    "v2_v3_support_identical": True,
    "regeneration_required": ("isotropic_direct",),
}


# ----------------------------------------------------------------------
# The committed catalog and family configuration (lazy singletons)
# ----------------------------------------------------------------------
_NESTED_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
_FAMILIES_PATH = (
    _NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
)
_COMMITTED: dict = {}


def _committed_catalog() -> HadronCatalog:
    """The committed catalog (``config/v2_singlet/resonance_catalog.
    json``), loaded once and cached (the runner's decay surface; the
    card-4 enumeration and the Round 2 kernel resolve against it)."""

    if "catalog" not in _COMMITTED:
        _COMMITTED["catalog"] = HadronCatalog.from_json(_CATALOG_PATH)
    return _COMMITTED["catalog"]


def _committed_fams():
    """The committed continuum family configuration
    (``config/v2_singlet/continuum_resonances.json``), loaded once and
    cached."""

    if "fams" not in _COMMITTED:
        _COMMITTED["fams"] = families_from_json(str(_FAMILIES_PATH))
    return _COMMITTED["fams"]


# ----------------------------------------------------------------------
# Stream spawning (the common-random-numbers contract)
# ----------------------------------------------------------------------
def _non_negative_int(value, name: str) -> int:
    """The non-negative Python ``int`` behind *value*; anything else
    raises ``ValueError`` naming *name* (booleans rejected)."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    converted = int(value)
    if converted < 0:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    return converted


def variant_stream(
    master_seed: int, event_index: int, variant_id: int
) -> tuple[np.random.Generator, np.random.Generator]:
    """The per-variant decay streams of one exchange event: the
    ``decay_channel`` and ``decay_kinematics`` generators of one
    variant.

    Child ``SeedSequence`` spawns keyed on ``(master_seed,
    event_index, variant_id)`` -- the one-step form
    ``SeedSequence(master_seed, spawn_key=(event_index, variant_id))``
    with two label children, bit-identical to the literal fresh-object
    spawn chain and stable across numpy versions (the
    :mod:`na49_mixing.rng_substreams` derivation contract).  The
    streams are disjoint across the variant ids (the decay uniforms of
    one variant never leak into another) and deterministic: the same
    triple reproduces the streams byte for byte.  The exchange layer is
    NOT part of this spawn tree: the exchange streams are
    :func:`exchange_substreams`'s, spawned once per ``(master_seed,
    event_index)`` outside the variant loop and shared across the
    variants (plan line 798).

    ``variant_id`` is the register index of :data:`ABLATION_VARIANTS`
    (0, 1, 2); a non-integer or negative argument, or a variant id
    outside the closed register, raises ``ValueError``.
    """

    ms = _non_negative_int(master_seed, "master_seed")
    ei = _non_negative_int(event_index, "event_index")
    vid = _non_negative_int(variant_id, "variant_id")
    if vid >= len(ABLATION_VARIANTS):
        raise ValueError(
            f"variant_id {variant_id!r} is outside the closed ablation "
            f"register (0..{len(ABLATION_VARIANTS) - 1} for "
            f"{ABLATION_VARIANTS})"
        )
    variant_seq = np.random.SeedSequence(ms, spawn_key=(ei, vid))
    channel_seq, kinematics_seq = variant_seq.spawn(2)
    return (
        np.random.default_rng(channel_seq),
        np.random.default_rng(kinematics_seq),
    )


def exchange_substreams(
    master_seed: int, event_index: int, replica_index: int = 0
) -> RngSubstreams:
    """The exchange-layer substreams of one exchange event (plan line
    798's identical exchange events): the house derivation
    :func:`~na49_mixing.rng_substreams.derive_substreams` at
    ``(master_seed, replica_index, event_index)``, spawned once per
    event outside the variant loop and shared across all three
    variants.

    The exchange generation consumes the ``topology_selection``,
    ``exchange_kinematics``, and ``spectral_state`` labels of the
    returned object (the first three entries of the closed
    :data:`~na49_mixing.rng_substreams.SUBSTREAM_LABELS` set); the
    decay labels of the object are unused -- the per-variant decay
    streams are :func:`variant_stream`'s spawns, a separate
    ``SeedSequence`` tree, so no decay uniform of the exchange
    derivation leaks into a variant stream or the other way around.
    """

    return derive_substreams(master_seed, replica_index, event_index)


# ----------------------------------------------------------------------
# The exchange event and the root state
# ----------------------------------------------------------------------
def _invariant_mass(p4) -> float:
    """The invariant mass of a four-momentum (the kernel's convention)."""

    e, px, py, pz = (float(x) for x in p4)
    m2 = e * e - px * px - py * py - pz * pz
    return math.sqrt(max(m2, 0.0))


def exchange_root_state(exchange_event) -> CascadeState:
    """The root :class:`~na49_mixing.cascade_decay.CascadeState` of one
    exchange event: the pure, draw-free function of the exchange-event
    mapping that all three variants decay.

    The mapping carries :data:`EXCHANGE_EVENT_KEYS` (all required) and
    the optional ``ancestry`` tuple (default ``(topology,)``):
    ``topology`` the exchange topology label, ``momentum`` the source
    four-momentum ``(E, px, py, pz)`` in the working frame, ``mass``
    the realized invariant mass, ``family_id`` the spectral-family
    identifier (``"catalog:<pdg>"`` or a continuum family id of the
    committed configuration), ``qn`` the integer ``(B, Q, S)``, and
    ``axis`` the unit production axis in the rest frame (the
    isolated-source lab z-axis of the R40 pre-registration, or card 5's
    :func:`~na49_mixing.cascade_axes.initial_source_axis` value for the
    p+p run cards).  The mapping is validated and never mutated:
    a missing key, a malformed field, a non-finite component, an
    invariant mass beyond :data:`~na49_mixing.resonance_decay.
    SOURCE_MASS_TOL` of ``mass``, a non-unit axis beyond
    :data:`~na49_mixing.cascade_axes.AXIS_NORM_TOL`, a quantum-number
    disagreement with the named record or family, or an unknown family
    id raises (checked, never repaired).  The root carries the
    card-4 isotropic label placeholder and depth 0.
    """

    if not isinstance(exchange_event, Mapping):
        raise ValueError(
            "exchange_event must be a mapping with the pinned keys "
            f"{EXCHANGE_EVENT_KEYS}, got {type(exchange_event).__name__}"
        )
    for key in EXCHANGE_EVENT_KEYS:
        if key not in exchange_event:
            raise ValueError(
                f"exchange_event is missing the pinned key {key!r} "
                f"(the pinned keys are {EXCHANGE_EVENT_KEYS})"
            )
    topology = str(exchange_event["topology"])
    if not topology:
        raise ValueError("exchange_event topology must be a non-empty label")
    momentum = tuple(float(x) for x in exchange_event["momentum"])
    if len(momentum) != 4:
        raise KinematicsError(
            f"exchange_event momentum must carry four components, got "
            f"{len(momentum)}"
        )
    for component in momentum:
        if not math.isfinite(component):
            raise KinematicsError(
                f"exchange_event momentum component {component!r} is "
                "not finite"
            )
    if not momentum[0] > 0.0:
        raise KinematicsError(
            f"exchange_event momentum energy {momentum[0]!r} is not "
            "positive; the four-momentum is not future-directed"
        )
    mass = float(exchange_event["mass"])
    if not (math.isfinite(mass) and mass > 0.0):
        raise KinematicsError(
            f"exchange_event mass {exchange_event['mass']!r} must be "
            "finite and positive"
        )
    invariant = _invariant_mass(momentum)
    if abs(invariant - mass) > SOURCE_MASS_TOL:
        raise KinematicsError(
            f"exchange_event mass {mass!r} disagrees with the invariant "
            f"mass {invariant!r} of its four-momentum (beyond "
            f"{SOURCE_MASS_TOL} GeV); checked, never repaired"
        )
    family_id = str(exchange_event["family_id"])
    qn = tuple(int(v) for v in exchange_event["qn"])
    if len(qn) != 3:
        raise ValueError(
            f"exchange_event qn {exchange_event['qn']!r} must carry the "
            "three integer quantum numbers (B, Q, S)"
        )
    axis = tuple(float(x) for x in exchange_event["axis"])
    if len(axis) != 3:
        raise KinematicsError(
            f"exchange_event axis must carry three components, got "
            f"{len(axis)}"
        )
    for component in axis:
        if not math.isfinite(component):
            raise KinematicsError(
                f"exchange_event axis component {component!r} is not "
                "finite"
            )
    norm = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if not abs(norm - 1.0) <= AXIS_NORM_TOL:
        raise KinematicsError(
            f"exchange_event axis is not unit (norm {norm!r} beyond the "
            f"card-5 tolerance {AXIS_NORM_TOL!r}); the production axis "
            "is a direction in the rest frame"
        )
    ancestry = exchange_event.get("ancestry", (topology,))
    ancestry = tuple(str(item) for item in ancestry)

    catalog = _committed_catalog()
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        record = catalog.get(int(family_id[len(CATALOG_FAMILY_PREFIX):]))
        record_qn = (int(record.B), int(record.Q), int(record.S))
        if record_qn != qn:
            raise ValueError(
                f"exchange_event qn {qn} disagrees with the catalog "
                f"record {record.name} (pdg {record.pdg_id}) carrying "
                f"{record_qn}; conservation is checked, never repaired"
            )
        quantum = {
            "I": float(record.isospin),
            "I3": int(record.isospin3),
            "J": float(record.spin),
            "P": record.parity,
        }
    else:
        family = next(
            (f for f in _committed_fams()
             if str(f.family_id) == family_id),
            None,
        )
        if family is None:
            raise ValueError(
                f"exchange_event family_id {family_id!r} names no "
                "continuum family of the committed configuration"
            )
        family_qn = tuple(int(v) for v in family.qn)
        if family_qn != qn:
            raise ValueError(
                f"exchange_event qn {qn} disagrees with the continuum "
                f"family {family_id!r} carrying {family_qn}; "
                "conservation is checked, never repaired"
            )
        quantum = {}
    return CascadeState(
        momentum=momentum,
        mass=mass,
        qn=qn,
        quantum=quantum,
        family_id=family_id,
        axis=axis,
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=ancestry,
        depth=0,
    )


# ----------------------------------------------------------------------
# The card-8 W_dec attachment (the pool-weight toggle point)
# ----------------------------------------------------------------------
def reference_direction(
    axis: tuple[float, float, float]
) -> tuple[float, float, float]:
    """The unit reference emission direction of the W_dec attachment:
    the pinned oblique configuration at
    :data:`W_DEC_REFERENCE_COS_THETA` against the unit inherited
    production axis.

    The direction is ``normalize(cos(theta_ref) * a +
    sin(theta_ref) * e1)`` with ``e1`` the normalized rejection of the
    first coordinate axis (x, then y, then z) whose rejection of *axis*
    has norm at least 0.5 -- a deterministic axis frame with no free
    azimuth, which is exact for the q_perp convention because q_perp is
    azimuth-invariant.  Pure: no random number is consumed; a
    degenerate axis raises the kernel's ``KinematicsError`` (checked,
    never repaired).
    """

    ax, ay, az = (float(v) for v in axis)
    e1: tuple[float, float, float] | None = None
    for seed in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        dot = seed[0] * ax + seed[1] * ay + seed[2] * az
        rx, ry, rz = (
            seed[0] - dot * ax, seed[1] - dot * ay, seed[2] - dot * az
        )
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        if norm >= 0.5:
            e1 = (rx / norm, ry / norm, rz / norm)
            break
    if e1 is None:
        raise KinematicsError(
            "reference_direction: the axis carries no coordinate "
            "rejection of norm >= 0.5 (a degenerate axis frame); "
            "checked, never repaired"
        )
    cx = W_DEC_REFERENCE_COS_THETA * ax + W_DEC_REFERENCE_SIN_THETA * e1[0]
    cy = W_DEC_REFERENCE_COS_THETA * ay + W_DEC_REFERENCE_SIN_THETA * e1[1]
    cz = W_DEC_REFERENCE_COS_THETA * az + W_DEC_REFERENCE_SIN_THETA * e1[2]
    norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    return (cx / norm, cy / norm, cz / norm)


def _reference_daughter_p4(
    channel: CascadeChannel, state: CascadeState
) -> tuple[float, float, float, float]:
    """The channel's leading-daughter four-momentum at the pinned
    reference emission configuration: the two-body momentum at the
    realized daughter masses (card 4's deterministic rule, the same
    masses the vertex realizes) emitted along
    :func:`reference_direction` of the parent state's inherited axis.
    Pure: no random number is consumed."""

    catalog = _committed_catalog()
    fams = _committed_fams()
    identities = [
        _daughter_identity(catalog, fams, daughter_id)
        for daughter_id in channel.daughters
    ]
    masses = _daughter_masses(state, identities)
    mass0, mass1 = masses
    q_star = decay_momentum(float(state.mass), mass0, mass1)
    direction = reference_direction(state.axis)
    return (
        math.sqrt(mass0 * mass0 + q_star * q_star),
        q_star * direction[0],
        q_star * direction[1],
        q_star * direction[2],
    )


def _attach_form_factor(
    pool: list[CascadeChannel], state: CascadeState, b_dec: float
) -> list[CascadeChannel]:
    """The attached pool at the pool-weight toggle point: every
    channel's weight is card 4's hooked weight scaled by the card-8
    W_dec factor at the channel's reference two-body configuration
    (the ``attach_to_channel_weight`` composition of the card-8
    attachment).

    The attached weights stay at the raw scale of the enumeration (no
    subset re-normalization): the vertex walk re-normalizes over the
    pool exactly like card 4's hook design, and at the off limit
    ``b_dec = 0`` the factor is exactly ``1.0``, so the attached
    weights are the unattached ones bit for bit (the R43 ablation
    baseline).  Pure: no random number is consumed."""

    wrapped = attach_to_channel_weight(channel_weight, float(b_dec))
    return [
        CascadeChannel(
            parent_id=channel.parent_id,
            daughters=channel.daughters,
            kind=channel.kind,
            weight=wrapped(channel, _reference_daughter_p4(channel, state),
                           state.axis),
        )
        for channel in pool
    ]


# ----------------------------------------------------------------------
# The per-vertex step (the cos(theta) toggle point; card 4's frozen
# 1 channel + 2 kinematics budget)
# ----------------------------------------------------------------------
def _daughter_cascade_state(
    state: CascadeState,
    identity,
    four_momentum: tuple[float, float, float, float],
    mass: float,
    catalog: HadronCatalog,
    alignment_label: str,
) -> CascadeState:
    """The daughter state of the runner's vertex: card 4's
    daughter-state construction (the realized four-momentum and mass,
    the daughter's quantum numbers and family identity, the parent
    link, the inherited exchange ancestry, the parent's axis verbatim,
    and the incremented depth) with the alignment label from the
    card-6/7 bookkeeping of the drawn vertex."""

    kind, pdg, obj = identity
    momentum = (
        float(four_momentum[0]),
        float(four_momentum[1]),
        float(four_momentum[2]),
        float(four_momentum[3]),
    )
    axis = (
        float(state.axis[0]),
        float(state.axis[1]),
        float(state.axis[2]),
    )
    if kind == "catalog":
        return CascadeState(
            momentum=momentum,
            mass=float(mass),
            qn=(int(obj.B), int(obj.Q), int(obj.S)),
            quantum=_quantum_of(catalog, pdg),
            family_id=f"{CATALOG_FAMILY_PREFIX}{int(pdg)}",
            axis=axis,
            alignment_label=str(alignment_label),
            parent=state,
            exchange_ancestry=tuple(state.exchange_ancestry),
            depth=int(state.depth) + 1,
        )
    return CascadeState(
        momentum=momentum,
        mass=float(mass),
        qn=(int(obj.qn[0]), int(obj.qn[1]), int(obj.qn[2])),
        quantum={},
        family_id=str(obj.family_id),
        axis=axis,
        alignment_label=str(alignment_label),
        parent=state,
        exchange_ancestry=tuple(state.exchange_ancestry),
        depth=int(state.depth) + 1,
    )


def _draw_attached_vertex(
    state: CascadeState,
    pool: list[CascadeChannel],
    catalog: HadronCatalog,
    fams,
    kappa_b: float,
    kappa_m: float,
    rng_channel,
    rng_kinematics,
) -> tuple[CascadeChannel, tuple[CascadeState, CascadeState]]:
    """One decay step of the sequential variants: the weighted channel
    draw over the attached pool (1 uniform on the ``decay_channel``
    substream, the Round 2 cumulative walk) and the two-body kinematics
    (2 uniforms on the ``decay_kinematics`` substream) with the
    cos(theta) draw dispatched through the card-6/7 kernels at
    ``(kappa_b, kappa_m)``.

    The walk, the daughter resolution, the kinematics construction, and
    the conservation checks are card 4's ``draw_decay`` expressions on
    the attached pool; at the isotropic point the dispatch is card 4's
    own ``2u - 1`` draw bit for bit, so the step reproduces card 4's
    cascade vertex exactly (the card test pins the bit identity against
    :func:`~na49_mixing.cascade_decay.draw_decay` and, at the off
    limit, against :func:`~na49_mixing.cascade_termination.run_cascade`).
    The per-vertex budget is exactly 1 + 2 uniforms on the closed
    labels; the dispatch and the attached weights consume none.
    """

    if not pool:
        raise ValueError(
            "draw_attached_vertex: the pool is empty (no open channel "
            f"at the realized mass {float(state.mass)!r} GeV of family "
            f"{state.family_id!r})"
        )
    total = math.fsum(channel.weight for channel in pool)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            "draw_attached_vertex: the pool weights do not sum to a "
            f"finite positive total ({total!r}); no channel is drawable"
        )

    # -- the channel draw: 1 uniform, the Round 2 cumulative walk -------
    u_channel = float(rng_channel.random())
    target = u_channel * total
    selected: CascadeChannel | None = None
    cumulative = 0.0
    for channel in pool:
        cumulative += channel.weight
        if cumulative >= target:
            selected = channel
            break
    if selected is None:
        selected = pool[-1]
    if selected.parent_id != str(state.family_id):
        raise ValueError(
            f"draw_attached_vertex: the drawn channel's parent_id "
            f"{selected.parent_id!r} does not match the state's "
            f"family_id {state.family_id!r}"
        )
    if len(selected.daughters) != 2:
        raise ValueError(
            "draw_attached_vertex: the drawn channel "
            f"{tuple(selected.daughters)!r} is not a quasi-two-body "
            "channel; the three-body clause is the Round 2 kernel's "
            "vertex, reached through the catalog handoff"
        )

    # -- the daughter identities and realized masses (card 4) -----------
    identities = [
        _daughter_identity(catalog, fams, daughter_id)
        for daughter_id in selected.daughters
    ]
    masses = _daughter_masses(state, identities)
    qn0 = _daughter_qn(state, identities[0], catalog)
    qn1 = _daughter_qn(state, identities[1], catalog)

    # -- the kinematics: 2 uniforms, the cos(theta) dispatch ------------
    u_cos = float(rng_kinematics.random())
    cos_theta = draw_vertex_cos_theta(u_cos, (qn0, qn1), kappa_b, kappa_m)
    u_phi = float(rng_kinematics.random())
    phi = 2.0 * math.pi * u_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = (
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    )

    parent_mass = float(state.mass)
    mass0, mass1 = masses
    q_star = decay_momentum(parent_mass, mass0, mass1)
    energy0 = (parent_mass ** 2 + mass0 ** 2 - mass1 ** 2) / (2.0 * parent_mass)
    energy1 = (parent_mass ** 2 - mass0 ** 2 + mass1 ** 2) / (2.0 * parent_mass)
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
    energy, px, py, pz = (float(x) for x in state.momentum)
    beta = (px / energy, py / energy, pz / energy)
    daughter_p4_0 = _boost_lab(rest0, beta)
    daughter_p4_1 = _boost_lab(rest1, beta)

    # -- conservation, checked never repaired (card 4's checks) ---------
    total_qn = (qn0[0] + qn1[0], qn0[1] + qn1[1], qn0[2] + qn1[2])
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    if total_qn != state_qn:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} "
            f"-> {tuple(selected.daughters)}: daughters sum to (B, Q, S) "
            f"= {total_qn} but the parent carries {state_qn}"
        )
    closure = (
        abs(daughter_p4_0[0] + daughter_p4_1[0] - energy),
        abs(daughter_p4_0[1] + daughter_p4_1[1] - px),
        abs(daughter_p4_0[2] + daughter_p4_1[2] - py),
        abs(daughter_p4_0[3] + daughter_p4_1[3] - pz),
    )
    worst = max(closure)
    if not worst <= DECAY_CLOSURE_TOL:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} "
            f"-> {tuple(selected.daughters)}: the daughter four-momenta "
            f"miss the parent four-momentum by {worst!r} GeV (beyond "
            f"{DECAY_CLOSURE_TOL}); no momentum repair"
        )
    for daughter_p4, mass in ((daughter_p4_0, mass0), (daughter_p4_1, mass1)):
        e, dpx, dpy, dpz = daughter_p4
        m2 = e * e - dpx * dpx - dpy * dpy - dpz * dpz
        deviation = abs(m2 - mass * mass)
        if not deviation <= DECAY_CLOSURE_TOL:
            raise KinematicsError(
                f"cascade vertex {state.family_id!r} at mass "
                f"{parent_mass!r} -> {tuple(selected.daughters)}: a "
                f"daughter is off shell (p^2 - m^2 = {deviation!r} "
                f"GeV^2, beyond {DECAY_CLOSURE_TOL}); no momentum repair"
            )

    # -- the daughter states with the governing-kernel labels -----------
    # A baryon-antibaryon pair (both daughters carry baryon number,
    # summing to zero: the Round 6 baryon-pair channels of a mesonic
    # parent, plan section 12.4) carries no baryonic line; the mesonic
    # kernel governs the vertex, matching draw_vertex_cos_theta.
    pair_vertex = qn0[0] != 0 and qn1[0] != 0
    if pair_vertex:
        label_kappa_b, label_kappa_m = 0.0, kappa_m
    elif qn0[0] != 0 or qn1[0] != 0:
        label_kappa_b, label_kappa_m = kappa_b, 0.0
    else:
        label_kappa_b, label_kappa_m = 0.0, kappa_m
    label0 = assign_alignment_label(
        qn0, label_kappa_b, label_kappa_m, pair_vertex=pair_vertex)
    label1 = assign_alignment_label(
        qn1, label_kappa_b, label_kappa_m, pair_vertex=pair_vertex)
    state0 = _daughter_cascade_state(
        state, identities[0], daughter_p4_0, mass0, catalog, label0)
    state1 = _daughter_cascade_state(
        state, identities[1], daughter_p4_1, mass1, catalog, label1)
    return selected, (state0, state1)


# ----------------------------------------------------------------------
# v1: the single-step direct decay
# ----------------------------------------------------------------------
def _terminal_daughter_state(
    root: CascadeState, daughter, catalog: HadronCatalog
) -> CascadeState:
    """The terminal :class:`~na49_mixing.cascade_decay.CascadeState` of
    one Round 2 kernel daughter (the v1 single-step output): the
    realized four-momentum and its invariant mass, the catalog quantum
    numbers, the root as the parent, and depth 1."""

    pdg = int(daughter.pdg_id)
    momentum = (
        float(daughter.four_momentum[0]),
        float(daughter.four_momentum[1]),
        float(daughter.four_momentum[2]),
        float(daughter.four_momentum[3]),
    )
    mass = _invariant_mass(momentum)
    try:
        record = catalog.get(pdg)
    except KeyError:
        qn_obj = quantum_numbers(pdg)
        qn = (int(qn_obj.B), int(qn_obj.Q), int(qn_obj.S))
        quantum: dict = {}
    else:
        qn = (int(record.B), int(record.Q), int(record.S))
        quantum = {
            "I": float(record.isospin),
            "I3": int(record.isospin3),
            "J": float(record.spin),
            "P": record.parity,
        }
    return CascadeState(
        momentum=momentum,
        mass=mass,
        qn=qn,
        quantum=quantum,
        family_id=f"{CATALOG_FAMILY_PREFIX}{pdg}",
        axis=(
            float(root.axis[0]),
            float(root.axis[1]),
            float(root.axis[2]),
        ),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=root,
        exchange_ancestry=tuple(root.exchange_ancestry),
        depth=int(root.depth) + 1,
    )


def _kernel_single_vertex(
    root: CascadeState,
    catalog: HadronCatalog,
    rng_channel,
    rng_kinematics,
) -> list[CascadeState]:
    """The v1 single step at a catalog handoff root: the byte-unchanged
    Round 2 kernel's own vertex, with the cascade's closed substream
    generators bound into the kernel's draw sites (the card-9 handoff
    binding pattern restricted to one vertex).

    The queued resonance daughters are NOT decayed further (the v1
    "no cascade chain" clause): they are the step's terminal output,
    the caller's event assembly (card 9's convention).  The step
    consumes the Round 2 single-vertex budget: 1 channel uniform, the
    two-body kinematics uniforms, and the kernel's documented mass
    draws per resonance daughter (the card-7 three-body budget at a
    three-body vertex).  Pure construction: the kernel instance
    consumes no random numbers."""

    pdg = int(root.family_id[len(CATALOG_FAMILY_PREFIX):])
    record = catalog.get(pdg)
    decayer = ResonanceDecayer(catalog, 0, 0, 0)  # pure: no random draws
    decayer._rng_channel = rng_channel
    decayer._rng_kinematics = rng_kinematics
    momentum = (
        float(root.momentum[0]),
        float(root.momentum[1]),
        float(root.momentum[2]),
        float(root.momentum[3]),
    )
    vertex, _queued = decayer._one_vertex(
        record, momentum, float(root.mass), (pdg,)
    )
    return [
        _terminal_daughter_state(root, daughter, catalog)
        for daughter in vertex.daughters
    ]


def _decay_isotropic_direct(
    exchange_event, rng_channel, rng_kinematics
) -> list[CascadeState]:
    """The v1 ``isotropic_direct`` variant: the single-step direct
    decay of the high-mass parent (the ablation floor of the "direct"
    family).

    At a catalog handoff root the step is the Round 2 kernel's own
    vertex (:func:`_kernel_single_vertex`); at a continuum root the
    Round 2 kernel has no record, so the step is card 4's
    ``draw_decay`` over the direct (established) pool -- the plan
    section 10.6 direct-channel rule at the isotropic point.  The line
    ends after the single step: the returned list carries the terminal
    daughters of the step and nothing else (no cascade chain, no
    alignment, no form factor).  A root with no open direct channel
    raises ``ValueError`` (checked, never repaired)."""

    catalog = _committed_catalog()
    root = exchange_root_state(exchange_event)
    if is_catalog_handoff(root, catalog):
        return _kernel_single_vertex(root, catalog, rng_channel,
                                     rng_kinematics)
    fams = _committed_fams()
    channels = enumerate_channels(root, catalog, fams)
    direct = [
        channel for channel in channels
        if channel.kind in (KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE)
    ]
    if not direct:
        raise ValueError(
            "isotropic_direct: the root state of family "
            f"{root.family_id!r} at mass {float(root.mass)!r} GeV "
            "carries no open direct hadronic channel, so the "
            "single-step direct decay has no channel (checked, never "
            "repaired)"
        )
    _channel, daughters = draw_decay(
        root, direct, catalog, fams, rng_channel, rng_kinematics)
    return list(daughters)


# ----------------------------------------------------------------------
# v2/v3: the sequential cascade with the attachments
# ----------------------------------------------------------------------
def _decay_sequential(
    exchange_event,
    policy: AblationPolicy,
    rng_channel,
    rng_kinematics,
) -> list[CascadeState]:
    """The v2/v3 sequential variants: card 4's cascade over the
    committed catalog and family configuration with the card-8
    form-factor attachment at ``policy.b_dec`` and the card-6/7
    alignment dispatch at ``(policy.kappa_b, policy.kappa_m)``.

    The routing is card 9's driver semantics, mirrored because the
    driver carries no attachment points: at every line state, in
    order, the hard depth guard (:class:`~na49_mixing.cascade_termination.
    CascadeDepthGuardError` at ``depth >
    :data:`~na49_mixing.cascade_decay.MAX_CASCADE_DEPTH``), the catalog
    handoff (the byte-unchanged kernel takes the remainder through
    :func:`~na49_mixing.cascade_termination.handoff_decay` and the line
    terminates), the continuation through one attached vertex over the
    open continuum pool, the direct-channel termination through one
    attached vertex over the direct pool, and the no-channel refusal
    (``ValueError``).  The returned list is the cascade line, root
    first, ending at the terminal state (card 9's convention).
    """

    catalog = _committed_catalog()
    fams = _committed_fams()
    root = exchange_root_state(exchange_event)
    trace: list[CascadeState] = [root]
    current = root
    while True:
        depth = int(current.depth)
        if depth > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"cascade depth guard: the line state of family "
                f"{current.family_id!r} at mass {float(current.mass)!r} "
                f"GeV carries depth {depth}, exceeding "
                f"MAX_CASCADE_DEPTH = {MAX_CASCADE_DEPTH} (plan section "
                "10.6, line 757); the cascade is stopped at the guard, "
                "no repair, no truncation"
            )
        if is_catalog_handoff(current, catalog):
            handoff_decay(current, catalog, rng_channel, rng_kinematics)
            return trace
        channels = enumerate_channels(current, catalog, fams)
        continuum = [
            channel for channel in channels
            if channel.kind == KIND_CONTINUUM
        ]
        pool = continuum if continuum else [
            channel for channel in channels
            if channel.kind in (KIND_ESTABLISHED_HADRON,
                                KIND_ESTABLISHED_RESONANCE)
        ]
        if not pool:
            raise ValueError(
                f"decay_variant: the line state of family "
                f"{current.family_id!r} at mass "
                f"{float(current.mass)!r} GeV carries no open continuum "
                "channel and no valid direct hadronic channel, so "
                "neither termination rule of plan section 10.6 applies "
                "(a state outside the accepted model; checked, never "
                "repaired)"
            )
        attached = _attach_form_factor(pool, current, policy.b_dec)
        _channel, daughters = _draw_attached_vertex(
            current, attached, catalog, fams,
            policy.kappa_b, policy.kappa_m,
            rng_channel, rng_kinematics,
        )
        if continuum:
            successor = next(
                daughter for daughter in daughters
                if not daughter.family_id.startswith(CATALOG_FAMILY_PREFIX)
            )
            trace.append(successor)
            current = successor
            continue
        return trace


# ----------------------------------------------------------------------
# The variant driver
# ----------------------------------------------------------------------
def decay_variant(
    exchange_event,
    policy: AblationPolicy,
    rng_channel,
    rng_kinematics,
) -> list:
    """Decay one identical exchange event through a named ablation
    variant.

    ``exchange_event`` is the exchange-layer mapping of plan line 798
    (the pinned keys :data:`EXCHANGE_EVENT_KEYS`; validated and never
    mutated), ``policy`` an :class:`AblationPolicy` of
    :data:`ABLATION_POLICIES`, and the two generators the per-variant
    decay streams of :func:`variant_stream` (``decay_channel`` first,
    ``decay_kinematics`` second).  Returns:

    - v1 ``isotropic_direct``: the terminal daughter states of the
      single direct step (the Round 2 kernel's vertex daughters at a
      catalog handoff root, card 4's direct-pool daughters at a
      continuum root), each carrying the root as its parent;
    - v2/v3: the cascade line as a list of
      :class:`~na49_mixing.cascade_decay.CascadeState`, root first,
      ending at the terminal state (card 9's driver convention).

    Every variant conserves ``(B, Q, S)`` exactly per event (checked,
    never repaired: the kernel's own checks at the v1 catalog vertex,
    card 4's checks at the card-4 vertices) and consumes exactly its
    documented budget: the Round 2 single-vertex budget for v1, card
    4's frozen 1 channel + 2 kinematics per vertex for v2/v3, with the
    alignment dispatch and the form factor adding zero uniforms.
    """

    if not isinstance(policy, AblationPolicy):
        raise ValueError(
            f"decay_variant: policy must be an AblationPolicy, got "
            f"{type(policy).__name__}"
        )
    if policy.variant == "isotropic_direct":
        return _decay_isotropic_direct(exchange_event, rng_channel,
                                       rng_kinematics)
    return _decay_sequential(exchange_event, policy, rng_channel,
                             rng_kinematics)
