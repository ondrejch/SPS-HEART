"""Parallel helpers for chunked event generation and deterministic merging.

This module keeps multiprocessing logic inside ``na49_mixing`` so callers can
request parallel generation without reimplementing process orchestration.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from . import pdg_database
from . import rng_substreams
from .breit_wigner import BreitWigner
from .decays import (
    DecayMode,
    MESTAR_PPBAR_PI0_VARIANT_KEY,
    MESTAR_PPBAR_STRENGTH_KEY,
    PPBAR_PAIR_EXPONENT,
    ParticleDecays,
    build_kaon_decays,
    build_soft_resonance_decays,
    ppbar_pair_threshold,
    read_decays,
)
from .event import Event
from .generator import Generator
from .glauber import NucleusSpec, WoundedNucleonSampler
from . import isospin
from .reaction import Reaction


# ---------------------------------------------------------------------------
# Typed payloads for chunk results, serialized metadata, and merged output.
#
# The dictionaries below are produced by the chunk workers, consumed by
# :func:`merge_track_chunks`, embedded into run metadata by callers (e.g.
# ``na49_mixing._cli``), and returned by :func:`generate_tracks_chunked`.
# They describe that schema statically; construction stays plain dict
# literals at runtime.
# ---------------------------------------------------------------------------


class RealizedPrimaryBranch(TypedDict):
    """Realized branching table of a generated primary parent."""

    parent_pdg: int
    modes: "List[List[Any]]"


class BreitWignerMassLaw(TypedDict):
    type: Literal["breit_wigner"]
    params: List[float]


class TabulatedMassLaw(TypedDict):
    type: Literal["tabulated_pdf"]


class FixedPDGMassLaw(TypedDict):
    type: Literal["fixed_pdg"]
    mass_gev: float


MassLaw = Union[BreitWignerMassLaw, TabulatedMassLaw, FixedPDGMassLaw]


class PANucleusMeta(TypedDict):
    """Serialized p+A target description of a p+A chunk."""

    name: str
    A: int
    Z: int
    profile: str
    sigma_nn_inel_mb: float
    target_wound_range: Optional[List[int]]


class ChannelAuditAccumulators(TypedDict):
    """Raw mass-conditioned channel audit accumulators of one chunk.

    Carries the raw state the chunk's shared :class:`Event` accumulated
    over its events (the same accumulators :meth:`Event.decay_stats`
    divides): per parent PDG, the channel-selection attempt count (the
    denominator of the attempt-normalized tables), the summed explicitly
    renormalized requested weights, the realized success count, the summed
    daughter-slot openness fractions, and the summed and counted
    parent-side line-shape openness. Carrying the raw sums and counts
    (rather than the derived per-chunk fractions) lets
    :func:`generate_tracks_chunked` re-derive the merged audit tables from
    the summed counts with the same divisions the single-process path
    uses, instead of averaging per-chunk fractions.
    """

    channel_attempts: Dict[int, int]
    channel_requested_sum: Dict[int, Dict[Tuple[int, ...], float]]
    channel_realized: Dict[int, Dict[Tuple[int, ...], int]]
    channel_openness_sum: Dict[int, Dict[Tuple[int, ...], float]]
    channel_parent_openness_sum: Dict[int, Dict[Tuple[int, ...], float]]
    channel_parent_openness_count: Dict[int, Dict[Tuple[int, ...], int]]


class _TrackChunkBase(TypedDict):
    """Fields every chunk worker reports."""

    chunk_index: int
    n_events: int
    seed: int
    pdg: np.ndarray      # dtype int32
    xf: np.ndarray       # dtype float32
    pt: np.ndarray       # dtype float32
    source_side: np.ndarray  # dtype int8
    weak_feed: np.ndarray    # dtype int8
    gen_depth: np.ndarray    # dtype int16
    n_decayed_parents: int
    n_decay_kinematic_failures: int
    n_decay_attempts: int
    primary_parent: Optional[RealizedPrimaryBranch]
    mass_law: MassLaw
    weak_feed_policy: str
    n_weak_dropped: int
    n_primary_generated: int
    n_primary_decayed: int
    primary_failed: int
    primary_pdg_stats: Dict[int, List[int]]
    n_terminal_unstable: int
    channel_audit_accumulators: ChannelAuditAccumulators


class TrackChunkResult(_TrackChunkBase, total=False):
    """One chunk worker payload.

    ``n_wounds`` and ``pa_nucleus`` appear only on p+A chunks; pp chunks
    omit them. Legacy cache chunks may additionally lack optional per-track
    arrays, which :func:`merge_track_chunks` synthesizes as zeros, and the
    ``channel_audit_accumulators`` payload, which
    :func:`generate_tracks_chunked` treats as an empty accumulator.
    """

    n_wounds: np.ndarray   # dtype int16
    pa_nucleus: PANucleusMeta


class ChunkSummary(TypedDict):
    """Per-chunk metadata block inside :func:`generate_tracks_chunked` output."""

    chunk_index: int
    n_events: int
    seed: int
    tracks: int
    n_decayed_parents: int
    n_decay_kinematic_failures: int
    n_decay_attempts: int


class MergedGenerationResult(TypedDict):
    """Merged result of :func:`generate_tracks_chunked` for one component.

    The channel audit tables (``channel_requested``, ``channel_realized``,
    ``openness_fraction``, ``parent_openness_fraction``) are merged across
    chunks with the same keying, denominators, and derivation as
    :meth:`Event.decay_stats` of the single-process path: counts are
    summed across chunks and the fractions are recomputed from the summed
    counts (``parent_openness_fraction`` divides by the summed per-channel
    selection count).
    """

    pdg: np.ndarray             # dtype int32
    xf: np.ndarray              # dtype float32
    pt: np.ndarray              # dtype float32
    source_side: np.ndarray     # dtype int8
    weak_feed: np.ndarray       # dtype int8
    gen_depth: np.ndarray       # dtype int16
    n_wounds: np.ndarray        # dtype int16
    pa_nucleus: Optional[PANucleusMeta]
    n_weak_feed_tracks: int
    n_terminal_unstable: int
    primary_parent: Optional[RealizedPrimaryBranch]
    mass_law: Optional[MassLaw]
    chunks: List[ChunkSummary]
    n_events: int
    n_decayed_parents: int
    n_decay_kinematic_failures: int
    n_decay_attempts: int
    weak_feed_policy: str
    n_weak_dropped: int
    n_primary_generated: int
    n_primary_decayed: int
    primary_failed: int
    primary_pdg_stats: Dict[int, List[int]]
    channel_requested: Dict[int, Dict[Tuple[int, ...], float]]
    channel_realized: Dict[int, Dict[Tuple[int, ...], float]]
    openness_fraction: Dict[int, Dict[Tuple[int, ...], float]]
    parent_openness_fraction: Dict[int, Dict[Tuple[int, ...], float]]


def _require_int(value: object, name: str) -> int:
    """Normalize *value* to a Python ``int`` without fractional truncation.

    Python and NumPy integer scalars are accepted. Integer-valued floats
    (``1.0``) are accepted because they lose no information. Booleans are
    rejected even though they subclass ``int``. Any other type, or a float
    with a fractional part, raises ``ValueError`` naming *name*.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"GenerationConfig {name} must be an integer, got {value!r}"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        converted = int(value)
        if converted == value:
            return converted
    raise ValueError(
        f"GenerationConfig {name} must be an integer, got {value!r}"
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for one resonance component generation task."""

    gen_file: str
    n_events: int
    seed: int
    beam_momentum: float = 158.0
    max_decay_passes: int = 8
    decay_files: Tuple[str, ...] = ()
    use_kaon_decay_factory: bool = False
    use_soft_decay_factory: bool = False
    soft_decay_overrides: Tuple[Tuple[str, float], ...] = ()
    inject_antibaryon_resonances: bool = False
    fragmentation_mode: str = "both"
    weak_feed_policy: str = "inclusive_weak"
    # p+A wounded-nucleon extension: when a nucleus is set the event is
    # a superposition of 1 projectile-side emission and K target-side
    # emissions, where K is the sampled number of wounded target
    # nucleons. ``target_wound_range`` selects the centrality slice
    # (inclusive [lo, hi] on K); None keeps minimum bias. Accepted
    # slices are generated from the interaction-weighted distribution
    # restricted to [lo, hi] via ``sample_event_in_range``, not by
    # throwing away minimum-bias events. Events are counted as p+A
    # events, not elementary collisions.
    nucleus: Optional[NucleusSpec] = None
    target_wound_range: Optional[Tuple[int, int]] = None
    # Elementary beam/target nucleon species for n+p / n+n reactions.
    # Defaults keep the historical p+p path bit-identical, including RNG
    # consumption order. When either side is a neutron (2112), the
    # corresponding fragmentation side uses the isospin-reflected
    # primary identity (see na49_mixing.isospin).
    beam_pdg: int = 2212
    target_pdg: int = 2212
    # RNG substream manifest hook (Round 1, card 5, additive): when set,
    # the value is the run directory under which each chunk worker
    # writes the per-event substream manifest of its events, one file
    # per replica (chunk) at
    # ``rng_substreams.substream_manifest_path(value, chunk_index)``.
    # The manifest is recorded, not enforced: writing it never consumes
    # randomness and never alters the generation stream.  ``None`` (the
    # default) keeps the legacy behavior bit-identical.
    substream_manifest_path: Optional[str] = None

    def __post_init__(self) -> None:
        n_events = _require_int(self.n_events, "n_events")
        if n_events < 0:
            raise ValueError(
                f"GenerationConfig n_events must be >= 0, got {self.n_events!r}"
            )
        object.__setattr__(self, "n_events", n_events)
        max_decay_passes = _require_int(self.max_decay_passes, "max_decay_passes")
        if max_decay_passes < 1:
            raise ValueError(
                f"GenerationConfig max_decay_passes must be >= 1, got "
                f"{self.max_decay_passes!r}"
            )
        object.__setattr__(self, "max_decay_passes", max_decay_passes)
        seed = _require_int(self.seed, "seed")
        if seed < 0:
            raise ValueError(
                f"GenerationConfig seed must be >= 0, got {self.seed!r}"
            )
        object.__setattr__(self, "seed", seed)
        beam_momentum = float(self.beam_momentum)
        if not math.isfinite(beam_momentum) or beam_momentum <= 0.0:
            raise ValueError(
                "GenerationConfig beam_momentum must be a finite positive "
                f"float, got {self.beam_momentum!r}"
            )
        object.__setattr__(self, "beam_momentum", beam_momentum)
        object.__setattr__(
            self, "weak_feed_policy", _normalize_weak_feed_policy(self.weak_feed_policy)
        )
        _validate_species_config(self)
        if self.nucleus is not None:
            if not isinstance(self.nucleus, NucleusSpec):
                raise ValueError(
                    "GenerationConfig.nucleus must be a "
                    f"NucleusSpec, got {type(self.nucleus)!r}"
                )
            if self.fragmentation_mode != "both":
                raise ValueError(
                    "GenerationConfig with a nucleus requires "
                    f"fragmentation_mode='both', got "
                    f"{self.fragmentation_mode!r}"
                )
        if self.target_wound_range is not None:
            if self.nucleus is None:
                raise ValueError(
                    "GenerationConfig target_wound_range requires nucleus, "
                    "got nucleus=None"
                )
            lo, hi = self.target_wound_range
            lo_i = _require_int(lo, "target_wound_range lo")
            hi_i = _require_int(hi, "target_wound_range hi")
            if lo_i < 1 or hi_i < lo_i:
                raise ValueError(
                    "GenerationConfig target_wound_range must satisfy "
                    f"1 <= lo <= hi, got ({lo!r}, {hi!r})"
                )
            object.__setattr__(
                self, "target_wound_range", (lo_i, hi_i)
            )
            if int(self.nucleus.A) == 1 and not (lo_i <= 1 <= hi_i):
                raise ValueError(
                    "GenerationConfig point target (A=1) always has K=1; "
                    f"target_wound_range must include 1, got ({lo_i}, {hi_i})"
                )
            cap_a = int(self.nucleus.A)
            if lo_i > cap_a:
                raise ValueError(
                    "GenerationConfig target_wound_range "
                    f"({lo_i}, {hi_i}) is unreachable for nucleus "
                    f"{self.nucleus.name} (A={cap_a}): lo > A, and "
                    "the truncated-at-A sampler assigns K > A zero "
                    "probability."
                )
            fraction = _wound_range_fraction(self.nucleus, lo_i, hi_i)
            if fraction <= MIN_WOUND_RANGE_FRACTION:
                raise ValueError(
                    "GenerationConfig target_wound_range "
                    f"({lo_i}, {hi_i}) is unreachable for nucleus "
                    f"{self.nucleus.name} (A={self.nucleus.A}): the "
                    f"interaction-weighted slice probability is "
                    f"{fraction:.3e}, identically zero within double "
                    "precision, so the conditioned sampler has "
                    "nothing to draw."
                )


FRAGMENTATION_MODES: tuple[str, ...] = ("projectile", "target", "both")
WEAK_FEED_POLICIES: tuple[str, ...] = ("inclusive_weak", "primary_only")

# Smallest acceptable interaction-weighted probability of a requested
# centrality slice ``target_wound_range``. Slices are generated directly
# from the interaction-weighted distribution restricted to [lo, hi]
# (``WoundedNucleonSampler.sample_event_in_range``), so reachability no
# longer depends on how rare a slice is under minimum bias: any slice
# with strictly positive probability produces events at two uniform
# draws per event. The check therefore only refuses slices whose
# restricted probability vanishes, either structurally (``lo > A``, and
# point-target ranges excluding K = 1, refused earlier) or through
# double-precision underflow of the analytic fraction.
MIN_WOUND_RANGE_FRACTION = 0.0


def _wound_range_fraction(nucleus: NucleusSpec, lo: int, hi: int) -> float:
    """Expected minimum-bias fraction of ``lo <= K <= hi`` for *nucleus*.

    The result is deterministic: building a ``WoundedNucleonSampler``
    consumes no random numbers, so the seeded generator only satisfies
    the constructor signature.
    """

    sampler = WoundedNucleonSampler(nucleus, rng=np.random.default_rng(0))
    return float(sampler.fraction_for_wound_range(lo, hi))


_NSTAR_LEVEL_IDS: tuple[tuple[int, ...], ...] = (
    (9911201, 9912201, -9911201),
    (9921201, 9922201, -9921201),
    (9931201, 9932201, -9931201),
)
_MSTAR_LEVEL_IDS: tuple[tuple[int, ...], ...] = (
    (9900113,),
    (9900213,),
    (9900313,),
)


# ---------------------------------------------------------------------------
# Stage-B candidate configuration (card TASK-2026-08-31-resonance-decay-fit-
# workflow rev 2; pre-registered constants, consumed by the staged fit
# driver).  The one-meson-at-a-time candidates are B-450 (level 1,
# meson-cascade only), B-850 (level 2, pair-capable) and B-1250 (level 3,
# pair-capable).  Every bound below is a named configuration constant, not
# an inline magic number, and the box payloads of
# :func:`stage_b_candidate_boxes` carry them in driver-consumable form.
# ---------------------------------------------------------------------------

# Pair-capable levels (card rev 2 working interpretation): l2 and l3.
# Level l1 registers no pair channel; raising it to pair-capable masses
# would compound the whole tower and is out of the card's scope.
STAGE_B_PAIR_CAPABLE_LEVELS: tuple[int, ...] = (2, 3)

# Daughter-side box for the pair-capable candidates: m_M*(k) lives in
# [2 m_p + eps, 2.5] GeV, so the two-body p pbar channel is open at the
# M* pole everywhere in the box.
STAGE_B_MESTAR_EPS_GEV = 0.005
STAGE_B_MESTAR_MASS_MAX_GEV = 2.5

# Parent-side box: m_N*(k) >= m_N*(k-1) + m_M*(k, upper) + delta, so the
# N* -> N*'M* channel is open at the N* pole everywhere in the box,
# including the corner where the meson mass is maximal.
STAGE_B_PARENT_MARGIN_GEV = 0.10

# Frozen chain anchors of this card's scope: m_N*(0) is the N*(1520)
# daughter of level 1; the lower N* parents of levels 2/3 and the l1
# meson stay at their default tower values (one candidate at a time).
STAGE_B_NSTAR0_FROZEN_GEV = 1.520
_STAGE_B_NSTAR_LOWER_FROZEN_GEV = {2: 1.900, 3: 2.100}
_STAGE_B_MESTAR_FROZEN_GEV = {1: 0.450, 2: 0.850, 3: 1.250}

# Window widening (once per candidate): tower-window upper bounds far
# enough out that neither the linear (lo, mid, hi) default spacing nor
# the .gen Breit-Wigner peaks clamp the box. The fitted spacing comes
# from the per-level mass keys, which take precedence over the window.
STAGE_B_NSTAR_WINDOW_MAX_GEV = 5.0
STAGE_B_MESTAR_WINDOW_MAX_GEV = 2.5

# D4 aggregate parent strength: the fitted s enters
# ``nstar_eff_l{k}_br_nstarmeson`` with the compensating
# ``nstar_eff_l{k}_br_nstarpi = base_k - s`` (see
# :func:`stage_b_strength_split`); base_k defaults to the frozen tower
# values. D4 (parent side) and D5 (daughter side) never share one budget.
STAGE_B_BASE_NSTARPI = {1: 0.15, 2: 0.30, 3: 0.45}
# Wide-budget round 2 (pre-registered 2026-09-02 in
# notes/resonance_fit_stage_progress.md): the cascade branch may grow past
# the frozen pi budget with br_nstarpi clamped at zero. Twice the frozen
# budget on the pair-capable levels; level 1 is unchanged (its round-1
# mass surface was flat and the level-1 candidate was not retained).
STAGE_B_STRENGTH_BUDGET_WIDE = {1: 0.15, 2: 0.60, 3: 0.90}

# D5 pair-family strength (daughter side): one aggregate c shared by l2
# and l3, thresholded shape P_ppbar(M) = c * (1 - (M_th/M)^2)^n evaluated
# at the realized parent mass; the exponent n and the channel daughters
# are recorded in na49_mixing.decays. The strength key and its default
# are defined there so the decay factory and the driver share one source
# of truth (review REV-5b2f37d-03: no local copy of the default here).


def stage_b_mestar_mass_range() -> tuple[float, float]:
    """Daughter-side box ``[2 m_p + eps, 2.5]`` GeV (pair-capable levels)."""
    return (
        ppbar_pair_threshold() + STAGE_B_MESTAR_EPS_GEV,
        STAGE_B_MESTAR_MASS_MAX_GEV,
    )


def _build_stage_b_boxes(
    meson_range: tuple[float, float],
    mestar_window_max: float,
    nstar_floor_meson_bound: float,
) -> Dict[int, Dict[str, object]]:
    """Build the pre-registered boxes for one M*-range variant.

    ``nstar_floor_meson_bound`` sets the parent-side floor via the bound
    chain (m_N* >= m_N*(k-1) + m_M*(bound) + margin). The wide-M* variant
    keeps the standard bound for the declared N* floor (the round-1 box
    arithmetic at m_M* = 2.5) while widening the M* axis: corners where
    m_N* < m_N*(k-1) + m_M* + margin are bound-chain-closed per point,
    which is the round-4 ridge probe (pre-registered 2026-09-03 in
    notes/resonance_fit_stage_progress.md); the openness machinery
    records them and the cascade channel is ineligible there.
    """
    boxes: Dict[int, Dict[str, object]] = {}
    for level in (1, 2, 3):
        if level in STAGE_B_PAIR_CAPABLE_LEVELS:
            level_meson_range: tuple[float, float] = meson_range
        else:
            frozen = float(_STAGE_B_MESTAR_FROZEN_GEV[level])
            level_meson_range = (frozen, frozen)
        lower_parent = (
            STAGE_B_NSTAR0_FROZEN_GEV
            if level == 1
            else float(_STAGE_B_NSTAR_LOWER_FROZEN_GEV[level])
        )
        floor_bound = (
            nstar_floor_meson_bound
            if level in STAGE_B_PAIR_CAPABLE_LEVELS
            else float(_STAGE_B_MESTAR_FROZEN_GEV[level])
        )
        nstar_min = lower_parent + floor_bound + STAGE_B_PARENT_MARGIN_GEV
        boxes[level] = {
            "level": level,
            "label": f"B-{int(round(_STAGE_B_MESTAR_FROZEN_GEV[level] * 1000))}",
            "pair_capable": level in STAGE_B_PAIR_CAPABLE_LEVELS,
            "mestar_mass_range": level_meson_range,
            "nstar_mass_range": (nstar_min, STAGE_B_NSTAR_WINDOW_MAX_GEV),
            "window_overrides": {
                "nstar_eff_tower_max_mass": STAGE_B_NSTAR_WINDOW_MAX_GEV,
                "mestar_eff_tower_max_mass": mestar_window_max,
            },
            "strength_keys": {
                "parent": f"nstar_eff_l{level}_br_nstarmeson",
                "parent_compensation": f"nstar_eff_l{level}_br_nstarpi",
                "pair": MESTAR_PPBAR_STRENGTH_KEY,
                "pair_pi0_variant": MESTAR_PPBAR_PI0_VARIANT_KEY,
            },
            "pair_exponent": PPBAR_PAIR_EXPONENT,
        }
    return boxes


def stage_b_candidate_boxes() -> Dict[int, Dict[str, object]]:
    """Pre-registered Stage-B candidate boxes, keyed by level.

    Each box records the free-mass ranges implied by the bound chain, the
    window-widening overrides, and the D4/D5 strength keys, so the staged
    fit driver (card phase P3/P4) consumes configuration instead of
    restating the arithmetic. Concrete parent-side minima from the bound
    chain: B-450 needs m_N*(1) >= 1.520 + 0.450 + 0.10 = 2.07 GeV (l1
    meson frozen), B-850 needs m_N*(2) >= 1.900 + 2.5 + 0.10 = 4.50 GeV
    (l1 frozen at 1.900), B-1250 needs m_N*(3) >= 2.100 + 2.5 + 0.10
    = 4.70 GeV (l2 frozen at 2.100).
    """
    return _build_stage_b_boxes(
        stage_b_mestar_mass_range(),
        STAGE_B_MESTAR_WINDOW_MAX_GEV,
        stage_b_mestar_mass_range()[1],
    )


def stage_b_candidate_boxes_wide_mestar() -> Dict[int, Dict[str, object]]:
    """Wide-M* box variant (round 4 E3, pre-registered 2026-09-03).

    The M* axis extends to 2.75 GeV with the M* tower window widened to
    match, while the declared N* range keeps the standard bound-chain
    floor (4.50 GeV for level 2, from the 2.5 GeV bound). Corners where
    the parent bound chain closes at the widened M* mass are handled per
    point by the C(M) channel construction and recorded by the openness
    machinery.
    """
    wide = (ppbar_pair_threshold() + STAGE_B_MESTAR_EPS_GEV, 2.75)
    return _build_stage_b_boxes(wide, 2.75, stage_b_mestar_mass_range()[1])


def stage_b_strength_split(
    level: int,
    strength: float,
    base_nstarpi: Optional[float] = None,
    budget: Optional[float] = None,
) -> tuple[float, float]:
    """D4 compensating split ``(br_nstarmeson, br_nstarpi)`` for one level.

    The aggregate parent strength s enters
    ``nstar_eff_l{level}_br_nstarmeson`` while
    ``nstar_eff_l{level}_br_nstarpi`` absorbs the complement ``base - s``,
    so the level branch budget stays physical by construction and
    ``s = 0`` reproduces the disabled state exactly. s is clamped into
    ``[0, budget]`` (the frozen ``base`` when no wide budget is given)
    and the complement clamps at zero, so a wide-budget round
    (``STAGE_B_STRENGTH_BUDGET_WIDE``, pre-registered in
    notes/resonance_fit_stage_progress.md) grows the cascade branch past
    the exhausted pi branch. Every s within ``[0, base]`` reproduces the
    frozen-budget split exactly, which keeps the round-1 grid points
    nested inside the wide round-2 strength axes.
    """
    base = float(
        STAGE_B_BASE_NSTARPI[int(level)] if base_nstarpi is None else base_nstarpi
    )
    cap = base if budget is None else float(budget)
    s = min(max(float(strength), 0.0), cap)
    return (s, max(base - s, 0.0))


def _normalize_fragmentation_mode(mode: str) -> str:
    val = str(mode).strip().lower()
    if val not in FRAGMENTATION_MODES:
        raise ValueError(
            f"Unsupported fragmentation_mode='{mode}'. Expected one of {FRAGMENTATION_MODES}."
        )
    return val


def _normalize_weak_feed_policy(policy: str) -> str:
    val = str(policy).strip().lower()
    if val not in WEAK_FEED_POLICIES:
        raise ValueError(
            f"Unsupported weak_feed_policy='{policy}'. Expected one of {WEAK_FEED_POLICIES}."
        )
    return val


_NUCLEON_PDGS = (2212, 2112)


def _card_primary_pdg(gen_file: str) -> int:
    """First-line PDG id of a ``.gen`` card (the generated primary)."""
    with open(str(gen_file), "r", errors="replace") as fh:
        token = fh.read().split()[0]
    return int(token)


def _validate_species_config(config: "GenerationConfig") -> None:
    """Validate beam/target species and neutron-beam card compatibility."""
    beam = int(getattr(config, "beam_pdg", 2212))
    target = int(getattr(config, "target_pdg", 2212))
    if beam not in _NUCLEON_PDGS or target not in _NUCLEON_PDGS:
        raise ValueError(
            "GenerationConfig beam_pdg/target_pdg must be nucleon ids "
            f"{_NUCLEON_PDGS}, got ({beam}, {target})."
        )
    if 2112 not in (beam, target):
        return
    primary = _card_primary_pdg(str(config.gen_file))
    if primary in isospin.ANTINEUTRON_PROHIBITED_PRIMARIES:
        raise ValueError(
            f"gen card primary {primary} has no charged anti-N* partner in "
            "the decay tables and is restricted to proton projectiles; "
            "neutron beams require an isospin-reflectable primary."
        )


def _mirror_projectile_to_target_in_place(part, reaction: Reaction) -> None:
    """Mirror a generated projectile-side parent to the target side in CMS.

    We keep transverse momentum and energy in CMS, and flip longitudinal CMS
    momentum sign. The mirrored state is then boosted back to LAB.
    """
    pz_cms = float(part.get_pz_cms(reaction))
    e_cms = float(part.get_e_cms(reaction))
    pz_lab_target = reaction.gamma() * (-pz_cms + reaction.betha() * e_cms)
    part.set_p(part.get_px(), part.get_py(), pz_lab_target)


def split_event_counts(total_events: int, n_chunks: int) -> List[int]:
    """Split ``total_events`` into ``n_chunks`` near-equal nonnegative counts."""

    total = max(int(total_events), 0)
    chunks = max(int(n_chunks), 1)
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def _soft_overrides_dict(config: GenerationConfig) -> Dict[str, float]:
    return {str(k): float(v) for k, v in config.soft_decay_overrides}


# Soft-override key for the resolved-region cap (card decision D3). Inactive
# by default: when the key is absent the resolved cap reduces to the
# realized parent mass inside the decay engine.
_RESOLVED_MATCH_MASS_KEY = "resolved_match_mass"


def _resolved_match_mass_override(config: GenerationConfig) -> Optional[float]:
    """Resolve the optional ``resolved_match_mass`` soft override (D3).

    Returns ``None`` (inactive) unless the soft-override set carries the
    key; an active value must be a finite positive float. When active,
    resolved-state line shapes truncate to ``min(M_source, M_match)``;
    when inactive the cap reduces to the realized parent mass.
    """
    val = _soft_overrides_dict(config).get(_RESOLVED_MATCH_MASS_KEY)
    if val is None:
        return None
    val = float(val)
    if not (math.isfinite(val) and val > 0.0):
        raise ValueError(
            f"soft override {_RESOLVED_MATCH_MASS_KEY} must be a finite "
            f"positive float, got {val!r}"
        )
    return val


def _compute_level_masses(min_mass: float, max_mass: float) -> tuple[float, float, float]:
    lo = float(min_mass)
    hi = float(max_mass)
    if hi < lo:
        hi = lo
    mid = 0.5 * (lo + hi)
    return lo, mid, hi


def _resolve_level_masses(
    overrides: Dict[str, float],
    *,
    prefix: str,
    default_masses: tuple[float, float, float],
) -> tuple[float, float, float]:
    masses = [float(default_masses[0]), float(default_masses[1]), float(default_masses[2])]
    lo_key = f"{prefix}_tower_min_mass"
    hi_key = f"{prefix}_tower_max_mass"

    if lo_key in overrides or hi_key in overrides:
        lo = float(overrides.get(lo_key, masses[0]))
        hi = float(overrides.get(hi_key, masses[2]))
        masses = list(_compute_level_masses(lo, hi))

    for i in range(3):
        key = f"{prefix}_l{i + 1}_mass"
        if key in overrides:
            masses[i] = float(overrides[key])
    return float(masses[0]), float(masses[1]), float(masses[2])


def _resolve_level_widths(
    overrides: Dict[str, float],
    *,
    prefix: str,
    default_widths: tuple[float, float, float],
) -> tuple[float, float, float]:
    widths = [float(default_widths[0]), float(default_widths[1]), float(default_widths[2])]
    for i in range(3):
        key = f"{prefix}_l{i + 1}_width"
        if key in overrides:
            widths[i] = max(float(overrides[key]), 0.0)
    return float(widths[0]), float(widths[1]), float(widths[2])


def _resolve_effective_tower_overrides(
    overrides: Dict[str, float],
) -> Dict[int, tuple[float, float]]:
    """Resolve soft-override keys into per-PDG (mass,width) updates.

    Supported keys:
    - `nstar_eff_tower_min_mass`, `nstar_eff_tower_max_mass`
    - `mestar_eff_tower_min_mass`, `mestar_eff_tower_max_mass`
    - `nstar_eff_l{1,2,3}_mass`, `nstar_eff_l{1,2,3}_width`
    - `mestar_eff_l{1,2,3}_mass`, `mestar_eff_l{1,2,3}_width`
    """
    nstar_masses = _resolve_level_masses(
        overrides,
        prefix="nstar_eff",
        default_masses=(1.9, 2.1, 2.3),
    )
    nstar_widths = _resolve_level_widths(
        overrides,
        prefix="nstar_eff",
        default_widths=(0.25, 0.30, 0.35),
    )
    mestar_masses = _resolve_level_masses(
        overrides,
        prefix="mestar_eff",
        default_masses=(0.45, 0.85, 1.25),
    )
    mestar_widths = _resolve_level_widths(
        overrides,
        prefix="mestar_eff",
        default_widths=(0.18, 0.22, 0.28),
    )

    payload: Dict[int, tuple[float, float]] = {}
    for level, ids in enumerate(_NSTAR_LEVEL_IDS):
        for pid in ids:
            payload[int(pid)] = (float(nstar_masses[level]), float(nstar_widths[level]))
    for level, ids in enumerate(_MSTAR_LEVEL_IDS):
        for pid in ids:
            payload[int(pid)] = (float(mestar_masses[level]), float(mestar_widths[level]))
    return payload


def _effective_tower_updates(config: GenerationConfig) -> Dict[int, tuple[float, float]]:
    """Resolve soft overrides into per-PDG (mass, width) without mutating PDG."""
    return _resolve_effective_tower_overrides(_soft_overrides_dict(config))


def _patch_generator_mass_width(
    gen: Generator,
    updates: Dict[int, tuple[float, float]],
) -> None:
    """Apply tower mass/width overrides to a loaded generator's sampling law.

    Parent masses for effective N*/M* states are typically sampled from the
    Breit-Wigner parameters embedded in the ``.gen`` card. Updating only the
    PDG table would not change those samples; this rewrites the BW peak/width
    (or fixed PDG mass cache) for the generator's PDG id when an override is
    present.

    The final parameter set is composed locally and committed through a
    single :meth:`BreitWigner.set_parameters` call. The update sequence
    passes through states that would fail eager per-parameter validation
    (the threshold is lowered before the peak moves; the damping window is
    widened in two steps), so one-at-a-time ``set_parameter`` updates are
    avoided here by design.

    Support coverage of grown masses (stage-B box verification, card rev
    2): after patching, the pole is always inside the sampling support.
    When the new peak reaches the old damping window (``mass >= dumplow``)
    the window is stretched to ``[mass + 0.2*span, mass + span]`` with
    ``span >= 0.3``, so ``dumphigh > mass`` holds; when it does not, the
    old window already covered the peak. The lower support edge
    (``schwel``) is the p-wave threshold and is lowered when the peak
    moves to or below it. A 2.5 GeV meson or a 4.7 GeV nucleon level
    therefore never samples outside the support, and the truncated draw
    helpers clip against the same ``[schwel, dumphigh]`` window, so no
    additional support widening is required here.
    """
    if not updates:
        return
    pid = int(getattr(gen, "_id", 0))
    pair = updates.get(pid)
    if pair is None:
        return
    mass, width = float(pair[0]), float(pair[1])
    bw = getattr(gen, "_bw", None)
    if bw is not None and gen._mass_type in (1, 3):
        # params: [schwel, rmas, brei, pbrei, dumplow, dumphigh]
        params = list(bw.params)
        if math.isfinite(mass) and mass > 0.0:
            if mass <= params[0]:
                # Keep sampling support valid if peak moves near/under threshold.
                params[0] = max(mass * 0.5, 0.0)
            params[1] = mass
            if mass >= params[4]:
                # Stretch damping window above the new peak.
                span = max(params[5] - params[4], 0.3)
                params[4] = mass + 0.2 * span
                params[5] = mass + span
        if math.isfinite(width) and width >= 0.0:
            params[2] = width
        bw.set_parameters(*params)
    if math.isfinite(mass) and mass > 0.0:
        gen._id_mass = float(mass)
        if gen._mass_type == 0:
            # Fixed-mass mode: nothing else to do; generate() uses _id_mass.
            pass


def _build_decay_chain(config: GenerationConfig, rng: np.random.Generator) -> dict:
    chain: dict = {}
    soft_decay_overrides = _soft_overrides_dict(config)
    if config.use_soft_decay_factory:
        chain.update(build_soft_resonance_decays(rng=rng, branching_overrides=soft_decay_overrides))
    if config.use_kaon_decay_factory:
        chain.update(build_kaon_decays(rng=rng))
    for path in config.decay_files:
        chain.update(read_decays(str(path), rng=rng))
    if config.inject_antibaryon_resonances:
        _inject_antibaryon_resonance_decays(chain, rng, soft_decay_overrides=soft_decay_overrides)
    return chain


def _inject_antibaryon_resonance_decays(
    chain: dict,
    rng: np.random.Generator,
    *,
    soft_decay_overrides: dict[str, float] | None = None,
) -> None:
    """Add anti-baryon channels needed for antiproton resonance components.

    Entries already present in *chain* win: the soft-resonance factory and
    decay files define physics deliberately.  PDG ``-1114`` is exclusively
    the physical anti-Delta- (charge +1, decaying to nbar pi+) in every
    builder; the historical matter Delta+ alias is retired and generator
    cards use 2214 directly.  Injection fills only the anti-baryon parents
    that are still missing (anti-Delta charge states, anti-N* and tower
    states).
    """

    def _ovr(key: str, default: float) -> float:
        if not soft_decay_overrides:
            return float(default)
        return float(soft_decay_overrides.get(key, default))

    # Anti-Delta daughters sample seeded Breit-Wigner spectral masses, matching
    # the treatment of matter Delta daughters in the soft-resonance factory.
    anti_delta_bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0, rng=rng)
    rho_bw = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5, rng=rng)

    m1_0 = 9900113
    m2_0 = 9900213
    m3_0 = 9900313

    def _add(parent_id: int, modes: List[Tuple[float, List[int]]]) -> None:
        if int(parent_id) in chain:
            # Existing definition from the soft-resonance factory or a decay
            # file takes precedence over injected anti-baryon defaults.
            return
        d = ParticleDecays(parent_id, rng=rng)
        cum = 0.0
        for prob, daughters in modes:
            prob = max(float(prob), 0.0)
            if prob <= 0.0:
                continue
            cum += prob
            bws = [
                anti_delta_bw if daughter in {-2224, -2214, -2114, -1114}
                else (rho_bw if daughter in {113, 213, -213} else None)
                for daughter in daughters
            ]
            d.add_mode(min(cum, 1.0), DecayMode(len(daughters), daughters, bws))
        if d._probabilities:
            d._probabilities[-1] = 1.0
        chain[parent_id] = d

    def _add_antinstar(
        parent_id: int,
        br_npi: float,
        br_deltapi: float = 0.0,
        br_neta: float = 0.0,
        br_nrho: float = 0.0,
        br_lambdak: float = 0.0,
        br_nstarpi: float = 0.0,
        br_nstarmeson: float = 0.0,
        lower_antinstar_id: int | None = None,
        lower_antinstar_meson_id: int | None = None,
        meson_tower_id: int | None = None,
        charged_lower_id: int | None = None,
    ) -> None:
        br_npi = max(float(br_npi), 0.0)
        br_deltapi = max(float(br_deltapi), 0.0)
        br_neta = max(float(br_neta), 0.0)
        br_nrho = max(float(br_nrho), 0.0)
        br_lambdak = max(float(br_lambdak), 0.0)
        br_nstarpi = max(float(br_nstarpi), 0.0)
        br_nstarmeson = max(float(br_nstarmeson), 0.0)
        total = br_npi + br_deltapi + br_neta + br_nrho + br_lambdak + br_nstarpi + br_nstarmeson
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            br_npi *= scale
            br_deltapi *= scale
            br_neta *= scale
            br_nrho *= scale
            br_lambdak *= scale
            br_nstarpi *= scale
            br_nstarmeson *= scale

        modes: List[Tuple[float, List[int]]] = [
            (br_npi * (2.0 / 3.0), [-2212, 211]),   # pbar pi+
            (br_npi * (1.0 / 3.0), [-2112, 111]),   # nbar pi0
            # Neutral anti-N* -> anti-Delta pi: exact charge conjugate of
            # N*0 -> Delta+ pi- (1/6), Delta0 pi0 (2/6), Delta- pi+ (3/6).
            (br_deltapi * (1.0 / 6.0), [-2214, 211]),   # anti-Delta- pi+
            (br_deltapi * (2.0 / 6.0), [-2114, 111]),   # anti-Delta0 pi0
            (br_deltapi * (3.0 / 6.0), [-1114, -211]),  # anti-Delta+ pi-
        ]
        if br_neta > 0.0:
            modes.append((br_neta, [-2112, 221]))   # nbar eta
        if br_nrho > 0.0:
            modes.append((br_nrho * (2.0 / 3.0), [-2212, 213]))  # pbar rho+
            modes.append((br_nrho * (1.0 / 3.0), [-2112, 113]))  # nbar rho0
        if br_lambdak > 0.0:
            modes.append((br_lambdak, [-3122, -311]))  # anti-Lambda anti-K0

        if br_nstarpi > 0.0 and lower_antinstar_id is not None:
            # Charge split mirrors the matter tower: 1/3 pi0 on the neutral
            # daughter, 2/3 pi+ on the charged anti-tower partner.
            modes.append((br_nstarpi * (1.0 / 3.0),
                          [lower_antinstar_id, 111]))
            charged = charged_lower_id or lower_antinstar_id
            modes.append((br_nstarpi * (2.0 / 3.0),
                          [charged, 211]))
        if br_nstarmeson > 0.0 and meson_tower_id is not None:
            target = lower_antinstar_meson_id if lower_antinstar_meson_id is not None else lower_antinstar_id
            if target is not None:
                modes.append((br_nstarmeson, [target, meson_tower_id]))  # effective Rbar -> Rbar' M*

        residual = max(
            1.0 - br_npi - br_deltapi - br_neta - br_nrho - br_lambdak - br_nstarpi - br_nstarmeson,
            0.0,
        )
        _add(
            parent_id,
            modes
            + [
                (residual * (2.0 / 3.0), [-2212, 211]),
                (residual * (1.0 / 3.0), [-2112, 111]),
            ],
        )

    _add(-2224, [(1.0, [-2212, -211])])  # anti-Delta--
    _add(-2214, [(2.0 / 3.0, [-2212, 111]), (1.0 / 3.0, [-2112, -211])])  # anti-Delta-
    _add(-2114, [(1.0 / 3.0, [-2212, 211]), (2.0 / 3.0, [-2112, 111])])   # anti-Delta0
    _add(-1114, [(1.0, [-2112, 211])])  # anti-Delta- (charge +1)

    _add_antinstar(
        -12112,
        br_npi=_ovr("n1440_br_npi", 0.65),
        br_deltapi=_ovr("n1440_br_deltapi", 0.165),
    )  # anti-N*(1440)0
    _add_antinstar(
        -2116,
        br_npi=_ovr("n1520_br_npi", 0.60),
        br_deltapi=_ovr("n1520_br_deltapi", 0.28),
    )  # anti-N*(1520)0
    _add_antinstar(
        -22112,
        br_npi=_ovr("n1680_br_npi", 0.65),
        br_deltapi=_ovr("n1680_br_deltapi", 0.17),
    )  # anti-N*(1680)0

    _add_antinstar(
        -102112,
        br_npi=_ovr("n1535_br_npi", 0.42),
        br_neta=_ovr("n1535_br_neta", 0.45),
    )  # anti-N(1535)0
    _add_antinstar(
        -122112,
        br_npi=_ovr("n1650_br_npi", 0.65),
        br_deltapi=_ovr("n1650_br_deltapi", 0.15),
        br_neta=_ovr("n1650_br_neta", 0.08),
        br_lambdak=_ovr("n1650_br_lambdak", 0.12),
    )  # anti-N(1650)0
    _add_antinstar(
        -212114,
        br_npi=_ovr("n1720_br_npi", 0.15),
        br_deltapi=_ovr("n1720_br_deltapi", 0.35),
        br_nrho=_ovr("n1720_br_nrho", 0.35),
        br_neta=_ovr("n1720_br_neta", 0.05),
        br_lambdak=_ovr("n1720_br_lambdak", 0.10),
    )  # anti-N(1720)0

    def _add_charged_antitower(pid, br_npi, br_deltapi, br_nrho,
                               br_neta, br_lambdak, br_nstarmeson=0.0,
                               lower_charged_id=None, meson_tower_id=None):
        """Charge conjugate of the positive matter tower level.

        Branch totals are passed in through the same soft-decay override
        keys used by the matter partner (see call sites below), so an
        override retunes matter and charged anti towers identically.
        The Nbar-rho split is the exact charge conjugate of the matter
        isospin split p rho0 (1/3) / n rho+ (2/3).
        Like decays.py::_add_nstar_extended and _add_antinstar above, an
        over-unity branch-total sum is renormalized proportionally before
        any mode is built, so shared override keys retune both towers by
        rescaling instead of silently truncating channels at the
        cumulative guard. The br_nstarmeson link joins the renormalization
        set, so an out-of-band override rescales the whole charged tower
        exactly as on the matter tower and the neutral anti tower (review
        REV-5b2f37d-02). The renormalization scale itself differs between
        matter and charged towers under such overrides by construction:
        the charged tower carries no br_nstarpi branch to renormalize
        against, so its branch total is smaller and the shared families
        keep their declared ratios tower-locally.

        The stage-B N* -> N*' + M* link (br_nstarmeson) is mirrored for
        every level whose charged lower anti-partner exists in the frozen
        PDG table: levels 2 and 3 feed [-9912201, M*(2)] and
        [-9922201, M*(3)]. The level-1 conjugate daughter of the matter
        channel [N*(1520)+, M_eff(450)] would be -2216, which the frozen
        table does not define, so the charged level-1 partner stays
        terminal — consistent with its already terminal cascade design
        (no br_nstarpi channel either); its cascade fraction folds into
        the Nbar-pi closure like on the neutral anti tower.
        """
        br_npi = max(float(br_npi), 0.0)
        br_deltapi = max(float(br_deltapi), 0.0)
        br_nrho = max(float(br_nrho), 0.0)
        br_neta = max(float(br_neta), 0.0)
        br_lambdak = max(float(br_lambdak), 0.0)
        br_nstarmeson = max(float(br_nstarmeson), 0.0)
        total = br_npi + br_deltapi + br_nrho + br_neta + br_lambdak + br_nstarmeson
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            br_npi *= scale
            br_deltapi *= scale
            br_nrho *= scale
            br_neta *= scale
            br_lambdak *= scale
            br_nstarmeson *= scale
        d = ParticleDecays(pid, rng=rng)
        state = {"cum": 0.0}
        def _add_mode(prob, daughters):
            prob = max(float(prob), 0.0)
            if prob <= 0.0 or state["cum"] >= 1.0 - 1e-12:
                return
            state["cum"] += prob
            bws = [
                anti_delta_bw if dd in {-2224, -2214, -2114, -1114}
                else (rho_bw if dd in {113, 213, -213} else None)
                for dd in daughters
            ]
            d.add_mode(min(state["cum"], 1.0),
                       DecayMode(len(daughters), daughters, bws))
        # Nbar pi (conjugate of p pi0 / n pi+)
        _add_mode(br_npi * (1.0/3.0), [-2212, 111])
        _add_mode(br_npi * (2.0/3.0), [-2112, -211])
        # anti-Delta pi (conjugate of Delta++ pi- / Delta+ pi0 / Delta0 pi+)
        _add_mode(br_deltapi * 0.50, [-2224, 211])
        _add_mode(br_deltapi * 0.33, [-2214, 111])
        _add_mode(br_deltapi * 0.17, [-2114, -211])
        # Nbar eta
        _add_mode(br_neta, [-2212, 221])
        # Nbar rho (spectral rho): exact charge conjugate of the matter
        # positive tower split p rho0 (1/3) / n rho+ (2/3).
        _add_mode(br_nrho * (1.0/3.0), [-2212, 113])
        _add_mode(br_nrho * (2.0/3.0), [-2112, -213])
        # Lambdabar K- (charge conjugate of Lambda K+)
        _add_mode(br_lambdak, [-3122, -321])
        # Nbar* -> Nbar*' + M* link: charge conjugate of the matter
        # positive-tower nstarmeson channel (daughters conserve charge
        # because the lower charged anti-partner carries the parent's
        # charge and the M* states are neutral).
        if br_nstarmeson > 0.0 and lower_charged_id is not None and meson_tower_id is not None:
            _add_mode(br_nstarmeson, [lower_charged_id, meson_tower_id])
        resid = max(1.0 - state["cum"], 0.0)
        _add_mode(resid * (1.0/3.0), [-2212, 111])
        _add_mode(resid * (2.0/3.0), [-2112, -211])
        d._probabilities[-1] = 1.0
        # Self-guard: electric charge must be conserved in every mode.
        from .pdg_database import get_charge as _qc
        prev = 0.0
        qp = _qc(pid)
        for pr, m in zip(d._probabilities, d._modes):
            inc = pr - prev; prev = pr
            if inc <= 0.0:
                continue
            tot = sum(_qc(dd) for dd in m.daughters)
            if abs(tot - qp) > 1e-9:
                raise ValueError(
                    f"charged anti-tower {pid}: mode {m.daughters} "
                    f"violates charge ({tot} != {qp})"
                )
        chain[pid] = d

    # Effective high-mass anti-N* tower for unresolved resonance feed-down.
    _add_antinstar(
        -9911201,
        br_npi=_ovr("nstar_eff_l1_br_npi", 0.12),
        br_deltapi=_ovr("nstar_eff_l1_br_deltapi", 0.35),
        br_nrho=_ovr("nstar_eff_l1_br_nrho", 0.28),
        br_neta=_ovr("nstar_eff_l1_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l1_br_lambdak", 0.05),
        br_nstarpi=_ovr("nstar_eff_l1_br_nstarpi", 0.15),
        br_nstarmeson=_ovr("nstar_eff_l1_br_nstarmeson", 0.0),
        lower_antinstar_id=-212114,
        lower_antinstar_meson_id=-2116,
        meson_tower_id=m1_0,
        charged_lower_id=-212214,
    )  # anti-N_eff(1900)0
    _add_antinstar(
        -9921201,
        br_npi=_ovr("nstar_eff_l2_br_npi", 0.08),
        br_deltapi=_ovr("nstar_eff_l2_br_deltapi", 0.30),
        br_nrho=_ovr("nstar_eff_l2_br_nrho", 0.22),
        br_neta=_ovr("nstar_eff_l2_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l2_br_lambdak", 0.05),
        br_nstarpi=_ovr("nstar_eff_l2_br_nstarpi", 0.30),
        br_nstarmeson=_ovr("nstar_eff_l2_br_nstarmeson", 0.0),
        lower_antinstar_id=-9911201,
        lower_antinstar_meson_id=-9911201,
        meson_tower_id=m2_0,
        charged_lower_id=-9912201,
    )  # anti-N_eff(2100)0
    _add_antinstar(
        -9931201,
        br_npi=_ovr("nstar_eff_l3_br_npi", 0.06),
        br_deltapi=_ovr("nstar_eff_l3_br_deltapi", 0.22),
        br_nrho=_ovr("nstar_eff_l3_br_nrho", 0.17),
        br_neta=_ovr("nstar_eff_l3_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l3_br_lambdak", 0.05),
        br_nstarpi=_ovr("nstar_eff_l3_br_nstarpi", 0.45),
        br_nstarmeson=_ovr("nstar_eff_l3_br_nstarmeson", 0.0),
        lower_antinstar_id=-9921201,
        lower_antinstar_meson_id=-9921201,
        meson_tower_id=m3_0,
        charged_lower_id=-9922201,
    )

    # Charged anti-tower partners: branch totals flow through the SAME
    # soft-decay override keys as their matter positive-tower partners
    # (n1720_* for the N(1720)+ conjugate, nstar_eff_l{1,2,3}_* for the
    # effective levels), so matter and anti towers stay conjugate under
    # any override tuning.
    _add_charged_antitower(
        -212214,
        br_npi=_ovr("n1720_br_npi", 0.15),
        br_deltapi=_ovr("n1720_br_deltapi", 0.35),
        br_nrho=_ovr("n1720_br_nrho", 0.35),
        br_neta=_ovr("n1720_br_neta", 0.05),
        br_lambdak=_ovr("n1720_br_lambdak", 0.10),
    )  # anti-N(1720)-
    _add_charged_antitower(
        -9912201,
        br_npi=_ovr("nstar_eff_l1_br_npi", 0.12),
        br_deltapi=_ovr("nstar_eff_l1_br_deltapi", 0.35),
        br_nrho=_ovr("nstar_eff_l1_br_nrho", 0.28),
        br_neta=_ovr("nstar_eff_l1_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l1_br_lambdak", 0.05),
        # No br_nstarmeson here: the level-1 conjugate daughter of
        # N*(1520)+ would be -2216, which the frozen PDG table does not
        # define (see the docstring above).
    )  # anti-N_eff(1900)- charged partner
    _add_charged_antitower(
        -9922201,
        br_npi=_ovr("nstar_eff_l2_br_npi", 0.08),
        br_deltapi=_ovr("nstar_eff_l2_br_deltapi", 0.30),
        br_nrho=_ovr("nstar_eff_l2_br_nrho", 0.22),
        br_neta=_ovr("nstar_eff_l2_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l2_br_lambdak", 0.05),
        br_nstarmeson=_ovr("nstar_eff_l2_br_nstarmeson", 0.0),
        lower_charged_id=-9912201,
        meson_tower_id=m2_0,
    )  # anti-N_eff(2100)- charged partner
    _add_charged_antitower(
        -9932201,
        br_npi=_ovr("nstar_eff_l3_br_npi", 0.06),
        br_deltapi=_ovr("nstar_eff_l3_br_deltapi", 0.22),
        br_nrho=_ovr("nstar_eff_l3_br_nrho", 0.17),
        br_neta=_ovr("nstar_eff_l3_br_neta", 0.05),
        br_lambdak=_ovr("nstar_eff_l3_br_lambdak", 0.05),
        br_nstarmeson=_ovr("nstar_eff_l3_br_nstarmeson", 0.0),
        lower_charged_id=-9922201,
        meson_tower_id=m3_0,
    )  # anti-N_eff(2300)- charged partner



def _chunk_seed(base_seed: int, chunk_index: int) -> int:
    # Prime stride to reduce accidental overlap in nearby seeds.
    seed = (int(base_seed) + 7919 * int(chunk_index)) % (2**31 - 1)
    return seed if seed > 0 else 1


def _build_side_generator(
    reaction: Reaction,
    gen_file: str,
    updates: Dict[int, tuple],
    rng,
    *,
    side_pdg: int,
):
    """Load one fragmentation-side generator from a card.

    A neutron side uses the isospin-reflected primary identity; proton
    sides load the card unchanged. The mass law comes from the card body
    and is identical either way, so RNG draw counts per ``generate()``
    match across sides.
    """
    gen = Generator(reaction=reaction, rng=rng)
    gen.set_all(str(gen_file))
    _patch_generator_mass_width(gen, updates)
    if int(side_pdg) == 2112:
        isospin.reflect_generator_identity(gen)
    return gen


def _parent_line_shape_map(gens) -> Dict[int, BreitWigner]:
    """Primary line shapes for the parent-side openness audit (stage B).

    Collects the Breit-Wigner mass law of each fragmentation-side
    generator (after ``_patch_generator_mass_width`` has applied the
    tower overrides, so the recorded openness follows the fitted
    masses). Parents without a card line shape (fixed-mass or tabulated
    mass laws) contribute no entry and simply stay absent from the
    audit's ``parent_openness_fraction`` table.
    """
    shapes: Dict[int, BreitWigner] = {}
    for gen in gens:
        if gen is None:
            continue
        bw = getattr(gen, "_bw", None)
        if bw is None or getattr(gen, "_mass_type", 0) not in (1, 3):
            continue
        shapes[int(getattr(gen, "_id", 0))] = bw
    return shapes


def _collect_channel_audit_accumulators(ev: Event) -> ChannelAuditAccumulators:
    """Snapshot the shared Event's channel audit accumulators for one chunk.

    The chunk worker's Event accumulates the raw sums and counts behind
    the per-channel audit tables across every event of the chunk
    (``reset_decay_stats`` runs once at chunk start, so the payload is
    self-contained per chunk). This returns copies of those raw
    accumulators so the chunk payload carries them across the worker
    boundary; the copies share no mutable state with the Event. Reading
    the accumulators consumes no random numbers and cannot alter any draw.
    """
    return {
        "channel_attempts": dict(ev._channel_attempts),
        "channel_requested_sum": {
            int(pdg_id): dict(tbl)
            for pdg_id, tbl in ev._channel_requested_sum.items()
        },
        "channel_realized": {
            int(pdg_id): dict(tbl)
            for pdg_id, tbl in ev._channel_realized.items()
        },
        "channel_openness_sum": {
            int(pdg_id): dict(tbl)
            for pdg_id, tbl in ev._channel_openness_sum.items()
        },
        "channel_parent_openness_sum": {
            int(pdg_id): dict(tbl)
            for pdg_id, tbl in ev._channel_parent_openness_sum.items()
        },
        "channel_parent_openness_count": {
            int(pdg_id): dict(tbl)
            for pdg_id, tbl in ev._channel_parent_openness_count.items()
        },
    }


def _merge_channel_audit_accumulators(
    chunks: Sequence[TrackChunkResult],
) -> ChannelAuditAccumulators:
    """Sum the per-chunk channel audit accumulators in ``chunk_index`` order.

    Counts and weight/openness sums are additive, so the merged
    accumulators are exactly the raw state one shared Event would have
    accumulated had every chunk's events run through it, which is what
    makes the derived merged tables the same fractions of the summed
    counts the single-process path reports. A chunk without the payload
    (legacy chunks, or chunks whose events never entered a decay chain)
    contributes nothing.
    """
    ordered = sorted(chunks, key=lambda x: int(x.get("chunk_index", 0)))
    attempts: Dict[int, int] = {}
    requested: Dict[int, Dict[Tuple[int, ...], float]] = {}
    realized: Dict[int, Dict[Tuple[int, ...], int]] = {}
    openness: Dict[int, Dict[Tuple[int, ...], float]] = {}
    parent_openness: Dict[int, Dict[Tuple[int, ...], float]] = {}
    parent_openness_cnt: Dict[int, Dict[Tuple[int, ...], int]] = {}

    for chunk in ordered:
        acc = chunk.get("channel_audit_accumulators")
        if not acc:
            continue
        # Distinct loop-variable names per field: mypy tracks function-scoped
        # names across loops, and the count fields (int) and the weight /
        # openness fields (float) must not share one declared type.
        for att_pdg, att_value in (acc.get("channel_attempts") or {}).items():
            key = int(att_pdg)
            attempts[key] = attempts.get(key, 0) + int(att_value)
        for req_pdg, req_inner in (
            acc.get("channel_requested_sum") or {}
        ).items():
            req_tbl = requested.setdefault(int(req_pdg), {})
            for req_key, req_value in req_inner.items():
                req_tbl[req_key] = req_tbl.get(req_key, 0.0) + float(req_value)
        for rea_pdg, rea_inner in (acc.get("channel_realized") or {}).items():
            rea_tbl = realized.setdefault(int(rea_pdg), {})
            for rea_key, rea_value in rea_inner.items():
                rea_tbl[rea_key] = rea_tbl.get(rea_key, 0) + int(rea_value)
        for opn_pdg, opn_inner in (
            acc.get("channel_openness_sum") or {}
        ).items():
            opn_tbl = openness.setdefault(int(opn_pdg), {})
            for opn_key, opn_value in opn_inner.items():
                opn_tbl[opn_key] = opn_tbl.get(opn_key, 0.0) + float(opn_value)
        for pop_pdg, pop_inner in (
            acc.get("channel_parent_openness_sum") or {}
        ).items():
            pop_tbl = parent_openness.setdefault(int(pop_pdg), {})
            for pop_key, pop_value in pop_inner.items():
                pop_tbl[pop_key] = pop_tbl.get(pop_key, 0.0) + float(pop_value)
        for poc_pdg, poc_inner in (
            acc.get("channel_parent_openness_count") or {}
        ).items():
            poc_tbl = parent_openness_cnt.setdefault(int(poc_pdg), {})
            for poc_key, poc_value in poc_inner.items():
                poc_tbl[poc_key] = poc_tbl.get(poc_key, 0) + int(poc_value)

    return {
        "channel_attempts": attempts,
        "channel_requested_sum": requested,
        "channel_realized": realized,
        "channel_openness_sum": openness,
        "channel_parent_openness_sum": parent_openness,
        "channel_parent_openness_count": parent_openness_cnt,
    }


def _derive_channel_audit_tables(
    acc: ChannelAuditAccumulators,
) -> Dict[str, Dict[int, Dict[Tuple[int, ...], float]]]:
    """Derive the per-channel audit tables from summed accumulators.

    Mirrors :meth:`Event.decay_stats` division for each parent PDG:
    ``channel_requested``, ``channel_realized`` and ``openness_fraction``
    divide by the summed channel-selection attempt count of that parent,
    and ``parent_openness_fraction`` divides by the summed per-channel
    selection count (the peak-openness diagnostic is a property of the
    channel and the parent line shape, not a frequency-weighted mean; it
    is absent, not zero, for channels without a supplied line shape).
    With a single chunk the summed accumulators are that chunk's own, so
    the derived tables are bit-identical to the single-process
    ``decay_stats`` output.
    """
    channel_requested_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
    channel_realized_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
    openness_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
    parent_openness_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
    for pdg_id, attempts in acc["channel_attempts"].items():
        denom = float(attempts)
        if denom <= 0.0:
            continue
        req_sum = acc["channel_requested_sum"].get(pdg_id, {})
        rea = acc["channel_realized"].get(pdg_id, {})
        open_sum = acc["channel_openness_sum"].get(pdg_id, {})
        parent_open_sum = acc["channel_parent_openness_sum"].get(pdg_id, {})
        parent_open_cnt = acc["channel_parent_openness_count"].get(pdg_id, {})
        keys = set(req_sum) | set(rea) | set(open_sum) | set(parent_open_sum)
        req_means: Dict[Tuple[int, ...], float] = {}
        rea_means: Dict[Tuple[int, ...], float] = {}
        open_means: Dict[Tuple[int, ...], float] = {}
        parent_open_means: Dict[Tuple[int, ...], float] = {}
        for key in keys:
            req_val = req_sum.get(key)
            if req_val is not None:
                req_means[key] = float(req_val) / denom
            rea_count = rea.get(key)
            if rea_count is not None:
                rea_means[key] = float(rea_count) / denom
            open_val = open_sum.get(key)
            if open_val is not None:
                open_means[key] = float(open_val) / denom
            if key in parent_open_sum:
                n_sel = int(parent_open_cnt.get(key, 0))
                if n_sel > 0:
                    parent_open_means[key] = (
                        float(parent_open_sum[key]) / float(n_sel)
                    )
        channel_requested_out[pdg_id] = req_means
        channel_realized_out[pdg_id] = rea_means
        openness_out[pdg_id] = open_means
        if parent_open_means:
            parent_openness_out[pdg_id] = parent_open_means
    return {
        "channel_requested": channel_requested_out,
        "channel_realized": channel_realized_out,
        "openness_fraction": openness_out,
        "parent_openness_fraction": parent_openness_out,
    }


def _generate_tracks_chunk_worker(
    *,
    config: GenerationConfig,
    chunk_index: int,
    n_events: int,
    chunk_seed: int,
) -> TrackChunkResult:
    if config.nucleus is not None:
        return _generate_pa_tracks_chunk_worker(
            config=config,
            chunk_index=chunk_index,
            n_events=n_events,
            chunk_seed=chunk_seed,
        )
    updates = _effective_tower_updates(config)
    with pdg_database.mass_width_overrides(updates):
        rng = np.random.default_rng(int(chunk_seed))
        reaction = Reaction(
            float(config.beam_momentum),
            beam_pdg=int(config.beam_pdg),
            target_pdg=int(config.target_pdg),
        )
        frag_mode = _normalize_fragmentation_mode(config.fragmentation_mode)

        gen = _build_side_generator(
            reaction, str(config.gen_file), updates, rng,
            side_pdg=int(reaction.beam_pdg()),
        )
        # Mixed-species reactions run one generator per side so each
        # hemisphere carries its own nucleon's charge content: the
        # projectile side uses the reflected primary for a neutron beam,
        # the target side mirrors whichever nucleon sits at rest. The
        # p+p default keeps the single shared generator and its
        # historical RNG consumption order untouched.
        gen_target = None
        if int(reaction.beam_pdg()) != int(reaction.target_pdg()):
            gen_target = _build_side_generator(
                reaction, str(config.gen_file), updates, rng,
                side_pdg=int(reaction.target_pdg()),
            )

        ev = Event(
            rng=rng,
            resolved_match_mass=_resolved_match_mass_override(config),
            parent_line_shapes=_parent_line_shape_map((gen, gen_target)),
        )
        ev._decay_chain = _build_decay_chain(config, rng)
        # Reset decay diagnostics so chunk stats are self-contained.
        ev.reset_decay_stats()

        pdg_vals: List[int] = []
        xf_vals: List[float] = []
        pt_vals: List[float] = []
        source_side_vals: List[int] = []
        weak_feed_vals: List[int] = []
        gen_depth_vals: List[int] = []

        n_terminal_unstable_vals: List[int] = []

        def _decay_and_collect(parent, side_tag: int) -> None:
            if parent is None:
                return
            ev.add_particle(parent)
            ev.mark_primary(parent)
            if int(config.max_decay_passes) <= 1:
                ev.decay()
            else:
                ev.decay_cascade(max_passes=int(config.max_decay_passes))
            # Use direct list reference to avoid per-event list copies.
            n_terminal_unstable_vals.append(
                sum(1 for p in ev.particles() if p.get_id() in ev._decay_chain)
            )
            for p in ev.particles():
                xv = float(p.get_xf())
                pv = float(p.get_pt())
                if not math.isfinite(xv) or not math.isfinite(pv):
                    continue
                pdg_vals.append(int(p.get_id()))
                xf_vals.append(xv)
                pt_vals.append(pv)
                source_side_vals.append(int(side_tag))
                weak_feed_vals.append(int(p.get_weak_feed()))
                gen_depth_vals.append(int(p.get_gen_depth()))
            ev.delete()

        # Sequential generate() preserves historical RNG consumption order used by
        # component caches. Use Generator.generate_n only for new/opt-in callers.
        # Substream manifest recording (Round 1, card 5, additive): the
        # derivation is pure SeedSequence hashing and consumes no draws
        # from ``rng``, so the generation stream is untouched.
        substream_records: Optional[List[Dict[str, Any]]] = (
            [] if config.substream_manifest_path is not None else None
        )
        for event_index_in_chunk in range(int(n_events)):
            if substream_records is not None:
                subs = rng_substreams.derive_substreams(
                    int(config.seed), int(chunk_index), event_index_in_chunk
                )
                substream_records.append(
                    {
                        "event_index": event_index_in_chunk,
                        "streams": dict(subs.streams),
                        "draws": [],
                    }
                )
            if frag_mode in ("projectile", "both"):
                part = gen.generate()
                _decay_and_collect(part, +1)

            if frag_mode in ("target", "both"):
                part_t = (gen_target or gen).generate()
                if part_t is not None:
                    _mirror_projectile_to_target_in_place(part_t, reaction)
                _decay_and_collect(part_t, -1)

        if substream_records is not None:
            rng_substreams.write_substream_manifest(
                rng_substreams.substream_manifest_path(
                    str(config.substream_manifest_path), int(chunk_index)
                ),
                substream_records,
            )

        # Weak-feed output policy: primary_only drops tracks with any weak
        # decay in their cascade ancestry.
        weak_feed_policy = _normalize_weak_feed_policy(config.weak_feed_policy)
        n_weak_dropped = 0
        if weak_feed_policy == "primary_only":
            keep = np.asarray(weak_feed_vals, dtype=np.int8) == 0
            n_weak_dropped = int(len(pdg_vals) - np.count_nonzero(keep))
            pdg_vals = list(np.asarray(pdg_vals, dtype=np.int64)[keep])
            xf_vals = list(np.asarray(xf_vals, dtype=np.float64)[keep])
            pt_vals = list(np.asarray(pt_vals, dtype=np.float64)[keep])
            source_side_vals = list(np.asarray(source_side_vals, dtype=np.int64)[keep])
            weak_feed_vals = list(np.asarray(weak_feed_vals, dtype=np.int8)[keep])
            gen_depth_vals = list(np.asarray(gen_depth_vals, dtype=np.int16)[keep])

        # Realized primary-parent branching table (post kinematic gating is
        # not observable here; these are the loaded cumulative probabilities)
        # plus mass-law metadata for the cache manifest.
        primary_id = int(getattr(gen, "_id", 0))
        realized: Optional[RealizedPrimaryBranch] = None
        dec = ev._decay_chain.get(primary_id) if isinstance(ev._decay_chain, dict) else None
        if dec is not None and dec.n_modes() > 0:
            inc: List[List[Any]] = []
            prev = 0.0
            for prob, mode in zip(dec._probabilities, dec._modes):
                inc.append([list(mode.daughters), float(prob - prev)])
                prev = prob
            realized = {"parent_pdg": primary_id, "modes": inc}
        mass_law: MassLaw
        if getattr(gen, "_mass_type", 0) in (1, 3) and getattr(gen, "_bw", None) is not None:
            mass_law = {"type": "breit_wigner", "params": list(gen._bw.params)}
        elif getattr(gen, "_mass_type", 0) == 2 and getattr(gen, "_mass_hist", None) is not None:
            mass_law = {"type": "tabulated_pdf"}
        else:
            mass_law = {"type": "fixed_pdg", "mass_gev": float(gen._id_mass)}

        decay_stats = ev.decay_stats()
        n_primary_generated = int(decay_stats["n_primary_generated"])
        n_primary_decayed = int(decay_stats["n_primary_decayed"])
        out: TrackChunkResult = {
            "chunk_index": int(chunk_index),
            "n_events": int(n_events),
            "seed": int(chunk_seed),
            "pdg": np.asarray(pdg_vals, dtype=np.int32),
            "xf": np.asarray(xf_vals, dtype=np.float32),
            "pt": np.asarray(pt_vals, dtype=np.float32),
            "source_side": np.asarray(source_side_vals, dtype=np.int8),
            "weak_feed": np.asarray(weak_feed_vals, dtype=np.int8),
            "gen_depth": np.asarray(gen_depth_vals, dtype=np.int16),
            "n_decayed_parents": int(decay_stats.get("n_decayed_parents", 0)),
            "n_decay_kinematic_failures": int(
                decay_stats.get("n_decay_kinematic_failures", 0)
            ),
            "n_decay_attempts": int(decay_stats.get("n_decay_attempts", 0)),
            "primary_parent": realized,
            "mass_law": mass_law,
            "weak_feed_policy": weak_feed_policy,
            "n_weak_dropped": int(n_weak_dropped),
            "n_primary_generated": n_primary_generated,
            "n_primary_decayed": n_primary_decayed,
            "primary_failed": max(n_primary_generated - n_primary_decayed, 0),
            "primary_pdg_stats": dict(decay_stats["primary_pdg_stats"]),
            "n_terminal_unstable": int(sum(n_terminal_unstable_vals)),
            "channel_audit_accumulators": _collect_channel_audit_accumulators(ev),
        }
        return out


def _generate_pa_tracks_chunk_worker(
    *,
    config: GenerationConfig,
    chunk_index: int,
    n_events: int,
    chunk_seed: int,
) -> TrackChunkResult:
    """Chunked p+A generation under the wounded-nucleon superposition.

    Each event is one p+A collision: a single projectile-side emission
    plus one target-side emission per wounded target nucleon ``K``.
    ``n_events`` counts p+A events, so track counts scale with
    ``<K> + 1`` relative to the pp worker at equal event counts. A
    requested centrality slice is sampled directly from the
    interaction-weighted distribution restricted to the slice
    (``WoundedNucleonSampler.sample_event_in_range``), so every event
    lands inside ``target_wound_range`` without an acceptance loop.
    For a point target (``A == 1``) every event has ``K = 1`` and the
    sampler consumes no random numbers, which makes the RNG consumption
    order identical to the pp ``fragmentation_mode='both'`` path;
    same-seed runs therefore produce identical tracks (pinned by
    regression test).
    """
    assert config.nucleus is not None
    nucleus = config.nucleus
    updates = _effective_tower_updates(config)
    with pdg_database.mass_width_overrides(updates):
        rng = np.random.default_rng(int(chunk_seed))
        reaction = Reaction(
            float(config.beam_momentum),
            beam_pdg=int(config.beam_pdg),
            target_pdg=int(config.target_pdg),
        )

        gen = _build_side_generator(
            reaction, str(config.gen_file), updates, rng,
            side_pdg=int(reaction.beam_pdg()),
        )
        gen_target = None
        if int(reaction.beam_pdg()) != int(reaction.target_pdg()):
            gen_target = _build_side_generator(
                reaction, str(config.gen_file), updates, rng,
                side_pdg=int(reaction.target_pdg()),
            )

        ev = Event(
            rng=rng,
            resolved_match_mass=_resolved_match_mass_override(config),
            parent_line_shapes=_parent_line_shape_map((gen, gen_target)),
        )
        ev._decay_chain = _build_decay_chain(config, rng)
        ev.reset_decay_stats()

        sampler: Optional[WoundedNucleonSampler] = (
            WoundedNucleonSampler(nucleus, rng=rng)
            if int(nucleus.A) > 1
            else None
        )
        wound_range = config.target_wound_range

        pdg_vals: List[int] = []
        xf_vals: List[float] = []
        pt_vals: List[float] = []
        source_side_vals: List[int] = []
        weak_feed_vals: List[int] = []
        gen_depth_vals: List[int] = []
        n_wounds_vals: List[int] = []

        n_terminal_unstable_vals: List[int] = []

        def _decay_and_collect(parent, side_tag: int, wounds: int) -> None:
            if parent is None:
                return
            ev.add_particle(parent)
            ev.mark_primary(parent)
            if int(config.max_decay_passes) <= 1:
                ev.decay()
            else:
                ev.decay_cascade(max_passes=int(config.max_decay_passes))
            n_terminal_unstable_vals.append(
                sum(1 for p in ev.particles() if p.get_id() in ev._decay_chain)
            )
            for p in ev.particles():
                xv = float(p.get_xf())
                pv = float(p.get_pt())
                if not math.isfinite(xv) or not math.isfinite(pv):
                    continue
                pdg_vals.append(int(p.get_id()))
                xf_vals.append(xv)
                pt_vals.append(pv)
                source_side_vals.append(int(side_tag))
                weak_feed_vals.append(int(p.get_weak_feed()))
                gen_depth_vals.append(int(p.get_gen_depth()))
                n_wounds_vals.append(int(wounds))
            ev.delete()

        # Substream manifest recording (Round 1, card 5, additive): the
        # derivation is pure SeedSequence hashing and consumes no draws
        # from ``rng``, so the generation stream is untouched.
        substream_records: Optional[List[Dict[str, Any]]] = (
            [] if config.substream_manifest_path is not None else None
        )
        for event_index_in_chunk in range(int(n_events)):
            if substream_records is not None:
                subs = rng_substreams.derive_substreams(
                    int(config.seed), int(chunk_index), event_index_in_chunk
                )
                substream_records.append(
                    {
                        "event_index": event_index_in_chunk,
                        "streams": dict(subs.streams),
                        "draws": [],
                    }
                )
            if sampler is None:
                wounds = 1
            elif wound_range is not None:
                # Range-conditioned draw: b and K come from the
                # interaction-weighted measure restricted to the slice,
                # so every event lands inside [lo, hi] by construction
                # and no acceptance-rejection loop runs.
                lo, hi = wound_range
                _, wounds = sampler.sample_event_in_range(lo, hi)
            else:
                _, wounds = sampler.sample_event()

            part = gen.generate()
            _decay_and_collect(part, +1, wounds)

            for _w in range(wounds):
                part_t = (gen_target or gen).generate()
                if part_t is not None:
                    _mirror_projectile_to_target_in_place(part_t, reaction)
                _decay_and_collect(part_t, -1, wounds)

        if substream_records is not None:
            rng_substreams.write_substream_manifest(
                rng_substreams.substream_manifest_path(
                    str(config.substream_manifest_path), int(chunk_index)
                ),
                substream_records,
            )

        weak_feed_policy = _normalize_weak_feed_policy(config.weak_feed_policy)
        n_weak_dropped = 0
        if weak_feed_policy == "primary_only":
            keep = np.asarray(weak_feed_vals, dtype=np.int8) == 0
            n_weak_dropped = int(len(pdg_vals) - np.count_nonzero(keep))
            pdg_vals = list(np.asarray(pdg_vals, dtype=np.int64)[keep])
            xf_vals = list(np.asarray(xf_vals, dtype=np.float64)[keep])
            pt_vals = list(np.asarray(pt_vals, dtype=np.float64)[keep])
            source_side_vals = list(np.asarray(source_side_vals, dtype=np.int64)[keep])
            weak_feed_vals = list(np.asarray(weak_feed_vals, dtype=np.int8)[keep])
            gen_depth_vals = list(np.asarray(gen_depth_vals, dtype=np.int16)[keep])
            n_wounds_vals = list(np.asarray(n_wounds_vals, dtype=np.int16)[keep])

        primary_id = int(getattr(gen, "_id", 0))
        realized: Optional[RealizedPrimaryBranch] = None
        dec = ev._decay_chain.get(primary_id) if isinstance(ev._decay_chain, dict) else None
        if dec is not None and dec.n_modes() > 0:
            inc: List[List[Any]] = []
            prev = 0.0
            for prob, mode in zip(dec._probabilities, dec._modes):
                inc.append([list(mode.daughters), float(prob - prev)])
                prev = prob
            realized = {"parent_pdg": primary_id, "modes": inc}
        mass_law: MassLaw
        if getattr(gen, "_mass_type", 0) in (1, 3) and getattr(gen, "_bw", None) is not None:
            mass_law = {"type": "breit_wigner", "params": list(gen._bw.params)}
        elif getattr(gen, "_mass_type", 0) == 2 and getattr(gen, "_mass_hist", None) is not None:
            mass_law = {"type": "tabulated_pdf"}
        else:
            mass_law = {"type": "fixed_pdg", "mass_gev": float(gen._id_mass)}

        decay_stats = ev.decay_stats()
        n_primary_generated = int(decay_stats["n_primary_generated"])
        n_primary_decayed = int(decay_stats["n_primary_decayed"])
        out: TrackChunkResult = {
            "chunk_index": int(chunk_index),
            "n_events": int(n_events),
            "seed": int(chunk_seed),
            "pdg": np.asarray(pdg_vals, dtype=np.int32),
            "xf": np.asarray(xf_vals, dtype=np.float32),
            "pt": np.asarray(pt_vals, dtype=np.float32),
            "source_side": np.asarray(source_side_vals, dtype=np.int8),
            "weak_feed": np.asarray(weak_feed_vals, dtype=np.int8),
            "gen_depth": np.asarray(gen_depth_vals, dtype=np.int16),
            "n_wounds": np.asarray(n_wounds_vals, dtype=np.int16),
            "pa_nucleus": {
                "name": str(nucleus.name),
                "A": int(nucleus.A),
                "Z": int(nucleus.Z),
                "profile": str(nucleus.resolved_profile()),
                "sigma_nn_inel_mb": float(nucleus.sigma_nn_inel_mb),
                "target_wound_range": (
                    [int(wound_range[0]), int(wound_range[1])]
                    if wound_range is not None
                    else None
                ),
            },
            "n_decayed_parents": int(decay_stats.get("n_decayed_parents", 0)),
            "n_decay_kinematic_failures": int(
                decay_stats.get("n_decay_kinematic_failures", 0)
            ),
            "n_decay_attempts": int(decay_stats.get("n_decay_attempts", 0)),
            "primary_parent": realized,
            "mass_law": mass_law,
            "weak_feed_policy": weak_feed_policy,
            "n_weak_dropped": int(n_weak_dropped),
            "n_primary_generated": n_primary_generated,
            "n_primary_decayed": n_primary_decayed,
            "primary_failed": max(n_primary_generated - n_primary_decayed, 0),
            "primary_pdg_stats": dict(decay_stats["primary_pdg_stats"]),
            "n_terminal_unstable": int(sum(n_terminal_unstable_vals)),
            "channel_audit_accumulators": _collect_channel_audit_accumulators(ev),
        }
        return out


# Per-track chunk arrays. The required fields must be present and equally
# long within every chunk; ``weak_feed``, ``gen_depth`` and ``n_wounds``
# are optional and are synthesized as zero-filled arrays of the declared
# dtype when a legacy chunk omits them, so merged metadata always aligns
# with the merged tracks. ``n_wounds`` carries the per-event number of
# wounded target nucleons of p+A events (0 for p+p chunks).
_TRACK_ARRAY_DTYPES: Dict[str, np.dtype] = {
    "pdg": np.dtype(np.int32),
    "xf": np.dtype(np.float32),
    "pt": np.dtype(np.float32),
    "source_side": np.dtype(np.int8),
    "weak_feed": np.dtype(np.int8),
    "gen_depth": np.dtype(np.int16),
    "n_wounds": np.dtype(np.int16),
}
_REQUIRED_TRACK_KEYS = ("pdg", "xf", "pt", "source_side")
_OPTIONAL_TRACK_KEYS = ("weak_feed", "gen_depth", "n_wounds")


def merge_track_chunks(chunks: Sequence[TrackChunkResult]) -> Dict[str, np.ndarray]:
    """Merge chunk outputs in ``chunk_index`` order.

    All returned per-track arrays share one length (the total number of
    tracks). Chunks may omit the optional ``weak_feed``/``gen_depth``
    fields; a missing field is synthesized as zeros matching that chunk's
    track count, which keeps legacy cache chunks mergeable. A chunk whose
    own per-track arrays disagree in length, or that lacks a required
    field, raises :class:`ValueError` naming the chunk and the offending
    array instead of returning silently misaligned metadata.
    """

    ordered = sorted(chunks, key=lambda x: int(x["chunk_index"]))
    if not ordered:
        return {
            key: np.empty(0, dtype=dtype)
            for key, dtype in _TRACK_ARRAY_DTYPES.items()
        }

    merged_lists: Dict[str, List[np.ndarray]] = {
        key: [] for key in _TRACK_ARRAY_DTYPES
    }

    for chunk in ordered:
        ci = chunk.get("chunk_index")
        raw_pdg = chunk.get("pdg")
        if raw_pdg is None:
            raise ValueError(
                f"merge_track_chunks: incomplete chunk schema at "
                f"chunk_index={ci}: missing required per-track array 'pdg'"
            )
        pdg = np.asarray(raw_pdg, dtype=_TRACK_ARRAY_DTYPES["pdg"])
        n_tracks = int(pdg.shape[0])
        arrays: Dict[str, np.ndarray] = {"pdg": pdg}

        for key in _REQUIRED_TRACK_KEYS[1:]:
            raw = chunk.get(key)
            if raw is None:
                raise ValueError(
                    f"merge_track_chunks: incomplete chunk schema at "
                    f"chunk_index={ci}: missing required per-track array "
                    f"'{key}'"
                )
            arr = np.asarray(raw, dtype=_TRACK_ARRAY_DTYPES[key])
            if arr.shape[0] != n_tracks:
                raise ValueError(
                    f"merge_track_chunks: inconsistent chunk schema at "
                    f"chunk_index={ci}: array '{key}' has length "
                    f"{arr.shape[0]} while 'pdg' has length {n_tracks}; "
                    f"all per-track arrays within a chunk must share one "
                    f"length"
                )
            arrays[key] = arr

        for key in _OPTIONAL_TRACK_KEYS:
            raw = chunk.get(key)
            if raw is None:
                # Legacy chunk without this optional field: synthesize
                # defaults so the merged metadata stays aligned.
                arrays[key] = np.zeros(n_tracks, dtype=_TRACK_ARRAY_DTYPES[key])
                continue
            arr = np.asarray(raw, dtype=_TRACK_ARRAY_DTYPES[key])
            if arr.shape[0] != n_tracks:
                raise ValueError(
                    f"merge_track_chunks: inconsistent chunk schema at "
                    f"chunk_index={ci}: array '{key}' has length "
                    f"{arr.shape[0]} while 'pdg' has length {n_tracks}; "
                    f"all per-track arrays within a chunk must share one "
                    f"length"
                )
            arrays[key] = arr

        for key, arr in arrays.items():
            merged_lists[key].append(arr)

    return {
        key: np.concatenate(arrs) if arrs else np.empty(0, dtype=_TRACK_ARRAY_DTYPES[key])
        for key, arrs in merged_lists.items()
    }


def generate_tracks_chunked(
    config: GenerationConfig,
    *,
    n_workers: int = 1,
    n_chunks: int | None = None,
) -> MergedGenerationResult:
    """Generate final-state tracks using chunked event generation.

    Parameters
    ----------
    config:
        Resonance generation config for one component.
    n_workers:
        Number of worker processes.
    n_chunks:
        Number of chunks to split ``config.n_events`` into. If omitted, uses
        ``n_workers``.
    """

    workers = max(int(n_workers), 1)
    chunks = max(int(n_chunks) if n_chunks is not None else workers, 1)
    counts = split_event_counts(config.n_events, chunks)

    # (chunk_index, n_events, chunk_seed) per non-empty chunk.
    tasks: List[Tuple[int, int, int]] = [
        (i, c, _chunk_seed(config.seed, i))
        for i, c in enumerate(counts)
        if c > 0
    ]

    results: List[TrackChunkResult] = []

    if workers == 1 or len(tasks) <= 1:
        for chunk_index, n_events, chunk_seed in tasks:
            results.append(
                _generate_tracks_chunk_worker(
                    config=config,
                    chunk_index=chunk_index,
                    n_events=n_events,
                    chunk_seed=chunk_seed,
                )
            )
    else:
        max_workers = min(workers, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _generate_tracks_chunk_worker,
                    config=config,
                    chunk_index=chunk_index,
                    n_events=n_events,
                    chunk_seed=chunk_seed,
                )
                for chunk_index, n_events, chunk_seed in tasks
            ]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

    merged = merge_track_chunks(results)
    ordered = sorted(results, key=lambda x: int(x["chunk_index"]))
    chunk_meta: List[ChunkSummary] = [
        {
            "chunk_index": int(r["chunk_index"]),
            "n_events": int(r["n_events"]),
            "seed": int(r["seed"]),
            "tracks": int(np.asarray(r["pdg"]).shape[0]),
            "n_decayed_parents": int(r.get("n_decayed_parents", 0)),
            "n_decay_kinematic_failures": int(r.get("n_decay_kinematic_failures", 0)),
            "n_decay_attempts": int(r.get("n_decay_attempts", 0)),
        }
        for r in ordered
    ]
    n_decayed = sum(int(c["n_decayed_parents"]) for c in chunk_meta)
    n_fail = sum(int(c["n_decay_kinematic_failures"]) for c in chunk_meta)
    n_attempts = sum(int(c["n_decay_attempts"]) for c in chunk_meta)

    n_weak_dropped = sum(int(r.get("n_weak_dropped", 0)) for r in ordered)
    n_primary_generated = sum(int(r.get("n_primary_generated", 0)) for r in ordered)
    n_primary_decayed = sum(int(r.get("n_primary_decayed", 0)) for r in ordered)
    primary_failed = sum(int(r.get("primary_failed", 0)) for r in ordered)
    primary_pdg_stats: Dict[int, List[int]] = {}
    for r in ordered:
        stats = r.get("primary_pdg_stats") or {}
        for pdg_id, pair in stats.items():
            reg, dec = pair
            entry = primary_pdg_stats.setdefault(int(pdg_id), [0, 0])
            entry[0] += int(reg)
            entry[1] += int(dec)

    # Per-channel audit tables: sum the raw per-chunk accumulators and
    # re-derive the tables from the summed counts with the same divisions
    # the single-process ``Event.decay_stats`` path uses (never an average
    # of per-chunk fractions).
    channel_audit = _derive_channel_audit_tables(
        _merge_channel_audit_accumulators(ordered)
    )

    return {
        "pdg": merged["pdg"],
        "xf": merged["xf"],
        "pt": merged["pt"],
        "source_side": merged["source_side"],
        "weak_feed": merged["weak_feed"],
        "gen_depth": merged["gen_depth"],
        "n_wounds": merged["n_wounds"],
        "pa_nucleus": next(
            (c.get("pa_nucleus") for c in ordered if c.get("pa_nucleus")),
            None,
        ),
        "n_weak_feed_tracks": int(np.count_nonzero(merged["weak_feed"])) if merged["weak_feed"].size else 0,
        "n_terminal_unstable": int(sum(int(c.get("n_terminal_unstable", 0)) for c in ordered)),
        "primary_parent": next((c.get("primary_parent") for c in ordered if c.get("primary_parent")), None),
        "mass_law": next((c.get("mass_law") for c in ordered if c.get("mass_law")), None),
        "chunks": chunk_meta,
        "n_events": int(sum(int(c["n_events"]) for c in chunk_meta)),
        "n_decayed_parents": int(n_decayed),
        "n_decay_kinematic_failures": int(n_fail),
        "n_decay_attempts": int(n_attempts),
        "weak_feed_policy": _normalize_weak_feed_policy(config.weak_feed_policy),
        "n_weak_dropped": int(n_weak_dropped),
        "n_primary_generated": int(n_primary_generated),
        "n_primary_decayed": int(n_primary_decayed),
        "primary_failed": int(primary_failed),
        "primary_pdg_stats": {
            pdg_id: [int(pair[0]), int(pair[1])]
            for pdg_id, pair in primary_pdg_stats.items()
        },
        "channel_requested": channel_audit["channel_requested"],
        "channel_realized": channel_audit["channel_realized"],
        "openness_fraction": channel_audit["openness_fraction"],
        "parent_openness_fraction": channel_audit["parent_openness_fraction"],
    }
