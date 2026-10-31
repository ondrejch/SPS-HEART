"""Round 3 card 9 phase 1: builder blend contract pins (test-author
register).

Task ``TASK-20260918-r3-run-r32-spectral-systematic-pp`` (plan section 9
run R32, line 644; phase 1, the hermetic pre-run phase).  This is the
test-author's independent pin register for the card-6 blend routing
added to the card-13 ``ResonancePPBuilder`` (nested commit ``b9df812``,
outer ``3713e34``) and for the dual-tree driver's seed/statistics/
spread pre-registration contract.  It is an independent cross-check of
the solver's register in
``tests/test_staged_r32_spectral_systematic.py`` and
``tests/test_staged_r32_spectral_systematic_pins.py``: the same
contract, pinned from an independently authored test with its own
fixture seeds, probe masses, and event streams (no solver constants
reused, no solver digests re-derived).

Part 1 (builder blend contract, pure engine, no sealed run dir, no
Monte Carlo beyond the pinned streams below):

- ``test_variant_constants_rederived``: the three variant ramp starts
  re-derived from the card's decimals (``B = 2.680194110727528``,
  ``B - 0.2 GeV = 2.480194110727528``, ``B + 0.2 GeV =
  2.880194110727528``; the exact-decimal arithmetic verified in the
  ``decimal`` module, the driver's v1 literal pinned as the correctly
  rounded double of the card's decimal, one ulp apart from the naive
  float ``B - 0.2``, while ``B + 0.2`` agrees bit-for-bit), the fixed
  ramp width ``W = 0.5``, the builder's ``KERNEL_BOUNDARY_M8``, the
  card-6 ``BLEND_BOUNDARY``, and the M8-frozen
  ``continuum.match_mass`` of the legacy_v1 manifest all equal; the
  driver's ``variant_blend`` ramp shape (``w(B) = 0``, ``w(B + W) =
  1``, linear inside).
- ``test_blend_none_pinned_stream_digest``: a 240-event stream at the
  test-author's own seed (20270421, master 20260913, replica 0) with
  ``blend = None`` reproduces the pinned per-event hash list and its
  pinned sha256 (the byte-stability regression anchor).
- ``test_blend_none_pre_extension_bit_identity``: the same 240-event
  stream is bit-identical (full ``as_dict`` content, event by event)
  between the post-extension builder at ``blend = None`` and the
  pre-extension R23 builder fetched from the nested repository at the
  pinned parent commit (blob digest verified); skips with a recorded
  reason when ``git`` is unavailable on this checkout (the pinned
  digest above then remains the byte-stability anchor).
- ``test_routing_below_boundary_zero_uniforms``: for every variant and
  three probe masses below the variant ramp start (all below the R23
  kernel boundary, so the no-blend builder takes the same catalog
  path), the blended builder consumes exactly one ``decay_channel``
  child uniform (the record draw; zero routing uniforms), the shared
  event generator is untouched, and the record contents and the
  ``decay_channel`` child state are bit-identical to the no-blend
  builder at the same seed.
- ``test_routing_band_exactly_one_routing_uniform``: eight probe
  masses per variant across the open transition interval, including
  the exact boundary ``M = B_v`` (``w = 0``: the continuum route is
  never taken there).  At every band mass the spawned ``decay_channel``
  child consumes exactly two uniforms, the first one is the routing
  uniform taken before any record draw, and the route is
  ``'continuum'`` exactly when that draw is below ``w(M) =
  (M - B_v) / W``; the record stamps ``route`` and the pinned
  ``blend_weight``.
- ``test_routing_above_ramp_end_zero_uniforms``: at
  ``M >= B_v + W`` the continuum route is taken with zero ramp
  uniforms: the ``decay_channel`` child consumes exactly one uniform
  (the class draw, the routing uniform is absent), and the drawn
  class equals the uniform over the family's eight
  ``allowed_decay_classes`` re-derived independently.
- ``test_boundary_matching_factor_never_consulted``: an independent
  spy on the card-6 blend-normalization surface (``blended_density``,
  ``blended_normalization_report``, ``_blended_pieces``,
  ``_blended_support``) records zero consultations during a
  220-event blended stream (several hundred continuum-routed
  sources in it), and every continuum-routed source conserves its
  ``(B, Q, S) = (1, 1, 0)``.
- ``test_continuum_route_record_structure``: the continuum-route
  record and root split (six probes at the test-author's own seed,
  source at rest so the boost is the identity): the record's
  ``kernel = 'continuum'`` / ``route = 'continuum'`` /
  ``family_id = 'baryonic_np'`` stamps, the root vertex's
  ``parent_pdg = 0`` sentinel (the R31 convention) with the source
  four-momentum and the drawn class, the Kaellen decay momentum and
  exact energy split of the root daughters, the root-split direction
  reconstructed from the first two uniforms of the spawned
  ``decay_kinematics`` child, the ``channel_family`` split
  (``continuum_decay`` root products vs ``resonance_decay`` cascade
  products), and per-source ``(B, Q, S)`` conservation.
- ``test_blend_argument_validation``: the builder refuses
  ``families`` without a ``BlendSpec`` (``TypeError``), a
  non-``BlendSpec`` blend (``TypeError``), a ``BlendSpec`` without
  ``families`` (``TypeError``), an empty family list (``ValueError``),
  non-``ContinuumFamily`` family items (``TypeError``), and two
  activated families for the source quantum numbers (``ValueError``).

Part 2 (driver and seed contract, no sealed run dir):

- ``test_driver_dual_tree_byte_identity``: the nested and outer driver
  copies are byte-identical at the pinned digest (post
  REV-b9df812-01).
- ``test_smoke_chunk_kernel_substream_smoke_block``: the smoke
  generation path derives its kernel substreams from the
  pre-registered smoke block ``5000 + 100*v + r`` (the
  REV-b9df812-01 contract): a construction monitor on the driver's
  chunk worker records the builder ``replica_index`` as 5000 for
  smoke v0 r0 and 5200 for smoke v2 r0 (never the production-style
  ``100*v`` indices 0/200), and the chunk's event hashes cross-check
  against direct builds at the smoke block (equal) and at the
  production indices (differ) under master 20260919.
- ``test_seed_map_contract``: the seed map re-derived independently
  from the card's rules (master 20260919; production raw seeds
  ``20301519 + 100*v + r`` for ``r < 8``; the +900 confirmation blocks
  ``20301819-20301834 / 20301919-20301934 / 20302019-20302034`` over
  16 replicas; the +5000 smoke block for variants 0 and 2): 74
  distinct raw seeds, the blocks pairwise disjoint and disjoint from
  the R23 raw seeds (20300913-20300920), the retired card 9's planned
  seeds (20301217-20301424), and the prior masters
  (20260913/20260917/20260918); the kernel replica-index spaces
  (``{0..7, 100..107, 200..207}`` vs ``{900..915, 1000..1015,
  1100..1115}`` vs ``{5000, 5200}``) pairwise disjoint; the driver's
  seed-map helpers agree with the re-derivation for every
  (variant, replica) pair.
- ``test_statistics_contract``: the T2 statistics (8 replicas x
  100,000 events per variant, 3 variants, 2,400,000 events total)
  from the driver constants and from the pre-registration schema's
  statistics block, checked against the card text.
- ``test_spread_bound_contract``: both pre-registered spread bounds
  (2.0 / 3.0) present in the driver constants and in the
  pre-registration schema, the bin bound in units of the pooled MC
  standard error ``sqrt(SE_0^2 + SE_1^2 + SE_2^2)`` with the 95th
  percentile aggregate and the deterministic-SE floor 1e-12; the
  card's literal "pooled MC standard error of the median variant"
  reading is recorded as informational (the median variant per bin is
  recorded as the reference beside the pooled unit).
- ``test_named_residual_and_robust_statistics_schema``: the
  named-residual precedent (the register shape of the sealed
  ``gate3_r31.json``) and the robust-statistics cross-reference
  (triggered only by finite-MC-scatter diagnostics; 16 x 100,000 x 3
  = 4,800,000 confirmation events at the +900 blocks; the
  confirmation run dir name) are present in the pre-registration
  schema.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules; no sealed run dir is read, no Monte Carlo beyond the pinned
streams, no writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import types
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import spectral_density as _spectral_density
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.source_event import (
    FourVector,
    QuantumNumbers,
    SourceRecord,
)
from na49_mixing.spectral_density import (
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    BlendSpec,
    ContinuumFamily,
    families_from_json,
)
from na49_mixing.resonance_pp_builder import (
    CONTINUUM_FAMILY,
    CONTINUUM_ROOT_SENTINEL,
)

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent

DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r32_spectral_systematic.py"
)
OUTER_DRIVER_PATH = MONO_ROOT / "helpers" / "benchmarking" / (
    "staged_r32_spectral_systematic.py"
)
BUILDER_PATH = NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "resonance_catalog.json"
)
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "continuum_resonances.json"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "legacy_v1_manifest.json"
)
R11_RANKED_MAP = MONO_ROOT / "dev" / "00runs" / (
    "r11_observable_decomposition_2026-09-11"
) / "ranked_residual_map.json"

# ---------------------------------------------------------------------------
# The pinned digests (the committed phase-1 substrate).
# ---------------------------------------------------------------------------
#: The driver's sha256 after the REV-b9df812-01 fix (nested ``7cc3337``,
#: outer ``c4a657c``): the smoke chunk worker now derives its kernel
#: substreams from the pre-registered smoke block ``5000 + 100*v + r``
#: (the stamped record field) instead of the production ``100*v + r``
#: indices that collided with production v0 r0 under master 20260919.
#: The pre-fix phase-1 digest was
#: ``84e4b4635b988309a3003fe6816bfbb7bc58d01ad75dc7b48530961d5aec27cf``,
#: superseded by that fix.
DRIVER_SHA256 = (
    "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
)
BUILDER_POST_EXTENSION_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)
BUILDER_PRE_EXTENSION_SHA256 = (
    "bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036"
)
#: The nested parent of the phase-1 commit ``b9df812`` (the
#: byte-unchanged R23 builder, fetched for the pre-extension
#: bit-identity check).
PRE_EXTENSION_COMMIT = "ff8c39709490464125f4f2624405c7d9924254f3"

# ---------------------------------------------------------------------------
# The card's variant constants, re-derived from the card's decimals.
# ---------------------------------------------------------------------------
B = 2.680194110727528
V1 = 2.480194110727528  # the card's decimal, its correctly rounded double
V2 = 2.880194110727528
RAMP_WIDTH = 0.5

# ---------------------------------------------------------------------------
# The test-author's pinned streams (independent of the solver's).
# ---------------------------------------------------------------------------
TA_STREAM_SEED = 20270421
TA_STREAM_EVENTS = 240
TA_STREAM_MASTER = 20260913
TA_STREAM_REPLICA = 0
TA_STREAM_STAGE = "r32_test_author_stream"
TA_STREAM_FIRST_HASH = (
    "1dd1dfdd02c28f01e5f029f53dd16259253c17431ef643a41e925b4a2c6c299d"
)
TA_STREAM_LAST_HASH = (
    "ce112af9012c72e025e7bf830cff38e0c822eabc5bc62c854e25fc5f540da818"
)
TA_STREAM_DIGEST = (
    "6464938b10f01a01fda3294f6d8e491e745ea9d935c1c8a5c5a916e5c4c8abfb"
)
TA_SPY_SEED = 20270422
TA_SPY_EVENTS = 220
TA_SPY_REPLICA = 3
TA_SPY_STAGE = "r32_test_author_spy_stream"
TA_PROBE_SEED = 20270423
TA_STRUCTURE_SEED = 20270424
TA_MASTER = 20260913
TA_REPLICA = 0

# ---------------------------------------------------------------------------
# The card's seed map, re-derived (master 20260919, the card's blocks).
# ---------------------------------------------------------------------------
TA_MASTER_SEED = 20260919
TA_SEED_BASE = TA_MASTER_SEED + 40_000
TA_PRODUCTION_BLOCKS = [
    [TA_SEED_BASE + 600 + 100 * v + r for r in range(8)] for v in range(3)
]
TA_CONFIRMATION_REPLICAS = 16
TA_CONFIRMATION_BLOCKS = [
    [TA_SEED_BASE + 900 + 100 * v + r
     for r in range(TA_CONFIRMATION_REPLICAS)]
    for v in range(3)
]
TA_SMOKE_VARIANTS = (0, 2)
TA_SMOKE_SEEDS = [TA_SEED_BASE + 5000 + 100 * v for v in TA_SMOKE_VARIANTS]
TA_R23_RAW_SEEDS = list(range(20300913, 20300921))
TA_RETIREDCARD9_SEEDS = list(range(20301217, 20301425))
TA_PRIOR_MASTERS = (20260913, 20260917, 20260918)
TA_PRODUCTION_KERNEL_INDICES = {100 * v + r for v in range(3) for r in range(8)}
TA_CONFIRMATION_KERNEL_INDICES = {
    100 * v + 900 + r
    for v in range(3) for r in range(TA_CONFIRMATION_REPLICAS)
}
TA_SMOKE_KERNEL_INDICES = {5000 + 100 * v for v in TA_SMOKE_VARIANTS}

# The card's statistics and spread contract (the card text, verbatim
# values).
CARD_T2_REPLICAS = 8
CARD_T2_EVENTS_PER_REPLICA = 100_000
CARD_N_VARIANTS = 3
CARD_TOTAL_EVENTS = 2_400_000
CARD_BIN_SPREAD_BOUND = 2.0
CARD_FEEDDOWN_SPREAD_BOUND = 3.0
CARD_SE_FLOOR = 1e-12
CARD_CONFIRMATION_RUN_DIR = (
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    """The committed dual-tree driver, loaded once under a unique
    module name (the driver's own ``_locate_roots`` wiring resolves
    the nested roots at import time)."""
    name = "staged_r32_spectral_systematic_test_author"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _load_families() -> list:
    return list(families_from_json(str(FAMILIES_PATH)))


def _load_config(drv):
    return drv._m8_config()


def _make_builder(drv, config, catalog, seed, event_id, variant=None,
                  master=TA_MASTER, replica=TA_REPLICA, families=None):
    """One ``ResonancePPBuilder`` at the test-author's fixture; a
    ``variant`` indexes the driver's pre-registered variant blend."""
    kwargs: dict = dict(
        seed=seed, event_id=event_id, master_seed=master, replica_index=replica,
    )
    if variant is not None:
        kwargs["blend"] = drv.variant_blend(variant)
        kwargs["families"] = families if families is not None else _load_families()
    return drv.ResonancePPBuilder(config, catalog, **kwargs)


def _hand_source(mass: float, event_id: int) -> SourceRecord:
    """One hand-realized source at rest at *mass* (the source layer is
    bypassed; the routing and decay contract is pinned at the frozen
    source mass)."""
    return SourceRecord(
        event_id,
        "projectile",
        FourVector(mass, 0.0, 0.0, 0.0),
        QuantumNumbers(1, 1, 0),
        collision_index=1,
        metadata={"origin": "test_author_probe"},
    )


class _CountingRNG:
    """A transparent wrapper that records every ``random()`` draw on
    the wrapped spawned child (the routing-uniform accounting probe)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def random(self):
        value = self._inner.random()
        self.calls.append(value)
        return value

    @property
    def bit_generator(self):
        return self._inner.bit_generator


def _probe_decay_channel_child(master, replica, event_id):
    """An independently re-derived spawned ``decay_channel`` child
    (the builder's own spawn chain, re-implemented here)."""
    subs = derive_substreams(master, replica, event_id)
    return np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
    )


def _probe_decay_kinematics_child(master, replica, event_id):
    """An independently re-derived spawned ``decay_kinematics``
    child (the root split's direction draw)."""
    subs = derive_substreams(master, replica, event_id)
    return np.random.default_rng(
        np.random.SeedSequence(
            subs.stream_for("decay_kinematics")
        ).spawn(1)[0]
    )


def _source_qn(finals) -> tuple:
    return tuple(
        int(sum(getattr(f.quantum_numbers, key) for f in finals))
        for key in ("B", "Q", "S")
    )


def _load_preextension_builder():
    """The pre-extension R23 builder fetched from the nested
    repository at the pinned parent commit, executed under a unique
    module name.  ``None`` when ``git`` is unavailable (fresh
    checkouts without the history)."""
    name = "na49_mixing._test_author_preextension_pp_builder"
    if name in sys.modules:
        return sys.modules[name]
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_EXTENSION_COMMIT}:na49_mixing/"
             "resonance_pp_builder.py"],
            capture_output=True, text=True, cwd=str(NESTED_ROOT),
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    blob = source.encode("utf-8")
    if hashlib.sha256(blob).hexdigest() != BUILDER_PRE_EXTENSION_SHA256:
        raise AssertionError(
            "the pre-extension builder blob at "
            f"{PRE_EXTENSION_COMMIT} no longer matches the pinned "
            f"digest {BUILDER_PRE_EXTENSION_SHA256}"
        )
    module = types.ModuleType(name)
    module.__package__ = "na49_mixing"
    module.__file__ = str(BUILDER_PATH)
    # Registered before the exec: Python 3.12's dataclass machinery
    # resolves ``sys.modules[cls.__module__]`` during class creation.
    sys.modules[name] = module
    exec(compile(source, str(BUILDER_PATH), "exec"), module.__dict__)
    return module


def _build_pre_registration(drv, catalog, families) -> dict:
    """The driver's pre-registration payload (hermetic on this
    checkout: the sealed R11 ranked-residual map it reads is present
    in the artifact store); skipped when absent."""
    if not R11_RANKED_MAP.is_file():
        pytest.skip(
            "the sealed R11 ranked-residual map is absent on this "
            "checkout; the pre-registration schema pins skip with a "
            "recorded reason"
        )
    return drv.build_pre_registration(catalog, families)


# ---------------------------------------------------------------------------
# Part 1: the builder blend contract
# ---------------------------------------------------------------------------
def test_variant_constants_rederived():
    """The card's variant decimals re-derived independently: the exact
    decimal arithmetic, the driver's literals (the v1 literal is the
    correctly rounded double of the card's decimal, one ulp apart from
    the naive float ``B - 0.2``; ``B + 0.2`` agrees bit-for-bit), the
    fixed ramp width, and the three boundary constants that must all
    equal (the builder's ``KERNEL_BOUNDARY_M8``, the card-6
    ``BLEND_BOUNDARY``, the M8-frozen ``continuum.match_mass``)."""
    # The card's exact-decimal arithmetic.
    assert Decimal("2.680194110727528") - Decimal("0.2") == (
        Decimal("2.480194110727528"))
    assert Decimal("2.680194110727528") + Decimal("0.2") == (
        Decimal("2.880194110727528"))
    # The float re-derivation: the v1 literal is the correctly rounded
    # double of the card's decimal; the naive float B - 0.2 is one ulp
    # lower (pinned, informational), B + 0.2 agrees bit-for-bit.
    assert V1 == float("2.480194110727528")
    assert V2 == B + 0.2 == float("2.880194110727528")
    assert V1 != B - 0.2
    assert V1.hex() != (B - 0.2).hex()

    import na49_mixing.resonance_pp_builder as builder_module

    assert builder_module.KERNEL_BOUNDARY_M8 == B
    assert BLEND_BOUNDARY == B
    assert BLEND_RAMP_WIDTH == RAMP_WIDTH
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["continuum.match_mass"] == B

    drv = _load_driver()
    assert drv.RAMP_WIDTH_GEV == RAMP_WIDTH
    starts = [entry["ramp_start_gev"] for entry in drv.VARIANTS]
    assert starts == [B, V1, V2]
    for variant, ramp_start in ((0, B), (1, V1), (2, V2)):
        spec = drv.variant_blend(variant)
        assert isinstance(spec, BlendSpec)
        assert spec.boundary == ramp_start
        assert spec.ramp_width == RAMP_WIDTH
        assert spec.weight(ramp_start) == 0.0
        assert spec.weight(ramp_start - 1.0) == 0.0
        assert spec.weight(ramp_start + RAMP_WIDTH) == 1.0
        assert spec.weight(ramp_start + RAMP_WIDTH + 1.0) == 1.0
        assert spec.weight(ramp_start + 0.25) == pytest.approx(0.5, rel=1e-15)
    with pytest.raises(ValueError):
        drv.variant_blend(3)


def test_blend_none_pinned_stream_digest():
    """The 240-event test-author stream at ``blend = None`` reproduces
    the pinned per-event hash list and its pinned sha256 (the
    byte-stability regression anchor: the R23 draw sequence and record
    contents are byte-stable under the additive extension)."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    hashes = []
    for event_id in range(TA_STREAM_EVENTS):
        builder = _make_builder(
            drv, config, catalog, TA_STREAM_SEED, event_id)
        result = builder.generate(stage=TA_STREAM_STAGE)
        hashes.append(result.event.as_dict()["event_hash"])
    assert len(hashes) == TA_STREAM_EVENTS
    assert hashes[0] == TA_STREAM_FIRST_HASH
    assert hashes[-1] == TA_STREAM_LAST_HASH
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == TA_STREAM_DIGEST


def test_blend_none_pre_extension_bit_identity():
    """The 240-event test-author stream at ``blend = None`` is
    bit-identical, event by event (full ``as_dict`` content), between
    the post-extension builder and the pre-extension R23 builder
    fetched from the nested repository at the pinned parent commit."""
    pre = _load_preextension_builder()
    if pre is None:
        pytest.skip(
            "git history unavailable on this checkout; the pre-"
            "extension bit-identity check skips (the pinned stream "
            "digest remains the byte-stability anchor)"
        )
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    for event_id in range(TA_STREAM_EVENTS):
        pre_builder = pre.ResonancePPBuilder(
            config, catalog, seed=TA_STREAM_SEED, event_id=event_id,
            master_seed=TA_STREAM_MASTER, replica_index=TA_STREAM_REPLICA,
        )
        post_builder = _make_builder(
            drv, config, catalog, TA_STREAM_SEED, event_id)
        pre_dict = pre_builder.generate(
            stage=TA_STREAM_STAGE).event.as_dict()
        post_dict = post_builder.generate(
            stage=TA_STREAM_STAGE).event.as_dict()
        assert pre_dict == post_dict, event_id


def test_routing_below_boundary_zero_uniforms():
    """Below the variant ramp start the routing consumes zero
    uniforms: exactly one ``decay_channel`` child draw (the record
    draw), the shared event generator untouched, and the record
    contents and child state bit-identical to the no-blend builder at
    the same seed (three probe masses per variant, all below the R23
    kernel boundary so both builders take the catalog path)."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    probe_masses = {
        0: (2.50, 2.58, 2.66),
        1: (2.30, 2.40, 2.46),
        2: (2.55, 2.60, 2.66),
    }
    for variant, masses in probe_masses.items():
        ramp_start = drv.variant_blend(variant).boundary
        assert all(mass < ramp_start for mass in masses)
        for i, mass in enumerate(masses):
            event_id = 4000 + 10 * variant + i
            plain = _make_builder(
                drv, config, catalog, TA_PROBE_SEED, event_id)
            blended = _make_builder(
                drv, config, catalog, TA_PROBE_SEED, event_id,
                variant=variant)
            plain_state = plain.rng.bit_generator.state
            plain_finals, plain_vertices, plain_record = plain.decay_source(
                _hand_source(mass, event_id), 0)
            blended_finals, blended_vertices, blended_record = (
                blended.decay_source(_hand_source(mass, event_id), 0))
            # The shared event generator is untouched by the decay
            # layer on both builders.
            assert plain.rng.bit_generator.state == plain_state
            assert blended.rng.bit_generator.state == plain_state
            # The decay_channel child state is bit-identical to the
            # no-blend convention at the same seed.
            assert (blended._rng_record.bit_generator.state
                    == plain._rng_record.bit_generator.state)
            # The record contents are bit-identical.
            assert [f.as_dict() for f in plain_finals] == (
                [f.as_dict() for f in blended_finals])
            assert blended_record.kernel == plain_record.kernel
            assert blended_record.route == "catalog"
            assert blended_record.blend_weight == 0.0
            # Exactly one child uniform (the record draw): an
            # instrumented fresh blended builder sees one call.
            fresh = _make_builder(
                drv, config, catalog, TA_PROBE_SEED, event_id,
                variant=variant)
            counter = _CountingRNG(fresh._rng_record)
            fresh._rng_record = counter
            fresh.decay_source(_hand_source(mass, event_id), 0)
            assert len(counter.calls) == 1, (variant, mass)


def test_routing_band_exactly_one_routing_uniform():
    """Inside the open transition interval ``B_v <= M < B_v + W``:
    exactly one routing uniform on the spawned ``decay_channel`` child
    before any record draw (two child draws per source in total), the
    route ``'continuum'`` exactly when the draw is below
    ``w(M) = (M - B_v) / W``, and the exact boundary ``M = B_v``
    (``w = 0``) never routes to the continuum.  Eight probe masses per
    variant."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    offsets = (0.0, 0.05, 0.125, 0.20, 0.275, 0.35, 0.425, 0.4999)
    for variant in range(3):
        ramp_start = drv.variant_blend(variant).boundary
        for i, offset in enumerate(offsets):
            mass = ramp_start + offset
            event_id = 5000 + 100 * variant + i
            builder = _make_builder(
                drv, config, catalog, TA_PROBE_SEED, event_id,
                variant=variant)
            counter = _CountingRNG(builder._rng_record)
            builder._rng_record = counter
            _, _, record = builder.decay_source(
                _hand_source(mass, event_id), 0)
            assert len(counter.calls) == 2, (variant, mass)
            weight = 0.0 if offset == 0.0 else (mass - ramp_start) / RAMP_WIDTH
            expected_route = "continuum" if counter.calls[0] < weight else "catalog"
            assert record.route == expected_route, (variant, mass,
                                                    counter.calls[0], weight)
            assert record.blend_weight == (
                0.0 if offset == 0.0 else (mass - ramp_start) / RAMP_WIDTH)
            if offset == 0.0:
                # The exact boundary: w = 0, the continuum is never
                # routed, the record is a catalog record.
                assert record.route == "catalog"
                assert record.kernel == "resonance"
            if record.route == "continuum":
                assert record.kernel == "continuum"
                assert record.family_id == "baryonic_np"
            else:
                assert record.kernel == "resonance"
                assert record.n_open_records > 0


def test_routing_above_ramp_end_zero_uniforms():
    """At ``M >= B_v + W`` the continuum route is taken with zero ramp
    uniforms: the ``decay_channel`` child consumes exactly one uniform
    (the class draw; the routing uniform is absent), and the drawn
    class equals the uniform over the family's ``allowed_decay_classes``
    re-derived independently (three probe masses per variant)."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    families = _load_families()
    active = [f for f in families
              if f.active and tuple(f.qn) == (1, 1, 0)]
    assert [f.family_id for f in active] == ["baryonic_np"]
    classes = active[0].allowed_decay_classes
    for variant in range(3):
        ramp_start = drv.variant_blend(variant).boundary
        for i, offset in enumerate((RAMP_WIDTH, RAMP_WIDTH + 0.05,
                                    RAMP_WIDTH + 0.3)):
            mass = ramp_start + offset
            event_id = 6000 + 100 * variant + i
            builder = _make_builder(
                drv, config, catalog, TA_PROBE_SEED, event_id,
                variant=variant)
            counter = _CountingRNG(builder._rng_record)
            builder._rng_record = counter
            _, vertices, record = builder.decay_source(
                _hand_source(mass, event_id), 0)
            assert len(counter.calls) == 1, (variant, mass)
            assert record.route == "continuum"
            assert record.blend_weight == 1.0
            index = min(int(counter.calls[0] * len(classes)),
                        len(classes) - 1)
            expected_daughters = tuple(
                int(value)
                for value in classes[index].strip("()").split(",")
            )
            assert vertices[0].channel.daughters == expected_daughters, (
                variant, mass, counter.calls[0], expected_daughters)


def test_boundary_matching_factor_never_consulted():
    """The routed Monte Carlo never consults the card-6 blend-
    normalization surface: an independent spy on ``blended_density``,
    ``blended_normalization_report``, ``_blended_pieces``, and
    ``_blended_support`` records zero consultations during a
    220-event v0 blended stream, and every continuum-routed source in
    the stream conserves its ``(B, Q, S) = (1, 1, 0)``."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    families = _load_families()
    counters = {name: 0 for name in (
        "blended_density", "blended_normalization_report",
        "_blended_pieces", "_blended_support")}

    def _make_spy(name):
        def _spy(*args, **kwargs):
            counters[name] += 1
            raise AssertionError(
                f"{name} consulted during the routed Monte Carlo")
        return _spy

    originals = {name: getattr(_spectral_density, name) for name in counters}
    for name in counters:
        setattr(_spectral_density, name, _make_spy(name))
    try:
        n_continuum = 0
        for event_id in range(TA_SPY_EVENTS):
            builder = _make_builder(
                drv, config, catalog, TA_SPY_SEED, event_id, variant=0,
                master=TA_STREAM_MASTER, replica=TA_SPY_REPLICA,
                families=families)
            result = builder.generate(stage=TA_SPY_STAGE)
            for record in result.source_records:
                if record.route != "continuum":
                    continue
                n_continuum += 1
                source_finals = [
                    f for f in result.event.final_particles
                    if int(f.source_id) == int(record.source_id)
                ]
                assert _source_qn(source_finals) == (1, 1, 0), (
                    event_id, record.source_id)
    finally:
        for name in counters:
            setattr(_spectral_density, name, originals[name])
    assert n_continuum > 0, "the spy stream must exercise the continuum route"
    assert all(count == 0 for count in counters.values()), counters


def test_continuum_route_record_structure():
    """The continuum-route record and root split at the test-author's
    own seed (six probes, source at rest so the lab/rest boost is the
    identity): the record stamps, the root vertex's ``parent_pdg = 0``
    sentinel with the source four-momentum and the drawn class, the
    Kaellen decay momentum and exact energy split, the root-split
    direction from the first two uniforms of the spawned
    ``decay_kinematics`` child, the ``channel_family`` split, and
    per-source ``(B, Q, S)`` conservation."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    from na49_mixing.resonance_widths import decay_momentum

    for i, mass in enumerate((3.2, 3.35, 3.6, 3.9, 4.2, 4.6)):
        event_id = 7001 + i
        builder = _make_builder(
            drv, config, catalog, TA_STRUCTURE_SEED, event_id, variant=0)
        finals, vertices, record = builder.decay_source(
            _hand_source(mass, event_id), 0)
        # The record stamps.
        assert record.kernel == "continuum"
        assert record.route == "continuum"
        assert record.family_id == "baryonic_np"
        assert record.blend_weight == 1.0
        assert record.boundary_new_kernel is False
        assert record.record_pdg is None
        # The root vertex: the sentinel parent, the source four-
        # momentum, the drawn class.
        root = vertices[0]
        assert root.parent_pdg == CONTINUUM_ROOT_SENTINEL == 0
        assert root.parent_mass == mass
        assert root.parent_four_momentum == (mass, 0.0, 0.0, 0.0)
        assert root.channel.parent_pdg == 0
        assert root.channel.branching_fraction == 1.0
        # The root daughters: momentum conservation, the Kaellen
        # decay momentum, the exact two-body energy split.
        daughter_a, daughter_b = root.daughters
        e_a = float(daughter_a.four_momentum[0])
        e_b = float(daughter_b.four_momentum[0])
        p_a = np.array(daughter_a.four_momentum[1:], dtype=float)
        p_b = np.array(daughter_b.four_momentum[1:], dtype=float)
        m_a = math.sqrt(e_a * e_a - float(p_a @ p_a))
        m_b = math.sqrt(e_b * e_b - float(p_b @ p_b))
        magnitude = float(np.linalg.norm(p_a))
        assert magnitude == pytest.approx(float(np.linalg.norm(p_b)),
                                          rel=1e-12)
        assert magnitude == pytest.approx(decay_momentum(mass, m_a, m_b),
                                          rel=1e-12)
        assert e_a == pytest.approx(
            (mass * mass + m_a * m_a - m_b * m_b) / (2.0 * mass),
            rel=1e-12)
        assert e_b == pytest.approx(
            (mass * mass - m_a * m_a + m_b * m_b) / (2.0 * mass),
            rel=1e-12)
        # The root-split direction from the first two uniforms of the
        # spawned decay_kinematics child (re-derived independently).
        kin = _probe_decay_kinematics_child(TA_MASTER, TA_REPLICA, event_id)
        u_cos, u_phi = kin.random(), kin.random()
        cos_theta = 2.0 * u_cos - 1.0
        phi = 2.0 * math.pi * u_phi
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        direction = np.array([
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        ])
        assert np.allclose(p_a, magnitude * direction, rtol=1e-12,
                           atol=1e-15)
        assert np.allclose(p_b, -magnitude * direction, rtol=1e-12,
                           atol=1e-15)
        # The channel-family split: the root-split products are
        # ``continuum_decay`` (direct source products), the cascade
        # products ``resonance_decay``.
        assert any(f.channel_family == CONTINUUM_FAMILY for f in finals)
        for final in finals:
            if int(final.parent_id) == 0:
                assert final.channel_family == CONTINUUM_FAMILY
            else:
                assert final.channel_family == "resonance_decay"
        # Per-source quantum-number conservation.
        assert _source_qn(finals) == (1, 1, 0), (mass, event_id)


def test_blend_argument_validation():
    """The builder's blend argument validation: ``families`` without a
    ``BlendSpec``, a non-``BlendSpec`` blend, a ``BlendSpec`` without
    ``families``, an empty family list, non-``ContinuumFamily`` items,
    and two activated families for the source quantum numbers are all
    refused before any draw."""
    drv = _load_driver()
    catalog = _load_catalog()
    config = _load_config(drv)
    families = _load_families()
    active = [f for f in families
              if f.active and tuple(f.qn) == (1, 1, 0)]

    with pytest.raises(TypeError, match="families is only valid"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, families=families)
    with pytest.raises(TypeError, match="blend must be a .*BlendSpec"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, blend=(B, RAMP_WIDTH),
            families=families)
    with pytest.raises(TypeError, match="families is required"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, blend=BlendSpec())
    with pytest.raises(ValueError, match="non-empty"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, blend=BlendSpec(), families=[])
    with pytest.raises(TypeError, match="ContinuumFamily records"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, blend=BlendSpec(),
            families=[object()])
    with pytest.raises(ValueError, match="exactly one activated"):
        drv.ResonancePPBuilder(
            config, catalog, seed=1, event_id=1, master_seed=TA_MASTER,
            replica_index=TA_REPLICA, blend=BlendSpec(),
            families=families + active[:1])
    # The validation consumes no draws: a refused construction leaves
    # no partial state behind (the construction itself raises).
    assert isinstance(active[0], ContinuumFamily)


# ---------------------------------------------------------------------------
# Part 2: the driver and seed contract
# ---------------------------------------------------------------------------
def test_driver_dual_tree_byte_identity():
    """The nested and outer driver copies are byte-identical at the
    pinned digest (the dual-tree contract; post REV-b9df812-01), and
    the builder module sits at its pinned post-extension digest."""
    assert _sha256_file(DRIVER_PATH) == DRIVER_SHA256
    assert _sha256_file(OUTER_DRIVER_PATH) == DRIVER_SHA256
    assert _sha256_file(DRIVER_PATH) == _sha256_file(OUTER_DRIVER_PATH)
    assert _sha256_file(BUILDER_PATH) == BUILDER_POST_EXTENSION_SHA256


class _RecordingBuilderFactory:
    """A transparent construction monitor: records the keyword
    arguments the chunk worker hands to the builder and delegates
    construction to the real class (the builder-construction probe of
    the smoke chunk path)."""

    def __init__(self, real):
        self._real = real
        self.constructions = []

    def __call__(self, *args, **kwargs):
        self.constructions.append(dict(kwargs))
        return self._real(*args, **kwargs)


def test_smoke_chunk_kernel_substream_smoke_block():
    """The smoke generation path derives its kernel substreams from the
    pre-registered smoke block ``5000 + 100*v + r`` (the
    REV-b9df812-01 contract): a construction monitor on the driver's
    chunk worker records the builder ``replica_index`` as 5000 for
    smoke v0 r0 and 5200 for smoke v2 r0 (never the production-style
    ``100*v`` indices 0/200 that collided with production under master
    20260919), and the chunk's event hashes cross-check against direct
    builds at the smoke block (equal: the same substream) and at the
    production indices (differ: a disjoint substream)."""
    drv = _load_driver()
    config = _load_config(drv)
    catalog = _load_catalog()
    families = _load_families()
    context = drv._build_generation_context(drv._smoke_tables(), smoke=True)
    previous_builder = drv.ResonancePPBuilder
    previous_context = drv._WORKER_CONTEXT
    monitor = _RecordingBuilderFactory(drv.ResonancePPBuilder)
    event_ids = {0: (7, 21), 2: (13, 42)}  # the test-author's fixture
    chunk_hashes: dict = {}
    drv.ResonancePPBuilder = monitor
    drv._set_worker_context(context)
    try:
        for variant in TA_SMOKE_VARIANTS:
            _start, hashes, _sums, _counts, _routing, _feeddown, n_events = (
                drv._r32_generate_chunk(
                    (
                        variant, 0,
                        drv.smoke_raw_seed(variant, 0),
                        list(event_ids[variant]),
                        float(drv.variant_blend(variant).boundary),
                    )
                )
            )
            assert n_events == len(event_ids[variant])
            chunk_hashes[variant] = list(hashes)
    finally:
        drv.ResonancePPBuilder = previous_builder
        drv._set_worker_context(previous_context)

    # The construction monitor: the smoke chunk worker hands the
    # pre-registered smoke-block kernel replica index to the builder,
    # with the smoke seed and master along with it.
    assert len(monitor.constructions) == 4
    for variant, ids in event_ids.items():
        for event_id in ids:
            construction = next(
                c for c in monitor.constructions
                if int(c["event_id"]) == event_id
            )
            assert int(construction["replica_index"]) == (
                5000 + 100 * variant)
            assert int(construction["replica_index"]) == (
                drv.smoke_kernel_replica_index(variant, 0))
            assert int(construction["replica_index"]) != 100 * variant
            assert int(construction["seed"]) == drv.smoke_raw_seed(variant, 0)
            assert int(construction["master_seed"]) == drv.MASTER_SEED
            assert drv.MASTER_SEED == TA_MASTER_SEED

    # Independent substream cross-check: the chunk's event hashes equal
    # the hashes of direct builds at the smoke block and differ from
    # the direct builds at the production-style indices (the pre-fix
    # collision), so the smoke kernel layer is disjoint from
    # production under master 20260919.
    for variant, ids in event_ids.items():
        seed = drv.smoke_raw_seed(variant, 0)
        blend = drv.variant_blend(variant)
        for event_id, chunk_hash in zip(ids, chunk_hashes[variant]):
            smoke_build = drv.ResonancePPBuilder(
                config, catalog, seed=seed, event_id=event_id,
                master_seed=drv.MASTER_SEED,
                replica_index=5000 + 100 * variant,
                blend=blend, families=families,
            ).generate(stage=drv.STAGE_LABEL)
            production_build = drv.ResonancePPBuilder(
                config, catalog, seed=seed, event_id=event_id,
                master_seed=drv.MASTER_SEED,
                replica_index=100 * variant,
                blend=blend, families=families,
            ).generate(stage=drv.STAGE_LABEL)
            assert smoke_build.event.as_dict()["event_hash"] == chunk_hash
            assert production_build.event.as_dict()["event_hash"] != (
                chunk_hash)


def test_seed_map_contract():
    """The seed map re-derived from the card's rules: 74 distinct raw
    seeds (24 production, 48 confirmation, 2 smoke), the blocks
    pairwise disjoint and disjoint from the R23 raw seeds, the retired
    card 9's planned seeds, and the prior masters; the kernel
    replica-index spaces pairwise disjoint; the driver's seed-map
    helpers agree with the re-derivation for every (variant, replica)
    pair."""
    all_seeds = (
        [seed for block in TA_PRODUCTION_BLOCKS for seed in block]
        + [seed for block in TA_CONFIRMATION_BLOCKS for seed in block]
        + list(TA_SMOKE_SEEDS)
    )
    assert len(all_seeds) == 74
    assert len(set(all_seeds)) == 74

    blocks = [set(block) for block in
              (TA_PRODUCTION_BLOCKS + TA_CONFIRMATION_BLOCKS)]
    blocks.append(set(TA_SMOKE_SEEDS))
    for i, block_a in enumerate(blocks):
        for block_b in blocks[i + 1:]:
            assert block_a.isdisjoint(block_b), (i, block_a, block_b)

    forbidden = (
        set(TA_R23_RAW_SEEDS) | set(TA_RETIREDCARD9_SEEDS)
        | set(TA_PRIOR_MASTERS)
    )
    assert set(all_seeds).isdisjoint(forbidden)

    for i, (a, b) in enumerate((
        (TA_PRODUCTION_KERNEL_INDICES, TA_CONFIRMATION_KERNEL_INDICES),
        (TA_PRODUCTION_KERNEL_INDICES, TA_SMOKE_KERNEL_INDICES),
        (TA_CONFIRMATION_KERNEL_INDICES, TA_SMOKE_KERNEL_INDICES),
    )):
        assert a.isdisjoint(b), i
    assert TA_PRODUCTION_KERNEL_INDICES == {0, 1, 2, 3, 4, 5, 6, 7,
                                            100, 101, 102, 103, 104, 105,
                                            106, 107, 200, 201, 202, 203,
                                            204, 205, 206, 207}
    assert TA_CONFIRMATION_KERNEL_INDICES == {
        900 + r for r in range(16)} | {
        1000 + r for r in range(16)} | {1100 + r for r in range(16)}
    assert TA_SMOKE_KERNEL_INDICES == {5000, 5200}

    drv = _load_driver()
    assert drv.MASTER_SEED == TA_MASTER_SEED
    for variant in range(3):
        for replica in range(8):
            assert drv.raw_replica_seed(variant, replica) == (
                TA_SEED_BASE + 600 + 100 * variant + replica)
            assert drv.kernel_replica_index(variant, replica) == (
                100 * variant + replica)
        for replica in range(TA_CONFIRMATION_REPLICAS):
            assert drv.confirmation_raw_seed(variant, replica) == (
                TA_SEED_BASE + 900 + 100 * variant + replica)
            assert drv.confirmation_kernel_replica_index(variant, replica) == (
                100 * variant + 900 + replica)
    for variant in TA_SMOKE_VARIANTS:
        assert drv.smoke_raw_seed(variant, 0) == (
            TA_SEED_BASE + 5000 + 100 * variant)
        assert drv.smoke_kernel_replica_index(variant, 0) == (
            5000 + 100 * variant)


def _pre_registration_fixture(request):
    """The (driver, pre-registration payload) pair, built once per
    module (the payload reads the sealed R11 ranked-residual map;
    skipped when absent on this checkout)."""
    drv = _load_driver()
    pre = _build_pre_registration(drv, _load_catalog(), _load_families())
    return drv, pre


def test_statistics_contract():
    """The T2 statistics from the driver constants and from the
    pre-registration schema's statistics block, checked against the
    card text: 8 replicas x 100,000 events per variant, 3 variants,
    2,400,000 events total."""
    drv = _load_driver()
    assert drv.T2_REPLICAS == CARD_T2_REPLICAS
    assert drv.T2_EVENTS_PER_REPLICA == CARD_T2_EVENTS_PER_REPLICA
    assert drv.N_VARIANTS == CARD_N_VARIANTS
    assert drv.TOTAL_EVENTS == CARD_TOTAL_EVENTS
    assert drv.TOTAL_EVENTS == (CARD_T2_REPLICAS
                                * CARD_T2_EVENTS_PER_REPLICA
                                * CARD_N_VARIANTS)
    _, pre = _pre_registration_fixture(None)
    stats = pre["statistics"]
    assert stats["tier"] == "T2"
    assert stats["replicas_per_variant"] == CARD_T2_REPLICAS
    assert stats["events_per_replica"] == CARD_T2_EVENTS_PER_REPLICA
    assert stats["events_per_variant"] == (
        CARD_T2_REPLICAS * CARD_T2_EVENTS_PER_REPLICA)
    assert stats["variants"] == CARD_N_VARIANTS
    assert stats["total_events"] == CARD_TOTAL_EVENTS
    variants = pre["variants"]
    assert [entry["variant"] for entry in variants] == [0, 1, 2]
    assert [entry["ramp_start_gev"] for entry in variants] == [B, V1, V2]
    assert all(entry["ramp_width_gev"] == RAMP_WIDTH for entry in variants)


def test_spread_bound_contract():
    """Both pre-registered spread bounds (2.0 / 3.0) present in the
    driver constants and in the pre-registration schema: the bin bound
    in units of the pooled MC standard error ``sqrt(SE_0^2 + SE_1^2 +
    SE_2^2)`` with the 95th percentile aggregate and the
    deterministic-SE floor 1e-12; the card's literal median-variant
    reading is recorded as informational (the median variant per bin
    is recorded as the reference beside the pooled unit)."""
    drv = _load_driver()
    assert drv.BIN_SPREAD_BOUND == CARD_BIN_SPREAD_BOUND
    assert drv.FEEDDOWN_SPREAD_BOUND == CARD_FEEDDOWN_SPREAD_BOUND
    assert drv.SE_FLOOR == CARD_SE_FLOOR
    _, pre = _pre_registration_fixture(None)
    spread = pre["cross_variant_spread"]
    assert spread["bin_spread_bound"] == CARD_BIN_SPREAD_BOUND
    assert spread["feeddown_spread_bound"] == CARD_FEEDDOWN_SPREAD_BOUND
    assert spread["se_floor"] == CARD_SE_FLOOR
    assert "sqrt(SE_0^2 + SE_1^2 + SE_2^2)" in spread["bin_spread_unit"]
    assert "sqrt(SE_0^2 + SE_1^2 + SE_2^2)" in spread["feeddown_spread_unit"]
    # The 95th percentile aggregate (the card's pre-registered
    # contract).
    assert "95th percentile" in spread["bin_spread_aggregate"]
    assert "< 2.0" in spread["bin_spread_aggregate"]
    # Informational: the card's literal "pooled MC standard error of
    # the median variant" reading; the schema records the median
    # variant per bin as the reference beside the pooled unit.
    assert "median variant per bin is recorded as the reference" in (
        spread["bin_spread_unit"])
    # The pure-scatter calibration note (the 1.8 inside the 2.0 bound).
    assert "1.8" in spread["bin_spread_calibration_note"]


def test_named_residual_and_robust_statistics_schema():
    """The named-residual precedent (the register shape of the sealed
    ``gate3_r31.json``) and the robust-statistics cross-reference
    (triggered only by finite-MC-scatter diagnostics; 16 x 100,000 x 3
    = 4,800,000 confirmation events at the +900 blocks; the
    confirmation run dir name) are present in the pre-registration
    schema."""
    drv = _load_driver()
    assert drv.CONFIRMATION_REPLICAS == TA_CONFIRMATION_REPLICAS
    assert drv.CONFIRMATION_RUN_NAME == (
        "r32_spectral_systematic_pp_confirmation_2026-09-19")
    _, pre = _pre_registration_fixture(None)
    precedent = pre["named_residual_precedent"]
    assert "gate3_r31.json" in precedent["register_shape"]
    cross = pre["robust_statistics_cross_reference"]
    assert "finite-MC-scatter" in cross["trigger"]
    stats = cross["statistics"]
    assert stats["replicas_per_variant"] == TA_CONFIRMATION_REPLICAS
    assert stats["events_per_replica"] == CARD_T2_EVENTS_PER_REPLICA
    assert stats["total_events"] == (
        CARD_N_VARIANTS * TA_CONFIRMATION_REPLICAS
        * CARD_T2_EVENTS_PER_REPLICA)
    assert stats["total_events"] == 4_800_000
    assert stats["master_seed"] == TA_MASTER_SEED
    # The +900 raw-seed blocks, re-derived.
    for variant in range(3):
        assert cross["raw_replica_seeds"][str(variant)] == (
            TA_CONFIRMATION_BLOCKS[variant])
    assert "100*v + 900 + r" in cross["kernel_replica_rule"]
    assert cross["run_dir"] == CARD_CONFIRMATION_RUN_DIR
