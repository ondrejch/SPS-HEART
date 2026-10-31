"""Hermetic tests for the R32 p+p spectral-systematic driver (Round 3, card 9).

Task ``TASK-20260918-r3-run-r32-spectral-systematic-pp`` (plan section 9
run R32, line 644), phase 1.  The tests exercise the card's
deterministic contract against the extended card-13 builder and the
committed driver, without any production Monte Carlo:

- ``test_variant_and_statistics_pins``: the three transition-window
  variants (round-3 index item 5) with the fixed ramp width, the T2
  statistics, the seed map, the spread bounds, the bit-identity clause,
  and the frozen substrate digests.
- ``test_seed_map_disjointness``: the 24 production raw seeds, the
  confirmation (+900) blocks, and the smoke (+5000) block are pairwise
  disjoint and disjoint from the R23 raw seeds, the retired card 9
  spec's planned seeds, and every prior master seed.
- ``test_kernel_layer_space_disjointness``: the kernel-layer substream
  replica indexes of the production, confirmation, and smoke tiers are
  pairwise disjoint.
- ``test_chunk_worker_kernel_index_by_mode``: the chunk worker's kernel
  substreams follow the mode (the smoke path draws at the
  pre-registered ``5000 + 100*v + r`` block, the value stamped on the
  replica record; the production path keeps ``100*v + r``), and the
  event metadata's ``kernel_replica_index`` equals the index the
  builder drew with.
- ``test_builder_blend_none_byte_identity``: the card-13 builder with
  ``blend = None`` regenerates the sealed R23 replica-0 stream's first
  1,000 event hashes bit for bit (the regression pin over the
  pre-extension pinned sample; the sample's full 100,000-event stream
  was verified against the sealed replica record's
  ``event_hash_set_sha256``).
- ``test_builder_blend_argument_validation``: families without a blend
  refused, a non-BlendSpec blend refused, a missing families list
  refused, and the explicit ``blend = None`` parity.
- ``test_blend_routing_contract``: the card-6 routing on
  hand-configured sources - zero routing uniforms below the boundary
  with bit-identical catalog finals, the transition-band pin (route
  ``'continuum'`` exactly when the spawned ``decay_channel`` uniform is
  below ``w(M)``), zero routing uniforms above the ramp end, and the
  boundary-matching factor never consulted by the routed builder.
- ``test_bit_identity_v0_10k``: the pre-registered 10,000-event
  comparison at the R23 replica-0 seed 20300913 (the driver's
  in-process instrument, 4 fork workers): the three source layers
  bit-identical across the legacy staged builder, the frozen R23
  builder, and the R32 v0 blend builder; the finals census recorded.
- ``test_pre_registration_structure``: the pre-registration schema
  (variants, seeds, statistics, spread bounds, named-residual
  precedent, robust-statistics trigger, bit-identity clause, gate
  shape).
- ``test_smoke_determinism`` / ``test_smoke_contract``: two identical
  CLI smoke runs produce byte-identical artifacts; the smoke records
  carry the pre-registered smoke seeds and the routing shift between
  the variants.
- ``test_layout_sync_inputs``: the nested and outer driver copies are
  byte-identical and the layout checker passes.

The smoke runs read the committed catalog, the committed continuum
config, and the committed production-config declarations only; no
NA49/SPS external tables are touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r32_spectral_systematic.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = NESTED_ROOT.parent.parent / DRIVER_REL
SYNC_CHECKER = NESTED_ROOT.parent.parent / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "continuum_resonances.json"
)
R11_MAP_PATH = NESTED_ROOT.parent.parent / "dev" / "00runs" / (
    "r11_observable_decomposition_2026-09-11"
) / "ranked_residual_map.json"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r32_spectral_systematic_under_test", NESTED_DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module: the fork worker pool resolves the
    # worker functions by module name (the sys.modules inheritance of
    # forked children makes the pickle reference resolvable).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture(scope="module")
def m8_config(driver):
    return driver._m8_config()


@pytest.fixture(scope="module")
def catalog():
    from na49_mixing.hadron_catalog import HadronCatalog

    return HadronCatalog.from_json(CATALOG_PATH)


# ----------------------------------------------------------------------
# Variant, statistics, seed, and substrate pins
# ----------------------------------------------------------------------
def test_variant_and_statistics_pins(driver):
    """The pre-registered variants, statistics, seed map, spread
    bounds, bit-identity clause, and frozen substrate digests."""
    from na49_mixing.spectral_density import BlendSpec

    assert driver.RUN_ID == "R32"
    assert driver.CARD == (
        "TASK-20260918-r3-run-r32-spectral-systematic-pp"
    )
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
    )
    assert driver.CONFIRMATION_RUN_NAME == (
        "r32_spectral_systematic_pp_confirmation_2026-09-19"
    )
    # The three transition-window variants (round-3 index item 5):
    # W = 0.5 GeV fixed, only the ramp start varies.
    assert driver.RAMP_WIDTH_GEV == 0.5
    assert [
        (int(v["variant"]), float(v["ramp_start_gev"]))
        for v in driver.VARIANTS
    ] == [
        (0, 2.680194110727528),
        (1, 2.480194110727528),
        (2, 2.880194110727528),
    ]
    for v in driver.VARIANTS:
        spec = driver.variant_blend(v["variant"])
        assert isinstance(spec, BlendSpec)
        assert spec.boundary == float(v["ramp_start_gev"])
        assert spec.ramp_width == 0.5
    # The T2 statistics.
    assert driver.T2_REPLICAS == 8
    assert driver.T2_EVENTS_PER_REPLICA == 100_000
    assert driver.TOTAL_EVENTS == 2_400_000
    assert driver.TOTAL_EVENTS == 3 * 8 * 100_000
    # The seed map (the card's seed scheme, verbatim).
    assert driver.MASTER_SEED == 20260919
    assert driver.raw_replica_seed(0, 0) == 20301519
    assert driver.raw_replica_seed(0, 7) == 20301526
    assert driver.raw_replica_seed(1, 0) == 20301619
    assert driver.raw_replica_seed(1, 7) == 20301626
    assert driver.raw_replica_seed(2, 0) == 20301719
    assert driver.raw_replica_seed(2, 7) == 20301726
    assert all(
        driver.raw_replica_seed(v, r) == 20301519 + 100 * v + r
        for v in range(3) for r in range(8)
    )
    assert driver.kernel_replica_index(2, 5) == 205
    # The confirmation (+900) blocks (acceptance (v)).
    assert driver.CONFIRMATION_REPLICAS == 16
    assert driver.confirmation_raw_seed(0, 0) == 20301819
    assert driver.confirmation_raw_seed(0, 15) == 20301834
    assert driver.confirmation_raw_seed(1, 0) == 20301919
    assert driver.confirmation_raw_seed(1, 15) == 20301934
    assert driver.confirmation_raw_seed(2, 0) == 20302019
    assert driver.confirmation_raw_seed(2, 15) == 20302034
    # The smoke (+5000) block (the hermetic driver validation).
    assert driver.SMOKE_RAW_SEED_BASE == 20305919
    assert driver.smoke_raw_seed(0, 0) == 20305919
    assert driver.smoke_raw_seed(2, 0) == 20306119
    assert driver.smoke_kernel_replica_index(0, 0) == 5000
    assert driver.smoke_kernel_replica_index(2, 0) == 5200
    assert driver.SMOKE_EVENTS == 2_000
    # The bit-identity clause.
    assert driver.BIT_IDENTITY_SEED == 20300913
    assert driver.BIT_IDENTITY_EVENTS == 10_000
    assert driver.BIT_IDENTITY_MASTER_SEED == 20260913
    assert driver.BIT_IDENTITY_REPLICA_INDEX == 0
    assert list(driver.LAYER_DEFINITIONS) == [
        "topology", "exchange", "spectral_state",
    ]
    # The spread bounds (the model-uncertainty contract).
    assert driver.BIN_SPREAD_BOUND == 2.0
    assert driver.FEEDDOWN_SPREAD_BOUND == 3.0
    assert driver.SE_FLOOR == 1e-12
    assert driver.FINITE_MC_SCATTER_MAX_SIGMA == 4.0
    # The frozen substrate digests (the card's substrate list).
    assert driver.ENGINE_SHA256 == (
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
    )
    assert driver.BUILDER_R23_PRE_EXTENSION_SHA256 == (
        "bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036"
    )
    assert driver.R31_DRIVER_SHA256 == (
        "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
    )
    assert driver.CATALOG_SHA256 == (
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
    )
    assert driver.FAMILIES_SHA256 == (
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
    )
    assert driver.SEALED_R23_EVENT_HASH_SET_SHA256 == (
        "503ab64e3ce23be92050e7e4e657a4aa4e276c2b9c267de7a14f86b7521221ec"
    )
    # The sealed run dirs the pre-registration verifies (the card's
    # read-only substrate).
    assert list(driver.SEALED_RUN_DIRS) == [
        "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13",
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r3_threshold_metric_reregistration_2026-09-17",
        "dev/00runs/r3_threshold_metric_engine_fix_2026-09-18",
        "dev/00runs/r3_se_audit_2026-09-18",
    ]


def test_seed_map_disjointness(driver):
    """The raw seed spaces are pairwise disjoint and disjoint from every
    prior-run seed space (the card's disjointness claim)."""
    production = [
        driver.raw_replica_seed(v, r)
        for v in range(driver.N_VARIANTS)
        for r in range(driver.T2_REPLICAS)
    ]
    confirmation = [
        driver.confirmation_raw_seed(v, r)
        for v in range(driver.N_VARIANTS)
        for r in range(driver.CONFIRMATION_REPLICAS)
    ]
    smoke = [
        driver.smoke_raw_seed(v, r)
        for v in driver.SMOKE_VARIANTS
        for r in range(driver.SMOKE_REPLICAS)
    ]
    r32_blocks = production + confirmation + smoke
    # Pairwise disjoint within R32.
    assert len(set(r32_blocks)) == len(r32_blocks) == (
        24 + 48 + len(driver.SMOKE_VARIANTS)
    )
    # Disjoint from the R23 raw replica seeds (20300913-20300920), the
    # bit-identity anchor seed included.
    r23_raw = set(range(20300913, 20300921))
    assert not (set(r32_blocks) & r23_raw)
    # Disjoint from the retired card 9 spec's planned seeds (never
    # drawn): 20301217-20301424.
    retired = set(range(20301217, 20301425))
    assert not (set(r32_blocks) & retired)
    # Disjoint from every prior master seed (the substream-derived
    # spaces of masters 20260913/20260917/20260918 are disjoint from
    # R32's kernel layer by the master seed alone).
    for master in (20260913, 20260917, 20260918):
        assert driver.MASTER_SEED != master
    # The production blocks sit between the retired block and the
    # confirmation block, all inside one documented window.
    assert min(production) == 20301519
    assert max(production) == 20301726
    assert min(confirmation) == 20301819
    assert max(confirmation) == 20302034
    assert min(smoke) == 20305919
    assert max(smoke) == 20306119
    # The R32 master seed differs from the R31 master (20260917) and
    # the card-8c master (20260918).
    assert driver.MASTER_SEED not in (20260913, 20260917, 20260918)


def test_kernel_layer_space_disjointness(driver):
    """The kernel-layer substream replica indexes are pairwise disjoint
    across the production, confirmation, and smoke tiers."""
    production = {
        driver.kernel_replica_index(v, r)
        for v in range(driver.N_VARIANTS)
        for r in range(driver.T2_REPLICAS)
    }
    confirmation = {
        driver.confirmation_kernel_replica_index(v, r)
        for v in range(driver.N_VARIANTS)
        for r in range(driver.CONFIRMATION_REPLICAS)
    }
    smoke = {
        driver.smoke_kernel_replica_index(v, r)
        for v in driver.SMOKE_VARIANTS
        for r in range(driver.SMOKE_REPLICAS)
    }
    assert production == set(range(8)) | set(range(100, 108)) | set(
        range(200, 208)
    )
    assert len(confirmation) == 48
    assert len(smoke) == len(driver.SMOKE_VARIANTS)
    assert not (production & confirmation)
    assert not (production & smoke)
    assert not (confirmation & smoke)


# ----------------------------------------------------------------------
# The chunk worker's kernel-index selection (the smoke substream contract)
# ----------------------------------------------------------------------
def test_chunk_worker_kernel_index_by_mode(driver):
    """The chunk worker's kernel substreams follow the mode (the
    REV-b9df812-01 contract): the smoke path draws at the pre-registered
    smoke block ``5000 + 100*v + r``, the value stamped on the replica
    record, so the smoke kernel layer stays disjoint from production and
    confirmation; the production path keeps ``100*v + r``.  The event
    metadata's ``kernel_replica_index`` equals the index the builder
    drew with.  The synthetic smoke grid serves as the binning tables on
    both legs (binning only; the index selection does not read the
    tables), so the test is hermetic."""
    class _Recorder(driver.ResonancePPBuilder):
        built: list = []
        event_indices: list = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).built.append(int(self.replica_index))

        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            type(self).event_indices.append(
                int(result.event.metadata["kernel_replica_index"])
            )
            return result

    previous_builder = driver.ResonancePPBuilder
    previous_context = driver._WORKER_CONTEXT
    driver.ResonancePPBuilder = _Recorder
    try:
        # The smoke leg: the chunk worker draws at 5000 + 100*v + r.
        context = driver._build_generation_context(
            driver._smoke_tables(), smoke=True
        )
        driver._set_worker_context(context)
        for variant in driver.SMOKE_VARIANTS:
            driver._r32_generate_chunk((
                variant, 0, driver.smoke_raw_seed(variant, 0), [0, 1],
                float(driver.variant_blend(variant).boundary),
            ))
        assert _Recorder.built == [5000, 5000, 5200, 5200]
        assert _Recorder.event_indices == [5000, 5000, 5200, 5200]
        # The production leg: the chunk worker keeps 100*v + r.
        context = driver._build_generation_context(
            driver._smoke_tables(), smoke=False
        )
        driver._set_worker_context(context)
        driver._r32_generate_chunk((
            0, 0, driver.raw_replica_seed(0, 0), [0, 1],
            float(driver.variant_blend(0).boundary),
        ))
        driver._r32_generate_chunk((
            2, 5, driver.raw_replica_seed(2, 5), [0],
            float(driver.variant_blend(2).boundary),
        ))
        assert _Recorder.built[-3:] == [0, 0, 205]
        assert _Recorder.event_indices[-3:] == [0, 0, 205]
    finally:
        driver.ResonancePPBuilder = previous_builder
        driver._set_worker_context(previous_context)


# ----------------------------------------------------------------------
# The card-13 builder: the blend=None byte-identity regression pin
# ----------------------------------------------------------------------
def test_builder_blend_none_byte_identity(driver):
    """The card-13 builder with ``blend = None`` regenerates the sealed
    R23 replica-0 stream's first 1,000 event hashes bit for bit (the
    additive blend parameter is a no-op on the default path).

    The pinned sample was generated pre-extension at the R23 replica-0
    conventions (seed 20300913, kernel master 20260913, replica index
    0, the R23 stage label) and its full 100,000-event stream was
    verified against the sealed R23 replica record's
    ``event_hash_set_sha256``; the pin here is the digest of the joined
    hash list plus the boundary hashes verbatim.
    """
    from na49_mixing.hadron_catalog import HadronCatalog

    pinned_digest = "496e68905e2a590ccaea03a6d44af254d43dbffbdf8335af513a57e5265fa634"
    pinned_first = (
        "2b757e6c0cdd52e525a51ddb6188c193b7dd3fb8bdaeb763d61a18ceb6d56a2b",
        "6fc52994cfdb0534d7715925c991f7f971c4d984ecf84ac3c157f8b619b2366a",
        "cba7069f6a7bc3cb99de130baaf84ae6f924c7f2a432ce800b486f37d30e443e",
    )
    pinned_last = "12844ecdc84b4decfa121287e273c7dec76e69cad528404bc620361cb822abb2"
    n = driver.BLEND_NONE_SAMPLE_N
    assert n == 1_000
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    config = driver._m8_config()
    hashes = []
    for event_id in range(n):
        builder = driver.ResonancePPBuilder(
            config, catalog,
            seed=driver.BLEND_NONE_SAMPLE_SEED, event_id=event_id,
            master_seed=driver.ROUND2_MASTER_SEED, replica_index=0,
        )
        result = builder.generate(stage="r23_resonance_ablation")
        assert builder.blend is None
        hashes.append(result.event.as_dict()["event_hash"])
    assert hashes[:3] == list(pinned_first)
    assert hashes[-1] == pinned_last
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == pinned_digest, (
        "the blend = None builder no longer reproduces the pre-extension "
        "R23 replica-0 stream (the card-13 byte-stability pin)"
    )


def test_builder_blend_argument_validation(driver, m8_config, catalog):
    """families without a blend refused, a non-BlendSpec blend refused,
    a missing families list refused, and the explicit ``blend = None``
    parity with the omitted default."""
    from na49_mixing.spectral_density import BlendSpec, families_from_json

    families = families_from_json(str(FAMILIES_PATH))
    kwargs = dict(seed=20300913, event_id=0, master_seed=20260913,
                  replica_index=0)
    # families without a blend is refused.
    with pytest.raises(TypeError):
        driver.ResonancePPBuilder(
            m8_config, catalog, families=families, **kwargs
        )
    # a non-BlendSpec blend is refused.
    with pytest.raises(TypeError):
        driver.ResonancePPBuilder(
            m8_config, catalog, blend=(2.68, 0.5), families=families,
            **kwargs
        )
    # a BlendSpec without families is refused.
    with pytest.raises(TypeError):
        driver.ResonancePPBuilder(
            m8_config, catalog, blend=BlendSpec(), **kwargs
        )
    # an empty family list is refused.
    with pytest.raises(ValueError):
        driver.ResonancePPBuilder(
            m8_config, catalog, blend=BlendSpec(), families=[], **kwargs
        )
    # the explicit blend = None parity: identical behavior to the
    # omitted default on a small sample.
    hashes_default = []
    hashes_explicit = []
    for event_id in range(40):
        default = driver.ResonancePPBuilder(m8_config, catalog, **{
            **kwargs, "event_id": event_id,
        })
        explicit = driver.ResonancePPBuilder(
            m8_config, catalog, blend=None, **{
                **kwargs, "event_id": event_id,
            }
        )
        assert explicit.blend is None
        hashes_default.append(
            default.generate().event.as_dict()["event_hash"]
        )
        hashes_explicit.append(
            explicit.generate().event.as_dict()["event_hash"]
        )
    assert hashes_default == hashes_explicit


def _hand_source(mass: float, source_id: int = 0):
    from na49_mixing.excitation_kinematics import FourVector
    from na49_mixing.quantum_numbers import QuantumNumbers
    from na49_mixing.source_event import SourceRecord

    return SourceRecord(
        source_id, "projectile",
        FourVector(float(mass), 0.0, 0.0, 0.0),
        QuantumNumbers(1, 1, 0),
        collision_index=1,
        metadata={"sampled_mass": float(mass), "t": 0.0},
    )


# ----------------------------------------------------------------------
# The card-6 blend routing contract (hand-configured sources)
# ----------------------------------------------------------------------
def test_blend_routing_contract(driver, m8_config, catalog):
    """The card-6 routing contract on hand-configured sources: below B
    the catalog route consumes zero uniforms and reproduces the
    blend=None catalog finals bit for bit; inside the band the route is
    'continuum' exactly when the spawned decay_channel uniform is below
    w(M); above B + W the continuum route consumes zero routing
    uniforms; the boundary-matching factor is never consulted."""
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.spectral_density import BlendSpec, families_from_json

    families = families_from_json(str(FAMILIES_PATH))
    seed, event_id = 20300913, 777
    below_mass = 2.5
    band_mass = 2.9
    above_mass = 3.5

    def make_builder(spec):
        return driver.ResonancePPBuilder(
            m8_config, catalog, seed=seed, event_id=event_id,
            master_seed=20260913, replica_index=0,
            blend=spec, families=families,
        )

    def make_builder_none():
        return driver.ResonancePPBuilder(
            m8_config, catalog, seed=seed, event_id=event_id,
            master_seed=20260913, replica_index=0,
        )

    # -- below B: zero routing uniforms, bit-identical catalog finals ---
    spec = BlendSpec(2.680194110727528, 0.5)
    plain = make_builder_none()
    plain_finals, _plain_vertices, _plain_record = plain.decay_source(
        _hand_source(below_mass)
    )
    plain_state_after = plain._rng_record.bit_generator.state
    blended = make_builder(spec)
    state_before = blended.rng.bit_generator.state
    finals, vertices, record = blended.decay_source(
        _hand_source(below_mass)
    )
    assert record.route == "catalog"
    assert record.kernel == driver.KERNEL_LABEL
    assert record.blend_weight == 0.0
    assert blended.rng.bit_generator.state == state_before
    # The record child's state after the decay equals the no-blend
    # builder's after its own record draw: the routing consumed zero
    # uniforms, so the catalog-path draw sequence is byte-identical.
    assert (
        blended._rng_record.bit_generator.state == plain_state_after
    )
    assert [f.as_dict() for f in finals] == [
        f.as_dict() for f in plain_finals
    ], (
        "the below-boundary catalog route is not bit-identical to the "
        "no-blend catalog branch"
    )

    # -- inside the band: the routing pin over 40 probes -----------------
    weight = spec.weight(band_mass)
    assert 0.0 < weight < 1.0
    n_continuum = 0
    for probe_index in range(40):
        eid = 900 + probe_index
        subs = derive_substreams(20260913, 0, eid)
        probe = np.random.default_rng(
            np.random.SeedSequence(
                subs.stream_for("decay_channel")).spawn(1)[0]
        )
        u = float(probe.random())
        builder = driver.ResonancePPBuilder(
            m8_config, catalog, seed=seed, event_id=eid,
            master_seed=20260913, replica_index=0,
            blend=spec, families=families,
        )
        _f, _v, rec = builder.decay_source(_hand_source(band_mass))
        expected_route = "continuum" if u < weight else "catalog"
        assert rec.route == expected_route, (
            f"probe {probe_index}: route {rec.route!r} != the pinned "
            f"{expected_route!r} (u={u!r}, w={weight!r})"
        )
        n_continuum += int(rec.route == "continuum")
        assert rec.blend_weight == weight
        if rec.route == "continuum":
            assert rec.kernel == driver.CONTINUUM_LABEL
            assert rec.family_id == "baryonic_np"
        else:
            assert rec.kernel == driver.KERNEL_LABEL
    assert 0 < n_continuum < 40

    # -- above B + W: the continuum route, zero routing uniforms ---------
    above_builder = make_builder(spec)
    state_before = above_builder.rng.bit_generator.state
    finals, vertices, record = above_builder.decay_source(
        _hand_source(above_mass)
    )
    assert record.route == "continuum"
    assert record.kernel == driver.CONTINUUM_LABEL
    assert record.blend_weight == 1.0
    assert record.family_id == "baryonic_np"
    assert above_builder.rng.bit_generator.state == state_before
    # The record child consumed exactly one uniform (the class draw):
    # the routing consumed zero at and above the ramp end.
    subs = derive_substreams(20260913, 0, event_id)
    probe = np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
    )
    probe.random()
    assert above_builder._rng_record.bit_generator.state == (
        probe.bit_generator.state
    )
    assert finals, "the continuum route must produce final particles"
    total = [0, 0, 0]
    for final in finals:
        qn = final.quantum_numbers
        total[0] += qn.B
        total[1] += qn.Q
        total[2] += qn.S
    assert tuple(total) == (1, 1, 0)
    # The root split is vertex 0 with the continuum root sentinel; the
    # root-split finals are direct source products (parent_id 0).
    assert vertices is not None and len(vertices) >= 1
    assert vertices[0].parent_pdg == driver.CONTINUUM_ROOT_SENTINEL

    # -- the boundary-matching factor is never consulted ------------------
    from na49_mixing import spectral_density

    calls = {"blended_pieces": 0, "blended_normalization": 0}
    original_pieces = spectral_density._blended_pieces
    original_report = spectral_density.blended_normalization_report

    def spy_pieces(*args, **kwargs):
        calls["blended_pieces"] += 1
        return original_pieces(*args, **kwargs)

    def spy_report(*args, **kwargs):
        calls["blended_normalization"] += 1
        return original_report(*args, **kwargs)

    spectral_density._blended_pieces = spy_pieces
    spectral_density.blended_normalization_report = spy_report
    try:
        for mass in (below_mass, band_mass, above_mass):
            builder = make_builder(spec)
            builder.decay_source(_hand_source(mass))
    finally:
        spectral_density._blended_pieces = original_pieces
        spectral_density.blended_normalization_report = original_report
    assert calls == {
        "blended_pieces": 0, "blended_normalization": 0,
    }, (
        "the routed builder must never consult the blend's "
        "boundary-matching factor (the card-6 contract)"
    )


# ----------------------------------------------------------------------
# The pre-registered 10,000-event bit identity (variant v0)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bit_identity_10k(driver):
    return driver.bit_identity_report(10_000, 4)


def test_bit_identity_v0_10k(driver, bit_identity_10k):
    """The pre-registered 10,000-event bit-identity check at the R23
    replica-0 seed 20300913: the three source layers bit-identical
    across the legacy staged builder, the frozen R23 builder
    (blend = None), and the R32 v0 blend builder; the finals layers
    recorded under the R23 perturbation-skip convention."""
    report = bit_identity_10k
    assert report["variant"] == 0
    assert report["seed"] == 20300913
    assert report["n_events"] == 10_000
    assert report["blend"] == {
        "boundary_gev": 2.680194110727528,
        "ramp_width_gev": 0.5,
    }
    assert report["pass"] is True
    for name in ("legacy_vs_builder", "builder_vs_v0"):
        for key in ("topology", "exchange", "spectral_state"):
            slot = report["asserted_layers"][name][key]
            assert slot["n_matched"] == slot["n_compared"] == 10_000, (
                f"{name}/{key}: {slot['n_matched']} of "
                f"{slot['n_compared']} matched"
            )
            assert slot["mismatches"] == []
    census = report["finals_census"]
    assert census["n_below_boundary_sources"] > 0
    assert census["n_in_band_sources"] > 0
    assert census["n_above_boundary_sources"] > 0
    assert census["v0_catalog_routed"] > 0
    assert census["v0_continuum_routed"] > 0
    assert (
        census["below_boundary_finals_compared"]
        == census["below_boundary_finals_matched"]
    ) and census["below_boundary_finals_compared"] > 0, (
        "the comparable below-boundary v0 catalog finals must match the "
        "frozen builder's bit for bit"
    )
    assert (
        census["legacy_finals_compared"] == census["legacy_finals_matched"]
    ) and census["legacy_finals_compared"] > 0, (
        "the frozen builder's legacy finals must match the legacy "
        "builder's bit for bit (the R23 legacy_kernel_finals layer)"
    )
    assert census["legacy_finals_skipped_new_kernel"] > 0
    assert census["below_boundary_finals_skipped_perturbed"] > 0


# ----------------------------------------------------------------------
# The pre-registration structure
# ----------------------------------------------------------------------
def test_pre_registration_structure(driver):
    """The pre-registration schema: the variants, the seed scheme (with
    the confirmation and smoke blocks), the statistics, the spread
    bounds, the named-residual precedent, the robust-statistics
    cross-reference trigger, the bit-identity clause, the no-refit
    clause, and the Gate 3 handoff shape."""
    if not R11_MAP_PATH.is_file():
        pytest.skip("the sealed R11 ranked residual map is absent")
    catalog = driver._load_catalog()
    families = driver._load_families()
    pre = driver.build_pre_registration(catalog, families)
    assert pre["schema"] == (
        "na49_mixing.staged_r32_spectral_systematic.pre_registration/1"
    )
    assert pre["run"] == "R32" and pre["card"] == driver.CARD
    assert pre["written_before_generation"] is True
    # The collision (the M8 beam momentum 158 GeV/c; sqrt(s) =
    # 17.270 GeV, the plan section 9 "sqrt(s) = 17.2 GeV" reference).
    collision = pre["collision"]
    assert collision["system"] == "p+p"
    assert abs(float(collision["sqrt_s_gev"]) - 17.2702) < 0.001
    assert float(collision["beam_momentum_gev_c"]) == 158.0
    assert "(1, 1, 0)" in collision["topology"]
    # The variants (round-3 index item 5).
    assert [
        (int(v["variant"]), float(v["ramp_start_gev"]),
         float(v["ramp_width_gev"]))
        for v in pre["variants"]
    ] == [
        (0, 2.680194110727528, 0.5),
        (1, 2.480194110727528, 0.5),
        (2, 2.880194110727528, 0.5),
    ]
    # The statistics (T2).
    stats = pre["statistics"]
    assert stats["replicas_per_variant"] == 8
    assert stats["events_per_replica"] == 100_000
    assert stats["total_events"] == 2_400_000
    # The seed map with the confirmation and smoke blocks.
    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == 20260919
    assert seed_map["raw_replica_seeds"]["0"] == list(
        range(20301519, 20301527)
    )
    assert seed_map["raw_replica_seeds"]["1"] == list(
        range(20301619, 20301627)
    )
    assert seed_map["raw_replica_seeds"]["2"] == list(
        range(20301719, 20301727)
    )
    confirmation = pre["robust_statistics_cross_reference"]
    assert confirmation["statistics"]["replicas_per_variant"] == 16
    assert confirmation["statistics"]["total_events"] == 4_800_000
    assert confirmation["raw_seed_rule"] == driver.CONFIRMATION_RAW_SEED_RULE
    assert confirmation["raw_replica_seeds"]["0"] == list(
        range(20301819, 20301835)
    )
    assert confirmation["raw_replica_seeds"]["1"] == list(
        range(20301919, 20301935)
    )
    assert confirmation["raw_replica_seeds"]["2"] == list(
        range(20302019, 20302035)
    )
    assert "triggered only by finite-MC-scatter diagnostics" in (
        confirmation["trigger"]
    )
    assert f"dev/00runs/{driver.CONFIRMATION_RUN_NAME}" in (
        confirmation["run_dir"]
    )
    # The builder extension and the blend surface.
    builder_block = pre["builder"]
    assert builder_block["pre_extension_r23_sha256"] == (
        driver.BUILDER_R23_PRE_EXTENSION_SHA256
    )
    assert "blend = None keeps the R23 behavior byte-stable" in (
        builder_block["extension"]
    )
    assert "na49_mixing/spectral_density.py" in builder_block[
        "blend_surface"
    ]
    # The bit-identity clause.
    bit = pre["bit_identity_check"]
    assert bit["seed"] == 20300913 and bit["n_events"] == 10_000
    assert list(bit["layers_checked"]) == [
        "topology", "exchange", "spectral_state",
    ]
    assert "recorded, not asserted" in bit["finals_layer"]
    # The comparisons and the spread bounds.
    comparisons = pre["comparisons"]
    assert comparisons["paper_window"]["total_bins"] == 1017
    assert len(comparisons["identified_set"]["entries"]) == 5
    assert "corrected lineage convention" in comparisons["feed_down"][
        "lineage_convention"
    ]
    spread = pre["cross_variant_spread"]
    assert spread["bin_spread_bound"] == 2.0
    assert spread["feeddown_spread_bound"] == 3.0
    assert "95th percentile" in spread["bin_spread_aggregate"]
    # The named-residual precedent and the diagnosis rule.
    register = pre["named_residual_precedent"]
    assert "gate3_r31.json" in register["register_shape"]
    assert "finite_mc_scatter" in register["diagnosis_rule"]
    assert "se_estimator_artifact" in register["diagnosis_rule"]
    # The conservation, routing census, and the no-refit clauses.
    assert "never repaired" in pre["conservation"]["statement"]
    assert "catalog vs continuum" in pre["routing_census"]["statement"]
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False
    # The Gate 3 handoff shape.
    gate = pre["gate_evidence_shape"]
    assert gate["file"] == "gate3_r32.json"
    assert gate["gate"] == "gate 3 (plan section 9, line 650)"
    assert gate["clauses"] == ["smoothness", "no_boundary_dominance"]
    # The pre-registration asserts against the driver constants.
    run_dir = None  # in-memory structural check only
    del run_dir


# ----------------------------------------------------------------------
# Reduced CLI smoke runs (hermetic; tmp run dirs)
# ----------------------------------------------------------------------
def _run_smoke(run_dir: Path, workers: int = 2) -> None:
    result = subprocess.run(
        [
            sys.executable, str(NESTED_DRIVER),
            "--smoke", "--run-dir", str(run_dir),
            "--workers", str(workers),
        ],
        capture_output=True, text=True, timeout=1800,
        cwd=str(NESTED_ROOT.parent.parent),
    )
    assert result.returncode == 0, (
        f"the smoke run failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two identical smoke run dirs for the byte-determinism check."""
    first = tmp_path_factory.mktemp("r32_smoke_first")
    second = tmp_path_factory.mktemp("r32_smoke_second")
    _run_smoke(first)
    _run_smoke(second)
    return first, second


def test_smoke_determinism(smoke_runs):
    """Two identical CLI smoke invocations produce byte-identical
    artifacts (the smoke records carry no wall-clock fields)."""
    first, second = smoke_runs
    names = sorted(p.name for p in first.iterdir())
    assert names == [
        "smoke_pre_registration.json",
        "smoke_spread_report.json",
        "smoke_variant_0_replica_0.json",
        "smoke_variant_2_replica_0.json",
    ]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), (
            f"{name} differs between two identical smoke runs"
        )


def test_smoke_contract(driver, smoke_runs):
    """The smoke records: the pre-registered smoke seeds, the smoke
    kernel replica indexes, the mode tags, the exact event/conservation
    counts, and the routing shift between the variants."""
    first, _ = smoke_runs
    pre = json.loads(
        (first / "smoke_pre_registration.json").read_text(encoding="utf-8")
    )
    assert pre["mode"] == "smoke"
    assert pre["smoke"]["raw_replica_seeds"]["0"] == [20305919]
    assert pre["smoke"]["raw_replica_seeds"]["2"] == [20306119]
    for variant, seed, kernel_index in ((0, 20305919, 5000),
                                        (2, 20306119, 5200)):
        record = json.loads(
            (first / f"smoke_variant_{variant}_replica_0.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["mode"] == "smoke"
        assert record["variant"] == variant
        assert record["seed"] == seed
        assert record["kernel_replica_index"] == kernel_index
        assert record["n_events"] == 2_000
        assert record["conservation"]["event_b_q_conserved"] is True
        assert record["conservation"]["events_checked"] == 2_000
        assert record["blend"]["ramp_width_gev"] == 0.5
        assert record["routing"]["n_sources"] == 4_000
        # No wall-clock fields in the smoke records.
        assert "wall_seconds" not in record
        assert "started_utc" not in record
    report = json.loads(
        (first / "smoke_spread_report.json").read_text(encoding="utf-8")
    )
    assert report["mode"] == "smoke"
    assert report["summary"]["events_checked"] == 4_000
    assert report["summary"]["bin_spread_bound"] == 2.0
    assert report["summary"]["feeddown_spread_bound"] == 3.0
    # The v2 (late) variant routes more sources to the catalog than v0
    # (the recorded routing shift; the smoke census is recorded, not
    # asserted against an analytic expectation).
    routing = report["routing_census"]
    assert routing["2"]["fraction_catalog"] > routing["0"][
        "fraction_catalog"
    ]


# ----------------------------------------------------------------------
# Layout sync
# ----------------------------------------------------------------------
def test_layout_sync_inputs():
    """The nested and outer copies of the driver are byte-identical and
    the layout checker passes."""
    assert NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file()
    nested = hashlib.sha256(NESTED_DRIVER.read_bytes()).hexdigest()
    outer = hashlib.sha256(OUTER_DRIVER.read_bytes()).hexdigest()
    assert nested == outer, (
        "the nested and outer staged_r32_spectral_systematic.py copies "
        "differ"
    )
    if SYNC_CHECKER.is_file():
        result = subprocess.run(
            [sys.executable, str(SYNC_CHECKER)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"check_helpers_layout_sync.py failed (exit "
            f"{result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
        assert "logic_drift=0" in result.stdout
