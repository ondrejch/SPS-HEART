"""Tests for na49_mixing.parallel chunked generation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from na49_mixing import pdg_database
from na49_mixing.event import Event
from na49_mixing.generator import Generator
from na49_mixing.parallel import (
    GenerationConfig,
    _build_decay_chain,
    _build_side_generator,
    _chunk_seed,
    _derive_channel_audit_tables,
    _effective_tower_updates,
    _merge_channel_audit_accumulators,
    _parent_line_shape_map,
    _patch_generator_mass_width,
    _resolve_effective_tower_overrides,
    _resolved_match_mass_override,
    _generate_tracks_chunk_worker,
    generate_tracks_chunked,
    merge_track_chunks,
    split_event_counts,
)
from na49_mixing.pdg_database import get_mass, get_width, mass_width_overrides
from na49_mixing.reaction import Reaction


def _delta_gen_file() -> str:
    base = Path(__file__).resolve().parents[1] / "na49_mixing" / "generators" / "michal" / "deltas-to-pion"
    return str(base / "delta++.gen")


def test_split_event_counts_balanced() -> None:
    assert split_event_counts(0, 4) == [0, 0, 0, 0]
    assert split_event_counts(10, 3) == [4, 3, 3]
    assert split_event_counts(3, 5) == [1, 1, 1, 0, 0]


def test_merge_track_chunks_orders_by_chunk_index() -> None:
    chunks = [
        {
            "chunk_index": 2,
            "pdg": np.array([3], dtype=np.int32),
            "xf": np.array([0.3], dtype=np.float32),
            "pt": np.array([3.0], dtype=np.float32),
            "source_side": np.array([1], dtype=np.int8),
        },
        {
            "chunk_index": 0,
            "pdg": np.array([1], dtype=np.int32),
            "xf": np.array([0.1], dtype=np.float32),
            "pt": np.array([1.0], dtype=np.float32),
            "source_side": np.array([-1], dtype=np.int8),
        },
        {
            "chunk_index": 1,
            "pdg": np.array([2], dtype=np.int32),
            "xf": np.array([0.2], dtype=np.float32),
            "pt": np.array([2.0], dtype=np.float32),
            "source_side": np.array([1], dtype=np.int8),
        },
    ]
    merged = merge_track_chunks(chunks)
    assert np.array_equal(merged["pdg"], np.array([1, 2, 3], dtype=np.int32))
    assert np.allclose(merged["xf"], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(merged["pt"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(merged["source_side"], np.array([-1, 1, 1], dtype=np.int8))
    # All merged arrays share one length; the optional metadata fields are
    # synthesized as zeros for chunks that omit them.
    for key in ("pdg", "xf", "pt", "source_side", "weak_feed", "gen_depth"):
        assert merged[key].shape[0] == 3, key
    assert merged["weak_feed"].dtype == np.int8
    assert merged["gen_depth"].dtype == np.int16
    assert np.all(merged["weak_feed"] == 0)
    assert np.all(merged["gen_depth"] == 0)


def test_merge_track_chunks_synthesizes_missing_optional_fields() -> None:
    chunks = [
        {   # legacy chunk without either optional field
            "chunk_index": 0,
            "pdg": np.array([1, 2], dtype=np.int32),
            "xf": np.array([0.1, 0.2], dtype=np.float32),
            "pt": np.array([1.0, 2.0], dtype=np.float32),
            "source_side": np.array([-1, 1], dtype=np.int8),
        },
        {   # weak_feed present, gen_depth missing
            "chunk_index": 1,
            "pdg": np.array([3], dtype=np.int32),
            "xf": np.array([0.3], dtype=np.float32),
            "pt": np.array([3.0], dtype=np.float32),
            "source_side": np.array([1], dtype=np.int8),
            "weak_feed": np.array([1], dtype=np.int8),
        },
        {   # empty chunk, only gen_depth present
            "chunk_index": 2,
            "pdg": np.array([], dtype=np.int32),
            "xf": np.array([], dtype=np.float32),
            "pt": np.array([], dtype=np.float32),
            "source_side": np.array([], dtype=np.int8),
            "gen_depth": np.array([], dtype=np.int16),
        },
    ]
    merged = merge_track_chunks(chunks)
    # Mixed presence/absence of optional fields still merges into arrays
    # that all share one length.
    for key in ("pdg", "xf", "pt", "source_side", "weak_feed", "gen_depth"):
        assert merged[key].shape[0] == 3, key
    assert np.array_equal(
        merged["weak_feed"], np.array([0, 0, 1], dtype=np.int8)
    )
    assert np.array_equal(
        merged["gen_depth"], np.array([0, 0, 0], dtype=np.int16)
    )
    assert np.array_equal(merged["pdg"], np.array([1, 2, 3], dtype=np.int32))


def test_merge_track_chunks_rejects_unequal_per_track_lengths() -> None:
    base = {
        "chunk_index": 0,
        "pdg": np.ones(2, dtype=np.int32),
        "xf": np.ones(2, dtype=np.float32),
        "source_side": np.ones(2, dtype=np.int8),
    }
    short_pt = dict(base, pt=np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError, match="'pt'"):
        merge_track_chunks([short_pt])

    long_weak_feed = dict(
        base,
        pt=np.ones(2, dtype=np.float32),
        weak_feed=np.ones(3, dtype=np.int8),  # present but wrong length
    )
    with pytest.raises(ValueError, match="'weak_feed'"):
        merge_track_chunks([long_weak_feed])

    long_gen_depth = dict(
        base,
        pt=np.ones(2, dtype=np.float32),
        gen_depth=np.ones(4, dtype=np.int16),
    )
    with pytest.raises(ValueError, match="'gen_depth'"):
        merge_track_chunks([long_gen_depth])


def test_merge_track_chunks_rejects_missing_required_field() -> None:
    incomplete = {
        "chunk_index": 7,
        "pdg": np.ones(1, dtype=np.int32),
        "xf": np.ones(1, dtype=np.float32),
        "pt": np.ones(1, dtype=np.float32),
        # source_side missing
    }
    with pytest.raises(ValueError, match="source_side"):
        merge_track_chunks([incomplete])


def test_merge_track_chunks_empty_input_returns_aligned_empty_arrays() -> None:
    merged = merge_track_chunks([])
    for key in ("pdg", "xf", "pt", "source_side", "weak_feed", "gen_depth"):
        assert merged[key].shape[0] == 0, key


def test_generate_tracks_chunked_deterministic_across_workers() -> None:
    cfg = GenerationConfig(
        gen_file=_delta_gen_file(),
        n_events=120,
        seed=20260329,
        beam_momentum=158.0,
        max_decay_passes=4,
        use_soft_decay_factory=True,
        fragmentation_mode="both",
    )

    serial = generate_tracks_chunked(cfg, n_workers=1, n_chunks=4)
    parallel = generate_tracks_chunked(cfg, n_workers=2, n_chunks=4)

    assert np.array_equal(serial["pdg"], parallel["pdg"])
    assert np.array_equal(serial["xf"], parallel["xf"])
    assert np.array_equal(serial["pt"], parallel["pt"])
    assert np.array_equal(serial["source_side"], parallel["source_side"])

    assert sum(c["n_events"] for c in serial["chunks"]) == cfg.n_events
    assert sum(c["n_events"] for c in parallel["chunks"]) == cfg.n_events


def test_fragmentation_mode_projectile_target_both() -> None:
    base = dict(
        gen_file=_delta_gen_file(),
        n_events=80,
        seed=20260329,
        beam_momentum=158.0,
        max_decay_passes=1,
        use_soft_decay_factory=False,
    )
    proj = generate_tracks_chunked(GenerationConfig(**base, fragmentation_mode="projectile"), n_workers=1, n_chunks=4)
    targ = generate_tracks_chunked(GenerationConfig(**base, fragmentation_mode="target"), n_workers=1, n_chunks=4)
    both = generate_tracks_chunked(GenerationConfig(**base, fragmentation_mode="both"), n_workers=1, n_chunks=4)

    # No decay chain in this setup: one parent track per side per event.
    assert proj["pdg"].shape[0] == base["n_events"]
    assert targ["pdg"].shape[0] == base["n_events"]
    assert both["pdg"].shape[0] == 2 * base["n_events"]
    assert np.all(proj["source_side"] == 1)
    assert np.all(targ["source_side"] == -1)
    assert np.count_nonzero(both["source_side"] == 1) == base["n_events"]
    assert np.count_nonzero(both["source_side"] == -1) == base["n_events"]

    # Target mode mirrors projectile parent longitudinal momentum in CMS.
    assert np.allclose(np.asarray(targ["xf"]), -np.asarray(proj["xf"]), rtol=0.0, atol=1e-6)


def test_resolve_effective_tower_overrides_supports_mass_ranges() -> None:
    payload = _resolve_effective_tower_overrides(
        {
            "nstar_eff_tower_min_mass": 1.9,
            "nstar_eff_tower_max_mass": 3.1,
            "mestar_eff_tower_min_mass": 0.45,
            "mestar_eff_tower_max_mass": 1.05,
            "nstar_eff_l2_width": 0.42,
            "mestar_eff_l3_width": 0.11,
        }
    )

    # Baryon tower levels follow min/mid/max interpolation.
    assert np.isclose(payload[9911201][0], 1.9)
    assert np.isclose(payload[9921201][0], 2.5)
    assert np.isclose(payload[9931201][0], 3.1)
    # Charged and anti levels mirror the same level mass.
    assert np.isclose(payload[9912201][0], 1.9)
    assert np.isclose(payload[-9931201][0], 3.1)
    # Width override applies to all level-2 baryon IDs.
    assert np.isclose(payload[9921201][1], 0.42)
    assert np.isclose(payload[9922201][1], 0.42)

    # Meson tower levels also follow min/mid/max interpolation.
    assert np.isclose(payload[9900113][0], 0.45)
    assert np.isclose(payload[9900213][0], 0.75)
    assert np.isclose(payload[9900313][0], 1.05)
    # Explicit level width override is honored.
    assert np.isclose(payload[9900313][1], 0.11)


def test_mass_width_overrides_do_not_mutate_pdg_table() -> None:
    base_mass = get_mass(9911201)
    base_width = get_width(9911201)
    with mass_width_overrides({9911201: (2.7, 0.55)}):
        assert np.isclose(get_mass(9911201), 2.7)
        assert np.isclose(get_width(9911201), 0.55)
    assert np.isclose(get_mass(9911201), base_mass)
    assert np.isclose(get_width(9911201), base_width)


def test_tower_mass_override_changes_parent_bw_samples() -> None:
    gen_file = str(
        Path(__file__).resolve().parents[1]
        / "na49_mixing"
        / "generators"
        / "benchmark"
        / "nstar_eff1900_zero.gen"
    )
    cfg0 = GenerationConfig(
        gen_file=gen_file,
        n_events=1,
        seed=1,
        soft_decay_overrides=(),
    )
    cfg1 = GenerationConfig(
        gen_file=gen_file,
        n_events=1,
        seed=1,
        soft_decay_overrides=(("nstar_eff_l1_mass", 2.6), ("nstar_eff_l1_width", 0.15)),
    )
    u0 = _effective_tower_updates(cfg0)
    u1 = _effective_tower_updates(cfg1)

    def _sample_mean(updates) -> float:
        with mass_width_overrides(updates):
            rng = np.random.default_rng(12345)
            gen = Generator(reaction=Reaction(158.0), rng=rng)
            gen.set_all(gen_file)
            _patch_generator_mass_width(gen, updates)
            masses = [float(gen.generate().get_mass()) for _ in range(200)]
        return float(np.mean(masses))

    m0 = _sample_mean(u0)
    m1 = _sample_mean(u1)
    # Peak moves from ~1.9 GeV to ~2.6 GeV; sample mean must shift materially.
    assert m1 > m0 + 0.3


def test_generate_n_matches_count_and_finite_kinematics() -> None:
    gen_file = _delta_gen_file()
    rng = np.random.default_rng(99)
    gen = Generator(reaction=Reaction(158.0), rng=rng)
    gen.set_all(gen_file)
    parts = gen.generate_n(64)
    assert len(parts) == 64
    for p in parts:
        assert np.isfinite(p.get_mass())
        assert np.isfinite(p.get_pt())
        assert np.isfinite(p.get_xf())


def test_generate_tracks_reports_decay_stats() -> None:
    cfg = GenerationConfig(
        gen_file=_delta_gen_file(),
        n_events=40,
        seed=7,
        max_decay_passes=4,
        use_soft_decay_factory=True,
        fragmentation_mode="projectile",
    )
    out = generate_tracks_chunked(cfg, n_workers=1, n_chunks=2)
    assert "n_decayed_parents" in out
    assert "n_decay_kinematic_failures" in out
    assert "n_decay_attempts" in out
    assert int(out["n_decay_attempts"]) >= int(out["n_decayed_parents"])
    assert int(out["n_decayed_parents"]) > 0
    for chunk in out["chunks"]:
        assert "n_decay_kinematic_failures" in chunk


# ---------------------------------------------------------------------------
# p+A wounded-nucleon extension
# ---------------------------------------------------------------------------

def _pb208_spec():
    from na49_mixing.glauber import NucleusSpec

    return NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )


def test_merge_track_chunks_synthesizes_n_wounds_for_legacy_chunks() -> None:
    chunk = {
        "chunk_index": 0,
        "pdg": np.asarray([211, -211], dtype=np.int32),
        "xf": np.zeros(2, dtype=np.float32),
        "pt": np.ones(2, dtype=np.float32),
        "source_side": np.asarray([1, -1], dtype=np.int8),
    }
    merged = merge_track_chunks([chunk])
    for key in ("pdg", "xf", "pt", "source_side", "weak_feed", "gen_depth", "n_wounds"):
        assert merged[key].shape[0] == 2, key
    assert np.all(merged["n_wounds"] == 0)

    bad = dict(chunk, n_wounds=np.zeros(3, dtype=np.int16))
    with pytest.raises(ValueError, match="n_wounds"):
        merge_track_chunks([bad])


def test_pa_generation_minbias_and_centrality_slice() -> None:
    base = dict(
        gen_file=_delta_gen_file(),
        n_events=60,
        seed=20260825,
        beam_momentum=158.0,
        max_decay_passes=1,
        use_soft_decay_factory=False,
        fragmentation_mode="both",
    )
    mb = generate_tracks_chunked(
        GenerationConfig(**base, nucleus=_pb208_spec()),
        n_workers=1,
        n_chunks=3,
    )
    assert mb["pa_nucleus"] is not None and mb["pa_nucleus"]["A"] == 208
    wounds = np.asarray(mb["n_wounds"], dtype=np.int64)
    sides = np.asarray(mb["source_side"], dtype=np.int64)
    assert wounds.min() >= 1
    # Delta+ card without decay files: every emission contributes exactly
    # one track (the undecayed parent). Track count = one projectile
    # track per event plus one target track per wound.
    n_events = 60
    proj_wounds = wounds[sides == 1]
    assert len(proj_wounds) == n_events
    # One undecayed-parent track per emission: projectile entries carry
    # one label per event, so their labels sum to the total wound count.
    assert mb["pdg"].shape[0] == n_events + int(proj_wounds.sum())

    central = generate_tracks_chunked(
        GenerationConfig(**base, nucleus=_pb208_spec(), target_wound_range=(4, 4)),
        n_workers=1,
        n_chunks=3,
    )
    cw = np.asarray(central["n_wounds"], dtype=np.int64)
    assert np.all(cw == 4)
    assert central["pdg"].shape[0] == 4 * n_events + n_events


def test_pa_rare_single_bin_slices_generate_without_rejection_budget() -> None:
    """Pb208 K = 21/24/25: one event each, no attempt cap in sight.

    These single-bin slices carry expected min-bias fractions of
    5.6e-7 down to 3.1e-9; per-event acceptance sampling exhausted any
    finite attempt budget on the first requested event. The
    range-conditioned draw makes the per-event cost independent of the
    slice rarity.
    """

    for k_single in (21, 24, 25):
        out = generate_tracks_chunked(
            GenerationConfig(
                gen_file=_delta_gen_file(),
                n_events=1,
                seed=20260826 + k_single,
                beam_momentum=158.0,
                max_decay_passes=1,
                fragmentation_mode="both",
                nucleus=_pb208_spec(),
                target_wound_range=(k_single, k_single),
            ),
            n_workers=1,
            n_chunks=1,
        )
        assert out["pa_nucleus"]["target_wound_range"] == [k_single, k_single]
        wounds = np.asarray(out["n_wounds"], dtype=np.int64)
        assert wounds.size == 1 + k_single
        assert np.all(wounds == k_single)


def test_pa_point_target_matches_pp_stream_exactly() -> None:
    from na49_mixing.glauber import NucleusSpec

    base = dict(
        gen_file=_delta_gen_file(),
        n_events=50,
        seed=123456789,
        beam_momentum=158.0,
        max_decay_passes=2,
        use_soft_decay_factory=True,
        fragmentation_mode="both",
    )
    pp = generate_tracks_chunked(GenerationConfig(**base), n_workers=1, n_chunks=2)
    pa = generate_tracks_chunked(
        GenerationConfig(**base, nucleus=NucleusSpec(name="p", A=1, Z=1)),
        n_workers=1,
        n_chunks=2,
    )
    for key in ("pdg", "xf", "pt", "source_side"):
        assert np.array_equal(pp[key], pa[key]), key
    # p+p chunks carry zero wounds; the A=1 target carries one per event.
    assert np.all(np.asarray(pa["n_wounds"]) >= 1)


def test_pa_config_rejects_non_both_fragmentation_and_bad_ranges() -> None:
    with pytest.raises(ValueError, match="fragmentation_mode='both'"):
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=1,
            seed=1,
            nucleus=_pb208_spec(),
            fragmentation_mode="projectile",
        )
    with pytest.raises(ValueError, match="1 <= lo <= hi"):
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=1,
            seed=1,
            nucleus=_pb208_spec(),
            fragmentation_mode="both",
            target_wound_range=(0, 3),
        )


# ---------------------------------------------------------------------------
# Chunked export of the mass-conditioned channel audit tables (REV-CUM-01):
# the chunk workers compute the single-process Event path's per-channel audit
# tables but drop them at the TrackChunkResult boundary. generate_tracks_chunked
# must carry the raw accumulators across that boundary and expose the merged
# tables at the same result level, with merge semantics that sum counts and
# re-derive fractions (never average fractions).
# ---------------------------------------------------------------------------


_AUDIT_TABLE_KEYS = (
    "channel_requested",
    "channel_realized",
    "openness_fraction",
    "parent_openness_fraction",
)


def _nstar_gen_file() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "na49_mixing" / "generators" / "benchmark" / "nstar_eff1900_zero.gen"
    )


def _audit_config(n_events: int = 80, seed: int = 555) -> GenerationConfig:
    """Small multi-channel component config for the audit-export tests.

    The N*(1900) benchmark card plus the soft-resonance decay chain yields
    several parent PDGs (the N* primary and its decaying rho daughters) and
    several decay channels per parent, so the audit tables are non-degenerate
    and every merge code path (including absent-not-zero parent openness for
    channels of line-shape-less parents) is exercised.
    """
    return GenerationConfig(
        gen_file=_nstar_gen_file(),
        n_events=n_events,
        seed=seed,
        beam_momentum=158.0,
        max_decay_passes=8,
        use_soft_decay_factory=True,
        fragmentation_mode="projectile",
    )


def _single_process_event_reference(config: GenerationConfig):
    """Run the single-process Event path over one chunk's events.

    Mirrors the chunk worker's construction order and event loop exactly
    (side generator, parent line shapes, decay chain built before the event
    loop, per-event generate + cascade decay), so its ``decay_stats()`` is
    the reference a single-chunk merged audit table must reproduce bit for
    bit. Returns ``(decay_stats, decay_chain)``.
    """
    updates = _effective_tower_updates(config)
    with pdg_database.mass_width_overrides(updates):
        rng = np.random.default_rng(int(_chunk_seed(config.seed, 0)))
        reaction = Reaction(
            float(config.beam_momentum),
            beam_pdg=int(config.beam_pdg),
            target_pdg=int(config.target_pdg),
        )
        gen = _build_side_generator(
            reaction, str(config.gen_file), updates, rng,
            side_pdg=int(reaction.beam_pdg()),
        )
        ev = Event(
            rng=rng,
            resolved_match_mass=_resolved_match_mass_override(config),
            parent_line_shapes=_parent_line_shape_map((gen, None)),
        )
        ev._decay_chain = _build_decay_chain(config, rng)
        ev.reset_decay_stats()
        for _ in range(int(config.n_events)):
            part = gen.generate()
            ev.add_particle(part)
            ev.mark_primary(part)
            if int(config.max_decay_passes) <= 1:
                ev.decay()
            else:
                ev.decay_cascade(max_passes=int(config.max_decay_passes))
            ev.delete()
        return ev.decay_stats(), ev._decay_chain


def test_chunked_result_exports_merged_channel_audit_tables() -> None:
    """A single chunk reproduces the Event path audit tables bit for bit."""
    config = _audit_config()
    out = generate_tracks_chunked(config, n_workers=1, n_chunks=1)

    for key in _AUDIT_TABLE_KEYS:
        assert key in out, f"merged result must expose the {key} audit table"

    ref_stats, chain = _single_process_event_reference(config)

    # Bit-identical to the single-process Event path audit payload: the
    # merge adds 0.0 to every accumulator, so one chunk changes nothing.
    for key in _AUDIT_TABLE_KEYS:
        assert out[key] == ref_stats[key], key

    # Non-degenerate structure: several parent PDGs, each with the channels
    # of the loaded decay table, and a parent-side openness entry for the
    # card's Breit-Wigner primary.
    parents = set(out["channel_requested"])
    assert len(parents) >= 2, parents
    for pdg_id in parents:
        selected = set(out["channel_realized"][pdg_id])
        assert set(out["openness_fraction"][pdg_id]) == selected, pdg_id
        requested = set(out["channel_requested"][pdg_id])
        assert selected <= requested, pdg_id
        assert requested <= set(chain[pdg_id].channel_keys()), pdg_id
        assert out["channel_requested"][pdg_id], pdg_id
    assert out["parent_openness_fraction"], "card carries a BW primary line shape"


def test_chunked_audit_merge_sums_counts_not_averages_fractions() -> None:
    """Merged tables equal fractions of the summed per-chunk raw counts."""
    config = _audit_config(n_events=80, seed=913)
    counts = [c for c in split_event_counts(config.n_events, 2) if c > 0]
    chunks = [
        _generate_tracks_chunk_worker(
            config=config,
            chunk_index=i,
            n_events=c,
            chunk_seed=_chunk_seed(config.seed, i),
        )
        for i, c in enumerate(counts)
    ]
    out = generate_tracks_chunked(config, n_workers=1, n_chunks=2)

    # Sum the raw per-chunk accumulators with plain arithmetic, exactly the
    # documented merge contract (counts sum; fractions are re-derived, never
    # averaged).
    attempts: dict[int, int] = {}
    requested: dict[int, dict] = {}
    realized: dict[int, dict] = {}
    openness: dict[int, dict] = {}
    p_open_sum: dict[int, dict] = {}
    p_open_cnt: dict[int, dict] = {}
    for chunk in chunks:
        acc = chunk["channel_audit_accumulators"]
        for pdg_id, value in acc["channel_attempts"].items():
            attempts[int(pdg_id)] = attempts.get(int(pdg_id), 0) + int(value)
        for field, sink in (
            ("channel_requested_sum", requested),
            ("channel_realized", realized),
            ("channel_openness_sum", openness),
            ("channel_parent_openness_sum", p_open_sum),
            ("channel_parent_openness_count", p_open_cnt),
        ):
            for pdg_id, inner in (acc.get(field) or {}).items():
                tgt = sink.setdefault(int(pdg_id), {})
                for key, value in inner.items():
                    if field in ("channel_realized", "channel_parent_openness_count"):
                        tgt[key] = tgt.get(key, 0) + int(value)
                    else:
                        tgt[key] = tgt.get(key, 0.0) + float(value)

    # Re-derive with the same divisions as the single-process Event path.
    exp_req: dict[int, dict] = {}
    exp_rea: dict[int, dict] = {}
    exp_open: dict[int, dict] = {}
    exp_popen: dict[int, dict] = {}
    for pdg_id, n_attempts in attempts.items():
        denom = float(n_attempts)
        if denom <= 0.0:
            continue
        keys = (
            set(requested.get(pdg_id, {}))
            | set(realized.get(pdg_id, {}))
            | set(openness.get(pdg_id, {}))
            | set(p_open_sum.get(pdg_id, {}))
        )
        r, a, o, po = {}, {}, {}, {}
        for key in keys:
            if key in requested.get(pdg_id, {}):
                r[key] = float(requested[pdg_id][key]) / denom
            if key in realized.get(pdg_id, {}):
                a[key] = float(realized[pdg_id][key]) / denom
            if key in openness.get(pdg_id, {}):
                o[key] = float(openness[pdg_id][key]) / denom
            if key in p_open_sum.get(pdg_id, {}):
                n_sel = int(p_open_cnt.get(pdg_id, {}).get(key, 0))
                if n_sel > 0:
                    po[key] = float(p_open_sum[pdg_id][key]) / float(n_sel)
        exp_req[pdg_id], exp_rea[pdg_id], exp_open[pdg_id] = r, a, o
        if po:
            exp_popen[pdg_id] = po

    assert out["channel_requested"] == exp_req
    assert out["channel_realized"] == exp_rea
    assert out["openness_fraction"] == exp_open
    assert out["parent_openness_fraction"] == exp_popen
    assert len(exp_req) >= 2
    assert any(len(tbl) >= 2 for tbl in exp_req.values())


def test_chunked_audit_tables_robust_to_legacy_chunks() -> None:
    """A chunk without the audit payload contributes nothing, not an error.

    Strips the payload from one chunk (simulating a pre-REV-CUM-01 legacy
    chunk) and checks that the audit merge and the track merge both
    tolerate the missing field.
    """
    config = _audit_config(n_events=40, seed=31)
    chunk = _generate_tracks_chunk_worker(
        config=config, chunk_index=0, n_events=40,
        chunk_seed=_chunk_seed(config.seed, 0),
    )
    full = chunk["channel_audit_accumulators"]
    assert full["channel_attempts"], "audit payload must be non-empty"
    legacy = {k: v for k, v in chunk.items() if k != "channel_audit_accumulators"}

    # The legacy chunk contributes nothing to the audit accumulators.
    mixed = _merge_channel_audit_accumulators([legacy, chunk])
    assert mixed == _merge_channel_audit_accumulators([chunk])
    # And the derived tables are exactly the single-chunk tables.
    assert _derive_channel_audit_tables(mixed) == _derive_channel_audit_tables(full)

    # Track merging is unaffected by the missing payload (chunk_index 0
    # for both; stable sort keeps the input order).
    merged = merge_track_chunks([chunk, legacy])
    assert np.array_equal(merged["pdg"], np.concatenate([
        np.asarray(chunk["pdg"], dtype=np.int64),
        np.asarray(legacy["pdg"], dtype=np.int64),
    ]))
