"""Round-2 generation tests: weak-feed policy hook and primary-parent wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked


def _nstar_gen_file() -> str:
    base = (
        Path(__file__).resolve().parents[1]
        / "na49_mixing"
        / "generators"
        / "michal"
        / "nstar-to-proton"
    )
    return str(base / "n1440-zero.gen")


def test_generation_config_rejects_unknown_weak_feed_policy() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(
            gen_file=_nstar_gen_file(),
            n_events=1,
            seed=1,
            weak_feed_policy="drop_everything",
        )


def test_weak_feed_policy_primary_only_drops_weak_ancestry_tracks() -> None:
    gen_file = (
        Path(__file__).resolve().parents[1]
        / "na49_mixing"
        / "generators"
        / "benchmark"
        / "nstar_eff1900_zero.gen"
    )
    base = dict(
        gen_file=str(gen_file),
        n_events=100,
        seed=3,
        beam_momentum=158.0,
        max_decay_passes=8,
        use_soft_decay_factory=True,
        fragmentation_mode="both",
    )
    inclusive = generate_tracks_chunked(
        GenerationConfig(**base, weak_feed_policy="inclusive_weak"),
        n_workers=1,
        n_chunks=2,
    )
    primary_only = generate_tracks_chunked(
        GenerationConfig(**base, weak_feed_policy="primary_only"),
        n_workers=1,
        n_chunks=2,
    )

    # The effective-tower cascade feeds Lambda/K0S daughters, so the inclusive
    # sample carries genuinely weak-fed tracks for this card and seed.
    inc_weak_feed = np.asarray(inclusive["weak_feed"], dtype=np.int8)
    assert int(np.count_nonzero(inc_weak_feed)) == int(inclusive["n_weak_feed_tracks"])
    assert int(inclusive["n_weak_feed_tracks"]) > 0

    keep = inc_weak_feed == 0
    dropped = int(primary_only["n_weak_dropped"])
    assert dropped == int(np.count_nonzero(inc_weak_feed))
    assert int(np.count_nonzero(primary_only["weak_feed"])) == 0
    assert np.array_equal(primary_only["pdg"], np.asarray(inclusive["pdg"])[keep])
    assert np.array_equal(primary_only["gen_depth"], np.asarray(inclusive["gen_depth"])[keep])


def test_weak_feed_policy_primary_only_end_to_end() -> None:
    base = dict(
        gen_file=_nstar_gen_file(),
        n_events=50,
        seed=20260823,
        beam_momentum=158.0,
        max_decay_passes=8,
        use_soft_decay_factory=True,
        fragmentation_mode="both",
    )
    inclusive = generate_tracks_chunked(
        GenerationConfig(**base, weak_feed_policy="inclusive_weak"),
        n_workers=1,
        n_chunks=2,
    )
    primary_only = generate_tracks_chunked(
        GenerationConfig(**base, weak_feed_policy="primary_only"),
        n_workers=1,
        n_chunks=2,
    )

    assert inclusive["weak_feed_policy"] == "inclusive_weak"
    assert primary_only["weak_feed_policy"] == "primary_only"

    n_inclusive = int(inclusive["pdg"].shape[0])
    n_primary_only = int(primary_only["pdg"].shape[0])
    assert n_primary_only <= n_inclusive

    # Identical seeds reproduce identical physics; primary_only only filters.
    inc_weak_feed = np.asarray(inclusive["weak_feed"], dtype=np.int8)
    keep = inc_weak_feed == 0
    dropped = int(primary_only["n_weak_dropped"])
    assert dropped == n_inclusive - int(np.count_nonzero(keep))
    assert (dropped > 0) == (bool(np.count_nonzero(inc_weak_feed)) > 0)
    assert int(np.count_nonzero(primary_only["weak_feed"])) == 0
    assert np.array_equal(primary_only["pdg"], np.asarray(inclusive["pdg"])[keep])
    assert np.array_equal(primary_only["xf"], np.asarray(inclusive["xf"])[keep])
    assert np.array_equal(primary_only["pt"], np.asarray(inclusive["pt"])[keep])

    # One registered primary parent per generated side (projectile + target).
    assert int(primary_only["n_primary_generated"]) == base["n_events"] * 2
    assert int(primary_only["primary_failed"]) == 0
    total_registered = sum(int(pair[0]) for pair in primary_only["primary_pdg_stats"].values())
    total_decayed = sum(int(pair[1]) for pair in primary_only["primary_pdg_stats"].values())
    assert total_registered == base["n_events"] * 2
    assert total_decayed == base["n_events"] * 2
