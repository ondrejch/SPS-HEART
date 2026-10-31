"""Tests for resonance cache worker resolution and batched scheduling."""

from __future__ import annotations

import json
from pathlib import Path

from helpers.benchmarking import benchmark_sps_species as bss


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mixing2006_dir() -> Path:
    return _repo_root() / "dev" / "mixing2006"


def _delta_gen_file() -> Path:
    return _mixing2006_dir() / "na49_mixing" / "generators" / "michal" / "deltas-to-pion" / "delta++.gen"


def test_resolve_resonance_worker_count_uses_gateway_reserved_tasks(monkeypatch) -> None:
    monkeypatch.setenv("NA49_CLUSTER_GW_TASKS", "24")
    assert bss._resolve_resonance_worker_count(1) == 24
    assert bss._resolve_resonance_worker_count(12) == 12
    assert bss._resolve_resonance_worker_count(64) == 24


def test_split_resonance_batches_covers_all_tasks() -> None:
    tasks = [{"spec_key": f"spec-{i}"} for i in range(23)]
    batches = bss._split_resonance_batches(tasks, n_workers=4)
    flat = [item["spec_key"] for batch in batches for item in batch]
    assert flat == [f"spec-{i}" for i in range(23)]
    assert max(len(batch) for batch in batches) <= bss._auto_resonance_batch_size(23, 4)


def test_build_resonance_cache_uses_batched_pool_with_gateway_tasks(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NA49_CLUSTER_GW_TASKS", "2")

    mixing2006_dir = _mixing2006_dir()
    gen_file = _delta_gen_file()
    specs = {
        f"spec-{i}::{gen_file.name}": bss.ResonanceSpec(
            label=f"spec_{i}",
            gen_file=gen_file,
            use_soft_decay_factory=True,
        )
        for i in range(3)
    }
    cache_dir = tmp_path / "res_cache"

    out = bss._build_resonance_component_cache(
        mixing2006_dir=mixing2006_dir,
        cache_dir=cache_dir,
        specs_by_key=specs,
        n_events_per_res=10,
        seed=20260330,
        max_decay_passes=4,
        fragmentation_mode="projectile",
        force_rebuild=True,
        res_workers=1,
        soft_decay_overrides=None,
    )

    assert set(out.keys()) == set(specs.keys())
    for cache_path in out.values():
        assert cache_path.exists()

    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    assert int(manifest["res_workers_requested"]) == 1
    assert int(manifest["res_workers_effective"]) == 2
    assert int(manifest["gateway_reserved_tasks"]) == 2
    assert manifest["schedule_mode"] == "batched_pool"
    assert int(manifest["schedule_batch_count"]) >= 2

