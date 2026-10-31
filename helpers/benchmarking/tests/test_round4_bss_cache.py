"""Round-4 cache-safety regression for benchmark_sps_species (bss).

Covers the source-fingerprint cache guard added on 2026-08-23:

1. ``_decay_lib_fingerprint`` hashes the imported na49_mixing decay-library
   and mass/pT-sampling sources (``decays``, ``parallel``, ``generator``,
   ``breit_wigner``, ``histogram``, ``glauber``, ``event``, ``isospin``,
   ``reaction``, ``pdg_database``), the spec's generator-card bytes, the
   sort-keys JSON of the soft-decay overrides, beam/target PDG ids, and
   the resolved effective-tower mass/width update set; it reacts to each
   input.
2. Component caches carrying a wrong fingerprint, and caches written before
   fingerprint tagging (field absent), mismatch in ``_cache_config_matches``
   and are rebuilt by the reuse path instead of being silently reused.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.benchmarking import benchmark_sps_species as bss

# Repo root derived from the imported module itself, so the path stays
# correct regardless of how pytest mangles test-module __file__.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(bss.__file__), "..", "..")
)
_MIXING2006_DIR = os.path.join(_REPO_ROOT, "dev", "mixing2006")
_GEN_CARD_1900 = os.path.join(
    _MIXING2006_DIR, "na49_mixing", "generators", "benchmark",
    "nstar_eff1900_zero.gen",
)
_GEN_CARD_2300 = os.path.join(
    _MIXING2006_DIR, "na49_mixing", "generators", "benchmark",
    "nstar_eff2300_zero.gen",
)


def _fingerprint(**kw):
    kw.setdefault("gen_file", _GEN_CARD_1900)
    kw.setdefault("soft_decay_overrides", {})
    kw.setdefault("mixing2006_dir", _MIXING2006_DIR)
    return bss._decay_lib_fingerprint(**kw)


def test_fingerprint_hashes_all_declared_inputs():
    base = _fingerprint()
    assert base is not None and len(base) == 64

    # Deterministic for identical inputs.
    assert _fingerprint() == base

    # Generator-card bytes enter the hash.
    assert _fingerprint(gen_file=_GEN_CARD_2300) != base

    # Soft-decay branch overrides enter the hash.
    assert _fingerprint(soft_decay_overrides={"nstar_eff_l1_br_nrho": 0.10}) != base

    # Overrides that change the resolved effective-tower mass/width update
    # set change the fingerprint too.
    assert _fingerprint(soft_decay_overrides={"nstar_eff_l1_mass": 2.05}) != base


def test_fingerprint_reacts_to_mass_sampling_module_bytes(tmp_path, monkeypatch):
    """generator/breit_wigner/histogram sources enter the hash.

    A stub module replaces na49_mixing.breit_wigner in sys.modules so the
    fingerprint hashes sentinel bytes from tmp_path instead of the real
    source file; touching those bytes must move the digest.
    """
    base = _fingerprint()

    src = tmp_path / "breit_wigner.py"
    payload = b"# mass-law source sentinel\n"
    src.write_bytes(payload)
    stub = types.ModuleType("na49_mixing.breit_wigner")
    stub.__file__ = str(src)
    monkeypatch.setitem(sys.modules, "na49_mixing.breit_wigner", stub)

    first = _fingerprint()
    assert first is not None and first != base

    src.write_bytes(payload + b"# touched\n")
    assert _fingerprint() != first


def test_fingerprint_reacts_to_event_and_isospin_bytes(tmp_path, monkeypatch):
    base = _fingerprint()
    for modname in ("na49_mixing.event", "na49_mixing.isospin"):
        src = tmp_path / (modname.split(".")[-1] + ".py")
        src.write_bytes(b"# sentinel\n")
        stub = types.ModuleType(modname)
        stub.__file__ = str(src)
        monkeypatch.setitem(sys.modules, modname, stub)
        moved = _fingerprint()
        assert moved is not None and moved != base
        src.write_bytes(b"# sentinel\n# touched\n")
        assert _fingerprint() != moved
        monkeypatch.undo()
        base = _fingerprint()


def test_uninspectable_library_yields_none(tmp_path):
    missing_card = tmp_path / "does_not_exist.gen"
    assert _fingerprint(gen_file=missing_card) is None


def _write_component_npz(path, fingerprint, *, spec_key="spec::tiny",
                         component_seed=11, n_events_per_res=3,
                         max_decay_passes=8, fragmentation_mode="projectile",
                         weak_feed_policy="primary_only"):
    """Fabricate a component cache with every field the reuse path reads."""
    payload = dict(
        pdg=np.zeros(0, dtype=np.int32),
        xf=np.zeros(0, dtype=np.float32),
        pt=np.zeros(0, dtype=np.float32),
        source_side=np.zeros(0, dtype=np.int8),
        weak_feed=np.zeros(0, dtype=np.int8),
        gen_depth=np.zeros(0, dtype=np.int16),
        meta_label=np.asarray("tiny"),
        meta_spec_key=np.asarray(str(spec_key)),
        meta_decay_lib_version=np.asarray(bss.RESONANCE_DECAY_LIB_VERSION),
        meta_component_seed=np.asarray(int(component_seed), dtype=np.int64),
        meta_n_events_per_res=np.asarray(int(n_events_per_res), dtype=np.int64),
        meta_max_decay_passes=np.asarray(int(max_decay_passes), dtype=np.int64),
        meta_fragmentation_mode=np.asarray(str(fragmentation_mode)),
        meta_soft_decay_overrides_json=np.asarray("{}"),
        meta_weak_feed_policy=np.asarray(str(weak_feed_policy)),
    )
    if fingerprint is not None:
        payload["meta_decay_lib_fingerprint"] = np.asarray(fingerprint)
    np.savez_compressed(path, **payload)


_MATCH_KW = dict(
    spec_key="spec::tiny",
    component_seed=11,
    n_events_per_res=3,
    max_decay_passes=8,
    fragmentation_mode="projectile",
    soft_decay_overrides_json="{}",
    weak_feed_policy="primary_only",
)


def test_cache_config_requires_matching_fingerprint(tmp_path):
    fp = _fingerprint()
    path = tmp_path / "component.npz"

    # Matching fingerprint reuses.
    _write_component_npz(path, fingerprint=fp)
    assert bss._cache_config_matches(
        path, decay_lib_fingerprint=fp, **_MATCH_KW)

    # Wrong fingerprint mismatches and triggers a rebuild.
    _write_component_npz(path, fingerprint="f" * 64)
    assert not bss._cache_config_matches(
        path, decay_lib_fingerprint=fp, **_MATCH_KW)

    # Absent fingerprint (pre-tagging cache layout) mismatches as well.
    _write_component_npz(path, fingerprint=None)
    assert not bss._cache_config_matches(
        path, decay_lib_fingerprint=fp, **_MATCH_KW)

    # An uninspectable current library can never validate a cache.
    assert not bss._cache_config_matches(
        path, decay_lib_fingerprint=None, **_MATCH_KW)


def test_reuse_path_rebuilds_on_fingerprint_mismatch(tmp_path, monkeypatch):
    """End-to-end: the scheduler rebuilds exactly when fingerprints fail."""
    build_calls = []

    def fake_builder(*, mixing2006_dir, task, n_events_per_res,
                     max_decay_passes, fragmentation_mode, chunk_workers=1):
        build_calls.append(str(task["cache_file"]))
        _write_component_npz(
            task["cache_file"],
            str(task.get("decay_lib_fingerprint", "")),
            spec_key=str(task["spec_key"]),
            component_seed=int(task["component_seed"]),
            n_events_per_res=int(n_events_per_res),
            max_decay_passes=int(max_decay_passes),
            fragmentation_mode=str(fragmentation_mode),
            weak_feed_policy=str(task.get("weak_feed_policy", "inclusive_weak")),
        )
        return {
            "ok": True,
            "spec_key": str(task["spec_key"]),
            "label": str(task["label"]),
            "cache_file": str(task["cache_file"]),
        }

    monkeypatch.setattr(
        bss, "_build_resonance_component_cache_one", fake_builder)
    specs_by_key = {
        "spec::tiny": bss.ResonanceSpec(
            label="tiny", gen_file=Path(_GEN_CARD_1900),
            use_soft_decay_factory=True,
        ),
    }
    common = dict(
        mixing2006_dir=Path(_MIXING2006_DIR),
        cache_dir=tmp_path,
        specs_by_key=specs_by_key,
        n_events_per_res=3,
        seed=5,
        max_decay_passes=8,
        fragmentation_mode="projectile",
        force_rebuild=False,
        res_workers=1,
        weak_feed_policy="primary_only",
    )

    out1 = bss._build_resonance_component_cache(**common)
    assert len(build_calls) == 1, "cold cache must build once"
    npz_path = Path(out1["spec::tiny"])
    assert npz_path.exists()

    out2 = bss._build_resonance_component_cache(**common)
    assert len(build_calls) == 1, "intact fingerprint must be reused"
    assert out2 == out1

    # Hand-corrupt the stored fingerprint: the reuse path must rebuild.
    with np.load(npz_path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    fields["meta_decay_lib_fingerprint"] = np.asarray("0" * 64)
    np.savez_compressed(npz_path, **fields)
    out3 = bss._build_resonance_component_cache(**common)
    assert len(build_calls) == 2, "wrong fingerprint must trigger a rebuild"
    assert Path(out3["spec::tiny"]) == npz_path

    # A rebuilt cache that lacks the fingerprint field entirely (the
    # pre-tagging layout) must also be rebuilt, not reused.
    with np.load(npz_path, allow_pickle=False) as z:
        fields = {
            k: z[k] for k in z.files if k != "meta_decay_lib_fingerprint"
        }
    np.savez_compressed(npz_path, **fields)
    out4 = bss._build_resonance_component_cache(**common)
    assert len(build_calls) == 3, "absent fingerprint must trigger a rebuild"
    assert Path(out4["spec::tiny"]) == npz_path
