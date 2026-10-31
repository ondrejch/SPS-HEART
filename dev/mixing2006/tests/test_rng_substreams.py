"""Deterministic tests for the RNG substream labeling (Round 1, card 5).

Plan section 6, task 5.  The pinned derivation table fixes the exact
``(master_seed, replica_index, event_index) -> streams`` mapping so any
future change to the derivation contract fails loudly.  The card's
feature is additive: when ``substreams.enabled`` is false (the legacy
default) the generation stream is bit-identical, asserted here via
event hashes of a 20-event ``PPEventBuilder`` build through the staged
smoke configuration.

Hermetic: no external tables; the staged smoke run below (20 events per
call, fit disabled) and the small bundled-card chunked runs in
``test_parallel_worker_manifest_additive`` are the only generation.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.glauber import NucleusSpec
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked
from na49_mixing.rng_substreams import (
    SUBSTREAM_LABELS,
    derive_substreams,
    substream_manifest_path,
    write_substream_manifest,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]

# Pinned derivation table (Round 1, card 5): five
# (master_seed, replica_index, event_index) -> stream id per label rows.
# Row values are the verbatim output of the SeedSequence spawn-chain
# contract documented in na49_mixing/rng_substreams.py; a change to the
# derivation (or to numpy's SeedSequence mixing) breaks this table.
PINNED_TABLE = [
    (
        20260831, 0, 0,
        [8615163616252788064, 14565340020608427291, 9904591513891838653,
         2392443123840094210, 14579774015383114706, 15056116069979542969,
         1825084508462397565],
    ),
    (
        20260831, 3, 17,
        [4999193161780055592, 15508480181631070499, 6481652778568646254,
         7260572705396731934, 5456666224609135427, 11777720112484403395,
         9104659688591761016],
    ),
    (
        1, 0, 1,
        [1044185198228299392, 3558795683043373558, 23650600771195422,
         5962497758155097386, 17001961675987965441, 655892175236301587,
         15633112903042822845],
    ),
    (
        7, 12, 345,
        [4812608840667394421, 4900567387100175985, 15459068893593001568,
         8363958293393795653, 1337963821020442534, 5888823153112430797,
         4602224263468941555],
    ),
    (
        2147483653, 9, 99,
        [13664014722412355096, 14311912194502637282, 12349681095173522965,
         10359547016193280233, 14947960173179448378, 3458499904431851432,
         16823207147404799298],
    ),
]


# ---------------------------------------------------------------------------
# test_derive_deterministic
# ---------------------------------------------------------------------------


def test_derive_deterministic() -> None:
    """The derived streams match the pinned table exactly, row by row."""

    assert len(PINNED_TABLE) == 5
    for master_seed, replica_index, event_index, expected in PINNED_TABLE:
        subs = derive_substreams(master_seed, replica_index, event_index)
        assert subs.master_seed == master_seed
        assert subs.replica_index == replica_index
        assert subs.event_index == event_index
        assert list(subs.streams.keys()) == list(SUBSTREAM_LABELS)
        assert [subs.streams[label] for label in SUBSTREAM_LABELS] == expected
        # Repeated derivation in the same process is identical.
        again = derive_substreams(master_seed, replica_index, event_index)
        assert again.streams == subs.streams


def test_derive_matches_literal_spawn_chain() -> None:
    """The one-step spawn_key form equals the literal fresh-object chain.

    This pins the equivalence claimed in the module docstring: the
    derivation is ``SeedSequence(master).spawn(r+1)[r].spawn(e+1)[e]
    .spawn(len(SUBSTREAM_LABELS))`` with each intermediate sequence
    constructed fresh.
    """

    import numpy as np

    for master_seed, replica_index, event_index, expected in PINNED_TABLE:
        literal = (
            np.random.SeedSequence(master_seed)
            .spawn(replica_index + 1)[replica_index]
            .spawn(event_index + 1)[event_index]
        )
        literal_ids = []
        for child in literal.spawn(len(SUBSTREAM_LABELS)):
            state = child.generate_state(2, dtype=np.uint32)
            literal_ids.append((int(state[0]) << 32) | int(state[1]))
        assert literal_ids == expected


# ---------------------------------------------------------------------------
# test_derive_cross_process
# ---------------------------------------------------------------------------


def _derive_row(row) -> tuple:
    """Spawn-pool target: derive one table row in a worker process."""

    from tests.test_rng_substreams import SUBSTREAM_LABELS as _LABELS

    master_seed, replica_index, event_index, _expected = row
    subs = derive_substreams(master_seed, replica_index, event_index)
    return (
        subs.master_seed,
        subs.replica_index,
        subs.event_index,
        tuple(subs.streams[label] for label in _LABELS),
    )


def test_derive_cross_process() -> None:
    """Derivation in fresh subprocesses matches the in-process values.

    The spawn context starts a new interpreter per worker, so the check
    rules out accidental dependence on import order or module state.
    """

    ctx = multiprocessing.get_context("spawn")
    rows = PINNED_TABLE
    local = [_derive_row(row) for row in rows]
    with ctx.Pool(2) as pool:
        remote = pool.map(_derive_row, rows)
    assert remote == local


# ---------------------------------------------------------------------------
# test_derive_non_overlap
# ---------------------------------------------------------------------------


def test_derive_non_overlap() -> None:
    """For a fixed (master_seed, replica_index), the 7 labels are distinct.

    Every event of the sampled block carries seven distinct stream ids;
    no two labels share a stream id for the same event.
    """

    master_seed, replica_index = 20260831, 4
    for event_index in range(32):
        subs = derive_substreams(master_seed, replica_index, event_index)
        values = list(subs.streams.values())
        assert len(values) == len(SUBSTREAM_LABELS) == 7
        assert len(set(values)) == 7, (master_seed, replica_index, event_index)
        # Cross-event identity of one label never collides with another
        # label's id within the same event (already asserted) and the
        # (replica, event, label) triple is reproducible:
        assert derive_substreams(master_seed, replica_index, event_index).streams == subs.streams


# ---------------------------------------------------------------------------
# test_stream_for_rejects_unknown_label
# ---------------------------------------------------------------------------


def test_stream_for_rejects_unknown_label() -> None:
    """The label set is closed: unknown labels raise ``KeyError``."""

    subs = derive_substreams(20260831, 0, 0)
    for bad in ("parton", "string_fragmentation"):
        with pytest.raises(KeyError):
            subs.stream_for(bad)


# ---------------------------------------------------------------------------
# test_legacy_bit_identity_with_feature_off
# ---------------------------------------------------------------------------


def _smoke_config(substreams_block: dict | None) -> dict:
    """The bundled smoke configuration, PPEventBuilder stage only.

    ``fit.enabled`` is false so no SPS tables are needed; ``events_per_stage``
    gives the 20-event builder call.  ``substreams_block`` is injected (or
    omitted entirely for the legacy default) as the additive config key.
    """

    config_path = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["through"] = "simple_excitation"
    config["fit"] = {**config.get("fit", {}), "enabled": False}
    if substreams_block is not None:
        config["substreams"] = substreams_block
    return config


def _stage_event_hashes(samples: dict) -> list[str]:
    return [event.as_dict()["event_hash"] for event in samples["simple_excitation"]]


def test_legacy_bit_identity_with_feature_off() -> None:
    """The feature is additive: event hashes are identical in all modes.

    Three staged smoke runs share one seed: the legacy default (no
    ``substreams`` key at all), the feature explicitly off, and the
    feature explicitly on.  The 20 PPEventBuilder event hashes are
    bit-identical across the three runs; the run metadata record gains
    the ``substreams`` block only when the feature is enabled.
    """

    from na49_mixing.staged_evaluation import run_staged_evaluation

    legacy_report, legacy_samples = run_staged_evaluation(_smoke_config(None))
    off_report, off_samples = run_staged_evaluation(_smoke_config({"enabled": False}))
    on_report, on_samples = run_staged_evaluation(_smoke_config({"enabled": True}))

    assert len(_stage_event_hashes(legacy_samples)) == 20
    assert (
        _stage_event_hashes(legacy_samples)
        == _stage_event_hashes(off_samples)
        == _stage_event_hashes(on_samples)
    )

    assert "substreams" not in legacy_report
    assert "substreams" not in off_report
    assert on_report["substreams"] == {
        "master_seed": 20260831,
        "replica_seed_rule": on_report["substreams"]["replica_seed_rule"],
        "substream_labels": list(SUBSTREAM_LABELS),
    }


# ---------------------------------------------------------------------------
# test_write_substream_manifest_roundtrip
# ---------------------------------------------------------------------------


def test_write_substream_manifest_roundtrip(tmp_path) -> None:
    """A written manifest reads back with identical contents."""

    events = []
    for event_index in range(4):
        subs = derive_substreams(20260831, 0, event_index)
        events.append(
            {
                "event_index": event_index,
                "streams": dict(subs.streams),
                "draws": [
                    {
                        "label": "topology_selection",
                        "stream_id": subs.stream_for("topology_selection"),
                        "n_draws": 3 + event_index,
                    }
                ],
            }
        )

    manifest = substream_manifest_path(str(tmp_path), 0)
    assert manifest.endswith("substream_manifest_replica_0000.json")
    write_substream_manifest(manifest, events)

    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert payload["schema"] == "na49_mixing.substream_manifest/1"
    assert payload["substream_labels"] == list(SUBSTREAM_LABELS)
    assert len(payload["events"]) == len(events)
    for written, original in zip(payload["events"], events):
        assert written["event_index"] == original["event_index"]
        assert written["streams"] == original["streams"]
        assert written["draws"] == original["draws"]


# ---------------------------------------------------------------------------
# test_parallel_worker_manifest_additive (hook coverage)
# ---------------------------------------------------------------------------


def _delta_gen_file() -> str:
    return str(
        NESTED_ROOT / "na49_mixing" / "generators" / "michal" / "deltas-to-pion" / "delta++.gen"
    )


def test_parallel_worker_manifest_additive(tmp_path) -> None:
    """The parallel.py manifest hook is additive and records per-event ids.

    With ``substream_manifest_path`` set the merged generation arrays are
    bit-identical to the reference run without the hook, and each chunk
    worker (one manifest per replica) writes the per-event stream ids of
    ``derive_substreams(seed, chunk_index, event_index)``.  Draw counts
    are recorded, not enforced, by this card, so ``draws`` is empty.
    """

    from na49_mixing.rng_substreams import MANIFEST_SCHEMA

    seed = 20260911
    reference = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=12,
            seed=seed,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
        ),
        n_workers=1,
        n_chunks=2,
    )

    run_dir = tmp_path / "substreams"
    with_manifest = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=12,
            seed=seed,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
            substream_manifest_path=str(run_dir),
        ),
        n_workers=1,
        n_chunks=2,
    )
    for key in ("pdg", "xf", "pt", "source_side"):
        assert np.array_equal(reference[key], with_manifest[key]), key

    chunk_counts = {c["chunk_index"]: c["n_events"] for c in with_manifest["chunks"]}
    for chunk_index in sorted(chunk_counts):
        path = Path(substream_manifest_path(str(run_dir), chunk_index))
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == MANIFEST_SCHEMA
        assert payload["substream_labels"] == list(SUBSTREAM_LABELS)
        assert len(payload["events"]) == chunk_counts[chunk_index]
        for record in payload["events"]:
            subs = derive_substreams(seed, chunk_index, record["event_index"])
            assert record["streams"] == dict(subs.streams)
            assert record["draws"] == []

    # The p+A chunk worker records the same contract (point target: no
    # sampler draws, so the run is cheap and deterministic).
    pa_run_dir = tmp_path / "pa_substreams"
    pa_out = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=12,
            seed=seed,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
            nucleus=NucleusSpec(name="p", A=1, Z=1),
            substream_manifest_path=str(pa_run_dir),
        ),
        n_workers=1,
        n_chunks=2,
    )
    for chunk_index in sorted(c["chunk_index"] for c in pa_out["chunks"]):
        path = Path(substream_manifest_path(str(pa_run_dir), chunk_index))
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["events"]) == 6
