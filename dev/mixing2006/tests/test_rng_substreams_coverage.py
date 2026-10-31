"""Coverage extension for the RNG substream labeling (Round 1, card 5).

Plan section 6, task 5.  ``tests/test_rng_substreams.py`` holds the six
card-named deterministic tests plus the spawn-chain equivalence pin and
the parallel-worker manifest hook test.  This file extends coverage of
the same committed behavior without touching that file:

- the closed-set assertion of ``SUBSTREAM_LABELS`` (a ``tuple`` with
  exactly the seven plan section 6, task 5 labels in order) and the
  identity of the additive package exports;
- derivation invariants beyond the pinned table: reproducibility, the
  strict Python-int type and 64-bit range of the stream ids, exact
  key ordering of ``streams``, NumPy-integer acceptance, argument
  validation (``bool``, negative, float, string, ``None`` rejected),
  distinct stream-id sets across events and replicas, and the literal
  spawn-chain equivalence for a master seed beyond the 32-bit range;
- ``RngSubstreams`` immutability (``dataclasses.FrozenInstanceError``)
  and ``stream_for`` against all seven labels plus further unknown-label
  cases (empty string, case variant);
- the ``substream_manifest_path`` naming contract (zero-padded replica
  index under the run directory) and the manifest writer's JSON schema,
  canonical key order, int coercion, and byte-deterministic
  serialization;
- the complementary parallel-path angle the committed hook test leaves
  open: a p+A chunk run with ``substream_manifest_path`` set is
  bit-identical to the same run without the hook across the *entire*
  merged output (the committed test compares four track arrays on the
  p+p path and only checks manifest presence on the p+A path), with the
  per-chunk manifest stream ids equal to
  ``derive_substreams(seed, chunk_index, event_index)``;
- the staged run-metadata ``substreams`` block gating: the block is
  absent for the legacy default *and* for an empty ``substreams`` block
  (``enabled`` defaults to false), carries exactly the three card keys
  when enabled, records the config seed and the module's replica-seed
  rule string verbatim, and leaves the 20-event PPEventBuilder event
  hashes untouched.

The cross-process determinism direction is not duplicated here:
``tests/test_rng_substreams.py::test_derive_cross_process`` already
derives all five pinned rows in fresh spawn-context subprocesses and
compares them against the in-process values.

Hermetic: no external tables; the generation below is one 12-event p+A
chunked run (point target, no sampler draws) per direction and two
20-event staged smoke builds, all with the bundled smoke configuration
(``config/staged_excitation_smoke.json``, ``fit.enabled`` false).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from na49_mixing.glauber import NucleusSpec
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked
from na49_mixing.rng_substreams import (
    MANIFEST_SCHEMA,
    SUBSTREAM_LABELS,
    RngSubstreams,
    derive_substreams,
    substream_manifest_path,
    write_substream_manifest,
)
from tests.test_rng_substreams import PINNED_TABLE

NESTED_ROOT = Path(__file__).resolve().parents[1]

# The seven per-track arrays of the merged chunked generation output
# (na49_mixing.parallel._TRACK_ARRAY_DTYPES); everything else in the
# merged result is scalar/metadata and compares by ``==``.
_TRACK_ARRAY_KEYS = (
    "pdg",
    "xf",
    "pt",
    "source_side",
    "weak_feed",
    "gen_depth",
    "n_wounds",
)


# ---------------------------------------------------------------------------
# Item 1: closed-set assertion
# ---------------------------------------------------------------------------


def test_substream_labels_closed_set() -> None:
    """``SUBSTREAM_LABELS`` is the closed seven-label set, in card order.

    The set is a ``tuple`` of exactly the seven plan section 6, task 5
    physical layers verbatim; no eighth label is reachable.  The
    additive package exports are the same objects, not copies.
    """

    import na49_mixing

    assert type(SUBSTREAM_LABELS) is tuple
    assert len(SUBSTREAM_LABELS) == 7
    assert list(SUBSTREAM_LABELS) == [
        "topology_selection",
        "exchange_kinematics",
        "spectral_state",
        "decay_channel",
        "decay_kinematics",
        "nuclear_geometry",
        "transport",
    ]
    assert len(set(SUBSTREAM_LABELS)) == 7, "the label set must have no duplicates"

    assert na49_mixing.SUBSTREAM_LABELS is SUBSTREAM_LABELS
    assert na49_mixing.derive_substreams is derive_substreams
    assert na49_mixing.RngSubstreams is RngSubstreams
    assert na49_mixing.substream_manifest_path is substream_manifest_path
    assert na49_mixing.write_substream_manifest is write_substream_manifest
    for name in (
        "SUBSTREAM_LABELS",
        "RngSubstreams",
        "derive_substreams",
        "substream_manifest_path",
        "write_substream_manifest",
    ):
        assert name in na49_mixing.__all__


# ---------------------------------------------------------------------------
# Item 2: derivation invariants beyond the pinned table
# ---------------------------------------------------------------------------


def test_derive_reproducible_typed_and_keyed() -> None:
    """Rows outside the pinned table: reproducible, typed, exact keys.

    Same ``(master_seed, replica_index, event_index)`` always yields the
    same streams; stream ids are strict Python ints within the 64-bit
    range; ``streams`` carries exactly one entry per label with keys in
    ``SUBSTREAM_LABELS`` order.
    """

    for master_seed, replica_index, event_index in (
        (20260831, 0, 0),
        (12345678, 2, 9),
        (99, 0, 0),
        (2**40 + 7, 5, 5),
    ):
        subs = derive_substreams(master_seed, replica_index, event_index)
        again = derive_substreams(master_seed, replica_index, event_index)
        assert subs.streams == again.streams
        assert subs.master_seed == master_seed
        assert subs.replica_index == replica_index
        assert subs.event_index == event_index
        assert list(subs.streams.keys()) == list(SUBSTREAM_LABELS)
        assert len(subs.streams) == len(SUBSTREAM_LABELS) == 7
        for value in subs.streams.values():
            assert type(value) is int, f"stream id {value!r} is not a Python int"
            assert 0 <= value < 2**64


def test_derive_accepts_numpy_integers() -> None:
    """NumPy integers normalize to the same derivation as Python ints."""

    subs_np = derive_substreams(np.int64(20260831), np.int64(3), np.int64(17))
    subs_py = derive_substreams(20260831, 3, 17)
    assert subs_np.streams == subs_py.streams
    assert type(subs_np.master_seed) is int
    assert subs_np.master_seed == 20260831
    # (20260831, 3, 17) is the second pinned row: the normalized
    # derivation must land on the pinned values.
    assert PINNED_TABLE[1][0] == 20260831
    assert [subs_np.streams[label] for label in SUBSTREAM_LABELS] == PINNED_TABLE[1][3]


def test_derive_matches_literal_chain_large_master() -> None:
    """The one-step form equals the literal chain for a > 32-bit master.

    The module docstring claims bit-identity with the literal fresh-object
    spawn chain; the committed pin covers the pinned table rows.  A master
    seed beyond the 32-bit range exercises the multi-word entropy
    coercion the same way, so the equivalence is re-checked there.
    """

    master_seed, replica_index, event_index = 2**40 + 7, 5, 5
    subs = derive_substreams(master_seed, replica_index, event_index)
    literal = (
        np.random.SeedSequence(master_seed)
        .spawn(replica_index + 1)[replica_index]
        .spawn(event_index + 1)[event_index]
    )
    literal_ids = []
    for child in literal.spawn(len(SUBSTREAM_LABELS)):
        state = child.generate_state(2, dtype=np.uint32)
        literal_ids.append((int(state[0]) << 32) | int(state[1]))
    assert [subs.streams[label] for label in SUBSTREAM_LABELS] == literal_ids


_BAD_ARGUMENTS = [
    True,
    False,
    np.bool_(True),
    -1,
    1.5,
    3.0,
    "20260831",
    None,
]


@pytest.mark.parametrize("arg_position", ["master_seed", "replica_index", "event_index"])
@pytest.mark.parametrize("bad", _BAD_ARGUMENTS)
def test_derive_rejects_non_integer_arguments(arg_position: str, bad) -> None:
    """The derivation is a pure function of non-negative integer arguments.

    ``bool`` (including ``np.bool_``), negative values, floats
    (including integer-valued ones), strings, and ``None`` are rejected
    with a ``ValueError`` naming the offending argument.
    """

    kwargs = {"master_seed": 20260831, "replica_index": 0, "event_index": 0}
    kwargs[arg_position] = bad
    with pytest.raises(ValueError, match=arg_position):
        derive_substreams(**kwargs)


def test_derive_distinct_events_and_replicas_distinct_id_sets() -> None:
    """Different events (and replicas) yield different stream-id sets.

    The card contract is per-event distinctness of the seven ids; across
    the small sampled block no two events of the same (master, replica)
    pair, nor two replicas of the same (master, event), share a full
    stream-id set.
    """

    master_seed, event_index = 20260831, 0
    event_sets = {
        event: frozenset(derive_substreams(master_seed, 4, event).streams.values())
        for event in range(16)
    }
    assert len(set(event_sets.values())) == len(event_sets) == 16

    replica_sets = {
        replica: frozenset(
            derive_substreams(master_seed, replica, event_index).streams.values()
        )
        for replica in range(8)
    }
    assert len(set(replica_sets.values())) == len(replica_sets) == 8


# ---------------------------------------------------------------------------
# Item 7: non-overlap on a second (master, replica) pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "master_seed, replica_index, n_events",
    [
        (1, 7, 32),
        (2**40 + 7, 3, 16),
    ],
)
def test_derive_non_overlap_additional_pairs(
    master_seed: int, replica_index: int, n_events: int
) -> None:
    """The seven per-label ids are pairwise distinct for sampled pairs.

    Widen the committed single-pair check to further (master_seed,
    replica_index) pairs, including a master seed beyond the 32-bit
    range.
    """

    for event_index in range(n_events):
        subs = derive_substreams(master_seed, replica_index, event_index)
        values = list(subs.streams.values())
        assert len(values) == 7
        assert len(set(values)) == 7, (master_seed, replica_index, event_index)


# ---------------------------------------------------------------------------
# Item 3: RngSubstreams immutability and stream_for
# ---------------------------------------------------------------------------


def test_rng_substreams_frozen_instance() -> None:
    """``RngSubstreams`` is a frozen dataclass: mutation is rejected."""

    subs = derive_substreams(20260831, 0, 0)
    # dataclasses.is_frozen is not available on Python 3.12; the frozen
    # flag set by the @dataclass(frozen=True) decorator is stable.
    assert RngSubstreams.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        subs.master_seed = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        subs.replica_index = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        subs.event_index = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        subs.streams = {}


def test_stream_for_returns_pinned_values_for_all_labels() -> None:
    """``stream_for`` returns the pinned row value for every one of the seven labels."""

    master_seed, replica_index, event_index, expected = PINNED_TABLE[0]
    subs = derive_substreams(master_seed, replica_index, event_index)
    for label, stream_id in zip(SUBSTREAM_LABELS, expected):
        assert subs.stream_for(label) == stream_id
        assert subs.streams[label] == stream_id


@pytest.mark.parametrize(
    "bad_label",
    ["parton", "string_fragmentation", "", "TOPOLOGY_SELECTION"],
)
def test_stream_for_rejects_further_unknown_labels(bad_label: str) -> None:
    """The label set is closed: empty, misspelled, and case variants raise."""

    subs = derive_substreams(20260831, 0, 0)
    with pytest.raises(KeyError):
        subs.stream_for(bad_label)


def test_stream_for_on_directly_constructed_instances() -> None:
    """``stream_for`` consults ``streams``: missing labels raise ``KeyError``.

    The dataclass constructor accepts a custom (or empty) ``streams``
    mapping; ``stream_for`` returns its values and raises ``KeyError``
    for labels it does not carry, and NumPy-integer fields normalize to
    Python ints at construction.
    """

    bare = RngSubstreams(master_seed=np.int64(5), replica_index=0, event_index=0)
    assert type(bare.master_seed) is int
    assert bare.master_seed == 5
    assert bare.streams == {}
    for label in SUBSTREAM_LABELS:
        with pytest.raises(KeyError):
            bare.stream_for(label)

    custom = RngSubstreams(
        master_seed=5,
        replica_index=0,
        event_index=0,
        streams={"transport": 42},
    )
    assert custom.stream_for("transport") == 42
    with pytest.raises(KeyError):
        custom.stream_for("decay_channel")


# ---------------------------------------------------------------------------
# Item 5: substream_manifest_path contract and manifest writer
# ---------------------------------------------------------------------------


def test_substream_manifest_path_contract(tmp_path) -> None:
    """The manifest path is the zero-padded replica file under the run dir."""

    base = str(tmp_path / "run_dir")
    path_0 = substream_manifest_path(base, 0)
    path_12 = substream_manifest_path(base, 12)
    assert path_0 == os.path.join(base, "substream_manifest_replica_0000.json")
    assert path_12 == os.path.join(base, "substream_manifest_replica_0012.json")
    assert os.path.dirname(path_0) == base
    assert os.path.dirname(path_12) == base
    # The zero-padded index keeps the per-replica files sorted.
    assert os.path.basename(path_0) < os.path.basename(path_12)
    # NumPy-integer replica indices are normalized to the same path.
    assert substream_manifest_path(base, np.int64(12)) == path_12
    with pytest.raises(ValueError, match="replica_index"):
        substream_manifest_path(base, -1)


def _parse_ordered(text: str):
    """Parse JSON keeping dicts as list-of-``(key, value)`` pairs.

    ``object_pairs_hook=list`` pins the on-file key order, which is what
    the byte-determinism of the writer depends on.
    """

    return json.loads(text, object_pairs_hook=list)


def _keys_of(node: list) -> list:
    return [key for key, _ in node]


def _value_of(node: list, key: str):
    for name, value in node:
        if name == key:
            return value
    raise AssertionError(f"key {key!r} not found in {node!r}")


def _hand_built_events() -> list:
    """Three events covering shuffled streams, mixed draws, and no draws.

    Event 0 carries the seven stream ids in reverse label order with two
    draw entries (one ``n_draws`` as a NumPy integer); event 1 carries
    them in canonical order with one draw; event 2 omits the ``draws``
    key entirely (the writer's committed default).
    """

    def streams_for(event_index: int) -> dict:
        subs = derive_substreams(20260911, 1, event_index)
        return dict(zip(SUBSTREAM_LABELS, (subs.streams[l] for l in SUBSTREAM_LABELS)))

    ordered = streams_for(0)
    shuffled = dict(reversed(list(ordered.items())))
    return [
        {
            "event_index": 0,
            "streams": shuffled,
            "draws": [
                {"label": "topology_selection", "stream_id": ordered["topology_selection"], "n_draws": 1},
                {"label": "decay_channel", "stream_id": ordered["decay_channel"], "n_draws": np.int64(3)},
            ],
        },
        {
            "event_index": 1,
            "streams": streams_for(1),
            "draws": [
                {"label": "transport", "stream_id": streams_for(1)["transport"], "n_draws": 5},
            ],
        },
        {
            "event_index": 2,
            "streams": streams_for(2),
        },
    ]


def test_write_substream_manifest_schema_and_bytes(tmp_path) -> None:
    """The manifest JSON matches the card schema and serializes deterministically."""

    events = _hand_built_events()
    path_a = tmp_path / "replica_0000.json"
    path_b = tmp_path / "nested" / "deeper" / "replica_0000.json"
    write_substream_manifest(str(path_a), events)
    write_substream_manifest(str(path_b), events)

    text_a = path_a.read_bytes()
    text_b = path_b.read_bytes()
    assert text_a == text_b, "identical inputs must produce identical bytes"

    # Valid JSON, top-level schema per the card.
    payload = _parse_ordered(text_a.decode("utf-8"))
    assert _keys_of(payload) == ["schema", "substream_labels", "events"]
    assert _value_of(payload, "schema") == MANIFEST_SCHEMA
    assert _value_of(payload, "substream_labels") == list(SUBSTREAM_LABELS)
    written_events = _value_of(payload, "events")
    assert len(written_events) == len(events)

    for written, original in zip(written_events, events):
        assert _keys_of(written) == ["event_index", "streams", "draws"]
        assert _value_of(written, "event_index") == original["event_index"]
        written_streams = _value_of(written, "streams")
        # Streams are emitted in canonical SUBSTREAM_LABELS order even
        # when the input mapping is shuffled.
        assert _keys_of(written_streams) == list(SUBSTREAM_LABELS)
        assert {k: v for k, v in written_streams} == dict(original["streams"])
        written_draws = _value_of(written, "draws")
        expected_draws = original.get("draws", [])
        assert len(written_draws) == len(expected_draws)
        for entry, expected in zip(written_draws, expected_draws):
            assert _keys_of(entry) == ["label", "stream_id", "n_draws"]
            assert _value_of(entry, "label") == expected["label"]
            assert _value_of(entry, "stream_id") == expected["stream_id"]
            n_draws = _value_of(entry, "n_draws")
            assert type(n_draws) is int, "n_draws is recorded as a JSON integer"
            assert n_draws == int(expected["n_draws"])

    # The omitted-draws event defaults to an empty draw list.
    assert _value_of(written_events[2], "draws") == []

    # Rewriting the same input over the same path is byte-idempotent.
    before = path_a.read_bytes()
    write_substream_manifest(str(path_a), events)
    assert path_a.read_bytes() == before

    # The input key order of the streams mapping is irrelevant: the
    # writer normalizes to canonical SUBSTREAM_LABELS order, so an
    # input with the same content in canonical key order serializes to
    # the same full-file bytes.
    canonical_events = [
        {
            **event,
            "streams": {label: event["streams"][label] for label in SUBSTREAM_LABELS},
        }
        for event in events
    ]
    path_c = tmp_path / "canonical_order.json"
    write_substream_manifest(str(path_c), canonical_events)
    assert path_c.read_bytes() == text_a


# ---------------------------------------------------------------------------
# Item 6: complementary parallel-path angle (p+A full-output bit identity)
# ---------------------------------------------------------------------------


def _delta_gen_file() -> str:
    return str(
        NESTED_ROOT / "na49_mixing" / "generators" / "michal" / "deltas-to-pion" / "delta++.gen"
    )


def _assert_generation_outputs_identical(reference: Any, actual: Any) -> None:
    """The merged chunked result is bit-identical across the two runs.

    All seven per-track arrays compare element-wise; every other key
    (chunk metadata, counters, audit tables, nucleus spec) compares by
    ``==``.  The key sets must match exactly as well.
    """

    assert set(reference) == set(actual), (
        f"merged result key sets differ: {sorted(set(reference) ^ set(actual))}"
    )
    for key in _TRACK_ARRAY_KEYS:
        assert np.array_equal(reference[key], actual[key]), key
    for key in reference:
        if key in _TRACK_ARRAY_KEYS:
            continue
        assert reference[key] == actual[key], key


def test_pa_manifest_hook_bit_identity(tmp_path) -> None:
    """The p+A manifest hook is additive: the whole output is unchanged.

    The committed hook test compares four track arrays on the p+p path
    and only checks manifest presence on the p+A path.  Here a 12-event
    p+A chunked run (point target: no sampler draws) with
    ``substream_manifest_path`` set is bit-identical to the same run
    without the hook across the entire merged output, and each chunk
    manifest records the per-event stream ids of
    ``derive_substreams(seed, chunk_index, event_index)`` with empty
    draw counts (recorded, not enforced, by this card).
    """

    seed = 20260911
    nucleus = NucleusSpec(name="p", A=1, Z=1)
    run_dir = tmp_path / "pa_substreams"
    reference = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=12,
            seed=seed,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
            nucleus=nucleus,
        ),
        n_workers=1,
        n_chunks=2,
    )
    with_manifest = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=12,
            seed=seed,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
            nucleus=nucleus,
            substream_manifest_path=str(run_dir),
        ),
        n_workers=1,
        n_chunks=2,
    )

    # The manifest write consumes no draws: the full merged result of the
    # hook run is bit-identical to the reference run.
    _assert_generation_outputs_identical(reference, with_manifest)

    chunk_counts = {c["chunk_index"]: c["n_events"] for c in with_manifest["chunks"]}
    for chunk_index in sorted(chunk_counts):
        path = Path(substream_manifest_path(str(run_dir), chunk_index))
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == MANIFEST_SCHEMA
        assert payload["substream_labels"] == list(SUBSTREAM_LABELS)
        records = payload["events"]
        assert [r["event_index"] for r in records] == list(
            range(chunk_counts[chunk_index])
        )
        for record in records:
            subs = derive_substreams(seed, chunk_index, record["event_index"])
            assert record["streams"] == dict(subs.streams)
            assert record["draws"] == []


# ---------------------------------------------------------------------------
# Staged run-metadata block: gating and content
# ---------------------------------------------------------------------------


def _smoke_config(substreams_block: dict | None) -> dict:
    """The bundled smoke configuration, PPEventBuilder stage only.

    ``fit.enabled`` is false so no SPS tables are needed.
    ``substreams_block`` is injected (or omitted entirely for the legacy
    default) as the additive config key.
    """

    config_path = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["through"] = "simple_excitation"
    config["fit"] = {**config.get("fit", {}), "enabled": False}
    if substreams_block is not None:
        config["substreams"] = substreams_block
    return config


def _stage_event_hashes(samples: dict) -> list:
    return [event.as_dict()["event_hash"] for event in samples["simple_excitation"]]


def test_staged_substreams_metadata_block_gating() -> None:
    """The run-metadata ``substreams`` block is gated on ``substreams.enabled``.

    The block is absent for the legacy default and for an empty
    ``substreams`` block (``enabled`` defaults to false); when enabled it
    carries exactly the three card keys, records the config seed and the
    module replica-seed rule string verbatim, and the 20 PPEventBuilder
    event hashes are bit-identical with and without it.
    """

    from na49_mixing.staged_evaluation import (
        _SUBSTREAM_REPLICA_SEED_RULE,
        run_staged_evaluation,
    )

    config_path = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
    config_seed = json.loads(config_path.read_text(encoding="utf-8"))["seed"]

    empty_block_report, empty_block_samples = run_staged_evaluation(
        _smoke_config({})
    )
    enabled_report, enabled_samples = run_staged_evaluation(
        _smoke_config({"enabled": True})
    )

    # The metadata block is run metadata only: the event hashes are
    # bit-identical with and without it.
    assert len(_stage_event_hashes(empty_block_samples)) == 20
    assert _stage_event_hashes(empty_block_samples) == _stage_event_hashes(
        enabled_samples
    )

    assert "substreams" not in empty_block_report
    block = enabled_report["substreams"]
    assert set(block) == {"master_seed", "replica_seed_rule", "substream_labels"}
    assert block["master_seed"] == config_seed
    assert block["substream_labels"] == list(SUBSTREAM_LABELS)
    assert block["replica_seed_rule"] == _SUBSTREAM_REPLICA_SEED_RULE
    assert "derive_substreams(master_seed, replica_index, event_index)" in block[
        "replica_seed_rule"
    ]
