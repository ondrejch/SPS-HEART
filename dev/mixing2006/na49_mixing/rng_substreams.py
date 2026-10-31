"""Deterministic RNG substream labeling for event generation.

Round 1, card 5 (plan section 6, task 5; plan section 4.3 names this
module).  Every random draw in event generation is assigned a labeled
substream, one per physical layer.  The seven labels below are the plan
section 6, task 5 physical layers verbatim and the set is closed: a new
layer files its own card.  The labels are draw-attribution labels only;
none of them implies strings, parton splitting, or constituent breakup.

This card records the substream map; it does not enforce it.  No draw
value changes when the feature is off, and the manifest writer is
additive: it never consumes randomness from a generation stream.

Derivation contract
-------------------

``derive_substreams(master_seed, replica_index, event_index)`` is a pure
function of its three non-negative integer arguments.  It chains
``numpy.random.SeedSequence`` spawn levels,

    SeedSequence(master_seed).spawn(replica_index + 1)[replica_index]
        .spawn(event_index + 1)[event_index]
        .spawn(len(SUBSTREAM_LABELS))

and takes one 64-bit stream id per label from the spawned children.  The
implementation uses the equivalent one-step form
``SeedSequence(master_seed, spawn_key=(replica_index, event_index))``,
which is bit-identical to the literal fresh-object spawn chain (each
intermediate sequence is constructed fresh, so its spawn counter starts
at zero) and avoids the O(replica_index + event_index) cost of
materializing the intermediate children.  The equivalence is pinned by
``tests/test_rng_substreams.py``.

Per-event seed contract: a draw in layer ``L`` of event ``e`` of replica
``k`` is seeded with ``(stream_for(L), event_index)``, so the same
(substream_label, event_index) pair reproduces the same draw sequence.
This matches the existing ``<config.seed>:<event_id>`` convention in
``na49_mixing.staged_evaluation`` but labels the physical layer.

The SeedSequence mixing algorithm uses fixed constants and is stable
across numpy versions and process invocations.  The derivation depends
on nothing but its arguments: no wall-clock time, no process id, no hash
randomization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

__all__ = [
    "SUBSTREAM_LABELS",
    "MANIFEST_SCHEMA",
    "RngSubstreams",
    "derive_substreams",
    "substream_manifest_path",
    "write_substream_manifest",
]


#: The closed set of physical layers a random draw can serve (plan
#: section 6, task 5).  A new layer files its own card.
SUBSTREAM_LABELS: tuple[str, ...] = (
    "topology_selection",
    "exchange_kinematics",
    "spectral_state",
    "decay_channel",
    "decay_kinematics",
    "nuclear_geometry",
    "transport",
)

#: Schema tag written into every substream manifest.
MANIFEST_SCHEMA = "na49_mixing.substream_manifest/1"


def _require_non_negative_int(value: Any, name: str) -> int:
    """Normalize *value* to a non-negative Python ``int``.

    Python and NumPy integers are accepted.  Booleans are rejected even
    though they subclass ``int``.  Anything else, or a negative value,
    raises ``ValueError`` naming *name*.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return converted


@dataclass(frozen=True)
class RngSubstreams:
    """Per-layer stream ids for one (master_seed, replica, event) triple."""

    master_seed: int
    replica_index: int
    event_index: int
    streams: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "master_seed", _require_non_negative_int(self.master_seed, "master_seed")
        )
        object.__setattr__(
            self, "replica_index", _require_non_negative_int(self.replica_index, "replica_index")
        )
        object.__setattr__(
            self, "event_index", _require_non_negative_int(self.event_index, "event_index")
        )

    def stream_for(self, label: str) -> int:
        """Return the stream id for *label*.

        Raises ``KeyError`` for a label outside ``SUBSTREAM_LABELS``:
        the label set is closed and a new layer files its own card.
        """

        if label not in self.streams:
            raise KeyError(
                f"unknown substream label {label!r}; the label set is "
                f"closed: {list(SUBSTREAM_LABELS)}"
            )
        return self.streams[label]


def derive_substreams(
    master_seed: int,
    replica_index: int,
    event_index: int,
) -> RngSubstreams:
    """Derive one stream id per ``SUBSTREAM_LABELS`` entry.

    Pure function of ``(master_seed, replica_index, event_index)``;
    see the module docstring for the spawn-chain contract and the
    one-step ``spawn_key`` form used here.
    """

    ms = _require_non_negative_int(master_seed, "master_seed")
    ri = _require_non_negative_int(replica_index, "replica_index")
    ei = _require_non_negative_int(event_index, "event_index")
    # One-step equivalent of SeedSequence(ms).spawn(ri + 1)[ri]
    # .spawn(ei + 1)[ei]: the spawn keys compose because every
    # intermediate sequence is freshly constructed.
    event_seq = np.random.SeedSequence(ms, spawn_key=(ri, ei))
    label_seqs = event_seq.spawn(len(SUBSTREAM_LABELS))
    streams: Dict[str, int] = {}
    for label, seq in zip(SUBSTREAM_LABELS, label_seqs):
        state = seq.generate_state(2, dtype=np.uint32)
        streams[label] = (int(state[0]) << 32) | int(state[1])
    return RngSubstreams(
        master_seed=ms, replica_index=ri, event_index=ei, streams=streams
    )


def substream_manifest_path(run_dir: str, replica_index: int) -> str:
    """Return the manifest file path for *replica_index* under *run_dir*.

    One manifest file per replica (a parallel generation chunk counts as
    the replica of its master stream); the zero-padded index keeps the
    files sorted.
    """

    ri = _require_non_negative_int(replica_index, "replica_index")
    return os.path.join(str(run_dir), f"substream_manifest_replica_{ri:04d}.json")


def write_substream_manifest(path: str, events: List[dict]) -> None:
    """Write the per-event substream manifest JSON file at *path*.

    *events* holds one record per event, each with ``event_index``
    (int), ``streams`` (label -> stream id), and ``draws`` (a list of
    ``{"label", "stream_id", "n_draws"}`` entries; the draw counts are
    recorded, not enforced, by the current card).  The file is purely
    additive bookkeeping: writing it never alters a generation stream.
    Stream entries are emitted in canonical ``SUBSTREAM_LABELS`` order
    (any additional keys follow in sorted order) so identical inputs
    produce identical bytes.
    """

    payload_events: List[Dict[str, Any]] = []
    for record in events:
        streams_in = record["streams"]
        ordered: Dict[str, int] = {
            label: int(streams_in[label])
            for label in SUBSTREAM_LABELS
            if label in streams_in
        }
        extra: Dict[str, int] = {
            key: int(streams_in[key])
            for key in sorted(streams_in)
            if key not in ordered
        }
        draws = [
            {
                "label": str(draw["label"]),
                "stream_id": int(draw["stream_id"]),
                "n_draws": int(draw["n_draws"]),
            }
            for draw in record.get("draws", ())
        ]
        payload_events.append(
            {
                "event_index": int(record["event_index"]),
                "streams": {**ordered, **extra},
                "draws": draws,
            }
        )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "substream_labels": list(SUBSTREAM_LABELS),
        "events": payload_events,
    }
    parent = os.path.dirname(os.path.abspath(str(path)))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
