"""Latent-event cache and reweight policy (Round 1, card 6).

Plan section 4.3 names this module ("latent-event caching and
reweighting"); plan section 5.3 defines the cached latent variables and
the reweight rule; the implementing card is
``TASK-20260911-r1-latent-event-cache``.  The module provides three
public objects:

- :class:`LatentEventRecord`, the frozen dataclass for one pre-fit
  event-level record (topology identity, exchange masses and transfers,
  candidate channel families, phase-space points, angular variables,
  bin mapping, generation weight, substream seeds, config hash, and
  library fingerprint).
- :func:`reweight_policy`, the decision rule that classifies a parameter
  change as ``"reweight"`` or ``"regenerate"``.
- :class:`LatentEventCache`, the on-disk store: one JSON file per record
  under ``<run_dir>/latent_cache/``, keyed by the sha256 of the record's
  canonical JSON serialization.

Reweight rule
-------------

The policy returns ``"reweight"`` exactly when the changed parameter is
in :data:`DENSITY_ONLY_PARAMETERS`: those four parameters scale the
numerator of a source-decay channel weight (a family strength) and leave
the kinematic support untouched.  Every other parameter returns
``"regenerate"``.  The six names in :data:`SUPPORT_ALTERING_PARAMETERS`
alter the kinematic support: they change the source-mass spectrum
(``regge.alpha_exc0``, ``regge.edge_power``), the transfer slope
(``regge.effective_transfer_slope``), the continuum mass window
(``continuum.match_mass``), the extra-pion multiplicity distribution
(``continuum.pion_multiplicity_slope``), or the mass dependence of an
opening factor (``continuum.pair_threshold_power``); the support
changes, so the cached records must be regenerated.  Any parameter
outside both lists also returns ``"regenerate"``: the conservative
default, so an unknown parameter is never reweighted.  The two lists
partition the frozen M8 ten-parameter set (pinned in
``tests/test_legacy_v1_freeze.py``); no new parameter is introduced.

One boundary belongs to the reweight step, not to this classification:
a family strength moved exactly to zero closes its channel (the
requested weight is absent from the family set at any mass), so a
record whose candidate channel would close under the new strength must
be regenerated.  Checking that per-record condition is the caller's
validity check; :func:`reweight_policy` classifies parameter names.

Cache mechanics
---------------

The store is pickle-free and JSON-only (the project ``data_io``
convention; no ``.npy``, no pickle).  ``write`` emits one file per
record named ``<key>.json`` where ``key = key_for(record)``; ``read``
reloads the records and returns them in ``event_index`` order.  A stored
file whose key does not match the recomputed key of its contents raises
``ValueError`` (a stale or edited record is a loud error, never a silent
reuse).  An absent or empty cache directory reads back as an empty list
(a cache miss, the regenerate path).

The cache is a new v2 infrastructure hook.  It is not wired into
``PPEventBuilder``, ``PAEventBuilder``, or the legacy ``Generator`` in
Round 1; a later-round card adds the builder hook that produces and
consumes the records.  The record is a bookkeeping object for a
color-singlet event: it introduces no colored object and no new state
type, and the module name implies no strings, parton splitting, or
constituent breakup.

Hash conventions (reused, not invented)
---------------------------------------

``config_hash`` is the canonical configuration hash of
``na49_mixing.staged_evaluation._canonical_config_hash``: sha256 over
``json.dumps(config, sort_keys=True, separators=(",", ":"))``.  Compute
it with :func:`config_hash_for`.  ``lib_fingerprint`` follows the module
byte-string hash convention of
``helpers/benchmarking/benchmark_sps_species._decay_lib_fingerprint``
(the cache invalidation fingerprint behind the
``RESONANCE_DECAY_LIB_VERSION`` salt): sha256 over the source bytes of a
fixed module set, each chunk length-prefixed, resolved through the
imported package via ``importlib``.  :func:`default_lib_fingerprint`
hashes the modules that define the staged two-source generation;
``staged_evaluation`` itself is excluded because its report and fit
helpers do not define generation behavior (a report edit must not
invalidate latent records).  Both helpers raise loudly on failure; the
``None``-on-failure contract of the helpers fingerprint belongs to that
caller, not to this one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

__all__ = [
    "RECORD_SCHEMA",
    "REWEIGHT",
    "REGENERATE",
    "DENSITY_ONLY_PARAMETERS",
    "SUPPORT_ALTERING_PARAMETERS",
    "LatentEventRecord",
    "reweight_policy",
    "LatentEventCache",
    "record_payload",
    "record_from_payload",
    "config_hash_for",
    "default_lib_fingerprint",
]


#: Schema tag written into every cached record file.
RECORD_SCHEMA = "na49_mixing.latent_event_record/1"

#: Policy verdict: the cached record stays valid and is reweighted.
REWEIGHT = "reweight"

#: Policy verdict: the cached record is invalid and must be regenerated.
REGENERATE = "regenerate"

#: Density-only parameters (card-pinned order): each scales a channel
#: probability, the numerator of a source-decay family weight, without
#: altering the kinematic support.
DENSITY_ONLY_PARAMETERS: tuple[str, ...] = (
    "continuum.pair_strength",
    "continuum.strange_pair_strength",
    "continuum.strange_lambda_strength",
    "continuum.strange_sigma_strength",
)

#: Support-altering parameters (card-pinned order): each alters the mass
#: window, the slope, the multiplicity distribution, or the threshold
#: dependence, so the kinematic support changes.
SUPPORT_ALTERING_PARAMETERS: tuple[str, ...] = (
    "regge.alpha_exc0",
    "regge.effective_transfer_slope",
    "regge.edge_power",
    "continuum.match_mass",
    "continuum.pion_multiplicity_slope",
    "continuum.pair_threshold_power",
)


def reweight_policy(
    record: LatentEventRecord,
    changed_parameter: str,
) -> str:
    """Classify a parameter change for one cached record.

    Returns :data:`REWEIGHT` exactly when *changed_parameter* is in
    :data:`DENSITY_ONLY_PARAMETERS`; every other name returns
    :data:`REGENERATE` (the conservative default: the support-altering
    list and all unknown parameter names regenerate).  *record* is
    accepted for signature compatibility with the v2 builder hook; the
    classification is a decision rule over parameter names and does not
    depend on the record contents.
    """

    if not isinstance(record, LatentEventRecord):
        raise TypeError(
            f"reweight_policy expects a LatentEventRecord, got {type(record)!r}"
        )
    if not isinstance(changed_parameter, str) or not changed_parameter:
        raise ValueError(
            "changed_parameter must be a non-empty dotted parameter name, "
            f"got {changed_parameter!r}"
        )
    if changed_parameter in DENSITY_ONLY_PARAMETERS:
        return REWEIGHT
    return REGENERATE


def _require_non_negative_int(value: Any, name: str) -> int:
    """Normalize *value* to a non-negative Python ``int``.

    Python and NumPy integers are accepted; booleans are rejected even
    though they subclass ``int``.  Anything else, or a negative value,
    raises ``ValueError`` naming *name*.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_hex_digest(value: Any, name: str) -> str:
    """Require a 64-character lowercase sha256 hex digest string."""

    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character sha256 hex string, got {value!r}")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a 64-character sha256 hex string, got {value!r}")
    return value


@dataclass(frozen=True)
class LatentEventRecord:
    """One pre-fit event-level record (the unit of cache invalidation).

    ``config_hash`` and ``lib_fingerprint`` pin the record to the
    configuration and to the installed generation code (see the module
    docstring for the reused conventions); ``substream_seeds`` carries
    the per-layer stream ids of card 5 (label -> stream id).
    """

    event_index: int
    topology_identity: str
    exchange_masses: tuple[float, ...]
    exchange_transfers: tuple[float, ...]
    candidate_channels: tuple[str, ...]
    phase_space_points: tuple[tuple[float, ...], ...]
    angular_variables: tuple[float, ...]
    bin_mapping: dict[str, int]
    generation_weight: float
    substream_seeds: dict[str, int]
    config_hash: str
    lib_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_index", _require_non_negative_int(self.event_index, "event_index")
        )
        for name in ("topology_identity",):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
            object.__setattr__(self, name, value)
        for name in ("exchange_masses", "exchange_transfers", "angular_variables"):
            values = tuple(float(v) for v in getattr(self, name))
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must hold finite floats, got {getattr(self, name)!r}")
            object.__setattr__(self, name, values)
        points = tuple(
            tuple(float(v) for v in point) for point in self.phase_space_points
        )
        for point in points:
            if not all(math.isfinite(v) for v in point):
                raise ValueError(
                    f"phase_space_points must hold finite floats, got "
                    f"{self.phase_space_points!r}"
                )
        object.__setattr__(self, "phase_space_points", points)
        channels = tuple(str(channel) for channel in self.candidate_channels)
        for channel in channels:
            if not channel:
                raise ValueError(f"candidate_channels entries must be non-empty, got {channel!r}")
        object.__setattr__(self, "candidate_channels", channels)
        weight = float(self.generation_weight)
        if not math.isfinite(weight):
            raise ValueError(
                f"generation_weight must be a finite float, got {self.generation_weight!r}"
            )
        object.__setattr__(self, "generation_weight", weight)
        bin_mapping = {
            str(key): _require_non_negative_int(value, f"bin_mapping[{key!r}]")
            for key, value in dict(self.bin_mapping).items()
        }
        object.__setattr__(self, "bin_mapping", bin_mapping)
        substream_seeds = {
            str(key): _require_non_negative_int(value, f"substream_seeds[{key!r}]")
            for key, value in dict(self.substream_seeds).items()
        }
        object.__setattr__(self, "substream_seeds", substream_seeds)
        object.__setattr__(
            self, "config_hash", _require_hex_digest(self.config_hash, "config_hash")
        )
        object.__setattr__(
            self,
            "lib_fingerprint",
            _require_hex_digest(self.lib_fingerprint, "lib_fingerprint"),
        )


#: The record field names, in declaration order (the card's 12 fields).
RECORD_FIELD_NAMES: tuple[str, ...] = tuple(
    field.name for field in fields(LatentEventRecord)
)


def record_payload(record: LatentEventRecord) -> dict[str, Any]:
    """The JSON-able payload of *record* (the canonical key input).

    Tuples become lists, dict fields become plain dicts with string keys,
    and every numeric value is coerced to a JSON-native ``int`` or
    ``float`` so NumPy scalars serialize.  The mapping from field name to
    payload key is the identity over :data:`RECORD_FIELD_NAMES`.
    """

    payload: dict[str, Any] = {}
    for name in RECORD_FIELD_NAMES:
        value = getattr(record, name)
        if name in ("exchange_masses", "exchange_transfers", "angular_variables"):
            payload[name] = [float(v) for v in value]
        elif name == "phase_space_points":
            payload[name] = [[float(v) for v in point] for point in value]
        elif name == "candidate_channels":
            payload[name] = [str(channel) for channel in value]
        elif name in ("bin_mapping", "substream_seeds"):
            payload[name] = {str(key): int(v) for key, v in value.items()}
        elif name == "event_index":
            payload[name] = int(value)
        elif name == "generation_weight":
            payload[name] = float(value)
        else:
            payload[name] = str(value)
    return payload


def record_from_payload(payload: dict[str, Any]) -> LatentEventRecord:
    """Rebuild a :class:`LatentEventRecord` from a stored file payload.

    The payload must be a dict with exactly the keys ``schema``, ``key``,
    and ``record``; the schema tag must equal :data:`RECORD_SCHEMA` and
    the inner record dict must carry exactly the
    :data:`RECORD_FIELD_NAMES` keys.  Violations raise ``ValueError``
    naming the problem.
    """

    if not isinstance(payload, dict) or set(payload) != {"schema", "key", "record"}:
        raise ValueError(
            "latent-event record payload must be a dict with exactly the "
            f"keys schema/key/record, got {sorted(payload) if isinstance(payload, dict) else payload!r}"
        )
    if payload["schema"] != RECORD_SCHEMA:
        raise ValueError(
            f"unknown latent-event record schema {payload['schema']!r}; "
            f"expected {RECORD_SCHEMA!r}"
        )
    record_data = payload["record"]
    if not isinstance(record_data, dict) or set(record_data) != set(RECORD_FIELD_NAMES):
        raise ValueError(
            "latent-event record must carry exactly the fields "
            f"{list(RECORD_FIELD_NAMES)}, got "
            f"{sorted(record_data) if isinstance(record_data, dict) else record_data!r}"
        )
    data = dict(record_data)
    for name in ("exchange_masses", "exchange_transfers", "angular_variables", "candidate_channels"):
        data[name] = tuple(data[name])
    data["phase_space_points"] = tuple(tuple(point) for point in data["phase_space_points"])
    data["bin_mapping"] = dict(data["bin_mapping"])
    data["substream_seeds"] = dict(data["substream_seeds"])
    return LatentEventRecord(**data)


def _canonical_json(payload: Any) -> str:
    """The canonical compact sort-keys JSON string (no non-finite values)."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash_for(config: Mapping[str, Any]) -> str:
    """The canonical configuration hash: sha256 over the canonical JSON.

    Reuses the convention of ``na49_mixing.staged_evaluation
    ._canonical_config_hash`` (sort-keys compact JSON, sha256); a record
    pinned to this hash is invalidated by any configuration change.
    """

    return hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()


#: The modules that define staged two-source generation behavior, in
#: hash order.  The set is closed; a new module files its own card.
#: ``staged_evaluation`` is excluded on purpose (see the module
#: docstring).
_FINGERPRINT_MODULES: tuple[str, ...] = (
    "na49_mixing.exchange_model",
    "na49_mixing.excitation_kinematics",
    "na49_mixing.continuum_channels",
    "na49_mixing.breit_wigner",
    "na49_mixing.pdg_database",
    "na49_mixing.event_builder_pp",
    "na49_mixing.event_builder_pA",
    "na49_mixing.source_event",
    "na49_mixing.rng_substreams",
    "na49_mixing.state_validation",
)


def default_lib_fingerprint() -> str:
    """The module byte-string hash of the staged generation modules.

    Mirrors ``helpers/benchmarking/benchmark_sps_species
    ._decay_lib_fingerprint``: sha256 over ``(tag, source bytes)``
    chunks, each tag and payload length-prefixed with a big-endian
    8-byte length, module sources resolved through the imported package
    via ``importlib`` (never through repo-root string arithmetic, so the
    monorepo tree and a standalone tree hash identical inputs).  A
    source edit to any fingerprinted module changes the fingerprint and
    invalidates the cache.  Failures raise; the caller should never
    cache a record whose fingerprint could not be computed.
    """

    import importlib

    chunks: list[tuple[str, bytes]] = []
    for modname in _FINGERPRINT_MODULES:
        module = importlib.import_module(modname)
        origin = getattr(module, "__file__", None)
        if not origin:
            raise RuntimeError(f"cannot resolve source file of {modname}")
        with open(origin, "rb") as handle:
            chunks.append((f"source:{modname}", handle.read()))
    digest = hashlib.sha256()
    for tag, payload in chunks:
        tag_bytes = tag.encode("utf-8")
        digest.update(len(tag_bytes).to_bytes(8, "big"))
        digest.update(tag_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class LatentEventCache:
    """JSON-only, sha256-keyed store of :class:`LatentEventRecord` files.

    One file per record under ``<run_dir>/latent_cache/``; the file name
    is ``key_for(record) + ".json"``.  Writing the same records twice is
    idempotent (identical records produce identical bytes).
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = str(run_dir)
        self.cache_dir = os.path.join(self.run_dir, "latent_cache")

    def key_for(self, record: LatentEventRecord) -> str:
        """The sha256 of the record's canonical JSON serialization."""

        if not isinstance(record, LatentEventRecord):
            raise TypeError(
                f"key_for expects a LatentEventRecord, got {type(record)!r}"
            )
        return hashlib.sha256(
            _canonical_json(record_payload(record)).encode("utf-8")
        ).hexdigest()

    def write(self, records: list[LatentEventRecord]) -> None:
        """Write one JSON file per record; creates the cache directory.

        The file payload is ``{"schema", "key", "record"}`` written with
        the byte-deterministic convention of the card 5 substream
        manifest (``json.dump`` with ``indent=2`` plus a trailing
        newline), so identical records produce identical bytes.
        """

        os.makedirs(self.cache_dir, exist_ok=True)
        for index, record in enumerate(records):
            if not isinstance(record, LatentEventRecord):
                raise TypeError(
                    f"records[{index}] is not a LatentEventRecord: {type(record)!r}"
                )
            key = self.key_for(record)
            path = os.path.join(self.cache_dir, f"{key}.json")
            payload = {
                "schema": RECORD_SCHEMA,
                "key": key,
                "record": record_payload(record),
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")

    def read(self) -> list[LatentEventRecord]:
        """Read all cached records, in ``event_index`` order.

        An absent or empty cache directory yields an empty list (a cache
        miss).  Every stored file is re-keyed from its contents; a file
        whose stored key differs from the recomputed key raises
        ``ValueError`` (a stale or edited record must never be silently
        reused).
        """

        records: list[LatentEventRecord] = []
        if not os.path.isdir(self.cache_dir):
            return records
        for name in sorted(os.listdir(self.cache_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.cache_dir, name)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record = record_from_payload(payload)
            stored_key = payload["key"]
            if stored_key != self.key_for(record):
                raise ValueError(
                    f"latent-event cache key mismatch for {path}: stored "
                    f"{stored_key!r}, recomputed {self.key_for(record)!r}; "
                    "the record is stale or was edited after write"
                )
            records.append(record)
        records.sort(key=lambda record: record.event_index)
        return records
