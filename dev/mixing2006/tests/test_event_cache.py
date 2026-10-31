"""Deterministic tests for the latent-event cache (Round 1, card 6).

Card ``TASK-20260911-r1-latent-event-cache``.  Seven named tests pin the
reweight policy's two card-pinned parameter lists (plus the conservative
default for unknown names), the JSON write/read roundtrip through
:class:`na49_mixing.event_cache.LatentEventCache`, and the sha256 cache
key (determinism, config sensitivity, library-fingerprint sensitivity).

Hermetic: no event generation.  The records are hand-constructed here;
the ``substream_seeds`` values come from the pure card 5 derivation
``derive_substreams`` (a SeedSequence computation, no generation stream),
and the fingerprints are computed from the installed modules by the
module's own helpers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from na49_mixing.event_cache import (
    DENSITY_ONLY_PARAMETERS,
    REGENERATE,
    RECORD_FIELD_NAMES,
    RECORD_SCHEMA,
    REWEIGHT,
    SUPPORT_ALTERING_PARAMETERS,
    LatentEventCache,
    LatentEventRecord,
    config_hash_for,
    default_lib_fingerprint,
    reweight_policy,
    record_payload,
)
from na49_mixing.rng_substreams import derive_substreams

# The frozen M8 ten-parameter set pinned by the legacy_v1 freeze (card 1).
# The two policy lists must partition exactly these names.
from tests.test_legacy_v1_freeze import FROZEN_M8_PARAMETERS

# Pinned parameter lists (card TASK-20260911-r1-latent-event-cache, verbatim
# order).  The report allow-lists must match these byte-for-byte.
PINNED_DENSITY_ONLY = (
    "continuum.pair_strength",
    "continuum.strange_pair_strength",
    "continuum.strange_lambda_strength",
    "continuum.strange_sigma_strength",
)
PINNED_SUPPORT_ALTERING = (
    "regge.alpha_exc0",
    "regge.effective_transfer_slope",
    "regge.edge_power",
    "continuum.match_mass",
    "continuum.pion_multiplicity_slope",
    "continuum.pair_threshold_power",
)

# The card's 12 record fields, exactly.
PINNED_RECORD_FIELDS = (
    "event_index",
    "topology_identity",
    "exchange_masses",
    "exchange_transfers",
    "candidate_channels",
    "phase_space_points",
    "angular_variables",
    "bin_mapping",
    "generation_weight",
    "substream_seeds",
    "config_hash",
    "lib_fingerprint",
)


def _record(
    event_index: int,
    config_hash: str | None = None,
    lib_fingerprint: str | None = None,
    generation_weight: float = 0.98765,
) -> LatentEventRecord:
    """One hand-constructed record with real channel-family labels."""

    return LatentEventRecord(
        event_index=event_index,
        topology_identity=f"pp:2source:exchange_m={1.242879 + event_index * 1e-4:.6f}",
        exchange_masses=(1.242879, 1.310231),
        exchange_transfers=(0.113, 0.113),
        # Real FinalParticle.channel_family labels (na49_mixing
        # .continuum_channels._family_spec); no new channel family.
        candidate_channels=("continuum_nonstrange", "associated_KLambda", "baryon_pair"),
        phase_space_points=((0.4321, 0.5123, 0.0811), (0.3978, 0.4811, 0.0791)),
        angular_variables=(0.3215, -0.1178, 1.0423),
        bin_mapping={"pi_minus": 12, "pi_plus": 44},
        generation_weight=generation_weight,
        # Card 5 linkage: per-layer stream ids from the pure derivation.
        substream_seeds=dict(derive_substreams(20260831, 0, event_index).streams),
        config_hash=config_hash
        or config_hash_for(
            {"beam_momentum": 158.0, "production_mode": "regge", "continuum": {}}
        ),
        lib_fingerprint=lib_fingerprint or default_lib_fingerprint(),
    )


# ---------------------------------------------------------------------------
# test_reweight_policy_density_only
# ---------------------------------------------------------------------------


def test_reweight_policy_density_only() -> None:
    """Each pinned density-only parameter returns ``"reweight"``.

    The module's allow-list constant equals the pinned card tuple
    verbatim (order included), the two lists are disjoint, and their
    union is exactly the frozen M8 ten-parameter set (no new parameter).
    """

    assert DENSITY_ONLY_PARAMETERS == PINNED_DENSITY_ONLY
    assert SUPPORT_ALTERING_PARAMETERS == PINNED_SUPPORT_ALTERING
    assert not set(DENSITY_ONLY_PARAMETERS) & set(SUPPORT_ALTERING_PARAMETERS)
    assert (
        set(DENSITY_ONLY_PARAMETERS) | set(SUPPORT_ALTERING_PARAMETERS)
    ) == set(FROZEN_M8_PARAMETERS)

    record = _record(0)
    for parameter in PINNED_DENSITY_ONLY:
        assert reweight_policy(record, parameter) == REWEIGHT, parameter


# ---------------------------------------------------------------------------
# test_reweight_policy_support_altering
# ---------------------------------------------------------------------------


def test_reweight_policy_support_altering() -> None:
    """Each pinned support-altering parameter returns ``"regenerate"``."""

    record = _record(0)
    for parameter in PINNED_SUPPORT_ALTERING:
        assert reweight_policy(record, parameter) == REGENERATE, parameter


# ---------------------------------------------------------------------------
# test_reweight_policy_unknown_parameter
# ---------------------------------------------------------------------------


def test_reweight_policy_unknown_parameter() -> None:
    """An unknown parameter returns ``"regenerate"`` (conservative default)."""

    record = _record(0)
    assert reweight_policy(record, "nonexistent.parameter") == REGENERATE
    # The declared-but-unfrozen threshold power is support-altering and is
    # not in either card list, so the conservative default applies.
    assert reweight_policy(record, "continuum.strange_threshold_power") == REGENERATE
    with pytest.raises(ValueError):
        reweight_policy(record, "")


# ---------------------------------------------------------------------------
# test_cache_write_read_roundtrip
# ---------------------------------------------------------------------------


def test_cache_write_read_roundtrip(tmp_path: Path) -> None:
    """Three records written in shuffled order read back by event_index.

    The card's 12 fields are pinned exactly, the record is frozen, and
    the roundtrip preserves every field, including ``config_hash``,
    ``lib_fingerprint``, and ``substream_seeds``.
    """

    assert RECORD_FIELD_NAMES == PINNED_RECORD_FIELDS
    assert dataclasses.is_dataclass(LatentEventRecord)
    frozen_probe = _record(1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen_probe.event_index = 1  # type: ignore[misc]

    config_hash = config_hash_for({"beam_momentum": 158.0, "production_mode": "regge"})
    lib_fingerprint = default_lib_fingerprint()
    records = [
        _record(7, config_hash=config_hash, lib_fingerprint=lib_fingerprint),
        _record(0, config_hash=config_hash, lib_fingerprint=lib_fingerprint),
        _record(3, config_hash=config_hash, lib_fingerprint=lib_fingerprint),
    ]
    cache = LatentEventCache(str(tmp_path / "run"))
    cache.write(records)

    cache_dir = tmp_path / "run" / "latent_cache"
    assert cache_dir.is_dir()
    for record in records:
        assert (cache_dir / f"{cache.key_for(record)}.json").is_file()

    loaded = cache.read()
    ordered = sorted(records, key=lambda record: record.event_index)
    assert [record.event_index for record in loaded] == [0, 3, 7]
    assert loaded == ordered
    assert sorted(loaded, key=lambda record: record.event_index) == loaded
    for written, original in zip(loaded, ordered):
        assert written.config_hash == original.config_hash
        assert written.lib_fingerprint == original.lib_fingerprint
        assert written.substream_seeds == original.substream_seeds
        assert list(written.substream_seeds) == list(original.substream_seeds)
    # Writing the same records again is idempotent.
    cache.write(records)
    assert cache.read() == loaded


# ---------------------------------------------------------------------------
# test_cache_key_deterministic
# ---------------------------------------------------------------------------


def test_cache_key_deterministic(tmp_path: Path) -> None:
    """The key is the sha256 of the canonical JSON; stable per record.

    Two calls with the same record return the same key; a mutated record
    returns a different key; and the key equals the sha256 of the
    canonical serialization (sort-keys compact JSON of the record
    payload).
    """

    record = _record(5)
    cache = LatentEventCache(str(tmp_path))
    key = cache.key_for(record)
    assert key == LatentEventCache(str(tmp_path / "other")).key_for(record)

    canonical = json.dumps(
        record_payload(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert key == expected
    assert len(key) == 64
    assert key == key.lower()
    int(key, 16)  # hex parseable

    mutated = _record(5, generation_weight=0.98764)
    assert cache.key_for(mutated) != key

    cache = LatentEventCache(str(tmp_path / "run"))
    cache.write([record])
    assert (Path(cache.cache_dir) / f"{key}.json").is_file()


# ---------------------------------------------------------------------------
# test_cache_key_includes_config_hash
# ---------------------------------------------------------------------------


def test_cache_key_includes_config_hash(tmp_path: Path) -> None:
    """Two records differing only in ``config_hash`` produce different keys.

    A configuration change invalidates the cache: the record's key is
    recomputed from a second, genuinely different configuration hash.
    """

    base_config = {"beam_momentum": 158.0, "production_mode": "regge"}
    changed_config = {"beam_momentum": 158.0, "production_mode": "regge",
                      "continuum": {"pair_strength": 0.08}}
    first = _record(2, config_hash=config_hash_for(base_config))
    second = _record(2, config_hash=config_hash_for(changed_config))
    assert first.config_hash != second.config_hash
    cache = LatentEventCache(str(tmp_path))
    assert cache.key_for(first) != cache.key_for(second)


# ---------------------------------------------------------------------------
# test_cache_key_includes_lib_fingerprint
# ---------------------------------------------------------------------------


def test_cache_key_includes_lib_fingerprint(tmp_path: Path) -> None:
    """Two records differing only in ``lib_fingerprint`` produce different keys.

    A code change invalidates the cache: the second fingerprint is a
    valid sha256 digest distinct from the installed module hash.
    """

    installed = default_lib_fingerprint()
    changed = hashlib.sha256(b"changed generation code").hexdigest()
    assert installed != changed
    first = _record(2, lib_fingerprint=installed)
    second = _record(2, lib_fingerprint=changed)
    cache = LatentEventCache(str(tmp_path))
    assert cache.key_for(first) != cache.key_for(second)
