"""Coverage extension for the latent-event cache and reweight policy.

Round 1, card 6 (``TASK-20260911-r1-latent-event-cache``).
``tests/test_event_cache.py`` holds the seven card-named deterministic
tests.  This file extends coverage of the same committed behavior
without touching that file:

- reweight-policy boundaries beyond the pinned lists: whitespace and
  case variants of density-only names are strict-name matches and take
  the conservative default (the committed policy does not normalize
  names); empty, non-string, and non-record inputs are rejected loudly
  (``ValueError`` / ``TypeError``) before the default applies;
- ``LatentEventRecord.__post_init__`` invariants: strict non-negative
  integer ``event_index`` (booleans, floats, strings, and ``None``
  rejected), non-empty ``topology_identity``, 64-character lowercase
  hex ``config_hash`` and ``lib_fingerprint`` (length, character set,
  and type), sequence fields coerced from lists to tuples, non-finite
  floats and non-numeric entries rejected, ``bin_mapping`` /
  ``substream_seeds`` value strictness with string-key coercion, and
  the committed ``float()`` leniency of ``generation_weight`` (numeric
  strings and booleans normalize);
- committed observation pinned: ``substream_seeds`` keys are *not*
  checked against the card 5 closed ``SUBSTREAM_LABELS`` set; a label
  outside the set is silently accepted (closedness is enforced at the
  derivation, and the module docstring assigns per-record validity
  checks to the caller);
- cache mechanics beyond the roundtrip: five records written in
  shuffled order read back in ``event_index`` order; a double write is
  byte-idempotent (same keys, no duplicate files); a second write of a
  subset is append-only (no deletions, no corruption of the earlier
  files); a missing or empty cache directory reads as a cache miss; a
  stray non-``.json`` file is ignored by ``read``; a file edited after
  write fails ``read`` with the key-mismatch ``ValueError``;
- ``key_for`` sensitivity to all 12 record fields (each field mutated
  one at a time via ``dataclasses.replace``; every mutation changes the
  key and the 13 keys are pairwise distinct);
- canonical-serialization stability: the key is invariant to dict-key
  permutation of ``bin_mapping`` / ``substream_seeds`` (content-equal
  records key identically).  Committed observation pinned alongside:
  the on-disk bytes are *not* permutation-stable, because ``write``
  serializes with ``json.dump(indent=2)`` without ``sort_keys``; the
  same key is what keeps the store consistent (same file path,
  content-equal read-back);
- on-disk JSON schema: one ``<key>.json`` file per record, each a
  single ``json.load``-able document with exactly the top-level
  ``schema``/``key``/``record`` payload, the pinned schema tag, the
  12-field record, and the card 5 trailing-newline convention; the
  directory holds no ``.npy`` or pickle files (the pickle-free
  ``data_io`` convention);
- the ``record_from_payload`` contract: exact top-level and record key
  sets, the pinned schema tag, ``ValueError`` on a foreign schema and
  on extra or missing fields;
- fingerprint helpers: ``config_hash_for`` is stable across
  content-equivalent dict constructions (including nested key order)
  and differs for different content; it matches the
  ``staged_evaluation._canonical_config_hash`` convention.
  ``default_lib_fingerprint`` is a 64-character lowercase hex digest,
  stable across in-process calls and across a fresh interpreter (the
  committed implementation is a pure function of the fingerprinted
  module source bytes, with no import-time state).

Committed behavior noted, not fixed:

- An empty or non-string ``changed_parameter`` raises ``ValueError``
  rather than returning ``"regenerate"``.  The card's conservative
  default applies to unknown but well-formed names; the committed
  function docstring documents the strict rejection of empty and
  non-string parameters.
- ``write``'s docstring claims "identical records produce identical
  bytes".  Two content-equal records whose dict fields differ only in
  insertion order are ``==``-equal yet serialize to different bytes
  (under the same key).  Cache identity is the key, which is
  permutation-stable, so this is a documentation tension, not a
  functional defect.

Hermetic: no event generation (hand-constructed records only), no
network, no new config files.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from na49_mixing.event_cache import (
    RECORD_FIELD_NAMES,
    RECORD_SCHEMA,
    REGENERATE,
    REWEIGHT,
    LatentEventCache,
    LatentEventRecord,
    config_hash_for,
    default_lib_fingerprint,
    record_from_payload,
    record_payload,
    reweight_policy,
)
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams

NESTED_ROOT = Path(__file__).resolve().parents[1]

# Fixed 64-character lowercase hex digests for the field-mutation and
# permutation tests (no fingerprint computation per record).
_ALT_CONFIG_HASH = "c" * 64
_ALT_LIB_FINGERPRINT = "d" * 64

# Module-level fingerprints, computed once at import: both helpers are
# pure (canonical JSON hash; module source bytes), so one computation
# serves every record in this file.
_CONFIG_HASH = config_hash_for(
    {"beam_momentum": 158.0, "production_mode": "regge", "continuum": {}}
)
_LIB_FINGERPRINT = default_lib_fingerprint()


def _record(event_index: int = 0, **overrides) -> LatentEventRecord:
    """One hand-constructed record; ``overrides`` replace any field.

    ``topology_identity`` is held constant so that a single-field
    ``event_index`` override is the only difference for the key
    sensitivity tests.  The ``substream_seeds`` values come from the
    pure card 5 derivation (a SeedSequence computation, no generation
    stream), matching the committed seven-test file.  When
    ``event_index`` is deliberately invalid (the validation tests),
    fixed values are used instead so that the card 5 derivation does
    not reject the index before the record's ``__post_init__`` does.
    """

    if (
        isinstance(event_index, int)
        and not isinstance(event_index, bool)
        and event_index >= 0
    ):
        substream_seeds = dict(derive_substreams(20260831, 0, event_index).streams)
    else:
        substream_seeds = {"topology_selection": 1, "transport": 2}
    base = dict(
        event_index=event_index,
        topology_identity="pp:2source:exchange_m=1.242879",
        exchange_masses=(1.242879, 1.310231),
        exchange_transfers=(0.113, 0.113),
        # Real channel_family labels (na49_mixing.continuum_channels);
        # no new channel family.
        candidate_channels=("continuum_nonstrange", "associated_KLambda", "baryon_pair"),
        phase_space_points=((0.4321, 0.5123, 0.0811), (0.3978, 0.4811, 0.0791)),
        angular_variables=(0.3215, -0.1178, 1.0423),
        bin_mapping={"pi_minus": 12, "pi_plus": 44},
        generation_weight=0.98765,
        substream_seeds=substream_seeds,
        config_hash=_CONFIG_HASH,
        lib_fingerprint=_LIB_FINGERPRINT,
    )
    base.update(overrides)
    return LatentEventRecord(**base)


# ---------------------------------------------------------------------------
# Item 1: policy boundaries beyond the pinned lists
# ---------------------------------------------------------------------------


def test_policy_verdict_strings() -> None:
    """The verdict strings and the schema tag are pinned verbatim."""

    assert REWEIGHT == "reweight"
    assert REGENERATE == "regenerate"
    assert RECORD_SCHEMA == "na49_mixing.latent_event_record/1"


@pytest.mark.parametrize(
    "parameter",
    [
        " continuum.pair_strength",  # leading whitespace
        "continuum.pair_strength ",  # trailing whitespace
        "continuum. pair_strength",  # internal whitespace
        "Continuum.pair_strength",  # case variant
        "CONTINUUM.PAIR_STRENGTH",  # case variant
        "continuum.pair_strength.extra",  # extra dotted component
        "continuumstrange_pair_strength",  # missing dot
    ],
)
def test_reweight_policy_name_variants_regenerate(parameter: str) -> None:
    """Whitespace and case variants regenerate (strict name matching).

    The committed policy matches names against
    ``DENSITY_ONLY_PARAMETERS`` without normalization: a name with
    leading or trailing whitespace, a different case, or a different
    dotted structure is an unknown name and takes the conservative
    default.
    """

    assert reweight_policy(_record(0), parameter) == REGENERATE, parameter


@pytest.mark.parametrize("bad", ["", None, 42, 0.5])
def test_reweight_policy_rejects_empty_or_non_string_parameter(bad) -> None:
    """An empty or non-string parameter raises ``ValueError``.

    Committed behavior: the conservative default applies to unknown but
    well-formed names; an empty or non-string parameter is rejected
    loudly rather than silently defaulted (the card wording "unknown
    parameter returns regenerate" is implemented as strict name
    validation plus the conservative default).
    """

    with pytest.raises(ValueError, match="changed_parameter"):
        reweight_policy(_record(0), bad)


def test_reweight_policy_rejects_non_record() -> None:
    """The first argument must be a ``LatentEventRecord``."""

    with pytest.raises(TypeError, match="LatentEventRecord"):
        reweight_policy(None, "continuum.pair_strength")


def test_policy_verdict_is_record_independent() -> None:
    """The classification depends on the parameter name only.

    The record is accepted for signature compatibility with the v2
    builder hook; two different records give the same verdict for the
    same name.
    """

    first = _record(0)
    second = _record(1)
    assert first != second
    for parameter in ("continuum.pair_strength", "regge.edge_power"):
        assert reweight_policy(first, parameter) == reweight_policy(second, parameter)


# ---------------------------------------------------------------------------
# Item 2: LatentEventRecord __post_init__ invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, True, False, 2.0, 3.5, "2", None])
def test_record_rejects_bad_event_index(bad) -> None:
    """``event_index`` is a strict non-negative integer.

    Booleans (which subclass ``int``), floats (including integer-valued
    ones), strings, and ``None`` are rejected with a ``ValueError``
    naming the field.
    """

    with pytest.raises(ValueError, match="event_index"):
        _record(event_index=bad)


def test_record_rejects_bad_topology_identity() -> None:
    """``topology_identity`` is a non-empty string."""

    with pytest.raises(ValueError, match="topology_identity"):
        _record(topology_identity="")
    with pytest.raises(ValueError, match="topology_identity"):
        _record(topology_identity=42)


@pytest.mark.parametrize(
    "bad",
    [
        "a" * 63,  # too short
        "a" * 65,  # too long
        "a" * 63 + "A",  # uppercase hex is not accepted
        "a" * 63 + "g",  # non-hex character
        12345,  # not a string
    ],
)
@pytest.mark.parametrize("field", ["config_hash", "lib_fingerprint"])
def test_record_rejects_malformed_hex_digests(field: str, bad) -> None:
    """``config_hash`` / ``lib_fingerprint`` are 64-char lowercase hex.

    Length, character set, and type are each enforced with a
    ``ValueError`` naming the field.
    """

    with pytest.raises(ValueError, match=field):
        _record(**{field: bad})


def test_record_coerces_sequence_fields_from_lists() -> None:
    """Sequence fields accept lists and coerce them to tuples.

    The committed ``__post_init__`` iterates and converts; a list where
    a tuple is required is accepted (coerced), not rejected.
    """

    record = _record(
        exchange_masses=[1.242879, 1.310231],
        exchange_transfers=[0.113],
        angular_variables=[0.3215],
        phase_space_points=[[0.4321, 0.5123, 0.0811]],
        candidate_channels=["continuum_nonstrange"],
    )
    assert type(record.exchange_masses) is tuple
    assert type(record.exchange_transfers) is tuple
    assert type(record.angular_variables) is tuple
    assert type(record.phase_space_points) is tuple
    assert type(record.phase_space_points[0]) is tuple
    assert type(record.candidate_channels) is tuple
    assert record.exchange_masses == (1.242879, 1.310231)


@pytest.mark.parametrize(
    "field", ["exchange_masses", "exchange_transfers", "angular_variables"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_record_rejects_nonfinite_sequence_values(field: str, bad: float) -> None:
    """Non-finite floats in the float sequence fields are rejected."""

    with pytest.raises(ValueError, match=field):
        _record(**{field: (bad,)})


def test_record_rejects_nonfinite_phase_space_points() -> None:
    """A non-finite entry anywhere in a phase-space point is rejected."""

    with pytest.raises(ValueError, match="phase_space_points"):
        _record(phase_space_points=((0.1, float("nan"), 0.3),))


def test_record_rejects_nonnumeric_sequence_entry() -> None:
    """A non-numeric sequence entry is rejected at coercion."""

    with pytest.raises(ValueError):
        _record(exchange_masses=("abc",))


def test_record_rejects_nonfinite_generation_weight() -> None:
    """``generation_weight`` must be finite after coercion."""

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="generation_weight"):
            _record(generation_weight=bad)


def test_record_coerces_generation_weight_via_float() -> None:
    """Committed leniency: ``generation_weight`` passes through ``float()``.

    A numeric string and a boolean normalize to the float the dataclass
    holds (``True`` is ``1.0``); pin the committed coercion rather than
    a strict-int style rejection.
    """

    assert _record(generation_weight="0.5").generation_weight == 0.5
    assert _record(generation_weight=True).generation_weight == 1.0


@pytest.mark.parametrize(
    "values", [{"pi_minus": -1}, {"pi_minus": True}, {"pi_minus": "12"}]
)
def test_record_rejects_bad_bin_mapping_values(values: dict) -> None:
    """``bin_mapping`` values are strict non-negative integers."""

    with pytest.raises(ValueError, match="bin_mapping"):
        _record(bin_mapping=values)


def test_record_coerces_bin_mapping_keys_to_strings() -> None:
    """``bin_mapping`` keys are coerced to strings; empty mapping is valid."""

    record = _record(bin_mapping={1: 12, "pi_plus": 44})
    assert record.bin_mapping == {"1": 12, "pi_plus": 44}
    assert list(record.bin_mapping) == ["1", "pi_plus"]
    assert _record(bin_mapping={}).bin_mapping == {}


@pytest.mark.parametrize(
    "seeds", [{"transport": -1}, {"transport": "1"}, {"transport": True}]
)
def test_record_rejects_bad_substream_seed_values(seeds: dict) -> None:
    """``substream_seeds`` stream ids are strict non-negative integers."""

    with pytest.raises(ValueError, match="substream_seeds"):
        _record(substream_seeds=seeds)


def test_record_substream_seed_labels_not_checked_against_closed_set() -> None:
    """Committed observation: an out-of-set seed label is silently accepted.

    The record does not validate ``substream_seeds`` keys against the
    card 5 closed ``SUBSTREAM_LABELS`` set (unlike
    ``RngSubstreams.stream_for``, which raises ``KeyError``); closedness
    is enforced at the derivation, and the module docstring assigns
    per-record validity checks to the caller.  Pin the committed
    accept behavior.
    """

    assert "parton" not in SUBSTREAM_LABELS
    record = _record(substream_seeds={"parton": 7})
    assert record.substream_seeds == {"parton": 7}
    assert _record(substream_seeds={}).substream_seeds == {}


# ---------------------------------------------------------------------------
# Item 3: cache mechanics beyond the roundtrip
# ---------------------------------------------------------------------------


def test_cache_reads_in_event_index_order_after_shuffled_write(tmp_path: Path) -> None:
    """Five records written in shuffled order read back by ``event_index``."""

    cache = LatentEventCache(str(tmp_path / "run"))
    indices = [9, 0, 4, 7, 2]
    cache.write([_record(index) for index in indices])
    loaded = cache.read()
    assert [record.event_index for record in loaded] == sorted(indices)
    assert loaded == sorted(
        (_record(index) for index in indices), key=lambda record: record.event_index
    )


def test_cache_double_write_is_idempotent(tmp_path: Path) -> None:
    """Writing the same records twice: same keys, no duplicates, same bytes."""

    cache = LatentEventCache(str(tmp_path / "run"))
    records = [_record(index) for index in (3, 1, 5)]
    cache.write(records)
    cache_dir = Path(cache.cache_dir)
    before = {name.name: name.read_bytes() for name in sorted(cache_dir.iterdir())}
    assert len(before) == 3

    cache.write(records)
    after = {name.name: name.read_bytes() for name in sorted(cache_dir.iterdir())}
    assert before == after, "a second identical write must be byte-idempotent"
    keys = {cache.key_for(record) for record in records}
    assert set(before) == {f"{key}.json" for key in keys}
    assert cache.read() == sorted(records, key=lambda record: record.event_index)


def test_cache_subset_rewrite_leaves_existing_files(tmp_path: Path) -> None:
    """A second write of a subset is append-only: no deletions, no corruption.

    The committed ``write`` adds (or overwrites) the files for the
    records it is given and never removes files for records it was not
    given; the directory still reads back as the full original set, in
    ``event_index`` order.
    """

    cache = LatentEventCache(str(tmp_path / "run"))
    full = [_record(index) for index in (0, 2, 4, 6, 8)]
    cache.write(full)
    cache.write(full[:2])  # a subset of the earlier write

    names = {name.name for name in Path(cache.cache_dir).iterdir()}
    expected = {f"{cache.key_for(record)}.json" for record in full}
    assert names == expected, "the earlier files must survive a subset rewrite"
    assert [record.event_index for record in cache.read()] == [0, 2, 4, 6, 8]


def test_cache_missing_and_empty_directories_are_cache_misses(tmp_path: Path) -> None:
    """An absent or empty cache directory reads back as an empty list."""

    missing = LatentEventCache(str(tmp_path / "never_written"))
    assert not Path(missing.cache_dir).is_dir()
    assert missing.read() == []

    empty = LatentEventCache(str(tmp_path / "empty"))
    Path(empty.cache_dir).mkdir(parents=True)
    assert empty.read() == []


def test_cache_read_ignores_non_json_files(tmp_path: Path) -> None:
    """A stray non-``.json`` file in the directory is ignored by ``read``."""

    cache = LatentEventCache(str(tmp_path / "run"))
    cache.write([_record(0)])
    (Path(cache.cache_dir) / "notes.txt").write_text("not a cache file", encoding="utf-8")
    loaded = cache.read()
    assert [record.event_index for record in loaded] == [0]


def test_cache_read_rejects_edited_record(tmp_path: Path) -> None:
    """A file edited after write fails ``read`` with the key-mismatch error.

    Every stored file is re-keyed from its contents; a stale or edited
    record is a loud error, never a silent reuse.
    """

    cache = LatentEventCache(str(tmp_path / "run"))
    record = _record(0)
    cache.write([record])
    path = Path(cache.cache_dir) / f"{cache.key_for(record)}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["record"]["generation_weight"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="key mismatch"):
        cache.read()


# ---------------------------------------------------------------------------
# Item 4: key sensitivity to all 12 fields
# ---------------------------------------------------------------------------

# One distinct mutation per field (each value differs from the base
# record's field and passes the record validation).
_FIELD_MUTATIONS = {
    "event_index": 99,
    "topology_identity": "pp:2source:exchange_m=9.999999",
    "exchange_masses": (1.5, 2.5),
    "exchange_transfers": (0.9,),
    "candidate_channels": ("h2_resonance",),
    "phase_space_points": ((0.9, 0.8, 0.7),),
    "angular_variables": (1.2345,),
    "bin_mapping": {"pi_minus": 99},
    "generation_weight": 0.11111,
    "substream_seeds": {"transport": 99},
    "config_hash": _ALT_CONFIG_HASH,
    "lib_fingerprint": _ALT_LIB_FINGERPRINT,
}


def test_cache_key_includes_every_record_field(tmp_path: Path) -> None:
    """Mutating each of the 12 fields one at a time changes the key.

    The committed seven-test file pins ``config_hash`` and
    ``lib_fingerprint`` sensitivity; here every remaining field is
    mutated in isolation via ``dataclasses.replace`` and the 13 keys
    (base plus 12 mutations) are pairwise distinct.
    """

    cache = LatentEventCache(str(tmp_path))
    base = _record(0)
    keys = {cache.key_for(base)}
    for field in RECORD_FIELD_NAMES:
        assert field in _FIELD_MUTATIONS
        mutated = dataclasses.replace(base, **{field: _FIELD_MUTATIONS[field]})
        assert mutated != base, field
        assert cache.key_for(mutated) != cache.key_for(base), field
        keys.add(cache.key_for(mutated))
    assert len(keys) == len(RECORD_FIELD_NAMES) + 1


# ---------------------------------------------------------------------------
# Item 5: canonical serialization and on-disk JSON schema
# ---------------------------------------------------------------------------


def test_cache_key_stable_under_dict_key_permutation(tmp_path: Path) -> None:
    """The key is invariant to dict-key permutation of the dict fields.

    The canonical keying JSON is serialized with ``sort_keys``, so two
    records equal in content but with permuted ``bin_mapping`` /
    ``substream_seeds`` insertion order must key identically.
    """

    cache = LatentEventCache(str(tmp_path))
    first = _record(
        0,
        bin_mapping={"pi_minus": 12, "pi_plus": 44, "pi0": 7},
        substream_seeds={"transport": 1, "decay_channel": 2, "spectral_state": 3},
    )
    second = _record(
        0,
        bin_mapping={"pi0": 7, "pi_plus": 44, "pi_minus": 12},
        substream_seeds={"spectral_state": 3, "transport": 1, "decay_channel": 2},
    )
    assert first == second
    assert cache.key_for(first) == cache.key_for(second)


def test_cache_file_bytes_depend_on_dict_insertion_order(tmp_path: Path) -> None:
    """Committed observation: the key is permutation-stable, the bytes are not.

    ``write`` serializes the payload with ``json.dump(indent=2)``
    without ``sort_keys``, so two content-equal records with permuted
    dict fields land in the *same* cache file path (same key) with
    *different* bytes; a rewrite from the second construction
    overwrites the first's bytes, and the read-back stays content-equal
    either way.
    """

    first = _record(
        0,
        bin_mapping={"pi_minus": 12, "pi_plus": 44},
        substream_seeds={"transport": 1, "decay_channel": 2},
    )
    second = _record(
        0,
        bin_mapping={"pi_plus": 44, "pi_minus": 12},
        substream_seeds={"decay_channel": 2, "transport": 1},
    )
    cache_a = LatentEventCache(str(tmp_path / "a"))
    cache_b = LatentEventCache(str(tmp_path / "b"))
    cache_a.write([first])
    cache_b.write([second])

    key_a = cache_a.key_for(first)
    key_b = cache_b.key_for(second)
    assert key_a == key_b
    bytes_a = (Path(cache_a.cache_dir) / f"{key_a}.json").read_bytes()
    bytes_b = (Path(cache_b.cache_dir) / f"{key_b}.json").read_bytes()
    assert bytes_a != bytes_b, (
        "the on-disk record dict keeps insertion order (no sort_keys)"
    )

    # Same key, same file path: the second construction overwrites the
    # first's bytes, and the read-back is content-equal either way.
    cache_a.write([second])
    assert (Path(cache_a.cache_dir) / f"{key_a}.json").read_bytes() == bytes_b
    assert cache_a.read() == [first]


def test_cache_files_are_single_valid_json_documents(tmp_path: Path) -> None:
    """One ``<key>.json`` file per record; each a single ``json.load`` document.

    The on-file payload carries exactly the top-level ``schema`` /
    ``key`` / ``record`` keys, the pinned schema tag, the 12-field
    record, a key matching the file name, and the card 5
    trailing-newline convention.
    """

    cache = LatentEventCache(str(tmp_path / "run"))
    records = [_record(index) for index in (5, 1, 3)]
    cache.write(records)

    names = sorted(Path(cache.cache_dir).iterdir())
    keys = {cache.key_for(record) for record in records}
    assert [name.name for name in names] == sorted(f"{key}.json" for key in keys)
    for name in names:
        text = name.read_text(encoding="utf-8")
        assert text.endswith("\n"), "card 5 trailing-newline convention"
        payload = json.loads(text)  # a single valid JSON document
        assert set(payload) == {"schema", "key", "record"}
        assert payload["schema"] == RECORD_SCHEMA
        assert payload["key"] == name.stem
        assert set(payload["record"]) == set(RECORD_FIELD_NAMES)


def test_cache_directory_is_pickle_free(tmp_path: Path) -> None:
    """No ``.npy`` or pickle files appear under ``latent_cache/``.

    The pickle-free ``data_io`` convention: after a write, every file
    in the cache directory is a ``.json`` document.
    """

    cache = LatentEventCache(str(tmp_path / "run"))
    cache.write([_record(index) for index in range(4)])
    names = [name.name for name in Path(cache.cache_dir).iterdir()]
    assert {Path(name).suffix for name in names} == {".json"}, names
    for name in names:
        assert not name.endswith(".npy")
        assert not name.endswith((".pkl", ".pickle"))


def test_record_from_payload_contract(tmp_path: Path) -> None:
    """The stored payload contract: exact top-level and record key sets.

    A valid payload rebuilds the record; a foreign schema tag, an
    extra or missing record field, or a missing top-level key raises
    ``ValueError`` naming the problem.
    """

    record = _record(0)
    payload = {
        "schema": RECORD_SCHEMA,
        "key": "e" * 64,
        "record": record_payload(record),
    }
    assert record_from_payload(payload) == record

    foreign = dict(payload, schema="na49_mixing.other_schema/9")
    with pytest.raises(ValueError, match="schema"):
        record_from_payload(foreign)

    extra = dict(payload)
    extra["record"] = dict(payload["record"], extra_field=1)
    with pytest.raises(ValueError, match="exactly the fields"):
        record_from_payload(extra)

    missing_top = dict(payload)
    del missing_top["key"]
    with pytest.raises(ValueError, match="schema/key/record"):
        record_from_payload(missing_top)


# ---------------------------------------------------------------------------
# Item 6: fingerprint helpers
# ---------------------------------------------------------------------------


def test_config_hash_for_stable_across_construction_order() -> None:
    """Same content in different construction order hashes identically.

    The canonical config hash (sort-keys compact JSON, sha256) is
    stable under dict-key permutation at every nesting level, differs
    for different content, and matches the
    ``staged_evaluation._canonical_config_hash`` convention.
    """

    config_a = {
        "beam_momentum": 158.0,
        "production_mode": "regge",
        "continuum": {"pair_strength": 0.084, "match_mass": 2.68},
    }
    config_b = {
        "continuum": {"match_mass": 2.68, "pair_strength": 0.084},
        "production_mode": "regge",
        "beam_momentum": 158.0,
    }
    # Dict == ignores insertion order; the unsorted canonical JSON
    # distinguishes the two construction orders.
    assert json.dumps(config_a) != json.dumps(config_b)
    hash_a = config_hash_for(config_a)
    assert config_hash_for(config_b) == hash_a
    assert len(hash_a) == 64
    assert hash_a == hash_a.lower()
    assert all(ch in "0123456789abcdef" for ch in hash_a)
    assert hash_a != config_hash_for(
        {"beam_momentum": 158.0, "production_mode": "regge"}
    )

    from na49_mixing.staged_evaluation import _canonical_config_hash

    assert hash_a == _canonical_config_hash(config_a)


def test_default_lib_fingerprint_in_process_stable() -> None:
    """Two in-process calls agree; 64-character lowercase hex digest.

    Cross-process stability is pinned separately; here the format and
    the in-process stability the cache relies on are what is asserted.
    """

    fingerprint_1 = default_lib_fingerprint()
    fingerprint_2 = default_lib_fingerprint()
    assert fingerprint_1 == fingerprint_2
    assert len(fingerprint_1) == 64
    assert all(ch in "0123456789abcdef" for ch in fingerprint_1)
    # A module byte-string hash, not a config hash: it differs from
    # config_hash_for for any configuration.
    assert fingerprint_1 != config_hash_for({})
    # The module-level record helper uses the same value.
    assert _LIB_FINGERPRINT == fingerprint_1


def test_default_lib_fingerprint_stable_across_processes() -> None:
    """The fingerprint is a pure function of the module source bytes.

    The committed implementation reads each fingerprinted module's
    file bytes at call time (no import-time state), so a fresh
    interpreter resolves the same digest.  Pinned with a bounded
    timeout; an import failure or digest drift fails the test.
    """

    code = (
        "from na49_mixing.event_cache import default_lib_fingerprint; "
        "print(default_lib_fingerprint())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(NESTED_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == default_lib_fingerprint()
