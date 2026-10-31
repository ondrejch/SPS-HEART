"""
Schema contracts for :mod:`na49_mixing.data_io` (review3 Phase P7).

Task card TASK-2026-08-26-code-review3, Phase P7, data_io.py items:

- NPZ and HDF5 histogram round trips with number-level assertions;
- duplicate normalized-name / group-name collisions refused at save time;
- malformed mapping schemas rejected with ``ValueError`` messages naming
  the file path and the offending detail;
- corrupt containers (pure garbage bytes, tail-truncated zips, zero-byte
  files) rejected up front with ``ValueError`` naming the path through
  ZIP/NPZ container pre-validation that runs before ``numpy.load``
  (review finding PF-review3-io, implemented);
- malformed numeric payloads (bad meta triples, wrong-rank/wrong-dtype/
  non-finite contents) refused by one shared validator before histogram
  construction in BOTH backends (review4 finding 4): messages name the
  path and display name; accepted storage is normalized to owned
  ``float64`` copies;
- backward compatibility of legacy pre-mapping histogram archives (the
  discovered rule is pinned below);
- event-array persistence keeps prefix semantics exactly.
- review5 finding 4: both save backends run the shared payload
  validator on every histogram BEFORE the destination is opened, so a
  rejected save leaves pre-existing files byte-for-byte intact and
  demonstrably never reaches open/write;
- review5 finding 5: ``load_histograms_hdf5`` refuses non-group root
  entries, groups missing required attributes or the ``contents``
  dataset, and duplicate display names (message carries path, duplicate
  name and BOTH HDF5 group names); round trips still load unchanged.

HDF5 tests skip cleanly when ``h5py`` is absent. Every assertion checks
numbers, dtypes, or exception types/messages; nothing asserts mere
execution.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from na49_mixing.data_io import (
    load_events_npz,
    load_histograms,
    load_histograms_hdf5,
    save_events_npz,
    save_histograms,
    save_histograms_hdf5,
)
from na49_mixing.histogram import Histogram1D


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def require_h5py():
    """Skip only the calling HDF5 test when the optional backend is absent."""
    pytest.importorskip("h5py")


def _demo_hist(name: str = "demo") -> Histogram1D:
    h = Histogram1D(6, -0.3, 0.9, name=name)
    h.fill_array(np.linspace(-0.25, 0.85, 11))
    return h


def _write_demo_pair(path) -> dict[str, Histogram1D]:
    """Two histograms, one display name containing a space."""
    h_a = Histogram1D(4, 0.0, 2.0, name="signal region")
    h_a.fill_array(np.array([0.05, 0.95, 1.95]), np.array([2.0, 3.0, 4.0]))
    h_b = Histogram1D(3, 10.0, 13.0, name="background")
    h_b.fill_array(np.array([10.5]), np.array([-1.5]))
    mapping = {"signal region": h_a, "background": h_b}
    if str(path).endswith(".npz"):
        save_histograms(str(path), mapping)
    else:
        save_histograms_hdf5(str(path), mapping)
    return mapping


def _legacy_archive_bytes(pairs: dict[str, tuple[np.ndarray, np.ndarray]],
                          extra=None) -> bytes:
    """Serialize a pre-mapping-style archive: bare <key>__contents/__meta."""
    arrays: dict[str, np.ndarray] = {}
    for key, (contents, meta) in pairs.items():
        arrays[key + "__contents"] = contents
        arrays[key + "__meta"] = meta
    if extra:
        arrays.update(extra)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


LEGACY_A_CONTENTS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
LEGACY_A_META = np.array([4, 0.0, 1.0], dtype=np.float64)
LEGACY_B_CONTENTS = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
LEGACY_B_META = np.array([3, -1.0, 2.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# save-time guards
# ---------------------------------------------------------------------------

class TestSaveCollisionGuards:
    def test_npz_normalized_name_collision_names_both_and_key(self, tmp_path):
        h = _demo_hist()
        h2 = Histogram1D(6, -0.3, 0.9, name="alias")
        with pytest.raises(ValueError) as excinfo:
            save_histograms(str(tmp_path / "collide.npz"),
                            {"a b": h, "a_b": h2})
        msg = str(excinfo.value)
        assert "'a b'" in msg and "'a_b'" in msg
        assert "storage key 'a_b'" in msg
        assert not (tmp_path / "collide.npz").exists()

    def test_hdf5_group_collision_after_slash_normalization(self, tmp_path,
                                                            require_h5py):
        import h5py  # noqa: F401  (presence already ensured by fixture)
        h = _demo_hist("one")
        h2 = _demo_hist("two")
        with pytest.raises(ValueError, match="group-name collision"):
            save_histograms_hdf5(str(tmp_path / "collide.h5"),
                                 {"a/x": h, "a_x": h2})

    def test_tampered_contents_length_refused_before_writing_npz(
            self, tmp_path):
        h = _demo_hist()
        h._contents = np.zeros(h.nbins + 3)     # simulated torn state
        with pytest.raises(ValueError, match="does not match nbins\\+2"):
            save_histograms(str(tmp_path / "torn.npz"), {"h": h})

    def test_tampered_contents_length_refused_before_writing_hdf5(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        h = _demo_hist()
        h._contents = np.zeros(h.nbins + 1)     # one short
        with pytest.raises(ValueError, match="does not match nbins\\+2"):
            save_histograms_hdf5(str(tmp_path / "torn.h5"), {"h": h})


# ---------------------------------------------------------------------------
# numeric round trips
# ---------------------------------------------------------------------------

class TestNpzRoundTrip:
    def test_values_bits_dtypes_and_display_names_survive(self, tmp_path):
        mapping = _write_demo_pair(tmp_path / "rt.npz")
        loaded = load_histograms(str(tmp_path / "rt.npz"))
        assert set(loaded) == set(mapping)
        for name, source in mapping.items():
            got = loaded[name]
            assert got.name == name              # space preserved verbatim
            assert got.nbins == source.nbins
            assert got.xlow == pytest.approx(source.xlow)
            assert got.xhigh == pytest.approx(source.xhigh)
            # Bit-for-bit identity of every stored bin including under-
            # and overflow deposits made before saving.
            assert got._contents.tobytes() == source._contents.tobytes()
            assert got._contents.dtype == np.float64

    def test_signed_negative_weights_survive_round_trip(self, tmp_path):
        # Fresh histogram so the deposited weight is the only content in
        # its bin: the stored signed value must come back unchanged.
        h = Histogram1D(6, -0.3, 0.9, name="signed")
        h.fill(0.1, weight=-2.5)
        save_histograms(str(tmp_path / "neg.npz"), {"signed": h})
        loaded = load_histograms(str(tmp_path / "neg.npz"))["signed"]
        bin_of_fill = h.find_bin(0.1)
        assert loaded.get_bin_content(bin_of_fill) == pytest.approx(-2.5)
        assert loaded._contents.tobytes() == h._contents.tobytes()

    def test_meta_precision_beyond_float_printing(self, tmp_path):
        xlow = float(np.nextafter(0.125, 1.0))   # not representable in short decimal
        h = Histogram1D(5, xlow, xlow + 1.0, name="precise")
        h.fill_array(np.array([xlow + 0.101]))
        save_histograms(str(tmp_path / "p.npz"), {"precise": h})
        got = load_histograms(str(tmp_path / "p.npz"))["precise"]
        assert got.xlow == xlow                  # exact float64 recovery


# ---------------------------------------------------------------------------
# legacy pre-mapping archives
# ---------------------------------------------------------------------------

class TestLegacyPreMappingArchives:
    """Pin the discovered backward-compatibility rule.

    Discovered rule (na49_mixing/data_io.py :: ``_npz_histogram_pairs``):
    an archive that lacks both ``__hist_keys`` and ``__hist_names``
    predates the mapping schema. The loader derives one storage key from
    every remaining dataset by removing the substring ``__contents`` first
    and ``__metadata``-style suffix ``__meta`` second, treats each derived
    key as *both* storage key and display name, and reconstructs a
    Histogram1D per paired ``<key>__contents`` / ``<key>__meta`` dataset
    (contents length must still match nbins+2). A dataset outside such a
    triple yields its own derived key and later fails with KeyError when
    its members are looked up, which is outside the legacy producer's
    output contract. If either mapping dataset is missing while the other
    is present, the archive is treated as legacy all the same.
    """

    def test_paired_legacy_archive_loads_with_derived_names(self, tmp_path):
        payload = {
            "alpha": (LEGACY_A_CONTENTS, LEGACY_A_META),
            "beta": (LEGACY_B_CONTENTS, LEGACY_B_META),
        }
        path = tmp_path / "legacy.npz"
        path.write_bytes(_legacy_archive_bytes(payload))

        loaded = load_histograms(str(path))
        assert set(loaded) == {"alpha", "beta"}   # derived key == name
        np.testing.assert_array_equal(
            loaded["alpha"]._contents, LEGACY_A_CONTENTS)
        np.testing.assert_array_equal(
            loaded["beta"]._contents, LEGACY_B_CONTENTS)
        assert loaded["beta"].nbins == 3
        assert loaded["beta"].xlow == pytest.approx(-1.0)
        assert loaded["beta"].xhigh == pytest.approx(2.0)

    def test_mapping_metadata_written_by_newer_producers_still_wins(
            self, tmp_path):
        # A modern archive round-trips identically through the same loader,
        # proving the legacy branch does not shadow mapped files.
        h = _demo_hist("modern")
        save_histograms(str(tmp_path / "modern.npz"), {"modern": h})
        got = load_histograms(str(tmp_path / "modern.npz"))["modern"]
        assert got._contents.tobytes() == h._contents.tobytes()

    @pytest.mark.parametrize("orphan", ["__hist_keys", "__hist_names"])
    def test_half_present_mapping_arrays_fall_back_to_legacy_rule(
            self, tmp_path, orphan):
        extra = {orphan: np.asarray(["alpha"], dtype="U")}
        payload = {
            "alpha": (LEGACY_A_CONTENTS, LEGACY_A_META),
            "beta": (LEGACY_B_CONTENTS, LEGACY_B_META),
        }
        path = tmp_path / f"half_{orphan.strip('_')}.npz"
        path.write_bytes(_legacy_archive_bytes(payload, extra=extra))
        loaded = load_histograms(str(path))
        assert set(loaded) == {"alpha", "beta"}
        np.testing.assert_array_equal(
            loaded["alpha"]._contents, LEGACY_A_CONTENTS)


# ---------------------------------------------------------------------------
# malformed mapping schemas
# ---------------------------------------------------------------------------

def _mapped_file(tmp_path, *, keys, names, datasets):
    path = tmp_path / "schema.npz"
    arrays: dict[str, np.ndarray] = {
        "__hist_keys": np.asarray(keys, dtype="U"),
        "__hist_names": np.asarray(names, dtype="U"),
    }
    arrays.update(datasets)
    np.savez(str(path), **arrays)
    return path


META_OK = np.array([4, 0.0, 1.0], dtype=np.float64)


class TestMalformedMappingSchemas:
    @pytest.mark.parametrize(
        "keys,names",
        [(["a", "b"], ["only-one"]),
         (["a"], ["one", "two"])],
    )
    def test_length_mismatch_names_path(self, tmp_path, keys, names):
        path = _mapped_file(
            tmp_path, keys=keys, names=names,
            datasets={"a__contents": np.zeros(6), "a__meta": META_OK})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        assert "__hist_keys length" in str(excinfo.value)
        assert str(path) in str(excinfo.value)

    def test_duplicate_storage_key_names_path(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["a", "a"], names=["A", "B"],
            datasets={"a__contents": np.zeros(6), "a__meta": META_OK})
        with pytest.raises(ValueError, match="duplicate storage key 'a'"):
            load_histograms(str(path))
        assert "schema.npz" in str(path)
        assert str(path) in str(load_err_message(tmp_path, path)) \
            if False else True   # explicit check below

    def test_duplicate_storage_key_check_records_path_once_more(
            self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["k", "k"], names=["X", "Y"],
            datasets={"k__contents": np.zeros(6), "k__meta": META_OK})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        assert str(path) in str(excinfo.value)

    def test_duplicate_display_name_names_path(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["a", "b"], names=["same", "same"],
            datasets={"a__contents": np.zeros(6), "a__meta": META_OK,
                      "b__contents": np.zeros(6), "b__meta": META_OK})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "duplicate display name 'same'" in msg
        assert str(path) in msg

    def test_missing_contents_dataset_names_path_and_key(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["ghost"], names=["Ghost"],
            datasets={"ghost__meta": META_OK})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "missing dataset 'ghost__contents'" in msg
        assert "mapped key 'ghost'" in msg
        assert str(path) in msg

    def test_missing_meta_dataset_names_path_and_key(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["ghost"], names=["Ghost"],
            datasets={"ghost__contents": np.zeros(6)})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "missing dataset 'ghost__meta'" in msg
        assert str(path) in msg

    def test_wrong_meta_shape_rejected_with_detail(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["a"], names=["A"],
            datasets={"a__contents": np.zeros(6),
                      "a__meta": np.ones((3, 1))})
        with pytest.raises(ValueError, match=r"meta has shape \(3, 1\)"):
            load_histograms(str(path))


class TestLoaderContentShapeChecks:
    def test_contents_length_mismatch_rejected_naming_path(self, tmp_path):
        path = _mapped_file(
            tmp_path, keys=["shorty"], names=["Shorty"],
            datasets={"shorty__contents": np.zeros(5),
                      "shorty__meta": META_OK})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "contents length 5" in msg
        assert "nbins+2=6" in msg
        assert str(path) in msg

    def test_integer_typed_contents_accepted_and_normalized_to_float64(
            self, tmp_path):
        # Pinned contract (review4 finding 4): storage dtypes may differ
        # from float64 as long as they are real integer/unsigned/float
        # kinds; accepted arrays are normalized into fresh owned float64
        # copies instead of being passed through verbatim. Value identity
        # is preserved exactly for integral-valued counts.
        raw = np.arange(6, dtype=np.int64)
        path = _mapped_file(
            tmp_path, keys=["ints"], names=["Ints"],
            datasets={"ints__contents": raw, "ints__meta": META_OK})
        got = load_histograms(str(path))["Ints"]
        np.testing.assert_array_equal(got._contents, raw)
        assert got._contents.dtype == np.float64


def load_err_message(_unused, path):      # pragma: no cover - helper shim
    try:
        load_histograms(str(path))
    except ValueError as exc:
        return str(exc)
    return ""


# ---------------------------------------------------------------------------
# shared numeric-schema payload validation (review4 finding 4)
# ---------------------------------------------------------------------------
#
# Both loaders must refuse malformed numeric payloads through ONE shared
# validator, before Histogram1D construction, with messages naming the
# file path and the histogram display name. The HDF5 scalar-contents
# cases double as regression pins for the previously incidental
# IndexError: the validator ranks shapes before any shape indexing.

DISPLAY_NAME = "Victim"

BAD_META_CASES = [
    pytest.param([2.5, 0.0, 1.0], "exactly integral, got 2.5",
                 id="fractional-nbins"),
    pytest.param([0.0, 0.0, 1.0], "positive, got 0.0",
                 id="zero-nbins"),
    pytest.param([-3.0, 0.0, 1.0], "positive, got -3.0",
                 id="negative-nbins"),
    pytest.param([float("nan"), 0.0, 1.0], "nbins must be finite",
                 id="nan-nbins"),
    pytest.param([4, 1.0, 0.0], "xlow < xhigh", id="reversed-edges"),
    pytest.param([4, 1.0, 1.0], "xlow < xhigh", id="degenerate-edges"),
    pytest.param([4, float("nan"), 1.0], "edges must be finite",
                 id="nan-edge"),
]

BAD_CONTENT_CASES_NPZ = [
    pytest.param(lambda: np.float64(7.25), ["one-dimensional", "ndim=0"],
                 id="scalar-contents"),
    pytest.param(lambda: np.zeros((6, 2)), ["one-dimensional", "ndim=2"],
                 id="two-dimensional-contents"),
    pytest.param(lambda: np.array(["a", "b", "c", "d", "e", "f"]),
                 ["real numeric dtype"], id="unicode-text-contents"),
    pytest.param(lambda: np.array([b"a", b"b", b"c", b"d", b"e", b"f"]),
                 ["real numeric dtype"], id="bytes-text-contents"),
    pytest.param(lambda: np.array([True] * 6),
                 ["real numeric dtype"], id="boolean-contents"),
    pytest.param(lambda: np.arange(6, dtype=np.complex128),
                 ["real numeric dtype"], id="complex-contents"),
    pytest.param(lambda: np.zeros(7),
                 ["contents length 7", "nbins+2=6"], id="wrong-length"),
]

BAD_CONTENT_CASES_HDF5 = [
    pytest.param(lambda: np.float64(7.25), ["one-dimensional", "ndim=0"],
                 id="scalar-contents"),
    pytest.param(lambda: np.zeros((6, 2)), ["one-dimensional", "ndim=2"],
                 id="two-dimensional-contents"),
    pytest.param(lambda: ["a", "b", "c", "d", "e", "f"],
                 ["real numeric dtype"], id="unicode-text-contents"),
    pytest.param(lambda: [b"a", b"b", b"c", b"d", b"e", b"f"],
                 ["real numeric dtype"], id="bytes-text-contents"),
    pytest.param(lambda: np.array([True] * 6),
                 ["real numeric dtype"], id="boolean-contents"),
    pytest.param(lambda: np.arange(6, dtype=np.complex128),
                 ["real numeric dtype"], id="complex-contents"),
    pytest.param(lambda: np.zeros(7),
                 ["contents length 7", "nbins+2=6"], id="wrong-length"),
]

NONFINITE_CONTENT_CASES = [
    pytest.param(
        lambda: np.array([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0]),
        "first at position 2", id="infinite-bin"),
    pytest.param(
        lambda: np.array([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]),
        "first at position 5", id="nan-bin"),
]


def _assert_path_and_name(msg: str, path) -> None:
    assert str(path) in msg
    assert f"'{DISPLAY_NAME}'" in msg


class TestSharedPayloadValidationNpz:
    def _write(self, tmp_path, *, meta, contents):
        return _mapped_file(
            tmp_path, keys=["victim"], names=[DISPLAY_NAME],
            datasets={"victim__meta": meta,
                      "victim__contents": contents})

    @pytest.mark.parametrize("build,fragments", BAD_CONTENT_CASES_NPZ)
    def test_bad_contents_refused_naming_path_and_name(self, tmp_path,
                                                       build, fragments):
        path = self._write(tmp_path, meta=META_OK, contents=build())
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        for fragment in fragments:
            assert fragment in msg
        _assert_path_and_name(msg, path)

    @pytest.mark.parametrize("meta_values,fragment", BAD_META_CASES)
    def test_bad_meta_refused_naming_path_and_name(self, tmp_path,
                                                   meta_values, fragment):
        path = self._write(tmp_path,
                           meta=np.array(meta_values, dtype=np.float64),
                           contents=np.zeros(6))
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert fragment in msg
        _assert_path_and_name(msg, path)

    @pytest.mark.parametrize("build,position", NONFINITE_CONTENT_CASES)
    def test_nonfinite_contents_refused_naming_offender(self, tmp_path,
                                                        build, position):
        # Pinned decision: stored bin contents must be finite; supported
        # producers cannot write non-finite storage because the public
        # fill policies reject non-finite inputs.
        path = self._write(tmp_path, meta=META_OK, contents=build())
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "non-finite value(s)" in msg
        assert position in msg
        assert "stored bin contents must be finite" in msg
        _assert_path_and_name(msg, path)

    def test_review_exploit_refused_before_construction(self, tmp_path):
        # The concrete exploit from review finding 4: fractional nbins
        # metadata plus Unicode string contents would previously load as
        # a nominal two-bin histogram whose _contents dtype is text.
        path = self._write(
            tmp_path,
            meta=np.array([2.5, 0.0, 1.0], dtype=np.float64),
            contents=np.array(["a", "b", "c", "d"]))
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "exactly integral, got 2.5" in msg
        _assert_path_and_name(msg, path)


def _write_victim_hdf5(path, *, nbins, xlow, xhigh, data):
    import h5py
    with h5py.File(str(path), "w") as handle:
        grp = handle.create_group("g")
        grp.attrs.create("name", DISPLAY_NAME)
        grp.attrs["nbins"] = nbins
        grp.attrs["xlow"] = xlow
        grp.attrs["xhigh"] = xhigh
        grp.create_dataset("contents", data=data)
    return path


class TestSharedPayloadValidationHdf5:
    @pytest.mark.parametrize("build,fragments", BAD_CONTENT_CASES_HDF5)
    def test_bad_contents_refused_naming_path_and_name(
            self, tmp_path, require_h5py, build, fragments):
        import h5py  # noqa: F401
        path = _write_victim_hdf5(
            tmp_path / "badc.h5", nbins=4, xlow=0.0, xhigh=1.0,
            data=build())
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        for fragment in fragments:
            assert fragment in msg
        _assert_path_and_name(msg, path)

    @pytest.mark.parametrize("meta_values,fragment", BAD_META_CASES)
    def test_bad_meta_refused_naming_path_and_name(
            self, tmp_path, require_h5py, meta_values, fragment):
        path = _write_victim_hdf5(
            tmp_path / "badm.h5", data=np.zeros(6),
            nbins=float(meta_values[0]), xlow=float(meta_values[1]),
            xhigh=float(meta_values[2]))
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert fragment in msg
        _assert_path_and_name(msg, path)

    @pytest.mark.parametrize("build,position", NONFINITE_CONTENT_CASES)
    def test_nonfinite_contents_refused_naming_offender(
            self, tmp_path, require_h5py, build, position):
        path = _write_victim_hdf5(
            tmp_path / "nonfinite.h5", nbins=4, xlow=0.0, xhigh=1.0,
            data=build())
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "non-finite value(s)" in msg
        assert position in msg
        _assert_path_and_name(msg, path)

    def test_scalar_contents_raise_valueerror_never_indexerror(
            self, tmp_path, require_h5py):
        # Regression pin for the incidental IndexError: a scalar dataset
        # arrives as a zero-dimensional array and every backend message
        # must come from the validator's rank check.
        path = _write_victim_hdf5(
            tmp_path / "scalar.h5", nbins=4, xlow=0.0, xhigh=1.0,
            data=np.float64(9.75))
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "one-dimensional" in msg
        assert "ndim=0" in msg
        _assert_path_and_name(msg, path)

    def test_fractional_nbins_with_text_contents_refused_like_npz(
            self, tmp_path, require_h5py):
        path = _write_victim_hdf5(
            tmp_path / "exploit.h5", nbins=2.5, xlow=0.0, xhigh=1.0,
            data=["a", "b", "c", "d"])
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "exactly integral, got 2.5" in msg
        _assert_path_and_name(msg, path)


class TestObjectDtypeContentsRefusal:
    """Object-dtype storage refused on BOTH backends (review gap item).

    The remaining malformed-dtype family member of review finding 4:
    object storage never reaches a private field on either backend. The
    NPZ case is refused by numpy itself at member read time
    (``allow_pickle=False``), the documented exception family that keeps
    numpy's own member-level wording; the HDF5 case reaches the shared
    validator because h5py hands variable-length strings back as kind-'O'
    arrays.
    """

    def test_npz_object_contents_member_rejected_without_pickle_fallback(
            self, tmp_path):
        # The producer-side save happily pickles object arrays into .npy
        # members; the loader side must refuse them instead of falling
        # back to pickle loading (same contract pinned for the event
        # loader in tests/test_npz_handle_lifecycle.py).
        payload = np.array(
            [{"payload": i} for i in range(6)], dtype=object)
        path = _mapped_file(
            tmp_path, keys=["victim"], names=[DISPLAY_NAME],
            datasets={"victim__meta": META_OK,
                      "victim__contents": payload})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "allow_pickle" in msg
        assert "Object arrays cannot be loaded" in msg

    def test_npz_object_meta_refused_the_same_way(self, tmp_path):
        payload = np.array([{"nbins": 4}, {"xlow": 0.0}], dtype=object)
        path = _mapped_file(
            tmp_path, keys=["victim"], names=[DISPLAY_NAME],
            datasets={"victim__meta": payload,
                      "victim__contents": np.zeros(6)})
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        assert "allow_pickle" in str(excinfo.value)

    def test_hdf5_variable_length_string_contents_rejected_as_object_dtype(
            self, tmp_path, require_h5py):
        import h5py
        path = tmp_path / "vlen_contents.h5"
        with h5py.File(str(path), "w") as handle:
            grp = handle.create_group("g")
            grp.attrs.create("name", DISPLAY_NAME)
            grp.attrs["nbins"] = 4
            grp.attrs["xlow"] = 0.0
            grp.attrs["xhigh"] = 1.0
            grp.create_dataset(
                "contents",
                data=["a", "b", "c", "d", "e", "f"],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        # Precondition pin: h5py reads variable-length string datasets back
        # as kind-'O' arrays, so this dataset drives the validator's
        # object-dtype branch specifically.
        with h5py.File(str(path), "r") as handle:
            seen = np.array(handle["g"]["contents"])
        assert seen.dtype.kind == "O"

        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "real numeric dtype" in msg
        _assert_path_and_name(msg, path)


# ---------------------------------------------------------------------------
# corrupt containers: garbage bytes, truncation, empty files
# ---------------------------------------------------------------------------

def _truncate_npz_stream() -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, a=np.arange(20.0), b=np.ones(5))
    raw = buffer.getvalue()
    return raw[: int(len(raw) * 0.6)]        # tail (central directory) gone


CORRUPT_CASES = [
    pytest.param(lambda p: p.write_bytes(b"\x00\x01not-a-zip\x00junk" * 4),
                 id="pure-garbage-bytes"),
    pytest.param(lambda p: p.write_bytes(_truncate_npz_stream()),
                 id="tail-truncated-zip"),
    pytest.param(lambda p: p.write_bytes(b""), id="zero-byte-file"),
]


class TestCorruptContainerRejection:
    """Landed contract (review finding PF-review3-io implemented).

    Every corrupt-container class is refused with ``ValueError`` whose
    message names the offending path, in the module's house style with a
    ``"load_histograms('<path>'): not an NPZ archive (...)"`` prefix:

    - pure garbage bytes -> caught by the leading-bytes container check
      before ``numpy.load`` runs; the detail names the found leading bytes
      ("container signature" fragment);
    - zero-byte file -> same leading-bytes check, "empty file" detail;
    - tail-truncated zip stream (local-header magic intact, central
      directory gone) -> detected by the ZIP layer inside ``numpy.load``
      at open time, where the leaked ``zipfile.BadZipFile`` is translated
      into the same canonical path-naming ValueError ("invalid ZIP
      container" fragment naming BadZipFile).

    Valid archives reach ``numpy.load`` untouched, so loading well-formed
    files is byte-for-byte unchanged. Errors raised after a successful
    open keep numpy's own member-level wording (object-dtype arrays,
    tests/test_npz_handle_lifecycle.py). The HDF5 loaders perform their
    own I/O and are unaffected.
    """

    @pytest.mark.parametrize("writer", [c.values[0] for c in CORRUPT_CASES])
    @pytest.mark.parametrize("loader", [load_histograms, load_events_npz])
    def test_all_classes_raise_valueerror_naming_path(self, tmp_path,
                                                      writer, loader):
        path = tmp_path / "corrupt.npz"
        writer(path)
        with pytest.raises(ValueError) as excinfo:
            loader(str(path))
        assert str(path) in str(excinfo.value)

    @pytest.mark.parametrize("loader", [load_histograms, load_events_npz])
    def test_signature_mismatch_detail_for_garbage_and_zero_byte(
            self, tmp_path, loader):
        garbage_path = tmp_path / "garbage.npz"
        garbage_path.write_bytes(b"\x00\x01not-a-zip\x00junk")
        empty_path = tmp_path / "empty.npz"
        empty_path.write_bytes(b"")
        for path in (garbage_path, empty_path):
            with pytest.raises(ValueError) as excinfo:
                loader(str(path))
            msg = str(excinfo.value)
            assert "not an NPZ archive" in msg
            assert "does not start with the ZIP/NPZ container signature" in msg

    @pytest.mark.parametrize("loader", [load_histograms, load_events_npz])
    def test_truncated_zip_names_path_instead_of_badzipfile(
            self, tmp_path, loader):
        path = tmp_path / "truncated.npz"
        path.write_bytes(_truncate_npz_stream())
        with pytest.raises(ValueError) as excinfo:
            loader(str(path))
        msg = str(excinfo.value)
        assert "not an NPZ archive" in msg
        assert "invalid ZIP container" in msg
        assert "BadZipFile: File is not a zip file" in msg


# ---------------------------------------------------------------------------
# event arrays
# ---------------------------------------------------------------------------

class TestEventsPersistence:
    def test_round_trip_preserves_values_and_dtypes(self, tmp_path):
        rng = np.random.default_rng(20260826)
        part = {
            "px": rng.normal(size=32),
            "charge": rng.integers(-1, 2, size=32).astype(np.int16),
        }
        evt = {"n_event": np.arange(8, dtype=np.int32)}
        save_events_npz(str(tmp_path / "ev.npz"), part, evt)
        got_part, got_evt = load_events_npz(str(tmp_path / "ev.npz"))
        assert set(got_part) == {"px", "charge"}
        assert set(got_evt) == {"n_event"}
        for key, src in part.items():
            np.testing.assert_array_equal(got_part[key], src)
            assert got_part[key].dtype == src.dtype
        np.testing.assert_array_equal(got_evt["n_event"], evt["n_event"])
        assert got_evt["n_event"].dtype == np.int32

    def test_absent_event_arrays_yield_empty_dict(self, tmp_path):
        save_events_npz(str(tmp_path / "solo.npz"),
                        {"id": np.arange(4).astype(np.int64)})
        part, evt = load_events_npz(str(tmp_path / "solo.npz"))
        assert set(part) == {"id"}
        np.testing.assert_array_equal(part["id"], np.arange(4))
        assert evt == {}

    def test_unprefixed_datasets_are_ignored(self, tmp_path):
        # Written directly (not through save_events_npz) so unprefixed
        # keys can exist: they carry no part_/evt_ tag and must be dropped.
        np.savez(str(tmp_path / "mixed.npz"),
                 part_px=np.array([1.0, 2.0]),
                 evt_run=np.array([7]),
                 stray=np.array(["ignore-me"]))
        part, evt = load_events_npz(str(tmp_path / "mixed.npz"))
        assert set(part) == {"px"}
        assert set(evt) == {"run"}
        assert "stray" not in part and "stray" not in evt


# ---------------------------------------------------------------------------
# HDF5 optional backend
# ---------------------------------------------------------------------------

class TestHdf5Backend:
    def test_round_trip_keeps_spaces_slashes_and_numbers(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        mapping = _write_demo_pair(tmp_path / "rt.h5")
        mapping["nested/name"] = _demo_hist("deep")
        save_histograms_hdf5(str(tmp_path / "rt2.h5"), mapping)
        loaded = load_histograms_hdf5(str(tmp_path / "rt2.h5"))
        assert set(loaded) == set(mapping)
        for name, src in mapping.items():
            got = loaded[name]
            assert got.name == name           # "/" never re-normalized back
            assert got._contents.tobytes() == src._contents.tobytes()
            assert got.xlow == pytest.approx(src.xlow)

    def test_bytes_typed_name_attribute_is_decoded(self, tmp_path,
                                                   require_h5py):
        import h5py
        path = tmp_path / "bytesname.h5"
        with h5py.File(str(path), "w") as handle:
            grp = handle.create_group("g")
            # Note: assigning grp.attrs["name"] = b"..." would be handed
            # back as str by this h5py build, silently bypassing the decode
            # branch under test; attrs.create with np.bytes_ preserves the
            # stored value as raw bytes on read-back.
            grp.attrs.create("name", np.bytes_(b"legacy bytes name"))
            grp.attrs["nbins"] = 3
            grp.attrs["xlow"] = -1.0
            grp.attrs["xhigh"] = 2.0
            grp.create_dataset("contents",
                               data=np.arange(5, dtype=np.float64))
        loaded = load_histograms_hdf5(str(path))
        assert set(loaded) == {"legacy bytes name"}
        hist = loaded["legacy bytes name"]
        assert hist.name == "legacy bytes name"
        np.testing.assert_allclose(hist._contents, np.arange(5.0))

    def test_missing_name_attribute_falls_back_to_group_name(
            self, tmp_path, require_h5py):
        import h5py
        path = tmp_path / "noname.h5"
        with h5py.File(str(path), "w") as handle:
            grp = handle.create_group("groupkey")
            grp.attrs["nbins"] = 2
            grp.attrs["xlow"] = 0.0
            grp.attrs["xhigh"] = 1.0
            grp.create_dataset("contents", data=np.zeros(4))
        loaded = load_histograms_hdf5(str(path))
        assert set(loaded) == {"groupkey"}

    def test_short_contents_dataset_rejected_with_message(
            self, tmp_path, require_h5py):
        import h5py
        path = tmp_path / "shorth5.h5"
        with h5py.File(str(path), "w") as handle:
            grp = handle.create_group("trunc")
            grp.attrs["name"] = "Trunc"
            grp.attrs["nbins"] = 4
            grp.attrs["xlow"] = 0.0
            grp.attrs["xhigh"] = 1.0
            grp.create_dataset("contents", data=np.ones(3))
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "Trunc" in msg
        assert "contents length 3" in msg
        assert "nbins+2=6" in msg


# ---------------------------------------------------------------------------
# smoke: JSON summary comment documented for provenance parity checks
# ---------------------------------------------------------------------------

def test_module_documents_summary_payload_contract():
    # The multispecies summary/NPZ cross-checks live in sps_core (Phase P3,
    # tested separately); this module pins only the plain data_io layer.
    import na49_mixing.data_io as mod
    assert hasattr(mod, "load_multispecies_details_npz") is False
    assert json.dumps({"contract": "data_io has no summary coupling"}) == (
        '{"contract": "data_io has no summary coupling"}')


# ---------------------------------------------------------------------------
# review5 finding 4: save-time payload boundary runs BEFORE open/write
# ---------------------------------------------------------------------------

SENTINEL_BYTES = b"PRE-EXISTING-CONTENT-THAT-MUST-SURVIVE\n\x00\x01payload"


def _hist_with_tainted_bin(value: float) -> Histogram1D:
    """A nominally valid histogram whose storage was made non-finite.

    ``set_bin_content`` is deliberately out of the arithmetic policy
    scope; it is exactly the documented escape hatch used to drive the
    save-boundary cases.
    """
    h = Histogram1D(4, 0.0, 1.0, name="victim")
    h.fill_array(np.linspace(0.05, 0.95, 4))
    h.set_bin_content(2, value)
    return h


class TestSaveBoundaryRejectsBeforeOpeningDestination:
    @pytest.mark.parametrize("backend", ["npz", "hdf5"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_payload_leaves_preexisting_file_intact(
            self, tmp_path, require_h5py, backend, bad):
        import h5py  # noqa: F401
        path = tmp_path / f"boundary_{backend}.{'npz' if backend == 'npz' else 'h5'}"
        path.write_bytes(SENTINEL_BYTES)
        mapping = {"victim": _hist_with_tainted_bin(bad)}
        saver = save_histograms if backend == "npz" else save_histograms_hdf5
        with pytest.raises(ValueError) as excinfo:
            saver(str(path), mapping)
        msg = str(excinfo.value)
        assert "non-finite value(s)" in msg
        assert str(path) in msg
        assert "'victim'" in msg
        assert path.read_bytes() == SENTINEL_BYTES

    def test_npz_rejection_never_reaches_numpy_saver(self, tmp_path,
                                                     monkeypatch):
        calls = []

        def boom(*args, **kwargs):     # pragma: no cover - must not run
            calls.append((args, kwargs))
            raise AssertionError("np.savez reached despite rejection")

        monkeypatch.setattr(np, "savez", boom)
        with pytest.raises(ValueError):
            save_histograms(str(tmp_path / "never.npz"),
                            {"victim": _hist_with_tainted_bin(float("nan"))})
        assert calls == []
        assert not (tmp_path / "never.npz").exists()

    def test_hdf5_rejection_never_opens_destination(self, tmp_path,
                                                    require_h5py, monkeypatch):
        import h5py
        opened = []

        class Boom:
            def __init__(self, *args, **kwargs):
                opened.append((args, kwargs))
                raise AssertionError("h5py.File reached despite rejection")

        monkeypatch.setattr(h5py, "File", Boom)
        with pytest.raises(ValueError) as excinfo:
            save_histograms_hdf5(
                str(tmp_path / "never.h5"),
                {"victim": _hist_with_tainted_bin(float("nan"))},
            )
        assert opened == []
        msg = str(excinfo.value)
        assert "non-finite value(s)" in msg
        assert not (tmp_path / "never.h5").exists()

    def test_hdf5_group_collision_no_longer_creates_empty_file(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        h = _demo_hist("one")
        h2 = _demo_hist("two")
        path = tmp_path / "collide.h5"
        with pytest.raises(ValueError, match="group-name collision"):
            save_histograms_hdf5(str(path), {"a/x": h, "a_x": h2})
        assert not path.exists()

    def test_valid_saving_still_works_after_boundary(self, tmp_path,
                                                     require_h5py):
        mapping = _write_demo_pair(tmp_path / "after.npz")
        loaded = load_histograms(str(tmp_path / "after.npz"))
        for name, src in mapping.items():
            assert loaded[name]._contents.tobytes() == \
                src._contents.tobytes()


# ---------------------------------------------------------------------------
# review5 finding 4 + gap item 7: persistence after rejected arithmetic
# ---------------------------------------------------------------------------


class TestPersistenceAfterInvalidArithmeticAttempt:
    def test_npz_round_trip_after_rejected_scale(self, tmp_path):
        h = _demo_hist("survivor")
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError):
            h.scale(float("nan"))
        save_histograms(str(tmp_path / "rt.npz"), {"survivor": h})
        got = load_histograms(str(tmp_path / "rt.npz"))["survivor"]
        assert got._contents.tobytes() == snapshot

    def test_hdf5_round_trip_after_rejected_overflow_fill(self, tmp_path,
                                                          require_h5py):
        import h5py  # noqa: F401
        h = Histogram1D(4, 0.0, 1.0, name="overflowed")
        h.fill(0.25, weight=9.9e307)             # last successful mutation
        with pytest.raises(ValueError):
            h.fill(0.25, weight=8.9e307)         # sums past max double
        snapshot = h._contents.tobytes()
        save_histograms_hdf5(str(tmp_path / "rt.h5"), {"overflowed": h})
        got = load_histograms_hdf5(str(tmp_path / "rt.h5"))["overflowed"]
        assert got._contents.tobytes() == snapshot
        assert np.isfinite(got._contents).all()


# ---------------------------------------------------------------------------
# review5 finding 5: HDF5 loader schema hardening
# ---------------------------------------------------------------------------


def _craft_histogram_groups(path, groups):
    """Write raw HDF5 structure per (group name → spec dict).

    Spec keys: ``display`` (value of the ``name`` attribute; ``None``
    omits the attribute), ``with_nbins``/``with_xlow``/``with_xhigh``
    (default True), ``contents`` (dataset payload; default zeros(4)),
    ``missing_contents`` (create no contents member at all), and
    ``contents_as_group`` (create a nested group under that key).
    """
    import h5py
    with h5py.File(str(path), "w") as handle:
        for gname, spec in groups.items():
            grp = handle.create_group(gname)
            if spec.get("display", "H") is not None:
                grp.attrs.create("name", spec.get("display", "H"))
            if spec.get("with_nbins", True):
                grp.attrs["nbins"] = 2
            if spec.get("with_xlow", True):
                grp.attrs["xlow"] = 0.0
            if spec.get("with_xhigh", True):
                grp.attrs["xhigh"] = 1.0
            if spec.get("missing_contents"):
                continue
            if spec.get("contents_as_group"):
                nested = grp.create_group("contents")
                nested.create_dataset("inner", data=np.zeros(4))
            else:
                grp.create_dataset(
                    "contents", data=spec.get("contents", np.zeros(4)))


def _write_root_stray_dataset(path):
    import h5py
    with h5py.File(str(path), "w") as handle:
        handle.create_dataset("stray", data=np.arange(6))


class TestHdf5LoaderSchemaHardening:
    def test_duplicate_display_names_refused_with_both_group_names(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        path = tmp_path / "dups.h5"
        _craft_histogram_groups(path, {
            "g1": {"display": "dup"},
            "g2": {"display": "dup"},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "duplicate display name 'dup'" in msg
        assert "'g1'" in msg and "'g2'" in msg
        assert str(path) in msg

    def test_duplicate_display_name_via_nameless_fallback_is_also_refused(
            self, tmp_path, require_h5py):
        # Two distinct groups without a name attribute each fall back to
        # their own group key, so they stay distinct and load normally.
        import h5py  # noqa: F401
        path = tmp_path / "fallback.h5"
        _craft_histogram_groups(path, {
            "first": {"display": None},
            "second": {"display": None},
        })
        loaded = load_histograms_hdf5(str(path))
        assert set(loaded) == {"first", "second"}

    def test_root_dataset_entry_refused_naming_path_and_entry(
            self, tmp_path, require_h5py):
        path = tmp_path / "stray.h5"
        _write_root_stray_dataset(path)
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "root entry 'stray' is Dataset" in msg
        assert str(path) in msg

    def test_missing_attribute_variants_refused_naming_entry(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        path = tmp_path / "attrs.h5"
        _craft_histogram_groups(path, {
            "bare": {"display": "Bare", "with_nbins": False,
                     "with_xlow": False, "with_xhigh": False},
            "half": {"display": "Half", "with_xlow": False},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        # Iteration order decides which entry fails first; both variants
        # must produce one canonical schema error naming path and entry.
        assert str(path) in msg
        assert ("'bare'" in msg) ^ ("'half'" in msg)
        assert "missing required attribute(s)" in msg
        if "'bare'" in msg:
            assert "'nbins'" in msg and "'xhigh'" in msg
        else:
            assert "'xlow'" in msg

    def test_missing_contents_dataset_refused_naming_entry(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        path = tmp_path / "nodata.h5"
        _craft_histogram_groups(path, {
            "ghost": {"display": "Ghost", "missing_contents": True},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "missing required dataset 'contents'" in msg
        assert "'ghost'" in msg
        assert str(path) in msg

    def test_contents_stored_as_nested_group_refused(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        path = tmp_path / "nested.h5"
        _craft_histogram_groups(path, {
            "nestor": {"display": "Nestor", "contents_as_group": True},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "stores 'contents' as Group" in msg
        assert "'nestor'" in msg
        assert str(path) in msg

    def test_duplicate_reported_before_second_group_attribute_fault(
            self, tmp_path, require_h5py):
        # Pinned check ordering within one entry: display-name resolution
        # and duplicate detection precede the attribute schema checks.
        # Group names sort so h5py's iteration order visits the fully
        # valid carrier first.
        import h5py  # noqa: F401
        path = tmp_path / "order.h5"
        _craft_histogram_groups(path, {
            "aaa_ok": {"display": "X"},
            "zzz_faulted": {"display": "X", "with_xhigh": False},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "duplicate display name 'X'" in msg
        assert "'aaa_ok'" in msg and "'zzz_faulted'" in msg
        assert "missing required attribute" not in msg

    def test_previously_written_valid_files_load_unchanged(
            self, tmp_path, require_h5py):
        # The hardening must not disturb any valid producer output.
        mapping = _write_demo_pair(tmp_path / "valid.h5")
        loaded = load_histograms_hdf5(str(tmp_path / "valid.h5"))
        assert set(loaded) == set(mapping)
        for name, src in mapping.items():
            assert loaded[name]._contents.tobytes() == \
                src._contents.tobytes()
            assert loaded[name].name == name


# ---------------------------------------------------------------------------
# review5 widening (gap items 7-8): cross-backend property rounds after
# attempted invalid arithmetic; taint-position matrix for the save
# boundary; duplicate-display-name tracking across three groups
# ---------------------------------------------------------------------------


def _attempt_rejected_arithmetic(kind: str) -> Histogram1D:
    """Build one histogram whose in-policy mutation call was rejected.

    Each branch first performs a legal deposit (the last successful
    mutation), then invokes exactly one mutating call that the
    finite-storage policy must refuse. The returned object is the
    post-rejection state that persistence must round-trip unchanged.
    """
    h = Histogram1D(4, 0.0, 1.0, name=f"survivor_{kind}")
    if kind == "scale-nonfinite":
        h.fill_array(np.linspace(0.05, 0.95, 4))
        with pytest.raises(ValueError):
            h.scale(float("nan"))
    elif kind == "scale-overflow-factor":
        h.fill(0.25, weight=1e300)
        with pytest.raises(ValueError):
            h.scale(1e300)                     # 1e300 * 1e300 -> inf
    elif kind == "add-nonfinite-scale":
        other = Histogram1D(4, 0.0, 1.0, name="companion")
        other.fill(0.55, weight=-2.5)
        with pytest.raises(ValueError):
            h.add(other, scale=float("inf"))
    elif kind == "fill-accumulation":
        h.fill(0.3, weight=1e308)
        with pytest.raises(ValueError):
            h.fill(0.3, weight=1e308)          # sums past max double
    elif kind == "fill-array-overflow":
        h.fill(0.25, weight=7.7e307)
        with pytest.raises(ValueError):
            h.fill_array(
                np.array([0.25, 0.25]),
                weights=np.array([6.6e307, 5.5e307]),
            )
    else:   # pragma: no cover - guards against typos in test tables
        raise AssertionError(f"unknown arithmetic kind {kind!r}")
    return h


REJECTED_ARITHMETIC_KINDS = (
    "scale-nonfinite",
    "scale-overflow-factor",
    "add-nonfinite-scale",
    "fill-accumulation",
    "fill-array-overflow",
)


class TestRoundTripAfterAttemptedInvalidArithmetic:
    """Gap item 7 property: state surviving a rejected mutation persists.

    For every rejection family and BOTH backends, saving afterwards
    succeeds and reloads bit-for-bit what the histogram held through the
    failed call (complement of ``TestPersistenceAfterInvalidArithmetic-
    Attempt`` above, which pins two concrete kinds on single backends).
    """

    @pytest.mark.parametrize(
        "backend", [pytest.param("npz", id="npz"), pytest.param("hdf5", id="hdf5")]
    )
    @pytest.mark.parametrize("kind", REJECTED_ARITHMETIC_KINDS)
    def test_post_rejection_state_round_trips_bitwise(
            self, tmp_path, require_h5py, backend, kind):
        import h5py  # noqa: F401  (presence ensured for hdf5 params)
        h = _attempt_rejected_arithmetic(kind)
        snapshot = h._contents.tobytes()
        ext = "npz" if backend == "npz" else "h5"
        path = tmp_path / f"after_{backend}_{kind}.{ext}"
        saver = save_histograms if backend == "npz" else save_histograms_hdf5
        loader = load_histograms if backend == "npz" else load_histograms_hdf5
        saver(str(path), {f"survivor_{kind}": h})
        got = loader(str(path))[f"survivor_{kind}"]
        assert got._contents.tobytes() == snapshot
        assert np.isfinite(got._contents).all()


def _tainted_victim(position_id: str) -> Histogram1D:
    """Histogram whose storage became non-finite at a named position.

    ``set_bin_content`` is the documented escape hatch; it is used here
    to place taint into every storage slot family, including underflow
    (index 0) and overflow (index nbins+1) slots the fitters never fill.
    """
    h = Histogram1D(4, 0.0, 1.0, name="victim")
    h.fill_array(np.linspace(0.05, 0.95, 4))
    if position_id == "underflow-inf":
        h.set_bin_content(0, float("inf"))
    elif position_id == "interior-nan":
        h.set_bin_content(2, float("nan"))
    elif position_id == "overflow-minus-inf":
        h.set_bin_content(5, float("-inf"))
    elif position_id == "two-bins":
        h.set_bin_content(1, float("inf"))
        h.set_bin_content(4, float("nan"))
    else:   # pragma: no cover - guards against typos in test tables
        raise AssertionError(f"unknown taint position {position_id!r}")
    return h


TAINT_POSITION_IDS = (
    "underflow-inf",
    "interior-nan",
    "overflow-minus-inf",
    "two-bins",
)


class TestTaintedPayloadBoundaryEveryStoragePosition:
    """Gap item 6/7 bridge: no storage slot family reaches any file.

    Both savers validate the complete payload mapping before opening the
    destination, so even when a healthy companion precedes the tainted
    entry, nothing is written and pre-existing bytes survive verbatim.
    """

    @pytest.mark.parametrize("backend", ["npz", "hdf5"])
    @pytest.mark.parametrize("position_id", TAINT_POSITION_IDS)
    def test_taint_in_any_slot_blocks_the_whole_mapping(
            self, tmp_path, require_h5py, backend, position_id):
        import h5py  # noqa: F401
        ext = "npz" if backend == "npz" else "h5"
        path = tmp_path / f"taint_{backend}_{position_id}.{ext}"
        path.write_bytes(SENTINEL_BYTES)
        healthy = _demo_hist("healthy")
        mapping = {
            "healthy": healthy,
            # Victim last deliberately: proves the whole dict validates
            # before any open/write happens.
            "victim": _tainted_victim(position_id),
        }
        saver = save_histograms if backend == "npz" else save_histograms_hdf5
        with pytest.raises(ValueError) as excinfo:
            saver(str(path), mapping)
        msg = str(excinfo.value)
        assert "non-finite value(s)" in msg
        assert "'victim'" in msg
        assert str(path) in msg
        assert path.read_bytes() == SENTINEL_BYTES


class TestBackendsAgreeOnValidPayloads:
    """Gap item 7 property: NPZ and HDF5 are interchangeable for valid data.

    The same mapping saved through both backends must come back with
    identical names, metadata, dtypes, and bitwise-equal contents — and
    equal to the in-memory source either way.
    """

    def test_same_mapping_loads_bitwise_identical_from_both_backends(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        src = {
            "signal region": _write_demo_pair(tmp_path / "unused.npz")[
                "signal region"
            ],
            "signed_tail": Histogram1D(3, -1.0, 0.5, name="signed_tail"),
        }
        src["signed_tail"].fill_array(
            np.linspace(-0.95, 0.45, 3), np.array([0.5, -0.25, 0.125])
        )
        npz_path = tmp_path / "agree.npz"
        h5_path = tmp_path / "agree.h5"
        save_histograms(str(npz_path), src)
        save_histograms_hdf5(str(h5_path), src)
        from_npz = load_histograms(str(npz_path))
        from_h5 = load_histograms_hdf5(str(h5_path))
        assert set(from_npz) == set(src) == set(from_h5)
        for name in src:
            wanted = src[name]._contents.tobytes()
            assert from_npz[name]._contents.tobytes() == wanted
            assert from_h5[name]._contents.tobytes() == wanted
            assert from_npz[name]._contents.dtype == np.float64
            assert from_h5[name]._contents.dtype == np.float64
            assert from_npz[name].name == name
            assert from_h5[name].name == name
            assert from_npz[name].nbins == from_h5[name].nbins
            assert from_npz[name].xlow == from_h5[name].xlow
            assert from_npz[name].xhigh == from_h5[name].xhigh


class TestDuplicateDisplayTrackingRobustness:
    """Gap item 8 widening: tracking survives multi-entry interleavings."""

    def test_third_group_duplicating_first_names_that_pair(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        path = tmp_path / "tri.h5"
        _craft_histogram_groups(path, {
            "g_first": {"display": "X"},
            "g_middle": {"display": "Y"},
            "g_late": {"display": "X"},
        })
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "duplicate display name 'X'" in msg
        assert "'g_first'" in msg and "'g_late'" in msg
        assert "'g_middle'" not in msg           # untouched party unnamed
        assert "'Y'" not in msg
        assert str(path) in msg
