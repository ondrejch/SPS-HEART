"""Handle-lifecycle and post-close usability tests for the NPZ loaders.

Covers :func:`na49_mixing.data_io.load_histograms` and
:func:`na49_mixing.data_io.load_events_npz`:

- repeated loads leave the process open-file count flat (the loaders close
  their ``NpzFile`` handles; measured with psutil, falling back to a
  ``/proc/self/fd`` listing);
- returned arrays stay fully usable and numerically identical to direct
  reads of the source file after the loader has returned;
- documented failure contracts are preserved after wrapping:
  ``load_histograms`` still raises mapping-schema/collision ``ValueError``
  messages naming the file path, and the events loader keeps its
  path-naming missing-file error plus unchanged lazy-member rejection of
  pickled object arrays (no pickle fallback).
"""

from __future__ import annotations

import gc
import io
import os
import re
import zipfile

import numpy as np
import pytest

from na49_mixing.data_io import (
    load_events_npz,
    load_histograms,
    save_events_npz,
    save_histograms,
)
from na49_mixing.histogram import Histogram1D


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

def _write_histogram_file(path):
    """Save two small histograms via the public save path; return them."""
    h_signal = Histogram1D(8, -1.0, 3.0, name="signal")
    h_background = Histogram1D(4, 0.0, 2.0, name="background b")
    h_signal.fill_array(np.linspace(-0.9, 2.9, 37))
    h_background.fill_array(np.array([0.1, 1.9, 0.5]))
    save_histograms(str(path), {"signal": h_signal, "background b": h_background})
    return h_signal, h_background


def _write_events_file(path):
    """Save deterministic particle/event arrays; return the plain dicts."""
    rng = np.random.default_rng(42)
    part = {
        "px": rng.normal(loc=10.0, scale=1.0, size=64),
        "charge": rng.integers(-1, 2, size=64).astype(np.int16),
        "mass": rng.uniform(0.05, 1.5, size=64),
    }
    evt = {
        "n_event": np.arange(16, dtype=np.int32),
        "vertex_z": rng.normal(scale=2.0, size=16),
    }
    save_events_npz(str(path), part, evt)
    return part, evt


def _count_open_fds():
    """Open-file count via psutil, else /proc/self/fd; None if unavailable."""
    try:
        import psutil
    except ImportError:
        pass
    else:
        return int(psutil.Process().num_fds())
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


_OPEN_FDS_MECHANISM = _count_open_fds() is not None

skipif_no_fd_mechanism = pytest.mark.skipif(
    not _OPEN_FDS_MECHANISM,
    reason="neither psutil nor /proc/self/fd is available for fd counting",
)


def _assert_flat_fd_count(load_once, n_iter=25, warmup=3):
    """Run load_once repeatedly; assert the open-file count does not grow."""
    for _ in range(warmup):
        load_once()
    gc.collect()
    before = _count_open_fds()
    assert before is not None
    for _ in range(n_iter):
        load_once()
    gc.collect()
    after = _count_open_fds()
    assert after is not None
    assert after <= before, (
        f"open-file count grew from {before} to {after} over "
        f"{n_iter} repeated loads"
    )


# ---------------------------------------------------------------------------
# fd-count smoke tests
# ---------------------------------------------------------------------------

@skipif_no_fd_mechanism
class TestRepeatedLoadsKeepFdCountFlat:
    def test_load_histograms(self, tmp_path):
        path = tmp_path / "histograms.npz"
        _write_histogram_file(path)
        _assert_flat_fd_count(lambda: load_histograms(str(path)))

    def test_load_events_npz(self, tmp_path):
        path = tmp_path / "events.npz"
        _write_events_file(path)
        _assert_flat_fd_count(lambda: load_events_npz(str(path)))

    def test_load_histograms_survives_mapping_errors_flat(self, tmp_path):
        """Failed loads must also release the handle before returning/raising."""
        path = tmp_path / "dupkey.npz"
        meta = np.array([4, 0.0, 1.0], dtype=np.float64)
        np.savez(
            path,
            __hist_keys=np.array(["a", "a"], dtype="U"),
            __hist_names=np.array(["A", "B"], dtype="U"),
            a__contents=np.zeros(6),
            a__meta=meta,
        )

        def fail_once():
            with pytest.raises(ValueError):
                load_histograms(str(path))

        _assert_flat_fd_count(fail_once)


# ---------------------------------------------------------------------------
# post-close usability and numeric identity
# ---------------------------------------------------------------------------

def _direct_npz_arrays(path):
    """Read reference copies through an independently managed NpzFile."""
    out = {}
    with np.load(str(path), allow_pickle=False) as ref:
        for key in ref.files:
            out[key] = ref[key].copy()
    return out


class TestHistogramsUsableAfterClose:
    def test_contents_identical_and_operable_after_return(self, tmp_path):
        path = tmp_path / "histograms.npz"
        h_signal, _ = _write_histogram_file(path)
        loaded = load_histograms(str(path))
        # The loader returned; its NpzFile handle is closed from here on.
        ref = _direct_npz_arrays(path)

        assert set(loaded) == {"signal", "background b"}
        got = loaded["signal"]
        assert got.nbins == 8
        assert got.xlow == -1.0
        assert got.xhigh == 3.0
        np.testing.assert_array_equal(got._contents, ref["signal__contents"])
        np.testing.assert_array_equal(loaded["background b"]._contents,
                                      ref["background_b__contents"])

        # Full usability: scalar reductions, boolean masks, in-place ops.
        total = got._contents.sum()
        interior = got._contents[1:-1].sum()
        mask = got._contents > 0
        masked_sum = got._contents[mask].sum()
        assert total == pytest.approx(h_signal._contents.sum())
        assert interior == pytest.approx(37.0)  # all filled values in range
        assert mask.sum() == 8  # bins 1..8 touched; edge bins stay empty
        assert masked_sum == total
        baseline = got._contents.copy()
        got._contents *= 2.0
        np.testing.assert_allclose(got._contents, 2.0 * baseline)


class TestEventsUsableAfterClose:
    def test_dicts_identical_and_operable_after_return(self, tmp_path):
        path = tmp_path / "events.npz"
        part_src, evt_src = _write_events_file(path)
        loaded_part, loaded_evt = load_events_npz(str(path))
        # The loader returned; its NpzFile handle is closed from here on.
        ref = _direct_npz_arrays(path)

        assert isinstance(loaded_part, dict)
        assert isinstance(loaded_evt, dict)
        assert set(loaded_part) == set(part_src)
        assert set(loaded_evt) == set(evt_src)

        expected_prefixes = {key[len("part_"):]: key for key in ref
                             if key.startswith("part_")}
        expected_prefixes.update({key[len("evt_"):]: key for key in ref
                                  if key.startswith("evt_")})
        all_loaded = dict(loaded_part)
        all_loaded.update(loaded_evt)
        for name, stored_key in expected_prefixes.items():
            np.testing.assert_array_equal(all_loaded[name], ref[stored_key])

        assert loaded_part["charge"].dtype == ref["part_charge"].dtype
        assert loaded_evt["n_event"].dtype == ref["evt_n_event"].dtype

        # Full usability: sums and boolean-mask selections after close.
        px_mask = loaded_part["px"] > 10.0
        assert px_mask.any()
        px_sum_selected = loaded_part["px"][px_mask].sum()
        assert px_sum_selected == pytest.approx(part_src["px"][px_mask].sum())
        evt_sum = float(loaded_evt["vertex_z"].sum())
        assert evt_sum == pytest.approx(float(evt_src["vertex_z"].sum()))

    def test_returned_arrays_are_independent_copies(self, tmp_path):
        path = tmp_path / "events.npz"
        _write_events_file(path)
        loaded_part, loaded_evt = load_events_npz(str(path))
        loaded_part["px"] += 100.0
        loaded_evt["vertex_z"][0] = -999.0

        fresh_part, fresh_evt = load_events_npz(str(path))
        # Fresh load reproduces the on-disk values untouched by mutation.
        with np.load(str(path), allow_pickle=False) as ref:
            np.testing.assert_array_equal(fresh_part["px"], ref["part_px"])
            np.testing.assert_array_equal(fresh_evt["vertex_z"],
                                          ref["evt_vertex_z"])


# ---------------------------------------------------------------------------
# failure contracts preserved under the with-block
# ---------------------------------------------------------------------------

class TestHistogramsMappingSchemaContract:
    def test_duplicate_storage_key_error_names_path(self, tmp_path):
        """Collision contract: ValueError message carries the file path."""
        path = tmp_path / "dupkey.npz"
        meta = np.array([4, 0.0, 1.0], dtype=np.float64)
        np.savez(
            path,
            __hist_keys=np.array(["a", "a"], dtype="U"),
            __hist_names=np.array(["A", "B"], dtype="U"),
            a__contents=np.zeros(6),
            a__meta=meta,
        )
        with pytest.raises(ValueError, match="duplicate storage key") as excinfo:
            load_histograms(str(path))
        assert str(path) in str(excinfo.value)

    def test_mismatched_mapping_lengths_error_names_path(self, tmp_path):
        """Schema contract: length mismatch raises ValueError naming path."""
        path = tmp_path / "mismatch.npz"
        np.savez(
            path,
            __hist_keys=np.array(["a", "b"], dtype="U"),
            __hist_names=np.array(["A"], dtype="U"),
            a__contents=np.zeros(6),
            a__meta=np.array([4, 0.0, 1.0], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="__hist_keys length") as excinfo:
            load_histograms(str(path))
        pattern = re.escape(str(path))
        assert re.search(pattern, str(excinfo.value))


class TestEventsLoaderFailureContract:
    def test_missing_file_error_names_path(self, tmp_path):
        """Documented failure surface: FileNotFoundError naming the path."""
        path = tmp_path / "absent_events.npz"
        with pytest.raises(FileNotFoundError) as excinfo:
            load_events_npz(str(path))
        assert str(path) in str(excinfo.value)

    def test_object_dtype_member_rejected_without_pickle_fallback(self, tmp_path):
        """Lazy member read inside the with-block still rejects object arrays.

        An object-dtype .npy member is crafted at container level (trusted
        producer side); reading it back with allow_pickle=False must raise
        the numpy ValueError unchanged through the wrapped loop, never fall
        back to pickle loading.
        """
        buffer = io.BytesIO()
        np.lib.format.write_array(
            buffer, np.array([{"payload": 1}], dtype=object), allow_pickle=True
        )
        path = tmp_path / "object_member.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("part_tag.npy", buffer.getvalue())
        with pytest.raises(ValueError, match="allow_pickle") as excinfo:
            load_events_npz(str(path))
        # The rejection happens while reading the member; numpy's own error
        # text does not embed the path, matching the pre-wrapping behavior.
        assert "allow_pickle" in str(excinfo.value)
