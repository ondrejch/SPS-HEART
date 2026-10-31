"""Data I/O utilities.

Provides HDF5- and NumPy-based persistence for events and histograms,
replacing ROOT ``TFile``/``TTree`` I/O.

Dependencies: ``h5py`` (optional for HDF5), ``numpy``.
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
import math
import zipfile

import numpy as np

from .histogram import (
    Histogram1D,
    _coerce_histogram_contents,
    _require_finite_bin_geometry,
)


# ---------------------------------------------------------------------------
# NPZ container pre-validation (review finding PF-review3-io)
# ---------------------------------------------------------------------------

_ZIP_LOCAL_FILE_HEADER = b"PK\x03\x04"
_ZIP_EMPTY_ARCHIVE = b"PK\x05\x06"


def _npz_not_an_archive_error(path: str, loader: str, detail: str) -> ValueError:
    return ValueError(
        f"{loader}({path!r}): not an NPZ archive ({detail}); the file "
        f"appears corrupt, truncated, or empty"
    )


def _validate_npz_container(path: str, loader: str) -> None:
    """Reject payloads that do not start like a ZIP (.npz) container.

    Raises :class:`ValueError` naming *path* before :func:`numpy.load` is
    called, so garbage-byte and zero-byte files produce one canonical error
    instead of leaking numpy/zipfile internals. Filesystem-level failures
    (missing file, wrong file type) are left to ``numpy.load`` unchanged.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(_ZIP_LOCAL_FILE_HEADER))
    except OSError:
        return
    if head.startswith(_ZIP_LOCAL_FILE_HEADER) or head.startswith(
        _ZIP_EMPTY_ARCHIVE
    ):
        return
    if head:
        found = f"leading bytes '{head.hex(' ')}'"
    else:
        found = "empty file"
    raise _npz_not_an_archive_error(
        path,
        loader,
        f"{found} does not start with the ZIP/NPZ container signature",
    )


@contextmanager
def _open_npz_archive(path: str, loader: str) -> Iterator[Any]:
    """Open an ``.npz`` file for reading behind container validation.

    Yields an open :class:`numpy.lib.npyio.NpzFile`; the handle is closed on
    exit whether the body raises or not. Non-zip payloads are refused with a
    :class:`ValueError` naming *path*; zip-headed but container-corrupt
    payloads (for example a tail-truncated stream whose central directory is
    gone) reach numpy's own ZIP layer, and that failure is translated into
    the same path-naming ValueError. Errors raised later at member access
    keep their original numpy wording.
    """
    _validate_npz_container(path, loader)
    try:
        archive = np.load(path, allow_pickle=False)
    except (zipfile.BadZipFile, EOFError) as exc:
        raise _npz_not_an_archive_error(
            path, loader, f"invalid ZIP container ({type(exc).__name__}: {exc})"
        ) from exc
    with archive as zarr:
        yield zarr


# ---------------------------------------------------------------------------
# Histogram persistence (NumPy .npz)
# ---------------------------------------------------------------------------

def save_histograms(path: str, histograms: Dict[str, Histogram1D]) -> None:
    """Save a dict of :class:`~na49_mixing.histogram.Histogram1D` to a ``.npz`` file.

    Histogram names are normalized to array keys by replacing spaces with
    underscores. Two distinct names that normalize to the same key
    (``"a b"`` and ``"a_b"``) would silently overwrite each other, so the
    save is refused up front.

    Save-time payload validation boundary: every histogram payload — the
    metadata triple plus contents rank, dtype, length and finiteness — is
    checked through :func:`_validate_histogram_payload` BEFORE the
    destination file is opened, applying the exact rule the loaders
    enforce on read-back. A rejected save therefore never truncates or
    partially writes the destination: pre-existing content survives
    byte-for-byte and no new file appears at *path*.

    Parameters
    ----------
    path : str
        Output file path (``'.npz'`` extension added automatically if absent).
    histograms : dict
        Mapping of name → histogram.

    Raises
    ------
    ValueError
        If two distinct names normalize to the same storage key, or if
        any histogram payload fails the shared numeric-schema validation
        (for example non-finite bin contents). The message names both
        offending names for key collisions and the path plus display name
        otherwise.
    """
    arrays: Dict[str, np.ndarray] = {}
    key_owner: Dict[str, str] = {}
    keys_in_order: list[str] = []
    names_in_order: list[str] = []
    context = f"save_histograms({path!r})"
    for name, h in histograms.items():
        key = name.replace(" ", "_")
        if key in key_owner:
            raise ValueError(
                "save_histograms(): normalized-key collision — "
                f"{key_owner[key]!r} and {name!r} both map to storage key "
                f"{key!r}; rename one of the histograms"
            )
        key_owner[key] = name
        # Payload boundary before any file is opened (review5 finding 4):
        # the written meta/contents pair is exactly what was validated.
        meta = np.array([h.nbins, h.xlow, h.xhigh], dtype=np.float64)
        _, _, _, stored = _validate_histogram_payload(
            context, name, meta, h._contents,
        )
        keys_in_order.append(key)
        names_in_order.append(name)
        arrays[key + "__contents"] = stored
        arrays[key + "__meta"] = meta
    arrays["__hist_keys"] = np.asarray(keys_in_order, dtype="U")
    arrays["__hist_names"] = np.asarray(names_in_order, dtype="U")
    np.savez(path, **arrays)


def _npz_histogram_pairs(path: str, data: np.lib.npyio.NpzFile) -> list[tuple[str, str]]:
    """Return ``(storage_key, display_name)`` pairs after schema checks."""

    files = set(data.files)
    if "__hist_keys" not in files or "__hist_names" not in files:
        derived = {
            k.replace("__contents", "").replace("__meta", "")
            for k in files
            if k not in ("__hist_keys", "__hist_names")
        }
        return [(name, name) for name in derived]

    storage_keys = [str(x) for x in data["__hist_keys"]]
    display_names = [str(x) for x in data["__hist_names"]]
    if len(storage_keys) != len(display_names):
        raise ValueError(
            f"load_histograms({path!r}): __hist_keys length "
            f"{len(storage_keys)} != __hist_names length {len(display_names)}"
        )
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    for key, name in zip(storage_keys, display_names):
        if key in seen_keys:
            raise ValueError(
                f"load_histograms({path!r}): duplicate storage key {key!r} "
                "in __hist_keys"
            )
        if name in seen_names:
            raise ValueError(
                f"load_histograms({path!r}): duplicate display name {name!r} "
                "in __hist_names"
            )
        seen_keys.add(key)
        seen_names.add(name)
        contents_name = key + "__contents"
        meta_name = key + "__meta"
        if contents_name not in files:
            raise ValueError(
                f"load_histograms({path!r}): missing dataset {contents_name!r} "
                f"for mapped key {key!r}"
            )
        if meta_name not in files:
            raise ValueError(
                f"load_histograms({path!r}): missing dataset {meta_name!r} "
                f"for mapped key {key!r}"
            )
    return list(zip(storage_keys, display_names))


# ---------------------------------------------------------------------------
# Shared numeric-schema validation for histogram payloads
# (review finding PF-review4-io)
# ---------------------------------------------------------------------------

_META_LENGTH = 3


def _validate_histogram_payload(
    context: str,
    name: str,
    meta: np.ndarray,
    contents: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    """Validate one persisted histogram ``(meta, contents)`` pair.

    Shared numeric-schema validator used by BOTH persistence backends,
    :func:`load_histograms` (NPZ) and :func:`load_histograms_hdf5`
    (HDF5), so every loader enforces the same schema *before* any
    :class:`~na49_mixing.histogram.Histogram1D` object is constructed and
    no unvalidated array is ever assigned to the private ``_contents``
    field:

    - *meta* must be a one-dimensional array of length three with a
      real numeric dtype (``kind`` in ``{'f', 'i', 'u'}``); its first
      entry (*nbins*) must be finite, positive and exactly integral
      (a fractional value such as ``2.5`` is refused outright instead of
      silently truncating); its last two entries are finite edges ordered
      ``xlow < xhigh``; the derived span ``xhigh - xlow`` and the bin
      width ``span / nbins`` must both be finite and positive;
    - *contents* must be a one-dimensional array of exact length
      ``nbins + 2`` whose dtype kind lies in ``{'f', 'i', 'u'}`` and
      whose values are all finite. Contents checks live in
      :func:`~na49_mixing.histogram._coerce_histogram_contents` so
      :meth:`~na49_mixing.histogram.Histogram1D.from_dict` and both
      persistence backends share one rule.

    Finiteness of stored contents is enforced under the same policy that
    rejects non-finite fill inputs. The public ``fill``/``fill_array``
    surface cannot write non-finite storage, so supported producers never
    produce such payloads; infinite or NaN bin contents would render the
    lazily built sampling CDF undefined while looking like an ordinary
    histogram, so only hand-crafted or corrupted files can carry them.

    On success the metadata is returned as ``(nbins: int, xlow: float,
    xhigh: float)`` and *contents* as a fresh owned native-endian
    ``float64`` copy. Storage dtypes other than float64 are normalized
    here rather than passed through verbatim; value identity is preserved
    exactly. All violations raise :class:`ValueError` naming *context*
    (which carries the file path) and the histogram display name.
    """
    label = f"{context}: histogram {name!r}"

    # -- metadata triple --------------------------------------------------
    # Rank and length first: report the geometry regardless of what the
    # individual entries contain.
    if meta.ndim != 1 or meta.shape[0] != _META_LENGTH:
        raise ValueError(
            f"{label} meta has shape {meta.shape}, expected ({_META_LENGTH},)"
        )
    if meta.dtype.kind not in "fiu":
        raise ValueError(
            f"{label} meta must have a real numeric dtype "
            f"(kind in {{'f', 'i', 'u'}}), got {meta.dtype!r}"
        )
    m_nbins = float(meta[0])
    if not math.isfinite(m_nbins):
        raise ValueError(f"{label} nbins must be finite, got {m_nbins!r}")
    if m_nbins <= 0:
        raise ValueError(f"{label} nbins must be positive, got {m_nbins!r}")
    if not m_nbins.is_integer():
        raise ValueError(
            f"{label} nbins must be exactly integral, got {m_nbins!r}; "
            "a fractional value would otherwise truncate silently"
        )
    nbins = int(m_nbins)
    xlow = float(meta[1])
    xhigh = float(meta[2])
    if not (math.isfinite(xlow) and math.isfinite(xhigh)):
        raise ValueError(
            f"{label} edges must be finite, got xlow={xlow!r}, "
            f"xhigh={xhigh!r}"
        )
    if xhigh <= xlow:
        raise ValueError(
            f"{label} edges must satisfy xlow < xhigh, got xlow={xlow!r}, "
            f"xhigh={xhigh!r}"
        )
    _require_finite_bin_geometry(nbins, xlow, xhigh, label=label)

    # -- contents ---------------------------------------------------------
    # Rank, dtype kind, exact length, and finiteness live in the shared
    # helper so from_dict() and both file backends cannot diverge.
    owned = _coerce_histogram_contents(contents, nbins, label=label)
    return nbins, xlow, xhigh, owned


def load_histograms(path: str) -> Dict[str, Histogram1D]:
    """Load histograms from a ``.npz`` file saved by :func:`save_histograms`.

    Returns a dict of name → :class:`~na49_mixing.histogram.Histogram1D`.

    Files are pre-validated as ZIP (.npz) containers before
    :func:`numpy.load` runs: garbage-byte, zero-byte, and truncated
    archives are refused with :class:`ValueError` naming *path* ("not an
    NPZ archive") rather than leaking numpy/zipfile internals. Valid
    archives load exactly as before.

    Files that carry ``__hist_keys`` / ``__hist_names`` are validated as a
    mapping schema before any histogram is reconstructed: equal lengths,
    unique storage keys, unique display names, and one contents/meta pair
    per listed key. Mapping errors name *path*.

    Every histogram entry then passes the shared numeric-schema validator
    (see :func:`_validate_histogram_payload`) before construction: a
    real-numeric one-dimensional ``__meta`` triple with finite, positive,
    exactly integral ``nbins``, finite ordered edges whose derived span
    and bin width stay finite and positive, plus a
    one-dimensional real-numeric ``__contents`` array of exact length
    ``nbins + 2`` whose values are all finite. Accepted contents come
    back as fresh owned ``float64`` arrays; malformed entries raise
    :class:`ValueError` naming *path* and the histogram display name.

    Each returned histogram owns its own copy of the stored contents, and
    the archive handle is closed before this function returns, whether
    loading succeeds or raises.
    """
    result: Dict[str, Histogram1D] = {}
    with _open_npz_archive(path, "load_histograms") as data:
        pairs = _npz_histogram_pairs(path, data)
        for key, name in pairs:
            meta = np.asarray(data[key + "__meta"])
            raw_contents = np.asarray(data[key + "__contents"])
            nbins, xlow, xhigh, contents = _validate_histogram_payload(
                f"load_histograms({path!r})", name, meta, raw_contents,
            )
            h = Histogram1D(nbins, xlow, xhigh, name=name)
            h._contents = contents
            result[name] = h
    return result


# ---------------------------------------------------------------------------
# HDF5 persistence (requires h5py)
# ---------------------------------------------------------------------------

def save_histograms_hdf5(path: str, histograms: Dict[str, Histogram1D]) -> None:
    """Save histograms to an HDF5 file using ``h5py``.

    Save-time payload validation boundary: every histogram payload — the
    metadata triple plus contents rank, dtype, length and finiteness — is
    checked through :func:`_validate_histogram_payload` BEFORE the
    destination file is opened, applying the exact rule
    :func:`load_histograms_hdf5` enforces on read-back. Group-name
    collisions after "/" normalization are likewise resolved up front.
    A rejected save therefore never creates a partial or empty file: the
    ``h5py.File(path, "w")`` open happens only after the whole mapping
    has passed validation.

    Parameters
    ----------
    path : str
        Output file path.
    histograms : dict
        Mapping of name → histogram.

    Raises
    ------
    ValueError
        If two distinct names map to the same HDF5 group name, or if any
        histogram payload fails the shared numeric-schema validation.
        Messages carry the path and the offending display name, matching
        :func:`save_histograms`.
    """
    import h5py  # optional dependency
    prepared: list[tuple[str, str, int, float, float, np.ndarray]] = []
    used: Dict[str, str] = {}
    context = f"save_histograms_hdf5({path!r})"
    for name, h in histograms.items():
        safe = name.replace("/", "_")
        if safe in used:
            raise ValueError(
                "save_histograms_hdf5(): group-name collision — "
                f"{used[safe]!r} and {name!r} both map to {safe!r}"
            )
        used[safe] = name
        # Payload boundary before the destination is opened (review5
        # finding 4): written values are exactly what was validated.
        meta = np.array([h.nbins, h.xlow, h.xhigh], dtype=np.float64)
        nbins, xlow, xhigh, stored = _validate_histogram_payload(
            context, name, meta, h._contents,
        )
        prepared.append((safe, name, nbins, xlow, xhigh, stored))
    with h5py.File(path, "w") as f:
        for safe, name, nbins, xlow, xhigh, stored in prepared:
            grp = f.create_group(safe)
            grp.create_dataset("contents", data=stored)
            grp.attrs["nbins"] = nbins
            grp.attrs["xlow"] = xlow
            grp.attrs["xhigh"] = xhigh
            grp.attrs["name"] = name


def load_histograms_hdf5(path: str) -> Dict[str, Histogram1D]:
    """Load histograms from an HDF5 file saved by :func:`save_histograms_hdf5`.

    Returns a dict of name → :class:`~na49_mixing.histogram.Histogram1D`,
    where the name is taken from the group's ``name`` attribute (decoded
    when stored as bytes) and falls back to the HDF5 group name.

    Schema enforcement, symmetric with the NPZ loader: each root entry in
    iteration order passes four checks before a histogram is built.

    - the entry must be an HDF5 group; datasets or other objects at the
      file root raise ``ValueError`` naming the path and the entry;
    - display names are tracked while iterating groups. A name already
      carried by an earlier group raises ``ValueError`` naming the path,
      the duplicate display name and both offending group names —
      previously the later group silently replaced the earlier histogram
      in the result;
    - every root group must carry the ``nbins``, ``xlow`` and ``xhigh``
      attributes; missing attributes raise a schema ``ValueError``
      naming the path, the entry and the absent attribute names;
    - the group must carry its bin contents as an actual ``contents``
      dataset (a missing member, or one stored as another group, raise a
      schema ``ValueError`` naming the path and the entry).

    Each group then passes the same shared numeric-schema validator as
    the NPZ backend (see :func:`_validate_histogram_payload`), applied
    before histogram construction: a real-numeric length-three metadata
    triple built from the ``nbins``/``xlow``/``xhigh`` attributes with a
    finite, positive, exactly integral ``nbins``, finite ordered edges
    (``xlow < xhigh``) whose derived span and bin width stay finite and
    positive, plus a one-dimensional real-numeric contents
    dataset of exact length ``nbins + 2`` whose values are all finite.
    Accepted contents come back as fresh owned ``float64`` arrays.
    Malformed entries raise :class:`ValueError` naming *path* and the
    histogram name.
    """
    import h5py
    result: Dict[str, Histogram1D] = {}
    seen_names: Dict[str, str] = {}
    with h5py.File(path, "r") as f:
        for entry_name in f.keys():
            node = f[entry_name]
            if not isinstance(node, h5py.Group):
                raise ValueError(
                    f"load_histograms_hdf5({path!r}): root entry "
                    f"{entry_name!r} is {type(node).__name__}, expected "
                    "an HDF5 group carrying one histogram (attributes "
                    "nbins/xlow/xhigh and a 'contents' dataset)"
                )
            orig = node.attrs.get("name", entry_name)
            if isinstance(orig, bytes):
                orig = orig.decode("utf-8")
            display = str(orig)
            if display in seen_names:
                raise ValueError(
                    f"load_histograms_hdf5({path!r}): duplicate display "
                    f"name {display!r} carried by groups "
                    f"{seen_names[display]!r} and {entry_name!r}; the "
                    "later group would silently overwrite the earlier "
                    "histogram, so nothing is loaded"
                )
            seen_names[display] = entry_name
            missing = [
                attr for attr in ("nbins", "xlow", "xhigh")
                if attr not in node.attrs
            ]
            if missing:
                raise ValueError(
                    f"load_histograms_hdf5({path!r}): group "
                    f"{entry_name!r} is missing required attribute(s) "
                    f"{', '.join(repr(a) for a in missing)}; a histogram "
                    "group must carry nbins, xlow and xhigh exactly as "
                    "save_histograms_hdf5() writes them"
                )
            if "contents" not in node:
                raise ValueError(
                    f"load_histograms_hdf5({path!r}): group "
                    f"{entry_name!r} is missing required dataset "
                    "'contents'; a histogram group must carry its bin "
                    "contents under that key"
                )
            member = node["contents"]
            if not isinstance(member, h5py.Dataset):
                raise ValueError(
                    f"load_histograms_hdf5({path!r}): group "
                    f"{entry_name!r} stores 'contents' as "
                    f"{type(member).__name__}, expected an HDF5 dataset "
                    "holding the bin contents"
                )
            meta = np.array([
                node.attrs["nbins"],
                node.attrs["xlow"],
                node.attrs["xhigh"],
            ])
            raw_contents = np.array(member)
            nbins, xlow, xhigh, contents = _validate_histogram_payload(
                f"load_histograms_hdf5({path!r})", display, meta, raw_contents,
            )
            h = Histogram1D(nbins, xlow, xhigh, name=display)
            h._contents = contents
            result[display] = h
    return result


# ---------------------------------------------------------------------------
# Simple event record persistence (NumPy structured arrays / npz)
# ---------------------------------------------------------------------------

def save_events_npz(
    path: str,
    particle_arrays: Dict[str, np.ndarray],
    event_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save particle and event arrays to a ``.npz`` file.

    Parameters
    ----------
    path : str
        Output file path.
    particle_arrays : dict
        Per-particle arrays, e.g. ``{'px': ..., 'py': ..., 'pz': ..., 'id': ...}``.
    event_arrays : dict, optional
        Per-event arrays, e.g. ``{'n_run': ..., 'n_event': ...}``.
    """
    to_save: Dict[str, np.ndarray] = {
        "part_" + k: v for k, v in particle_arrays.items()
    }
    if event_arrays:
        to_save.update({"evt_" + k: v for k, v in event_arrays.items()})
    np.savez_compressed(path, **to_save)


def load_events_npz(path: str):
    """Load arrays from a ``.npz`` file saved by :func:`save_events_npz`.

    Returns ``(particle_arrays, event_arrays)`` as plain dicts. Each dict
    holds its own copy of the stored arrays; keys without the ``part_`` /
    ``evt_`` prefix are ignored. Files are pre-validated as ZIP (.npz)
    containers before :func:`numpy.load` runs: garbage-byte, zero-byte, and
    truncated archives are refused with :class:`ValueError` naming *path*
    ("not an NPZ archive") rather than leaking numpy/zipfile internals. The
    archive handle is closed before this function returns, whether loading
    succeeds or raises.
    """
    part: Dict[str, np.ndarray] = {}
    evt: Dict[str, np.ndarray] = {}
    with _open_npz_archive(path, "load_events_npz") as data:
        for k in data.files:
            if k.startswith("part_"):
                part[k[5:]] = data[k].copy()
            elif k.startswith("evt_"):
                evt[k[4:]] = data[k].copy()
    return part, evt
