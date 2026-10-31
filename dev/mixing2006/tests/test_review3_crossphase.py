"""Cross-phase probes beyond the landed review3 phases (Phase P7 extensions).

Covers the focused cases from the assignment's cross-phase list:

- **P3 species-order parity** — behavior when the companion
  ``multispecies_constrained_summary.json`` disagrees with the details NPZ
  about *species order* (not length). ``validate_multispecies_details_npz``
  compares species keys against the summary as an ordered list, mirroring
  the component_keys/component_labels rule. Permutations of either artifact
  are rejected with ``ValueError`` naming the details path and the
  offending key, while a legitimately produced run still loads end-to-end
  through ``bootstrap._load_run`` (every production consumer addresses
  per-species arrays by tag, never by position).
- **P6 round trip under frozen finite routing** — histograms filled at the
  range edges (value exactly at xlow, just below xlow, exactly at xhigh,
  just above xhigh) survive ``save_histograms`` / ``load_histograms``
  bit-for-bit including under-/overflow bins, mirrored on the HDF5 backend
  when h5py is present.
- **P5/P1 interplay** — non-finite fit-range endpoints raising from the
  histogram policy is owned by ``tests/test_fits_behavior.py``
  (classes ``TestMultiFitterFitRangeThroughFindBin`` and
  ``TestSpectraFitInterplayWithPolicy``), which exercise every public
  consumer routing ranges through ``find_bin``; no duplication here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import helpers.benchmarking.bootstrap_multispecies_robustness as bootstrap_mod
from helpers.benchmarking.benchmark_multispecies_constrained import (
    SpeciesData,
    build_details_npz_payload,
)
from helpers.benchmarking.sps_core import (
    DETAILS_NPZ_SCHEMA_VERSION,
    DETAILS_NPZ_REQUIRED_STRING_KEYS,
    load_multispecies_details_npz,
)
from na49_mixing.histogram import Histogram1D

DETAILS_NAME = "multispecies_constrained_details.npz"
SUMMARY_NAME = "multispecies_constrained_summary.json"

COMPONENT_KEYS = ["rho0", "omega"]
COMPONENT_WEIGHTS = [0.6, 0.4]

# Two species with clearly different value levels so any cross-wiring
# would be visible immediately.
SPECIES_SPEC = {
    # key: (tag, pdg, mass, y_values, pred_values)
    "pi-": ("pim", -211, 0.13957, [1.0, 2.0], [4.0, 5.0]),
    "k+": ("kp", 321, 0.493677, [10.0], [11.0]),
}


def _species_data(key: str) -> SpeciesData:
    tag, pdg, mass, y_vals, res_vals = SPECIES_SPEC[key]
    n = len(y_vals)
    return SpeciesData(
        key=key,
        name=key,
        tag=tag,
        pdg=int(pdg),
        mass_gev=float(mass),
        na49_path=None,  # type: ignore[arg-type]
        xf=np.linspace(0.05, 0.15, n),
        pt=np.full(n, 0.20),
        y=np.asarray(y_vals, dtype=float),
        sigma=np.asarray([0.1 + 0.01 * i for i in range(n)], dtype=float),
        xedges=np.asarray([0.0, 0.2, 0.4], dtype=float),
        pedges=np.asarray([0.1, 0.5], dtype=float),
        ix_data=np.arange(n, dtype=np.int64) % 2,
        ip_data=np.zeros(n, dtype=np.int64),
        train_idx=np.arange(n, dtype=np.int64),
        test_idx=np.arange(n, dtype=np.int64),
        matrix=np.zeros((n, len(COMPONENT_KEYS)), dtype=float),
        pred_res=np.asarray(res_vals, dtype=float),
        pred_res_sigma=np.full(n, 0.05),
    )


def _external_maps() -> tuple[dict, dict]:
    ext_models = ("PYTHIA8_external", "EPOS_external", "UrQMD_external")
    ext_pred: dict[str, dict[str, np.ndarray]] = {m: {} for m in ext_models}
    ext_sigma: dict[str, dict[str, np.ndarray]] = {m: {} for m in ext_models}
    level = 1.0
    for model in ext_models:
        for key in SPECIES_SPEC:
            n = len(SPECIES_SPEC[key][3])
            ext_pred[model][key] = np.full(n, level)
            ext_sigma[model][key] = np.full(n, 0.02 * level)
            level += 1.0
    return ext_pred, ext_sigma


def _payload(species_keys_order=None) -> dict[str, np.ndarray]:
    order = species_keys_order or list(SPECIES_SPEC)
    species_map = {key: _species_data(key) for key in order}
    ext_pred, ext_sigma = _external_maps()
    return build_details_npz_payload(
        component_weights=COMPONENT_WEIGHTS,
        component_keys=list(COMPONENT_KEYS),
        component_labels=list(COMPONENT_KEYS),
        species_data_by_key=species_map,
        pred_full_stack=np.concatenate(
            [species_map[k].pred_res for k in order]),
        y_full_stack=np.concatenate([species_map[k].y for k in order]),
        sigma_full_stack=np.concatenate(
            [species_map[k].sigma for k in order]),
        n_effective=len(order),
        prior_report={"nstar_smooth": {"ok": False}, "prior_total_rows": 0},
        fit_solver={"method": "scipy.optimize.nnls", "rnorm": 0.25},
        fit_weight_mode="relative",
        fit_objective="nnls_weighted",
        weak_feed_policy="subtract",
        histogram_convention="invariant",
        external_pred_by_model=ext_pred,
        external_sigma_by_model=ext_sigma,
    )


def _summary(species_order=None) -> dict[str, object]:
    order = species_order or list(SPECIES_SPEC)
    return {
        "weak_feed_policy": "subtract",
        "histogram_convention": "invariant",
        "species": {
            key: {
                "name": key,
                "tag": SPECIES_SPEC[key][0],
                "pdg": int(SPECIES_SPEC[key][1]),
                "na49_table": "in-memory-fixture",
                "n_points": len(SPECIES_SPEC[key][3]),
                "full_points": len(SPECIES_SPEC[key][3]),
            }
            for key in order
        },
        "resonance": {
            "n_components": len(COMPONENT_KEYS),
            "component_keys": list(COMPONENT_KEYS),
            "component_labels": list(COMPONENT_KEYS),
            "weights_by_key": dict(zip(COMPONENT_KEYS, COMPONENT_WEIGHTS)),
            "global_metrics": {"n_params": len(COMPONENT_KEYS)},
            "priors": {},
        },
        "fit_weighting": {"mode": "relative"},
        "fit_solver": {"method": "scipy.optimize.nnls"},
    }


def _write_run_dir(tmp_path: Path, payload: dict, summary: dict) -> Path:
    run_dir = tmp_path / "constrained_refit_order_fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(run_dir / DETAILS_NAME), **payload)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# P3: species-order divergence between summary JSON and details NPZ
# ---------------------------------------------------------------------------

class TestSpeciesOrderParity:
    def test_ordered_production_run_loads_end_to_end_with_correct_tag_data(
            self, tmp_path):
        """Control: a legitimately produced run passes and feeds consumers.

        The producer writes ``species_keys`` into the details NPZ and the
        ``species`` mapping into the summary JSON from one common ordering,
        so both carry the same sequence here. Ordered enforcement therefore
        never rejects such a run, and every species array reaches the
        consumer under its own tag, asserted end-to-end through
        ``bootstrap._load_run``.
        """
        run_dir = _write_run_dir(tmp_path, _payload(), _summary())
        summary = json.loads((run_dir / SUMMARY_NAME).read_text("utf-8"))
        with load_multispecies_details_npz(
                run_dir / DETAILS_NAME, summary_payload=summary) as zarr:
            npz_order = [str(x) for x in zarr["species_keys"]]
        assert npz_order == list(SPECIES_SPEC)          # same order as JSON

        rp = bootstrap_mod._load_run(run_dir=run_dir, label="ordered",
                                     species_csv="")
        for key, (_tag, _pdg, _mass, y_vals, res_vals) in SPECIES_SPEC.items():
            got = rp.species_arrays[key]
            np.testing.assert_allclose(got.y, y_vals)
            got_res = got.pred_by_model["Resonance_NNLS_shared"]
            np.testing.assert_allclose(got_res, res_vals)

    def test_permuted_summary_species_order_is_rejected(self, tmp_path):
        """Summary JSON written in another order than the NPZ is refused.

        ``validate_multispecies_details_npz`` compares the species keys as
        an ordered list against the summary JSON (the same rule applied to
        component_keys/component_labels); the error names the details path
        and the offending key.
        """
        permuted = list(reversed(list(SPECIES_SPEC)))
        assert permuted != list(SPECIES_SPEC)

        run_dir = _write_run_dir(
            tmp_path,
            _payload(),                    # NPZ keeps production order
            _summary(species_order=permuted),   # JSON written permuted
        )
        summary = json.loads((run_dir / SUMMARY_NAME).read_text("utf-8"))
        assert list(summary["species"]) == permuted     # JSON reordered

        details_path = run_dir / DETAILS_NAME
        with pytest.raises(ValueError) as excinfo:
            with load_multispecies_details_npz(
                    details_path, summary_payload=summary):
                pass
        msg = str(excinfo.value)
        assert "species_keys" in msg
        assert "order does not match summary JSON" in msg
        assert str(details_path) in msg

    def test_permuted_npz_species_keys_are_rejected_too(self, tmp_path):
        """Symmetric direction: NPZ writes species_keys reversed."""
        permuted = list(reversed(list(SPECIES_SPEC)))
        run_dir = _write_run_dir(
            tmp_path,
            _payload(species_keys_order=permuted),
            _summary(),                     # JSON keeps production order
        )
        details_path = run_dir / DETAILS_NAME
        with pytest.raises(ValueError) as excinfo:
            with load_multispecies_details_npz(
                    details_path, summary_payload=_summary()):
                pass
        msg = str(excinfo.value)
        assert "species_keys" in msg
        assert "order does not match summary JSON" in msg
        assert str(details_path) in msg

    def test_desired_end_state_rejects_species_order_divergence(
            self, tmp_path):
        """Review-parity end state: species order checked like components.

        Formerly carried as ``xfail(strict=False)`` (REVIEW3-P3-GAP) while
        the comparison was set-based; the ordered comparison landed, so
        this pins the contract permanently alongside the tighter
        assertions in the neighboring tests.
        """
        permuted = list(reversed(list(SPECIES_SPEC)))
        run_dir = _write_run_dir(tmp_path, _payload(),
                                 _summary(species_order=permuted))
        summary = json.loads((run_dir / SUMMARY_NAME).read_text("utf-8"))
        with pytest.raises(ValueError, match="order does not match") as (
                excinfo):
            with load_multispecies_details_npz(
                    run_dir / DETAILS_NAME, summary_payload=summary):
                pass
        msg = str(excinfo.value)
        assert "species order" in msg
        assert str(run_dir / DETAILS_NAME) in msg

    def test_component_order_divergence_is_still_rejected_control(self,
                                                                  tmp_path):
        """Control: proves the guard difference is real, not assumed."""
        flipped = list(reversed(COMPONENT_KEYS))
        payload = _payload()
        payload["component_keys"] = np.asarray(flipped, dtype=str)
        payload["component_labels"] = np.asarray(flipped, dtype=str)
        run_dir = _write_run_dir(tmp_path, payload, _summary())
        summary = json.loads((run_dir / SUMMARY_NAME).read_text("utf-8"))
        with pytest.raises(ValueError) as excinfo:
            with load_multispecies_details_npz(
                    run_dir / DETAILS_NAME, summary_payload=summary):
                pass
        msg = str(excinfo.value)
        assert "component" in msg
        assert "does not match summary JSON" in msg
        assert str(run_dir / DETAILS_NAME) in msg

    def test_baseline_payload_carries_expected_schema_markers(self):
        payload = _payload()
        assert int(payload["schema_version"]) == DETAILS_NPZ_SCHEMA_VERSION
        for key in DETAILS_NPZ_REQUIRED_STRING_KEYS:
            assert payload[key].dtype.kind == "U", key


# ---------------------------------------------------------------------------
# P6: edge-heavy contents survive the save/load round trip bit-for-bit
# ---------------------------------------------------------------------------

W_JUST_BELOW = 7.0     # lands in underflow bin 0


@pytest.fixture
def require_h5py():
    """Skip only the calling HDF5 test when the optional backend is absent."""
    pytest.importorskip("h5py")
W_AT_XLOW = 13.0       # bin 1
W_INTERIOR = 21.0      # middle bin
W_AT_XHIGH = 3.0       # overflow bin nbins+1
W_ABOVE_XHIGH = 5.0    # overflow bin nbins+1 (deposited separately)


def _edge_histogram(name: str) -> Histogram1D:
    h = Histogram1D(8, 1.0, 3.0, name=name)
    width = h.bin_width
    h.fill_array(
        np.array([1.0 - 0.25 * width,      # strictly below xlow
                  1.0,                      # exactly the lower edge
                  2.0,                      # plain interior bin
                  3.0,                      # exactly the upper edge
                  3.0 + 0.25 * width]),     # strictly above xhigh
        np.array([W_JUST_BELOW, W_AT_XLOW, W_INTERIOR,
                  W_AT_XHIGH, W_ABOVE_XHIGH]),
    )
    return h


class TestEdgeHeavyRoundTrip:
    def test_npz_round_trip_preserves_boundary_routing_exactly(self,
                                                               tmp_path):
        from na49_mixing.data_io import load_histograms, save_histograms
        source = _edge_histogram("edges")
        # Frozen routing sanity before saving: where did each value land?
        assert source.get_bin_content(0) == pytest.approx(W_JUST_BELOW)
        assert source.get_bin_content(1) == pytest.approx(W_AT_XLOW)
        interior_bin = source.find_bin(2.0)
        assert source.get_bin_content(interior_bin) == pytest.approx(
            W_INTERIOR)
        overflow_bin = source.nbins + 1
        assert source.get_bin_content(overflow_bin) == pytest.approx(
            W_AT_XHIGH + W_ABOVE_XHIGH)

        path = tmp_path / "edges.npz"
        save_histograms(str(path), {"edges": source})
        loaded = load_histograms(str(path))["edges"]

        # Bit-for-bit identity across every slot, underflow included.
        assert loaded._contents.tobytes() == source._contents.tobytes()
        # Re-derive the placements on the loaded object through find_bin,
        # proving the frozen finite routing survives persistence too.
        assert loaded.get_bin_content(loaded.find_bin(1.0)) == pytest.approx(
            W_AT_XLOW)
        assert loaded.get_bin_content(
            loaded.find_bin(1.0 - 0.25 * loaded.bin_width)) == pytest.approx(
            W_JUST_BELOW)
        assert loaded.get_bin_content(
            loaded.find_bin(3.0 + 0.25 * loaded.bin_width)) == pytest.approx(
            W_AT_XHIGH + W_ABOVE_XHIGH)
        assert loaded.name == "edges"

    def test_hdf5_round_trip_preserves_boundary_routing_too(
            self, tmp_path, require_h5py):
        import h5py  # noqa: F401
        from na49_mixing.data_io import (load_histograms_hdf5,
                                         save_histograms_hdf5)
        source = _edge_histogram("edges")
        path = tmp_path / "edges.h5"
        save_histograms_hdf5(str(path), {"edges": source})
        loaded = load_histograms_hdf5(str(path))["edges"]
        assert loaded._contents.tobytes() == source._contents.tobytes()
        assert loaded.nbins == source.nbins
        assert loaded.xlow == pytest.approx(source.xlow)
        assert loaded.xhigh == pytest.approx(source.xhigh)
