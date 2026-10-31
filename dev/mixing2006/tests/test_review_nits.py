"""Nits from friend reviews: validation, I/O schema, and hang guards."""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.data_io import load_histograms, save_histograms
from na49_mixing.decays import DecayMode, ParticleDecays
from na49_mixing.glauber import NucleusSpec
from na49_mixing.histogram import Histogram1D
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked


_DELTA = os.path.join(
    os.path.dirname(__file__),
    "..",
    "na49_mixing",
    "generators",
    "michal",
    "deltas-to-pion",
    "delta++.gen",
)


class TestDecayTableValidation:
    def test_rejects_unsupported_daughter_count(self):
        with pytest.raises(ValueError, match="2 or 3"):
            DecayMode(1, [211])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="expected 2 daughters"):
            DecayMode(2, [211])

    def test_rejects_decreasing_cumulative(self):
        d = ParticleDecays(113)
        d.add_mode(0.5, DecayMode(2, [211, -211]))
        with pytest.raises(ValueError, match="nondecreasing"):
            d.add_mode(0.2, DecayMode(2, [111, 111]))

    def test_get_mode_rejects_empty_and_unclosed_tables(self):
        empty = ParticleDecays(113)
        with pytest.raises(ValueError, match="no modes"):
            empty.get_mode()
        open_table = ParticleDecays(113)
        open_table.add_mode(0.5, DecayMode(2, [211, -211]))
        with pytest.raises(ValueError, match="final cumulative"):
            open_table.get_mode()


class TestGenerationConfigValidation:
    def test_rejects_negative_events_and_passes(self):
        with pytest.raises(ValueError, match="n_events"):
            GenerationConfig(gen_file=_DELTA, n_events=-1, seed=1)
        with pytest.raises(ValueError, match="max_decay_passes"):
            GenerationConfig(
                gen_file=_DELTA, n_events=1, seed=1, max_decay_passes=0
            )
        with pytest.raises(ValueError, match="seed"):
            GenerationConfig(gen_file=_DELTA, n_events=1, seed=-3)

    def test_rejects_fractional_and_boolean_integers(self):
        with pytest.raises(ValueError, match="n_events"):
            GenerationConfig(gen_file=_DELTA, n_events=1.5, seed=1)
        with pytest.raises(ValueError, match="n_events"):
            GenerationConfig(gen_file=_DELTA, n_events=True, seed=1)
        with pytest.raises(ValueError, match="seed"):
            GenerationConfig(gen_file=_DELTA, n_events=1, seed=1.2)
        with pytest.raises(ValueError, match="max_decay_passes"):
            GenerationConfig(
                gen_file=_DELTA, n_events=1, seed=1, max_decay_passes=2.5
            )

    def test_normalizes_numpy_integers_and_integral_floats(self):
        cfg = GenerationConfig(
            gen_file=_DELTA,
            n_events=np.int64(2),
            seed=np.int32(3),
            max_decay_passes=4.0,
        )
        assert cfg.n_events == 2
        assert cfg.seed == 3
        assert cfg.max_decay_passes == 4
        assert type(cfg.n_events) is int
        assert type(cfg.seed) is int
        assert type(cfg.max_decay_passes) is int

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("158", id="str-numeric"),
            pytest.param(158, id="plain-int"),
            pytest.param(157.0, id="integral-float"),
            pytest.param(np.int64(320), id="np-int64"),
            pytest.param(np.float32(157.5), id="np-float32"),
        ],
    )
    def test_beam_momentum_stored_normalized_to_builtin_float(self, raw):
        """The converted value is stored like the sibling integer fields."""
        cfg = GenerationConfig(
            gen_file=_DELTA, n_events=1, seed=1, beam_momentum=raw
        )
        assert type(cfg.beam_momentum) is float
        assert cfg.beam_momentum == (158.0 if isinstance(raw, str) else float(raw))

    def test_beam_momentum_invalid_inputs_keep_today_message(self):
        with pytest.raises(
            ValueError, match="beam_momentum must be a finite positive"
        ):
            GenerationConfig(
                gen_file=_DELTA, n_events=1, seed=1, beam_momentum=0.0
            )
        # Non-numeric strings keep the plain float() conversion error.
        with pytest.raises(ValueError, match="could not convert string to float"):
            GenerationConfig(
                gen_file=_DELTA, n_events=1, seed=1,
                beam_momentum="not-a-number",
            )

    def test_beam_momentum_normalization_keeps_config_frozen(self):
        """Storing via ``object.__setattr__`` preserves frozen semantics."""
        cfg = GenerationConfig(
            gen_file=_DELTA, n_events=1, seed=1,
            beam_momentum=np.float64(157.5),
        )
        assert type(cfg.beam_momentum) is float
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.beam_momentum = 200.0  # type: ignore[misc]
        replaced = dataclasses.replace(cfg, beam_momentum=np.int64(200))
        assert replaced.beam_momentum == 200.0
        assert type(replaced.beam_momentum) is float
        assert cfg.beam_momentum == 157.5

    def test_zero_events_reports_zero(self):
        cfg = GenerationConfig(gen_file=_DELTA, n_events=0, seed=1)
        out = generate_tracks_chunked(cfg)
        assert int(out["n_events"]) == 0
        assert np.asarray(out["pdg"]).size == 0

    def test_point_target_range_must_include_one(self):
        spec = NucleusSpec(name="p1", A=1, Z=1)
        with pytest.raises(ValueError, match="must include 1"):
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                nucleus=spec,
                target_wound_range=(2, 4),
            )


class TestWoundRangeConfigValidation:
    """Config-level acceptance rules for ``target_wound_range``.

    Slices are generated from the interaction-weighted distribution
    restricted to [lo, hi] (``WoundedNucleonSampler.sample_event_in_range``),
    so acceptance no longer depends on how rare a slice is under
    minimum bias: only structurally or numerically empty slices are
    refused at construction.
    """

    def _c12_spec(self) -> NucleusSpec:
        return NucleusSpec(
            name="C12",
            A=12,
            Z=6,
            profile="harmonic_oscillator",
            radius_fm=1.88,
            sigma_nn_inel_mb=31.7,
        )

    def _pb208_spec(self) -> NucleusSpec:
        return NucleusSpec(
            name="Pb208",
            A=208,
            Z=82,
            profile="woods_saxon",
            radius_fm=6.62,
            diffuseness_fm=0.546,
            sigma_nn_inel_mb=31.7,
        )

    def test_unreachable_range_fails_at_config_time(self):
        """C12 K=50 exceeds the truncated support (A=12)."""

        with pytest.raises(ValueError, match="lo > A") as excinfo:
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                nucleus=self._c12_spec(),
                target_wound_range=(50, 50),
                use_soft_decay_factory=True,
            )
        assert "unreachable" in str(excinfo.value)

    def test_range_above_support_fails_at_config_time_and_never_reaches_generation(self, monkeypatch):
        """C12 (13, 13): lo > A is structurally impossible.

        The truncated-at-A sampler gives K > A zero probability, so the
        config must be rejected before any chunk worker can run.
        """
        import na49_mixing.parallel as parallel

        entered = []
        original_worker = parallel._generate_pa_tracks_chunk_worker

        def spy_worker(**kwargs):
            entered.append(kwargs)
            return original_worker(**kwargs)

        monkeypatch.setattr(parallel, "_generate_pa_tracks_chunk_worker", spy_worker)
        with pytest.raises(ValueError, match="lo > A"):
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                nucleus=self._c12_spec(),
                target_wound_range=(13, 13),
            )
        assert entered == []

    def test_pb208_low_central_slice_accepted(self):
        cfg = GenerationConfig(
            gen_file=_DELTA,
            n_events=1,
            seed=1,
            nucleus=self._pb208_spec(),
            target_wound_range=(1, 8),
        )
        assert cfg.target_wound_range == (1, 8)

    @pytest.mark.parametrize("k_single", [21, 24, 25])
    def test_pb208_rare_single_bin_accepted_and_generated(self, k_single):
        """Pb208 single-bin slices K = 21/24/25 generate directly.

        Their expected min-bias fractions are 5.6e-7, 1.2e-8, and
        3.1e-9; under per-event rejection each requested event failed
        with near certainty. The range-conditioned draw populates the
        slice in one pass, so one event comes out immediately.
        """

        cfg = GenerationConfig(
            gen_file=_DELTA,
            n_events=1,
            seed=k_single,
            nucleus=self._pb208_spec(),
            target_wound_range=(k_single, k_single),
        )
        out = generate_tracks_chunked(cfg)
        wounds = np.asarray(out["n_wounds"], dtype=np.int64)
        # One projectile track plus one target track per wound.
        assert wounds.size == 1 + k_single
        assert np.all(wounds == k_single)

    def test_pb208_ultrarare_slice_inside_support_generates(self):
        """Pb208 (50, 60): expected min-bias fraction ~3.7e-28.

        The slice sits far below any min-bias reach, yet the
        conditioned sampler produces its events at fixed cost.
        """

        cfg = GenerationConfig(
            gen_file=_DELTA,
            n_events=1,
            seed=2,
            nucleus=self._pb208_spec(),
            target_wound_range=(50, 60),
            use_soft_decay_factory=True,
        )
        out = generate_tracks_chunked(cfg)
        wounds = np.asarray(out["n_wounds"], dtype=np.int64)
        assert wounds.size >= 51
        assert np.all((wounds >= 50) & (wounds <= 60))

    def test_zero_probability_slice_rejected_at_config_time(self, monkeypatch):
        """A slice whose restricted probability vanishes is refused.

        The analytic fraction is exactly zero when the conditioned
        sampler has nothing to draw; the config must reject before
        generation instead of producing an empty stream.
        """
        import na49_mixing.parallel as parallel

        monkeypatch.setattr(
            parallel, "_wound_range_fraction", lambda nucleus, lo, hi: 0.0
        )
        with pytest.raises(ValueError, match="identically zero"):
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                nucleus=self._c12_spec(),
                target_wound_range=(2, 4),
            )

    def test_moderate_c12_slice_accepted(self):
        cfg = GenerationConfig(
            gen_file=_DELTA,
            n_events=1,
            seed=1,
            nucleus=self._c12_spec(),
            target_wound_range=(2, 4),
        )
        assert cfg.target_wound_range == (2, 4)

    def test_wound_range_requires_nucleus(self):
        with pytest.raises(ValueError, match="requires nucleus"):
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                target_wound_range=(2, 4),
            )

    def test_wound_range_rejects_fractional_endpoints(self):
        spec = self._c12_spec()
        with pytest.raises(ValueError, match="target_wound_range lo"):
            GenerationConfig(
                gen_file=_DELTA,
                n_events=1,
                seed=1,
                nucleus=spec,
                target_wound_range=(1.5, 3),
            )


class TestHistogramNameRoundTrip:
    def test_spaces_preserved(self, tmp_path):
        h = Histogram1D(4, 0.0, 1.0, name="mass spectrum")
        h.fill(0.25)
        path = tmp_path / "h.npz"
        save_histograms(str(path), {"mass spectrum": h})
        loaded = load_histograms(str(path))
        assert "mass spectrum" in loaded
        assert loaded["mass spectrum"].name == "mass spectrum"
        np.testing.assert_allclose(loaded["mass spectrum"]._contents, h._contents)

    def test_rejects_mismatched_mapping_lengths(self, tmp_path):
        path = tmp_path / "mismatch.npz"
        contents = np.zeros(6)
        np.savez(
            path,
            __hist_keys=np.array(["a", "b"], dtype="U"),
            __hist_names=np.array(["A"], dtype="U"),
            a__contents=contents,
            a__meta=np.array([4, 0.0, 1.0], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="__hist_keys length"):
            load_histograms(str(path))

    def test_rejects_duplicate_storage_keys(self, tmp_path):
        path = tmp_path / "dupkey.npz"
        contents = np.zeros(6)
        meta = np.array([4, 0.0, 1.0], dtype=np.float64)
        np.savez(
            path,
            __hist_keys=np.array(["a", "a"], dtype="U"),
            __hist_names=np.array(["A", "B"], dtype="U"),
            a__contents=contents,
            a__meta=meta,
        )
        with pytest.raises(ValueError, match="duplicate storage key"):
            load_histograms(str(path))

    def test_rejects_duplicate_display_names(self, tmp_path):
        path = tmp_path / "dupname.npz"
        contents = np.zeros(6)
        meta = np.array([4, 0.0, 1.0], dtype=np.float64)
        np.savez(
            path,
            __hist_keys=np.array(["a", "b"], dtype="U"),
            __hist_names=np.array(["same", "same"], dtype="U"),
            a__contents=contents,
            a__meta=meta,
            b__contents=contents,
            b__meta=meta,
        )
        with pytest.raises(ValueError, match="duplicate display name"):
            load_histograms(str(path))

    def test_rejects_missing_mapped_dataset(self, tmp_path):
        path = tmp_path / "missing.npz"
        np.savez(
            path,
            __hist_keys=np.array(["a"], dtype="U"),
            __hist_names=np.array(["A"], dtype="U"),
            a__meta=np.array([4, 0.0, 1.0], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="missing dataset"):
            load_histograms(str(path))

    def test_rejects_malformed_meta_shape(self, tmp_path):
        path = tmp_path / "badmeta.npz"
        np.savez(
            path,
            __hist_keys=np.array(["a"], dtype="U"),
            __hist_names=np.array(["A"], dtype="U"),
            a__contents=np.zeros(6),
            a__meta=np.array([4, 0.0], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="meta has shape"):
            load_histograms(str(path))
