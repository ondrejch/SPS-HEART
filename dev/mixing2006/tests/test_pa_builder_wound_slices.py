"""p+A builder centrality-slice pins (card TASK-20260911-centrality-sets-1-5, M13 P1).

Pins the generation wiring of ``na49_mixing.event_builder_pA`` (the
optional ``PAExcitationConfig.target_wound_range`` field and the routed
draw in ``PAEventBuilder.generate``), card requirement 8:

- (i) the minimum-bias stream is unchanged: at fixed seeds with no
  ``target_wound_range`` the builder reproduces the pre-existing
  ``sample_event`` stream; the (b, K) draw-list literals below were
  captured from the builder before the routed draw existed (the
  foundations section 5 "min-bias (b, K) stream frozen against
  pre-conditioning literals" pin extended to the builder path);
- (ii) the sliced stream matches the pinned
  ``WoundedNucleonSampler.sample_event_in_range`` behavior exactly (the
  stubbed-attempt builder draw list equals the standalone conditioned
  stream at the same seed; ``sample_event_in_range`` consumes exactly two
  uniforms per draw, pinned in ``test_glauber``);
- (iii) the rare-slice (Pb208 K = 21, expected min-bias fraction 5.6e-7)
  and the point-target (A = 1, K = 1) events generate through the builder
  path (the foundations section 5 pins, exercised on the builder);
- (iv) equivalence: the builder's sliced stream and the chunked-worker's
  ``GenerationConfig.target_wound_range`` stream, at the same nucleus
  fields and fixed seeds, agree on the slice-mean wound-count statistics
  and with the closed-form slice mean (the M12 P3 "equivalent generation
  entry" requirement);
- the ``PAExcitationConfig.target_wound_range`` construction rules mirror
  the ``GenerationConfig`` validation (integer 1 <= lo <= hi, lo <= A,
  point target includes K = 1, non-zero restricted probability at builder
  construction), and the min-bias default (``None``) is unchanged.

Hermetic and fast: the event samples are test-scale (tens of events), the
equivalence sample 100 events per path; no Pb208 production statistics.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
from na49_mixing.exchange_model import ReggeExcitationParameters
from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

DR_FM = 0.05

# Light test-scale Regge parameters (the pinned audit configuration; the
# wound-count statistics below do not depend on the decay parameters).
REGGE = ReggeExcitationParameters(
    alpha_p0=1.0808,
    alpha_p_prime=0.25,
    alpha_exc0=0.5,
    transfer_slope=4.5,
    edge_power=2.0,
    s0=1.0,
    grid_size=96,
    effective_transfer_slope=7.3489827740435505,
)


def _c12_spec() -> NucleusSpec:
    return NucleusSpec(
        name="C12", A=12, Z=6, profile="woods_saxon", radius_fm=2.2,
        diffuseness_fm=0.546, quad_alpha=0.0, sigma_nn_inel_mb=31.7,
    )


def _pb208_spec() -> NucleusSpec:
    return NucleusSpec(
        name="Pb208", A=208, Z=82, profile="woods_saxon", radius_fm=6.62,
        diffuseness_fm=0.546, quad_alpha=0.0, sigma_nn_inel_mb=31.7,
    )


def _config(nucleus: NucleusSpec, wound_range=None) -> PAExcitationConfig:
    return PAExcitationConfig(
        production_mode="regge",
        regge=REGGE,
        continuum=ContinuumParameters(),
        nucleus=nucleus,
        glauber_dr_fm=DR_FM,
        max_event_attempts=100,
        target_wound_range=wound_range,
    )


def _delta_gen_file() -> str:
    from pathlib import Path

    base = (Path(__file__).resolve().parents[1] / "na49_mixing" /
            "generators" / "michal" / "deltas-to-pion")
    return str(base / "delta++.gen")


# ---------------------------------------------------------------------------
# (i) The min-bias (b, K) stream frozen against pre-conditioning literals
# ---------------------------------------------------------------------------

# Captured from PAEventBuilder.generate (glauber_draws) before the routed
# draw existed: no target_wound_range, fixed seeds, no failed attempts, so
# the draw list equals the accepted sequence.
MINBIAS_DRAW_STREAM = {
    ("C12", 20260911): [
        (2.6, 1), (1.25, 1), (3.1, 1), (1.85, 1), (2.4, 1), (3.35, 1),
        (3.25, 1), (1.3, 2), (2.2, 1), (0.7, 3), (3.95, 1), (3.15, 3),
    ],
    ("C12", 7): [
        (2.5, 2), (1.2, 2), (2.6, 1), (1.35, 3), (2.9, 1), (1.9, 3),
        (1.7, 1), (2.6, 2), (2.15, 1), (1.55, 4), (2.7, 2), (2.7, 1),
    ],
    ("Pb208", 20260911): [
        (6.0, 2), (1.8, 3), (6.6, 3), (2.75, 11), (7.7, 1), (3.05, 6),
        (6.25, 1), (5.4, 3), (9.05, 1), (6.05, 2), (4.95, 4), (6.0, 5),
    ],
    ("Pb208", 7): [
        (5.85, 5), (2.7, 3), (4.1, 6), (5.3, 2), (6.3, 5), (5.8, 3),
        (2.4, 8), (5.25, 5), (6.8, 1), (6.05, 2), (3.4, 6), (1.85, 5),
    ],
}


@pytest.mark.parametrize("label,seed", [
    ("C12", 20260911), ("C12", 7), ("Pb208", 20260911), ("Pb208", 7),
])
def test_builder_minbias_stream_unchanged(label, seed):
    """The builder min-bias (b, K) stream is frozen against the
    pre-change literals (the routed draw leaves the min-bias path
    byte-identical)."""
    spec = _c12_spec() if label == "C12" else _pb208_spec()
    builder = PAEventBuilder(_config(spec), np.random.default_rng(seed))
    pairs = MINBIAS_DRAW_STREAM[(label, seed)]
    for event_id in range(len(pairs)):
        event = builder.generate(event_id)
        event.validate()
    draws = [(round(float(b), 10), int(w)) for b, w, _ok in builder.glauber_draws]
    assert draws == [(float(b), int(w)) for b, w in pairs]
    # Every draw lands exactly on the Glauber b grid.
    for b, _w, _ok in builder.glauber_draws:
        assert abs(b / DR_FM - round(b / DR_FM)) < 1e-9


def _stub_attempt(event_id, impact_parameter, wounds):
    """A stub _attempt for stream pins: generate() then only consumes the
    Glauber draw and records the bookkeeping."""
    return types.SimpleNamespace(metadata={})


def test_builder_minbias_draw_equals_sample_event_stream():
    """With no target_wound_range the builder draws sample_event exactly:
    the stubbed-attempt draw list equals the standalone min-bias stream at
    the same seed."""
    spec = _pb208_spec()
    seed = 4242
    builder = PAEventBuilder(_config(spec), np.random.default_rng(seed))
    builder._attempt = _stub_attempt  # type: ignore[assignment]
    n = 30
    for event_id in range(n):
        builder.generate(event_id)
    draws = [(float(b), int(w)) for b, w, _ok in builder.glauber_draws]
    reference = WoundedNucleonSampler(
        spec, np.random.default_rng(seed), dr_fm=0.05)
    expected = [reference.sample_event() for _ in range(n)]
    assert draws == expected


# ---------------------------------------------------------------------------
# (ii) The sliced stream matches sample_event_in_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wound_range,seed", [((4, 7), 20260911), ((2, 2), 99)])
def test_builder_sliced_stream_matches_sample_event_in_range(wound_range, seed):
    """With a target_wound_range, the builder's (b, K) draws are exactly
    the pinned sample_event_in_range behavior at the same seed (two
    uniforms per draw, no acceptance loop)."""
    spec = _pb208_spec()
    builder = PAEventBuilder(_config(spec, wound_range),
                             np.random.default_rng(seed))
    builder._attempt = _stub_attempt  # type: ignore[assignment]
    n = 25
    for event_id in range(n):
        event = builder.generate(event_id)
        # The stub carries no target_wounds metadata; the draw list is the
        # pin surface.
        assert wound_range[0] <= builder.glauber_draws[-1][1] <= wound_range[1]
    draws = [(float(b), int(w)) for b, w, _ok in builder.glauber_draws]
    reference = WoundedNucleonSampler(spec, np.random.default_rng(seed),
                                      dr_fm=DR_FM)
    expected = [reference.sample_event_in_range(*wound_range)
                for _ in range(n)]
    assert draws == expected
    assert len(builder.glauber_draws) == n  # no extra draws, no failures


def test_builder_sliced_events_lands_inside_the_slice_and_stays_consistent():
    """The unstubbed sliced stream: every event inside [lo, hi], the wound
    bookkeeping counters unchanged, conservation at 1e-10, and the
    generated slice-mean wound count consistent with the closed form."""
    spec = _pb208_spec()
    wound_range = (4, 7)
    builder = PAEventBuilder(_config(spec, wound_range),
                             np.random.default_rng(20260911))
    n = 120
    ks = []
    for event_id in range(n):
        event = builder.generate(event_id)
        event.validate()
        ks.append(int(event.metadata["target_wounds"]))
    assert min(ks) >= wound_range[0] and max(ks) <= wound_range[1]
    # Closed-form slice mean and variance of the mapped band [4, 7]
    # (fraction_for_wound_range arithmetic, independent of the builder).
    analytic = WoundedNucleonSampler(spec, np.random.default_rng(0),
                                     dr_fm=DR_FM)
    pk = {k: analytic.fraction_for_wound_range(k, k) for k in range(1, 15)}
    mass = sum(pk[k] for k in range(4, 8))
    mean_closed = sum(k * pk[k] for k in range(4, 8)) / mass
    var = sum(k * k * pk[k] for k in range(4, 8)) / mass - mean_closed**2
    assert abs(float(np.mean(ks)) - mean_closed) <= \
        4.0 * math.sqrt(var / n) + 1e-9
    # The counters and the draw bookkeeping are unchanged by the slice.
    assert builder.generation_accepted == n
    assert builder.generation_attempts == len(builder.glauber_draws)
    assert builder.generation_rejected == 0
    event = builder.generate(n)
    assert event.metadata["generation_attempt"] >= 1
    assert event.metadata["generation_last_failure_reason"] is None


# ---------------------------------------------------------------------------
# (iii) The rare-slice and point-target pins on the builder path
# ---------------------------------------------------------------------------


def test_builder_rare_slice_pb208_k21_generates_without_rejection():
    """Pb208 K = 21 (expected min-bias fraction 5.6e-7): one builder event
    lands exactly on the requested wound count with no attempt-cap issue."""
    builder = PAEventBuilder(_config(_pb208_spec(), (21, 21)),
                             np.random.default_rng(424242))
    event = builder.generate(0)
    event.validate()
    assert event.metadata["target_wounds"] == 21
    assert event.conservation_residuals()["relative_max"] < 1e-10
    assert event.incoming_quantum_numbers == _pb208_incoming_qn()


def _pb208_incoming_qn():
    from na49_mixing.quantum_numbers import QuantumNumbers

    return QuantumNumbers(209, 83, 0)


def test_builder_point_target_sliced_stream():
    """A = 1: the sliced builder stream has K = 1 everywhere and the
    point-target draw consumes no Glauber random numbers (the glauber
    sampler's own point-target pin, exercised through the builder)."""
    point_spec = NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=31.7)
    builder = PAEventBuilder(_config(point_spec, (1, 1)),
                             np.random.default_rng(5))
    rng_state_before = builder.rng.bit_generator.state
    event = builder.generate(0)
    event.validate()
    assert event.metadata["target_wounds"] == 1
    assert builder.glauber_draws[0][:2] == (0.0, 1)
    with pytest.raises(ValueError, match="K=1"):
        PAExcitationConfig(
            production_mode="regge", regge=REGGE,
            nucleus=NucleusSpec(name="p", A=1, Z=1),
            target_wound_range=(2, 3),
        ).validate()


# ---------------------------------------------------------------------------
# (iv) Builder/worker equivalence on the slice-mean wound-count statistics
# ---------------------------------------------------------------------------


def test_builder_sliced_stream_equivalent_to_chunked_worker_slice():
    """The builder's sliced stream and the chunked-worker's
    GenerationConfig.target_wound_range stream, at the same nucleus fields
    and fixed seeds, agree on the slice-mean wound count statistics (the
    M12 P3 'equivalent generation entry' requirement) and with the
    closed-form slice mean of the band."""
    spec = _pb208_spec()
    wound_range = (2, 4)
    n = 100
    builder = PAEventBuilder(_config(spec, wound_range),
                             np.random.default_rng(20771))
    ks_builder = [
        int(builder.generate(event_id).metadata["target_wounds"])
        for event_id in range(n)
    ]
    out = generate_tracks_chunked(
        GenerationConfig(
            gen_file=_delta_gen_file(),
            n_events=n,
            seed=20771,
            beam_momentum=158.0,
            max_decay_passes=1,
            fragmentation_mode="both",
            nucleus=_pb208_spec(),
            target_wound_range=wound_range,
        ),
        n_workers=1,
        n_chunks=2,
    )
    wounds = np.asarray(out["n_wounds"], dtype=np.int64)
    sides = np.asarray(out["source_side"], dtype=np.int64)
    per_event_k = wounds[sides == 1]
    assert per_event_k.size == n
    assert int(per_event_k.min()) >= wound_range[0]
    assert int(per_event_k.max()) <= wound_range[1]
    mean_builder = float(np.mean(ks_builder))
    mean_worker = float(np.mean(per_event_k))
    # Closed-form slice statistics of the band [2, 4].
    analytic = WoundedNucleonSampler(spec, np.random.default_rng(0),
                                     dr_fm=DR_FM)
    pk = {k: analytic.fraction_for_wound_range(k, k) for k in range(1, 15)}
    lo, hi = wound_range
    mass = sum(pk[k] for k in range(lo, hi + 1))
    mean_closed = sum(k * pk[k] for k in range(lo, hi + 1)) / mass
    var = sum(k * k * pk[k] for k in range(lo, hi + 1)) / mass - mean_closed**2
    se = math.sqrt(var / n)
    assert mean_builder == pytest.approx(mean_closed, abs=4.0 * se)
    assert mean_worker == pytest.approx(mean_closed, abs=4.0 * se)
    assert mean_builder == pytest.approx(mean_worker, abs=4.0 * se * math.sqrt(2))


# ---------------------------------------------------------------------------
# The PAExcitationConfig slice-construction rules
# ---------------------------------------------------------------------------


def test_config_slice_validation_refuses_bad_ranges():
    spec = _pb208_spec()
    # The dataclass does not validate at construction; validate() carries
    # the rules (the builder calls it in __init__).
    with pytest.raises(ValueError, match="1 <= lo <= hi"):
        _config(spec, (0, 3)).validate()
    with pytest.raises(ValueError, match="1 <= lo <= hi"):
        _config(spec, (3, 2)).validate()
    with pytest.raises(ValueError, match="must be an integer"):
        _config(spec, (1.5, 3)).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _config(spec, (True, 3)).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lo > A"):
        _config(spec, (209, 210)).validate()
    with pytest.raises(ValueError, match=r"\(lo, hi\) pair"):
        PAExcitationConfig(regge=REGGE, nucleus=spec,
                           target_wound_range=(5,)).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\(lo, hi\) pair"):
        PAExcitationConfig(regge=REGGE, nucleus=spec,
                           target_wound_range="4-7").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="K=1"):
        _config(NucleusSpec(name="p", A=1, Z=1), (2, 3)).validate()


def test_config_normalizes_integer_valued_floats_and_lists():
    cfg = PAExcitationConfig(regge=REGGE, nucleus=_pb208_spec(),
                             target_wound_range=(2.0, 4.0))  # type: ignore[arg-type]
    cfg.validate()
    assert cfg.target_wound_range == (2, 4)
    cfg_list = PAExcitationConfig(regge=REGGE, nucleus=_pb208_spec(),
                                  target_wound_range=[3, 5])  # type: ignore[arg-type]
    cfg_list.validate()
    normalized = cfg_list.target_wound_range
    assert normalized is not None and normalized == (3, 5)
    assert all(type(value) is int for value in normalized)


def test_builder_refuses_zero_probability_slice_but_takes_pileup_range():
    # A slice inside the truncated support whose restricted probability
    # underflows to zero is refused at builder construction (the
    # MIN_WOUND_RANGE_FRACTION = 0.0 semantics): for Pb208, K = 171 is the
    # first bin whose pmf underflows double precision, so (171, 10^6)
    # passes the config rules (lo <= A) and the builder refuses.
    spec = _pb208_spec()
    with pytest.raises(ValueError, match="nothing to draw"):
        PAEventBuilder(_config(spec, (171, 10**6)), np.random.default_rng(1))
    # A range reaching beyond A carries the tail pile-up and generates
    # directly: on the compact A = 8 nucleus the range (6, 10^6) draws
    # (the sampler's truncated-tail pin) and constructs events.
    compact = NucleusSpec(name="A8compact", A=8, Z=4,
                          profile="harmonic_oscillator", radius_fm=1.0,
                          quad_alpha=0.0, sigma_nn_inel_mb=31.7)
    builder = PAEventBuilder(_config(compact, (6, 10**6)),
                             np.random.default_rng(3))
    event = builder.generate(0)
    event.validate()
    assert 6 <= event.metadata["target_wounds"] <= 8


def test_pa_acceptance_bookkeeping_unchanged_on_the_sliced_path():
    """The resampling loop and counters stay measurable on the sliced
    path: a forced failure redraws the slice-conditioned event and the
    bookkeeping records it exactly as on the min-bias path."""
    spec = _c12_spec()
    builder = PAEventBuilder(_config(spec, (1, 2)), np.random.default_rng(11))
    real_attempt = builder._attempt
    calls = {"n": 0}

    def flaky_attempt(event_id, impact_parameter, wounds):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic failure")
        return real_attempt(event_id, impact_parameter, wounds)

    builder._attempt = flaky_attempt
    event = builder.generate(0)
    event.validate()
    assert calls["n"] == 2
    assert builder.failure_reason_counts == {"ValueError": 1}
    assert [item[2] for item in builder.glauber_draws] == [False, True]
    assert event.metadata["generation_attempt"] == 2
    assert event.metadata["generation_last_failure_reason"] == \
        "ValueError: synthetic failure"
    assert builder.config.target_wound_range == (1, 2)
