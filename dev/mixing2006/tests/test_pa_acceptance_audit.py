"""p+A generation acceptance audit bookkeeping tests (card TASK-20260905-physics-review, M5).

Pins the committed behavior of the nested tree at 9f1bf25:

- ``PAEventBuilder`` records every Glauber draw: per-event metadata
  ``generation_attempt`` (1-based; 1 when the first attempt succeeds)
  and ``generation_last_failure_reason`` (the last failed attempt's
  ``"<ExceptionClass>: <message>"`` when any attempt failed before
  acceptance, ``None`` otherwise); builder-level counters
  ``generation_attempts`` / ``generation_accepted`` /
  ``generation_rejected`` / ``failure_reason_counts`` (per exception
  class name) / ``accepted_generation_attempts`` (1-based attempt
  index per accepted event) / ``glauber_draws`` (one
  ``(b, wounds, accepted)`` tuple per draw in draw order);
  ``reset_generation_counters`` empties all of them.
- A failed kinematic construction redraws the whole Glauber event
  (the physics path is unchanged): a forced failure before acceptance
  advances the draw list and the attempt index, and an exhausted
  ``max_event_attempts`` raises ``RuntimeError`` and is counted in
  ``generation_rejected``.
- The hermetic small-N p+C12 audit (500 accepted events, fixed seed,
  Woods-Saxon C12, no Pb208, no long run) drives
  ``helpers.benchmarking.pa_acceptance_audit.summarize_acceptance``
  end to end: counter/draw-list consistency, the expected payload
  keys, the conservation residual of every accepted event below
  1e-10, the total event baryon number A+1 = 13 and charge
  Z+1 = 7, and the pre-registered P(nu) decision fields are
  internally consistent (``decision_passed == (not violating_bins)``).
- Determinism: rerunning the same-seed small-N generation reproduces
  the audit payload bit for bit.

Hermetic and fast: no Pb208, no long run.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from helpers.benchmarking import pa_acceptance_audit
from na49_mixing.quantum_numbers import QuantumNumbers

SEED = 20260905
N_ACCEPTED = 500
DR_FM = 0.05
# Total event quantum numbers of p + C12: baryon number A+1, charge Z+1.
P_C12_QN = QuantumNumbers(13, 7, 0)
FAILURE_REASON_PATTERN = re.compile(r"^(ValueError|RuntimeError|FloatingPointError): ")


def _small_c12_config() -> dict:
    """A small p+C12 configuration for the hermetic audit sample.

    Woods-Saxon C12 (A = 12, Z = 6, small radius) at the committed
    bundled Regge parameter set; the audit measures generation
    acceptance, not production shape, so only the nucleus differs
    from the bundled p+Pb208 configuration.
    """
    return {
        "beam_momentum": 158.0,
        "regge": {
            "alpha_p0": 1.0808,
            "alpha_p_prime": 0.25,
            "alpha_exc0": 0.5,
            "transfer_slope": 4.5,
            "edge_power": 2.0,
            "s0": 1.0,
            "grid_size": 96,
            "effective_transfer_slope": 7.3489827740435505,
        },
        "nucleus": {
            "name": "C12",
            "A": 12,
            "Z": 6,
            "profile": "woods_saxon",
            "radius_fm": 2.2,
            "diffuseness_fm": 0.546,
            "quad_alpha": 0.0,
            "sigma_nn_inel_mb": 31.7,
        },
        "pA": {"glauber_dr_fm": DR_FM, "max_event_attempts": 100},
    }


@pytest.fixture(scope="module")
def small_audit() -> tuple:
    """Generate the small-N p+C12 sample once per module and summarize it.

    Returns ``(config, builder, events, payload)``; the payload is
    deterministic for this seed, so the fixture doubles as the
    determinism reference.
    """
    config = _small_c12_config()
    builder = pa_acceptance_audit.build_pa_builder(config, seed=SEED)
    events = pa_acceptance_audit.run_acceptance_sample(builder, N_ACCEPTED)
    payload = pa_acceptance_audit.summarize_acceptance(builder, events)
    return config, builder, events, payload


# ---------------------------------------------------------------------------
# Per-event metadata and conservation of the small-N sample
# ---------------------------------------------------------------------------


def test_small_c12_sample_is_closed_and_carries_attempt_metadata(small_audit):
    """Every accepted event is closed, carries the attempt bookkeeping, and
    the p+C12 total event baryon number is A+1 and charge Z+1."""
    config, builder, events, payload = small_audit
    assert len(events) == N_ACCEPTED
    for event in events:
        attempt = event.metadata["generation_attempt"]
        assert isinstance(attempt, int) and attempt >= 1
        reason = event.metadata["generation_last_failure_reason"]
        assert reason is None or (
            isinstance(reason, str)
            and FAILURE_REASON_PATTERN.match(reason) is not None
        )
        # Per-event conservation at the 1e-10 tolerance (card M5,
        # acceptance criterion 4).
        assert event.conservation_residuals()["relative_max"] < 1e-10
        event.validate()  # default relative tolerance 1e-10
        assert event.incoming_quantum_numbers == P_C12_QN
        assert event.outgoing_quantum_numbers() == P_C12_QN
    assert payload["max_conservation_residual"] < 1e-10


# ---------------------------------------------------------------------------
# Counter / draw-list consistency
# ---------------------------------------------------------------------------


def test_counter_draw_list_consistency(small_audit):
    """The builder counters and the per-draw list agree with each other
    and with the accepted events."""
    _config, builder, events, _payload = small_audit

    draws = builder.glauber_draws
    # One entry per draw, in draw order.
    assert builder.generation_attempts == len(draws)
    accepted_marks = [item[2] for item in draws]
    assert sum(accepted_marks) == builder.generation_accepted == len(events)
    # Every draw either produced an accepted event or was recorded as a
    # failed attempt under its exception class name.
    assert (
        builder.generation_attempts
        == builder.generation_accepted
        + sum(builder.failure_reason_counts.values())
    )
    # The per-event attempt indices are stored per accepted event, in
    # acceptance order: each event's generation_attempt equals its entry,
    # and each generate() call counts attempts from 1, so a first-attempt
    # success records 1.
    assert len(builder.accepted_generation_attempts) == len(events)
    accepted_positions = [
        index for index, accepted in enumerate(accepted_marks) if accepted
    ]
    # Each accepted event carries the (b, wounds) of its own draw: the
    # k-th accepted event pairs with the k-th accepted draw in draw order,
    # and its attempt count is the k-th per-event attempt index.
    for k, event in enumerate(events):
        b_fm, wounds, _accepted = draws[accepted_positions[k]]
        assert event.metadata["impact_parameter_fm"] == b_fm
        assert event.metadata["target_wounds"] == wounds
        assert (
            event.metadata["generation_attempt"]
            == builder.accepted_generation_attempts[k]
        )
    # Draws land exactly on the Glauber b grid (b = i*dr_fm).
    for b_fm, wounds, _accepted in draws:
        assert isinstance(wounds, int) and wounds >= 1
        assert abs(b_fm / DR_FM - round(b_fm / DR_FM)) < 1e-9
        assert b_fm >= 0.0


# ---------------------------------------------------------------------------
# summarize_acceptance payload on the real small-N audit
# ---------------------------------------------------------------------------


def test_summarize_payload_keys_and_wound_decision(small_audit):
    """The audit payload carries the expected keys and the pre-registered
    P(nu) decision fields are internally consistent."""
    _config, builder, _events, payload = small_audit

    expected_keys = {
        "n_accepted",
        "n_glauber_draws",
        "n_rejected_events",
        "overall_acceptance",
        "failure_reason_counts",
        "attempt_distribution",
        "mean_generation_attempt",
        "max_generation_attempt",
        "wound",
        "b_table_native",
        "b_table_coarse",
        "nb_joint_table",
        "nb_failed_cells",
        "centrality_shift",
        "max_conservation_residual",
    }
    assert expected_keys <= set(payload)
    assert payload["n_accepted"] == N_ACCEPTED
    assert payload["n_glauber_draws"] == builder.generation_attempts
    # The per-event attempt distribution tallies the accepted events.
    assert sum(payload["attempt_distribution"].values()) == len(
        builder.accepted_generation_attempts
    )
    assert payload["max_generation_attempt"] >= 1
    assert (
        payload["max_generation_attempt"]
        == max(builder.accepted_generation_attempts)
    )

    wound = payload["wound"]
    # The pre-registered decision: passes exactly when no qualifying bin
    # violates its band.
    assert wound["decision_passed"] == (len(wound["violating_bins"]) == 0)
    assert wound["min_accepted_for_qualification"] == 30
    assert wound["n_qualifying_bins"] == sum(
        entry["qualifies"] for entry in wound["entries"]
    )
    assert wound["n_qualifying_bins"] >= 1  # 500 events populate the low-nu bins
    assert sum(entry["accepted"] for entry in wound["entries"]) == wound[
        "n_accepted_total"
    ] == N_ACCEPTED
    assert (
        sum(entry["reference_draws"] for entry in wound["entries"])
        == wound["n_reference_total"]
        == builder.generation_attempts
    )
    for entry in wound["entries"]:
        # The decision fields are mutually consistent on every bin.
        assert entry["inside"] == (
            entry["band_low"] <= entry["p_acc"] <= entry["band_high"]
        )
        assert entry["qualifies"] == (
            entry["accepted"] >= wound["min_accepted_for_qualification"]
        )
        assert entry["nu"] >= 1
    assert "Wilson" in wound["ci_convention"]
    assert wound["decision_rule"]

    shift = payload["centrality_shift"]
    assert 0.0 <= shift["kolmogorov_distance_b"] <= 1.0
    assert shift["mean_nu_accepted"] > 0.0
    assert shift["mean_nu_all_draws"] > 0.0
    assert shift["mean_b_accepted_fm"] > 0.0
    assert shift["mean_b_all_draws_fm"] > 0.0


# ---------------------------------------------------------------------------
# Forced-failure bookkeeping (deterministic, no long run)
# ---------------------------------------------------------------------------


def test_forced_failures_record_reasons_and_attempt_index():
    """A failed attempt redraws the whole Glauber event and is counted
    under its exception class name; the accepted event carries the 1-based
    attempt index and the last failure reason."""
    builder = pa_acceptance_audit.build_pa_builder(_small_c12_config(), seed=SEED)
    real_attempt = builder._attempt
    calls = {"n": 0}

    def flaky_attempt(event_id, impact_parameter, wounds):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError(f"synthetic failure {calls['n']}")
        return real_attempt(event_id, impact_parameter, wounds)

    builder._attempt = flaky_attempt  # instance attribute shadows the method
    event = builder.generate(0)

    assert calls["n"] == 3
    assert event.metadata["generation_attempt"] == 3
    assert event.metadata["generation_last_failure_reason"] == (
        "ValueError: synthetic failure 2"
    )
    assert builder.failure_reason_counts == {"ValueError": 2}
    assert builder.generation_attempts == 3
    assert builder.generation_accepted == 1
    assert builder.generation_rejected == 0
    assert builder.accepted_generation_attempts == [3]
    draws = builder.glauber_draws
    assert [item[2] for item in draws] == [False, False, True]
    # The accepted event carries the geometry of its own (third) draw.
    b_fm, wounds, _accepted = draws[2]
    assert event.metadata["impact_parameter_fm"] == b_fm
    assert event.metadata["target_wounds"] == wounds
    event.validate()


def test_exhausted_attempts_raise_and_count_rejection():
    """Exhausting max_event_attempts raises RuntimeError and increments
    generation_rejected; every draw stays unaccepted."""
    config = _small_c12_config()
    config["pA"] = {"glauber_dr_fm": DR_FM, "max_event_attempts": 3}
    builder = pa_acceptance_audit.build_pa_builder(config, seed=SEED)

    def always_fails(event_id, impact_parameter, wounds):
        raise RuntimeError("always fails")

    builder._attempt = always_fails
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        builder.generate(0)
    assert builder.generation_rejected == 1
    assert builder.generation_attempts == 3
    assert builder.generation_accepted == 0
    assert builder.accepted_generation_attempts == []
    assert builder.failure_reason_counts == {"RuntimeError": 3}
    assert [item[2] for item in builder.glauber_draws] == [False, False, False]


def test_reset_generation_counters_starts_from_empty():
    """reset_generation_counters empties the bookkeeping; counters restart
    from one after a fresh generate."""
    builder = pa_acceptance_audit.build_pa_builder(_small_c12_config(), seed=SEED)
    builder.generate(0)
    assert builder.generation_attempts >= 1
    builder.reset_generation_counters()
    assert builder.generation_attempts == 0
    assert builder.generation_accepted == 0
    assert builder.generation_rejected == 0
    assert builder.failure_reason_counts == {}
    assert builder.accepted_generation_attempts == []
    assert builder.glauber_draws == []
    builder.generate(1)
    assert builder.generation_attempts >= 1
    assert builder.generation_accepted == 1
    assert len(builder.glauber_draws) == builder.generation_attempts


# ---------------------------------------------------------------------------
# Determinism and the consistency guard
# ---------------------------------------------------------------------------


def test_small_c12_rerun_reproduces_payload_bitwise(small_audit):
    """The same seed and configuration reproduce the audit payload.

    ``summarize_acceptance`` is a pure function of the generated sample
    (no randomness is consumed), and the sample is deterministic in the
    seed, so the rerun payload is bit-equal to the fixture's.
    """
    config, _builder, _events, reference = small_audit
    rerun_builder = pa_acceptance_audit.build_pa_builder(config, seed=SEED)
    rerun_events = pa_acceptance_audit.run_acceptance_sample(rerun_builder, N_ACCEPTED)
    rerun_payload = pa_acceptance_audit.summarize_acceptance(rerun_builder, rerun_events)
    assert rerun_payload == reference


def test_summarize_acceptance_rejects_inconsistent_bookkeeping():
    """Counter/draw-list mismatches raise instead of producing a report."""
    builder = pa_acceptance_audit.build_pa_builder(_small_c12_config(), seed=SEED)
    events = pa_acceptance_audit.run_acceptance_sample(builder, 5)

    original_draws = list(builder.glauber_draws)
    builder.glauber_draws = original_draws[:-1]
    with pytest.raises(ValueError, match="draw list"):
        pa_acceptance_audit.summarize_acceptance(builder, events)
    builder.glauber_draws = original_draws  # restore the tampered list

    over = builder
    over.generation_accepted = builder.generation_accepted + 1
    with pytest.raises(ValueError, match="accepted-count mismatch"):
        pa_acceptance_audit.summarize_acceptance(over, events)

    with pytest.raises(ValueError, match="at least one event"):
        pa_acceptance_audit.summarize_acceptance(builder, [])
