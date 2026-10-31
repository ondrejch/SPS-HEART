"""Tests for the terminal-unstable residual gate in component builds."""

from __future__ import annotations

from helpers.benchmarking import benchmark_sps_species as bss


def test_gate_passes_at_zero_residuals() -> None:
    assert bss._terminal_unstable_gate(0, 120_000) is None


def test_gate_passes_at_exactly_allowed() -> None:
    n_gen = 120_000
    allowed = int(bss.TERMINAL_UNSTABLE_TOLERANCE * n_gen)
    assert allowed == 24
    assert bss._terminal_unstable_gate(allowed, n_gen) is None


def test_gate_fails_one_over_allowed() -> None:
    n_gen = 120_000
    allowed = int(bss.TERMINAL_UNSTABLE_TOLERANCE * n_gen)
    reason = bss._terminal_unstable_gate(allowed + 1, n_gen)
    assert reason is not None
    assert reason.startswith("terminal_unstable_tracks:")
    assert f":{allowed + 1}" in reason
    assert f"> {allowed}" in reason


def test_gate_fails_with_zero_generated_primaries() -> None:
    reason = bss._terminal_unstable_gate(1, 0)
    assert reason is not None
    assert reason.startswith("terminal_unstable_tracks:")


def test_gate_passes_measured_production_case() -> None:
    # N*(1440) p-species component, 2026-09-02, seed 20260831:
    # 8 terminal-unstable tracks out of 120k generated primaries.
    assert bss._terminal_unstable_gate(8, 120_000) is None
