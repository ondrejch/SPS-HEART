"""Tests for the p+A acceptance-audit statistics (card TASK-20260905-physics-review, M5).

Pins the committed pure-statistics surface of
``helpers.benchmarking.pa_acceptance_audit`` (outer tree 6083566):

- ``wilson_interval``: the vacuous ``n = 0`` interval, containment of
  the observed proportion, shrinkage as ``n`` grows, the two-sided
  symmetry at ``p_hat = 1/2``, and the closed-form value at two pinned
  ``(x, n)`` pairs.
- ``wound_comparison``: the bin-by-bin accepted-versus-Glauber P(nu)
  comparison with the pre-registered decision rule — a proportional
  accepted distribution passes, a constructed out-of-band bin fails
  with the right ``violating_bins`` while a sub-``min_accepted`` bin
  does not qualify, and unequal count-array lengths are refused.
- ``b_grid_index``: the Glauber b-grid arithmetic (draws land exactly
  on ``i*dr_fm``) and the negative / non-finite rejection.
- ``configuration_hash``: the stable sorted-key canonical SHA-256.
- ``summarize_acceptance``: the counter/draw-list cross-checks raise on
  inconsistent bookkeeping (no event generation involved: a stub
  builder and stub events exercise the guard).

Data-independent and unmarked; no NA49 tables are read.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers.benchmarking.pa_acceptance_audit as audit  # noqa: E402


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------


def test_wilson_interval_is_vacuous_for_zero_trials() -> None:
    """n = 0 (and n < 0) return the vacuous full interval."""
    assert audit.wilson_interval(0, 0) == (0.0, 1.0)
    assert audit.wilson_interval(3, -5) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("x", "n"),
    [(1, 10), (5, 10), (9, 10), (30, 100), (999, 1000)],
)
def test_wilson_interval_contains_the_observed_proportion(x: int, n: int) -> None:
    """The observed proportion p_hat = x/n lies strictly inside the band."""
    lo, hi = audit.wilson_interval(x, n)
    p_hat = x / n
    assert 0.0 < lo < p_hat < hi < 1.0
    assert math.isfinite(lo) and math.isfinite(hi)


def test_wilson_interval_endpoints_pin_the_unit_interval() -> None:
    """x = 0 and x = n: the mathematical endpoints (0 and 1) are pinned to
    within floating-point rounding of the clamped interval."""
    lo_zero, hi_zero = audit.wilson_interval(0, 100)
    assert lo_zero <= 1e-12  # mathematically 0, clamped at the lower edge
    assert hi_zero < 0.2  # a real 95% band, not the vacuous interval
    lo_one, hi_one = audit.wilson_interval(100, 100)
    assert lo_one > 0.9  # the band approaches 1 without collapsing
    assert hi_one >= 1.0 - 1e-12  # mathematically 1, clamped at the upper edge


def test_wilson_interval_shrinks_and_nests_as_n_grows() -> None:
    """At the same p_hat the interval narrows and nests for larger n."""
    lo_small, hi_small = audit.wilson_interval(30, 100)
    lo_large, hi_large = audit.wilson_interval(3000, 10000)
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_large < width_small
    # Same p_hat (0.3): the larger-n interval nests inside the smaller-n one.
    assert lo_small < lo_large
    assert hi_large < hi_small


def test_wilson_interval_is_two_sided_symmetric_at_half() -> None:
    """p_hat = 1/2: the band is symmetric about 0.5 to working precision."""
    lo, hi = audit.wilson_interval(50, 100)
    center = 0.5 * (lo + hi)
    assert center == pytest.approx(0.5, abs=1e-12)
    assert hi - 0.5 == pytest.approx(0.5 - lo, rel=1e-9)


def test_wilson_interval_matches_the_documented_closed_form() -> None:
    """Pinned (x, n) values against hand-computed closed-form intervals.

    The literals were computed once from the documented Wilson score
    formulas (module docstring) at the default z and pin the committed
    arithmetic bit for bit.
    """
    assert audit.wilson_interval(1, 100) == (
        0.0017674320641406505,
        0.054486196178705315,
    )
    assert audit.wilson_interval(30, 100) == (
        0.2189488529493276,
        0.3958485463334666,
    )


def test_wilson_interval_independent_formula_cross_check() -> None:
    """An in-test reimplementation of the documented formulas agrees."""
    for x, n in [(7, 40), (131, 500), (1, 1), (100, 100)]:
        z = audit.WILSON_Z_95
        p_hat = x / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
        expected = (max(center - half, 0.0), min(center + half, 1.0))
        got = audit.wilson_interval(x, n)
        assert got[0] == pytest.approx(expected[0], abs=1e-12)
        assert got[1] == pytest.approx(expected[1], abs=1e-12)


# ---------------------------------------------------------------------------
# wound_comparison
# ---------------------------------------------------------------------------


def test_wound_comparison_proportional_acceptance_passes() -> None:
    """accepted = 0.5 * reference per bin: every bin inside, decision passes."""
    result = audit.wound_comparison([200, 150, 150], [400, 300, 300])
    assert result["n_reference_total"] == 1000
    assert result["n_accepted_total"] == 500
    assert result["min_accepted_for_qualification"] == 30
    assert result["decision_passed"] is True
    assert result["violating_bins"] == []
    assert result["n_qualifying_bins"] == 3
    assert "monotonic projectile-mass rule retained" in result["verdict"]
    assert "Wilson" in result["ci_convention"]
    assert result["decision_rule"]

    entries = result["entries"]
    assert [entry["nu"] for entry in entries] == [1, 2, 3]
    for entry in entries:
        # The accepted proportions equal the reference proportions
        # (proportional counts), which the Wilson band of the reference
        # proportion always contains.
        assert entry["p_acc"] == pytest.approx(entry["p_ref"], rel=1e-15)
        assert entry["inside"] is True
        assert entry["qualifies"] is True
        assert entry["acceptance"] == pytest.approx(0.5, rel=1e-15)
        # The band is the Wilson interval of the reference proportion.
        assert entry["band_low"] == pytest.approx(
            audit.wilson_interval(entry["reference_draws"], 1000)[0], abs=1e-15
        )
        assert entry["band_high"] == pytest.approx(
            audit.wilson_interval(entry["reference_draws"], 1000)[1], abs=1e-15
        )
        # The per-bin acceptance CI is a real interval around the ratio.
        assert 0.0 <= entry["acceptance_ci_low"] <= entry["acceptance"] <= entry[
            "acceptance_ci_high"
        ] <= 1.0


def test_wound_comparison_constructed_violation_pinpoints_the_bin() -> None:
    """An out-of-band qualifying bin fails; a sub-30 bin cannot qualify.

    Reference: 900 / 100 draws (p_ref = 0.9 / 0.1). Accepted: 450 / 5
    (p_acc = 0.989 / 0.011). Bin nu=1 sits 0.07 above the top of its
    band (band_top ~ 0.917) and qualifies (450 >= 30): it is the single
    violating bin. Bin nu=2 is also out of band (0.011 below a band
    whose bottom is ~0.083) but holds only 5 accepted events, so it does
    not qualify and does not violate the decision.
    """
    result = audit.wound_comparison([450, 5], [900, 100])
    assert result["n_reference_total"] == 1000
    assert result["n_accepted_total"] == 455
    assert result["decision_passed"] is False
    assert result["violating_bins"] == [1]
    assert result["n_qualifying_bins"] == 1
    assert "de-excitation" in result["verdict"]

    entry_1, entry_2 = result["entries"]
    assert entry_1["nu"] == 1 and entry_1["accepted"] == 450
    assert entry_1["p_acc"] == pytest.approx(450.0 / 455.0, rel=1e-15)
    assert entry_1["band_low"] > 0.87  # the band stays far below p_acc
    assert entry_1["inside"] is False
    assert entry_1["qualifies"] is True

    assert entry_2["nu"] == 2 and entry_2["accepted"] == 5
    assert entry_2["p_acc"] == pytest.approx(5.0 / 455.0, rel=1e-15)
    assert entry_2["band_high"] < 0.13  # the band stays far above p_acc
    assert entry_2["inside"] is False
    assert entry_2["qualifies"] is False  # 5 < 30 accepted


def test_wound_comparison_unequal_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="equal length"):
        audit.wound_comparison([1, 2], [1])


# ---------------------------------------------------------------------------
# b_grid_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b_fm", "dr_fm", "expected"),
    [
        (0.0, 0.05, 0),
        (0.05, 0.05, 1),
        (0.1, 0.05, 2),
        (1.0, 0.05, 20),
        (4.45, 0.05, 89),
    ],
)
def test_b_grid_index_lands_on_grid_multiples(b_fm, dr_fm, expected) -> None:
    """Draws land exactly on i*dr_fm: the index is the exact multiple."""
    assert audit.b_grid_index(b_fm, dr_fm) == expected
    for i in range(0, 41):
        assert audit.b_grid_index(i * dr_fm, dr_fm) == i
    for i in range(0, 21):
        assert audit.b_grid_index(i * 0.1, 0.05) == 2 * i


@pytest.mark.parametrize("bad_b", [-0.01, -1.0, float("nan"), float("inf"), float("-inf")])
def test_b_grid_index_rejects_negative_and_non_finite(bad_b) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        audit.b_grid_index(bad_b, 0.05)


# ---------------------------------------------------------------------------
# configuration_hash
# ---------------------------------------------------------------------------


def test_configuration_hash_is_stable_and_key_order_invariant() -> None:
    a = {"b": 2, "a": {"y": 1, "x": [1, 2]}}
    b = {"a": {"x": [1, 2], "y": 1}, "b": 2}
    h_a = audit.configuration_hash(a)
    h_b = audit.configuration_hash(b)
    assert h_a == h_b
    assert isinstance(h_a, str) and len(h_a) == 64
    assert all(c in "0123456789abcdef" for c in h_a)
    # A different value changes the digest.
    assert audit.configuration_hash({"a": {"x": [1, 2], "y": 1}, "b": 3}) != h_a


def test_configuration_hash_is_the_sorted_key_sha256() -> None:
    """The digest is the SHA-256 of the canonical sorted JSON form."""
    config = {
        "seed": 20260831,
        "pA": {"glauber_dr_fm": 0.05, "max_event_attempts": 100},
        "nucleus": {"name": "C12", "A": 12, "Z": 6},
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert audit.configuration_hash(config) == expected


# ---------------------------------------------------------------------------
# summarize_acceptance consistency guard (stub builder, no generation)
# ---------------------------------------------------------------------------


def _stub_event(wounds: int, b_fm: float, residual: float) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"target_wounds": wounds, "impact_parameter_fm": b_fm},
        conservation_residuals=lambda: {"relative_max": residual},
    )


def _stub_builder() -> SimpleNamespace:
    """A minimal stand-in for PAEventBuilder's bookkeeping surface.

    Three Glauber draws (two accepted), so every marginal is populated:
    nu = 1 twice (both accepted), nu = 2 once (rejected), b bins 0/1/2.
    """
    return SimpleNamespace(
        config=SimpleNamespace(glauber_dr_fm=0.05),
        generation_attempts=3,
        generation_accepted=2,
        generation_rejected=0,
        failure_reason_counts={"ValueError": 1},
        accepted_generation_attempts=[1, 3],
        glauber_draws=[(0.0, 1, True), (0.05, 2, False), (0.1, 1, True)],
    )


def test_summarize_acceptance_aggregates_stub_bookkeeping() -> None:
    builder = _stub_builder()
    events = [_stub_event(1, 0.0, 1e-15), _stub_event(1, 0.1, 2e-15)]
    payload = audit.summarize_acceptance(builder, events, min_shift_draws=1)
    assert payload["n_accepted"] == 2
    assert payload["n_glauber_draws"] == 3
    assert payload["n_rejected_events"] == 0
    assert payload["overall_acceptance"] == pytest.approx(2.0 / 3.0, rel=1e-15)
    assert payload["failure_reason_counts"] == {"ValueError": 1}
    assert payload["attempt_distribution"] == {"1": 1, "3": 1}
    assert payload["mean_generation_attempt"] == pytest.approx(2.0, rel=1e-15)
    assert payload["max_generation_attempt"] == 3
    assert payload["max_conservation_residual"] == pytest.approx(2e-15, rel=1e-15)
    # The wound decision runs on the stub marginals (nu=1 accepted 2/3,
    # nu=2 accepted 0/1) and carries the pinned decision fields.
    wound = payload["wound"]
    assert wound["decision_passed"] == (len(wound["violating_bins"]) == 0)
    assert wound["n_accepted_total"] == 2
    assert wound["n_reference_total"] == 3


def test_summarize_acceptance_rejects_inconsistent_bookkeeping() -> None:
    builder = _stub_builder()
    events = [_stub_event(1, 0.0, 1e-15), _stub_event(1, 0.1, 2e-15)]

    # A draw list shorter than generation_attempts is refused.
    short = copy.deepcopy(builder)
    short.glauber_draws = list(short.glauber_draws)[:-1]
    with pytest.raises(ValueError, match="draw list"):
        audit.summarize_acceptance(short, events)

    # A counter that disagrees with the accepted draw marks is refused.
    over = copy.deepcopy(builder)
    over.generation_accepted = 3
    with pytest.raises(ValueError, match="accepted-count mismatch"):
        audit.summarize_acceptance(over, events)

    # An empty sample has nothing to summarize.
    with pytest.raises(ValueError, match="at least one event"):
        audit.summarize_acceptance(builder, [])
