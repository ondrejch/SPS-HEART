"""Line-shape moment pins for the committed resonance catalog (2026-09-28).

Three defects in the Round 2 kernels were found by comparing drawn mass
distributions with their stated laws (review record
``.collab/reviews/fable-speedup-2026-09-28.md``, Section 8):

- FAB-09: the spectral function squared the width in the numerator, the
  formation cross-section shape, which put the Delta(1232) mean at
  1.62 GeV with 81% of its mass above pole + Gamma;
- FAB-08: the L = 2 barrier factor grew with momentum;
- FAB-07: the line-shape support started at the highest channel
  threshold, leaving the Delta(1600), Delta(1700), and K1(1270) poles
  outside their own support.

These tests pin the corrected forms by their consequences, so a change
of form is caught at landing: barrier monotonicity, support at the
lowest threshold with the pole inside, and the drawn-mass moments of
the broad states against the analytic one-power shape and a
constant-width plausibility band.  Hermetic: the committed catalog and
the standard library plus NumPy only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: numpy 2 renamed trapz to trapezoid (the code reviews of 2026-11-04)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_widths import (
    BLATT_WEISSKOPF_BARRIER,
    _unnormalized_shape,
    draw_mass,
    spectral_function,
    total_width,
)
from na49_mixing.spectral_density import _daughter_masses

CATALOG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "v2_singlet"
    / "resonance_catalog.json"
)

#: (pdg, mean-mass band [GeV], maximum fraction above pole + Gamma) for
#: the broad states whose shapes the squared-width form distorted most.
#: The bands hold the analytic one-power shape on the committed support
#: (Delta 1.388, N*(1440) 1.847, rho 0.877, K* 0.938, f2 1.462) with room
#: for the barrier and support corrections, and exclude the squared-width
#: values (1.624, 2.319, 0.962, 1.001, 1.779).
BROAD_STATE_BANDS = {
    2214: (1.30, 1.45, 0.55),    # Delta(1232)+
    12212: (1.70, 1.95, 0.60),   # N*(1440)+
    113: (0.83, 0.92, 0.40),     # rho(770)0
    323: (0.91, 0.97, 0.40),     # K*(892)+
    225: (1.35, 1.55, 0.55),     # f2(1270)
}


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(str(CATALOG_PATH))


def _grid_moments(record, masses):
    lo, hi = float(record.mass_range[0]), float(record.mass_range[1])
    xs = np.linspace(lo + 1e-9, hi, 20001)
    ys = np.array([_unnormalized_shape(float(x), record, masses,
                                       BLATT_WEISSKOPF_BARRIER) for x in xs])
    ys = ys / _trapezoid(ys, xs)
    cut = record.pole_mass + record.nominal_width
    return float(_trapezoid(xs * ys, xs)), float(_trapezoid(ys[xs > cut], xs[xs > cut]))


def test_barrier_factors_never_grow_with_momentum():
    """Every supported L falls monotonically in q and starts at 1
    (FAB-08 pinned the L = 2 member, which grew from 1 to 3)."""
    qs = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0)
    for L in range(0, 5):
        values = [BLATT_WEISSKOPF_BARRIER(L, q) for q in qs]
        assert values[0] == 1.0
        assert all(a >= b for a, b in zip(values, values[1:])), (L, values)
    # the D-wave member at z = 1 is the standard 9 / (1 + 3 + 9)
    assert abs(BLATT_WEISSKOPF_BARRIER(2, 1.0) - math.sqrt(9.0 / 13.0)) <= 1e-12


def test_support_starts_at_the_lowest_threshold_and_holds_the_pole(catalog):
    """Every decaying record's support begins at its lowest channel
    threshold, the pole mass lies inside the support, and the shape is
    positive at the pole (FAB-07 pinned Delta(1600), Delta(1700), and
    K1(1270), whose poles sat below their support)."""
    for record in catalog.records():
        if record.stable_for_decay:
            continue
        masses = _daughter_masses(catalog, record)
        thresholds = [catalog.threshold(record.pdg_id, ch) for ch in record.decay_channels]
        assert record.mass_range[0] == pytest.approx(min(thresholds), abs=1e-12), record.name
        assert record.mass_range[0] < record.pole_mass < record.mass_range[1], record.name
        assert spectral_function(float(record.pole_mass), record, masses) > 0.0, record.name


def test_one_power_numerator_and_pole_normalization(catalog):
    """The shape at the pole is M0**2 / (M0**2 Gamma0) = 1 / Gamma0 up to
    the normalization constant, so the ratio of the unnormalized shape at
    the pole to the value with the width doubled is 2 (one power of the
    width in the numerator; the squared form gives 4)."""
    record = catalog.get(2214)
    masses = _daughter_masses(catalog, record)
    M0 = float(record.pole_mass)
    gamma0 = total_width(M0, record, masses, BLATT_WEISSKOPF_BARRIER)
    assert abs(gamma0 - record.nominal_width) <= 1e-9 * record.nominal_width
    value = _unnormalized_shape(M0, record, masses, BLATT_WEISSKOPF_BARRIER)
    assert abs(value * gamma0 - 1.0) <= 1e-9


@pytest.mark.parametrize("pdg", sorted(BROAD_STATE_BANDS))
def test_broad_state_moments_within_band(catalog, pdg):
    """The mean mass and the fraction above pole + Gamma of the analytic
    shape lie in the physical band (FAB-09)."""
    record = catalog.get(pdg)
    masses = _daughter_masses(catalog, record)
    mean, tail = _grid_moments(record, masses)
    lo, hi, tail_max = BROAD_STATE_BANDS[pdg]
    assert lo <= mean <= hi, (record.name, mean)
    assert tail <= tail_max, (record.name, tail)


def test_drawn_masses_follow_the_analytic_shape(catalog):
    """20,000 ``draw_mass`` draws of the Delta(1232) reproduce the analytic
    mean within 3 standard errors and the tail fraction within 0.01."""
    record = catalog.get(2214)
    masses = _daughter_masses(catalog, record)
    rng = np.random.default_rng(20260928)
    lo, hi = record.mass_range
    drawn = np.array([draw_mass((lo, hi), record, masses, rng) for _ in range(20000)])
    mean, tail = _grid_moments(record, masses)
    se = drawn.std(ddof=1) / math.sqrt(drawn.size)
    assert abs(drawn.mean() - mean) <= 3.0 * se + 1e-3, (drawn.mean(), mean, se)
    cut = record.pole_mass + record.nominal_width
    assert abs((drawn > cut).mean() - tail) <= 0.01
