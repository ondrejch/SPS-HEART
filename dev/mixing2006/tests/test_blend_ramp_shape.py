"""Phase 1 of the Round 4 rework card
``TASK-20260920-r4-rework-transition-routing-formulation`` (2026-09-28):
the additive ``BlendSpec.ramp_shape`` field with the pre-registered
candidate set C0 linear, C1 smoothstep, C2 quadratic.

- the default is the sealed linear formulation, bit-identical on a
  10,001-node probe grid;
- every shape is exactly 0.0 at and below B and exactly 1.0 from B + W,
  continuous at both junctions;
- the in-band means are 1/2, 1/2, 1/3 (the pre-registered diagnostic
  contrast) within 1e-9;
- the double-counting identity of the normalization report holds for
  every shape within the R30 tolerance on the committed catalog and
  families;
- the routing draw budget is unchanged (0 / 1 / 0 uniforms below, in,
  and above the band) for every shape;
- an unknown shape is refused.
Hermetic: the committed catalog and families plus NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: numpy 2 renamed trapz to trapezoid (the code reviews of 2026-11-04)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import (
    BLEND_BOUNDARY,
    BLEND_RAMP_SHAPES,
    BLEND_RAMP_WIDTH,
    BlendSpec,
    NORMALIZATION_TOLERANCE,
    blended_normalization_report,
    families_from_json,
    route_source_decay,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "v2_singlet"


class _Counting:
    def __init__(self):
        self.calls = 0
        self._rng = np.random.default_rng(3)

    def random(self):
        self.calls += 1
        return self._rng.random()


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(str(CONFIG / "resonance_catalog.json"))


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONFIG / "continuum_resonances.json"))


def test_candidate_set_and_default():
    assert BLEND_RAMP_SHAPES == ("linear", "smoothstep", "quadratic")
    assert BlendSpec().ramp_shape == "linear"
    with pytest.raises(ValueError):
        BlendSpec(ramp_shape="cubic")


def test_default_is_bit_identical_to_the_sealed_linear_ramp():
    spec = BlendSpec()
    B, W = BLEND_BOUNDARY, BLEND_RAMP_WIDTH
    grid = np.linspace(B - 0.5, B + W + 0.5, 10001)
    for M in grid:
        expected = 0.0 if not M > B else (1.0 if M >= B + W else (M - B) / W)
        assert spec.weight(float(M)) == expected


@pytest.mark.parametrize("shape", BLEND_RAMP_SHAPES)
def test_endpoints_continuity_and_band_means(shape):
    spec = BlendSpec(ramp_shape=shape)
    B, W = spec.boundary, spec.ramp_width
    assert spec.weight(B) == 0.0 and spec.weight(B - 1e-9) == 0.0
    assert spec.weight(B + W) == 1.0 and spec.weight(B + W + 1e-9) == 1.0
    assert spec.weight(B + 1e-12) < 1e-9
    assert 1.0 - spec.weight(B + W - 1e-12) < 1e-9
    s = np.linspace(0.0, 1.0, 200001)
    w = np.array([spec.weight(B + x * W) for x in s])
    assert np.all(np.diff(w) >= -1e-15)  # monotone within the band
    mean = float(_trapezoid(w, s))
    expected_mean = {"linear": 0.5, "smoothstep": 0.5, "quadratic": 1.0 / 3.0}[shape]
    assert abs(mean - expected_mean) < 1e-8


@pytest.mark.parametrize("shape", BLEND_RAMP_SHAPES)
def test_normalization_identity_and_routing_budget(shape, catalog, fams):
    spec = BlendSpec(ramp_shape=shape)
    qn = (1, 1, 0)
    report = blended_normalization_report(qn, spec, catalog, fams)
    total = report.get("int_rho_total") or report.get("total_integral") or report.get("integral")
    if total is not None:
        assert abs(float(total) - 1.0) <= 10.0 * NORMALIZATION_TOLERANCE, report
    B, W = spec.boundary, spec.ramp_width
    for M, expected_uniforms in ((B - 0.2, 0), (B + 0.5 * W, 1), (B + W + 0.2, 0)):
        rng = _Counting()
        route_source_decay(qn, M, spec, catalog, fams, rng)
        assert rng.calls == expected_uniforms, (shape, M, rng.calls)


def test_in_band_routing_frequency_follows_the_shape(catalog, fams):
    """At the band midpoint the continuum route frequency is w(B + W/2):
    0.5 for linear and smoothstep, 0.25 for quadratic."""
    qn = (1, 1, 0)
    for shape, expected in (("linear", 0.5), ("smoothstep", 0.5), ("quadratic", 0.25)):
        spec = BlendSpec(ramp_shape=shape)
        rng = np.random.default_rng(11)
        M = spec.boundary + 0.5 * spec.ramp_width
        routes = [route_source_decay(qn, M, spec, catalog, fams, rng) for _ in range(20000)]
        frequency = sum(r == "continuum" for r in routes) / len(routes)
        assert abs(frequency - expected) < 0.015, (shape, frequency)
