"""Independent complementary coverage for the smooth catalog/continuum
matching (Round 3, card 6).

Round 3, card 6 (``TASK-20260917-r3-smooth-catalog-continuum-matching``;
plan section 22 task 20, plan section 9.3; nested feat commit
``0c5f21c``) extended ``na49_mixing/spectral_density.py`` with
``BlendSpec`` (the pre-registered piecewise ramp ``w(M)``,
``B = 2.680194110727528 GeV``, ``W = 0.5 GeV``), ``catalog_density``,
``continuum_density``, ``blended_density``, ``route_source_decay``,
and ``blended_normalization_report``, with the solver's card tests in
``tests/test_catalog_continuum_matching.py``.  This file is the
test-author's independent complement: it pins the same committed
behavior from a different angle, without touching the solver's file and
without duplicating its assertions.  The coverage added here:

- ``test_blend_weight_edge_semantics``: the ramp's edge values at the
  exact junctions (``w(B) == 0.0``, ``w(B + W) == 1.0``) and one
  representable float past each junction (``nextafter`` probes), plus
  monotone non-decreasing behavior on a 10,000-point probe across
  ``[B - 1e-9, B + W + 1e-9]``.
- ``test_blend_spec_frozen_record_and_pinned_constants``: ``BlendSpec``
  is a frozen dataclass (both fields raise
  ``dataclasses.FrozenInstanceError`` on mutation,
  ``dataclasses.is_frozen`` reports True), the module constants are
  pinned to the pre-registered values (``BLEND_BOUNDARY ==
  2.680194110727528``, ``BLEND_RAMP_WIDTH == 0.5``), the default-
  constructed spec carries exactly those values, and the boundary is
  the M8-frozen ``continuum.match_mass`` of the ``legacy_v1`` manifest
  (independent cross-link to the freeze record).
- ``test_density_edge_behavior``: for all three activated QN sets the
  continuum density is exactly ``0.0`` at and below ``m_min = B`` (a
  probe set including ``nextafter(B, -inf)``) and strictly positive
  just above the boundary; the catalog density is finite and
  non-negative at ``B``, ``B + W``, and ``m_max``; the strange-baryonic
  set (``(1, 0, -1)``) carries no record support above the boundary, so
  its catalog density is exactly ``0.0`` at all three probe points
  (pinned), while the baryonic and neutral-mesonic sets are strictly
  positive at ``B`` (record supports crossing the boundary).
- ``test_blended_density_triple_algebra_and_catalog_scaling``: the
  ``blended_density`` triple satisfies
  ``rho_total == (1 - w) * rho_catalog + w * rho_cont`` to 1e-15 at
  the probe set ``{B, B + 0.1, B + 0.25, B + 0.45, B + W}`` for all
  three activated QN sets, the returned ``rho_cont`` is the pinned
  unit-normalized family sum (bit-identical to ``continuum_density``),
  and the returned ``rho_catalog`` is the raw catalog mixture at the
  report's boundary-matching strength (bit-identical to
  ``matching_factor * catalog_density``; the shape is untouched, the
  factor fixes the strength only).
- ``test_routing_no_activated_family_never_continuum``: for the QN set
  with catalog records but no activated continuum family
  (``(1, -1, 0)``), the routing never returns ``'continuum'``: below
  ``B`` it returns ``'catalog'`` with zero uniforms, in the transition
  band it consumes exactly one uniform and either returns
  ``'catalog'`` (the draw is not below ``w(M)``) or raises the named
  ``ValueError`` (pinned message) when the draw routes to the
  continuum, and at or above ``B + W`` it raises the same named
  ``ValueError`` with zero uniforms; for a QN with neither catalog
  records nor a loaded family (``(2, 0, 0)``) every mass region raises
  a named ``ValueError`` (the pinned "no open record" / "no activated
  family" messages) with the documented uniform budget, and the
  ``blended_normalization_report`` raises a named ``ValueError`` for
  both sets.
- ``test_routing_determinism_cumulative_uniforms``: at the probe seed
  ``20260920`` (disjoint from the card's master seed ``20260917``),
  10,000 routing calls over the fixed mass grid spanning
  ``[B, B + W)`` consume exactly 10,000 uniforms in total (cumulative
  checkpoints at 2,000/4,000/6,000/8,000/10,000), the routing sequence
  is bit-identical across two independent runs (routes and final
  generator state), the empirical continuum count is pinned
  (5,013 of 10,000), and the sequence matches the independent
  recomputation from the raw uniform stream (``u < w(M)``).
- ``test_report_structure_and_integrals_recompute``: the
  ``blended_normalization_report`` output carries the pinned key
  structure (top level, ``integrals``, ``split_integrals``,
  ``catalog_matching``, ``double_counting_identity``), the double-
  counting identity deviations (both forms) are within 1e-6 for all
  three activated QN sets, the reported per-density integrals
  reproduce the trapezoid recomputed from ``blended_density`` on the
  10,001-node grid over the report support to 1e-12 (same for the
  split sum), and the report is JSON-serializable.

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import (
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    BlendSpec,
    blended_density,
    blended_normalization_report,
    catalog_density,
    continuum_density,
    families_from_json,
    route_source_decay,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent  # dev/mixing2006
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

#: The activated source QN sets of the card (the R31 windows, round 3
#: index item 1): baryonic, strange baryonic, neutral mesonic.
ACTIVATED_QN = ((1, 1, 0), (1, 0, -1), (0, 0, 0))

#: The pre-registered blend boundary and ramp width (the M8-frozen
#: ``continuum.match_mass``; round 3 index item 4).
BOUNDARY = 2.680194110727528
RAMP_WIDTH = 0.5

#: The probe seed for the routing determinism test, disjoint from the
#: card's master seed 20260917.
PROBE_SEED = 20260920

#: The QN set with committed catalog records (Delta-, Delta(1600)-,
#: Delta(1700)-) but no activated continuum family (the declared
#: mesonic families are neutral-sector only).
NO_FAMILY_QN = (1, -1, 0)

#: The QN set with neither catalog records nor a loaded family at all.
NO_RECORD_QN = (2, 0, 0)

#: The pinned named-value-error messages of ``route_source_decay`` for
#: the no-state routing (quantum numbers rendered by the f-string
#: ``{tuple(qn)!r}``).
NO_FAMILY_MSG = (
    "route_source_decay: the continuum path has no activated "
    "family with the source quantum numbers (1, -1, 0) (the "
    "declared inactive families are never sampled)"
)
NO_RECORD_MSG_AT_M = (
    "route_source_decay: the catalog path at mass 2.5801941107275277 GeV "
    "has no open record with the source quantum numbers (2, 0, 0) "
    "(the card-9/R23 open set is empty; no fallback route exists)"
)
NO_RECORD_FAMILY_MSG = (
    "route_source_decay: the continuum path has no activated "
    "family with the source quantum numbers (2, 0, 0) (the "
    "declared inactive families are never sampled)"
)

#: The pinned ``blended_normalization_report`` rejection messages for
#: the no-state QN sets.
REPORT_NO_FAMILY_MSG = (
    "blended matching: no activated continuum family carries "
    "the source quantum numbers (1, -1, 0)"
)
REPORT_NO_RECORD_MSG = (
    "blended matching: the catalog carries no record with the "
    "source quantum numbers (2, 0, 0)"
)

#: The empirical continuum-routing count over the 10,000-node grid of
#: ``test_routing_determinism_cumulative_uniforms`` at seed
#: ``PROBE_SEED`` (pinned; the sequence is bit-identical).
DETERMINISM_N = 10_000
DETERMINISM_CONTINUUM_COUNT = 5_013


def _load_families() -> list:
    return families_from_json(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def families() -> list:
    return _load_families()


@pytest.fixture(scope="module")
def spec() -> BlendSpec:
    return BlendSpec()


class _CountingRng:
    """A uniform source that counts its draws (the established card-4
    counting-stub pattern: the same bit stream, one counter)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())

    @property
    def state(self) -> dict:
        return self._rng.bit_generator.state


# ----------------------------------------------------------------------
# Blend weight
# ----------------------------------------------------------------------
def test_blend_weight_edge_semantics(spec):
    """The ramp's exact edge values at the junctions and one
    representable float past each, plus monotone non-decreasing
    behavior on a 10,000-point probe across
    ``[B - 1e-9, B + W + 1e-9]``."""
    assert spec.boundary == BOUNDARY
    assert spec.ramp_width == RAMP_WIDTH
    # Exact junction values (the branches close at the junctions).
    assert spec.weight(BOUNDARY) == 0.0
    assert spec.weight(BOUNDARY + RAMP_WIDTH) == 1.0
    # One representable float past each junction stays on the flat
    # branches.
    assert spec.weight(np.nextafter(BOUNDARY, -np.inf)) == 0.0
    assert spec.weight(np.nextafter(BOUNDARY + RAMP_WIDTH, np.inf)) == 1.0
    # Monotone non-decreasing on the probe window.
    probe = np.linspace(BOUNDARY - 1e-9, BOUNDARY + RAMP_WIDTH + 1e-9, 10_000)
    assert probe[0] < BOUNDARY and probe[-1] > BOUNDARY + RAMP_WIDTH
    weights = np.array([spec.weight(float(M)) for M in probe])
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0)
    assert np.all(np.diff(weights) >= 0.0)
    assert weights.min() == 0.0 and weights.max() == 1.0


def test_blend_spec_frozen_record_and_pinned_constants():
    """``BlendSpec`` is a frozen value object; the module constants are
    the pre-registered values; the default spec carries them; and the
    boundary is the M8-frozen ``continuum.match_mass`` of the
    ``legacy_v1`` manifest."""
    assert BlendSpec.__dataclass_params__.frozen is True
    spec = BlendSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.boundary = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.ramp_width = 1.0  # type: ignore[misc]
    # The record is unchanged by the attempted mutations.
    assert spec.boundary == BOUNDARY
    assert spec.ramp_width == RAMP_WIDTH
    # The pinned constants.
    assert BLEND_BOUNDARY == 2.680194110727528
    assert BLEND_RAMP_WIDTH == 0.5
    assert BlendSpec().boundary == BLEND_BOUNDARY
    assert BlendSpec().ramp_width == BLEND_RAMP_WIDTH
    # Independent cross-link: the boundary is the M8-frozen
    # continuum.match_mass recorded in the legacy_v1 manifest.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["continuum.match_mass"] == BLEND_BOUNDARY


# ----------------------------------------------------------------------
# Density edge behavior
# ----------------------------------------------------------------------
def test_density_edge_behavior(spec, catalog, families):
    """Continuum density is exactly 0.0 at and below ``m_min = B`` and
    strictly positive just above it, for all three activated QN sets;
    catalog density is finite and non-negative at ``B``, ``B + W``,
    and ``m_max`` (the strange-baryonic set is exactly 0.0 at all
    three, every record support ending below the boundary; the other
    two sets are strictly positive at ``B``)."""
    below_probes = (
        BOUNDARY - 0.5,
        BOUNDARY - 1e-6,
        BOUNDARY - 1e-12,
        BOUNDARY,
        np.nextafter(BOUNDARY, -np.inf),
    )
    for qn in ACTIVATED_QN:
        # The continuum side: no strength at or below the boundary.
        for M in below_probes:
            assert continuum_density(qn, M, families) == 0.0
        # Strictly positive just above the boundary (the ramp opens).
        assert continuum_density(qn, BOUNDARY + 0.01, families) > 0.0
        # The catalog side: finite and non-negative at the blend
        # junctions and the support upper end.
        for M in (BOUNDARY, BOUNDARY + RAMP_WIDTH, SUPPORT_M_MAX):
            value = catalog_density(qn, M, catalog)
            assert math.isfinite(value)
            assert value >= 0.0
    # The strange-baryonic set: every record support ends below the
    # boundary (the highest m_up is 1.59750 GeV), so the catalog
    # density is exactly 0.0 at all three blend probe points.
    for M in (BOUNDARY, BOUNDARY + RAMP_WIDTH, SUPPORT_M_MAX):
        assert catalog_density((1, 0, -1), M, catalog) == 0.0
    # The baryonic and neutral-mesonic sets: record supports cross the
    # boundary, so the catalog density is strictly positive at B.
    for qn in ((1, 1, 0), (0, 0, 0)):
        assert catalog_density(qn, BOUNDARY, catalog) > 0.0


# ----------------------------------------------------------------------
# Blended density triple
# ----------------------------------------------------------------------
def test_blended_density_triple_algebra_and_catalog_scaling(
    spec, catalog, families
):
    """The ``blended_density`` triple satisfies the plan section 9.3
    blend algebra to 1e-15 at the probe set
    ``{B, B + 0.1, B + 0.25, B + 0.45, B + W}`` for all three
    activated QN sets; ``rho_cont`` is bit-identical to
    ``continuum_density``; ``rho_catalog`` is the raw catalog mixture
    at the report's boundary-matching strength (the factor fixes the
    strength only, the shape is untouched)."""
    probes = (
        BOUNDARY,
        BOUNDARY + 0.1,
        BOUNDARY + 0.25,
        BOUNDARY + 0.45,
        BOUNDARY + RAMP_WIDTH,
    )
    for qn in ACTIVATED_QN:
        report = blended_normalization_report(qn, spec, catalog, families)
        factor = report["catalog_matching"]["matching_factor"]
        for M in probes:
            rho_total, rho_catalog, rho_cont = blended_density(
                qn, M, spec, catalog, families
            )
            w = spec.weight(M)
            # The blend algebra on the returned triple, to 1e-15.
            assert math.isfinite(rho_total)
            assert rho_total >= 0.0
            assert (
                abs(rho_total - ((1.0 - w) * rho_catalog + w * rho_cont))
                <= 1e-15
            )
            # The continuum component is the pinned family sum.
            assert rho_cont == continuum_density(qn, M, families)
            # The catalog component is the raw mixture at the
            # boundary-matching strength.
            raw = catalog_density(qn, M, catalog)
            assert rho_catalog == factor * raw
        # The scaling factor is a positive finite strength.
        assert math.isfinite(factor) and factor > 0.0


# ----------------------------------------------------------------------
# Routing with no activated continuum family
# ----------------------------------------------------------------------
def test_routing_no_activated_family_never_continuum(
    spec, catalog, families
):
    """For the QN set with catalog records but no activated continuum
    family (``(1, -1, 0)``), the routing never returns
    ``'continuum'``: below ``B`` it is ``'catalog'`` with zero
    uniforms; in the band it consumes exactly one uniform and is
    either ``'catalog'`` (at ``B``, where ``w(M) = 0``) or the named
    ``ValueError`` when the draw routes to the continuum; at and
    above ``B + W`` it is the named ``ValueError`` with zero
    uniforms.  For a QN with neither records nor a family
    (``(2, 0, 0)``) every region raises a named ``ValueError`` with
    the documented uniform budget, and the report raises for both
    sets."""
    # --- (1, -1, 0): catalog records open, no activated family ------
    # Below the boundary: 'catalog', zero uniforms (state untouched).
    rng = np.random.default_rng(PROBE_SEED)
    state_before = rng.bit_generator.state
    assert (
        route_source_decay(NO_FAMILY_QN, BOUNDARY - 0.1, spec,
                           catalog, families, rng)
        == "catalog"
    )
    assert rng.bit_generator.state == state_before

    # At B the weight is exactly 0, so the one band uniform can never
    # route to the continuum: deterministic 'catalog', one uniform.
    counter = _CountingRng(PROBE_SEED)
    assert route_source_decay(NO_FAMILY_QN, BOUNDARY, spec,
                             catalog, families, counter) == "catalog"
    assert counter.calls == 1

    # In the band at the probe seed: the first uniform
    # 0.043746673856235097 lies below w(M) at every pinned probe, so
    # the routed continuum path is rejected by the named ValueError,
    # exactly one uniform consumed.
    u0 = float(np.random.default_rng(PROBE_SEED).random())
    assert u0 == 0.043746673856235097
    for M in (BOUNDARY + 0.1, BOUNDARY + 0.25, BOUNDARY + 0.45):
        assert u0 < spec.weight(M)
        counter = _CountingRng(PROBE_SEED)
        with pytest.raises(ValueError, match="no activated family"):
            route_source_decay(NO_FAMILY_QN, M, spec,
                               catalog, families, counter)
        assert counter.calls == 1
        with pytest.raises(ValueError) as excinfo:
            route_source_decay(NO_FAMILY_QN, M, spec,
                               catalog, families,
                               np.random.default_rng(PROBE_SEED))
        assert str(excinfo.value) == NO_FAMILY_MSG

    # At and above B + W: named ValueError, zero uniforms.
    for M in (BOUNDARY + RAMP_WIDTH, BOUNDARY + RAMP_WIDTH + 0.1):
        rng = np.random.default_rng(PROBE_SEED)
        state_before = rng.bit_generator.state
        with pytest.raises(ValueError) as excinfo:
            route_source_decay(NO_FAMILY_QN, M, spec,
                               catalog, families, rng)
        assert str(excinfo.value) == NO_FAMILY_MSG
        assert rng.bit_generator.state == state_before

    # --- (2, 0, 0): no records, no family --------------------------
    # Below B: the catalog path has no open record; zero uniforms.
    rng = np.random.default_rng(PROBE_SEED)
    state_before = rng.bit_generator.state
    with pytest.raises(ValueError) as excinfo:
        route_source_decay(NO_RECORD_QN, BOUNDARY - 0.1, spec,
                           catalog, families, rng)
    assert str(excinfo.value) == NO_RECORD_MSG_AT_M
    assert rng.bit_generator.state == state_before

    # At B: w(B) = 0, so the catalog route is taken: the
    # no-open-record message, one uniform consumed.
    counter = _CountingRng(PROBE_SEED)
    with pytest.raises(ValueError) as excinfo:
        route_source_decay(NO_RECORD_QN, BOUNDARY, spec,
                           catalog, families, counter)
    assert "no open record" in str(excinfo.value)
    assert counter.calls == 1
    # At B + 0.1: u0 < w(B + 0.1), the continuum route is taken and
    # rejected: the no-family message, one uniform.
    counter = _CountingRng(PROBE_SEED)
    with pytest.raises(ValueError) as excinfo:
        route_source_decay(NO_RECORD_QN, BOUNDARY + 0.1, spec,
                           catalog, families, counter)
    assert str(excinfo.value) == NO_RECORD_FAMILY_MSG
    assert counter.calls == 1

    # At and above B + W: named ValueError, zero uniforms.
    rng = np.random.default_rng(PROBE_SEED)
    state_before = rng.bit_generator.state
    with pytest.raises(ValueError) as excinfo:
        route_source_decay(NO_RECORD_QN, BOUNDARY + RAMP_WIDTH, spec,
                           catalog, families, rng)
    assert str(excinfo.value) == NO_RECORD_FAMILY_MSG
    assert rng.bit_generator.state == state_before

    # The report requires both sides: named ValueError for both sets.
    with pytest.raises(ValueError) as excinfo:
        blended_normalization_report(NO_FAMILY_QN, spec, catalog, families)
    assert str(excinfo.value) == REPORT_NO_FAMILY_MSG
    with pytest.raises(ValueError) as excinfo:
        blended_normalization_report(NO_RECORD_QN, spec, catalog, families)
    assert str(excinfo.value) == REPORT_NO_RECORD_MSG


# ----------------------------------------------------------------------
# Routing determinism
# ----------------------------------------------------------------------
def test_routing_determinism_cumulative_uniforms(spec, catalog, families):
    """At seed ``20260920``, 10,000 routing calls over the fixed mass
    grid spanning ``[B, B + W)`` consume exactly 10,000 uniforms in
    total (cumulative checkpoints), the routing sequence is
    bit-identical across two independent runs, the empirical
    continuum count is pinned, and the sequence matches the
    independent recomputation from the raw uniform stream
    (``u < w(M)``)."""
    grid = np.linspace(BOUNDARY, BOUNDARY + RAMP_WIDTH - 1e-12, DETERMINISM_N)
    assert grid[0] == BOUNDARY and grid[-1] < BOUNDARY + RAMP_WIDTH

    def run() -> tuple[list, _CountingRng]:
        counter = _CountingRng(PROBE_SEED)
        routes = [
            route_source_decay((1, 1, 0), float(M), spec,
                               catalog, families, counter)
            for M in grid
        ]
        return routes, counter

    routes_a, counter_a = run()
    routes_b, counter_b = run()

    # Exactly one uniform per band call; cumulative checkpoints.
    for k in (2_000, 4_000, 6_000, 8_000, DETERMINISM_N):
        counter = _CountingRng(PROBE_SEED)
        for i in range(k):
            route_source_decay((1, 1, 0), float(grid[i]), spec,
                               catalog, families, counter)
        assert counter.calls == k
    assert counter_a.calls == DETERMINISM_N
    assert counter_b.calls == DETERMINISM_N

    # Bit-identical routing across two independent runs: the route
    # sequence, the call count, and the final generator state.
    assert routes_a == routes_b
    assert counter_a.state == counter_b.state

    # Every route is a closed-set value and the empirical continuum
    # count is pinned.
    assert set(routes_a) <= {"catalog", "continuum"}
    assert sum(route == "continuum" for route in routes_a) \
        == DETERMINISM_CONTINUUM_COUNT

    # The sequence matches the independent recomputation from the raw
    # uniform stream of the probe seed (u < w(M) to 'continuum').
    uniforms = np.random.default_rng(PROBE_SEED).random(DETERMINISM_N)
    expected = [
        "continuum" if u < spec.weight(float(M)) else "catalog"
        for u, M in zip(uniforms, grid)
    ]
    assert routes_a == expected

    # A plain np.random.Generator run consumes the identical bit
    # stream: the final state of two independently seeded-and-driven
    # generators is bit-identical.
    rng_one = np.random.default_rng(PROBE_SEED)
    rng_two = np.random.default_rng(PROBE_SEED)
    for M in grid:
        route_source_decay((1, 1, 0), float(M), spec,
                           catalog, families, rng_one)
    for M in grid:
        route_source_decay((1, 1, 0), float(M), spec,
                           catalog, families, rng_two)
    assert rng_one.bit_generator.state == rng_two.bit_generator.state


# ----------------------------------------------------------------------
# Report structure
# ----------------------------------------------------------------------
_REPORT_TOP_KEYS = frozenset({
    "qn", "grid_points", "support", "boundary", "ramp_width",
    "integrals", "split_integrals", "catalog_matching",
    "double_counting_identity", "tolerance",
})
_REPORT_INTEGRAL_KEYS = frozenset({"rho_total", "rho_catalog", "rho_cont"})
_REPORT_SPLIT_KEYS = frozenset({
    "catalog_below_boundary", "rho_total_at_or_above_boundary", "sum",
})
_REPORT_MATCHING_KEYS = frozenset({
    "cont_integral", "ramp_complement", "captured_raw", "matching_factor",
})
_REPORT_IDENTITY_FORM_KEYS = frozenset({
    "integral", "deviation", "tolerance", "within_tolerance",
})


def test_report_structure_and_integrals_recompute(spec, catalog, families):
    """The ``blended_normalization_report`` carries the pinned key
    structure; the double-counting identity deviations (both forms)
    are within 1e-6 for all three activated QN sets; and the reported
    per-density integrals reproduce the trapezoid recomputed from
    ``blended_density`` on the 10,001-node grid over the report
    support to 1e-12 (same for the split sum)."""
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for qn in ACTIVATED_QN:
        report = blended_normalization_report(qn, spec, catalog, families)

        # The pinned key structure at every level.
        assert set(report) == _REPORT_TOP_KEYS
        assert set(report["integrals"]) == _REPORT_INTEGRAL_KEYS
        assert set(report["split_integrals"]) == _REPORT_SPLIT_KEYS
        assert set(report["catalog_matching"]) == _REPORT_MATCHING_KEYS
        assert set(report["double_counting_identity"]) == {"total", "split"}
        for form in ("total", "split"):
            assert set(report["double_counting_identity"][form]) \
                == _REPORT_IDENTITY_FORM_KEYS

        # The echoes.
        assert report["qn"] == list(qn)
        assert report["grid_points"] == REFERENCE_GRID_POINTS == 10_001
        assert report["boundary"] == BOUNDARY
        assert report["ramp_width"] == RAMP_WIDTH
        assert report["tolerance"] == NORMALIZATION_TOLERANCE

        # The support: the union-support upper end is the pre-
        # registered support upper end; the lower end is the lowest
        # matching record threshold, recomputed independently from
        # the raw catalog JSON.
        lo, hi = report["support"]
        assert hi == SUPPORT_M_MAX
        expected_lo = min(
            float(record["mass_range"][0])
            for record in raw["records"]
            if (int(record["B"]), int(record["Q"]), int(record["S"])) == qn
        )
        assert lo == expected_lo

        # The double-counting identity, both forms, within 1e-6.
        identity = report["double_counting_identity"]
        assert identity["total"]["deviation"] <= 1e-6
        assert identity["total"]["within_tolerance"] is True
        assert identity["split"]["deviation"] <= 1e-6
        assert identity["split"]["within_tolerance"] is True
        assert abs(report["integrals"]["rho_total"] - 1.0) <= 1e-6
        assert abs(report["split_integrals"]["sum"] - 1.0) <= 1e-6

        # The matching pieces recombine to the identity exactly:
        # factor * captured_raw == 1 - D + G, to float rounding.
        matching = report["catalog_matching"]
        lhs = matching["matching_factor"] * matching["captured_raw"]
        rhs = 1.0 - matching["cont_integral"] + matching["ramp_complement"]
        assert abs(lhs - rhs) <= 1e-12
        assert math.isfinite(matching["matching_factor"])
        assert matching["matching_factor"] > 0.0

        # The reported per-density integrals reproduce the trapezoid
        # recomputed from blended_density on the 10,001-node grid
        # over the report support, to 1e-12.
        xs = np.linspace(lo, hi, REFERENCE_GRID_POINTS)
        total = np.empty(xs.size)
        cat = np.empty(xs.size)
        cont = np.empty(xs.size)
        for i, M in enumerate(xs):
            rho_total, rho_catalog, rho_cont = blended_density(
                qn, float(M), spec, catalog, families
            )
            total[i] = rho_total
            cat[i] = rho_catalog
            cont[i] = rho_cont
        widths = xs[1:] - xs[:-1]

        def trapezoid(ys: np.ndarray) -> float:
            return float((0.5 * (ys[1:] + ys[:-1]) * widths).sum())

        assert abs(trapezoid(total) - report["integrals"]["rho_total"]) \
            <= 1e-12
        assert abs(trapezoid(cat) - report["integrals"]["rho_catalog"]) \
            <= 1e-12
        assert abs(trapezoid(cont) - report["integrals"]["rho_cont"]) \
            <= 1e-12
        below = xs < BOUNDARY
        split = trapezoid(np.where(below, cat, 0.0)) + trapezoid(
            np.where(below, 0.0, total)
        )
        assert abs(split - report["split_integrals"]["sum"]) <= 1e-12
        assert abs(
            report["split_integrals"]["catalog_below_boundary"]
            - trapezoid(np.where(below, cat, 0.0))
        ) <= 1e-12
        assert abs(
            report["split_integrals"]["rho_total_at_or_above_boundary"]
            - trapezoid(np.where(below, 0.0, total))
        ) <= 1e-12

        # JSON-serializable (the report is a reviewable artifact).
        json.dumps(report)
