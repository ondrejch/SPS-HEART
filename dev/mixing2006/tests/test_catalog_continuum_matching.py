"""Card tests for the smooth catalog/continuum matching.

Round 3, card 6 (``TASK-20260917-r3-smooth-catalog-continuum-matching``;
plan section 22 task 20, plan section 9.3).  The card's deterministic
tests against the committed catalog and the committed families of
``config/v2_singlet/continuum_resonances.json``:

- ``test_w_ramp``: the blend weight ``w(M)`` is 0 below the boundary
  ``B``, linear on ``[B, B + W]``, 1 above, and continuous at ``B`` and
  at ``B + W`` (one-sided values agree to 1e-12).
- ``test_blended_density_continuity``: ``rho_total`` is continuous at
  ``B`` and at ``B + W``; ``rho_total(B-) == rho_catalog(B)`` and
  ``rho_total(B+W+) == rho_cont(B+W)`` (the card's one-sided
  identities), and the blend algebra holds pointwise.
- ``test_no_double_counting_identity``: the
  ``blended_normalization_report`` identity holds within 1e-6 for all
  three activated source QN sets (no strength is lost or doubled at
  the boundary).
- ``test_routing_bit_identity_below_boundary``: for ``M < B``,
  ``route_source_decay`` consumes zero uniforms and returns
  ``'catalog'``; the draw sequence of the catalog path is byte-identical
  to the R23 builder convention (fixture: fixed seed, the record draw
  and the mass draw match the pinned R23-convention reference values).
- ``test_routing_transition_uniform``: for ``M`` in ``[B, B + W)``,
  exactly one uniform is consumed and the routing matches
  ``u < w(M)`` to ``'continuum'``; for ``M >= B + W`` zero uniforms and
  ``'continuum'``.
- ``test_routing_lineage_traceable``: the routing result is returned
  (not just consumed), so the caller can stamp the source's
  ``state_type`` (catalog-mixture vs ``continuum_resonance``).

plus the card's statistical test ``test_routing_fractions`` (a
100,000-source mass grid spanning ``[B - 0.3, B + W + 0.3]`` per
activated QN set; the empirical continuum-routing fraction agrees with
the analytic ``int w(M) rho(M) / int rho(M)`` over the span to 3 sigma
of ``1/sqrt(N)``).

The blend normalization: the continuum side is pinned (every family is
unit-normalized over its support, the cards 4-5 contract), so the one
free normalization of the blend is the overall strength of the catalog
mixture.  The boundary-matching factor fixes that strength so
``int(rho_total) = 1``: the strength the blend takes from the catalog
(below the boundary plus the ``1 - w`` share of the transition band)
exactly complements the routed continuum strength.  The factor fixes
the strength only -- the catalog-density shape (the sum of the
unit-normalized record line shapes) and every family record are
untouched, and the routing weight is ``w(M)`` alone, so no Monte Carlo
draw depends on the factor.

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.resonance_widths import draw_mass
from na49_mixing.spectral_density import (
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    CONTINUUM_STATE_TYPE,
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    BlendSpec,
    _blended_pieces,
    _daughter_masses,
    blended_density,
    blended_normalization_report,
    catalog_density,
    continuum_density,
    families_from_json,
    route_source_decay,
)
from na49_mixing.state_validation import STATE_TYPES, validate_state_type

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: The activated source QN sets of the card (the R31 windows, round 3
#: index item 1): baryonic, strange baryonic, neutral mesonic.
ACTIVATED_QN = ((1, 1, 0), (1, 0, -1), (0, 0, 0))

#: The master seed of the round (round 3 index item 7), fixed for the
#: deterministic and statistical tests.
MASTER_SEED = 20260917

#: The pre-registered blend boundary and ramp width (the M8-frozen
#: ``continuum.match_mass``; round 3 index item 4).
BOUNDARY = 2.680194110727528
RAMP_WIDTH = 0.5

#: The below-boundary fixture: fixed (master seed, replica, event id),
#: source QN, and source mass ``M = B - 0.1`` (the catalog path).
FIXTURE_MASTER_SEED = 20260917
FIXTURE_REPLICA = 0
FIXTURE_EVENT_ID = 0
FIXTURE_QN = (1, 1, 0)
FIXTURE_MASS = BOUNDARY - 0.1

#: The pinned R23-convention reference draws at the fixture point (the
#: catalog-path draw sequence of the card's routing rule: the record
#: draw on the spawned ``decay_channel`` child -- the R23 builder's
#: expression ``min(int(u * len(open)), len(open) - 1)`` over the open
#: set at the source mass in catalog record order -- then the mass draw
#: on the spawned ``decay_kinematics`` child, the card-3 truncated
#: ``draw_mass`` on the record support truncated at the boundary ``B``).
#: Computed once from the committed catalog and pinned here; the
#: routing must leave both values bit-identical (zero uniforms).
FIXTURE_N_OPEN = 10
FIXTURE_RECORD_UNIFORM = 0.3201829545267483
FIXTURE_RECORD_INDEX = 3
FIXTURE_RECORD_PDG = 2216  # N*(1520)+
FIXTURE_RECORD_MASS = 1.6591845169259503  # re-pinned 2026-09-28 (corrected line shapes, FAB-07/08/09)

#: The pinned mass draw for a record whose support extends above the
#: boundary (N*(1440)+, pdg 12212): the window is the record support
#: truncated at ``B`` -- ``[1.0732489, 2.680194110727528]`` -- so the
#: pin demonstrates the truncation at the boundary itself.  Fixed seed
#: ``MASTER_SEED``, the card-3 truncated draw.
TRUNCATION_RECORD_PDG = 12212  # N*(1440)+
TRUNCATION_WINDOW = (1.07324891, 2.680194110727528)  # re-pinned 2026-09-28: the support starts at the lowest channel threshold (FAB-07)
TRUNCATION_SEED = 20260917
TRUNCATION_MASS = 1.7253944136502413

#: The statistical routing-fraction test (the card's statistical test).
FRACTION_N = 100_000
FRACTION_SIGMA = 3.0
FRACTION_SPAN_HALF_WIDTH = 0.3


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


# ----------------------------------------------------------------------
# Deterministic tests
# ----------------------------------------------------------------------
def test_w_ramp(spec):
    """``w(M)`` is 0 below ``B``, linear on ``[B, B + W]``, 1 above, and
    continuous at both junctions (one-sided values agree to 1e-12);
    the defaults are the pre-registered values and the ramp is fixed
    (the frozen dataclass carries no mutable surface)."""
    assert spec.boundary == BOUNDARY == BLEND_BOUNDARY
    assert spec.ramp_width == RAMP_WIDTH == BLEND_RAMP_WIDTH
    # The pre-registered piecewise values, exact at the branches.
    assert spec.weight(2.0) == 0.0
    assert spec.weight(BOUNDARY) == 0.0
    assert spec.weight(BOUNDARY + RAMP_WIDTH) == 1.0
    assert spec.weight(3.5) == 1.0
    # Linearity on the open interval: w(B + x W) = x to 1e-12.
    for k in range(1, 100):
        x = k / 100.0
        M = BOUNDARY + x * RAMP_WIDTH
        assert abs(spec.weight(M) - x) <= 1e-12
    # Continuity at B: the left side is the zero function; the one-sided
    # values agree to 1e-12 (the probe step is 1e-13, so the linear
    # branch moves by 2e-13).
    h = 1e-13
    assert spec.weight(BOUNDARY - h) == 0.0
    assert abs(spec.weight(BOUNDARY + h) - spec.weight(BOUNDARY)) <= 1e-12
    # Continuity at B + W: the one-sided values agree to 1e-12 and the
    # constant branch above is exact.
    assert abs(spec.weight(BOUNDARY + RAMP_WIDTH - h) - 1.0) <= 1e-12
    assert spec.weight(BOUNDARY + RAMP_WIDTH + h) == 1.0
    # The ramp is fixed: a second spec with the pre-registered values
    # weights identically, and the record is frozen.
    assert BlendSpec().weight(BOUNDARY + 0.13) == spec.weight(BOUNDARY + 0.13)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.ramp_width = 1.0  # type: ignore[misc]


def test_blended_density_continuity(spec, catalog, families):
    """``rho_total`` is continuous at ``B`` and at ``B + W``; the card's
    one-sided identities hold (``rho_total(B-) == rho_catalog(B)``,
    ``rho_total(B+W+) == rho_cont(B+W)``), the continuum density is
    exactly zero at ``B``, and the blend algebra holds pointwise."""
    h = 1e-14
    for qn in ACTIVATED_QN:
        _, cat_at_b, cont_at_b = blended_density(
            qn, BOUNDARY, spec, catalog, families
        )
        _, _, cont_at_top = blended_density(
            qn, BOUNDARY + RAMP_WIDTH, spec, catalog, families
        )
        # The continuum density is exactly 0.0 at B (no continuum
        # strength at or below the boundary, the card-4/5 threshold).
        assert cont_at_b == 0.0
        # Continuity at B (one-sided probes 1e-14 around the boundary).
        left = blended_density(qn, BOUNDARY - h, spec, catalog, families)[0]
        right = blended_density(qn, BOUNDARY + h, spec, catalog, families)[0]
        assert abs(left - right) <= 1e-12
        # rho_total(B-) == rho_catalog(B): below the boundary the weight
        # is exactly 0, so the left value is the catalog mixture.
        assert abs(left - cat_at_b) <= 1e-12
        # Continuity at B + W.
        below_top = blended_density(
            qn, BOUNDARY + RAMP_WIDTH - h, spec, catalog, families
        )[0]
        above_top = blended_density(
            qn, BOUNDARY + RAMP_WIDTH + h, spec, catalog, families
        )[0]
        assert abs(below_top - above_top) <= 1e-12
        # rho_total(B+W+) == rho_cont(B+W): above the ramp end the
        # weight is exactly 1, so the value is the continuum mixture.
        assert abs(above_top - cont_at_top) <= 1e-12
        # The blend algebra holds pointwise at probe masses on both
        # sides of both junctions.
        for M in (BOUNDARY - 0.1, BOUNDARY, BOUNDARY + 0.13,
                  BOUNDARY + RAMP_WIDTH, BOUNDARY + RAMP_WIDTH + 0.1):
            rho_total, rho_catalog, rho_cont = blended_density(
                qn, M, spec, catalog, families
            )
            w = spec.weight(M)
            expected = (1.0 - w) * rho_catalog + w * rho_cont
            assert rho_total == expected


def test_no_double_counting_identity(spec, catalog, families):
    """The ``blended_normalization_report`` identity holds within 1e-6
    for all three activated source QN sets: ``int(rho_total) = 1`` and
    ``int_{M<B}(rho_catalog) + int_{M>=B}(rho_total) = 1`` (no strength
    is lost or doubled at the boundary); the pinned continuum side
    stays unit-normalized and the matching pieces are reviewable."""
    assert REFERENCE_GRID_POINTS == 10_001
    for qn in ACTIVATED_QN:
        report = blended_normalization_report(qn, spec, catalog, families)
        assert report["qn"] == list(qn)
        assert report["grid_points"] == REFERENCE_GRID_POINTS
        lo, hi = report["support"]
        assert lo <= BOUNDARY and hi >= BOUNDARY + RAMP_WIDTH
        assert hi == SUPPORT_M_MAX
        # The pinned continuum side: the family sum stays unit-normalized.
        assert (
            abs(report["integrals"]["rho_cont"] - 1.0)
            <= NORMALIZATION_TOLERANCE
        )
        # The double-counting identity, both forms.
        identity = report["double_counting_identity"]
        assert identity["total"]["within_tolerance"] is True
        assert identity["split"]["within_tolerance"] is True
        assert identity["total"]["deviation"] <= 1e-6
        assert identity["split"]["deviation"] <= 1e-6
        assert abs(report["integrals"]["rho_total"] - 1.0) <= 1e-6
        split = report["split_integrals"]
        assert abs(split["sum"] - 1.0) <= 1e-6
        # The matching pieces recombine to the identity exactly: the
        # factor times the raw captured mass equals the routed
        # continuum's complement.
        matching = report["catalog_matching"]
        lhs = matching["matching_factor"] * matching["captured_raw"]
        rhs = 1.0 - matching["cont_integral"] + matching["ramp_complement"]
        assert abs(lhs - rhs) <= 1e-12
        # The matching factor is a positive strength scale; the catalog
        # and continuum records themselves are untouched (the report
        # integrals are of the blended densities only).
        assert math.isfinite(matching["matching_factor"])
        assert matching["matching_factor"] > 0.0


# ----------------------------------------------------------------------
# Routing tests
# ----------------------------------------------------------------------
def test_routing_bit_identity_below_boundary(spec, catalog, families):
    """For ``M < B`` the routing consumes zero uniforms and returns
    ``'catalog'``; the catalog-path draw sequence is byte-identical to
    the R23 builder convention at the same seed (the pinned record draw
    and mass draw), with and without a preceding routing call."""
    # Zero uniforms: the generator state is bit-unchanged.
    rng = np.random.default_rng(MASTER_SEED)
    state_before = rng.bit_generator.state
    route = route_source_decay(
        FIXTURE_QN, FIXTURE_MASS, spec, catalog, families, rng
    )
    assert route == "catalog"
    assert rng.bit_generator.state == state_before

    # The R23-convention catalog path (the card's routing rule): the
    # open set at the source mass in catalog record order, the record
    # draw on the spawned decay_channel child (the R23 builder's
    # expression), then the card-3 mass draw on the spawned
    # decay_kinematics child over the record support truncated at B.
    subs = derive_substreams(
        FIXTURE_MASTER_SEED, FIXTURE_REPLICA, FIXTURE_EVENT_ID
    )
    open_records = [
        record
        for record in catalog.records()
        if (int(record.B), int(record.Q), int(record.S)) == FIXTURE_QN
        and float(record.mass_range[0]) <= FIXTURE_MASS
    ]
    assert len(open_records) == FIXTURE_N_OPEN

    def catalog_path() -> tuple[int, int, float]:
        rng_record = np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
        )
        rng_mass = np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_kinematics")).spawn(1)[0]
        )
        u = rng_record.random()
        index = min(int(u * len(open_records)), len(open_records) - 1)
        record = open_records[int(index)]
        lo = float(record.mass_range[0])
        hi = min(float(record.mass_range[1]), BOUNDARY)
        dmap = _daughter_masses(catalog, record)
        drawn = draw_mass((lo, hi), record, dmap, rng_mass)
        assert drawn is not None
        return int(index), int(record.pdg_id), float(drawn)

    # The pinned R23-convention reference values.
    index, pdg, mass = catalog_path()
    assert index == FIXTURE_RECORD_INDEX
    assert pdg == FIXTURE_RECORD_PDG
    assert mass == FIXTURE_RECORD_MASS

    # Byte-identity: the same sequence preceded by the routing call on
    # the substream generators is bit-identical (the routing consumed
    # no uniforms, so the record draw, its uniform, and the mass draw
    # are exactly the no-blend values).
    rng_a = np.random.default_rng(MASTER_SEED)
    route_a = route_source_decay(
        FIXTURE_QN, FIXTURE_MASS, spec, catalog, families, rng_a
    )
    assert route_a == "catalog"
    assert rng_a.bit_generator.state == state_before
    assert catalog_path() == (index, pdg, mass)

    # The pinned record uniform reproduces the pinned index (the R23
    # builder's expression, pinned end to end).
    assert FIXTURE_RECORD_UNIFORM * len(open_records) >= FIXTURE_RECORD_INDEX
    assert (
        min(
            int(FIXTURE_RECORD_UNIFORM * len(open_records)),
            len(open_records) - 1,
        )
        == FIXTURE_RECORD_INDEX
    )

    # The truncation pin: a record whose support extends above the
    # boundary draws inside the record support truncated at B.
    record = catalog.get(TRUNCATION_RECORD_PDG)
    assert float(record.mass_range[1]) > BOUNDARY
    rng_trunc = np.random.default_rng(TRUNCATION_SEED)
    drawn = draw_mass(
        TRUNCATION_WINDOW, record, _daughter_masses(catalog, record), rng_trunc
    )
    assert drawn is not None
    assert float(drawn) == TRUNCATION_MASS
    assert TRUNCATION_WINDOW[0] <= drawn <= TRUNCATION_WINDOW[1]


def test_routing_transition_uniform(spec, catalog, families):
    """For ``M`` in ``[B, B + W)`` exactly one uniform is consumed and
    the routing matches ``u < w(M)`` to ``'continuum'``; at and above
    ``B + W`` zero uniforms and ``'continuum'``; below ``B`` zero
    uniforms and ``'catalog'``."""
    probes = (
        BOUNDARY,
        BOUNDARY + 0.05,
        BOUNDARY + 0.25,
        BOUNDARY + 0.45,
        BOUNDARY + RAMP_WIDTH - 1e-9,
    )
    for M in probes:
        counter = _CountingRng(MASTER_SEED)
        route = route_source_decay(qn=(1, 1, 0), M=M, spec=spec,
                                   catalog=catalog, fams=families, rng=counter)
        assert counter.calls == 1
        u = float(np.random.default_rng(MASTER_SEED).random())
        expected = "continuum" if u < spec.weight(M) else "catalog"
        assert route == expected
    # At B the weight is exactly 0, so the routing draw never routes to
    # the continuum (deterministic 'catalog' on the one-uniform branch).
    counter = _CountingRng(MASTER_SEED)
    assert route_source_decay(
        (1, 1, 0), BOUNDARY, spec, catalog, families, counter
    ) == "catalog"
    assert counter.calls == 1
    # At and above B + W: zero uniforms (state unchanged), 'continuum'.
    for M in (BOUNDARY + RAMP_WIDTH, 3.5):
        rng = np.random.default_rng(MASTER_SEED)
        state_before = rng.bit_generator.state
        route = route_source_decay((1, 1, 0), M, spec, catalog, families, rng)
        assert route == "continuum"
        assert rng.bit_generator.state == state_before
    # Just below B: zero uniforms, 'catalog'.
    rng = np.random.default_rng(MASTER_SEED)
    state_before = rng.bit_generator.state
    assert route_source_decay(
        (1, 1, 0), BOUNDARY - 1e-9, spec, catalog, families, rng
    ) == "catalog"
    assert rng.bit_generator.state == state_before


def test_routing_lineage_traceable(spec, catalog, families):
    """The routing result is returned (not just consumed), so the
    caller can stamp the source's ``state_type``: a ``'catalog'`` route
    keeps the catalog-mixture lineage (the drawn record's
    established-resonance identity) and a ``'continuum'`` route stamps
    ``continuum_resonance`` (the Round 1 card-4 stamp)."""
    lineage_stamp = {
        "catalog": "established_resonance",
        "continuum": CONTINUUM_STATE_TYPE,
    }
    assert CONTINUUM_STATE_TYPE == "continuum_resonance"
    for stamp in lineage_stamp.values():
        assert stamp in STATE_TYPES
        validate_state_type(stamp)  # raises when invalid

    routes = []
    for M in (BOUNDARY - 0.1, BOUNDARY + 0.45, BOUNDARY + RAMP_WIDTH + 0.1):
        route = route_source_decay(
            (1, 1, 0), M, spec, catalog, families,
            np.random.default_rng(MASTER_SEED),
        )
        assert route in ("catalog", "continuum")
        routes.append(route)
    # The returned route decides the stamp, so the lineage is
    # traceable from the return value alone.
    stamps = [lineage_stamp[route] for route in routes]
    assert stamps[0] == "established_resonance"
    assert stamps[1] in ("established_resonance", "continuum_resonance")
    assert stamps[2] == "continuum_resonance"
    # Every stamp of every route is a closed-set state type.
    for stamp in stamps:
        validate_state_type(stamp)


# ----------------------------------------------------------------------
# Statistical test
# ----------------------------------------------------------------------
def test_routing_fractions(spec, catalog, families):
    """On a 100,000-source mass grid spanning
    ``[B - 0.3, B + W + 0.3]`` per activated QN set, the empirical
    continuum-routing fraction agrees with the analytic
    ``int w(M) rho(M) / int rho(M)`` over the span to 3 sigma of
    ``1/sqrt(N)``.  The source masses are drawn from ``rho_total``
    restricted to the span (the fixed 10,001-node report-grid samples,
    the piecewise-linear inverse-CDF draw convention), so the empirical
    fraction estimates the analytic ratio; the routing consumes its own
    uniform per transition-band source."""
    tolerance = FRACTION_SIGMA / math.sqrt(FRACTION_N)
    for qn in ACTIVATED_QN:
        pieces = _blended_pieces(qn, spec, catalog, families)
        xs = pieces["xs"]
        rho_catalog = pieces["matching_factor"] * pieces["cat_raw"]
        rho_total = (
            pieces["one_minus_w"] * rho_catalog + pieces["w"] * pieces["cont"]
        )
        # The span sub-slice of the fixed report grid (the node spacing
        # is ~4e-4 GeV, so the clipped edges stay below one cell).
        lo_span = BOUNDARY - FRACTION_SPAN_HALF_WIDTH
        hi_span = BOUNDARY + spec.ramp_width + FRACTION_SPAN_HALF_WIDTH
        mask = (xs >= lo_span) & (xs <= hi_span)
        xs_s = xs[mask]
        rho_s = rho_total[mask]
        w_s = pieces["w"][mask]
        assert xs_s.size > 100
        widths = xs_s[1:] - xs_s[:-1]
        norm = float((0.5 * (rho_s[1:] + rho_s[:-1]) * widths).sum())
        weighted = float(
            (0.5 * ((w_s * rho_s)[1:] + (w_s * rho_s)[:-1]) * widths).sum()
        )
        p_analytic = weighted / norm
        assert 0.0 < p_analytic < 1.0

        # The 100,000-source mass grid: masses drawn from rho_total on
        # the span (inverse CDF on the piecewise-linear density model,
        # two uniforms per mass).
        cell_masses = 0.5 * (rho_s[1:] + rho_s[:-1]) * widths
        cdf = np.cumsum(cell_masses) / norm
        cell_width = (xs_s[-1] - xs_s[0]) / cdf.size
        rng_draw = np.random.default_rng(MASTER_SEED)
        u_segment = rng_draw.random(FRACTION_N)
        u_within = rng_draw.random(FRACTION_N)
        idx = np.searchsorted(cdf, u_segment, side="right")
        idx = np.minimum(idx, cdf.size - 1)
        masses = xs_s[0] + (idx + u_within) * cell_width
        assert np.all(masses >= xs_s[0]) and np.all(masses <= xs_s[-1])

        # The routing: one generator, the card-6 rule per source mass.
        rng_route = np.random.default_rng(MASTER_SEED + 1)
        n_continuum = 0
        for mass in masses:
            route = route_source_decay(
                qn, float(mass), spec, catalog, families, rng_route
            )
            if route == "continuum":
                n_continuum += 1
        empirical = n_continuum / FRACTION_N
        assert abs(empirical - p_analytic) <= tolerance, (
            f"qn {qn}: empirical continuum fraction {empirical!r} vs "
            f"analytic {p_analytic!r} (tolerance {tolerance!r})"
        )
