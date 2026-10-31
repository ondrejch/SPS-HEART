"""Card tests for the baryonic continuum spectral family.

Round 3, card 4 (``TASK-20260917-r3-baryonic-continuum-spectral-family``;
plan section 22 task 18, plan sections 9.1-9.2).  The card's
deterministic tests against the committed families of
``config/v2_singlet/continuum_resonances.json``:

- ``test_threshold_zero_below_m_min``: the density is exactly 0 strictly
  below ``m_min`` and exactly 0 at ``m_min`` (no continuum strength at
  or below the catalog-completeness boundary, plan section 9.3).
- ``test_ramp_continuity``: the density is C^1 at ``m_min`` and at
  ``m_min + ramp_width`` (one-sided derivatives agree to 1e-12).
- ``test_unit_normalization``: the report integral is within 1e-6 of 1
  for both families on the fixed 10,001-node reference grid.
- ``test_draw_determinism_and_support``: 1,000 draws at a fixed seed are
  reproducible, lie in ``[m_min, m_max]`` with ``m_max = 5.0 GeV``, and
  consume exactly two uniforms per draw (the card-3 ``draw_mass``
  convention) on the 10,001-node grid.
- ``test_allowed_decay_classes_conserve_qn``: every allowed class of
  both families conserves the family ``(B, Q, S)`` at the vertex,
  checked against the committed catalog's quantum numbers.
- ``test_color_singlet_audit``: every family record carries the
  ``continuum_resonance`` state stamp and no name implies strings,
  parton splitting, or constituent breakup (plan section 4.3, line 220).

plus the card's statistical test ``test_draw_moments`` (100,000 draws
per family; sample mean and variance against the quadrature moments at
3 sigma).

Card-5 accommodation (``TASK-20260917-r3-mesonic-continuum-spectral-
family``, the established later-card pattern): the committed config now
carries the card-5 mesonic families after the baryonic section.  This
file's tests stay scoped to the card-4 baryonic surface; the mesonic
families' own pins live in ``tests/test_spectral_density_mesonic.py``.

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import SPECTRAL_QUADRATURE_NODES
from na49_mixing.spectral_density import (
    CONTINUUM_STATE_TYPE,
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    ContinuumFamily,
    family_density,
    family_draw,
    family_normalization_report,
    families_from_json,
)
from na49_mixing.state_validation import (
    STATE_TYPES,
    is_color_singlet,
    validate_state_type,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: The pre-registered family records of this card: the baryonic
#: N*_cont-like family (1, 1, 0) and the strange baryonic family
#: (1, 0, -1), both active, both on the shared baryonic scale.  (The
#: card-5 mesonic families follow them in the config; this file's
#: tests stay scoped to these two baryonic families.)
EXPECTED_FAMILY_IDS = ("baryonic_np", "baryonic_strange")

#: The resolved allowed-decay-class identifiers per family (the committed
#: catalog's multiplet-class keys, daughter flavor pairs in the recorded
#: baryon-first order).  The QN audit below re-derives the conservation
#: from the catalog; this pin makes the resolution explicit and
#: reviewable.  For baryonic_np the card-named classes resolve to the
#: (1, 1, 0)-conserving charge states of N-pi, Delta-pi (the N-pi-pi
#: cascade class), K-Lambda, and K-Sigma.  For baryonic_strange no
#: charge state of the S = 0 K-Lambda / K-Sigma / N-pi classes conserves
#: S = -1 at the vertex, so the allowed set resolves to the
#: strangeness-conserving catalog classes of the established S = -1
#: baryon spectrum: Lambda-pi, Sigma-pi, and N-anti-K (all six are
#: recorded channels of the Lambda(1520) and Sigma(1385) records).
EXPECTED_CLASSES = {
    "baryonic_np": (
        "(2212,111)",   # p pi0        [N-pi]
        "(2112,211)",   # n pi+        [N-pi isospin partner]
        "(2224,-211)",  # Delta++ pi-  [N-pi-pi via cascade]
        "(2214,111)",   # Delta+ pi0   [cascade partner]
        "(2114,211)",   # Delta0 pi+   [cascade partner]
        "(3122,321)",   # Lambda K+    [K-Lambda]
        "(3212,321)",   # Sigma0 K+    [K-Sigma]
        "(3222,311)",   # Sigma+ K0    [K-Sigma isospin partner]
        "(3224,311)",  # Sigma(1385)+ K0 [Round 6 tasks 41-42]
        "(3214,321)",  # Sigma(1385)0 K+ [Round 6]
        "(102134,321)", # Lambda(1520) K+ [Round 6]
        "(2212,113)",  # proton rho0 [Block B stage 1]
        "(2112,213)",  # neutron rho+ [Block B stage 1]
        "(2212,223)",  # proton omega [Block B stage 1]
        "(2212,221)",  # proton eta [Block B stage 1]
        "(12212,111)",  # N*(1440)+ pi0 [Block B stage 1]
        "(12112,211)",  # N*(1440)0 pi+ [Block B stage 1]
        "(2216,111)",  # N*(1520)+ pi0 [Block B stage 1]
        "(2116,211)",  # N*(1520)0 pi+ [Block B stage 1]
        "(102212,111)",  # N(1535)+ pi0 [Block B stage 1]
        "(102112,211)",  # N(1535)0 pi+ [Block B stage 1]
        "(122212,111)",  # N(1650)+ pi0 [Block B stage 1]
        "(122112,211)",  # N(1650)0 pi+ [Block B stage 1]
        "(22212,111)",  # N*(1680)+ pi0 [Block B stage 1]
        "(22112,211)",  # N*(1680)0 pi+ [Block B stage 1]
        "(212214,111)",  # N(1720)+ pi0 [Block B stage 1]
        "(212114,211)",  # N(1720)0 pi+ [Block B stage 1]
        "(202224,-211)",  # Delta(1600)++ pi- [Block B stage 1]
        "(202214,111)",  # Delta(1600)+ pi0 [Block B stage 1]
        "(202114,211)",  # Delta(1600)0 pi+ [Block B stage 1]
        "(122224,-211)",  # Delta(1700)++ pi- [Block B stage 1]
        "(122214,111)",  # Delta(1700)+ pi0 [Block B stage 1]
        "(122114,211)",  # Delta(1700)0 pi+ [Block B stage 1]
        "(2224,-213)",  # Delta++ rho- [Block B stage 1]
        "(2214,113)",  # Delta+ rho0 [Block B stage 1]
        "(2114,213)",  # Delta0 rho+ [Block B stage 1]
        "(12212,113)",  # N*(1440)+ rho0 [Block B stage 1]
        "(12112,213)",  # N*(1440)0 rho+ [Block B stage 1]
        "(2216,113)",  # N*(1520)+ rho0 [Block B stage 1]
        "(2116,213)",  # N*(1520)0 rho+ [Block B stage 1]
        "(12212,223)",  # N*(1440)+ omega [Block B stage 1]
    ),
    "baryonic_strange": (
        "(3122,111)",   # Lambda pi0   [Lambda-pi]
        "(3212,111)",   # Sigma0 pi0   [Sigma-pi]
        "(3112,211)",   # Sigma- pi+   [Sigma-pi isospin partner]
        "(3222,-211)",  # Sigma+ pi-   [Sigma-pi isospin partner]
        "(2212,-321)",  # p K-         [N-anti-K]
        "(2112,-311)",  # n Kbar0      [N-anti-K isospin partner]
        "(3114,211)",  # Sigma(1385)- pi+ [Round 6 task 42]
        "(3214,111)",  # Sigma(1385)0 pi0 [Round 6]
        "(3224,-211)", # Sigma(1385)+ pi- [Round 6]
        "(102134,111)", # Lambda(1520) pi0 [Round 6]
    ),
}

#: The master seed of the round (round 3 index item 7), fixed for the
#: deterministic and statistical draw tests.
MASTER_SEED = 20260917

_CLASS_ID_PATTERN = re.compile(r"\((-?\d+),(-?\d+)\)")

#: Plan section 4.3, line 220: no module or configuration name implies
#: strings, parton splitting, or constituent breakup.
_BANNED_NAME_TOKENS = (
    "string", "parton", "constituent", "quark", "gluon",
    "breakup", "splitting",
)


def _load_families() -> list[ContinuumFamily]:
    return families_from_json(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def families() -> list[ContinuumFamily]:
    return _load_families()


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _class_daughters(class_id: str) -> tuple[int, int]:
    """Parse one multiplet-class identifier into its daughter pair."""
    match = _CLASS_ID_PATTERN.fullmatch(class_id)
    if match is None:
        raise AssertionError(
            f"allowed-decay-class identifier {class_id!r} is not a "
            "catalog multiplet-class key '(daughter0,daughter1)'"
        )
    return int(match.group(1)), int(match.group(2))


# ----------------------------------------------------------------------
# Deterministic tests
# ----------------------------------------------------------------------
def test_threshold_zero_below_m_min(families):
    """The density is exactly 0 strictly below m_min and exactly 0 at
    m_min (no continuum strength below the boundary), and positive
    above it."""
    probes = (1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.5)
    for family in families:
        for delta in probes:
            assert family_density(family, family.m_min - delta) == 0.0
        assert family_density(family, family.m_min) == 0.0
        # Positive just above the threshold and on the constant branch.
        assert family_density(family, family.m_min + 1e-9) > 0.0
        assert family_density(family, family.m_min + family.ramp_width + 0.1) > 0.0


def test_ramp_continuity(families):
    """The density is C^1 at m_min and at m_min + ramp_width.

    The smoothstep polynomial s^2 (3 - 2 s) has exact zero derivative at
    s = 0 and s = 1, so the density is C^1 at both junctions once the
    implemented ramp reproduces the smoothstep.  Verified here three
    ways, all deterministic:

    1. the implemented ramp equals the analytic smoothstep to <= 1e-12
       (relative) at 100 probe points of the ramp interval, and equals
       1.0 exactly at the upper junction;
    2. one-sided derivative stencils agree to <= 1e-12 at both
       junctions: at m_min the left side is the zero function (left
       derivative exactly 0) and the 3-point right stencil is 0 to
       1e-12; at m_min + W the 3-point one-sided stencils on each side
       agree to 1e-12 (step 3e-6 GeV, where the O(h^2) stencil bias and
       the float rounding of the numerator are both below the
       tolerance);
    3. the junction values are continuous: the density at m_min is
       exactly 0.0, and the one-sided value offsets at the upper
       junction stay within h * (local slope bound).
    """
    h = 3e-6
    for family in families:
        m_min = family.m_min
        junction = m_min + family.ramp_width

        def dens(M: float) -> float:
            return family_density(family, M)

        # (1) The implemented ramp is the analytic smoothstep.
        n_norm = family.n_norm
        for k in range(1, 101):
            x = k / 100.0
            M = m_min + x * family.ramp_width
            shape = M ** (-family.b_power) * math.exp(M / family.t_h)
            ramp_measured = dens(M) / (n_norm * shape)
            ramp_reference = x * x * (3.0 - 2.0 * x)
            assert abs(ramp_measured - ramp_reference) <= 1e-12
        upper_shape = junction ** (-family.b_power) * math.exp(
            junction / family.t_h
        )
        assert dens(junction) / (n_norm * upper_shape) == 1.0

        # (2) One-sided derivative stencils at both junctions.
        # Lower junction: the left derivative is exactly 0 (the density
        # is the zero function below m_min); the right 3-point stencil
        # vanishes to 1e-12 because the smoothstep slope is exactly 0
        # at s = 0.
        assert dens(m_min - h) == 0.0
        left_lower = 0.0
        right_lower = (-3.0 * dens(m_min)
                       + 4.0 * dens(m_min + h)
                       - dens(m_min + 2.0 * h)) / (2.0 * h)
        assert abs(right_lower - left_lower) <= 1e-12
        # Upper junction: 3-point one-sided stencils on each side.
        left_upper = (3.0 * dens(junction)
                      - 4.0 * dens(junction - h)
                      + dens(junction - 2.0 * h)) / (2.0 * h)
        right_upper = (-3.0 * dens(junction)
                       + 4.0 * dens(junction + h)
                       - dens(junction + 2.0 * h)) / (2.0 * h)
        assert abs(right_upper - left_upper) <= 1e-12

        # (3) Junction values are continuous.
        assert dens(m_min) == 0.0
        assert abs(dens(junction - h) - dens(junction)) <= 1e-7
        assert abs(dens(junction + h) - dens(junction)) <= 1e-7


def test_unit_normalization(families):
    """The report integral is within 1e-6 of 1 for both baryonic
    families on the 10,001-node reference grid, and the baryonic
    section of the config stays exactly the two active baryonic
    families (the card-5 mesonic families extend the config behind
    them; their normalization is pinned in
    tests/test_spectral_density_mesonic.py)."""
    baryonic = [f for f in families if f.family_id in EXPECTED_FAMILY_IDS]
    assert [f.family_id for f in baryonic] == list(EXPECTED_FAMILY_IDS)
    assert all(f.active for f in baryonic)
    for family in baryonic:
        report = family_normalization_report(family)
        assert report["family_id"] == family.family_id
        assert report["grid_points"] == REFERENCE_GRID_POINTS == 10_001
        assert report["support"] == [family.m_min, SUPPORT_M_MAX]
        assert abs(report["integral"] - 1.0) <= NORMALIZATION_TOLERANCE
        assert report["within_tolerance"] is True


def test_draw_determinism_and_support(families):
    """1,000 draws at a fixed seed are reproducible, all lie in
    [m_min, m_max] (m_max = 5.0), the draw runs on the 10,001-node
    reference grid, and each draw consumes exactly two uniforms (the
    card-3 draw_mass convention)."""
    assert REFERENCE_GRID_POINTS == SPECTRAL_QUADRATURE_NODES
    assert SUPPORT_M_MAX == 5.0
    n_draws = 1_000
    # Card-5 accommodation: the draw tests stay scoped to the baryonic
    # families (the card-5 declared-inactive families are never
    # sampled; the neutral mesonic family's draw contract is pinned in
    # tests/test_spectral_density_mesonic.py).
    baryonic = [f for f in families if f.family_id in EXPECTED_FAMILY_IDS]
    for family in baryonic:
        rng_a = np.random.default_rng(MASTER_SEED)
        draws_a = [family_draw(family, rng_a) for _ in range(n_draws)]
        rng_b = np.random.default_rng(MASTER_SEED)
        draws_b = [family_draw(family, rng_b) for _ in range(n_draws)]
        assert draws_a == draws_b  # reproducible at the fixed seed
        assert all(family.m_min <= m <= SUPPORT_M_MAX for m in draws_a)

        # Exactly two uniforms per draw: a counting stub around the same
        # bit stream must reproduce the same masses.
        class _CountingRng:
            def __init__(self, seed: int) -> None:
                self._rng = np.random.default_rng(seed)
                self.calls = 0

            def random(self) -> float:
                self.calls += 1
                return float(self._rng.random())

        counter = _CountingRng(MASTER_SEED)
        draws_c = [
            family_draw(family, counter)  # type: ignore[arg-type]
            for _ in range(n_draws)
        ]
        assert counter.calls == 2 * n_draws
        assert draws_c == draws_a


# ----------------------------------------------------------------------
# Catalog audit
# ----------------------------------------------------------------------
def test_allowed_decay_classes_conserve_qn(families, catalog):
    """Every allowed class of both baryonic families conserves the
    family (B, Q, S) at the vertex, checked against the committed
    catalog's quantum numbers, and every class is kinematically open
    across the family support (the daughter pole-mass sum lies below
    m_min)."""
    # Card-5 accommodation: the baryonic classes are pinned here; the
    # mesonic families' conserving classes are pinned in
    # tests/test_spectral_density_mesonic.py.
    baryonic = [f for f in families if f.family_id in EXPECTED_CLASSES]
    for family in baryonic:
        assert family.allowed_decay_classes == EXPECTED_CLASSES[family.family_id]
        assert len(set(family.allowed_decay_classes)) == len(
            family.allowed_decay_classes
        )
        for class_id in family.allowed_decay_classes:
            d0, d1 = _class_daughters(class_id)
            rec0 = catalog.get(d0)  # KeyError would fail the audit
            rec1 = catalog.get(d1)
            vertex = (int(rec0.B) + int(rec1.B),
                      int(rec0.Q) + int(rec1.Q),
                      int(rec0.S) + int(rec1.S))
            assert vertex == tuple(family.qn), (
                f"family {family.family_id}: class {class_id} carries "
                f"vertex (B, Q, S) = {vertex}, family requires "
                f"{tuple(family.qn)}"
            )
            threshold = float(rec0.pole_mass) + float(rec1.pole_mass)
            assert threshold < family.m_min, (
                f"family {family.family_id}: class {class_id} has "
                f"threshold {threshold} at or above m_min {family.m_min}"
            )


def test_color_singlet_audit(families):
    """Every family record carries the ``continuum_resonance`` state
    stamp (the Round 1 card-4 label), integer color-singlet quantum
    numbers, and no name implies strings, parton splitting, or
    constituent breakup (plan section 4.3, line 220)."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = {record["family_id"]: record for record in config["families"]}
    # Card-5 accommodation: the baryonic records stay exactly the two
    # card-4 families at the head of the families array; the card-5
    # mesonic records extend the config behind them (their state-stamp
    # and naming audit runs below over every record).
    assert [r["family_id"] for r in config["families"]][:2] == list(
        EXPECTED_FAMILY_IDS
    )
    assert set(EXPECTED_FAMILY_IDS) < set(records)

    # The audited names: the record keys and identifier-valued values,
    # the module file, and the config file.
    audited = []
    for family_id, record in records.items():
        for key in record:
            audited.append((f"{family_id}.<key>", str(key)))
        audited.append((f"{family_id}.family_id", str(record["family_id"])))
        audited.append((f"{family_id}.state_type", str(record["state_type"])))
        for class_id in record["allowed_decay_classes"]:
            audited.append((f"{family_id}.class", str(class_id)))
    audited.append(("module file", "spectral_density.py"))
    audited.append(("config file", CONFIG_PATH.name))
    for label, name in audited:
        for token in _BANNED_NAME_TOKENS:
            assert token not in name.lower(), (
                f"{label} {name!r} carries the banned name token "
                f"{token!r} (plan section 4.3, line 220)"
            )

    for family in families:
        record = records[family.family_id]
        # The Round 1 card-4 stamp, verbatim and inside the closed set.
        assert record["state_type"] == CONTINUUM_STATE_TYPE
        assert record["state_type"] in STATE_TYPES
        validate_state_type(record["state_type"])  # raises when invalid
        # Integer color-singlet quantum numbers.
        b, q, s = (int(v) for v in record["qn"])
        assert tuple(record["qn"]) == tuple(family.qn)
        assert is_color_singlet(b, q, s)
        # The frozen dataclass carries no mutable surface.
        with pytest.raises(dataclasses.FrozenInstanceError):
            family.m_min = 1.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# Statistical test
# ----------------------------------------------------------------------
def test_draw_moments(families):
    """100,000 draws per baryonic family: the sample mean and variance
    of the drawn mass agree with the quadrature moments to 3 sigma
    (standard errors of 1/sqrt(N) scaling, the variance through the
    kurtosis), and the relative mean deviation stays within 1e-3."""
    n_draws = 100_000
    sigma_tolerance = 3.0
    rng = np.random.default_rng(MASTER_SEED)
    # Card-5 accommodation: the moment test stays scoped to the
    # baryonic families (the neutral mesonic family's moment test is
    # tests/test_spectral_density_mesonic.py::test_neutral_meson_draw_moments).
    baryonic = [f for f in families if f.family_id in EXPECTED_FAMILY_IDS]
    for family in baryonic:
        xs = np.linspace(family.m_min, SUPPORT_M_MAX, REFERENCE_GRID_POINTS)
        ys = np.array([family_density(family, float(x)) for x in xs])
        widths = xs[1:] - xs[:-1]
        mid = 0.5 * (ys[1:] + ys[:-1])

        def trapz(values: np.ndarray) -> float:
            return float((0.5 * (values[1:] + values[:-1]) * widths).sum())

        mean_q = trapz(ys * xs)
        var_q = trapz(ys * xs * xs) - mean_q * mean_q
        mu4_q = trapz(ys * (xs - mean_q) ** 4)

        draws = np.array(
            [family_draw(family, rng) for _ in range(n_draws)]
        )
        mean_s = float(draws.mean())
        var_s = float(draws.var())

        se_mean = math.sqrt(var_q / n_draws)
        z_mean = (mean_s - mean_q) / se_mean
        se_var = math.sqrt(max(mu4_q - var_q * var_q, 0.0) / n_draws)
        z_var = (var_s - var_q) / se_var
        assert abs(z_mean) <= sigma_tolerance, (
            f"family {family.family_id}: mean z = {z_mean:.3f}"
        )
        assert abs(z_var) <= sigma_tolerance, (
            f"family {family.family_id}: variance z = {z_var:.3f}"
        )
        assert abs(mean_s - mean_q) / mean_q <= 1e-3
