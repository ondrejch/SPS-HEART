"""Card tests for the mesonic continuum spectral family.

Round 3, card 5 (``TASK-20260917-r3-mesonic-continuum-spectral-family``;
plan section 22 task 19, plan sections 9.1-9.2).  The card's
deterministic tests against the committed families of
``config/v2_singlet/continuum_resonances.json``:

- ``test_neutral_meson_threshold_zero_below_m_min``: the neutral mesonic
  density is exactly 0 strictly below ``m_min`` and exactly 0 at
  ``m_min`` (no continuum strength below the catalog-completeness
  boundary, plan section 9.3).
- ``test_neutral_meson_ramp_continuity``: the density is C^1 at
  ``m_min`` and at ``m_min + ramp_width`` (one-sided derivatives agree
  to 1e-12).
- ``test_neutral_meson_unit_normalization``: the report integral is
  within 1e-6 of 1 on the fixed 10,001-node reference grid.
- ``test_neutral_meson_allowed_classes_conserve_qn``: every allowed
  class of the neutral mesonic family conserves ``(0, 0, 0)`` at the
  vertex against the committed catalog's quantum numbers.
- ``test_inactive_families_not_sampleable``: ``active_families``
  returns no charged/strange mesonic family, and a draw for an inactive
  family raises ``ValueError`` naming the family.
- ``test_baryonic_section_byte_stable``: the baryonic section of the
  config is unchanged from card 4's committed bytes (the re-pin absorbs
  the card-5 config extension).

plus the card's statistical test ``test_neutral_meson_draw_moments``
(100,000 draws of the neutral mesonic family; sample mean and variance
against the quadrature moments at 3 sigma), the config-surface pins
(``test_config_family_surface``: the seven committed families in order,
the active flags, and the one common mesonic scale shared by the three
mesonic families), and the neutral-family draw determinism and support
check (``test_neutral_meson_draw_determinism_and_support``).

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import SPECTRAL_QUADRATURE_NODES
from na49_mixing.spectral_density import (
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    ContinuumFamily,
    active_families,
    family_density,
    family_draw,
    family_normalization_report,
    families_from_json,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: The committed family records in file order: the card-4 baryonic
#: section (byte-stable) followed by the card-5 mesonic section (the
#: activated neutral family, then the four declared-inactive families).
EXPECTED_FAMILY_IDS = (
    "baryonic_np",
    "baryonic_strange",
    "mesonic_neutral",
    "mesonic_charged",
    "mesonic_anticharged",
    "mesonic_strange",
    "mesonic_antistrange",
)
EXPECTED_QNS = {
    "baryonic_np": (1, 1, 0),
    "baryonic_strange": (1, 0, -1),
    "mesonic_neutral": (0, 0, 0),
    "mesonic_charged": (0, 1, 0),
    "mesonic_anticharged": (0, -1, 0),
    "mesonic_strange": (0, 0, -1),
    "mesonic_antistrange": (0, 0, 1),
}
EXPECTED_ACTIVE = (True, True, True, False, False, False, False)
INACTIVE_FAMILY_IDS = (
    "mesonic_charged",
    "mesonic_anticharged",
    "mesonic_strange",
    "mesonic_antistrange",
)
NEUTRAL_FAMILY_ID = "mesonic_neutral"

#: The resolved allowed-decay-class identifiers of the neutral mesonic
#: family (the committed catalog's multiplet-class keys, daughter pairs
#: in the recorded channel order).  The QN audit below re-derives the
#: conservation from the catalog; this pin makes the resolution
#: explicit and reviewable.  Card-named resolution: the pi-pi classes
#: (the two-pion system of total isospin I = 0, 1, 2), the K-Kbar
#: classes (the neutral (K0, Kbar0) flavor pair in its weak-eigenstate
#: recorded (310, 130) form, the card-2 _concrete_pair convention), and
#: the scalar/tensor and other neutral meson classes (rho-pi, omega-pi,
#: eta-pi, eta-eta) with the recorded charge-conjugate partners.
EXPECTED_NEUTRAL_CLASSES = (
    "(211,-211)",   # pi+ pi-     [pi-pi, I = 0, 1, 2]
    "(111,111)",    # pi0 pi0     [pi-pi, I = 0, 1, 2]
    "(321,-321)",   # K+ K-       [K-Kbar]
    "(310,130)",    # K0S K0L     [K-Kbar, weak-eigenstate K0-Kbar0]
    "(113,111)",    # rho0 pi0    [rho-pi]
    "(213,-211)",   # rho+ pi-    [rho-pi]
    "(-213,211)",   # rho- pi+    [rho-pi charge conjugate]
    "(223,111)",    # omega pi0   [omega-pi]
    "(221,111)",    # eta pi0     [eta-pi]
    "(221,221)",    # eta eta     [eta-eta]
    "(323,-321)",   # K*+ K-      [K*-Kbar, Round 6 task 39]
    "(-323,321)",   # K*- K+      [K*-Kbar charge conjugate]
    "(313,-311)",   # K*0 Kbar0   [K*-Kbar neutral partner, task 38]
    "(-313,311)",   # Kbar*0 K0   [neutral partner]
    "(333,111)",   # phi pi0     [phi channel, task 40]
    "(333,221)",   # phi eta     [phi channel]
    "(2212,-2212)",   # p pbar      [baryon pair, task 43]
    "(2112,-2112)",   # n nbar      [baryon pair]
    "(113,113)",  # rho0 rho0 [Block B stage 1]
    "(213,-213)",  # rho+ rho- [Block B stage 1]
    "(223,223)",  # omega omega [Block B stage 1]
    "(113,223)",  # rho0 omega [Block B stage 1]
    "(221,223)",  # eta omega [Block B stage 1]
    "(225,111)",  # f2(1270) pi0 [Block B stage 1]
    "(115,111)",  # a2(1320)0 pi0 [Block B stage 1]
    "(215,-211)",  # a2(1320)+ pi- [Block B stage 1]
    "(-215,211)",  # a2(1320)- pi+ [Block B stage 1]
    "(20113,111)",  # a1(1260)0 pi0 [Block B stage 1]
    "(20213,-211)",  # a1(1260)+ pi- [Block B stage 1]
    "(-20213,211)",  # a1(1260)- pi+ [Block B stage 1]
    "(10113,111)",  # b1(1235)0 pi0 [Block B stage 1]
    "(10213,-211)",  # b1(1235)+ pi- [Block B stage 1]
    "(-10213,211)",  # b1(1235)- pi+ [Block B stage 1]
    "(323,-323)",  # K*(892)+ K*(892)- [Block B stage 1]
    "(313,-313)",  # K*(892)0 aK*(892)0 [Block B stage 1]
    "(100113,111)",  # rho(1450)0 pi0 [Block B stage 1]
    "(100213,-211)",  # rho(1450)+ pi- [Block B stage 1]
    "(-100213,211)",  # rho(1450)- pi+ [Block B stage 1]
    "(100223,111)",  # omega(1420) pi0 [Block B stage 1]
    "(113,225)",  # rho0 f2(1270) [Block B stage 1]
    "(223,225)",  # omega f2(1270) [Block B stage 1]
)

#: The analog conserving classes of the four declared-inactive families
#: (never sampled in Round 3; the classes are declared for schema
#: completeness and audited here for conservation).
EXPECTED_INACTIVE_CLASSES = {
    "mesonic_charged": (
        "(211,111)",    # pi+ pi0     [pi-pi charged state]
        "(223,211)",    # omega pi+   [omega-pi]
        "(213,111)",    # rho+ pi0    [rho-pi]
        "(113,211)",    # rho0 pi+    [rho-pi isospin partner]
        "(221,211)",    # eta pi+     [eta-pi]
        "(-311,321)",   # Kbar0 K+    [K-Kbar charged state]
        "(323,-311)",   # K*+ Kbar0   [K*-Kbar charge partner, task 39]
        "(321,-313)",   # K+ Kbar*0   [partner]
    ),
    "mesonic_anticharged": (
        "(-211,111)",   # pi- pi0     [pi-pi charged state]
        "(223,-211)",   # omega pi-   [omega-pi]
        "(-213,111)",   # rho- pi0    [rho-pi]
        "(113,-211)",   # rho0 pi-    [rho-pi isospin partner]
        "(221,-211)",   # eta pi-     [eta-pi]
        "(311,-321)",   # K0 K-       [K-Kbar charged state]
        "(-323,311)",   # K*- K0      [K*-Kbar charge partner, task 39]
        "(-321,313)",   # K- K*0      [partner]
    ),
    "mesonic_strange": (
        "(-321,211)",   # K- pi+      [K-pi]
        "(-311,111)",   # Kbar0 pi0   [K-pi isospin partner]
        "(213,-321)",   # rho+ K-     [rho-K]
        "(113,-311)",   # rho0 Kbar0  [rho-K isospin partner]
        "(-323,211)",   # K*- pi+     [K*-pi]
        "(-313,111)",   # K*0bar pi0  [K*-pi isospin partner]
        "(223,-311)",   # omega Kbar0 [omega-K]
    ),
    "mesonic_antistrange": (
        "(321,-211)",   # K+ pi-      [K-pi]
        "(311,111)",    # K0 pi0      [K-pi isospin partner]
        "(-213,321)",   # rho- K+     [rho-K]
        "(113,311)",    # rho0 K0     [rho-K isospin partner]
        "(323,-211)",   # K*+ pi-     [K*-pi]
        "(313,111)",    # K*0 pi0     [K*-pi isospin partner]
        "(223,311)",    # omega K0    [omega-K]
    ),
}

#: The byte-span digests of the two baryonic family records, computed
#: from card 4's committed config (nested commits e08ed99 / 5fb63e0).
#: The card-5 re-pin absorbs the config extension; this pin holds the
#: baryonic section to card 4's exact committed bytes.
BARYONIC_SPAN_SHA256 = {
    "baryonic_np": "ef02a19490ab14ecac388cba39898c80a687eda6bec322b6357f03e9d5a0eb71",
    "baryonic_strange": (
        "c805833b43127705ffb571f9e2433d290990361db4bae4daa60787fde81a4bff"
    ),
}

#: The pre-registered parameters shared by every family (index items
#: 2-3): the exponential shape, the M8-frozen minimum mass, the 0.5 GeV
#: smoothstep ramp and nominal width, and the 5.0 GeV support upper end.
SHARED_PARAMETERS = {
    "m_min": 2.680194110727528,
    "t_h": 0.165,
    "b_power": 0.0,
    "ramp_width": 0.5,
    "nominal_width": 0.5,
    "m_max": 5.0,
}

#: The master seed of the round (round 3 index item 7), fixed for the
#: deterministic and statistical draw tests.
MASTER_SEED = 20260917

_CLASS_ID_PATTERN = re.compile(r"\((-?\d+),(-?\d+)\)")


def _load_families() -> list[ContinuumFamily]:
    return families_from_json(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def families() -> list[ContinuumFamily]:
    return _load_families()


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _neutral_family(families: list[ContinuumFamily]) -> ContinuumFamily:
    (family,) = [f for f in families if f.family_id == NEUTRAL_FAMILY_ID]
    return family


def _class_daughters(class_id: str) -> tuple[int, int]:
    """Parse one multiplet-class identifier into its daughter pair."""
    match = _CLASS_ID_PATTERN.fullmatch(class_id)
    if match is None:
        raise AssertionError(
            f"allowed-decay-class identifier {class_id!r} is not a "
            "catalog multiplet-class key '(daughter0,daughter1)'"
        )
    return int(match.group(1)), int(match.group(2))


def _family_record_spans(path: Path) -> dict[str, bytes]:
    """The raw byte span of each family record object in the config.

    The records are located with ``json.JSONDecoder.raw_decode`` over
    the ``families`` array, so the returned bytes are the exact file
    bytes of each record object (the file is ASCII, so character and
    byte offsets coincide).
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert len(raw) == len(text), "the config must be ASCII"
    decoder = json.JSONDecoder()
    key_index = text.index('"families"')
    array_index = text.index("[", key_index)
    pos = array_index + 1
    spans: dict[str, bytes] = {}
    while True:
        while text[pos] in " \t\r\n,":
            pos += 1
        if text[pos] == "]":
            break
        start = pos
        obj, end = decoder.raw_decode(text, pos)
        spans[obj["family_id"]] = raw[start:end]
        pos = end
    return spans


# ----------------------------------------------------------------------
# Deterministic tests
# ----------------------------------------------------------------------
def test_neutral_meson_threshold_zero_below_m_min(families):
    """The neutral mesonic density is exactly 0 strictly below m_min and
    exactly 0 at m_min (no continuum strength below the boundary), and
    positive above it."""
    family = _neutral_family(families)
    probes = (1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.5)
    for delta in probes:
        assert family_density(family, family.m_min - delta) == 0.0
    assert family_density(family, family.m_min) == 0.0
    # Positive just above the threshold and on the constant branch.
    assert family_density(family, family.m_min + 1e-9) > 0.0
    assert family_density(
        family, family.m_min + family.ramp_width + 0.1
    ) > 0.0


def test_neutral_meson_ramp_continuity(families):
    """The neutral mesonic density is C^1 at m_min and at
    m_min + ramp_width.

    The smoothstep polynomial s^2 (3 - 2 s) has exact zero derivative at
    s = 0 and s = 1, so the density is C^1 at both junctions.  Verified
    three ways, all deterministic:

    1. the implemented ramp equals the analytic smoothstep to <= 1e-12
       (relative) at 100 probe points of the ramp interval, and equals
       1.0 exactly at the upper junction;
    2. one-sided derivative stencils agree to <= 1e-12 at both
       junctions (step 3e-6 GeV, where the O(h^2) stencil bias and the
       float rounding of the numerator are both below the tolerance);
    3. the junction values are continuous: the density at m_min is
       exactly 0.0, and the one-sided value offsets at the upper
       junction stay within h * (local slope bound).
    """
    family = _neutral_family(families)
    h = 3e-6
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
    assert dens(m_min - h) == 0.0
    left_lower = 0.0
    right_lower = (-3.0 * dens(m_min)
                   + 4.0 * dens(m_min + h)
                   - dens(m_min + 2.0 * h)) / (2.0 * h)
    assert abs(right_lower - left_lower) <= 1e-12
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


def test_neutral_meson_unit_normalization(families):
    """The report integral is within 1e-6 of 1 on the 10,001-node
    reference grid for the neutral mesonic family, and every active
    family of the config normalizes to 1 on the same grid."""
    family = _neutral_family(families)
    report = family_normalization_report(family)
    assert report["family_id"] == NEUTRAL_FAMILY_ID
    assert report["qn"] == [0, 0, 0]
    assert report["grid_points"] == REFERENCE_GRID_POINTS == 10_001
    assert report["support"] == [family.m_min, SUPPORT_M_MAX]
    assert abs(report["integral"] - 1.0) <= NORMALIZATION_TOLERANCE
    assert report["within_tolerance"] is True
    for active in [f for f in families if f.active]:
        active_report = family_normalization_report(active)
        assert abs(active_report["integral"] - 1.0) <= NORMALIZATION_TOLERANCE
        assert active_report["within_tolerance"] is True


def test_config_family_surface(families):
    """The config loads with exactly the seven committed families in
    file order, the card-5 active flags, and the one common mesonic
    scale: the three mesonic families share every density parameter
    (t_h = 0.165 GeV fixed and shared, b_a = 0, m_min = B, W = 0.5 GeV,
    nominal_width = 0.5 GeV, m_max = 5.0 GeV), so the neutral family's
    derived normalization equals the baryonic value."""
    assert tuple(f.family_id for f in families) == EXPECTED_FAMILY_IDS
    assert tuple(f.qn for f in families) == tuple(
        EXPECTED_QNS[fid] for fid in EXPECTED_FAMILY_IDS
    )
    assert tuple(f.active for f in families) == EXPECTED_ACTIVE

    mesonic = [f for f in families if f.family_id.startswith("mesonic_")]
    assert len(mesonic) == 5
    assert SUPPORT_M_MAX == SHARED_PARAMETERS["m_max"]
    for family in mesonic:
        assert family.t_h == 0.165, (
            f"family {family.family_id}: the mesonic scale must be the one "
            "common mesonic continuum scale T_H = 165 MeV shared by every "
            "mesonic family (plan section 9.2; no per-sector fit)"
        )
        assert family.m_min == SHARED_PARAMETERS["m_min"]
        assert family.b_power == SHARED_PARAMETERS["b_power"]
        assert family.ramp_width == SHARED_PARAMETERS["ramp_width"]
        assert family.nominal_width == SHARED_PARAMETERS["nominal_width"]
    neutral = _neutral_family(families)
    baryonic_np = families[0]
    assert neutral.n_norm == baryonic_np.n_norm


# ----------------------------------------------------------------------
# Catalog audit
# ----------------------------------------------------------------------
def test_neutral_meson_allowed_classes_conserve_qn(families, catalog):
    """Every allowed class of the neutral mesonic family conserves
    (0, 0, 0) at the vertex, checked against the committed catalog's
    quantum numbers, and every class is kinematically open across the
    family support (the daughter pole-mass sum lies below m_min).  The
    declared-inactive families carry the analog conserving classes,
    audited by the same rule."""
    neutral = _neutral_family(families)
    assert neutral.allowed_decay_classes == EXPECTED_NEUTRAL_CLASSES
    assert len(set(neutral.allowed_decay_classes)) == len(
        neutral.allowed_decay_classes
    )
    inactive = [f for f in families if not f.active]
    assert tuple(f.family_id for f in inactive) == INACTIVE_FAMILY_IDS
    for family in [neutral, *inactive]:
        expected = (
            EXPECTED_NEUTRAL_CLASSES
            if family.family_id == NEUTRAL_FAMILY_ID
            else EXPECTED_INACTIVE_CLASSES[family.family_id]
        )
        assert family.allowed_decay_classes == expected, (
            f"family {family.family_id}: the recorded classes are not the "
            "card's named conserving set"
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


# ----------------------------------------------------------------------
# Sampling surface
# ----------------------------------------------------------------------
def test_inactive_families_not_sampleable(families):
    """``active_families`` returns no charged/strange mesonic family,
    and a draw for an inactive family raises ``ValueError`` naming the
    family (the active flag controls sampling; the declared families
    are never sampled in Round 3)."""
    # No active family for any charged/strange mesonic qn.
    for family_id in INACTIVE_FAMILY_IDS:
        qn = EXPECTED_QNS[family_id]
        assert active_families(families, qn) == [], (
            f"{family_id}: the charged/strange mesonic qn {qn} must have "
            "no active family"
        )
    # The selector returns exactly the activated families.
    assert [f.family_id for f in active_families(families, (0, 0, 0))] == [
        NEUTRAL_FAMILY_ID
    ]
    assert [f.family_id for f in active_families(families, (1, 1, 0))] == [
        "baryonic_np"
    ]
    assert [f.family_id for f in active_families(families, (1, 0, -1))] == [
        "baryonic_strange"
    ]
    # A draw for an inactive family raises ValueError naming it.
    rng = np.random.default_rng(MASTER_SEED)
    for family_id in INACTIVE_FAMILY_IDS:
        (family,) = [f for f in families if f.family_id == family_id]
        with pytest.raises(ValueError, match=re.escape(family_id)):
            family_draw(family, rng)
        with pytest.raises(ValueError, match="not active"):
            family_draw(family, rng)


def test_neutral_meson_draw_determinism_and_support(families):
    """1,000 draws of the neutral mesonic family at a fixed seed are
    reproducible, all lie in [m_min, m_max] (m_max = 5.0), and consume
    exactly two uniforms per draw (the card-3 draw_mass convention) on
    the 10,001-node reference grid."""
    assert REFERENCE_GRID_POINTS == SPECTRAL_QUADRATURE_NODES
    family = _neutral_family(families)
    n_draws = 1_000
    rng_a = np.random.default_rng(MASTER_SEED)
    draws_a = [family_draw(family, rng_a) for _ in range(n_draws)]
    rng_b = np.random.default_rng(MASTER_SEED)
    draws_b = [family_draw(family, rng_b) for _ in range(n_draws)]
    assert draws_a == draws_b  # reproducible at the fixed seed
    assert all(family.m_min <= m <= SUPPORT_M_MAX for m in draws_a)

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
# Byte stability
# ----------------------------------------------------------------------
def test_baryonic_section_byte_stable():
    """The baryonic section of the config is unchanged from card 4's
    committed bytes (the re-pin absorbs the card-5 config extension):
    the raw file-byte span of each baryonic family record hashes to the
    digest computed from card 4's committed config, and the baryonic
    records remain the first two entries of the families array."""
    spans = _family_record_spans(CONFIG_PATH)
    ids = tuple(EXPECTED_QNS)
    assert tuple(spans)[:2] == ids[:2] == ("baryonic_np", "baryonic_strange")
    for family_id, digest in BARYONIC_SPAN_SHA256.items():
        assert hashlib.sha256(spans[family_id]).hexdigest() == digest, (
            f"the baryonic section record {family_id} is not byte-stable "
            "against card 4's committed config"
        )


# ----------------------------------------------------------------------
# Statistical test
# ----------------------------------------------------------------------
def test_neutral_meson_draw_moments(families):
    """100,000 draws of the neutral mesonic family: the sample mean and
    variance of the drawn mass agree with the quadrature moments to
    3 sigma (standard errors of 1/sqrt(N) scaling, the variance through
    the kurtosis), and the relative mean deviation stays within 1e-3."""
    family = _neutral_family(families)
    n_draws = 100_000
    sigma_tolerance = 3.0
    rng = np.random.default_rng(MASTER_SEED)
    xs = np.linspace(family.m_min, SUPPORT_M_MAX, REFERENCE_GRID_POINTS)
    ys = np.array([family_density(family, float(x)) for x in xs])
    widths = xs[1:] - xs[:-1]

    def trapz(values: np.ndarray) -> float:
        return float((0.5 * (values[1:] + values[:-1]) * widths).sum())

    mean_q = trapz(ys * xs)
    var_q = trapz(ys * xs * xs) - mean_q * mean_q
    mu4_q = trapz(ys * (xs - mean_q) ** 4)

    draws = np.array([family_draw(family, rng) for _ in range(n_draws)])
    mean_s = float(draws.mean())
    var_s = float(draws.var())

    se_mean = math.sqrt(var_q / n_draws)
    z_mean = (mean_s - mean_q) / se_mean
    se_var = math.sqrt(max(mu4_q - var_q * var_q, 0.0) / n_draws)
    z_var = (var_s - var_q) / se_var
    assert abs(z_mean) <= sigma_tolerance, f"mean z = {z_mean:.3f}"
    assert abs(z_var) <= sigma_tolerance, f"variance z = {z_var:.3f}"
    assert abs(mean_s - mean_q) / mean_q <= 1e-3
