"""Independent complementary coverage for the baryonic continuum
spectral family (Round 3, card 4).

Round 3, card 4 (``TASK-20260917-r3-baryonic-continuum-spectral-family``;
plan section 22 task 18, plan sections 9.1-9.2) landed
``na49_mixing/spectral_density.py`` and the family section of
``config/v2_singlet/continuum_resonances.json`` with the card's own test
file ``tests/test_spectral_density.py``.  This file is the test-author's
independent complement: it pins the same committed behavior from a
different angle, without touching the solver's file and without
duplicating its assertions.  The coverage added here:

- ``test_loader_rejects_missing_state_stamp``: the loader fails with a
  distinct, named ``ValueError`` for a record missing the
  ``continuum_resonance`` state stamp (missing-key message) and for a
  record carrying a foreign state stamp (named-stamp message).
- ``test_loader_rejects_m_max_below_m_min``: a recorded ``m_max`` below
  ``m_min`` fails with the named support-upper-end message (the
  pre-registered ``m_max = 5.0`` check fires before any geometry).
- ``test_loader_rejects_mismatched_n_norm``: a recorded ``n_norm``
  displaced beyond the module's recorder tolerance (relative 1e-9 vs
  the module's relative 1e-12) fails with the named
  does-not-reproduce-the-unit-normalization message.
- ``test_draw_contract_two_uniforms_per_draw_support_and_repro``: a
  proxy ``np.random.Generator`` wrapper counting uniform draws proves
  the per-draw contract at cumulative checkpoints (1, 100, 1,000,
  10,000 draws: exactly ``2n`` uniforms after the n-th draw, so no
  hidden rejection loop can hide in the total count), 10,000 draws all
  lie in ``[m_min, m_max]`` for both families, and draws at a fixed
  probe seed (distinct from the card's master seed, so the two test
  files exercise disjoint bit streams) are bit-identical across two
  independent runs.
- ``test_inverse_cdf_consistency_at_fixed_quantiles``: at fixed
  quantiles (0.05 to 0.95 in steps of 0.05), the module-convention mass
  (read from the module's internal CDF table, the public draw pinned by
  a scripted two-uniform probe) agrees with the exact root of the
  piecewise-linear grid CDF to grid-resolution precision.  Documented
  bound: one grid cell, i.e. ``(m_max - m_min) / 10,000`` = 1/10,000 of
  the support span (about 2.32e-4 GeV) in mass units, and the CDF mass
  of the cell containing the quantile in quantile units.
- ``test_grid_matches_round2_reference_quadrature``: the module's
  10,001-node grid node masses match the Round 2 reference quadrature
  (``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay`` and
  ``_NORMALIZATION_GRID_POINTS`` of the card-3 width law
  ``na49_mixing.resonance_widths``: the same 10,001-node uniform
  linspace construction over the support) exactly, with the endpoints
  at the config's ``m_min`` and ``m_max``.
- ``test_config_integrity_pre_registered_values_and_classes``: the
  committed config opens with the two baryonic families, both active,
  with the pre-registered values (``b_a = 0``, ``t_h = 0.165`` shared,
  ``m_min = 2.680194110727528``, ``ramp_width = 0.5``,
  ``m_max = 5.0``, ``nominal_width = 0.5``) and exactly the card's
  8 + 6 strangeness-conserving catalog class pairs, in card order; the
  recorded catalog reference digest matches the committed catalog
  file.
- ``test_class_pairs_conserve_family_qn_independent_catalog_audit``:
  every class pair conserves the family (B, Q, S), audited against the
  committed catalog read as raw JSON (independent of the
  ``HadronCatalog`` layer the card test uses).
- ``test_density_edge_values_and_grid_finiteness``: the density is
  exactly 0 at ``m_min``, equals the independently computed ramp-top
  value at ``m_min + ramp_width``, equals the independently computed
  constant-branch value at ``m_max`` (expected values recomputed from
  the raw config values, not the loaded record), and no NaN/inf appears
  anywhere on the full 10,001-node grid for either family.

Card-5 accommodation (``TASK-20260917-r3-mesonic-continuum-spectral-
family``, the established later-card pattern): the committed config now
carries the card-5 mesonic families behind the baryonic section.  The
loader-backed tests run over the active families the public loader
returns (the declared-inactive families are never sampled, card 5), and
the config pins stay scoped to the baryonic section; the mesonic
families' own pins live in ``tests/test_spectral_density_mesonic.py``.

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.  The
loader-failure variants are written under the repo scratch area
``dev/00runs/tmp/`` (never ``/tmp``) and removed after use.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from na49_mixing.resonance_decay import SPECTRAL_QUADRATURE_NODES
from na49_mixing.resonance_widths import _NORMALIZATION_GRID_POINTS
from na49_mixing import spectral_density as sd
from na49_mixing.spectral_density import (
    CONTINUUM_STATE_TYPE,
    REFERENCE_GRID_POINTS,
    family_density,
    family_draw,
    families_from_json,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent  # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent  # monorepo root: holds dev/
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: Scratch area for the loader-failure variant configs (repo scratch,
#: never /tmp).  Removed after each test; the directory itself is
#: exempt from the freeze inventory (dev/00runs/tmp).
SCRATCH_DIR = MONO_ROOT / "dev" / "00runs" / "tmp" / "test_spectral_density_independent"

#: The probe seed of this file: distinct from the card's master seed
#: 20260917 (tests/test_spectral_density.py) so the two files draw from
#: disjoint bit streams.
PROBE_SEED = 20260918

#: The pre-registered parameter values of this card (index items 2-4,
#: minimal choice), pinned here against the raw config JSON.
PRE_REGISTERED = {
    "b_power": 0.0,
    "t_h": 0.165,
    "m_min": 2.680194110727528,
    "ramp_width": 0.5,
    "m_max": 5.0,
    "nominal_width": 0.5,
}

#: The card's named class-pair resolution, in card order: the
#: (1, 1, 0)-conserving charge states of N-pi, Delta-pi (the N-pi-pi
#: cascade class), K-Lambda, and K-Sigma for baryonic_np, and the
#: strangeness-conserving catalog classes (Lambda-pi, Sigma-pi,
#: N-anti-K) for baryonic_strange.
EXPECTED_CLASS_PAIRS = {
    "baryonic_np": (
        "(2212,111)",   # p pi0        [N-pi]
        "(2112,211)",   # n pi+        [N-pi isospin partner]
        "(2224,-211)",  # Delta++ pi-  [N-pi-pi via cascade]
        "(2214,111)",   # Delta+ pi0   [cascade partner]
        "(2114,211)",   # Delta0 pi+   [cascade partner]
        "(3122,321)",   # Lambda K+    [K-Lambda]
        "(3212,321)",   # Sigma0 K+    [K-Sigma]
        "(3222,311)",   # Sigma+ K0    [K-Sigma isospin partner]
        "(3224,311)",   # Sigma(1385)+ K0  [K-Sigma* feed-down, Round 6 tasks 41-42]
        "(3214,321)",   # Sigma(1385)0 K+  [K-Sigma* feed-down]
        "(102134,321)",   # Lambda(1520) K+  [K-Lambda* feed-down]
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
        "(3114,211)",   # Sigma(1385)- pi+ [hyperon feed-down, task 42]
        "(3214,111)",   # Sigma(1385)0 pi0 [hyperon feed-down]
        "(3224,-211)",   # Sigma(1385)+ pi- [hyperon feed-down]
        "(102134,111)",   # Lambda(1520) pi0 [hyperon feed-down]
    ),
}

#: Fixed quantiles of the inverse-CDF consistency probe.
QUANTILES = tuple(round(0.05 * k, 2) for k in range(1, 20))  # 0.05..0.95

_CLASS_ID_PATTERN = re.compile(r"\((-?\d+),(-?\d+)\)")


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def config() -> dict:
    """The raw committed config payload."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog_raw() -> dict:
    """The raw committed catalog payload (independent of
    HadronCatalog)."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def families() -> list:
    """The committed families, loaded through the public loader (the
    card-4 baryonic pair followed by the card-5 mesonic families)."""
    return families_from_json(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def scratch_dir() -> Iterator[Path]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    yield SCRATCH_DIR
    for child in SCRATCH_DIR.iterdir():
        if child.is_file():
            child.unlink()


# ----------------------------------------------------------------------
# Loader validation (distinct, named failures)
# ----------------------------------------------------------------------
def _variant_config_path(
    scratch_dir: Path, name: str, raw: dict, mutate
) -> str:
    """Write one mutated copy of the committed config and return its
    path.  *mutate* receives the list of family records in place."""
    variant = json.loads(json.dumps(raw))
    mutate(variant["families"])
    path = scratch_dir / name
    path.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
    return str(path)


def test_loader_rejects_missing_state_stamp(config, scratch_dir):
    """A record missing the ``continuum_resonance`` state stamp, and a
    record carrying a foreign stamp, each fail with a distinct named
    ``ValueError``."""
    # (a) Missing stamp: the required-key check names the stamp key.
    def drop_stamp(records):
        del records[0]["state_type"]

    path = _variant_config_path(scratch_dir, "missing_state_stamp.json",
                                config, drop_stamp)
    try:
        with pytest.raises(ValueError) as excinfo:
            families_from_json(path)
        message = str(excinfo.value)
    finally:
        Path(path).unlink()
    assert "missing required keys" in message, (
        f"missing-stamp failure must carry the named required-key "
        f"message, got: {message}"
    )
    assert "'state_type'" in message, (
        f"missing-stamp failure must name the state_type key, got: "
        f"{message}"
    )

    # (b) Foreign stamp: the named-stamp check fires.
    def foreign_stamp(records):
        records[0]["state_type"] = "established_resonance"

    path = _variant_config_path(scratch_dir, "foreign_state_stamp.json",
                                config, foreign_stamp)
    try:
        with pytest.raises(ValueError) as excinfo:
            families_from_json(path)
        message = str(excinfo.value)
    finally:
        Path(path).unlink()
    assert CONTINUUM_STATE_TYPE in message and "state_type" in message, (
        f"foreign-stamp failure must name the required continuum stamp, "
        f"got: {message}"
    )
    # The failure is not the missing-key variant (the stamp is present,
    # just foreign).
    assert "missing required keys" not in message


def test_loader_rejects_m_max_below_m_min(config, scratch_dir):
    """A recorded ``m_max`` below ``m_min`` fails with the named
    support-upper-end message (``m_max`` must equal the pre-registered
    5.0, so an inverted support cannot load)."""

    def invert_support(records):
        records[0]["m_max"] = records[0]["m_min"] - 0.1

    path = _variant_config_path(scratch_dir, "m_max_below_m_min.json",
                                config, invert_support)
    try:
        with pytest.raises(ValueError) as excinfo:
            families_from_json(path)
        message = str(excinfo.value)
    finally:
        Path(path).unlink()
    assert "m_max" in message, (
        f"m_max failure must name the offending field, got: {message}"
    )
    assert "does not equal the pre-registered support upper end" in message, (
        f"inverted-support failure must carry the named support-upper-end "
        f"message, got: {message}"
    )


def test_loader_rejects_mismatched_n_norm(config, scratch_dir):
    """A recorded ``n_norm`` displaced beyond the module's recorder
    tolerance (relative 1e-9 vs the module's relative 1e-12) fails with
    the named does-not-reproduce-the-unit-normalization message."""

    def displace_norm(records):
        records[0]["n_norm"] = records[0]["n_norm"] * (1.0 + 1e-9)

    path = _variant_config_path(scratch_dir, "mismatched_n_norm.json",
                                config, displace_norm)
    try:
        with pytest.raises(ValueError) as excinfo:
            families_from_json(path)
        message = str(excinfo.value)
    finally:
        Path(path).unlink()
    assert "n_norm" in message, (
        f"n_norm failure must name the recorded field, got: {message}"
    )
    assert "does not reproduce the unit normalization" in message, (
        f"n_norm failure must carry the named unit-normalization message, "
        f"got: {message}"
    )


# ----------------------------------------------------------------------
# Draw contract audit
# ----------------------------------------------------------------------
class _CountingGenerator:
    """A proxy ``np.random.Generator`` wrapper counting uniform draws.

    Delegates to a real ``np.random.default_rng(seed)`` bit stream and
    counts every ``random()`` call, so the per-draw uniform contract of
    ``family_draw`` is auditable on a genuine generator."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._generator.random())


def test_draw_contract_two_uniforms_per_draw_support_and_repro(
    families, config
):
    """The draw contract on 10,000 draws per family: exactly two
    uniforms per draw (checked at cumulative checkpoints 1, 100, 1,000,
    and 10,000, so a hidden rejection loop cannot hide in the total),
    every draw in ``[m_min, m_max]``, and bit-identical reproducibility
    at the fixed probe seed across two independent runs."""
    n_total = 10_000
    checkpoints = {1, 100, 1_000, n_total}
    records = {r["family_id"]: r for r in config["families"]}
    # Card-5 accommodation: the declared-inactive families are never
    # sampled (the active flag controls the draw), so the draw contract
    # runs over the active families.
    sampleable = [f for f in families if f.active]
    for family in sampleable:
        m_min = float(records[family.family_id]["m_min"])
        m_max = float(records[family.family_id]["m_max"])

        # Per-draw uniform count at the checkpoints, on the proxy
        # counting wrapper.
        proxy = _CountingGenerator(PROBE_SEED)
        draws = []
        for i in range(n_total):
            draws.append(family_draw(family, proxy))  # type: ignore[arg-type]
            if (i + 1) in checkpoints:
                assert proxy.calls == 2 * (i + 1), (
                    f"family {family.family_id}: after {i + 1} draws the "
                    f"proxy consumed {proxy.calls} uniforms, expected "
                    f"{2 * (i + 1)} (exactly two per draw)"
                )
        assert proxy.calls == 2 * n_total

        # Support: every draw in [m_min, m_max].
        assert all(m_min <= m <= m_max for m in draws), (
            f"family {family.family_id}: a draw left "
            f"[{m_min}, {m_max}]"
        )

        # Bit-identical reproducibility at the fixed probe seed: two
        # independent runs on the same bit stream (one fresh generator
        # per run, created before the draw loop).
        runs = []
        for _ in range(2):
            stream = np.random.default_rng(PROBE_SEED)
            runs.append([
                family_draw(family, stream) for _ in range(n_total)
            ])
        assert runs[0] == runs[1], (
            f"family {family.family_id}: draws at seed {PROBE_SEED} are "
            "not reproducible"
        )
        assert runs[0] == draws, (
            f"family {family.family_id}: the proxy draw stream diverged "
            "from the plain-generator stream at the same seed"
        )


# ----------------------------------------------------------------------
# Inverse-CDF consistency
# ----------------------------------------------------------------------
class _ScriptedGenerator:
    """A two-draw probe: returns exactly the scripted uniform values."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self.calls = 0

    def random(self) -> float:
        value = self._values[self.calls]
        self.calls += 1
        return value


def _pl_cdf_root(
    y0: float, y1: float, dx: float, target_mass: float
) -> float:
    """The exact fraction ``t`` in [0, 1] along a cell such that the
    piecewise-linear grid CDF has accumulated *target_mass* from the
    cell's lower node.

    The cell integral from the lower node at fraction ``t`` is
    ``y0 * t * dx + (y1 - y0) * t^2 * dx / 2``; solving the quadratic
    for the in-cell root (valid for rising and falling cells alike).
    """
    if y1 == y0:
        return target_mass / (y0 * dx)
    discriminant = y0 * y0 + 2.0 * (y1 - y0) * target_mass / dx
    return (math.sqrt(discriminant) - y0) / (y1 - y0)


def test_inverse_cdf_consistency_at_fixed_quantiles(families, config):
    """At fixed quantiles, the drawn mass agrees with the grid CDF to
    grid-resolution precision.

    For each quantile ``q`` the module-convention mass is the cell
    lower node plus the in-cell fraction times the cell width (the
    public ``family_draw`` output for the scripted probe
    ``(u_segment = q, u_within = f)``, with the module's internal CDF
    table read from ``spectral_density._draw_table``).  The exact
    piecewise-linear grid CDF root of ``q`` is solved independently
    (per-cell quadratic).  Documented grid-resolution bounds: the mass
    difference is at most one grid cell, i.e. ``(m_max - m_min) /
    10,000`` = 1/10,000 of the support span (about 2.32e-4 GeV), and the
    quantile difference is at most the CDF mass of the cell containing
    the quantile (the cell's share of the total, about 5e-4 here).
    """
    records = {r["family_id"]: r for r in config["families"]}
    # Card-5 accommodation: the declared-inactive families are never
    # sampled; the inverse-CDF consistency runs over the active
    # families.
    sampleable = [f for f in families if f.active]
    for family in sampleable:
        m_min = float(records[family.family_id]["m_min"])
        m_max = float(records[family.family_id]["m_max"])
        lo, hi, cdf, cell_width = sd._draw_table(family)
        assert lo == m_min and hi == m_max

        xs = np.linspace(m_min, m_max, REFERENCE_GRID_POINTS)
        ys = np.array([family_density(family, float(x)) for x in xs])
        dx = xs[1] - xs[0]
        cell_masses = 0.5 * (ys[1:] + ys[:-1]) * dx
        total = float(cell_masses.sum())
        assert total > 0.0 and math.isfinite(total)
        # The grid-resolution bound in mass units: one cell.
        grid_bound_mass = (m_max - m_min) / (REFERENCE_GRID_POINTS - 1)

        def pl_cdf_at(mass: float) -> float:
            """The piecewise-linear grid CDF at *mass* (independent
            re-evaluation on the same grid)."""
            i = int(np.searchsorted(xs, mass, side="right")) - 1
            t = (mass - xs[i]) / dx
            y0, y1 = ys[i], ys[i + 1]
            below = float(cdf[i - 1]) * total if i > 0 else 0.0
            return (
                below
                + y0 * t * dx
                + (y1 - y0) * t * t * dx / 2.0
            ) / total

        for q in QUANTILES:
            idx = int(np.searchsorted(cdf, q, side="right"))
            assert 1 <= idx < cdf.size, f"quantile {q} left the grid"
            cell_prev = float(cdf[idx - 1])
            cell_cdf_mass = float(cdf[idx] - cdf[idx - 1])
            fraction = (q - cell_prev) / cell_cdf_mass

            # Module-convention mass, pinned through the public draw
            # with a scripted two-uniform probe.  The expected value
            # reproduces the module's own association order
            # (``lo + (idx + u_within) * cell_width``), so the equality
            # is bit-exact.
            probe = _ScriptedGenerator([q, fraction])
            m_mod = family_draw(family, probe)  # type: ignore[arg-type]
            assert probe.calls == 2
            expected = float(lo + (idx + fraction) * cell_width)
            assert m_mod == expected, (
                f"family {family.family_id}: the scripted probe draw "
                f"{m_mod!r} is not the module-convention mass "
                f"{expected!r} at quantile {q}"
            )

            # Exact grid-CDF root of the quantile, solved independently.
            t_root = _pl_cdf_root(
                float(ys[idx]), float(ys[idx + 1]), dx,
                (q - cell_prev) * total,
            )
            m_root = float(xs[idx] + t_root * dx)
            assert 0.0 <= t_root <= 1.0

            # The root reproduces the quantile (independent CDF sanity).
            assert abs(pl_cdf_at(m_root) - q) <= 1e-12

            # Grid-resolution bounds (documented above).
            assert abs(m_mod - m_root) <= grid_bound_mass + 1e-15, (
                f"family {family.family_id}: quantile {q}: the "
                f"module-convention mass {m_mod!r} misses the grid-CDF "
                f"root {m_root!r} by more than one grid cell "
                f"({grid_bound_mass!r})"
            )
            assert abs(pl_cdf_at(m_mod) - q) <= cell_cdf_mass, (
                f"family {family.family_id}: quantile {q}: the CDF of the "
                f"module-convention mass misses {q} by more than the "
                f"cell's own CDF mass ({cell_cdf_mass!r})"
            )


# ----------------------------------------------------------------------
# Grid consistency with the Round 2 reference quadrature
# ----------------------------------------------------------------------
def test_grid_matches_round2_reference_quadrature(families, config):
    """The module's 10,001-node grid node masses match the Round 2
    reference quadrature.

    The reference quadrature is the fixed 10,001-node count
    (``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay``;
    the card-3 width law normalizes on the same count via
    ``_NORMALIZATION_GRID_POINTS`` of
    ``na49_mixing.resonance_widths``) realized as the uniform linspace
    over the support.  The module grid must instantiate exactly that
    construction over ``[m_min, m_max]``: same node count, the same
    node masses (the linspace values), the endpoints at the config's
    support ends.
    """
    assert (
        REFERENCE_GRID_POINTS == SPECTRAL_QUADRATURE_NODES
        == _NORMALIZATION_GRID_POINTS == 10_001
    ), (
        "the spectral-density grid must be the Round 2 reference "
        f"quadrature: {REFERENCE_GRID_POINTS} vs "
        f"{SPECTRAL_QUADRATURE_NODES} vs {_NORMALIZATION_GRID_POINTS}"
    )
    records = {r["family_id"]: r for r in config["families"]}
    for family in families:
        m_min = float(records[family.family_id]["m_min"])
        m_max = float(records[family.family_id]["m_max"])
        lo, hi, cdf, cell_width = sd._draw_table(family)
        assert lo == m_min
        assert hi == m_max
        assert cdf.size == REFERENCE_GRID_POINTS - 1
        # The module's grid node masses, reconstructed from the draw
        # table.
        nodes = lo + np.arange(REFERENCE_GRID_POINTS) * cell_width
        # The Round 2 reference quadrature over the family support: the
        # card-3 width law's node-mass construction (uniform 10,001-node
        # linspace over the support).
        reference = np.linspace(m_min, m_max, SPECTRAL_QUADRATURE_NODES)
        assert nodes[0] == m_min
        assert nodes[-1] == m_max
        assert np.array_equal(nodes, reference), (
            f"family {family.family_id}: the module grid node masses "
            "differ from the Round 2 reference quadrature node masses"
        )


# ----------------------------------------------------------------------
# Config integrity
# ----------------------------------------------------------------------
def test_config_integrity_pre_registered_values_and_classes(config):
    """The committed config opens with the two baryonic families, both
    active, with the pre-registered values and exactly the card's
    8 + 6 class pairs, and its catalog reference digest matches the
    committed catalog file.  (Card-5 accommodation: the card-5 mesonic
    families follow the baryonic section in the families array; their
    pins live in tests/test_spectral_density_mesonic.py.)"""
    assert config["schema_version"] == 1
    records = config["families"]
    assert [r["family_id"] for r in records] == [
        "baryonic_np",
        "baryonic_strange",
        "mesonic_neutral",
        "mesonic_charged",
        "mesonic_anticharged",
        "mesonic_strange",
        "mesonic_antistrange",
    ], "the baryonic section must open the families array, followed by the card-5 mesonic families"
    baryonic_records = records[:2]
    for record in baryonic_records:
        assert record["active"] is True, (
            f"family {record['family_id']} must be active"
        )
        assert record["state_type"] == CONTINUUM_STATE_TYPE
        for field, value in PRE_REGISTERED.items():
            assert record[field] == value, (
                f"family {record['family_id']}: pre-registered "
                f"{field} = {value!r}, recorded {record[field]!r}"
            )
    # The one common baryonic continuum scale, shared verbatim.
    assert records[0]["t_h"] == records[1]["t_h"] == PRE_REGISTERED["t_h"]
    # Exactly the card's named class pairs, in card order.
    for record in baryonic_records:
        expected = list(EXPECTED_CLASS_PAIRS[record["family_id"]])
        assert record["allowed_decay_classes"] == expected, (
            f"family {record['family_id']}: the recorded class pairs "
            f"{record['allowed_decay_classes']!r} are not the card's "
            f"named set {expected!r}"
        )
    # The recorded catalog reference matches the committed catalog.
    catalog_digest = hashlib.sha256(
        CATALOG_PATH.read_bytes()
    ).hexdigest()
    reference = config["catalog_reference"]
    assert reference["path"] == "config/v2_singlet/resonance_catalog.json"
    assert reference["sha256"] == catalog_digest, (
        f"the config's catalog_reference sha256 "
        f"{reference['sha256']!r} does not match the committed catalog "
        f"({catalog_digest!r})"
    )


def _class_daughters(class_id: str) -> tuple[int, int]:
    """Parse one multiplet-class identifier into its daughter pair."""
    match = _CLASS_ID_PATTERN.fullmatch(class_id)
    assert match is not None, (
        f"class identifier {class_id!r} is not a catalog multiplet-class "
        "key '(daughter0,daughter1)'"
    )
    return int(match.group(1)), int(match.group(2))


def test_class_pairs_conserve_family_qn_independent_catalog_audit(
    config, catalog_raw
):
    """Every class pair of both baryonic families conserves the family
    (B, Q, S), audited against the committed catalog read as raw JSON
    (independent of the ``HadronCatalog`` layer the card test uses).
    (Card-5 accommodation: the mesonic families' conserving classes are
    audited in tests/test_spectral_density_mesonic.py.)"""
    records_by_pdg = {r["pdg_id"]: r for r in catalog_raw["records"]}
    baryonic_records = config["families"][:2]
    for record in baryonic_records:
        family_qn = tuple(record["qn"])
        assert record["allowed_decay_classes"] == list(
            EXPECTED_CLASS_PAIRS[record["family_id"]]
        )
        for class_id in record["allowed_decay_classes"]:
            daughter0, daughter1 = _class_daughters(class_id)
            assert daughter0 in records_by_pdg, (
                f"family {record['family_id']}: daughter {daughter0} of "
                f"class {class_id} is not a catalog record"
            )
            assert daughter1 in records_by_pdg, (
                f"family {record['family_id']}: daughter {daughter1} of "
                f"class {class_id} is not a catalog record"
            )
            rec0 = records_by_pdg[daughter0]
            rec1 = records_by_pdg[daughter1]
            vertex = (
                int(rec0["B"]) + int(rec1["B"]),
                int(rec0["Q"]) + int(rec1["Q"]),
                int(rec0["S"]) + int(rec1["S"]),
            )
            assert vertex == family_qn, (
                f"family {record['family_id']}: class {class_id} "
                f"({rec0['name']} + {rec1['name']}) carries vertex "
                f"(B, Q, S) = {vertex}, family requires {family_qn}"
            )


# ----------------------------------------------------------------------
# Edge behavior
# ----------------------------------------------------------------------
def test_density_edge_values_and_grid_finiteness(families, config):
    """The density at the edge masses and grid-wide finiteness.

    Exactly 0 at ``m_min`` (no continuum strength below the
    catalog-completeness boundary, plan section 9.3), exactly the
    ramp-top value at ``m_min + ramp_width`` (the ramp is exactly 1
    from there on), and exactly the constant-branch value at
    ``m_max``.  Expected values are recomputed from the raw config
    values (not the loaded record).  No NaN/inf appears anywhere on
    the full 10,001-node grid for either family.
    """
    records = {r["family_id"]: r for r in config["families"]}
    for family in families:
        record = records[family.family_id]
        m_min = float(record["m_min"])
        m_max = float(record["m_max"])
        t_h = float(record["t_h"])
        b_power = float(record["b_power"])
        n_norm = float(record["n_norm"])

        # Edge masses.
        assert family_density(family, m_min) == 0.0
        junction = m_min + float(record["ramp_width"])
        ramp_top = n_norm * junction ** (-b_power) * math.exp(
            junction / t_h
        )
        assert family_density(family, junction) == ramp_top, (
            f"family {family.family_id}: the density at the ramp-top "
            f"mass {junction} is {family_density(family, junction)!r}, "
            f"expected {ramp_top!r} (the ramp is exactly 1 there)"
        )
        constant_branch = n_norm * m_max ** (-b_power) * math.exp(
            m_max / t_h
        )
        assert family_density(family, m_max) == constant_branch

        # Grid-wide finiteness: every node value finite and
        # non-negative.
        for x in np.linspace(m_min, m_max, REFERENCE_GRID_POINTS):
            value = family_density(family, float(x))
            assert math.isfinite(value), (
                f"family {family.family_id}: non-finite density "
                f"{value!r} at {float(x)!r}"
            )
            assert value >= 0.0, (
                f"family {family.family_id}: negative density "
                f"{value!r} at {float(x)!r}"
            )
