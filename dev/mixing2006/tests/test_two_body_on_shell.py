"""Two-body on-shell repair tests (Round 1, card
TASK-20260911-r1-two-body-on-shell-repair).

Gate 0 off-shell clause (plan section 6): no two-body daughter may be
off shell beyond a documented float tolerance.  The repaired
``two_body_split_safe`` contract:

- every emitted daughter satisfies ``m_i^2 = p_i · p_i`` equal to the
  configured ``mass_i`` to within the relative tolerance
  ``1e-10 * max(mass_i, 1.0)`` (the same semantics as
  ``ExcitationEvent._validate_pdg_on_shell``);
- a mass draw with ``mass1 + mass2`` above the parent beyond
  ``two_body_threshold_tolerance`` raises ``TwoBodyBelowThreshold`` so
  the staged builders reject and retry; the mass-moving clip and the
  at-rest fallback branch are removed;
- an overshoot inside the tolerance is emitted with zero relative
  momentum at the threshold point and the daughters keep their
  configured masses.

The card's five deterministic tests and the statistical rejection-rate
sanity test live here.  The M8 configuration hashes pinned in
``test_config_hash_unchanged`` are the legacy_v1 freeze record (card 1,
nested commit ``17a01c8``): they are the "before" values of the repair,
so the assertion is a genuine before/after comparison and not a
re-read of the re-pinned manifest.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.continuum_channels import (
    ContinuumParameters,
    SourceDecayer,
)
import na49_mixing.continuum_channels as continuum_channels_module
from na49_mixing.excitation_kinematics import (
    FourVector,
    TwoBodyBelowThreshold,
    audit_two_body_sites,
    two_body_split_safe,
    two_body_threshold_tolerance,
)
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.source_event import SourceRecord

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # monorepo root

M_PI = get_mass(211)
M_PI0 = get_mass(111)
M_P = get_mass(2212)

# On-shell tolerance: same semantics as
# ExcitationEvent._validate_pdg_on_shell (na49_mixing/source_event.py).
ON_SHELL_REL_TOL = 1e-10

# Statistical test (pre-registered in the card): 1e6 seeded H3 mass
# draws, rejection fraction for below-threshold draws < 1e-2.
STATISTICAL_SEED = 20260911
STATISTICAL_DRAWS = 1_000_000
STATISTICAL_PARENTS = (1.9, 2.2, 2.6, 3.0, 3.5, 4.1, 5.0, 6.0)
REJECTION_FRACTION_BOUND = 1e-2

# legacy_v1 freeze record (card 1, nested commit 17a01c8): staged config
# digests and the canonical M8 parameter-map digest at freeze time.
CARD1_CONFIG_SHA256 = {
    "dev/mixing2006/config/staged_excitation.json": (
        "154e41028fc68d63daf74af73ab812789ae11915fcdabb3b1a6d32deabf6693f"
    ),
    "dev/mixing2006/config/staged_excitation_smoke.json": (
        "7d36656a128a42da9f0c2342680befe6c81b500c432df97be689d61131881035"
    ),
}
CARD1_M8_PARAMETERS_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_on_shell(pair: tuple[FourVector, FourVector], mass1: float, mass2: float) -> None:
    """Every daughter reconstructed on its configured mass shell."""
    a, b = pair
    assert abs(a.mass() - mass1) <= ON_SHELL_REL_TOL * max(mass1, 1.0)
    assert abs(b.mass() - mass2) <= ON_SHELL_REL_TOL * max(mass2, 1.0)


def _assert_parent_conservation(
    pair: tuple[FourVector, FourVector], parent_mass: float, tolerance: float = 1e-10
) -> None:
    a, b = pair
    total = a + b
    assert abs(total.e - parent_mass) <= tolerance
    assert abs(total.px) <= tolerance
    assert abs(total.py) <= tolerance
    assert abs(total.pz) <= tolerance


# ---------------------------------------------------------------------------
# Deterministic tests (card)
# ---------------------------------------------------------------------------


def test_two_body_split_on_shell_grid():
    """Fixed grid: near-threshold, threshold, well-above; on-shell + conservation."""
    # (parent_mass, mass1, mass2); the threshold-adjacent rows are built
    # from the exact float sums of the daughter masses.
    pairs = [
        (M_PI, M_PI0),
        (M_P, M_PI),
        (M_P, M_P),
        (M_PI, M_PI),
        (1.2, 0.7),
    ]
    grid: list[tuple[float, float, float]] = [
        # well above threshold
        (3.0, M_PI, M_PI0),
        (3.0, M_P, M_PI),
        (30.0, M_P, M_P),
        (1.3, M_P, M_PI),
        (1.232, M_P, M_PI0),
        (2.0, M_PI, M_PI),
        (2.0, 1.2, 0.7),
        (5.5, 2.4, 1.9),
        (2.0 * (M_P + M_PI), M_P, M_PI),
    ]
    for m1, m2 in pairs:
        sum_float = m1 + m2
        grid.extend(
            [
                # threshold point: m1 + m2 == parent, zero relative momentum
                (sum_float, m1, m2),
                # one ulp above the daughters' float sum: just above
                # threshold (m1 + m2 < parent), moving split with tiny p*
                (float(np.nextafter(sum_float, np.inf)), m1, m2),
                # just above threshold: moving split with tiny p*
                (sum_float + 1e-12, m1, m2),
                (sum_float + 1e-9, m1, m2),
                (sum_float + 1e-6, m1, m2),
                # within-tolerance overshoot (m1 + m2 above the parent):
                # still allowed, threshold point with zero p*
                (sum_float - 1e-15, m1, m2),
            ]
        )

    for parent_mass, mass1, mass2 in grid:
        rng = np.random.default_rng(hash((parent_mass, mass1, mass2)) % (2**32))
        excess = (mass1 + mass2) - parent_mass
        tolerance = two_body_threshold_tolerance(parent_mass, mass1, mass2)
        assert excess <= tolerance, f"grid row must be legal: {parent_mass, mass1, mass2}"
        pair = two_body_split_safe(parent_mass, mass1, mass2, rng)
        _assert_on_shell(pair, mass1, mass2)
        _assert_parent_conservation(pair, parent_mass)
        if excess >= 0.0:
            # threshold draw: zero relative momentum, configured masses
            a, b = pair
            assert float(np.linalg.norm(a.spatial())) == 0.0
            assert float(np.linalg.norm(b.spatial())) == 0.0
            assert a.e == mass1
            assert b.e == mass2
        else:
            # above-threshold draw (m1 + m2 < parent): back-to-back moving split
            a, b = pair
            assert a.px + b.px == 0.0
            assert a.py + b.py == 0.0
            assert a.pz + b.pz == 0.0
            assert a.e >= mass1
            assert b.e >= mass2


def test_below_threshold_rejects():
    """Below-threshold draws beyond the float tolerance raise the new exception."""
    assert issubclass(TwoBodyBelowThreshold, ValueError)
    with pytest.raises(TwoBodyBelowThreshold):
        two_body_split_safe(1.0, 0.8, 0.4, np.random.default_rng(1))
    # The staged builders' retry convention catches ValueError; the new
    # exception keeps that convention working.
    with pytest.raises(ValueError):
        two_body_split_safe(1.0, 0.8, 0.4, np.random.default_rng(2))
    # A draw one nano-unit below the daughters' float sum is a physical
    # threshold failure, not roundoff.
    m1, m2 = M_PI, M_PI0
    with pytest.raises(TwoBodyBelowThreshold, match="two-body split below threshold"):
        two_body_split_safe(m1 + m2 - 1e-9, m1, m2, np.random.default_rng(3))
    # Source inspection: no off-shell emission path remains.  The
    # mass-moving clip and the at-rest fallback branch are absent.
    source = inspect.getsource(two_body_split_safe)
    assert "parent_mass - mass1" not in source
    assert "FourVector(parent_mass" not in source


def test_threshold_point():
    """At exact threshold the daughters are at rest with configured masses."""
    m1, m2 = M_P, M_PI
    parent_mass = m1 + m2  # the exact float sum of the configured masses
    a, b = two_body_split_safe(parent_mass, m1, m2, np.random.default_rng(0))
    assert a.e == m1
    assert b.e == m2
    assert a.px == a.py == a.pz == 0.0
    assert b.px == b.py == b.pz == 0.0
    # Zero relative momentum: the spatial sum vanishes exactly.
    assert np.array_equal(a.spatial() + b.spatial(), np.zeros(3))
    _assert_parent_conservation((a, b), parent_mass)

    # A within-tolerance overshoot (one ulp below the daughters' float
    # sum) is a threshold draw: configured masses, at rest.
    m2_over = float(np.nextafter(m2, np.inf))
    sum_over = m1 + m2_over
    parent_below = float(np.nextafter(sum_over, -np.inf))
    excess = (m1 + m2_over) - parent_below
    assert 0.0 < excess <= two_body_threshold_tolerance(parent_below, m1, m2_over)
    a2, b2 = two_body_split_safe(parent_below, m1, m2_over, np.random.default_rng(1))
    assert a2.e == m1
    assert b2.e == m2_over
    assert float(np.linalg.norm(a2.spatial())) == 0.0
    assert float(np.linalg.norm(b2.spatial())) == 0.0


def test_audit_inventory_complete():
    """The pre-registered site inventory is pinned as of card execution."""
    # Pinned at card execution (2026-09-11): (module, function, site_kind,
    # on_shell, action) in audit order.  A new 4-vector construction site
    # in an audited function makes audit_two_body_sites() raise; a
    # re-classified site breaks this list.
    expected = [
        ("na49_mixing.excitation_kinematics", "two_body_split_rest_frame", "staged", "by_construction", "unchanged"),
        ("na49_mixing.excitation_kinematics", "two_body_split_rest_frame", "staged", "by_construction", "unchanged"),
        ("na49_mixing.excitation_kinematics", "two_body_split_safe", "staged", "rejection", "repaired"),
        ("na49_mixing.excitation_kinematics", "two_body_split_safe", "staged", "by_construction", "repaired"),
        ("na49_mixing.excitation_kinematics", "two_body_split_safe", "staged", "by_construction", "repaired"),
        ("na49_mixing.excitation_kinematics", "scatter_two_body", "staged", "by_construction", "unchanged"),
        ("na49_mixing.excitation_kinematics", "scatter_two_body", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_rest_frame", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_rest_frame", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_rest_frame", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_phsp", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_phsp", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_phsp", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_phsp", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_decay_resonant", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_h2_resonance_mass", "staged", "by_construction", "documented"),
        ("na49_mixing.continuum_channels", "_h2_resonance_mass", "staged", "by_construction", "documented"),
        ("na49_mixing.continuum_channels", "_h2_emit", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_h2_cascade", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_h2_cascade", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_h2_cascade", "staged", "by_construction", "unchanged"),
        ("na49_mixing.continuum_channels", "_h2_cascade", "staged", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_2body", "legacy", "rejection", "unchanged"),
        ("na49_mixing.event", "_decay_2body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_2body", "legacy", "rejection", "unchanged"),
        ("na49_mixing.event", "_decay_2body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_2body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "rejection", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
        ("na49_mixing.event", "_decay_3body", "legacy", "by_construction", "unchanged"),
    ]
    sites = audit_two_body_sites()
    got = [
        (site["module"], site["function"], site["site_kind"], site["on_shell"], site["action"])
        for site in sites
    ]
    assert got == expected
    assert all(site["documented"] is True for site in sites)
    # No site may keep the removed clip/at-rest fallback classification.
    assert all(site["on_shell"] != "clip_fallback" for site in sites)
    # Every reported line resolves inside the registered function body.
    for site in sites:
        module = __import__(site["module"], fromlist=["_"])
        owner = getattr(module, site["function"], None)
        if owner is None:
            for value in vars(module).values():
                if inspect.isclass(value) and site["function"] in vars(value):
                    owner = vars(value)[site["function"]]
                    break
        assert owner is not None, site
        source = inspect.getsource(owner)
        first = owner.__code__.co_firstlineno
        assert first <= site["line"] <= first + len(source.splitlines())


def test_config_hash_unchanged():
    """The staged config JSONs are bit-identical before and after the repair."""
    for rel_path, expected in CARD1_CONFIG_SHA256.items():
        config_path = MONO_ROOT / rel_path
        assert config_path.is_file(), f"staged config missing: {rel_path}"
        assert _sha256_file(config_path) == expected, (
            f"staged config drifted from the legacy_v1 freeze record: {rel_path}"
        )
    # The frozen M8 parameter set stays bit-exact (canonical-JSON digest,
    # same convention as the card-1 freeze report).
    manifest = json.loads(
        (NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    canonical = json.dumps(m8, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == CARD1_M8_PARAMETERS_HASH


# ---------------------------------------------------------------------------
# Statistical test (card): rejection-rate sanity
# ---------------------------------------------------------------------------


def _h3_mass_draws(n_draws: int, seed: int, parents: tuple[float, ...]):
    """Run the H3 kernel mass draws against fixed parents.

    The decay kinematics are stubbed at the module boundary so the loop
    measures exactly the H3 mass draw (channel family, extra pions, mass
    budget clamp, daughter permutation) that feeds
    ``_decay_rest_frame``; the stub records every drawn mass set.
    """
    rng = np.random.default_rng(seed)
    decayer = SourceDecayer(ContinuumParameters(decay_hypothesis="H3"), rng)
    drawn: list[tuple[float, tuple[float, ...]]] = []

    def _record_masses(parent_mass, masses, rng_):
        drawn.append((parent_mass, tuple(masses)))
        return []

    original = continuum_channels_module._decay_rest_frame
    continuum_channels_module._decay_rest_frame = _record_masses
    try:
        per_parent = n_draws // len(parents)
        for mass in parents:
            source = SourceRecord(
                source_id=0,
                role="excitation",
                p4=FourVector(mass, 0.0, 0.0, 0.0),
                quantum_numbers=QuantumNumbers(1, 1, 0),
            )
            for _ in range(per_parent):
                decayer.decay(source, 0)
    finally:
        continuum_channels_module._decay_rest_frame = original
    return drawn


def test_rejection_rate_sanity():
    """1e6 seeded H3 mass draws: the below-threshold rejection fraction
    must be finite and below the pre-registered 1e-2 bound."""
    drawn = _h3_mass_draws(STATISTICAL_DRAWS, STATISTICAL_SEED, STATISTICAL_PARENTS)
    assert len(drawn) == STATISTICAL_DRAWS
    # The H3 kernel's own threshold guard: a drawn mass set above the
    # parent beyond the kernel's 1e-12 tolerance is a below-threshold
    # rejection (TwoBodyBelowThreshold after the repair).
    below = sum(
        1
        for parent_mass, masses in drawn
        if sum(masses) > parent_mass + 1e-12
    )
    fraction = below / len(drawn)
    assert math.isfinite(fraction)
    assert 0.0 <= fraction < REJECTION_FRACTION_BOUND, (
        f"below-threshold rejection fraction {fraction:.6g} exceeds the "
        f"pre-registered bound {REJECTION_FRACTION_BOUND:g} "
        f"({below} of {len(drawn)} draws)"
    )