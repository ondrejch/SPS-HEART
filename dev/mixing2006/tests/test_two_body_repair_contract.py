"""Contract extensions of the two-body on-shell repair (Round 1, card
TASK-20260911-r1-two-body-on-shell-repair, nested commit ``8577213``).

Companion to ``tests/test_two_body_on_shell.py``: the card's suite pins
the repaired ``two_body_split_safe`` contract on a fixed grid, the
36-site audit inventory, the config/M8 hashes, and the 1e6-draw H3
rejection fraction.  This file pins the committed behavior that the
card's suite does not exercise, at grid/scale sizes only:

- exact-threshold determinism: identical (and seed-independent)
  daughter four-momentums at the threshold point, and no isotropic-angle
  random numbers consumed (the documented RNG-stream change);
- the documented tolerance band boundary: an excess exactly equal to
  ``two_body_threshold_tolerance`` is emitted at the threshold point,
  and an excess one ulp beyond it raises ``TwoBodyBelowThreshold``
  (the contract rejects on a strict ``excess > tolerance``);
- the staged builders' retry convention: a rejected
  ``TwoBodyBelowThreshold`` draw is a counted rejection (``PPEventBuilder``
  attempt bookkeeping, ``PAEventBuilder.failure_reason_counts``) and the
  fresh draw produces a valid event whose daughters are all on their
  configured mass shells;
- the H1 recursion path: ``_decay_phsp`` routes its two-body splits
  through the repaired ``two_body_split_safe``, including the
  near-threshold terminal split of a three-body recursion and the
  exact-threshold two-body draw;
- audit robustness: a vanished 4-vector construction site, a new
  unclaimed construction line, and a vanished audited function each
  fail ``audit_two_body_sites()``.  The probes serve mutated source
  lines from a temporary ``linecache`` entry (a temp copy of the file
  lines in memory); the committed tree is never modified, and the
  audit passes again once the probe is removed.
"""

from __future__ import annotations

import contextlib
import inspect
import linecache
import math
import os
import re

import numpy as np
import pytest

import na49_mixing.continuum_channels as continuum_channels_module
import na49_mixing.excitation_kinematics as excitation_kinematics_module
from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import (
    FourVector,
    TwoBodyBelowThreshold,
    audit_two_body_sites,
    two_body_split_safe,
    two_body_threshold_tolerance,
)
from na49_mixing.glauber import NucleusSpec
from na49_mixing.pdg_database import get_mass

M_PI = get_mass(211)
M_PI0 = get_mass(111)
M_P = get_mass(2212)

# On-shell tolerance: same semantics as
# ExcitationEvent._validate_pdg_on_shell (na49_mixing/source_event.py).
ON_SHELL_REL_TOL = 1e-10

# Pinned seeds (test-scale runs only; the 1e6-draw statistical test
# stays in tests/test_two_body_on_shell.py).
BUILDER_SEED = 20260911
H1_SEED = 23

# The repair report's full 36-site inventory (nested commit 8577213).
AUDIT_SITE_COUNT = 36


def _assert_on_shell(daughters: list[FourVector], masses: list[float]) -> None:
    """Every daughter reconstructed on its configured mass shell."""
    assert len(daughters) == len(masses)
    for vector, mass in zip(daughters, masses):
        assert abs(vector.mass() - mass) <= ON_SHELL_REL_TOL * max(mass, 1.0)


def _assert_sum_conservation(
    daughters: list[FourVector], parent_mass: float, tolerance: float = 1e-10
) -> None:
    total = daughters[0]
    for vector in daughters[1:]:
        total = total + vector
    assert abs(total.e - parent_mass) <= tolerance
    assert abs(total.px) <= tolerance
    assert abs(total.py) <= tolerance
    assert abs(total.pz) <= tolerance


# ---------------------------------------------------------------------------
# Committed behavior 2: exact-threshold determinism and RNG-stream change
# ---------------------------------------------------------------------------


def test_exact_threshold_is_seed_independent_and_draws_no_angles():
    """The threshold point is bit-identical across seeds and consumes no rng."""
    m1, m2 = M_P, M_PI
    parent_mass = m1 + m2  # the exact float sum: excess == 0.0
    tuples = set()
    for seed in (1, 42, 98765):
        a, b = two_body_split_safe(parent_mass, m1, m2, np.random.default_rng(seed))
        assert a == FourVector(m1, 0.0, 0.0, 0.0)
        assert b == FourVector(m2, 0.0, 0.0, 0.0)
        tuples.add((a.e, a.px, a.py, a.pz, b.e, b.px, b.py, b.pz))
    assert len(tuples) == 1  # bit-identical daughters for every seed

    # The documented RNG-stream change: at the threshold point the two
    # isotropic-angle random numbers are no longer consumed, while the
    # kinematically allowed moving split of the same function (parent
    # above the daughters' mass sum) still draws both angles.
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    two_body_split_safe(parent_mass, m1, m2, rng)
    assert rng.bit_generator.state == before
    before = rng.bit_generator.state
    two_body_split_safe(parent_mass + 1e-6, m1, m2, rng)
    assert rng.bit_generator.state != before


# ---------------------------------------------------------------------------
# Committed behavior 1/2 boundary: the strict ``excess > tolerance`` band
# ---------------------------------------------------------------------------


def test_threshold_tolerance_band_boundary():
    """Excess exactly at the band edge is emitted; one ulp beyond is rejected."""
    m1, m2 = 1.4, 1.6
    daughters_sum = m1 + m2
    assert daughters_sum == 3.0  # a clean float sum; every row below is exact

    tol = two_body_threshold_tolerance(daughters_sum, m1, m2)
    assert tol == 8.0 * math.ulp(daughters_sum)
    assert tol > 1e-15 * daughters_sum  # the ulp branch dominates the band

    # excess == tol exactly: inside the band (the contract rejects only on
    # a strict ``excess > tol``), emitted at the threshold point with the
    # configured masses and no random numbers consumed.
    p_edge = daughters_sum - tol
    assert (m1 + m2) - p_edge == two_body_threshold_tolerance(p_edge, m1, m2) == tol
    rng = np.random.default_rng(3)
    state_before = rng.bit_generator.state
    a, b = two_body_split_safe(p_edge, m1, m2, rng)
    assert a == FourVector(m1, 0.0, 0.0, 0.0)
    assert b == FourVector(m2, 0.0, 0.0, 0.0)
    assert rng.bit_generator.state == state_before

    # Half the band inside: also the threshold point.
    p_half = daughters_sum - tol / 2.0
    a, b = two_body_split_safe(p_half, m1, m2, np.random.default_rng(4))
    assert a == FourVector(m1, 0.0, 0.0, 0.0)
    assert b == FourVector(m2, 0.0, 0.0, 0.0)

    # One ulp below the band edge: excess == tol + ulp > tol, a physical
    # threshold failure and a counted rejection.
    p_out = float(np.nextafter(p_edge, -np.inf))
    tol_out = two_body_threshold_tolerance(p_out, m1, m2)
    assert (m1 + m2) - p_out == tol + math.ulp(daughters_sum)
    assert (m1 + m2) - p_out > tol_out
    with pytest.raises(TwoBodyBelowThreshold):
        two_body_split_safe(p_out, m1, m2, np.random.default_rng(5))


# ---------------------------------------------------------------------------
# Committed behavior 3: staged-builder retry on TwoBodyBelowThreshold
# ---------------------------------------------------------------------------


def _once_rejecting_phsp(counter: dict[str, int]):
    """Patch ``_decay_phsp`` so its first call rejects below threshold.

    The first decay of the first attempt raises
    ``TwoBodyBelowThreshold`` (the repaired exception the event builder's
    ``ValueError`` retry convention must catch); every later call
    delegates to the real kernel, so the accepted event is a genuine
    on-shell one.
    """
    real_phsp = continuum_channels_module._decay_phsp

    def once_rejecting(parent_mass, masses, rng):
        counter["total"] += 1
        if counter["rejected"] == 0:
            counter["rejected"] = 1
            raise TwoBodyBelowThreshold("test-injected below-threshold mass draw")
        return real_phsp(parent_mass, masses, rng)

    return real_phsp, once_rejecting


def test_pp_builder_counts_below_threshold_retry_and_emits_on_shell_event():
    """A rejected draw is a counted attempt; the fresh draw is a valid event."""
    config = PPExcitationConfig(
        production_mode="simple",
        continuum=ContinuumParameters(decay_hypothesis="H1"),
        max_event_attempts=50,
    )
    builder = PPEventBuilder(config, np.random.default_rng(BUILDER_SEED))
    counter = {"total": 0, "rejected": 0}
    real_phsp, once_rejecting = _once_rejecting_phsp(counter)
    continuum_channels_module._decay_phsp = once_rejecting
    try:
        event = builder.generate(event_id=0)
    finally:
        continuum_channels_module._decay_phsp = real_phsp

    # First attempt: one source decay hit the injected rejection. Second
    # attempt: fresh source-mass draw, both sources decay for real.
    assert counter["rejected"] == 1
    assert counter["total"] == 3  # 1 rejected decay + 2 real decays
    assert event.metadata["generation_attempt"] == 2

    # Gate 0: no daughter is off its configured mass shell, and the
    # accepted event conserves the incoming four-momentum.
    _assert_on_shell(
        [particle.p4 for particle in event.final_particles],
        [get_mass(particle.pdg_id) for particle in event.final_particles],
    )
    incoming = event.incoming[0] + event.incoming[1]
    _assert_sum_conservation(
        [particle.p4 for particle in event.final_particles],
        incoming.e,
        tolerance=1e-10,
    )
    total = event.final_particles[0].p4
    for particle in event.final_particles[1:]:
        total = total + particle.p4
    assert abs(total.px - incoming.px) <= 1e-10
    assert abs(total.py - incoming.py) <= 1e-10
    assert abs(total.pz - incoming.pz) <= 1e-10


def test_pa_builder_counts_two_body_below_threshold_as_rejection_reason():
    """The p+A builder records the exception class as a counted rejection."""
    config = PAExcitationConfig(
        production_mode="simple",
        continuum=ContinuumParameters(decay_hypothesis="H1"),
        nucleus=NucleusSpec(name="p", A=1, Z=1),  # point target: no Glauber draws
        glauber_dr_fm=0.05,
        max_event_attempts=50,
    )
    builder = PAEventBuilder(config, np.random.default_rng(BUILDER_SEED))
    counter = {"total": 0, "rejected": 0}
    real_phsp, once_rejecting = _once_rejecting_phsp(counter)
    continuum_channels_module._decay_phsp = once_rejecting
    try:
        event = builder.generate(event_id=0)
    finally:
        continuum_channels_module._decay_phsp = real_phsp

    # The exception class name is the counted rejection reason of the
    # failed attempt; the accepted event is the second attempt.
    assert counter["rejected"] == 1
    assert builder.failure_reason_counts == {"TwoBodyBelowThreshold": 1}
    assert builder.generation_accepted == 1
    assert builder.accepted_generation_attempts == [2]
    assert event.metadata["generation_attempt"] == 2
    assert str(event.metadata["generation_last_failure_reason"]).startswith(
        "TwoBodyBelowThreshold"
    )
    _assert_on_shell(
        [particle.p4 for particle in event.final_particles],
        [get_mass(particle.pdg_id) for particle in event.final_particles],
    )


# ---------------------------------------------------------------------------
# Committed behavior 4: H1 _decay_phsp routes through the repaired function
# ---------------------------------------------------------------------------


def _spyed_split_safe(counter: list[tuple[float, float, float]]):
    """Wrap ``two_body_split_safe`` to record its arguments and delegate."""
    real_split = continuum_channels_module.two_body_split_safe

    def spy(parent, mass_a, mass_b, rng):
        counter.append((parent, mass_a, mass_b))
        return real_split(parent, mass_a, mass_b, rng)

    return real_split, spy


def test_h1_phsp_two_body_threshold_routes_through_split_safe():
    """The exact-threshold two-body H1 draw splits through the repaired
    function at the threshold point, with no random numbers consumed."""
    m1, m2 = M_P, M_PI
    parent_mass = m1 + m2  # the exact float sum
    counter: list[tuple[float, float, float]] = []
    real_split, spy = _spyed_split_safe(counter)
    continuum_channels_module.two_body_split_safe = spy
    rng = np.random.default_rng(H1_SEED)
    state_before = rng.bit_generator.state
    try:
        daughters = continuum_channels_module._decay_phsp(parent_mass, [m1, m2], rng)
    finally:
        continuum_channels_module.two_body_split_safe = real_split

    assert counter == [(parent_mass, m1, m2)]  # routed, with the configured masses
    assert daughters == [
        FourVector(m1, 0.0, 0.0, 0.0),
        FourVector(m2, 0.0, 0.0, 0.0),
    ]
    assert rng.bit_generator.state == state_before


def test_h1_phsp_three_body_near_threshold_terminal_split():
    """The H1 recursion reaches the terminal ``two_body_split_safe`` with a
    near-threshold cluster mass: every split argument stays inside the
    documented band, the daughters are on shell, and the stream is
    deterministic at a fixed seed."""
    masses = [M_PI, M_PI0, M_PI]
    parent_mass = sum(masses) + 1e-4  # near-threshold total kinetic margin
    real_split = continuum_channels_module.two_body_split_safe

    counter: list[tuple[float, float, float]] = []

    def run() -> list[FourVector]:
        counter.clear()
        continuum_channels_module.two_body_split_safe = _spyed_split_safe(counter)[1]
        rng = np.random.default_rng(H1_SEED)
        try:
            return continuum_channels_module._decay_phsp(parent_mass, list(masses), rng)
        finally:
            continuum_channels_module.two_body_split_safe = real_split

    first = run()
    second = run()

    # The three-body recursion splits through two_body_split_safe twice:
    # the top-level split and the terminal two-body split of the drawn
    # cluster.
    assert len(counter) == 2
    for parent, mass_a, mass_b in counter:
        # No draw moves a daughter off its configured mass shell: every
        # split argument is at or above threshold within the band, so the
        # repair never clips and never rejects here.
        assert (mass_a + mass_b) - parent <= two_body_threshold_tolerance(
            parent, mass_a, mass_b
        )
    _assert_on_shell(first, masses)
    _assert_sum_conservation(first, parent_mass)
    # Deterministic at the seed: bit-identical daughters across runs.
    assert [v.as_dict() for v in first] == [v.as_dict() for v in second]


# ---------------------------------------------------------------------------
# Committed behavior 5: the audit fails on source mutation (temp copies)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _serving_mutated_source(function, line_index_from_def: int, replacement_text: str):
    """Serve *function*'s source from linecache with one line replaced.

    ``audit_two_body_sites`` reads the source through ``inspect`` and
    ``linecache``.  Injecting a mutated copy of the file's lines into
    ``linecache.cache`` (a temp copy in memory, with the file's real size
    and mtime so ``linecache.checkcache`` accepts the entry) probes the
    audit's sensitivity to source mutation without ever touching the file
    on disk.  The previous cache entry is restored on exit.

    The cache entry is keyed by the function's own ``co_filename``
    verbatim: ``linecache`` keys on the literal import-time path and
    performs no normalization, while the path may carry ``..`` components
    when the package was imported through a relative parent directory.
    """
    filename = function.__code__.co_filename
    with open(filename, encoding="utf-8") as handle:
        lines = handle.readlines()
    index = function.__code__.co_firstlineno - 1 + line_index_from_def
    original = lines[index]
    lines[index] = replacement_text + "\n"
    stat = os.stat(filename)
    cached = linecache.cache.get(filename)
    linecache.cache[filename] = (stat.st_size, stat.st_mtime, lines, filename)
    try:
        yield
    finally:
        lines[index] = original
        if cached is not None:
            linecache.cache[filename] = cached
        else:
            linecache.cache.pop(filename, None)


def test_audit_fails_when_a_registered_site_vanishes():
    """A vanished 4-vector construction site fails the pinned inventory."""
    function = excitation_kinematics_module.two_body_split_safe
    source = inspect.getsource(function).splitlines()
    index = next(
        i
        for i, line in enumerate(source)
        if re.search(r"raise TwoBodyBelowThreshold\(", line)
    )
    with _serving_mutated_source(function, index, "        _audit_vanish_probe = 0"):
        with pytest.raises(RuntimeError, match="audit site vanished"):
            audit_two_body_sites()
    # Restored: the pinned inventory passes again.
    assert len(audit_two_body_sites()) == AUDIT_SITE_COUNT


def test_audit_fails_on_unclaimed_construction_line():
    """A new construction line no registered site claims fails the audit."""
    function = excitation_kinematics_module.two_body_split_rest_frame
    source = inspect.getsource(function).splitlines()
    index = next(
        i for i, line in enumerate(source) if line.strip().startswith("e1 = math.sqrt(mass1")
    )
    with _serving_mutated_source(
        function, index, "    ghost_probe = FourVector(1.0, 0.0, 0.0, 0.0)"
    ):
        with pytest.raises(RuntimeError, match="not in the audit inventory"):
            audit_two_body_sites()
    assert len(audit_two_body_sites()) == AUDIT_SITE_COUNT


def test_audit_fails_when_an_audited_function_vanishes():
    """A function no longer defined in its module fails the audit."""
    original = excitation_kinematics_module.two_body_split_safe
    excitation_kinematics_module.two_body_split_safe = None
    try:
        with pytest.raises(RuntimeError, match="is not defined in the module"):
            audit_two_body_sites()
    finally:
        excitation_kinematics_module.two_body_split_safe = original
    assert len(audit_two_body_sites()) == AUDIT_SITE_COUNT
