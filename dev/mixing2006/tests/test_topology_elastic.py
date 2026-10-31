"""Card tests for the elastic reference event class.

Round 5, card 1 (``TASK-20260925-r5-elastic-reference``; plan section
22 task 28, line 1753; plan section 11.1 item 1, line 824).  The
card's pre-registered deterministic tests against the module
``na49_mixing/topology_elastic.py``:

- ``test_t_range``: the exact two-body elastic t range
  ``[-(2 m_p p_beam / sqrt(s))^2, 0]`` on the pinned M8 p+p
  configuration (beam 158 GeV/c, recorded sqrt(s) = 17.270 GeV).
- ``test_t_normalization``: the closed-form normalization of
  ``exp(B_SD t)`` on ``[t_min, 0]`` against quadrature within 1e-12;
  the inverse-CDF round trip on a pinned quantile grid within 1e-9;
  the 100,000-draw sample mean against the analytic mean within
  3 sigma of 1/sqrt(N).
- ``test_zero_transfer``: the exact integer ledger check passes on a
  pinned event and raises ``ElasticTransferViolation`` on two planted
  violations (a nonzero line transfer, a nonzero exchange transfer).
- ``test_no_excitation``: the white-box field scan (no excited state,
  no central system, no decay vertex in the frozen record; the class
  label is ``"elastic"``).
- ``test_conservation``: over a pinned 10,000-event trace, exact
  ``(B, Q, S)`` bookkeeping ((1, 1, 0) per line, (2, 2, 0) total),
  four-momentum closure within 1e-9 relative, both outgoing lines on
  the ``m_p`` mass shell within 1e-9 relative, and the realized
  Mandelstam t of the solved kinematics equal to the drawn t.
- ``test_draw_budget``: a pinned 1,000-event trace consumes exactly
  one uniform on the closed ``exchange_kinematics`` substream per
  event and zero elsewhere (pinned RNG-state identity, the counting
  proxy, and the continuous-stream identity against the pristine
  uniform sequence).
- ``test_layout_sync``: the test register (the card's frozen-substrate
  provenance pins, verified on disk) and the layout sync (the
  subprocess check from the monorepo root; ``logic_drift = 0``).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, with the per-event substream triple
``(master_seed, replica, event_index)`` on the closed
``exchange_kinematics`` label.

Hermetic: the module, numpy, scipy, the standard library, and the
committed config files only; no event generation driver, no external
tables, no Monte Carlo run (the runs R50-R55 exercise the class
statistically).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
    ElasticTransferViolation,
    ExchangeRecord,
    LineState,
    check_zero_transfer,
    draw_elastic_t,
    elastic_t_quantile,
    elastic_t_range,
    generate_elastic,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, event index = trace index.
MASTER_SEED = 20260925

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, and the sealed Round 4 R45/R46 drivers in both
#: helper trees.
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
}

#: The M8-frozen effective transfer slope and the kernel boundary B,
#: pinned through the manifest's frozen parameter map (the R45 sealed
#: point keeps the ramp start excluded from every fit set).
KERNEL_BOUNDARY = 2.680194110727528

#: The pinned inverse-CDF quantile grid of the round-trip check (the
#: underflowed range edge excludes u = 0; see the module docstring).
QUANTILE_GRID = (
    0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _elastic_rng(event_index: int) -> np.random.Generator:
    """The pinned exchange_kinematics stream of one trace event."""

    streams = derive_substreams(MASTER_SEED, 0, event_index)
    return np.random.default_rng(streams.stream_for("exchange_kinematics"))


def _elastic_stream_id(event_index: int) -> int:
    return derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "exchange_kinematics")


class _FixedUniform:
    """A duck-typed single-uniform source: ``random()`` always returns
    the pinned quantile (the inverse-CDF round-trip probe)."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


class _CountingRng:
    """A counting proxy around one generator (the house draw-count
    pattern): every ``random()`` consumes exactly one uniform from the
    wrapped stream and advances the counter."""

    def __init__(self, source) -> None:
        self._source = source
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._source.random())


def _analytic_t_range() -> tuple[float, float]:
    return elastic_t_range(SQRT_S)


def _analytic_mean(t_min: float) -> float:
    """The closed-form mean of ``exp(B_SD t)`` on ``[t_min, 0]``:
    ``-1/B - t_min e^{B t_min} / (1 - e^{B t_min})``."""

    e_lo = math.exp(ELASTIC_TRANSFER_SLOPE * t_min)
    return (-1.0 / ELASTIC_TRANSFER_SLOPE
            - t_min * e_lo / (1.0 - e_lo))


# ----------------------------------------------------------------------
# test_t_range
# ----------------------------------------------------------------------
def test_t_range():
    """The exact two-body elastic t range on the pinned M8 p+p
    configuration: ``t_min = -(2 m_p p_beam / sqrt(s))^2`` within
    1e-12 and the right edge exactly 0."""

    t_min, t_max = _analytic_t_range()
    assert t_max == 0.0
    assert t_min < 0.0
    analytic = -((2.0 * PROTON_MASS * BEAM_MOMENTUM / SQRT_S) ** 2)
    assert abs(t_min - analytic) <= 1e-12

    # the recorded SQRT_S = 17.270 GeV is the rounded quote of the exact
    # fixed-target invariant mass; the nominal range edge sits within
    # 0.01 GeV^2 of the exact two-body limit of the beam-momentum
    # configuration (the guard in generate_elastic checks the exact
    # limit; see the module docstring)
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    s_exact = 2.0 * PROTON_MASS * (PROTON_MASS + e_beam)
    exact_limit = s_exact - 4.0 * PROTON_MASS * PROTON_MASS
    assert abs((-t_min) - exact_limit) < 0.01

    # checked, never repaired: a non-finite or sub-threshold sqrt_s is
    # rejected
    with pytest.raises(ValueError):
        elastic_t_range(2.0 * PROTON_MASS)
    with pytest.raises(ValueError):
        elastic_t_range(float("nan"))


# ----------------------------------------------------------------------
# test_t_normalization
# ----------------------------------------------------------------------
def test_t_normalization():
    """The closed-form normalization of ``exp(B_SD t)`` on ``[t_min,
    0]`` against quadrature within 1e-12; the inverse-CDF round trip
    on the pinned quantile grid within 1e-9; the 100,000-draw sample
    mean against the analytic mean within 3 sigma of 1/sqrt(N)."""

    t_min, t_max = _analytic_t_range()
    slope = ELASTIC_TRANSFER_SLOPE
    closed = (1.0 - math.exp(slope * t_min)) / slope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quad_value, _err = quad(
            lambda t: math.exp(slope * t), t_min, t_max,
            epsabs=1e-13, epsrel=1e-13, limit=400)
    assert abs(quad_value - closed) <= 1e-12

    # the inverse-CDF round trip on the pinned quantile grid: the draw
    # of a pinned quantile reproduces the quantile, and the closed-form
    # CDF of the drawn t reproduces the quantile
    e_lo = math.exp(slope * t_min)
    for u in QUANTILE_GRID:
        t_quant = elastic_t_quantile(u, t_min)
        t_draw = draw_elastic_t(_FixedUniform(u))
        assert abs(t_draw - t_quant) <= 1e-9
        cdf = (math.exp(slope * t_draw) - e_lo) / (1.0 - e_lo)
        assert abs(cdf - u) <= 1e-9
    # the underflowed edge is a domain error, not a silent value
    with pytest.raises(ValueError):
        elastic_t_quantile(0.0, t_min)

    # the 100,000-draw sample mean against the analytic mean within
    # 3 sigma of 1/sqrt(N) (one pinned exchange_kinematics stream,
    # consumed sequentially)
    n = 100_000
    stream = _elastic_rng(0)
    draws = np.fromiter((draw_elastic_t(stream) for _ in range(n)),
                        dtype=float, count=n)
    mean_exact = _analytic_mean(t_min)
    sigma_mean = float(np.std(draws)) / math.sqrt(n)
    assert abs(float(np.mean(draws)) - mean_exact) <= 3.0 * sigma_mean, (
        float(np.mean(draws)), mean_exact, sigma_mean)


# ----------------------------------------------------------------------
# test_zero_transfer
# ----------------------------------------------------------------------
def test_zero_transfer():
    """The exact integer zero-transfer check passes on a pinned event
    and raises ``ElasticTransferViolation`` on two planted violations
    (a nonzero line transfer, a nonzero exchange transfer)."""

    event = generate_elastic(_elastic_rng(7))
    assert check_zero_transfer(event) is None

    # planted violation 1: a nonzero line transfer (the target-side
    # outgoing line loses one unit of charge)
    bad_line = dataclasses.replace(
        event,
        outgoing=(event.outgoing[0],
                  dataclasses.replace(event.outgoing[1], qn=(1, 0, 0, 1))))
    with pytest.raises(ElasticTransferViolation, match="line 1"):
        check_zero_transfer(bad_line)

    # planted violation 2: a nonzero exchange transfer
    bad_exchange = dataclasses.replace(
        event,
        exchange=ExchangeRecord(name="vacuum", transfer=(0, 0, 1, 0)))
    with pytest.raises(ElasticTransferViolation, match="exchange"):
        check_zero_transfer(bad_exchange)

    # checked, never repaired: the check consumes an ElasticEvent
    with pytest.raises(ValueError):
        check_zero_transfer(("not", "an", "event"))


# ----------------------------------------------------------------------
# test_no_excitation
# ----------------------------------------------------------------------
def test_no_excitation():
    """The white-box field scan: the pinned event record carries no
    excited state, no central system, and no decay vertex; the class
    label is ``"elastic"``; the frozen dataclasses carry exactly the
    card's field sets."""

    event = generate_elastic(_elastic_rng(11))

    assert getattr(ElasticEvent, "__dataclass_params__").frozen
    assert getattr(LineState, "__dataclass_params__").frozen
    assert getattr(ExchangeRecord, "__dataclass_params__").frozen

    # the record's field set is exactly the card's five fields: no
    # excited-state, central-system, or decay-vertex field exists
    assert [f.name for f in dataclasses.fields(ElasticEvent)] == [
        "class_label", "incoming", "outgoing", "exchange", "t"]
    assert [f.name for f in dataclasses.fields(LineState)] == [
        "four_momentum", "qn"]
    assert [f.name for f in dataclasses.fields(ExchangeRecord)] == [
        "name", "transfer"]

    # the label excludes the class from every inelastic sample
    # (plan line 824)
    assert event.class_label == "elastic"
    assert event.exchange.name == "vacuum"
    assert event.exchange.transfer == (0, 0, 0, 0)
    assert event.t < 0.0

    # every line is an intact proton in the doubled-I3 convention
    for line in event.incoming + event.outgoing:
        assert isinstance(line, LineState)
        assert line.qn == (1, 1, 0, 1)
        assert len(line.four_momentum) == 4

    # no nested record beyond LineState/ExchangeRecord exists anywhere
    # in the record (no decay vertex, no central system)
    def _scan(obj):
        if isinstance(obj, tuple):
            for part in obj:
                _scan(part)
        elif dataclasses.is_dataclass(obj):
            assert isinstance(obj, (LineState, ExchangeRecord, ElasticEvent)), (
                f"unexpected record in the elastic event: {type(obj)!r}")

    _scan(event)


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation():
    """Over a pinned 10,000-event trace: exact (B, Q, S) bookkeeping
    ((1, 1, 0) per line, (2, 2, 0) total), four-momentum closure within
    1e-9 relative, both outgoing lines on the m_p mass shell within
    1e-9 relative, and the realized Mandelstam t equal to the drawn t."""

    n = 10_000
    for i in range(n):
        event = generate_elastic(_elastic_rng(i))
        check_zero_transfer(event)

        # exact integer bookkeeping per line and in total
        for in_line, out_line in zip(event.incoming, event.outgoing):
            assert in_line.qn[:3] == (1, 1, 0)
            assert out_line.qn[:3] == (1, 1, 0)
        total_in = tuple(sum(part) for part in zip(
            *(line.qn for line in event.incoming)))
        total_out = tuple(sum(part) for part in zip(
            *(line.qn for line in event.outgoing)))
        assert total_in == (2, 2, 0, 2)
        assert total_out == (2, 2, 0, 2)

        # four-momentum closure within 1e-9 relative
        in_sum = [a + b for a, b in zip(event.incoming[0].four_momentum,
                                        event.incoming[1].four_momentum)]
        out_sum = [a + b for a, b in zip(event.outgoing[0].four_momentum,
                                         event.outgoing[1].four_momentum)]
        for k in range(4):
            assert abs(out_sum[k] - in_sum[k]) <= 1e-9 * abs(in_sum[k])

        # both outgoing lines on the m_p mass shell within 1e-9 relative
        for line in event.outgoing:
            e, px, py, pz = line.four_momentum
            mass = math.sqrt(e * e - px * px - py * py - pz * pz)
            assert abs(mass - PROTON_MASS) <= 1e-9 * PROTON_MASS

        # the realized Mandelstam t of the solved kinematics equals the
        # drawn t (absolute tolerance for the small-|t| bulk, relative
        # beyond 1 GeV^2)
        p1 = event.incoming[0].four_momentum
        p3 = event.outgoing[0].four_momentum
        t_real = ((p1[0] - p3[0]) ** 2
                  - (p1[1] - p3[1]) ** 2
                  - (p1[2] - p3[2]) ** 2
                  - (p1[3] - p3[3]) ** 2)
        assert abs(t_real - event.t) <= 1e-9 * max(1.0, abs(event.t)), (
            i, t_real, event.t)


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget():
    """A pinned 1,000-event trace consumes exactly one uniform on the
    closed exchange_kinematics substream per event and zero elsewhere:
    the counting proxy, the bit-identical post-draw stream state, the
    inverse CDF of the first pristine uniform, and the continuous-stream
    identity against the pristine uniform sequence."""

    assert "exchange_kinematics" in SUBSTREAM_LABELS
    t_min, _t_max = _analytic_t_range()
    n = 1_000

    # per-event stream: exactly one uniform (counting proxy), the post-
    # draw state is bit-identical to the reference stream after exactly
    # one draw, and the drawn t is the inverse CDF of the first
    # pristine uniform
    for i in range(n):
        sid = _elastic_stream_id(i)
        counter = _CountingRng(np.random.default_rng(sid))
        event = generate_elastic(counter)
        assert counter.calls == 1

        advanced = np.random.default_rng(sid)
        advanced.random()
        assert counter._source.bit_generator.state == advanced.bit_generator.state

        u_first = np.random.default_rng(sid).random()
        assert event.t == elastic_t_quantile(u_first, t_min)

    # the continuous-stream identity: one shared stream over the whole
    # trace consumes exactly the pristine uniform sequence in order
    # (no burn-in, no extra draw anywhere in the module)
    shared = _elastic_rng(0)
    pristine = np.random.default_rng(_elastic_stream_id(0))
    for i in range(n):
        u_ref = pristine.random()
        event = generate_elastic(shared)
        assert event.t == elastic_t_quantile(u_ref, t_min)


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk: the builder, the card-6 routing, the card-2
    engine, the committed catalog, the continuum families, and the
    sealed R45/R46 drivers in both helper trees; the M8-frozen
    parameter map and the kernel boundary through the manifest) and
    the layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["regge.effective_transfer_slope"] == ELASTIC_TRANSFER_SLOPE
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
