"""Hermetic tests for the R46 driver (the simplified recoil/decay
re-profile).

Task ``TASK-20260920-r4-gate4-followup-recoil-decay-reprofile`` (the
Gate 4 follow-up obligation 2 of the written Gate 4 decision; plan
section 23, lines 1821-1823; plan section 10 run R42, line 786, re-run
under the simplified form), phase 1 (no Monte Carlo): the profile,
curvature, and boundary tests with named re-derivations.  The frozen
simplified alignment (the card-20 substrate: kappa_B = 0.0, kappa_m =
4.0, the frozen b_dec 2.0 GeV^-2 on the profile grid, the frozen width
0.3 GeV) is asserted against the pinned corrected R41 fit-report
fixture with the same predicate the production pre-registration runs
against the sealed archive; the unchanged 9 x 9 grid and its bounds
are never relaxed; the exact transfer-law and decay form-factor
reweighting identities are re-derived on pinned fixtures; the plan-
5.4-line-311 quoting gate, the per-direction curvature spread, the
boundary constructs, the correlation algebra, and the named-residual
register are exercised on synthetic surfaces; the seed block
20301076-20301083 and the 900M event-index partition are checked for
disjointness against every previously used block; the superseded
corrected R42 profile pins are verified read-only (the sealed-archive
legs skip on a fresh checkout).  No Monte Carlo generation at run
scale, no SPS table: the acceptance is hermetic.
"""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from tests._freeze_inventory import inventory_count as _inventory_count


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r46_simplified_recoil_decay_profile.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r46_simplified_recoil_decay_profile", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R46 driver was not found next to the test file")


r46 = _load_driver()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def by_kind(register, kind):
    """The register entry of the given kind (the register names each
    kind at most once per axis for the boundary findings)."""
    for entry in register:
        if entry["kind"] == kind:
            return entry
    raise AssertionError(f"no {kind} entry")


# ---------------------------------------------------------------------------
# The run identity and the substrate (the card-20 sibling pins)
# ---------------------------------------------------------------------------
def test_run_identity_constants():
    """The R46 run identity and the card chain (the Gate 4 decision, the
    card-20 sibling, and this card appended)."""
    assert r46.RUN_ID == "R46"
    assert r46.CARD == \
        "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile"
    assert r46.RUN_NAME == "r46_simplified_recoil_decay_profile_2026-09-24"
    assert r46.DEFAULT_RUN_DIR == (
        "dev/00runs/r46_simplified_recoil_decay_profile_2026-09-24")
    assert r46.GATE_RECORD_NAME == "gate4_followup_reprofile.json"
    assert r46.PROFILE_REPORT_NAME == "profile_report.json"
    assert r46.DRIVER_REL == (
        "helpers/benchmarking/"
        "staged_r46_simplified_recoil_decay_profile.py")
    # the card-20 substrate sibling digest (the frozen R45 driver)
    assert r46.FROZEN_SUBSTRATE["r45_driver"] == (
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py",
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    )
    assert r46.FROZEN_SUBSTRATE["r44_driver"][1] == (
        "6d77c7137dceb5bb52f05c6c37f701c250b63756618197f2cc454b4ca3cbf1e5")
    assert r46.FROZEN_SUBSTRATE["r42_driver"][1] == (
        "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd")
    assert r46.CARD_CHAIN[-3:] == (
        "TASK-20260920-r4-gate4-decision",
        "TASK-20260920-r4-gate4-followup-simplified-alignment",
        "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile",
    )
    # the sealed run dirs include the card-20 substrate and the
    # superseded corrected R42 archive
    assert r46.SEALED_RUN_DIRS[-2:] == (
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23",
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24",
    )
    assert "dev/00runs/r42_recoil_decay_separation_2026-09-23" in \
        r46.SEALED_RUN_DIRS
    assert r46.SUPERSEDED_R42_RUN_DIR == \
        "dev/00runs/r42_recoil_decay_separation_2026-09-23"


def test_card20_substrate_sibling_pins():
    """The card-20 substrate cross-check ran at import: the sealed R45
    driver is byte-identical in both trees at the pinned digest, its
    self-digest agrees, its frozen policy equals this card's frozen
    simplified alignment, and its run dir is the card-20 archive."""
    status = r46._CARD20_SUBSTRATE_STATUS
    assert status["verified"] is True
    assert status["sibling_driver_rel"] == (
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py")
    assert status["sibling_driver_sha256"] == (
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10")
    assert status["sibling_self_digest_match"] is True
    assert status["sibling_frozen_policy"] == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
    }
    assert status["sibling_frozen_width_gev"] == 0.3
    assert status["sibling_run_dir"] == \
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"
    # the frozen simplified alignment itself (the card-20 tuple)
    assert r46.FROZEN_SIMPLIFIED_ALIGNMENT == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
        "baryonic_nominal_width": 0.3,
    }
    assert (r46.GENERATION_KAPPA_B, r46.GENERATION_KAPPA_M) == (0.0, 4.0)
    assert r46.FROZEN_WIDTH_GEV == 0.3
    assert "card-20 frozen simplified alignment" in r46.FROZEN_ALIGNMENT_NOTE
    assert "named deviation" in r46.FROZEN_ALIGNMENT_NOTE
    assert "r45_simplified_alignment_freeze_2026-09-24" in \
        r46.CARD20_SUBSTRATE_NOTE


def test_frozen_parameter_assertion_fixture_and_refusals():
    """The frozen-parameter assertion runs the same predicate against
    the pinned R41 fixture that the production pre-registration runs
    against the sealed archive; every drift refuses."""
    frozen = r46.assert_frozen_parameters(r46.R41_FIT_REPORT_FIXTURE)
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == (0.0, 4.0, 2.0, 0.3)
    assert frozen["refit"] is False
    fixture = r46.R41_FIT_REPORT_FIXTURE
    drifted = dict(fixture)
    drifted["fitted"] = dict(fixture["fitted"], kappa_B=0.1)
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(drifted)
    width_drift = dict(fixture)
    width_drift["fitted"] = dict(fixture["fitted"],
                                 baryonic_nominal_width=0.4)
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(width_drift)
    wrong_card = dict(fixture, card="TASK-20260920-r4-run-r42-")
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(wrong_card)
    wrong_run = dict(fixture, run="R42")
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(wrong_run)
    smoke_mode = dict(fixture, mode="smoke")
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(smoke_mode)
    no_block = {"card": fixture["card"], "run": "R41",
                "mode": "production"}
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(no_block)


def _superseded_report_path():
    return (r46.MONO_ROOT / r46.SUPERSEDED_R42_RUN_DIR /
            r46.SUPERSEDED_R42_REPORT_NAME)


def test_sealed_archives_and_superseded_profile():
    """The sealed R30-R32 / R41-R45 archives verify read-only on this
    checkout (skips on a fresh checkout), and the superseded corrected
    R42 profile report sits at its pinned file digest and self-digest
    with the boundary minimum (0.5, 2.6) and the failing eigenvalues."""
    if not all(
        (r46.MONO_ROOT / rel / r46.SHA256SUMS_NAME).is_file()
        for rel in r46.SEALED_RUN_DIRS
    ):
        pytest.skip("the sealed archives are absent on this checkout")
    seals = r46.verify_sealed_inputs()
    for rel, status in seals.items():
        assert status["verified"] is True and status["mismatches"] == 0, rel
    assert seals["dev/00runs/r45_simplified_alignment_freeze_2026-09-24"] \
        == {"entries": 25, "mismatches": 0, "verified": True}
    assert seals["dev/00runs/r42_recoil_decay_separation_2026-09-23"] == \
        {"entries": 13, "mismatches": 0, "verified": True}
    # the frozen-parameter source (the sealed corrected R41 fit report)
    fit_path = (r46.MONO_ROOT / r46.R41_RUN_DIR / r46.R41_FIT_REPORT_NAME)
    assert r46.R41_FIT_REPORT_FILE_SHA256 == _sha256_file(fit_path)
    report = json.loads(fit_path.read_text(encoding="utf-8"))
    assert report.get("digest") == r46.R41_FIT_REPORT_DIGEST
    frozen = r46.assert_frozen_parameters(report)
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == (0.0, 4.0, 2.0, 0.3)
    # the superseded corrected R42 profile report pins
    superseded_path = _superseded_report_path()
    assert r46.SUPERSEDED_R42_REPORT_FILE_SHA256 == _sha256_file(
        superseded_path)
    superseded = json.loads(superseded_path.read_text(encoding="utf-8"))
    assert superseded.get("digest") == r46.SUPERSEDED_R42_REPORT_DIGEST
    assert superseded["minimum"]["b_eff_multiplier"] == 0.5
    assert superseded["minimum"]["b_dec"] == 2.6
    assert superseded["hessian"]["eigenvalues"] == \
        r46.SUPERSEDED_R42_EIGENVALUES
    assert superseded["correlation_b_eff_b_dec"] is None
    block4 = superseded["curvature_stability"]["4"]
    assert block4["eigenvalues"] == r46.SUPERSEDED_R42_BLOCK4_EIGENVALUES
    assert block4["positive_definite"] is False
    assert r46.SUPERSEDED_R42_STATUS == "superseded by re-profile"
    assert r46.SUPERSEDED_R42_SEALED_UTC == "2026-09-24T01:53:32Z"


# ---------------------------------------------------------------------------
# The profile grid (unchanged; the bounds never relaxed)
# ---------------------------------------------------------------------------
def test_profile_grid_unchanged_and_bounds():
    """The unchanged 9 x 9 grid: B_eff multipliers 0.5..2.5 in 0.25
    steps, b_dec 0.2..2.6 in 0.3 GeV^-2 steps; the card-20 frozen
    b_dec 2.0 GeV^-2 sits on the grid (the generation point multiplier
    1.0 / b_dec 2.0); the bounds are never relaxed."""
    assert r46.B_EFF_MULTIPLIERS == (
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
    assert r46.B_DEC_GRID == (
        0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6)
    assert r46.B_DEC_MULT_STEP == 0.25 and r46.B_DEC_STEP_GEV2 == 0.3
    assert r46.B_EFF_FROZEN_GEV2 == 8.504140762083612
    assert r46.B_DEC_GENERATION_GEV2 == 2.0
    assert 2.0 in r46.B_DEC_GRID and 1.0 in r46.B_EFF_MULTIPLIERS
    assert "never relaxed" in r46.PROFILE_GRID_NOTE
    assert "never extended" in r46.PROFILE_GRID_NOTE
    # the blend ramp start stays fixed at B (obligation 4)
    assert r46.BLEND_RAMP_START_GEV == 2.680194110727528
    assert r46.M8_PARAMETER_HASH.startswith("358989d84cb03cff")


def test_validity_pinned_points_and_bound():
    """The two pinned reweighting-validity grid points (unchanged from
    the corrected R42) and the 1e-6 bound are pre-registered; the
    pinned sub-sample rule names the 900M partition and seed 20301076."""
    assert r46.VALIDITY_GRID_POINTS == ((1.5, 1.7), (2.0, 0.5))
    assert r46.VALIDITY_BOUND == 1e-6
    assert r46.VALIDITY_EVENTS == 2_000
    assert "900_000_000 + i" in r46.VALIDITY_SUBSAMPLE_RULE
    assert "20301076" in r46.VALIDITY_SUBSAMPLE_RULE


# ---------------------------------------------------------------------------
# The exact reweighting algebra (named re-derivations on pinned data)
# ---------------------------------------------------------------------------
def _pinned_block():
    """A hand-built two-event BlockLatent: event 0 carries one
    two-channel continuum vertex and two bin rows, event 1 carries one
    bin row and no vertex; the transfer supports are pinned."""
    return r46.BlockLatent(
        replica=0, seed=20301076, n_events=2, route_census=(2, 0, 2),
        row_flat=np.asarray([5, 9, 14], dtype=np.int64),
        row_weight=np.asarray([1.25, 0.75, 2.0], dtype=float),
        row_ptr=np.asarray([0, 2, 3], dtype=np.int64),
        event_ids=np.asarray([900_000_000, 900_000_001], dtype=np.int64),
        t_scatter=np.asarray([-1.0, -2.0], dtype=float),
        t_minus=np.asarray([-3.0, -4.0], dtype=float),
        t_plus=np.asarray([0.0, 0.0], dtype=float),
        vtx_ptr=np.asarray([0, 1, 1], dtype=np.int64),
        vtx_nchan=np.asarray([2], dtype=np.int64),
        vtx_chosen=np.asarray([0], dtype=np.int64),
        ch_w0=np.asarray([0.5, 0.5], dtype=float),
        ch_q2=np.asarray([0.25, 0.75], dtype=float),
    )


def _manual_channel_ratio(b_dec, w0, q2, chosen, generation=2.0):
    """The hand-computed card-8 channel-choice ratio at the trial b_dec
    against the generation b_dec (the R41 algebra: the chosen share's
    w0 factor cancels between the trial and generation densities, so
    the ratio is exp(-(b'-b) q2_chosen) over the trial pool total)."""
    delta = float(b_dec) - float(generation)
    ex = [math.exp(-delta * q) for q in q2]
    pool = sum(w * e for w, e in zip(w0, ex))
    return ex[chosen] / pool


def _truncated_density(t, b, t_lo, t_hi):
    return b * math.exp(b * t) / (
        math.exp(b * t_hi) - math.exp(b * t_lo))


def test_transfer_log_weight_exact_density_ratio():
    """The closed-form transfer weight is the exact truncated-exponential
    density ratio: on a pinned support, exp(log w) equals the numeric
    ratio p_B'(t) / p_B(t) of the normalized laws, and the weight is
    exactly 1 at the frozen slope and on a degenerate support."""
    t_lo, t_hi = -3.0, 0.0
    b_gen = r46.B_EFF_FROZEN_GEV2
    t_grid = np.linspace(t_lo, t_hi, 25)
    for multiplier in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        b_trial = float(multiplier * b_gen)
        log_w = r46.transfer_log_weight(
            t_grid, np.asarray([t_lo] * t_grid.size),
            np.asarray([t_hi] * t_grid.size), b_gen, b_trial)
        for t, lw in zip(t_grid, log_w):
            if multiplier == 1.0:
                assert float(lw) == 0.0
                continue
            numeric = math.log(
                (b_trial * math.exp(b_trial * float(t)))
                / (math.exp(b_trial * t_hi) - math.exp(b_trial * t_lo))
                / ((b_gen * math.exp(b_gen * float(t)))
                   / (math.exp(b_gen * t_hi) - math.exp(b_gen * t_lo))))
            assert float(lw) == pytest.approx(numeric, rel=1e-9,
                                              abs=1e-12), (t, multiplier)
    # the degenerate zero-width support carries weight 1
    log_w = r46.transfer_log_weight(
        np.asarray([1.0]), np.asarray([1.0]), np.asarray([1.0]),
        b_gen, b_gen * 0.5)
    assert float(log_w[0]) == 0.0


def _support_cdf(t, b):
    t_lo, t_hi = -3.0, 0.0
    return (math.exp(b * (t - t_hi)) - math.exp(-b * (t_hi - t_lo))) / (
        1.0 - math.exp(-b * (t_hi - t_lo)))


def test_transfer_implied_t_cdf_identity():
    """The implied trial realization preserves the generating uniform:
    G_B'(t_implied) equals G_B(t) per pinned event (the closed-form
    inverse-CDF identity the pinned validity probe asserts)."""
    b_gen = r46.B_EFF_FROZEN_GEV2
    t_lo, t_hi = -3.0, 0.0
    for t_gen in (-2.5, -1.5, -0.5, -0.05):
        for multiplier in (0.5, 1.5, 2.5):
            b_trial = float(multiplier * b_gen)
            t_implied = r46.transfer_implied_t(
                t_gen, b_gen, b_trial, t_lo, t_hi)
            if multiplier == 1.0:
                assert t_implied == t_gen
                continue
            assert _support_cdf(t_implied, b_trial) == pytest.approx(
                _support_cdf(t_gen, b_gen), rel=1e-9, abs=1e-12)


def test_event_weights_generation_point_exact_one():
    """At the generation point (multiplier 1.0, b_dec 2.0 GeV^-2) every
    per-event importance weight is exactly 1: the transfer-law ratio
    and the channel-choice ratio are both identically 1 there (the
    card-20 frozen point is on the grid)."""
    block = _pinned_block()
    eta = r46.event_weights(block, r46.B_EFF_FROZEN_GEV2,
                            r46.B_DEC_GENERATION_GEV2)
    assert eta.shape == (2,)
    assert float(eta[0]) == 1.0 and float(eta[1]) == 1.0
    # at the frozen slope the B_eff factor is 1 and the channel factor
    # follows the card-8 ratio
    eta_frozen_b = r46.event_weights(block, r46.B_EFF_FROZEN_GEV2, 1.7)
    expected_channel = _manual_channel_ratio(
        1.7, (0.5, 0.5), (0.25, 0.75), 0)
    assert float(eta_frozen_b[0]) == pytest.approx(
        expected_channel, rel=1e-14)
    assert float(eta_frozen_b[1]) == 1.0  # event 1 carries no vertex


def test_event_weights_channel_ratio_identity():
    """The per-vertex channel-choice ratio reweights exactly: the
    recorded (w0, q2, chosen) sufficient statistics reproduce the
    hand-computed card-8 ratio at every off-generation b_dec."""
    block = _pinned_block()
    for b_dec in (0.2, 0.8, 1.1, 1.4, 1.7, 2.3, 2.6):
        eta = r46.event_weights(block, r46.B_EFF_FROZEN_GEV2,
                                float(b_dec))
        expected = _manual_channel_ratio(b_dec, (0.5, 0.5),
                                         (0.25, 0.75), 0)
        assert float(eta[0]) == pytest.approx(expected, rel=1e-13), b_dec
        assert float(eta[1]) == 1.0
    # a trial b_dec far outside the pre-registered domain overflows the
    # pool total and refuses (checked, never repaired)
    bad = _pinned_block()
    with pytest.raises(SystemExit):
        r46.event_weights(bad, r46.B_EFF_FROZEN_GEV2, -1.0e6)


def test_event_weights_transfer_law_composition():
    """The composed per-event weight at a pinned grid point equals the
    product of the exact transfer-law ratio and the channel ratio
    (the trial law is exactly normalized on the unchanged support)."""
    block = _pinned_block()
    b_trial = float(1.5 * r46.B_EFF_FROZEN_GEV2)
    eta = r46.event_weights(block, b_trial, 1.7)
    log_wb = float(r46.transfer_log_weight(
        np.asarray([block.t_scatter[0]]),
        np.asarray([block.t_minus[0]]),
        np.asarray([block.t_plus[0]]),
        r46.B_EFF_FROZEN_GEV2, b_trial)[0])
    expected = math.exp(log_wb) * _manual_channel_ratio(
        1.7, (0.5, 0.5), (0.25, 0.75), 0)
    assert float(eta[0]) == pytest.approx(expected, rel=1e-12)
    # the ESS diagnostic is shift-invariant and 1 for constant weights
    assert r46.ess_fraction(np.ones(10)) == pytest.approx(1.0)
    assert r46.ess_fraction(np.asarray([1.0, 0.0, 0.0, 0.0])) == \
        pytest.approx(0.25)


def test_pack_unpack_roundtrip():
    """The base64 latent container is a lossless transport: the packed
    arrays unpack bit for bit."""
    arrays = [
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([0.5, -0.25], dtype=np.float64),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    ]
    packed = r46.pack_arrays(arrays)
    out = r46.unpack_arrays(packed)
    for a, b in zip(arrays, out):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The grid algebra (argmin, curvature, boundary constructs, correlation)
# ---------------------------------------------------------------------------
def _quadratic_surface(a: float, b: float, c: float, i0: int, j0: int):
    """A synthetic -2lnL surface with the curvature matrix
    [[2a, c], [c, 2b]] in the (multiplier, b_dec) coordinates."""
    db, dk = r46.B_DEC_MULT_STEP, r46.B_DEC_STEP_GEV2
    surface = []
    for i in range(9):
        row = []
        for j in range(9):
            x = db * (i - i0)
            y = dk * (j - j0)
            row.append(a * x * x + b * y * y + c * x * y)
        surface.append(row)
    return surface


def test_grid_argmin_and_hessian_interior():
    """The deterministic lexicographic argmin and the interior grid
    second differences reproduce the quadratic's curvature matrix
    exactly (the spacings 0.25 and 0.3 GeV^-2)."""
    surface = _quadratic_surface(2.0, 3.0, 0.5, 3, 4)
    i0, j0 = r46.grid_argmin(surface)
    assert (i0, j0) == (3, 4)
    h, flags = r46.grid_hessian(surface, i0, j0)
    assert h[0][0] == pytest.approx(4.0, rel=1e-12)
    assert h[1][1] == pytest.approx(6.0, rel=1e-12)
    assert h[0][1] == pytest.approx(0.5, rel=1e-12)
    assert h[1][0] == pytest.approx(0.5, rel=1e-12)
    assert flags["all_interior"] is True
    eigen = r46.hessian_eigenvalues(h)
    assert all(v > 0.0 for v in eigen)
    assert r46.positive_definite_2x2(h) is True
    rho = r46.correlation_from_hessian(h)
    assert rho == pytest.approx(
        -0.5 / math.sqrt(4.0 * 6.0), rel=1e-12)


def test_grid_hessian_boundary_constructs():
    """A minimum on a grid boundary carries the honest one-sided
    second-difference construct over the nearest interior-side triplet,
    and the interior flags name the boundary axes."""
    surface = _quadratic_surface(2.0, 3.0, 0.5, 0, 8)
    i0, j0 = r46.grid_argmin(surface)
    assert (i0, j0) == (0, 8)
    h, flags = r46.grid_hessian(surface, i0, j0)
    assert h[0][0] == pytest.approx(4.0, rel=1e-12)
    assert h[1][1] == pytest.approx(6.0, rel=1e-12)
    assert flags["all_interior"] is False
    assert flags["interior_bb"] is False and flags["interior_kk"] is False
    # a non-positive diagonal quotes a None correlation
    bad = [[-1.0, 0.0], [0.0, 2.0]]
    assert r46.correlation_from_hessian(bad) is None


def test_minima_agreement_gate():
    """The plan-5.4-line-311 quoting gate: the 8 per-block argmins must
    equal the pooled argmin; a single disagreeing block flips the
    gate."""
    agreeing = {str(k): _quadratic_surface(2.0, 3.0, 0.0, 3, 4)
                for k in range(8)}
    report = r46.minima_agreement(agreeing, 3, 4)
    assert report["minima_agree"] is True
    disagreeing = dict(agreeing)
    disagreeing["4"] = _quadratic_surface(2.0, 3.0, 0.0, 5, 8)
    report = r46.minima_agreement(disagreeing, 3, 4)
    assert report["minima_agree"] is False
    assert report["per_block_argmins"]["4"] == [5, 8]
    assert report["pooled_argmin"] == [3, 4]
    assert "line 311" in report["note"]


def _stability_fixture(diag_bb: float, diag_kk: float, spread: float):
    """Eight per-block Hessians around the pooled diagonals with a
    controlled relative spread on the multiplier axis (block 7)."""
    stability = {}
    for k in range(8):
        offset = spread if k == 7 else 0.0
        stability[str(k)] = {
            "hessian": [[diag_bb * (1.0 + offset), 0.0],
                        [0.0, diag_kk]],
            "eigenvalues": [min(diag_bb, diag_kk),
                            max(diag_bb, diag_kk)],
            "positive_definite": True,
            "argmin": [3, 4],
            "argmin_agrees_with_pooled": True,
        }
    return stability


def test_per_direction_curvature_spread():
    """The per-direction spread helper: a 10% spread is below the
    pre-registered 0.20 fraction, a 50% spread fails it, and the
    recorded fraction is the max over the blocks."""
    stability = _stability_fixture(4.0, 6.0, 0.0)
    pooled = [[4.0, 0.0], [0.0, 6.0]]
    spread = r46.per_direction_curvature_spread(stability, pooled)
    assert spread["b_eff_multiplier"]["spread_fraction"] == 0.0
    assert spread["b_eff_multiplier"]["below_fraction"] is True
    assert spread["b_dec"]["below_fraction"] is True
    # a 50% spread on the multiplier axis (block 7 at 6.0 vs pooled 4.0)
    stability_high = _stability_fixture(4.0, 6.0, 0.5)
    spread = r46.per_direction_curvature_spread(stability_high, pooled)
    assert spread["b_eff_multiplier"]["spread_fraction"] == \
        pytest.approx(0.5)
    assert spread["b_eff_multiplier"]["below_fraction"] is False
    assert spread["b_dec"]["below_fraction"] is True
    # a zero pooled diagonal quotes an infinite fraction (never relaxed)
    spread = r46.per_direction_curvature_spread(
        stability, [[0.0, 0.0], [0.0, 6.0]])
    assert math.isinf(spread["b_eff_multiplier"]["spread_fraction"])
    assert spread["b_eff_multiplier"]["below_fraction"] is False


def test_blocks_failing_census():
    """The per-block positive-definiteness census names the failing
    block indices with their eigenvalues."""
    stability = _stability_fixture(4.0, 6.0, 0.0)
    stability["4"] = {
        "hessian": [[1.0, 0.5], [0.5, -2.0]],
        "eigenvalues": [-2.0639347, 1.0639347],
        "positive_definite": False,
        "argmin": [3, 4],
        "argmin_agrees_with_pooled": True,
    }
    failing = r46.blocks_failing(stability)
    assert list(failing) == ["4"]
    assert failing["4"][0] < 0.0
    assert r46.blocks_failing(_stability_fixture(4.0, 6.0, 0.0)) == {}


def test_se_from_floored_between_replica_errors():
    """The between-replica SE convention: std ddof=1 over the blocks
    divided by sqrt(k), floored at the engine's deterministic SE
    floor; fewer than two blocks carry the floor."""
    stack = np.asarray([[0.25 * k] for k in range(1, 9)])
    expected = float(np.std(stack, axis=0, ddof=1)[0].item()) / math.sqrt(8)
    assert float(r46.se_from(stack)[0]) == pytest.approx(
        expected, rel=1e-12)
    single = np.asarray([[1.0]])
    assert float(r46.se_from(single)[0]) == r46.SE_FLOOR
    nan_stack = np.asarray([[float("nan")], [1.0]])
    assert float(r46.se_from(nan_stack)[0]) == float("inf")


def test_nuisance_columns_and_priors():
    """The nuisance structure stands: one normalization nuisance on the
    b_transport spectrum (prior 0.35) plus the obligation-4 carry on
    the blend-band-sensitive bins (prior 0.25); the seagulls carry
    none."""
    assert r46.NUISSANCE_COLUMNS == (
        "norm_b_transport_spectrum", "carry")
    assert r46.STAT_BLOCK_COLUMN == {"b_transport_spectrum": 0}
    assert r46.NORM_NUISANCE_PRIOR == 0.35
    assert r46.CARRY_NUISANCE_PRIOR == 0.25
    assert r46.CARRY_SPECIES == ("proton", "antiproton")
    assert r46.OBLIGATION4_R32_SPREAD == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    assert "blend-band-sensitive bins" in r46.OBLIGATION4_FORM
    assert "the original R32 spread stands" in r46.OBLIGATION4_FORM


def test_standing_residuals_recorded():
    """The R43 b_transport_seagull finding and the R32 spread carry
    appear as named entries (the card's standing-residual clause)."""
    standing = r46.STANDING_RESIDUALS
    assert len(standing) == 2
    assert standing[0]["kind"] == \
        "b_transport_seagull_beyond_isotropic_not_positive"
    assert standing[0]["observed"] == -2.3959765824349295e-05
    assert standing[1]["kind"] == "r32_model_uncertainty_carry"
    assert standing[1]["r32_spread"] == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }


# ---------------------------------------------------------------------------
# The named-residual register (the card's named findings)
# ---------------------------------------------------------------------------
def _failing_report() -> dict:
    """A synthetic report that fails every acceptance-(ii) condition:
    the minimum on both grid boundaries, the minima disagreement, a
    non-positive-definite pooled Hessian, a failing block, a spread
    failure, a degenerate ESS, and a non-finite surface point."""
    stability = {}
    for k in range(8):
        entry = {
            "hessian": [[4.0, 0.0], [0.0, 6.0]],
            "eigenvalues": [4.0, 6.0],
            "positive_definite": True,
            "argmin": [3, 4],
            "argmin_agrees_with_pooled": True,
        }
        if k == 4:
            entry = {
                "hessian": [[1.0, 0.5], [0.5, -2.0]],
                "eigenvalues": [-2.0639347, 1.0639347],
                "positive_definite": False,
                "argmin": [0, 8],
                "argmin_agrees_with_pooled": False,
            }
        stability[str(k)] = entry
    return {
        "minimum": {"i": 0, "j": 8, "b_eff_multiplier": 0.5,
                    "b_dec": 2.6},
        "profile_grid": {
            "b_eff_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                  2.0, 2.25, 2.5],
            "b_dec_gev2": [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6],
        },
        "correlation_b_eff_b_dec": None,
        "hessian": {
            "hessian": [[-181606.31, -13089.49],
                        [-13089.49, -935.70]],
            "eigenvalues": [-182549.71, 7.70],
            "positive_definite": False,
            "quoted": False,
            "interior_flags": {"all_interior": False},
        },
        "minima_agreement": {
            "minima_agree": False,
            "pooled_argmin": [0, 8],
            "per_block_argmins": {str(k): [3, 4] for k in range(8)},
        },
        "curvature_stability": stability,
        "per_direction_curvature": {
            "b_eff_multiplier": {
                "pooled": 4.0,
                "per_block": {str(k): 4.0 for k in range(8)},
                "max_abs_spread": 0.0,
                "spread_fraction": 0.0,
                "bound_fraction": 0.20,
                "below_fraction": True,
            },
            "b_dec": {
                "pooled": 6.0,
                "per_block": {**{str(k): 6.0 for k in range(8)},
                              "7": 9.0},
                "max_abs_spread": 3.0,
                "spread_fraction": 0.5,
                "bound_fraction": 0.20,
                "below_fraction": False,
            },
        },
        "objective_at_minimum": {"ess": {"min": 0.004}},
        "nonfinite_surface_points": [{"i": 0, "j": 8}],
    }


def test_named_residual_register_failure_kinds():
    """Every failing acceptance-(ii) condition is a named finding with
    the card's names: boundary_minimum (per axis),
    replica_block_minima_disagree, curvature_not_positive,
    block_curvature_failure (with the block index and eigenvalues
    quoted), weight_degeneracy, and nonfinite_surface_points; the
    correlation entry is suppressed while the minima gate fails (the
    not-quoted curvature is the root cause, named by its own entry)."""
    register = r46._assemble_named_residuals(_failing_report())
    kinds = [entry["kind"] for entry in register]
    assert kinds == [
        "boundary_minimum",          # the b_eff multiplier lower bound
        "boundary_minimum",          # the b_dec upper bound
        "replica_block_minima_disagree",
        "block_curvature_failure",
        "weight_degeneracy",
        "nonfinite_surface_points",
    ]
    boundary = [entry for entry in register
                if entry["kind"] == "boundary_minimum"]
    assert [(entry["axis"], entry["observed"]) for entry in boundary] == [
        ("b_eff_multiplier", 0.5), ("b_dec", 2.6)]
    block_failure = by_kind(register, "block_curvature_failure")
    assert block_failure["observed"]["blocks_not_positive_definite"][
        "4"] == [-2.0639347, 1.0639347]
    spread_failure = block_failure["observed"][
        "per_direction_spread_failures"]["b_dec"]
    assert spread_failure["spread_fraction"] == 0.5
    assert spread_failure["offending_blocks"] == ["7"]
    minima = by_kind(register, "replica_block_minima_disagree")
    assert minima["observed"]["pooled_argmin"] == [0, 8]
    assert "line 311" in minima["diagnosis"]
    # the index sequence is 1..n
    assert [entry["index"] for entry in register] == list(
        range(1, len(register) + 1))


def test_named_residual_register_quoted_curvature_failure():
    """With the replica-block minima agreeing, a quoted non-positive-
    definite pooled Hessian is the curvature_not_positive finding, an
    undefined correlation is correlation_undefined_or_high, and the
    failing block is block_curvature_failure (the eigenvalues quoted)."""
    report = _failing_report()
    report["minima_agreement"] = {
        "minima_agree": True,
        "pooled_argmin": [0, 8],
        "per_block_argmins": {str(k): [0, 8] for k in range(8)},
    }
    report["hessian"]["quoted"] = True
    for entry in report["curvature_stability"].values():
        entry["argmin"] = [0, 8]
        entry["argmin_agrees_with_pooled"] = True
    register = r46._assemble_named_residuals(report)
    kinds = [entry["kind"] for entry in register]
    assert kinds == [
        "boundary_minimum",
        "boundary_minimum",
        "correlation_undefined_or_high",
        "curvature_not_positive",
        "block_curvature_failure",
        "weight_degeneracy",
        "nonfinite_surface_points",
    ]
    curvature = by_kind(register, "curvature_not_positive")
    assert curvature["observed"]["eigenvalues"][0] < 0.0


def test_named_residual_register_clean_report():
    """A passing surface carries no named finding: an interior minimum,
    a defined correlation below the bound, a quoted positive-definite
    pooled Hessian, all blocks positive definite with the spreads below
    the fraction, a healthy ESS, and a finite surface."""
    report = {
        "minimum": {"i": 3, "j": 4, "b_eff_multiplier": 1.25,
                    "b_dec": 1.1},
        "profile_grid": {
            "b_eff_multipliers": [0.5, 2.5],
            "b_dec_gev2": [0.2, 2.6],
        },
        "correlation_b_eff_b_dec": 0.316,
        "hessian": {
            "hessian": [[4.0, 0.5], [0.5, 6.0]],
            "eigenvalues": [3.78, 6.22],
            "positive_definite": True,
            "quoted": True,
            "interior_flags": {"all_interior": True},
        },
        "minima_agreement": {
            "minima_agree": True,
            "pooled_argmin": [3, 4],
            "per_block_argmins": {str(k): [3, 4] for k in range(8)},
        },
        "curvature_stability": _stability_fixture(4.0, 6.0, 0.0),
        "per_direction_curvature": {
            "b_eff_multiplier": {
                "pooled": 4.0,
                "per_block": {str(k): 4.0 for k in range(8)},
                "max_abs_spread": 0.0,
                "spread_fraction": 0.0,
                "bound_fraction": 0.20,
                "below_fraction": True,
            },
            "b_dec": {
                "pooled": 6.0,
                "per_block": {str(k): 6.0 for k in range(8)},
                "max_abs_spread": 0.0,
                "spread_fraction": 0.0,
                "bound_fraction": 0.20,
                "below_fraction": True,
            },
        },
        "objective_at_minimum": {"ess": {"min": 0.42}},
        "nonfinite_surface_points": [],
    }
    assert r46._assemble_named_residuals(report) == []


def test_named_residual_register_boundary_only():
    """A boundary minimum on one axis is named even when every other
    condition passes (the profile range does not extend)."""
    report = _failing_report()
    report["minimum"] = {"i": 8, "j": 3, "b_eff_multiplier": 2.5,
                         "b_dec": 1.1}
    report["correlation_b_eff_b_dec"] = 0.2
    report["hessian"] = {
        "hessian": [[4.0, 0.1], [0.1, 6.0]],
        "eigenvalues": [3.98, 6.02],
        "positive_definite": True,
        "quoted": True,
        "interior_flags": {"all_interior": False},
    }
    report["minima_agreement"] = {
        "minima_agree": True,
        "pooled_argmin": [8, 3],
        "per_block_argmins": {str(k): [8, 3] for k in range(8)},
    }
    report["curvature_stability"] = _stability_fixture(4.0, 6.0, 0.0)
    for entry in report["curvature_stability"].values():
        entry["argmin"] = [8, 3]
        entry["argmin_agrees_with_pooled"] = True
    report["per_direction_curvature"] = {
        "b_eff_multiplier": {"pooled": 4.0,
                             "per_block": {str(k): 4.0 for k in range(8)},
                             "max_abs_spread": 0.0,
                             "spread_fraction": 0.0,
                             "bound_fraction": 0.20,
                             "below_fraction": True},
        "b_dec": {"pooled": 6.0,
                  "per_block": {str(k): 6.0 for k in range(8)},
                  "max_abs_spread": 0.0,
                  "spread_fraction": 0.0,
                  "bound_fraction": 0.20,
                  "below_fraction": True},
    }
    report["objective_at_minimum"] = {"ess": {"min": 0.4}}
    report["nonfinite_surface_points"] = []
    register = r46._assemble_named_residuals(report)
    kinds = [entry["kind"] for entry in register]
    assert kinds == ["boundary_minimum"]
    assert register[0]["axis"] == "b_eff_multiplier"
    assert register[0]["observed"] == 2.5


def test_correlation_bound_register():
    """A defined correlation at or above the pre-registered 0.9 bound is
    the correlation_undefined_or_high finding; the bound is never
    relaxed."""
    report = _failing_report()
    report["minima_agreement"]["minima_agree"] = True
    report["correlation_b_eff_b_dec"] = 0.95
    report["hessian"]["quoted"] = True
    report["hessian"]["positive_definite"] = True
    h01 = -0.95 * math.sqrt(4.0 * 6.0)
    report["hessian"]["hessian"] = [[4.0, h01], [h01, 6.0]]
    report["hessian"]["eigenvalues"] = [1.15, 8.85]
    report["hessian"]["interior_flags"] = {"all_interior": True}
    report["curvature_stability"] = _stability_fixture(4.0, 6.0, 0.0)
    for entry in report["curvature_stability"].values():
        entry["argmin"] = [3, 4]
        entry["argmin_agrees_with_pooled"] = True
    report["per_direction_curvature"] = {
        "b_eff_multiplier": {"pooled": 4.0,
                             "per_block": {str(k): 4.0 for k in range(8)},
                             "spread_fraction": 0.0,
                             "bound_fraction": 0.20,
                             "below_fraction": True},
        "b_dec": {"pooled": 6.0,
                  "per_block": {str(k): 6.0 for k in range(8)},
                  "spread_fraction": 0.0,
                  "bound_fraction": 0.20,
                  "below_fraction": True},
    }
    report["objective_at_minimum"] = {"ess": {"min": 0.4}}
    report["nonfinite_surface_points"] = []
    register = r46._assemble_named_residuals(report)
    kinds = [entry["kind"] for entry in register]
    assert "correlation_undefined_or_high" in kinds
    entry = by_kind(register, "correlation_undefined_or_high")
    assert entry["observed"] == 0.95
    assert entry["bound"] == 0.9


def test_curvature_quoting_gate_semantics():
    """The hessian.quoted flag follows the minima gate: with the minima
    agreeing the curvature is quoted and the correlation defined; with
    the gate failing no correlation is quoted (the stable-curvature
    claim is not established)."""
    agreement = r46.minima_agreement(
        {str(k): _quadratic_surface(2.0, 3.0, 0.0, 3, 4)
         for k in range(8)}, 3, 4)
    assert agreement["minima_agree"] is True
    quoted = bool(agreement["minima_agree"])
    correlation = (r46.correlation_from_hessian([[4.0, 0.5], [0.5, 6.0]])
                   if quoted else None)
    assert correlation == pytest.approx(
        -0.5 / math.sqrt(24.0), rel=1e-12)
    disagreement = r46.minima_agreement(
        {**{str(k): _quadratic_surface(2.0, 3.0, 0.0, 3, 4)
            for k in range(8)},
         "4": _quadratic_surface(2.0, 3.0, 0.0, 5, 8)}, 3, 4)
    quoted = bool(disagreement["minima_agree"])
    assert quoted is False
    assert (r46.correlation_from_hessian([[4.0, 0.5], [0.5, 6.0]])
            if quoted else None) is None


# ---------------------------------------------------------------------------
# The seeds, the smoke block, and the CLI
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    """The new raw block 20301076-20301083, the 900M event-index
    partition, and the +12000 smoke block (seed 20312920, event index
    990_000_000), all disjoint from every previously used block."""
    assert r46.MASTER_SEED == 20260920
    assert r46.RAW_REPLICA_SEEDS == tuple(20301076 + k for k in range(8))
    assert r46.RAW_REPLICA_SEEDS[0] == 20301076
    assert r46.RAW_REPLICA_SEEDS[-1] == 20301083
    assert r46.EVENT_INDEX_RULE == "900_000_000 + i, i < 100,000"
    assert r46.EVENT_INDEX_BASE == 900_000_000
    assert r46.T2_REPLICAS == 8 and r46.T2_EVENTS_PER_REPLICA == 100_000
    assert r46.T2_REPLICAS * r46.T2_EVENTS_PER_REPLICA == 800_000
    assert r46.SMOKE_RAW_SEED == 20260920 + 40_000 + 12_000 == 20312920
    assert r46.SMOKE_KERNEL_REPLICA_SLOT == 12_000
    assert r46.SMOKE_EVENT_INDEX_BASE == 990_000_000
    assert r46.SMOKE_EVENT_INDEX_RULE == "990_000_000 + i, i < 2,000"
    assert r46.SMOKE_RAW_SEED not in r46.RAW_REPLICA_SEEDS
    assert r46.SMOKE_RAW_SEED not in range(20301020, 20301084)
    assert r46.SMOKE_RAW_SEED != 20311920  # the R45 smoke block
    assert r46.SMOKE_RAW_SEED != 20310920  # the R44 smoke block
    assert r46.SMOKE_RAW_SEED != 20309920  # the R42/R43 smoke block
    assert r46.SMOKE_RAW_SEED != 20308920  # the R41 smoke block
    assert "20301076-20301083" in r46.RAW_SEED_RULE
    assert "R40-R44 block 20301020-20301039" in r46.RAW_SEED_RULE
    assert "20301040 + k" in r46.RAW_SEED_RULE
    assert "20301056-20301075" in r46.RAW_SEED_RULE
    assert "derive_substreams(20260920, k, 900_000_000 + i)" in \
        r46.KERNEL_SEED_RULE
    assert [r46.raw_replica_seed(k) for k in range(8)] == [
        20301076 + k for k in range(8)]
    with pytest.raises(SystemExit):
        r46.raw_replica_seed(8)
    assert r46.event_index(0) == 900_000_000


def test_seed_disjointness_report():
    """The seed-block disjointness check passes: the R46 raw block
    shares no raw seed with any previously used block, the audit block
    is disjoint, and the event-index partitions are pairwise disjoint
    (the card-20 R45 800M partition included); any collision refuses
    the run (never a silent re-block)."""
    report = r46.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["audit_block_disjoint"] is True
    assert report["r46_raw_seed_block"] == [20301076, 20301083]
    assert report["r46_event_index_range"] == [900_000_000, 900_100_000]
    assert report["shared_raw_seeds"] == {}
    blocks = report["previously_used_blocks"]
    assert blocks["r45"]["event_index_partition"] == \
        "800_000_000 + i, i < 250,000"
    assert blocks["r45"]["event_index_range"] == [800_000_000, 800_250_000]
    assert blocks["r45"]["smoke_raw_seed"] == 20311920
    assert blocks["r44"]["event_index_range"] == [700_000_000, 700_250_000]
    assert blocks["r43"]["event_index_partition"] == \
        "600_000_000 + i, i < 100,000"
    assert blocks["r42"]["event_index_partition"] == \
        "500_000_000 + i, i < 100,000"
    assert blocks["r41"]["event_index_partition"] == \
        "400_000_000 + i, i < 100,000"
    assert blocks["r40"]["event_index_range"] == [300_000_000, 389_025_000]
    assert blocks["audit_reserved"]["fired"] is False
    # a colliding partition is refused (never a silent re-block)
    saved = copy.deepcopy(r46.PREVIOUS_USED_BLOCKS)
    try:
        collided = copy.deepcopy(r46.PREVIOUS_USED_BLOCKS)
        collided["r45"]["event_index_range"] = [900_000_000, 900_050_000]
        r46.PREVIOUS_USED_BLOCKS.clear()
        r46.PREVIOUS_USED_BLOCKS.update(collided)
        with pytest.raises(SystemExit):
            r46.seed_disjointness_report()
    finally:
        r46.PREVIOUS_USED_BLOCKS.clear()
        r46.PREVIOUS_USED_BLOCKS.update(saved)
    assert r46.AUDIT_RESERVED_BLOCK == (20301040, 20301055)
    assert min(r46.RAW_REPLICA_SEEDS) == 20301076 > 20301075
    assert r46.event_index(0) == 900_000_000


def test_stream_tree_disjointness():
    """The (master, replica slot, event index) exchange spaces are
    pairwise disjoint across the R46 production slots and the smoke
    slot, and disjoint from the card-20 R45 (800M), R44 (700M), R43
    (600M), R42 (500M), R41 (400M), and R40 (300M) spaces."""
    from na49_mixing.rng_substreams import derive_substreams

    labels = ("topology_selection", "exchange_kinematics",
              "spectral_state", "decay_channel", "decay_kinematics")
    streams = set()
    for replica in range(r46.T2_REPLICAS):
        subs = derive_substreams(
            r46.MASTER_SEED, replica, r46.EVENT_INDEX_BASE)
        for label in labels:
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    subs = derive_substreams(
        r46.MASTER_SEED, r46.SMOKE_KERNEL_REPLICA_SLOT,
        r46.SMOKE_EVENT_INDEX_BASE)
    for label in labels:
        assert int(subs.stream_for(label)) not in streams
    for replica in range(r46.T2_REPLICAS):
        for base in (800_000_000, 700_000_000, 600_000_000,
                     500_000_000, 400_000_000, 300_000_000):
            prior = derive_substreams(r46.MASTER_SEED, replica, base)
            for label in labels:
                assert int(prior.stream_for(label)) not in streams
    seq = lambda seed, event: np.random.SeedSequence(  # noqa: E731
        seed, spawn_key=(event,))
    for seed in (20301076, 20301083):
        r46_stream = seq(seed, r46.EVENT_INDEX_BASE).generate_state(1)
        for event in (800_000_000, 700_000_000, 600_000_000,
                      500_000_000, 400_000_000, 300_000_000):
            assert seq(seed, event).generate_state(1) != r46_stream


def test_cli_surface():
    """The seven-mode CLI surface (exactly one mode per invocation)."""
    parser = r46._build_parser()
    modes = {
        "--pre-registration", "--smoke", "--generate", "--profile",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {"--replica", "--run-dir", "--workers"}
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    assert seen == modes | auxiliary | {"-h", "--help"}, (
        f"the CLI surface drifted: {sorted(seen)}")
    with pytest.raises(SystemExit) as exc:
        r46.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        r46.main(["--profile", "--gate"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and sit
    at the driver's self-computed identity digest."""
    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r46_simplified_recoil_decay_profile.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r46_simplified_recoil_decay_profile.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest
    assert nested_digest == r46.DRIVER_SHA256


# ---------------------------------------------------------------------------
# The smoke refusal convention
# ---------------------------------------------------------------------------
def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r46.MONO_ROOT / r46.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r46.run_smoke(default_dir, workers=1)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention)."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r46.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r46._refuse_smoke_artifacts(run_dir)
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r46.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_replica_0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r46._refuse_smoke_artifacts(run_dir2)
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r46._refuse_smoke_artifacts(run_dir3)


def test_smoke_payload_constants():
    """The smoke payload: 1 replica x 2,000 events at seed 20312920 on
    the +90M smoke partition (990_000_000 + i), with the synthetic
    truth point and the pinned validity grid."""
    payload = r46.SMOKE_PAYLOAD
    assert payload["replicas"] == 1
    assert payload["events_per_replica"] == 2_000
    assert payload["raw_seed"] == 20312920
    assert payload["kernel_replica_slot"] == 12_000
    assert payload["synthetic_truth"] == {
        "b_eff_multiplier": 1.3, "b_dec": 1.2}
    assert payload["validity_events"] == 200
    assert payload["validity_grid_points"] == [
        [1.5, 1.7], [2.0, 0.5]]
    assert r46.SMOKE_EVENT_INDEX_BASE == 990_000_000
    assert r46.SMOKE_EVENT_INDEX_RULE == "990_000_000 + i, i < 2,000"
    assert r46.SMOKE_VALIDITY_EVENTS == 200


# ---------------------------------------------------------------------------
# The hessian-discipline constants and the optimizer pins
# ---------------------------------------------------------------------------
def test_hessian_discipline_and_optimizer_pins():
    """The plan-5.4-line-311 quoting gate, the 20% per-direction spread
    bound, the 0.9 correlation bound, and the obligation-4 form are the
    pre-registered constants; the blend ramp start stays fixed at B."""
    assert r46.CURVATURE_SPREAD_FRACTION == 0.20
    assert r46.CORRELATION_BOUND == 0.9
    assert "distinguishable-effects" in r46.CORRELATION_NOTE
    assert "line 311" in r46.HESSIAN_DISCIPLINE_NOTE
    assert "20%" in r46.HESSIAN_DISCIPLINE_NOTE
    assert "never relaxed" in r46.CURVATURE_SPREAD_NOTE
    assert r46.ESS_FLOOR_FRACTION == 0.01
    assert r46.SIGMA_BAND == 3.0
    assert r46.BLEND_RAMP_START_GEV == 2.680194110727528
    assert "blend-band-sensitive bins" in r46.OBLIGATION4_FORM
    assert "the original R32 spread stands" in r46.OBLIGATION4_FORM


# ---------------------------------------------------------------------------
# The freeze re-pin register (the card-21 landing)
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r46_simplified_recoil_decay_profile.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py"
)
MY_PINS_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py"
)
SOLVER_CARD = "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile"

#: The twenty-three chain count-pin files whose inventory-count pins
#: this card's re-pin bump (495 -> 499, +4 entries): the card-20 chain
#: (twenty-one files, cards 1-15's registers plus the card-13/14/15/19
#: solver tests' own count constants and the card-11/12/13/14
#: test-author registers' own count constants) plus the card-20 solver
#: test's and the card-20 test-author register's own count constants.
#: Every chain file carries the card-21 naming comment appended to the
#: card-20 history.
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
)
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned in place by card 19 (no new files; the count stays 491).
# re-pinned 491 -> 495 by card 20 (+4 entries: the dual-tree R45
# driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-
# recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both
# helper trees and the two new R46 test files).
# re-pinned 499 -> 501 by Round 5 card 1 (TASK-20260925-r5-elastic-reference; +2 entries: the elastic reference module na49_mixing/topology_elastic.py and the new elastic test file).
# re-pinned 501 -> 502 by Round 5 card 1 test-author (TASK-20260925-r5-elastic-reference-test-author; +1 entry: the card-1 test-author register file).
# re-pinned 502 -> 504 by Round 5 card 2 (TASK-20260925-r5-exchange-quantum-number-ledger; +2 entries: the exchange quantum-number ledger module na49_mixing/exchange_ledger.py and the new ledger test file tests/test_exchange_ledger.py).
# re-pinned 504 -> 505 by Round 5 card 2 test-author (TASK-20260925-r5-exchange-quantum-number-ledger-test-author; +1 entry: the card-2 test-author register file).
# re-pinned 505 -> 508 by the test-protocol tiering change (2026-09-25; +3 entries: helpers/benchmarking/select_tests_for_changes.py, its vendored copy, and helpers/benchmarking/tests/test_select_tests_for_changes.py).
# re-pinned 508 -> 510 by Round 5 card 3 (TASK-20260925-r5-intact-low-mass-remnants; +2 entries: the remnant module na49_mixing/topology_remnants.py and the new remnant test file tests/test_topology_remnants.py).
# re-pinned 510 -> 511 by Round 5 card 3 test-author (TASK-20260925-r5-intact-low-mass-remnants-test-author; +1 entry: the card-3 test-author register file tests/test_topology_remnants_contract_pins.py).
# re-pinned 511 -> 513 by Round 5 card 4 (TASK-20260925-r5-topology-mass-laws; +2 entries: the mass-laws module na49_mixing/topology_mass_laws.py and the new mass-laws test file tests/test_topology_mass_laws.py).
# re-pinned 513 -> 514 by Round 5 card 4 test-author (TASK-20260925-r5-topology-mass-laws-test-author; +1 entry: the card-4 test-author register file tests/test_topology_mass_laws_contract_pins.py).
# re-pinned 514 -> 516 by Round 5 card 5 (TASK-20260925-r5-single-diffraction; +2 entries: the single-diffraction module na49_mixing/topology_single_diffraction.py and the new single-diffraction test file tests/test_topology_single_diffraction.py).
# re-pinned 516 -> 517 by Round 5 card 5 test-author (TASK-20260925-r5-single-diffraction-test-author; +1 entry: the card-5 test-author register file tests/test_topology_single_diffraction_contract_pins.py).
# re-pinned 519 -> 520 by Round 5 card 6 test-author (TASK-20260925-r5-double-diffraction-test-author; +1 entry: the card-6 test-author register file tests/test_topology_double_diffraction_contract_pins.py).
# re-pinned 520 -> 522 by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; +2 entries: the module na49_mixing/topology_non_diffractive.py and the test file tests/test_topology_non_diffractive.py).
# re-pinned 522 -> 523 by Round 5 card 7 test-author (TASK-20260925-r5-non-diffrative-central-excitation-test-author; +1 entry: the card-7 test-author register file tests/test_topology_non_diffractive_contract_pins.py).
# re-pinned 523 -> 525 by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; +2 entries: the module na49_mixing/topology_multi_neutral.py and the test file tests/test_topology_multi_neutral.py).
# re-pinned 525 -> 526 by Round 5 card 8 test-author (TASK-20260925-r5-multiple-neutral-excitations-test-author; +1 entry: the card-8 test-author register file tests/test_topology_multi_neutral_contract_pins.py).
# re-pinned 526 -> 528 by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; +2 entries: the runner module na49_mixing/topology_runner.py and the test file tests/test_topology_runner.py).
# re-pinned 528 -> 529 by Round 5 card 9 test-author (TASK-20260925-r5-topology-template-runner-test-author; +1 entry: the card-9 test-author register file tests/test_topology_runner_contract_pins.py).
# re-pinned 529 -> 532 by Round 5 card 10 (TASK-20260925-r5-run-r50-pure-topology-templates; +3 entries: the dual-tree driver helpers/benchmarking/staged_r50_pure_topology_templates.py, its nested copy, and the test file tests/test_staged_r50_pure_topology_templates.py).
# re-pinned 532 -> 533 by Round 5 card 10 test-author (TASK-20260925-r5-run-r50-pure-topology-templates-test-author; +1 entry: the card-10 test-author register file tests/test_staged_r50_pure_topology_templates_contract_pins.py).
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-topology-fraction-fit; +3 entries: the dual-tree driver helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested copy, and the test file tests/test_staged_r51_topology_fraction_fit.py).
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()