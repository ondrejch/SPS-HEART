"""Hermetic tests for the R45 driver (the simplified-alignment freeze).

Task ``TASK-20260920-r4-gate4-followup-simplified-alignment`` (the
Gate 4 follow-up obligation 1; the written Gate 4 decision
``TASK-20260920-r4-gate4-decision``), phase 1 (no Monte Carlo): the
frozen-parameter consistency test (the frozen tuple equals the sealed
corrected R41 fit output exactly, the fit-report digest matches the
sealed archive byte for byte), the constant-kappa / mass-independence
tests (the alignment weight is independent of parent mass at fixed
angle; no mass-dependence coefficient differs from zero; the committed
kernel signatures carry no mass argument), the conservation and
determinism pins, the pre-registered ranges (unchanged, never
extended, never relaxed), the no-optimizer pin, the CLI surface, the
layout sync, and the freeze re-pin register.  No Monte Carlo
generation at run scale, no SPS table, no sealed archive read by the
acceptance legs (the sealed-dir pins skip on a fresh checkout): the
acceptance is hermetic.
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
        / "staged_r45_simplified_alignment_freeze.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r45_simplified_alignment_freeze", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R45 driver was not found next to the test file")


r45 = _load_driver()


# ---------------------------------------------------------------------------
# The simplified-form pin (the card's acceptance (ii))
# ---------------------------------------------------------------------------
def test_simplified_form_import_pin():
    """The import-time constant-kappa enforcement ran: the simplified
    form is constant kappa_B / kappa_M, every mass-dependence
    coefficient is exactly zero, and the enforcement status carries the
    mass-free kernel signatures and the flat mass scan."""
    status = r45._SIMPLIFIED_FORM_STATUS
    assert status["pass"] is True
    assert status["simplified_form"] == {
        "kappa_B": "constant", "kappa_M": "constant",
        "mass_dependence": "fixed at zero, not fit",
    }
    assert status["mass_dependence_coefficients"] == {
        "kappa_b_mass_slope_per_gev": 0.0,
        "kappa_m_mass_slope_per_gev": 0.0,
    }
    assert "constant" in status["note"]
    assert "plan section 10.4" in status["note"]
    # the committed kernels carry no mass argument
    for name, params in status["kernel_signatures_mass_free"].items():
        assert all("mass" not in p.lower() for p in params), (name, params)
    assert status["kernel_signature_note"]
    scan = status["mass_independence_scan"]
    assert scan["pass"] is True
    assert scan["worst_mass_spread"] == 0.0
    assert all(row["zero_mass_slope"] is True for row in scan["rows"])
    # the scan grid: twenty parent masses, eleven fixed angles, both laws
    assert len(scan["mass_grid"]) == 20
    assert len(scan["angles"]) == 11
    assert len(scan["rows"]) == 22


def test_mass_independence_scan_explicit():
    """The dedicated scan (the card: a dedicated test scans parent mass
    at fixed cos(theta) and asserts zero mass slope) at a custom mass
    grid and angle set: the alignment weight is identical at every
    parent mass for both laws, at every fixed angle."""
    masses = [0.3, 0.5, 0.9382720, 1.4, 2.0, 3.5, 6.0]
    angles = (-1.0, -0.5, 0.0, 0.5, 1.0)
    scan = r45.mass_independence_scan(masses=masses, angles=angles)
    assert scan["pass"] is True
    assert scan["worst_mass_spread"] == 0.0
    for row in scan["rows"]:
        assert row["zero_mass_slope"] is True
        assert row["mass_spread"] == 0.0
    # the weights themselves are the normalized kernel densities at the
    # frozen constant kappa values
    for row in scan["rows"]:
        c = row["cos_theta"]
        if row["kind"] == "baryonic":
            expected = (math.exp(r45.ALIGNED_POLICY.kappa_b * c)
                        / r45.cascade_alignment.baryonic_normalization(
                            r45.ALIGNED_POLICY.kappa_b))
        else:
            expected = (math.exp(r45.ALIGNED_POLICY.kappa_m * c * c)
                        / r45.cascade_alignment.mesonic_normalization(
                            r45.ALIGNED_POLICY.kappa_m))
        assert row["weight_at_first_mass"] == pytest.approx(
            expected, rel=1e-15)
    # no mass-dependence coefficient differs from zero (the card)
    assert all(float(v) == 0.0
               for v in r45.MASS_DEPENDENCE_COEFFICIENTS.values())


def test_mass_independence_refusals(monkeypatch):
    """A nonzero mass-dependence coefficient or a mass-dependent kernel
    signature refuses the run (checked, never repaired)."""
    # a nonzero coefficient is refused
    saved = dict(r45.MASS_DEPENDENCE_COEFFICIENTS)
    try:
        r45.MASS_DEPENDENCE_COEFFICIENTS[
            "kappa_b_mass_slope_per_gev"] = 0.1
        with pytest.raises(SystemExit):
            r45.mass_independence_scan()
    finally:
        r45.MASS_DEPENDENCE_COEFFICIENTS.clear()
        r45.MASS_DEPENDENCE_COEFFICIENTS.update(saved)
    # a mass argument on a kernel signature is refused (the live kernel
    # attribute is monkeypatched; the real module attribute is restored)
    def _mass_kernel(c, kappa, parent_mass_gev=0.0):
        return 0.0
    monkeypatch.setattr(
        r45.cascade_alignment, "baryonic_kernel_density", _mass_kernel)
    with pytest.raises(SystemExit):
        r45.assert_constant_kappa_form()
    # a frozen value off its pre-registered bound is refused
    monkeypatch.setattr(
        r45.cascade_alignment, "baryonic_kernel_density",
        r45.cascade_alignment.baryonic_kernel_density, raising=False)
    monkeypatch.undo()
    saved_kappa = r45.ALIGNED_POLICY.kappa_b
    try:
        r45.ALIGNED_POLICY.kappa_b = 0.5
        with pytest.raises(SystemExit):
            r45.assert_constant_kappa_form()
    finally:
        r45.ALIGNED_POLICY.kappa_b = saved_kappa
    # bad inputs raise ValueError
    with pytest.raises(ValueError):
        r45.alignment_weight_at(0.5, 1.0, "bogus")
    with pytest.raises(ValueError):
        r45.alignment_weight_at(1.5, 1.0, "baryonic")
    with pytest.raises(ValueError):
        r45.alignment_weight_at(0.5, 0.0, "baryonic")


def test_alignment_weight_no_mass_argument():
    """The alignment-weight helper pins its mass argument unused: the
    weight at a fixed angle is bit-identical across the mass grid (the
    card's zero-mass-slope clause; the committed kernels are the plan
    section 10.4 mass-independent angular laws)."""
    for c_value in (-1.0, -0.3, 0.0, 0.4, 1.0):
        for kind in ("baryonic", "mesonic"):
            weights = [r45.alignment_weight_at(c_value, m, kind)
                       for m in (0.3, 1.0, 2.5, 8.0)]
            assert all(w == weights[0] for w in weights)


def test_pre_registered_ranges_unchanged():
    """The pre-registered ranges are unchanged (the card): kappa_B in
    [0.0, 4.0], kappa_m in [0.0, 4.0], b_dec in [0.0, 2.0] GeV^-2,
    baryonic nominal_width in [0.3, 0.7] GeV; no range extension; the
    frozen values sit on these bounds by construction; the bounds are
    never relaxed."""
    assert r45.PRE_REGISTERED_RANGES == {
        "kappa_B": (0.0, 4.0),
        "kappa_m": (0.0, 4.0),
        "b_dec_gev2": (0.0, 2.0),
        "baryonic_nominal_width_gev": (0.3, 0.7),
    }
    record = r45.simplified_form_record()
    assert record["pre_registered_ranges"] == {
        "kappa_B": [0.0, 4.0],
        "kappa_m": [0.0, 4.0],
        "b_dec_gev2": [0.0, 2.0],
        "baryonic_nominal_width_gev": [0.3, 0.7],
    }
    # the simplified-form import check carries the frozen values on
    # their pre-registered bounds
    on_bounds = r45._SIMPLIFIED_FORM_STATUS["frozen_values_on_bounds"]
    assert on_bounds["kappa_B"] == {"value": 0.0, "range": [0.0, 4.0]}
    assert on_bounds["kappa_m"] == {"value": 4.0, "range": [0.0, 4.0]}
    assert on_bounds["b_dec"] == {"value": 2.0, "range": [0.0, 2.0]}
    assert on_bounds["baryonic_nominal_width"] == {
        "value": 0.3, "range": [0.3, 0.7]}
    assert "never" in record["pre_registered_ranges_note"]


# ---------------------------------------------------------------------------
# The frozen-parameter consistency (the card's acceptance (i))
# ---------------------------------------------------------------------------
def test_frozen_parameter_assertion_pinned_fixture():
    """The frozen parameters (kappa_B, kappa_m, b_dec, baryonic
    nominal_width) are the corrected R41 fitted values (0.0, 4.0,
    2.0 GeV^-2, 0.3 GeV; the sealed r41_nonstrange_pilot_fit_
    2026-09-23 archive), pinned as a fixture and asserted by the same
    predicate the production pre-registration runs against the sealed
    archive; the aligned policy carries them and refit is false."""
    frozen = r45.assert_frozen_parameters(r45.R41_FIT_REPORT_FIXTURE)
    assert frozen["kappa_B"] == 0.0
    assert frozen["kappa_m"] == 4.0
    assert frozen["b_dec_gev2"] == 2.0
    assert frozen["baryonic_nominal_width_gev"] == 0.3
    assert frozen["refit"] is False
    assert "R41" in frozen["source"]
    assert r45.R41_FITTED == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
        "baryonic_nominal_width": 0.3,
    }
    assert r45.FROZEN_WIDTH_GEV == 0.3
    assert (r45.ALIGNED_POLICY.kappa_b, r45.ALIGNED_POLICY.kappa_m,
            r45.ALIGNED_POLICY.b_dec) == (0.0, 4.0, 2.0)
    assert r45.ALIGNED_POLICY.variant_id == 0
    assert r45.ALIGNED_POLICY.name == "aligned_simplified"
    # the simplified-form record carries the sealed R41 source digests
    record = r45.simplified_form_record()
    assert record["frozen_source"]["file_sha256"] == \
        r45.R41_FIT_REPORT_FILE_SHA256
    assert record["frozen_source"]["self_digest"] == \
        r45.R41_FIT_REPORT_DIGEST
    assert record["no_fitting"] is True
    assert record["optimizer_invoked"] is False
    # the sealed R41 archive cross-check (read-only; skips on a fresh
    # checkout without the sealed archive)
    fit_path = r45.MONO_ROOT / r45.R41_RUN_DIR / r45.R41_FIT_REPORT_NAME
    if fit_report_is_sealed():
        report = json.loads(
            (r45.MONO_ROOT / r45.R41_RUN_DIR
             / r45.R41_FIT_REPORT_NAME).read_text(encoding="utf-8"))
        frozen_sealed = r45.assert_frozen_parameters(report)
        assert frozen_sealed_values(frozen_sealed) == \
            frozen_sealed_values(frozen)
        assert r45.R41_FIT_REPORT_DIGEST == report["digest"]
        assert r45.R41_FIT_REPORT_FILE_SHA256 == _sha256_file(fit_path)
        # the sealed tuple is byte-identical to the frozen tuple (the
        # card's frozen-parameter consistency clause)
        assert tuple(report["fitted"][k] for k in (
            "kappa_B", "kappa_m", "b_dec", "baryonic_nominal_width")) \
            == (0.0, 4.0, 2.0, 0.3)


def fit_report_is_sealed() -> bool:
    """The sealed R41 fit report is present on this checkout."""
    path = r45.MONO_ROOT / r45.R41_RUN_DIR / r45.R41_FIT_REPORT_NAME
    return path.is_file()


def frozen_sealed_values(frozen: dict) -> tuple:
    return (
        float(frozen["kappa_B"]), float(frozen["kappa_m"]),
        float(frozen["b_dec_gev2"]),
        float(frozen["baryonic_nominal_width_gev"]),
    )


def test_frozen_parameter_assertion_refusals():
    """Any drift in the frozen-parameter source refuses the run: a
    drifted fitted value, a wrong card, a wrong run, a non-production
    mode, a missing fitted block, and a width drift."""
    fixture = r45.R41_FIT_REPORT_FIXTURE
    drifted = dict(fixture)
    drifted["fitted"] = dict(fixture["fitted"], kappa_B=0.1)
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(drifted)
    wrong_card = dict(fixture, card="TASK-20260920-r4-run-r42-")
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(wrong_card)
    wrong_run = dict(fixture, run="R42")
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(wrong_run)
    smoke_mode = dict(fixture, mode="smoke")
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(smoke_mode)
    no_block = {"card": fixture["card"], "run": "R41",
                "mode": "production"}
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(no_block)
    width_drift = dict(fixture)
    width_drift["fitted"] = dict(fixture["fitted"],
                                 baryonic_nominal_width=0.4)
    with pytest.raises(SystemExit):
        r45.assert_frozen_parameters(width_drift)


def test_no_optimizer_pin():
    """The card's fit-set clause: no_fitting true, optimizer_invoked
    false everywhere they are recorded (the round's two pre-registered
    optimizer uses were consumed by cards 16 and 17)."""
    pre, _frozen, _disjointness = r45._pre_registration_payload(smoke=True)
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False
    record = r45.simplified_form_record()
    assert record["no_fitting"] is True
    assert record["optimizer_invoked"] is False
    assert "card 16 the first, card 17 the second and final" in \
        record["optimizer_note"]
    # the blend ramp start stays fixed at B (obligation 4)
    assert r45.BLEND_RAMP_START_GEV == 2.680194110727528
    assert "obligation 4" in r45.BLEND_RAMP_START_STATUS


# ---------------------------------------------------------------------------
# The run identity, the substrate, and the sealed-archive pins
# ---------------------------------------------------------------------------
def test_run_identity_constants():
    """The R45 run identity and the frozen-substrate byte stability of
    every pinned surface (the corrected R41/R42/R43 driver digests, the
    cascade surfaces, the builder, the card-6 routing, the catalog, the
    continuum config)."""
    assert r45.RUN_ID == "R45"
    assert r45.CARD == "TASK-20260920-r4-gate4-followup-simplified-alignment"
    assert r45.RUN_NAME == "r45_simplified_alignment_freeze_2026-09-24"
    assert r45.DEFAULT_RUN_DIR == (
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24")
    assert r45.GATE_RECORD_NAME == "gate4_followup_simplified.json"
    assert r45.REPORT_NAME == "frozen_report.json"
    # the corrected sibling substrate pins (the card-16/17/18 digests)
    assert r45.FROZEN_SUBSTRATE["r41_driver"] == (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py",
        "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc",
    )
    assert r45.FROZEN_SUBSTRATE["r42_driver"] == (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py",
        "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd",
    )
    assert r45.FROZEN_SUBSTRATE["r43_driver"] == (
        "helpers/benchmarking/staged_r43_cascade_ablation.py",
        "dfb5ee3f4ee6598698faaf3da641fbf315ba21d95f2203792a2414a72e11801c",
    )
    status = r45.verify_frozen_substrate()
    assert set(status) == set(r45.FROZEN_SUBSTRATE)
    assert all(entry["verified"] is True for entry in status.values())
    # the sealed-archive pins include the corrected R44 archive (the
    # prior frozen-point validation at the same tuple)
    assert "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23" in \
        r45.SEALED_RUN_DIRS
    assert r45.R44_RUN_DIR == \
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"
    assert r45.R44_GATE_NAME == "gate4_r44.json"
    # the card chain names the Gate 4 decision and this card
    assert r45.CARD_CHAIN[-2:] == (
        "TASK-20260920-r4-gate4-decision",
        "TASK-20260920-r4-gate4-followup-simplified-alignment",
    )


def test_sealed_archives_verify():
    """The sealed R30-R32 and R40-R44 archives verify read-only on this
    checkout (their seals are intact; skips on a fresh checkout)."""
    if not all(
        (r45.MONO_ROOT / rel / r45.SHA256SUMS_NAME).is_file()
        for rel in r45.SEALED_RUN_DIRS
    ):
        pytest.skip("the sealed archives are absent on this checkout")
    seals = r45.verify_sealed_inputs()
    for rel, status in seals.items():
        assert status["verified"] is True and status["mismatches"] == 0, rel
    # the corrected R41 fit-report digest pins (byte for byte)
    fit_path = r45.MONO_ROOT / r45.R41_RUN_DIR / r45.R41_FIT_REPORT_NAME
    assert r45.R41_FIT_REPORT_FILE_SHA256 == (
        "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
    )
    assert r45.R41_FIT_REPORT_FILE_SHA256 == _sha256_file(fit_path)


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    assert r45.MASTER_SEED == 20260920
    assert r45.RAW_REPLICA_SEEDS == tuple(20301056 + k for k in range(20))
    assert r45.RAW_REPLICA_SEEDS[0] == 20301056
    assert r45.RAW_REPLICA_SEEDS[-1] == 20301075
    assert r45.EVENT_INDEX_RULE == "800_000_000 + i, i < 250,000"
    assert r45.EVENT_INDEX_BASE == 800_000_000
    assert r45.T3_REPLICAS == 20 and r45.T3_EVENTS_PER_REPLICA == 250_000
    assert r45.T3_REPLICAS * r45.T3_EVENTS_PER_REPLICA == 5_000_000
    assert [r45.raw_replica_seed(k) for k in range(20)] == [
        20301056 + k for k in range(20)]
    with pytest.raises(SystemExit):
        r45.raw_replica_seed(20)
    # the reserved audit block 20301040 + k (k = 0..15) is disjoint from
    # the raw block and from the smoke seed
    assert r45.AUDIT_RESERVED_BLOCK == (20301040, 20301055)
    assert min(r45.RAW_REPLICA_SEEDS) == 20301056 > 20301055
    assert r45.SMOKE_RAW_SEED == 20260920 + 40_000 + 11_000 == 20311920
    assert r45.SMOKE_KERNEL_REPLICA_SLOT == 11_000
    assert r45.SMOKE_RAW_SEED not in r45.RAW_REPLICA_SEEDS
    assert r45.SMOKE_RAW_SEED not in range(20301040, 20301056)
    assert r45.SMOKE_RAW_SEED != 20310920  # the R44 smoke block
    assert r45.SMOKE_RAW_SEED != 20309920  # the R42/R43 smoke block
    assert r45.SMOKE_RAW_SEED != 20308920  # the R41 smoke block
    assert r45.SMOKE_EVENT_INDEX_BASE == 890_000_000
    assert r45.SMOKE_EVENT_INDEX_RULE == "890_000_000 + i, i < 2,000"
    # the production event-index partition is disjoint from R44's
    # (700M-based), R43's (600M-based), R42's (500M-based), R41's
    # (400M-based), and R40's (300M-based, below 389,025,000)
    assert r45.EVENT_INDEX_BASE == 800_000_000
    assert r45.EVENT_INDEX_BASE > 700_000_000
    # the rules name the disjointness and the new block
    assert "20301056-20301075" in r45.RAW_SEED_RULE
    assert "R40-R44 block 20301020-20301039" in r45.RAW_SEED_RULE
    assert "20301040 + k" in r45.RAW_SEED_RULE
    assert "derive_substreams(20260920, k, 800_000_000 + i)" in \
        r45.EXCHANGE_SEED_RULE
    assert "variant_stream(20260920, 800_000_000 + i, v)" in \
        r45.VARIANT_STREAM_RULE
    # the previously-used blocks register carries the sealed R44 block
    blocks = r45.PREVIOUS_USED_BLOCKS
    assert blocks["r44"]["event_index_partition"] == \
        "700_000_000 + i, i < 250,000"
    assert blocks["r44"]["source"].endswith(
        "r44_frozen_nonstrange_validation_2026-09-23/pre_registration.json")
    assert blocks["r43"]["event_index_partition"] == \
        "600_000_000 + i, i < 100,000"
    assert blocks["r42"]["event_index_partition"] == \
        "500_000_000 + i, i < 100,000"
    assert blocks["r41"]["event_index_partition"] == \
        "400_000_000 + i, i < 100,000"
    assert blocks["r40"]["event_index_range"] == [300_000_000, 389_025_000]
    assert blocks["audit_reserved"]["raw_seeds"] == [20301040, 20301055]
    assert blocks["audit_reserved"]["fired"] is False


def test_seed_disjointness_report():
    """The seed-block disjointness check passes: the R45 raw block
    shares no raw seed with any previously used block, the audit block
    is disjoint, and the event-index partitions are pairwise disjoint;
    any substream collision refuses the run (never a silent re-block)."""
    report = r45.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["audit_block_disjoint"] is True
    assert report["r45_raw_seed_block"] == [20301056, 20301075]
    assert report["r45_event_index_range"] == [800_000_000, 800_250_000]
    assert report["shared_raw_seeds"] == {}
    # a colliding partition is refused (never a silent re-block)
    saved = dict(r45.PREVIOUS_USED_BLOCKS)
    try:
        collided = copy.deepcopy(r45.PREVIOUS_USED_BLOCKS)
        collided["r44"]["event_index_range"] = [800_000_000, 800_100_000]
        r45.PREVIOUS_USED_BLOCKS.clear()
        r45.PREVIOUS_USED_BLOCKS.update(collided)
        with pytest.raises(SystemExit):
            r45.seed_disjointness_report()
    finally:
        r45.PREVIOUS_USED_BLOCKS.clear()
        r45.PREVIOUS_USED_BLOCKS.update(saved)
    # the raw seeds helpers
    assert r45.raw_replica_seed(0) == 20301056
    assert r45.raw_replica_seed(19) == 20301075
    with pytest.raises(SystemExit):
        r45.raw_replica_seed(20)
    assert r45.event_index(0) == 800_000_000
    assert r45.smoke_event_index(1) == 890_000_001
    assert r45.smoke_raw_seed() == 20311920


def test_stream_tree_disjointness():
    """The (master, replica slot, event index) exchange spaces are
    pairwise disjoint across the R45 production slots and the smoke
    slot, and disjoint from the R44 (700M), R43 (600M), R42 (500M),
    R41 (400M), and R40 (300M) spaces; the variant-spawn tree is a
    separate tree (card 10's contract)."""
    from na49_mixing.rng_substreams import derive_substreams

    labels = ("topology_selection", "exchange_kinematics",
              "spectral_state", "decay_channel", "decay_kinematics")
    streams = set()
    for replica in range(r45.T3_REPLICAS):
        subs = derive_substreams(
            r45.MASTER_SEED, replica, r45.EVENT_INDEX_BASE)
        for label in labels:
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    subs = derive_substreams(
        r45.MASTER_SEED, r45.SMOKE_KERNEL_REPLICA_SLOT,
        r45.SMOKE_EVENT_INDEX_BASE)
    for label in labels:
        assert int(subs.stream_for(label)) not in streams
    for replica in range(r45.T3_REPLICAS):
        r44_subs = derive_substreams(
            r45.MASTER_SEED, replica, 700_000_000)
        r43_subs = derive_substreams(
            r45.MASTER_SEED, replica, 600_000_000)
        r42_subs = derive_substreams(
            r45.MASTER_SEED, replica, 500_000_000)
        r41_subs = derive_substreams(
            r45.MASTER_SEED, replica, 400_000_000)
        r40_subs = derive_substreams(
            r45.MASTER_SEED, replica, 300_000_000)
        for label in labels:
            assert int(r44_subs.stream_for(label)) not in streams
            assert int(r43_subs.stream_for(label)) not in streams
            assert int(r42_subs.stream_for(label)) not in streams
            assert int(r41_subs.stream_for(label)) not in streams
            assert int(r40_subs.stream_for(label)) not in streams
    # the builder's legacy generator spawn keys are disjoint across the
    # event-index partitions (the same raw seed, different blocks)
    seq = lambda seed, event: np.random.SeedSequence(  # noqa: E731
        seed, spawn_key=(event,))
    for seed in (20301056, 20301075):
        r45_stream = seq(seed, r45.EVENT_INDEX_BASE).generate_state(1)
        for event in (700_000_000, 600_000_000, 500_000_000,
                      400_000_000, 300_000_000):
            assert seq(seed, event).generate_state(1) != \
                seq(seed, r45.EVENT_INDEX_BASE).generate_state(1)


# ---------------------------------------------------------------------------
# The variant policies (the two configurations on identical events)
# ---------------------------------------------------------------------------
def test_variant_policies():
    """The card's two configurations: the frozen simplified point (the
    corrected R41 fitted values under the constant-kappa form, no
    refit) and the card-10 v2 isotropic baseline (read from the frozen
    card-10 register); both record the corrected frozen nominal width
    0.3 GeV."""
    register = r45.VARIANT_NAMES
    assert register == ("aligned_simplified",
                        "isotropic_sequential_baseline")
    assert r45.ALIGNED_POLICY.variant_id == 0
    assert r45.BASELINE_POLICY.variant_id == 1
    assert (r45.BASELINE_POLICY.kappa_b, r45.BASELINE_POLICY.kappa_m,
            r45.BASELINE_POLICY.b_dec) == (0.0, 0.0, 1.0)
    from na49_mixing.cascade_ablation import ABLATION_POLICIES

    v2 = ABLATION_POLICIES["isotropic_sequential"]
    assert (r45.BASELINE_POLICY.kappa_b, r45.BASELINE_POLICY.kappa_m,
            r45.BASELINE_POLICY.b_dec) == (
        float(v2.kappa_b), float(v2.kappa_m), float(v2.b_dec))
    # the aligned variant is the frozen R41 point, not the R43 nominal
    assert (r45.ALIGNED_POLICY.kappa_b, r45.ALIGNED_POLICY.kappa_m,
            r45.ALIGNED_POLICY.b_dec) != (1.0, 1.0, 1.0)


def test_variant_stream_disjointness_pinned_key():
    """The per-variant decay streams are child spawns keyed on the
    variant id, disjoint across the two variants, deterministic on the
    pinned key, and sensitive to the event index and the master seed
    (card 10's variant_stream at the production event-index base)."""
    pinned = r45.EVENT_INDEX_BASE
    for policy in r45.VARIANTS:
        first = r45.variant_stream(r45.MASTER_SEED, pinned,
                                   policy.variant_id)
        second = r45.variant_stream(r45.MASTER_SEED, pinned,
                                    policy.variant_id)
        for gen_a, gen_b in zip(first, second):
            assert [gen_a.random() for _ in range(6)] == [
                gen_b.random() for _ in range(6)]
    streams = {
        policy.variant_id: r45.variant_stream(
            r45.MASTER_SEED, pinned, policy.variant_id)
        for policy in r45.VARIANTS
    }
    states_before = {
        vid: tuple(gen.bit_generator.state for gen in pair)
        for vid, pair in streams.items()
    }
    for _ in range(16):
        streams[0][0].random()
        streams[0][1].random()
    states_after = tuple(
        gen.bit_generator.state for gen in streams[1])
    assert states_after == states_before[1], (
        "variant 1: its stream advanced while variant 0 drew")
    fresh = {
        policy.variant_id: r45.variant_stream(
            r45.MASTER_SEED, pinned, policy.variant_id)
        for policy in r45.VARIANTS
    }
    channel_sequences = [
        tuple(fresh[vid][0].random() for _ in range(8))
        for vid in (0, 1)
    ]
    assert channel_sequences[0] != channel_sequences[1]
    seq_a = tuple(r45.variant_stream(r45.MASTER_SEED, pinned, 0)[0]
                  .random() for _ in range(4))
    seq_b = tuple(r45.variant_stream(r45.MASTER_SEED, pinned + 1, 0)[0]
                  .random() for _ in range(4))
    seq_c = tuple(r45.variant_stream(r45.MASTER_SEED + 1, pinned, 0)[0]
                  .random() for _ in range(4))
    assert seq_a != seq_b and seq_a != seq_c
    with pytest.raises(ValueError):
        r45.variant_stream(r45.MASTER_SEED, pinned, 3)


# ---------------------------------------------------------------------------
# The held-out set invariants (species-separated, disjoint from R41's
# training combinations) -- unchanged from the card-15/19 contract
# ---------------------------------------------------------------------------
def test_held_out_set_invariants():
    """The held-out register is species-separated and disjoint from the
    R41 training combinations (the card-15/19 contract the R45 freeze
    validates)."""
    check = r45.held_out_disjointness_check()
    assert check["disjoint"] is True
    assert check["species_separated"] is True
    assert check["overlap"] == []
    assert check["combination_overlap"] == []
    assert tuple(r45.HELD_OUT_KEYS) == (
        "pi_plus_spectrum", "pi_minus_spectrum", "proton_spectrum",
        "antiproton_spectrum",
        "pi_plus_multiplicity", "pi_minus_multiplicity",
        "proton_multiplicity", "antiproton_multiplicity",
    )
    assert tuple(r45.MODEL_ONLY_KEYS) == (
        "pi_zero_spectrum", "pi_zero_multiplicity")
    assert r45.HELD_OUT_SPECIES_NAMED == ("pi+", "pi-", "pi0", "p", "pbar")
    assert r45.R41_TRAINING_OBSERVABLES == (
        "pi_sigma_spectrum", "b_transport_spectrum", "pi_sigma_seagull",
        "b_transport_seagull", "pi_sigma_multiplicity",
        "b_transport_multiplicity",
    )
    assert r45.R41_TRAINING_COMBINATIONS["b_transport"][2] == -1.0
    assert r45.MODEL_ONLY_SPECIES == ("pi_zero",)
    assert "no published pi0 table" in r45.HELD_OUT_SET_NOTE
    assert r45.PI_ZERO_MASS_GEV == 0.1349766
    assert r45.DATA_SPECIES == ("pi_plus", "pi_minus", "proton",
                                "antiproton")


# ---------------------------------------------------------------------------
# The deviance and delta-AIC algebra on pinned fixtures
# ---------------------------------------------------------------------------
def test_deviance_algebra_pinned_fixture():
    d = np.asarray([1.0, 2.0, 3.0], dtype=float)
    v_data = np.asarray([1.0, 1.0, 1.0], dtype=float)
    se = np.asarray([0.1, 0.2, 0.1], dtype=float)
    m = np.asarray([1.2, 0.8, 1.5], dtype=float)
    expected = math.fsum(
        (float(d[i]) - float(m[i])) ** 2
        / (float(v_data[i]) + float(se[i]) ** 2)
        for i in range(3)
    )
    got = r45.observable_deviance("fixture", d, v_data, m, se)
    assert got == pytest.approx(expected, rel=1e-12)
    se_zero = np.asarray([0.0, 0.0, 0.0], dtype=float)
    expected_zero = math.fsum(
        (float(d[i]) - float(m[i])) ** 2
        / (float(v_data[i]) + r45.SE_FLOOR ** 2)
        for i in range(3)
    )
    assert r45.observable_deviance(
        "fixture", d, v_data, m, se_zero) == pytest.approx(
        expected_zero, rel=1e-12)
    m_bad = np.asarray([1.0, float("nan"), 3.0], dtype=float)
    dropped = []
    r45.record_deviance_drops("fixture", d, v_data, m_bad, se, dropped)
    got_bad = r45.observable_deviance(
        "fixture", d, v_data, m_bad, se)
    assert len(dropped) == 1 and dropped[0]["index"] == 1
    expected_bad = (float(d[0]) - 1.0) ** 2 / (1.0 + 0.01) + (
        float(d[2]) - 3.0) ** 2 / (1.0 + 0.01)
    assert got_bad == pytest.approx(expected_bad, rel=1e-12)
    deltas = [0.25 * k for k in range(1, 21)]
    expected_se = float(np.std(deltas, ddof=1)) / math.sqrt(20)
    assert r45.pooled_difference_se(deltas) == pytest.approx(
        expected_se, rel=1e-12)
    assert r45.pooled_difference_se([1.0]) == r45.SE_FLOOR
    assert r45.pooled_difference_se([]) == r45.SE_FLOOR
    assert r45.pooled_difference_se([float("inf"), 1.0]) == r45.SE_FLOOR
    assert r45.HELD_OUT_DEGRADATION_FACTOR == 2.0
    assert r45.SECTOR_DEGRADATION_FRACTION == 0.05


def test_delta_aic_algebra_pinned_fixture():
    """AIC = -2ln L + 2k with the aligned configuration carrying
    kappa_B, kappa_m, b_dec, nominal_width (k = 4) and the baseline
    carrying nominal_width (k = 1); the parameter cost is included and
    favored is delta < 0."""
    aligned_total = 100.0
    baseline_total = 100.0
    block = r45.delta_aic_block(aligned_total, baseline_total)
    assert block["k_aligned"] == 4 and block["k_baseline"] == 1
    assert block["parameter_cost"] == "included"
    assert block["delta_aic"] == pytest.approx(6.0, rel=1e-12)
    assert block["favored"] is False
    block = r45.delta_aic_block(100.0, 110.0)
    assert block["delta_aic"] == pytest.approx(-4.0, rel=1e-12)
    assert block["favored"] is True
    block = r45.delta_aic_block(94.0, 100.0)
    assert block["delta_aic"] == pytest.approx(0.0, abs=1e-12)
    assert block["favored"] is False
    block = r45.delta_aic_block(93.999, 100.0)
    assert block["favored"] is True


def test_held_out_degradation_rows():
    """The held-out row verdict: a held-out observable whose deviance
    degrades by more than 2.0 x the pooled MC SE of the difference is a
    named finding (the card: a degrading observable is a named finding
    that card 21 and any later comparison carry)."""
    row = r45.held_out_row(10.0, 4.0, 2.0)
    assert row["delta_aligned_minus_baseline"] == pytest.approx(6.0)
    assert row["beyond_2x_se"] is True and row["named_finding"] is True
    row = r45.held_out_row(5.0, 4.0, 2.0)
    assert row["degrades"] is True and row["beyond_2x_se"] is False
    assert row["named_finding"] is False
    row = r45.held_out_row(4.0, 5.0, 2.0)
    assert row["degrades"] is False and row["named_finding"] is False
    row = r45.held_out_row(8.0, 4.0, 2.0)
    assert row["beyond_2x_se"] is False
    assert r45.SIGMA_TOLERANCE == 3.0


def test_depth_stats_and_pinned_conventions():
    hist = {0: 30, 1: 40, 2: 20, 3: 10}
    stats = r45.depth_stats(hist, 100)
    assert stats["mean"] == pytest.approx(1.1)
    assert stats["p95"] == 3
    assert stats["max"] == 3
    assert stats["guard_activations"] == 0
    with pytest.raises(ValueError):
        r45.depth_stats(hist, 99)
    assert r45.MAX_CASCADE_DEPTH >= 1


# ---------------------------------------------------------------------------
# The standing residuals (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_standing_residuals_recorded():
    """The R43 b_transport_seagull finding and the R32 spread carry
    appear as named entries in the frozen evaluation's comparison
    record (the card's standing-residual clause; the evaluation is
    re-run over the smoke machinery with the standing-residual blocks
    present regardless of smoke or production)."""
    assert r45.OBLIGATION4_R32_SPREAD == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    # the report builder carries both blocks (checked on the smoke path
    # in test_smoke_mode_refuses_default_run_dir's siblings; the
    # constants are pinned here without sealed reads)


# ---------------------------------------------------------------------------
# The exchange bit-identity on a pinned event
# ---------------------------------------------------------------------------
BARYONIC_QN = (1, 1, 0)


def _exchange_event(family_id: str, mass: float, qn) -> dict:
    return {
        "topology": "exchange_probe",
        "momentum": (mass, 0.0, 0.0, 0.0),
        "family_id": family_id,
        "mass": mass,
        "qn": tuple(int(v) for v in qn),
        "axis": (0.0, 0.0, 1.0),
        "ancestry": ("exchange_probe",),
    }


def test_exchange_event_hash_identity():
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    first = r45.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
    second = r45.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
    assert first == second
    for mutate in (
        lambda e: e.__setitem__("mass", 4.0000001),
        lambda e: e.__setitem__("momentum", (4.0000001, 0.0, 0.0, 0.0)),
        lambda e: e.__setitem__("family_id", "baryonic_strange"),
        lambda e: e.__setitem__("axis", (0.0, 0.0, 0.9999999)),
        lambda e: e.__setitem__("topology", "other"),
    ):
        mutated = json.loads(json.dumps(event))
        mutate(mutated)
        try:
            digest = r45.exchange_event_hash([mutated], 42, 0.5, [1, 0, 1])
        except Exception:
            continue
        assert digest != first


def test_exchange_bit_identity_pinned_event():
    """On a pinned synthetic exchange event, both variants receive the
    identical exchange mapping, the mapping is never mutated, and both
    variants conserve the event exactly (the card-15/19 contract)."""
    from na49_mixing.cascade_ablation import exchange_root_state
    from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL

    catalog = r45._load_catalog()
    fams = r45._load_families()
    tables = _synthetic_tables()
    r45._init_species_layout(tables)
    r45._set_worker_context(r45._worker_context(tables, fams, smoke=True))
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    snapshot = copy.deepcopy(event)
    exchange = r45._EventCapture()
    exchange.event_id = 42
    exchange.t = 0.1
    exchange.route_census = [1, 0, 1]
    exchange.sources = [{
        "source_id": 0, "role": "projectile", "exchange": event,
        "route": "continuum", "record_pdg": None, "record_name": None,
        "n_open_records": None, "stable_identity": False, "energy": 4.0,
    }, {
        "source_id": 1, "role": "target",
        "exchange": _exchange_event("baryonic_np", 4.0, BARYONIC_QN),
        "route": "continuum", "record_pdg": None, "record_name": None,
        "n_open_records": None, "stable_identity": False, "energy": 4.0,
    }]
    exchange.energy_expected = 8.0
    exchange.exchange_hash = r45.exchange_event_hash(
        [entry["exchange"] for entry in exchange.sources],
        42, 0.1, exchange.route_census)
    variants = r45.decay_event_variants(
        catalog, fams, exchange, smoke=True)
    assert event == snapshot, "the exchange event was mutated"
    root = r45.exchange_root_state(event)
    assert root.family_id == "baryonic_np" and root.mass == 4.0
    assert root.qn == BARYONIC_QN
    assert root.axis == (0.0, 0.0, 1.0)
    assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert root.depth == 0
    for name in r45.VARIANT_NAMES:
        run = variants[name]["run"]
        assert run.bqs == (2, 2, 0)
        assert run.worst_closure <= r45.EVENT_CLOSURE_TOL
    r45._set_worker_context(None)


# ---------------------------------------------------------------------------
# The CLI surface (the seven modes)
# ---------------------------------------------------------------------------
def test_cli_surface():
    parser = r45._build_parser()
    modes = {
        "--pre-registration", "--smoke", "--generate", "--evaluate",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {"--replica", "--run-dir", "--workers"}
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    assert seen == modes | auxiliary | {"-h", "--help"}, (
        f"the CLI surface drifted: {sorted(seen)}")
    with pytest.raises(SystemExit) as exc:
        r45.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        r45.main(["--evaluate", "--gate"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and sit
    at the driver's self-computed identity digest."""
    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r45_simplified_alignment_freeze.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r45_simplified_alignment_freeze.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest
    assert nested_digest == r45.DRIVER_SHA256


# ---------------------------------------------------------------------------
# The smoke refusal convention
# ---------------------------------------------------------------------------
def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r45.MONO_ROOT / r45.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r45.run_smoke(default_dir, workers=1)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention)."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r45.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r45._refuse_smoke_artifacts(run_dir)
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r45.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_replica_0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r45._refuse_smoke_artifacts(run_dir2)
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r45._refuse_smoke_artifacts(run_dir3)


# ---------------------------------------------------------------------------
# The synthetic fixtures (the hermetic helpers)
# ---------------------------------------------------------------------------
def _synthetic_tables():
    """A small hermetic seven-species table set on a 4x3 grid (the R43
    test's synthetic-grid pattern plus the model-only pi0 block)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    tables = []
    for species, pdg, mass in (
        ("pi_plus", 211, 0.1395699), ("pi_minus", -211, 0.1395699),
        ("k_plus", 321, 0.4936770), ("k_minus", -321, 0.4936770),
        ("proton", 2212, 0.9382720), ("antiproton", -2212, 0.9382720),
    ):
        n_x, n_p = 4, 3
        xf = np.linspace(0.0, 0.4, n_x + 2)[1:-1]
        pt = np.linspace(0.05, 1.5, n_p + 2)[1:-1]
        tables.append(_build_table(
            species, pdg, mass, xf, pt,
            window={"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
        ))
    tables.append(r45._pi_zero_model_table(tables))
    return tables


def _build_table(species: str, pdg_id: int, mass: float,
                 xf_centers: np.ndarray, pt_centers: np.ndarray,
                 window: dict):
    """One BinnedFitTable on the given grid geometry (the per-bin point
    arrays in the pT-major flat layout)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    n_x = len(xf_centers)
    n_p = len(pt_centers)
    xf_pts = np.tile(xf_centers, n_p)
    pt_pts = np.repeat(pt_centers, n_x)
    ones = np.ones(n_x * n_p, dtype=float)
    return BinnedFitTable.from_arrays(
        species=species, pdg_id=pdg_id, mass_gev=float(mass),
        normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
        path="test://synthetic",
        xf=xf_pts, pt=pt_pts, value=ones, error=ones,
        window=window,
    )


# ---------------------------------------------------------------------------
# The freeze re-pin register (the card-20 landing)
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r45_simplified_alignment_freeze.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze.py"
)
SOLVER_CARD = "TASK-20260920-r4-gate4-followup-simplified-alignment"

#: The chain count pins this card's re-pin bump (491 -> 495, +4 entries:
#: the dual-tree R45 driver in both helper trees and the two new R45
#: test files), plus this file's own count constant.  The chain files
#: were re-pinned 488 -> 491 by card 15 with their naming comments; the
#: card-20 bump appends the card-20 comment to every chain file.
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
)
# re-pinned 487 -> 488 by card 14 test-author (TASK-20260920-r4-run-
# r43-cascade-ablation-test-author; +1 entry: the card-14
# test-author register file).
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned in place by card 19 (TASK-20260920-r4-run-r44-repoint;
# no new files: the same three entries re-pinned to the card-19
# on-disk digests, the count stays 491).
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-
# simplified-alignment; +4 entries: the dual-tree R45 driver in both
# helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).
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
def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




