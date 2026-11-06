"""Hermetic tests for the R44 driver (the frozen nonstrange validation).

Task ``TASK-20260920-r4-run-r44-frozen-nonstrange-validation`` (Round 4,
card 15; plan section 10 run R44, line 800), re-pointed by card 19
(``TASK-20260920-r4-run-r44-repoint``): the seed-map and event-index
partition, the frozen-parameter assertion against a pinned R41 fixture
(the corrected card-16 fit output of the sealed
r41_nonstrange_pilot_fit_2026-09-23 archive; the superseded card-15
tuple is retained provenance), the held-out set invariants
(species-separated, disjoint from the R41 training combinations), the
variant stream disjointness on a pinned key, the deviance and delta-AIC
algebra on pinned fixtures, the CLI surface, the layout sync, and the
freeze re-pin register.  No Monte Carlo generation at run scale, no SPS
table, no sealed archive read by the acceptance legs (the sealed-dir
pins skip on a fresh checkout): the acceptance is hermetic.
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
        / "staged_r44_frozen_nonstrange_validation.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r44_frozen_nonstrange_validation", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R44 driver was not found next to the test file")


r44 = _load_driver()


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    assert r44.MASTER_SEED == 20260920
    assert r44.RAW_REPLICA_SEEDS == tuple(20301020 + k for k in range(20))
    assert r44.RAW_REPLICA_SEEDS[0] == 20301020
    assert r44.RAW_REPLICA_SEEDS[-1] == 20301039
    assert r44.EVENT_INDEX_RULE == "700_000_000 + i, i < 250,000"
    assert r44.EVENT_INDEX_BASE == 700_000_000
    assert r44.T3_REPLICAS == 20 and r44.T3_EVENTS_PER_REPLICA == 250_000
    assert r44.T3_REPLICAS * r44.T3_EVENTS_PER_REPLICA == 5_000_000
    # the per-replica seeds are the block 20301020-20301039 (the
    # R41/R42/R43 block 20301020-20301027 extended to twenty slots; the
    # (seed, event-index) substreams are disjoint by the event-index
    # partition)
    assert [r44.raw_replica_seed(k) for k in range(20)] == [
        20301020 + k for k in range(20)]
    with pytest.raises(SystemExit):
        r44.raw_replica_seed(20)
    # the reserved audit block 20301040 + k (k = 0..15) is disjoint from
    # the raw block and from the smoke seed
    assert r44.AUDIT_RESERVED_BLOCK == (20301040, 20301055)
    assert max(r44.RAW_REPLICA_SEEDS) == 20301039 < 20301040
    assert r44.SMOKE_RAW_SEED == 20260920 + 40_000 + 10_000 == 20310920
    assert r44.SMOKE_KERNEL_REPLICA_SLOT == 10_000
    assert r44.SMOKE_RAW_SEED not in r44.RAW_REPLICA_SEEDS
    assert r44.SMOKE_RAW_SEED not in range(20301040, 20301056)
    assert r44.SMOKE_RAW_SEED != 20309920  # the R42/R43 smoke block
    assert r44.SMOKE_RAW_SEED != 20308920  # the R41 smoke block
    assert r44.SMOKE_EVENT_INDEX_BASE == 790_000_000
    assert r44.SMOKE_EVENT_INDEX_RULE == "790_000_000 + i, i < 2,000"
    # the production event-index partition is disjoint from R43's
    # (600M-based), R42's (500M-based), R41's (400M-based), and R40's
    # (300M-based, below 389,025,000)
    assert r44.EVENT_INDEX_BASE == 700_000_000
    assert r44.EVENT_INDEX_BASE > 600_000_000
    # the rules name the disjointness, the block sharing, and the
    # previously used blocks
    assert "R41" in r44.RAW_SEED_RULE and "R42" in r44.RAW_SEED_RULE
    assert "R43" in r44.RAW_SEED_RULE and "R40" in r44.RAW_SEED_RULE
    assert "20301040 + k" in r44.RAW_SEED_RULE
    assert "derive_substreams(20260920, k, 700_000_000 + i)" in \
        r44.EXCHANGE_SEED_RULE
    assert "variant_stream(20260920, 700_000_000 + i, v)" in \
        r44.VARIANT_STREAM_RULE
    assert "700_000_000 + i" in r44.EXCHANGE_SEED_RULE
    # the recorded previously-used blocks name the sealed
    # pre-registrations and the audit reserved block
    blocks = r44.PREVIOUS_USED_BLOCKS
    assert blocks["r41"]["event_index_partition"] == \
        "400_000_000 + i, i < 100,000"
    assert blocks["r42"]["event_index_partition"] == \
        "500_000_000 + i, i < 100,000"
    assert blocks["r43"]["event_index_partition"] == \
        "600_000_000 + i, i < 100,000"
    assert blocks["r43"]["source"].endswith(
        "r43_cascade_ablation_2026-09-23/pre_registration.json")
    assert blocks["r42"]["source"].endswith(
        "r42_recoil_decay_separation_2026-09-23/pre_registration.json")
    assert blocks["r41"]["source"].endswith(
        "r41_nonstrange_pilot_fit_2026-09-23/pre_registration.json")
    assert blocks["r40"]["event_index_range"] == [300_000_000, 389_025_000]
    assert blocks["audit_reserved"]["raw_seeds"] == [20301040, 20301055]
    assert blocks["audit_reserved"]["fired"] is False


def test_seed_disjointness_report():
    """The seed-block disjointness check passes (the audit block
    disjoint from the R44 raw block, the shared raw seeds named, the
    event-index partitions pairwise disjoint) and any substream
    collision refuses the run."""
    report = r44.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["audit_block_disjoint"] is True
    assert report["r44_raw_seed_block"] == [20301020, 20301039]
    assert report["r44_event_index_range"] == [700_000_000, 700_250_000]
    shared = report["shared_raw_seeds"]
    assert shared["r40"] == [20301020, 20301023]
    assert shared["r41"] == [20301020, 20301027]
    assert shared["r42"] == [20301020, 20301027]
    assert shared["r43"] == [20301020, 20301027]
    # a colliding partition is refused (never a silent re-block)
    saved = dict(r44.PREVIOUS_USED_BLOCKS)
    try:
        collided = copy.deepcopy(r44.PREVIOUS_USED_BLOCKS)
        collided["r41"]["event_index_range"] = [700_000_000, 700_100_000]
        r44.PREVIOUS_USED_BLOCKS.clear()
        r44.PREVIOUS_USED_BLOCKS.update(collided)
        with pytest.raises(SystemExit):
            r44.seed_disjointness_report()
    finally:
        r44.PREVIOUS_USED_BLOCKS.clear()
        r44.PREVIOUS_USED_BLOCKS.update(saved)
    # the raw seeds helpers
    assert r44.raw_replica_seed(0) == 20301020
    assert r44.raw_replica_seed(19) == 20301039
    with pytest.raises(SystemExit):
        r44.raw_replica_seed(20)
    assert r44.event_index(0) == 700_000_000
    assert r44.smoke_event_index(1) == 790_000_001
    assert r44.smoke_raw_seed() == 20310920


def test_stream_tree_disjointness():
    """The (master, replica slot, event index) exchange spaces are
    pairwise disjoint across the R44 production slots and the smoke
    slot, and disjoint from the R41 (400M), R42 (500M), R43 (600M), and
    R40 (300M) spaces; the variant-spawn tree is a separate tree (card
    10's contract)."""
    from na49_mixing.rng_substreams import derive_substreams

    labels = ("topology_selection", "exchange_kinematics",
              "spectral_state", "decay_channel", "decay_kinematics")
    streams = set()
    for replica in range(r44.T3_REPLICAS):
        subs = derive_substreams(
            r44.MASTER_SEED, replica, r44.EVENT_INDEX_BASE)
        for label in labels:
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    subs = derive_substreams(
        r44.MASTER_SEED, r44.SMOKE_KERNEL_REPLICA_SLOT,
        r44.SMOKE_EVENT_INDEX_BASE)
    for label in labels:
        assert int(subs.stream_for(label)) not in streams
    for replica in range(r44.T3_REPLICAS):
        r43_subs = derive_substreams(
            r44.MASTER_SEED, replica, 600_000_000)
        r42_subs = derive_substreams(
            r44.MASTER_SEED, replica, 500_000_000)
        r41_subs = derive_substreams(
            r44.MASTER_SEED, replica, 400_000_000)
        r40_subs = derive_substreams(
            r44.MASTER_SEED, replica, 300_000_000)
        for label in labels:
            assert int(r43_subs.stream_for(label)) not in streams
            assert int(r42_subs.stream_for(label)) not in streams
            assert int(r41_subs.stream_for(label)) not in streams
            assert int(r40_subs.stream_for(label)) not in streams
    # the builder's legacy generator spawn keys are disjoint across the
    # event-index partitions (the same raw seed, different blocks)
    seq = lambda seed, event: np.random.SeedSequence(  # noqa: E731
        seed, spawn_key=(event,))
    for seed in (20301020, 20301027, 20301039):
        r44_stream = seq(seed, r44.EVENT_INDEX_BASE).generate_state(1)
        for event in (400_000_000, 500_000_000, 600_000_000,
                      300_000_000):
            assert seq(seed, event).generate_state(1) != \
                seq(seed, r44.EVENT_INDEX_BASE).generate_state(1)


# ---------------------------------------------------------------------------
# The frozen-parameter assertion (the sealed R41 archive is the source)
# ---------------------------------------------------------------------------
def test_frozen_parameter_assertion_pinned_fixture():
    """The frozen parameters (kappa_B, kappa_m, b_dec, baryonic
    nominal_width) are the corrected R41 fitted values (0.0, 4.0,
    2.0 GeV^-2, 0.3 GeV; the card-19 re-point to the card-16
    relitigated fit output of the sealed r41_nonstrange_pilot_fit_
    2026-09-23 archive), pinned as a fixture and asserted by the same
    predicate the production pre-registration runs against the sealed
    archive; the aligned policy carries them and refit is false."""
    frozen = r44.assert_frozen_parameters(r44.R41_FIT_REPORT_FIXTURE)
    assert frozen["kappa_B"] == 0.0
    assert frozen["kappa_m"] == 4.0
    assert frozen["b_dec_gev2"] == 2.0
    assert frozen["baryonic_nominal_width_gev"] == 0.3
    assert frozen["refit"] is False
    assert "R41" in frozen["source"]
    assert r44.R41_FITTED == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
        "baryonic_nominal_width": 0.3,
    }
    assert r44.FROZEN_WIDTH_GEV == 0.3
    assert (r44.ALIGNED_POLICY.kappa_b, r44.ALIGNED_POLICY.kappa_m,
            r44.ALIGNED_POLICY.b_dec) == (0.0, 4.0, 2.0)
    assert r44.ALIGNED_POLICY.variant_id == 0
    # the sealed R41 archive cross-check (read-only; skips on a fresh
    # checkout without the sealed archive)
    fit_path = r44.MONO_ROOT / r44.R41_RUN_DIR / r44.R41_FIT_REPORT_NAME
    if fit_report_is_sealed():
        report = json.loads(
            (r44.MONO_ROOT / r44.R41_RUN_DIR
             / r44.R41_FIT_REPORT_NAME).read_text(encoding="utf-8"))
        frozen_sealed = r44.assert_frozen_parameters(report)
        assert frozen_sealed_values(frozen_sealed) == \
            frozen_sealed_values(frozen)
        assert r44.R41_FIT_REPORT_DIGEST == report["digest"]
        assert r44.R41_FIT_REPORT_FILE_SHA256 == _sha256_file(fit_path)


def fit_report_is_sealed() -> bool:
    """The sealed R41 fit report is present on this checkout."""
    path = r44.MONO_ROOT / r44.R41_RUN_DIR / r44.R41_FIT_REPORT_NAME
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
    mode, a missing fitted block, a width drift, and the superseded
    card-15 fitted tuple (retained provenance, never consumed)."""
    fixture = r44.R41_FIT_REPORT_FIXTURE
    drifted = dict(fixture)
    drifted["fitted"] = dict(fixture["fitted"], kappa_B=0.1)
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(drifted)
    wrong_card = dict(fixture, card="TASK-20260920-r4-run-r42-")
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(wrong_card)
    wrong_run = dict(fixture, run="R42")
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(wrong_run)
    smoke_mode = dict(fixture, mode="smoke")
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(smoke_mode)
    no_block = {"card": fixture["card"], "run": "R41",
                "mode": "production"}
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(no_block)
    # a width drift off the corrected fitted value refuses
    width_drift = dict(fixture)
    width_drift["fitted"] = dict(fixture["fitted"],
                                 baryonic_nominal_width=0.4)
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(width_drift)
    # the superseded card-15 fitted tuple (0.5 GeV) refuses: the run
    # never consumes the superseded archive's values
    superseded = dict(fixture)
    superseded["fitted"] = dict(r44.SUPERSEDED_R41_FITTED)
    assert superseded["fitted"] != dict(r44.R41_FITTED)
    with pytest.raises(SystemExit):
        r44.assert_frozen_parameters(superseded)


# ---------------------------------------------------------------------------
# The card-19 re-point pins (the corrected constants, the retained
# superseded constants, and the frozen-substrate stability)
# ---------------------------------------------------------------------------
def test_frozen_parameter_repoint_pins():
    """The card-19 re-point: the corrected frozen-parameter source pins
    (the sealed r41_nonstrange_pilot_fit_2026-09-23 archive's fit
    report file digest and self-digest), the corrected run-dir pins,
    the corrected R41/R42/R43 sibling substrate digests, the new run
    identity, and the frozen-substrate byte stability of every pinned
    surface."""
    # the corrected run-dir pins (the superseded 2026-09-22 pins are
    # retained as SUPERSEDED_* constants, asserted separately)
    assert r44.R41_RUN_DIR == \
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"
    assert r44.R42_RUN_DIR == \
        "dev/00runs/r42_recoil_decay_separation_2026-09-23"
    assert r44.R43_RUN_DIR == \
        "dev/00runs/r43_cascade_ablation_2026-09-23"
    for rel in r44.SEALED_RUN_DIRS:
        assert "2026-09-23" in rel or rel in (
            "dev/00runs/r30_spectral_normalization_2026-09-17",
            "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
            "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
            "dev/00runs/r32_spectral_systematic_pp_confirmation_"
            "2026-09-19",
        ), rel
    assert "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23" in \
        r44.SEALED_RUN_DIRS
    assert "dev/00runs/r42_recoil_decay_separation_2026-09-23" in \
        r44.SEALED_RUN_DIRS
    assert "dev/00runs/r43_cascade_ablation_2026-09-23" in \
        r44.SEALED_RUN_DIRS
    for superseded_dir in (r44.SUPERSEDED_R41_RUN_DIR,
                           *r44.SUPERSEDED_R42_RUN_DIRS,
                           r44.SUPERSEDED_R43_RUN_DIR):
        assert superseded_dir not in r44.SEALED_RUN_DIRS
    # the corrected fit-report digest pins
    assert r44.R41_FIT_REPORT_FILE_SHA256 == (
        "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
    )
    assert r44.R41_FIT_REPORT_DIGEST == (
        "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
    )
    # the corrected sibling substrate pins (the card-16/17/18 digests)
    assert r44.FROZEN_SUBSTRATE["r41_driver"] == (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py",
        "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc",
    )
    assert r44.FROZEN_SUBSTRATE["r42_driver"] == (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py",
        "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd",
    )
    assert r44.FROZEN_SUBSTRATE["r43_driver"] == (
        "helpers/benchmarking/staged_r43_cascade_ablation.py",
        "dfb5ee3f4ee6598698faaf3da641fbf315ba21d95f2203792a2414a72e11801c",
    )
    # the r32 / r40 substrate pins and the other entries stay
    assert r44.FROZEN_SUBSTRATE["r32_driver"] == (
        "helpers/benchmarking/staged_r32_spectral_systematic.py",
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    )
    assert r44.FROZEN_SUBSTRATE["r40_driver"] == (
        "helpers/benchmarking/staged_r40_alignment_scan.py",
        "7366a55e0e15032f6d4c9194adce7537ea5239075e34f921900b75be64787354",
    )
    # the frozen-substrate byte stability: every pinned surface,
    # corrected siblings included, matches its on-disk bytes
    status = r44.verify_frozen_substrate()
    assert set(status) == set(r44.FROZEN_SUBSTRATE)
    assert all(entry["verified"] is True for entry in status.values())
    # the new run identity (the aborted card-15 2026-09-22 dir was
    # deleted and must not reappear)
    assert r44.RUN_NAME == "r44_frozen_nonstrange_validation_2026-09-23"
    assert r44.DEFAULT_RUN_DIR == (
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23")
    assert r44.RUN_NAME != "r44_frozen_nonstrange_validation_2026-09-22"
    assert not (r44.MONO_ROOT / "dev/00runs/"
                "r44_frozen_nonstrange_validation_2026-09-22").exists()


def test_superseded_constants_retained():
    """The superseded card-15 constants are retained for the sealed
    provenance comparisons (read-only; nothing in the superseded
    archives is consumed): the superseded frozen tuple, the superseded
    R41 fit-report digests and run dir, the superseded sibling driver
    digests, the superseded sibling archive run dirs, and the
    pre-registration's re-point record (superseded vs corrected)."""
    assert r44.SUPERSEDED_R41_FITTED == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
        "baryonic_nominal_width": 0.5,
    }
    # the re-point moved exactly the width: the other three fitted
    # values agree between the corrected and superseded tuples
    assert r44.R41_FITTED["kappa_B"] == \
        r44.SUPERSEDED_R41_FITTED["kappa_B"] == 0.0
    assert r44.R41_FITTED["kappa_m"] == \
        r44.SUPERSEDED_R41_FITTED["kappa_m"] == 4.0
    assert r44.R41_FITTED["b_dec"] == \
        r44.SUPERSEDED_R41_FITTED["b_dec"] == 2.0
    assert r44.R41_FITTED["baryonic_nominal_width"] == 0.3 != 0.5 == \
        r44.SUPERSEDED_R41_FITTED["baryonic_nominal_width"]
    assert r44.SUPERSEDED_R41_RUN_DIR == \
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-22"
    assert r44.SUPERSEDED_R41_FIT_REPORT_FILE_SHA256 == (
        "69851f1ac29804d2745de7efabffe332be928e77c0474f353c8338a7bc1ebcfe"
    )
    assert r44.SUPERSEDED_R41_FIT_REPORT_DIGEST == (
        "2c0755e6370739b3089120d432e447433b48eed7441281f58103a78457a990b4"
    )
    assert r44.SUPERSEDED_R41_DRIVER_SHA256 == (
        "573d833bfbc659885fb83e6c0794207947fbe9e5a6dafe6159650bd45097172b"
    )
    assert r44.SUPERSEDED_R42_DRIVER_SHA256 == (
        "30e97f18653129ba3e7f1e5d8af782c819a1a7b189413032925ec35b26b0afa0"
    )
    assert r44.SUPERSEDED_R43_DRIVER_SHA256 == (
        "4c6bc0f32430577ff80e607e19f2512bf2ed1c40701f1f115ff82a88c975bf3a"
    )
    assert r44.SUPERSEDED_R42_RUN_DIRS == (
        "dev/00runs/r42_recoil_decay_separation_2026-09-22",
        "dev/00runs/r42_recoil_decay_separation_2026-09-22r2",
    )
    assert r44.SUPERSEDED_R43_RUN_DIR == \
        "dev/00runs/r43_cascade_ablation_2026-09-22"
    assert r44.SUPERSEDED_ARCHIVE_RUN_DIRS == (
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-22",
        "dev/00runs/r42_recoil_decay_separation_2026-09-22",
        "dev/00runs/r42_recoil_decay_separation_2026-09-22r2",
        "dev/00runs/r43_cascade_ablation_2026-09-22",
    )
    # the pre-registration carries the re-point record (superseded vs
    # corrected frozen tuple, digests, archives); the smoke payload
    # includes it too (pure constants, no sealed reads)
    pre, frozen, _disjointness = r44._pre_registration_payload(smoke=True)
    repoint = pre["frozen_parameter_repoint"]
    assert repoint["card"] == "TASK-20260920-r4-run-r44-repoint"
    assert repoint["corrected_frozen"] == dict(r44.R41_FITTED)
    assert repoint["superseded_frozen"] == dict(r44.SUPERSEDED_R41_FITTED)
    assert repoint["corrected_fit_report_file_sha256"] == \
        r44.R41_FIT_REPORT_FILE_SHA256
    assert repoint["superseded_fit_report_file_sha256"] == \
        r44.SUPERSEDED_R41_FIT_REPORT_FILE_SHA256
    assert repoint["corrected_fit_report_self_digest"] == \
        r44.R41_FIT_REPORT_DIGEST
    assert repoint["superseded_fit_report_self_digest"] == \
        r44.SUPERSEDED_R41_FIT_REPORT_DIGEST
    assert repoint["superseded_status"] == (
        "superseded by the card-16 R41 relitigation; the values stay "
        "intact in the sealed superseded archive (read-only, the seal "
        "verifies) and are not consumed by this run")
    assert repoint["corrected_sibling_driver_sha256s"] == {
        "r41": r44.FROZEN_SUBSTRATE["r41_driver"][1],
        "r42": r44.FROZEN_SUBSTRATE["r42_driver"][1],
        "r43": r44.FROZEN_SUBSTRATE["r43_driver"][1],
    }
    assert repoint["superseded_sibling_driver_sha256s"] == {
        "r41": r44.SUPERSEDED_R41_DRIVER_SHA256,
        "r42": r44.SUPERSEDED_R42_DRIVER_SHA256,
        "r43": r44.SUPERSEDED_R43_DRIVER_SHA256,
    }
    assert repoint["superseded_sibling_run_dirs"] == \
        list(r44.SUPERSEDED_ARCHIVE_RUN_DIRS)
    assert frozen["baryonic_nominal_width_gev"] == 0.3
    # the superseded archives verify byte-for-byte on this checkout
    # (the seal states are read-only; skips on a fresh checkout)
    if superseded_archives_present():
        for rel in r44.SUPERSEDED_ARCHIVE_RUN_DIRS:
            seal = r44.verify_sealed_dir(rel)
            assert seal["verified"] is True and seal["mismatches"] == 0
        # the superseded sealed R41 fit report still carries the
        # superseded tuple at the retained digests (byte-for-byte)
        report = json.loads(
            (r44.MONO_ROOT / r44.SUPERSEDED_R41_RUN_DIR
             / r44.R41_FIT_REPORT_NAME).read_text(encoding="utf-8"))
        assert report["fitted"] == dict(r44.SUPERSEDED_R41_FITTED)
        assert r44.SUPERSEDED_R41_FIT_REPORT_FILE_SHA256 == _sha256_file(
            r44.MONO_ROOT / r44.SUPERSEDED_R41_RUN_DIR
            / r44.R41_FIT_REPORT_NAME)
        assert r44.SUPERSEDED_R41_FIT_REPORT_DIGEST == report["digest"]
        # the superseded sealed R41 archive is not the frozen-parameter
        # source: its values refuse the production assertion
        with pytest.raises(SystemExit):
            r44.assert_frozen_parameters(report)


def superseded_archives_present() -> bool:
    """The superseded sealed archives are present on this checkout."""
    return all(
        (r44.MONO_ROOT / rel / r44.SHA256SUMS_NAME).is_file()
        for rel in r44.SUPERSEDED_ARCHIVE_RUN_DIRS
    )


# ---------------------------------------------------------------------------
# The variant policies (the two configurations on identical events)
# ---------------------------------------------------------------------------
def test_variant_policies():
    """The card's two configurations: the frozen aligned point (the
    corrected R41 fitted values, no refit) and the card-10 v2 isotropic
    baseline (read from the frozen card-10 register); both record the
    corrected frozen nominal width 0.3 GeV."""
    register = r44.VARIANT_NAMES
    assert register == ("aligned_frozen", "isotropic_sequential_baseline")
    assert r44.ALIGNED_POLICY.variant_id == 0
    assert r44.BASELINE_POLICY.variant_id == 1
    assert (r44.BASELINE_POLICY.kappa_b, r44.BASELINE_POLICY.kappa_m,
            r44.BASELINE_POLICY.b_dec) == (0.0, 0.0, 1.0)
    from na49_mixing.cascade_ablation import ABLATION_POLICIES

    v2 = ABLATION_POLICIES["isotropic_sequential"]
    assert (r44.BASELINE_POLICY.kappa_b, r44.BASELINE_POLICY.kappa_m,
            r44.BASELINE_POLICY.b_dec) == (
        float(v2.kappa_b), float(v2.kappa_m), float(v2.b_dec))
    # the aligned variant is the frozen R41 point, not the R43 nominal
    assert (r44.ALIGNED_POLICY.kappa_b, r44.ALIGNED_POLICY.kappa_m,
            r44.ALIGNED_POLICY.b_dec) != (1.0, 1.0, 1.0)
    # the blend ramp start stays fixed at B (obligation 4; excluded from
    # the R41 fit set)
    assert r44.BLEND_RAMP_START_GEV == 2.680194110727528
    assert "obligation 4" in r44.BLEND_RAMP_START_STATUS


def test_variant_stream_disjointness_pinned_key():
    """The per-variant decay streams are child spawns keyed on the
    variant id, disjoint across the two variants (a draw on variant 0
    never advances variant 1's stream), deterministic on the pinned key,
    and sensitive to the event index and the master seed (card 10's
    variant_stream at the production event-index base)."""
    pinned = r44.EVENT_INDEX_BASE
    # determinism on the pinned key
    for policy in r44.VARIANTS:
        first = r44.variant_stream(r44.MASTER_SEED, pinned,
                                   policy.variant_id)
        second = r44.variant_stream(r44.MASTER_SEED, pinned,
                                    policy.variant_id)
        for gen_a, gen_b in zip(first, second):
            assert [gen_a.random() for _ in range(6)] == [
                gen_b.random() for _ in range(6)]
    # disjointness: a draw on variant 0 never advances variant 1
    streams = {
        policy.variant_id: r44.variant_stream(
            r44.MASTER_SEED, pinned, policy.variant_id)
        for policy in r44.VARIANTS
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
    # the two variant streams differ
    fresh = {
        policy.variant_id: r44.variant_stream(
            r44.MASTER_SEED, pinned, policy.variant_id)
        for policy in r44.VARIANTS
    }
    channel_sequences = [
        tuple(fresh[vid][0].random() for _ in range(8))
        for vid in (0, 1)
    ]
    assert channel_sequences[0] != channel_sequences[1]
    # the key is sensitive to the event index and the master seed
    seq_a = tuple(r44.variant_stream(r44.MASTER_SEED, pinned, 0)[0]
                  .random() for _ in range(4))
    seq_b = tuple(r44.variant_stream(r44.MASTER_SEED, pinned + 1, 0)[0]
                  .random() for _ in range(4))
    seq_c = tuple(r44.variant_stream(r44.MASTER_SEED + 1, pinned, 0)[0]
                  .random() for _ in range(4))
    assert seq_a != seq_b and seq_a != seq_c
    # the R44 variant ids stay inside the card-10 closed register and
    # outside ids the register refuses
    with pytest.raises(ValueError):
        r44.variant_stream(r44.MASTER_SEED, pinned, 3)


# ---------------------------------------------------------------------------
# The held-out set invariants (species-separated, disjoint from R41's
# training combinations)
# ---------------------------------------------------------------------------
def test_held_out_set_invariants():
    """The held-out register is species-separated and disjoint from the
    R41 training combinations: no held-out key is an R41 training
    observable, a charge-symmetric combination of plan section 7.3, or a
    pooled moment; the pi0 rows carry the model-only marker."""
    check = r44.held_out_disjointness_check()
    assert check["disjoint"] is True
    assert check["species_separated"] is True
    assert check["overlap"] == []
    assert check["combination_overlap"] == []
    assert tuple(r44.HELD_OUT_KEYS) == (
        "pi_plus_spectrum", "pi_minus_spectrum", "proton_spectrum",
        "antiproton_spectrum",
        "pi_plus_multiplicity", "pi_minus_multiplicity",
        "proton_multiplicity", "antiproton_multiplicity",
    )
    assert tuple(r44.MODEL_ONLY_KEYS) == (
        "pi_zero_spectrum", "pi_zero_multiplicity")
    # the card's species naming (round 4 index item 11)
    assert r44.HELD_OUT_SPECIES_NAMED == ("pi+", "pi-", "pi0", "p", "pbar")
    # the R41 training register (the complement) is the card's list
    assert r44.R41_TRAINING_OBSERVABLES == (
        "pi_sigma_spectrum", "b_transport_spectrum", "pi_sigma_seagull",
        "b_transport_seagull", "pi_sigma_multiplicity",
        "b_transport_multiplicity",
    )
    # the combinations of plan section 7.3 (pi_Sigma = pi+ + pi-,
    # B_transport = p - pbar with the minus sign) are the R41 training
    # combinations and are not held-out keys
    assert set(r44.R41_TRAINING_COMBINATIONS) == {"pi_sigma",
                                                  "b_transport"}
    assert r44.R41_TRAINING_COMBINATIONS["b_transport"][2] == -1.0
    # the model-only pi0 territory carries no published data vector
    assert r44.MODEL_ONLY_SPECIES == ("pi_zero",)
    assert "no published pi0 table" in r44.HELD_OUT_SET_NOTE
    assert r44.PI_ZERO_MASS_GEV == 0.1349766
    # the published surface is the six paper-window species
    assert r44.DATA_SPECIES == ("pi_plus", "pi_minus", "proton",
                                "antiproton")
    for held_key in list(r44.HELD_OUT_KEYS) + list(r44.MODEL_ONLY_KEYS):
        assert held_key not in r44.R41_TRAINING_OBSERVABLES


def test_held_out_statistics_fixture():
    """The held-out statistic construction on a pinned fixture: the
    spectrum d/v_data pass through from the sealed surface, the moment
    is the window integral (sum d * area with the delta-method variance
    sum v * area^2), and the register order is the pre-registered one."""
    n_bins = 6
    sealed = {
        "pi_plus": {
            "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "v_data": [0.01] * n_bins,
            "xF_mid": [0.1] * n_bins,
            "pT_mid": [0.2] * n_bins,
        },
    }
    tables_by_species = {}
    for species in r44.DATA_SPECIES:
        if species == "pi_plus":
            continue
        sealed[species] = {
            "d": [1.0] * n_bins, "v_data": [0.01] * n_bins,
            "xF_mid": [0.1] * n_bins, "pT_mid": [0.2] * n_bins,
        }
    for species in r44.DATA_SPECIES:
        tables_by_species[species] = _fake_table(species, np.ones(n_bins))
    areas = np.asarray([
        r44._cell_area_from_table(tables_by_species["pi_plus"], i)
        for i in range(n_bins)
    ])
    stats = r44._build_held_out_statistics(sealed, tables_by_species)
    keys = [stat["key"] for stat in stats]
    assert keys == list(r44.HELD_OUT_KEYS)
    spectrum = stats[0]
    assert spectrum["key"] == "pi_plus_spectrum"
    assert list(spectrum["d"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.allclose(spectrum["areas"], areas)
    moment = stats[4]
    assert moment["key"] == "pi_plus_multiplicity"
    d_expected = float(np.sum(
        np.asarray(sealed["pi_plus"]["d"]) * areas))
    v_expected = float(np.sum(
        np.asarray(sealed["pi_plus"]["v_data"]) * areas ** 2))
    assert float(moment["d"][0]) == pytest.approx(d_expected, rel=1e-12)
    assert float(moment["v_data"][0]) == pytest.approx(
        max(0.01 * float(np.sum(areas ** 2)), r44.SE_FLOOR), rel=1e-12)
    # the model vector of the moment is the same integral of the model
    model = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    values = {s: model for s in r44.DATA_SPECIES}
    vector = r44._stat_vector(moment, values)
    assert float(vector[0]) == pytest.approx(
        float(np.sum(model * areas)), rel=1e-12)


def _fake_table(species: str, areas: np.ndarray):
    """A minimal table stand-in carrying only what _cell_area_from_table
    needs: the unique centers, the per-bin indices, and the bin count on
    the pinned-grid geometry."""
    n_x, n_p = 3, 2
    xf = np.linspace(0.0, 0.4, n_x + 2)[1:-1]
    pt = np.linspace(0.05, 1.5, n_p + 2)[1:-1]
    assert areas.size == n_x * n_p
    return _FakeBinnedTable(species, 211 if species == "pi_plus" else 0,
                            0.1395699 if species.startswith("pi") else 0.9)


class _FakeBinnedTable:
    """A hermetic table stand-in with the pinned-grid geometry."""

    def __init__(self, species: str, pdg_id: int, mass: float) -> None:
        self.species = species
        self.pdg_id = int(pdg_id)
        self.mass_gev = float(mass)
        n_x, n_p = 3, 2
        xf = np.linspace(0.0, 0.4, n_x + 2)[1:-1]
        pt = np.linspace(0.05, 1.5, n_p + 2)[1:-1]
        self.xf_centers = xf
        self.pt_centers = pt
        self.bin_index_xf = np.tile(np.arange(n_x), n_p)
        self.bin_index_pt = np.repeat(np.arange(n_p), n_x)
        self.data = np.ones(n_x * n_p)


# ---------------------------------------------------------------------------
# The deviance and delta-AIC algebra on pinned fixtures
# ---------------------------------------------------------------------------
def test_deviance_algebra_pinned_fixture():
    """The deviance algebra against a hand-computed fixture: the summed
    squared normalized residual with sigma^2 = v_data + se^2 (the SE
    floored), the structural drops, the pooled SE of the difference, and
    the held-out degradation verdict predicate."""
    d = np.asarray([1.0, 2.0, 3.0], dtype=float)
    v_data = np.asarray([1.0, 1.0, 1.0], dtype=float)
    se = np.asarray([0.1, 0.2, 0.1], dtype=float)
    m = np.asarray([1.2, 0.8, 1.5], dtype=float)
    expected = math.fsum(
        (float(d[i]) - float(m[i])) ** 2
        / (float(v_data[i]) + float(se[i]) ** 2)
        for i in range(3)
    )
    got = r44.observable_deviance("fixture", d, v_data, m, se)
    assert got == pytest.approx(expected, rel=1e-12)
    # the SE floor enters the sigma when the between-replica SE
    # collapses below it
    se_zero = np.asarray([0.0, 0.0, 0.0], dtype=float)
    expected_zero = math.fsum(
        (float(d[i]) - float(m[i])) ** 2
        / (float(v_data[i]) + r44.SE_FLOOR ** 2)
        for i in range(3)
    )
    assert r44.observable_deviance(
        "fixture", d, v_data, m, se_zero) == pytest.approx(
        expected_zero, rel=1e-12)
    # the structural drop: a non-finite model bin is dropped and listed
    m_bad = np.asarray([1.0, float("nan"), 3.0], dtype=float)
    dropped = []
    r44.record_deviance_drops("fixture", d, v_data, m_bad, se, dropped)
    got_bad = r44.observable_deviance(
        "fixture", d, v_data, m_bad, se)
    assert len(dropped) == 1 and dropped[0]["index"] == 1
    expected_bad = (float(d[0]) - 1.0) ** 2 / (1.0 + 0.01) + (
        float(d[2]) - 3.0) ** 2 / (1.0 + 0.01)
    assert got_bad == pytest.approx(expected_bad, rel=1e-12)
    # the pooled SE of the difference: std ddof=1 / sqrt(20), floored
    deltas = [0.25 * k for k in range(1, 21)]
    expected_se = float(np.std(deltas, ddof=1)) / math.sqrt(20)
    assert r44.pooled_difference_se(deltas) == pytest.approx(
        expected_se, rel=1e-12)
    assert r44.pooled_difference_se([1.0]) == r44.SE_FLOOR
    assert r44.pooled_difference_se([]) == r44.SE_FLOOR
    assert r44.pooled_difference_se([float("inf"), 1.0]) == r44.SE_FLOOR
    assert r44.HELD_OUT_DEGRADATION_FACTOR == 2.0
    assert r44.SECTOR_DEGRADATION_FRACTION == 0.05


def test_sigma_components_data_and_mc_contributions():
    """The -2ln L register reports the data and MC contributions
    separately (the card's likelihood clause): the total form
    sigma^2 = v_data + se^2, the data-only form sigma^2 = v_data, and
    the MC-only form sigma^2 = se^2 (floored), each over the same
    finite bins."""
    d = np.asarray([1.0, 2.0, 3.0], dtype=float)
    v_data = np.asarray([1.0, 1.0, 1.0], dtype=float)
    se = np.asarray([0.1, 0.2, 0.1], dtype=float)
    m = np.asarray([1.2, 0.8, 1.5], dtype=float)
    comp = r44.sigma_components(d, v_data, m, se)
    total = math.fsum(
        (float(d[i]) - float(m[i])) ** 2
        / (float(v_data[i]) + float(se[i]) ** 2)
        for i in range(3)
    )
    data_only = math.fsum(
        (float(d[i]) - float(m[i])) ** 2 / float(v_data[i])
        for i in range(3)
    )
    mc_only = math.fsum(
        (float(d[i]) - float(m[i])) ** 2 / float(se[i]) ** 2
        for i in range(3)
    )
    assert comp["minus2lnl"] == pytest.approx(total, rel=1e-12)
    assert comp["minus2lnl_data_only"] == pytest.approx(
        data_only, rel=1e-12)
    assert comp["minus2lnl_mc_only"] == pytest.approx(mc_only, rel=1e-12)
    assert comp["n_bins"] == 3
    # the total sits below both single-variance forms here (the added
    # variance lowers every squared normalized residual)
    assert comp["minus2lnl"] < comp["minus2lnl_data_only"]
    assert comp["minus2lnl"] < comp["minus2lnl_mc_only"]
    # a non-finite model bin drops from all three forms together
    comp_bad = r44.sigma_components(
        d, v_data, m_bad := np.asarray([1.0, float("nan"), 3.0]), se)
    assert comp_bad["n_bins"] == 2
    assert math.isfinite(comp_bad["minus2lnl"])
    assert math.isfinite(comp_bad["minus2lnl_data_only"])
    assert math.isfinite(comp_bad["minus2lnl_mc_only"])


def test_delta_aic_algebra_pinned_fixture():
    """The delta-AIC algebra on a pinned likelihood fixture (plan
    section 5.5 condition 6, line 322): AIC = -2ln L + 2k with the
    aligned configuration carrying kappa_B, kappa_m, b_dec,
    nominal_width (k = 4) and the baseline carrying nominal_width
    (k = 1); the parameter cost is included and favored is delta < 0."""
    aligned_total = 100.0
    baseline_total = 100.0
    block = r44.delta_aic_block(aligned_total, baseline_total)
    assert block["k_aligned"] == 4 and block["k_baseline"] == 1
    assert block["parameter_cost"] == "included"
    # equal likelihoods: the parameter cost alone gives delta = 6
    assert block["delta_aic"] == pytest.approx(6.0, rel=1e-12)
    assert block["favored"] is False
    # the aligned configuration wins by more than the cost: favored
    block = r44.delta_aic_block(100.0, 110.0)
    assert block["aic_aligned"] == pytest.approx(108.0, rel=1e-12)
    assert block["aic_baseline"] == pytest.approx(112.0, rel=1e-12)
    assert block["delta_aic"] == pytest.approx(-4.0, rel=1e-12)
    assert block["favored"] is True
    # the exact break-even: the aligned configuration is favored only
    # when its -2ln L is better by more than the cost difference
    block = r44.delta_aic_block(94.0, 100.0)
    assert block["delta_aic"] == pytest.approx(0.0, abs=1e-12)
    assert block["favored"] is False
    block = r44.delta_aic_block(93.999, 100.0)
    assert block["favored"] is True


def test_held_out_degradation_rows():
    """The held-out row verdict (condition iii): a held-out observable
    whose deviance degrades by more than 2.0 x the pooled MC SE of the
    difference is a named finding; a degradation inside the band and an
    improvement are not."""
    row = r44.held_out_row(10.0, 4.0, 2.0)
    assert row["delta_aligned_minus_baseline"] == pytest.approx(6.0)
    assert row["beyond_2x_se"] is True and row["named_finding"] is True
    row = r44.held_out_row(5.0, 4.0, 2.0)
    assert row["degrades"] is True and row["beyond_2x_se"] is False
    assert row["named_finding"] is False
    row = r44.held_out_row(4.0, 5.0, 2.0)
    assert row["degrades"] is False and row["named_finding"] is False
    # the exact threshold: delta == 2.0 x SE is not a finding (beyond,
    # not at)
    row = r44.held_out_row(8.0, 4.0, 2.0)
    assert row["beyond_2x_se"] is False
    assert r44.SIGMA_TOLERANCE == 3.0


def test_depth_stats_and_pinned_conventions():
    """The depth statistics (the R40 convention) and the pre-registered
    bounds recorded on the driver."""
    hist = {0: 30, 1: 40, 2: 20, 3: 10}
    stats = r44.depth_stats(hist, 100)
    assert stats["mean"] == pytest.approx(1.1)
    assert stats["p95"] == 3
    assert stats["max"] == 3
    assert stats["guard_activations"] == 0
    with pytest.raises(ValueError):
        r44.depth_stats(hist, 99)
    assert r44.MAX_CASCADE_DEPTH >= 1


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
    """The canonical exchange-layer hash is a pure function of the
    mapping bytes (deterministic) and sensitive to every pinned field;
    two recomputations of the same mapping agree and any component
    change moves the digest."""
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    first = r44.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
    second = r44.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
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
            digest = r44.exchange_event_hash([mutated], 42, 0.5, [1, 0, 1])
        except Exception:
            continue
        assert digest != first


def test_exchange_bit_identity_pinned_event():
    """On a pinned synthetic exchange event, both variants receive the
    identical exchange mapping (the canonical digest of each variant's
    own view equals the shared digest), the mapping is never mutated,
    and the exchange root is the draw-free pure function of the event;
    both variants conserve the event exactly."""
    from na49_mixing.cascade_ablation import exchange_root_state
    from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL

    catalog = r44._load_catalog()
    fams = r44._load_families()
    tables = _synthetic_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams, smoke=True))
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    snapshot = copy.deepcopy(event)
    exchange = r44._EventCapture()
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
    exchange.exchange_hash = r44.exchange_event_hash(
        [entry["exchange"] for entry in exchange.sources],
        42, 0.1, exchange.route_census)
    variants = r44.decay_event_variants(
        catalog, fams, exchange, smoke=True)
    assert event == snapshot, "the exchange event was mutated"
    root = r44.exchange_root_state(event)
    assert root.family_id == "baryonic_np" and root.mass == 4.0
    assert root.qn == BARYONIC_QN
    assert root.axis == (0.0, 0.0, 1.0)
    assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert root.depth == 0
    # both variants conserved the event exactly
    for name in r44.VARIANT_NAMES:
        run = variants[name]["run"]
        assert run.bqs == (2, 2, 0)
        assert run.worst_closure <= r44.EVENT_CLOSURE_TOL
    # a mutated mapping is refused (the per-variant recomputation leg)
    bad = r44._EventCapture()
    bad.event_id = 42
    bad.t = 0.1
    bad.route_census = [1, 0, 1]
    mutated = json.loads(json.dumps(event))
    mutated["axis"] = (0.0, 1.0, 0.0)
    bad.sources = [{
        "source_id": 0, "role": "projectile", "exchange": mutated,
        "route": "continuum", "record_pdg": None, "record_name": None,
        "n_open_records": None, "stable_identity": False, "energy": 4.0,
    }]
    bad.energy_expected = 4.0
    bad.exchange_hash = r44.exchange_event_hash(
        [event], 42, 0.1, exchange.route_census)
    with pytest.raises(r44.ConservationViolation):
        r44.decay_event_variants(catalog, fams, bad, smoke=True)
    r44._set_worker_context(None)


def test_feed_down_lineage_tally():
    """The card-9 feed-down lineage convention carried on the walk
    machinery: a final is resonant feed-down when its producing vertex's
    parent is a resonance other than the drawn root (the catalog route's
    drawn record, the continuum route's sentinel 0); the stable-identity
    path (parent None) is a direct source product."""
    catalog = r44._load_catalog()
    tables = _synthetic_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams=None,
                                                smoke=True))
    run = r44._VariantRun(0)
    # a continuum-route final produced by a non-root record: resonant
    run.add_final(211, (1.0, 0.1, 0.1, 0.2),
                  parent_pdg=213, root_pdg=0)
    slot = run.feeddown["per_species"]["pi_plus"]
    assert slot["n"] == 1 and slot["n_resonant"] == 1
    assert slot["by_parent"] == {"213": 1}
    # a continuum-route final produced by the root sentinel: direct
    run.add_final(-211, (0.2, -0.1, 0.0, 0.1),
                  parent_pdg=0, root_pdg=0)
    slot = run.feeddown["per_species"]["pi_minus"]
    assert slot["n"] == 1 and slot["n_resonant"] == 0
    # a catalog-route final produced by the drawn root record: direct
    run.add_final(2212, (1.0, 0.0, 0.0, 0.9),
                  parent_pdg=2212, root_pdg=2212)
    slot = run.feeddown["per_species"]["proton"]
    assert slot["n"] == 1 and slot["n_resonant"] == 0
    # the stable-identity path: parent None, never resonant
    run.add_final(2212, (1.5, 0.0, 0.0, 1.2),
                  parent_pdg=None, root_pdg=2212)
    slot = run.feeddown["per_species"]["proton"]
    assert slot["n"] == 2 and slot["n_resonant"] == 0
    assert run.feeddown["n_finals"] == 4
    assert run.bqs == (2, 2, 0)
    r44._set_worker_context(None)


# ---------------------------------------------------------------------------
# The CLI surface (the seven modes)
# ---------------------------------------------------------------------------
def test_cli_surface():
    parser = r44._build_parser()
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
        r44.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        r44.main(["--evaluate", "--gate"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and sit
    at the driver's self-computed identity digest."""
    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r44_frozen_nonstrange_validation.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r44_frozen_nonstrange_validation.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest
    assert nested_digest == r44.DRIVER_SHA256


# ---------------------------------------------------------------------------
# The freeze re-pin register (the card-15 landing)
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r44_frozen_nonstrange_validation.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r44_frozen_nonstrange_validation.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r44_frozen_nonstrange_validation.py"
)
SOLVER_CARD = "TASK-20260920-r4-run-r44-frozen-nonstrange-validation"
REPOINT_CARD = "TASK-20260920-r4-run-r44-repoint"

#: The twenty register-chain files whose inventory-count pins card 15
#: re-pins (488 -> 491, +3 entries): the sixteen chain register files
#: (cards 1, 4, 5, 6, 7, 8, 9, 10, 11, 12) plus the card-13 solver
#: test's own count constant (test_staged_r42_recoil_decay_separation.py),
#: the card-13 test-author register's own count constant
#: (test_staged_r42_recoil_decay_separation_contract_pins.py), the
#: card-14 solver test's own count constant
#: (test_staged_r43_cascade_ablation.py), and the card-14 test-author
#: register's own count constant
#: (test_staged_r43_cascade_ablation_contract_pins.py).
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
)
# re-pinned 487 -> 488 by card 14 test-author (TASK-20260920-r4-run-
# r43-cascade-ablation-test-author; +1 entry: the card-14
# test-author register file).
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned in place by card 19 (TASK-20260920-r4-run-r44-repoint;
# no new files: the same three entries re-pinned to the card-19
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).
# on-disk digests, the count stays 491).
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
    tables.append(r44._pi_zero_model_table(tables))
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


def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r44.MONO_ROOT / r44.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r44.run_smoke(default_dir, workers=1)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention): a run dir whose pre-registration names the smoke
    mode is refused."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r44.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r44._refuse_smoke_artifacts(run_dir)
    # a smoke-named artifact is refused even with a production
    # pre-registration
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r44.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_replica_0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r44._refuse_smoke_artifacts(run_dir2)
    # a run dir with no pre-registration is refused too
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r44._refuse_smoke_artifacts(run_dir3)


# ---------------------------------------------------------------------------
# The non-rectangular published geometry (the R01 pick convention)
# ---------------------------------------------------------------------------
#: The hole-geometry fixture: a 4x3 lattice per species whose published
#: rows are a NON-rectangular, NON-pT-major subset of the lattice (the
#: real published R01 tables' shape: pi_minus 229 of 324 cells, rows not
#: pT-major).  Per species the holes (dense cells ip*n_x + ix with no
#: published row) differ, and the row order is deterministically
#: shuffled, so a dense cell address and a published row address are
#: different addresses and the dense-rectangle fixtures above cannot
#: see the pick convention.
HOLEY_N_X = 4
HOLEY_N_P = 3
#: The per-species hole cells (dense lattice indices ip * n_x + ix).
HOLEY_HOLES = {
    "pi_plus": {1, 5},
    "pi_minus": {2, 7, 11},
    "k_plus": {3, 9},
    "k_minus": {5, 6},
    "proton": {0, 10},
    "antiproton": {6, 8, 10},
}


def _holey_tables():
    """The seven-species hole-geometry table set (six paper-window
    species with per-species holes and shuffled row order plus the
    model-only pi0 block on the published pi+ grid geometry)."""
    xf = np.linspace(0.0, 0.4, HOLEY_N_X + 2)[1:-1]
    pt = np.linspace(0.05, 1.5, HOLEY_N_P + 2)[1:-1]
    window: dict = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    tables = []
    for species, pdg, mass in (
        ("pi_plus", 211, 0.1395699), ("pi_minus", -211, 0.1395699),
        ("k_plus", 321, 0.4936770), ("k_minus", -321, 0.4936770),
        ("proton", 2212, 0.9382720), ("antiproton", -2212, 0.9382720),
    ):
        holes = HOLEY_HOLES[species]
        cells = [c for c in range(HOLEY_N_X * HOLEY_N_P)
                 if c not in holes]
        # the deterministic shuffle: rows not pT-major (the published
        # R01 tables' row order)
        order = np.random.default_rng(
            20260920 + HOLEY_HOLES[species].__len__()).permutation(
            len(cells))
        rows = [cells[int(i)] for i in order]
        xf_pts = np.asarray([float(xf[c % HOLEY_N_X]) for c in rows])
        pt_pts = np.asarray([float(pt[c // HOLEY_N_X]) for c in rows])
        ones = np.ones(len(rows), dtype=float)
        from na49_mixing.staged_evaluation import BinnedFitTable

        tables.append(BinnedFitTable.from_arrays(
            species=species, pdg_id=pdg, mass_gev=float(mass),
            normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
            path="test://holey-grid",
            xf=xf_pts, pt=pt_pts, value=ones, error=ones,
            window=window,
        ))
    tables.append(r44._pi_zero_model_table(tables))
    return tables


def _holey_cell_of(table, row: int) -> tuple[int, int]:
    """The dense lattice cell (ix, ip) of one published row."""
    return (int(table.bin_index_xf[row]), int(table.bin_index_pt[row]))


def _holey_final(ix: int, ip: int, xf_centers, pt_centers, scale: float):
    """A final four-vector whose (x_F, pT) is the (ix, ip) cell center,
    plus the exact Jacobian weight ``e / pT`` the driver records for it
    (the same double operands, so the expected weight matches the
    recorded weight bit for bit)."""
    x_f = float(xf_centers[ix])
    p_t = float(pt_centers[ip])
    sqrt_s = r44._sqrt_s()
    pz = x_f * sqrt_s / 2.0
    # e/pT = scale with pT = pt: e = scale * pt
    e = scale * p_t
    weight = e / p_t
    return (float(e), float(p_t), 0.0, float(pz)), weight


def _holey_sample(tables):
    """One final per PUBLISHED row (distinct exact weight per row) plus
    one final per HOLE cell (distinct weight, >= 1000).  Returns the
    per-species cell -> weight maps."""
    xf_by = {t.species: t.xf_centers for t in tables}
    pt_by = {t.species: t.pt_centers for t in tables}
    finals = []
    counter = 0
    for table in tables:
        species = table.species
        for row in range(int(table.data.size)):
            ix, ip = _holey_cell_of(table, row)
            scale = 1.0 + 0.125 * counter
            counter += 1
            finals.append((species, ix, ip, scale))
    hole_finals = []
    h_counter = 0
    for species, holes in HOLEY_HOLES.items():
        for cell in sorted(holes):
            ix, ip = cell % HOLEY_N_X, cell // HOLEY_N_X
            scale = 1000.0 + 0.125 * h_counter
            h_counter += 1
            hole_finals.append((species, ix, ip, scale))
    return finals, hole_finals


def _holey_record(tables, finals, hole_finals):
    """Drive the finals through ``_VariantRun.add_final`` (dense
    accumulation) and assemble the record (the pick), returning the
    record, the expected published-row weights, and the hole weights."""
    xf_by = {t.species: t.xf_centers for t in tables}
    pt_by = {t.species: t.pt_centers for t in tables}
    run = r44._VariantRun(0)
    cell_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in finals:
        pdg = r44.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        run.add_final(pdg, p4, parent_pdg=None, root_pdg=0)
        cell_weights[species][(ix, ip)] = weight
    hole_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in hole_finals:
        pdg = r44.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        run.add_final(pdg, p4, parent_pdg=None, root_pdg=0)
        hole_weights[species][(ix, ip)] = weight
    slot = {
        "hashes": [], "row_flat": list(run.flat),
        "row_weight": list(run.weights), "row_ptr": [0, len(run.flat)],
        "routing": [1, 0, 1], "depth_hist": {}, "mode_census": {},
        "mult": {k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                 for k in r44.SPECIES_KEYS},
        "feeddown": r44._empty_feeddown(r44.SPECIES_KEYS),
        "worst_closure": 0.0, "n_finals": int(run.n_finals),
        "event_digest": "0" * 64, "exchange_digest": "0" * 64,
    }
    aggregated = {
        "n_events": 1,
        "shared_exchange_digest": "0" * 64,
        "pre_registration_sha256": "0" * 64,
        "per_variant": {name: slot for name in r44.VARIANT_NAMES},
    }
    record = r44._replica_record(0, 20301020, 0, 1, aggregated)
    return record, cell_weights, hole_weights


def _holey_dense_reference(context, cell_weights, hole_weights):
    """The independent dense totals: every published cell and hole cell
    weight placed at its dense lattice address (the R01 accumulation
    space), built from the cell maps alone."""
    dense = np.zeros(int(context["dense_total"]), dtype=float)
    for species, cells in cell_weights.items():
        base = int(context["dense_offsets_by_species"][species])
        n_x = int(context["n_x_by_species"][species])
        for (ix, ip), scale in cells.items():
            dense[base + ip * n_x + ix] += scale
    for species, cells in hole_weights.items():
        base = int(context["dense_offsets_by_species"][species])
        n_x = int(context["n_x_by_species"][species])
        for (ix, ip), scale in cells.items():
            dense[base + ip * n_x + ix] += scale
    return dense


def test_holey_geometry_holes_dropped_and_rows_aligned():
    """The non-rectangular published geometry (holes; rows not
    pT-major): the finals accumulate densely and the record picks the
    published rows, so (a) a hole cell's weight is dropped (recorded
    nowhere, bled into no neighboring row or species) and (b)
    ``pooled_sums`` align 1:1 with the published rows -- equal to the
    direct pick of the independent dense totals, and NOT equal to the
    dense-position-consumed reading (the pre-fix convention)."""
    tables = _holey_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams=None,
                                                smoke=True))
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        context = r44._WORKER_CONTEXT
        dense = _holey_dense_reference(context, cell_weights,
                                       hole_weights)
        picks = context["pick_by_species"]
        sizes = context["dense_sizes_by_species"]
        offsets = context["dense_offsets_by_species"]
        for name in r44.VARIANT_NAMES:
            pooled = record["variants"][name]["pooled_sums"]
            for table in tables:
                species = table.species
                values = pooled[species]
                assert len(values) == int(table.data.size), (
                    f"{species}: pooled_sums must be published-row sized")
                # (b) row r holds the weight of row r's own lattice cell
                for row in range(int(table.data.size)):
                    ix, ip = _holey_cell_of(table, row)
                    assert values[row] == cell_weights[species][(ix, ip)], (
                        f"{species} row {row}: cell (ix={ix}, ip={ip}) "
                        "weight drifted")
                # the direct pick of the independent dense totals agrees
                # exactly
                dense_slice = dense[
                    int(offsets[species]):
                    int(offsets[species]) + int(sizes[species])]
                direct_pick = [float(v) for v in dense_slice[picks[species]]]
                assert direct_pick == values
                # (the pre-fix convention would consume dense cell
                # addresses as published rows; on this geometry that is
                # a different vector)
                consumed = [
                    float(v) for v in dense_slice[:int(table.data.size)]]
                assert consumed != values, (
                    f"{species}: the dense-position-consumed reading "
                    "equals the picked rows; the fixture does not "
                    "discriminate the two conventions")
                # (a) every hole cell's weight is recorded nowhere
                flat_all = [v for key in r44.SPECIES_FLAT_ORDER
                            for v in pooled[key]]
                for (ix, ip), scale in hole_weights[species].items():
                    assert scale not in flat_all, (
                        f"{species}: hole cell (ix={ix}, ip={ip}) weight "
                        "bled into a recorded row")
                # the recorded total carries the published cells only
                assert math.fsum(values) == pytest.approx(
                    math.fsum(cell_weights[species].values()), rel=1e-12,
                    abs=0.0)
    finally:
        r44._set_worker_context(None)


def test_holey_geometry_pi_zero_follows_published_pick():
    """The pi0 model-only diagnostic rows use the published pi+ grid
    geometry at the pi0 mass and pdg and go through the same pick
    mapping: the pi0 pick equals the pi+ pick (same published rows),
    the pi0 pooled_sums align with the pi0 rows, and a pi0 final in a
    pi0 hole cell (a pi+ hole) is dropped."""
    tables = _holey_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams=None,
                                                smoke=True))
    try:
        pi0 = next(t for t in tables if t.species == "pi_zero")
        pi_plus = next(t for t in tables if t.species == "pi_plus")
        context = r44._WORKER_CONTEXT
        # the same published grid geometry as pi+ (same rows, same pick)
        assert int(pi0.data.size) == int(pi_plus.data.size)
        assert (pi0.bin_index_xf == pi_plus.bin_index_xf).all()
        assert (pi0.bin_index_pt == pi_plus.bin_index_pt).all()
        assert (context["pick_by_species"]["pi_zero"]
                == context["pick_by_species"]["pi_plus"]).all()
        # pi0 finals on every pi0 published row plus the two pi0 hole
        # cells (the pi+ holes {1, 5})
        finals = []
        for row in range(int(pi0.data.size)):
            ix, ip = _holey_cell_of(pi0, row)
            finals.append(("pi_zero", ix, ip, 3.0 + 0.25 * row))
        hole_finals = [
            ("pi_zero", 1 % HOLEY_N_X, 1 // HOLEY_N_X, 2000.0),
            ("pi_zero", 5 % HOLEY_N_X, 5 // HOLEY_N_X, 2000.125),
        ]
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        for name in r44.VARIANT_NAMES:
            pooled = record["variants"][name]["pooled_sums"]
            values = pooled["pi_zero"]
            assert len(values) == int(pi0.data.size)
            for row in range(int(pi0.data.size)):
                ix, ip = _holey_cell_of(pi0, row)
                assert values[row] == cell_weights["pi_zero"][(ix, ip)]
            # the pi0 hole-cell finals are dropped
            flat_all = [v for key in r44.SPECIES_FLAT_ORDER
                        for v in pooled[key]]
            for scale in (2000.0, 2000.125):
                assert scale not in flat_all
            # the other species carry nothing (no cross-species bleed)
            for key in r44.SPECIES_FLAT_ORDER:
                if key != "pi_zero":
                    assert all(v == 0.0 for v in pooled[key])
    finally:
        r44._set_worker_context(None)


def test_holey_geometry_evaluation_row_areas_agree():
    """The evaluation's per-published-row area division agrees with the
    recorded rows end to end: ``_species_bin_values`` on the record's
    pooled_sums returns, for every published row r, the row's own
    Jacobian sum divided by the row's own cell area (the self-normalized
    estimator on the published bins), with no hole-cell contamination."""
    tables = _holey_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams=None,
                                                smoke=True))
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r44._sqrt_s()
        for name in r44.VARIANT_NAMES:
            weighted = r44._variant_sums(record, name)
            values = r44._species_bin_values(
                weighted, tables_by_species, sqrt_s, 1.0)
            for table in tables:
                species = table.species
                got = np.asarray(values[species], dtype=float)
                assert got.size == int(table.data.size)
                for row in range(int(table.data.size)):
                    ix, ip = _holey_cell_of(table, row)
                    area = r44._cell_area_from_table(table, row)
                    expected = cell_weights[species][(ix, ip)] / (
                        math.pi * float(sqrt_s) * 1.0 * float(area))
                    assert got[row] == pytest.approx(
                        expected, rel=1e-15, abs=0.0), (
                        f"{species} row {row}: the evaluation's "
                        "per-published-row division disagrees with the "
                        "recorded row")
                # a hole cell's scaled weight appears in no row value
                # (the published weights stay below the hole weights)
                flat_all = np.concatenate([
                    np.asarray(values[key], dtype=float)
                    for key in r44.SPECIES_FLAT_ORDER])
                for (_ix, _ip), scale in hole_weights[species].items():
                    assert not np.any(flat_all * math.pi * float(sqrt_s)
                                      >= scale * 0.999)
    finally:
        r44._set_worker_context(None)


def test_holey_geometry_pick_map_and_duplicate_refusal():
    """The pick map is built in the worker context from the tables'
    bin_index arrays (pick = bin_index_pt * n_x + bin_index_xf), the
    dense layout covers every lattice cell, and a table with two
    published rows on one dense lattice cell refuses (the pick map must
    stay one-to-one)."""
    tables = _holey_tables()
    r44._init_species_layout(tables)
    r44._set_worker_context(r44._worker_context(tables, fams=None,
                                                smoke=True))
    try:
        context = r44._WORKER_CONTEXT
        total = 0
        for table in tables:
            species = table.species
            n_x = int(context["n_x_by_species"][species])
            n_p = int(context["n_p_by_species"][species])
            expected = (np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
                        + np.asarray(table.bin_index_xf, dtype=np.int64))
            pick = context["pick_by_species"][species]
            assert (pick == expected).all()
            assert int(context["dense_sizes_by_species"][species]) == \
                n_x * n_p
            assert int(pick.max()) < n_x * n_p
            total += n_x * n_p
        assert int(context["dense_total"]) == total
        # the dense accumulation space exceeds the published-row space
        # on this geometry (the holes)
        assert int(context["dense_total"]) > int(r44._flat_total())
    finally:
        r44._set_worker_context(None)
    # a duplicated lattice cell refuses (the pick map must be 1:1)
    dup = _holey_tables()[0]
    same_xf = float(dup.xf_centers[int(dup.bin_index_xf[0])])
    same_pt = float(dup.pt_centers[int(dup.bin_index_pt[0])])
    from na49_mixing.staged_evaluation import BinnedFitTable

    window: dict = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    dup_table = BinnedFitTable.from_arrays(
        species="dup_probe", pdg_id=991, mass_gev=0.5,
        normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
        path="test://duplicate-cell",
        xf=np.asarray([same_xf, same_xf]),
        pt=np.asarray([same_pt, same_pt]),
        value=np.ones(2), error=np.ones(2),
        window=window,
    )
    r44._set_worker_context(None)
    with pytest.raises(SystemExit):
        r44._worker_context([dup_table], fams=None, smoke=True)
    r44._set_worker_context(None)
