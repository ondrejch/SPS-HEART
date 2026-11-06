"""Hermetic tests for the R43 driver (the cascade ablation).

Task ``TASK-20260920-r4-run-r43-cascade-ablation`` (Round 4, card 14;
plan section 10 run R43, lines 790-798), with the recording surface
reworked to the R01 accumulate-dense-then-pick convention by the card-18
relitigation (TASK-20260920-r4-run-r43-relitigation): the seed-map and
event-index partition, the exchange bit-identity on a pinned event, the
variant stream disjointness on a pinned key, the deviance algebra on a
pinned fixture, the four hole-geometry tests of the corrected recording
convention (mirrored from the R44 driver's card-15 coverage and the
card-16/card-17 R41/R42 adaptations, adapted to the R43 three-variant
recording path), the layout sync, and the freeze re-pin register.  No
Monte Carlo generation at run scale, no SPS table, no sealed archive
read by the acceptance legs (the sealed-dir pins skip on a fresh
checkout): the acceptance is hermetic.
"""

import hashlib
import importlib.util
import copy
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
        / "staged_r43_cascade_ablation.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r43_cascade_ablation", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R43 driver was not found next to the test file")


r43 = _load_driver()


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    assert r43.MASTER_SEED == 20260920
    assert r43.RAW_REPLICA_SEEDS == tuple(20301020 + r for r in range(8))
    assert r43.EVENT_INDEX_RULE == "600_000_000 + i, i < 100,000"
    assert r43.EVENT_INDEX_BASE == 600_000_000
    assert r43.T2_REPLICAS == 8 and r43.T2_EVENTS_PER_REPLICA == 100_000
    # the per-replica seeds are the block 20301020-20301027 (the R41/R42
    # block, shared by pre-registration; the exchange and decay-layer
    # spaces are disjoint by the event-index partition)
    assert [r43.raw_replica_seed(r) for r in range(8)] == [
        20301020 + r for r in range(8)]
    with pytest.raises(SystemExit):
        r43.raw_replica_seed(8)
    # the smoke block (+9000 raw seed; the +90M event-index block) is
    # disjoint from the production slots and spaces, from the R41 +8000
    # and R40 +7000 smoke blocks, and from the audit block
    assert r43.SMOKE_SEED_STRIDE == 9_000
    assert r43.SMOKE_RAW_SEED == 20260920 + 40_000 + 9_000 == 20309920
    assert r43.SMOKE_KERNEL_REPLICA_SLOT == 9_000
    assert 20309920 not in r43.RAW_REPLICA_SEEDS
    assert 20309920 not in range(20301040, 20301056)  # the audit block
    assert 20309920 != 20308920  # the R41 smoke block
    assert r43.SMOKE_EVENT_INDEX_BASE == 690_000_000
    assert r43.SMOKE_EVENT_INDEX_RULE == "690_000_000 + i, i < 2,000"
    # the production event-index partition is disjoint from R42's
    # (500M-based), R41's (400M-based), and R40's (300M-based)
    assert r43.EVENT_INDEX_BASE > 500_000_000
    # the rules name the disjointness and the block sharing
    assert "R41" in r43.RAW_SEED_RULE and "R42" in r43.RAW_SEED_RULE
    assert "derive_substreams(20260920, r, 600_000_000 + i)" in \
        r43.EXCHANGE_SEED_RULE
    assert "variant_stream(20260920, 600_000_000 + i, v)" in \
        r43.VARIANT_STREAM_RULE
    assert "20301040 + k" in r43.RAW_SEED_RULE


def test_stream_tree_disjointness():
    """The (master, replica slot, event index) exchange spaces are
    pairwise disjoint across the R43 production slots and the smoke
    slot, and disjoint from the R41 (400M), R42 (500M), and R40 (300M)
    spaces; the variant-spawn tree is a separate tree (card 10's
    contract)."""
    from na49_mixing.rng_substreams import derive_substreams

    streams = set()
    for replica in range(r43.T2_REPLICAS):
        subs = derive_substreams(
            r43.MASTER_SEED, replica, r43.EVENT_INDEX_BASE)
        for label in ("topology_selection", "exchange_kinematics",
                      "spectral_state", "decay_channel",
                      "decay_kinematics"):
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    subs = derive_substreams(
        r43.MASTER_SEED, r43.SMOKE_KERNEL_REPLICA_SLOT,
        r43.SMOKE_EVENT_INDEX_BASE)
    for label in ("topology_selection", "exchange_kinematics",
                  "spectral_state", "decay_channel",
                  "decay_kinematics"):
        assert int(subs.stream_for(label)) not in streams
    for replica in range(r43.T2_REPLICAS):
        r42_subs = derive_substreams(
            r43.MASTER_SEED, replica, 500_000_000)
        r41_subs = derive_substreams(
            r43.MASTER_SEED, replica, 400_000_000)
        r40_subs = derive_substreams(
            r43.MASTER_SEED, replica, 300_000_000)
        for label in ("topology_selection", "exchange_kinematics",
                      "spectral_state", "decay_channel",
                      "decay_kinematics"):
            assert int(r42_subs.stream_for(label)) not in streams
            assert int(r41_subs.stream_for(label)) not in streams
            assert int(r40_subs.stream_for(label)) not in streams


# ---------------------------------------------------------------------------
# The variant stream contract on a pinned key (card 10's spawn contract)
# ---------------------------------------------------------------------------
def test_variant_stream_disjointness_pinned_key():
    """The per-variant decay streams are child spawns keyed on the
    variant id, disjoint across variants (a draw on variant 0 never
    advances variant 1's or variant 2's stream), deterministic on the
    pinned key, and sensitive to the event index and the master seed
    (card 10's variant_stream at the production event-index base)."""
    pinned = r43.EVENT_INDEX_BASE
    # determinism on the pinned key
    for variant_id in range(len(r43.ABLATION_VARIANTS)):
        first = r43.variant_stream(r43.MASTER_SEED, pinned, variant_id)
        second = r43.variant_stream(r43.MASTER_SEED, pinned, variant_id)
        for gen_a, gen_b in zip(first, second):
            assert [gen_a.random() for _ in range(6)] == [
                gen_b.random() for _ in range(6)]
    # disjointness: a draw on variant 0 never advances the others
    streams = {
        vid: r43.variant_stream(r43.MASTER_SEED, pinned, vid)
        for vid in range(len(r43.ABLATION_VARIANTS))
    }
    states_before = {
        vid: tuple(gen.bit_generator.state for gen in pair)
        for vid, pair in streams.items()
    }
    for _ in range(16):
        streams[0][0].random()
        streams[0][1].random()
    for vid in (1, 2):
        states_after = tuple(
            gen.bit_generator.state for gen in streams[vid])
        assert states_after == states_before[vid], (
            f"variant {vid}: its stream advanced while variant 0 drew")
    # the three variant streams differ
    fresh = {vid: r43.variant_stream(pinned, vid, vid)
             for vid in range(len(r43.ABLATION_VARIANTS))}
    channel_sequences = [
        tuple(fresh[vid][0].random() for _ in range(8))
        for vid in range(len(r43.ABLATION_VARIANTS))
    ]
    assert len(set(channel_sequences)) == len(r43.ABLATION_VARIANTS)
    # the key is sensitive to the event index and the master seed
    seq_a = tuple(r43.variant_stream(r43.MASTER_SEED, pinned, 0)[0]
                  .random() for _ in range(4))
    seq_b = tuple(r43.variant_stream(r43.MASTER_SEED, pinned + 1, 0)[0]
                  .random() for _ in range(4))
    seq_c = tuple(r43.variant_stream(r43.MASTER_SEED + 1, pinned, 0)[0]
                  .random() for _ in range(4))
    assert seq_a != seq_b and seq_a != seq_c
    # the smoke event indices are outside the production partition
    with pytest.raises(ValueError):
        r43.variant_stream(r43.MASTER_SEED, pinned, 3)


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
    first = r43.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
    second = r43.exchange_event_hash([event], 42, 0.5, [1, 0, 1])
    assert first == second
    for mutate in (
        lambda e: e.__setitem__("mass", 4.0000001),
        lambda e: e.__setitem__("momentum", (4.0000001, 0.0, 0.0, 0.0)),
        lambda e: e.__setitem__("family_id", "baryonic_strange"),
        lambda e: e.__setitem__("axis", (0.0, 0.0, 0.9999999)),
        lambda e: e.__setitem__("topology", "other"),
    ):
        mutated = json.loads(json.dumps(event))
        mutated["momentum"] = [
            mutated["mass"], 0.0, 0.0, 0.0]
        mutated["mass"] = math.hypot(mutated["momentum"][0], 0.0)
        mutated = json.loads(json.dumps(event))
        mutate(mutated)
        try:
            digest = r43.exchange_event_hash([mutated], 42, 0.5, [1, 0, 1])
        except Exception:
            continue
        assert digest != first


def test_exchange_bit_identity_pinned_event():
    """On a pinned synthetic exchange event, all three variants receive
    the identical exchange mapping (the canonical digest of each
    variant's own view equals the shared digest), the mapping is never
    mutated, and the exchange root is the draw-free pure function of
    the event."""
    from na49_mixing.cascade_ablation import exchange_root_state
    from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL

    catalog = r43._load_catalog()
    fams = r43._load_families()
    tables = _synthetic_tables()
    r43._init_species_layout(tables)
    r43._set_worker_context(r43._worker_context(tables, fams, smoke=True))
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    snapshot = copy.deepcopy(event)
    exchange = r43._EventCapture()
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
    exchange.exchange_hash = r43.exchange_event_hash(
        [entry["exchange"] for entry in exchange.sources],
        42, 0.1, exchange.route_census)
    variants = r43.decay_event_variants(
        catalog, fams, exchange, smoke=True)
    assert event == snapshot, "the exchange event was mutated"
    root = r43.exchange_root_state(event)
    assert root.family_id == "baryonic_np" and root.mass == 4.0
    assert root.qn == BARYONIC_QN
    assert root.axis == (0.0, 0.0, 1.0)
    assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert root.depth == 0
    # every variant conserved the event exactly
    for name in r43.ABLATION_VARIANTS:
        run = variants[name]["run"]
        assert run.bqs == (2, 2, 0)
        assert run.worst_closure <= r43.EVENT_CLOSURE_TOL
    # a mutated mapping is refused (the per-variant recomputation leg)
    bad = r43._EventCapture()
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
    bad.exchange_hash = r43.exchange_event_hash(
        [event], 42, 0.1, exchange.route_census)
    with pytest.raises(r43.ConservationViolation):
        r43.decay_event_variants(catalog, fams, bad, smoke=True)
    r43._set_worker_context(None)


def test_support_boundary_recorded():
    """The pre-registered support boundary is card 10's block verbatim:
    v1's single-step support differs from v2/v3's multi-vertex support,
    the exchange event is identical across the variants, and no
    regeneration is required at the nominal point."""
    from na49_mixing.cascade_ablation import SUPPORT_BOUNDARY

    assert r43.SUPPORT_BOUNDARY is SUPPORT_BOUNDARY
    variants = r43.SUPPORT_BOUNDARY["variants"]
    assert variants["isotropic_direct"]["common_random_numbers"] == \
        "regenerate"
    assert variants["isotropic_sequential"][
        "common_random_numbers"] == "shared"
    assert variants["aligned_sequential"]["common_random_numbers"] == \
        "shared"
    assert r43.SUPPORT_BOUNDARY["v2_v3_support_identical"] is True
    assert r43.SUPPORT_BOUNDARY["regeneration_required"] == \
        ("isotropic_direct",)


def test_nominal_point_policies():
    """The card-10 policies: v1/v2 at the isotropic point, v3 at the
    pre-registered nominal point (kappa_B, kappa_M, b_dec) = (1, 1,
    1.0 GeV^-2), not the R41 fitted values."""
    register = r43.ABLATION_VARIANTS
    assert register == ("isotropic_direct", "isotropic_sequential",
                        "aligned_sequential")
    v1 = r43.ABLATION_POLICIES["isotropic_direct"]
    v2 = r43.ABLATION_POLICIES["isotropic_sequential"]
    v3 = r43.ABLATION_POLICIES["aligned_sequential"]
    assert (v1.kappa_b, v1.kappa_m, v1.b_dec) == (0.0, 0.0, 1.0)
    assert (v2.kappa_b, v2.kappa_m, v2.b_dec) == (0.0, 0.0, 1.0)
    assert (v3.kappa_b, v3.kappa_m, v3.b_dec) == (1.0, 1.0, 1.0)
    # the R41 fitted tuple (0.0, 4.0, 2.0 GeV^-2, 0.5 GeV) is not consumed
    assert (v3.kappa_b, v3.kappa_m) != (0.0, 4.0)


# ---------------------------------------------------------------------------
# The deviance algebra on a pinned fixture
# ---------------------------------------------------------------------------
def test_deviance_algebra_pinned_fixture():
    """The deviance algebra against a hand-computed fixture: the summed
    squared normalized residual with sigma^2 = v_data + se^2 (the SE
    floored), the structural drops, the pooled SE of the difference,
    and the beyond-isotropic verdict predicate."""
    d = np.asarray([1.0, 2.0, 3.0], dtype=float)
    v_data = np.asarray([1.0, 1.0, 1.0], dtype=float)
    se = np.asarray([0.1, 0.2, 0.1], dtype=float)
    dropped = []
    # hand-computed v2 deviance: sigma_i^2 = v_data_i + se_i^2
    m_v2 = np.asarray([1.2, 0.8, 1.5], dtype=float)
    expected_v2 = math.fsum(
        (float(d[i]) - float(m_v2[i])) ** 2
        / (float(v_data[i]) + float(se[i]) ** 2)
        for i in range(3)
    )
    got_v2 = r43.observable_deviance(
        "fixture", d, v_data, m_v2, se, dropped)
    assert got_v2 == pytest.approx(expected_v2, rel=1e-12)
    assert dropped == []
    # the SE floor enters the sigma when the between-replica SE
    # collapses below it
    m_v3 = np.asarray([1.1, 1.9, 2.8], dtype=float)
    se_zero = np.asarray([0.0, 0.0, 0.0], dtype=float)
    expected_v3 = math.fsum(
        (float(d[i]) - float(m_v3[i])) ** 2
        / (float(v_data[i]) + r43.SE_FLOOR ** 2)
        for i in range(3)
    )
    got_v3 = r43.observable_deviance(
        "fixture", d, v_data, m_v3, se_zero, dropped)
    assert got_v3 == pytest.approx(expected_v3, rel=1e-12)
    # the structural drop: a non-finite model bin is dropped and listed
    m_bad = np.asarray([1.0, float("nan"), 3.0], dtype=float)
    dropped = []
    r43.record_deviance_drops("fixture", d, v_data, m_bad, se, dropped)
    got_bad = r43.observable_deviance(
        "fixture", d, v_data, m_bad, se, dropped)
    assert len(dropped) == 1 and dropped[0]["index"] == 1
    expected_bad = (float(d[0]) - 1.0) ** 2 / (1.0 + 0.01) + (
        float(d[2]) - 3.0) ** 2 / (1.0 + 0.01)
    assert got_bad == pytest.approx(expected_bad, rel=1e-12)
    # the pooled SE of the difference: std ddof=1 / sqrt(n), floored
    deltas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    expected_se = float(np.std(deltas, ddof=1)) / math.sqrt(8)
    assert r43.pooled_difference_se(deltas) == pytest.approx(
        expected_se, rel=1e-12)
    assert r43.pooled_difference_se([1.0]) == r43.SE_FLOOR
    assert r43.pooled_difference_se([]) == r43.SE_FLOOR
    # the verdict predicate: positive AND beyond 2.0 x the pooled SE
    row = r43.beyond_isotropic_row(1.0, 0.4)
    assert row["pass"] is True and row["positive"] is True
    row = r43.beyond_isotropic_row(0.7, 0.4)
    assert row["pass"] is False and row["beyond_2x_se"] is False
    row = r43.beyond_isotropic_row(-1.0, 0.5)
    assert row["pass"] is False and row["positive"] is False
    assert r43.beyond_isotropic_row(1.01, 0.5)["pass"] is True
    assert r43.BEYOND_ISOTROPIC_FACTOR == 2.0


# ---------------------------------------------------------------------------
# The depth statistics (the R40 convention)
# ---------------------------------------------------------------------------
def test_depth_stats_and_first_moment():
    hist = {0: 30, 1: 40, 2: 20, 3: 10}
    stats = r43.depth_stats(hist, 100)
    assert stats["mean"] == pytest.approx(1.1)
    assert stats["p95"] == 3  # the cumulative count reaches 95 at depth 3
    assert stats["max"] == 3
    assert stats["guard_activations"] == 0
    with pytest.raises(ValueError):
        r43.depth_stats(hist, 99)
    cells = [{"xF_mid": 0.1}, {"xF_mid": 0.3}, {"xF_mid": 0.5}]
    model = np.asarray([1.0, 1.0, 2.0], dtype=float)
    moment = r43.first_moment(cells, model)
    assert moment == pytest.approx(
        (0.1 * 1.0 + 0.3 * 1.0 + 0.5 * 2.0) / 4.0)
    assert r43.first_moment(cells, np.zeros(3)) is None
    assert r43.first_moment(cells, np.full(3, float("nan"))) is None


# ---------------------------------------------------------------------------
# The CLI surface (the seven modes)
# ---------------------------------------------------------------------------
def test_cli_surface():
    parser = r43._build_parser()
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
        r43.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        r43.main(["--evaluate", "--gate"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and sit
    at the driver's self-computed identity digest."""
    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r43_cascade_ablation.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r43_cascade_ablation.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest
    assert nested_digest == r43.DRIVER_SHA256


# ---------------------------------------------------------------------------
# The freeze re-pin register (the card-14 landing)
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r43_cascade_ablation.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r43_cascade_ablation.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation.py"
)
SOLVER_CARD = "TASK-20260920-r4-run-r43-cascade-ablation"

#: The seventeen register-chain files whose inventory-count pins card 13
#: re-pinned (483 -> 484, then 484 -> 487 by this card, +3 entries):
#: the twelve register-chain files (cards 1, 4, 5, 6, 7, 8, 9, 10) plus
#: the card-10 solver test's own count constant
#: (test_cascade_ablation.py), the card-11 test-author register's own
#: count constant (test_staged_r40_alignment_scan_contract_pins.py),
#: the card-12 test-author register's own count constant
#: (test_staged_r41_nonstrange_pilot_fit_contract_pins.py), the card-13
#: solver test's own count constant
#: (test_staged_r42_recoil_decay_separation.py), and the card-13
#: test-author register's own count constant
#: (test_staged_r42_recoil_decay_separation_contract_pins.py).
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
)
# re-pinned 484 -> 487 by card 14 (TASK-20260920-r4-run-r43-
# cascade-ablation; +3 entries: the dual-tree R43 driver in
# both helper trees and the hermetic R43 test file).
# re-pinned 487 -> 488 by card 14 test-author (TASK-20260920-r4-run-
# r43-cascade-ablation-test-author; +1 entry: the card-14
# test-author register file).
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
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






# ---------------------------------------------------------------------------
# The hole-geometry suite (the R01 accumulate-dense-then-pick recording
# convention; mirrored from the R44 driver's card-15 test coverage and
# the card-16/card-17 R41/R42 adaptations, adapted to the R43 recording
# path and species surface: the six paper-window species, the three
# per-variant pooled assemblies, and the pooled-combination spectra and
# seagulls on the aligned cells; the pick map and the hole dropping must
# hold identically for all three variants, so the variant comparison
# stays common-random-number fair)
# ---------------------------------------------------------------------------
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
    """The six-species hole-geometry table set (the R43 recording
    surface: per-species holes and shuffled row order, rows not
    pT-major)."""
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
            20260920 + len(holes)).permutation(len(cells))
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
    sqrt_s = r43._sqrt_s()
    pz = x_f * sqrt_s / 2.0
    # e/pT = scale with pT = pt: e = scale * pt
    e = scale * p_t
    weight = e / p_t
    return (float(e), float(p_t), 0.0, float(pz)), weight


def _holey_sample(tables):
    """One final per PUBLISHED row (distinct exact weight per row) plus
    one final per HOLE cell (distinct weight, >= 1000).  Returns the
    per-species cell -> weight maps."""
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
    """Drive the same finals through ``_VariantRun.add_final`` (the
    dense recording) once per variant and assemble the record (the
    per-variant pick), returning the record, the expected published-cell
    weights, and the hole-cell weights."""
    xf_by = {t.species: t.xf_centers for t in tables}
    pt_by = {t.species: t.pt_centers for t in tables}
    per_variant: dict[str, dict] = {}
    cell_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    hole_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for name in r43.ABLATION_VARIANTS:
        run = r43._VariantRun(
            r43.ABLATION_VARIANTS.index(name))
        for species, ix, ip, scale in finals:
            pdg = r43.SPECIES_PDGS[species]
            p4, weight = _holey_final(
                ix, ip, xf_by[species], pt_by[species], scale)
            run.add_final(pdg, p4)
            cell_weights[species][(ix, ip)] = weight
        for species, ix, ip, scale in hole_finals:
            pdg = r43.SPECIES_PDGS[species]
            p4, weight = _holey_final(
                ix, ip, xf_by[species], pt_by[species], scale)
            run.add_final(pdg, p4)
            hole_weights[species][(ix, ip)] = weight
        per_variant[name] = {
            "hashes": [], "row_flat": list(run.flat),
            "row_weight": list(run.weights), "row_ptr": [0, len(run.flat)],
            "routing": [1, 0, 1], "depth_hist": {}, "mode_census": {},
            "mult": {k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                     for k in r43.SPECIES_KEYS},
            "worst_closure": 0.0, "n_finals": int(run.n_finals),
            "event_digest": "0" * 64, "exchange_digest": "0" * 64,
        }
    aggregated = {
        "n_events": 1,
        "shared_exchange_digest": "0" * 64,
        "pre_registration_sha256": "0" * 64,
        "per_variant": per_variant,
        "smoke": True,
    }
    record = r43._replica_record(0, 20301020, 0, 1, aggregated)
    return record, cell_weights, hole_weights


def _holey_context(tables):
    """The worker context of the holey tables (the dense layout and the
    pick maps), with the context left set for the recording path."""
    r43._init_species_layout(tables)
    r43._set_worker_context(r43._worker_context(tables, fams=None,
                                                smoke=True))
    return r43._WORKER_CONTEXT


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
    pT-major): the finals accumulate densely and every variant's record
    picks the published rows, so (a) a hole cell's weight is dropped
    (recorded nowhere, bled into no neighboring row or species) and (b)
    each variant's ``pooled_sums`` align 1:1 with the published rows --
    equal to the direct pick of the independent dense totals, and NOT
    equal to the dense-position-consumed reading (the superseded pre-fix
    convention).  Variant-invariant by construction (the common-random-
    number fairness)."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        context = r43._WORKER_CONTEXT
        dense = _holey_dense_reference(context, cell_weights,
                                       hole_weights)
        picks = context["pick_by_species"]
        sizes = context["dense_sizes_by_species"]
        offsets = context["dense_offsets_by_species"]
        for name in r43.ABLATION_VARIANTS:
            pooled = record["variants"][name]["pooled_sums"]
            for table in tables:
                species = table.species
                values = pooled[species]
                assert len(values) == int(table.data.size), (
                    f"{name}/{species}: pooled_sums must be published-"
                    "row sized")
                # (b) row r holds the weight of row r's own lattice cell
                for row in range(int(table.data.size)):
                    ix, ip = _holey_cell_of(table, row)
                    assert values[row] == cell_weights[species][(ix, ip)], (
                        f"{name}/{species} row {row}: cell (ix={ix}, "
                        f"ip={ip}) weight drifted")
                # the direct pick of the independent dense totals agrees
                # exactly
                dense_slice = dense[
                    int(offsets[species]):
                    int(offsets[species]) + int(sizes[species])]
                direct_pick = [
                    float(v) for v in dense_slice[picks[species]]]
                assert direct_pick == values
                # (the superseded pre-fix convention would consume dense
                # cell addresses as published rows; on this geometry that
                # is a different vector)
                consumed = [
                    float(v) for v in dense_slice[:int(table.data.size)]]
                assert consumed != values, (
                    f"{name}/{species}: the dense-position-consumed "
                    "reading equals the picked rows; the fixture does "
                    "not discriminate the two conventions")
                # (a) every hole cell's weight is recorded nowhere
                flat_all = [v for key in r43.SPECIES_FLAT_ORDER
                            for v in pooled[key]]
                for (ix, ip), scale in hole_weights[species].items():
                    assert scale not in flat_all, (
                        f"{name}/{species}: hole cell (ix={ix}, "
                        f"ip={ip}) weight bled into a recorded row")
                # the recorded total carries the published cells only
                assert math.fsum(values) == pytest.approx(
                    math.fsum(cell_weights[species].values()), rel=1e-12,
                    abs=0.0)
    finally:
        r43._set_worker_context(None)


def test_holey_geometry_variant_sums_follow_published_pick():
    """The per-variant pooled sums follow the published pick through the
    evaluation boundary (the R43 analog of the card-15/16/17
    combination case): ``_variant_sums`` re-assembles the published-row
    flat vector from the record's ``pooled_sums`` and
    ``_species_bin_values`` divides by the per-published-row areas, so
    every species row and every pooled-combination cell (pi_Sigma =
    pi+ + pi-, B_transport = p - pbar, and the pooled seagull moments
    built on them) equals its own-row contributions, and the
    dense-position-consumed reading gives a different combination (the
    superseded convention discriminated on this geometry)."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, _hole_weights = _holey_record(
            tables, finals, hole_finals)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r43._sqrt_s()
        sealed = {
            t.species: {
                "d": [float(2 + i % 3) for i in range(int(t.data.size))],
                "v_data": [0.25] * int(t.data.size),
                "xF_mid": [float(v) for v in t.xf_centers],
                "pT_mid": [float(v) for v in t.pt_centers],
            } for t in tables
        }
        stats = r43._build_statistics(sealed, tables_by_species, sqrt_s)
        for name in r43.ABLATION_VARIANTS:
            variant_sums = r43._variant_sums(record, name)
            assert variant_sums.size == int(r43._flat_total())
            values = r43._species_bin_values(
                variant_sums, tables_by_species, sqrt_s, 1.0)
            for table in tables:
                species = table.species
                got = np.asarray(values[species], dtype=float)
                assert got.size == int(table.data.size)
                for row in range(int(table.data.size)):
                    ix, ip = _holey_cell_of(table, row)
                    area = r43._cell_area_from_table(table, row)
                    expected = cell_weights[species][(ix, ip)] / (
                        math.pi * float(sqrt_s) * 1.0 * float(area))
                    assert got[row] == pytest.approx(
                        expected, rel=1e-15, abs=0.0), (
                        f"{name}/{species} row {row}: the evaluation's "
                        "per-published-row division disagrees with the "
                        "recorded row")
            for stat in stats:
                if stat.kind == "spectrum":
                    combo = r43._combination_model_statistic(
                        stat.combo,
                        values[stat.combo["pair"][0]],
                        values[stat.combo["pair"][1]])
                    cells = stat.combo["cells"]
                    assert combo.size == len(cells)
                    for index, cell in enumerate(cells):
                        pair_a = stat.combo["pair"][0]
                        pair_b = stat.combo["pair"][1]
                        ix_a = int(tables_by_species[pair_a].
                                   bin_index_xf[cell["bin_a"]])
                        ip_a = int(tables_by_species[pair_a].
                                   bin_index_pt[cell["bin_a"]])
                        weight_a = cell_weights[pair_a][(ix_a, ip_a)]
                        ix_b = int(tables_by_species[pair_b].
                                   bin_index_xf[cell["bin_b"]])
                        ip_b = int(tables_by_species[pair_b].
                                   bin_index_pt[cell["bin_b"]])
                        weight_b = cell_weights[pair_b][(ix_b, ip_b)]
                        expected = (
                            weight_a / (math.pi * float(sqrt_s)
                                        * r43._cell_area_from_table(
                                            tables_by_species[pair_a],
                                            cell["bin_a"]))
                            + stat.combo["pair"][2] * weight_b / (
                                math.pi * float(sqrt_s)
                                * r43._cell_area_from_table(
                                    tables_by_species[pair_b],
                                    cell["bin_b"])))
                        assert combo[index] == pytest.approx(
                            expected, rel=1e-15, abs=0.0), (
                            f"{name}/{stat.key} cell {index}: the "
                            "combination does not follow the two "
                            "species' published rows")
                else:
                    model = r43._seagull_model(stat, values)
                    finite = [v for v in model if v is not None]
                    assert all(math.isfinite(v) for v in finite)
        # the dense-position-consumed reading gives a different
        # combination (the superseded convention discriminated on this
        # geometry)
        context = r43._WORKER_CONTEXT
        dense = _holey_dense_reference(context, cell_weights, {})
        consumed_values = {}
        for species in r43.SPECIES_FLAT_ORDER:
            dense_slice = dense[
                int(context["dense_offsets_by_species"][species]):
                int(context["dense_offsets_by_species"][species])
                + int(context["dense_sizes_by_species"][species])]
            table = tables_by_species[species]
            areas = np.asarray([
                r43._cell_area_from_table(table, i)
                for i in range(int(table.data.size))
            ], dtype=float)
            consumed_values[species] = (
                dense_slice[:int(table.data.size)]
                / (math.pi * float(sqrt_s) * areas))
        spectrum = next(s for s in stats if s.kind == "spectrum")
        picked = r43._variant_sums(record, "isotropic_sequential")
        picked_values = r43._species_bin_values(
            picked, tables_by_species, sqrt_s, 1.0)
        picked_combo = r43._combination_model_statistic(
            spectrum.combo,
            picked_values[spectrum.combo["pair"][0]],
            picked_values[spectrum.combo["pair"][1]])
        consumed_combo = r43._combination_model_statistic(
            spectrum.combo,
            consumed_values[spectrum.combo["pair"][0]],
            consumed_values[spectrum.combo["pair"][1]])
        assert not np.allclose(picked_combo, consumed_combo), (
            "the dense-position-consumed combination equals the "
            "picked one; the fixture does not discriminate the two "
            "conventions")
    finally:
        r43._set_worker_context(None)


def test_holey_geometry_evaluation_row_areas_agree():
    """The evaluation's per-published-row area division agrees with the
    recorded rows end to end for every variant: ``_species_bin_values``
    on the variant's flat sums returns, for every published row r, the
    row's own Jacobian sum divided by the row's own cell area (the
    self-normalized estimator on the published bins), with no
    hole-cell contamination."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r43._sqrt_s()
        for name in r43.ABLATION_VARIANTS:
            variant_sums = r43._variant_sums(record, name)
            values = r43._species_bin_values(
                variant_sums, tables_by_species, sqrt_s, 1.0)
            for table in tables:
                species = table.species
                got = np.asarray(values[species], dtype=float)
                assert got.size == int(table.data.size)
                for row in range(int(table.data.size)):
                    ix, ip = _holey_cell_of(table, row)
                    area = r43._cell_area_from_table(table, row)
                    expected = cell_weights[species][(ix, ip)] / (
                        math.pi * float(sqrt_s) * 1.0 * float(area))
                    assert got[row] == pytest.approx(
                        expected, rel=1e-15, abs=0.0), (
                        f"{name}/{species} row {row}: the evaluation's "
                        "per-published-row division disagrees with the "
                        "recorded row")
                # a hole cell's scaled weight appears in no row value
                # (the published weights stay below the hole weights)
                flat_all = np.concatenate([
                    np.asarray(values[key], dtype=float)
                    for key in r43.SPECIES_FLAT_ORDER])
                for (_ix, _ip), scale in hole_weights[species].items():
                    assert not np.any(flat_all * math.pi * float(sqrt_s)
                                      >= scale * 0.999)
    finally:
        r43._set_worker_context(None)


def test_holey_geometry_pick_map_and_duplicate_refusal():
    """The pick map is built in the worker context from the tables'
    bin_index arrays (pick = bin_index_pt * n_x + bin_index_xf), the
    dense layout covers every lattice cell, and a table with two
    published rows on one dense lattice cell refuses (the pick map must
    stay one-to-one).  The pick map is variant-invariant: one map serves
    all three variants."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        context = r43._WORKER_CONTEXT
        total = 0
        for table in tables:
            species = table.species
            n_x = int(context["n_x_by_species"][species])
            n_p = int(context["n_p_by_species"][species])
            expected = (
                np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
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
        assert int(context["dense_total"]) > int(r43._flat_total())
    finally:
        r43._set_worker_context(None)
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
    r43._set_worker_context(None)
    with pytest.raises(SystemExit):
        r43._worker_context([dup_table], fams=None, smoke=True)
    r43._set_worker_context(None)


# ---------------------------------------------------------------------------
# The synthetic fixtures (the hermetic helpers)
# ---------------------------------------------------------------------------
def _synthetic_tables():
    """A small hermetic six-species table set on a 4x3 grid (the R42
    test's synthetic-grid pattern)."""
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
        tables.append(BinnedFitTable.from_arrays(
            species=species, pdg_id=pdg, mass_gev=mass,
            normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
            path="test://synthetic",
            xf=np.tile(xf, n_p), pt=np.repeat(pt, n_x),
            value=np.ones(n_x * n_p), error=np.ones(n_x * n_p),
            window={"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
        ))
    return tables


def test_deviance_pooled_difference_se_floor():
    """The pooled SE floor: fewer than two blocks carry no between-
    replica estimate and the floor stands; a non-finite spread falls
    back to the floor."""
    assert r43.pooled_difference_se([]) == r43.SE_FLOOR
    assert r43.pooled_difference_se([2.0]) == r43.SE_FLOOR
    assert r43.pooled_difference_se(
        [float("inf"), 1.0]) == r43.SE_FLOOR
