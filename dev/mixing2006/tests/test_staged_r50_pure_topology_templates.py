"""Hermetic tests for the R50 run driver (the pure topology templates).

Task ``TASK-20260925-r5-run-r50-pure-topology-templates`` (plan section
11 run R50, lines 898-907): the pre-registered slot seed map (20 seeds,
class order elastic, sd_projectile, sd_target, dd, nd) and the
event-index partition; the nominal-parameter assertion against
``TopologyNominalParams`` (card 9); the class-set invariants (five
classes, nd_multi absent); the conservation algebra on a pinned
fixture; the seed-block disjointness; the deterministic per-event
record bytes; the smoke-mode refusal conventions; the sealed run-dir
artifact pins (read-only, skipped on a fresh checkout); the layout
sync; and the freeze re-pin register.  No Monte Carlo generation at run
scale in the deterministic legs (the sealed-dir pins read the sealed
run dir when the artifact store is present).
"""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tests._freeze_inventory import inventory_count as _inventory_count


#: The sealed driver's per-event mass-shell check divides the deviation by
#: M^2, and the kernel's radiative channels (2026-10-02,
#: TASK-20261002-kernel-feedover-threshold-radiative) put massless photons
#: in the final state, so the driver's hermetic generation cannot pass on
#: the current kernel.  The sealed archive tests stay active; the current
#: pipeline is the R71 register's.
_SEALED_KERNEL_SKIP = pytest.mark.skip(
    reason="the sealed driver predates the radiative channels (massless photons) of the current kernel")


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r50_pure_topology_templates.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r50_pure_topology_templates", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R50 driver was not found next to the test file")


r50 = _load_driver()


# ---------------------------------------------------------------------------
# The run identity constants (the card's pre-registered identity)
# ---------------------------------------------------------------------------
def test_run_identity_constants():
    """The run identity: R50, the card, the execution-date run name (the
    card moves the date with the actual execution; the pre-registered
    2026-09-25 identity re-pins to the execution date), the statistics,
    and the smoke block."""
    assert r50.RUN_ID == "R50"
    assert r50.CARD == "TASK-20260925-r5-run-r50-pure-topology-templates"
    assert r50.RUN_NAME.startswith("r50_pure_topology_templates_")
    assert r50.DEFAULT_RUN_DIR == f"dev/00runs/{r50.RUN_NAME}"
    assert r50.MASTER_SEED == 20260925
    assert r50.REPLICAS_PER_CLASS == 4
    assert r50.EVENTS_PER_SLOT == 25_000
    assert r50.N_SLOTS == 20
    assert r50.N_EVENTS_PER_CLASS == 100_000
    assert r50.N_EVENTS_TOTAL == 500_000
    assert r50.SMOKE_SEED == 20260925 + 40_000 + 12_000 == 20312925
    assert r50.SMOKE_EVENTS == 2_000
    assert r50.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r50.EVENT_INDEX_BASE == 1_000_000_000
    assert r50.PROBE_SLOT == 16
    assert r50.CLASS_ORDER == (
        "elastic", "sd_projectile", "sd_target", "dd", "nd")
    assert r50.INELASTIC_CLASS_ORDER == (
        "sd_projectile", "sd_target", "dd", "nd")


def test_slot_seed_map_and_event_index_partition():
    """The pre-registered slot map (index item 9): raw seeds
    20300925-20300944 (k = 0..19), one seed per (class, replica) pair in
    class order elastic, SD_p, SD_t, DD, ND with slot
    k = class_index * 4 + replica_index; the event-index partition
    1_000_000_000 + i with i < 25,000 per slot."""
    assert r50.SLOT_SEEDS == tuple(20300925 + k for k in range(20))
    assert r50.SLOT_SEEDS == tuple(20260925 + 40_000 + k for k in range(20))
    for slot in range(20):
        assert r50.slot_class(slot) == r50.CLASS_ORDER[slot // 4]
        assert r50.slot_replica(slot) == slot % 4
        assert r50.slot_seed(slot) == 20300925 + slot
        assert r50.slot_name(slot) == (
            f"slot_{r50.slot_class(slot)}_{slot % 4}.json")
    assert r50.event_id(0) == 1_000_000_000
    assert r50.event_id(24_999) == 1_000_024_999
    blocks = {
        label: [20300925 + 4 * index, 20300925 + 4 * index + 3]
        for index, label in enumerate(r50.CLASS_ORDER)
    }
    for label, (lo, hi) in blocks.items():
        seeds = [r50.slot_seed(k) for k in range(20)
                 if r50.slot_class(k) == label]
        assert seeds == list(range(lo, hi + 1))
    # the partition disjointness from the Round 4 partitions and the
    # reserved round-5 partitions
    report = r50.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["shared_raw_seeds"] == {}
    r50_lo, r50_hi = report["r50_event_index_range"]
    assert r50_lo == 1_000_000_000 and r50_hi == 1_000_025_000
    for name, (lo, hi) in r50.RESERVED_ROUND5_PARTITIONS.items():
        assert r50_hi <= lo or hi <= r50_lo, name
    for name, block in r50.PREVIOUS_USED_BLOCKS.items():
        p_range = block.get("event_index_range")
        if p_range is not None:
            assert r50_hi <= p_range[0] or p_range[1] <= r50_lo, name
    # the smoke block is disjoint from the production block by seed and
    # by partition
    assert r50.SMOKE_SEED not in set(r50.SLOT_SEEDS)
    assert r50.SMOKE_EVENT_INDEX_BASE >= r50_hi


def test_nominal_parameter_assertion():
    """The generation parameters equal TopologyNominalParams exactly
    (acceptance (iii)): epsilon = 0, B_SD byte-identical to the
    M8-frozen effective_transfer_slope, the inelastic fractions 0.25
    each, the remnant weights 0.5/0.5/0.0, P(n = 2) = 0.0."""
    params = r50.NOMINAL_PARAMS
    assert params.epsilon == 0.0
    assert params.b_sd == 8.504140762083612
    # byte-identical to the M8-frozen effective_transfer_slope: the
    # repr form is the byte-level identity
    assert repr(params.b_sd) == repr(8.504140762083612)
    assert params.b_sd.hex() == float(8.504140762083612).hex()
    assert dict(params.class_fractions) == {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}
    assert dict(params.remnant_weights) == {
        "intact_proton": 0.5,
        "low_mass_baryonic_resonance": 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }
    assert params.p_n2 == 0.0
    status = r50.assert_nominal_parameters()
    assert status["pass"] is True
    assert status["elastic_fraction"] == 0.0
    m8 = r50.m8_frozen_set_status()
    assert m8["verified"] is True
    assert m8["hash"] == (
        "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
    assert m8["effective_transfer_slope_gev2"] == params.b_sd
    assert m8["blend_ramp_start_gev"] == 2.680194110727528


def test_class_set_invariants():
    """The class-set invariants (acceptance (i) and (iii)): the six
    registry entries with nd_multi retained: False, the elastic class
    outside the inelastic set, and the R50 class order of five classes
    with nd_multi absent."""
    assert set(r50.TOPOLOGY_CLASSES) == {
        "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}
    assert r50.TOPOLOGY_CLASSES["nd_multi"]["retained"] is False
    assert all(r50.TOPOLOGY_CLASSES[label]["retained"] is True
               for label in r50.TOPOLOGY_CLASSES if label != "nd_multi")
    assert r50.TOPOLOGY_CLASSES["elastic"]["inelastic"] is False
    assert set(r50.INELASTIC_SET) == {
        "sd_projectile", "sd_target", "dd", "nd"}
    assert len(r50.CLASS_ORDER) == 5
    assert "nd_multi" not in r50.CLASS_ORDER
    assert "elastic" not in r50.INELASTIC_CLASS_ORDER


# ---------------------------------------------------------------------------
# The conservation algebra on a pinned fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pinned_fixture():
    """A pinned 3-event-per-class fixture generated through the driver's
    own path (the fixed-class dispatch at the pinned slot seeds)."""
    catalog = r50._load_catalog()
    fams = r50._load_families()
    fixture = {}
    for label in r50.CLASS_ORDER:
        class_index = r50.CLASS_ORDER.index(label)
        slot = class_index * 4
        records, census, conservation, hash_set = r50.generate_slot_events(
            label, r50.slot_seed(slot), 3, r50.EVENT_INDEX_BASE,
            catalog, fams)
        fixture[label] = (records, census, conservation, hash_set)
    return fixture


@_SEALED_KERNEL_SKIP
def test_conservation_algebra_pinned_fixture(pinned_fixture):
    """The conservation algebra on the pinned fixture: exact
    (B, Q, S) = (2, 2, 0) per event, four-momentum closure within 1e-9
    relative, worst mass shell within 1e-9; the aggregate census
    matches; a planted quantum-number violation and a planted
    four-momentum violation are each detected by name."""
    for label, (records, census, conservation, hash_set) in \
            pinned_fixture.items():
        assert census == {label: 3}
        assert conservation["violations"] == 0
        assert conservation["b_q_s_census"] == {"2,2,0": 3}
        assert conservation["worst_closure_relative"] <= 1e-9
        assert conservation["worst_mass_shell_relative"] <= 1e-9
        for record in records:
            assert record["bqs"] == [2, 2, 0]
            assert max(record["clo"]) <= 1e-9
            # the per-event hash is the sha256 of the canonical record
            # without the h key
            payload = {k: v for k, v in record.items() if k != "h"}
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8")
            assert hashlib.sha256(canonical).hexdigest() == record["h"]
        # the event-hash set digest is the sha256 over the joined
        # per-event hashes
        joined = "\n".join(r["h"] for r in records).encode("utf-8")
        assert hashlib.sha256(joined).hexdigest() == hash_set
    # a planted violation: a mutated ledger quantum number breaks the
    # exact (2, 2, 0) bookkeeping, a mutated four-momentum breaks the
    # closure
    from na49_mixing.topology_elastic import ElasticEvent, LineState

    catalog = r50._load_catalog()
    fams = r50._load_families()
    subs = r50.derive_substreams(r50.slot_seed(0), 0, r50.EVENT_INDEX_BASE)
    rng = np.random.default_rng(subs.stream_for("exchange_kinematics"))
    event = r50.run_topology("elastic", r50.NOMINAL_PARAMS, catalog, fams,
                             rng_exchange_kinematics=rng)
    assert isinstance(event, ElasticEvent)
    broken_qn = ElasticEvent(
        class_label=event.class_label,
        incoming=event.incoming,
        outgoing=(
            LineState(four_momentum=event.outgoing[0].four_momentum,
                      qn=(1, 1, 0, 1)),
            LineState(four_momentum=event.outgoing[1].four_momentum,
                      qn=(1, 1, 1, 1)),  # planted S = 1
        ),
        exchange=event.exchange,
        t=event.t,
    )
    with pytest.raises(r50.ConservationViolation):
        r50.event_conservation(broken_qn)
    broken_p = ElasticEvent(
        class_label=event.class_label,
        incoming=event.incoming,
        outgoing=(
            LineState(four_momentum=event.outgoing[0].four_momentum,
                      qn=(1, 1, 0, 1)),
            LineState(four_momentum=(
                event.outgoing[1].four_momentum[0] + 1e-6,
                event.outgoing[1].four_momentum[1],
                event.outgoing[1].four_momentum[2],
                event.outgoing[1].four_momentum[3]), qn=(1, 1, 0, 1)),
        ),
        exchange=event.exchange,
        t=event.t,
    )
    with pytest.raises(r50.ConservationViolation):
        r50.event_conservation(broken_p)


@_SEALED_KERNEL_SKIP
def test_per_event_record_determinism(pinned_fixture):
    """The same (seed, event index) regenerates the identical record
    bytes (the determinism contract; the per-event record is a pure
    function of the pre-registered seed and the committed code)."""
    catalog = r50._load_catalog()
    fams = r50._load_families()
    label = "nd"
    records_a, _census_a, _cons_a, hash_a = r50.generate_slot_events(
        label, r50.slot_seed(16), 5, r50.EVENT_INDEX_BASE, catalog, fams)
    records_b, _census_b, _cons_b, hash_b = r50.generate_slot_events(
        label, r50.slot_seed(16), 5, r50.EVENT_INDEX_BASE, catalog, fams)
    assert hash_a == hash_b
    assert records_a == records_b


def test_elastic_normalization_check_algebra():
    """The elastic normalization check's analytic pieces: the analytic
    mean of exp(B t) on [t_min, 0] is -1/B at the pinned configuration
    (the underflow domain fact), the closed-form normalization is
    1/B there, and the check on a small pinned sample stays inside its
    recorded tolerances."""
    t_min, _t_max = r50.elastic_t_range(r50.SQRT_S)
    b_slope = r50.ELASTIC_TRANSFER_SLOPE
    assert r50._analytic_t_mean(t_min, b_slope) == -1.0 / b_slope
    quadrature = r50._quadrature_normalization(t_min, b_slope)
    closed = r50.elastic_t_normalization(t_min)
    assert abs(quadrature - closed) <= 1e-12


# ---------------------------------------------------------------------------
# The smoke-mode refusal conventions (the R31/8c convention)
# ---------------------------------------------------------------------------
def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r50.MONO_ROOT / r50.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r50.run_smoke(default_dir)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention): a smoke pre-registration and any smoke_-prefixed
    file are refused; a dir without a pre-registration is refused."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r50.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r50._refuse_smoke_artifacts(run_dir)
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r50.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_slot.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r50._refuse_smoke_artifacts(run_dir2)
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r50._refuse_smoke_artifacts(run_dir3)


# ---------------------------------------------------------------------------
# The sealed run-dir artifact pins (read-only; skipped on a fresh
# checkout without the artifact store)
# ---------------------------------------------------------------------------
RUN_DIR = r50.MONO_ROOT / r50.DEFAULT_RUN_DIR


@pytest.fixture(scope="module")
def sealed_run():
    if not (RUN_DIR / "SHA256SUMS.txt").is_file():
        pytest.skip("the sealed R50 run dir is absent on this checkout")
    # verify the seal first (the sha256sum -c equivalent)
    for line in (RUN_DIR / "SHA256SUMS.txt").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = RUN_DIR / name
        assert payload.is_file(), f"sealed file missing: {name}"
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == digest, \
            f"sealed file drifted: {name}"
    return RUN_DIR


def test_sealed_run_dir_inventory(sealed_run):
    """The sealed inventory: the 8 meta files, the 20 slot files, and
    the seal manifest's own record count."""
    names = sorted(p.name for p in sealed_run.iterdir())
    slot_files = [n for n in names if n.startswith("slot_")
                  and n.endswith(".json")]
    assert len(slot_files) == 20
    for name in ("pre_registration.json", "provenance.json",
                 "template_report.json", "REPORT.md", "gate5_r50.json",
                 "clean_environment_commands.json", "seal_manifest.json",
                 "SHA256SUMS.txt"):
        assert name in names, name
    seal = json.loads((sealed_run / "seal_manifest.json").read_text(
        encoding="utf-8"))
    assert seal["run"] == "R50"
    assert seal["record_count"] == 20
    sums_entries = len(
        [l for l in (sealed_run / "SHA256SUMS.txt")
         .read_text(encoding="utf-8").splitlines() if l.strip()])
    assert seal["sha256sums_entries"] == sums_entries == 26


def test_sealed_pre_registration_predates_the_run_data(sealed_run):
    """The pre-registration predates the run data: every slot record's
    identity block carries the pre-registration digest (the slot was
    generated against exactly this pre-registration) and the
    pre-registration file's mtime precedes every slot file's mtime."""
    pre_path = sealed_run / "pre_registration.json"
    pre_digest = hashlib.sha256(pre_path.read_bytes()).hexdigest()
    pre_mtime = pre_path.stat().st_mtime
    for k in range(20):
        slot_path = sealed_run / r50.slot_name(k)
        record = json.loads(slot_path.read_text(encoding="utf-8"))
        assert record["identity"]["pre_registration_sha256"] == pre_digest
        assert slot_path.stat().st_mtime >= pre_mtime


def test_sealed_census_and_conservation(sealed_run):
    """The sealed run's census and conservation (the card's acceptance
    (i) and (v)): 20 slots x 25,000 events, 100,000 per class, 500,000
    total; zero B/Q/S violations and zero four-momentum closure
    failures at tolerance 1e-09; the worst closure recorded in GeV."""
    report = json.loads((sealed_run / "template_report.json").read_text(
        encoding="utf-8"))
    assert report["run"] == "R50"
    assert report["statistics"]["events_total"] == 500_000
    census = report["class_draw_census"]
    assert len(census) == 5
    for label, row in census.items():
        assert row["events"] == 100_000
        assert row["replicas"] == 4
        if row["inelastic"]:
            assert row["nominal_inelastic_fraction"] == 0.25
            assert row["realized_inelastic_fraction"] == 0.25
    cons = report["conservation"]
    assert cons["events_checked"] == 500_000
    assert cons["violations"] == 0
    assert cons["pass"] is True
    assert cons["tolerance"] == 1e-9
    assert cons["worst_closure_relative"] <= 1e-9
    assert cons["worst_mass_shell_relative"] <= 1e-9
    assert cons["b_q_s_census"] == {"2,2,0": 500_000}
    elastic = report["elastic_normalization_check"]
    assert elastic["verdict"] == "pass"
    assert report["optimizer_invoked"] is False
    assert report["optimizer_uses"] == 0


def test_sealed_gate_record(sealed_run):
    """The sealed Gate 5 evidence record (the card's template): the
    class list, the excluded set, the nominal parameters, and the
    optimizer clause."""
    gate = json.loads((sealed_run / "gate5_r50.json").read_text(
        encoding="utf-8"))
    assert gate["gate"] == "gate 5 (plan section 11, line 938)"
    assert gate["run"] == "R50"
    assert gate["clause"] == "per-class template evidence (training side)"
    assert gate["classes"] == list(r50.CLASS_ORDER)
    assert gate["excluded_from_inelastic"] == ["elastic", "nd_multi"]
    assert gate["nominal_parameters"] == {
        "epsilon": 0.0,
        "b_sd": 8.504140762083612,
        "inelastic_fractions": {
            "sd_projectile": 0.25, "sd_target": 0.25,
            "dd": 0.25, "nd": 0.25},
        "p_n2": 0.0,
    }
    assert gate["events_checked"] == 500_000
    assert gate["optimizer_invoked"] is False
    assert gate["optimizer_uses"] == 0
    assert gate["conservation"]["violations"] == 0
    assert gate["conservation"]["pass"] is True


def test_sealed_slot_identity_chain(sealed_run):
    """The per-slot identity chain: every slot carries the current
    driver digest and the pre-registration digest, the exact seed, and
    the event-hash set digest its stored events reproduce."""
    driver_digest = hashlib.sha256(
        (r50.MONO_ROOT / r50.DRIVER_REL_OUTER).read_bytes()).hexdigest()
    nested_digest = hashlib.sha256(
        (r50.MONO_ROOT / r50.DRIVER_REL_NESTED).read_bytes()).hexdigest()
    assert driver_digest == nested_digest
    for k in range(20):
        record = json.loads(
            (sealed_run / r50.slot_name(k)).read_text(encoding="utf-8"))
        assert record["identity"]["driver_sha256"] == driver_digest
        assert record["seed"] == r50.slot_seed(k)
        assert record["class_label"] == r50.slot_class(k)
        assert record["n_events"] == 25_000
        recomputed = hashlib.sha256(
            "\n".join(e["h"] for e in record["events"]).encode(
                "utf-8")).hexdigest()
        assert recomputed == record["event_hash_set_sha256"]


# ---------------------------------------------------------------------------
# The layout sync and the freeze re-pin register
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r50_pure_topology_templates.py")
DRIVER_REL_OUTER = "helpers/benchmarking/staged_r50_pure_topology_templates.py"
MY_TEST_REL = "dev/mixing2006/tests/test_staged_r50_pure_topology_templates.py"
SOLVER_CARD = "TASK-20260925-r5-run-r50-pure-topology-templates"

#: The chain count pins this card's re-pin bump: the card's
#: pre-registered 517 -> 520 literal is noted; the operative growth
#: followed the card-9 test-author re-pin (529 at this card's start) and
#: grows by exactly 3 entries (the dual-tree driver in both helper
#: trees and this test file).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the dual-tree driver, this test file, and the re-pinned chain count
#: pin files.
CARD_EXCEPTION_FILES = tuple(sorted((
    MANIFEST_REL,
    DRIVER_REL_NESTED,
    DRIVER_REL_OUTER,
    MY_TEST_REL,
    *CHAIN_COUNT_PIN_FILES,
)))

# re-pinned 517 -> 520 by Round 5 card 10 (the card's pre-registered
# literal; the operative growth followed the interim re-pins)
# re-pinned 508 -> 510 by Round 5 card 3 ...
# (the chain naming comments carry the full history; see the manifest's
# freeze_exceptions entries)
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
M8_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
KERNEL_BOUNDARY = 2.680194110727528


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _source_of(rel):
    return (r50.MONO_ROOT / rel).read_text(encoding="utf-8")


def source_of(rel):
    return _source_of(rel)


def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""
    checker = r50.MONO_ROOT / "helpers" / "benchmarking" / \
        "check_helpers_layout_sync.py"
    probe = subprocess.run(
        [sys.executable, str(checker)],
        capture_output=True, text=True, cwd=str(r50.MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
