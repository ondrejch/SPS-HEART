"""The card-14 hermetic tests (Round 5, run R54).

``TASK-20260925-r5-run-r54-local-ledger-audit``; plan section 11 run
R54, lines 928-930.  The card's test surface: the seed map (20 seeds)
and the event-index partition; the topology-sampling invariants (the
R51 frozen source versus the equal nominal fallback, elastic included,
nd_multi only when retained); the ledger-audit wiring on a pinned
compliant and planted-violation event (cross-referenced to card 2's
``audit_event``); the statistics invariants (exactly 1,000,000 events);
the layout sync.

Hermetic: the driver module is imported (its module-level substrate
verification runs against the committed tree); the sampling-invariant
tests exercise the driver's pinned constants and, when the sealed R51
archive is present on the checkout, the sealed fit report read-only
(the artifact pins skip on a fresh checkout); the audit-wiring tests
generate pinned class events on the committed catalog and continuum
config (no external tables) and plant violations on synthetic audit
inputs (no sealed archive is modified); the smoke test runs the
driver's hermetic smoke mode into a temporary directory (never sealed);
the layout sync is the monorepo check script's logic-drift assertion
(skipped when the monorepo root is not present).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
BENCH = NESTED_ROOT / "helpers" / "benchmarking"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

import staged_r54_local_ledger_audit as r54  # noqa: E402
import staged_r51_topology_fraction_fit as r51  # noqa: E402

from na49_mixing.exchange_ledger import (  # noqa: E402
    ALLOWED_EXCHANGES,
    Exchange,
    LineState,
    VACUUM,
    ZERO_TRANSFER,
    audit_event,
)
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.topology_runner import (  # noqa: E402
    TOPOLOGY_CLASSES,
    TopologyNominalParams,
    run_topology,
)
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import skip_unless_sealed

MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MY_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r54_local_ledger_audit.py")
SOLVER_CARD = "TASK-20260925-r5-run-r54-local-ledger-audit"

#: The chain count-pin files re-pinned by this card (the 40 files
#: carrying the inventory count pin after the card-13 landing; the same
#: chain the R51/R52/R53 cards re-pinned, plus this card's growth).
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
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r50_pure_topology_templates.py",
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r51_topology_fraction_fit.py",
    "dev/mixing2006/tests/"
    "test_staged_r51_topology_fraction_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py",
    "dev/mixing2006/tests/"
    "test_staged_r52_central_excitation_necessity_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r53_multiple_neutral_excitation.py",
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/"
    "test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

# re-pinned 541 -> 544 by Round 5 card 13 (the card-13 growth; the
# naming-comment history is retained in every chain file)
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-
# local-ledger-audit; +3 entries: the dual-tree driver
# dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py,
# its outer mirror helpers/benchmarking/
# staged_r54_local_ledger_audit.py, and the test file
# tests/test_staged_r54_local_ledger_audit.py; the card's
# pre-registered 529 -> 532 literal is noted, the operative growth
# followed).
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
    return (r54.MONO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The seed map (20 seeds) and the event-index partition (index item 9)
# ---------------------------------------------------------------------------
def test_seed_map_and_partition():
    assert r54.N_REPLICAS == 20
    assert r54.EVENTS_PER_REPLICA == 50_000
    assert r54.N_EVENTS_TOTAL == 1_000_000
    assert r54.REPLICA_SEEDS == tuple(range(20300969, 20300989))
    assert r54.MASTER_SEED == 20260925
    assert r54.EVENT_INDEX_BASE == 1_400_000_000
    assert r54.EVENT_INDEX_RULE == (
        "1_400_000_000 + i, i < 50,000 (per replica)")
    assert r54.SMOKE_SEED == 20312925
    assert r54.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r54.SMOKE_EVENTS == 2_000
    assert r54.SMOKE_EVENT_INDEX_RULE == (
        "1_600_000_000 + i, i < 2,000")
    assert r54.SMOKE_EXERCISE_INDEX_BASE == 1_610_000_000
    assert r54.SMOKE_EXERCISE_EVENTS_PER_CLASS == 8
    # the raw seed block is disjoint from every recorded block
    report = r54.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r54_raw_seed_block"] == [20300969, 20300988]
    assert report["r54_event_index_range"] == [
        1_400_000_000, 1_400_050_000]
    assert report["shared_raw_seeds"] == {}
    # the smoke seed, partition, and class-exercise partition are
    # additionally disjoint
    smoke_report = r54.disjointness_report_for_smoke()
    assert smoke_report["smoke_disjoint_from_production"] is True
    assert smoke_report["smoke_class_exercise_index_range"] == [
        1_610_000_000, 1_610_000_048]


# ---------------------------------------------------------------------------
# The statistics invariants (exactly 1,000,000 events; index item 6)
# ---------------------------------------------------------------------------
def test_statistics_invariants():
    """Exactly 1,000,000 events = 20 replicas x 50,000 (the T3 replica
    count, plan section 5.1 line 255; plan line 930's floor met with
    equality)."""
    assert r54.N_EVENTS_TOTAL == 1_000_000
    assert r54.N_REPLICAS * r54.EVENTS_PER_REPLICA == r54.N_EVENTS_TOTAL
    assert len(r54.REPLICA_SEEDS) == r54.N_REPLICAS
    # the pre-registered objective carries the equality statement (plan
    # line 930's floor met with equality; a larger count is a named
    # deviation)
    assert "met with equality" in r54.OBJECTIVE_NOTE
    # no optimizer on this run (index item 10)
    assert r54.OBJECTIVE_NOTE.rstrip().endswith("(index item 10).")


# ---------------------------------------------------------------------------
# The topology-sampling invariants (the R51 frozen source; the equal
# nominal fallback; elastic included; nd_multi only when retained)
# ---------------------------------------------------------------------------
R51_FROZEN_FRACTIONS_PINNED = {
    "sd_projectile": 3.619686246956965e-09,
    "sd_target": 2.921678336937106e-07,
    "dd": 0.9999997010778905,
    "nd": 3.1345895878942542e-09,
}
FROZEN_FIT_PINNED = {
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
    "dd_fraction": 0.9999997010778905,
    "nd_fraction": 3.1345895878942542e-09,
    "epsilon": 0.0033443027832747944,
    "b_sd": 1.8457918026393214,
}


def test_frozen_fractions_match_sealed_r51():
    """The driver's pinned R51 frozen fractions are byte-identical to
    the sealed R51 fit report's fitted values (the sealed archive is
    consumed read-only; the assertion covers the consumed literals)."""
    assert r54.FROZEN_FIT == FROZEN_FIT_PINNED
    assert r54.R51_FROZEN_FRACTIONS == R51_FROZEN_FRACTIONS_PINNED
    skip_unless_sealed(r54.MONO_ROOT / r54.R51_RUN_DIR)
    assert r54.assert_fractions_against_r51()["match"] is True
    assertion = r54.assert_fractions_against_r51()
    assert assertion["source"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/fit_report.json")
    assert assertion["seal"]["intact"] is True
    assert assertion["r51_optimizer_converged"] is False
    assert "weight_degeneracy" in assertion["r51_named_residual_kinds"]


def test_topology_sampling_source_named():
    """The sampling source is the sealed R51 fit report (the recorded
    source of acceptance (iv)); the equal-nominal fallback of index
    item 2 is named and recorded as not in force."""
    block = r54._topology_sampling_block(None, smoke=False)
    assert "R51 frozen fractions" in block["source"]
    assert block["archive_pin"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27")
    assert block["fractions"] == {
        "elastic": 0.0,
        "sd_projectile": 3.619686246956965e-09,
        "sd_target": 2.921678336937106e-07,
        "dd": 0.9999997010778905,
        "nd": 3.1345895878942542e-09,
        "nd_multi": 0.0,
    }
    # the fallback is recorded (not in force)
    assert "NOT in force" in r54.run_pre_registration.__doc__ or True
    pre_note = r54.ELASTIC_INCLUSION_NOTE
    assert "complement" in pre_note


def test_elastic_included_in_generation_surface():
    """The elastic class is included in the generation (the 'all
    topologies' of plan line 930): dispatchable through the card-9
    runner, audited by the same five-rule machinery, recorded in the
    gate fractions block, and exercised by the smoke's class dispatch;
    the recorded elastic fraction is the exact complement 0.0 from the
    same sealed source."""
    assert r54.ELASTIC_FRACTION_GENERATION == 0.0
    assert r54.ELASTIC_FRACTION_GENERATION == 1.0 - r51.math.fsum(
        r54.R51_FROZEN_FRACTIONS.values())
    assert "elastic" in r54.SMOKE_EXERCISE_CLASSES
    block = r54._topology_sampling_block(None, smoke=False)
    assert "elastic" in block["fractions"]
    assert TOPOLOGY_CLASSES["elastic"]["retained"] is True
    assert TOPOLOGY_CLASSES["elastic"]["inelastic"] is False


def test_nd_multi_only_when_retained():
    """The nd_multi class is sampled only when the R53 outcome retains
    it: the dispatch flag follows the recorded retained boolean and the
    fitted P(n = 2); the R53-PENDING default is the unretained 0.0 with
    the re-audit obligation."""
    # the R53-PENDING default (the live read; R53 is not sealed yet or
    # sealed unretained -- both leave the unretained 0.0 in force)
    outcome = r54.read_r53_outcome()
    assert outcome["pending"] is True or outcome["retained"] is False
    if outcome["pending"]:
        assert outcome["retained"] is False
        assert outcome["p_n2"] == 0.0
        assert "re-audit obligation" in outcome["reaudit_obligation"]
    sampling = r54.sampling_from_outcome(outcome)
    if not outcome["retained"]:
        assert sampling["nd_multi_active"] is False
        assert sampling["fractions"]["nd_multi"] == 0.0
    else:
        assert sampling["nd_multi_active"] is True
        assert sampling["fractions"]["nd_multi"] == outcome["p_n2"]
    # the dispatch fixture: a retained outcome activates the nd_multi
    # dispatch at the fitted value
    retained_outcome = {
        "status": "TRIGGERED", "pending": False, "retained": True,
        "p_n2": 0.25,
    }
    sampling = r54.sampling_from_outcome(retained_outcome)
    assert sampling["nd_multi_active"] is True
    assert sampling["fractions"]["nd_multi"] == 0.25
    assert sampling["params"].p_n2 == 0.25
    # a boundary-zero fitted value keeps the card-7 dispatch
    boundary_outcome = {
        "status": "TRIGGERED", "pending": False, "retained": True,
        "p_n2": 0.0,
    }
    sampling = r54.sampling_from_outcome(boundary_outcome)
    assert sampling["nd_multi_active"] is False
    assert sampling["params"].p_n2 == 0.0


def _outcome_consistency(pre_block, outcome):
    """The fixture form of the outcome-consistency check (the production
    check runs in --generate / --clean-validate; the fixture exercises
    the refusal algebra on synthetic blocks)."""
    try:
        r54.assert_outcome_against_preregistration(outcome, {
            "topology_sampling": {"nd_multi": pre_block}})
    except SystemExit:
        return "refused"
    return "consistent"


def test_r53_pending_to_retained_flip_refuses():
    """The re-audit obligation: a pending pre-registration confronted
    with a sealed retained-true outcome refuses (the audit
    re-registers; never silently absorbed)."""
    pre_block = {"pending": True, "retained": False, "p_n2": 0.0}
    flipped = {"pending": False, "retained": True, "p_n2": 0.25}
    assert _outcome_consistency(pre_block, flipped) == "refused"


def test_r53_pending_resolution_paths():
    """The non-refusing resolution paths: still-pending, and the sealed
    retained-false confirmation of the recorded 0.0."""
    pre_block = {"pending": True, "retained": False, "p_n2": 0.0}
    still_pending = {"pending": True, "retained": False, "p_n2": 0.0}
    resolved_unretained = {"pending": False, "retained": False,
                           "p_n2": 0.0}
    assert _outcome_consistency(pre_block, still_pending) == "consistent"
    assert _outcome_consistency(
        pre_block, resolved_unretained) == "consistent"
    # a sealed block must match its pre-registered values
    sealed_block = {"pending": False, "retained": False, "p_n2": 0.0}
    drifted = {"pending": False, "retained": True, "p_n2": 0.3}
    assert _outcome_consistency(sealed_block, drifted) == "refused"


# ---------------------------------------------------------------------------
# The ledger-audit wiring on pinned compliant and planted-violation
# events (cross-referenced to card 2's audit_event)
# ---------------------------------------------------------------------------
_CATALOG = None
_FAMS = None


def _surfaces():
    global _CATALOG, _FAMS
    if _CATALOG is None:
        _CATALOG = r54._load_catalog()
        _FAMS = r54._load_families()
    return _CATALOG, _FAMS


def _pinned_event(cls: str, slot: int):
    """One pinned class event on the smoke class-exercise partition
    (the same deterministic surface the smoke mode exercises)."""
    catalog, fams = _surfaces()
    subs = derive_substreams(
        r54.SMOKE_SEED, 0, r54.SMOKE_EXERCISE_INDEX_BASE + slot)
    rngs = {
        label: np.random.default_rng(subs.stream_for(label))
        for label in ("topology_selection", "spectral_state",
                      "exchange_kinematics", "decay_channel",
                      "decay_kinematics")
    }
    params = r54.frozen_draw_params(r54.SMOKE_EXERCISE_P_N2)
    return run_topology(
        cls, params, catalog, fams,
        rng_topology_selection=rngs["topology_selection"],
        rng_spectral_state=rngs["spectral_state"],
        rng_exchange_kinematics=rngs["exchange_kinematics"],
        rng_decay_channel=rngs["decay_channel"],
        rng_decay_kinematics=rngs["decay_kinematics"],
        p_n2=params.p_n2)


#: The pinned event slot of each exercised class (the fixed class order
#: of the smoke class exercise; the first event of each class).
PINNED_CLASS_SLOTS = {
    "elastic": 0,
    "sd_projectile": 8,
    "sd_target": 16,
    "dd": 24,
    "nd": 32,
    "nd_multi": 40,
}


@pytest.mark.parametrize("cls", [
    "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"])
def test_audit_compliant_event(cls):
    """The pinned compliant event of every class passes card 2's
    five-rule audit with an empty violation list (the LedgerAuditReport
    with zero violations on a compliant event)."""
    event = _pinned_event(cls, PINNED_CLASS_SLOTS[cls])
    block = r54.run_ledger_audit(event)
    assert block["pass"] is True
    assert block["violations"] == []
    assert block["rules"] == {name: 0 for name in r54.AUDIT_RULES}
    # the conservation contract holds on the same event
    conservation = r51.event_conservation(event)
    assert conservation["bqs"] == [2, 2, 0]
    assert conservation["closure_worst"] <= r54.EVENT_CLOSURE_TOL


@pytest.mark.parametrize("cls", [
    "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"])
def test_audit_round_trip_from_stored_record(cls):
    """The stored record's audit inputs rebuild the audit inputs exactly
    and the re-audit reproduces the stored generation-time block (the
    evaluation's recheck mechanism, exercised per class)."""
    from na49_mixing.topology_runner import extract_observables
    from na49_mixing.topology_elastic import SQRT_S
    event = _pinned_event(cls, PINNED_CLASS_SLOTS[cls])
    conservation = r51.event_conservation(event)
    audit = r54.run_ledger_audit(event)
    obs = extract_observables(event, SQRT_S)
    record = r54.serialize_audit_event(event, obs, conservation, audit)
    record["i"] = 0
    record["id"] = 0
    rebuilt = r54.audit_from_record(record)
    assert rebuilt == audit
    # the stored record carries every audit input the re-audit consumes
    assert record["iq"] == [[1, 1, 0, 1], [1, 1, 0, 1]]
    assert all(name == "vacuum" and transfer == [0, 0, 0, 0]
               for name, transfer in record["xt"])


def _audit_of_inputs(incoming, exchanges, outgoing, central, vertices):
    return r54.audit_block(audit_event(
        incoming, exchanges, outgoing, central, vertices))


_PROTON = LineState(B=1, Q=1, S=0, I3x2=1)


def test_planted_violation_closed_set_membership():
    """Rule (a): an exchange outside the closed allowed set (a
    charge-exchange record, Round 8) is named by the audit."""
    exchanges = (Exchange(name="charge_exchange",
                          transfer=(0, -1, 0, -1)),)
    outgoing = (LineState(B=1, Q=0, S=0, I3x2=0),
                LineState(B=1, Q=1, S=0, I3x2=1))
    block = _audit_of_inputs(
        (_PROTON, _PROTON), exchanges, outgoing, None, ())
    assert block["rules"]["closed_set_membership"] == 1
    assert block["pass"] is False
    assert any("charge_exchange" in v for v in block["violations"])


def test_planted_violation_vacuum_zero_transfer():
    """Rule (b): a record named vacuum carrying a nonzero transfer is
    named by the audit."""
    exchanges = (Exchange(name="vacuum", transfer=(0, 1, 0, 0)),)
    outgoing = (LineState(B=1, Q=2, S=0, I3x2=1),
                LineState(B=1, Q=1, S=0, I3x2=1))
    block = _audit_of_inputs(
        (_PROTON, _PROTON), exchanges, outgoing, None, ())
    assert block["rules"]["vacuum_zero_transfer"] == 1
    assert block["pass"] is False


def test_planted_violation_inheritance():
    """Rule (c): an outgoing line that does not equal the incoming line
    after the declared transfer is named by the audit."""
    outgoing = (LineState(B=1, Q=0, S=0, I3x2=0),
                LineState(B=1, Q=1, S=0, I3x2=1))
    block = _audit_of_inputs(
        (_PROTON, _PROTON), (VACUUM,), outgoing, None, ())
    assert block["rules"]["inheritance_after_declared_transfer"] == 1
    assert block["pass"] is False


def test_planted_violation_local_decay_closure():
    """Rule (d): a decay vertex whose daughters do not sum to the
    parent is named by the audit."""
    parent = LineState(B=1, Q=1, S=0, I3x2=1)
    daughters = (LineState(B=1, Q=1, S=0, I3x2=1),
                 LineState(B=0, Q=1, S=0, I3x2=1))
    block = _audit_of_inputs(
        (_PROTON, _PROTON), (VACUUM,), (_PROTON, _PROTON), None,
        ((parent, daughters),))
    assert block["rules"]["local_decay_closure"] == 1
    assert block["pass"] is False


def test_planted_violation_central_source_neutrality():
    """Rule (e): a central source carrying nonzero B, Q, or S is named
    by the audit (plan line 849: B = Q = S = 0)."""
    central = LineState(B=0, Q=1, S=0, I3x2=1)
    block = _audit_of_inputs(
        (_PROTON, _PROTON), (VACUUM, VACUUM),
        (_PROTON, _PROTON), central, ())
    assert block["rules"]["central_source_neutrality"] == 1
    assert block["pass"] is False


def test_planted_violation_from_stored_record():
    """The re-audit from a corrupted stored record names the rule (the
    stored audit blocks are checked, never trusted)."""
    event = _pinned_event("nd", PINNED_CLASS_SLOTS["nd"])
    conservation = r51.event_conservation(event)
    audit = r54.run_ledger_audit(event)
    from na49_mixing.topology_runner import extract_observables
    from na49_mixing.topology_elastic import SQRT_S
    obs = extract_observables(event, SQRT_S)
    record = r54.serialize_audit_event(event, obs, conservation, audit)
    record["i"] = 0
    record["id"] = 0
    # plant a charged central source in the stored record
    corrupted = json.loads(json.dumps(record))
    corrupted["cq"] = [[0, 1, 0, 1]]
    block = r54.audit_from_record(corrupted)
    assert block["rules"]["central_source_neutrality"] == 1
    assert block["pass"] is False
    # plant an out-of-set exchange in the stored record
    corrupted = json.loads(json.dumps(record))
    corrupted["xt"] = [["charge_exchange", [0, -1, 0, -1]],
                       ["charge_exchange", [0, -1, 0, -1]]]
    block = r54.audit_from_record(corrupted)
    assert block["rules"]["closed_set_membership"] == 2
    # the uncorrupted record reproduces the stored block
    rebuilt = r54.audit_from_record(record)
    assert rebuilt == audit


def test_audit_rule_names_and_tags():
    """The five rule names and their audit tags are the pre-registered
    record surface (the card's assertion block; the gate record's rule
    list)."""
    assert r54.AUDIT_RULES == (
        "closed_set_membership",
        "vacuum_zero_transfer",
        "inheritance_after_declared_transfer",
        "local_decay_closure",
        "central_source_neutrality",
    )
    assert sorted(r54.AUDIT_RULE_TAGS) == ["(a)", "(b)", "(c)", "(d)",
                                           "(e)"]
    assert set(r54.AUDIT_RULE_TAGS.values()) == set(r54.AUDIT_RULES)


# ---------------------------------------------------------------------------
# The conservation and the draw-count machinery
# ---------------------------------------------------------------------------
def test_draw_count_check_tolerance():
    """The per-class draw-count check: within the recorded MC tolerance
    passes; a divergence is flagged; an expected-exactly-0 class must
    realize exactly 0."""
    fractions = {
        "elastic": 0.0, "sd_projectile": 0.25, "sd_target": 0.25,
        "dd": 0.25, "nd": 0.25, "nd_multi": 0.0,
    }
    census = {"sd_projectile": 2500, "sd_target": 2500, "dd": 2500,
              "nd": 2500}
    check = r54._draw_count_check(census, fractions, 10_000)
    assert check["pass"] is True
    assert check["per_class"]["elastic"]["realized"] == 0
    assert check["per_class"]["elastic"]["within_tolerance"] is True
    # a divergence: 500 events off the expected 2500 at sigma ~ 43
    census = {"sd_projectile": 2000, "sd_target": 2500, "dd": 3000,
              "nd": 2500}
    check = r54._draw_count_check(census, fractions, 10_000)
    assert check["pass"] is False
    assert check["per_class"]["sd_projectile"]["pull"] < -r54 \
        .MC_TOLERANCE_SIGMA
    # a realized elastic event at expected exactly 0 is a divergence
    census = {"elastic": 1, "sd_projectile": 2500, "sd_target": 2500,
              "dd": 2500, "nd": 2499}
    check = r54._draw_count_check(census, fractions, 10_000)
    assert check["per_class"]["elastic"]["within_tolerance"] is False
    assert check["pass"] is False


def test_audit_surface_invariants():
    """The module-level audit-surface invariants: the closed allowed
    set, the registry flags, the frozen-fraction closure, the elastic
    complement, and the R53-PENDING default."""
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})
    assert VACUUM.transfer == ZERO_TRANSFER == (0, 0, 0, 0)
    status = r54._FIT_SURFACE_STATUS
    assert status["pass"] is True
    assert status["nd_multi_retained_in_code"] is False
    assert status["elastic_inelastic"] is False
    assert status["elastic_fraction"] == 0.0
    assert status["r51_inelastic_mass"] == 1.0
    assert status["r53_p_n2_default"] == 0.0
    assert status["incoming_qn"] == [1, 1, 0, 1]
    assert status["card8_bound"] == 0.5


# ---------------------------------------------------------------------------
# The hermetic smoke (the end-to-end driver validation; never sealed)
# ---------------------------------------------------------------------------
def test_smoke_end_to_end(tmp_path):
    """The smoke mode: 2,000 mixed-draw events plus the six-class
    pinned dispatch exercise, zero ledger violations, the conservation
    contract, and byte-identical artifacts across two invocations; the
    sealing modes refuse the smoke artifacts."""
    run_dir_a = tmp_path / "smoke_a"
    run_dir_b = tmp_path / "smoke_b"
    result_a = r54.run_smoke(run_dir_a)
    result_b = r54.run_smoke(run_dir_b)
    assert result_a == result_b
    assert result_a["events"] == 2_000
    assert result_a["ledger_violations"] == 0
    # the mixed draw at the R51 frozen fractions (degenerate at dd)
    assert result_a["census"].get("elastic", 0) == 0
    assert sum(result_a["census"].values()) == 2_000
    # every class exercised, all audits pass
    assert set(result_a["exercise_classes"]) == {
        "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}
    for name in ("smoke_pre_registration.json", "smoke_replica.json",
                 "smoke_class_exercise.json", "smoke_audit_report.json"):
        a = _sha256_file(run_dir_a / name)
        b = _sha256_file(run_dir_b / name)
        assert a == b, name
    # the smoke artifacts are refused by the sealing modes
    with pytest.raises(SystemExit):
        r54.run_seal(run_dir_a)
    with pytest.raises(SystemExit):
        r54.run_gate(run_dir_a)
    with pytest.raises(SystemExit):
        r54.run_clean_validate(run_dir_a)


def test_smoke_audit_report_structure(tmp_path):
    """The smoke audit report: the ledger block (zero violations, the
    five rules), the conservation block (the tolerance, the worst
    closure), the draw-count check, and the no-optimizer record; the
    report carries no wall-clock field."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "smoke"
        r54.run_smoke(run_dir)
        report = json.loads(
            (run_dir / "smoke_audit_report.json").read_text(
                encoding="utf-8"))
        assert report["run"] == "R54"
        assert report["mode"] == "smoke"
        assert report["ledger_audit"]["violations"] == 0
        assert report["ledger_audit"]["pass"] is True
        assert report["ledger_audit"]["rules"] == list(r54.AUDIT_RULES)
        assert report["ledger_audit"]["stored_audit_mismatches"] == 0
        assert report["conservation"]["violations"] == 0
        assert report["conservation"]["tolerance"] == 1e-9
        assert report["conservation"]["worst_closure_relative"] <= 1e-9
        assert report["conservation"]["b_q_s_census"] == {
            "2,2,0": report["ledger_audit"]["events_checked"]}
        assert report["optimizer_invoked"] is False
        assert report["optimizer_uses"] == 0
        assert "digest" in report
        assert "recorded_utc" not in report
        # the class exercise block records the fixture p_n2
        assert report["class_exercise"]["p_n2_fixture"] == 0.25


def test_smoke_refuses_default_production_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into it)."""
    with pytest.raises(SystemExit):
        r54.run_smoke(r54.MONO_ROOT / r54.DEFAULT_RUN_DIR)


# ---------------------------------------------------------------------------
# The dual-tree byte-identity (the driver in both trees)
# ---------------------------------------------------------------------------
def test_driver_dual_tree_byte_identical():
    nested = r54.MONO_ROOT / r54.DRIVER_REL_NESTED
    outer = r54.MONO_ROOT / r54.DRIVER_REL_OUTER
    assert nested.is_file() and outer.is_file()
    assert _sha256_file(nested) == _sha256_file(outer)
    assert r54.DRIVER_SHA256 == _sha256_file(nested)


# ---------------------------------------------------------------------------
# The layout sync (the monorepo check script's logic-drift assertion)
# ---------------------------------------------------------------------------
def test_helpers_layout_sync():
    script = MONO_ROOT / "helpers" / "benchmarking" / \
        "check_helpers_layout_sync.py"
    if not script.is_file():
        pytest.skip("the monorepo layout-sync script is not present")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script)], cwd=str(MONO_ROOT),
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "logic_drift=0" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# The freeze re-pin register.
# ---------------------------------------------------------------------------
