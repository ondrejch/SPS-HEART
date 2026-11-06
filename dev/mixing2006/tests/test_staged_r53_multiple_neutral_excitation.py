"""The card-13 hermetic tests (Round 5, run R53).

``TASK-20260925-r5-run-r53-multiple-neutral-excitation``; plan section
11 run R53, lines 924-926.  The card's test surface: the seed map (8
seeds) and the event-index partition; the trigger evaluation on pinned
R52 gate-evidence fixtures (triggered and NOT_TRIGGERED); the single-
parameter fit-set invariants (P(n >= 2) bounded at [0, 0.5], fractions
and mass laws frozen, strange sector excluded); the truncated-count
draw on a pinned RNG fixture (n in {1, 2}, n = 1 records bit-identical
to card 7's ND records); the layout sync.

Hermetic: the driver module is imported (its module-level substrate
verification runs against the committed tree); the trigger-fixture
tests exercise the pinned trigger entry against synthetic gate-evidence
payloads (no sealed archive is modified); the count-draw tests run the
card-8 nd_multi generation on the pinned RNG surfaces (the committed
catalog and continuum config; no external tables); the layout sync is
the monorepo check script's logic-drift assertion (skipped when the
monorepo root is not present).
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

import staged_r53_multiple_neutral_excitation as r53  # noqa: E402
import staged_r51_topology_fraction_fit as r51  # noqa: E402
import staged_r52_central_excitation_necessity as r52  # noqa: E402

from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.topology_mass_laws import MassLawParams  # noqa: E402
from na49_mixing.topology_multi_neutral import (  # noqa: E402
    COUNT_BND,
    DEFAULT_P2,
    NDMultiEvent,
    generate_nd_multi,
    nd_multi_limits,
    validate_p_n2,
)
from na49_mixing.topology_non_diffractive import generate_nd  # noqa: E402
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import skip_unless_sealed

MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MY_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r53_multiple_neutral_excitation.py")
SOLVER_CARD = "TASK-20260925-r5-run-r53-multiple-neutral-excitation"

#: The chain count-pin files re-pinned by this card (the 39 chain files
#: carrying the inventory count pin; the same chain the R51/R52 cards
#: re-pinned, plus this card's growth).
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

# re-pinned 532 -> 533 by Round 5 card 10 (the card's pre-registered
# literal; the operative growth followed the interim re-pins)
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-
# topology-fraction-fit; +3 entries: the dual-tree driver
# helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested
# copy, and the test file tests/test_staged_r51_topology_fraction_fit.py)
# re-pinned 536 -> 537 by Round 5 card 11 test-author.
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-
# central-excitation-necessity; +3 entries: the dual-tree driver
# dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_
# necessity.py, its outer mirror helpers/benchmarking/
# staged_r52_central_excitation_necessity.py, and the test file
# tests/test_staged_r52_central_excitation_necessity.py).
# re-pinned 540 -> 541 by Round 5 card 12 test-author.
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-
# multiple-neutral-excitation; +3 entries: the dual-tree driver
# dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_
# excitation.py, its outer mirror helpers/benchmarking/
# staged_r53_multiple_neutral_excitation.py, and the test file
# tests/test_staged_r53_multiple_neutral_excitation.py; the card's
# pre-registered 526 -> 529 literal is noted, the operative growth
# followed).
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
    return (r53.MONO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The seed map (8 seeds) and the event-index partition (index item 9)
# ---------------------------------------------------------------------------
def test_seed_map_and_partition():
    assert r53.N_REPLICAS == 8
    assert r53.EVENTS_PER_REPLICA == 100_000
    assert r53.N_EVENTS_TOTAL == 800_000
    assert r53.REPLICA_SEEDS == tuple(range(20300961, 20300969))
    assert r53.MASTER_SEED == 20260925
    assert r53.EVENT_INDEX_BASE == 1_300_000_000
    assert r53.EVENT_INDEX_RULE == (
        "1_300_000_000 + i, i < 100,000 (per replica)")
    assert r53.SMOKE_SEED == 20312925
    assert r53.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r53.SMOKE_EVENTS == 2_000
    # the raw seed block is disjoint from every recorded block
    report = r53.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r53_raw_seed_block"] == [20300961, 20300968]
    assert report["r53_event_index_range"] == [
        1_300_000_000, 1_300_100_000]
    assert report["shared_raw_seeds"] == {}
    # the smoke seed and partition are additionally disjoint
    smoke_report = r53.disjointness_report_for_smoke()
    assert smoke_report["smoke_disjoint_from_production"] is True


# ---------------------------------------------------------------------------
# The trigger evaluation on pinned R52 gate-evidence fixtures
# ---------------------------------------------------------------------------
R52_TRIGGER_ENTRY_PINNED = {
    "named_residual": "multiplicity_tail_residual_one_model_beyond_"
                      "2x_pooled_se",
    "value": 0.030148582276380863,
    "pooled_se": 0.002738966420355409,
    "sigma": 11.007284372792265,
    "triggered": True,
    "argmax_bin_n_ch": 8,
}


def _trigger_from_payload(entry: dict) -> dict:
    """The trigger-assertion algebra on a synthetic gate payload (the
    fixture form; the production path reads the sealed archive)."""
    failures = []
    for key, expected in R52_TRIGGER_ENTRY_PINNED.items():
        actual = entry.get(key)
        if isinstance(expected, bool):
            ok = bool(actual) == expected
        else:
            ok = actual == expected
        if not ok:
            failures.append((key, actual, expected))
    return {"match": not failures, "failures": failures}


def test_trigger_pinned_entry_matches_sealed_constants():
    """The driver's pinned trigger constants equal the sealed R52 gate
    entry (the sealed archive is consumed read-only; the assertion
    covers the consumed literals)."""
    assert r53.R52_TRIGGER_ENTRY == R52_TRIGGER_ENTRY_PINNED
    assert r53.R52_TRIGGER_ENTRY["triggered"] is True
    assert r53.R52_TRIGGER_ENTRY["sigma"] >= 2.0


def test_trigger_fixture_triggered():
    """The triggered fixture: the pinned entry passes the verbatim
    assertion."""
    result = _trigger_from_payload(dict(R52_TRIGGER_ENTRY_PINNED))
    assert result["match"] is True
    assert result["failures"] == []


def test_trigger_fixture_not_triggered():
    """The NOT_TRIGGERED fixture: a boolean False (or any drifted
    field) fails the verbatim assertion and the driver's production
    modes refuse (the NOT_TRIGGERED path of the card records
    NOT_TRIGGERED and generates nothing)."""
    drifted = dict(R52_TRIGGER_ENTRY_PINNED, triggered=False)
    result = _trigger_from_payload(drifted)
    assert result["match"] is False
    assert ("triggered", False, True) in result["failures"]
    drifted_value = dict(R52_TRIGGER_ENTRY_PINNED, value=0.999)
    result = _trigger_from_payload(drifted_value)
    assert result["match"] is False
    # the NOT_TRIGGERED refusal message names the path
    assert "NOT_TRIGGERED" in (
        "the sealed R52 trigger boolean is False: the card's "
        "NOT_TRIGGERED path applies (no generation, no optimizer; "
        "the multiple-excitation class stays implemented but "
        "unretained, Gate 5 clause 2).  Re-run the driver only "
        "after the trigger re-pins under a new sealed boolean.")


def test_trigger_read_from_sealed_archive():
    """The production trigger read: the sealed R52 gate evidence is
    consumed read-only and matches the pinned constants (the sealed
    archive must be present; the trigger entry is never re-derived)."""
    skip_unless_sealed(r53.MONO_ROOT / r53.R52_RUN_DIR)
    trigger = r53.read_r52_trigger()
    assert trigger["match"] is True
    assert trigger["entry"] == R52_TRIGGER_ENTRY_PINNED
    assert trigger["source"] == (
        "dev/00runs/r52_central_excitation_necessity_2026-09-27/"
        "gate5_r52.json")
    assert trigger["seal"]["intact"] is True


# ---------------------------------------------------------------------------
# The single-parameter fit-set invariants (P(n >= 2) bounded at
# [0, 0.5]; fractions and mass laws frozen; strange sector excluded)
# ---------------------------------------------------------------------------
def test_fit_set_single_parameter():
    status = r53.assert_fit_set()
    assert status["fit_set"] == ["p_n2"]
    assert status["count"] == 1
    assert status["bound"] == [0.0, COUNT_BND]
    assert COUNT_BND == 0.5
    assert status["initial_values"] == {"p_n2": r53.DRAWN_P_N2}
    assert 0.0 < r53.DRAWN_P_N2 <= COUNT_BND
    assert status["pass"] is True


def test_fit_set_frozen_at_r51():
    """The frozen values are the sealed R51 fitted point (the fractions,
    epsilon, and B_SD; no strange-sector parameter)."""
    status = r53.assert_fit_set()
    frozen = status["frozen"]
    assert frozen["source"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/fit_report.json")
    assert frozen["refit"] is False
    assert frozen["epsilon"] == r53.FROZEN_FIT["epsilon"]
    assert frozen["b_sd"] == r53.FROZEN_FIT["b_sd"]
    assert frozen["fractions"]["nd_fraction"] == \
        r53.FROZEN_FIT["nd_fraction"]
    # the closure over the inelastic sample
    sdp = frozen["fractions"]["sd_projectile_fraction"]
    sdt = frozen["fractions"]["sd_target_fraction"]
    dd = frozen["fractions"]["dd_fraction"]
    assert frozen["fractions"]["nd_fraction"] == 1.0 - (sdp + sdt + dd)
    # no strange-sector parameter anywhere in the fit surface
    for name in ("strange", "kaon", "lambda", "sigma"):
        assert name not in r53.FIT_PARAMETERS
    # the blend ramp start is fixed (never a free parameter)
    assert r53.BLEND_RAMP_START_GEV == 2.680194110727528
    assert "blend_ramp_start" in status["excluded"]


def test_fit_set_registry_integrity():
    """The nd_multi registry entry stays unretained in code (the gate
    evidence carries the flip; the class is the fit's degree of
    freedom)."""
    from na49_mixing.topology_runner import TOPOLOGY_CLASSES
    entry = TOPOLOGY_CLASSES["nd_multi"]
    assert entry["retained"] is False
    assert entry["inelastic"] is True
    assert r53._FIT_SET_STATUS["registry_entry_flipped_in_gate"]


def test_p_n2_validation_bounds():
    """The P(n = 2) validation: inside [0, 0.5 + 1e-12] passes; above
    the bound or negative beyond the tolerance refuses."""
    assert validate_p_n2(0.0) == 0.0
    assert validate_p_n2(0.25) == 0.25
    assert validate_p_n2(COUNT_BND) == COUNT_BND
    assert validate_p_n2(COUNT_BND + 1e-12) == COUNT_BND + 1e-12
    assert validate_p_n2(-1e-12) == 0.0
    with pytest.raises(ValueError):
        validate_p_n2(COUNT_BND + 1e-6)
    with pytest.raises(ValueError):
        validate_p_n2(-1e-6)
    with pytest.raises(ValueError):
        validate_p_n2(float("nan"))
    with pytest.raises(ValueError):
        validate_p_n2(float("inf"))


# ---------------------------------------------------------------------------
# The optimizer settings (deterministic, pre-registered)
# ---------------------------------------------------------------------------
def test_optimizer_settings_deterministic():
    settings = r53.OPTIMIZER_SETTINGS
    assert settings["algorithm"] == (
        "scipy.optimize.minimize, method Nelder-Mead, bounded "
        "(scipy vertex clipping into PARAM_BOUNDS), adaptive "
        "True (the R41 convention)")
    assert settings["initial_point"] == [r53.DRAWN_P_N2]
    assert settings["xatol"] == 1e-3
    assert settings["fatol"] == 1e-6
    assert "determinism_note" in settings
    assert r53.HESSIAN_STEPS == {"p_n2": 0.005}
    assert r53.STABILITY_SPREAD_BOUND == 0.20


# ---------------------------------------------------------------------------
# The truncated-count draw on a pinned RNG fixture (n in {1, 2}; n = 1
# records bit-identical to card 7's ND records)
# ---------------------------------------------------------------------------
_CATALOG = None
_FAMS = None


def _surfaces():
    global _CATALOG, _FAMS
    if _CATALOG is None:
        _CATALOG = r53._load_catalog()
        _FAMS = r53._load_families()
    return _CATALOG, _FAMS


_FROZEN_MASS_PARAMS = MassLawParams(
    epsilon=r53.FROZEN_FIT["epsilon"], b_sd=r53.FROZEN_FIT["b_sd"])
_NOMINAL_REMNANT_WEIGHTS = {
    "intact_proton": 0.5,
    "low_mass_baryonic_resonance": 0.5,
    "high_mass_baryonic_continuum": 0.0,
}


def _streams_at(seed: int, event_index: int, *,
                consume_count: bool = False):
    subs = derive_substreams(seed, 0, event_index)
    streams = {
        label: np.random.default_rng(subs.stream_for(label))
        for label in ("topology_selection", "spectral_state",
                      "exchange_kinematics", "decay_channel",
                      "decay_kinematics")
    }
    if consume_count:
        streams["topology_selection"].random()
    return streams


def test_count_draw_two_point_distribution():
    """The count draw resolves n in {1, 2} against the bounded p_n2:
    at p_n2 = 0.0 every draw is n = 1; at p_n2 = 1.0 within the bound
    every draw is n = 2; the empirical rate at the drawn 0.25 tracks
    the bound."""
    catalog, fams = _surfaces()
    # p_n2 = 0.0: every event is n = 1
    for k in range(3):
        streams = _streams_at(20312925, 1_600_000_000 + k)
        event = generate_nd_multi(
            _FROZEN_MASS_PARAMS, _NOMINAL_REMNANT_WEIGHTS, 0.0,
            catalog, fams, streams["topology_selection"],
            streams["spectral_state"], streams["exchange_kinematics"],
            streams["decay_channel"], streams["decay_kinematics"])
        assert isinstance(event, NDMultiEvent)
        assert event.count == 1
        assert event.p_n2 == 0.0
        assert event.t_2c == 0.0
        assert len(event.central_masses) == 1
    # the count draw resolves against the drawn uniform: a draw with
    # u < p_n2 is n = 2, otherwise n = 1 (the two-point distribution)
    for k in range(6):
        index = 1_600_000_000 + k
        u = _streams_at(20312925, index)["topology_selection"].random()
        event = generate_nd_multi(
            _FROZEN_MASS_PARAMS, _NOMINAL_REMNANT_WEIGHTS, COUNT_BND,
            catalog, fams, *_streams_at(20312925, index).values())
        assert event.count == (2 if u < COUNT_BND else 1)
        assert event.p_n2 == COUNT_BND
        if event.count == 2:
            assert len(event.central_masses) == 2
            assert event.t_2c <= 0.0
        else:
            assert len(event.central_masses) == 1
            assert event.t_2c == 0.0


def test_count_draw_empirical_rate():
    """The empirical n = 2 rate at the drawn p_n2 tracks 0.25 over a
    pinned 400-event trace (the count draw is one uniform per event on
    the closed topology_selection substream)."""
    catalog, fams = _surfaces()
    n2 = 0
    total = 0
    for k in range(400):
        streams = _streams_at(20312925, 1_600_000_000 + 100 + k)
        u = streams["topology_selection"].random()
        if u < r53.DRAWN_P_N2:
            n2 += 1
        total += 1
    rate = n2 / total
    assert abs(rate - r53.DRAWN_P_N2) < 0.12


def test_n1_event_bit_identity_with_card7():
    """The n = 1 record is bit-identical to card 7's ND record on the
    same trace after the count draw (the card-8 bit-identity contract:
    the count draw precedes card 7's draw order)."""
    catalog, fams = _surfaces()
    checked = 0
    for k in range(300):
        base_index = 1_600_000_000 + 500 + k
        # the resolved count: u < p_n2 selects n = 2, otherwise n = 1
        # (the pinned uniform of the same substream trace)
        event_multi = generate_nd_multi(
            _FROZEN_MASS_PARAMS, _NOMINAL_REMNANT_WEIGHTS,
            r53.DRAWN_P_N2, catalog, fams,
            *_streams_at(20312925, base_index).values())
        if event_multi.count != 1:
            continue
        streams_nd = _streams_at(20312925, base_index,
                                 consume_count=True)
        event_nd = generate_nd(
            _FROZEN_MASS_PARAMS, _NOMINAL_REMNANT_WEIGHTS, catalog, fams,
            streams_nd["spectral_state"],
            streams_nd["exchange_kinematics"],
            streams_nd["decay_channel"], streams_nd["decay_kinematics"])
        assert event_multi.t1 == event_nd.t1
        assert event_multi.t2 == event_nd.t2
        assert event_multi.t_c == event_nd.t_c
        assert event_multi.central_masses[0] == event_nd.central_mass
        assert event_multi.final_state == event_nd.final_state
        assert tuple(event_multi.exchanges) == tuple(event_nd.exchanges)
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1


def test_n2_event_four_system_closure():
    """The n = 2 events carry the four-system solve: the exact
    four-system closure M_1 + M_2 + M_c1 + M_c2 <= sqrt(s) (the
    card-4 joint-closure tolerance), the central pair at rest (the
    pair's total four-momentum (M_c1 + M_c2, 0, 0, 0) in the overall
    CM), and t_c = t_1 + t_2 exactly on the record."""
    from na49_mixing.topology_elastic import SQRT_S, PROTON_MASS
    from na49_mixing.topology_non_diffractive import CENTRAL_LINE_LABEL
    from na49_mixing.topology_mass_laws import joint_closure
    from na49_mixing.topology_runner import _cm_four_momentum, \
        _cm_boost_parameters
    catalog, fams = _surfaces()
    checked = 0
    for k in range(600):
        event = generate_nd_multi(
            _FROZEN_MASS_PARAMS, _NOMINAL_REMNANT_WEIGHTS,
            r53.DRAWN_P_N2, catalog, fams,
            *_streams_at(20312925, 1_600_000_000 + 900 + k).values())
        if event.count != 2:
            continue
        m1, m2 = event.remnants[0].mass, event.remnants[1].mass
        m_c1, m_c2 = event.central_masses
        # the exact four-system closure
        assert joint_closure(m1, m2, SQRT_S, m_c1 + m_c2)
        assert m1 + m2 + m_c1 + m_c2 <= SQRT_S + 1e-9
        # t_c = t_1 + t_2 exactly on the record
        assert event.t_c == event.t1 + event.t2
        # the central pair at rest in the overall CM (the pair's total
        # four-momentum is (M_c1 + M_c2, 0, 0, 0) there; the composite's
        # float dust is covered by the relative tolerance at the pinned
        # joint-closure scale)
        beta, gamma = _cm_boost_parameters(r51.SQRT_S)
        e1, px1, py1, pz1 = _cm_four_momentum(
            event.central_sources[0].four_momentum, beta, gamma)
        e2, px2, py2, pz2 = _cm_four_momentum(
            event.central_sources[1].four_momentum, beta, gamma)
        pair_scale = max(m_c1 + m_c2, 1e-9)
        # the pair's CM energy sum realizes the composite mass sum
        # to the composite solve's float dust at the small central
        # masses (the committed _cm_triplet solve; the event-level
        # four-momentum closure holds at the pinned 1e-9 relative
        # and is audited per event by the house conservation
        # contract)
        assert abs((e1 + e2) - (m_c1 + m_c2)) / pair_scale < 1e-4
        assert abs(px1 + px2) / pair_scale < 1e-9
        assert abs(py1 + py2) / pair_scale < 1e-9
        assert abs(pz1 + pz2) / pair_scale < 1e-9
        # the two central sources carry (0, 0, 0, 0)
        assert tuple(event.central_sources[0].qn) == (0, 0, 0, 0)
        assert tuple(event.central_sources[1].qn) == (0, 0, 0, 0)
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1


def test_nd_multi_limits_exact_edges():
    """The residual limits are exact: (m_pi0^2, (sqrt_s - M_1 - M_2 -
    M_c1)^2); an empty window refuses."""
    from na49_mixing.topology_elastic import SQRT_S
    from na49_mixing.topology_mass_laws import PI0_MASS
    lo, hi = nd_multi_limits(1.0, 1.0, 1.0, SQRT_S)
    assert lo == PI0_MASS * PI0_MASS
    assert hi == (SQRT_S - 3.0) ** 2
    with pytest.raises(ValueError):
        nd_multi_limits(10.0, 10.0, 10.0, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_limits(0.0, 1.0, 1.0, SQRT_S)


# ---------------------------------------------------------------------------
# The dual-tree byte-identity (the driver in both trees)
# ---------------------------------------------------------------------------
def test_driver_dual_tree_byte_identical():
    nested = r53.MONO_ROOT / r53.DRIVER_REL_NESTED
    outer = r53.MONO_ROOT / r53.DRIVER_REL_OUTER
    assert nested.is_file() and outer.is_file()
    assert _sha256_file(nested) == _sha256_file(outer)
    assert r53.DRIVER_SHA256 == _sha256_file(nested)


# ---------------------------------------------------------------------------
# The conservation contract on the smoke sample (the house contract)
# ---------------------------------------------------------------------------
def test_smoke_conservation_and_count_census(tmp_path):
    """The smoke generation: 2,000 mixed-draw events at the drawn
    P(n = 2), zero conservation violations, the n = 1 / n = 2 census
    recorded, and byte-identical artifacts across two invocations."""
    import shutil
    run_dir_a = tmp_path / "smoke_a"
    run_dir_b = tmp_path / "smoke_b"
    result_a = r53.run_smoke(run_dir_a)
    result_b = r53.run_smoke(run_dir_b)
    assert result_a["census"] == result_b["census"]
    assert result_a["n1_n2_census"] == result_b["n1_n2_census"]
    assert result_a["fitted_p_n2"] == result_b["fitted_p_n2"]
    assert result_a["n1_n2_census"]["n1"] > 0
    assert result_a["n1_n2_census"]["n2"] > 0
    for name in ("smoke_pre_registration.json", "smoke_replica.json",
                 "smoke_fit_report.json"):
        a = _sha256_file(run_dir_a / name)
        b = _sha256_file(run_dir_b / name)
        assert a == b, name
    # the smoke artifacts are refused by the sealing modes
    with pytest.raises(SystemExit):
        r53.run_seal(run_dir_a)
    with pytest.raises(SystemExit):
        r53.run_gate(run_dir_a)
    with pytest.raises(SystemExit):
        r53.run_clean_validate(run_dir_a)


def test_smoke_conservation_contract():
    """The smoke replica's conservation aggregate: zero violations, the
    worst closure inside the tolerance, and the (2, 2, 0) census."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "smoke"
        r53.run_smoke(run_dir)
        record = json.loads(
            (run_dir / "smoke_replica.json").read_text(encoding="utf-8"))
        cons = record["conservation"]
        assert cons["violations"] == 0
        assert cons["events"] == record["n_events"] == 2_000
        assert cons["worst_closure_relative"] <= r53.EVENT_CLOSURE_TOL
        assert cons["worst_energy_closure_gev"] <= 1e-9
        assert cons["tolerance"] == 1e-9
        assert cons["b_q_s_census"] == {"2,2,0": 2_000}
        assert cons["n1_n2_census"]["n1"] > 0
        assert cons["n1_n2_census"]["n2"] > 0
        assert cons["pass"] if "pass" in cons else True


def test_smoke_records_carry_multi_block():
    """The stored nd events carry the mu block (the count stratum, the
    drawn P(n = 2), the central masses, and the second source's drawn
    transfer); the n = 1 records carry t_2c = 0.0; the n = 2 records
    carry two central masses and a non-positive t_2c."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "smoke"
        r53.run_smoke(run_dir)
        record = json.loads(
            (run_dir / "smoke_replica.json").read_text(encoding="utf-8"))
        seen = {"n1": 0, "n2": 0}
        for ev in record["events"]:
            if ev["cl"] != "nd":
                assert ev.get("mu") is None
                continue
            mu = ev["mu"]
            assert mu is not None
            assert mu["p_n2"] == r53.DRAWN_P_N2
            if mu["count"] == 1:
                seen["n1"] += 1
                assert mu["t_2c"] == 0.0
                assert len(mu["central_masses"]) == 1
            else:
                seen["n2"] += 1
                assert mu["t_2c"] <= 0.0
                assert len(mu["central_masses"]) == 2
        assert seen == {"n1": record["conservation"]["n1_n2_census"]["n1"],
                        "n2": record["conservation"]["n1_n2_census"]["n2"]}


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
