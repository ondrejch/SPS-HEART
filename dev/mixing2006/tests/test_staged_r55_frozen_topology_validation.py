"""The card-15 hermetic tests (Round 5, run R55).

``TASK-20260925-r5-run-r55-frozen-topology-validation``; plan section 11
run R55, lines 932-934.  The card's test surface: the seed map (20
seeds) and the event-index partition; the frozen-set assertion against
pinned R51/R52/R53 archive fixtures (both the retained and the
not-retained R52 outcomes, and the R53-absent and R53-present cases);
the validation observable set invariants (leading baryons, gap size,
charged and species-separated multiplicities, nonstrange spectra,
disjointness from the R51 training observables); the R54 audit
precondition guard; the accumulation and deviance machinery on
synthetic data; the layout sync.

Hermetic: the driver module is imported (its module-level substrate
verification runs against the committed tree); the frozen-set tests
exercise the driver's pinned constants and, when the sealed R51/R52/R53
archives are present on the checkout, the sealed gate records read-only
(the artifact pins skip on a fresh checkout); the update-rule tests
exercise the pure two-case function on synthetic verdict fixtures (no
sealed archive is modified); the machinery tests generate pinned class
events on the committed catalog and continuum config (no external
tables) and run the accumulation and deviance algebra on synthetic
data; the smoke test runs the driver's hermetic smoke mode into a
temporary directory (never sealed); the layout sync is the monorepo
check script's logic-drift assertion (skipped when the monorepo root is
not present).
"""

from __future__ import annotations

import hashlib
import json
import math
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

import staged_r55_frozen_topology_validation as r55  # noqa: E402
import staged_r51_topology_fraction_fit as r51  # noqa: E402

from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.topology_runner import (  # noqa: E402
    TOPOLOGY_CLASSES,
    TopologyNominalParams,
    run_topology,
)
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import skip_unless_sealed

MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MY_TEST_REL = "dev/mixing2006/tests/test_staged_r55_frozen_topology_validation.py"
MY_DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r55_frozen_topology_validation.py")
MY_DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r55_frozen_topology_validation.py")
SOLVER_CARD = "TASK-20260925-r5-run-r55-frozen-topology-validation"

#: The chain count-pin files re-pinned by this card (the 46 files
#: carrying the inventory count pin after the card-13 test-author
#: landing; the same chain the R50-R54 cards re-pinned, plus this
#: card's growth).
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
    "dev/mixing2006/tests/"
    "test_staged_r53_multiple_neutral_excitation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r54_local_ledger_audit.py",
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
# local-ledger-audit; +3 entries: the dual-tree driver, its outer
# mirror, and the card-14 test file)
# re-pinned 547 -> 548 by Round 5 card 13 test-author
# (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author;
# +1 entry: the card-13 test-author register file)
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-
# frozen-topology-validation; +3 entries: the dual-tree driver
# dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_
# validation.py, its outer mirror helpers/benchmarking/
# staged_r55_frozen_topology_validation.py, and the test file
# tests/test_staged_r55_frozen_topology_validation.py; the card's
# pre-registered 532 -> 535 literal is noted, the operative growth
# followed).
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
    return (r55.MONO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The seed map (20 seeds) and the event-index partition (index item 9)
# ---------------------------------------------------------------------------
def test_seed_map_and_partition():
    assert r55.N_REPLICAS == 20
    assert r55.EVENTS_PER_REPLICA == 250_000
    assert r55.N_EVENTS_TOTAL == 5_000_000
    assert r55.N_REPLICAS * r55.EVENTS_PER_REPLICA == r55.N_EVENTS_TOTAL
    assert r55.REPLICA_SEEDS == tuple(range(20300989, 20301009))
    assert r55.MASTER_SEED == 20260925
    assert r55.EVENT_INDEX_BASE == 1_500_000_000
    assert r55.EVENT_INDEX_RULE == (
        "1_500_000_000 + i, i < 250,000 (per replica)")
    assert r55.SMOKE_SEED == 20312925
    assert r55.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r55.SMOKE_EVENTS == 2_000
    assert r55.SMOKE_EVENT_INDEX_RULE == (
        "1_600_000_000 + i, i < 2,000")
    assert r55.SMOKE_EXERCISE_INDEX_BASE == 1_610_000_000
    assert r55.SMOKE_EXERCISE_EVENTS_PER_CLASS == 8
    # the raw seed block is disjoint from every recorded block
    report = r55.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r55_raw_seed_block"] == [20300989, 20301008]
    assert report["r55_event_index_range"] == [
        1_500_000_000, 1_500_250_000]
    assert report["shared_raw_seeds"] == {}
    # the smoke seed, partition, and class-exercise partition are
    # additionally disjoint
    smoke_report = r55.disjointness_report_for_smoke()
    assert smoke_report["smoke_disjoint_from_production"] is True
    assert smoke_report["smoke_class_exercise_index_range"] == [
        1_610_000_000, 1_610_000_048]


def test_statistics_invariants():
    """Exactly 5,000,000 events = 20 replicas x 250,000 (T3, plan
    section 5.1 line 255; the card's pre-registered instantiation)."""
    assert r55.N_EVENTS_TOTAL == 5_000_000
    assert r55.N_REPLICAS * r55.EVENTS_PER_REPLICA == r55.N_EVENTS_TOTAL
    assert len(r55.REPLICA_SEEDS) == r55.N_REPLICAS
    assert "20 replicas x 250,000" in r55.OBJECTIVE_NOTE
    # no optimizer on this run (index item 10)
    assert "No optimizer (index item 10)" in r55.OBJECTIVE_NOTE
    assert "frozen-parameter evaluation" in r55.OBJECTIVE_NOTE


# ---------------------------------------------------------------------------
# The frozen-set invariants (the round's frozen point; index item 7)
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


def test_frozen_point_matches_sealed_r51():
    """The driver's pinned frozen point is byte-identical to the sealed
    R51 fit report's fitted values (the sealed archive is consumed
    read-only; the assertion covers the consumed literals)."""
    assert r55.FROZEN_FIT == FROZEN_FIT_PINNED
    assert r55.R51_FROZEN_FRACTIONS == R51_FROZEN_FRACTIONS_PINNED
    skip_unless_sealed(r55.MONO_ROOT / r55.R51_RUN_DIR)
    assertion = r55.assert_fractions_against_r51()
    assert assertion["match"] is True
    assert assertion["source"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/fit_report.json")
    assert assertion["seal"]["intact"] is True
    assert assertion["r51_optimizer_converged"] is False
    assert "weight_degeneracy" in assertion["r51_named_residual_kinds"]


def test_frozen_elastic_complement():
    """The elastic fraction is the exact complement 0.0 of the frozen
    inelastic mass (the recorded source convention); the elastic class
    stays in the generation surface."""
    assert r55.ELASTIC_FRACTION_GENERATION == 0.0
    assert r55.ELASTIC_FRACTION_GENERATION == 1.0 - r51.math.fsum(
        r55.R51_FROZEN_FRACTIONS.values())
    assert "elastic" in r55.SMOKE_EXERCISE_CLASSES
    assert TOPOLOGY_CLASSES["elastic"]["retained"] is True
    assert TOPOLOGY_CLASSES["elastic"]["inelastic"] is False


# ---------------------------------------------------------------------------
# The R52 update rule (the pure two-case function; both outcomes plus
# the mixed-verdict refusal)
# ---------------------------------------------------------------------------
R52_RENORM_PINNED = {
    "dd": 0.9999997042124792,
    "nd": 0.0,
    "sd_projectile": 3.619686258303196e-09,
    "sd_target": 2.9216783460953685e-07,
}


def test_r52_update_rule_not_retained():
    """The sealed R52 outcome's shape: the zero model retained by no
    family keeps the ND fraction at the R51 fit value (the card's
    update rule; the live outcome's recorded verdicts)."""
    verdicts = {f: "not retained" for f in (
        "central_spectrum_k_minus", "central_spectrum_k_plus",
        "central_spectrum_pi_minus", "central_spectrum_pi_plus",
        "multiplicity_tails")}
    fractions, zero_retained, note = r55.r52_update_rule(
        verdicts, R52_RENORM_PINNED)
    assert zero_retained is False
    assert fractions == R51_FROZEN_FRACTIONS_PINNED
    assert "one-excitation topology is retained" in note
    # the live sealed outcome agrees (when present on the checkout)
    skip_unless_sealed(r55.MONO_ROOT / r55.R52_RUN_DIR)
    outcome = r55.read_r52_outcome()
    assert outcome["zero_model_retained"] is False
    assert outcome["frozen_fractions_after_r52"] == \
        R51_FROZEN_FRACTIONS_PINNED
    assert outcome["zero_model_renormalized_fractions"] == R52_RENORM_PINNED
    assert outcome["retention_verdicts"] == verdicts
    assert outcome["seal"]["intact"] is True


def test_r52_update_rule_retained():
    """The counterfactual outcome: the zero model retained by every
    family moves the ND fraction to zero and the remaining three
    fractions to the R52 renormalized values (the card's update rule)."""
    verdicts = {f: "retained" for f in (
        "central_spectrum_k_minus", "central_spectrum_k_plus",
        "central_spectrum_pi_minus", "central_spectrum_pi_plus",
        "multiplicity_tails")}
    fractions, zero_retained, note = r55.r52_update_rule(
        verdicts, R52_RENORM_PINNED)
    assert zero_retained is True
    assert fractions == {
        "sd_projectile": 3.619686258303196e-09,
        "sd_target": 2.9216783460953685e-07,
        "dd": 0.9999997042124792,
        "nd": 0.0,
    }
    assert math.fsum(fractions.values()) == 1.0
    assert "renormalized zero-model values" in note


def test_r52_update_rule_mixed_refuses():
    """A mixed per-family verdict is outside the card's two-case rule:
    the run refuses (never silently absorbed)."""
    verdicts = {
        "central_spectrum_pi_plus": "not retained",
        "central_spectrum_pi_minus": "retained",
        "central_spectrum_k_plus": "not retained",
        "central_spectrum_k_minus": "not retained",
        "multiplicity_tails": "not retained",
    }
    with pytest.raises(SystemExit):
        r55.r52_update_rule(verdicts, R52_RENORM_PINNED)


def test_r52_outcome_consistency_guard():
    """The generation-time consistency guard: a live outcome drifting
    against the pre-registered block refuses."""
    pre = {"topology_sampling": {"r52_outcome": {
        "zero_model_retained": False,
        "frozen_fractions_after_r52": dict(R51_FROZEN_FRACTIONS_PINNED)}}}
    drifted = {"zero_model_retained": True,
               "frozen_fractions_after_r52": {
                   "sd_projectile": 3.619686258303196e-09,
                   "sd_target": 2.9216783460953685e-07,
                   "dd": 0.9999997042124792, "nd": 0.0}}
    with pytest.raises(SystemExit):
        r55.assert_r52_outcome_against_preregistration(drifted, pre)
    skip_unless_sealed(r55.MONO_ROOT / r55.R52_RUN_DIR)
    outcome = r55.read_r52_outcome()
    check = r55.assert_r52_outcome_against_preregistration(outcome, pre)
    assert check["consistent"] is True


# ---------------------------------------------------------------------------
# The R53 outcome (the nd_multi dispatch; the absent and present cases)
# ---------------------------------------------------------------------------
def test_r53_outcome_present_unretained():
    """The sealed R53 archive (present on this checkout): retained
    false, the unretained default P(n = 2) = 0.0; the sampling surface
    keeps the nd dispatch off and the six-entry fractions block records
    nd_multi 0.0."""
    for rel in (r55.R52_RUN_DIR, r55.R53_RUN_DIR):
        skip_unless_sealed(r55.MONO_ROOT / rel)
    outcome = r55.read_r53_outcome()
    assert outcome["pending"] is False
    assert outcome["retained"] is False
    assert outcome["p_n2"] == 0.0
    assert outcome["status"] in ("TRIGGERED", "NOT_TRIGGERED")
    assert outcome["seal"]["intact"] is True
    sampling = r55.sampling_from_outcomes(
        r55.read_r52_outcome(), outcome)
    assert sampling["nd_multi_active"] is False
    assert sampling["fractions"]["nd_multi"] == 0.0
    assert sampling["params"].p_n2 == 0.0
    assert sampling["fractions"]["dd"] == \
        R51_FROZEN_FRACTIONS_PINNED["dd"]


def test_r53_outcome_absent_pending_default(monkeypatch, tmp_path):
    """The R53-PENDING default (the archive-absent case, exercised on a
    monkeypatched run dir): retained false, P(n = 2) = 0.0, the
    re-audit obligation recorded."""
    monkeypatch.setattr(r55, "R53_RUN_DIR",
                        str(tmp_path / "absent_r53"))
    outcome = r55.read_r53_outcome()
    assert outcome["pending"] is True
    assert outcome["retained"] is False
    assert outcome["p_n2"] == 0.0
    assert "re-audit obligation" in outcome["reaudit_obligation"]


def test_r53_retained_fixture_dispatch():
    """The dispatch fixture: a retained R53 outcome activates the
    nd_multi dispatch at the fitted P(n = 2); a boundary-zero fitted
    value keeps the card-7 dispatch."""
    r52 = {"frozen_fractions_after_r52": dict(R51_FROZEN_FRACTIONS_PINNED)}
    retained = {"status": "TRIGGERED", "pending": False, "retained": True,
                "p_n2": 0.25}
    sampling = r55.sampling_from_outcomes(r52, retained)
    assert sampling["nd_multi_active"] is True
    assert sampling["fractions"]["nd_multi"] == 0.25
    assert sampling["params"].p_n2 == 0.25
    boundary = {"status": "TRIGGERED", "pending": False, "retained": True,
                "p_n2": 0.0}
    sampling = r55.sampling_from_outcomes(r52, boundary)
    assert sampling["nd_multi_active"] is False
    assert sampling["params"].p_n2 == 0.0


def test_r53_pending_to_retained_flip_refuses():
    """The re-audit obligation: a pending pre-registration confronted
    with a sealed retained-true outcome refuses (never silently
    absorbed)."""
    pre = {"topology_sampling": {"nd_multi": {
        "pending": True, "retained": False, "p_n2": 0.0}}}
    flipped = {"pending": False, "retained": True, "p_n2": 0.25}
    with pytest.raises(SystemExit):
        r55.assert_outcome_against_preregistration(flipped, pre)
    resolved = {"pending": False, "retained": False, "p_n2": 0.0}
    check = r55.assert_outcome_against_preregistration(resolved, pre)
    assert check["consistent"] is True


# ---------------------------------------------------------------------------
# The R54 audit precondition guard (the card's dependency)
# ---------------------------------------------------------------------------
def test_r54_audit_precondition():
    """The sealed R54 ledger audit passes on the frozen topology (zero
    violations on the five ledger rules over 1,000,000 events; the
    recorded topology sampling fractions equal the frozen fractions
    block; the recorded R53 outcome matches the live read)."""
    skip_unless_sealed(r55.MONO_ROOT / r55.R54_RUN_DIR)
    precondition = r55.assert_r54_audit_precondition()
    assert precondition["pass"] is True
    assert precondition["ledger_audit_pass"] is True
    assert precondition["violations"] == 0
    assert precondition["events_checked"] == 1_000_000
    assert precondition["topology_sampling_matches_frozen_fractions"] is True
    assert precondition["r53_outcome_consistent"] is True
    assert precondition["source"] == (
        "dev/00runs/r54_local_ledger_audit_2026-09-28/gate5_r54.json")
    assert precondition["seal"]["intact"] is True


# ---------------------------------------------------------------------------
# The validation-observable register invariants (index item 7)
# ---------------------------------------------------------------------------
def test_validation_register():
    """The validation observable set: the leading baryon x_F spectra of
    p and p-bar, the gap-size distribution, the charged and
    species-separated multiplicity distributions, the nonstrange
    spectra of pi+, pi-, pi0, p, p-bar; the data-backed / model-only
    split; the named-residual bounds; the pinned binnings."""
    surface = r55._FIT_SURFACE_STATUS
    assert surface["pass"] is True
    register = surface["validation_register"]
    assert register["leading_baryon_xf"] == ["proton", "antiproton"]
    assert register["gap_size_distribution"] is True
    assert register["multiplicity"] == "charged and species-separated"
    assert register["nonstrange_spectra"] == [
        "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton"]
    assert r55.DATA_SPECIES == ("pi_plus", "pi_minus", "proton",
                                "antiproton")
    assert r55.MODEL_ONLY_SPECIES == ("pi_zero",)
    assert r55.SIGMA_TOLERANCE == 3.0
    assert r55.NAMED_FINDING_SE_FACTOR == 2.0
    assert r55.SECTOR_DEGRADATION_FRACTION == 0.05
    assert len(r55.GAP_HISTOGRAM_EDGES) == 51
    assert len(r55.LEAD_XF_HISTOGRAM_EDGES) == 43
    assert r55.NCH_SUPPORT_LO == 2 and r55.NCH_SUPPORT_HI == 20
    assert r55.SPECIES_MULT_SUPPORT_HI == 20
    assert r55.SPECTRUM_PDGS == {
        "pi_plus": 211, "pi_minus": -211, "pi_zero": 111,
        "proton": 2212, "antiproton": -2212}


def test_validation_register_disjoint_from_r51_training():
    """The register's disjointness from the R51 training observables:
    the R51 training statistics were the leading-proton and B_transport
    published-cell spectra and the R50-pinned gap-fraction and charged-
    multiplicity references; the R55 register's 1D leading-baryon x_F
    spectra, gap-size distribution, and species-separated
    multiplicities are new statistics on the pinned R50 binnings, and
    the nonstrange published-cell spectra extend the R51 proton cell
    set to the four published nonstrange species (the recorded overlap
    note)."""
    note = r55.R51_TRAINING_NOTE
    assert "leading-proton spectrum" in note
    assert "B_transport" in note
    assert "gap fraction" in note
    assert "charged-multiplicity" in note
    # the R55 register's statistic keys carry no R51 training key
    r51_training_keys = {
        "leading_proton_spectrum", "b_transport_spectrum",
        "gap_fraction_abs_y_lt_0.5", "charged_multiplicity_distribution"}
    r55_keys = (
        {f"leading_baryon_xf_{sp}" for sp in ("proton", "antiproton")}
        | {"gap_size_distribution", "charged_multiplicity_distribution"}
        | {f"species_multiplicity_{sp}" for sp in r55.SPECIES_MULT_ORDER}
        | {f"{sp}_spectrum" for sp in r55.SPECTRUM_SPECIES_ORDER})
    overlap = r55_keys & r51_training_keys
    assert overlap == {"charged_multiplicity_distribution"}, (
        "the recorded overlap: the charged-multiplicity family shares "
        "the R51 training statistic's bins and reference (recorded in "
        "R51_TRAINING_NOTE); no other register key overlaps")
    # the pion spectra are not R51 training observables
    assert "pi_plus_spectrum" not in r51_training_keys
    assert "pi_minus_spectrum" not in r51_training_keys


# ---------------------------------------------------------------------------
# The accumulation and deviance machinery (synthetic data)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_geometry():
    return r55.build_geometry(r55._load_smoke_tables())


def test_accumulator_histogram_semantics(smoke_geometry):
    """The pinned histogram semantics: the np.histogram bin edges (the
    last bin closed on the right), the out-of-range drops counted, and
    the support refusals."""
    acc = r55._ReplicaAccumulator(smoke_geometry)
    # the gap-size edges: the first, an interior, and the last bin
    assert acc._bin_index(0.0, r55.GAP_HISTOGRAM_EDGES) == 0
    assert acc._bin_index(0.3456, r55.GAP_HISTOGRAM_EDGES) == 1
    assert acc._bin_index(17.28, r55.GAP_HISTOGRAM_EDGES) == 49
    assert acc._bin_index(17.29, r55.GAP_HISTOGRAM_EDGES) is None
    assert acc._bin_index(-0.01, r55.LEAD_XF_HISTOGRAM_EDGES) == 20
    assert acc._bin_index(1.05, r55.LEAD_XF_HISTOGRAM_EDGES) == 41
    assert acc._bin_index(-1.06, r55.LEAD_XF_HISTOGRAM_EDGES) is None
    acc.add_gap(None)
    assert acc.gap["events_without_field"] == 1
    acc.add_gap(1.0)
    assert acc.gap["hist"][2] == 1
    assert acc.gap["events_with_field"] == 1
    acc.add_nch(2)
    acc.add_nch(20)
    assert acc.nch[0] == 1 and acc.nch[18] == 1
    with pytest.raises(r55.ValidationRefusal):
        acc.add_nch(21)
    with pytest.raises(r55.ValidationRefusal):
        acc.add_nch(1)
    acc.add_species_count("pi_plus", 20)
    with pytest.raises(r55.ValidationRefusal):
        acc.add_species_count("pi_plus", 21)
    # the spectra rows land inside the species' dense slice (a forward
    # particle whose CM x_F ~ 0.1 and p_T 0.3 sit inside the smoke
    # grid's window; an out-of-window particle drops without a row)
    acc.add_spectrum_final("pi_plus", (17.56, 0.3, 0.0, 17.55))
    assert len(acc.rows_flat) == 1
    assert 0 <= acc.rows_flat[0] < smoke_geometry["dense_total"]
    n_rows = len(acc.rows_flat)
    acc.add_spectrum_final("pi_plus", (1.0, 0.3, 0.0, 1.0))
    assert len(acc.rows_flat) == n_rows


def test_deviance_machinery():
    """The deviance algebra: the sigma^2 = v + se^2 total form with the
    data and MC contributions reported separately; the structural drop
    rule; the beyond-bound flags."""
    d = np.array([1.0, 2.0, np.nan])
    v = np.array([0.5, 0.5, 0.5])
    m = np.array([1.5, 1.0, 1.0])
    se = np.array([0.1, 0.1, 0.1])
    components = r55.sigma_components(d, v, m, se)
    # bin 0: (1 - 1.5)^2 / (0.5 + 0.01) ; bin 1: (2 - 1)^2 / (0.5 + 0.01)
    assert components["deviance"] == pytest.approx(
        0.25 / 0.51 + 1.0 / 0.51)
    assert components["n_kept"] == 2
    block = r55._family_block(
        "k", "distribution", "synthetic", d, v, m, se, "note")
    assert block["n_kept"] == 2
    assert len(block["dropped_bins"]) == 1
    assert block["dropped_bins"][0]["index"] == 2
    # the differences -0.5 and 1.0 sit inside the 2 x sqrt(0.51) ~ 1.429
    # bound: no beyond-bound bin
    assert block["beyond_2x_se_count"] == 0
    assert block["house_form_count"] == 0
    assert block["per_bin"][0]["beyond_2x_se"] is False
    assert block["per_bin"][1]["beyond_2x_se"] is False
    # a planted beyond-bound bin: the difference 3.0 exceeds 2 x 0.714,
    # its sigma 4.2 exceeds the 3.0 house tolerance; the -0.5 bin stays
    # inside both bounds
    m2 = np.array([1.5, -1.0])
    block2 = r55._family_block(
        "k2", "distribution", "synthetic", d[:2], v[:2], m2, se[:2], "n")
    assert block2["beyond_2x_se_count"] == 1
    assert block2["house_form_count"] == 1
    assert block2["per_bin"][0]["beyond_2x_se"] is False
    assert block2["per_bin"][1]["beyond_2x_se"] is True
    assert block2["per_bin"][1]["house_form"] is True
    assert block2["per_bin"][0]["sigma"] == pytest.approx(
        -0.5 / math.sqrt(0.51), rel=1e-12)
    assert block2["per_bin"][1]["sigma"] == pytest.approx(
        3.0 / math.sqrt(0.51), rel=1e-12)


def test_mixture_reference_weights():
    """The reference mixture: the values are the weight-weighted
    per-class fractions (normalized over the subset), the variances the
    weight-squared binomial diagonals; the 0.25-each weights reproduce
    the R51 reference construction."""
    per_class = {
        "sd_projectile": {"hist": [10, 30], "n_events": 100},
        "sd_target": {"hist": [50, 50], "n_events": 100},
        "dd": {"hist": [0, 100], "n_events": 100},
        "nd": {"hist": [25, 75], "n_events": 100},
    }
    values, variances, weights = r55._mixture_reference(
        per_class, tuple(per_class),
        {c: 0.25 for c in per_class}, "hist")
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert values[0] == pytest.approx(0.25 * 0.1 + 0.25 * 0.5 + 0.25 * 0.0
                                      + 0.25 * 0.25)
    assert values[1] == pytest.approx(0.25 * 0.3 + 0.25 * 0.5
                                      + 0.25 * 1.0 + 0.25 * 0.75)
    # the variance of a degenerate class bin (p = 1) contributes zero
    assert variances[1] == pytest.approx(
        0.25 ** 2 * (0.3 * 0.7 / 100)
        + 0.25 ** 2 * (0.5 * 0.5 / 100)
        + 0.25 ** 2 * (0.75 * 0.25 / 100))
    # the gap subset renormalizes the weights over its classes
    values2, _v2, weights2 = r55._mixture_reference(
        per_class, ("sd_projectile", "sd_target", "dd"),
        {c: 0.25 for c in per_class}, "hist")
    assert abs(sum(weights2.values()) - 1.0) < 1e-12
    assert weights2["dd"] == pytest.approx(1.0 / 3.0)


def test_pinned_class_event_accumulation(smoke_geometry):
    """One pinned class event of every inelastic class runs through the
    live accumulation: conservation holds, the register surfaces fill,
    and the rolling digest is deterministic."""
    catalog = r55._load_catalog()
    fams = r55._load_families()
    sampling = r55.sampling_from_outcomes(
        {"frozen_fractions_after_r52": dict(r55.R51_FROZEN_FRACTIONS)},
        {"status": "R53-PENDING", "pending": True, "retained": False,
         "p_n2": 0.0})
    digests = set()
    for cls in ("sd_projectile", "sd_target", "dd", "nd"):
        subs = derive_substreams(
            r55.SMOKE_SEED, 0, r55.SMOKE_EXERCISE_INDEX_BASE + 24)
        rngs = {
            label: np.random.default_rng(subs.stream_for(label))
            for label in ("topology_selection", "spectral_state",
                          "exchange_kinematics", "decay_channel",
                          "decay_kinematics")
        }
        event = run_topology(
            cls, sampling["params"], catalog, fams,
            rng_topology_selection=rngs["topology_selection"],
            rng_spectral_state=rngs["spectral_state"],
            rng_exchange_kinematics=rngs["exchange_kinematics"],
            rng_decay_channel=rngs["decay_channel"],
            rng_decay_kinematics=rngs["decay_kinematics"],
            p_n2=sampling["params"].p_n2)
        conservation = r51.event_conservation(event)
        assert conservation["bqs"] == [2, 2, 0]
        obs = r55.extract_observables(event, r55.SQRT_S)
        acc = r55._ReplicaAccumulator(smoke_geometry)
        r55.accumulate_live_event(acc, event, obs, 24)
        assert acc.gap["events_with_field"] + \
            acc.gap["events_without_field"] == 1
        assert sum(acc.nch) == 1
        assert sum(acc.spec_mult["proton"]) >= 0
        digests.add(acc.event_rolling_digest())
        # determinism: the same event re-accumulates to the same digest
        acc2 = r55._ReplicaAccumulator(smoke_geometry)
        r55.accumulate_live_event(acc2, event, obs, 24)
        assert acc2.event_rolling_digest() == \
            acc.event_rolling_digest()
    assert len(digests) == 4


# ---------------------------------------------------------------------------
# The hermetic smoke (the end-to-end driver validation; never sealed)
# ---------------------------------------------------------------------------
def test_smoke_end_to_end(tmp_path):
    """The smoke mode: 2,000 mixed-draw events plus the six-class
    pinned dispatch exercise, zero conservation violations, and
    byte-identical artifacts across two invocations; the sealing modes
    refuse the smoke artifacts."""
    run_dir_a = tmp_path / "smoke_a"
    run_dir_b = tmp_path / "smoke_b"
    result_a = r55.run_smoke(run_dir_a)
    result_b = r55.run_smoke(run_dir_b)
    assert result_a == result_b
    assert result_a["events"] == 2_000
    # the mixed draw at the frozen fractions (degenerate at dd)
    assert result_a["census"].get("elastic", 0) == 0
    assert sum(result_a["census"].values()) == 2_000
    # every class exercised
    assert set(result_a["exercise_classes"]) == {
        "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}
    for name in ("smoke_pre_registration.json", "smoke_replica.json",
                 "smoke_class_exercise.json",
                 "smoke_validation_report.json"):
        a = _sha256_file(run_dir_a / name)
        b = _sha256_file(run_dir_b / name)
        assert a == b, name
    # the smoke artifacts are refused by the sealing modes
    with pytest.raises(SystemExit):
        r55.run_seal(run_dir_a)
    with pytest.raises(SystemExit):
        r55.run_gate(run_dir_a)
    with pytest.raises(SystemExit):
        r55.run_clean_validate(run_dir_a)


def test_smoke_validation_report_structure(tmp_path):
    """The smoke validation report: the four families present, the
    self-consistency deviances exactly zero on the reference-side
    families, the conservation block, and the no-optimizer record; the
    report carries no wall-clock field."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "smoke"
        r55.run_smoke(run_dir)
        report = json.loads(
            (run_dir / "smoke_validation_report.json").read_text(
                encoding="utf-8"))
        assert report["run"] == "R55"
        assert report["mode"] == "smoke"
        obs = report["validation_observables"]
        assert set(obs["nonstrange_spectra"]) == {
            "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton"}
        assert set(obs["leading_baryon_xf"]) == {"proton", "antiproton"}
        assert obs["gap_size_distribution"]["n_bins"] == 50
        assert obs["multiplicity_charged"]["n_bins"] == 19
        assert set(obs["multiplicity_species"]) == set(
            r55.SPECIES_MULT_ORDER)
        # the reference-side families: the smoke's self-consistency side
        assert obs["gap_size_distribution"]["deviance"] == 0.0
        assert obs["multiplicity_charged"]["deviance"] == 0.0
        for sp in ("proton", "antiproton"):
            assert obs["leading_baryon_xf"][sp]["deviance"] == 0.0
        for sp in r55.SPECIES_MULT_ORDER:
            assert obs["multiplicity_species"][sp]["deviance"] == 0.0
        # the pi0 model-only spectrum carries no deviance
        pi0 = obs["nonstrange_spectra"]["pi_zero"]
        assert pi0["deviance"] is None and pi0["d"] is None
        assert pi0["data_source"] == r55.PI_ZERO_GEOMETRY_NOTE
        assert report["conservation"]["violations"] == 0
        assert report["conservation"]["tolerance"] == 1e-9
        assert report["conservation"]["b_q_s_census"] == {
            "2,2,0": report["conservation"]["events_checked"]}
        assert report["optimizer_invoked"] is False
        assert report["optimizer_uses"] == 0
        assert report["frozen_point"]["refit"] is False
        assert report["frozen_point"]["fractions"]["nd_multi"] == 0.0
        assert "digest" in report
        assert "recorded_utc" not in report


def test_smoke_refuses_default_production_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into it)."""
    with pytest.raises(SystemExit):
        r55.run_smoke(r55.MONO_ROOT / r55.DEFAULT_RUN_DIR)


# ---------------------------------------------------------------------------
# The dual-tree byte-identity (the driver in both trees)
# ---------------------------------------------------------------------------
def test_driver_dual_tree_byte_identical():
    nested = r55.MONO_ROOT / r55.DRIVER_REL_NESTED
    outer = r55.MONO_ROOT / r55.DRIVER_REL_OUTER
    assert nested.is_file() and outer.is_file()
    assert _sha256_file(nested) == _sha256_file(outer)
    assert r55.DRIVER_SHA256 == _sha256_file(nested)


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
