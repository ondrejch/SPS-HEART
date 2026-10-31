"""M10 helper: committed behavior of the mass-law comparison machinery.

Pins the pure, data-independent surface of
``helpers/benchmarking/staged_mass_law_compare.py`` (the M10 mass-law
comparison helper, card TASK-20260905-physics-review; this monorepo copy
is byte-identical to the vendored mirror at dev/mixing2006, sha256
6477bfe585f4db46abc114eb2ad371bf669d088653271eefa955e9b91e06d921):

- the tree layout and the mirrored helper file (byte identity and sha256,
  the layout-sync rule);
- the arm parameter set: the arm's decay hypothesis is applied to the
  frozen continuum before the frozen-record snapshot, so
  frozen_record.continuum.decay_hypothesis equals the arm's hypothesis for
  every (law, hypothesis) arm (the H3 arms carry the dataclass default);
- the pre-registered protocol constants as literals (the form of record,
  the split windows and bin counts, the M8 parameter set, free
  parameters, bounds, starting points, the main and H1 optimizer caps,
  the H1 menu, the seed grid and arm ordering, the M8 record sha256, the
  verdict rule, the H1 caveat);
- the train/held-out split machinery on synthetic NA49-format tables (the
  strict xF-center < 0.2 rule, the bin centered exactly at 0.2 held out,
  and the production bin-count pin refusing a non-production split);
- the verdict machinery: the branch boundaries exactly at and just
  beyond the combined band, the band/combined-band arithmetic
  hand-computed, and the production comparison re-derived from the
  recorded law values;
- the pre-registration cross-check: it passes on the run-record values
  (with the recorded config canonical hash) and refuses seven distinct
  tampers (the form choice, the split boundary, the seed base, the main
  cap, the H1 menu, k_replicas, and an arm seed);
- the H1 declared-reduction bookkeeping: the MC-sigma scaling factor
  sqrt(2000/50) = 6.324555320336759 at the chosen E1 = 50, the caveat of
  record, and the refusal of an H1 arm whose chosen_E1 is undecided or
  off-menu (the refusal fires before any table I/O);
- the seed derivation (base 20320831 + 10 per arm, replicas seed + 1 ..
  seed + 8) with the M9 non-collision (the M9 streams end at the
  bootstrap seed 20312831);
- the run-arm bookkeeping with a stubbed objective (no events are
  generated): the best-of-trace convention (the arm reports at the best
  evaluation of the whole trace, not the last) and the cap-hit record (an
  H1 arm resumed with an exhausted 150-evaluation budget records
  success = False with the cap-hit message, distinct from the converged
  success = True record);
- the held-out metric edge case: an empty held-out set returns the
  defined zero metric (the sps_core degenerate convention; the helper
  does not refuse at the metric level, and the production path is
  guarded upstream by the bin-count pin).

Data-independent and unmarked: no NA49 tables are read, no events are
generated, and no network is touched. The run-record numbers (the law
values, the comparison, the config canonical hash, the seeds, the bin
counts) are pinned as hermetic constants; their source is
dev/00runs/mass_law_compare_2026-09-07/ (pre_registration.json and
mass_law_compare_report.json), which the test does not read.

Pre-registered tolerances (one rationale each):
- closed-form float agreement (the band arithmetic, the production
  re-derivation): absolute 1e-12; the expressions are algebraically
  identical and the difference is pure double rounding.
- exact integer / equality pins (counts, seeds, split assignment, the
  verdict boundary, the sigma factor, the stubbed run-arm bookkeeping):
  no tolerance; the math is exact in double precision and the stubs are
  deterministic quadratic objectives.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.benchmarking import staged_mass_law_compare as h  # noqa: E402

MIXING2006 = REPO_ROOT / "dev" / "mixing2006"
BUNDLED_CONFIG = MIXING2006 / "config" / "staged_excitation.json"
HELPER_PATH = REPO_ROOT / "helpers" / "benchmarking" / "staged_mass_law_compare.py"
VENDORED_PATH = (MIXING2006 / "helpers" / "benchmarking"
                 / "staged_mass_law_compare.py")

HELPER_SHA256 = (
    "6477bfe585f4db46abc114eb2ad371bf669d088653271eefa955e9b91e06d921"
)
TOL_CLOSED_FORM = 1e-12

PI_MASS_GEV = 0.1395699
SYNTH_XF = (0.10, 0.199, 0.200, 0.201, 0.30)
SYNTH_PT = (0.30, 0.50, 0.70, 0.90, 1.10)

CONFIG_CANONICAL_SHA256 = (
    "3a83d7c8817ef934656c5b45d27d7b9da53af7b51152fab29b4aafa497c7bfee"
)
M8_RECORD_SHA256 = (
    "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65"
)

# The M9 seed block of record (source: the M9 pre-registration seed
# protocol, pinned by helpers/benchmarking/tests/
# test_staged_heldout_correlations.py): the production arm seed, the 16
# shuffle seeds, and the bootstrap seed. The highest consumed M9 seed is
# the bootstrap seed.
M9_PRODUCTION_SEEDS = ((20310831,) + tuple(range(20310832, 20310848))
                       + (20312831,))
M9_HIGHEST_CONSUMED_SEED = 20312831

# The production split of record (pre-registration split_of_record).
PRODUCTION_SPLIT_TOTAL = {"train": 676, "heldout": 341, "sum": 1017}
PRODUCTION_SPLIT_PER_SPECIES = {
    "pi_minus": {"total": 229, "train": 158, "heldout": 71},
    "pi_plus": {"total": 230, "train": 158, "heldout": 72},
    "k_plus": {"total": 153, "train": 105, "heldout": 48},
    "k_minus": {"total": 148, "train": 105, "heldout": 43},
    "proton": {"total": 135, "train": 75, "heldout": 60},
    "antiproton": {"total": 122, "train": 75, "heldout": 47},
}

# The per-hypothesis pooled held-out WRRMSE of record (the merged
# artifact law_metrics.values).
PRODUCTION_LAW_VALUES = {
    "regge": {"H1": 1.9942530538358547, "H2": 1.2604935086628049,
              "H3": 1.047704796409968},
    "simple": {"H1": 1.848541320852332, "H2": 0.8301533535849469,
               "H3": 0.8380219363648985},
}
PRODUCTION_COMPARISON = {
    "verdict": "not_distinguished",
    "delta": -0.20968286004506942,
    "band_regge": 0.9465482574258868,
    "band_simple": 1.018387967267385,
    "combined_band": 1.3903480339508447,
}
PRODUCTION_DIAGNOSTIC_BANDS = {
    "band_regge": 0.2127887122528369,
    "band_simple": 0.007868582779951616,
}

VERDICT_STATEMENTS = {
    "prefer_regge": (
        "delta > combined band: the data prefer the Regge-informed mass law"
    ),
    "prefer_simple": (
        "delta < -combined band: the data prefer the flexible empirical "
        "mass law"
    ),
    "not_distinguished": (
        "|delta| <= combined band: the data do not distinguish the two "
        "mass laws within the bands"
    ),
}

H1_CAVEAT_TEXT = (
    "the H1 arms enter only the cross-hypothesis spread, never the primary "
    "verdict at H3. The H1 replica-mean MC sigma is sqrt(2000/E1) times the "
    "M8-statistics value (sigma_MC scales as 1/sqrt(events x K)), and the H1 "
    "fitted values are 2-iteration descents from the starting point, not "
    "converged fits (the M7 H1-cap honesty convention). Both caveats are "
    "quoted with the spread."
)
CAP_HIT_MESSAGE = (
    "resumed with exhausted evaluation budget; best of prior trace"
)


def _load_config() -> dict:
    return json.loads(BUNDLED_CONFIG.read_text())


def _write_table(path: Path, rows) -> None:
    """A synthetic NA49-format table (4 columns: xF pT value error).

    xF is written at 4 decimals so the 0.199/0.200/0.201 split-boundary
    centers survive the write/read round trip.
    """
    lines = ["# synthetic M10 table (4 columns: xF pT value error)"]
    for xf, pt, value, error in rows:
        lines.append(f"{xf:.4f} {pt:.2f} {value!r} {error!r}")
    path.write_text("\n".join(lines) + "\n")


def _scratch_config(tmp_path: Path, tables: list[dict]) -> dict:
    config = _load_config()
    config["fit"]["binned"]["path_root"] = str(tmp_path)
    config["fit"]["binned"]["tables"] = tables
    return config


def _two_species_config(tmp_path: Path) -> dict:
    """Two synthetic species on the five split-boundary xF centers."""
    rows = [
        (xf, pt, 1.0e-2 + 1.0e-3 * i, 1.0e-3)
        for i, (xf, pt) in enumerate(zip(SYNTH_XF, SYNTH_PT))
    ]
    _write_table(tmp_path / "synth_pi_minus.txt", rows)
    _write_table(tmp_path / "synth_pi_plus.txt", rows)
    tables = [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_pi_minus.txt", "normalization": "f"},
        {"species": "pi_plus", "pdg": 211, "mass_gev": PI_MASS_GEV,
         "path": "synth_pi_plus.txt", "normalization": "f"},
    ]
    return _scratch_config(tmp_path, tables)


def _prereg_record() -> dict:
    """The pre-registration record rebuilt from the harness constants."""
    return {
        "form_of_record": {"choice": h.FORM_CHOICE},
        "split_of_record": {
            "train_window": dict(h.TRAIN_WINDOW),
            "heldout_window": dict(h.HELDOUT_WINDOW),
            "bin_counts_measured": {
                "per_species": {k: dict(v) for k, v in h.BIN_COUNTS.items()},
                "total": dict(h.BIN_COUNTS_TOTAL),
            },
        },
        "identical_protocol": {
            "m8_fitted_parameters": {"values": dict(h.M8_PARAMETER_SET)},
            "frozen_continuum_m8_values": {
                "values": {k: v for k, v in h.M8_PARAMETER_SET.items()
                           if k.startswith("continuum.")}
            },
            "free_parameters": {k: list(v) for k, v in h.FREE_PARAMETERS.items()},
            "bounds": {k: {kk: list(vv) for kk, vv in v.items()}
                       for k, v in h.BOUNDS.items()},
            "starting_points": {k: dict(v)
                                for k, v in h.STARTING_POINTS.items()},
            "events_per_replica_main": h.EVENTS_PER_REPLICA_MAIN,
            "k_replicas": h.K_REPLICAS,
            "optimizer": {"xtol": h.XTOL, "ftol": h.FTOL,
                          "cap_max_iterations_main": h.CAP_MAX_ITERATIONS_MAIN},
        },
        "seed_protocol": {
            "m10_seed_base": h.M10_SEED_BASE,
            "stage_label": h.STAGE_LABEL,
            "arm_seeds": dict(h.ARM_SEEDS),
            "replica_seed_blocks": {k: list(v)
                                    for k, v in h.REPLICA_SEED_BLOCKS.items()},
        },
        "arm_grid": {"ordering": [dict(a) for a in h.ARM_ORDERING]},
        "h1_declared_reduction": {
            "chosen_E1": 50,
            "menu_events_per_replica": list(h.H1_MENU_EVENTS_PER_REPLICA),
            "max_iterations": h.H1_MAX_ITERATIONS,
            "max_evaluations": h.H1_MAX_EVALUATIONS,
            "max_projected_arm_wall_s": h.H1_MAX_PROJECTED_ARM_WALL_S,
        },
        "verdict_rule": dict(h.VERDICT_RULE),
        "provenance": {
            "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
            "m8_record_sha256": h.M8_RECORD_SHA256,
        },
    }


def _write_trace(trace_dir: Path, arm_name: str, entries: list[dict]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trace_{arm_name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n"
    )
    return path


def _stub_objective(target: list[float]):
    """A deterministic quadratic objective standing in for evaluate_point."""

    def stub(x, arm, tables, train_masks, n_events, replica_seeds, beam):
        xv = [float(v) for v in np.asarray(x, dtype=float)]
        chi2 = float(sum((a - b) ** 2 for a, b in zip(xv, target)) + 1.0)
        return {
            "chi2": chi2,
            "per_species_train_chi2": {t.species: chi2 / 2.0 for t in tables},
            "max_conservation_residual": 0.0,
            "_m_mean": {t.species: np.full(t.data.size, 0.5) for t in tables},
            "_sigma_mc": {t.species: np.zeros(t.data.size) for t in tables},
            "_sigma_replica": {t.species: np.zeros(t.data.size)
                               for t in tables},
        }

    return stub


def test_m10_helper_layout_and_byte_identity():
    """The monorepo copy resolves the monorepo tree and pins byte identity.

    The two helper trees must carry the same file (the layout-sync rule):
    same sha256, same bytes, and the monorepo copy resolves its own tree
    (MIXING2006_DIR = dev/mixing2006, RUNS_ROOT = dev/00runs).
    """
    assert h.__file__ == str(HELPER_PATH)
    assert h.MIXING2006_DIR == MIXING2006
    assert h.RUNS_ROOT == REPO_ROOT / "dev" / "00runs"
    assert h._sha256_file(HELPER_PATH) == HELPER_SHA256
    assert h._sha256_file(VENDORED_PATH) == HELPER_SHA256
    assert HELPER_PATH.read_bytes() == VENDORED_PATH.read_bytes()


def test_m10_arm_frozen_record_carries_arm_decay_hypothesis():
    """The frozen record reports the kernel each arm actually runs.

    Regression pin for the round-1 fix (REV-0092af6-01): the arm's decay
    hypothesis is applied to the frozen continuum before the frozen_record
    snapshot, so for every (law, hypothesis) arm the frozen record's
    continuum carries the arm's hypothesis; for the H3 arms it is "H3",
    which is also the ContinuumParameters dataclass default. The frozen
    record is therefore self-sufficient per arm. This needs a full
    bundled-config load (a hermetic JSON read, no event generation):
    arm_parameters builds the base parameter set from the config.
    """
    config = _load_config()
    for law in ("regge", "simple"):
        for hypothesis in ("H1", "H2", "H3"):
            payload = h.arm_parameters(law, hypothesis, config)
            assert payload["hypothesis"] == hypothesis
            assert payload["continuum"].decay_hypothesis == hypothesis
            assert payload["frozen_record"]["continuum"]["decay_hypothesis"] \
                == hypothesis


def test_m10_protocol_constants_pinned():
    """The pre-registered protocol constants, as literals of record."""
    assert h.CARD == "TASK-20260905-physics-review"
    assert h.MILESTONE == "M10"
    assert h.PHASE == "A"
    assert h.FORM_CHOICE == "form2_predictive_validation"
    assert h.SPLIT_XF == 0.2
    assert h.TRAIN_WINDOW == {"xf_min": 0.0, "xf_max": 0.2,
                              "pt_min": 0.05, "pt_max": 1.5}
    assert h.HELDOUT_WINDOW == {"xf_min": 0.2, "xf_max": 0.4,
                                "pt_min": 0.05, "pt_max": 1.5}
    assert h.BIN_COUNTS_TOTAL == PRODUCTION_SPLIT_TOTAL
    assert h.BIN_COUNTS == PRODUCTION_SPLIT_PER_SPECIES
    assert h.SPECIES_ORDER == ("pi_minus", "pi_plus", "k_plus", "k_minus",
                               "proton", "antiproton")
    assert h.M8_PARAMETER_SET == {
        "regge.alpha_exc0": 1.1418864566363858,
        "regge.effective_transfer_slope": 8.504140762083612,
        "regge.edge_power": 2.3769493784466396,
        "continuum.match_mass": 2.680194110727528,
        "continuum.pion_multiplicity_slope": 1.5226475471033067,
        "continuum.strange_lambda_strength": 0.30184606498191385,
        "continuum.strange_sigma_strength": 0.3777282734799746,
        "continuum.strange_pair_strength": 0.19187855210738686,
        "continuum.pair_strength": 0.08411858717003319,
        "continuum.pair_threshold_power": 1.7610429093564994,
    }
    assert h.FREE_PARAMETERS == {
        "regge": ["regge.alpha_exc0", "regge.effective_transfer_slope",
                  "regge.edge_power"],
        "simple": ["simple.mass_scale", "simple.edge_power",
                   "simple.transfer_slope"],
    }
    assert h.BOUNDS == {
        "regge": {"regge.alpha_exc0": [0.0, 1.2],
                  "regge.effective_transfer_slope": [5.07, 10.2],
                  "regge.edge_power": [0.25, 6.0]},
        "simple": {"simple.mass_scale": [0.1, 5.0],
                   "simple.edge_power": [0.25, 6.0],
                   "simple.transfer_slope": [5.07, 10.2]},
    }
    assert h.STARTING_POINTS == {
        "regge": {"regge.alpha_exc0": 0.5,
                  "regge.effective_transfer_slope": 7.3489827740435505,
                  "regge.edge_power": 2.0},
        "simple": {"simple.mass_scale": 0.65,
                   "simple.edge_power": 1.5,
                   "simple.transfer_slope": 5.5},
    }
    assert h.EVENTS_PER_REPLICA_MAIN == 2000
    assert h.K_REPLICAS == 8
    assert h.XTOL == 1e-3 and h.FTOL == 1e-3
    assert h.CAP_MAX_ITERATIONS_MAIN == 10
    assert h.N_PARAMS_FIT == 3
    assert h.H1_MENU_EVENTS_PER_REPLICA == [200, 100, 50]
    assert h.H1_MAX_ITERATIONS == 2
    assert h.H1_MAX_EVALUATIONS == 150
    assert h.H1_MAX_PROJECTED_ARM_WALL_S == 18000.0
    assert h.M10_SEED_BASE == 20320831
    assert h.ARM_SEED_STRIDE == 10
    assert h.REPLICA_SEED_OFFSET == 1
    assert h.ARM_ORDERING == [
        {"a": 0, "law": "regge", "hypothesis": "H1"},
        {"a": 1, "law": "regge", "hypothesis": "H2"},
        {"a": 2, "law": "regge", "hypothesis": "H3"},
        {"a": 3, "law": "simple", "hypothesis": "H1"},
        {"a": 4, "law": "simple", "hypothesis": "H2"},
        {"a": 5, "law": "simple", "hypothesis": "H3"},
    ]
    assert h.ARM_SEEDS == {
        "regge_H1": 20320831,
        "regge_H2": 20320841,
        "regge_H3": 20320851,
        "simple_H1": 20320861,
        "simple_H2": 20320871,
        "simple_H3": 20320881,
    }
    assert h.REPLICA_SEED_BLOCKS == {
        "regge_H1": [20320832, 20320839],
        "regge_H2": [20320842, 20320849],
        "regge_H3": [20320852, 20320859],
        "simple_H1": [20320862, 20320869],
        "simple_H2": [20320872, 20320879],
        "simple_H3": [20320882, 20320889],
    }
    assert h.STAGE_LABEL == "mass_law_compare_replica"
    assert h.CONFIG_SEED == 20260831
    assert h.BEAM_MOMENTUM_GEV_C == 158.0
    assert h.SQRT_S_GEV == 17.270205201372406
    assert h.M8_RECORD_SHA256 == M8_RECORD_SHA256
    # The verdict rule is written before fitting and pinned as a literal.
    assert h.VERDICT_RULE["written_before_fitting"] is True
    for key, text in VERDICT_STATEMENTS.items():
        assert h.VERDICT_RULE[key] == text
    assert h.H1_CAVEAT_TEXT == H1_CAVEAT_TEXT


def test_m10_split_mask_assigns_train_heldout_by_xf_center(tmp_path):
    """The split is the strict xF-center < 0.2 rule, per table."""
    config = _two_species_config(tmp_path)
    tables = h.build_tables(config)
    assert [t.species for t in tables] == ["pi_minus", "pi_plus"]
    train_masks, heldout_masks = h.split_masks(tables)
    for table, train, heldout in zip(tables, train_masks, heldout_masks):
        assert table.data.size == 5
        assert table.xf_centers.tolist() == [0.10, 0.199, 0.2, 0.201, 0.3]
        assert train.tolist() == [True, True, False, False, False]
        assert heldout.tolist() == [False, False, True, True, True]
        assert int(train.sum()) == 2 and int(heldout.sum()) == 3
        for i in range(table.data.size):
            assert bool(train[i]) == (table.xf_centers[i] < 0.2)
    assert sum(int(m.sum()) for m in train_masks) == 4
    assert sum(int(m.sum()) for m in heldout_masks) == 6


def test_m10_split_boundary_bin_at_0_2_is_heldout(tmp_path):
    """A bin centered exactly at xF = 0.2 is held out (the train side is strict).

    On the xF centers 0.199, 0.200, 0.201: 0.199 is train, and the bin
    centered exactly at 0.2 (and everything at or above it) is held out.
    """
    rows = [(xf, 0.50, 1.0e-2, 1.0e-3) for xf in (0.199, 0.200, 0.201)]
    _write_table(tmp_path / "synth_boundary.txt", rows)
    config = _scratch_config(tmp_path, [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_boundary.txt", "normalization": "f"},
    ])
    tables = h.build_tables(config)
    train_masks, heldout_masks = h.split_masks(tables)
    assert train_masks[0].tolist() == [True, False, False]
    assert heldout_masks[0].tolist() == [False, True, True]


def test_m10_bin_count_pin_refuses_non_production_split(tmp_path):
    """The pre-registered bin-count pin refuses any non-production split."""
    config = _two_species_config(tmp_path)
    tables = h.build_tables(config)
    train_masks, heldout_masks = h.split_masks(tables)
    with pytest.raises(SystemExit,
                       match="do not match the pre-registered pin"):
        h._assert_bin_counts(tables, train_masks, heldout_masks)


def test_m10_verdict_rule_branches_and_boundaries():
    """The verdict branches exactly at and just beyond the combined band.

    At the boundary one law is flat (band 0) and the other spans the band
    with H3 interior to its range, so a 1e-9 perturbation of H3 moves the
    delta without moving the band.
    """
    def verdict_for(regge, simple):
        return h.compare_laws({"regge": regge, "simple": simple})

    at_plus = verdict_for(
        {"H1": 1.0, "H2": 1.0, "H3": 1.0},
        {"H1": 1.5, "H2": 3.5, "H3": 3.0})
    assert at_plus["delta"] == 2.0 and at_plus["combined_band"] == 2.0
    assert at_plus["verdict"] == "not_distinguished"
    assert at_plus["verdict_statement"] == VERDICT_STATEMENTS[
        "not_distinguished"]
    above = verdict_for(
        {"H1": 1.0, "H2": 1.0, "H3": 1.0},
        {"H1": 1.5, "H2": 3.5, "H3": 3.0 + 1e-9})
    assert abs(above["delta"] - (2.0 + 1e-9)) < TOL_CLOSED_FORM
    assert above["combined_band"] == 2.0
    assert above["verdict"] == "prefer_regge"
    assert above["verdict_statement"] == VERDICT_STATEMENTS["prefer_regge"]
    at_minus = verdict_for(
        {"H1": 1.5, "H2": 3.5, "H3": 3.0},
        {"H1": 1.0, "H2": 1.0, "H3": 1.0})
    assert at_minus["delta"] == -2.0 and at_minus["combined_band"] == 2.0
    assert at_minus["verdict"] == "not_distinguished"
    below = verdict_for(
        {"H1": 1.5, "H2": 3.5, "H3": 3.0 + 1e-9},
        {"H1": 1.0, "H2": 1.0, "H3": 1.0})
    assert below["verdict"] == "prefer_simple"
    assert below["verdict_statement"] == VERDICT_STATEMENTS["prefer_simple"]


def test_m10_band_arithmetic_hand_computed():
    """The band/combined-band arithmetic re-derived by hand in the test."""
    v = h.compare_laws({
        "regge": {"H1": 0.5, "H2": 2.5, "H3": 1.5},
        "simple": {"H1": 1.25, "H2": 0.75, "H3": 2.75},
    })
    assert abs(v["band_regge"] - (2.5 - 0.5)) < TOL_CLOSED_FORM
    assert abs(v["band_simple"] - (2.75 - 0.75)) < TOL_CLOSED_FORM
    assert abs(v["combined_band"] - math.sqrt(2.0 ** 2 + 2.0 ** 2)) \
        < TOL_CLOSED_FORM
    # delta = M_simple(H3) - M_regge(H3) = 2.75 - 1.5.
    assert abs(v["delta"] - (2.75 - 1.5)) < TOL_CLOSED_FORM
    assert v["verdict"] == "not_distinguished"
    # The identical-statistics diagnostic uses the H2/H3 range only.
    assert abs(v["diagnostic_h2h3_only"]["band_regge"] - (2.5 - 1.5)) \
        < TOL_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_simple"] - (2.75 - 0.75)) \
        < TOL_CLOSED_FORM
    assert v["h1_caveat"] == H1_CAVEAT_TEXT
    assert v["metric"] == h.VERDICT_RULE["primary_metric"]


def test_m10_production_comparison_rederived_from_law_values():
    """The recorded production comparison re-derived from the law values."""
    lv = PRODUCTION_LAW_VALUES
    v = h.compare_laws(lv)
    rec = PRODUCTION_COMPARISON
    assert v["verdict"] == rec["verdict"]
    assert abs(v["delta"] - rec["delta"]) < TOL_CLOSED_FORM
    assert abs(v["band_regge"] - rec["band_regge"]) < TOL_CLOSED_FORM
    assert abs(v["band_simple"] - rec["band_simple"]) < TOL_CLOSED_FORM
    assert abs(v["combined_band"] - rec["combined_band"]) < TOL_CLOSED_FORM
    assert v["m_regge_h3"] == lv["regge"]["H3"]
    assert v["m_simple_h3"] == lv["simple"]["H3"]
    assert v["verdict_statement"] == VERDICT_STATEMENTS["not_distinguished"]
    # Hand re-derivation from the pinned law values (no helper call).
    assert abs(lv["simple"]["H3"] - lv["regge"]["H3"] - rec["delta"]) \
        < TOL_CLOSED_FORM
    assert abs(max(lv["regge"].values()) - min(lv["regge"].values())
               - rec["band_regge"]) < TOL_CLOSED_FORM
    assert abs(max(lv["simple"].values()) - min(lv["simple"].values())
               - rec["band_simple"]) < TOL_CLOSED_FORM
    hand_combined = math.sqrt(rec["band_regge"] ** 2 + rec["band_simple"] ** 2)
    assert abs(hand_combined - rec["combined_band"]) < TOL_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_regge"]
               - PRODUCTION_DIAGNOSTIC_BANDS["band_regge"]) < TOL_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_simple"]
               - PRODUCTION_DIAGNOSTIC_BANDS["band_simple"]) < TOL_CLOSED_FORM


def test_m10_cross_check_preregistration_passes_on_recorded_values():
    """The cross-check passes on the recorded pre-registration values."""
    config = _load_config()
    assert h.canonical_configuration_hash(config) == CONFIG_CANONICAL_SHA256
    h.cross_check_preregistration(_prereg_record(), config, m8_record=None)


TAMPER_CASES = (
    ("form_of_record",
     lambda p: p["form_of_record"].__setitem__(
         "choice", "form1_penalized_likelihood")),
    ("split boundary",
     lambda p: p["split_of_record"]["train_window"].__setitem__(
         "xf_max", 0.3)),
    ("seed base",
     lambda p: p["seed_protocol"].__setitem__("m10_seed_base", 20320832)),
    ("main cap",
     lambda p: p["identical_protocol"]["optimizer"].__setitem__(
         "cap_max_iterations_main", 8)),
    ("h1 menu",
     lambda p: p["h1_declared_reduction"].__setitem__(
         "menu_events_per_replica", [200, 100, 40])),
    ("k_replicas",
     lambda p: p["identical_protocol"].__setitem__("k_replicas", 7)),
    ("arm seed",
     lambda p: p["seed_protocol"]["arm_seeds"].__setitem__("regge_H3",
                                                          20320852)),
)


@pytest.mark.parametrize("label,mutate", TAMPER_CASES,
                         ids=[c[0] for c in TAMPER_CASES])
def test_m10_cross_check_preregistration_refuses_tamperings(label, mutate):
    """A tampered pre-registration is refused (the negative controls)."""
    config = _load_config()
    prereg = _prereg_record()
    mutate(prereg)
    with pytest.raises(SystemExit, match="pre-registration mismatch"):
        h.cross_check_preregistration(prereg, config, m8_record=None)


def test_m10_h1_sigma_factor_and_caveat():
    """The H1 declared-reduction sigma factor at the chosen E1 = 50."""
    assert 50 in h.H1_MENU_EVENTS_PER_REPLICA
    # The replica-mean MC sigma scales as 1/sqrt(events x K): at E1 = 50
    # against the M8 statistics (2000 events, the same K) the factor is
    # sqrt(2000 / 50), exact in double precision.
    assert math.sqrt(2000 / 50) == 6.324555320336759
    assert h.H1_CAVEAT_TEXT == H1_CAVEAT_TEXT
    assert "sqrt(2000/E1)" in h.H1_CAVEAT_TEXT


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p["h1_declared_reduction"].__setitem__("chosen_E1", None),
        lambda p: p["h1_declared_reduction"].__setitem__("chosen_E1", 250),
    ),
    ids=("missing", "off-menu"),
)
def test_m10_h1_arm_refuses_undecided_reduction(tmp_path, mutate):
    """An H1 arm refuses to run before the declared reduction is decided.

    The refusal fires before any table I/O, so this needs no tables and
    no events.
    """
    config = _two_species_config(tmp_path)
    prereg = _prereg_record()
    mutate(prereg)
    with pytest.raises(SystemExit,
                       match="the H1 declared reduction must be decided"):
        h.run_arm("regge", "H1", tmp_path / "arm_out", config, prereg)


def test_m10_arm_seed_derivation_and_m9_non_collision():
    """The seed grid derivation, and non-collision with the M9 block."""
    base = h.M10_SEED_BASE
    for a, item in enumerate(h.ARM_ORDERING):
        name = f"{item['law']}_{item['hypothesis']}"
        assert item["a"] == a
        assert h.ARM_SEEDS[name] == base + h.ARM_SEED_STRIDE * a
        seed = h.ARM_SEEDS[name]
        # The replica seed block records the first and last of the
        # K = 8 replica seeds seed + 1 .. seed + 8.
        assert h.REPLICA_SEED_BLOCKS[name] == [seed + 1, seed + 8]
        replicas = [seed + h.REPLICA_SEED_OFFSET + k
                    for k in range(h.K_REPLICAS)]
        assert len(set(replicas)) == h.K_REPLICAS
    # The per-arm replica streams are pairwise disjoint (stride 10 > 9).
    all_replicas = []
    for item in h.ARM_ORDERING:
        name = f"{item['law']}_{item['hypothesis']}"
        first = h.REPLICA_SEED_BLOCKS[name][0]
        all_replicas.extend(first + k for k in range(h.K_REPLICAS))
    assert len(set(all_replicas)) == 6 * h.K_REPLICAS
    # Non-collision with the M9 seed block of record: every M10 seed sits
    # strictly above the highest consumed M9 seed (the bootstrap seed).
    m10_seeds = set(h.ARM_SEEDS.values()) | set(all_replicas)
    assert M9_HIGHEST_CONSUMED_SEED < min(m10_seeds)
    assert all(seed < min(m10_seeds) for seed in M9_PRODUCTION_SEEDS)


def test_m10_best_of_trace_reports_at_best_evaluation(tmp_path, monkeypatch):
    """The arm reports at the best evaluation of the whole trace.

    A pre-written 3-entry scratch trace whose minimum is the middle entry
    (not the last) is resumed; the stubbed objective is a deterministic
    quadratic whose minimum is exactly the pre-written best entry. The arm
    must report that entry (best-of-trace convention), with the converged
    success record, the determinism check consistent, and the final
    re-evaluation reproducing the best chi2.
    """
    config = _two_species_config(tmp_path)
    prereg = _prereg_record()
    names = list(h.FREE_PARAMETERS["regge"])
    target = {"regge.alpha_exc0": 1.0,
              "regge.effective_transfer_slope": 8.5,
              "regge.edge_power": 2.4}
    prior = (
        {"regge.alpha_exc0": 0.5,
         "regge.effective_transfer_slope": 7.3489827740435505,
         "regge.edge_power": 2.0},
        target,
        {"regge.alpha_exc0": 0.7,
         "regge.effective_transfer_slope": 8.0,
         "regge.edge_power": 2.2},
    )
    entries = [
        {"parameters": {n: p[n] for n in names},
         "chi2": float(sum((p[n] - target[n]) ** 2 for n in names) + 1.0),
         "per_species": {"pi_minus": 1.0, "pi_plus": 1.0},
         "wall_s": 0.0}
        for p in prior
    ]
    out_dir = tmp_path / "run_arm"
    trace_path = _write_trace(out_dir / "scratch_traces", "regge_H2", entries)
    monkeypatch.setattr(h, "evaluate_point",
                        _stub_objective([target[n] for n in names]))
    monkeypatch.setattr(h, "_assert_bin_counts", lambda *a, **k: None)
    payload = h.run_arm("regge", "H2", out_dir, config, prereg, resume=True)
    fit = payload["fit"]
    assert fit["success"] is True
    assert fit["message"] == "Optimization terminated successfully."
    assert fit["prior_evaluations_from_restarts"] == 3
    assert fit["restarts"] == 1
    assert payload["fitted_parameters"] == target
    assert fit["train_chi2_combined_best"] == 1.0
    assert payload["final_evaluation"]["train_chi2_combined"] == 1.0
    assert fit["determinism_consistent"] is True
    final_trace = [json.loads(line)
                   for line in trace_path.read_text().splitlines()
                   if line.strip()]
    assert len(final_trace) == fit["n_objective_evaluations"] > 3
    # The reported best is the first occurrence of the trace minimum, which
    # is the pre-written entry, not the last evaluation.
    best_idx = min(range(len(final_trace)),
                   key=lambda i: final_trace[i]["chi2"])
    assert best_idx < len(final_trace) - 1
    assert final_trace[best_idx]["parameters"] == target


def test_m10_cap_hit_arm_recorded_distinct_from_converged(tmp_path, monkeypatch):
    """A budget-exhausted H1 arm records the cap-hit record.

    A pre-written 150-entry scratch trace fills the pre-registered H1
    budget (max_evaluations = 150); on resume the remaining budget is
    zero, the optimizer is skipped, and the arm records success = False
    with the cap-hit message, the best-of-trace fitted point, and the
    declared reduction with its caveat. This is distinct from the
    converged success = True record pinned above.
    """
    config = _two_species_config(tmp_path)
    prereg = _prereg_record()
    names = list(h.FREE_PARAMETERS["simple"])
    target = {"simple.mass_scale": 1.0,
              "simple.edge_power": 2.4,
              "simple.transfer_slope": 8.5}

    def chi2_at(p):
        return float(sum((p[n] - target[n]) ** 2 for n in names) + 1.0)

    entries = []
    for k in range(h.H1_MAX_EVALUATIONS):
        if k == 7:
            p = dict(target)
        else:
            p = {"simple.mass_scale": 0.5 + 0.001 * k,
                 "simple.edge_power": 1.5,
                 "simple.transfer_slope": 7.0 + 0.001 * k}
        entries.append({"parameters": {n: p[n] for n in names},
                        "chi2": chi2_at(p),
                        "per_species": {"pi_minus": 1.0, "pi_plus": 1.0},
                        "wall_s": 0.0})
    out_dir = tmp_path / "run_arm_h1"
    _write_trace(out_dir / "scratch_traces", "simple_H1", entries)
    monkeypatch.setattr(h, "evaluate_point",
                        _stub_objective([target[n] for n in names]))
    monkeypatch.setattr(h, "_assert_bin_counts", lambda *a, **k: None)
    payload = h.run_arm("simple", "H1", out_dir, config, prereg, resume=True)
    fit = payload["fit"]
    assert fit["success"] is False
    assert fit["message"] == CAP_HIT_MESSAGE
    assert fit["n_objective_evaluations"] == h.H1_MAX_EVALUATIONS
    assert fit["prior_evaluations_from_restarts"] == h.H1_MAX_EVALUATIONS
    assert payload["fitted_parameters"] == target
    assert fit["train_chi2_combined_best"] == 1.0
    assert payload["final_evaluation"]["train_chi2_combined"] == 1.0
    assert fit["determinism_consistent"] is True
    # The declared reduction is recorded with the caveat of record.
    assert payload["statistics"]["declared_reduction"] == {
        "events_per_replica": 50,
        "max_iterations": h.H1_MAX_ITERATIONS,
        "max_evaluations": h.H1_MAX_EVALUATIONS,
        "caveat": H1_CAVEAT_TEXT,
    }
    assert payload["statistics"]["events_per_replica"] == 50
    assert payload["protocol"]["max_iterations"] == h.H1_MAX_ITERATIONS
    assert payload["protocol"]["max_evaluations"] == h.H1_MAX_EVALUATIONS
    # The simple_H1 seed block of the recorded run.
    assert payload["seed_protocol"]["arm_seed"] == h.ARM_SEEDS["simple_H1"]
    assert payload["seed_protocol"]["replica_seeds"] == [
        h.ARM_SEEDS["simple_H1"] + 1 + k for k in range(h.K_REPLICAS)]


def test_m10_empty_heldout_is_defined_zero_metric(tmp_path):
    """An empty held-out set scores the defined zero metric (no refusal).

    sps_core.relative_rmse_weighted returns 0.0 for empty input (its
    documented degenerate convention); the helper's heldout_metrics
    inherits it. The production path is guarded upstream by the
    _assert_bin_counts refusal pinned in
    test_m10_bin_count_pin_refuses_non_production_split.
    """
    rows = [(xf, pt, 1.0e-2, 1.0e-3)
            for xf, pt in zip((0.05, 0.15), SYNTH_PT[:2])]
    _write_table(tmp_path / "synth_alltrain.txt", rows)
    config = _scratch_config(tmp_path, [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_alltrain.txt", "normalization": "f"},
    ])
    tables = h.build_tables(config)
    train_masks, heldout_masks = h.split_masks(tables)
    assert int(train_masks[0].sum()) == 2
    assert int(heldout_masks[0].sum()) == 0
    m_mean = {t.species: np.ones(t.data.size) for t in tables}
    held = h.heldout_metrics(m_mean, tables, heldout_masks)
    assert held["pooled"] == {"wrrmse": 0.0, "n_bins": 0,
                              "chi2_data_only": 0.0}
    assert held["per_species_wrrmse"] == {"pi_minus": 0.0}
    assert held["per_species_chi2_data_only"] == {"pi_minus": 0.0}
    assert held["per_species_bins"] == {"pi_minus": 0}
    assert all(math.isfinite(v) for v in held["pooled"].values())
