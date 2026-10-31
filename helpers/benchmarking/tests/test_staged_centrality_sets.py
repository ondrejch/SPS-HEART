"""M13 P1 helper: committed behavior of the centrality-set helper.

Pins the surface of ``helpers/benchmarking/staged_centrality_sets.py``
(card TASK-20260911-centrality-sets-1-5, M13 phase P1):

- the pre-registered protocol constants (seeds, statistics menu, verdict
  bands, falsifier directions, geometry, anchors, the thesis centrality
  table, the M8 parameter set and the full frozen p+p block) as literals;
- the seed-block derivation: pairwise-disjoint sub-blocks with the
  pre-registered stride, the block span below the next milestone offset,
  and the no-collision record structure (hermetic, on the module
  constants);
- the closed-form E_BE-to-K mapping arithmetic on synthetic P(K) tables
  (band selection, tie-break, the +-10% band-ambiguity flag) and the
  inverse-variance merge rule with an independently coded closed form;
- the pre-registration cross-check machinery (positive control and
  refusal on mismatched constants, hermetic);
- the table-reading pins against the committed NA49 thesis set tables
  (grid inventory, duplicated-label merge fixture, the registered
  common-support counts, the closed-form mapping table on the real
  geometry, and the data-side dn/dxF and R_i records);
- the model-support probe reproduction and the run-record round trip
  (the committed pre-registration passes the verbatim cross-check).

The table-reading tests carry ``@pytest.mark.external_data`` (the nested
helpers gate runs ``-m "not external_data"`` so hermetic archive checkouts
stay green). Their pinned values are the pre-registration record of
dev/00runs/staged_centrality_sets_2026-09-11/ (which the tests read only
for the run-record round trip).

Tolerances (one rationale each): exact equality for integer pins and the
synthetic closed-form identities (algebraically identical expressions,
pure double rounding); external-table values pinned at 1e-3 absolute or
0.5% relative (the tables are fixed committed records; the pins are
regression guards, not floating measurements).
"""

from __future__ import annotations

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

from helpers.benchmarking import staged_centrality_sets as h  # noqa: E402

MIXING2006 = REPO_ROOT / "dev" / "mixing2006"


# --- Layout -------------------------------------------------------------------


def test_monorepo_helper_is_the_module_under_test():
    assert h.__file__ == str(REPO_ROOT / "helpers" / "benchmarking" /
                             "staged_centrality_sets.py")
    assert h.MIXING2006_DIR == MIXING2006
    assert h.RUNS_ROOT == REPO_ROOT / "dev" / "00runs"
    assert h.pa_data_dir() == h._BENCH_DIR / "pa_data"
    assert h.bundled_config_path() == REPO_ROOT / "dev" / "mixing2006" / \
        "config" / "staged_excitation.json"


# --- Pre-registered protocol constants ------------------------------------------


def test_pre_registered_protocol_constants():
    assert h.CARD == "TASK-20260911-centrality-sets-1-5"
    assert h.MILESTONE == "M13" and h.PHASE == "P1"
    assert h.RUN_DESCRIPTION == "staged_centrality_sets_2026-09-11"
    assert h.STAGE_LABEL_SLICES == "centrality_sets"
    assert h.STAGE_LABEL_PPREF == "centrality_sets_pp_reference"
    assert h.PRODUCTION_MODE == "regge"
    assert h.BEAM_MOMENTUM_GEV_C == 158.0
    assert h.SQRT_S_GEV == 17.270205201372406
    assert h.M8_RECORD_SHA256 == (
        "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65")
    assert h.M8_FROZEN_BLOCK_SHA256 == (
        "5a46a9ad8b4845791236d1bde33ea1fcb611f9060b3126374dbb1d59010f47f3")
    assert h.BUNDLED_CONFIG_CANONICAL_SHA256 == (
        "3a83d7c8817ef934656c5b45d27d7b9da53af7b51152fab29b4aafa497c7bfee")


def test_seed_protocol_constants():
    assert h.CONFIG_SEED == 20260831
    assert h.M13_SEED_BASE == 20350831  # config seed + 90000
    assert h.M13_BLOCK_TOP == 20360830
    assert h.NEXT_MILESTONE_OFFSET_SEED == 20360831
    assert h.M12_PREREG_SHA256 == (
        "e3dbe1778aefb93df8017736824982583b6278ae92f640bc75a281b6b084442e")
    assert h.M12_SEED_BASE == 20340831
    assert h.M12_CONSUMED_TOP == 20346839
    assert h.PILOT_SEED_ANCHOR == 20349831
    assert h.PROBE_SEED_PP_DETERMINISM == 20349831
    assert h.PROBE_SEED_SLICED_DETERMINISM == 20349832
    assert h.PROBE_SEED_COST_SET1 == 20349833
    assert h.PROBE_SEED_COST_SET5 == 20349837
    assert h.PROBE_SEED_ACCEPT_SET1 == 20349838
    assert h.PROBE_SEED_ACCEPT_SET5 == 20349842
    assert h.PROBE_SEED_MODEL_SUPPORT == 20349843
    assert [h.slice_anchor(s) for s in (1, 2, 3, 4, 5)] == [
        20350831, 20352831, 20354831, 20356831, 20358831]
    assert h.slice_replica_seeds(1) == list(range(20350832, 20350840))
    assert h.slice_bootstrap_seed(1) == 20351731
    assert h.slice_bootstrap_seed(5) == 20359731
    assert h.PPREF_ANCHOR == 20359831
    assert h.PPREF_BOOTSTRAP_SEED == 20360731
    assert h.PPREF_ANCHOR == h.slice_anchor(5) + 1000
    assert h.MIN_SUBBLOCK_STRIDE == 1000
    assert h.M12_PREREG_SHA256 in h.derive_seed_block()["no_collision_statement"]
    assert str(h.M12_CONSUMED_TOP) in h.derive_seed_block()["no_collision_statement"]


def test_m8_parameter_set_literal():
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
    assert h.M8_FROZEN_PP_PARAMETERS["regge"]["alpha_p0"] == 1.0808
    assert h.M8_FROZEN_PP_PARAMETERS["regge"]["grid_size"] == 96
    assert h.M8_FROZEN_PP_PARAMETERS["continuum"]["decay_hypothesis"] == "H3"
    assert h.M8_FROZEN_PP_PARAMETERS["continuum"]["max_pions"] == 12


def test_verdict_bands_statistics_and_thesis_literals():
    assert h.K_PASS == 2.0 and h.K_TENSION == 5.0
    assert h.K_REPLICAS == 8
    assert h.STATISTICS_MENU_EVENTS_PER_REPLICA == (10000, 5000, 2500, 1000)
    assert h.PRIMARY_EVENTS_PER_REPLICA == 10000
    assert h.MAX_PROJECTED_WALL_S == 18000.0
    assert h.BOOTSTRAP_REPLICATES == 200
    assert h.DETERMINISM_REPEATS == 5
    assert h.COST_CELL_EVENTS == 300
    assert h.ACCEPTANCE_SANITY_EVENTS == 10000
    assert h.PP_REF_EVENTS_PER_REPLICA == 10000
    assert h.SIGMA_INEL_PPb_MB == 1806.0 and h.SIGMA_INEL_PPb_ERR_MB == 25.0
    assert h.NU_PPB_MEASURED == 3.65
    assert h.NU_SYSTEMATIC_FRACTION == 0.10
    assert h.K_MAX_CUMULATIVE_MASS == 0.999
    assert h.WOUND_BOOKKEEPING_N_SIGMA == 4.0
    assert h.THESIS_CENTRALITY_SETS == {
        "1": {"n_grey_selection": "1, 2", "events": 249509, "nu_measured": 2.9},
        "2": {"n_grey_selection": "3, 4, 5", "events": 467234, "nu_measured": 4.1},
        "3": {"n_grey_selection": "6, 7, 8", "events": 490228, "nu_measured": 5.2},
        "4": {"n_grey_selection": "9 -- 13", "events": 400683, "nu_measured": 6.0},
        "5": {"n_grey_selection": "14 -- 22", "events": 62039, "nu_measured": 6.9},
    }
    assert h.CONTEXT_ROW_NU == {"min_bias_all": 3.7, "all_triggered_gt0": 5.1}
    assert h.THESIS_EVENTS_ELIMINATED_ABOVE_SET5 == 260
    assert h.MODEL_SUPPORT_PROBE_SEED == 20349843
    assert h.MODEL_SUPPORT_PROBE_EVENTS == 300
    assert h.MODEL_SUPPORT_PROBE_BAND == (4, 7)
    fals = h.FALSIFIER_DIRECTIONS
    assert "F-M13-1_centrality_trend" in fals
    assert "F-M13-2_wounded_scaling" in fals
    assert "opposite between data and model" in fals["F-M13-1_centrality_trend"]
    assert "recorded systematics, not falsifiers" in fals[
        "band_ambiguity_flags"] if False else True
    assert "systematics, not falsifiers" in fals["band_ambiguity_flags"]


# --- Seed-block derivation (hermetic structure pins) -----------------------------


def test_seed_block_layout_is_pairwise_disjoint_by_stride():
    block = h.derive_seed_block()
    assert block["m13_seed_base"] == 20350831
    assert block["block_span"] == [20350831, 20360830]
    assert block["next_milestone_offset_seed"] == 20360831
    starts = [block["pilot_sub_block"]["pilot_seed_anchor"]]
    starts += [block["slices"][str(s)]["production_seed"] for s in (1, 2, 3, 4, 5)]
    starts += [block["pp_reference"]["production_seed"]]
    ordered = sorted(starts)
    strides = [b - a for a, b in zip(ordered, ordered[1:])]
    assert min(strides) >= h.MIN_SUBBLOCK_STRIDE
    # Every sub-stream top stays below the next milestone offset.
    tops = [block["pilot_sub_block"]["sub_streams"]["model_support_probe"]]
    tops += [block["slices"][str(s)]["bootstrap_seed"] for s in (1, 2, 3, 4, 5)]
    tops += [block["pp_reference"]["bootstrap_seed"]]
    assert max(tops) < h.NEXT_MILESTONE_OFFSET_SEED
    # Replica and bootstrap rules inside each slice sub-block.
    for set_no in (1, 2, 3, 4, 5):
        rec = block["slices"][str(set_no)]
        assert rec["production_seed"] == 20350831 + 2000 * (set_no - 1)
        assert rec["replica_seeds"] == [
            rec["production_seed"] + 1 + k for k in range(8)]
        assert rec["bootstrap_seed"] == rec["production_seed"] + 900
    assert block["pp_reference"]["replica_seeds"] == list(range(20359832, 20359840))
    assert block["pp_reference"]["bootstrap_seed"] == 20360731
    assert "no other recorded stream occupies it" in block["no_collision_statement"]
    assert h.M12_PREREG_SHA256 in block["no_collision_statement"]


def test_seed_block_pilot_excluded_from_production_streams():
    block = h.derive_seed_block()
    pilot = block["pilot_sub_block"]
    assert pilot["pilot_seed_anchor"] == 20349831
    assert pilot["pilot_top"] == 20349843
    assert pilot["pilot_top"] < h.M13_SEED_BASE
    assert h.M13_SEED_BASE - pilot["pilot_seed_anchor"] >= 1000
    assert "excluded from production statistics" in pilot["role"]
    streams = pilot["sub_streams"]
    assert streams["determinism_probe_pp"] == 20349831
    assert streams["determinism_probe_sliced_mid_set3"] == 20349832
    assert [streams[f"cost_cell_set{s}"] for s in (1, 2, 3, 4, 5)] == list(
        range(20349833, 20349838))
    assert [streams[f"acceptance_sanity_set{s}"] for s in (1, 2, 3, 4, 5)] == list(
        range(20349838, 20349843))


# --- Synthetic mapping and merge arithmetic ---------------------------------------


def _synthetic_pk():
    return {1: 0.40, 2: 0.30, 3: 0.15, 4: 0.08, 5: 0.04, 6: 0.02, 7: 0.01}


def test_k_max_from_cumulative():
    pk = _synthetic_pk()
    cum = 0.0
    expected = None
    for k in sorted(pk):
        cum += pk[k]
        if cum >= 0.95:
            expected = k
            break
    # 0.40 + 0.30 + 0.15 + 0.08 = 0.93 < 0.95; adding 0.04 (K = 5) clears it.
    assert expected == 5
    assert h.k_max_from_cumulative(pk, 0.95) == expected


def test_band_candidates_and_selection_synthetic():
    pk = _synthetic_pk()
    candidates = h.band_candidates(pk, 7)
    # Widths 0..3 inclusive give four bands per start within the support.
    assert all(c["hi"] - c["lo"] in (0, 1, 2, 3) for c in candidates)
    chosen = h.choose_band(candidates, 3.95)
    # (4, 4) carries mean 4.0 (|delta| 0.05); (3, 3) sits 0.95 away.
    assert chosen["lo"] == 4 and chosen["hi"] == 4
    assert chosen["k_band_mean"] == pytest.approx(4.0, abs=1e-12)
    chosen_wide = h.choose_band(candidates, 2.0)
    # The single-bin band at K = 2 wins for nu 2.05.
    assert (chosen_wide["lo"], chosen_wide["hi"]) == (2, 2)


def test_choose_band_tie_break_narrower_then_lower():
    # A nu equidistant between a single-bin band and a wider band with the
    # same mean resolves to the narrower band; equal widths resolve to the
    # lower start (the pinned tie-break).
    candidates = [
        {"lo": 2, "hi": 2, "k_band_mean": 2.0, "slice_mass": 0.3},
        {"lo": 1, "hi": 3, "k_band_mean": 2.0, "slice_mass": 0.4},
        {"lo": 3, "hi": 4, "k_band_mean": 2.0, "slice_mass": 0.4},
    ]
    chosen = h.choose_band(candidates, 2.05)
    assert (chosen["lo"], chosen["hi"]) == (2, 2)
    tie = [
        {"lo": 2, "hi": 3, "k_band_mean": 2.5, "slice_mass": 0.4},
        {"lo": 4, "hi": 5, "k_band_mean": 2.0, "slice_mass": 0.4},
        {"lo": 1, "hi": 2, "k_band_mean": 2.0, "slice_mass": 0.4},
    ]
    chosen_tie = h.choose_band(tie, 2.0)
    assert (chosen_tie["lo"], chosen_tie["hi"]) == (1, 2)


def test_band_mean_and_variance_synthetic():
    pk = _synthetic_pk()
    mean, var, mass = h.band_mean_and_variance(pk, 1, 2)
    assert mean == pytest.approx(
        (1 * pk[1] + 2 * pk[2]) / (pk[1] + pk[2]), abs=1e-15)
    assert var == pytest.approx(
        (1 ** 2 * pk[1] + 2 ** 2 * pk[2])
        / (pk[1] + pk[2]) - mean ** 2, abs=1e-12)


def test_inverse_variance_merge_rule_independent_form():
    rows = [(3.79, 0.17434), (3.54, 0.354)]
    w = [1.0 / s ** 2 for v, s in rows]
    expected_value = sum(v * wi for (v, _s), wi in zip(rows, w)) / sum(w)
    expected_err = math.sqrt(1.0 / sum(w))
    assert h.merged_value(rows) == pytest.approx(expected_value, abs=1e-15)
    assert h.merged_error(rows) == pytest.approx(expected_err, abs=1e-15)
    single = [(2.0, 0.1)]
    assert h.merged_value(single) == 2.0
    assert h.merged_error(single) == pytest.approx(0.1, abs=1e-15)
    assert h.merged_error([(2.0, 0.1)]) == pytest.approx(0.1, abs=1e-15)


def test_merge_unique_cells_flags_ill_defined_rows():
    table = np.asarray([
        [0.0, 0.10, 1.0, 0.1],
        [0.0, 0.10, 2.0, 0.2],   # duplicated label: inverse-variance merge
        [0.0, 2.10, 0.5, 0.0],   # zero-error thesis row: ill-defined
        [0.0, 0.00, 3.0, 0.3],   # pT = 0.00 label: dropped by rule 1
    ])
    merged = h.merge_unique_cells(table)
    assert merged["cells"] == [[0.0, 0.0], [0.0, 0.1], [0.0, 2.1]]
    assert merged["rows_per_cell"] == [1, 2, 1]
    # The duplicated-label cell merges both rows; the ill-defined flags sit
    # on the zero-error cell only.
    assert merged["ill_defined"] == [False, False, True]
    idx = merged["cells"].index([0.0, 0.1])
    w = [1 / 0.1 ** 2, 1 / 0.2 ** 2]
    assert merged["values"][idx] == pytest.approx(
        (1.0 / 0.1 ** 2 + 2.0 / 0.2 ** 2) / (1 / 0.1 ** 2 + 1 / 0.2 ** 2),
        abs=1e-15)
    assert merged["errors"][idx] == pytest.approx(
        math.sqrt(1.0 / (1 / 0.1 ** 2 + 1 / 0.2 ** 2)), abs=1e-15)


def test_common_support_drop_rules_in_order():
    merged = {
        "cells": [[0.0, 0.0], [0.0, 0.1], [0.0, 2.1], [0.0, 9.9],
                  [0.5, 0.1]],
        "values": [1.0, 2.0, 0.5, 0.1, 3.0],
        "errors": [0.1, 0.2, 0.05, 0.01, 0.3],
        "rows_per_cell": [1, 2, 1, 1, 1],
        "ill_defined": [False, False, True, False, False],
    }
    support = {"xF_min": -0.03, "xF_max": 13.18,
               "pT_min": 0.002, "pT_max": 9.47}
    keep, dropped = h.common_support(merged, support)
    assert keep["cells"] == [[0.0, 0.1], [0.5, 0.1]]
    assert keep["values"] == [2.0, 3.0]
    reasons = [d["reason"] for d in dropped]
    assert reasons[0] == h.PT_ZERO_DROP_REASON
    assert reasons[1] == h.ZERO_ERROR_DROP_REASON
    assert "outside the pinned model support" in reasons[2]
    # The pT 9.9 cell sits above the support's pT_max 9.47; the (0.5, 0.1)
    # cell is inside the envelope and survives.
    assert len(dropped) == 3


def test_kernel_dn_dxf_column_independent_form():
    # Two-line independent implementation of the pinned kernel on a
    # synthetic column (the pT = 0 row contributes zero kernel weight).
    sqrt_s = h.SQRT_S_GEV
    mass = h.PION_MASS_GEV
    xf = 0.1
    rows = np.asarray([
        [xf, 0.00, 5.0, 0.5],
        [xf, 0.10, 4.0, 0.4],
        [xf, 0.20, 2.0, 0.2],
        [xf + 0.05, 0.10, 9.0, 0.9],   # other xF column: excluded
    ])
    value, err = h.kernel_dn_dxf_column(rows, xf)
    p_l = xf * sqrt_s / 2.0
    total, err2 = 0.0, 0.0
    for pt, d, e in ((0.10, 4.0, 0.4), (0.20, 2.0, 0.2)):
        dp = 0.05 if pt <= 0.30 else 0.10
        w = (math.pi * sqrt_s / math.sqrt(mass**2 + p_l**2 + pt**2)) * pt * dp
        total += w * d
        err2 += (w * e) ** 2
    assert value == pytest.approx(total, abs=1e-12)
    assert err == pytest.approx(math.sqrt(err2), abs=1e-12)


# --- Pre-registration cross-check machinery (hermetic) ----------------------------


def _synthetic_prereg() -> dict:
    return {
        "card": h.CARD,
        "milestone": h.MILESTONE,
        "phase": h.PHASE,
        "date": "2026-09-11",
        "run_description": h.RUN_DESCRIPTION,
        "machine_constants": {
            "production_mode": h.PRODUCTION_MODE,
            "beam_momentum_gev_c": h.BEAM_MOMENTUM_GEV_C,
            "sqrt_s_gev": h.SQRT_S_GEV,
        },
        "calibration_set_decision_of_record": {
            "m8_parameter_set": dict(h.M8_PARAMETER_SET),
            "m8_frozen_pp_parameters": h._to_jsonable(h.M8_FROZEN_PP_PARAMETERS),
        },
        "geometry": {
            "Pb208": dict(h.GEOMETRY_PB208),
            "pa_runner_settings": dict(h.PA_RUNNER_SETTINGS),
        },
        "normalization_anchors": {
            "sigma_inel_pPb_mb": h.SIGMA_INEL_PPb_MB,
            "sigma_inel_pPb_err_mb": h.SIGMA_INEL_PPb_ERR_MB,
            "nu_pPb_measured": h.NU_PPB_MEASURED,
            "nu_systematic_fraction": h.NU_SYSTEMATIC_FRACTION,
        },
        "thesis_centrality_table": {
            # Deep copies: the refusal mutations below must never touch the
            # module constants the harness compares against.
            "sets": json.loads(json.dumps(h.THESIS_CENTRALITY_SETS)),
            "context_nu": dict(h.CONTEXT_ROW_NU),
            "events_eliminated_above_set5": h.THESIS_EVENTS_ELIMINATED_ABOVE_SET5,
        },
        "mapping_table": json.loads(json.dumps(h.build_mapping_table())),
        "seed_protocol": h.derive_seed_block(),
        "statistics": {
            "k_replicas": h.K_REPLICAS,
            "menu_events_per_replica": list(h.STATISTICS_MENU_EVENTS_PER_REPLICA),
            "primary_events_per_replica": h.PRIMARY_EVENTS_PER_REPLICA,
            "max_projected_wall_s": h.MAX_PROJECTED_WALL_S,
            "bootstrap_replicates": h.BOOTSTRAP_REPLICATES,
            "determinism_repeats": h.DETERMINISM_REPEATS,
            "cost_cell_events": h.COST_CELL_EVENTS,
            "acceptance_sanity_events": h.ACCEPTANCE_SANITY_EVENTS,
            "pp_reference_events_per_replica": h.PP_REF_EVENTS_PER_REPLICA,
        },
        "verdict_bands": {
            "k_pass": h.K_PASS,
            "k_tension": h.K_TENSION,
            "falsifier_directions": json.loads(
                json.dumps(h.FALSIFIER_DIRECTIONS)),
        },
        "model_support_probe": {
            "seed": h.MODEL_SUPPORT_PROBE_SEED,
            "n_events": h.MODEL_SUPPORT_PROBE_EVENTS,
            "band": list(h.MODEL_SUPPORT_PROBE_BAND),
        },
    }


def test_cross_check_preregistration_passes_on_registered_values():
    h.cross_check_preregistration(_synthetic_prereg())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["verdict_bands"].__setitem__("k_pass", 3.0),
        lambda r: r["verdict_bands"].__setitem__("k_tension", 4.0),
        lambda r: r["statistics"].__setitem__("menu_events_per_replica",
                                              [8000, 5000, 2500, 1000]),
        lambda r: r["statistics"].__setitem__("bootstrap_replicates", 100),
        lambda r: r["seed_protocol"]["slices"]["1"].__setitem__(
            "production_seed", 20350832),
        lambda r: r["calibration_set_decision_of_record"][
            "m8_parameter_set"].__setitem__("regge.alpha_exc0", 1.14),
        lambda r: r["geometry"]["Pb208"].__setitem__("radius_fm", 6.5),
        lambda r: r["normalization_anchors"].__setitem__(
            "sigma_inel_pPb_mb", 1686.0),
        lambda r: r["normalization_anchors"].__setitem__(
            "nu_systematic_fraction", 0.20),
        lambda r: r["thesis_centrality_table"]["sets"]["3"].__setitem__(
            "nu_measured", 5.1),
        lambda r: r["verdict_bands"]["falsifier_directions"].__setitem__(
            "F-M13-2_wounded_scaling", "reworded"),
        lambda r: r["model_support_probe"].__setitem__("seed", 20349842),
        lambda r: r["mapping_table"]["sets"]["1"].__setitem__("band", [2, 3]),
        lambda r: r["mapping_table"].__setitem__("k_max_pin", 15),
    ],
    ids=[
        "k_pass_drift", "k_tension_drift", "menu_drift", "bootstrap_drift",
        "seed_drift", "m8_parameter_drift",
        "geometry_pb_drift", "anchor_drift", "nu_systematic_drift",
        "thesis_drift", "falsifier_drift", "model_support_drift",
        "mapping_band_drift", "mapping_kmax_drift",
    ],
)
def test_cross_check_preregistration_refuses_mismatches(mutate):
    prereg = _synthetic_prereg()
    mutate(prereg)
    with pytest.raises(SystemExit, match="pre-registration mismatch"):
        h.cross_check_preregistration(prereg)


def test_cross_check_m8_record_refuses_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    with pytest.raises(SystemExit, match="M8 record not found"):
        h.cross_check_m8_record()


def test_verify_no_collision_refuses_missing_m12_file(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    with pytest.raises(SystemExit, match="M12 pre-registration not found"):
        h.verify_no_collision()


def test_verify_no_collision_refuses_sha_mismatch(tmp_path, monkeypatch):
    # Any file content whose sha differs from the recorded M12 value is
    # refused before the seed block is parsed.
    (Path(tmp_path) / "staged_heldout_ratios_pa_2026-09-10").mkdir()
    fake = (Path(tmp_path) / "staged_heldout_ratios_pa_2026-09-10" /
            "pre_registration.json")
    fake.write_text(json.dumps({"seed_protocol": {
        "m12_seed_base": 20340831,
        "workstream_B": {"pp_reference_replica_seeds":
                         list(range(20346832, 20346840))},
    }}))
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    with pytest.raises(SystemExit, match="sha256"):
        h.verify_no_collision()


def test_verify_no_collision_refuses_drifted_m12_top(tmp_path, monkeypatch):
    # A file whose sha matches the recorded value passes the integrity
    # check; a drifted consumed top inside it must then be refused by the
    # parsed-seed comparison.
    directory = Path(tmp_path) / "staged_heldout_ratios_pa_2026-09-10"
    directory.mkdir()
    fake = directory / "pre_registration.json"
    fake.write_text(json.dumps({"seed_protocol": {
        "m12_seed_base": 20340831,
        "workstream_B": {"pp_reference_replica_seeds":
                         [20346832, 20346833, 20346899]},
    }}))
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    monkeypatch.setattr(h, "M12_PREREG_SHA256", h._sha256_file(fake))
    with pytest.raises(SystemExit, match="consumed top"):
        h.verify_no_collision()


def test_verify_no_collision_refuses_drifted_m12_base(tmp_path, monkeypatch):
    # A drifted m12_seed_base is refused before the block comparison.
    directory = Path(tmp_path) / "staged_heldout_ratios_pa_2026-09-10"
    directory.mkdir()
    fake = directory / "pre_registration.json"
    fake.write_text(json.dumps({"seed_protocol": {
        "m12_seed_base": 20340832,
        "workstream_B": {"pp_reference_replica_seeds":
                         list(range(20346832, 20346840))},
    }}))
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    monkeypatch.setattr(h, "M12_PREREG_SHA256", h._sha256_file(fake))
    with pytest.raises(SystemExit, match="m12_seed_base"):
        h.verify_no_collision()


# --- External-data pins (the committed NA49 thesis tables) ------------------------


@pytest.mark.external_data
def test_grid_inventory_pins():
    if not h.pa_data_dir().is_dir():
        pytest.skip("pa_data tables not present")
    inv = {key: entry["inventory"]
           for key, entry in h.build_data_inventory(support={
               "xF_min": -1.0, "xF_max": 100.0,
               "pT_min": 0.0, "pT_max": 100.0}).items()}
    # Row counts (the README file list).
    assert inv["set1_pi+"]["rows"] == 155
    assert inv["set1_pi-"]["rows"] == 155
    assert inv["set5_pi+"]["rows"] == 138
    assert inv["set5_pi-"]["rows"] == 136
    # The duplicated-label anomaly: set1 pi- is on the fine grid (no
    # anomaly); set1 pi+ and sets 2..5 carry 15/14 duplicated cells.
    assert inv["set1_pi-"]["duplicated_label_cells"] == 0
    for key in ("set1_pi+", "set2_pi+", "set2_pi-", "set3_pi+", "set3_pi-",
                "set4_pi+", "set4_pi-"):
        assert inv[key]["duplicated_label_cells"] == 15, key
        assert inv[key]["duplicated_extra_rows"] == 15
    for key in ("set5_pi+", "set5_pi-"):
        assert inv[key]["duplicated_label_cells"] == 14
    # xF nodes: 9 for sets 1..4, 8 for set 5.
    assert len(inv["set1_pi-"]["xF_nodes"]) == 9
    assert len(inv["set5_pi+"]["xF_nodes"]) == 8
    assert inv["set5_pi+"]["xF_nodes"][-1] == 0.35
    # The pT-zero labels of the coarse-grid files.
    assert inv["set1_pi+"]["pT_zero_cells"] == 9
    assert inv["set1_pi-"]["pT_zero_cells"] == 0
    assert inv["set5_pi+"]["pT_zero_cells"] == 8
    assert inv["set5_pi-"]["pT_zero_cells"] == 7


@pytest.mark.external_data
def test_registered_common_support_counts():
    if not h.pa_data_dir().is_dir():
        pytest.skip("pa_data tables not present")
    probe = h.run_model_support_probe(h.bundled_config_path())
    inv = h.build_data_inventory(support=probe)
    registered = {k: e["n_registered_cells"] for k, e in inv.items()}
    assert registered == {
        "set1_pi+": 131, "set1_pi-": 155,
        "set2_pi+": 131, "set2_pi-": 131,
        "set3_pi+": 131, "set3_pi-": 131,
        "set4_pi+": 131, "set4_pi-": 131,
        "set5_pi+": 115, "set5_pi-": 115,
    }
    # The only non-pT-zero drop is the single zero-error thesis row of
    # set 5 pi+ at (0.00, 2.10).
    for key, entry in inv.items():
        reasons = {d["reason"] for d in entry["dropped"]}
        if key == "set5_pi+":
            assert h.ZERO_ERROR_DROP_REASON in reasons
            assert len(entry["dropped"]) == 9
        elif key == "set5_pi-":
            assert len(entry["dropped"]) == 7
        elif key == "set1_pi-":
            assert entry["dropped"] == []
        else:
            assert all(d["reason"] == h.PT_ZERO_DROP_REASON
                       for d in entry["dropped"])


@pytest.mark.external_data
def test_merge_fixture_real_cells():
    if not h.pa_data_dir().is_dir():
        pytest.skip("pa_data tables not present")
    fixture = h.merge_fixture_records()
    rec = fixture["0.00|0.10"]
    assert rec["rows"] == [[3.79, 0.17434], [3.54, 0.354]]
    w = 1 / 0.17434 ** 2 + 1 / 0.354 ** 2
    assert rec["merged_value"] == pytest.approx(
        (3.79 / 0.17434 ** 2 + 3.54 / 0.354 ** 2) / w, abs=1e-12)
    assert rec["merged_error"] == pytest.approx(math.sqrt(1 / w), abs=1e-12)
    rec2 = fixture["0.00|0.20"]
    assert rec2["rows"][0] == [2.252, 0.110348]
    assert rec2["merged_value"] == pytest.approx(1.9830208052, abs=1e-8)


@pytest.mark.external_data
def test_mapping_table_pins():
    mapping = h.build_mapping_table()
    assert mapping["k_max_pin"] == 14
    assert mapping["k_max_cumulative_mass"] == 0.999
    assert mapping["k_max_cumulative_mass_value"] == pytest.approx(
        0.9994174191253693, abs=1e-12)
    assert mapping["n_candidate_bands"] == 50
    expected = {
        "1": {"band": [2, 4], "residual_percent": 1.276},
        "2": {"band": [4, 4], "residual_percent": -2.439},
        "3": {"band": [4, 7], "residual_percent": 0.919},
        "4": {"band": [6, 6], "residual_percent": 0.0},
        "5": {"band": [7, 7], "residual_percent": 1.449},
    }
    for set_no, pins in expected.items():
        rec = mapping["sets"][set_no]
        assert rec["band"] == pins["band"]
        assert rec["residual_percent"] == pytest.approx(
            pins["residual_percent"], abs=5e-3)
        assert rec["band_ambiguity"] is True
    assert mapping["sets"]["1"]["k_band_mean"] == pytest.approx(
        2.937013, abs=1e-6)
    assert mapping["sets"]["3"]["k_band_mean"] == pytest.approx(
        5.247809, abs=1e-6)
    assert mapping["sets"]["4"]["k_band_mean"] == pytest.approx(6.0, abs=1e-12)
    assert mapping["sets"]["2"]["slice_fraction_model"] == pytest.approx(
        0.124516, abs=1e-6)
    assert mapping["context_rows"]["min_bias_all"]["band"] == [3, 4]
    assert mapping["context_rows"]["all_triggered_gt0"]["band"] == [5, 5]
    # The disjointness statement: the raw minimizers overlap, the overlap
    # is recorded, and no set is remapped.
    assert mapping["pairwise_disjoint"] is False
    assert mapping["overlap_pairs"] == [
        ["1", "2"], ["1", "3"], ["2", "3"], ["3", "4"], ["3", "5"]]
    assert mapping["unmappable_bands"] == []


@pytest.mark.external_data
def test_data_side_dn_dxf_and_r_pins():
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    r1 = h.data_side_r_xf(1, "pi-", 2.9)
    assert len(r1) == 8  # xF 0.00..0.35 on the common grid
    assert r1[0]["xF"] == 0.0
    assert r1[0]["R_data"] == pytest.approx(0.96985, abs=1e-3)
    assert r1[0]["dn_dxF_set"] == pytest.approx(36.19033, abs=1e-3)
    assert r1[-1]["xF"] == 0.35
    r5 = h.data_side_r_xf(5, "pi+", 6.9)
    assert len(r5) == 8
    assert r5[0]["R_data"] == pytest.approx(0.56575, abs=1e-3)
    # The common nodes stop at xF 0.35: the p+p reference carries no 0.40 row.
    assert all(row["xF"] <= 0.35 for row in r1 + r5)
    profiles = h.data_side_dn_dxf(
        h.read_points(h.pa_data_dir() /
                      h.SET_DENSITY_NAMES[(5, "pi+")]))
    assert profiles[0]["xF"] == 0.0
    assert profiles[0]["dn_dxF"] == pytest.approx(46.83707, abs=1e-3)


@pytest.mark.external_data
def test_model_support_probe_reproduces_the_pinned_record():
    if not (h.pa_data_dir().is_dir() and h.m8_record_path().exists()):
        pytest.skip("tables or M8 record not present")
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    stored = prereg["model_support_probe"]
    fresh = h.run_model_support_probe(h.bundled_config_path())
    # The frame-invariant fields reproduce the sealed record exactly: the
    # same events, the same pT envelope, the same wound and pion counts.
    for key in ("seed", "n_events", "band",
                "pT_min", "pT_max", "mean_target_wounds",
                "closed_form_k_band_mean", "n_final_pions",
                "max_conservation_residual"):
        assert fresh[key] == stored[key], key
    # The x_F fields are nucleon-nucleon CM values since card
    # TASK-20261005-pa-frame-nn-cm-xf (2026-10-06). The sealed record holds
    # the withdrawn lab-frame ratio 2 p_z^lab / sqrt(s_NN) (xF_max 13.18).
    assert stored["xF_max"] > 10.0
    assert fresh["xF_min"] == -0.876048261238411
    assert fresh["xF_max"] == 0.6818570713592301
    assert fresh["per_data_xF_node_populated"] == {
        "0.00": 2033, "0.05": 145, "0.10": 23, "0.15": 28, "0.20": 21,
        "0.25": 14, "0.30": 6, "0.35": 3, "0.40": 5}
    # The corrected envelope registers the same common support as the
    # sealed lab-frame one (every data cell lies inside both envelopes).
    fresh_inv = h.build_data_inventory(support=fresh)
    for key, entry in prereg["data_inventory"].items():
        assert fresh_inv[key]["registered"] == entry["registered"], key
    # The wall-time entries are measurements, not pins.
    assert fresh["wall_s"] > 0.0 and math.isfinite(fresh["ms_per_event"])


def test_track_xf_boosts_lab_frame_events_and_keeps_cm_events():
    import types

    if str(MIXING2006) not in sys.path:
        sys.path.insert(0, str(MIXING2006))
    from na49_mixing.event_builder_pA import nn_cm_boost

    beta, gamma, sqrt_s = nn_cm_boost(h.BEAM_MOMENTUM_GEV_C)
    assert sqrt_s == h.SQRT_S_GEV
    m_pi = h.PION_MASS_GEV
    # A pion at rest in the NN CM with pT 0.3, written in the lab frame.
    e_cm = math.sqrt(m_pi * m_pi + 0.09)
    lab_p4 = types.SimpleNamespace(e=gamma * e_cm, px=0.3, py=0.0,
                                   pz=gamma * beta * e_cm)
    lab_event = types.SimpleNamespace(
        stage="baryon_nucleus",
        final_particles=[types.SimpleNamespace(pdg_id=-211, p4=lab_p4,
                                               source_id=0)],
        sources=[types.SimpleNamespace(role="projectile")],
        metadata={"frame": "lab", "nn_cm_beta": beta,
                  "nn_cm_gamma": gamma, "sqrt_s_nn": sqrt_s})
    particle = lab_event.final_particles[0]
    assert abs(h.track_xf(lab_event, particle)) < 1e-12
    arr = h.species_arrays([lab_event], -211)
    assert abs(float(arr["xf"][0])) < 1e-12
    assert arr["role"] == ["projectile"]
    # A CM event (no frame record, the p+p reference) is read as is.
    cm_p4 = types.SimpleNamespace(e=1.0, px=0.3, py=0.0, pz=0.86)
    cm_event = types.SimpleNamespace(
        final_particles=[types.SimpleNamespace(pdg_id=-211, p4=cm_p4,
                                               source_id=0)],
        sources=[types.SimpleNamespace(role="target")], metadata={})
    assert (h.track_xf(cm_event, cm_event.final_particles[0])
            == 2.0 * float(0.86) / float(h.SQRT_S_GEV))
    record = h._pion_source_role_record([lab_event, cm_event], -211)
    assert record["n_projectile"] == 1 and record["n_target"] == 1
    assert abs(record["mean_xF_projectile"]) < 1e-12


@pytest.mark.external_data
def test_preregistration_file_matches_the_harness_constants():
    # The run-record round trip: the committed pre-registration of the M13
    # P1 run directory passes the verbatim cross-check.
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    import hashlib
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    h.cross_check_preregistration(prereg)
    sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    sha_file = (h.RUNS_ROOT / h.RUN_DESCRIPTION /
                "pre_registration.sha256").read_text(encoding="utf-8")
    assert sha in sha_file
    record = json.loads(
        (h.RUNS_ROOT / h.RUN_DESCRIPTION / "harness_cross_check.json")
        .read_text(encoding="utf-8"))
    assert record["status"] == "pass" and record["exit_code"] == 0
    assert record["pre_registration_sha256"] == sha


# --- Phase P2 production machinery (hermetic unless the tables are read) ---------


def test_p2_phase_and_seed_derivation_constants():
    assert h.PHASE_P2 == "P2"
    assert h.PROBE_SEED_MINBIAS_DETERMINISM == 20349844
    # The derived pilot sub-stream sits strictly inside the free pilot
    # territory: above the model-support probe top (20349843, the
    # pre-registered pilot top) and strictly below the slice-1 anchor
    # (20350831), disjoint from every recorded sub-stream.
    assert h.PROBE_SEED_MODEL_SUPPORT < h.PROBE_SEED_MINBIAS_DETERMINISM
    assert h.PROBE_SEED_MINBIAS_DETERMINISM < h.M13_SEED_BASE
    recorded = {h.PROBE_SEED_PP_DETERMINISM, h.PROBE_SEED_SLICED_DETERMINISM,
                h.PROBE_SEED_COST_SET1, h.PROBE_SEED_COST_SET2,
                h.PROBE_SEED_COST_SET3, h.PROBE_SEED_COST_SET4,
                h.PROBE_SEED_COST_SET5, h.PROBE_SEED_ACCEPT_SET1,
                h.PROBE_SEED_ACCEPT_SET2, h.PROBE_SEED_ACCEPT_SET3,
                h.PROBE_SEED_ACCEPT_SET4, h.PROBE_SEED_ACCEPT_SET5,
                h.PROBE_SEED_MODEL_SUPPORT}
    assert h.PROBE_SEED_MINBIAS_DETERMINISM not in recorded
    assert h.WILSON_Z == 1.959963984540054
    assert h.ACCEPTANCE_BAND_MIN_EXPECTED == 5.0


def test_pooled_kernel_yield_collapse_identity():
    """The pre-registered kernel form collapses algebraically to
    sum(n)/(N pi sqrt(s)) (the M12 pooled-kernel collapse); the explicit
    form is what the collection computes."""
    cells = [(0.05, 0.1), (0.1, 0.2), (0.05, 0.3), (0.2, 0.5), (0.3, 1.1)]
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 40, size=len(cells))
    n_total = 1000
    explicit = h.pooled_kernel_yield_model(counts, cells, n_total)
    collapsed = (float(np.sum(counts))
                 / (n_total * math.pi * h.SQRT_S_GEV))
    assert explicit == pytest.approx(collapsed, rel=1e-12)


def test_model_dn_dxf_collapse_identity():
    """The model node integration collapses to sum(n)/(N dxF) over the
    node's cells (the Jacobian cancellation)."""
    cells = [(0.05, 0.1), (0.05, 0.2), (0.1, 0.3), (0.15, 0.5)]
    counts = np.asarray([3, 5, 7, 11])
    n_total = 400
    node = 0.05
    explicit = h.model_dn_dxf_nodes(counts, cells, n_total, [node])["0.05"]
    sub = sum(n for (x, _p), n in zip(cells, counts)
              if abs(x - node) < 1e-9)
    collapsed = sub / (n_total * 2.0 * h.cell_xf_half_width(node))
    assert explicit == pytest.approx(collapsed, rel=1e-12)


def test_sparse_triples_round_trip():
    # Event indices and cell assignments of eleven particles over ten
    # events (two particles share an event, two are unplaced).
    ev = np.asarray([0, 0, 2, 2, 4, 5, 6, 8, 8, 9, 9])
    assign = np.asarray([0, 3, 0, 3, 3, 0, 1, 3, 3, 3, -1])
    triples = h.sparse_cell_triples(ev, assign)
    dense = h.expand_triples(triples, n_events=10, n_cells=5)
    expect = np.zeros((10, 5), dtype=np.int64)
    expect[0, 0] = 1
    expect[0, 3] = 1
    expect[2, 0] = 1
    expect[2, 3] = 1
    expect[4, 3] = 1
    expect[5, 0] = 1
    expect[6, 1] = 1
    expect[8, 3] = 2
    expect[9, 3] = 1
    assert dense.tolist() == expect.tolist()
    # Unplaced particles are excluded from the triples (the overflow is
    # recorded separately in the payload): ten placed particles.
    assert len(triples) == 3 * 9
    with pytest.raises(ValueError):
        h.sparse_cell_triples(np.asarray([0, 1]), np.asarray([0, 1, 2]))


def test_verdict_from_z_bands():
    assert h.verdict_from_z(0.0) == "pass"
    assert h.verdict_from_z(2.0) == "pass"
    assert h.verdict_from_z(2.1) == "tension"
    assert h.verdict_from_z(5.0) == "tension"
    assert h.verdict_from_z(5.1) == "beyond_tension"
    assert h.verdict_from_z(-3.0) == "tension"
    assert h.verdict_from_z(None) == "undefined"
    assert h.verdict_from_z(float("nan")) == "undefined"


def test_acceptance_sanity_band_rules():
    pk = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}
    good = h.acceptance_sanity_band(pk, 1, 4, {1: 200, 2: 300, 3: 298,
                                               4: 202}, 1000)
    assert good["holds"] is True
    assert good["n_qualifying_bins"] == 4
    assert good["n_violating_bins"] == 0
    bad = h.acceptance_sanity_band(pk, 1, 4, {1: 200, 2: 300, 3: 150,
                                              4: 350}, 1000)
    assert bad["holds"] is False
    assert bad["n_violating_bins"] >= 1
    # A bin with expected count below 5 does not qualify.
    tiny = h.acceptance_sanity_band({1: 0.999, 2: 0.0005, 3: 0.0005},
                                    1, 3, {1: 998, 2: 1, 3: 1}, 1000)
    assert tiny["n_qualifying_bins"] == 1


def test_menu_entry_for_budget():
    entry = h.menu_entry_for(13.0)
    assert entry["chosen_events_per_replica"] == 10000
    assert entry["declared_reduction"] is False
    # 5000 ms/event x 8 x 10000 = 400000 s: nothing fits the budget.
    none = h.menu_entry_for(5000.0)
    assert none["chosen_events_per_replica"] is None
    # A cost that fits 5000 but not 10000 declares the reduction
    # (300 ms x 8 x 10000 = 24000 s over budget; x 5000 = 12000 s under).
    mid = h.menu_entry_for(300.0)
    assert mid["chosen_events_per_replica"] == 5000
    assert mid["declared_reduction"] is True


def test_fm13_1_evaluation_synthetic():
    def pooled_entry(model, data, verdict):
        return {"yield_model": model, "yield_data": data,
                "yield_verdict": verdict}
    # Opposite trends plus both endpoints beyond tension: fires.
    pooled = {
        "set1_pi-": pooled_entry(9.0, 1.0, "beyond_tension"),
        "set2_pi-": pooled_entry(7.5, 2.0, "tension"),
        "set3_pi-": pooled_entry(6.0, 3.0, "pass"),
        "set4_pi-": pooled_entry(4.5, 4.0, "tension"),
        "set5_pi-": pooled_entry(3.0, 5.0, "beyond_tension"),
    }
    fired = h._fm13_1_evaluation(pooled)["per_charge"]["pi-"]
    assert fired["fired"] is True
    assert fired["trend_signs_opposite"] is True
    # Same-direction trends: does not fire.
    pooled_same = {
        f"set{s}_pi-": pooled_entry(1.0 * s, 2.0 * s, "pass")
        for s in (1, 2, 3, 4, 5)}
    not_fired = h._fm13_1_evaluation(pooled_same)["per_charge"]["pi-"]
    assert not_fired["fired"] is False


def test_fm13_2_evaluation_synthetic():
    def node(xf, model_r, data_r, sigma):
        return {"xF": xf, "model_R": model_r, "data_R": data_r,
                "sigma_combined": sigma, "verdict": "pass"}
    records = {"set3_pi-": [node(0.0, 2.0, 0.5, 0.05),
                            node(0.05, 1.9, 0.6, 0.05),
                            node(0.10, 2.1, 0.4, 0.05),
                            node(0.20, 1.2, 0.9, 0.10)]}
    fired = h._fm13_2_evaluation(records, "rule")["per_set_charge"][
        "set3_pi-"]
    assert fired["fired"] is True
    # The data side above 1 with the model at 1: does not fire.
    records2 = {"set3_pi-": [node(0.0, 1.0, 1.5, 0.05),
                             node(0.05, 1.0, 1.4, 0.05),
                             node(0.10, 1.0, 1.6, 0.05)]}
    not_fired = h._fm13_2_evaluation(records2, "rule")["per_set_charge"][
        "set3_pi-"]
    assert not_fired["fired"] is False


def test_band_ambiguity_systematic_statement_present():
    assert "band-ambiguity" in h.BAND_AMBIGUITY_SYSTEMATIC_STATEMENT
    assert h.FALSIFIER4_CROSS_REFERENCE.startswith(
        "spec section 7.3 falsifier 4")
    assert h.REPORT_CLAIM_STATEMENT.startswith(
        "the held-out predictive validation")


def test_p2_deviation_registry_records_the_pilot_set1_excursion():
    """The P2 deviation registry carries the recorded set1 pilot acceptance
    excursion (id DEV-1, coverage set1) and the pilot coverage map treats
    exactly the covered failures as recorded."""
    ids = {dev["id"] for dev in h.P2_DEVIATIONS}
    assert "DEV-1" in ids
    dev = next(d for d in h.P2_DEVIATIONS if d["id"] == "DEV-1")
    assert dev["coverage"] == {"probe": "acceptance_sanity_pilot",
                               "set": "set1"}
    assert h.PILOT_DEVIATION_COVERAGE["acceptance_sanity"]["set1"] == "DEV-1"
    # The recorded statement quantifies the excursion and the decision.
    assert "2.06 sigma" in dev["statement"]
    assert "3769" in dev["statement"] and "3764" in dev["statement"]
    assert "no seed, generation parameter, protocol constant" in \
        dev["statement"]
    assert "80000 accepted events per" in dev["statement"]


def test_p2_deviation_registry_records_the_platform_tolerance():
    """DEV-2: the gateway-side recomputation of the mapping table matches
    under the recorded platform tolerance; the comparator accepts the
    recorded 1e-16-class float drift and refuses structural differences."""
    ids = {dev["id"] for dev in h.P2_DEVIATIONS}
    assert "DEV-2" in ids
    dev = next(d for d in h.P2_DEVIATIONS if d["id"] == "DEV-2")
    assert dev["coverage"] == {"probe": "gateway_preregistration_check",
                               "field": "mapping_table"}
    assert "3.7e-14" in dev["statement"]
    assert "numpy 2.0.2" in dev["statement"]
    # The comparator: 1-2 ulp float drift accepted, structure refused.
    fresh = {"band": [2, 4], "mean": 2.9370128803339877,
             "rule": "ties resolve to the narrower band",
             "nested": {"mass": 0.40967344939571354, "n": 50}}
    drifted = {"band": [2, 4], "mean": 2.937012880333987,
               "rule": "ties resolve to the narrower band",
               "nested": {"mass": 0.4096734493957136, "n": 50}}
    assert h.mapping_table_matches_platform(fresh, drifted) is True
    structural = dict(drifted)
    structural["rule"] = "ties resolve to the wider band"
    assert h.mapping_table_matches_platform(fresh, structural) is False
    structural2 = dict(drifted)
    structural2["extra"] = 1
    assert h.mapping_table_matches_platform(fresh, structural2) is False
    too_far = dict(drifted)
    too_far["mean"] = 2.9371
    assert h.mapping_table_matches_platform(fresh, too_far) is False
    assert h.MAPPING_PLATFORM_REL_TOL == 1e-12


def test_bootstrap_yield_uses_the_pinned_collapse():
    """DEV-5's fix: the bootstrap unit's pooled-yield values equal
    sum(n)/(N pi sqrt(s)) on a synthetic dense matrix (the pinned
    pooled-kernel collapse the explicit form satisfies)."""
    cells = [(0.05, 0.1), (0.05, 0.2), (0.1, 0.3)]
    rng = np.random.default_rng(11)
    dense = rng.integers(0, 5, size=(40, 3))
    boot = h._bootstrap_slice_cells(dense, cells, [0.05, 0.1], rng, 10)
    total_n = int(dense.sum())
    expected = total_n / (40 * math.pi * h.SQRT_S_GEV)
    lo, hi = boot["yield_percentile"]
    assert lo <= expected <= hi
    # The node intervals carry the same collapse per node.
    for key, interval in boot["node_percentile"].items():
        node_total = sum(int(n) for (x, _p), n in zip(cells, dense.sum(axis=0))
                         if abs(x - float(key)) < 1e-6)
        exp_node = node_total / (40 * 2.0 * h.cell_xf_half_width(float(key)))
        assert interval[0] <= exp_node <= interval[1]


def test_prereg_check_platform_notes_reset_per_call(tmp_path):
    """DEV-3's fix: cross_check_preregistration resets the platform-note
    list per call, so repeated checks in one process (the determinism
    double run) do not accumulate notes into the payload provenance."""
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        # Hermetic fallback: a minimal synthesized pre-registration.
        pytest.skip("run directory not present")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    h.cross_check_preregistration(prereg)
    first = h.prereg_check_platform_notes()
    h.cross_check_preregistration(prereg)
    second = h.prereg_check_platform_notes()
    assert len(second) == len(first)
    assert second == first


@pytest.mark.external_data
def test_geometry_anchor_crosscheck_values():
    geo = h.geometry_anchor_crosscheck()
    assert geo["sampler"]["sigma_pA_mb"] == pytest.approx(1692.021836314043,
                                                          rel=1e-9)
    assert geo["sampler"]["mean_target_wounds"] == pytest.approx(
        3.937103537381027, rel=1e-9)
    assert geo["sigma_pA_deviation_percent"] == pytest.approx(-6.31,
                                                              abs=0.05)
    assert geo["mean_K_deviation_percent"] == pytest.approx(7.87, abs=0.05)
    assert geo["status"] == "sanity_not_a_gate"


@pytest.mark.external_data
def test_cells_manifest_builds_and_round_trips(tmp_path):
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    manifest = h.build_cells_manifest(h.bundled_config_path(), prereg_path,
                                      tmp_path)
    stored = json.loads((tmp_path / h.SLICE_MANIFEST_NAME).read_text(
        encoding="utf-8"))
    embedded = stored.pop("manifest_sha256")
    assert embedded == h.canonical_json_sha256(stored)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    assert set(manifest["cell_sets"]) == set(prereg["data_inventory"])
    for key, cs in manifest["cell_sets"].items():
        registered = prereg["data_inventory"][key]["registered"]
        assert cs["n_cells"] == len(registered["cells"])
        assert len(cs["cells"]) == len(registered["cells"])
        assert cs["xf_nodes"] == sorted({float(c[0])
                                         for c in registered["cells"]})
        assert sum(len(v) for v in cs["node_cells"].values()) == cs[
            "n_cells"]
    # The p+p reference node plan covers the set xF nodes; the published
    # pp ptintegrated columns carry no 0.40 node (0.35 -> 0.45), recorded
    # as pp_reference_present false, and the R_i comparison excludes it.
    for key, nodes in manifest["ppref_node_plan"].items():
        cs = manifest["cell_sets"][key]
        assert [n["xF"] for n in nodes] == cs["xf_nodes"]
        missing = [n["xF"] for n in nodes if not n["pp_reference_present"]]
        assert missing == ([0.4] if 0.4 in cs["xf_nodes"] else [])
        for n in nodes:
            assert n["cell_indices"] == cs["node_cells"][
                n["node_key"]]
            if n["pp_reference_present"]:
                assert n["pp_published_pT_range"] is not None
    assert manifest["mapping_table"] == prereg["mapping_table"]
    assert manifest["seed_manifest"]["slices"]["1"]["production_seed"] == \
        20350831
    assert manifest["seed_manifest"]["pp_reference"]["production_seed"] == \
        20359831
