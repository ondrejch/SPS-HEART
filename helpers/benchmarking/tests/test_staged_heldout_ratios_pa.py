"""M12 P1 helper: committed behavior of the held-out ratios and p+A trends helper.

Pins the surface of ``helpers/benchmarking/staged_heldout_ratios_pa.py``
(card TASK-20260910-heldout-ratios-pa-trends, M12 phase P1):

- the pre-registered protocol constants (seeds, statistics menu, verdict
  bands, window classes, geometry, normalization anchors, the M8 parameter
  set and the full frozen p+p block) as literals;
- the seed-block derivation: pairwise-disjoint sub-blocks with the
  pre-registered stride, the block span below the next milestone offset,
  and the no-collision record structure (hermetic, on the module constants);
- the pinned seagull kernel arithmetic (the pT/E-weighted invariant-density
  mean with E = sqrt(m^2 + pL^2 + pT^2), pL = xF sqrt(s)/2, and the pT bin
  widths) reproduced exactly by an independently coded two-line closed form
  on a synthetic column, and the model-side statistic path using the same
  closed form on a synthetic event sample (card P1 requirement 1);
- the common-support and window-class arithmetic on synthetic tables, the
  pa_data duplicated-pT-label disambiguation, and the pre-registration
  cross-check (positive control and refusal on mismatched constants).

The table-reading tests carry ``@pytest.mark.external_data`` (the nested
helpers gate runs ``-m "not external_data"`` so hermetic archive checkouts
stay green): the p+p mirror tables, the p+Pb thesis tables, the p+C mirror
tables, the data-side R(xF) recomputation, the data-data unit checks, the
seagull cross-check against the published ptintegrated columns, and the
Glauber geometry anchors. Their pinned values are the pre-registration
record of dev/00runs/staged_heldout_ratios_pa_2026-09-10/ (which the tests
themselves do not read, except the no-collision test, which reads the M11
pre-registration seed_protocol block exactly as the harness does). The
marker contract itself is pinned behaviorally: a hermetic-selection
collection query must keep the mirror-reading seagull closed-form test
deselected while unmarked siblings still collect.

Tolerances (one rationale each): exact equality for integer pins and the
synthetic closed-form identity (algebraically identical expressions, pure
double rounding); 1e-12 absolute for the two-line seagull form; the
pre-registered band boundaries pinned by equality; external-table values
pinned at 1e-3 absolute or 0.5% relative (the tables are fixed committed
records; the pins are regression guards, not floating measurements).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.benchmarking import staged_heldout_ratios_pa as h  # noqa: E402

MIXING2006 = REPO_ROOT / "dev" / "mixing2006"


# --- Layout -------------------------------------------------------------------


def test_monorepo_helper_is_the_module_under_test():
    assert h.__file__ == str(REPO_ROOT / "helpers" / "benchmarking" /
                             "staged_heldout_ratios_pa.py")
    assert h.MIXING2006_DIR == MIXING2006
    assert h.RUNS_ROOT == REPO_ROOT / "dev" / "00runs"
    assert h.pa_data_dir() == h._BENCH_DIR / "pa_data"


# --- Pre-registered protocol constants ------------------------------------------


def test_pre_registered_protocol_constants():
    assert h.CARD == "TASK-20260910-heldout-ratios-pa-trends"
    assert h.MILESTONE == "M12" and h.PHASE == "P1"
    assert h.RUN_DESCRIPTION == "staged_heldout_ratios_pa_2026-09-10"
    assert h.STAGE_LABEL_RATIOS == "heldout_ratios"
    assert h.STAGE_LABEL_PA == "heldout_pa_trends"
    assert h.PRODUCTION_MODE == "regge"
    assert h.BEAM_MOMENTUM_GEV_C == 158.0
    assert h.SQRT_S_GEV == 17.270205201372406
    assert h.CONFIG_SEED == 20260831
    assert h.M12_SEED_BASE == 20340831
    assert h.NEXT_MILESTONE_OFFSET_SEED == 20350831
    assert h.M12_BLOCK_TOP == 20350830
    assert h.M11_PREREG_SHA256 == (
        "d2e206c9bc5b7cf66ca64fbb6358dd739005585968faeb1deb149dc16788b57a")
    assert h.M11_SEED_BASE == 20330831
    assert h.M11_CONSUMED_TOP == 20330839
    assert h.PILOT_SEED_ANCHOR == 20339831
    assert h.PROBE_SEED_PP_DETERMINISM == 20339831
    assert h.PROBE_SEED_PA_DETERMINISM == 20339832
    assert h.PROBE_SEED_PP_COST == 20339833
    assert h.PROBE_SEED_PPB_COST == 20339834
    assert h.PROBE_SEED_PC_COST == 20339835
    assert h.WS_A_PRODUCTION_SEED == 20340831
    assert h.WS_A_REPLICA_SEEDS == tuple(range(20340832, 20340840))
    assert h.WS_A_SHUFFLE_SEED_BASE == 20340841
    assert h.WS_A_SHUFFLE_REALIZATIONS == 16
    assert h.WS_A_BOOTSTRAP_SEED == 20342831
    assert h.WS_B_PRODUCTION_SEED == 20343831
    assert h.WS_B_REPLICA_SEEDS == tuple(range(20343832, 20343840))
    assert h.WS_B_BOOTSTRAP_SEED == 20345831
    assert h.WS_B_PPREF_PRODUCTION_SEED == 20346831
    assert h.WS_B_PPREF_REPLICA_SEEDS == tuple(range(20346832, 20346840))
    assert h.BOOTSTRAP_REPLICATES == 200
    assert h.MIN_SUBBLOCK_STRIDE == 1000
    assert h.M8_RECORD_SHA256 == (
        "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65")
    assert h.M8_FROZEN_BLOCK_SHA256 == (
        "5a46a9ad8b4845791236d1bde33ea1fcb611f9060b3126374dbb1d59010f47f3")


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
    assert h.M8_FROZEN_PP_PARAMETERS["regge"] == {
        "alpha_exc0": 1.1418864566363858,
        "alpha_p0": 1.0808,
        "alpha_p_prime": 0.25,
        "edge_power": 2.3769493784466396,
        "effective_transfer_slope": 8.504140762083612,
        "grid_size": 96,
        "s0": 1.0,
        "transfer_slope": 4.5,
    }
    assert h.M8_FROZEN_PP_PARAMETERS["continuum"]["decay_hypothesis"] == "H3"


def test_verdict_bands_and_statistics_literals():
    assert h.K_PASS == 2.0 and h.K_TENSION == 5.0
    assert h.K_REPLICAS == 8
    assert h.STATISTICS_MENU_EVENTS_PER_REPLICA == (10000, 5000, 2500, 1000)
    assert h.PRIMARY_EVENTS_PER_REPLICA == 10000
    assert h.MAX_PROJECTED_WALL_S == 18000.0
    assert h.DETERMINISM_REPEATS == 5
    assert h.COST_CELL_EVENTS == 300
    assert h.DETERMINISM_PA_EVENTS == 50
    assert h.WINDOW_CLASSES == ("in_window", "forward_xF_gt_0.4",
                                "high_pT_gt_1.5", "negative_xF_out_of_windows")
    assert h.PREDICTIVE_CONTENT["forward_xF_gt_0.4"] == \
        "out_of_window_extrapolation"
    assert h.PREDICTIVE_CONTENT["high_pT_gt_1.5"] == \
        "out_of_window_extrapolation"
    assert h.PREDICTIVE_CONTENT["in_window"] == "in_window_reduced_content"
    assert h.SIGMA_INEL_PPb_MB == 1806.0 and h.SIGMA_INEL_PPb_ERR_MB == 25.0
    assert h.SIGMA_INEL_PC_MB == 226.0 and h.SIGMA_INEL_PC_ERR_MB == 7.0
    assert h.NU_PPB_MEASURED == 3.65 and h.NU_PC_MEASURED == 1.69
    assert h.SQRT_S_GEV == 17.270205201372406


# --- Seed-block derivation (hermetic structure pins) -----------------------------


def test_seed_block_layout_is_pairwise_disjoint_by_stride():
    block = h.derive_seed_block()
    assert block["m12_seed_base"] == 20340831
    starts = [
        block["pilot_sub_block"]["pilot_seed_anchor"],
        block["workstream_A"]["production_seed"],
        block["workstream_A"]["bootstrap_seed"],
        block["workstream_B"]["production_seed"],
        block["workstream_B"]["bootstrap_seed"],
        block["workstream_B"]["pp_reference_production_seed"],
    ]
    ordered = sorted(starts)
    strides = [b - a for a, b in zip(ordered, ordered[1:])]
    assert min(strides) >= h.MIN_SUBBLOCK_STRIDE
    # Every sub-stream top stays below the next milestone offset.
    tops = [
        block["pilot_sub_block"]["pilot_top"],
        block["workstream_A"]["shuffle_seed_base"] + 15,
        block["workstream_A"]["bootstrap_seed"],
        block["workstream_B"]["production_seed"] + 8,
        block["workstream_B"]["bootstrap_seed"],
        block["workstream_B"]["pp_reference_production_seed"] + 8,
    ]
    assert max(tops) < h.NEXT_MILESTONE_OFFSET_SEED
    assert block["workstream_A"]["replica_seeds"] == list(range(
        20340832, 20340840))
    assert block["workstream_B"]["replica_seeds"] == list(range(
        20343832, 20343840))
    assert block["workstream_B"]["pp_reference_replica_seeds"] == list(range(
        20346832, 20346840))
    assert block["workstream_A"]["bootstrap_seed"] == 20342831
    assert block["workstream_B"]["bootstrap_seed"] == 20345831
    assert "no other recorded stream occupies it" in block["no_collision_statement"]
    assert h.M11_PREREG_SHA256 in block["no_collision_statement"]


def test_seed_block_pilot_excluded_from_production_streams():
    block = h.derive_seed_block()
    pilot = block["pilot_sub_block"]
    assert pilot["pilot_seed_anchor"] == 20339831
    assert pilot["pilot_top"] == 20339835
    # The pilot block sits a full stride below the production base and is
    # disjoint (the stride rule is between block starts).
    assert pilot["pilot_top"] < h.WS_A_PRODUCTION_SEED
    assert h.WS_A_PRODUCTION_SEED - pilot["pilot_seed_anchor"] >= 1000
    assert "excluded from production statistics" in pilot["role"]


# --- Window and support arithmetic (synthetic, hermetic) -------------------------


def _synthetic_table(rows):
    return np.asarray(rows, dtype=float)


def test_common_support_is_exact_grid_intersection():
    a = _synthetic_table([
        [0.0, 0.05, 1.0, 0.1], [0.0, 0.10, 2.0, 0.2],
        [0.5, 0.05, 3.0, 0.3],
    ])
    b = _synthetic_table([
        [0.0, 0.05, 4.0, 0.1], [0.0, 0.20, 5.0, 0.2],
        [0.5, 0.05, 6.0, 0.3], [0.6, 0.05, 7.0, 0.3],
    ])
    assert h.common_support(a, b) == [(0.0, 0.05), (0.5, 0.05)]


def test_window_class_classes():
    assert h.window_class(0.0, 0.5) == "in_window"
    assert h.window_class(0.4, 1.5) == "in_window"
    assert h.window_class(0.45, 0.5) == "forward_xF_gt_0.4"
    assert h.window_class(0.2, 1.7) == "high_pT_gt_1.5"
    assert h.window_class(-0.05, 0.5) == "negative_xF_out_of_windows"


def test_window_support_separates_classes():
    tables = {
        "a": _synthetic_table([[0.0, 0.5, 1.0, 0.1], [0.45, 0.5, 2.0, 0.1],
                               [0.2, 1.7, 3.0, 0.1]]),
        "b": _synthetic_table([[0.0, 0.5, 1.0, 0.1], [0.45, 0.5, 2.0, 0.1],
                               [0.2, 1.7, 3.0, 0.1]]),
    }
    sup = h.window_support(tables, "a", "b")
    assert sup["in_window"]["n_cells"] == 1
    assert sup["forward_xF_gt_0.4"]["n_cells"] == 1
    assert sup["high_pT_gt_1.5"]["n_cells"] == 1
    assert sup["negative_xF_out_of_windows"]["n_cells"] == 0


def test_window_yield_kernel_weights_by_pT_over_E():
    # Y_h = sum n (pT/E) at one cell: the kernel factor at xF=0, pT=0.1,
    # m=0.13957039 is 0.1/sqrt(0.13957039^2 + 0.1^2).
    counts = np.asarray([7], dtype=np.int64)
    cells = [(0.0, 0.1)]
    expected = 7.0 * (0.1 / math.sqrt(0.13957039**2 + 0.1**2))
    assert h.model_window_yield(counts, cells, 0.13957039) == \
        pytest.approx(expected, abs=1e-15)


def test_kernel_weight_pL_convention():
    # E = sqrt(m^2 + pL^2 + pT^2) with pL = xF sqrt(s)/2: at xF = 0.4,
    # pL = 0.4 * 17.270205201372406 / 2 = 3.4540410402744812.
    pt = np.asarray([0.1])
    w = h.kernel_weight(pt, 0.93827231, 0.4)
    p_l = 0.4 * h.SQRT_S_GEV / 2.0
    expected = 0.1 / math.sqrt(0.93827231**2 + p_l**2 + 0.1**2)
    assert float(w[0]) == pytest.approx(expected, abs=1e-15)


# --- The pinned seagull closed form (synthetic reproduction test) -----------------


def test_seagull_mean_pt_matches_independent_closed_form():
    # The card P1 requirement 1 pin: the model side reproduces the data-side
    # arithmetic on a synthetic sample. Two-line independent implementation.
    pts = np.asarray([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    f = np.exp(-pts / 0.2)
    mass, xf = 0.13957039, 0.3
    p_l = xf * h.SQRT_S_GEV / 2.0
    dp = np.array([h.standard_dp(p) for p in pts])
    w = f * (pts / np.sqrt(mass**2 + p_l**2 + pts**2)) * dp
    expected = float(np.sum(pts * w) / np.sum(w))
    assert h.seagull_mean_pt(pts, f, mass, xf) == \
        pytest.approx(expected, abs=1e-15)
    # Explicit dp matches the standard-grid default.
    assert h.seagull_mean_pt(pts, f, mass, xf, dp) == \
        pytest.approx(expected, abs=1e-15)


def test_seagull_mean_pt_zero_column_is_nan():
    pts = np.asarray([0.05, 0.1])
    assert math.isnan(h.seagull_mean_pt(pts, np.zeros(2), 0.13957039, 0.0))
    assert math.isnan(h.seagull_mean_pt(pts, np.asarray([1.0, float("nan")]),
                                        0.13957039, 0.0))


def test_seagull_weighting_favors_higher_pT_than_unweighted_mean():
    # The pT/E kernel down-weights the soft end: the kernel mean must sit
    # above the plain f-weighted mean on a falling spectrum.
    pts = np.asarray([0.05, 0.1, 0.2, 0.4, 0.8])
    f = np.exp(-pts / 0.15)
    kernel_mean = h.seagull_mean_pt(pts, f, 0.13957039, 0.0)
    plain_mean = float(np.sum(pts * f) / np.sum(f))
    assert kernel_mean > plain_mean


def _stub_event(particles):
    final = []
    for pdg, px, py, pz in particles:
        p4 = types.SimpleNamespace(px=px, py=py, pz=pz)
        final.append(types.SimpleNamespace(pdg_id=pdg, p4=p4))
    return types.SimpleNamespace(final_particles=final,
                                 metadata={"target_wounds": 1})


@pytest.mark.external_data
def test_model_side_seagull_uses_the_data_side_closed_form():
    # Card P1 requirement 1 pin, in the DEV-1-corrected form (P2): the
    # identical-kernel value in count space is the plain count-weighted pT
    # mean sum(n pT)/sum(n), because the pinned kernel form multiplies the
    # per-cell invariant density (with its thesis Jacobian
    # E/(pi sqrt(s) pT)) by (pT/E) and the pT bin width, and both factors
    # cancel against the Jacobian.
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    pts = [0.05, 0.10, 0.15, 0.20, 0.40]
    counts = [3, 5, 2, 1, 1]
    events = []
    for pt, n in zip(pts, counts):
        for _ in range(n):
            events.append(_stub_event([(-211, 0.0, pt, 0.0)]))
    values = h.pp_seagull_model_values(events)
    counts_arr = np.asarray(counts, dtype=float)
    pts_arr = np.asarray(pts, dtype=float)
    expected = float(np.sum(pts_arr * counts_arr) / np.sum(counts_arr))
    assert values[0] == pytest.approx(expected, abs=1e-15)
    # Algebraic identity with the f-space kernel form applied to the
    # Jacobian-corrected model densities (the literal identical kernel).
    cells = [(0.0, p) for p in pts]
    jac = (math.pi * h.SQRT_S_GEV * pts_arr
           / np.sqrt(0.13957039**2 + pts_arr**2))
    n_ev = len(events)
    dxf = 2.0 * h.cell_xf_half_width(0.0)
    # The model's per-cell invariant density (the f/sigma scale): the cell
    # count divided by the thesis Jacobian, the event count, and the areas.
    densities = counts_arr / (jac * n_ev * dxf
                              * np.asarray([h.standard_dp(p) for p in pts]))
    assert values[0] == pytest.approx(
        h.seagull_mean_pt(pts, densities, 0.13957039, 0.0), abs=1e-15)
    # The spurious-k form the DEV-1 fix removes differs on a mixed-width
    # column (the coarse cells are up-weighted).
    w_counts_dp = counts_arr * np.asarray(
        [h.standard_dp(p) for p in pts]) * np.asarray(
        [h.standard_dp(p) for p in pts])
    biased = float(np.sum(pts_arr * w_counts_dp) / np.sum(w_counts_dp))
    assert biased != pytest.approx(expected, abs=1e-9)


@pytest.mark.external_data
def test_seagull_count_mean_matches_the_f_space_kernel_on_data_tables():
    # The algebraic pin of the identical-kernel equivalence on a real table
    # column: the f-space pinned form (which reproduces the published
    # <pT> at the percent level) equals the plain count-weighted mean when
    # the counts carry the thesis Jacobian; the DEV-1 k-weighted form does
    # not.
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    spectra, integrated = h.load_pp_tables()
    col = spectra["pi-"]
    sel = (np.isclose(col[:, 0], 0.0, atol=1e-6)
           & (col[:, 1] <= h.IN_WINDOW_PT_MAX + 1e-9))
    col = col[sel]
    pts = col[:, 1]
    fs = col[:, 2]
    dp = np.asarray([h.standard_dp(p) for p in pts])
    k = h.kernel_weight(pts, h.SPECIES_MASS_GEV["pi-"], 0.0)
    f_space = float(np.sum(pts * fs * k * dp) / np.sum(fs * k * dp))
    jac = (math.pi * h.SQRT_S_GEV * pts
           / np.sqrt(h.SPECIES_MASS_GEV["pi-"]**2 + pts**2))
    counts = jac * (fs / h.SIGMA_INEL_PP_MB) * dp
    cells = [(0.0, float(p)) for p in pts]
    assert h.seagull_count_mean_pt(cells, counts) == pytest.approx(
        f_kernel := float(np.sum(pts * fs * k * dp)
                          / np.sum(fs * k * dp)), rel=1e-12)
    k_weighted = float(np.sum(pts * counts * k) / np.sum(counts * k))
    assert k_weighted != pytest.approx(f_kernel, abs=1e-3)
    # The f-space form reproduces the published value at the percent level.
    published = float(integrated["pi-"][
        np.isclose(integrated["pi-"][:, 0], 0.0)][0][5])
    assert abs(f_kernel - published) / published < 0.01


@pytest.mark.external_data
def test_pp_window_model_ratios_returns_one_value_per_registered_window():
    # DEV-2 pin: the model ratio path consumes the pre-registered cell sets
    # and returns exactly one value per registered ratio window (11).
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    rng = np.random.default_rng(3)
    events = []
    for _ in range(400):
        particles = []
        for pdg, n in ((-211, 2), (211, 2), (321, 1), (-321, 1),
                       (2212, 1), (-2212, 1)):
            for _s in range(n):
                # Half the particles in the low-pT band, half reaching the
                # high-pT windows, so every registered window fills.
                pt = (float(rng.uniform(0.05, 0.9)) if rng.random() < 0.5
                      else float(rng.uniform(1.55, 2.0)))
                particles.append((pdg, 0.0, pt,
                                  float(rng.uniform(0.0, 7.0))))
        events.append(_stub_event(particles))
    values = h.pp_window_model_ratios(events)
    assert len(values) == 11
    assert all(math.isfinite(v) for v in values)


def test_bin_species_counts_first_match_and_overflow():
    pts = [0.05, 0.10]
    cells = [(0.0, p) for p in pts]
    # One pi- exactly at pT=0.05 (inside the first cell), one at pT=0.001
    # (below the first cell's half-width: overflow).
    events = [
        _stub_event([(-211, 0.0, 0.05, 0.0)]),
        _stub_event([(-211, 0.0, 0.001, 0.0)]),
        _stub_event([(211, 0.0, 0.05, 0.0)]),  # wrong charge: never counted
    ]
    counts, overflow = h.bin_species_counts(events, cells, -211)
    assert counts.tolist() == [1, 0]
    assert h.bin_species_counts(events, cells, -211)[1] == 1
    counts_p, _ = h.bin_species_counts(events, cells, 211)
    assert counts_p.tolist() == [1, 0]


# --- pa_data duplicated-pT-label disambiguation ----------------------------------


def test_fine_pt_labels_maps_doubled_labels():
    xf = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pt = np.asarray([0.0, 0.10, 0.10, 0.20, 0.20, 0.30, 0.40, 0.50])
    out = h.fine_pt_labels(xf, pt)
    np.testing.assert_allclose(
        out, [0.0, 0.10, 0.05, 0.20, 0.15, 0.30, 0.40, 0.50])


# --- Pre-registration cross-check machinery (hermetic) ----------------------------


def _synthetic_prereg() -> dict:
    return {
        "card": h.CARD,
        "milestone": h.MILESTONE,
        "phase": h.PHASE,
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
            "C12": dict(h.GEOMETRY_C12),
            "pa_runner_settings": dict(h.PA_RUNNER_SETTINGS),
        },
        "normalization_anchors": {
            "sigma_inel_pp_mb": h.SIGMA_INEL_PP_MB,
            "sigma_inel_pPb_mb": h.SIGMA_INEL_PPb_MB,
            "sigma_inel_pC_mb": h.SIGMA_INEL_PC_MB,
            "nu_pPb_measured": h.NU_PPB_MEASURED,
            "nu_pC_measured": h.NU_PC_MEASURED,
        },
        "seed_protocol": h.derive_seed_block(),
        "statistics": {
            "k_replicas": h.K_REPLICAS,
            "menu_events_per_replica": list(h.STATISTICS_MENU_EVENTS_PER_REPLICA),
            "primary_events_per_replica": h.PRIMARY_EVENTS_PER_REPLICA,
            "max_projected_wall_s": h.MAX_PROJECTED_WALL_S,
            "bootstrap_replicates": h.BOOTSTRAP_REPLICATES,
            "determinism_repeats": h.DETERMINISM_REPEATS,
            "cost_cell_events": h.COST_CELL_EVENTS,
            "determinism_pa_events": h.DETERMINISM_PA_EVENTS,
        },
        "verdict_bands": {"k_pass": h.K_PASS, "k_tension": h.K_TENSION},
        "verdict_bands": {"k_pass": h.K_PASS, "k_tension": h.K_TENSION},
        "protocol_A": {
            "workstream": "A",
            "window_definitions": h._to_jsonable(h.WINDOW_DEFINITIONS_LITERAL),
        },
        "protocol_B": {
            "workstream": "B",
            "normalization_anchors": h._to_jsonable(h.PA_ANCHORS_LITERAL),
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
        lambda r: r["seed_protocol"]["workstream_A"].__setitem__(
            "production_seed", 20340832),
        lambda r: r["calibration_set_decision_of_record"][
            "m8_parameter_set"].__setitem__("regge.alpha_exc0", 1.14),
        lambda r: r["geometry"]["Pb208"].__setitem__("radius_fm", 6.5),
        lambda r: r["geometry"]["C12"].__setitem__("radius_fm", 1.07),
        lambda r: r["protocol_A"]["window_definitions"].__setitem__(
            "in_window_xf_max", 0.5),
        lambda r: r["protocol_A"]["window_definitions"][
            "predictive_content"].__setitem__("in_window", "extrapolation"),
        lambda r: r["protocol_B"]["normalization_anchors"].__setitem__(
            "sigma_inel_pPb_mb", 1686.0),
    ],
    ids=[
        "k_pass_drift", "k_tension_drift", "menu_drift", "bootstrap_drift",
        "seed_drift", "m8_parameter_drift",
        "geometry_pb_drift", "geometry_c12_drift", "window_drift",
        "content_class_drift", "anchor_drift",
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


def test_verify_no_collision_refuses_missing_m11_file(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    with pytest.raises(SystemExit, match="M11 pre-registration not found"):
        h.verify_no_collision()


def test_verify_no_collision_refuses_sha_mismatch(tmp_path, monkeypatch):
    # Any file content whose sha differs from the recorded M11 value is
    # refused before the seed block is parsed.
    (Path(tmp_path) / "parameter_profiles_2026-09-07").mkdir()
    fake = (Path(tmp_path) / "parameter_profiles_2026-09-07" /
            "pre_registration.json")
    fake.write_text(json.dumps({"seed_protocol": {
        "m11_seed_base": 20330831,
        "shared_replica_block": [20330832, 20330839],
    }}))
    monkeypatch.setattr(h, "RUNS_ROOT", Path(tmp_path))
    with pytest.raises(SystemExit, match="sha256"):
        h.verify_no_collision()


# --- External-data pins (the committed NA49 tables) -------------------------------


@pytest.mark.external_data
def test_pp_tables_load_and_pin_ratio_windows():
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    stats = h.all_ratio_window_statistics()
    pp = stats["pi-_over_pi+"]
    assert pp["total_common_cells"] == 276
    assert pp["windows"]["in_window"]["n_cells"] == 229
    assert pp["windows"]["in_window"]["ratio_data"] == pytest.approx(
        0.8427905676670945, abs=1e-9)
    assert pp["windows"]["forward_xF_gt_0.4"]["n_cells"] == 29
    assert pp["windows"]["forward_xF_gt_0.4"]["xF_min"] == 0.45
    assert pp["windows"]["forward_xF_gt_0.4"]["xF_max"] == 0.55
    assert pp["windows"]["high_pT_gt_1.5"]["n_cells"] == 18
    assert pp["windows"]["negative_xF_out_of_windows"]["n_cells"] == 0
    pk = stats["p_over_pi-"]
    assert pk["windows"]["in_window"]["n_cells"] == 102
    assert pk["windows"]["forward_xF_gt_0.4"]["n_cells"] == 49
    assert pk["windows"]["forward_xF_gt_0.4"]["xF_max"] == 0.85
    assert pk["windows"]["high_pT_gt_1.5"]["n_cells"] == 12
    ppp = stats["p_over_pi+"]
    assert ppp["windows"]["forward_xF_gt_0.4"]["n_cells"] == 24
    assert ppp["windows"]["forward_xF_gt_0.4"]["xF_max"] == 0.55
    kk = stats["k+_over_k-"]
    assert kk["windows"]["in_window"]["n_cells"] == 148
    assert kk["windows"]["forward_xF_gt_0.4"]["status"] == \
        "dropped_no_common_support"
    assert kk["windows"]["high_pT_gt_1.5"]["n_cells"] == 4
    pbar = stats["p_over_pbar"]
    assert pbar["windows"]["in_window"]["n_cells"] == 122
    assert pbar["windows"]["forward_xF_gt_0.4"]["status"] == \
        "dropped_no_common_support"
    assert pbar["windows"]["high_pT_gt_1.5"]["status"] == \
        "dropped_no_common_support"
    assert pbar["windows"]["negative_xF_out_of_windows"]["n_cells"] == 21
    assert len(h.dropped_windows_record()) >= 3


@pytest.mark.external_data
def test_seagull_kernel_reproduces_the_published_pion_means():
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    spectra, integrated = h.load_pp_tables()
    for sp, table_ref in (("pi-", 0.2579), ("pi+", 0.2590)):
        cc = h.data_side_seagull_crosscheck(spectra, integrated, sp, 0.0)
        assert cc["n_cells"] > 10
        assert cc["rel_deviation"] == pytest.approx(
            (cc["kernel_mean_pt"] - table_ref) / table_ref, abs=1e-12)
        assert abs(cc["rel_deviation"]) < 0.01


@pytest.mark.external_data
def test_data_data_unit_checks_rederive_the_cross_system_pins():
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    unit = h.data_data_unit_checks()
    assert unit["mean_pt_xF0_pp_pi-"] == 0.2579
    assert unit["mean_pt_xF0_pPb_pi-"] == 0.2578
    f_over = unit["f_over_sigma_times_d_pi-"]
    assert f_over["n_cells"] == 145
    assert f_over["median"] == pytest.approx(0.9956, abs=1e-3)
    # The density-integrated dn/dxF chain against the thesis ptintegrated
    # column: recorded at the percent level near mid-xF, with the known
    # table-grid deviations at the edges.
    per_xf = {row["xF"]: row for row in unit["dn_dxF_cross_check_pi-"]}
    row = per_xf[0.05]
    assert row["dn_dxF_from_density"] == pytest.approx(
        row["dn_dxF_ptintegrated"], rel=0.05)


@pytest.mark.external_data
def test_data_side_r_xf_pPb_pin():
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    geo = h.geometry_anchors()
    r = h.data_side_r_xf("pPb", geo["Pb208"]["mean_target_wounds"])
    pi = r["pi-"]
    assert len(pi) == 8  # xF 0.00..0.35 on the common grid
    assert pi[0]["R_samplerK"] == pytest.approx(0.7981, abs=1e-3)
    assert pi[0]["R_measuredAnchor"] == pytest.approx(0.8474, abs=1e-3)
    devs = [row["anchor_deviation"] for row in pi]
    # The systematic is a uniform <K> rescaling: every row carries the same
    # deviation up to double rounding.
    assert max(abs(d + 0.058152221278577) for d in devs) < 1e-9


@pytest.mark.external_data
def test_data_side_r_xf_pC_pin():
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    geo = h.geometry_anchors()
    r = h.data_side_r_xf("pC", geo["C12"]["mean_target_wounds"])
    pi = r["pi-"]
    assert pi[0]["xF"] == 0.0
    assert pi[0]["R_samplerK"] == pytest.approx(0.9852, abs=1e-3)
    # The common grid stops at xF 0.30: the p+C table continues to 0.5, the
    # p+p table has no 0.40/0.50 rows.
    assert pi[-1]["xF"] == 0.3
    assert len(pi) == 11


@pytest.mark.external_data
def test_geometry_anchors_at_the_pinned_values():
    geo = h.geometry_anchors()
    pb = geo["Pb208"]
    assert pb["sigma_pA_mb"] == pytest.approx(1692.0, abs=1.0)
    assert pb["mean_target_wounds"] == pytest.approx(3.9371, abs=1e-3)
    c = geo["C12"]
    assert c["sigma_pA_mb"] == pytest.approx(205.4, abs=1.0)
    assert c["mean_target_wounds"] == pytest.approx(1.9030, abs=1e-3)
    # The foundations reference numbers: sigma_pA(C12) ~ 0.20 b, T(0) ~ 1.1.
    assert 0.19 < c["sigma_pA_mb"] / 1000.0 < 0.21


@pytest.mark.external_data
def test_preregistration_file_matches_the_harness_constants():
    # The run-record round trip: the committed pre-registration of the M12
    # P1 run directory passes the verbatim cross-check.
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    import hashlib
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    h.cross_check_preregistration(prereg)
    sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    record = json.loads(
        (h.RUNS_ROOT / h.RUN_DESCRIPTION / "harness_cross_check.json")
        .read_text(encoding="utf-8"))
    assert record["status"] == "pass" and record["exit_code"] == 0
    assert record["pre_registration_sha256"] == sha


# --- P2 production path (hermetic) -----------------------------------------------


def test_p2_deviations_recorded():
    ids = [d["id"] for d in h.P2_DEVIATIONS]
    assert ids == ["DEV-1", "DEV-2", "DEV-3"]
    dev1 = h.P2_DEVIATIONS[0]["statement"]
    assert "raw binned cell counts" in dev1
    assert "plain count-weighted pT mean" in dev1
    assert "Jacobian" in dev1
    assert "no seed" in h.P2_DEVIATIONS[0]["statement"]
    assert "cell list" in h.P2_DEVIATIONS[1]["statement"]
    dev3 = h.P2_DEVIATIONS[2]["statement"]
    assert "with-areas" in dev3 and "pre-registered" in dev3
    assert "sha256 is re-registered" in dev3
    assert "no seed" in dev3 or "no seed changes" in dev3


def test_verdict_from_z_band_branches():
    assert h.verdict_from_z(0.0) == "pass"
    assert h.verdict_from_z(1.999) == "pass"
    assert h.verdict_from_z(2.0) == "pass"
    assert h.verdict_from_z(-2.0001) == "tension"
    assert h.verdict_from_z(5.0) == "tension"
    assert h.verdict_from_z(-5.0001) == "beyond_tension"
    assert h.verdict_from_z(float("nan")) == "no_model_support"
    assert h.verdict_from_z(float("inf")) == "no_model_support"


def test_wilson_ratio_interval_closed_form_and_odds_mapping():
    rec = h.wilson_ratio_interval(30, 70)
    assert rec is not None
    z = 1.959963984540054
    n, p = 100, 0.30
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    assert rec["fraction_interval"] == pytest.approx(
        [center - half, center + half], abs=1e-15)
    lo, hi = rec["fraction_interval"]
    assert rec["ratio_interval"] == [lo / (1 - lo), hi / (1 - hi)]
    # The interval contains the observed ratio and is inside the score band.
    assert rec["ratio_interval"][0] < 30 / 70 < rec["ratio_interval"][1]
    assert h.wilson_ratio_interval(0, 0) is None


def test_bootstrap_ratio_interval_seeded_and_deterministic():
    rng = np.random.default_rng(7)
    num = rng.random(500)
    den = rng.random(500) + 0.1
    a = h.bootstrap_ratio_interval(num, den, replicates=50, seed=12345)
    b = h.bootstrap_ratio_interval(num, den, replicates=50, seed=12345)
    assert a["percentile_interval"] == b["percentile_interval"]
    assert a["seed"] == 12345 and a["replicates"] == 50
    lo, hi = a["percentile_interval"]
    point = float(num.sum() / den.sum())
    assert lo <= point <= hi
    # A zero-denominator resample is invalid, not a crash.
    den_bad = den.copy()
    den_den = h.bootstrap_ratio_interval(
        num, np.asarray([0.0] * 499 + [1.0]), replicates=50, seed=7)
    assert den_den["invalid_replicates"] >= 1


def test_bin_first_match_matches_the_scalar_rule():
    rng = np.random.default_rng(11)
    events = []
    for _ in range(40):
        particles = []
        for _n in range(rng.integers(1, 5)):
            particles.append((int(rng.choice([-211, 211, 2212])),
                              float(rng.uniform(-0.1, 0.9)),
                              float(rng.uniform(0.0, 1.6)),
                              float(rng.uniform(-3, 3))))
        events.append(_stub_event(particles))
    cells = [(0.0, 0.05), (0.0, 0.10), (0.0, 0.20), (0.45, 0.30),
             (0.45, 0.50), (0.2, 1.0)]
    counts_vec, assign = h._bin_first_match(
        np.asarray([h.particle_xf(p, event=e) for e in events
                    for p in e.final_particles
                    if int(p.pdg_id) == -211]),
        np.asarray([h.particle_pt(p) for e in events
                    for p in e.final_particles
                    if int(p.pdg_id) == -211]),
        cells)
    counts_scalar, overflow_scalar = h.bin_species_counts(
        events, cells, -211)
    assert counts_vec.tolist() == counts_scalar.tolist()
    assert int((assign < 0).sum()) == overflow_scalar


def _ratio_cell_events(species_pdg, xf, pt, source_id=0):
    return _stub_event([(species_pdg, xf, pt, 0.0)])


def _synthetic_manifest(tmp_path):
    """A minimal cells manifest covering one registered ratio window, one
    seagull node, and the symmetry grid, with the embedded sha."""
    manifest = {
        "schema": h.MANIFEST_SCHEMA,
        "card": h.CARD, "milestone": h.MILESTONE, "phase": h.PHASE_P2,
        "run_description": h.RUN_DESCRIPTION,
        "provenance": {"configuration_hash": "x"},
        "m8_crosscheck": {"verified": True},
        "binning_rule": "synthetic",
        "yield_kernel": "synthetic",
        "seagull_kernel": "synthetic",
        "ratio_windows": {
            "pi-_over_pi+|in_window": {
                "numerator": "pi-", "denominator": "pi+",
                "predictive_content": "in_window_reduced_content",
                "status": "registered", "n_cells": 2,
                "xF_min": 0.0, "xF_max": 0.05,
                "pT_min": 0.05, "pT_max": 0.10,
                "cells": [[0.0, 0.05], [0.05, 0.10]],
                "Y_numerator": 1.0, "Y_denominator": 1.0,
                "ratio_data": 1.0,
            },
            "pi-_over_pi+|forward_xF_gt_0.4": {
                "numerator": "pi-", "denominator": "pi+",
                "predictive_content": "out_of_window_extrapolation",
                "status": "dropped_no_common_support",
                "drop_reason": "synthetic drop",
            },
        },
        "seagull": {
            sp: {"species": sp, "nodes": {}}
            for sp in h.PP_TABLE_NAMES
        },
        "symmetry_grid": {
            "xF_nodes": [0.05], "pT_nodes": [0.05, 0.10],
            "cells": [[0.05, 0.05], [0.05, 0.10],
                      [-0.05, 0.05], [-0.05, 0.10]],
            "statement": "synthetic",
        },
    }
    manifest["seagull"]["pi-"] = {"species": "pi-", "nodes": {
        "0.0": {"xF": 0.0, "n_cells": 2, "pT_min": 0.05,
                "pT_max": 0.10,
                "cells": [[0.0, 0.05], [0.0, 0.10]],
                "published_mean_pt": 0.25,
                "published_err_abs": 0.01,
                "kernel_mean_pt": 0.25,
                "kernel_err_abs": 0.01,
                "reconstruction_deviation": 0.0,
                "supported": True, "reason": ""},
        "0.4": {"xF": 0.4, "n_cells": 0, "supported": False,
                "reason": "no 2D-table cells"},
    }}
    manifest["manifest_sha256"] = h.canonical_json_sha256(manifest)
    path = tmp_path / "cells_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest, path


def _rich_stub_events():
    """Synthetic events with sources, conservation residuals, antibaryons,
    and both source labels - enough surface for build_replica_payload.

    Kinematics: xF = 2 pz / sqrt(s) with sqrt(s) = 17.270205201372406, so
    xF = 0.05 corresponds to pz = 0.4317551300343101.
    """
    sys.path.insert(0, str(h.MIXING2006_DIR))
    from na49_mixing.quantum_numbers import QuantumNumbers

    sqrt_s = h.SQRT_S_GEV
    pz_005 = 0.05 * sqrt_s / 2.0

    def part(pdg, pz, pt, src, baryon=None):
        qn = (QuantumNumbers(1, -1, 0) if pdg == -2212
              else types.SimpleNamespace(B=0))
        return types.SimpleNamespace(
            pdg_id=pdg,
            p4=types.SimpleNamespace(px=0.0, py=0.0 if pt == 0 else pt,
                                     pz=pz),
            source_id=src,
            quantum_numbers=qns_of(pdg))

    def qns_of(pdg):
        from na49_mixing.quantum_numbers import quantum_numbers as qn_fn
        return qn_fn(pdg)

    events = []
    # Event 0: pi- at the (0.0, 0.05) and (0.05, 0.10) window cells from
    # source 0, and a pi+ at the (0.05, 0.10) cell from source 1.
    events.append(types.SimpleNamespace(
        event_id=0,
        conservation_residuals=lambda: {"relative_max": 1e-16},
        sources=[
            types.SimpleNamespace(source_id=0, mass=3.0,
                                  quantum_numbers=QuantumNumbers(1, 1, 0)),
            types.SimpleNamespace(source_id=1, mass=3.0,
                                  quantum_numbers=QuantumNumbers(1, 1, 0)),
        ],
        final_particles=[
            types.SimpleNamespace(pdg_id=-211,
                                  p4=types.SimpleNamespace(px=0.0, py=0.05,
                                                           pz=0.0),
                                  source_id=0,
                                  quantum_numbers=qns_of(-211)),
            types.SimpleNamespace(pdg_id=-211,
                                  p4=types.SimpleNamespace(px=0.0, py=0.10,
                                                           pz=pz_005),
                                  source_id=0,
                                  quantum_numbers=qns_of(-211)),
            types.SimpleNamespace(pdg_id=211,
                                  p4=types.SimpleNamespace(px=0.0, py=0.10,
                                                           pz=pz_005),
                                  source_id=1,
                                  quantum_numbers=qns_of(211)),
        ]))
    # Event 1: a pbar whose source sits below the pair threshold (test 7
    # violation) plus an overflow pion far outside every cell.
    events.append(types.SimpleNamespace(
        event_id=1,
        conservation_residuals=lambda: {"relative_max": 2e-16},
        sources=[
            types.SimpleNamespace(source_id=0, mass=1.5,
                                  quantum_numbers=QuantumNumbers(1, 1, 0)),
            types.SimpleNamespace(source_id=1, mass=3.0,
                                  quantum_numbers=QuantumNumbers(1, 1, 0)),
        ],
        final_particles=[
            types.SimpleNamespace(pdg_id=-2212,
                                  p4=types.SimpleNamespace(px=0.0, py=0.0,
                                                           pz=0.0),
                                  source_id=0,
                                  quantum_numbers=qns_of(-2212)),
            types.SimpleNamespace(pdg_id=-211,
                                  p4=types.SimpleNamespace(px=0.0, py=0.30,
                                                           pz=8.0),
                                  source_id=1,
                                  quantum_numbers=qns_of(-211)),
        ]))
    return events


def test_build_replica_payload_on_synthetic_events(tmp_path, monkeypatch):
    manifest, manifest_path = _synthetic_manifest(tmp_path)
    monkeypatch.setattr(h, "cross_check_m8_record",
                        lambda *a, **k: {"verified": True})
    monkeypatch.setattr(h, "canonical_configuration_hash",
                        lambda p: "x")
    events = _rich_stub_events()
    payload = h.build_replica_payload(
        20340832, tmp_path / "config.json", manifest_path,
        events=events)
    assert payload["n_events"] == 2
    # The window counts equal the scalar first-match rule.
    cells = [(0.0, 0.05), (0.05, 0.10)]
    counts_num, ov_num = h.bin_species_counts(events, cells, -211)
    counts_den, ov_den = h.bin_species_counts(events, cells, 211)
    assert counts_num.tolist() == [1, 1]
    assert counts_den.tolist() == [0, 1]
    rec = payload["window_records"]["pi-_over_pi+|in_window"]
    assert rec["num"]["counts"] == counts_num.tolist()
    assert rec["den"]["counts"] == counts_den.tolist()
    # The per-event contributions sum to the binned-count yield.
    k_cell = h._cell_kernel_factors(cells, h.SPECIES_MASS_GEV["pi-"])
    expected_yield = float(np.sum(np.asarray(counts_num) * k_cell))
    assert rec["num"]["yield"] == pytest.approx(expected_yield, abs=1e-15)
    assert sum(payload["per_event"]["pi-_over_pi+|in_window"]["num"]) == \
        pytest.approx(expected_yield, abs=1e-15)
    # The dropped window carries no per-event contributions.
    assert "pi-_over_pi+|forward_xF_gt_0.4" not in payload["per_event"]
    # Seagull counts on the supported node; the unsupported node is skipped.
    sg = payload["species_binning"]["pi-"]["seagull_counts"]
    assert sg["0.0"] == [1, 0]
    assert "0.4" not in sg
    # The overflow: the pion at xF=0.93/pT=0.30 matches no cell anywhere.
    assert payload["species_binning"]["pi-"]["overflow"] == 1
    # The antibaryon traceability (test 7): the light-source pbar violates.
    cons = payload["conservation"]
    assert cons["n_antibaryons"] == 1
    assert cons["antibaryon_pair_violations"] == 1
    assert cons["source_quantum_number_violations"] == 0
    assert cons["min_antibaryon_pair_margin_gev"] == pytest.approx(
        1.5 - h._antibaryon_pair_margin(-2212), abs=1e-12)
    # Symmetry grid: the (xF=0.05, pT=0.10) pi- from source 0.
    sym = np.asarray(payload["species_binning"]["pi-"]["symmetry_counts"])
    assert sym[0].sum() + sym[1].sum() == 1


def test_bin_first_match_and_overflow_unchanged():
    # The pre-existing pin, repeated for the vectorized path: wrong charge is
    # never counted; an out-of-bin particle overflows.
    pts = [0.05, 0.10]
    cells = [(0.0, p) for p in pts]
    events = [
        _stub_event([(-211, 0.0, 0.05, 0.0)]),
        _stub_event([(-211, 0.0, 0.001, 0.0)]),
        _stub_event([(211, 0.0, 0.05, 0.0)]),
    ]
    counts, overflow = h.bin_species_counts(events, cells, -211)
    assert counts.tolist() == [1, 0] and overflow == 1


def test_seagull_node_support_synthetic():
    # Synthetic species tables: the support flags and the reconstruction
    # arithmetic.
    spectra = {"s": np.asarray([
        [0.0, 0.05, 10.0, 0.1], [0.0, 0.10, 5.0, 0.1],
        [0.0, 0.20, 1.0, 0.1],
        [0.4, 0.40, 1.0, 0.1], [0.4, 0.50, 0.5, 0.1],
    ])}
    integrated = {"s": np.asarray([
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.10, 5.0, 0.05, 1.0],
        [0.2, 1.0, 1.0, 1.0, 1.0, 0.15, 5.0, 0.05, 1.0],
    ])}
    h.SPECIES_MASS_GEV["s"] = 0.13957039
    try:
        rec = h.seagull_node_support(spectra, integrated, "s")
    finally:
        del h.SPECIES_MASS_GEV["s"]
    n0 = rec["nodes"]["0.0"]
    assert n0["n_cells"] == 3 and n0["supported"] is True
    # The kernel reconstruction at xF=0 over the synthetic column, with the
    # per-cell error propagation (all three cells sit below pT 0.3, so the
    # standard widths are 0.05).
    w = (np.asarray([10.0, 5.0, 1.0])
         * h.kernel_weight(np.asarray([0.05, 0.10, 0.20]), 0.13957039, 0.0)
         * np.asarray([0.05, 0.05, 0.05]))
    expected = float(np.sum(np.asarray([0.05, 0.10, 0.20]) * w) / np.sum(w))
    assert n0["kernel_mean_pt"] == pytest.approx(expected, abs=1e-15)
    # The percent-error convention of the table header: the absolute <pT>
    # error is (pct / 100) * value.
    assert n0["published_err_abs"] == pytest.approx(
        (5.0 / 100.0) * 0.10, rel=1e-12)
    # The xF=0.2 ptintegrated node has no 2D-table cells: recorded with the
    # reason, unsupported. The 2D column at xF=0.4 has no ptintegrated node
    # and therefore no node record at all.
    n02 = rec["nodes"]["0.2"]
    assert n02["n_cells"] == 0 and n02["supported"] is False
    assert sorted(rec["nodes"]) == ["0.0", "0.2"]


def test_falsifier_cross_reference_no_fire_and_fire():
    def rec(delta, z, model, data):
        return {"status": "registered", "delta": delta,
                "delta_over_sigma_combined": z,
                "model_ratio": model, "data_ratio": data}

    base = {}
    for win in ("in_window", "forward_xF_gt_0.4", "high_pT_gt_1.5"):
        base[f"pi-_over_pi+|{win}"] = rec(0.01, 0.5, 0.85, 0.84)
    base["k+_over_k-|in_window"] = rec(0.0, 0.5, 1.5, 1.5)
    out = h.falsifier_cross_reference(base)
    assert out["falsifier_1_ratios"]["fired"] is False
    assert out["falsifier_5_parameter_variation"]["fired"] is False
    # A constructed joint violation pattern fires falsifier 1: all three
    # pi-/pi+ windows far above the data away from R = 1 and K+/K- below.
    fired = {}
    for win in ("in_window", "forward_xF_gt_0.4", "high_pT_gt_1.5"):
        fired[f"pi-_over_pi+|{win}"] = rec(+0.30, 9.0, 1.20, 0.84)
    fired["k+_over_k-|in_window"] = rec(-0.6, -9.0, 0.9, 1.5)
    out2 = h.falsifier_cross_reference(fired)
    assert out2["falsifier_1_ratios"]["fired"] is True
    # Falsifier 5: one registered ratio flipping sign beyond tension between
    # adjacent windows (the falsifier walks the five registered ratios).
    flip = {"p_over_pi+|in_window": rec(0.5, 8.0, 1.0, 0.5),
            "p_over_pi+|forward_xF_gt_0.4": rec(-0.5, -8.0, 0.1, 0.5)}
    out3 = h.falsifier_cross_reference(flip)
    assert out3["falsifier_5_parameter_variation"]["fired"] is True
    assert out3["falsifier_5_parameter_variation"]["flips"][0]["ratio"] == \
        "p_over_pi+"


def test_symmetry_grid_nodes():
    nodes = h.SYMMETRY_GRID_PT_NODES
    assert nodes[0] == 0.05 and nodes[-1] == 1.40
    # 0.05 steps below 0.3, 0.10 above.
    assert list(nodes[:6]) == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert list(nodes[6:9]) == [0.40, 0.50, 0.60]
    assert len(h.SYMMETRY_GRID_XF_NODES) == 8
    # The stored grid (built in the manifest) covers both signs for every
    # xF node; check the construction rule on a fresh node pair.
    cells = sorted(
        [(round(float(x), 4), round(float(p), 4))
         for x in h.SYMMETRY_GRID_XF_NODES for p in nodes]
        + [(round(-float(x), 4), round(float(p), 4))
           for x in h.SYMMETRY_GRID_XF_NODES for p in nodes])
    assert (0.05, 0.05) in cells and (-0.05, 0.05) in cells
    assert (-0.40, 1.40) in cells and (0.40, 1.40) in cells
    assert len(cells) == 2 * len(h.SYMMETRY_GRID_XF_NODES) * len(nodes)


# --- P2 external-data pins (the committed tables and the run record) -------------


@pytest.mark.external_data
def test_seagull_support_record_pins():
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    spectra, integrated = h.load_pp_tables()
    rec = h.seagull_node_support(spectra, integrated, "pi-")["nodes"]
    # The pre-registration's recorded reproduction at xF=0 (-0.21%).
    n0 = rec["0.0"]
    assert n0["published_mean_pt"] == 0.2579
    assert n0["kernel_mean_pt"] == pytest.approx(0.2574, abs=5e-4)
    assert n0["reconstruction_deviation"] == pytest.approx(-0.0021, abs=5e-4)
    assert n0["published_err_abs"] == pytest.approx(
        0.06 / 100.0 * 0.2579, rel=1e-12)
    # The truncated column at xF=0.075 starts at pT=0.40 and its
    # reconstruction is far off (+51%): still a supported node, the
    # deviation is the recorded reconstruction limitation.
    n075 = rec["0.075"]
    assert n075["n_cells"] == 12 and n075["pT_min"] == 0.40
    assert n075["reconstruction_deviation"] == pytest.approx(0.5154, abs=2e-3)
    assert n075["supported"] is True
    # pi+ carries no 2D columns at its forward ptintegrated nodes.
    rec_p = h.seagull_node_support(spectra, integrated, "pi+")["nodes"]
    assert rec_p["0.45"]["supported"] is True
    for node in ("0.65", "0.75", "0.85"):
        assert rec_p[node]["n_cells"] == 0
        assert rec_p[node]["supported"] is False
    # The coarse-grid species carry the recorded 3-8% scale at xF=0.
    rec_k = h.seagull_node_support(spectra, integrated, "k+")["nodes"]["0.0"]
    assert -0.05 < rec_k["reconstruction_deviation"] < -0.02
    rec_b = h.seagull_node_support(spectra, integrated, "pbar")["nodes"]["0.0"]
    assert 0.04 < rec_b["reconstruction_deviation"] < 0.08


@pytest.mark.external_data
def test_cells_manifest_matches_a_fresh_build(tmp_path):
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    stored_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "cells_manifest.json"
    if not stored_path.exists():
        pytest.skip("run manifest not present")
    h.build_cells_manifest(
        h.MIXING2006_DIR / "config" / "staged_excitation.json",
        h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json",
        tmp_path)
    fresh = json.loads((tmp_path / "cells_manifest.json").read_text(
        encoding="utf-8"))
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    # Each manifest's embedded sha verifies against its own body.
    for manifest in (fresh, stored):
        sha = manifest.pop("manifest_sha256")
        body = {k: v for k, v in manifest.items()}
        assert h.canonical_json_sha256(body) == sha
    # Everything except the volatile provenance timestamp is identical.
    fresh.pop("provenance", None)
    stored.pop("provenance", None)
    assert fresh == stored


@pytest.mark.external_data
def test_cells_manifest_registry_gating():
    if not h.spshadrons_data_dir().is_dir():
        pytest.skip("spshadrons mirror not present")
    path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "cells_manifest.json"
    if not path.exists():
        pytest.skip("run manifest not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prereg = json.loads(
        (h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json")
        .read_text(encoding="utf-8"))
    registry = prereg["protocol_A"]["registered_and_dropped_windows"]
    registered = []
    for win_key, entry in manifest["ratio_windows"].items():
        key, win_name = win_key.split("|", 1)
        reg = registry[key][win_name]
        assert entry["status"] == reg["status"]
        assert entry["predictive_content"] == reg["predictive_content"]
        if reg["status"] == "registered":
            assert entry["cells"]
            registered.append(win_key)
        else:
            assert not entry.get("cells")
            assert entry["drop_reason"]
    # Exactly the pre-registered verdict set: three pi-/pi+, three p/pi-,
    # three p/pi+, one k+/k-, one p/pbar.
    assert len(registered) == 11
    # The dropped windows carry their pre-registered statuses.
    assert manifest["ratio_windows"][
        "k+_over_k-|high_pT_gt_1.5"]["status"] == "dropped_single_pT_label"
    assert manifest["ratio_windows"][
        "p_over_pbar|negative_xF_out_of_windows"][
        "status"] == "dropped_outside_registered_windows"
    # The seagull support records agree with a fresh recomputation.
    spectra, integrated = h.load_pp_tables()
    for sp in h.PP_TABLE_NAMES:
        fresh = h.seagull_node_support(spectra, integrated, sp)["nodes"]
        assert fresh == manifest["seagull"][sp]["nodes"]


# --- P3 production path (workstream B) --------------------------------------------


def test_p3_constants():
    assert h.PHASE_P3 == "P3"
    assert h.PA_REPLICA_SCHEMA == "staged_heldout_ratios_pa_replica_pa/v1"
    assert h.PPREF_REPLICA_SCHEMA == \
        "staged_heldout_ratios_pa_replica_ppref/v1"
    assert h.PA_MANIFEST_SCHEMA == \
        "staged_heldout_ratios_pa_pa_cells_manifest/v1"
    assert h.PA_REPORT_SCHEMA == \
        "staged_heldout_ratios_pa_pa_trends_report/v1"
    assert h.PA_MANIFEST_NAME == "pa_cells_manifest.json"
    assert h.PILOT_RECORD_NAME == "pilot_pa_pins.json"
    assert h.PA_DETERMINISM_NAME == "probe_determinism_production_pa.json"
    assert h.PA_SYSTEMS == ("pPb208", "C12")
    assert h.PA_SYSTEM_GEOMETRY["pPb208"] == h.GEOMETRY_PB208
    assert h.PA_SYSTEM_GEOMETRY["C12"] == h.GEOMETRY_C12
    assert h.PA_SYSTEM_DATA_KEY == {"pPb208": "pPb", "C12": "pC"}
    assert h.PA_F_SPECIES["pPb208"] == ("pi-", "pi+")
    assert h.PA_F_SPECIES["C12"] == ("pi-", "pi+", "p", "pbar")
    assert h.PPREF_SPECIES == ("pi-", "pi+", "p", "pbar")
    assert h.PPB_F_PT_NODE_MIN == 1e-9
    assert h.FALSIFIER4_FORWARD_XF_MIN == 0.2
    assert h.P3_DEVIATIONS == []
    # The foundations section 5 regression pins, verbatim.
    assert h.RARE_SLICE_KS == (21, 24, 25)
    assert h.RARE_SLICE_FRACTIONS == {21: 5.6e-7, 24: 1.2e-8, 25: 3.1e-9}
    assert h.A_SCALING_XF_NODES == (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def test_grid_edges_reproduce_the_pinned_half_widths():
    # The standard-grid coincidence pin: on the 0.05-spaced xF grid the
    # table's own midpoint edges equal the pinned half-width windows. The
    # pT grid coincides only away from the 0.30 knot (the pinned rule
    # keeps 0.025 half-width at the node 0.30 while the midpoint edge to
    # the 0.40 neighbor is 0.05), so the p+A cell sets pin the pinned
    # half-widths and record the coincidence flags.
    std_xf = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    assert h._grid_matches_pinned(std_xf, h.cell_xf_half_width)
    std_pt = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    assert not h._grid_matches_pinned(std_pt, h.cell_pt_half_width)
    # Away from the knot the midpoint edges equal the pinned windows.
    soft_pt = [0.05, 0.10, 0.15, 0.20, 0.25]
    assert h._grid_matches_pinned(soft_pt, h.cell_pt_half_width)
    # A non-standard grid (uneven spacings) does not.
    assert not h._grid_matches_pinned([0.0, 0.025, 0.07],
                                      h.cell_xf_half_width)


def test_column_pT_range_rules():
    # The pinned rule on the standard pT grid: outermost pinned windows.
    lo, hi = h.column_pT_range([0.05, 0.10, 0.30, 2.1], h.PA_WIDTH_RULE_PINNED)
    assert lo == 0.025 and hi == 2.15
    # The table's own midpoint rule on a variable grid.
    lo, hi = h.column_pT_range([0.2, 0.3, 0.5], h.PA_WIDTH_RULE_OWN)
    assert lo == 0.15 and hi == 0.6


def test_bin_variable_grid_exact_binning():
    cells = [
        {"lo_xF": -0.05, "hi_xF": -0.025, "lo_pT": 0.3, "hi_pT": 0.5},
        {"lo_xF": -0.025, "hi_xF": 0.0, "lo_pT": 0.3, "hi_pT": 0.5},
        {"lo_xF": 0.0, "hi_xF": 0.025, "lo_pT": 0.1, "hi_pT": 0.2},
    ]
    xf = np.asarray([-0.03, -0.03, -0.024, 0.01, 0.5])
    pt = np.asarray([0.35, 0.6, 0.4, 0.15, 0.2])
    counts, assign = h._bin_variable_grid(xf, pt, cells)
    assert counts.tolist() == [1, 1, 1]
    assert assign.tolist() == [0, -1, 1, 2, -1]


def _lab_frame_event(particles):
    """A stub p+A event in the laboratory frame (the builder's frame
    record with the 158 GeV/c NN CM boost); ``particles`` are
    (pdg, px, py, pz_cm, mass) tracks given by their CM p_z and boosted
    into the lab."""
    if str(MIXING2006) not in sys.path:
        sys.path.insert(0, str(MIXING2006))
    from na49_mixing.event_builder_pA import nn_cm_boost

    beta, gamma, sqrt_s = nn_cm_boost(h.BEAM_MOMENTUM_GEV_C)
    final = []
    for pdg, px, py, pz_cm, mass in particles:
        e_cm = math.sqrt(mass * mass + px * px + py * py + pz_cm * pz_cm)
        p4 = types.SimpleNamespace(e=gamma * (e_cm + beta * pz_cm), px=px,
                                   py=py, pz=gamma * (pz_cm + beta * e_cm))
        final.append(types.SimpleNamespace(pdg_id=pdg, p4=p4))
    return types.SimpleNamespace(
        final_particles=final, stage="baryon_nucleus",
        metadata={"target_wounds": 1, "frame": "lab", "nn_cm_beta": beta,
                  "nn_cm_gamma": gamma, "sqrt_s_nn": sqrt_s})


def test_particle_xf_is_the_nn_cm_xf_of_lab_frame_pa_events():
    # Card TASK-20261005-pa-frame-nn-cm-xf: the p+A builder emits lab-frame
    # events; x_F is 2 p_z^CM / sqrt(s_NN) after the NN CM boost.
    assert h.SQRT_S_GEV == 17.270205201372406
    m_pi = h.SPECIES_MASS_GEV["pi-"]
    pz_02 = 0.2 * h.SQRT_S_GEV / 2.0
    event = _lab_frame_event([(-211, 0.1, 0.0, 0.0, m_pi),
                              (-211, 0.0, 0.3, pz_02, m_pi),
                              (211, 0.0, 0.2, -pz_02, m_pi)])
    xs = [h.particle_xf(p, event=event) for p in event.final_particles]
    assert abs(xs[0]) < 1e-12
    assert abs(xs[1] - 0.2) < 1e-12
    assert abs(xs[2] + 0.2) < 1e-12
    # The withdrawn lab-frame ratio of the CM-rest pion is far from 0.
    raw = 2.0 * event.final_particles[0].p4.pz / h.SQRT_S_GEV
    assert raw > 0.1
    arr = h._species_arrays([event], -211)
    assert np.allclose(arr["xf"], [0.0, 0.2], atol=1e-12, rtol=0.0)
    assert arr["pt"].tolist() == [0.1, 0.3]
    counts, overflow = h.bin_species_counts(
        [event], [(0.0, 0.10), (0.2, 0.30)], -211)
    assert counts.tolist() == [1, 1] and overflow == 0


def test_particle_xf_of_cm_events_is_bit_for_bit_and_needs_the_event():
    rng = np.random.default_rng(5)
    pzs = rng.uniform(-8.0, 8.0, size=50)
    event = _stub_event([(-211, 0.0, 0.1, float(pz)) for pz in pzs])
    for particle, pz in zip(event.final_particles, pzs):
        assert (h.particle_xf(particle, event=event)
                == 2.0 * float(pz) / float(h.SQRT_S_GEV))
    with pytest.raises(TypeError):
        h.particle_xf(event.final_particles[0])


def test_species_arrays_structure():
    pz_02 = 0.2 * h.SQRT_S_GEV / 2.0
    events = [_stub_event([(-211, 0.0, 0.05, 0.0),
                           (211, 0.0, 0.2, pz_02)]),
              _stub_event([(-211, 0.0, 0.3, pz_02)])]
    arr = h._species_arrays(events, -211)
    assert arr["xf"].tolist() == [0.0, 0.2]
    assert arr["pt"].tolist() == [0.05, 0.3]
    assert arr["ev"].tolist() == [0, 1]


def test_pa_falsifier4_cross_reference_branches():
    def r_node(xF, z, model, data):
        return {"xF": xF, "delta_over_sigma_combined": z,
                "model_R": model, "data_R": data}

    r_xf = {"pPb208": {"pi-": [r_node(0.1, 0.5, 0.5, 0.5),
                               r_node(0.3, 7.0, 1.0, 0.3)],
                      "pi+": [r_node(0.2, -7.0, 0.1, 0.4)]},
            "C12": {"pi-": [r_node(0.1, 8.0, 2.0, 0.9)]}}
    a_scaling = {"pi-": {"nodes": [
        {"xF": 0.1, "ordering_z": 0.5},
        {"xF": 0.2, "ordering_z": 6.0},
    ]}}
    out = h.pa_falsifier4_cross_reference(r_xf, a_scaling)
    # Projectile stopping: the pPb forward node at z=+7 (xF >= 0.2) fires;
    # the C12 node is the wrong system; the pPb pi+ node is below the data.
    assert out["projectile_stopping_nodes"] == [
        {"species": "pi-", "xF": 0.3, "z": 7.0, "model_R": 1.0,
         "data_R": 0.3}]
    assert out["inverted_ordering_nodes"] == [
        {"species": "pi-", "xF": 0.2, "ordering_z": 6.0}]
    assert out["fired"] is True
    quiet = h.pa_falsifier4_cross_reference(
        {"pPb208": {"pi-": [r_node(0.3, 1.0, 0.5, 0.3)]}},
        {"pi-": {"nodes": [{"xF": 0.1, "ordering_z": 1.0}]}})
    assert quiet["fired"] is False
    assert "does not block the land gate" in out["policy"]


@pytest.mark.external_data
def test_p3_r_node_entries_pins():
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    geo = h.geometry_anchors()
    entries = h._r_node_entries("pPb208", "pi-",
                                geo["Pb208"]["mean_target_wounds"])
    assert len(entries) == 8
    first = entries[0]
    assert first["xF"] == 0.0 and first["supported"] is True
    assert first["pT_lo"] == 0.025 and first["pT_hi"] == 2.15
    assert first["data_R_samplerK"] == pytest.approx(0.7981, abs=1e-3)
    entries_c = h._r_node_entries("C12", "pi-", geo["C12"]["mean_target_wounds"])
    assert len(entries_c) == 11
    assert entries_c[-1]["xF"] == 0.3
    # The R plan mirrors the pre-registered recomputed rows exactly.
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    for system, data_key in (("pPb208", "pPb"), ("C12", "pC")):
        sys_e = h.PA_F_SPECIES[system]
        for sp in sys_e:
            fresh = h._r_node_entries(
                system, sp,
                geo["Pb208" if system == "pPb208" else "C12"][
                    "mean_target_wounds"])
            h._refuse_r_rows_vs_prereg(system, sp, fresh, prereg)


@pytest.mark.external_data
def test_pa_r_data_error_pins():
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    # p+Pb: the converted table carries absolute errors; the p+p mirror
    # percent errors; the combined relative error times R_samplerK.
    geo = h.geometry_anchors()
    entries = h._r_node_entries("pPb208", "pi-",
                                geo["Pb208"]["mean_target_wounds"])
    err = h._pa_r_data_error(entries[0], "pPb")
    # The p+p dN/dxF error at xF=0 is 0.13 percent.
    expected = (math.sqrt((0.8294 / 37.7) ** 2 + (0.13 / 100.0) ** 2)
                * entries[0]["data_R_samplerK"])
    assert err == pytest.approx(expected, rel=1e-12)
    # The p+C side converts the percent error of the mirror table:
# relative errors 0.3% (p+C) and 0.13% (p+p) in quadrature, times R.
    entries_pc = h._r_node_entries("C12", "pi-", geo["C12"][
        "mean_target_wounds"])
    err_pc = h._pa_r_data_error(entries_pc[0], "pC")
    expected_pc = (math.sqrt((0.3 / 100.0) ** 2 + (0.13 / 100.0) ** 2)
                   * entries_pc[0]["data_R_samplerK"])
    assert err_pc == pytest.approx(expected_pc, rel=1e-12)


def _p3_synthetic_pilot():
    return {
        "schema": h.PILOT_SCHEMA,
        "probe": "pilot_pins",
        "all_pins_hold": True,
        "c12_spec_pin": {"holds": True},
        "rare_slice_pins": [{"k": k, "holds": True} for k in h.RARE_SLICE_KS],
        "point_target_pin": {"holds": True},
        "model_backward_xF_support": {
            "pPb208": {"min_pion_xF": -0.3, "n_events": 50},
            "C12": {"min_pion_xF": -0.3, "n_events": 50},
            "statement": "synthetic",
        },
        "determinism_re_assertion": {"deterministic": True},
        "p1_expectation_re_assertions": {"holds": True},
    }


def test_pa_manifest_refuses_missing_or_failing_pilot(tmp_path,
                                                      monkeypatch):
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    config = h.MIXING2006_DIR / "config" / "staged_excitation.json"
    # Missing pilot record.
    with pytest.raises(SystemExit, match="pilot pin record not found"):
        h.build_pa_cells_manifest(config, prereg_path, tmp_path,
                                  pilot_record_path=tmp_path / "nope.json")
    # A pilot record whose pins do not hold.
    pilot = _p3_synthetic_pilot()
    pilot["all_pins_hold"] = False
    p = tmp_path / h.PILOT_RECORD_NAME
    p.write_text(json.dumps(pilot))
    with pytest.raises(SystemExit, match="does not hold"):
        h.build_pa_cells_manifest(config, prereg_path, tmp_path,
                                  pilot_record_path=p)


@pytest.mark.external_data
def test_pa_manifest_builds_against_the_tables(tmp_path):
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    pilot = _p3_synthetic_pilot()
    # The synthetic support carries backward support, so the restriction
    # does not fire (mirroring the recorded pilot outcome).
    p = tmp_path / h.PILOT_RECORD_NAME
    p.write_text(json.dumps(pilot))
    manifest = h.build_pa_cells_manifest(
        h.MIXING2006_DIR / "config" / "staged_excitation.json",
        prereg_path, tmp_path, pilot_record_path=p)
    assert manifest["manifest_sha256"] == h.canonical_json_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})
    pPb = manifest["systems"]["pPb208"]["cell_sets"]
    assert pPb["density_pi-"]["n_cells"] == 155
    assert pPb["f_pi-"]["n_cells"] == 145
    assert pPb["f_pi-"]["n_cells_excluded_pT0"] == 9
    assert pPb["density_pi-"]["grid_coincides_with_pinned_xF"] is True
    C12 = manifest["systems"]["C12"]["cell_sets"]
    assert C12["f_pi-"]["n_cells"] == 270
    assert manifest["pC_xF_restriction"]["restricted"] is False
    # The registered R-node registry: 8 pPb rows per charge, 11 pC pi-,
    # 22 pC p, 7 pC pbar, plus the pp reference side.
    order = manifest["r_node_plan"]["registered_node_order"]
    systems = [row[0] for row in order]
    assert systems.count("pPb208") == 16
    assert systems.count("C12") == 51
    assert systems.count("ppref") == 53
    # Every pPb supported node carries a common model window.
    for entry in manifest["systems"]["pPb208"]["r_nodes"]["pi-"]:
        if entry["supported"]:
            assert entry["model_window"] is not None
            assert entry["model_window"][0] < entry["model_window"][1]


def _pa_stub_event(particles, wounds=2):
    """A p+A-shaped stub event: sources with roles and quantum numbers,
    conservation residuals, no spectators, and the wound metadata.

    Particles are (pdg, pt, pz) tuples; xF follows from pz through the
    pinned convention.
    """
    sys.path.insert(0, str(h.MIXING2006_DIR))
    from na49_mixing.quantum_numbers import QuantumNumbers, quantum_numbers

    def qn(pdg):
        return quantum_numbers(pdg)

    sources = [
        types.SimpleNamespace(
            source_id=0, role="projectile", mass=3.0,
            quantum_numbers=QuantumNumbers(1, 1, 0),
            metadata={"sampled_mass": 3.0, "n_encounters": wounds,
                      "history": []}),
    ]
    for index in range(1, wounds + 1):
        sources.append(types.SimpleNamespace(
            source_id=index, role="target", mass=2.5,
            quantum_numbers=QuantumNumbers(1, 1, 0),
            metadata={"sampled_mass": 2.5, "target_pdg": 2212, "t": 0.5}))
    final = []
    for pdg, pt, pz in particles:
        final.append(types.SimpleNamespace(
            pdg_id=pdg,
            p4=types.SimpleNamespace(px=0.0, py=pt, pz=pz),
            source_id=0,
            quantum_numbers=qn(pdg)))
    return types.SimpleNamespace(
        event_id=0,
        conservation_residuals=lambda: {"relative_max": 1e-16},
        sources=sources,
        spectators=[],
        final_particles=final,
        metadata={"target_wounds": wounds},
    )


def test_pa_replica_payload_on_synthetic_events(tmp_path, monkeypatch):
    manifest, manifest_path = _synthetic_pa_manifest(tmp_path)
    monkeypatch.setattr(h, "cross_check_m8_record",
                        lambda *a, **k: {"verified": True})
    monkeypatch.setattr(h, "canonical_configuration_hash",
                        lambda p: "x")
    events = [_pa_stub_event([(-211, 0.4, 0.0),
                              (211, 0.4, 0.0),
                              (2212, 0.2, 3.0)], wounds=2)]
    payload = h.build_pa_replica_payload(
        20343832, tmp_path / "config.json", manifest_path, "C12",
        events=events)
    assert payload["n_events"] == 1
    cons = payload["conservation"]
    # The stub carries one pi-, one pi+, one proton: B = 3, Q = 2 (the
    # produced particles) plus no spectators; the accounting rule for C12
    # demands B = 13, Q = 7, so the synthetic event is recorded as a
    # violation (the test pins the accounting bookkeeping, not physics).
    assert cons["accounting_rule"].startswith(
        "spec 7.1 test 8: total event baryon number 13")
    assert cons["accounting_violations"] == 1
    assert cons["n_spectators_pooled"] == 0
    assert cons["source_quantum_number_violations"] == 0
    assert payload["wounds"]["mean"] == 2.0
    # The cell counts: the single p+C f cell set carries one placed pi-.
    assert payload["cell_counts"]["f_pi-"] == [1]
    # The R node per-event record: one pi- inside the xF=0.0 window
    # (the pion at pt 0.4 sits inside the common pT window).
    arr = payload["species_binning"]["pi-"]["r_node_per_event"]
    assert "0" in arr
    assert sum(arr["0"]) == 1


def _synthetic_pa_manifest(tmp_path):
    """A minimal p+A cells manifest covering one p+C f cell set and R
    nodes, with the embedded sha."""
    manifest = {
        "schema": h.PA_MANIFEST_SCHEMA,
        "card": h.CARD, "milestone": h.MILESTONE, "phase": h.PHASE_P3,
        "run_description": h.RUN_DESCRIPTION,
        "provenance": {"configuration_hash": "x"},
        "m8_crosscheck": {"verified": True},
        "binning_statement": "synthetic",
        "model_backward_xF_support": {"pPb208": None, "C12": None,
                                      "statement": "synthetic"},
        "pC_xF_restriction": {"restricted": False, "min_xF": None},
        "a_scaling": {"xF_nodes": list(h.A_SCALING_XF_NODES)},
        "r_node_plan": {"definition": "synthetic",
                        "registered_node_order": []},
        "pooled_yield_kernel": "synthetic",
        "systems": {
            "C12": {
                "nucleus": dict(h.GEOMETRY_C12),
                "stage_label": h.STAGE_LABEL_PA,
                "runner_settings": dict(h.PA_RUNNER_SETTINGS),
                "analytic_mean_target_wounds": 1.903,
                "cell_sets": {},
                "r_nodes": {
                    "pi-": [
                        {"xF": 0.0, "xF_half_width": 0.025,
                         "supported": True,
                         "support_source": "pC_f_table",
                         "pT_lo": 0.025, "pT_hi": 1.9,
                         "n_pT_cells": 2,
                         "model_window": [0.025, 1.9],
                         "data_R_samplerK": 0.985,
                         "data_R_measuredAnchor": 1.063,
                         "anchor_deviation": -0.0734,
                         "data_dn_dxF_pa": 27.364,
                         "data_dn_dxF_err_raw": 0.3,
                         "data_dn_dxF_pp": 19.136},
                    {"xF": 0.05, "supported": False,
                     "reason": "synthetic unsupported"},
                ],
            },
        },
    }
    }
    manifest["systems"]["C12"]["cell_sets"]["f_pi-"] = {        "species": "pi-",
        "width_rule": h.PA_WIDTH_RULE_OWN,
        "n_cells": 1,
        "cells": [{"xF": 0.0, "pT": 0.4, "lo_xF": -0.0125, "hi_xF": 0.0125,
                   "lo_pT": 0.35, "hi_pT": 0.45, "dxf": 0.025, "dpt": 0.1,
                   "data_value": 100.0, "data_err": 5.0}],
        "anchor_mb": h.SIGMA_INEL_PC_MB,
    }
    manifest["manifest_sha256"] = h.canonical_json_sha256(manifest)
    path = tmp_path / "pa_cells_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest, path


def test_produce_pa_replica_deterministic_payload(tmp_path, monkeypatch):
    manifest, manifest_path = _synthetic_pa_manifest(tmp_path)
    monkeypatch.setattr(h, "cross_check_m8_record",
                        lambda *a, **k: {"verified": True})
    monkeypatch.setattr(h, "canonical_configuration_hash",
                        lambda p: "x")
    events = [_pa_stub_event([(-211, 0.4, 0.0)]),
              _pa_stub_event([(-211, 0.6, 0.0)])]
    a = h.build_pa_replica_payload(20343832, tmp_path / "c.json",
                                   manifest_path, "C12", events=events)
    b = h.build_pa_replica_payload(20343832, tmp_path / "c.json",
                                   manifest_path, "C12", events=events)
    canon = lambda p: json.dumps(  # noqa: E731
        h._to_jsonable(p), sort_keys=True, separators=(",", ":"))
    assert canon(a) == canon(b)
    assert a["provenance"]["system"] == "C12"
    assert a["provenance"]["phase"] == h.PHASE_P3
    assert a["schema"] == h.PA_REPLICA_SCHEMA
    assert a["wounds"]["mean"] == 2.0


def test_pa_pilot_pin_constants_match_the_foundations():
    # The foundations section 5 rare-slice fractions, verbatim.
    assert h.RARE_SLICE_FRACTIONS[21] == 5.6e-7
    assert h.RARE_SLICE_FRACTIONS[24] == 1.2e-8
    assert h.RARE_SLICE_FRACTIONS[25] == 3.1e-9
    assert h.RARE_SLICE_SEED_BASE == 20260826
    assert h.POINT_TARGET_SEED == 123456789


@pytest.mark.external_data
def test_pa_pilot_pins_run(tmp_path):
    """The pilot runs at the recorded pins (the full pin battery at the
    pilot-scale statistics, in a scratch directory)."""
    if not (h.spshadrons_data_dir().is_dir() and h.pa_data_dir().is_dir()):
        pytest.skip("tables not present")
    prereg_path = h.RUNS_ROOT / h.RUN_DESCRIPTION / "pre_registration.json"
    if not prereg_path.exists():
        pytest.skip("run directory not present")
    # The pilot's determinism unit is the pre-registered scale; run it
    # into the scratch directory and re-assert the expectations.
    import shutil
    (tmp_path / "probe_cost_cells.json").write_text(
        (h.RUNS_ROOT / h.RUN_DESCRIPTION / "probe_cost_cells.json")
        .read_text(encoding="utf-8"))
    (tmp_path / "probe_determinism.json").write_text(
        (h.RUNS_ROOT / h.RUN_DESCRIPTION / "probe_determinism.json")
        .read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    record = h.run_pilot_pins(tmp_path, h.MIXING2006_DIR / "config" /
                              "staged_excitation.json", prereg)
    assert record["all_pins_hold"] is True
    assert record["c12_spec_pin"]["holds"] is True
    assert all(r["holds"] for r in record["rare_slice_pins"])
    assert record["point_target_pin"]["streams_bit_identical"] is True


def test_seagull_closed_form_test_stays_external_data():
    # Archive self-containment regression (F1): the seagull closed-form
    # test reads the mirror tables transitively through
    # load_pp_tables(), so it must keep the external_data marker that
    # makes hermetic checkouts deselect it (and the mirror-absent skip
    # guard). On a live tree with the mirror present a lost marker is
    # silent, so the pin is behavioral: under the hermetic selection a
    # collection query must not collect the test, while an unmarked
    # sibling in this file must (positive control).
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-m", "not external_data", str(Path(__file__).resolve())],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-1000:]
    collected = {
        line.split("::")[-1]
        for line in r.stdout.splitlines()
        if "::" in line
    }
    assert "test_model_side_seagull_uses_the_data_side_closed_form" not in collected
    assert "test_seagull_weighting_favors_higher_pT_than_unweighted_mean" in collected
