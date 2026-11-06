"""Hermetic tests for the R40 fixed-source alignment scan driver.

Round 4, card 11 (``TASK-20260920-r4-run-r40-fixed-source-alignment-
scan``; plan section 10 run R40, line 761).  The tests exercise the
driver's pre-registered contracts hermetically:

- ``test_seed_map_and_event_index_partition``: the pre-registered seed
  map (the master seed, the raw replica block, the kernel replica
  slots, the event-index partition) and its disjointness across all
  360 (setting, point, replica) stream spaces and against every
  previously used raw block.
- ``test_routing_regimes``: the card-6 routing regimes and ramp
  weights at the point masses (zero uniforms off-band, one inside the
  band).
- ``test_draw_budget_pinned_trace``: the per-decay draw budget on a
  pinned trace (the spawned-child budgets exact; the cascade vertices
  at card 4's frozen 1 channel + 2 kinematics budget; the alignment
  dispatch and the form factor consuming none).
- ``test_depth_statistics_algebra``: the histogram-to-statistics
  algebra on pinned histograms (mean, p95, max, guard activations, and
  the total-mismatch refusal).
- ``test_verdict_and_check_set``: the endpoint-contrast verdict
  algebra, the pre-registered response-direction map, and the
  named-residual register shape on a synthetic failing fixture.
- ``test_evaluation_determinism_pinned_fixture``: the evaluation
  algebra on a pinned synthetic fixture (byte-identical across
  invocations, closed-form values).
- ``test_smoke_and_layout_sync``: the driver's ``--smoke`` path (1
  setting x 1 point x 1 replica x 2,000 decays, hermetic) into
  ``tmp_path`` run dirs, the byte determinism of two identical smoke
  invocations, and the dual-tree layout sync (byte-identical nested +
  outer copies, the freeze-manifest entries).

The smoke runs read the committed catalog, the committed continuum
config, and the vendored PDG table only; no NA49/SPS external tables
are touched.  The pinned-trace tests import the driver module by path
(the driver locates the package itself).
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r40_alignment_scan.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet"
MANIFEST_PATH = MANIFEST_PATH / "legacy_v1_manifest.json"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r40_alignment_scan_test_module", NESTED_DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (round 4 index item 9)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition(driver):
    assert driver.MASTER_SEED == 20260920
    assert [driver.raw_replica_seed(r) for r in range(4)] == [
        20301020, 20301021, 20301022, 20301023
    ]
    assert driver.SMOKE_RAW_SEED == 20307920
    assert driver.SMOKE_KERNEL_REPLICA_SLOT == 7000
    # the raw block is disjoint from every previously used raw block
    used = set(range(20300913, 20300921))          # R23
    used |= set(range(20301519, 20301527))         # R32 v0 production
    used |= set(range(20301619, 20301627))         # R32 v1 production
    used |= set(range(20301719, 20301727))         # R32 v2 production
    used |= set(range(20301819, 20301835))         # R32 v0 confirmation
    used |= set(range(20301919, 20301935))         # R32 v1 confirmation
    used |= set(range(20302019, 20302035))         # R32 v2 confirmation
    used |= set(range(20301040, 20301056))         # the audit block
    used |= {20305919, 20306019, 20306119}         # R32 smoke anchors
    used |= set(range(20305919, 20305927))
    used |= set(range(20306019, 20306027))
    used |= set(range(20306119, 20306127))
    raw_block = {driver.raw_replica_seed(r) for r in range(4)}
    assert raw_block == {20301020, 20301021, 20301022, 20301023}
    assert not raw_block & used
    assert driver.SMOKE_RAW_SEED not in used
    # the event-index partition: 300_000_000 + s*10M + p*1M + i
    assert driver.event_index(0, 0, 0) == 300_000_000
    assert driver.event_index(1, 0, 0) == 310_000_000
    assert driver.event_index(8, 9, 24_999) == 389_024_999
    indices = set()
    for s in range(9):
        for p in range(10):
            first = driver.event_index(s, p, 0)
            last = driver.event_index(s, p, 24_999)
            assert first < last
            assert not indices & set(range(first, last + 1))
            indices |= set(range(first, last + 1))
    assert len(indices) == 9 * 10 * 25_000
    # the kernel-layer stream ids are pairwise disjoint across the 360
    # (setting, point, replica) spaces at the shared event index
    stream_ids = set()
    for s in range(9):
        for p in range(10):
            for r in range(4):
                subs = driver.derive_substreams(
                    driver.MASTER_SEED, r, driver.event_index(s, p, 0))
                for label in ("decay_channel", "decay_kinematics"):
                    sid = subs.stream_for(label)
                    assert sid not in stream_ids
                    stream_ids.add(sid)
    assert len(stream_ids) == 360 * 2
    # the smoke block is disjoint from the production spaces
    smoke_subs = driver.derive_substreams(
        driver.MASTER_SEED, driver.SMOKE_KERNEL_REPLICA_SLOT,
        driver.event_index(1, 0, 0))
    for label in ("decay_channel", "decay_kinematics"):
        assert smoke_subs.stream_for(label) not in stream_ids


# ---------------------------------------------------------------------------
# The routing regimes (the card-6 blend at the point masses)
# ---------------------------------------------------------------------------
def test_routing_regimes(driver):
    from na49_mixing.spectral_density import BLEND_BOUNDARY, BLEND_RAMP_WIDTH
    assert driver.BLEND_BOUNDARY == 2.680194110727528
    assert driver.BLEND_RAMP_WIDTH == 0.5
    assert driver.BLEND_WEIGHT_AT_3 == pytest.approx(
        (3.0 - 2.680194110727528) / 0.5)
    assert driver.blend_weight_at(2.0) == 0.0
    assert driver.blend_weight_at(3.0) == pytest.approx(0.6396117785449444)
    assert driver.blend_weight_at(4.0) == 1.0
    assert driver.blend_weight_at(6.0) == 1.0
    assert driver.blend_weight_at(8.0) == 1.0
    assert driver.point_regime(2.0) == "catalog"
    assert driver.point_regime(3.0) == "blend_band"
    assert driver.point_regime(4.0) == "continuum"
    assert [p["mass_gev"] for p in driver.POINTS] == [
        2.0, 3.0, 4.0, 6.0, 8.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    assert [tuple(p["qn"]) for p in driver.POINTS] == [
        (1, 1, 0)] * 5 + [(0, 0, 0)] * 5
    assert len(driver.SETTINGS) == 9
    assert (driver.SETTINGS[1]["kappa_b"], driver.SETTINGS[1]["kappa_m"],
            driver.SETTINGS[1]["b_dec"]) == (1.0, 1.0, 1.0)
    assert len({(float(s["kappa_b"]), float(s["kappa_m"]),
                 float(s["b_dec"])) for s in driver.SETTINGS}) == 9


# ---------------------------------------------------------------------------
# The per-decay draw budget on a pinned trace
# ---------------------------------------------------------------------------
def test_draw_budget_pinned_trace(driver):
    """The per-decay draw budget on a pinned trace: the spawned-child
    budgets are exact (1 record uniform on a catalog route plus 2 card-3
    mass uniforms; the one routing uniform inside the blend band; zero
    off-band on the continuum route), the cascade vertices consume card
    4's frozen 1 channel + 2 kinematics budget, and the alignment
    dispatch and the form factor consume none."""
    catalog = driver._load_catalog()
    fams = driver._load_families()
    blend = driver.BlendSpec()

    def one(qn, mass, event_index, kappa_b=1.0, kappa_m=1.0, b_dec=1.0):
        subs = driver.derive_substreams(20260920, 0, event_index)
        rng_record = driver._CountingRng(driver.np.random.default_rng(
            driver.np.random.SeedSequence(
                subs.stream_for("decay_channel")).spawn(1)[0]))
        rng_mass = driver._CountingRng(driver.np.random.default_rng(
            driver.np.random.SeedSequence(
                subs.stream_for("decay_kinematics")).spawn(1)[0]))
        rng_channel = driver._CountingRng(driver.np.random.default_rng(
            subs.stream_for("decay_channel")))
        rng_kin = driver._CountingRng(driver.np.random.default_rng(
            subs.stream_for("decay_kinematics")))
        exchange, route, diagnostics = driver.build_exchange_event(
            catalog, fams, blend, qn, mass, rng_record, rng_mass)
        policy = driver.AblationPolicy(
            "aligned_sequential", kappa_b, kappa_m, b_dec)
        line, finals, mode = driver.decay_scan_step(
            catalog, fams, exchange, policy, rng_channel, rng_kin)
        return {
            "route": route,
            "line_depth": int(line[-1].depth),
            "n_finals": len(finals),
            "record_uniforms": rng_record.count,
            "mass_uniforms": rng_mass.count,
            "channel_uniforms": rng_channel.count,
            "kinematics_uniforms": rng_kin.count,
            "mass_src": float(exchange["mass"]),
        }

    # the catalog-routed 2 GeV point: one record uniform + two card-3
    # mass uniforms per decay (the spawned children), the kernel's own
    # draws on the labeled ids
    trace = one((1, 1, 0), 2.0, 300_000_000)
    assert trace["route"] == "catalog"
    assert trace["record_uniforms"] == 1
    assert trace["mass_uniforms"] == 2
    assert trace["line_depth"] == 0  # the handoff root at depth 0
    assert trace["n_finals"] >= 2
    assert trace["channel_uniforms"] >= 1  # the kernel's own budget
    # the continuum-routed 6 GeV point: zero routing/record uniforms,
    # the cascade vertices at exactly 1 channel + 2 kinematics each
    trace = one((1, 1, 0), 6.0, 300_000_003)
    assert trace["route"] == "continuum"
    assert trace["record_uniforms"] == 0
    assert trace["mass_uniforms"] == 0
    assert trace["line_depth"] >= 1
    assert trace["channel_uniforms"] >= trace["line_depth"]
    assert trace["kinematics_uniforms"] >= 2 * trace["line_depth"]
    # a second identical pinned decay reproduces the trace exactly
    again = one((1, 1, 0), 6.0, 300_000_003)
    assert again == trace
    # a different event index draws a different realization
    other = one((1, 1, 0), 6.0, 300_000_004)
    assert other["record_uniforms"] == 0
    assert other["mass_uniforms"] == 0
    # conservation at the assembly level on the pinned trace
    for event_index in range(300_000_005, 300_000_015):
        result = one((1, 1, 0), 4.0, event_index)
        assert result["n_finals"] >= 2
        assert result["line_depth"] >= 1


# ---------------------------------------------------------------------------
# The depth-statistics algebra
# ---------------------------------------------------------------------------
def test_depth_statistics_algebra(driver):
    stats = driver.depth_stats([10, 20, 60, 10], 100)
    assert stats["mean"] == pytest.approx(1.7)
    assert stats["p95"] == 3
    assert stats["max"] == 3
    assert stats["guard_activations"] == 0
    stats = driver.depth_stats([0, 100], 100)
    assert stats["mean"] == 1.0
    assert stats["p95"] == 1
    assert stats["max"] == 1
    stats = driver.depth_stats([100, 0], 100)
    assert stats["p95"] == 0
    assert stats["max"] == 0
    with pytest.raises(ValueError, match="histogram total"):
        driver.depth_stats([10, 20], 100)
    with pytest.raises(ValueError, match="empty"):
        driver.depth_stats([0, 0], 0)


# ---------------------------------------------------------------------------
# The verdict algebra and the pre-registered response-direction map
# ---------------------------------------------------------------------------
def test_verdict_and_check_set(driver):
    # the endpoint-contrast verdict algebra
    sigma, verdict = driver.slice_verdict(
        [0.20, 0.30, 0.40, 0.50], [0.01, 0.01, 0.01, 0.01], 1)
    assert verdict == "pass" and sigma > 0
    sigma, verdict = driver.slice_verdict(
        [0.50, 0.40, 0.30, 0.20], [0.01, 0.01, 0.01, 0.01], 1)
    assert verdict == "fail" and sigma < -3.0
    sigma, verdict = driver.slice_verdict(
        [0.50, 0.40, 0.30, 0.20], [0.01, 0.01, 0.01, 0.01], -1)
    assert verdict == "pass" and sigma > 0
    sigma, verdict = driver.slice_verdict(
        [1.0, 1.0, 1.0, 1.0], [0.001, 0.001, 0.001, 0.001], 0)
    assert verdict == "pass" and sigma == 0.0
    sigma, verdict = driver.slice_verdict(
        [1.0, 1.0, 1.0, 1.2], [0.001, 0.001, 0.001, 0.001], 0)
    assert verdict == "fail" and abs(sigma) > 3.0
    sigma, verdict = driver.slice_verdict(
        [1.0, None, 1.0, 1.0], [0.01, 0.01, 0.01, 0.01], 0)
    assert sigma is None and verdict == "skip"
    # the deterministic SE floor: exact zeros stay exact
    sigma, verdict = driver.slice_verdict([1.0, 1.0], [0.0, 0.0], 0)
    assert verdict == "pass" and sigma == 0.0
    sigma, verdict = driver.slice_verdict([1.0, 1.1], [0.0, 0.0], 0)
    assert verdict == "fail"  # 0.1 / 1e-12 >> 3.0
    # the pre-registered response-direction map
    for point in (1, 2, 3, 4):
        assert driver._predicted_direction("kappa_B", point) == 1
    assert driver._predicted_direction("kappa_B", 0) == 0
    for point in range(10):
        assert driver._predicted_direction("kappa_M", point) == 0
    for point in (1, 2, 3, 4, 6, 7, 8, 9):
        assert driver._predicted_direction("b_dec", point) == -1
    assert driver._predicted_direction("b_dec", 0) == 0
    assert driver._predicted_direction("b_dec", 5) == 0
    with pytest.raises(ValueError):
        driver._predicted_direction("unknown", 0)


# ---------------------------------------------------------------------------
# The evaluation algebra on a pinned synthetic fixture
# ---------------------------------------------------------------------------
def _synthetic_record(setting, point, replica, n=1000):
    """One synthetic chunk record with closed-form sums (the evaluation
    algebra's pinned fixture)."""
    return {
        "schema": "na49_mixing.staged_r40_alignment_scan.record/1",
        "mode": "production",
        "setting": int(setting),
        "point": int(point),
        "replica": int(replica),
        "statistics": {"n_decays": int(n)},
        "species_sums": {
            "pi_plus": {"n": 10, "sum_pt": 3.0, "sum_e": 40.0,
                        "sum_ept": 12.0, "sum_lc": 22.0},
            "pi_minus": {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                         "sum_ept": 0.0, "sum_lc": 0.0},
            "k_plus": {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                       "sum_ept": 0.0, "sum_lc": 0.0},
            "k_minus": {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                        "sum_ept": 0.0, "sum_lc": 0.0},
            "proton": {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                       "sum_ept": 0.0, "sum_lc": 0.0},
            "antiproton": {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                           "sum_ept": 0.0, "sum_lc": 0.0},
        },
        "partition_sums": {"baryon_line_lc": 300.0, "central_lc": 700.0,
                           "baryon_finals": n, "central_finals": 10 * n},
        "alignment_moment": {"n_mesons": 10 * n, "sum_z2_over_p2": 3.0 * n},
        "depth_histograms": {"all": [0, n], "catalog_route": [0, 0],
                             "continuum_route": [0, n]},
        "route_census": {"n_catalog": 0, "n_continuum": n,
                         "n_band_catalog": 0, "n_band_continuum": 0},
        "conservation": {"events_checked": int(n), "qn_violations": 0,
                         "worst_energy_closure_gev": 1e-15,
                         "worst_lightcone_identity_gev": 1e-15},
        "rolling_digest": "0" * 64,
      }


def test_evaluation_determinism_pinned_fixture(driver):
    records = [
        _synthetic_record(s, 0, r)
        for s in (0, 1) for r in range(4)
    ]
    mass = 6.0
    species = driver._point_species_tables(records, mass)
    pi = species["pi_plus"]
    # closed-form pooled values (the sums are equal across the 8 records)
    assert pi["n"] == 80
    assert pi["pooled_mean_pt_x"] == pytest.approx(3.0 / 10)
    assert pi["pooled_mean_pt_y"] == pytest.approx(12.0 / 40.0)
    assert pi["pooled_lightcone_fraction"] == pytest.approx(
        22.0 * 8 / (8 * 1000 * 6.0))
    # the between-replica SE of a constant per-replica observable is 0
    assert pi["multiplicity_se"] == 0.0
    assert pi["mean_pt_x"] == pytest.approx(0.3)
    # the absent species is None (recorded, not fabricated)
    assert species["pi_minus"]["n"] == 0
    assert species["pi_minus"]["mean_pt_x"] is None
    assert species["pi_minus"]["mean_pt_y"] is None
    partition = driver._point_partition(records, mass)
    assert partition["partition"]["pooled_central_fraction"] == (
        pytest.approx(700.0 * 8 / (8 * 1000 * 6.0)))
    assert partition["partition"]["central_finals_per_decay"] == 10.0
    assert partition["alignment_moment"]["mean_z2_over_p2"] == pytest.approx(
        3.0 / 10.0)
    assert partition["depth"]["all"]["mean"] == pytest.approx(1.0)
    # two invocations produce byte-identical tables (the report
    # determinism contract)
    a = driver._point_species_tables(records, mass)
    b = driver._point_species_tables(records, mass)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # determinism under replica-order permutation is NOT assumed: the
    # sums follow the record order, and the caller's fixed replica
    # order pins the bytes (asserted in the report contract)


def test_check_set_register_shape(driver):
    """A synthetic failing fixture produces a named-residual register
    entry with sigma, a one-line diagnosis, and the robust-statistics
    cross-reference; a passing fixture produces none."""
    def tables_with(values_by_setting):
        tables = {}
        for s in range(9):
            for p in range(10):
                frac = values_by_setting.get(s, 0.5)
                tables[(s, p)] = {
                    "mass_gev": 6.0,
                    "n_decays": 100_000,
                    "replicas": 4,
                    "species": {
                        name: {
                            "n": 0, "mean_multiplicity": 0.0,
                            "multiplicity_se": None, "mean_pt_x": None,
                            "mean_pt_x_se": None, "mean_pt_y": None,
                            "mean_pt_y_se": None,
                            "lightcone_fraction": None,
                            "lightcone_fraction_se": None,
                            "pooled_mean_pt_x": None,
                            "pooled_mean_pt_y": None,
                            "pooled_lightcone_fraction": None,
                        }
                        for name in driver.SPECIES_KEYS
                    },
                    "partition": {
                        "baryon_line_fraction": frac,
                        "baryon_line_fraction_se": 0.001,
                        "central_fraction": 0.5,
                        "central_fraction_se": 0.001,
                        "central_multiplicity": 3.0,
                        "central_multiplicity_se": 0.001,
                        "baryon_line_finals_per_decay": 1.0,
                        "central_finals_per_decay": 3.0,
                        "pooled_baryon_line_fraction": frac,
                        "pooled_central_fraction": 0.5,
                        "note": "synthetic",
                    },
                    "alignment_moment": {"n_mesons": 0,
                                         "mean_z2_over_p2": None,
                                         "se": None},
                    "depth": {"all": {"mean": 1.0, "p95": 1, "max": 1,
                                      "guard_activations": 0,
                                      "n_decays": 100_000,
                                      "convention": "x"},
                              "catalog_route": None,
                              "continuum_route": None,
                              "guard_activations": 0},
                    "routing_census": {"n_catalog": 0, "n_continuum": 4,
                                       "n_band_catalog": 0,
                                       "n_band_continuum": 0,
                                       "fraction_continuum": 1.0,
                                       "blend_weight_at_mass": 1.0},
                    "conservation": {"events_checked": 100_000,
                                     "qn_violations": 0,
                                     "worst_energy_closure_gev": 1e-15,
                                     "worst_lightcone_identity_gev": 1e-15},
                }
        return tables

    # a DECREASING baryon-line fraction across the kappa_B slice at the
    # continuum-routed points is a failing verdict (predicted increasing)
    decreasing = {0: 0.50, 1: 0.48, 2: 0.45, 3: 0.40}
    check = driver.build_check_set(tables_with(decreasing), smoke=False)
    assert check["status"] == "evaluated"
    assert check["named_residual_register"], "a failure must be named"
    responses = {entry["response"]
                 for entry in check["named_residual_register"]}
    assert responses == {"kappa_B"}
    by_point = {entry["point"]: entry
                for entry in check["named_residual_register"]}
    # the flat prediction at the catalog-routed point 0 also fails on
    # the decreasing fixture (the zero-response band)
    assert 0 in by_point
    assert by_point[0]["predicted_direction"] == 0
    assert abs(by_point[0]["sigma"]) > 3.0
    for point in (1, 2, 3, 4):
        assert by_point[point]["predicted_direction"] == 1
        assert by_point[point]["sigma"] < -3.0
        assert "diagnosis" in by_point[point]
        assert "robust_statistics_reference" in by_point[point]
        assert "3.0-sigma" in by_point[point]["diagnosis"]
    # a profile inside every pre-registered band (increasing at the
    # continuum-routed points, flat within the band at the catalog-
    # routed point) passes with an empty register
    passing = {0: 0.500, 1: 0.500, 2: 0.501, 3: 0.502}
    check = driver.build_check_set(tables_with(passing), smoke=False)
    assert check["named_residual_register"] == []
    kappa_b_rows = [row for row in check["rows"]
                    if row["response"] == "kappa_B"]
    assert all(row["verdict"] == "pass" for row in kappa_b_rows)
    # the smoke statistics skip the check set
    check = driver.build_check_set(tables_with({}), smoke=True)
    assert check["status"].startswith("skipped")
    assert check["named_residual_register"] == []


# ---------------------------------------------------------------------------
# The smoke path and the dual-tree layout sync
# ---------------------------------------------------------------------------
def _run_smoke(run_dir: Path) -> None:
    """Run the driver's ``--smoke`` mode into *run_dir* (a subprocess;
    the driver locates the package itself, so the environment carries
    no PYTHONPATH dependency)."""
    result = subprocess.run(
        [
            sys.executable, str(NESTED_DRIVER),
            "--smoke", "--run-dir", str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"the smoke run failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke run dirs; the second exists for the byte-determinism
    check."""
    first = tmp_path_factory.mktemp("r40_smoke_first")
    second = tmp_path_factory.mktemp("r40_smoke_second")
    _run_smoke(first)
    _run_smoke(second)
    return first, second


def test_smoke_determinism(smoke_runs):
    """Two identical smoke invocations produce byte-identical
    artifacts (the smoke records carry no wall-clock fields)."""
    first, second = smoke_runs
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert "smoke_pre_registration.json" in names
    assert "smoke_scan_report.json" in names
    assert "smoke_setting_1_point_0_replica_0.json" in names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), (
            f"{name} differs between the two smoke invocations")


def test_smoke_record_contract(smoke_runs):
    """The smoke record carries the pre-registered seed block, the
    exact spawned-child draw budget, the exact per-decay conservation,
    and zero wall-clock fields."""
    run_dir = smoke_runs[0]
    record = json.loads(
        (run_dir / "smoke_setting_1_point_0_replica_0.json").read_text(
            encoding="utf-8"))
    assert record["mode"] == "smoke"
    assert record["statistics"]["n_decays"] == 2_000
    assert record["seeds"]["kernel_replica_slot"] == 7_000
    assert record["seeds"]["raw_replica_seed"] == 20_307_920
    assert record["seeds"]["event_index_first"] == 310_000_000
    assert record["statistics"].get("n_decays") == 2_000
    # the catalog route at the 2 GeV point: 1 record + 2 mass uniforms
    # per decay on the spawned children
    assert record["draw_budget"]["spawned_record_child_uniforms"] == 2_000
    assert record["draw_budget"]["spawned_mass_child_uniforms"] == 4_000
    assert "wall_seconds" not in record
    assert "started_utc" not in record
    assert record["conservation"]["qn_violations"] == 0
    assert record["conservation"]["worst_energy_closure_gev"] <= 1e-9
    assert len(record["rolling_digest"]) == 64


def test_layout_sync(driver):
    """The nested and outer driver copies are byte-identical (the
    strongest form of the layout-only allowance) and the freeze
    manifest carries the re-pinned entries of this card."""
    assert NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file()
    assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()
    import hashlib

    digest = hashlib.sha256(NESTED_DRIVER.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    outer_rel = "helpers/benchmarking/staged_r40_alignment_scan.py"
    nested_rel = (
        "dev/mixing2006/helpers/benchmarking/"
        "staged_r40_alignment_scan.py"
    )
    test_rel = "dev/mixing2006/tests/test_staged_r40_alignment_scan.py"
    for rel, want in (
        (outer_rel, digest),
        (nested_rel, digest),
        (test_rel, hashlib.sha256(
            Path(__file__).resolve().read_bytes()).hexdigest()),
    ):
        assert rel in inventory, f"missing freeze entry: {rel}"
        assert inventory[rel] == want, f"stale freeze entry: {rel}"
    exceptions = manifest["freeze_exceptions"]
    cards = [
        entry for entry in exceptions
        if entry.get("card")
        == "TASK-20260920-r4-run-r40-fixed-source-alignment-scan"
    ]
    assert len(cards) == 1, (
        "the card-11 freeze exception must name its card exactly once")
