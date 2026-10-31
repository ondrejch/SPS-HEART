"""Hermetic smoke test for the R31 matching-scan driver (Round 3, card 8).

Task ``TASK-20260917-r3-run-r31-catalog-continuum-matching-scan`` (plan
section 9 run R31, line 642).  The tests exercise the card's
deterministic contract against the committed driver and the committed
inputs:

- ``test_grid_pins``: the pre-registered windows, node grid, blend
  boundary and ramp, expected routing weights, and boundary-straddling
  pair indices.
- ``test_seed_map``: the card-9 seed map with the window folded into
  the point index, the closed substream labels, and the disjointness
  of the R31 master seed from the R20-R23 seeds.
- ``test_routing_zero_uniform_below_boundary``: the card-6 routing
  consumes zero uniforms and returns ``'catalog'`` at every
  below-boundary node of every window.
- ``test_routing_transition_pin``: at the transition node 2.78 the
  route is ``'continuum'`` exactly when the first spawned
  ``decay_channel`` uniform is below ``w(M)``, pinned event by event
  on the fixed seed map.
- ``test_smoke_determinism``: two reduced CLI smoke runs with the same
  arguments produce byte-identical node records (a below-boundary
  catalog-only node and a transition node exercising the continuum
  route).
- ``test_smoke_contract``: the smoke records' charge balance is exact,
  the below-boundary node routed everything to the catalog, and the
  transition node's recorded continuum count equals the pinned
  expected count.
- ``test_analytic_engine_determinism``: the continuum analytic engine
  and the card-2 catalog payload reproduce identical bytes on repeated
  evaluation.
- ``test_layout_sync``: the nested and outer driver copies are
  byte-identical and the layout checker passes.
- ``test_production_pre_registration_pins`` (read-only, skips when the
  sealed run dir is absent): the sealed pre-registration carries the
  card's statistics, seeds, routing weights, and the re-registered
  metric.

The smoke runs read the committed catalog
``config/v2_singlet/resonance_catalog.json`` and the committed
continuum config only; no NA49/SPS external tables are touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


#: The sealed R31 engine's species accumulator knows the hadron species of
#: the Round 3 catalog; the kernel's radiative channels (2026-10-02,
#: TASK-20261002-kernel-feedover-threshold-radiative) put photons in the
#: final state, so its hermetic smoke run cannot pass on the current
#: kernel.  The sealed archive tests stay active.
_SEALED_KERNEL_SKIP = pytest.mark.skip(
    reason="the sealed R31 engine predates the radiative channels (photons) of the current kernel")

NESTED_ROOT = Path(__file__).resolve().parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r31_matching_scan.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = NESTED_ROOT.parent.parent / DRIVER_REL
SYNC_CHECKER = NESTED_ROOT.parent.parent / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "continuum_resonances.json"
)
SEALED_RUN_DIR = NESTED_ROOT.parent.parent / "dev" / "00runs" / (
    "r31_catalog_continuum_matching_scan_2026-09-17"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r31_matching_scan_under_test", NESTED_DRIVER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


# ----------------------------------------------------------------------
# Grid, routing, and seed pins
# ----------------------------------------------------------------------
def test_grid_pins(driver):
    """The pre-registered windows, node grid, blend values, expected
    routing weights, and boundary-straddling pair indices."""
    assert [tuple(w["qn"]) for w in driver.WINDOWS] == [
        (1, 1, 0), (1, 0, -1), (0, 0, 0)
    ]
    assert [w["family_expectation"] for w in driver.WINDOWS] == [
        "baryonic_np", "baryonic_strange", "mesonic_neutral"
    ]
    assert driver.N_WINDOWS == 3
    assert driver.N_NODES == 5
    assert driver.NODE_MASSES == [2.48, 2.58, 2.68, 2.78, 2.88]
    assert driver.NODE_START_GEV == 2.48
    assert driver.NODE_STEP_GEV == 0.1
    # The card-6 blend values (the M8-frozen continuum.match_mass and
    # the pre-registered W).
    from na49_mixing.spectral_density import BlendSpec

    spec = BlendSpec()
    assert driver.BOUNDARY == spec.boundary == 2.680194110727528
    assert driver.RAMP_WIDTH == spec.ramp_width == 0.5
    # The expected routing weights: exactly 0 at and below the boundary
    # (node 2.68 lies below B), (M - B)/W inside the transition band.
    expected = [0.0, 0.0, 0.0,
                (2.78 - 2.680194110727528) / 0.5,
                (2.88 - 2.680194110727528) / 0.5]
    assert driver.EXPECTED_W == expected
    assert round(driver.EXPECTED_W[3], 12) == 0.199611778545
    assert round(driver.EXPECTED_W[4], 12) == 0.399611778545
    # The boundary-straddling pair is the third pair (j = 2) of every
    # window: global indices 2, 6, 10.
    assert driver.BOUNDARY_PAIR_INDICES == [2, 6, 10]
    # The T1 statistics and the seed map constants.
    assert driver.T1_REPLICAS == 4
    assert driver.T1_DECAYS_PER_REPLICA == 25_000
    assert driver.TOTAL_DECAYS == 1_500_000
    assert driver.MASTER_SEED == 20260917
    assert driver.SEED_RULE == (
        "derive_substreams(20260917, r, (w*5 + j) * 1_000_000 + i) per "
        "(window w, node j, replica r < 4, decay i < 25,000) on the "
        "closed decay_channel/decay_kinematics labels (the card-9 seed "
        "map)"
    )
    assert driver.METRIC_ID == "delta_analytic_full"
    assert driver.METRIC_N_SIGMA == 3.0
    assert driver.RUN_ID == "R31"
    assert driver.CARD == (
        "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan"
    )
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
    )


def test_seed_map(driver):
    """The per-decay event index is ``(w*5 + j) * 1_000_000 + i`` and
    the two labeled substream ids are the card-9 closed labels; the
    master seed 20260917 is disjoint from the R20-R23 seeds."""
    from na49_mixing.rng_substreams import derive_substreams

    subs = derive_substreams(driver.MASTER_SEED, 0, (0 * 5 + 0) * 1_000_000)
    first_channel = subs.stream_for("decay_channel")
    first_kinematics = subs.stream_for("decay_kinematics")
    # Pinned stream ids of the first decay of the run.
    assert first_channel == 3742437277880976286
    assert first_kinematics == 5234642022647842082
    # The event-index partition: window 1 node 2 replica 3 decay 7.
    subs2 = derive_substreams(driver.MASTER_SEED, 3, (1 * 5 + 2) * 1_000_000 + 7)
    assert subs2.master_seed == 20260917
    assert subs2.replica_index == 3
    assert subs2.event_index == 7_000_007
    # The R31 master seed differs from the R20-R22 master (20260913)
    # and from R23's raw replica seeds (20300913..20300920).
    assert driver.MASTER_SEED != 20260913
    assert driver.MASTER_SEED not in range(20300913, 20300921)


def test_routing_zero_uniform_below_boundary(driver):
    """At every below-boundary node of every window the card-6 routing
    consumes zero uniforms and returns ``'catalog'`` (the generator
    state is bit-unchanged)."""
    from na49_mixing.spectral_density import BlendSpec, families_from_json

    from na49_mixing.hadron_catalog import HadronCatalog

    spec = BlendSpec()
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    families = families_from_json(str(FAMILIES_PATH))
    for spec_window in driver.WINDOWS:
        qn = tuple(int(v) for v in spec_window["qn"])
        for j, mass in enumerate(driver.NODE_MASSES):
            if not mass < driver.BOUNDARY:
                continue
            rng = np.random.default_rng(driver.MASTER_SEED)
            state_before = rng.bit_generator.state
            route = driver.route_source_decay(
                qn, mass, spec, catalog, families, rng
            )
            assert route == "catalog"
            assert rng.bit_generator.state == state_before
            # The driver's own expected weight is exactly zero there.
            assert driver.EXPECTED_W[j] == 0.0


def test_routing_transition_pin(driver):
    """At the transition node 2.78 the route is ``'continuum'`` exactly
    when the first spawned ``decay_channel`` uniform is below ``w(M)``:
    pinned event by event on the fixed seed map (replica 0, the first
    40 decays of window 0 node 3)."""
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.spectral_density import BlendSpec, families_from_json
    from na49_mixing.hadron_catalog import HadronCatalog

    spec = BlendSpec()
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    families = families_from_json(str(FAMILIES_PATH))
    qn = (1, 1, 0)
    mass = 2.78
    w = spec.weight(mass)
    assert 0.0 < w < 1.0
    n_expected_continuum = 0
    for i in range(40):
        subs = derive_substreams(
            driver.MASTER_SEED, 0, (0 * 5 + 3) * 1_000_000 + i
        )
        # The routing uniform is the FIRST draw of the spawned
        # decay_channel child: a probe generator pins it, the routing
        # call consumes its own.
        probe = np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
        )
        u = float(probe.random())
        rng = np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
        )
        route = driver.route_source_decay(qn, mass, spec, catalog, families,
                                          rng)
        assert route == ("continuum" if u < w else "catalog")
        n_expected_continuum += int(route == "continuum")
    # The pinned expected count over the 40 events.
    assert n_expected_continuum == 9


# ----------------------------------------------------------------------
# Reduced CLI smoke runs (hermetic; tmp run dirs)
# ----------------------------------------------------------------------
SMOKE_DECAYS = 40
SMOKE_REPLICAS = 1


def _run_smoke(run_dir: Path, window: int, node: int) -> None:
    result = subprocess.run(
        [
            sys.executable, str(NESTED_DRIVER),
            "--generate", "--smoke",
            "--window", str(window), "--node", str(node),
            "--smoke-decays", str(SMOKE_DECAYS),
            "--smoke-replicas", str(SMOKE_REPLICAS),
            "--workers", "2",
            "--run-dir", str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"the smoke run (window {window} node {node}) failed "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke run dirs per probed node; the second exists for the
    byte-determinism check.  Node (0, 0) is catalog-only; node (2, 3)
    exercises the continuum route at the transition node."""
    runs = {}
    for window, node in ((0, 0), (2, 3)):
        first = tmp_path_factory.mktemp(f"smoke_first_{window}_{node}")
        second = tmp_path_factory.mktemp(f"smoke_second_{window}_{node}")
        _run_smoke(first, window, node)
        _run_smoke(second, window, node)
        runs[(window, node)] = (first, second)
    return runs


@_SEALED_KERNEL_SKIP
def test_smoke_determinism(smoke_runs):
    """Two smoke runs with the same arguments produce byte-identical
    node and replica records."""
    for (window, node), (first, second) in smoke_runs.items():
        for name in (
            f"window_{window}_node_{node}.json",
            f"window_{window}_node_{node}_replica_0.json",
        ):
            a = (first / name).read_bytes()
            b = (second / name).read_bytes()
            assert a == b, (
                f"window {window} node {node}: {name} differs between "
                "two identical smoke runs"
            )


@_SEALED_KERNEL_SKIP
def test_smoke_contract(driver, smoke_runs):
    """The smoke records: exact charge balance, the below-boundary node
    routed everything to the catalog, and the transition node's
    continuum count equals the pinned expected count."""
    first, _ = smoke_runs[(0, 0)]
    record = json.loads(
        (first / "window_0_node_0.json").read_text(encoding="utf-8")
    )
    assert record["mode"] == "smoke"
    assert record["charge_balance"] == {
        "total_Q_exact": 1, "total_B_exact": 1,
        "events_checked": SMOKE_DECAYS,
    }
    assert record["routing"]["n_catalog"] == SMOKE_DECAYS
    assert record["routing"]["n_continuum"] == 0
    assert record["routing"]["expected_w_at_node"] == 0.0
    assert record["source_mass_gev"] == 2.48

    first3, _ = smoke_runs[(2, 3)]
    record3 = json.loads(
        (first3 / "window_2_node_3.json").read_text(encoding="utf-8")
    )
    assert record3["charge_balance"] == {
        "total_Q_exact": 0, "total_B_exact": 0,
        "events_checked": SMOKE_DECAYS,
    }
    assert record3["source_mass_gev"] == 2.78
    # The recorded continuum count is the pinned per-event route count.
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.spectral_density import BlendSpec

    spec = BlendSpec()
    w = spec.weight(2.78)
    expected = sum(
        int(float(np.random.default_rng(np.random.SeedSequence(
            derive_substreams(driver.MASTER_SEED, 0,
                              (2 * 5 + 3) * 1_000_000 + i)
            .stream_for("decay_channel")).spawn(1)[0]
        ).random()) < w)
        for i in range(SMOKE_DECAYS)
    )
    assert record3["routing"]["n_continuum"] == expected
    assert 0 < expected < SMOKE_DECAYS
    # Both routes produced final particles with recorded moments.
    assert record3["momentum_moments"]


# ----------------------------------------------------------------------
# Analytic engine determinism
# ----------------------------------------------------------------------
def test_analytic_engine_determinism(driver):
    """The continuum analytic engine and the card-2 catalog payload
    reproduce identical values on repeated evaluation (pure compute)."""
    from na49_mixing.hadron_catalog import HadronCatalog
    from na49_mixing.spectral_density import families_from_json
    from na49_mixing.threshold_metric import RecordSet

    catalog = HadronCatalog.from_json(CATALOG_PATH)
    families = families_from_json(str(FAMILIES_PATH))
    qn = (1, 1, 0)
    mass = 2.78
    open_records = driver._open_set(catalog, qn, mass)
    record_set = RecordSet(catalog, open_records)
    family = driver.active_families(families, qn)[0]
    first = driver.continuum_payload(record_set, family, mass)
    second = driver.continuum_payload(record_set, family, mass)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )
    total = sum(block["n"] for block in first.values())
    assert total > 2.0
    cat_first = record_set.mixture_payload(open_records, mass)
    cat_second = record_set.mixture_payload(open_records, mass)
    assert json.dumps(cat_first, sort_keys=True) == json.dumps(
        cat_second, sort_keys=True
    )


# ----------------------------------------------------------------------
# Layout sync
# ----------------------------------------------------------------------
def test_layout_sync_inputs():
    """The nested and outer copies of the driver are byte-identical and
    the layout checker passes."""
    assert NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file()
    nested = hashlib.sha256(NESTED_DRIVER.read_bytes()).hexdigest()
    outer = hashlib.sha256(OUTER_DRIVER.read_bytes()).hexdigest()
    assert nested == outer, (
        "the nested and outer staged_r31_matching_scan.py copies differ"
    )
    if SYNC_CHECKER.is_file():
        result = subprocess.run(
            [sys.executable, str(SYNC_CHECKER)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"check_helpers_layout_sync.py failed (exit "
            f"{result.returncode}):\n{result.stdout}\n{result.stderr}"
        )


# ----------------------------------------------------------------------
# Sealed production artifacts (read-only; skip when absent)
# ----------------------------------------------------------------------
def test_production_pre_registration_pins():
    """The sealed pre-registration carries the card's statistics, seed
    map, routing weights, metric, and per-node open sets (skips when
    the sealed run dir is absent)."""
    pre_path = SEALED_RUN_DIR / "pre_registration.json"
    if not pre_path.is_file():
        pytest.skip("the sealed R31 run dir is absent on this checkout")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    stats = pre["statistics"]
    assert stats["replicas"] == 4
    assert stats["decays_per_replica"] == 25_000
    assert stats["windows"] == 3
    assert stats["nodes"] == 5
    assert stats["total_decays"] == 1_500_000
    assert pre["nodes"]["masses_gev"] == [2.48, 2.58, 2.68, 2.78, 2.88]
    assert pre["nodes"]["boundary_gev"] == 2.680194110727528
    routing = pre["decay_routing"]
    assert routing["boundary_gev"] == 2.680194110727528
    assert routing["ramp_width_gev"] == 0.5
    assert routing["expected_w_at_nodes"][:3] == [0.0, 0.0, 0.0]
    assert round(routing["expected_w_at_nodes"][3], 12) == 0.199611778545
    assert round(routing["expected_w_at_nodes"][4], 12) == 0.399611778545
    assert pre["seed_map"]["master_seed"] == 20260917
    metric = pre["discontinuity_metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == 3.0
    assert metric["boundary_straddling_pairs"] == [2, 6, 10]
    assert pre["no_fitting"] is True
    # The three windows carry the pre-registered quantum numbers, five
    # open-set entries per window, and no empty open set.
    assert [tuple(w["qn"]) for w in pre["windows"]] == [
        (1, 1, 0), (1, 0, -1), (0, 0, 0)
    ]
    for w in pre["windows"]:
        open_by_node = pre["open_sets"][str(w["window"])]
        assert sorted(open_by_node) == ["0", "1", "2", "3", "4"]
        for pdgs in open_by_node.values():
            assert pdgs, "an empty open set in the sealed pre-registration"
