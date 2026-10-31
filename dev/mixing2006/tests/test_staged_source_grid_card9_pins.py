"""Card-9 staged source-grid driver: committed-behavior coverage pins.

Task ``TASK-20260913-r2-source-grid-runner`` (plan section 22 task 17; the
runs R21/R22 of plan section 8, lines 559-565; round 2 index items 6, 9,
and 10). Sibling of ``tests/test_staged_source_grid.py`` (the solver's
card file, which pins the hermetic ``--smoke`` path: determinism, charge
balance at the smoke points, the smoke open sets, the degenerate window,
and the dual-tree layout sync): this module pins the pre-registered
contract of the committed driver
``helpers/benchmarking/staged_source_grid.py`` (nested 9293638, outer
3daba09, byte-identical) that the smoke test does not cover, on the
committed catalog ``config/v2_singlet/resonance_catalog.json``.

Solver scope decision pinned here (#4, the mixture-layer stream split):
the mixture layer consumes one-step spawned children
(``SeedSequence(stream_id).spawn(1)``) of the two labeled stream ids while
the card-6/7 kernel keeps the labeled ids themselves. Sharing the labeled
ids directly between the two layers made the record-draw uniform and the
kernel's first vertex-channel uniform the same value and correlated the
record selection with the root channel selection (the Delta+ charge split
reversed to 1/3:2/3 in the first smoke). The pins below pin both the
mechanism (white-box, on the driver's actual per-event seed construction)
and the observable regression (the pi_0 : pi_plus = 2 : 1 split at the
baryon grid point k = 1).

Pins:

- ``test_grids_are_pre_registered``: the baryonic grid is exactly
  ``M_k = 1.07324891 + 0.1 k`` for ``k = 0..19`` (20 points, 1.07324891 to
  2.97324891 GeV, source (B, Q, S) = (1, 1, 0) at rest) and the mesonic
  grid is exactly ``M_k = 0.2699532 + 0.1 k`` for ``k = 0..27`` (28
  points, 0.2699532 to 2.9699532 GeV, source (0, 0, 0) at rest); start,
  step, count, and every grid mass are exact, and every point file's
  ``source_mass_gev`` sits on its grid mass.
- ``test_seed_map_and_closed_labels``: the master seed is 20260913, the
  seed rule is ``derive_substreams(20260913, r, k * 1_000_000 + i)``, the
  seed map runs on the closed labels ``decay_channel`` /
  ``decay_kinematics``, and ``SUBSTREAM_LABELS`` is still the closed
  seven-label set (no extension); the driver's pre-registration echoes
  the seed map.
- ``test_t1_statistics_and_points_enforcement``: T1 statistics (4
  replicas x 25,000 decays per grid point) are the enforced defaults, a
  mismatched ``--points`` and non-positive ``--decays-per-replica`` /
  ``--replicas`` are rejected with ``SystemExit``, and the pre-registered
  point counts (20 baryonic, 28 mesonic) are accepted.
- ``test_mixture_stream_split_is_spawned_children``: the solver scope
  decision #4, pinned white-box by capturing every ``np.random.default_rng``
  seed the driver and the card-6 decayer consume for four committed
  events: the mixture record draw and the card-3 mass draw run on the
  spawned children of the two labeled stream ids, the kernel's two
  substreams run on the labeled ids themselves re-derived from
  ``derive_substreams(20260913, r, k * 1_000_000 + i)`` (the card's seed
  map, pinned behaviorally), and the two layers draw disjoint uniforms.
- ``test_delta_plus_split_two_thirds_to_one_third``: the regression pin.
  At the baryon grid point k = 1 (M = 1.17324891 GeV, the first point
  where the Delta+ joins the open set) over 2,000 committed decays the
  pi_0 : pi_plus ratio is 2 : 1 (the 2/3 p pi0 vs 1/3 n pi+ split of the
  card-5 charge weights inside the uniform record mixture over
  {proton, Delta+}); the shared-stream correlation bug of the first smoke
  reverses it to about 1/3 : 2/3.
- ``test_open_set_assert``: the open set is non-empty at every point of
  both committed grids (the card's assert holds by construction through
  the stable termination records), and a synthetic catalog whose
  (1, 1, 0) support lies above every baryon grid mass raises the driver's
  ``RuntimeError`` and stops the run before any file is written (no
  fallback decay path).
- ``test_pre_registration_and_provenance_fields``: ``pre_registration.json``
  (grids, mixture model, seed map, statistics, output schema, per-point
  open sets) and ``provenance.json`` (catalog SHA-256, card chain) carry
  the expected fields and are written before any generation (mtime
  ordering against the first point file).
- ``test_no_gate_or_seal_files``: the driver writes no gate JSON and no
  seal files (cards 11 and 12's scope): every run dir holds exactly
  ``pre_registration.json``, ``provenance.json``, the per-point
  ``point_<k:02d>.json`` files, and the per-replica
  ``replica_<r>_point_<k:02d>.json`` files.
- ``test_charge_balance_on_full_grids``: the exact per-event (B, Q)
  charge balance extended from the solver's smoke points to every point
  of both full grids (20 baryonic and 28 mesonic points at 100 decays
  each): every point file and its replica file show the exact source
  (B, Q) over all events, and the multiplicity key set is the catalog's
  stable termination set (all listed).

Hermetic: committed catalog and vendored PDG table only; no NA49/SPS
external tables. The generation budgets are small (the full grids at 100
decays per point, the split at 2,000 decays at one point) and
deterministic for the committed seed and catalog.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog, HadronRecord
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams

NESTED_ROOT = Path(__file__).resolve().parent.parent
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_source_grid.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

# The pre-registered contract of the task card (the driver enforces it).
CARD_ID = "TASK-20260913-r2-source-grid-runner"
MASTER_SEED = 20260913
SEED_RULE = "derive_substreams(20260913, r, k * 1_000_000 + i)"
SEED_LABELS = ("decay_channel", "decay_kinematics")
T1_REPLICAS = 4
T1_DECAYS_PER_REPLICA = 25_000
T1_LABEL = "T1: 4 replicas x 25,000 decays per grid point (plan section 5.1)"

#: The closed physical-layer label set (card 5; the card forbids any
#: extension of SUBSTREAM_LABELS).
CLOSED_SUBSTREAM_LABELS = (
    "topology_selection",
    "exchange_kinematics",
    "spectral_state",
    "decay_channel",
    "decay_kinematics",
    "nuclear_geometry",
    "transport",
)

GRID_START = {"baryonic": 1.07324891, "mesonic": 0.2699532}
GRID_STEP = 0.1
GRID_POINTS = {"baryonic": 20, "mesonic": 28}
GRID_QN = {"baryonic": (1, 1, 0), "mesonic": (0, 0, 0)}
GRID_ENDPOINTS = {"baryonic": 2.97324891, "mesonic": 2.9699532}

#: The card chain the driver records in provenance.json (cards 1-9 of the
#: Round 2 set, in order).
CARD_CHAIN = (
    "TASK-20260913-r2-hadron-resonance-schema",
    "TASK-20260913-r2-provenance-catalog-import",
    "TASK-20260913-r2-mass-dependent-widths",
    "TASK-20260913-r2-barrier-factors",
    "TASK-20260913-r2-isospin-charge-completion",
    "TASK-20260913-r2-recursive-two-body-decays",
    "TASK-20260913-r2-selected-three-body-decays",
    "TASK-20260913-r2-branching-spectral-normalization-tests",
    "TASK-20260913-r2-source-grid-runner",
)

#: The card's per-point output schema key list (verbatim order).
SCHEMA_KEYS = [
    "point", "source_mass_gev", "source_qn", "open_records", "n_decays",
    "multiplicities", "momentum_moments", "feed_down", "charge_balance",
    "pairs",
]

# Generation budgets (deterministic for the committed seed and catalog).
SPLIT_POINT = 1          # the first baryon point with the Delta+ open
SPLIT_DECAYS = 2_000     # the pi_0 : pi_plus = 2 : 1 regression sample
FULL_GRID_DECAYS = 100   # the exact (B, Q) balance over every grid point
CAPTURE_DECAYS = 4       # the white-box stream-split capture sample


@lru_cache(maxsize=1)
def _driver():
    """The committed driver module, loaded from its nested-tree path."""
    spec = importlib.util.spec_from_file_location(
        "staged_source_grid_card9_pins", str(DRIVER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _grid_masses(source_type: str) -> list[float]:
    """The card's grid masses ``M_k = start + step * k`` (independent of
    the driver's computation)."""
    return [GRID_START[source_type] + GRID_STEP * k
            for k in range(GRID_POINTS[source_type])]


def _expected_open_set(source_type: str, k: int) -> list[int]:
    """The card's open set at point *k*, computed independently from the
    committed catalog with the truncated-spectral-window rule: the
    records with the source (B, Q, S) whose window
    ``[mass_range[0], min(mass_range[1], M_k)]`` is non-empty, in catalog
    record order."""
    catalog = _catalog()
    qn = GRID_QN[source_type]
    mass = GRID_START[source_type] + GRID_STEP * k
    out: list[int] = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != qn:
            continue
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        if min(hi, mass) >= lo:
            out.append(int(record.pdg_id))
    return out


def _species_key(name: str) -> str:
    """The driver's species-key transliteration (pi+ -> pi_plus, pi- ->
    pi_minus, a trailing 0 -> _0, otherwise verbatim)."""
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _stable_species_keys() -> set[str]:
    """The multiplicity key set of the card's schema: the catalog's stable
    termination set, all listed."""
    stable = _catalog().stable_states()
    return {
        _species_key(str(record.name))
        for record in _catalog().records()
        if int(record.pdg_id) in stable
    }


def _point(run_dir: Path, k: int) -> dict:
    return json.loads(
        (run_dir / f"point_{k:02d}.json").read_text(encoding="utf-8")
    )


def _pre_registration(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "pre_registration.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def full_grid_runs(tmp_path_factory) -> dict[str, Path]:
    """Both full pre-registered grids at 100 decays x 1 replica each
    (the exact (B, Q) balance extended from the smoke points to every
    grid point; deterministic for the committed seed and catalog)."""
    driver = _driver()
    runs: dict[str, Path] = {}
    for source_type in ("baryonic", "mesonic"):
        run_dir = tmp_path_factory.mktemp(f"card9_full_grid_{source_type}")
        driver._run_grid(
            catalog=_catalog(),
            source_type=source_type,
            point_indices=list(range(GRID_POINTS[source_type])),
            decays_per_replica=FULL_GRID_DECAYS,
            replicas=1,
            master_seed=MASTER_SEED,
            run_dir=run_dir,
            mode="production",
        )
        runs[source_type] = run_dir
    return runs


@pytest.fixture(scope="module")
def split_run(tmp_path_factory) -> Path:
    """The baryon grid point k = 1 (M = 1.17324891 GeV; the first point
    with the Delta+ open, the open set {proton, Delta+}) at 2,000 decays x
    1 replica: the pi_0 : pi_plus = 2 : 1 split sample."""
    driver = _driver()
    run_dir = tmp_path_factory.mktemp("card9_delta_plus_split")
    driver._run_grid(
        catalog=_catalog(),
        source_type="baryonic",
        point_indices=[SPLIT_POINT],
        decays_per_replica=SPLIT_DECAYS,
        replicas=1,
        master_seed=MASTER_SEED,
        run_dir=run_dir,
        mode="production",
    )
    return run_dir


@pytest.fixture(scope="module")
def production_runs(tmp_path_factory) -> dict[str, Path]:
    """The driver's production path at the pre-registered point counts
    with minimal statistics (1 decay x 1 replica per point): the
    pre-registration/provenance contract and the no-gate/no-seal file set
    on the full 20-point baryonic and 28-point mesonic grids."""
    driver = _driver()
    runs: dict[str, Path] = {}
    for source_type in ("baryonic", "mesonic"):
        run_dir = tmp_path_factory.mktemp(f"card9_production_{source_type}")
        result = driver.main([
            "--source-type", source_type,
            "--points", str(GRID_POINTS[source_type]),
            "--replicas", "1",
            "--decays-per-replica", "1",
            "--run-dir", str(run_dir),
        ])
        assert result == 0
        runs[source_type] = run_dir
    return runs


def test_grids_are_pre_registered(full_grid_runs):
    """The driver's grids are exactly the card's pre-registered grids:
    baryonic ``M_k = 1.07324891 + 0.1 k`` for k = 0..19 (20 points) and
    mesonic ``M_k = 0.2699532 + 0.1 k`` for k = 0..27 (28 points), both
    at rest; start, step, count, and every grid mass are exact."""
    driver = _driver()
    assert set(driver.GRIDS) == {"baryonic", "mesonic"}
    for source_type in ("baryonic", "mesonic"):
        grid = driver.GRIDS[source_type]
        assert tuple(int(q) for q in grid["source_qn"]) == GRID_QN[source_type]
        assert grid["start_gev"] == GRID_START[source_type]
        assert grid["step_gev"] == GRID_STEP
        assert grid["points"] == GRID_POINTS[source_type]
        # every grid mass is exactly start + step * k (the card's M_k)
        masses = driver._grid_masses(grid)
        assert len(masses) == GRID_POINTS[source_type]
        assert masses == _grid_masses(source_type)
        assert masses[0] == pytest.approx(GRID_START[source_type], abs=1e-9)
        assert masses[-1] == pytest.approx(GRID_ENDPOINTS[source_type],
                                           abs=1e-9)
        # every generated point file sits on its grid mass and carries the
        # source quantum numbers
        run_dir = full_grid_runs[source_type]
        for k in range(GRID_POINTS[source_type]):
            payload = _point(run_dir, k)
            assert payload["point"] == k
            assert payload["source_mass_gev"] == pytest.approx(masses[k],
                                                               abs=0.0)
            assert payload["source_qn"] == list(GRID_QN[source_type])


def test_seed_map_and_closed_labels(production_runs):
    """The seed map is exactly the card's: master seed 20260913, the rule
    ``derive_substreams(20260913, r, k * 1_000_000 + i)``, the closed
    labels ``decay_channel`` / ``decay_kinematics``, and no extension of
    ``SUBSTREAM_LABELS``; the driver's pre-registration echoes it."""
    driver = _driver()
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.SEED_RULE == SEED_RULE
    assert tuple(driver.SEED_LABELS) == SEED_LABELS
    # the card forbids any extension of SUBSTREAM_LABELS: the label set is
    # still the closed seven-label set of card 5
    assert tuple(SUBSTREAM_LABELS) == CLOSED_SUBSTREAM_LABELS
    for source_type, run_dir in production_runs.items():
        seed_map = _pre_registration(run_dir)["seed_map"]
        assert seed_map["master_seed"] == MASTER_SEED
        assert seed_map["rule"] == SEED_RULE
        assert seed_map["substream_labels"] == list(SEED_LABELS)


def test_t1_statistics_and_points_enforcement(production_runs):
    """The T1 statistics (4 replicas x 25,000 decays per grid point) are
    the enforced defaults; a mismatched --points and non-positive
    --decays-per-replica / --replicas are rejected with SystemExit; the
    pre-registered point counts are accepted."""
    driver = _driver()
    assert driver.T1_REPLICAS == T1_REPLICAS
    assert driver.T1_DECAYS_PER_REPLICA == T1_DECAYS_PER_REPLICA
    assert driver.T1_LABEL == T1_LABEL
    args = driver._build_parser().parse_args(
        ["--source-type", "baryonic", "--run-dir", "unused"]
    )
    assert args.replicas == T1_REPLICAS
    assert args.decays_per_replica == T1_DECAYS_PER_REPLICA
    assert args.master_seed == MASTER_SEED
    assert args.points is None
    # a mismatched --points is rejected (the pre-registered count is
    # enforced per source type, including the cross-type mismatches)
    for source_type, wrong_points in (
        ("baryonic", 19), ("baryonic", 21), ("baryonic", 28),
        ("mesonic", 20), ("mesonic", 27), ("mesonic", 29),
    ):
        with pytest.raises(SystemExit):
            driver.main([
                "--source-type", source_type,
                "--points", str(wrong_points),
                "--run-dir", "unused",
            ])
    # non-positive statistics are rejected
    for extra in (["--replicas", "0"], ["--decays-per-replica", "-1"]):
        with pytest.raises(SystemExit):
            driver.main([
                "--source-type", "baryonic", *extra, "--run-dir", "unused",
            ])
    # the pre-registered point counts are accepted (the production_runs
    # fixture ran the driver's production path at exactly 20 baryonic and
    # 28 mesonic points) and the pre-registration records the standard
    # and the statistics used
    for source_type, run_dir in production_runs.items():
        pre = _pre_registration(run_dir)
        assert pre["mode"] == "production"
        assert pre["grid_points"] == list(range(GRID_POINTS[source_type]))
        assert pre["grid_masses_gev"] == _grid_masses(source_type)
        assert pre["statistics"] == {
            "standard": T1_LABEL,
            "replicas": 1,
            "decays_per_replica": 1,
        }


def test_mixture_stream_split_is_spawned_children(tmp_path):
    """Scope decision #4, pinned white-box: the mixture layer consumes
    one-step spawned children of the two labeled stream ids while the
    kernel keeps the labeled ids themselves, so the two layers draw
    disjoint uniforms. Every ``np.random.default_rng`` seed handed to the
    driver and the card-6 decayer is captured over four committed events
    at baryon point k = 1 (the card's seed map re-derived independently):
    the record draw and the card-3 mass draw run on the spawned children,
    the kernel's two substreams on the labeled ids."""
    driver = _driver()
    catalog = _catalog()
    captured: list = []
    real_default_rng = np.random.default_rng

    def capture(seed, *args, **kwargs):
        captured.append(seed)
        return real_default_rng(seed, *args, **kwargs)

    np.random.default_rng = capture
    try:
        driver._run_grid(
            catalog=catalog,
            source_type="baryonic",
            point_indices=[SPLIT_POINT],
            decays_per_replica=CAPTURE_DECAYS,
            replicas=1,
            master_seed=MASTER_SEED,
            run_dir=tmp_path,
            mode="production",
        )
    finally:
        np.random.default_rng = real_default_rng

    # exactly four generators per event: the mixture record draw, the
    # mixture mass draw, and the kernel's two labeled substreams
    assert len(captured) == CAPTURE_DECAYS * 4
    for i in range(CAPTURE_DECAYS):
        # the card's seed map, re-derived independently
        subs = derive_substreams(
            MASTER_SEED, 0, SPLIT_POINT * 1_000_000 + i
        )
        labeled_channel = subs.stream_for("decay_channel")
        labeled_kinematics = subs.stream_for("decay_kinematics")
        child_channel = np.random.SeedSequence(labeled_channel).spawn(1)[0]
        child_kinematics = (
            np.random.SeedSequence(labeled_kinematics).spawn(1)[0]
        )
        record_seed, mass_seed, kernel_channel, kernel_kinematics = (
            captured[4 * i:4 * i + 4]
        )
        # the mixture draw runs on the spawned children, not the labeled
        # ids directly (the scope decision #4 regression)
        assert isinstance(record_seed, np.random.SeedSequence)
        assert isinstance(mass_seed, np.random.SeedSequence)
        assert record_seed.state == child_channel.state, (
            "the mixture record draw does not run on "
            "SeedSequence(stream_id).spawn(1)[0] of the decay_channel id"
        )
        assert mass_seed.state == child_kinematics.state, (
            "the mixture mass draw does not run on "
            "SeedSequence(stream_id).spawn(1)[0] of the "
            "decay_kinematics id"
        )
        assert record_seed.state != (
            np.random.SeedSequence(labeled_channel).state
        ), "the mixture record draw shares the labeled decay_channel id"
        assert mass_seed.state != (
            np.random.SeedSequence(labeled_kinematics).state
        ), "the mixture mass draw shares the labeled decay_kinematics id"
        # the kernel keeps the labeled ids themselves (the card-6/7
        # decayer serves one decay per (master_seed, replica, event))
        assert kernel_channel == labeled_channel
        assert kernel_kinematics == labeled_kinematics
        # the two layers draw disjoint uniform sequences under the same
        # labeled anchors: the first uniform of each layer differs
        assert real_default_rng(record_seed).random() != (
            real_default_rng(kernel_channel).random()
        )
        assert real_default_rng(mass_seed).random() != (
            real_default_rng(kernel_kinematics).random()
        )
    # the run itself completed (the captured seeds produced a valid run)
    assert (tmp_path / "point_01.json").is_file()


def test_delta_plus_split_two_thirds_to_one_third(split_run):
    """The regression pin for scope decision #4: at baryon point k = 1
    (M = 1.17324891 GeV; the open set {proton, Delta+}, each drawn with
    the flat mixture weight 1/2) the Delta+ charge split is the card-5
    2/3 p pi0 vs 1/3 n pi+ over the 2,000 committed decays. The proton
    identity events carry no pions, so the pi_0 : pi_plus ratio is the
    split itself: 2 : 1 (the shared-stream correlation bug of the first
    smoke reverses it to about 1/3 : 2/3, ratio about 0.5)."""
    payload = _point(split_run, SPLIT_POINT)
    assert payload["open_records"] == _expected_open_set("baryonic",
                                                         SPLIT_POINT)
    assert payload["n_decays"] == SPLIT_DECAYS
    means = {key: float(block["mean"])
             for key, block in payload["multiplicities"].items()}
    pi_0, pi_plus = means["pi_0"], means["pi_plus"]
    # the 2 : 1 split (pi_0 from the 2/3 p pi0 channel, pi_plus from the
    # 1/3 n pi+ channel); the buggy shared-stream split reads about 0.5
    assert 1.5 < pi_0 / pi_plus < 2.6, (
        f"the Delta+ charge split at k = {SPLIT_POINT} reads "
        f"pi_0 : pi_plus = {pi_0 / pi_plus:.3f} : 1; the committed "
        "stream split (spawned mixture children, labeled kernel ids) "
        "reads 2 : 1, the shared-stream correlation bug about 0.5 : 1"
    )
    # the per-event means sit on the card's expectations: the uniform
    # record mixture over {proton, Delta+} with the 2/3 : 1/3 channel
    # split gives <proton> = 1/2 + 1/2 * 2/3 = 5/6,
    # <neutron> = 1/2 * 1/3 = 1/6, <pi_0> = 1/2 * 2/3 = 1/3,
    # <pi_plus> = 1/2 * 1/3 = 1/6 (8-se bands over SPLIT_DECAYS events;
    # the committed seed and catalog make the sample deterministic)
    for key, expected in (
        ("proton", 5.0 / 6.0),
        ("neutron", 1.0 / 6.0),
        ("pi_0", 1.0 / 3.0),
        ("pi_plus", 1.0 / 6.0),
    ):
        se = math.sqrt(expected * (1.0 - expected) / SPLIT_DECAYS)
        assert expected - 8.0 * se < means[key] < expected + 8.0 * se, (
            f"the {key} mean at k = {SPLIT_POINT} reads {means[key]:.4f} "
            f"outside the 8-se band of the card's expectation "
            f"{expected:.4f}"
        )


def test_open_set_assert(tmp_path):
    """The card's open-set assert: non-empty at every grid point of both
    committed grids (it holds by construction through the stable
    termination records carrying the source quantum numbers), and an
    empty open set raises the driver's RuntimeError and stops the run
    before any file is written (no fallback decay path)."""
    driver = _driver()
    # the committed catalog: the open set is non-empty at every point of
    # both grids, and the driver's open set matches the independent
    # recomputation at every point
    catalog = _catalog()
    for source_type in ("baryonic", "mesonic"):
        masses = _grid_masses(source_type)
        for k, mass in enumerate(masses):
            expected = _expected_open_set(source_type, k)
            assert expected, (
                f"the recomputed {source_type} open set at k = {k} is "
                "empty; the card's assert would raise on the committed "
                "catalog"
            )
            got = [
                int(record.pdg_id)
                for record in driver._open_set(catalog,
                                              GRID_QN[source_type], mass)
            ]
            assert got == expected
    # the unit assert path: an empty open set raises, a non-empty one
    # passes
    with pytest.raises(RuntimeError, match="open set .* is empty"):
        driver._assert_open_set([], GRID_QN["baryonic"],
                                GRID_START["baryonic"])
    driver._assert_open_set([catalog.get(2212)], GRID_QN["baryonic"],
                            GRID_START["baryonic"])
    # the run-level path: a synthetic catalog whose (1, 1, 0) support lies
    # above every baryon grid mass has an empty open set at every grid
    # point; the driver raises and writes nothing (the assert stops the
    # run before any generation and before any pre-registration write)
    synth = HadronRecord(
        pdg_id=900001, name="synth_baryon", B=1, Q=1, S=0,
        isospin=0.5, isospin3=1, spin=0.5, parity="+",
        pole_mass=3.2, nominal_width=0.1,
        mass_range=(3.0, 3.5), decay_channels=(),
        stable_for_decay=False, provenance=None,
    )
    synth_catalog = HadronCatalog([synth])
    run_dir = tmp_path / "card9_pins_empty_open_set"
    with pytest.raises(RuntimeError, match="no fallback decay path"):
        driver._run_grid(
            catalog=synth_catalog,
            source_type="baryonic",
            point_indices=[0],
            decays_per_replica=10,
            replicas=1,
            master_seed=MASTER_SEED,
            run_dir=run_dir,
            mode="production",
        )
    assert not (run_dir / "pre_registration.json").exists()
    assert not (run_dir / "point_00.json").exists()


def test_pre_registration_and_provenance_fields(production_runs):
    """pre_registration.json (grids, mixture model, seed map, statistics,
    output schema, per-point open sets) and provenance.json (catalog
    SHA-256, card chain) carry the expected fields and are written before
    any generation."""
    driver = _driver()
    catalog_bytes = CATALOG_PATH.read_bytes()
    for source_type, run_dir in production_runs.items():
        pre = _pre_registration(run_dir)
        assert pre["schema"] == "na49_mixing.staged_source_grid.pre_registration/1"
        assert pre["card"] == CARD_ID
        assert pre["mode"] == "production"
        assert pre["source_type"] == source_type
        assert pre["source_qn"] == list(GRID_QN[source_type])
        # the pre-registered grids, both, verbatim
        for grid_name, grid in pre["grids"].items():
            assert grid["start_gev"] == GRID_START[grid_name]
            assert grid["step_gev"] == GRID_STEP
            assert grid["points"] == GRID_POINTS[grid_name]
            assert tuple(grid["source_qn"]) == GRID_QN[grid_name]
        # the mixture model (the card's flat record mixture), all clauses
        assert set(pre["mixture_model"]) == {
            "open_set_rule", "open_set_nonempty_note", "record_draw",
            "mass_rule", "decay", "kernels",
        }
        # the per-point open sets match the independent recomputation
        for k in range(GRID_POINTS[source_type]):
            assert pre["open_sets"][str(k)] == _expected_open_set(
                source_type, k
            )
        # the output schema key list, verbatim order
        assert pre["output_schema"]["keys"] == SCHEMA_KEYS
        assert pre["written_before_generation"] is True

        prov = json.loads(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        assert prov["schema"] == "na49_mixing.staged_source_grid.provenance/1"
        assert prov["card"] == CARD_ID
        assert prov["card_chain"] == list(CARD_CHAIN)
        assert prov["catalog"]["path"] == "config/v2_singlet/resonance_catalog.json"
        assert prov["catalog"]["sha256"] == (
            hashlib.sha256(catalog_bytes).hexdigest()
        )
        assert prov["driver"] == "helpers/benchmarking/staged_source_grid.py"
        assert prov["written_before_generation"] is True

        # written before any generation: the ordering of the file writes
        # (the pre-registration and provenance precede the first point
        # file; separate sequential writes keep the mtime ordering)
        pre_mtime = (run_dir / "pre_registration.json").stat().st_mtime_ns
        prov_mtime = (run_dir / "provenance.json").stat().st_mtime_ns
        first_point_mtime = (run_dir / "point_00.json").stat().st_mtime_ns
        assert pre_mtime <= prov_mtime <= first_point_mtime


def test_no_gate_or_seal_files(full_grid_runs, production_runs):
    """The driver writes no gate JSON and no seal files (cards 11 and 12's
    scope): every run dir holds exactly pre_registration.json,
    provenance.json, the per-point point_<k:02d>.json files, and the
    per-replica replica_<r>_point_<k:02d>.json files."""
    cases: list[tuple[Path, int]] = [
        (run_dir, GRID_POINTS[source_type])
        for source_type, run_dir in full_grid_runs.items()
    ] + [
        (run_dir, GRID_POINTS[source_type])
        for source_type, run_dir in production_runs.items()
    ]
    for run_dir, n_points in cases:
        names = sorted(path.name for path in run_dir.iterdir())
        # both fixtures run 1 replica per point
        expected = (
            {"pre_registration.json", "provenance.json"}
            | {f"point_{k:02d}.json" for k in range(n_points)}
            | {f"replica_0_point_{k:02d}.json" for k in range(n_points)}
        )
        assert set(names) == expected, (
            "the run dir holds an unexpected file set: "
            + ", ".join(sorted(set(names) ^ expected))
        )
        for forbidden in ("gate", "seal"):
            assert not any(
                forbidden in name for name in names
            ), f"a {forbidden} file is written: {names}"


def test_charge_balance_on_full_grids(full_grid_runs):
    """The exact per-event (B, Q) charge balance over every point of both
    full grids (20 baryonic and 28 mesonic points at 100 decays each):
    every point file and its replica file show the exact source (B, Q),
    and the multiplicity key set is the catalog's stable termination set
    (all listed)."""
    stable_keys = _stable_species_keys()
    for source_type, run_dir in full_grid_runs.items():
        qn = GRID_QN[source_type]
        baryon, charge, _strangeness = qn
        for k in range(GRID_POINTS[source_type]):
            payload = _point(run_dir, k)
            assert payload["n_decays"] == FULL_GRID_DECAYS
            assert payload["charge_balance"] == {
                "total_Q_exact": charge,
                "total_B_exact": baryon,
                "events_checked": FULL_GRID_DECAYS,
            }
            # all stable species of the catalog stable set are listed
            assert set(payload["multiplicities"]) == stable_keys
            # the per-replica raw diagnostics carry the same exact (B, Q)
            replica = json.loads(
                (run_dir / f"replica_0_point_{k:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert replica["n_decays"] == FULL_GRID_DECAYS
            assert replica["charge_balance"]["total_Q_exact"] == charge
            assert replica["charge_balance"]["total_B_exact"] == baryon
