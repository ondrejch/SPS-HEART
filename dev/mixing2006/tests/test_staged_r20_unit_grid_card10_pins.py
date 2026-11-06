"""Card-10 staged R20 single-resonance unit grid: committed-behavior pins.

Task ``TASK-20260913-r2-run-r20-single-resonance-unit-grid`` (round 2,
card 10; the run R20 of plan section 8, line 551). This module pins the
committed driver ``helpers/benchmarking/staged_r20_unit_grid.py``
(nested 3a9b15b, outer e4e76ca, byte-identical in the two trees) against
the sealed run R20 evidence under
``dev/00runs/r20_single_resonance_unit_grid_2026-09-13/``: 80 sealed
state files, the pre-registration, provenance, aggregate, the Gate 2
report, the seal manifest, and ``SHA256SUMS.txt``.

The run card's procedure is staged (each invocation performs exactly
one step and stops: pre-registration, generation, verification, seal,
clean-environment validation; the run dir is immutable after the
seal).  These pins are read-only over the sealed artifacts plus
parser-level CLI checks and the driver's pre-registration
recomputation (no decays are generated, the sealed run is never
re-sealed, and ``--verify`` / ``--seal`` / ``--clean-validate`` are
never invoked by the tests).

Pins:

- ``test_driver_module_contract_constants``: the committed run
  contract constants (run id, card id, run name, default run dir,
  100 grid points, master seed 20260913, replica index 0, the
  event-index rule, the closed seed labels, the 10,000/2,000
  per-point counts, the 3-sigma / 1e-9 / 10-bin tolerances) and the
  pre-registered important/minor state split (59 important, 21
  minor, disjoint, exactly the catalog's non-stable records).
- ``test_event_index_formula_partitions_states``: the committed
  ``event_index = s * 1_000_000 + j * 10_000 + i`` slot arithmetic:
  with 100 points (``j`` below 100) and per-point counts below 10,000
  (``i`` below 10,000), each state's event indices occupy the
  disjoint block ``[s * 1_000_000, (s + 1) * 1_000_000)``, so the 80
  states partition ``[0, 80 * 1_000_000)`` without overlap.
- ``test_cli_modes_and_flags``: the CLI exposes exactly the staged
  modes ``--pre-registration`` / ``--generate`` / ``--verify`` /
  ``--seal`` / ``--clean-validate`` (a required mutually exclusive
  group) plus ``--state`` (int, only with ``--generate``) and
  ``--run-dir`` (default ``dev/00runs/
  r20_single_resonance_unit_grid_2026-09-13``); ``--state`` without
  ``--generate`` and a bare invocation are rejected.
- ``test_cli_pre_registration_writes_only_pre_registration``: the
  ``--pre-registration`` step writes exactly
  ``pre_registration.json`` and ``provenance.json`` (nothing is
  generated), and is immutable once a state file exists.
- ``test_cli_generate_requires_pre_registration_and_known_state``:
  ``--generate`` refuses to run without the pre-registration, and a
  ``--state`` that is not a pre-registered decayable catalog state
  (a made-up pdg, a stable endpoint) is rejected before any file is
  written.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (schema, run, card, the 80 state
  rows with ranks 0..79, 59 important / 21 minor, 10,000/2,000
  per-point counts, 100 grid points with both endpoints inclusive,
  master seed 20260913, replica index 0, the event-index rule, the
  closed seed labels, the per-state expected degenerate-point
  inventory (17 skips: 7 important, 10 minor), the decay budget
  (63,200,000 all points; 63,110,000 over the open points), the
  machine-enumerated executed three-body channel set, and the 18
  no-decay stable endpoints).
- ``test_pre_registration_is_deterministic``: the committed driver
  recomputes the pre-registration byte-identical to the sealed file
  (the payload carries no wall-clock fields), and the provenance
  records the catalog SHA-256, the driver SHA-256 (nested tree and
  monorepo mirror byte-identical), and the 10-entry card chain.
- ``test_state_file_schema``: every one of the 80 sealed
  ``state_<pdg>.json`` files carries schema
  ``na49_mixing.staged_r20_unit_grid.state/1`` and the committed
  field set: the ``state`` block (pdg_id, name, rank,
  classification, decays_per_point, B/Q/S, pole_mass,
  nominal_width, mass_range, channels), the ``seed_map`` echo, the
  100-point grid on the committed formula
  ``M_j = mass_range[0] + (mass_range[1] - mass_range[0]) * j / 99``,
  the degenerate-point inventory, the per-point records (0 decays +
  skip reason at degenerate points, the full per-point count at open
  points, channel counts summing to the count, the 10-bin
  ``cos(theta)`` histogram), ``totals`` (n_decays, n_open_points,
  n_degenerate_points), and the ``deterministic`` closure /
  on-shell maxima at the 1e-9 tolerance.
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records 84 sealed files (pre-registration, provenance, aggregate,
  gate report, and the 80 state files; the manifest and the sums
  file exclude themselves) and the digest of ``SHA256SUMS.txt``,
  which is in ``sha256sum -c`` format (``<digest>  <relpath>``) and
  verifies against every sealed file on disk.
- ``test_gate2_report_structure``: ``gate2_r20.json`` carries the
  Gate 2 clauses with the committed verdict:
  ``local_conservation`` passes (all maxima at the 1e-9 tolerance),
  ``branching_test`` fails with 41 of 80 states passing,
  ``threshold_discontinuity`` fails, ``observables`` is the null
  placeholder and ``no_branching_fit`` records no fitting; the
  sealed form is marked and its ``aggregate_sha256`` reproduces the
  canonical bytes of ``aggregate.json``.  The overall pass=False
  verdict is the conjunction of the clause pass fields (the
  committed driver records the verdict per clause and writes no
  ``overall`` key).
- ``test_aggregate_report_structure``: ``aggregate.json`` carries
  the per-state verification entries (80, the committed field set),
  the sealed-input digests, the committed tolerances and notes, and
  totals consistent with the gate report (41 of 80 states pass,
  63,110,000 decays, the 1e-9 deterministic maxima, overall pass
  False).

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and the committed catalog; no generation runs.  If
the run dir is absent (a fresh checkout), the artifact pins skip
with a clear message; the driver-contract and CLI pins still run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r20_unit_grid.py"
MIRROR_PATH = MONO_ROOT / "helpers" / "benchmarking" / "staged_r20_unit_grid.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
RUN_NAME = "r20_single_resonance_unit_grid_2026-09-13"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

# ---------------------------------------------------------------------------
# The committed run contract (the driver enforces it; the sealed run
# realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R20"
CARD_ID = "TASK-20260913-r2-run-r20-single-resonance-unit-grid"
MASTER_SEED = 20260913
REPLICA_INDEX = 0
EVENT_INDEX_RULE = "event_index = s * 1_000_000 + j * 10_000 + i"
SEED_LABELS = ("decay_channel", "decay_kinematics")
N_GRID_POINTS = 100
IMPORTANT_DECAYS_PER_POINT = 10_000
MINOR_DECAYS_PER_POINT = 2_000
N_IMPORTANT = 59
N_MINOR = 21
N_STATES = N_IMPORTANT + N_MINOR
N_NO_DECAY_STATES = 18
N_SEALED_FILES = 4 + N_STATES  # + pre-registration, provenance, aggregate, gate
N_BRANCHING_PASS = 41
EXPECTED_SKIP_TOTAL = 17
EXPECTED_SKIPS_BY_CLASS = {"important": 7, "minor": 10}
BUDGET_ALL_POINTS = 63_200_000
EXPECTED_DECAYS_OPEN_POINTS = 63_110_000

PRE_SCHEMA = "na49_mixing.staged_r20_unit_grid.pre_registration/1"
PROV_SCHEMA = "na49_mixing.staged_r20_unit_grid.provenance/1"
STATE_SCHEMA = "na49_mixing.staged_r20_unit_grid.state/1"
AGG_SCHEMA = "na49_mixing.staged_r20_unit_grid.aggregate/1"
SEAL_SCHEMA = "na49_mixing.staged_r20_unit_grid.seal_manifest/1"

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")

#: The committed executed three-body channel set (machine-enumerated
#: from the committed catalog's ``is_three_body`` channels).
EXECUTED_THREE_BODY = [
    {"parent_pdg": 223, "parent": "omega",
     "daughters": [211, -211, 111],
     "branching_fraction": 0.9831367794555275},
    {"parent_pdg": 221, "parent": "eta",
     "daughters": [211, -211, 111], "branching_fraction": 0.2302},
    {"parent_pdg": 221, "parent": "eta",
     "daughters": [111, 111, 111], "branching_fraction": 0.3256},
    {"parent_pdg": 377, "parent": "eta-prime(958)",
     "daughters": [211, -211, 221], "branching_fraction": 0.425},
    {"parent_pdg": 377, "parent": "eta-prime(958)",
     "daughters": [111, 111, 221], "branching_fraction": 0.224},
    {"parent_pdg": 117, "parent": "rho3(1690)0",
     "daughters": [321, -321, 111],
     "branching_fraction": 0.08448199199644287},
    {"parent_pdg": -217, "parent": "rho3(1690)-",
     "daughters": [311, -321, 111],
     "branching_fraction": 0.08448199199644287},
    {"parent_pdg": 217, "parent": "rho3(1690)+",
     "daughters": [-311, 321, 111],
     "branching_fraction": 0.08448199199644287},
    {"parent_pdg": 102134, "parent": "Lambda(1520)",
     "daughters": [3122, 211, -211],
     "branching_fraction": 0.10309278350515465},
]

STATE_BLOCK_KEYS = {
    "pdg_id", "name", "rank", "classification", "decays_per_point",
    "B", "Q", "S", "pole_mass", "nominal_width", "mass_range",
    "channels",
}
DETERMINISTIC_KEYS = {
    "max_four_momentum_closure_deviation_gev",
    "max_on_shell_deviation_gev2",
    "max_on_shell_deviation_gev",
    "n_vertices_checked",
    "n_two_body_vertices",
    "n_three_body_vertices",
    "tolerance",
}
OPEN_POINT_KEYS = {
    "j", "mass_gev", "degenerate", "n_decays", "channel_counts",
    "n_two_body_vertices", "n_three_body_vertices", "cos_theta_hist",
    "max_four_momentum_closure_deviation_gev",
    "max_on_shell_deviation_gev2",
    "max_on_shell_deviation_gev",
}
DEGENERATE_POINT_KEYS = {
    "j", "mass_gev", "degenerate", "n_decays", "skip_reason",
}
AGG_PER_STATE_KEYS = {
    "name", "rank", "classification", "pole_point", "n_open_points",
    "n_degenerate_points", "n_decays", "branching_pole_max_abs_sigma",
    "branching_open_max_abs_sigma", "charge_max_abs_sigma",
    "angular_max_abs_sigma", "threshold_continuity_max_abs_sigma",
    "degenerate_points", "max_four_momentum_closure_deviation_gev",
    "max_on_shell_deviation_gev2", "max_on_shell_deviation_gev",
    "pass",
}
GATE_PER_STATE_KEYS = {
    "branching_max_abs_sigma", "charge_max_abs_sigma",
    "angular_max_abs_sigma", "degenerate_points", "pass",
}


@lru_cache(maxsize=1)
def _driver():
    """The committed driver module, loaded from its nested-tree path."""
    spec = importlib.util.spec_from_file_location(
        "staged_r20_unit_grid_card10_pins", str(DRIVER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict) -> str:
    """The driver's canonical report hash (indent-2 JSON, trailing
    newline, UTF-8)."""
    return hashlib.sha256(
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R20 run dir, skipped with a clear message when the
    gitignored artifact is absent (a fresh checkout)."""
    pre_path = RUN_DIR / "pre_registration.json"
    required = [
        "pre_registration.json",
        "provenance.json",
        "aggregate.json",
        "gate2_r20.json",
        "seal_manifest.json",
        "SHA256SUMS.txt",
    ]
    if not pre_path.is_file():
        pytest.skip(
            f"the sealed R20 run dir {RUN_DIR} is absent (gitignored "
            "artifact); restore the sealed run to execute the "
            "artifact pins"
        )
    missing = [name for name in required if not (RUN_DIR / name).is_file()]
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    missing += [
        f"state_{entry['pdg_id']}.json"
        for entry in pre["state_coverage"]["states"]
        if not (RUN_DIR / f"state_{entry['pdg_id']}.json").is_file()
    ]
    if missing:
        pytest.skip(
            f"the sealed R20 run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing) + "); restore the sealed run to "
            "execute the artifact pins"
        )
    return RUN_DIR


@pytest.fixture(scope="module")
def pre_registration(run_dir) -> dict:
    return _load("pre_registration.json")


@pytest.fixture(scope="module")
def state_files(pre_registration) -> dict[int, dict]:
    """All 80 sealed ``state_<pdg>.json`` payloads, keyed by pdg id."""
    return {
        entry["pdg_id"]: _load(f"state_{entry['pdg_id']}.json")
        for entry in pre_registration["state_coverage"]["states"]
    }


# ---------------------------------------------------------------------------
# Driver contract (no sealed artifact needed)
# ---------------------------------------------------------------------------
def test_driver_module_contract_constants():
    """The committed run contract: run identity, the 100-point grid,
    the 10,000/2,000 per-point counts, master seed 20260913 with
    replica index 0, the event-index rule, the closed seed labels,
    the tolerances, and the 59/21 important/minor split that exactly
    covers the catalog's non-stable records."""
    driver = _driver()
    assert driver.RUN_ID == RUN_ID
    assert driver.CARD_ID == CARD_ID
    assert driver.RUN_NAME == RUN_NAME
    assert driver.DEFAULT_RUN_DIR == f"dev/00runs/{RUN_NAME}"
    assert driver.GRID_POINTS == N_GRID_POINTS
    assert driver.IMPORTANT_DECAYS_PER_POINT == IMPORTANT_DECAYS_PER_POINT
    assert driver.MINOR_DECAYS_PER_POINT == MINOR_DECAYS_PER_POINT
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.REPLICA_INDEX == REPLICA_INDEX
    assert driver.SEED_RULE == EVENT_INDEX_RULE
    assert tuple(driver.SEED_LABELS) == SEED_LABELS
    assert driver.N_SIGMA == 3.0
    assert driver.DETERMINISTIC_TOL == 1e-9
    assert driver.ANGULAR_BINS == 10
    # the pre-registered important/minor split: 59 + 21, disjoint
    assert len(driver.IMPORTANT_STATES) == N_IMPORTANT
    assert len(driver.MINOR_STATES) == N_MINOR
    assert not (driver.IMPORTANT_STATES & driver.MINOR_STATES)
    # the split exactly covers the committed catalog's non-stable
    # records, without the state the orphan-record revision of 2026-10-29
    # removed (the driver's own row builder asserts this; run it)
    catalog = driver._load_catalog()
    non_stable = {
        int(record.pdg_id)
        for record in catalog.records()
        if not record.stable_for_decay
    }
    stable = {
        int(record.pdg_id)
        for record in catalog.records()
        if record.stable_for_decay
    }
    assert driver.REMOVED_BY_REVISION == {377}
    assert (driver.IMPORTANT_STATES | driver.MINOR_STATES) - driver.REMOVED_BY_REVISION == non_stable
    assert len(stable) == N_NO_DECAY_STATES
    rows = driver._state_rows(catalog)
    assert len(rows) == N_STATES - len(driver.REMOVED_BY_REVISION)
    assert sum(1 for row in rows
               if row["classification"] == "important") == N_IMPORTANT
    assert sum(1 for row in rows
               if row["classification"] == "minor") == N_MINOR - len(driver.REMOVED_BY_REVISION & driver.MINOR_STATES)


def test_event_index_formula_partitions_states():
    """The committed event-index rule partitions the index space: with
    100 points (j below 100) and per-point counts below 10,000 (i
    below 10,000), the term ``j * 10_000 + i`` stays below 1_000_000,
    so state s occupies the disjoint block
    ``[s * 1_000_000, (s + 1) * 1_000_000)`` and the 80 states
    partition ``[0, 80 * 1_000_000)`` without overlap (79 states on the
    catalog after the orphan-record revision of 2026-10-29)."""
    driver = _driver()
    assert driver.SEED_RULE == EVENT_INDEX_RULE
    catalog = driver._load_catalog()
    rows = driver._state_rows(catalog)
    blocks: list[tuple[int, int]] = []
    for row in rows:
        s = row["rank"]
        i_max = row["decays_per_point"] - 1  # the decay index bound
        assert i_max < 10_000, (
            f"state rank {s} has {row['decays_per_point']} decays per "
            "point; the decay index would spill into the point slot"
        )
        low = s * 1_000_000
        high = low + (N_GRID_POINTS - 1) * 10_000 + i_max
        blocks.append((low, high))
        assert high < (s + 1) * 1_000_000, (
            f"state rank {s} spills into the next state's block"
        )
    # the blocks are disjoint and cover the index space in rank order
    for (low_a, high_a), (low_b, high_b) in zip(blocks, blocks[1:]):
        assert high_a + 1 <= low_b, (
            f"state blocks overlap or leave a gap: "
            f"[{low_a}, {high_a}] vs [{low_b}, {high_b}]"
        )
    assert blocks[0][0] == 0
    assert blocks[-1][1] == (N_STATES - len(driver.REMOVED_BY_REVISION)) * 1_000_000 - 1


def test_cli_modes_and_flags():
    """The CLI exposes exactly the staged modes (a required mutually
    exclusive group) plus --state (int, only with --generate) and
    --run-dir; --state without --generate and a bare invocation are
    rejected."""
    driver = _driver()
    parser = driver._build_parser()
    # every staged mode parses and selects exactly one step flag
    for mode in (
        ["--pre-registration"],
        ["--generate"],
        ["--generate", "--state", "1114"],
        ["--verify"],
        ["--seal"],
        ["--clean-validate"],
    ):
        args = parser.parse_args(mode)
        flags = [
            value for name, value in vars(args).items()
            if name in (
                "pre_registration", "generate", "verify", "seal",
                "clean_validate",
            )
        ]
        assert sum(bool(flag) for flag in flags) == 1, (
            f"{mode}: the mutually exclusive group must select exactly "
            "one mode"
        )
    args = parser.parse_args(["--generate", "--state", "1114"])
    assert args.generate is True
    assert args.state == 1114  # --state is parsed as an int
    args = parser.parse_args(["--verify", "--run-dir", "some/other/dir"])
    assert args.run_dir == "some/other/dir"
    assert parser.parse_args(["--verify"]).run_dir == driver.DEFAULT_RUN_DIR
    # --state is only valid together with --generate
    with pytest.raises(SystemExit):
        driver.main(["--state", "1114"])
    # a bare invocation (no staged mode) is rejected by the required
    # mutually exclusive group
    with pytest.raises(SystemExit):
        driver.main([])
    # a non-integer --state is rejected by argparse
    with pytest.raises(SystemExit):
        parser.parse_args(["--generate", "--state", "not-a-pdg"])


def test_cli_pre_registration_writes_only_pre_registration(tmp_path):
    """The --pre-registration step writes exactly pre_registration.json
    and provenance.json (nothing is generated), and is immutable once
    a state file exists in the run dir."""
    driver = _driver()
    run_dir = tmp_path / "r20_pins_pre_registration"
    assert driver.main([
        "--pre-registration", "--run-dir", str(run_dir),
    ]) == 0
    names = sorted(path.name for path in run_dir.iterdir())
    assert names == ["pre_registration.json", "provenance.json"], (
        "the pre-registration step must write exactly the "
        "pre-registration and the provenance (nothing is generated)"
    )
    pre = json.loads(
        (run_dir / "pre_registration.json").read_text(encoding="utf-8")
    )
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_generation"] is True
    assert pre["state_coverage"]["n_decay_states"] == N_STATES - len(driver.REMOVED_BY_REVISION)
    assert pre["seeds"]["master_seed"] == MASTER_SEED
    assert pre["seeds"]["replica_index"] == REPLICA_INDEX
    # the pre-registration is immutable after generation has started
    (run_dir / "state_1114.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="immutable"):
        driver.main([
            "--pre-registration", "--run-dir", str(run_dir),
        ])


def test_cli_generate_requires_pre_registration_and_known_state(tmp_path):
    """--generate refuses to run without the pre-registration, and a
    --state that is not a pre-registered decayable catalog state (a
    made-up pdg, a stable endpoint) is rejected before any state file
    is written."""
    driver = _driver()
    run_dir = tmp_path / "r20_pins_generate_guard"
    # no pre-registration: the run stops before any generation
    for argv in (
        ["--generate", "--run-dir", str(run_dir)],
        ["--generate", "--state", "999999", "--run-dir", str(run_dir)],
    ):
        with pytest.raises(RuntimeError, match="pre-registration"):
            driver.main(argv)
    assert not list(run_dir.glob("state_*.json"))
    # with the pre-registration: an unknown pdg and a stable endpoint
    # are rejected (stable endpoints have no decays to generate)
    assert driver.main([
        "--pre-registration", "--run-dir", str(run_dir),
    ]) == 0
    for pdg in ("999999", "2212"):
        with pytest.raises(SystemExit,
                           match="not a pre-registered decayable"):
            driver.main([
                "--generate", "--state", pdg, "--run-dir", str(run_dir),
            ])
    assert not list(run_dir.glob("state_*.json"))


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_pre_registration_structure(run_dir, pre_registration):
    """The sealed pre-registration carries the committed contract:
    80 state rows (59 important / 21 minor, ranks 0..79), 10,000/2,000
    per-point counts, the 100-point inclusive grid, master seed
    20260913 with replica index 0, the event-index rule, the closed
    seed labels, the 17 expected degenerate-point skips (7 important,
    10 minor), the decay budget, the executed three-body channel set,
    and the 18 no-decay stable endpoints."""
    driver = _driver()
    pre = pre_registration
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_generation"] is True

    coverage = pre["state_coverage"]
    assert coverage["n_important"] == N_IMPORTANT
    assert coverage["n_minor"] == N_MINOR
    assert coverage["n_decay_states"] == N_STATES
    assert coverage["n_no_decay_states"] == N_NO_DECAY_STATES
    assert coverage["important_pdgs"] == sorted(driver.IMPORTANT_STATES)
    assert coverage["minor_pdgs"] == sorted(driver.MINOR_STATES)
    no_decay = coverage["no_decay_states"]
    assert len(no_decay) == N_NO_DECAY_STATES
    assert all(set(entry) == {"pdg_id", "name"} for entry in no_decay)
    catalog = driver._load_catalog()
    assert {entry["pdg_id"] for entry in no_decay} == {
        int(record.pdg_id)
        for record in catalog.records()
        if record.stable_for_decay
    }

    states = coverage["states"]
    assert len(states) == N_STATES
    assert [entry["rank"] for entry in states] == list(range(N_STATES))
    for entry in states:
        assert set(entry) == {
            "rank", "pdg_id", "name", "classification",
            "decays_per_point", "pole_mass", "nominal_width",
            "mass_range", "channel_branching_sum", "channels",
            "expected_skip_points", "expected_skip_count",
            "open_points", "expected_decays",
        }
        assert entry["classification"] in ("important", "minor")
        expected_count = {
            "important": IMPORTANT_DECAYS_PER_POINT,
            "minor": MINOR_DECAYS_PER_POINT,
        }[entry["classification"]]
        assert entry["decays_per_point"] == expected_count
        lo, hi = (float(x) for x in entry["mass_range"])
        assert lo < hi
        # the pole may sit outside the mass range (committed catalog
        # data, e.g. Delta(1600) at 1.6 below its 1.7135 range low);
        # the driver's pole-point rule is the closest open grid point
        assert float(entry["pole_mass"]) > 0.0
        assert float(entry["nominal_width"]) >= 0.0
        assert float(entry["channel_branching_sum"]) > 0.0
        assert entry["channels"], (
            f"state {entry['pdg_id']} has no pre-registered channels"
        )
        for channel in entry["channels"]:
            assert set(channel) == {
                "daughters", "branching_fraction", "is_three_body",
                "orbital_angular_momentum",
            }
        assert entry["expected_skip_count"] == len(
            entry["expected_skip_points"]
        )
        assert entry["open_points"] == N_GRID_POINTS - entry[
            "expected_skip_count"
        ]
        assert entry["expected_decays"] == (
            entry["open_points"] * entry["decays_per_point"]
        )
    assert sum(1 for entry in states
               if entry["classification"] == "important") == N_IMPORTANT
    assert sum(1 for entry in states
               if entry["classification"] == "minor") == N_MINOR

    counts = pre["counts"]
    assert counts["important_decays_per_point"] == (
        IMPORTANT_DECAYS_PER_POINT
    )
    assert counts["minor_decays_per_point"] == MINOR_DECAYS_PER_POINT

    seeds = pre["seeds"]
    assert seeds["master_seed"] == MASTER_SEED
    assert seeds["replica_index"] == REPLICA_INDEX
    assert seeds["event_index_rule"] == EVENT_INDEX_RULE
    assert seeds["substream_labels"] == list(SEED_LABELS)

    grid = pre["grid"]
    assert grid["points_per_state"] == N_GRID_POINTS
    assert grid["endpoints_inclusive"] is True
    assert "j / 99" in grid["formula"]

    degenerate = pre["degenerate_point_rule"]
    assert degenerate["expected_skip_count_total"] == EXPECTED_SKIP_TOTAL
    per_state_skips = degenerate["expected_skip_points_per_state"]
    assert set(per_state_skips) == {str(entry["pdg_id"]) for entry in states}
    for entry in states:
        assert per_state_skips[str(entry["pdg_id"])] == list(
            entry["expected_skip_points"]
        )

    three_body = pre["three_body_channels"]
    approved = three_body["approved_card7_set"]
    assert [row["parent_pdg"] for row in approved] == [
        223, 333, 20113, 113, 2224,
    ]
    assert [row for row in approved if row["in_committed_catalog"]] == [
        row for row in approved if row["parent_pdg"] == 223
    ]
    assert three_body["executed_set"] == EXECUTED_THREE_BODY

    plan = pre["generation_plan"]
    assert plan["total_decays_budget_all_points"] == BUDGET_ALL_POINTS
    assert BUDGET_ALL_POINTS == (
        N_IMPORTANT * N_GRID_POINTS * IMPORTANT_DECAYS_PER_POINT
        + N_MINOR * N_GRID_POINTS * MINOR_DECAYS_PER_POINT
    )
    assert plan["total_expected_decays_open_points"] == (
        EXPECTED_DECAYS_OPEN_POINTS
    )
    assert plan["total_expected_decays_open_points"] == (
        sum(entry["expected_decays"] for entry in states)
    )
    assert plan["expected_skip_points_total"] == EXPECTED_SKIP_TOTAL
    assert plan["expected_skips_by_class"] == EXPECTED_SKIPS_BY_CLASS
    assert plan["expected_skip_points_total"] == (
        sum(entry["expected_skip_count"] for entry in states)
    )


def test_pre_registration_is_deterministic(run_dir, pre_registration):
    """The committed driver recomputes the pre-registration
    byte-identical to the sealed file (the payload carries no
    wall-clock fields), and the provenance records the catalog
    SHA-256, the driver SHA-256 (the nested tree and the monorepo
    mirror are byte-identical), and the 10-entry card chain ending at
    the run card."""
    driver = _driver()
    recomputed = driver._build_pre_registration(driver._load_catalog())
    recomputed_bytes = (json.dumps(recomputed, indent=2) + "\n").encode(
        "utf-8"
    )
    assert recomputed_bytes == (
        run_dir / "pre_registration.json"
    ).read_bytes(), (
        "the committed driver does not reproduce the sealed "
        "pre-registration byte for byte"
    )

    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert prov["card_chain"] == list(driver.CARD_CHAIN)
    assert len(prov["card_chain"]) == 10
    assert prov["card_chain"][0] == (
        "TASK-20260913-r2-hadron-resonance-schema"
    )
    assert prov["card_chain"][-1] == CARD_ID
    assert prov["catalog"]["path"] == (
        "config/v2_singlet/resonance_catalog.json"
    )
    assert prov["catalog"]["sha256"] == _file_sha256(CATALOG_PATH)
    assert prov["driver"] == "helpers/benchmarking/staged_r20_unit_grid.py"
    assert prov["driver_sha256"] == _file_sha256(DRIVER_PATH)
    # dual-tree driver: the monorepo mirror is byte-identical
    assert _file_sha256(MIRROR_PATH) == _file_sha256(DRIVER_PATH)
    assert prov["written_before_generation"] is True
    for key in ("nested_head", "outer_head",
                "nested_dirty_entries", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in prov["environment"]


def test_state_file_schema(run_dir, pre_registration, state_files):
    """Every one of the 80 sealed state files carries schema
    na49_mixing.staged_r20_unit_grid.state/1 and the committed field
    set: the state block (pdg_id, name, rank, classification,
    decays_per_point, B/Q/S, pole_mass, nominal_width, mass_range,
    channels), the seed_map echo, the 100-point grid on the committed
    formula, the degenerate-point inventory, the per-point records
    (0 decays + skip reason at degenerate points, the full per-point
    count at open points, channel counts summing to the count, the
    10-bin cos(theta) histogram), totals (n_decays, n_open_points,
    n_degenerate_points), and the deterministic closure / on-shell
    maxima at the 1e-9 tolerance."""
    driver = _driver()
    pre_by_pdg = {
        entry["pdg_id"]: entry
        for entry in pre_registration["state_coverage"]["states"]
    }
    assert set(state_files) == set(pre_by_pdg)
    assert len(state_files) == N_STATES
    for pdg, payload in state_files.items():
        pre_entry = pre_by_pdg[pdg]
        assert payload["schema"] == STATE_SCHEMA, (
            f"state_{pdg}.json: wrong schema"
        )
        assert payload["run"] == RUN_ID
        assert payload["card"] == CARD_ID

        state = payload["state"]
        assert set(state) == STATE_BLOCK_KEYS
        assert state["pdg_id"] == pdg, (
            f"state_{pdg}.json: the state block carries pdg "
            f"{state['pdg_id']}"
        )
        assert state["name"] == pre_entry["name"]
        assert state["rank"] == pre_entry["rank"]
        assert state["classification"] == pre_entry["classification"]
        assert state["decays_per_point"] == pre_entry["decays_per_point"]
        assert all(isinstance(state[key], int)
                   for key in ("B", "Q", "S"))
        assert state["pole_mass"] == pre_entry["pole_mass"]
        assert state["nominal_width"] == pre_entry["nominal_width"]
        assert state["mass_range"] == list(pre_entry["mass_range"])
        assert state["channels"] == pre_entry["channels"]

        seed_map = payload["seed_map"]
        assert seed_map["master_seed"] == MASTER_SEED
        assert seed_map["replica_index"] == REPLICA_INDEX
        assert seed_map["event_index_rule"] == EVENT_INDEX_RULE
        assert seed_map["substream_labels"] == list(SEED_LABELS)

        grid = payload["grid"]
        assert grid["points"] == N_GRID_POINTS
        lo, hi = (float(x) for x in state["mass_range"])
        expected_masses = [lo + (hi - lo) * j / 99
                           for j in range(N_GRID_POINTS)]
        assert [float(x) for x in grid["masses_gev"]] == expected_masses

        degenerate_js = sorted(
            int(entry["j"]) for entry in payload["degenerate_points"]
        )
        assert degenerate_js == sorted(pre_entry["expected_skip_points"])
        for entry in payload["degenerate_points"]:
            assert entry["reason"] == driver.DEGENERATE_SKIP_REASON

        points = payload["points"]
        assert sorted(int(key) for key in points) == list(
            range(N_GRID_POINTS)
        )
        dpp = state["decays_per_point"]
        n_two_total = 0
        n_three_total = 0
        for j in range(N_GRID_POINTS):
            point = points[str(j)]
            if j in degenerate_js:
                assert set(point) == DEGENERATE_POINT_KEYS
                assert point["degenerate"] is True
                assert point["n_decays"] == 0
                assert point["skip_reason"] == (
                    driver.DEGENERATE_SKIP_REASON
                )
            else:
                assert set(point) == OPEN_POINT_KEYS
                assert point["degenerate"] is False
                assert point["n_decays"] == dpp
                assert point["mass_gev"] == expected_masses[j]
                counts = {
                    str(key): int(value)
                    for key, value in point["channel_counts"].items()
                }
                assert sum(counts.values()) == dpp
                hist = [int(x) for x in point["cos_theta_hist"]]
                assert len(hist) == driver.ANGULAR_BINS
                assert sum(hist) == int(point["n_two_body_vertices"])
                n_two_total += int(point["n_two_body_vertices"])
                n_three_total += int(point["n_three_body_vertices"])
                for key in (
                    "max_four_momentum_closure_deviation_gev",
                    "max_on_shell_deviation_gev2",
                    "max_on_shell_deviation_gev",
                ):
                    assert float(point[key]) <= driver.DETERMINISTIC_TOL

        totals = payload["totals"]
        assert set(totals) == {
            "n_decays", "n_open_points", "n_degenerate_points",
        }
        n_open = N_GRID_POINTS - len(degenerate_js)
        assert totals["n_open_points"] == n_open
        assert totals["n_degenerate_points"] == len(degenerate_js)
        assert totals["n_open_points"] == pre_entry["open_points"]
        assert totals["n_decays"] == n_open * dpp
        assert totals["n_decays"] == pre_entry["expected_decays"]

        deterministic = payload["deterministic"]
        assert set(deterministic) == DETERMINISTIC_KEYS
        assert deterministic["tolerance"] == driver.DETERMINISTIC_TOL
        for key in (
            "max_four_momentum_closure_deviation_gev",
            "max_on_shell_deviation_gev2",
            "max_on_shell_deviation_gev",
        ):
            assert 0.0 <= float(deterministic[key]) <= (
                driver.DETERMINISTIC_TOL
            )
        # every vertex is two-body or three-body: the checked vertex
        # count is the sum of the per-point vertex counts
        assert deterministic["n_vertices_checked"] == (
            n_two_total + n_three_total
        )
        # the state-level two/three counters are the driver's
        # per-event running sums (each per-event addition carries the
        # point-level accumulator), so they are never below the true
        # per-point totals
        assert int(deterministic["n_two_body_vertices"]) >= n_two_total
        assert int(deterministic["n_three_body_vertices"]) >= n_three_total


def test_seal_manifest_and_sha256_sums(run_dir, pre_registration):
    """seal_manifest.json records the 84 sealed files (pre-registration,
    provenance, aggregate, gate report, and the 80 state files; the
    manifest and the sums file exclude themselves) and the digest of
    SHA256SUMS.txt, which is in sha256sum -c format and verifies
    against every sealed file on disk."""
    manifest = _load("seal_manifest.json")
    assert manifest["schema"] == SEAL_SCHEMA
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert manifest["file_count"] == N_SEALED_FILES
    assert manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt",
    ]

    sums_path = run_dir / "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == _file_sha256(sums_path)
    text = sums_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    lines = text.splitlines()
    assert len(lines) == N_SEALED_FILES
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format: {line!r}"
        )

    expected = (
        {"pre_registration.json", "provenance.json", "aggregate.json",
         "gate2_r20.json"}
        | {f"state_{entry['pdg_id']}.json"
           for entry in pre_registration["state_coverage"]["states"]}
    )
    assert len(expected) == N_SEALED_FILES
    entries = [line.split("  ", 1) for line in lines]
    assert {rel for _digest, rel in entries} == expected
    for digest, rel in entries:
        assert _file_sha256(run_dir / rel) == digest, (
            f"the sealed digest of {rel} does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    assert manifest["total_bytes"] == sum(
        (run_dir / rel).stat().st_size for _digest, rel in entries
    )


def test_gate2_report_structure(run_dir, pre_registration):
    """gate2_r20.json carries the Gate 2 clauses with the committed
    verdict: local_conservation passes (all maxima at the 1e-9
    tolerance), branching_test fails with 41 of 80 states passing,
    threshold_discontinuity fails, observables is the null
    placeholder, no_branching_fit records no fitting, the sealed form
    is marked, and aggregate_sha256 reproduces the canonical bytes of
    aggregate.json.  The overall pass=False verdict is the
    conjunction of the clause pass fields (the committed driver
    records the verdict per clause and writes no overall key)."""
    driver = _driver()
    gate = _load("gate2_r20.json")
    assert gate["gate"] == "gate 2 (plan section 8, line 575)"
    assert gate["run"] == RUN_ID
    assert gate["sealed"] is True
    assert "note" in gate
    assert "overall" not in gate
    aggregate = _load("aggregate.json")
    assert gate["aggregate_sha256"] == _canonical_sha256(aggregate)

    clauses = gate["clauses"]
    assert set(clauses) == {
        "local_conservation", "branching_test",
        "threshold_discontinuity", "observables", "no_branching_fit",
    }

    local = clauses["local_conservation"]
    assert set(local) == {
        "max_four_momentum_closure_deviation_gev",
        "max_on_shell_deviation_gev",
        "max_on_shell_deviation_gev2",
        "pass",
    }
    assert local["pass"] is True
    for key in (
        "max_four_momentum_closure_deviation_gev",
        "max_on_shell_deviation_gev",
        "max_on_shell_deviation_gev2",
    ):
        assert 0.0 <= float(local[key]) <= driver.DETERMINISTIC_TOL

    branching = clauses["branching_test"]
    per_state = branching["per_state"]
    assert set(per_state) == {
        str(entry["pdg_id"])
        for entry in pre_registration["state_coverage"]["states"]
    }
    assert len(per_state) == N_STATES
    n_pass = 0
    for str_pdg, entry in per_state.items():
        assert set(entry) == GATE_PER_STATE_KEYS
        # the committed per-state gate logic: branching, charge, and
        # angular all within 3 sigma
        assert entry["pass"] == bool(
            entry["branching_max_abs_sigma"] <= driver.N_SIGMA
            and entry["charge_max_abs_sigma"] <= driver.N_SIGMA
            and entry["angular_max_abs_sigma"] <= driver.N_SIGMA
        )
        n_pass += bool(entry["pass"])
    assert n_pass == N_BRANCHING_PASS
    assert branching["pass"] is False

    threshold = clauses["threshold_discontinuity"]
    assert set(threshold) == {
        "two_lowest_points_max_abs_sigma", "pass",
    }
    assert threshold["pass"] is False
    assert float(threshold["two_lowest_points_max_abs_sigma"]) > (
        driver.N_SIGMA
    )

    assert clauses["observables"] is None
    assert clauses["no_branching_fit"] == {"no_fitting": True}

    # the overall pass=False verdict: the conjunction of the clause
    # pass fields (local conservation passes; the branching test and
    # the threshold-discontinuity clause fail)
    overall = all(
        clauses[name]["pass"]
        for name in (
            "local_conservation", "branching_test",
            "threshold_discontinuity",
        )
    )
    assert overall is False


def test_aggregate_report_structure(run_dir, pre_registration):
    """aggregate.json carries the per-state verification entries (80,
    the committed field set), the sealed-input digests, the committed
    tolerances and notes, and totals consistent with the gate report
    (41 of 80 states pass, 63,110,000 decays, the 1e-9 deterministic
    maxima, overall pass False)."""
    driver = _driver()
    aggregate = _load("aggregate.json")
    assert aggregate["schema"] == AGG_SCHEMA
    assert aggregate["run"] == RUN_ID
    assert aggregate["card"] == CARD_ID
    assert aggregate["pre_registration_sha256"] == _file_sha256(
        run_dir / "pre_registration.json"
    )
    assert aggregate["provenance_sha256"] == _file_sha256(
        run_dir / "provenance.json"
    )
    assert aggregate["tolerances"] == {
        "n_sigma": driver.N_SIGMA,
        "deterministic_tol": driver.DETERMINISTIC_TOL,
        "angular_bins": driver.ANGULAR_BINS,
    }
    assert set(aggregate["notes"]) == {"pole_expected", "continuity"}

    per_state = aggregate["per_state"]
    assert set(per_state) == {
        str(entry["pdg_id"])
        for entry in pre_registration["state_coverage"]["states"]
    }
    assert len(per_state) == N_STATES
    pre_by_pdg = {
        entry["pdg_id"]: entry
        for entry in pre_registration["state_coverage"]["states"]
    }
    n_pass = 0
    for str_pdg, entry in per_state.items():
        assert set(entry) == AGG_PER_STATE_KEYS
        assert entry["rank"] == pre_by_pdg[int(str_pdg)]["rank"]
        assert entry["n_degenerate_points"] == (
            pre_by_pdg[int(str_pdg)]["expected_skip_count"]
        )
        assert entry["n_open_points"] == (
            pre_by_pdg[int(str_pdg)]["open_points"]
        )
        n_pass += bool(entry["pass"])
    assert n_pass == N_BRANCHING_PASS
    assert aggregate["pass"] is False

    totals = aggregate["totals"]
    assert totals["n_states"] == N_STATES
    assert totals["n_states_pass"] == N_BRANCHING_PASS
    assert totals["n_decays"] == EXPECTED_DECAYS_OPEN_POINTS
    assert totals["n_decays"] == sum(
        int(entry["n_decays"]) for entry in per_state.values()
    )
    # the deterministic maxima agree between the two sealed reports
    gate = _load("gate2_r20.json")
    local = gate["clauses"]["local_conservation"]
    for key in (
        "max_four_momentum_closure_deviation_gev",
        "max_on_shell_deviation_gev",
        "max_on_shell_deviation_gev2",
    ):
        assert float(totals[key]) == float(local[key])
        assert 0.0 <= float(totals[key]) <= driver.DETERMINISTIC_TOL
