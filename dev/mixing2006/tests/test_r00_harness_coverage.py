"""Coverage pins for the R00 deterministic-correctness driver.

Companion to ``tests/test_r00_correctness_harness.py`` (the three
card-named hermetic checks of card
``TASK-20260911-r1-run-r00-deterministic-correctness``); this module
pins the committed behavior around them.  All checks are hermetic and
small-N: the largest generation here is a 20-event audit seed and a
4-event deterministic seed, driven either through the driver's
subprocess CLI or through its in-process helpers.  The 50,000-event
core and the 1,000,000-event audit are the R00 run itself, not tests.

CLI surface (subprocess, the card's smoke-config path):

- ``test_cli_audit_mode_tiny_n``: ``--mode audit`` at 20 events on one
  pre-registered audit seed (20312000) writes the audit stage and the
  per-seed file with the 1M-audit field names at small scale, the
  canonical pre-registration is written to a fresh ``--out`` directory
  that the driver creates, and no ``gate0_r00.json`` and no seal pair
  exist for a single-stage run.
- ``test_cli_all_mode_runs_both_stages_assembles_gate0_and_seals``:
  ``--mode all`` runs the deterministic core first and the audit
  second, assembles ``gate0_r00.json`` (the card's acceptance schema,
  with the per-seed projection of ``gate0_per_seed_fields``), seals the
  run directory, and exits non-zero because the ad-hoc single-seed
  registration plus the smoke config cannot verify against the
  pre-registered production record (verdict ``fail`` is the committed
  outcome for any non-pre-registered run).
- ``test_cli_unknown_mode_exits_nonzero``: an unknown ``--mode`` is an
  argparse refusal (exit 2, the three choices named) before the
  ``--out`` directory is created.
- ``test_cli_zero_events_refused_before_generation`` and
  ``test_cli_negative_events_refused_before_generation``: a zero or
  negative ``--events-per-seed`` is refused with exit 2 before the
  ``--out`` directory is created or any event is generated; the
  pre-registration pins positive event counts per seed (5,000 core /
  100,000 audit), so a degenerate count cannot mask a misconfigured run.
- ``test_cli_unregistered_config_name_refused``: any config file name
  other than the two bundled legacy_v1 configs is refused with exit 2
  before verification.
- ``test_cli_seed_outside_preregistration_recorded_not_refused``: an
  ad-hoc seed outside the pre-registered lists (999999) is accepted by
  the single-stage CLI, runs, and is recorded in its per-seed file; the
  pre-registration file is written unchanged.  The deviation is only
  adjudicated when both stages assemble the gate record (see the
  all-mode test above).

Pre-registration enforcement (in-process, no generation):

- ``test_verify_config_hash_preregistration_mismatch``: the pristine
  production config verifies, and a mutated ``config_hash`` in the
  pre-registration refuses with exit code 2.
- ``test_verify_m8_parameter_hash_mismatch``: the M8 parameter hash is
  verified at runtime against card 1's manifest (read from the fixed
  nested-root path) and the driver's verbatim frozen copy, not from the
  config file; a mismatched hash or a drifted parameter copy refuses
  with exit code 2 before any generation.

Aggregation and mutation sensitivity (hand-constructed staged events,
no generation at all):

- ``test_hand_constructed_bqs_residual_isolated``: a clean two-source
  batch audits to zero counters; swapping one daughter's PDG identity
  (pi+ -> pi-, same mass, different charge) on a copy injects exactly
  one per-vertex B/Q/S residual while the four-momentum and mass-shell
  counters stay at zero.
- ``test_hand_constructed_offshell_daughter``: scaling one daughter's
  four-vector by ``1 + 1e-6`` on a copy (a ~1e-6 relative mass
  deviation) reports ``offshell_daughters == 1``.
- ``test_mutation_tolerance_boundary``: on the same hand-constructed
  batch a 1e-6 component shift is flagged while a 1e-12 shift is not,
  pinning both sides of the existing 1e-10 relative tolerance default.

Reproducibility machinery and output schema (driver helpers):

- ``test_reproducibility_machinery_in_process``:
  ``run_deterministic_seed`` on one 4-event seed batch returns
  ``bit_identical`` true with equal event-hash and four-vector digests
  across the two worker-count runs; a p4 mutation injected between the
  two runs (through the driver's own ``_map_chunks`` seam) flips the
  bit-identical flag while the unmutated run's digest stays
  run-to-run stable.
- ``test_deterministic_output_payload_across_runs``: two driver runs on
  the same tiny input write identical physics payload (event-hash
  digests, four-vector digests, audit summaries, pre-registration
  bytes); the written files are not byte-identical across runs because
  they embed wall-clock and UTC timing fields, which is recorded here
  as a finding rather than fixed.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.benchmarking import staged_r00_correctness as r00
from na49_mixing.event_conservation import audit_staged_event, vertex_audit_summary
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.source_event import ExcitationEvent, FinalParticle, SourceRecord

NESTED_ROOT = Path(r00.NESTED_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r00_correctness.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
PRODUCTION_CONFIG = NESTED_ROOT / "config" / "staged_excitation.json"

CORE_SEED = 20311000
AUDIT_SEED = 20312000
OUT_OF_RANGE_SEED = 999999
SUBPROCESS_TIMEOUT = 600

TOLERANCE = 1e-10
ABOVE_TOLERANCE_SHIFT = 1e-6
BELOW_TOLERANCE_SHIFT = 1e-12

# The four-momentum and mass-shell counters of the run-level summary.
RUN_COUNTERS = (
    "conservation_failures",
    "offshell_daughters",
    "bqs_residual_nonzero",
    "ancestry_violations",
)


def _run_driver(extra_args, out_dir: Path):
    """One driver subprocess invocation from the nested tree root."""
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--config", str(SMOKE_CONFIG)]
        + [str(a) for a in extra_args]
        + ["--out", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Hand-constructed staged batch: the frozen legacy_v1 two-source ontology
# (two (1,1,0) sources) with exact four-momentum closure, so mutations can
# be injected one check at a time without any event generation.
# ---------------------------------------------------------------------------


def _on_shell(px: float, py: float, pz: float, mass: float) -> FourVector:
    return FourVector(math.sqrt(mass * mass + px * px + py * py + pz * pz), px, py, pz)


def _make_hand_event(event_id: int = 0) -> ExcitationEvent:
    """Two (1,1,0) sources; source 0 -> p + pi0, source 1 -> n + pi+.

    Each source four-vector is the exact floating-point sum of its
    daughters, so the clean batch audits to zero four-momentum
    residual; every daughter sits on its particle-database mass shell.
    """
    proton = _on_shell(0.1, 0.0, 0.2, get_mass(2212))
    pi0 = _on_shell(-0.1, 0.0, -0.2, get_mass(111))
    neutron = _on_shell(0.0, 0.15, -0.1, get_mass(2112))
    pi_plus = _on_shell(0.0, -0.15, 0.1, get_mass(211))
    source0 = SourceRecord(
        source_id=0,
        role="source",
        p4=proton + pi0,
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )
    source1 = SourceRecord(
        source_id=1,
        role="source",
        p4=neutron + pi_plus,
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )
    return ExcitationEvent(
        event_id=event_id,
        stage="r00_coverage_hand",
        incoming=[proton, proton],  # not read by the staged audit
        incoming_quantum_numbers=QuantumNumbers(2, 2, 0),
        sources=[source0, source1],
        final_particles=[
            FinalParticle(
                particle_id=0, pdg_id=2212, p4=proton,
                source_id=0, parent_id=0, channel_family="continuum",
            ),
            FinalParticle(
                particle_id=1, pdg_id=111, p4=pi0,
                source_id=0, parent_id=0, channel_family="continuum",
            ),
            FinalParticle(
                particle_id=2, pdg_id=2112, p4=neutron,
                source_id=1, parent_id=1, channel_family="continuum",
            ),
            FinalParticle(
                particle_id=3, pdg_id=211, p4=pi_plus,
                source_id=1, parent_id=1, channel_family="continuum",
            ),
        ],
    )


def _summary_of(events: list) -> dict:
    return vertex_audit_summary(
        [
            audit_staged_event(event, event_index=event.event_id)
            for event in events
        ]
    )


def _assert_clean(summary: dict, n_events: int) -> None:
    assert summary["n_events"] == n_events
    assert summary["n_vertices"] == 2 * n_events
    for counter in RUN_COUNTERS:
        assert summary[counter] == 0
    assert summary["all_pass"] is True


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_audit_mode_tiny_n(tmp_path):
    """--mode audit at 20 events writes the 1M-audit schema at small scale."""
    out_dir = tmp_path / "audit_run"
    assert not out_dir.is_dir()
    proc = _run_driver(
        ["--mode", "audit", "--events-per-seed", "20", "--seeds", str(AUDIT_SEED)],
        out_dir,
    )
    assert proc.returncode == 0, proc.stderr
    # The driver creates the fresh --out directory.
    assert out_dir.is_dir()
    assert _load(out_dir / r00.PREREGISTRATION_NAME) == r00.default_pre_registration()
    assert (out_dir / f"audit_seed_{AUDIT_SEED}.json").is_file()
    assert not (out_dir / r00.GATE_RECORD_NAME).is_file()
    # A single-stage run is not sealed; the seal pair needs both stages.
    assert not (out_dir / r00.SHA256SUMS_NAME).is_file()
    assert not (out_dir / r00.SEAL_NAME).is_file()

    stage = _load(out_dir / r00.STAGE_AUDIT_NAME)
    # The 1M-audit field names at the small scale.
    assert stage["seeds"] == [AUDIT_SEED]
    assert stage["events_per_seed"] == 20
    assert stage["total_events"] == 20
    assert stage["config_sha256"] == r00.SMOKE_CONFIG_SHA256
    assert stage["pre_registered_config_hash"] == r00.PRODUCTION_CONFIG_SHA256
    assert stage["m8_parameter_hash"] == r00.M8_PARAMETER_HASH
    assert stage["event_seed_rule"] == r00.EVENT_SEED_RULE
    for field in RUN_COUNTERS + (
        "max_four_momentum_relative_residual",
        "max_mass_shell_relative_deviation",
        "all_pass",
        "per_seed",
    ):
        assert field in stage
    for counter in RUN_COUNTERS:
        assert stage[counter] == 0
    assert stage["max_four_momentum_relative_residual"] <= TOLERANCE
    assert stage["max_mass_shell_relative_deviation"] <= TOLERANCE
    assert stage["all_pass"] is True
    entry = stage["per_seed"][0]
    assert entry["seed"] == AUDIT_SEED
    assert entry["n_events"] == 20
    for counter in RUN_COUNTERS:
        assert entry[counter] == 0
    assert "wall_seconds" in entry

    seed_file = _load(out_dir / f"audit_seed_{AUDIT_SEED}.json")
    assert seed_file["n_events"] == 20
    assert seed_file["audit_summary"]["n_vertices"] == 2 * 20
    assert seed_file["audit_summary"]["all_pass"] is True


def test_cli_all_mode_runs_both_stages_assembles_gate0_and_seals(tmp_path):
    """--mode all runs both stages, assembles the gate record, seals."""
    out_dir = tmp_path / "all_run"
    proc = _run_driver(
        ["--mode", "all", "--events-per-seed", "4", "--seeds", str(CORE_SEED)],
        out_dir,
    )
    # Both stages passed their own checks, but the ad-hoc single-seed
    # registration plus the smoke config cannot verify against the
    # pre-registered production record: the committed verdict for any
    # non-pre-registered run is fail, hence the non-zero exit.
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert (out_dir / r00.STAGE_CORE_NAME).is_file()
    assert (out_dir / r00.STAGE_AUDIT_NAME).is_file()

    core = _load(out_dir / r00.STAGE_CORE_NAME)
    audit = _load(out_dir / r00.STAGE_AUDIT_NAME)
    assert core["seeds"] == [CORE_SEED] and audit["seeds"] == [CORE_SEED]
    assert core["total_events"] == 4 and audit["total_events"] == 4

    gate = _load(out_dir / r00.GATE_RECORD_NAME)
    assert gate["gate"] == "gate0"
    assert gate["clause"] == r00.GATE_CLAUSE
    assert gate["verdict"] == "fail"
    assert gate["config_hash_verified"] is False  # smoke config, production pin
    assert gate["m8_parameter_hash_verified"] is True
    assert set(gate["deterministic_core"].keys()) == {
        "seeds", "events_per_seed", "total_events", "per_seed", "all_pass"
    }
    assert set(gate["conservation_audit_1m"].keys()) == {
        "seeds",
        "events_per_seed",
        "total_events",
        "conservation_failures",
        "offshell_daughters",
        "bqs_residual_nonzero",
        "ancestry_violations",
        "max_four_momentum_relative_residual",
        "max_mass_shell_relative_deviation",
    }
    # The per-seed entry is the card's gate0 schema, the driver's own
    # projection of the full per-seed record.
    seed_file = _load(out_dir / f"deterministic_seed_{CORE_SEED}.json")
    assert gate["deterministic_core"]["per_seed"][0] == r00.gate0_per_seed_fields(
        seed_file
    )
    assert gate["deterministic_core"]["per_seed"][0]["bit_identical"] is True

    # The seal pair is written after every other file and covers the
    # payload plus SHA256SUMS.txt, never itself.
    seal = _load(out_dir / r00.SEAL_NAME)
    assert (out_dir / r00.SHA256SUMS_NAME).is_file()
    sealed_names = {entry["path"] for entry in seal["files"]}
    payload_names = {p.name for p in out_dir.iterdir() if p.is_file()} - {
        r00.SHA256SUMS_NAME,
        r00.SEAL_NAME,
    }
    assert sealed_names == payload_names | {r00.SHA256SUMS_NAME}
    for entry in seal["files"]:
        assert entry["sha256"] == r00.sha256_file(out_dir / entry["path"])


def test_cli_unknown_mode_exits_nonzero(tmp_path):
    """An unknown --mode is an argparse refusal before any output."""
    out_dir = tmp_path / "unknown_mode_run"
    proc = _run_driver(["--mode", "bogus"], out_dir)
    assert proc.returncode == 2
    assert "invalid choice: 'bogus'" in proc.stderr
    for mode in ("deterministic", "audit", "all"):
        assert mode in proc.stderr
    assert not out_dir.is_dir()


def test_cli_zero_events_refused_before_generation(tmp_path):
    """--events-per-seed 0 is refused (exit 2) before any generation."""
    out_dir = tmp_path / "zero_run"
    proc = _run_driver(
        ["--mode", "deterministic", "--events-per-seed", "0",
         "--seeds", str(CORE_SEED)],
        out_dir,
    )
    assert proc.returncode == 2, proc.stderr
    assert "--events-per-seed must be a positive integer, got 0" in proc.stderr
    # The refusal fires before the --out directory is created, so no
    # stage file and no per-seed file can exist.
    assert not out_dir.is_dir()
    assert not (out_dir / r00.STAGE_CORE_NAME).is_file()
    assert not (out_dir / f"deterministic_seed_{CORE_SEED}.json").is_file()


def test_cli_negative_events_refused_before_generation(tmp_path):
    """A negative --events-per-seed is refused (exit 2) before generation."""
    out_dir = tmp_path / "negative_run"
    proc = _run_driver(
        ["--mode", "deterministic", "--events-per-seed", "-3",
         "--seeds", str(CORE_SEED)],
        out_dir,
    )
    assert proc.returncode == 2, proc.stderr
    assert "--events-per-seed must be a positive integer, got -3" in proc.stderr
    # The refusal fires before the --out directory is created, so no
    # stage file and no per-seed file can exist.
    assert not out_dir.is_dir()
    assert not (out_dir / r00.STAGE_CORE_NAME).is_file()
    assert not (out_dir / f"deterministic_seed_{CORE_SEED}.json").is_file()


def test_cli_unregistered_config_name_refused(tmp_path):
    """Only the two bundled legacy_v1 config names are accepted."""
    other = tmp_path / "other_config.json"
    shutil.copyfile(SMOKE_CONFIG, other)
    out_dir = tmp_path / "config_name_run"
    proc = subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--config", str(other),
         "--mode", "deterministic", "--events-per-seed", "2",
         "--seeds", str(CORE_SEED), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )
    assert proc.returncode == 2
    assert "refusing config" in proc.stderr
    assert "other_config.json" in proc.stderr
    assert not (out_dir / r00.STAGE_CORE_NAME).is_file()


def test_cli_seed_outside_preregistration_recorded_not_refused(tmp_path):
    """An ad-hoc seed outside the pre-registered lists runs and is recorded.

    The single-stage CLI accepts any seed: 999999 is generated, audited,
    and written to ``deterministic_seed_999999.json`` with the
    pre-registration file written unchanged beside it.  The deviation
    from the pre-registered seed lists is only adjudicated when both
    stages assemble the gate record (``registration_ok``), where it
    flips the verdict to fail (see the all-mode test).
    """
    out_dir = tmp_path / "seed_run"
    proc = _run_driver(
        ["--mode", "deterministic", "--events-per-seed", "3",
         "--seeds", str(OUT_OF_RANGE_SEED)],
        out_dir,
    )
    assert proc.returncode == 0, proc.stderr
    entry = _load(out_dir / f"deterministic_seed_{OUT_OF_RANGE_SEED}.json")
    assert entry["seed"] == OUT_OF_RANGE_SEED
    assert entry["n_events"] == 3
    assert entry["bit_identical"] is True
    assert entry["all_pass"] is True
    assert not (out_dir / f"deterministic_seed_{CORE_SEED}.json").is_file()
    assert _load(out_dir / r00.PREREGISTRATION_NAME) == r00.default_pre_registration()


# ---------------------------------------------------------------------------
# Pre-registration enforcement (in-process; no event generation)
# ---------------------------------------------------------------------------


def test_verify_config_hash_preregistration_mismatch(capsys):
    """The pristine production config verifies; a drifted hash refuses."""
    pre_registration = r00.default_pre_registration()
    verification = r00.verify_config_and_m8(PRODUCTION_CONFIG, pre_registration)
    assert verification["config_sha256"] == r00.PRODUCTION_CONFIG_SHA256
    assert verification["production_config"] is True

    drifted = r00.default_pre_registration()
    drifted["config_hash"] = "0" * 64
    with pytest.raises(SystemExit) as excinfo:
        r00.verify_config_and_m8(PRODUCTION_CONFIG, drifted)
    assert excinfo.value.code == 2
    # The refusal text goes to stderr (the exception carries the code only).
    message = capsys.readouterr().err
    assert "refusing to run" in message
    assert "pre-registration config_hash" in message


def test_verify_m8_parameter_hash_mismatch(monkeypatch, capsys):
    """The M8 parameter hash is verified from card 1's manifest, not the config.

    The committed driver reads the M8 frozen block from the fixed
    nested-root manifest path at runtime and compares its canonical
    sha256 (and its values) against the driver's verbatim copy; the
    config file carries no M8 check of its own.  Either a mismatched
    hash or a drifted parameter copy refuses the run with exit code 2
    before generation.
    """
    pre_registration = r00.default_pre_registration()

    monkeypatch.setattr(r00, "M8_PARAMETER_HASH", "0" * 64)
    with pytest.raises(SystemExit) as excinfo:
        r00.verify_config_and_m8(SMOKE_CONFIG, pre_registration)
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "refusing to run" in message
    assert "m8_frozen_pp_parameters" in message

    mutated = dict(r00.M8_FROZEN_PP_PARAMETERS)
    mutated["continuum.match_mass"] = float(mutated["continuum.match_mass"]) + 1e-3
    monkeypatch.setattr(r00, "M8_FROZEN_PP_PARAMETERS", mutated)
    with pytest.raises(SystemExit) as excinfo:
        r00.verify_config_and_m8(SMOKE_CONFIG, pre_registration)
    assert excinfo.value.code == 2
    assert "verbatim frozen copy" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Aggregation and mutation sensitivity on the hand-constructed batch
# ---------------------------------------------------------------------------


def test_hand_constructed_bqs_residual_isolated():
    """Exactly one injected per-vertex B/Q/S residual is the only flag."""
    batch = [_make_hand_event(event_id=i) for i in range(3)]
    _assert_clean(_summary_of(batch), 3)

    # Staged quantum numbers derive from the daughter PDG identity, so the
    # residual is injected by swapping one daughter's PDG id on a copy:
    # pi+ -> pi- keeps the four-vector and the mass shell intact (the
    # charged pion masses are identical) and breaks the charge closure.
    mutated = copy.deepcopy(batch[0])
    mutated.final_particles[3].pdg_id = -211
    report = audit_staged_event(mutated, event_index=mutated.event_id)
    summary = vertex_audit_summary([report])
    assert summary["bqs_residual_nonzero"] == 1
    assert report.bqs["residuals"] == [(0, 2, 0)]
    assert summary["conservation_failures"] == 0
    assert summary["offshell_daughters"] == 0
    assert summary["ancestry_violations"] == 0
    assert summary["all_pass"] is False
    # The unmutated batch members stay clean.
    clean = vertex_audit_summary(
        [audit_staged_event(event, event_index=event.event_id)
         for event in batch]
    )
    assert clean["bqs_residual_nonzero"] == 0
    assert clean["all_pass"] is True


def test_hand_constructed_offshell_daughter():
    """A ~1e-6 relative mass deviation on one daughter is off-shell."""
    batch = [_make_hand_event(event_id=i) for i in range(3)]
    _assert_clean(_summary_of(batch), 3)

    mutated = copy.deepcopy(batch[0])
    particle = mutated.final_particles[0]
    factor = 1.0 + ABOVE_TOLERANCE_SHIFT
    particle.p4 = FourVector(
        particle.p4.e * factor,
        particle.p4.px * factor,
        particle.p4.py * factor,
        particle.p4.pz * factor,
    )
    report = audit_staged_event(mutated, event_index=mutated.event_id)
    summary = vertex_audit_summary([report])
    assert summary["offshell_daughters"] == 1
    assert report.mass_shell["max_relative_deviation"] > TOLERANCE
    assert summary["bqs_residual_nonzero"] == 0
    assert summary["all_pass"] is False
    # The vertex the off-shell daughter belongs to also fails its
    # four-momentum closure (the shifted vector leaves the vertex sum).
    assert summary["conservation_failures"] >= 1


def test_mutation_tolerance_boundary():
    """1e-6 is flagged, 1e-12 is not: both sides of the 1e-10 default."""
    # The audit's relative tolerance default is the existing 1e-10
    # conservation_tolerance convention; no new tolerance here.
    default_tolerance = inspect.signature(audit_staged_event).parameters[
        "relative_tolerance"
    ].default
    assert default_tolerance == TOLERANCE

    event = _make_hand_event()
    clean = _summary_of([event])
    _assert_clean(clean, 1)

    above = copy.deepcopy(event)
    above.final_particles[0].p4 = FourVector(
        above.final_particles[0].p4.e,
        above.final_particles[0].p4.px + ABOVE_TOLERANCE_SHIFT,
        above.final_particles[0].p4.py,
        above.final_particles[0].p4.pz,
    )
    summary_above = _summary_of([above])
    assert summary_above["conservation_failures"] >= 1
    assert summary_above["all_pass"] is False

    below = copy.deepcopy(event)
    below.final_particles[0].p4 = FourVector(
        below.final_particles[0].p4.e,
        below.final_particles[0].p4.px + BELOW_TOLERANCE_SHIFT,
        below.final_particles[0].p4.py,
        below.final_particles[0].p4.pz,
    )
    summary_below = _summary_of([below])
    for counter in RUN_COUNTERS:
        assert summary_below[counter] == 0
    assert summary_below["max_four_momentum_relative_residual"] < TOLERANCE
    assert summary_below["max_mass_shell_relative_deviation"] < TOLERANCE
    assert summary_below["all_pass"] is True


# ---------------------------------------------------------------------------
# Reproducibility machinery and output schema
# ---------------------------------------------------------------------------


def test_reproducibility_machinery_in_process(monkeypatch):
    """run_deterministic_seed: bit-identical across worker counts; a
    mutation between the two runs flips the flag."""
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload = r00.config_payload(raw_config)

    real = r00.run_deterministic_seed(CORE_SEED, 4, payload)
    assert real["n_events"] == 4
    assert real["bit_identical"] is True
    assert real["audit_summaries_identical"] is True
    assert real["run_a"]["repro_run_a_sha256"] == real["run_b"]["repro_run_b_sha256"]
    assert real["run_a"]["p4_set_sha256"] == real["run_b"]["p4_set_sha256"]
    assert real["worker_counts"] == {
        "run_a": r00.CORE_WORKERS_RUN_A,
        "run_b": r00.CORE_WORKERS_RUN_B,
    }
    for counter in RUN_COUNTERS:
        assert real[counter] == 0

    # Inject a different generated event into the second (4-worker) run
    # only, through the driver's own _map_chunks seam: the stored
    # ``event_hash`` is a snapshot of the generated payload (it is not
    # recomputed from the mutated four-vectors), so a different event
    # means a different hash and a different p4 together.  The
    # comparison machinery must flip the bit-identical flag while the
    # first run's digest stays stable.
    original_map_chunks = r00._map_chunks

    def flaky_map_chunks(tasks, workers, label):
        results = list(original_map_chunks(tasks, workers, label))
        if workers == r00.CORE_WORKERS_RUN_B:
            for records, _summary in results:
                for record in records:
                    record["p4"][0][1] += ABOVE_TOLERANCE_SHIFT
                    record["event_hash"] += "flaky"
        for result in results:
            yield result

    monkeypatch.setattr(r00, "_map_chunks", flaky_map_chunks)
    flaky = r00.run_deterministic_seed(CORE_SEED, 4, payload)
    assert flaky["bit_identical"] is False
    assert flaky["all_pass"] is False
    assert flaky["run_a"]["repro_run_a_sha256"] != flaky["run_b"][
        "repro_run_b_sha256"
    ]
    assert flaky["run_a"]["p4_set_sha256"] != flaky["run_b"]["p4_set_sha256"]
    # The unmutated run reproduces the first invocation bit for bit:
    # the seeding rule is a pure function of (seed, event_id).
    assert flaky["run_a"]["repro_run_a_sha256"] == real["run_a"][
        "repro_run_a_sha256"
    ]
    assert flaky["run_a"]["p4_set_sha256"] == real["run_a"]["p4_set_sha256"]
    # The audit summaries were not touched by the record mutation.
    assert flaky["audit_summaries_identical"] is True


def test_deterministic_output_payload_across_runs(tmp_path):
    """Two runs on the same tiny input agree on every physics field.

    Finding (recorded, not fixed): the written files are not
    byte-identical across runs because they embed wall-clock
    (``wall_seconds``) and UTC (``started_utc``/``finished_utc``)
    timing fields; the physics payload is run-to-run stable.
    """
    args = ["--mode", "deterministic", "--events-per-seed", "4",
            "--seeds", str(CORE_SEED)]
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    proc_a = _run_driver(args, out_a)
    proc_b = _run_driver(args, out_b)
    assert proc_a.returncode == 0, proc_a.stderr
    assert proc_b.returncode == 0, proc_b.stderr

    seed_a = _load(out_a / f"deterministic_seed_{CORE_SEED}.json")
    seed_b = _load(out_b / f"deterministic_seed_{CORE_SEED}.json")
    assert seed_a["run_a"]["repro_run_a_sha256"] == seed_b["run_a"][
        "repro_run_a_sha256"
    ]
    assert seed_a["run_b"]["repro_run_b_sha256"] == seed_b["run_b"][
        "repro_run_b_sha256"
    ]
    assert seed_a["run_a"]["p4_set_sha256"] == seed_b["run_a"]["p4_set_sha256"]
    assert seed_a["run_a"]["audit_summary"] == seed_b["run_a"]["audit_summary"]
    assert r00.gate0_per_seed_fields(seed_a) == r00.gate0_per_seed_fields(seed_b)
    assert (out_a / r00.PREREGISTRATION_NAME).read_bytes() == (
        out_b / r00.PREREGISTRATION_NAME
    ).read_bytes()

    # The non-determinism is confined to the timing fields.
    stage_a = _load(out_a / r00.STAGE_CORE_NAME)
    stage_b = _load(out_b / r00.STAGE_CORE_NAME)
    for field in ("wall_seconds", "started_utc", "finished_utc"):
        assert field in stage_a and field in stage_b
    for field in ("seed", "n_events", "repro_run_a_sha256",
                  "repro_run_b_sha256", "bit_identical"):
        assert stage_a["per_seed"][0][field] == stage_b["per_seed"][0][field]
