"""Hermetic self-check suite for the R00 deterministic-correctness driver.

Three checks (card ``TASK-20260911-r1-run-r00-deterministic-correctness``):

- ``test_r00_harness_small_n``: the driver's generation-plus-audit path
  on one seed with 10 events from the archive-runnable smoke config
  (``config/staged_excitation_smoke.json``, ``fit.enabled: false``);
  the aggregated ``vertex_audit_summary`` carries ``all_pass: true``
  and zero conservation, off-shell, B/Q/S, and ancestry counters.
- ``test_r00_harness_mutation_detected``: shifting one final-particle
  four-momentum component of a generated event by 1e-6 must flag the
  mutation (``conservation_failures >= 1`` or ``offshell_daughters >= 1``).
- ``test_r00_config_hash_verified``: a mutated config copy fails the
  driver with a non-zero exit code before any event is generated; an
  unmutated copy of the same config passes the same invocation.

Hermetic: the 10-event builder call is the only generation inside the
first two checks and the two-event control run inside the third; the
50,000-event core and the 1,000,000-event audit are the R00 run itself,
not tests.  Standard library, the driver module, and ``na49_mixing``
only; the smoke config needs no external tables.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

from helpers.benchmarking import staged_r00_correctness as r00
from na49_mixing.event_builder_pp import PPEventBuilder
from na49_mixing.event_conservation import audit_staged_event, vertex_audit_summary
from na49_mixing.excitation_kinematics import FourVector

NESTED_ROOT = Path(r00.NESTED_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r00_correctness.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"

HARNESS_SEED = 20311000
HARNESS_N_EVENTS = 10
MUTATION_SHIFT = 1e-6


def _smoke_payload() -> dict:
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    assert raw_config["fit"]["enabled"] is False
    return raw_config


def _generate_harness_batch(raw_config: dict, n_events: int) -> tuple[list, dict]:
    """The driver's builder-plus-audit path on one seed, n_events events."""
    pp_config, _ = r00.build_pp_config(raw_config)
    events = []
    for event_id in range(n_events):
        builder = PPEventBuilder(pp_config, r00._event_rng(HARNESS_SEED, event_id))
        events.append(builder.generate(event_id, stage=r00.CORE_STAGE_LABEL))
    reports = [
        audit_staged_event(
            event,
            relative_tolerance=pp_config.conservation_tolerance,
            event_index=event.event_id,
        )
        for event in events
    ]
    return events, vertex_audit_summary(reports)


def _assert_clean_summary(summary: dict, n_events: int) -> None:
    assert summary["n_events"] == n_events
    assert summary["n_vertices"] == 2 * n_events  # two decaying sources per event
    assert summary["conservation_failures"] == 0
    assert summary["offshell_daughters"] == 0
    assert summary["bqs_residual_nonzero"] == 0
    assert summary["ancestry_violations"] == 0
    assert summary["all_pass"] is True


def test_r00_harness_small_n():
    """10 staged events from the smoke config audit clean end to end."""
    raw_config = _smoke_payload()
    events, summary = _generate_harness_batch(raw_config, HARNESS_N_EVENTS)
    assert len(events) == HARNESS_N_EVENTS
    _assert_clean_summary(summary, HARNESS_N_EVENTS)

    # The driver's own chunk path (the code the run itself executes) must
    # return the identical aggregate for the same batch.
    payload = r00.config_payload(raw_config)
    records, chunk_summary = r00._generate_chunk(
        (
            HARNESS_SEED,
            list(range(HARNESS_N_EVENTS)),
            payload,
            r00.CORE_STAGE_LABEL,
            True,
        )
    )
    assert len(records) == HARNESS_N_EVENTS
    _assert_clean_summary(chunk_summary, HARNESS_N_EVENTS)
    assert chunk_summary == summary


def test_r00_harness_mutation_detected():
    """A 1e-6 four-momentum shift on one final particle is flagged."""
    raw_config = _smoke_payload()
    pp_config, _ = r00.build_pp_config(raw_config)
    events, _ = _generate_harness_batch(raw_config, 1)
    event = events[0]
    clean = vertex_audit_summary(
        [
            audit_staged_event(
                event,
                relative_tolerance=pp_config.conservation_tolerance,
                event_index=event.event_id,
            )
        ]
    )
    assert clean["conservation_failures"] == 0
    assert clean["offshell_daughters"] == 0

    mutated = copy.deepcopy(event)
    particle = mutated.final_particles[0]
    shifted = FourVector(
        particle.p4.e,
        particle.p4.px + MUTATION_SHIFT,
        particle.p4.py,
        particle.p4.pz,
    )
    particle.p4 = shifted
    report = audit_staged_event(mutated, event_index=mutated.event_id)
    summary = vertex_audit_summary([report])
    assert (
        summary["conservation_failures"] >= 1 or summary["offshell_daughters"] >= 1
    )
    assert summary["all_pass"] is False


def test_r00_config_hash_verified(tmp_path):
    """A mutated config copy fails the driver; the pristine copy passes."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / r00.SMOKE_CONFIG_NAME
    # The pin is over the exact file bytes, so the control copy must be a
    # byte-identical copy of the bundled smoke config (not a re-serialized
    # JSON dump).
    shutil.copyfile(SMOKE_CONFIG, config_path)
    assert json.loads(config_path.read_text(encoding="utf-8"))["fit"][
        "enabled"
    ] is False

    pristine_out = tmp_path / "run_pristine"
    pristine = subprocess.run(
        [
            sys.executable,
            str(DRIVER_PATH),
            "--config",
            str(config_path),
            "--mode",
            "deterministic",
            "--events-per-seed",
            "2",
            "--seeds",
            str(HARNESS_SEED),
            "--out",
            str(pristine_out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(NESTED_ROOT),
    )
    assert pristine.returncode == 0, pristine.stderr
    written = json.loads(
        (pristine_out / r00.PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )
    assert written == r00.default_pre_registration()

    mutated_raw = json.loads(config_path.read_text(encoding="utf-8"))
    mutated_raw["continuum"]["match_mass"] = float(
        mutated_raw["continuum"]["match_mass"]
    ) + MUTATION_SHIFT
    config_path.write_text(
        json.dumps(mutated_raw, indent=2), encoding="utf-8"
    )
    mutated_out = tmp_path / "run_mutated"
    mutated = subprocess.run(
        [
            sys.executable,
            str(DRIVER_PATH),
            "--config",
            str(config_path),
            "--mode",
            "deterministic",
            "--events-per-seed",
            "2",
            "--seeds",
            str(HARNESS_SEED),
            "--out",
            str(mutated_out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(NESTED_ROOT),
    )
    assert mutated.returncode != 0, mutated.stdout
    assert "config sha256" in mutated.stderr
    # The refusal fires before generation: no deterministic stage file exists.
    assert not (mutated_out / r00.STAGE_CORE_NAME).is_file()
