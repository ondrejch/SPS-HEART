"""Release-blocker pins for the 2026-09-11 source review (review 9).

TASK-2026-09-11-code-review9-fixes. Hermetic: no SPS tables, no network.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.continuum_channels import ContinuumParameters, SourceDecayer
from na49_mixing.excitation_kinematics import (
    FourVector,
    two_body_split_safe,
    two_body_threshold_tolerance,
)
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.source_event import ExcitationEvent, FinalParticle, SourceRecord
from na49_mixing.staged_evaluation import (
    _BINNED_DATA_PREREQUISITE,
    preflight_binned_tables,
    run_staged_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "config" / "staged_excitation_smoke.json"
PROD_CONFIG = REPO_ROOT / "config" / "staged_excitation.json"


def test_two_body_split_safe_raises_on_the_review_example():
    with pytest.raises(ValueError, match="two-body split below threshold"):
        two_body_split_safe(1.0, 0.8, 0.4, np.random.default_rng(0))


def test_two_body_threshold_tolerance_covers_one_ulp():
    parent, m1, m2 = 1.0, 0.1, 0.9000000000000001
    excess = (m1 + m2) - parent
    assert excess > 0.0
    assert excess <= two_body_threshold_tolerance(parent, m1, m2)


def _on_shell_event(mass: float | None = None) -> ExcitationEvent:
    pdg_mass = get_mass(211) if mass is None else float(mass)
    incoming = FourVector(pdg_mass, 0.0, 0.0, 0.0)
    particle = FinalParticle(
        particle_id=0,
        pdg_id=211,
        p4=FourVector(pdg_mass, 0.0, 0.0, 0.0),
        source_id=0,
        parent_id=None,
        channel_family="test",
    )
    source = SourceRecord(
        source_id=0,
        role="projectile",
        p4=incoming,
        quantum_numbers=QuantumNumbers(0, 1, 0),
    )
    return ExcitationEvent(
        event_id=0,
        stage="test",
        incoming=[incoming],
        incoming_quantum_numbers=QuantumNumbers(0, 1, 0),
        sources=[source],
        final_particles=[particle],
    )


def test_validate_accepts_pdg_on_shell_final_particles():
    _on_shell_event().validate()


def test_validate_rejects_an_off_shell_final_particle():
    event = _on_shell_event(mass=0.2)
    with pytest.raises(ValueError, match="off shell"):
        event.validate()


def test_extra_pion_clamp_drops_a_charged_pair_that_exceeds_the_budget():
    decayer = SourceDecayer(ContinuumParameters(), np.random.default_rng(1))
    mpi0 = get_mass(111)
    mpi_charged = get_mass(211)
    remaining = 2.0 * mpi0 + 0.001
    assert 2.0 * mpi_charged > remaining
    clamped = decayer._clamp_extra_to_mass_budget([211, -211], remaining)
    assert clamped == []
    two_pi0 = decayer._clamp_extra_to_mass_budget([111, 111], remaining)
    assert two_pi0 == [111, 111]


def test_extra_pion_clamp_preserves_charge_when_dropping_pairs():
    decayer = SourceDecayer(ContinuumParameters(), np.random.default_rng(2))
    remaining = get_mass(111) + 0.001
    clamped = decayer._clamp_extra_to_mass_budget(
        [211, -211, 111, 111], remaining
    )
    assert clamped == [111]
    assert sum(1 for pid in clamped if pid == 211) == sum(
        1 for pid in clamped if pid == -211
    )


def test_near_threshold_charged_composition_is_mass_safe():
    """A source that can fit two pi0 but not a charged pair must not raise."""
    mpi0 = get_mass(111)
    proton = get_mass(2212)
    source_mass = proton + mpi0 + 2.0 * mpi0 + 0.002
    source = SourceRecord(
        source_id=0,
        role="projectile",
        p4=FourVector(source_mass, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )
    parameters = ContinuumParameters(
        match_mass=10.0,
        pion_multiplicity_slope=100.0,
        strange_lambda_strength=0.0,
        strange_sigma_strength=0.0,
        strange_pair_strength=0.0,
        pair_strength=0.0,
        max_pions=2,
    )
    decayer = SourceDecayer(parameters, np.random.default_rng(3))
    extra = decayer._clamp_extra_to_mass_budget(
        [211, -211], source.mass - proton - mpi0
    )
    daughters = decayer.decay(source)
    ids = [p.pdg_id for p in daughters]
    assert sum(get_mass(pid) for pid in extra) <= (
        source.mass - proton - mpi0 + 1e-12
    )
    assert sum(get_mass(p.pdg_id) for p in daughters) <= source.mass + 1e-12
    assert 2212 in ids or 2112 in ids


def test_preflight_names_missing_tables_and_the_prerequisite(tmp_path: Path):
    binned = {
        "tables": [
            {"species": "pi_minus", "path": "spshadrons.web.cern.ch/data/xsect_piminus.txt"},
            {"species": "pi_plus", "path": "spshadrons.web.cern.ch/data/xsect_piplus.txt"},
        ]
    }
    with pytest.raises(FileNotFoundError, match="binned-fit data preflight failed") as excinfo:
        preflight_binned_tables(binned, path_root=tmp_path)
    message = str(excinfo.value)
    assert "xsect_piminus.txt" in message
    assert "xsect_piplus.txt" in message
    assert "staged_excitation_smoke.json" in message
    assert _BINNED_DATA_PREREQUISITE.split("Provide", 1)[0].strip() in message or (
        "source-only archives" in message
    )


def test_preflight_accepts_existing_tables(tmp_path: Path):
    table = tmp_path / "xsect_piminus.txt"
    table.write_text("# xf pt val err\n0.05 0.2 1.0 0.1\n")
    binned = {"tables": [{"species": "pi_minus", "path": "xsect_piminus.txt"}]}
    resolved = preflight_binned_tables(binned, path_root=tmp_path)
    assert resolved == [table]


def test_smoke_config_disables_the_binned_fit():
    smoke = json.loads(SMOKE_CONFIG.read_text())
    prod = json.loads(PROD_CONFIG.read_text())
    assert smoke["fit"]["enabled"] is False
    assert prod["fit"]["enabled"] is True
    assert smoke["fit"]["objective"] == "binned"
    assert smoke["through"] == prod["through"]


def test_smoke_config_runs_without_sps_tables():
    config = json.loads(SMOKE_CONFIG.read_text())
    report, samples = run_staged_evaluation(config)
    assert report["through"] == "baryon_nucleus"
    assert report["regge_fit"]["fit"]["performed"] is False
    assert "simple_excitation" in samples
    assert "regge_fit" in samples
    assert "baryon_nucleus" in samples
    for event in samples["regge_fit"]:
        event.validate()


def test_staged_evaluation_does_not_mutate_sys_path():
    import na49_mixing.staged_evaluation as staged

    source = inspect.getsource(staged)
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source
