import json

import numpy as np

from na49_mixing.continuum_channels import ContinuumParameters, SourceDecayer
from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.glauber import NucleusSpec
from na49_mixing.quantum_numbers import QuantumNumbers, quantum_numbers
from na49_mixing.source_event import SourceRecord
from na49_mixing.staged_evaluation import run_staged_evaluation


def test_quantum_number_table_covers_pair_and_associated_strangeness():
    assert quantum_numbers(2212) == QuantumNumbers(1, 1, 0)
    assert quantum_numbers(-2212) == QuantumNumbers(-1, -1, 0)
    assert quantum_numbers(321) == QuantumNumbers(0, 1, 1)
    assert quantum_numbers(3122) == QuantumNumbers(1, 0, -1)


def test_pp_simple_and_regge_events_close_and_sources_are_baryonic():
    for mode in ("simple", "regge"):
        builder = PPEventBuilder(
            PPExcitationConfig(production_mode=mode),
            np.random.default_rng(1234),
        )
        for event_id in range(20):
            event = builder.generate(event_id)
            event.validate()
            assert len(event.sources) == 2
            assert all(
                source.quantum_numbers == QuantumNumbers(1, 1, 0)
                for source in event.sources
            )
            for particle in event.final_particles:
                if particle.pdg_id < 0 and abs(particle.pdg_id) in {2212, 2112}:
                    assert particle.channel_family == "baryon_pair"


def test_forced_antiproton_channel_is_legal_and_above_threshold():
    parameters = ContinuumParameters(
        match_mass=2.0,
        strange_lambda_strength=0.0,
        strange_sigma_strength=0.0,
        strange_pair_strength=0.0,
        pair_strength=1e12,
        pair_threshold_power=0.0,
        pion_multiplicity_slope=0.0,
    )
    source = SourceRecord(
        source_id=7,
        role="projectile",
        p4=FourVector(3.2, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )
    daughters = SourceDecayer(parameters, np.random.default_rng(8)).decay(source)
    assert any(p.pdg_id == -2212 for p in daughters)
    assert all(p.channel_family == "baryon_pair" for p in daughters)
    assert sum((p.quantum_numbers for p in daughters), QuantumNumbers()) == source.quantum_numbers


def test_pa_has_one_projectile_and_one_source_per_wounded_target():
    nucleus = NucleusSpec(
        name="C12",
        A=12,
        Z=6,
        profile="harmonic_oscillator",
        radius_fm=1.7,
        diffuseness_fm=0.5,
    )
    builder = PAEventBuilder(
        PAExcitationConfig(
            nucleus=nucleus,
            glauber_dr_fm=0.1,
            max_event_attempts=100,
        ),
        np.random.default_rng(44),
    )
    event = builder.generate(3)
    event.validate()
    wounds = event.metadata["target_wounds"]
    projectile = [s for s in event.sources if s.role == "projectile"]
    targets = [s for s in event.sources if s.role == "target"]
    assert len(projectile) == 1
    assert len(targets) == wounds
    assert len(projectile[0].metadata["history"]) == wounds
    assert event.outgoing_quantum_numbers() == QuantumNumbers(13, 7, 0)


def test_staged_runner_can_stop_after_each_gate(tmp_path):
    base = {
        "seed": 91,
        "events_per_stage": 4,
        "beam_momentum": 158.0,
        "fit": {"enabled": False},
        "nucleus": {
            "name": "C12",
            "A": 12,
            "Z": 6,
            "profile": "harmonic_oscillator",
            "radius_fm": 1.7,
            "diffuseness_fm": 0.5,
            "quad_alpha": 0.0,
            "sigma_nn_inel_mb": 31.7,
        },
        "pA": {"glauber_dr_fm": 0.1, "max_event_attempts": 100},
    }
    for through, expected in (
        ("simple_excitation", ["simple_excitation"]),
        ("regge_fit", ["simple_excitation", "regge_fit"]),
        (
            "baryon_nucleus",
            ["simple_excitation", "regge_fit", "baryon_nucleus"],
        ),
    ):
        report, samples = run_staged_evaluation({**base, "through": through})
        assert report["stage_order"] == expected
        assert list(samples) == expected
        assert all(
            report[stage]["summary"]["max_conservation_residual"] < 1e-10
            for stage in expected
        )
