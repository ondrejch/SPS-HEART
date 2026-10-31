"""The p+A event frame and the nucleon-nucleon CM x_F (card TASK-20261005-pa-frame-nn-cm-xf).

``na49_mixing.event_builder_pA`` builds every p+A event in the
laboratory frame (beam proton ``(E_beam, 0, 0, p_beam)``, target nucleons
at rest) and records the frame in the event metadata (``frame = "lab"``
with ``nn_cm_beta``, ``nn_cm_gamma``, ``sqrt_s_nn``). The NA49 p+A
observables use x_F = 2 p_z^CM / sqrt(s_NN) in the nucleon-nucleon CM
frame, so every consumer reads the tracks through ``nn_cm_p4`` /
``nn_cm_xf`` / ``event_in_nn_cm``. Pins:

- the boost parameters: beta = p_beam / (E_beam + m_p), gamma = (E_beam
  + m_p) / sqrt(s_NN), consistent with each other and with the p+p
  builder's sqrt(s);
- a particle at rest in the NN CM gives x_F = 0;
- the boosted final-state p_z sum of a single-wounded-nucleon (nu = 1)
  event with a proton struck target is zero to 1e-9 (the spectators are
  excluded from the sum; the test also accounts for them explicitly:
  each spectator at rest in the lab carries p_z^CM = -beta gamma m);
- a p+p event read through the p+A x_F path gives the p+p x_F bit for
  bit (the CM event is returned unchanged);
- the x_F of generated p+C and p+Pb events lies in the NN kinematic range
  (all pions in [-1, 1]; every track of a nu = 1 proton-struck event in
  [-1, 1]); the raw lab-frame ratio 2 p_z^lab / sqrt(s_NN) exceeds 1 for
  a sizable fraction of the tracks (the withdrawn convention);
- the (b, K) redraw loop of ``PAEventBuilder.generate`` never redraws at
  the production beam momentum: 200 p+Pb208 and 200 p+C12 events at
  158 GeV/c at the frozen M8 set are all accepted at the first attempt,
  so the accepted (b, K) stream is the Glauber stream and the loop
  cannot bias the production wound distribution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.event_builder_pA import (
    FRAME_LAB,
    FRAME_NN_CM,
    PAEventBuilder,
    PAExcitationConfig,
    event_frame,
    event_in_nn_cm,
    lab_to_nn_cm,
    nn_cm_boost,
    nn_cm_p4,
    nn_cm_xf,
)
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.exchange_model import ReggeExcitationParameters
from na49_mixing.glauber import NucleusSpec
from na49_mixing.pdg_database import get_mass
from na49_mixing.reaction import Reaction

BEAM = 158.0
SQRT_S_NN = 17.270205201372406  # Reaction(158.0, 2212, 2212).sqrt_s()
M_P = get_mass(2212)
M_N = get_mass(2112)
M_PI = get_mass(211)
SEED = 20261006

C12_FIELDS = {"name": "C12", "A": 12, "Z": 6, "radius_fm": 1.88,
              "diffuseness_fm": 0.546, "sigma_nn_inel_mb": 31.7,
              "profile": "harmonic_oscillator", "quad_alpha": 0.0}
PB208_FIELDS = {"name": "Pb208", "A": 208, "Z": 82, "radius_fm": 6.62,
                "diffuseness_fm": 0.546, "sigma_nn_inel_mb": 31.7,
                "profile": "woods_saxon", "quad_alpha": 0.0}
POINT_PROTON_FIELDS = {"name": "p", "A": 1, "Z": 1, "radius_fm": 1.0,
                       "diffuseness_fm": 0.546, "sigma_nn_inel_mb": 31.7,
                       "profile": "woods_saxon", "quad_alpha": 0.0}


def _light_regge() -> ReggeExcitationParameters:
    return ReggeExcitationParameters(
        alpha_p0=1.0808, alpha_p_prime=0.25, alpha_exc0=0.5,
        transfer_slope=4.5, edge_power=2.0, s0=1.0, grid_size=96,
        effective_transfer_slope=7.3489827740435505)


def _pa_events(fields: dict, n_events: int, seed: int = SEED) -> list:
    builder = PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=BEAM, production_mode="regge",
            regge=_light_regge(), continuum=ContinuumParameters(),
            nucleus=NucleusSpec(**fields), glauber_dr_fm=0.05),
        np.random.default_rng(seed))
    return [builder.generate(i) for i in range(n_events)]


@pytest.fixture(scope="module")
def c12_events() -> list:
    return _pa_events(C12_FIELDS, 40)


@pytest.fixture(scope="module")
def pb_events() -> list:
    return _pa_events(PB208_FIELDS, 20, SEED + 1)


def test_nn_cm_boost_parameters_are_consistent():
    beta, gamma, sqrt_s = nn_cm_boost(BEAM)
    e_beam = math.sqrt(M_P * M_P + BEAM * BEAM)
    assert sqrt_s == Reaction(BEAM, 2212, 2212).sqrt_s() == SQRT_S_NN
    assert beta == BEAM / (e_beam + M_P)
    assert gamma == (e_beam + M_P) / sqrt_s
    assert abs(gamma * gamma * (1.0 - beta * beta) - 1.0) < 1e-12
    reaction = Reaction(BEAM, 2212, 2212)
    assert abs(beta - reaction.betha()) < 1e-15
    assert abs(gamma - reaction.gamma()) < 1e-12
    # The pinned 158 GeV/c values (beta gamma of the NN CM).
    assert abs(beta - 0.99407920) < 1e-8
    assert abs(gamma - 9.2031945) < 1e-7


def test_builder_records_the_lab_frame(c12_events):
    beta, gamma, sqrt_s = nn_cm_boost(BEAM)
    for event in c12_events:
        assert event.metadata["frame"] == FRAME_LAB
        assert event.metadata["nn_cm_beta"] == beta
        assert event.metadata["nn_cm_gamma"] == gamma
        assert event.metadata["sqrt_s_nn"] == sqrt_s
        assert event_frame(event) == FRAME_LAB
        # The lab frame of construction: beam along +z, targets at rest.
        assert event.incoming[0].pz == BEAM
        assert all(p4.pz == 0.0 for p4 in event.incoming[1:])


@pytest.mark.parametrize("mass", [M_PI, M_P, 1.232])
@pytest.mark.parametrize("pt", [0.0, 0.4])
def test_particle_at_rest_in_nn_cm_gives_zero_xf(c12_events, mass, pt):
    beta, gamma, _sqrt_s = nn_cm_boost(BEAM)
    e_cm = math.sqrt(mass * mass + pt * pt)
    lab = FourVector(gamma * e_cm, pt, 0.0, gamma * beta * e_cm)
    event = c12_events[0]
    assert abs(nn_cm_xf(event, lab, SQRT_S_NN)) < 1e-12
    cm = nn_cm_p4(event, lab)
    assert abs(cm.e - e_cm) < 1e-12
    assert cm.px == pt and cm.py == 0.0
    # The raw lab ratio of the same track is far from zero.
    assert 2.0 * lab.pz / SQRT_S_NN > 0.5 * mass


def test_beam_and_target_nucleon_map_to_the_nn_kinematic_limits(c12_events):
    event = c12_events[0]
    limit = math.sqrt(1.0 - 4.0 * M_P * M_P / (SQRT_S_NN * SQRT_S_NN))
    beam = event.incoming[0]
    target_at_rest = FourVector(M_P, 0.0, 0.0, 0.0)
    assert abs(nn_cm_xf(event, beam, SQRT_S_NN) - limit) < 1e-12
    assert abs(nn_cm_xf(event, target_at_rest, SQRT_S_NN) + limit) < 1e-12
    # The NN pair is at rest in the frame.
    pair = lab_to_nn_cm(beam + target_at_rest, *nn_cm_boost(BEAM)[:2])
    assert abs(pair.pz) < 1e-12
    assert abs(pair.e - SQRT_S_NN) < 1e-12


def _boosted_final_pz_sum(event) -> float:
    return math.fsum(nn_cm_p4(event, p.p4).pz for p in event.final_particles)


def test_nu1_proton_struck_event_boosted_pz_sum_is_zero():
    # The A = 1 point target: every event is a nu = 1 proton-struck event
    # without spectators.
    events = _pa_events(POINT_PROTON_FIELDS, 25)
    for event in events:
        assert event.metadata["target_wounds"] == 1
        assert event.metadata["struck_target_pdgs"] == [2212]
        assert not event.spectators
        assert abs(_boosted_final_pz_sum(event)) < 1e-9


def test_nu1_event_in_a_nucleus_with_spectators_accounted(c12_events):
    beta, gamma, _sqrt_s = nn_cm_boost(BEAM)
    checked = {2212: 0, 2112: 0}
    for event in c12_events:
        if event.metadata["target_wounds"] != 1:
            continue
        struck = event.metadata["struck_target_pdgs"][0]
        # Spectators excluded: the final particles carry the beam plus the
        # struck nucleon, which is at rest in the NN CM for a proton and
        # moves with p_z = -beta gamma (m_n - m_p) for a neutron.
        expected = -beta * gamma * (get_mass(struck) - M_P)
        assert abs(_boosted_final_pz_sum(event) - expected) < 1e-9
        # Spectators accounted: each one at rest in the lab carries
        # p_z^CM = -beta gamma m, so the full final-state sum is the boosted
        # total incoming p_z.
        total = math.fsum(
            [nn_cm_p4(event, p.p4).pz for p in event.final_particles]
            + [nn_cm_p4(event, s.p4).pz for s in event.spectators])
        spectator_mass = math.fsum(s.p4.e for s in event.spectators)
        assert abs(total - (expected - beta * gamma * spectator_mass)) < 1e-9
        checked[struck] += 1
    assert checked[2212] >= 1


def test_event_in_nn_cm_boosts_a_copy_and_conserves(c12_events):
    for event in c12_events[:10]:
        before = event.as_dict()["event_hash"]
        cm = event_in_nn_cm(event)
        assert cm is not event
        assert cm.metadata["frame"] == FRAME_NN_CM
        assert event.metadata["frame"] == FRAME_LAB
        assert event.as_dict()["event_hash"] == before  # input untouched
        cm.validate()
        for lab_p, cm_p in zip(event.final_particles, cm.final_particles):
            boosted = nn_cm_p4(event, lab_p.p4)
            assert cm_p.p4 == boosted
            assert cm_p.pdg_id == lab_p.pdg_id
            assert cm_p.source_id == lab_p.source_id
        # A CM event passes through unchanged.
        assert event_in_nn_cm(cm) is cm
        assert nn_cm_p4(cm, cm.final_particles[0].p4) is cm.final_particles[0].p4


def test_pp_event_through_the_pa_path_is_bit_for_bit():
    builder = PPEventBuilder(
        PPExcitationConfig(beam_momentum=BEAM, production_mode="regge",
                           regge=_light_regge(),
                           continuum=ContinuumParameters()),
        np.random.default_rng(SEED))
    for event_id in range(20):
        event = builder.generate(event_id)
        assert "frame" not in event.metadata
        assert event_frame(event) == FRAME_NN_CM
        assert event_in_nn_cm(event) is event
        for particle in event.final_particles:
            assert nn_cm_p4(event, particle.p4) is particle.p4
            raw = 2.0 * float(particle.p4.pz) / float(SQRT_S_NN)
            assert nn_cm_xf(event, particle.p4, SQRT_S_NN) == raw


def test_pa_stage_event_without_frame_record_is_refused(c12_events):
    event = c12_events[0]
    stripped = type(event)(
        event_id=event.event_id, stage=event.stage,
        incoming=event.incoming,
        incoming_quantum_numbers=event.incoming_quantum_numbers,
        sources=event.sources, spectators=event.spectators,
        final_particles=event.final_particles,
        metadata={k: v for k, v in event.metadata.items() if k != "frame"})
    with pytest.raises(ValueError, match="frame record"):
        nn_cm_xf(stripped, stripped.final_particles[0].p4, SQRT_S_NN)


@pytest.mark.parametrize("system", ["C12", "Pb208"])
def test_pa_xf_lies_in_the_nn_kinematic_range(c12_events, pb_events, system):
    events = c12_events if system == "C12" else pb_events
    pion_xf = []
    raw_xf = []
    for event in events:
        nu1_proton = (event.metadata["target_wounds"] == 1
                      and event.metadata["struck_target_pdgs"] == [2212])
        for particle in event.final_particles:
            xf = nn_cm_xf(event, particle.p4, SQRT_S_NN)
            raw_xf.append(2.0 * particle.p4.pz / SQRT_S_NN)
            # The projectile side is bounded by the NN limit.
            assert xf <= 1.0
            if nu1_proton:
                # One NN collision: every track lies inside the NN
                # kinematic limit |p| <= sqrt(s)/2.
                assert -1.0 <= xf <= 1.0
            if particle.pdg_id in (-211, 211):
                pion_xf.append(xf)
    pion_xf = np.asarray(pion_xf)
    assert pion_xf.size > 100
    assert pion_xf.min() >= -1.0 and pion_xf.max() <= 1.0
    # Both hemispheres are populated in the NN CM.
    assert (pion_xf < -0.05).any() and (pion_xf > 0.05).any()
    # The withdrawn lab-frame ratio leaves the physical range.
    raw = np.asarray(raw_xf)
    assert (raw > 1.0).mean() > 0.05


def _m8_builder(fields: dict, seed: int) -> PAEventBuilder:
    """The staged p+A builder at the frozen M8 set on the bundled config
    (the M12/M13 production configuration)."""
    from helpers.benchmarking import staged_heldout_ratios_pa as h

    config_path = Path(h.MIXING2006_DIR) / "config" / "staged_excitation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    simple, regge, continuum, _applied = h.build_parameter_set(config)
    pa_cfg = dict(config.get("pA") or h.PA_RUNNER_SETTINGS)
    return PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=float(config.get("beam_momentum", BEAM)),
            production_mode=h.PRODUCTION_MODE, simple=simple, regge=regge,
            continuum=continuum, nucleus=NucleusSpec(**fields),
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100))),
        np.random.default_rng(int(seed)))


@pytest.mark.parametrize("fields", [PB208_FIELDS, C12_FIELDS],
                         ids=["Pb208", "C12"])
def test_redraw_loop_never_redraws_at_158_gev(fields):
    # The first 200 events of the M12 workstream-B replica-0 stream.
    builder = _m8_builder(fields, 20343832)
    assert builder.config.beam_momentum == BEAM
    n_events = 200
    for event_id in range(n_events):
        event = builder.generate(event_id)
        assert event.metadata["generation_attempt"] == 1
        assert event.metadata["generation_last_failure_reason"] is None
    assert builder.generation_attempts == n_events
    assert builder.generation_accepted == n_events
    assert builder.generation_rejected == 0
    assert builder.failure_reason_counts == {}
    assert builder.accepted_generation_attempts == [1] * n_events
    # Every Glauber draw was accepted, so the accepted (b, K) stream is the
    # unconditioned Glauber stream.
    assert len(builder.glauber_draws) == n_events
    assert all(accepted for _b, _k, accepted in builder.glauber_draws)


def test_r02_hemisphere_sums_read_nn_cm_tracks(c12_events):
    """The R02 hemisphere split classifies p+A tracks by their NN CM x_F
    and weights them with the CM E/pT: a lab-frame event and its
    ``event_in_nn_cm`` copy give identical sums, and the target hemisphere
    (x_F < 0) is populated."""
    from helpers.benchmarking import staged_r02_reference as r02

    species = {"pi_minus": (-211,), "pi_plus": (211,), "proton": (2212,)}
    lab_sums, lab_counts = r02._hemisphere_track_sums(
        c12_events, species, SQRT_S_NN)
    cm_events = [event_in_nn_cm(event) for event in c12_events]
    cm_sums, cm_counts = r02._hemisphere_track_sums(
        cm_events, species, SQRT_S_NN)
    for name in species:
        assert np.array_equal(lab_counts[name], cm_counts[name])
        assert np.array_equal(lab_sums[name], cm_sums[name])
    assert cm_counts["pi_minus"][2] > 0
    assert cm_counts["proton"][2] > 0
