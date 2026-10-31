"""Tests for the kernel configuration (TASK-20261004-refactor-run-engine-
kernel-config, P1): the profiles reproduce the committed kernel and the
kernel of the sealed R74 run, the options validate, and the committed
configuration sets none of them."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import LEGACY, PROFILES, R74, KernelConfig, apply_kernel_config, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
R74_DRIVER = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r74_central_component_barrier_step.py"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


def test_profiles():
    assert PROFILES == {"legacy": LEGACY, "r74": R74}
    assert LEGACY.as_dict() == {"continuation_floor": None, "continuation_ceiling": None,
                                "continuation_mass_law": None, "continuation_barrier_l": None, "isospin_completion": [],
                                "delta_lines": False, "isovector_meson_pairs": False,
                                "continuation_sibling_weight": None, "three_body_phase_space": False,
                                "continuation_daughters": None, "continuation_t_h_alpha": None, "leading_split_fraction": None, "leading_split_kappa": None,
                                "leading_split_slope": None, "leading_split_epsilon": None, "continuation_q_min": None, "sibling_line_width": None, "meson_split_fraction": None, "meson_split_slope": None, "string_step_mass": None, "string_step_slope": None, "string_step_epsilon": None, "leading_split_strange": None, "leading_split_regge": None, "xi_tower": None, "isovector_neutral_mesons": False, "isoscalar_dipion": False,
                                "floor_pools": [], "conjugate_towers": False,
                                "baryon_resonance_pairs": False, "non_diffractive_isospin_mixture": False,
                                "light_scalar_mesons": False, "nucleon_kstar_pairs": False,
                               "pdg_hyperons": False, "eta_prime": False, "hyperon_meson_partners": False,
                               "delta_isoscalar_pairs": False, "charged_scalar_pairs": False,
                               "xi_baryons": False, "pdg_meson_modes": False, "nstar_range_rule": False,
                               "three_body_endings": None, "dalitz_matrix_elements": False, "axial_isoscalars": False, "excited_kaons": False,
                               "pdg_nucleon_resonances": False, "associated_kstar_pairs": False,
                               "pdg_nucleon_resonance_siblings": False, "baryon_table_corrections": False,
                               "baryon_pdg_widths": False, "resonance_pair_line_shapes": False,
                               "direct_baryon_pairs_in_gap": False, "baryon_pair_s_wave": False,
                               "baryon_forward_hemisphere": False}
    assert R74.as_dict() == {"continuation_floor": 1.7, "continuation_ceiling": 17.3,
                             "continuation_mass_law": "density", "continuation_barrier_l": 1, "isospin_completion": [],
                             "delta_lines": False, "isovector_meson_pairs": False,
                             "continuation_sibling_weight": None, "three_body_phase_space": False,
                                "continuation_daughters": None, "continuation_t_h_alpha": None, "leading_split_fraction": None, "leading_split_kappa": None,
                                "leading_split_slope": None, "leading_split_epsilon": None, "continuation_q_min": None, "sibling_line_width": None, "meson_split_fraction": None, "meson_split_slope": None, "string_step_mass": None, "string_step_slope": None, "string_step_epsilon": None, "leading_split_strange": None, "leading_split_regge": None, "xi_tower": None, "isovector_neutral_mesons": False, "isoscalar_dipion": False,
                                "floor_pools": [], "conjugate_towers": False,
                                "baryon_resonance_pairs": False, "non_diffractive_isospin_mixture": False,
                                "light_scalar_mesons": False, "nucleon_kstar_pairs": False,
                               "pdg_hyperons": False, "eta_prime": False, "hyperon_meson_partners": False,
                               "delta_isoscalar_pairs": False, "charged_scalar_pairs": False,
                               "xi_baryons": False, "pdg_meson_modes": False, "nstar_range_rule": False,
                               "three_body_endings": None, "dalitz_matrix_elements": False, "axial_isoscalars": False, "excited_kaons": False,
                               "pdg_nucleon_resonances": False, "associated_kstar_pairs": False,
                               "pdg_nucleon_resonance_siblings": False, "baryon_table_corrections": False,
                               "baryon_pdg_widths": False, "resonance_pair_line_shapes": False,
                               "direct_baryon_pairs_in_gap": False, "baryon_pair_s_wave": False,
                               "baryon_forward_hemisphere": False}


def test_legacy_is_the_committed_kernel():
    committed = families_from_json(str(FAMILIES_PATH))
    _catalog, fams = load_kernel(LEGACY, NESTED_ROOT)
    assert fams == committed


def test_r74_profile_is_the_sealed_r74_kernel():
    spec = importlib.util.spec_from_file_location("staged_r74_for_kernel_config", R74_DRIVER)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    _c, theirs = driver._load_physics(1.7)
    _c2, mine = load_kernel(R74, NESTED_ROOT)
    assert mine == theirs
    # a floor grid varies the floor per branch
    _c3, other_floor = load_kernel(dataclasses.replace(R74, continuation_floor=1.5), NESTED_ROOT)
    assert all(f.continuation_floor == 1.5 for f in other_floor)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_events_are_bit_identical_through_the_profile(cls):
    """The same families give the same events: the profile is a pure
    reorganization of the in-memory options."""
    spec = importlib.util.spec_from_file_location("staged_r74_for_kernel_config_events", R74_DRIVER)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    cat_a, fams_a = driver._load_physics(1.7)
    cat_b, fams_b = load_kernel(R74, NESTED_ROOT)
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.63, epsilon_nd=0.23)
    for i in range(6):
        events = []
        for catalog, fams in ((cat_a, fams_a), (cat_b, fams_b)):
            subs = derive_substreams(20261004, 7, 9_800_000_000 + i)
            rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}
            ev = run_topology(cls, params, catalog, fams, **rngs)
            events.append([(h.pdg_id, h.four_momentum, h.line) for h in ev.final_state])
        assert events[0] == events[1]


@pytest.mark.parametrize("kwargs", [
    {"continuation_floor": 0.0}, {"continuation_floor": -1.0}, {"continuation_floor": float("nan")},
    {"continuation_ceiling": 4.0}, {"continuation_ceiling": float("inf")},
    {"continuation_mass_law": "thermal"}, {"continuation_barrier_l": -1}, {"continuation_barrier_l": True},
    {"continuation_barrier_l": 1}, {"continuation_barrier_l": 1, "continuation_mass_law": "edge"},
])
def test_options_validate(kwargs):
    with pytest.raises(ValueError):
        KernelConfig(**kwargs)


def test_a_committed_option_is_refused():
    fams = families_from_json(str(FAMILIES_PATH))
    fams[0] = dataclasses.replace(fams[0], continuation_ceiling=17.3)
    with pytest.raises(ValueError):
        apply_kernel_config(fams, LEGACY)
