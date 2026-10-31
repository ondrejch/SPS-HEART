"""Importance-ensemble discipline and the exact mass-law density ratios
(2026-09-28; review record ``.collab/reviews/fable-speedup-2026-09-28.md``,
FAB-02 and FAB-04).

The unit-mean test is the reweighting-validity check the R51 fit lacked:
for a sample generated at the nominal point, the mean of
``exp(log_density_ratio)`` at any trial point is exactly 1 in
expectation, class by class.  The sealed R51 driver's double-diffraction
branch carried an extra single-marginal normalization ratio and gave
2.03 at epsilon = 0.3 and 0.39 at epsilon = -0.3 on the same check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.importance import (
    DEFAULT_ESS_FLOOR,
    RegenerateAt,
    check_trust_region,
    ess_fraction,
    logits_from_simplex,
    simplex_from_logits,
)
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_mass_laws import (
    log_density_ratio,
    log_dd_joint_mass,
    log_power_integral,
)
from na49_mixing.topology_runner import (
    ELASTIC_TRANSFER_SLOPE,
    NOMINAL_PARAMS,
    SQRT_S,
    extract_observables,
    run_topology,
)
from na49_mixing.topology_remnants import REMNANT_KIND_INDEX, KIND_HIGH_MASS_BARYONIC_CONTINUUM
from na49_mixing.topology_runner import TopologyNominalParams
from na49_mixing.topology_mass_laws import EXCITED_REMNANT_KIND_INDEX
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES = Path(__file__).resolve().parent.parent / "config" / "v2_singlet" / "continuum_resonances.json"


def test_ess_fraction_limits():
    assert ess_fraction(np.zeros(100)) == 1.0
    assert ess_fraction([]) == 1.0
    one_heavy = np.array([0.0] + [-50.0] * 99)
    assert abs(ess_fraction(one_heavy) - 0.01) < 1e-9
    assert ess_fraction([-math.inf, -math.inf]) == 0.0
    # shift invariance
    lw = np.random.default_rng(1).normal(size=500)
    assert abs(ess_fraction(lw) - ess_fraction(lw + 123.4)) < 1e-12


def test_trust_region_refuses_degenerate_ensembles():
    assert check_trust_region(np.zeros(10)) == 1.0
    with pytest.raises(RegenerateAt) as info:
        check_trust_region(np.array([0.0] + [-50.0] * 99), point=("theta", 1))
    assert info.value.point == ("theta", 1)
    assert info.value.ess < DEFAULT_ESS_FLOOR
    with pytest.raises(ValueError):
        check_trust_region(np.zeros(3), floor=0.0)


def test_simplex_parametrization_round_trip():
    fractions = np.array([0.7, 0.2, 0.09, 0.01])
    z = logits_from_simplex(fractions)
    assert abs(z.mean()) < 1e-12
    back = simplex_from_logits(z)
    assert np.allclose(back, fractions, atol=1e-12)
    assert abs(simplex_from_logits([3.0, -1.0, 0.5]).sum() - 1.0) < 1e-12
    assert np.all(simplex_from_logits([100.0, -100.0]) > 0.0)
    with pytest.raises(ValueError):
        logits_from_simplex([0.5, 0.6])


def test_power_integral_and_joint_mass_limits():
    a, c = 0.88, 266.7
    assert abs(math.exp(log_power_integral(0.0, a, c)) - math.log(c / a)) < 1e-12
    eps = 0.3
    exact = (a ** (-eps) - c ** (-eps)) / eps
    assert abs(math.exp(log_power_integral(eps, a, c)) - exact) < 1e-12 * exact
    # the joint mass is continuous through eps = 0
    assert abs(log_dd_joint_mass(1e-9, SQRT_S) - log_dd_joint_mass(0.0, SQRT_S)) < 1e-6


def _nominal_sample(label: str, n: int, catalog, fams, seed: int = 20260928, params=NOMINAL_PARAMS):
    rows = []
    for i in range(n):
        subs = derive_substreams(seed, 0, 1_950_000_000 + i)
        rngs = {k: np.random.default_rng(subs.stream_for(k)) for k in
                ("topology_selection", "spectral_state", "exchange_kinematics",
                 "decay_channel", "decay_kinematics")}
        ev = run_topology(label, params, catalog, fams,
                          rng_topology_selection=rngs["topology_selection"],
                          rng_spectral_state=rngs["spectral_state"],
                          rng_exchange_kinematics=rngs["exchange_kinematics"],
                          rng_decay_channel=rngs["decay_channel"],
                          rng_decay_kinematics=rngs["decay_kinematics"],
                          p_n2=NOMINAL_PARAMS.p_n2)
        obs = extract_observables(ev, SQRT_S)
        ex = [float(m) for m in obs.excitation_masses]
        tr = [float(t) for t in obs.transfers]
        if label == "sd_projectile":
            rows.append((ex[0] ** 2, tr[0], float(ev.remnant.mass)))
        elif label == "dd":
            rows.append((ex[0] ** 2, ex[1] ** 2, tr[0], tr[1]))
        else:
            (lo1, hi1), (lo2, hi2) = ev.remnant_windows
            rows.append((tr[0], tr[1], tr[2], ex[0] ** 2, float(ev.remnants[0].mass), float(ev.remnants[1].mass),
                         float(REMNANT_KIND_INDEX[ev.remnants[0].kind]), float(REMNANT_KIND_INDEX[ev.remnants[1].kind]),
                         float(lo1), float(hi1), float(lo2), float(hi2)))
    cols = {"sd_projectile": ("m2", "t", "m_rem"), "dd": ("m12", "m22", "t1", "t2"),
            "nd": ("t1", "t2", "t_c", "mc2", "m1", "m2", "k1", "k2", "lo1", "hi1", "lo2", "hi2")}[label]
    return {k: np.array([r[i] for r in rows]) for i, k in enumerate(cols)}


@pytest.mark.parametrize("eps, b_sd", [(0.3, ELASTIC_TRANSFER_SLOPE), (-0.3, ELASTIC_TRANSFER_SLOPE), (0.0, 4.0)])
def test_unit_mean_weights_per_class(eps, b_sd):
    """For every class the mean importance weight over a nominal sample
    is 1 within 3 standard errors (the reweighting-validity check)."""
    catalog = HadronCatalog.from_json(str(CATALOG))
    fams = families_from_json(str(FAMILIES))
    for label in ("sd_projectile", "dd", "nd"):
        arrays = _nominal_sample(label, 1200, catalog, fams)
        lw = log_density_ratio(label, arrays, eps, b_sd, eps0=NOMINAL_PARAMS.epsilon,
                               b_sd0=NOMINAL_PARAMS.b_sd, sqrt_s=SQRT_S)
        w = np.exp(lw)
        mean = w.mean()
        se = w.std(ddof=1) / math.sqrt(w.size)
        assert abs(mean - 1.0) <= 3.0 * se + 0.01, (label, eps, b_sd, mean, se)
        assert ess_fraction(lw) > 0.05, (label, eps, b_sd, ess_fraction(lw))


@pytest.mark.parametrize("eps", [0.3, -0.3])
def test_unit_mean_weights_nd_with_excited_remnants(eps):
    """The non-diffractive ratio carries the excited remnants' own mass law
    (2026-09-29): over a sample with released remnant weights the mean
    importance weight is 1 within 3 standard errors, and the ratio refuses
    arrays without the kind and window columns."""
    assert REMNANT_KIND_INDEX[KIND_HIGH_MASS_BARYONIC_CONTINUUM] == EXCITED_REMNANT_KIND_INDEX
    catalog = HadronCatalog.from_json(str(CATALOG))
    fams = families_from_json(str(FAMILIES))
    released = TopologyNominalParams(remnant_weights={"intact_proton": 1 / 3, "low_mass_baryonic_resonance": 1 / 3,
                                                      "high_mass_baryonic_continuum": 1 / 3})
    arrays = _nominal_sample("nd", 1500, catalog, fams, seed=20260929, params=released)
    assert np.any(arrays["k1"] == 2) and np.any(arrays["k2"] == 2)
    lw = log_density_ratio("nd", arrays, eps, NOMINAL_PARAMS.b_sd, eps0=NOMINAL_PARAMS.epsilon,
                           b_sd0=NOMINAL_PARAMS.b_sd, sqrt_s=SQRT_S)
    w = np.exp(lw)
    mean = w.mean()
    se = w.std(ddof=1) / math.sqrt(w.size)
    assert abs(mean - 1.0) <= 3.0 * se + 0.02, (eps, mean, se)
    assert ess_fraction(lw) > 0.02
    stripped = {k: v for k, v in arrays.items() if k not in ("k1", "k2", "lo1", "hi1", "lo2", "hi2")}
    with pytest.raises(ValueError):
        log_density_ratio("nd", stripped, eps, NOMINAL_PARAMS.b_sd, eps0=NOMINAL_PARAMS.epsilon,
                          b_sd0=NOMINAL_PARAMS.b_sd, sqrt_s=SQRT_S)
