"""Tests for the two-component variant of the non-diffractive class
(2026-10-02): ``generate_nd(..., central=False)`` and
``TopologyNominalParams(nd_central=False)`` draw the class without its
central source, so a fit can measure the strength of the central
component against the pure two-component picture.

The default path keeps the card's event bit for bit (the contract pins
cover it); the variant shares the remnant selections, transfers and
remnant decay draws with the default event of the same seed (common
random numbers), conserves (B, Q, S) and four-momentum exactly, carries
no central-line hadron, and reweights in the exponents without the
central mass law.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_mass_laws import log_density_ratio
from na49_mixing.topology_non_diffractive import (
    CENTRAL_LINE_LABEL,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    check_central_neutrality,
)
from na49_mixing.topology_runner import NOMINAL_PARAMS, extract_observables, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
SEED = 20261002
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
EXCITED = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1, "high_mass_baryonic_continuum": 0.9}


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return [dataclasses.replace(f, continuation_floor=1.7) for f in families_from_json(str(FAMILIES_PATH))]


def _params(central: bool):
    return dataclasses.replace(NOMINAL_PARAMS, remnant_weights=dict(EXCITED), epsilon=0.5, b_sd=2.4,
                               epsilon_nd=0.4, nd_central=central)


def _event(i: int, central: bool, catalog, fams):
    subs = derive_substreams(SEED, 0, 9_300_000_000 + i)
    rngs = {f"rng_{label}": np.random.default_rng(subs.stream_for(label)) for label in LABELS}
    return run_topology("nd", _params(central), catalog, fams, **rngs)


def _cm_total(event):
    return [math.fsum(float(h.four_momentum[j]) for h in event.final_state) for j in range(4)]


def test_default_is_the_central_event(catalog, fams):
    assert NOMINAL_PARAMS.nd_central is True
    a = _event(0, True, catalog, fams)
    subs = derive_substreams(SEED, 0, 9_300_000_000)
    rngs = {f"rng_{label}": np.random.default_rng(subs.stream_for(label)) for label in LABELS}
    b = run_topology("nd", dataclasses.replace(_params(True)), catalog, fams, **rngs)
    assert a == b and a.central_component and a.central_source is not None and a.central_mass > 0.0


def test_two_component_events_conserve_and_carry_no_central_line(catalog, fams):
    for i in range(120):
        event = _event(i, False, catalog, fams)
        assert event.central_component is False and event.central_source is None and event.central_mass == 0.0
        assert all(h.line in (PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL) for h in event.final_state)
        assert not any(h.line == CENTRAL_LINE_LABEL for h in event.final_state)
        bqs = tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3))
        assert bqs == (2, 2, 0)
        reference = _cm_total(_event(i, True, catalog, fams))
        total = _cm_total(event)
        assert all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(total, reference))
        check_central_neutrality(event)
        obs = extract_observables(event, 17.27)
        assert obs.excitation_masses == ()


def test_common_random_numbers_with_the_central_event(catalog, fams):
    """The same seed draws the same remnants, transfers and remnant
    decay chains with and without the central source; only the recoil
    momenta and the central products differ."""

    same_species = 0
    for i in range(60):
        on, off = _event(i, True, catalog, fams), _event(i, False, catalog, fams)
        assert [r.kind for r in on.remnants] == [r.kind for r in off.remnants]
        assert [r.mass for r in on.remnants] == [r.mass for r in off.remnants]
        assert (on.t1, on.t2) == (off.t1, off.t2)
        for line in (PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL):
            if sorted(h.pdg_id or 0 for h in on.final_state if h.line == line) == \
                    sorted(h.pdg_id or 0 for h in off.final_state if h.line == line):
                same_species += 1
    # the remnant decay draws are shared; the species lists agree line by
    # line (a channel threshold crossed by the slightly larger recoil
    # energy may change a rare chain)
    assert same_species >= 110


def test_two_component_events_carry_more_baryonic_momentum(catalog, fams):
    """Without the central source the two baryonic systems share the full
    sqrt(s): their summed CM energy is sqrt(s) exactly."""

    from na49_mixing.topology_non_diffractive import _cm_pair, _cm_triplet
    s1, s2 = _cm_pair(2.1, -0.2, 1.6, 17.27)
    t1, t2, c = _cm_triplet(2.1, -0.2, 1.6, 0.5, 17.27)
    assert math.isclose(s1[0] + s2[0], 17.27, rel_tol=1e-3)
    assert abs(s1[3]) > abs(t1[3]) and math.isclose(s1[3], -s2[3], rel_tol=1e-12)


def _shape(event):
    """The non-diffractive mass-law arrays (run_observables.shape_arrays)."""
    windows = event.remnant_windows
    kind = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 1.0, "high_mass_baryonic_continuum": 2.0}
    return {"t1": event.t1, "t2": event.t2, "t_c": event.t_c, "mc2": event.central_mass ** 2,
            "m1": event.remnants[0].mass, "m2": event.remnants[1].mass,
            "k1": kind[event.remnants[0].kind], "k2": kind[event.remnants[1].kind],
            "lo1": windows[0][0], "hi1": windows[0][1], "lo2": windows[1][0], "hi2": windows[1][1]}


def test_density_ratio_mixed_events(catalog, fams):
    """Mixed arrays: the central events reproduce the all-central
    evaluation, the two-component events carry the transfer and remnant
    terms only, and the generation point gives zero."""

    on = [_shape(_event(i, True, catalog, fams)) for i in range(40)]
    off = [_shape(_event(i, False, catalog, fams)) for i in range(40)]
    cols = lambda rows: {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}  # noqa: E731
    mixed = cols(on[:20] + off[20:])
    central_only = cols(on[:20])
    point = dict(eps0=0.4, b_sd0=2.4, sqrt_s=17.27)
    for eps, b in ((0.4, 2.4), (0.3, 3.0), (0.55, 2.0)):
        got = log_density_ratio("nd", mixed, eps, b, **point)
        assert np.all(np.isfinite(got))
        ref = log_density_ratio("nd", central_only, eps, b, **point)
        assert np.allclose(got[:20], ref, rtol=1e-12, atol=1e-12)
        if (eps, b) == (0.4, 2.4):
            assert np.allclose(got, 0.0, atol=1e-12)
    # the two-component rows equal the all-central expression without the
    # central terms: at fixed b and t the eps dependence comes from the
    # remnant law alone
    off_only = cols(off[20:])
    for eps in (0.3, 0.55):
        a = log_density_ratio("nd", off_only, eps, 2.4, **point)
        b_ = log_density_ratio("nd", mixed, eps, 2.4, **point)[20:]
        assert np.allclose(a, b_, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        bad = dict(mixed); bad["mc2"] = mixed["mc2"].copy(); bad["mc2"][0] = -1.0
        log_density_ratio("nd", bad, 0.3, 2.4, **point)
