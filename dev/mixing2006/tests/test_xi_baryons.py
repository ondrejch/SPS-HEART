"""The S = -2 baryons (2026-10-16): ``KernelConfig.xi_baryons`` adds the
Xi0, the Xi- (stable) and the Xi(1530) of the PDG 2025 summary tables with
the (Xi, K) and (Xi(1530), K) classes at the strange baryonic families,
before the conjugate towers; off, the committed kernel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.flavor_weights import flavor_exponents
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R98 = json.loads((NESTED_ROOT / "config" / "runs" / "r98.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
XI = (3322, 3312, 3324, 3314)


@pytest.fixture(scope="module")
def kernels():
    off = load_kernel(KernelConfig(**R98["kernel"]), NESTED_ROOT)
    on = load_kernel(KernelConfig(**{**R98["kernel"], "xi_baryons": True}), NESTED_ROOT)
    return off, on


def test_off_is_the_committed_kernel(kernels):
    (cat_off, fams_off), _on = kernels
    ids = {abs(int(r.pdg_id)) for r in cat_off.records()}
    assert not ids & set(XI)
    tokens = {t.strip() for f in fams_off for c in f.allowed_decay_classes for t in str(c).strip("()").split(",")}
    assert not {t for t in tokens if t.lstrip("-").isdigit() and abs(int(t)) in XI}


def test_the_records(kernels):
    """The ground states stable at their masses, the Xi(1530) to Xi pi at
    the Clebsch-Gordan shares 2/3 : 1/3 in the P wave, the antibaryons the
    conjugates; the catalog validates."""
    _off, (cat, _fams) = kernels
    assert cat.validate().errors == ()
    for pid, q, mass in ((3322, 0, 1.31486), (3312, -1, 1.32171)):
        rec = cat.get(pid)
        assert (rec.B, rec.Q, rec.S, rec.isospin) == (1, q, -2, 0.5) and rec.pole_mass == mass
        assert rec.stable_for_decay and not rec.decay_channels and pid in cat.stable_states()
        anti = cat.get(-pid)
        assert (anti.B, anti.Q, anti.S) == (-1, -q, 2) and anti.stable_for_decay
    star0 = {tuple(c.daughters): c.branching_fraction for c in cat.get(3324).decay_channels}
    assert star0 == {(3312, 211): pytest.approx(2 / 3), (3322, 111): pytest.approx(1 / 3)}
    starm = {tuple(c.daughters): c.branching_fraction for c in cat.get(3314).decay_channels}
    assert starm == {(3322, -211): pytest.approx(2 / 3), (3312, 111): pytest.approx(1 / 3)}
    assert all(c.orbital_angular_momentum == 1 for c in cat.get(3324).decay_channels)
    rec = cat.get(3324)
    assert rec.mass_range == (pytest.approx(1.31486 + 0.1349768, abs=1e-4), pytest.approx(1.5318 + 5 * 0.0091))
    assert math.isclose(math.fsum(c.branching_fraction for c in cat.get(-3314).decay_channels), 1.0)


def test_the_classes(kernels):
    """(Xi, K) and (Xi(1530), K) at the strange baryonic families of the
    matching charge, their conjugates at the conjugate towers, and nothing
    else changed."""
    (_c0, fams_off), (_c1, fams) = kernels
    off = {str(f.family_id): set(f.allowed_decay_classes) for f in fams_off}
    on = {str(f.family_id): set(f.allowed_decay_classes) for f in fams}
    added = {k: on[k] - off.get(k, set()) for k in on if on[k] - off.get(k, set())}
    assert added == {
        "baryonic_strange": {"(3312,321)", "(3322,311)", "(3314,321)", "(3324,311)"},
        "baryonic_strange_q1": {"(3322,321)", "(3324,321)"},
        "baryonic_strange_qm1": {"(3312,311)", "(3314,311)"},
        "anti_baryonic_strange": {"(-3312,-321)", "(-3322,-311)", "(-3314,-321)", "(-3324,-311)"},
        "anti_baryonic_strange_q1": {"(-3322,-321)", "(-3324,-321)"},
        "anti_baryonic_strange_qm1": {"(-3312,-311)", "(-3314,-311)"},
    }


def test_one_created_pair(kernels):
    """A (Xi, K) channel of a strange baryonic line creates one s-sbar pair
    beside the line's strange quark, so its flavor weight is gamma_s."""
    _off, (cat, fams) = kernels
    family = next(f for f in fams if str(f.family_id) == "baryonic_strange")

    class _Channel:
        daughters = ("catalog:3312", "catalog:321")
    exps = flavor_exponents((1, 0, -1), _Channel(), cat, fams)
    assert (exps.n_ssbar, exps.n_v, exps.n_bbar, exps.n_assoc) == (1.0, 0, 0, 0)
    assert family.qn[2] == -1


def test_the_refusal():
    iso = [x for x in R98["kernel"]["isospin_completion"] if x != "strange_baryonic"]
    bare = {**R98["kernel"], "xi_baryons": True, "pdg_hyperons": False, "isospin_completion": iso}
    with pytest.raises(ValueError, match="xi_baryons"):
        KernelConfig(**bare)


def test_events_conserve(kernels):
    """Non-diffractive events with the Xi conserve baryon number, charge and
    strangeness."""
    _off, (cat, fams) = kernels
    params = TopologyNominalParams(
        epsilon=0.525, b_sd=6.11, epsilon_nd=0.249, nd_central=False, exact_transfer_range=True, line_thresholds=True,
        transfer_slope_mass_exponent=1.25, low_mass_spin_isospin=True, nd_duality_mass=2.0, nd_duality_shape_mass=1.4,
        remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 1.0})
    for seed in range(25):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 98, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", params, cat, fams, **rngs)
        assert sum(int(h.qn[0]) for h in ev.final_state) == 2
        assert sum(int(h.qn[1]) for h in ev.final_state) == 2
        assert sum(int(h.qn[2]) for h in ev.final_state) == 0


def test_the_r99_spec_is_r98_with_the_xi():
    """R99: the R98 contract with the S = -2 baryons, generated at the R98
    optimum with every free coordinate from the fit (no judgment)."""
    runs = NESTED_ROOT / "config" / "runs"
    r99 = json.loads((runs / "r99.json").read_text())
    assert r99["kernel"] == {**R98["kernel"], "xi_baryons": True}
    assert r99["topology"] == R98["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r99[block] == R98[block]
    assert [p["name"] for p in r99["parameters"]] == [p["name"] for p in R98["parameters"]]
    g0 = r99["generation_0"]
    assert g0["run"] == "r98_completion_2026-10-15" and g0["fit"] == "free" and "judgment" not in g0
    assert r99["seeds"]["raw_first"] > R98["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r98_completion_2026-10-15" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r99["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name
