"""The duality picks on the records' own line shapes
(``TopologyNominalParams.duality_line_shapes``, 2026-10-18): each low-mass
N* and Delta record weighted by its normalized running-width line shape,
the one the decayer samples for it, in place of the fixed-width
Breit-Wigner.  At the R100 optimum the paired probe gave -197 +- 47 with the
normalizations fitted (the external review's part C, Delta(1232) x 1.5 at
-64 +- 9)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.nd_duality import duality_pick, duality_records, isoscalar_exchange_records
from na49_mixing.resonance_decay import _check_daughter_masses
from na49_mixing.resonance_widths import spectral_function
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


@pytest.fixture(scope="module")
def kernel():
    r100 = json.loads((RUNS / "r100.json").read_text())
    return load_kernel(KernelConfig(**r100["kernel"]), NESTED_ROOT)


def _shares(records, mass):
    """The pick's probability per record at *mass* (a fine uniform scan of
    the pick, exact up to the scan step)."""
    out = {}
    n = 20000
    for k in range(n):
        pdg = duality_pick(mass, records, (k + 0.5) / n)
        out[pdg] = out.get(pdg, 0) + 1.0 / n
    return out


def test_the_default_records_carry_no_shape(kernel):
    cat, _fams = kernel
    assert all(r.shape is None for r in duality_records(cat))
    assert all(r.shape is None for r in isoscalar_exchange_records(cat))


def test_the_shapes_are_the_decayers(kernel):
    cat, _fams = kernel
    recs = duality_records(cat, line_shapes=True)
    for r in recs:
        rec = cat.get(r.pdg_id)
        masses = _check_daughter_masses(cat, rec)
        for m in (1.15, 1.3, 1.5, 1.7):
            if r.lo < m < r.hi:
                assert r.shape(m) == spectral_function(m, rec, masses, catalog=cat)
    i12 = isoscalar_exchange_records(cat, line_shapes=True)
    assert {r.pdg_id for r in i12} == {r.pdg_id for r in isoscalar_exchange_records(cat)}
    assert all(r.shape is not None for r in i12)


def test_the_delta_takes_the_threshold_region(kernel):
    """Near the N pi threshold the running widths suppress the broad
    N(1440)'s tail: the Delta(1232) share rises at 1.15-1.3 GeV."""
    cat, _fams = kernel
    fixed, shaped = duality_records(cat), duality_records(cat, line_shapes=True)
    for m in (1.15, 1.25):
        a, b = _shares(fixed, m), _shares(shaped, m)
        assert b[2214] > a[2214] and b.get(12212, 0.0) < a.get(12212, 0.0), (m, a, b)
        assert math.isclose(sum(b.values()), 1.0, rel_tol=1e-9)


def test_the_option_needs_a_duality_and_conserves(kernel):
    cat, fams = kernel
    r100 = json.loads((RUNS / "r100.json").read_text())
    topo = {k: v for k, v in r100["topology"].items()}
    base = TopologyNominalParams(epsilon=0.51, b_sd=6.15, epsilon_nd=0.249, nd_central=False,
                                 remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                  "high_mass_baryonic_continuum": 1.0}, **topo)
    with pytest.raises(ValueError, match="duality_line_shapes"):
        bad = dataclasses.replace(base, nd_duality_mass=None, nd_duality_shape_mass=None, diffractive_duality_mass=None,
                                  duality_line_shapes=True)
        rngs = {f"rng_{lab}": np.random.default_rng([1, k]) for k, lab in enumerate(LABELS)}
        run_topology("nd", bad, cat, fams, **rngs)
    on = dataclasses.replace(base, duality_line_shapes=True)
    for cls in ("nd", "sd_projectile", "dd"):
        params = on if cls == "nd" else dataclasses.replace(on, remnant_weights={
            "intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
        for seed in range(12):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, 2048, k]) for k, lab in enumerate(LABELS)}
            ev = run_topology(cls, params, cat, fams, **rngs)
            for i in range(3):
                assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]


def test_the_r102_spec_is_r100_with_the_line_shapes():
    r100, r102 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r100", "r102"))
    assert r102["kernel"] == r100["kernel"]
    assert r102["topology"] == {**r100["topology"], "duality_line_shapes": True}
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r102[block] == r100[block]
    assert [p["name"] for p in r102["parameters"]] == [p["name"] for p in r100["parameters"]]
    assert "flavor_channel_factors" not in r102
    g0 = r102["generation_0"]
    assert g0["run"] == "r100_pdg_tables_2026-10-17"
    assert r102["seeds"]["raw_first"] > 20337000 + 511
    branch = NESTED_ROOT.parent / "00runs" / "r100_pdg_tables_2026-10-17" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r102["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name


def test_the_redraw_on_the_line_shapes(kernel):
    """``duality_redraw_shapes`` (2026-10-20): below the shape mass the line
    mass follows the records' own line shapes; the running widths empty the
    region just above the N pi threshold, and the map keeps the continuum's
    share below the shape mass (no mass is lost or added)."""
    from na49_mixing.nd_duality import duality_redraw_mass, duality_redraw_mass_shapes
    cat, _fams = kernel
    shaped = duality_records(cat, line_shapes=True)
    lo, ms, eps = 1.08, 1.4, 0.247
    rng = np.random.default_rng(11)
    old, new = [], []
    for v in rng.random(3000):
        a2, c2 = lo * lo, ms * ms
        m = math.sqrt((a2 ** (-eps) + v * (c2 ** (-eps) - a2 ** (-eps))) ** (-1.0 / eps))
        old.append(duality_redraw_mass(m, lo, ms, eps, shaped))
        new.append(duality_redraw_mass_shapes(m, lo, ms, eps, shaped))
    assert None not in old and None not in new
    old, new = np.array(old), np.array(new)
    assert ((new > lo) & (new < ms)).all()
    assert (new < 1.16).sum() < 0.5 * (old < 1.16).sum()
    assert duality_redraw_mass_shapes(1.5, lo, ms, eps, shaped) is None
    with pytest.raises(ValueError, match="line shapes"):
        duality_redraw_mass_shapes(1.2, lo, ms, eps, duality_records(cat))


def test_the_redraw_option_needs_the_line_shapes(kernel):
    cat, fams = kernel
    r102 = json.loads((RUNS / "r102.json").read_text())
    base = TopologyNominalParams(epsilon=0.53, b_sd=6.44, epsilon_nd=0.247, nd_central=False,
                                 remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                  "high_mass_baryonic_continuum": 1.0}, **r102["topology"])
    rngs = {f"rng_{lab}": np.random.default_rng([3, k]) for k, lab in enumerate(LABELS)}
    with pytest.raises(ValueError, match="duality_redraw_shapes"):
        run_topology("nd", dataclasses.replace(base, duality_line_shapes=False, duality_redraw_shapes=True), cat, fams, **rngs)
    on = dataclasses.replace(base, duality_redraw_shapes=True)
    for seed in range(12):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 4096, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", on, cat, fams, **rngs)
        for i in range(3):
            assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]


def test_the_r103_spec_is_r102_with_the_redraw():
    r102, r103 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r102", "r103"))
    assert r103["kernel"] == r102["kernel"]
    assert r103["topology"] == {**r102["topology"], "duality_redraw_shapes": True}
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r103[block] == r102[block]
    assert [p["name"] for p in r103["parameters"]] == [p["name"] for p in r102["parameters"]]
    assert r103["generation_0"]["run"] == "r102_duality_shapes_2026-10-18"
    assert r103["seeds"]["raw_first"] > r102["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r102_duality_shapes_2026-10-18" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r103["parameters"]}
        for name in r103["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name
