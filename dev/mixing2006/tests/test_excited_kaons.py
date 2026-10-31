"""The excited kaons K*(1410), K0*(1430), K1(1400) and K*(1680) (2026-10-28, ``KernelConfig.excited_kaons``): the
sixteen records at their PDG 2024 values validate in the catalog, the classes follow every baryon-kaon class of the
baryonic families, the mesonic families and the committed kernel are unchanged, and events conserve."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import _parse_class
from na49_mixing.excited_kaons import EXCITED_KAON_PARTNERS, excited_kaon_classes, load_excited_kaons
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R125 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r125.json").read_text())["kernel"])
HEAVY = {p for ps in EXCITED_KAON_PARTNERS.values() for p in ps}


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(R125, NESTED_ROOT), load_kernel(dataclasses.replace(R125, excited_kaons=True), NESTED_ROOT)


def test_the_option_is_off_by_default_and_validated():
    assert KernelConfig().excited_kaons is False and R125.excited_kaons is False
    assert R125.as_dict()["excited_kaons"] is False
    with pytest.raises(ValueError, match="excited_kaons"):
        KernelConfig(excited_kaons=1)


def test_the_records(kernels):
    (cat0, _f0), (cat, _fams) = kernels
    have0 = {int(r.pdg_id) for r in cat0.records()}
    assert not HEAVY & have0
    recs = {int(r.pdg_id): r for r in cat.records()}
    assert HEAVY <= set(recs)
    config = load_excited_kaons(NESTED_ROOT)
    assert len(config["records"]) == 16
    for spec in config["records"]:
        r = recs[int(spec["pdg_id"])]
        assert (r.Q, r.S, r.isospin, r.isospin3, r.spin, r.parity) == (spec["Q"], spec["S"], 0.5, spec["isospin3"],
                                                                       spec["spin"], spec["parity"]), r.name
        assert r.pole_mass == pytest.approx(spec["pole_mass"]) and r.nominal_width == pytest.approx(spec["nominal_width"])
        fractions = [c.branching_fraction for c in r.decay_channels]
        assert sum(fractions) == pytest.approx(1.0) and all(f > 0 for f in fractions)
        assert r.mass_range[0] < r.pole_mass < r.mass_range[1] == pytest.approx(r.pole_mass + 5 * r.nominal_width)
        # (B, Q, S) conservation of every mode
        for c in r.decay_channels:
            q = sum(recs[d].Q if d in recs else cat.get(d).Q for d in c.daughters)
            s = sum(recs[d].S if d in recs else cat.get(d).S for d in c.daughters)
            assert (q, s) == (r.Q, r.S), (r.name, c.daughters)
    # the vectors and the others, by spin and parity (the ledger's r_V follows spin 1, parity -)
    assert {recs[p].spin for p in (100323, 30323)} == {1.0} and {recs[p].parity for p in (100323, 30323)} == {"-"}
    assert (recs[10321].spin, recs[10321].parity, recs[20323].spin, recs[20323].parity) == (0.0, "+", 1.0, "+")
    report = cat.validate()
    assert not report.errors, report.errors[:5]


def test_the_classes_follow_every_baryon_kaon_class(kernels):
    (cat0, f0), (cat, fams) = kernels
    by0 = {f.family_id: f for f in f0}
    assert [f.family_id for f in fams] == [f.family_id for f in f0]
    for f in fams:
        old = list(by0[f.family_id].allowed_decay_classes)
        new = [c for c in f.allowed_decay_classes if c not in old]
        assert list(f.allowed_decay_classes)[:len(old)] == old
        if int(f.qn[0]) == 0:
            assert not new, f.family_id
            continue
        baryon_kaon = set()
        recs = {int(r.pdg_id): r for r in cat0.records()}
        for c in old:
            d0, d1 = _parse_class(c)
            for b, k in ((d0, d1), (d1, d0)):
                if k in EXCITED_KAON_PARTNERS and b in recs and int(recs[b].B) != 0:
                    baryon_kaon.add((b, k))
        expected = {tuple(sorted((b, h))) for b, k in baryon_kaon for h in EXCITED_KAON_PARTNERS[k]}
        assert {tuple(sorted(_parse_class(c))) for c in new} == expected, f.family_id
        assert new == excited_kaon_classes(old, cat)
        for c in new:
            d0, d1 = _parse_class(c)
            assert (d0 in HEAVY) != (d1 in HEAVY)
    strange_baryonic = [f for f in fams if int(f.qn[0]) == 1 and int(f.qn[2]) == -1]
    non_strange = [f for f in fams if int(f.qn[0]) == 1 and int(f.qn[2]) == 0]
    assert strange_baryonic and non_strange
    for f in strange_baryonic + non_strange:
        assert any(d in HEAVY for c in f.allowed_decay_classes for d in _parse_class(c)), f.family_id


def test_the_r127_spec_is_r126_with_the_excited_kaons():
    r126, r127 = (json.loads((NESTED_ROOT / "config" / "runs" / f"{r}.json").read_text()) for r in ("r126", "r127"))
    assert r127["kernel"] == {**r126["kernel"], "excited_kaons": True} and not r126["kernel"].get("excited_kaons")
    for block in ("topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope", "fit"):
        assert r127[block] == r126[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r127["parameters"]] == [strip(p) for p in r126["parameters"]]
    assert r127["generation_0"]["run"] == "r126_correlated_errors_replica_2026-10-28"
    assert r127["seeds"]["raw_first"] > r126["seeds"]["raw_first"] + 511
    assert r127["seeds"]["partition_base"] != r126["seeds"]["partition_base"]
    assert r127["identity"]["run_id"] == "r127_excited_kaons_2026-10-28"
    assert KernelConfig(**r127["kernel"]).excited_kaons is True


def test_events_conserve_with_the_excited_kaons(kernels):
    from na49_mixing.topology_runner import TopologyNominalParams, run_topology
    _c0, (cat, fams) = kernels
    params = TopologyNominalParams(epsilon=0.1, b_sd=5.65, epsilon_nd=0.248, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0},
                                   exact_transfer_range=True, line_thresholds=True)
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    seen = set()
    for cls in ("nd", "sd_projectile"):
        p = params if cls == "nd" else dataclasses.replace(params, remnant_weights={
            "intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
        for seed in range(20):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, 28102026, k]) for k, lab in enumerate(labels)}
            ev = run_topology(cls, p, cat, fams, **rngs)
            for i in range(3):
                assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]
            seen |= {int(h.pdg_id) for h in ev.final_state}
    assert 321 in seen or -321 in seen
