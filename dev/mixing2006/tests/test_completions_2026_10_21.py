"""The completions of 2026-10-21 (the external reviews of R104, `.collab/`):
the PDG 2025 N and Delta resonances up to the 2.0 GeV floor
(``KernelConfig.pdg_nucleon_resonances``), the axial isoscalars f1(1285) and
h1(1170) (``axial_isoscalars``) and the (Y, K*(892)) partners of the
associated pairs (``associated_kstar_pairs``)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import CascadeState, _charge_conjugate_pdg, enumerate_channels
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.nd_duality import duality_records
from na49_mixing.pdg_nucleon_resonances import load_pdg_nucleon_resonances, new_resonance_ids
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
OPTIONS = {"pdg_nucleon_resonances": True, "axial_isoscalars": True, "associated_kstar_pairs": True}


@pytest.fixture(scope="module")
def kernels():
    r104 = json.loads((RUNS / "r104.json").read_text())
    base = load_kernel(KernelConfig(**r104["kernel"]), NESTED_ROOT)
    full = load_kernel(KernelConfig(**{**r104["kernel"], **OPTIONS}), NESTED_ROOT)
    return r104, base, full


def _qn_sum(cat, daughters):
    from na49_mixing.cascade_decay import _qn_of
    return tuple(sum(int(_qn_of(cat, d)[i]) for d in daughters) for i in range(3))


def test_the_options_are_off_by_default_and_validated():
    cfg = KernelConfig()
    for name in (*OPTIONS, "pdg_nucleon_resonance_siblings"):
        assert getattr(cfg, name) is False and cfg.as_dict()[name] is False
    with pytest.raises(ValueError, match="pdg_nucleon_resonances needs"):
        KernelConfig(pdg_nucleon_resonances=True)
    with pytest.raises(ValueError, match="needs pdg_nucleon_resonances"):
        KernelConfig(pdg_nucleon_resonance_siblings=True)
    with pytest.raises(ValueError, match="axial_isoscalars"):
        KernelConfig(axial_isoscalars=1)


def test_the_nucleon_resonance_records(kernels):
    _r, (cat0, _f0), (cat, _f) = kernels
    config = load_pdg_nucleon_resonances(NESTED_ROOT)
    ids = new_resonance_ids(config)
    assert len(ids) == len(set(ids)) == 7 * 2 + 6 * 4
    held0 = {int(r.pdg_id) for r in cat0.records()}
    assert not set(ids) & held0
    for pdg in ids + tuple(-p for p in ids):
        rec = cat.get(pdg)
        assert math.isclose(math.fsum(c.branching_fraction for c in rec.decay_channels), 1.0, rel_tol=1e-12)
        for c in rec.decay_channels:
            assert _qn_sum(cat, c.daughters) == (int(rec.B), int(rec.Q), int(rec.S)), (rec.name, c.daughters)
    # the duality picks hold the new charge-one records
    recs = {r.pdg_id for r in duality_records(cat)}
    assert {p for p in ids if int(cat.get(p).Q) == 1} <= recs
    # Delta(1950) carries its N pi pi remainder as a three-body mode; N(1895) its N eta', self-conjugate in the antibaryon
    assert any(c.is_three_body for c in cat.get(9241950).decay_channels)
    assert any(331 in c.daughters for c in cat.get(-9121895).decay_channels)
    assert _charge_conjugate_pdg(331, cat0) == 331 and _charge_conjugate_pdg(211, cat0) == -211


def test_the_isospin_splits(kernels):
    _r, _b, (cat, _f) = kernels
    # N(1675)+ -> N pi: n pi+ : p pi0 = 2 : 1; Delta(1620)++ -> N pi only p pi+
    n1675 = {c.daughters: c.branching_fraction for c in cat.get(9121675).decay_channels}
    assert math.isclose(n1675[(2112, 211)] / n1675[(2212, 111)], 2.0, rel_tol=1e-9)
    d1620 = [c.daughters for c in cat.get(9241620).decay_channels if set(map(abs, c.daughters)) <= {2212, 2112, 211, 111}]
    assert d1620 == [(2212, 211)]
    # f1(1285): 2pi+ 2pi- : pi+ pi- 2pi0 = 1 : 2 through rho pi pi
    f1 = {c.daughters: c.branching_fraction for c in cat.get(20223).decay_channels}
    assert math.isclose(f1[(213, -211, 111)] + f1[(-213, 211, 111)], 2.0 * f1[(113, 211, -211)], rel_tol=1e-9)


def test_the_axial_classes_follow_the_omega(kernels):
    _r, (_c0, f0), (cat, fams) = kernels
    by0 = {f.family_id: f for f in f0}
    for f in fams:
        new = [c for c in f.allowed_decay_classes if c not in by0[f.family_id].allowed_decay_classes]
        axial = [c for c in new if "20223" in c or "10223" in c]
        omega_pi = [c for c in by0[f.family_id].allowed_decay_classes
                    if "223" in c.strip("()").split(",") and any(p in c.strip("()").split(",") for p in ("111", "211", "-211"))]
        assert len(axial) == 2 * len(omega_pi), f.family_id
    st = CascadeState(momentum=(2.1, 0.0, 0.0, 0.0), mass=2.1, qn=(0, 0, 0), quantum={}, family_id="mesonic_neutral",
                      axis=(0.0, 0.0, 1.0), alignment_label="none", parent=None, exchange_ancestry=(), depth=2)
    # the I = 0 line reaches no (f1, pi0), (h1, pi0)
    assert not any(("catalog:20223" in c.daughters or "catalog:10223" in c.daughters) for c in enumerate_channels(st, cat, fams))


def test_the_associated_kstar_classes(kernels):
    _r, (_c0, f0), (cat, fams) = kernels
    by0 = {f.family_id: f for f in f0}
    for f in fams:
        new = [c for c in f.allowed_decay_classes if c not in by0[f.family_id].allowed_decay_classes]
        kstar = [c for c in new if any(t in ("323", "313", "-323", "-313") for t in c.strip("()").split(","))]
        if int(f.qn[0]) == 0 or int(f.qn[2]) != 0:
            assert not kstar, f.family_id
        else:
            assert kstar, f.family_id


def test_events_conserve_with_the_completions(kernels):
    r104, _b, (cat, fams) = kernels
    params = TopologyNominalParams(epsilon=0.52, b_sd=6.43, epsilon_nd=0.247, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0}, **r104["topology"])
    for cls in ("nd", "sd_projectile", "dd"):
        p = params if cls == "nd" else dataclasses.replace(params, remnant_weights={
            "intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
        for seed in range(15):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, 21021, k]) for k, lab in enumerate(LABELS)}
            ev = run_topology(cls, p, cat, fams, **rngs)
            for i in range(3):
                assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]


def test_the_r105_spec_is_r104_with_the_completions():
    r104, r105 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r104", "r105"))
    assert r105["kernel"] == {**r104["kernel"], **OPTIONS}
    assert r105["topology"] == r104["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r105[block] == r104[block]
    assert [p["name"] for p in r105["parameters"]] == [p["name"] for p in r104["parameters"]]
    assert r105["generation_0"]["run"] == "r104_dalitz_2026-10-20"
    assert r105["seeds"]["raw_first"] > r104["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r104_dalitz_2026-10-20" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r105["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r105["parameters"]}
        for name in r105["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
