"""The (N, Kbar*(892)) pairs of the strange baryonic lines (2026-10-12):
``KernelConfig.nucleon_kstar_pairs`` adds the class (N, Kbar*) beside
every (N, Kbar) class of a strange baryonic family, the completion
families and the conjugate towers included; off, the committed classes."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL, KIND_CONTINUUM, CascadeState, enumerate_channels
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.nucleon_kstar_pairs import KSTAR_OF_KAON, nucleon_kstar_classes, nucleon_kstar_families
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R92 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r92.json").read_text())["kernel"])
ON = dataclasses.replace(R92, nucleon_kstar_pairs=True)
KSTARS = frozenset(KSTAR_OF_KAON.values())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
#: The classes the option adds at the R92 kernel, family by family.
ADDED = {
    "baryonic_strange": ("(2212,-323)", "(2112,-313)"),
    "baryonic_strange_qm1": ("(2112,-323)",),
    "baryonic_strange_q1": ("(2212,-313)",),
    "anti_baryonic_strange": ("(-2212,323)", "(-2112,313)"),
    "anti_baryonic_strange_qm1": ("(-2112,323)",),
    "anti_baryonic_strange_q1": ("(-2212,313)",),
}


@pytest.fixture(scope="module")
def kernels():
    return {"off": load_kernel(R92, NESTED_ROOT), "on": load_kernel(ON, NESTED_ROOT)}


def _root(family_id, mass, qn):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                        exchange_ancestry=("kstar_probe",), depth=1)


def test_off_is_the_committed_kernel(kernels):
    """Off, no family carries an (N, K*) class; on, the six strange
    baryonic families append theirs and the catalog and every other class
    stay."""
    cat_off, fams_off = kernels["off"]
    cat_on, fams_on = kernels["on"]
    assert [r.pdg_id for r in cat_on.records()] == [r.pdg_id for r in cat_off.records()]
    for f_off, f_on in zip(fams_off, fams_on, strict=True):
        assert f_off.family_id == f_on.family_id
        added = ADDED.get(str(f_off.family_id), ())
        assert tuple(f_on.allowed_decay_classes) == tuple(f_off.allowed_decay_classes) + added, f_off.family_id
        assert dataclasses.replace(f_on, allowed_decay_classes=f_off.allowed_decay_classes) == f_off
        assert not any(c in ADDED.get(str(f_off.family_id), ()) for c in f_off.allowed_decay_classes)
    assert R92.as_dict()["nucleon_kstar_pairs"] is False and ON.as_dict()["nucleon_kstar_pairs"] is True


def test_the_class_rule():
    """One (N, K*) class beside each (N, K) class of a strange baryonic
    family, the nucleon first, in class order and once; none at a
    non-strange or mesonic family."""
    family = SimpleNamespace(qn=(1, 0, -1), allowed_decay_classes=("(3122,111)", "(-321,2212)", "(2112,-311)",
                                                                  "(2212,-321)", "(2212,-323)"))
    assert nucleon_kstar_classes(family) == ("(2112,-313)",)
    family = SimpleNamespace(qn=(1, 0, -1), allowed_decay_classes=("(-321,2212)", "(2112,-311)"))
    assert nucleon_kstar_classes(family) == ("(2212,-323)", "(2112,-313)")
    for qn in ((1, 1, 0), (0, 0, -1), (0, 0, 0)):
        family = SimpleNamespace(qn=qn, allowed_decay_classes=("(2212,-321)", "(321,-321)"))
        assert nucleon_kstar_classes(family) == ()
    family = SimpleNamespace(qn=(-1, 0, 1), allowed_decay_classes=("(-2212,321)", "(-2112,311)", "(-3122,111)"))
    assert nucleon_kstar_classes(family) == ("(-2212,323)", "(-2112,313)")


def test_the_channels(kernels):
    """The class opens as a direct pair at a floor state and as
    continuation channels above it, with the charge partners of the
    isospin factor; a non-strange baryonic vertex keeps its channels."""
    catalog, fams = kernels["on"]
    family = {str(f.family_id): f for f in fams}
    for family_id, mass in (("baryonic_strange", 2.1), ("baryonic_strange_qm1", 2.1), ("anti_baryonic_strange", 2.1)):
        chans = enumerate_channels(_root(family_id, mass, family[family_id].qn), catalog, fams)
        kstar = [c for c in chans if any(d in {f"catalog:{k}" for k in KSTARS} for d in c.daughters)]
        assert kstar and all(c.kind != KIND_CONTINUUM for c in kstar), family_id
        share = math.fsum(c.weight for c in kstar) / math.fsum(c.weight for c in chans)
        assert 0.1 < share < 0.6, (family_id, share)
    chans = enumerate_channels(_root("baryonic_strange", 3.5, (1, 0, -1)), catalog, fams)
    cont = [c for c in chans if c.kind == KIND_CONTINUUM]
    assert any(set(c.daughters) & {"catalog:-323", "catalog:-313"} for c in cont)
    _off_cat, off_fams = kernels["off"]
    off = enumerate_channels(_root("baryonic_np", 3.0, (1, 1, 0)), catalog, off_fams)
    on = enumerate_channels(_root("baryonic_np", 3.0, (1, 1, 0)), catalog, fams)
    assert [(c.daughters, c.weight) for c in on] == [(c.daughters, c.weight) for c in off]


def test_the_refusals(kernels):
    catalog, fams = kernels["off"]
    bare = SimpleNamespace(records=lambda: [r for r in catalog.records() if int(r.pdg_id) not in KSTARS])
    with pytest.raises(ValueError, match="lacks the K"):
        nucleon_kstar_families(fams, bare)
    with pytest.raises(ValueError, match="no strange baryonic family"):
        nucleon_kstar_families([f for f in fams if int(f.qn[2]) == 0 or int(f.qn[0]) == 0], catalog)
    for value in ("yes", 1, None):
        with pytest.raises(ValueError, match="nucleon_kstar_pairs"):
            KernelConfig(nucleon_kstar_pairs=value)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_events_conserve_charge(kernels, cls):
    """Events under the option conserve the charge and baryon number of the
    p+p state, and the K*(892) never reaches the final state."""
    catalog, fams = kernels["on"]
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.05, "high_mass_baryonic_continuum": 0.95} if cls == "nd"
               else {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0})
    params = TopologyNominalParams(epsilon=0.5, b_sd=6.9, epsilon_nd=0.24, remnant_weights=weights, nd_central=False,
                                   exact_transfer_range=True, line_thresholds=True)
    for seed in range(8):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 43, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology(cls, params, catalog, fams, **rngs)
        qns = [tuple(int(v) for v in h.qn) if h.pdg_id is None else
               (catalog._daughter_quantum_numbers(h.pdg_id).B, catalog._daughter_quantum_numbers(h.pdg_id).Q)
               for h in ev.final_state]
        assert sum(q[1] for q in qns) == 2 and sum(q[0] for q in qns) == 2
        assert all(h.pdg_id not in KSTARS for h in ev.final_state)


def test_the_run_spec_option(tmp_path):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_nucleon_kstar_pairs", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r92.json").read_text())
    good = dict(base, kernel={**base["kernel"], "nucleon_kstar_pairs": True})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.KERNEL.nucleon_kstar_pairs is True
    _catalog, fams = engine._load_physics(engine.FLOOR_GRID[0])
    assert "(2212,-323)" in {str(f.family_id): f for f in fams}["baryonic_strange"].allowed_decay_classes


def test_the_r93_spec_is_r92_with_the_pairs_at_the_r92_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r92, r93 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r92", "r93"))
    assert r93["kernel"] == {**r92["kernel"], "nucleon_kstar_pairs": True}
    for block in ("topology", "fit", "diffractive_split", "isovector"):
        assert r93[block] == r92[block]
    assert r93["generation_0"]["run"] == "r92_light_scalars_2026-10-12" and r93["generation_0"]["fit"] == "free"
    assert [p["name"] for p in r93["parameters"]] == [p["name"] for p in r92["parameters"]]
    assert r93["seeds"]["master"] == r92["seeds"]["master"]
    assert r93["seeds"]["raw_first"] > r92["seeds"]["raw_first"] + 511
    assert r93["seeds"]["partition_base"] != r92["seeds"]["partition_base"]
    assert KernelConfig(**r93["kernel"]).nucleon_kstar_pairs is True
