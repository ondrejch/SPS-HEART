"""The S = -2 excited baryonic continuum (``KernelConfig.xi_tower``, 2026-10-30)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.xi_tower import XI_TOWER_FAMILIES, xi_tower_families

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


def _r128_kernel(**options):
    return KernelConfig(**{**json.loads((RUNS / "r128.json").read_text())["kernel"], **options})


def test_the_option_is_checked():
    for bad in (1, "yes"):
        with pytest.raises(ValueError):
            _r128_kernel(xi_tower=bad)
    with pytest.raises(ValueError):
        _r128_kernel(xi_tower=True, xi_baryons=False)
    with pytest.raises(ValueError):
        _r128_kernel(xi_tower=True, conjugate_towers=False, baryon_resonance_pairs=False)
    assert _r128_kernel().xi_tower is None and _r128_kernel(xi_tower=True).as_dict()["xi_tower"] is True


def test_the_tower_is_an_s_minus_two_doublet_on_the_strange_family():
    """Four families after the committed ones, none a root: the doublet at (1, 0, -2) and (1, -1, -2) and its
    conjugates, each class conserving its family's quantum numbers, the spectral law and the options of the strange
    baryonic family; without the option the kernel is unchanged."""
    cat0, fams0 = load_kernel(_r128_kernel(), NESTED_ROOT)
    cat, fams = load_kernel(_r128_kernel(xi_tower=True), NESTED_ROOT)
    assert [f.family_id for f in fams[:len(fams0)]] == [f.family_id for f in fams0] and fams[:len(fams0)] == fams0
    new = {str(f.family_id): f for f in fams[len(fams0):]}
    assert tuple(new) == ("baryonic_xi_q0", "anti_baryonic_xi_q0", "baryonic_xi_qm1", "anti_baryonic_xi_qm1")
    assert set(new) == set(XI_TOWER_FAMILIES)
    base = next(f for f in fams0 if str(f.family_id) == "baryonic_strange")
    for fid, f in new.items():
        assert not f.active and f.isospin == 0.5
        assert f == dataclasses.replace(base, family_id=fid, qn=f.qn, allowed_decay_classes=f.allowed_decay_classes,
                                        active=False, isospin=0.5)
        sign = -1 if fid.startswith("anti_") else 1
        assert tuple(f.qn) == tuple(sign * v for v in ((1, 0, -2) if fid.endswith("q0") else (1, -1, -2)))
        assert len(f.allowed_decay_classes) == 9
        for cls in f.allowed_decay_classes:
            ids = [int(t) for t in cls.strip("()").split(",")]
            q = sum((cat._daughter_quantum_numbers(pid) for pid in ids[1:]), cat._daughter_quantum_numbers(ids[0]))
            assert (q.B, q.Q, q.S) == tuple(f.qn), (fid, cls)
    with pytest.raises(ValueError):
        xi_tower_families(cat, fams)


def test_the_r130_spec_is_r129_with_the_tower():
    import importlib.util
    r129, r130 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r129", "r130"))
    assert {k: v for k, v in r130["kernel"].items() if k != "xi_tower"} == r129["kernel"] and r130["kernel"]["xi_tower"] is True
    for block in ("topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r130[block] == r129[block], block
    assert r130["generation_0"]["run"] == "r129_pdg_elastic_profile_2026-10-30"
    assert r130["seeds"]["raw_first"] > r129["seeds"]["raw_first"] + 511
    spec = importlib.util.spec_from_file_location("run_engine_r130_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e.configure(RUNS / "r130.json")
    assert e.KERNEL.xi_tower is True
