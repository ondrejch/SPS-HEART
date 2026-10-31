"""The isospin-resolved continuation daughters and the I = 1 neutral mesonic
family (2026-10-06; card TASK-20261005-r8-isovector-mesonic-completion, steps
2 and 3): under ``continuation_daughters = "isospin"`` a continuation step
enumerates every family of the daughter's (B, Q, S) with its own isospin
share; the I = 1 neutral family lets an I = 0 neutral state emit a rho0,
which isospin forbids next to an I = 0 daughter; the committed first-family
rule is the default bit for bit."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.continuum_isospin import ISOVECTOR_NEUTRAL_MESONIC_ID
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.spectral_density import _family_from_record

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True)


def _state(M, family_id, qn):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=qn, quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("test",),
                          depth=0)


@pytest.fixture(scope="module")
def kernels():
    return {"default": load_kernel(BASE, NESTED_ROOT),
            "first": load_kernel(dataclasses.replace(BASE, continuation_daughters="first"), NESTED_ROOT),
            "isospin": load_kernel(dataclasses.replace(BASE, continuation_daughters="isospin", isovector_neutral_mesons=True),
                                   NESTED_ROOT)}


def _sibling(ch):
    return next(d for d in ch.daughters if d.startswith(C.CATALOG_FAMILY_PREFIX))


def _family(ch):
    return next(d for d in ch.daughters if not d.startswith(C.CATALOG_FAMILY_PREFIX))


@pytest.mark.parametrize("fid, qn", [("mesonic_neutral", (0, 0, 0)), ("baryonic_np", (1, 1, 0)), ("mesonic_charged", (0, 1, 0))])
def test_first_is_the_committed_rule_bit_for_bit(kernels, fid, qn):
    for M in (2.6, 4.0, 9.0):
        a = C.enumerate_channels(_state(M, fid, qn), *kernels["default"])
        b = C.enumerate_channels(_state(M, fid, qn), *kernels["first"])
        assert [(c.daughters, c.kind, c.weight) for c in a] == [(c.daughters, c.kind, c.weight) for c in b]


def test_an_isoscalar_state_emits_rho0_only_with_the_isovector_daughter(kernels):
    state = _state(4.0, "mesonic_neutral", (0, 0, 0))
    first = [c for c in C.enumerate_channels(state, *kernels["default"]) if c.kind == C.KIND_CONTINUUM]
    iso = [c for c in C.enumerate_channels(state, *kernels["isospin"]) if c.kind == C.KIND_CONTINUUM]
    assert not any(_sibling(c) == "catalog:113" for c in first)
    rho0 = [c for c in iso if _sibling(c) == "catalog:113"]
    assert rho0 and all(_family(c) == ISOVECTOR_NEUTRAL_MESONIC_ID for c in rho0)
    # an omega next to the isovector daughter is closed for an isoscalar parent
    assert not any(_sibling(c) == "catalog:223" and _family(c) == ISOVECTOR_NEUTRAL_MESONIC_ID for c in iso)
    assert sum(c.weight for c in C.enumerate_channels(state, *kernels["isospin"])) == pytest.approx(1.0)


def test_the_delta_like_families_become_daughters(kernels):
    iso = C.enumerate_channels(_state(4.0, "baryonic_np", (1, 1, 0)), *kernels["isospin"])
    assert any(_family(c) == "baryonic_np_delta_q1" for c in iso if c.kind == C.KIND_CONTINUUM)
    first = C.enumerate_channels(_state(4.0, "baryonic_np", (1, 1, 0)), *kernels["default"])
    assert not any(_family(c) == "baryonic_np_delta_q1" for c in first if c.kind == C.KIND_CONTINUUM)


def test_the_isovector_neutral_family(kernels):
    _catalog, fams = kernels["isospin"]
    fam = next(f for f in fams if f.family_id == ISOVECTOR_NEUTRAL_MESONIC_ID)
    src = next(f for f in fams if f.family_id == "mesonic_neutral")
    assert fam.qn == src.qn and float(fam.isospin) == 1.0 and fam.allowed_decay_classes == src.allowed_decay_classes
    # a daughter only: the sampled root surfaces keep the one neutral family
    from na49_mixing.spectral_density import active_families, continuum_density
    assert fam.active is False
    assert [f.family_id for f in active_families(fams, (0, 0, 0))] == ["mesonic_neutral"]
    _c, fams_off = kernels["default"]
    assert continuum_density((0, 0, 0), 3.0, fams) == continuum_density((0, 0, 0), 3.0, fams_off)
    # the isovector state's own open channels: pi0 pi0 and rho0 pi0 closed, rho+- pi-+ and omega pi0 open
    chans = C.enumerate_channels(_state(1.9, ISOVECTOR_NEUTRAL_MESONIC_ID, (0, 0, 0)), *kernels["isospin"])
    pairs = {tuple(sorted(c.daughters)) for c in chans if c.kind != C.KIND_CONTINUUM}
    assert ("catalog:111", "catalog:111") not in pairs and ("catalog:111", "catalog:113") not in pairs
    assert ("catalog:111", "catalog:223") in pairs


def test_the_options_validate():
    with pytest.raises(ValueError):
        KernelConfig(continuation_daughters="all")
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, isovector_neutral_mesons=True)
    assert dataclasses.replace(BASE, continuation_daughters="isospin").as_dict()["continuation_daughters"] == "isospin"
    record = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json").read_text())["families"][0]
    assert _family_from_record(record).continuation_daughters is None
    assert _family_from_record({**record, "continuation_daughters": "isospin"}).continuation_daughters == "isospin"
    with pytest.raises(ValueError):
        _family_from_record({**record, "continuation_daughters": "all"})
