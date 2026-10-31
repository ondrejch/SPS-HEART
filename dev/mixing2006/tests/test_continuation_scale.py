"""The continuation scale of the density step (2026-10-07; card
TASK-20261005-hadron-count-vs-mass): ``KernelConfig.continuation_t_h_alpha``
sets t_h = max(t_h, alpha * M_parent) in the daughter-mass draw and in the
density-integrated sibling weight alike; the absent option is the committed
kernel bit for bit."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.spectral_density import _family_from_record

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True, continuation_sibling_weight="density")


def _state(M, family_id, qn):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=qn, quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("test",), depth=0)


@pytest.fixture(scope="module")
def kernels():
    return {"plain": load_kernel(BASE, NESTED_ROOT),
            "alpha": load_kernel(dataclasses.replace(BASE, continuation_t_h_alpha=0.25), NESTED_ROOT)}


def test_the_option_sets_every_family_and_validates(kernels):
    _c, fams = kernels["alpha"]
    assert all(f.continuation_t_h_alpha == 0.25 for f in fams)
    assert all(getattr(f, "continuation_t_h_alpha", None) is None for f in kernels["plain"][1])
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, continuation_t_h_alpha=-0.1)
    with pytest.raises(ValueError):
        KernelConfig(continuation_t_h_alpha=0.25)  # needs the density law
    assert dataclasses.replace(BASE, continuation_t_h_alpha=0.25).as_dict()["continuation_t_h_alpha"] == 0.25
    record = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json").read_text())["families"][0]
    assert _family_from_record({**record, "continuation_t_h_alpha": 0.3}).continuation_t_h_alpha == 0.3
    with pytest.raises(ValueError):
        _family_from_record({**record, "continuation_t_h_alpha": "big"})


def test_the_draw_and_the_weight_use_one_scale(kernels):
    """At a heavy parent the scale is alpha * M; the draw and the weight
    equal those of the family with that t_h written in."""
    _c, fams = kernels["alpha"]
    fam = next(f for f in fams if f.family_id == "baryonic_np")
    M, m_sib, floor = 6.0, 0.13957, 1.7
    edge = M - m_sib
    scaled = dataclasses.replace(fam, continuation_t_h_alpha=None, t_h=0.25 * M)
    assert C._density_daughter_mass(fam, M, m_sib, floor, edge, 0.37) == C._density_daughter_mass(scaled, M, m_sib, floor, edge, 0.37)
    assert C._density_integrated_phase_space(fam, M, m_sib, floor, edge) == C._density_integrated_phase_space(
        scaled, M, m_sib, floor, edge)
    # near the floor the committed scale stays: alpha * M below t_h
    light = 0.5
    assert 0.25 * light < float(fam.t_h)
    plain = dataclasses.replace(fam, continuation_t_h_alpha=None)
    assert C._with_continuation_scale(fam, light) == fam
    assert C._density_daughter_mass(fam, 2.2, m_sib, floor, 2.2 - m_sib, 0.6) == C._density_daughter_mass(
        dataclasses.replace(plain, t_h=max(float(fam.t_h), 0.25 * 2.2)), 2.2, m_sib, floor, 2.2 - m_sib, 0.6)


def test_a_larger_scale_moves_the_emission_from_pions_to_resonances(kernels):
    """The density weight carries the Boltzmann-like factor exp(-m_sib / t_h)
    between siblings; at a heavy parent the larger scale weakens it, and the
    many high-spin resonances (weighted by 2J+1) take the emission over from
    the pions (2026-10-07: at 8 GeV the pion share falls from 0.3 to below
    0.05)."""
    def pion_share(kernel):
        cat, fams = kernel
        cont = [c for c in C.enumerate_channels(_state(8.0, "mesonic_neutral", (0, 0, 0)), cat, fams)
                if c.kind == C.KIND_CONTINUUM]
        tot = sum(c.weight for c in cont)
        return sum(c.weight for c in cont if any(d in ("catalog:211", "catalog:-211", "catalog:111") for d in c.daughters)) / tot
    plain, scaled = pion_share(kernels["plain"]), pion_share(kernels["alpha"])
    assert plain > 0.2 and scaled < 0.1
