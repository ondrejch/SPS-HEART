"""The density-integrated sibling weight of a continuation step (2026-10-06;
TASK-20261005-cascade-vertex-rules item 2, the code and physics review's
M-2): under ``KernelConfig.continuation_sibling_weight = "density"`` the
weight of a continuation channel is ``int rho_fam(m) q*^(2L+1) F_L^2 dm`` on
the density step's window with ``rho_fam`` normalized on the family support;
``"edge"`` and the absent option are the card-4 weight bit for bit; the
option needs the density mass law; the closed-window assertion of the density
step (L-4) raises."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

#: numpy 2 renamed trapz to trapezoid (the code reviews of 2026-11-04)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.resonance_widths import BLATT_WEISSKOPF_BARRIER, decay_momentum
from na49_mixing.spectral_density import _family_from_record

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True)


def _state(M, family_id="baryonic_np", qn=(1, 1, 0)):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=qn, quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("test",),
                          depth=0)


@pytest.fixture(scope="module")
def kernels():
    return {law: load_kernel(dataclasses.replace(BASE, continuation_sibling_weight=law), NESTED_ROOT)
            for law in (None, "edge", "density")}


def test_edge_is_the_card4_weight_bit_for_bit(kernels):
    for M in (2.4, 4.0, 9.0):
        a = C.enumerate_channels(_state(M), *kernels[None])
        b = C.enumerate_channels(_state(M), *kernels["edge"])
        assert [(c.daughters, c.kind, c.weight) for c in a] == [(c.daughters, c.kind, c.weight) for c in b]


def _independent_weight(fam, M, m_other, floor, edge):
    """The integral on a fine grid in m (numpy, no shared code path)."""
    t_h, ceiling, L = float(fam.t_h), C._continuation_ceiling(fam), int(fam.continuation_barrier_l or 0)
    low = max(floor, edge - 16.0 * t_h)
    m = np.linspace(low, edge, 40001)
    q = np.array([decay_momentum(M, x, m_other) for x in m])
    barrier = np.array([BLATT_WEISSKOPF_BARRIER(L, x) ** 2 for x in q]) if L else 1.0
    integrand = np.exp((m - ceiling) / t_h) * q ** (2 * L + 1) * barrier
    norm = t_h * (1.0 - math.exp((floor - ceiling) / t_h))
    return float(_trapezoid(integrand, m)) / norm


def test_the_density_weight(kernels):
    catalog, fams = kernels["density"]
    for M in (4.0, 8.0):
        chans = [c for c in C.enumerate_channels(_state(M), catalog, fams) if c.kind == C.KIND_CONTINUUM]
        ratios = []
        for ch in chans:
            fam_id = next(d for d in ch.daughters if not d.startswith(C.CATALOG_FAMILY_PREFIX))
            pdg = int(next(d for d in ch.daughters if d.startswith(C.CATALOG_FAMILY_PREFIX))[len(C.CATALOG_FAMILY_PREFIX):])
            fam = C._family_by_id(fams, fam_id)
            m_other = C._daughter_pole_mass(catalog, pdg)
            edge = min(C._continuation_ceiling(fam), M - m_other)
            floor = C._continuation_floor(fam, catalog)
            got = C._density_integrated_phase_space(fam, M, m_other, floor, edge)
            assert got == pytest.approx(_independent_weight(fam, M, m_other, floor, edge), rel=2e-3)
            ratios.append(ch.weight / got)
        # the vertex shares are the integrals times spin and isospin factors, normalized
        assert all(math.isfinite(r) and r > 0.0 for r in ratios)
        # the light siblings dominate: the pion share of the continuation set at 4 and 8 GeV
        def sibling(ch):
            return next(d for d in ch.daughters if d.startswith(C.CATALOG_FAMILY_PREFIX))
        pions = sum(c.weight for c in chans if sibling(c).endswith((":211", ":-211", ":111")))
        assert 0.45 < pions / sum(c.weight for c in chans) < 0.6


def test_the_option_needs_the_density_law():
    with pytest.raises(ValueError):
        KernelConfig(continuation_sibling_weight="density")
    with pytest.raises(ValueError):
        dataclasses.replace(R74, continuation_sibling_weight="flat")
    assert dataclasses.replace(R74, continuation_sibling_weight="density").as_dict()["continuation_sibling_weight"] == "density"


def test_the_closed_window_raises(kernels):
    catalog, fams = kernels[None]
    fam = C._family_by_id(fams, "baryonic_np")
    with pytest.raises(ValueError):
        C._density_daughter_mass(fam, 3.0, 0.14, 2.0, 1.9, 0.5)
    assert C._density_daughter_mass(fam, 3.0, 0.14, 2.0, 2.0, 0.5) == 2.0


def test_the_loader_reads_and_validates_the_option():
    import json
    record = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json").read_text())["families"][0]
    assert _family_from_record(record).continuation_sibling_weight is None
    for law in ("edge", "density"):
        assert _family_from_record({**record, "continuation_sibling_weight": law}).continuation_sibling_weight == law
    for bad in ("flat", None, 1):
        with pytest.raises(ValueError):
            _family_from_record({**record, "continuation_sibling_weight": bad})
