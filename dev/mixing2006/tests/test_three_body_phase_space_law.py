"""The phase-space law of the three-body decay vertex (2026-10-06;
TASK-20261005-three-body-phase-space, the code and physics review's M-3):
``three_body_phase_space(..., law="phase_space")`` draws the (b, c) mass from
the flat three-body phase-space density, so the Dalitz plot of omega and eta
-> pi+ pi- pi0 is flat and each daughter's mean momentum matches flat phase
space; the draw budget stays four uniforms; the default law is the Round 2
vertex bit for bit; the run selects the law through the catalog
(``KernelConfig.three_body_phase_space``) without touching the payload."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.resonance_decay import ResonanceDecayer, three_body_phase_space
from na49_mixing.resonance_widths import decay_momentum

NESTED_ROOT = Path(__file__).resolve().parents[1]
M_PIC, M_PI0 = 0.13957039, 0.1349768
OMEGA, ETA = 0.78266, 0.547862


def _round2_vertex(M, mb, mc, md, rng):
    """The Round 2 vertex as committed before the law option (flat in M_bc^2)."""
    lo, hi = (mb + mc) ** 2, (M - md) ** 2
    m_bc = math.sqrt(lo + float(rng.random()) * (hi - lo))
    u_cos, u_phi, u_d = float(rng.random()), float(rng.random()), float(rng.random())
    cos_t = 2.0 * u_cos - 1.0
    phi = 2.0 * math.pi * u_phi
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    return m_bc, (sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t), 2.0 * u_d - 1.0


def _dalitz(points):
    pb, pc, pd = (np.array([p[k] for p in points]) for k in range(3))

    def m2(a, b):
        s = a + b
        return s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2

    return m2(pb, pc), m2(pc, pd), pb, pc, pd


def _flat_reference(M, mb, mc, md, n, rng):
    """Independent flat phase space: (m12^2, m23^2) uniform in the bounding
    box, accepted inside the physical Dalitz boundary (PDG kinematics)."""
    x = rng.uniform((mb + mc) ** 2, (M - md) ** 2, n)
    y = rng.uniform((mc + md) ** 2, (M - mb) ** 2, n)
    m12 = np.sqrt(x)
    e2 = (x - mb * mb + mc * mc) / (2.0 * m12)
    e3 = (M * M - x - md * md) / (2.0 * m12)
    p2 = np.sqrt(np.maximum(e2 * e2 - mc * mc, 0.0))
    p3 = np.sqrt(np.maximum(e3 * e3 - md * md, 0.0))
    lo = (e2 + e3) ** 2 - (p2 + p3) ** 2
    hi = (e2 + e3) ** 2 - (p2 - p3) ** 2
    keep = (y >= lo) & (y <= hi)
    return x[keep], y[keep]


def _flat_momenta(M, mb, mc, md, x, y):
    """|p| of each daughter in the parent frame at the Dalitz point (x = m12^2, y = m23^2)."""
    z = M * M + mb * mb + mc * mc + md * md - x - y   # m13^2
    e_b = (M * M + mb * mb - y) / (2.0 * M)
    e_c = (M * M + mc * mc - z) / (2.0 * M)
    e_d = (M * M + md * md - x) / (2.0 * M)
    return [np.sqrt(np.maximum(e * e - m * m, 0.0)) for e, m in ((e_b, mb), (e_c, mc), (e_d, md))]


@pytest.mark.parametrize("M", [OMEGA, ETA], ids=["omega", "eta"])
def test_the_dalitz_plot_is_flat(M):
    mb, mc, md = M_PIC, M_PIC, M_PI0
    rng = np.random.default_rng(20261006)
    n = 160_000
    ref_x, ref_y = _flat_reference(M, mb, mc, md, 8_000_000, np.random.default_rng(7))
    xe = np.linspace((mb + mc) ** 2, (M - md) ** 2, 11)
    ye = np.linspace((mc + md) ** 2, (M - mb) ** 2, 11)
    ref, _, _ = np.histogram2d(ref_x, ref_y, bins=(xe, ye))
    expected_share = ref / ref.sum()
    chi2 = {}
    for law in ("phase_space", "flat_m2"):
        points = [three_body_phase_space(M, mb, mc, md, rng, law=law) for _ in range(n)]
        x, y, pb, pc, pd = _dalitz(points)
        counts, _, _ = np.histogram2d(x, y, bins=(xe, ye))
        expect = n * expected_share
        cells = expect > 20.0
        chi2[law] = (float(np.sum((counts[cells] - expect[cells]) ** 2 / expect[cells])), int(cells.sum()))
        if law == "phase_space":
            # the mean momentum of each daughter against flat phase space
            flat = _flat_momenta(M, mb, mc, md, ref_x, ref_y)
            for p4, ref_p in zip((pb, pc, pd), flat):
                p = np.sqrt(p4[:, 1] ** 2 + p4[:, 2] ** 2 + p4[:, 3] ** 2)
                assert p.mean() == pytest.approx(ref_p.mean(), abs=4.0 * p.std() / math.sqrt(n))
    value, dof = chi2["phase_space"]
    assert value / dof < 1.5
    assert chi2["flat_m2"][0] / chi2["flat_m2"][1] > 10.0  # the Round 2 draw is far from flat


def test_the_default_law_is_the_round2_vertex_bit_for_bit():
    mb, mc, md = M_PIC, M_PIC, M_PI0
    for seed in range(5):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        points = three_body_phase_space(OMEGA, mb, mc, md, a)
        m_bc, direction, cos_d = _round2_vertex(OMEGA, mb, mc, md, b)
        pair = points[0] + points[1]
        assert math.sqrt(pair[0] ** 2 - pair[1] ** 2 - pair[2] ** 2 - pair[3] ** 2) == pytest.approx(m_bc, abs=1e-12)
        assert points[2][3] / math.sqrt(points[2][1] ** 2 + points[2][3] ** 2) == pytest.approx(cos_d, abs=1e-12)
        assert a.random() == b.random()


@pytest.mark.parametrize("law", ["flat_m2", "phase_space"])
def test_the_draw_budget_is_four_uniforms(law):
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    three_body_phase_space(OMEGA, M_PIC, M_PIC, M_PI0, a, law=law)
    b.random(4)
    assert a.random() == b.random()


def test_closure_shell_and_a_massless_daughter():
    rng = np.random.default_rng(5)
    for masses in ((M_PIC, M_PIC, M_PI0), (M_PIC, M_PIC, 0.0), (0.938272, 0.13957039, 0.4936770)):
        M = 0.547862 if masses[2] == 0.0 else (OMEGA if masses[0] < 0.2 else 1.72)
        for _ in range(200):
            points = three_body_phase_space(M, *masses, rng, law="phase_space")
            total = sum(points)
            assert total == pytest.approx([M, 0.0, 0.0, 0.0], abs=1e-9)
            for p4, m in zip(points, masses):
                assert p4[0] ** 2 - p4[1] ** 2 - p4[2] ** 2 - p4[3] ** 2 == pytest.approx(m * m, abs=1e-9)
    with pytest.raises(ValueError):
        three_body_phase_space(OMEGA, M_PIC, M_PIC, M_PI0, rng, law="dalitz")


def test_the_pair_mass_density():
    """The drawn M_bc follows q*(M_bc) q(M; M_bc) on [mb + mc, M - md]."""
    mb, mc, md = M_PIC, M_PIC, M_PI0
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(30_000):
        pb, pc, _pd = three_body_phase_space(OMEGA, mb, mc, md, rng, law="phase_space")
        pair = pb + pc
        draws.append(math.sqrt(pair[0] ** 2 - pair[1] ** 2 - pair[2] ** 2 - pair[3] ** 2))
    grid = np.linspace(mb + mc, OMEGA - md, 20001)
    density = np.array([decay_momentum(m, mb, mc) * decay_momentum(OMEGA, m, md) for m in grid])
    mean = np.trapz(grid * density, grid) / np.trapz(density, grid)
    draws = np.asarray(draws)
    assert draws.mean() == pytest.approx(mean, abs=4.0 * draws.std() / math.sqrt(draws.size))


def test_the_kernel_option_sets_the_catalog_law():
    committed = HadronCatalog.from_json(NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json")
    assert committed.three_body_law == "flat_m2"
    catalog, _fams = load_kernel(dataclasses.replace(R74, three_body_phase_space=True), NESTED_ROOT)
    assert catalog.three_body_law == "phase_space"
    assert catalog.to_dict() == committed.to_dict()
    assert ResonanceDecayer(catalog, 0, 0, 0)._three_body_law == "phase_space"
    assert load_kernel(R74, NESTED_ROOT)[0].three_body_law == "flat_m2"
    assert R74.as_dict()["three_body_phase_space"] is False
    with pytest.raises(ValueError):
        committed.with_three_body_law("dalitz")
