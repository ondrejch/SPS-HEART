"""Regressions for the 08rev physics/functionality bugs."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.constants import EMOMEGA
from na49_mixing.cuts import EventCut, PairCut
from na49_mixing.decays import DecayMode, ParticleDecays
from na49_mixing.event import Event
from na49_mixing.histogram import Histogram1D
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_mass
from na49_mixing.parallel import GenerationConfig, _build_decay_chain
from na49_mixing.reaction import Reaction
from na49_mixing._cli import generate_main


def _kallen(x: float, y: float, z: float) -> float:
    return x * x + y * y + z * z - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _pstar_weight(s: float, m0: float, m1: float, m2: float, m3: float) -> float:
    """The flat Dalitz projection on s = m23^2: |p1| |p2*| / sqrt(s), the
    m12^2 band width 4 |p1| |p2*| at fixed s up to constants (dGamma/dm23 is
    |p1| |p2*|, and ds = 2 m23 dm23)."""
    a = _kallen(m0 * m0, m1 * m1, s)
    b = _kallen(s, m2 * m2, m3 * m3)
    if a <= 0.0 or b <= 0.0 or s <= 0.0:
        return 0.0
    return (math.sqrt(a) / (2.0 * m0)) * (math.sqrt(b) / (2.0 * math.sqrt(s))) / math.sqrt(s)


def _dalitz_inside(s12: float, s23: float, m0: float, m1: float, m2: float, m3: float) -> bool:
    """Whether (m12^2, m23^2) lies inside the Dalitz boundary (PDG
    kinematics: the m12^2 range at fixed m23^2 from the 23 rest frame)."""
    m23 = math.sqrt(s23)
    e2 = (s23 - m3 * m3 + m2 * m2) / (2.0 * m23)
    e1 = (m0 * m0 - s23 - m1 * m1) / (2.0 * m23)
    if e2 < m2 or e1 < m1:
        return False
    p2, p1 = math.sqrt(e2 * e2 - m2 * m2), math.sqrt(e1 * e1 - m1 * m1)
    centre = (e1 + e2) ** 2
    return centre - (p1 + p2) ** 2 <= s12 <= centre - (p1 - p2) ** 2


class TestThreeBodyPhaseSpace:
    def test_compmass_matches_pstar_product(self):
        m0, m1, m2, m3 = EMOMEGA, get_mass(211), get_mass(-211), get_mass(111)
        s_lo = (m2 + m3) ** 2
        s_hi = (m0 - m1) ** 2
        params = (m0, m1, m2, m3)
        ratios = []
        for s in np.linspace(s_lo, s_hi, 21)[1:-1]:
            w = Event._compmass(float(s), params)
            p = _pstar_weight(float(s), m0, m1, m2, m3)
            assert w > 0.0 and p > 0.0
            ratios.append(w / p)
        assert max(ratios) / min(ratios) == pytest.approx(1.0, rel=1e-9)

    def test_compmass_matches_a_uniform_dalitz_plot(self):
        """Independent of any formula: points uniform in (m12^2, m23^2)
        inside the boundary, by rejection, give the m23^2 marginal that
        the sampler weight must reproduce."""
        m0, m1, m2, m3 = EMOMEGA, get_mass(211), get_mass(-211), get_mass(111)
        s_lo, s_hi = (m2 + m3) ** 2, (m0 - m1) ** 2
        t_lo, t_hi = (m1 + m2) ** 2, (m0 - m3) ** 2
        rng = np.random.default_rng(20261007)
        s12 = rng.uniform(t_lo, t_hi, 400_000)
        s23 = rng.uniform(s_lo, s_hi, 400_000)
        keep = np.array([_dalitz_inside(a, b, m0, m1, m2, m3) for a, b in zip(s12, s23)])
        counts, edges = np.histogram(s23[keep], bins=20, range=(s_lo, s_hi))
        # bin-integrated expectation (the density has square-root edges)
        w = np.array([np.mean([Event._compmass(float(s), (m0, m1, m2, m3)) for s in np.linspace(a, b, 102)[1:-1]])
                      for a, b in zip(edges[:-1], edges[1:])])
        expected = w / w.sum() * counts.sum()
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        assert chi2 < 45.0  # 20 bins; the 1/sqrt(s) weight of the earlier port gives about 1.1e4 here

    def test_omega_m23_mean_matches_constant_matrix_element(self):
        rng = np.random.default_rng(20260825)
        m0 = EMOMEGA
        m1 = get_mass(211)
        m2 = get_mass(-211)
        m3 = get_mass(111)
        s_lo = (m2 + m3) ** 2
        s_hi = (m0 - m1) ** 2
        ss = np.linspace(s_lo, s_hi, 2001)
        w = np.array([_pstar_weight(float(s), m0, m1, m2, m3) for s in ss])
        expected = float(np.average(np.sqrt(ss), weights=w))  # <m23>, the sampled mean

        reaction = Reaction(158.0)
        table = ParticleDecays(223, rng=rng)
        table.add_mode(1.0, DecayMode(3, [211, -211, 111]))
        masses = []
        e_plus = []
        e_minus = []
        n_ok = 0
        for _ in range(4000):
            ev = Event(rng=rng)
            ev._decay_chain = {223: table}
            parent = Particle(0.0, 0.0, 0.0, 0.0, m0, 223, reaction)
            ev.add_particle(parent)
            ev.decay()
            parts = {p.get_id(): p for p in ev.particles()}
            if not {211, -211, 111} <= set(parts):
                continue
            n_ok += 1
            masses.append(parts[-211].get_inv_mass(parts[111]))
            e_plus.append(parts[211].get_e())
            e_minus.append(parts[-211].get_e())
        assert n_ok > 3000
        assert float(np.mean(masses)) == pytest.approx(expected, abs=0.008)
        # Sequential 1+(23) keeps π⁺ as the bachelor, so ⟨E(π⁺)⟩ and
        # ⟨E(π⁻)⟩ need not match; both must still sit in the physical
        # rest-frame window.
        assert 0.15 < float(np.mean(e_plus)) < 0.40
        assert 0.15 < float(np.mean(e_minus)) < 0.40


def _npi_increment(decays) -> float:
    prev = 0.0
    total = 0.0
    nucleons = {-2212, -2112, 2212, 2112}
    pions = {211, -211, 111}
    for prob, mode in zip(decays._probabilities, decays._modes):
        inc = float(prob) - prev
        prev = float(prob)
        if inc <= 0.0:
            continue
        d = list(mode.daughters)
        if len(d) == 2 and d[0] in nucleons and d[1] in pions:
            total += inc
    return total


class TestAntiNstarOverrides:
    def test_n1440_override_moves_matter_and_anti(self):
        rng = np.random.default_rng(1)
        cfg0 = GenerationConfig(
            gen_file="x", n_events=1, seed=1,
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=True,
        )
        cfg1 = GenerationConfig(
            gen_file="x", n_events=1, seed=1,
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=True,
            soft_decay_overrides=(("n1440_br_npi", 0.55), ("n1440_br_deltapi", 0.27)),
        )
        c0 = _build_decay_chain(cfg0, rng)
        c1 = _build_decay_chain(cfg1, np.random.default_rng(1))
        m0 = _npi_increment(c0[12112])
        m1 = _npi_increment(c1[12112])
        a0 = _npi_increment(c0[-12112])
        a1 = _npi_increment(c1[-12112])
        assert m0 == pytest.approx(a0, abs=1e-12)
        assert m1 == pytest.approx(a1, abs=1e-12)
        assert m1 != pytest.approx(m0, abs=1e-4)
        assert a1 != pytest.approx(a0, abs=1e-4)

    def test_n1720_override_moves_neutral_anti(self):
        rng = np.random.default_rng(2)
        cfg0 = GenerationConfig(
            gen_file="x", n_events=1, seed=2,
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=True,
        )
        cfg1 = GenerationConfig(
            gen_file="x", n_events=1, seed=2,
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=True,
            soft_decay_overrides=(("n1720_br_npi", 0.40), ("n1720_br_deltapi", 0.20)),
        )
        c0 = _build_decay_chain(cfg0, rng)
        c1 = _build_decay_chain(cfg1, np.random.default_rng(2))
        assert _npi_increment(c1[-212114]) != pytest.approx(
            _npi_increment(c0[-212114]), abs=1e-4
        )
        assert _npi_increment(c1[212114]) == pytest.approx(
            _npi_increment(c1[-212114]), abs=1e-12
        )


class TestHistogramFillParity:
    def test_fill_and_fill_array_share_edges(self):
        h_s = Histogram1D(4, 0.0, 1.0)
        h_v = Histogram1D(4, 0.0, 1.0)
        values = np.array([-0.2, 0.0, 0.25, 1.0, 1.2])
        for v in values:
            h_s.fill(v)
        h_v.fill_array(values)
        np.testing.assert_allclose(h_s._contents, h_v._contents)
        assert h_s.get_bin_content(0) == pytest.approx(1.0)  # underflow
        assert h_s.get_bin_content(5) == pytest.approx(2.0)  # 1.0 and 1.2 overflow


class TestEventAndPairCuts:
    def test_mult_cut_counts_all_tracks_without_particle_filter(self):
        ev = Event()
        for _ in range(3):
            ev.add_particle(Particle(0.1, 0.0, 1.0, 1.0, 0.14, 211))
        cut = EventCut()
        cut.set_mult_cut(10, 100)
        assert cut.check_event(ev) is False
        cut.set_mult_cut(1, 5)
        assert cut.check_event(ev) is True

    def test_pair_angle_cut_rejects_zero_momentum(self):
        cut = PairCut()
        cut.set_opening_angle_cut(0.0, 0.1)
        a = Particle(0.0, 0.0, 0.0, 1.0, 0.14, 211)
        b = Particle(1.0, 0.0, 0.0, -1.0, 0.14, -211)
        assert cut.check_pair(a, b) is False
        cut2 = PairCut()
        cut2.set_cos_theta_cut(0.9, 1.0)
        assert cut2.check_pair(a, b) is False


class TestCliTrackSchema:
    def test_cli_writes_ancestry_arrays(self, tmp_path):
        gen = os.path.join(
            os.path.dirname(__file__),
            "..",
            "na49_mixing",
            "generators",
            "michal",
            "deltas-to-pion",
            "delta++.gen",
        )
        out = tmp_path / "tracks.npz"
        rc = generate_main(
            [
                "--genfile", gen,
                "--outfile", str(out),
                "--nevents", "5",
                "--seed", "1",
                "--use-soft-decay-factory",
            ]
        )
        assert rc == 0
        data = np.load(out)
        for key in ("pdg", "xf", "pt", "source_side", "weak_feed", "gen_depth", "n_wounds"):
            assert key in data.files
        assert data["pdg"].shape == data["weak_feed"].shape
        meta = json.loads(str(data["meta_json"]))
        assert meta["weak_feed_policy"] == "inclusive_weak"
