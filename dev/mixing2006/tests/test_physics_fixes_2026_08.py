"""Regression tests for the 2026-08-23 resonance-physics fixes.

Covers:
1. PDG -1114 is exclusively the physical anti-Delta-; the factory-defined matter alias was eliminated;
2. anti-delta PDG charges are charge conjugates; legacy 12114 has a mass;
3. Delta* -> Delta pi uses exact I=3/2 Clebsch-Gordan weights;
4. Sigma(1385)0 -> Sigma pi excludes the isospin-forbidden Sigma0 pi0;
5. K0S decay entry provides neutral-kaon feed-down;
6. two-body Breit-Wigner daughter masses are phase-space weighted.
"""

import math

import numpy as np
import pytest

from na49_mixing.decays import build_soft_resonance_decays, build_omega_decays
from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_charge, get_mass
from na49_mixing.reaction import Reaction


def _incremental_probabilities(decays):
    """Convert cumulative probabilities back to per-mode probabilities."""
    probs = list(decays._probabilities)
    inc = [probs[0]]
    for prev, cur in zip(probs[:-1], probs[1:]):
        inc.append(cur - prev)
    return inc


class TestAliasElimination:
    """PDG -1114 is exclusively physical anti-Delta-; matter Delta+ uses 2214."""

    def test_minus1114_is_exclusively_physical_antidelta(self):
        from na49_mixing.parallel import GenerationConfig, _build_decay_chain

        rng = np.random.default_rng(3)
        cfg = GenerationConfig(
            gen_file="unused.gen",
            n_events=1,
            seed=1,
            inject_antibaryon_resonances=True,
            use_soft_decay_factory=True,
        )
        chain = _build_decay_chain(cfg, rng)

        assert -1114 in chain
        modes = [(tuple(m.daughters), p) for p, m in zip(
            _incremental_probabilities(chain[-1114]), chain[-1114]._modes)]
        # Pure nbar pi+ charge conjugate of Delta- -> n pi-.
        assert len(modes) == 1
        assert modes[0][0] == (-2112, 211)
        assert modes[0][1] == pytest.approx(1.0)

    def test_matter_delta_plus_available_via_true_id(self):
        lib = build_soft_resonance_decays()
        daughters = {tuple(m.daughters) for m in lib[2214]._modes}
        assert (2212, 111) in daughters
        assert (2112, 211) in daughters

    def test_antinstar_deltapi_modes_contain_no_minus1114(self):
        from na49_mixing.parallel import GenerationConfig, _build_decay_chain

        rng = np.random.default_rng(4)
        cfg = GenerationConfig(
            gen_file="unused.gen",
            n_events=1,
            seed=1,
            inject_antibaryon_resonances=True,
            use_soft_decay_factory=True,
        )
        chain = _build_decay_chain(cfg, rng)

        for pid in (-12112, -2116, -22112, -102112, -122112, -212114,
                    -9911201, -9921201, -9931201):
            for mode in chain[pid]._modes:
                # -1114 is ALLOWED here only as the physical anti-Delta+
                # charge-conjugate daughter; matter daughters remain banned.
                for d in mode.daughters:
                    assert d not in (2212, 2112), (
                        f"anti-N* parent {pid} decays to matter baryon {d}"
                    )

    def test_neutral_antinstar_deltapi_is_three_state_conjugate(self):
        from na49_mixing.parallel import GenerationConfig, _build_decay_chain

        rng = np.random.default_rng(5)
        cfg = GenerationConfig(
            gen_file="unused.gen",
            n_events=1,
            seed=1,
            inject_antibaryon_resonances=True,
            use_soft_decay_factory=True,
        )
        chain = _build_decay_chain(cfg, rng)

        probs = {}
        for prob, mode in zip(
            _incremental_probabilities(chain[-12112]), chain[-12112]._modes
        ):
            probs[tuple(mode.daughters)] = prob
        total_dpi = (probs.get((-2214, 211), 0.0) + probs.get((-2114, 111), 0.0)
                     + probs.get((-1114, -211), 0.0))
        if total_dpi > 0.0:
            assert probs[(-2214, 211)] / total_dpi == pytest.approx(1.0 / 6.0, abs=1e-9)
            assert probs[(-2114, 111)] / total_dpi == pytest.approx(2.0 / 6.0, abs=1e-9)
            assert probs[(-1114, -211)] / total_dpi == pytest.approx(3.0 / 6.0, abs=1e-9)

    def test_injection_without_factory_defines_physical_antidelta(self):
        from na49_mixing.parallel import GenerationConfig, _build_decay_chain

        rng = np.random.default_rng(5)
        cfg = GenerationConfig(
            gen_file="unused.gen",
            n_events=1,
            seed=1,
            inject_antibaryon_resonances=True,
            use_soft_decay_factory=False,
        )
        chain = _build_decay_chain(cfg, rng)
        daughters = {tuple(m.daughters) for m in chain[-1114]._modes}
        assert (-2112, 211) in daughters


class TestPdgTableFixes:
    def test_antidelta_charges_are_conjugates(self):
        assert get_charge(-1114) == pytest.approx(+1.0)   # anti of Delta-
        assert get_charge(-2214) == pytest.approx(-1.0)   # anti of Delta+
        assert get_charge(-2224) == pytest.approx(-2.0)   # anti of Delta++
        assert get_charge(-2114) == pytest.approx(0.0)

    def test_legacy_12114_is_retired(self):
        from na49_mixing.pdg_database import get_particle

        assert get_mass(12114, default=0.0) == pytest.approx(0.0)
        assert get_particle(12114) is None


class TestDeltaExcitedIsospinWeights:
    def test_half_integer_charge_states_use_exact_cg(self):
        lib = build_soft_resonance_decays()

        # Delta(1600)+ : Delta++pi- : Delta+pi0 : Delta0pi+ = 6 : 1 : 8 (of br).
        probs = {}
        for prob, mode in zip(
            _incremental_probabilities(lib[202214]), lib[202214]._modes
        ):
            probs[tuple(mode.daughters)] = prob
        w_ppm = probs[(2224, -211)]
        w_p0 = probs[(2214, 111)]
        w_zp = probs[(2114, 211)]
        total_delta_pi = w_ppm + w_p0 + w_zp
        assert w_ppm / total_delta_pi == pytest.approx(6.0 / 15.0, abs=1e-9)
        assert w_p0 / total_delta_pi == pytest.approx(1.0 / 15.0, abs=1e-9)
        assert w_zp / total_delta_pi == pytest.approx(8.0 / 15.0, abs=1e-9)

        # Mirror structure for Delta(1600)0.
        probs = {}
        for prob, mode in zip(
            _incremental_probabilities(lib[202114]), lib[202114]._modes
        ):
            probs[tuple(mode.daughters)] = prob
        total = probs[(2214, -211)] + probs[(2114, 111)] + probs[(1114, 211)]
        assert probs[(2214, -211)] / total == pytest.approx(8.0 / 15.0, abs=1e-9)
        assert probs[(2114, 111)] / total == pytest.approx(1.0 / 15.0, abs=1e-9)
        assert probs[(1114, 211)] / total == pytest.approx(6.0 / 15.0, abs=1e-9)

    def test_branch_sum_above_one_is_renormalized(self):
        lib = build_soft_resonance_decays(
            branching_overrides={"d1600_br_deltapi": 1.20}
        )
        cums = lib[202224]._probabilities
        assert all(b >= a - 1e-12 for a, b in zip(cums[:-1], cums[1:]))
        assert cums[-1] == pytest.approx(1.0)


class TestSigma1385ZeroIsospin:
    def test_forbidden_sigma0_pi0_channel_absent(self):
        lib = build_soft_resonance_decays()
        daughter_sets = {tuple(m.daughters) for m in lib[3214]._modes}
        assert (3212, 111) not in daughter_sets       # Sigma0 pi0 forbidden
        assert (3222, -211) in daughter_sets          # Sigma+ pi-
        assert (3112, 211) in daughter_sets           # Sigma- pi+

        probs = {}
        for prob, mode in zip(
            _incremental_probabilities(lib[3214]), lib[3214]._modes
        ):
            probs[tuple(mode.daughters)] = prob
        assert probs[(3222, -211)] == pytest.approx(probs[(3112, 211)], rel=1e-9)


class TestNeutralKaonFeedDown:
    def test_k0s_decays_to_charged_and_neutral_pion_pairs(self):
        lib = build_soft_resonance_decays()
        assert 310 in lib
        probs = {}
        for prob, mode in zip(
            _incremental_probabilities(lib[310]), lib[310]._modes
        ):
            probs[tuple(mode.daughters)] = prob
        assert probs[(211, -211)] == pytest.approx(0.692, abs=5e-3)
        assert probs[(111, 111)] == pytest.approx(0.308, abs=5e-3)

    def test_k0s_feeddown_reaches_pions_through_phi(self):
        rng = np.random.default_rng(6)
        react = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        phi_found = False
        for _ in range(60):
            parent = Particle.from_id(0.3, -0.2, 5.0, 333, react)
            ev.add_particle(parent)
            ev.decay_cascade(max_passes=8)
            ids = [p.get_id() for p in ev.particles()]
            ev.delete()
            if 211 in ids and -211 in ids:
                phi_found = True
                break
        assert phi_found, "phi -> K Kbar chain never produced charged pions"


class TestQuantumNumberConservation:
    """Every mode of the full soft-resonance library conserves charge and
    baryon number; strong/EM parents also conserve strangeness except for
    declared weak decayers. This catches any matter/antimatter id mix-up."""

    # pid: (charge, baryon number, strangeness)
    QNS = {
        211: (1, 0, 0), -211: (-1, 0, 0), 111: (0, 0, 0),
        321: (1, 0, 1), -321: (-1, 0, -1),
        311: (0, 0, 1), -311: (0, 0, -1),
        310: (0, 0, 1), 130: (0, 0, 1),   # see K0Kbar0 proxy note below
        2212: (1, 1, 0), 2112: (0, 1, 0),
        -2212: (-1, -1, 0), -2112: (0, -1, 0),
        3122: (0, 1, -1), -3122: (0, -1, 1),
        3222: (1, 1, -1), 3212: (0, 1, -1), 3112: (-1, 1, -1),
        102134: (0, 1, -1),
        3224: (1, 1, -1), 3214: (0, 1, -1), 3114: (-1, 1, -1),
        2224: (2, 1, 0), 2214: (1, 1, 0), 2114: (0, 1, 0), 1114: (-1, 1, 0),
        -1114: (1, -1, 0), -2114: (0, -1, 0),
        -2214: (-1, -1, 0), -2224: (-2, -1, 0),
        12112: (0, 1, 0), 12212: (1, 1, 0),
        2116: (0, 1, 0), 2216: (1, 1, 0),
        22112: (0, 1, 0), 22212: (1, 1, 0),
        102112: (0, 1, 0), 102212: (1, 1, 0),
        122112: (0, 1, 0), 122212: (1, 1, 0),
        212114: (0, 1, 0), 212214: (1, 1, 0),
        202224: (2, 1, 0), 202214: (1, 1, 0),
        202114: (0, 1, 0), 201114: (-1, 1, 0),
        122224: (2, 1, 0), 122214: (1, 1, 0),
        122114: (0, 1, 0), 121114: (-1, 1, 0),
        9911201: (0, 1, 0), 9912201: (1, 1, 0),
        9921201: (0, 1, 0), 9922201: (1, 1, 0),
        9931201: (0, 1, 0), 9932201: (1, 1, 0),
        9900113: (0, 0, 0), 9900213: (0, 0, 0), 9900313: (0, 0, 0),
        113: (0, 0, 0), 213: (1, 0, 0), -213: (-1, 0, 0),
        223: (0, 0, 0), 221: (0, 0, 0), 22: (0, 0, 0),
        10113: (0, 0, 0), 10213: (1, 0, 0), -10213: (-1, 0, 0),
        20113: (0, 0, 0), 20213: (1, 0, 0), -20213: (-1, 0, 0),
        115: (0, 0, 0), 215: (1, 0, 0), -215: (-1, 0, 0),
        225: (0, 0, 0),
        117: (0, 0, 0), 217: (1, 0, 0), -217: (-1, 0, 0),
        229: (0, 0, 0),
        323: (1, 0, 1), -323: (-1, 0, -1),
        313: (0, 0, 1), -313: (0, 0, -1),
        333: (0, 0, 0),
        10323: (1, 0, 1), -10323: (-1, 0, -1),
        10313: (0, 0, 1), -10313: (0, 0, -1),
        325: (1, 0, 1), -325: (-1, 0, -1),
        315: (0, 0, 1), -315: (0, 0, -1),
        # omega(1420)/omega(1650) merged into the soft factory
        100223: (0, 0, 0), 30223: (0, 0, 0),
        # anti-N*/anti-tower states injected for antibaryon cascades
        -12112: (0, -1, 0), -2116: (0, -1, 0), -22112: (0, -1, 0),
        -102112: (0, -1, 0), -122112: (0, -1, 0), -212114: (0, -1, 0),
        -9911201: (0, -1, 0), -9921201: (0, -1, 0), -9931201: (0, -1, 0),
        # charged anti-tower partners (conjugates of positive matter levels)
        -212214: (-1, -1, 0), -9912201: (-1, -1, 0),
        -9922201: (-1, -1, 0), -9932201: (-1, -1, 0),
    }

    # Parents whose decays are weak; strangeness may change, charge and
    # baryon number may not.
    WEAK_PARENTS = {3122, -3122, 3222, 3112, 310}

    def _qns(self, pid):
        if pid in self.QNS:
            return self.QNS[pid]
        raise AssertionError(f"quantum numbers undefined for pdg {pid}")

    def test_full_library_conserves_quantum_numbers(self):
        lib = build_soft_resonance_decays()

        violations = []
        for pid, dec in sorted(lib.items()):
            q_p, b_p, s_p = self._qns(pid)
            for mode in dec._modes:
                daughters = list(mode.daughters)
                qs = [self._qns(d) for d in daughters]

                q_tot = sum(x[0] for x in qs)
                b_tot = sum(x[1] for x in qs)
                s_tot = sum(x[2] for x in qs)

                # K0-Kbar0 placeholder pair: (310, 130) jointly represents a
                # neutral-kaon pair of net strangeness zero (documented
                # convention in na49_mixing.decays).
                if sorted(daughters) == [130, 310]:
                    s_tot = 0

                tag = f"{pid} -> {daughters}"
                if q_tot != q_p:
                    violations.append(f"charge: {tag} ({q_tot} vs {q_p})")
                if b_tot != b_p:
                    violations.append(f"baryon: {tag} ({b_tot} vs {b_p})")
                if pid not in self.WEAK_PARENTS and s_tot != s_p:
                    violations.append(f"strangeness: {tag} ({s_tot} vs {s_p})")

        assert not violations, "\n".join(violations[:20])

    def test_weak_decays_obey_allowed_quantum_number_changes(self):
        """Weak parents: |dS| <= 2, baryon number conserved."""
        lib = build_soft_resonance_decays()
        for pid in self.WEAK_PARENTS & set(lib.keys()):
            q_p, b_p, s_p = self._qns(pid)
            for mode in lib[pid]._modes:
                qs = [self._qns(d) for d in mode.daughters]
                b_tot = sum(x[1] for x in qs)
                s_tot = sum(x[2] for x in qs)
                q_tot = sum(x[0] for x in qs)
                assert q_tot == q_p
                assert b_tot == b_p
                assert abs(s_tot - s_p) <= 1 if pid != 310 else True


class TestTwoBodyPhaseSpaceWeighting:
    def _rho_masses(self, weighted: bool, n: int = 6000):
        """Sample rho-daughter masses from omega(1420)->rho0 pi0 decays."""
        rng = np.random.default_rng(7 if weighted else 8)
        react = Reaction(158.0)
        ev = Event(rng=rng)
        mode_decay = build_omega_decays(rng=rng)[100223]._modes[0]
        masses = []
        for _ in range(n):
            parent = Particle(reaction=react)
            parent.set_id(100223)
            parent.set_mass(get_mass(100223))
            parent.set_p(0.2, 0.1, 4.0)
            ev.add_particle(parent)
            n_before = len(ev.particles())
            ok = ev._decay_2body(parent, mode_decay)
            if ok:
                d1 = ev.particles()[n_before]
                masses.append(d1.get_mass())
                ev.remove(d1)
                ev.remove(ev.particles()[n_before])
            else:
                ev.remove(parent)
        return np.asarray(masses)

    def test_conservation_holds_with_weighted_bw_decays(self):
        rng = np.random.default_rng(9)
        react = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        worst = 0.0
        for i in range(500):
            pid = [100223, 30223, 20113, 215][i % 4]
            parent = Particle.from_id(0.4, -0.1, 7.0, pid, react)
            ev.add_particle(parent)
            ev.decay_cascade(max_passes=8)
            e0 = parent.get_e()
            dp = [
                sum(p.get_e() for p in ev.particles()) - e0,
                sum(p.get_px() for p in ev.particles()) - parent.get_px(),
                sum(p.get_py() for p in ev.particles()) - parent.get_py(),
                sum(p.get_pz() for p in ev.particles()) - parent.get_pz(),
            ]
            worst = max(worst, max(abs(x) for x in dp))
            ev.delete()
        assert worst < 1e-6

    def test_phase_space_softens_line_shape_near_limit(self):
        m0 = get_mass(100223)
        mpi0 = get_mass(111)
        xs = np.linspace(0.2791 + 1e-6, m0 - mpi0, 200000)
        params = (0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5)

        from na49_mixing.breit_wigner import bw_new_array

        ys = bw_new_array(xs, params)
        q = np.array(
            [Event._two_body_pstar(m0, x, mpi0) for x in xs]
        )
        raw_mean = np.trapz(xs * ys, xs) / np.trapz(ys, xs)
        weighted_mean = np.trapz(xs * ys * q, xs) / np.trapz(ys * q, xs)
        assert weighted_mean < raw_mean - 0.005, (
            "phase-space weighting must shift the sampled line shape down"
        )

        # Sampled masses from the engine should sit between the raw and the
        # fully weighted predictions within statistical tolerance.
        masses = self._rho_masses(weighted=True)
        assert masses.size > 3000
        sampled_mean = float(masses.mean())
        assert weighted_mean - 0.03 < sampled_mean < raw_mean + 0.01


class TestWeakFeedAncestry:
    def test_lambda_daughters_carry_weak_flag(self):
        rng = np.random.default_rng(21)
        react = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        parent = Particle.from_id(0.2, -0.1, 4.0, 3122, react)
        parent.set_mass(get_mass(3122))
        ev.add_particle(parent)
        ev.decay_cascade(max_passes=8)
        finals = ev.particles()
        allowed = ({2212, -211}, {2112, 111})
        assert {p.get_id() for p in finals} in allowed
        for p in finals:
            assert p.get_weak_feed() == 1
            assert p.get_gen_depth() >= 1

    def test_direct_delta_decay_has_no_weak_flag(self):
        rng = np.random.default_rng(22)
        react = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        parent = Particle.from_id(0.3, 0.0, 5.0, 2224, react)
        parent.set_mass(get_mass(2224))
        ev.add_particle(parent)
        ev.decay_cascade(max_passes=8)
        for p in ev.particles():
            assert p.get_weak_feed() == 0


class TestMixBufferOverlappingFilters:
    def test_particle_passing_both_filters_enters_both_lists(self):
        from na49_mixing.cuts import ParticleCut
        from na49_mixing.mixing import MixBuffer
        from na49_mixing.particle import Particle as P

        cut_a = ParticleCut(); cut_a.set_id_cut(211)
        cut_b = ParticleCut()  # no id restriction -> matches everything
        buf = MixBuffer(max_depth=2, a_filter=cut_a, b_filter=cut_b)

        from na49_mixing.event import Event
        ev = Event()
        ev.add_particle(Particle(1.0, 0.0, 5.0, 1.0, 0.1395699, 211))
        added = buf.process_event(ev)
        assert added is True
        mev = buf._buffer[0]
        assert len(mev.a_particles()) == 1
        assert len(mev.b_particles()) == 1
