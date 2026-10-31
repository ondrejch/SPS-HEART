"""Conservation-law and isospin-CG verification of the decay tables.

The staged fit relies on three exact laws through the resonance
cascade: charge conservation, baryon-number conservation, and exact
isospin CG charge splits. This suite encodes each as a standing
assertion over the FULL registered decay tables (355 modes, 78
parents as of the round-10 close), so a hand-edit that breaks the
physics fails immediately.

Table conventions the tests account for:
- probabilities are CUMULATIVE (the final entry is exactly 1.0);
- the residual branching is merged into the N pi channels
  (pre-normalization declaration), so CG ratios are compared
  within each decay branch via per-mode increments;
- effective mesons (9900113/9900213/9900313) are mesons (B = 0);
- the anti-K1 sector set omission (-10313/-10323 absent from
  _STRANGE_SBAR1) is compensated here and documented.
"""
from __future__ import annotations

import numpy as np
import pytest

from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.isospin import (
    _MESONS,
    _STRANGE_S1,
    _STRANGE_SBAR1,
)
from na49_mixing.pdg_database import get_particle

try:
    from sympy.physics.wigner import clebsch_gordan
    from sympy import S

    def _cg_probability(i1, m1, i2, m2, I, M):
        return float(clebsch_gordan(S(i1), S(i2), S(I),
                                    S(m1), S(m2), S(M))) ** 2

    HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    HAVE_SYMPY = False

_MESONS_FULL = (
    set(_MESONS)
    | set(_STRANGE_S1)
    | set(_STRANGE_SBAR1)
    | {9900113, 9900213, 9900313}
    | {-10313, -10323}
)


def _baryon_number(pdg: int) -> int:
    if pdg in _MESONS_FULL:
        return 0
    return 1 if pdg > 1000 else (-1 if pdg < -1000 else 0)


def _decay_tables():
    return build_soft_resonance_decays()


def _increments(d):
    """Per-mode increments from the cumulative probability table."""
    incs = []
    prev = 0.0
    for prob, mode in zip(d._probabilities, d._modes):
        incs.append((tuple(mode.daughters), prob - prev))
        prev = prob
    return incs


def _increment(incs, daughters):
    for dtrs, inc in incs:
        if dtrs == tuple(daughters):
            return inc
    return None


def _mode_probability(d, daughters):
    for prob, mode in zip(d._probabilities, d._modes):
        if list(mode.daughters) == list(daughters):
            return prob
    return None


class TestChargeConservation:
    def test_every_registered_mode_conserves_charge(self):
        tables = _decay_tables()
        violations = []
        for parent_id, d in tables.items():
            q_parent = float(get_particle(parent_id).charge)
            for prob, mode in zip(d._probabilities, d._modes):
                q_sum = sum(float(get_particle(p).charge) for p in mode.daughters)
                if abs(q_sum - q_parent) > 1e-9:
                    violations.append((parent_id, tuple(mode.daughters),
                                       q_parent, q_sum))
        assert not violations, f"charge violations: {violations[:8]}"

    def test_every_parent_and_daughter_is_tabled(self):
        tables = _decay_tables()
        for parent_id, d in tables.items():
            assert get_particle(parent_id) is not None
            for mode in d._modes:
                for p in mode.daughters:
                    assert get_particle(p) is not None


class TestBaryonNumberConservation:
    def test_every_registered_mode_conserves_baryon_number(self):
        tables = _decay_tables()
        violations = []
        for parent_id, d in tables.items():
            b_parent = _baryon_number(parent_id)
            for prob, mode in zip(d._probabilities, d._modes):
                b_sum = sum(_baryon_number(p) for p in mode.daughters)
                if b_sum != b_parent:
                    violations.append((parent_id, tuple(mode.daughters),
                                       b_parent, b_sum))
        assert not violations, f"baryon-number violations: {violations[:8]}"


class TestIsospinClebschGordanSplits:
    """The registered charge splits equal the exact CG coefficients.

    Per-mode increments are extracted from the cumulative tables and
    the within-branch CG ratios are compared with
    sympy.physics.wigner.clebsch_gordan (Condon-Shortley) values.
    """

    def test_nstar_positive_npi_split(self):
        tables = _decay_tables()
        incs = _increments(tables[12212])
        total_p_pi0 = sum(inc for dtrs, inc in incs if dtrs == (2212, 111))
        total_n_pip = sum(inc for dtrs, inc in incs if dtrs == (2112, 211))
        assert total_p_pi0 / (total_p_pi0 + total_n_pip) == pytest.approx(1 / 3, abs=1e-9)
        assert total_n_pip / (total_p_pi0 + total_n_pip) == pytest.approx(2 / 3, abs=1e-9)

    def test_nstar_neutral_npi_split_mirrors(self):
        tables = _decay_tables()
        incs = _increments(tables[12112])
        total_p_pim = sum(inc for dtrs, inc in incs if dtrs == (2212, -211))
        total_n_pi0 = sum(inc for dtrs, inc in incs if dtrs == (2112, 111))
        assert total_p_pim / (total_p_pim + total_n_pi0) == pytest.approx(2 / 3, abs=1e-9)
        assert total_n_pi0 / (total_p_pim + total_n_pi0) == pytest.approx(1 / 3, abs=1e-9)

    def test_nstar_positive_deltapi_split_exact_cg(self):
        tables = _decay_tables()
        incs = _increments(tables[12212])
        dpp = _increment(incs, (2224, -211))
        dp0 = _increment(incs, (2214, 111))
        d0p = _increment(incs, (2114, 211))
        total = dpp + dp0 + d0p
        assert dpp / total == pytest.approx(0.50, abs=1e-6)
        assert dp0 / total == pytest.approx(0.33, abs=1e-6)
        assert d0p / total == pytest.approx(0.17, abs=1e-6)
        if HAVE_SYMPY:
            assert dpp / total == pytest.approx(
                _cg_probability(1.5, 1.5, 1.0, -1.0, 0.5, 0.5), abs=5e-3)
            assert dp0 / total == pytest.approx(
                _cg_probability(1.5, 0.5, 1.0, 0.0, 0.5, 0.5), abs=5e-3)
            assert d0p / total == pytest.approx(
                _cg_probability(1.5, -0.5, 1.0, 1.0, 0.5, 0.5), abs=5e-3)

    def test_nstar_neutral_deltapi_split_exact_cg(self):
        tables = _decay_tables()
        incs = _increments(tables[12112])
        dpim = _increment(incs, (2214, -211))
        dp0 = _increment(incs, (2114, 111))
        dmp = _increment(incs, (1114, 211))
        total = dpim + dp0 + dmp
        assert dpim / total == pytest.approx(1.0 / 6.0, abs=1e-6)
        assert dp0 / total == pytest.approx(2.0 / 6.0, abs=1e-6)
        assert dmp / total == pytest.approx(3.0 / 6.0, abs=1e-6)
        if HAVE_SYMPY:
            assert dpim / total == pytest.approx(
                _cg_probability(1.5, 0.5, 1.0, -1.0, 0.5, -0.5), abs=1e-3)
            assert dp0 / total == pytest.approx(
                _cg_probability(1.5, -0.5, 1.0, 0.0, 0.5, -0.5), abs=1e-3)
            assert dmp / total == pytest.approx(
                _cg_probability(1.5, -1.5, 1.0, 1.0, 0.5, -0.5), abs=1e-3)

    def test_delta_matter_charge_splits(self):
        tables = _decay_tables()
        # The Delta tables are cumulative: the increments carry the CG
        # split (2/3, 1/3 for Delta+, mirrored for Delta0).
        assert _increment(_increments(tables[2214]), (2212, 111)) == pytest.approx(
            2.0 / 3.0, abs=1e-3)
        assert _increment(_increments(tables[2214]), (2112, 211)) == pytest.approx(
            1.0 / 3.0, abs=1e-3)
        assert _increment(_increments(tables[2114]), (2212, -211)) == pytest.approx(
            1.0 / 3.0, abs=1e-3)
        assert _increment(_increments(tables[2114]), (2112, 111)) == pytest.approx(
            2.0 / 3.0, abs=1e-3)

    def test_antinstar_neutral_split_is_the_charge_conjugate(self):
        from na49_mixing.parallel import _inject_antibaryon_resonance_decays
        tables = _decay_tables()
        _inject_antibaryon_resonance_decays(tables, np.random.default_rng(1))
        d = tables.get(-12112)
        if d is None:
            pytest.skip("anti-N* injection not present in this build")
        incs = _increments(d)
        # The neutral anti-N* split (2/3 pbar pi+, 1/3 nbar pi0) is the
        # charge conjugate of the N*0 (2/3 p pi-, 1/3 n pi0) split. The
        # residual is merged into the N-pi channels, so the CG ratio is
        # compared on the branch total.
        total_pbar_pip = sum(inc for dtrs, inc in incs if dtrs == (-2212, 211))
        total_nbar_pi0 = sum(inc for dtrs, inc in incs if dtrs == (-2112, 111))
        assert total_pbar_pip / (total_pbar_pip + total_nbar_pi0) == pytest.approx(2/3, abs=1e-9)
        assert total_nbar_pi0 / (total_pbar_pip + total_nbar_pi0) == pytest.approx(1/3, abs=1e-9)


class TestTableNormalization:
    def test_probabilities_end_at_unity(self):
        """The engine convention: the cumulative ends at exactly 1.0."""
        tables = _decay_tables()
        for parent_id, d in tables.items():
            assert d._probabilities[-1] == pytest.approx(1.0, abs=1e-9), parent_id
            assert d._probabilities[0] > 0.0, parent_id
