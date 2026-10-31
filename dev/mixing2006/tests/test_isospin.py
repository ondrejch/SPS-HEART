"""Tests for isospin reflection (neutron-beam support).

Covers the curated I3 -> -I3 map, the GenerationConfig species plumbing,
and the physical reflection-equivalence gate connecting a Delta- component
on p+p with its conjugate Delta++ component on n+p.

The equivalence gate is statistical rather than bitwise: isospin
multiplet mass splittings (m_n - m_p = 1.293 MeV) shift decay
phase-space floors, so the Breit-Wigner samplers legitimately diverge
between conjugate channels. Charge content must conjugate exactly.
"""
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.isospin import (
    ANTINEUTRON_PROHIBITED_PRIMARIES,
    reflect_pdg,
    validate_reflection_domain,
    has_reflection,
    primary_ids_from_cards,
)
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked


class TestReflectionMap:
    def test_domain_validation_passes(self):
        validate_reflection_domain()

    @pytest.mark.parametrize("a,b", [
        (2212, 2112), (-2212, -2112),
        (211, -211), (111, 111),
        (213, -213), (113, 113),
        (321, 310), (323, 313), (-323, -313),
        (1114, 2224), (2114, 2214),
        (-2224, -1114), (-2214, -2114),
        (12112, 12212), (212214, 212114),
        (9911201, 9912201), (-9912201, -9911201),
        (10213, -10213),
        (3222, 3112), (3224, 3114), (3212, 3212), (3122, 3122),
    ])
    def test_involutive_anchors(self, a, b):
        assert reflect_pdg(a) == b
        assert reflect_pdg(b) == a

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            reflect_pdg(999999)

    def test_prohibited_set_are_neutral_anti_nstars(self):
        for pid in ANTINEUTRON_PROHIBITED_PRIMARIES:
            assert pid < 0

    def test_card_primaries_covered(self):
        cards = primary_ids_from_cards()
        assert cards, "no generator cards found"
        missing = [i for i in cards if not has_reflection(i)]
        assert set(missing) <= set(ANTINEUTRON_PROHIBITED_PRIMARIES)


_DELTA_MINUS_CARD = os.path.join(
    os.path.dirname(__file__), "..",
    "na49_mixing", "generators", "michal", "deltas-to-pion", "delta-.gen",
)


def _run_delta_component(**kw):
    cfg = GenerationConfig(
        gen_file=_DELTA_MINUS_CARD,
        n_events=4000,
        seed=424242,
        use_soft_decay_factory=True,
        fragmentation_mode="projectile",
        **kw,
    )
    return generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)


class TestSpeciesConfig:
    def test_defaults_are_pp(self):
        cfg = GenerationConfig(
            gen_file=_DELTA_MINUS_CARD, n_events=10, seed=1,
        )
        assert cfg.beam_pdg == 2212
        assert cfg.target_pdg == 2212

    def test_unknown_species_rejected(self):
        with pytest.raises(ValueError):
            GenerationConfig(
                gen_file=_DELTA_MINUS_CARD, n_events=10, seed=1,
                beam_pdg=3122,
            )

    def test_neutron_beam_with_prohibited_primary_rejected(self):
        anti_card = os.path.join(
            os.path.dirname(__file__), "..",
            "na49_mixing", "generators", "benchmark", "anti_n1440_zero.gen",
        )
        with pytest.raises(ValueError):
            GenerationConfig(
                gen_file=anti_card, n_events=10, seed=1,
                beam_pdg=2112,
            )
        # Proton projectile stays allowed for the same card.
        GenerationConfig(gen_file=anti_card, n_events=10, seed=1)


class TestReflectionEquivalenceGate:
    """Delta-(p+p) versus conjugated Delta++(n+p), projectile side."""

    @pytest.fixture(scope="class")
    def runs(self):
        return _run_delta_component(), _run_delta_component(beam_pdg=2112)

    def test_track_counts_equal(self, runs):
        pp, np_ = runs
        assert len(np.asarray(pp["pdg"])) == len(np.asarray(np_["pdg"]))

    def test_charge_content_conjugates_exactly(self, runs):
        pp, np_ = runs
        pdg_pp = np.asarray(pp["pdg"])
        pdg_np = np.asarray(np_["pdg"])
        mirrored = np.vectorize(reflect_pdg)(pdg_pp.astype(np.int64))
        assert bool((mirrored == pdg_np).all())
        # The conjugate pairs realized by Delta- -> n pi- vs Delta++ -> p pi+.
        assert {int(p) for p in np.unique(pdg_pp)} == {2112, -211}
        assert {int(p) for p in np.unique(pdg_np)} == {2212, 211}

    def test_kinematics_agree_statistically(self, runs):
        """Means agree within conservative MC tolerance.

        Conjugate runs diverge beyond the sqrt(s) offset because the
        n/p mass split moves the two-body phase-space floor feeding the
        Breit-Wigner sampler; the residual is a genuine isospin-breaking
        effect of order (m_n - m_p) / Q-value, far below data precision.
        """
        pp, np_ = runs
        xf_pp = np.abs(np.asarray(pp["xf"], dtype=float))
        xf_np = np.abs(np.asarray(np_["xf"], dtype=float))
        pt_pp = np.asarray(pp["pt"], dtype=float)
        pt_np = np.asarray(np_["pt"], dtype=float)

        n = len(xf_pp)
        d_mean_xf = abs(xf_pp.mean() - xf_np.mean())
        tol_xf = 5.0 * float(xf_pp.std()) / math.sqrt(n)
        d_mean_pt = abs(pt_pp.mean() - pt_np.mean())
        tol_pt = 5.0 * float(pt_pp.std()) / math.sqrt(n)

        assert d_mean_xf < max(tol_xf, 1e-3), (d_mean_xf, tol_xf)
        assert d_mean_pt < max(tol_pt, 5e-4), (d_mean_pt, tol_pt)
