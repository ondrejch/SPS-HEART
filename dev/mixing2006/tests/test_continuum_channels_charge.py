"""Charge-complete decay channel basis (M2) pinning tests.

Task TASK-20260905-physics-review, milestone M2, encoding the committed
behavior of the nested tree at cde6f9b ("fix(excitation): complete the
source-decay channel basis with charge-balanced extra pions and
strange-pair channels"):

- the extra pions beyond a base channel are sampled over every charge
  state consistent with the source: ``k`` (pi+, pi-) pairs plus
  ``n_extra - 2k`` pi0 with ``k`` uniform over ``0..n_extra // 2``;
  every realized composition carries ``(B, Q, S) = (0, 0, 0)``;
- each baryonic source type carries the strangeness-zero strange-pair
  families (p K+ K-, p K0 aK0, n K+ aK0 for p-like ``(1,1,0)``; n K+ K-,
  n K0 aK0, p K0 K- for n-like ``(1,0,0)``), all at the shared
  ``strange_pair_strength`` prior (0.05, documented default), so
  isospin-degenerate charge partners carry the same requested weight;
- ``family_requested_weights`` returns the pre-normalization requested
  weights over the complete family set: the non-strange candidate with
  unit weight plus, per strange/pair family, its strength times the
  thresholded opening factor, hard zero at and below the physical
  threshold (the sum of the daughter PDG masses).

Statistical pre-registrations:

- uniform-``k`` chi-square (``K_DRAWS = 4000`` draws per extra-pion
  count ``n_extra`` in ``2..9``): the chi-square over the ``k`` counts
  must stay below the chi-square quantile with upper-tail probability
  ``0.00135`` (the one-sided 3-sigma Gaussian tail, ``P(Z > 3)``) for
  ``df = n_extra // 2``; the quantiles are pre-registered in
  ``CHI2_3SIGMA`` (scipy ``chi2.ppf(1 - 1.35e-3, df)``).
- decay-draw audit (``DECAY_DRAWS = 20000`` decays of one p-like source
  at 3.0 GeV, all families open): each family count within ``3 sigma``
  of the renormalized requested weight, multinomial variance
  ``N p (1 - p)``.
- seeded baseline sample (``SAMPLE_EVENTS = 20000`` p+p events, seed
  ``20260905``, regge mode, bundled configuration, matching the run
  report's sample): realized per-family fractions within ``3 sigma`` of
  the renormalized requested weights using the exact per-source
  multinomial variance ``sum p(1-p)`` over the sources.

Run artifacts: ``dev/00runs/charge_complete_basis_2026-09-05/``
(``REPORT.md``, ``after_summary.json``; all 8 p-like families within 3
sigma there, max ``|z|`` 1.60). Hermetic: reads only
``config/staged_excitation.json`` and the ``na49_mixing`` sources; no
network, no external data tables, deterministic seeds.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.continuum_channels import (
    ContinuumParameters,
    SourceDecayer,
    _family_spec,
    family_requested_weights,
)
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import QuantumNumbers, sum_quantum_numbers
from na49_mixing.source_event import SourceRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CONFIG = REPO_ROOT / "config" / "staged_excitation.json"

P_LIKE = QuantumNumbers(1, 1, 0)
N_LIKE = QuantumNumbers(1, 0, 0)

# Pre-registered seeds and statistics (see the module docstring).
SAMPLE_SEED = 20260905
SAMPLE_EVENTS = 20_000
K_SEED = 20260907
K_DRAWS = 4_000
DECAY_SEED = 20260908
DECAY_DRAWS = 20_000
DECAY_SOURCE_MASS = 3.0
Z_TOL = 3.0

# Pre-registered 3-sigma chi-square upper bounds for df = 1..4: the
# quantiles with upper-tail probability 0.00135 = P(Z > 3), the
# one-sided 3-sigma Gaussian tail (scipy chi2.ppf(1 - 1.35e-3, df)).
CHI2_3SIGMA = {1: 10.272879, 2: 13.215301, 3: 15.630403, 4: 17.800413}

# Threshold literals from the run report (four decimals), drift guards
# against the recomputed PDG mass sums.
REPORT_THRESHOLDS = {
    P_LIKE: {
        "associated_KLambda": 1.6094,
        "associated_KSigma": 1.6863,
        "strange_pair_pKpKm": 1.9256,
        "strange_pair_pK0K0bar": 1.9335,
        "strange_pair_nKpK0bar": 1.9309,
        "baryon_pair": 2.8148,
    },
    N_LIKE: {
        "associated_KLambda": 1.6133,
        "associated_KSigma": 1.6911,
        "strange_pair_nKpKm": 1.9269,
        "strange_pair_nK0K0bar": 1.9348,
        "strange_pair_pK0Km": 1.9296,
        "baryon_pair": 2.8161,
    },
}
REPORT_NONSTRANGE_THRESHOLDS = {
    P_LIKE: (1.0732, 1.0791),
    N_LIKE: (1.0745, 1.0778),
}
PINNED_STRANGE_PAIR_NAMES = {
    P_LIKE: {
        "strange_pair_pKpKm",
        "strange_pair_pK0K0bar",
        "strange_pair_nKpK0bar",
    },
    N_LIKE: {
        "strange_pair_nKpKm",
        "strange_pair_nK0K0bar",
        "strange_pair_pK0Km",
    },
}


def _load_bundled_config() -> dict:
    return json.loads(BUNDLED_CONFIG.read_text())


def _bundled_continuum() -> ContinuumParameters:
    return ContinuumParameters(**_load_bundled_config()["continuum"])


def test_extra_pion_compositions_are_charge_balanced_and_k_is_uniform():
    """The extra-pion set is pi0 singles plus pi+ pi- pairs, k uniform.

    For every extra-pion count ``n_extra`` in ``2..9`` with ``K_DRAWS``
    seeded draws: every composition uses only {pi0, pi+, pi-}, the
    charged pions come in equal numbers (pairs only), the composition
    carries ``(B, Q, S) = (0, 0, 0)``, all admissible ``k`` values
    ``0..n_extra // 2`` occur, and the ``k`` counts stay inside the
    pre-registered 3-sigma chi-square bound (module docstring).
    """
    decayer = SourceDecayer(ContinuumParameters(), np.random.default_rng(K_SEED))
    assert decayer._extra_pions(0) == []
    assert decayer._extra_pions(1) == [111]
    for n_extra in range(2, 10):
        k_counts = Counter()
        for _ in range(K_DRAWS):
            ids = decayer._extra_pions(n_extra)
            n_plus = ids.count(211)
            n_minus = ids.count(-211)
            n_zero = ids.count(111)
            assert set(ids) <= {211, -211, 111}
            assert n_plus == n_minus
            assert n_plus + n_minus + n_zero == n_extra
            assert sum_quantum_numbers(ids) == QuantumNumbers()
            k_counts[n_plus] += 1
        max_pairs = n_extra // 2
        assert set(k_counts) == set(range(max_pairs + 1))
        expected = K_DRAWS / (max_pairs + 1)
        chi2 = sum((count - expected) ** 2 / expected for count in k_counts.values())
        assert chi2 <= CHI2_3SIGMA[max_pairs], (
            f"n_extra={n_extra}: k chi-square {chi2:.2f} exceeds the "
            f"3-sigma bound {CHI2_3SIGMA[max_pairs]:.4f} for "
            f"df={max_pairs}; the k distribution is not uniform"
        )


def test_family_bqs_sums_conserve_source_and_strange_pairs_cover_kminus_anti_k0():
    """Every family conserves the source (B, Q, S) exactly.

    The 8 charge states per source type (2 non-strange partners plus the
    6 strange/pair families) each sum to the source quantum numbers;
    every strange-pair family sums to S = 0, is carried at the shared
    ``strange_pair_strength`` prior, and contains K- or anti-K0; the
    strange-pair set of each source type covers both K- and anti-K0.
    """
    params = ContinuumParameters()
    for qn in (P_LIKE, N_LIKE):
        nonstrange, strange_and_pair = _family_spec(qn, params)
        assert len(nonstrange) == 2
        assert len(strange_and_pair) == 6
        for ids in nonstrange:
            assert sum_quantum_numbers(ids) == qn
        names = [entry[0] for entry in strange_and_pair]
        assert {name for name in names if name.startswith("strange_pair")} == (
            PINNED_STRANGE_PAIR_NAMES[qn]
        )
        for name, ids, strength, _ in strange_and_pair:
            total = sum_quantum_numbers(ids)
            assert total == qn
            assert total.S == 0
            if name.startswith("strange_pair"):
                assert strength == params.strange_pair_strength
                assert -321 in ids or -311 in ids
        pair_ids = [
            ids
            for name, ids, _, _ in strange_and_pair
            if name.startswith("strange_pair")
        ]
        assert any(-321 in ids for ids in pair_ids)
        assert any(-311 in ids for ids in pair_ids)


def test_family_thresholds_match_pdg_mass_sums_and_gate_the_weight():
    """No family opens at or below the sum of its daughter PDG masses.

    The threshold of each strange/pair family is recomputed
    independently from the public PDG mass lookup and pinned to the
    run-report literal (abs 5e-4). ``family_requested_weights`` omits
    the family (weight exactly zero, hence absent) below and at the
    threshold, and includes it with a positive weight just above. Both
    non-strange partners of each type are legal at the continuum match
    point, so the full family set opens at ``match_mass``.
    """
    params = ContinuumParameters()
    for qn in (P_LIKE, N_LIKE):
        nonstrange, strange_and_pair = _family_spec(qn, params)
        for ids, literal in zip(nonstrange, REPORT_NONSTRANGE_THRESHOLDS[qn]):
            threshold = sum(get_mass(pid) for pid in ids)
            assert threshold == pytest.approx(literal, abs=5e-4)
            assert threshold <= params.match_mass
        for name, ids, _, _ in strange_and_pair:
            threshold = sum(get_mass(pid) for pid in ids)
            assert threshold == pytest.approx(
                REPORT_THRESHOLDS[qn][name], abs=5e-4
            )
            for mass in (threshold - 2e-3, threshold - 1e-3, threshold):
                weights = family_requested_weights(mass, qn, params)
                assert name not in weights or weights[name] == 0.0, (
                    f"{name} ({qn.as_tuple()}): open at mass {mass:.4f} at "
                    f"or below its threshold {threshold:.4f}"
                )
            mass_above = max(threshold + 1e-3, params.match_mass + 1e-3)
            weights = family_requested_weights(mass_above, qn, params)
            assert weights.get(name, 0.0) > 0.0, (
                f"{name} ({qn.as_tuple()}): closed just above its "
                f"threshold at mass {mass_above:.4f}"
            )


def test_requested_weights_equal_strength_times_threshold_factor():
    """``family_requested_weights`` is the explicit pre-normalization audit.

    At a series of masses the returned dict equals, per open family, the
    strength times the independently recomputed thresholded factor
    ``max(0, 1 - (T/m)^2)^power``, with the non-strange candidate at unit
    weight; the sum therefore exceeds one (pre-normalization), and below
    ``match_mass`` the only family is the resolved non-strange proxy at
    unit weight. The bundled configuration pins the documented
    defaults, including ``strange_pair_strength = 0.05``.
    """
    config = _load_bundled_config()
    assert config["continuum"]["strange_pair_strength"] == 0.05
    params = _bundled_continuum()
    assert params == ContinuumParameters()
    for qn in (P_LIKE, N_LIKE):
        _, strange_and_pair = _family_spec(qn, params)
        for mass in (2.4, 2.5, 3.0, 3.5, 4.0):
            weights = family_requested_weights(mass, qn, params)
            expected = {"continuum_nonstrange": 1.0}
            for name, ids, strength, power in strange_and_pair:
                threshold = sum(get_mass(pid) for pid in ids)
                if mass > threshold:
                    factor = max(
                        1.0 - (threshold * threshold) / (mass * mass), 0.0
                    ) ** power
                    expected[name] = strength * factor
            assert set(weights) == set(expected)
            for name, value in expected.items():
                assert weights[name] == pytest.approx(value, abs=1e-12)
            assert weights["continuum_nonstrange"] == 1.0
            assert sum(weights.values()) > 1.0
        below = family_requested_weights(params.match_mass - 1e-3, qn, params)
        assert below == {"resolved_nonstrange": 1.0}


def test_seeded_decay_draws_match_renormalized_requested_weights():
    """One fixed source: the decay family draw follows the renormalized weights.

    ``DECAY_DRAWS`` seeded decays of one p-like source at
    ``DECAY_SOURCE_MASS = 3.0`` GeV (all families open) count the
    ``channel_family`` metadata; each family count must lie within
    ``Z_TOL`` of the renormalized requested weight using the multinomial
    variance ``N p (1 - p)``. Every decayed source sum is exactly
    ``(1, 1, 0)``.
    """
    params = ContinuumParameters()
    source = SourceRecord(
        source_id=0,
        role="projectile",
        p4=FourVector(DECAY_SOURCE_MASS, 0.0, 0.0, 0.0),
        quantum_numbers=P_LIKE,
    )
    decayer = SourceDecayer(params, np.random.default_rng(DECAY_SEED))
    family_counts = Counter()
    for _ in range(DECAY_DRAWS):
        daughters = decayer.decay(source)
        family_counts[daughters[0].channel_family] += 1
        assert (
            sum((p.quantum_numbers for p in daughters), QuantumNumbers())
            == source.quantum_numbers
        )
    weights = family_requested_weights(DECAY_SOURCE_MASS, P_LIKE, params)
    total = sum(weights.values())
    for name in sorted(weights):
        p = weights[name] / total
        expected = DECAY_DRAWS * p
        realized = float(family_counts.get(name, 0))
        sigma = math.sqrt(DECAY_DRAWS * p * (1.0 - p))
        if sigma > 0.0:
            assert abs(realized - expected) <= Z_TOL * sigma, (
                f"{name}: {realized:.0f} decays deviate from {expected:.1f} "
                f"by more than {Z_TOL} sigma ({sigma:.1f})"
            )
        else:
            assert realized == expected


def test_seeded_pp_baseline_sample_is_charge_complete_and_conserved():
    """The M2 acceptance sample: 20000 seeded p+p events, bundled config.

    (a) the final set contains pi- (-211), K- (-321) and anti-K0 (-311);
    (b) at least one p-like source decays through a strange-pair family
    carrying K- and/or anti-K0 with the daughter (B, Q, S) sum exactly
    (1, 1, 0); (c) every event closes four-momentum at the default
    1e-10 and the global integer balance is exact; (d) the realized
    per-family fractions match the renormalized requested weights within
    ``Z_TOL`` using the exact per-source multinomial variance
    ``sum p(1-p)``; (e) every final antibaryon traces to the
    ``baryon_pair`` channel.
    """
    config = _load_bundled_config()
    continuum = ContinuumParameters(**config["continuum"])
    builder = PPEventBuilder(
        PPExcitationConfig(
            beam_momentum=float(config["beam_momentum"]),
            production_mode="regge",
            simple=SimpleExcitationParameters(**config["simple"]),
            regge=ReggeExcitationParameters(**config["regge"]),
            continuum=continuum,
        ),
        np.random.default_rng(SAMPLE_SEED),
    )
    species = Counter()
    family_counts = Counter()
    expected_counts: dict[str, float] = {}
    family_variance: dict[str, float] = {}
    max_residual = 0.0
    global_qn = QuantumNumbers()
    p_like_strange_pair_with_kminus = 0
    n_antibaryons = 0
    for event_id in range(SAMPLE_EVENTS):
        event = builder.generate(event_id)
        max_residual = max(max_residual, event.conservation_residuals()["relative_max"])
        event.validate()
        for particle in event.final_particles:
            species[particle.pdg_id] += 1
            global_qn = global_qn + particle.quantum_numbers
            if particle.pdg_id < 0 and abs(particle.pdg_id) in (2212, 2112):
                assert particle.channel_family == "baryon_pair", (
                    f"event {event_id}: final antibaryon {particle.pdg_id} "
                    f"not traceable to the baryon_pair channel "
                    f"({particle.channel_family})"
                )
                n_antibaryons += 1
        for source in event.sources:
            daughters = [
                p for p in event.final_particles if p.source_id == source.source_id
            ]
            family = daughters[0].channel_family
            family_counts[family] += 1
            if family.startswith("strange_pair") and source.quantum_numbers == P_LIKE:
                if any(p.pdg_id in (-321, -311) for p in daughters):
                    daughter_qn = sum(
                        (p.quantum_numbers for p in daughters), QuantumNumbers()
                    )
                    assert daughter_qn == P_LIKE
                    p_like_strange_pair_with_kminus += 1
            weights = family_requested_weights(
                float(source.mass), source.quantum_numbers, continuum
            )
            total = sum(weights.values())
            for name, weight in weights.items():
                p = weight / total
                expected_counts[name] = expected_counts.get(name, 0.0) + p
                family_variance[name] = family_variance.get(name, 0.0) + p * (1.0 - p)

    assert max_residual < 1e-10
    assert global_qn == QuantumNumbers(2 * SAMPLE_EVENTS, 2 * SAMPLE_EVENTS, 0)
    assert species[-211] > 1000
    assert species[-321] > 200
    assert species[-311] > 400
    assert p_like_strange_pair_with_kminus > 1000
    assert n_antibaryons > 30
    for name in sorted(set(expected_counts) | set(family_counts)):
        expected = expected_counts.get(name, 0.0)
        realized = float(family_counts.get(name, 0))
        sigma = math.sqrt(family_variance.get(name, 0.0))
        if sigma > 0.0:
            z = (realized - expected) / sigma
            assert abs(z) <= Z_TOL, (
                f"{name}: realized {realized:.0f} vs requested {expected:.1f} "
                f"deviates by |z| = {abs(z):.2f} (tolerance {Z_TOL})"
            )
        else:
            assert realized == expected
