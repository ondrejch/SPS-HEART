"""The flavor ledger (Round 6 runs R61 to R65): per-vertex sufficient
statistics for reweighting a generated topology sample to another
flavor point without regeneration.

Checks:

- passing a ledger changes no draw: the final state of every event is
  identical with and without the ledger on the same substreams;
- the log weight from a point to itself is exactly zero, and a vertex
  whose pool spans one exponent key records ``None``;
- consistency: reweighting an identity-point sample to a suppressed
  strangeness point reproduces the mean strange-hadron count of a
  direct generation at that point within the combined Monte Carlo
  error, with a healthy effective sample size.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import topology_runner as TR
from na49_mixing.flavor_weights import (
    FlavorParams,
    NOMINAL_FLAVOR,
    ledger_log_weight,
    vertex_ledger_record,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import families_from_json

NESTED = Path(__file__).resolve().parents[1]
STRANGE_IDS = {321, 311, 310, 130, 3122, 3222, 3212, 3112, 333, 323, 313}


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(NESTED / "config" / "v2_singlet" / "resonance_catalog.json")


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(NESTED / "config" / "v2_singlet" / "continuum_resonances.json"))


def _rngs(seed: int) -> dict:
    labels = ("rng_topology_selection", "rng_spectral_state", "rng_exchange_kinematics",
              "rng_decay_channel", "rng_decay_kinematics")
    return {k: np.random.default_rng(1_000_003 * seed + i) for i, k in enumerate(labels)}


def _signature(event) -> list:
    return [(h.pdg_id, tuple(float(x) for x in h.four_momentum)) for h in event.final_state]


def _n_strange(event) -> int:
    return sum(1 for h in event.final_state if h.pdg_id is not None and abs(int(h.pdg_id)) in STRANGE_IDS)


def test_ledger_changes_no_draw(catalog, fams):
    informative = 0
    for cls in ("sd_projectile", "dd", "nd"):
        for seed in range(40):
            ledger: list = []
            with_ledger = TR.run_topology(cls, TR.NOMINAL_PARAMS, catalog, fams, **_rngs(seed), ledger=ledger)
            without = TR.run_topology(cls, TR.NOMINAL_PARAMS, catalog, fams, **_rngs(seed))
            assert _signature(with_ledger) == _signature(without)
            assert ledger_log_weight(ledger, None, NOMINAL_FLAVOR) == 0.0
            assert ledger_log_weight(ledger, NOMINAL_FLAVOR, FlavorParams()) == 0.0
            informative += sum(1 for record in ledger if record is not None)
    assert informative > 0


def test_single_key_pool_records_none(catalog, fams):
    class _Channel:
        def __init__(self, daughters, weight):
            self.daughters = daughters
            self.weight = weight
    pool = [_Channel(("catalog:211", "catalog:-211"), 0.4), _Channel(("catalog:111", "catalog:111"), 0.6)]
    assert vertex_ledger_record((0, 0, 0), pool, pool[0], catalog, fams, None) is None
    pool.append(_Channel(("catalog:321", "catalog:-321"), 0.2))
    record = vertex_ledger_record((0, 0, 0), pool, pool[2], catalog, fams, None)
    assert record is not None
    chosen_key, groups = record
    assert chosen_key == (1.0, 0, 0, 0)
    assert set(groups) == {(0.0, 0, 0, 0), (1.0, 0, 0, 0)}
    assert groups[(0.0, 0, 0, 0)] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        vertex_ledger_record((0, 0, 0), pool, _Channel(("x",), 1.0), catalog, fams, None)


def test_reweighting_reproduces_direct_generation(catalog, fams):
    n = 1500
    target = FlavorParams(gamma_s_had=0.5)
    params_target = TR.TopologyNominalParams(
        epsilon=TR.NOMINAL_PARAMS.epsilon, b_sd=TR.NOMINAL_PARAMS.b_sd,
        class_fractions=TR.NOMINAL_PARAMS.class_fractions,
        remnant_weights=TR.NOMINAL_PARAMS.remnant_weights, p_n2=TR.NOMINAL_PARAMS.p_n2,
        flavor=target)
    counts = []
    log_w = []
    for seed in range(n):
        ledger: list = []
        event = TR.run_topology("dd", TR.NOMINAL_PARAMS, catalog, fams, **_rngs(seed), ledger=ledger)
        counts.append(_n_strange(event))
        log_w.append(ledger_log_weight(ledger, None, target))
    counts = np.asarray(counts, dtype=float)
    w = np.exp(np.asarray(log_w) - np.max(log_w))
    w /= np.sum(w)
    ess = 1.0 / np.sum(w * w)
    assert ess / n > 0.2, ess / n
    reweighted_mean = float(np.sum(w * counts))
    reweighted_se = float(math.sqrt(np.sum(w * w * (counts - reweighted_mean) ** 2)))
    direct = np.asarray(
        [_n_strange(TR.run_topology("dd", params_target, catalog, fams, **_rngs(10_000 + seed)))
         for seed in range(n)], dtype=float)
    direct_mean = float(np.mean(direct))
    direct_se = float(np.std(direct, ddof=1) / math.sqrt(n))
    identity_mean = float(np.mean(counts))
    # the suppression lowers the strange yield, and the reweighted sample
    # agrees with the direct sample at the suppressed point
    assert direct_mean < identity_mean
    pull = (reweighted_mean - direct_mean) / math.sqrt(reweighted_se ** 2 + direct_se ** 2)
    assert abs(pull) < 4.0, (reweighted_mean, direct_mean, reweighted_se, direct_se, pull)
