"""The baryon-antibaryon pair vertex of the Round 4 alignment law.

The Round 6 baryon-pair channels (plan section 12.4; ``H_0 -> p pbar``
and ``H_0 -> n nbar`` of the neutral mesonic family, 2026-09-28) make
a decay vertex with two baryon-carrying daughters reachable through
the declared families.  Round 4 pinned that vertex as unreachable and
rejected it.  The plan states that the pair channels carry net baryon
number zero and do not alter the baryonic beam lines, so the vertex
has no baryonic line: the alignment law dispatches it like the
symmetric mesonic vertex (the even mesonic law while that kernel is
active, else the isotropic branch), and the label bookkeeping follows
the same kernel.  A two-baryon pair of one sign stays rejected.

Checks:

- ``draw_vertex_cos_theta``: the net-zero pair reproduces the
  symmetric mesonic vertex bit for bit at the same quantile, in either
  daughter order and under either sign of the pair; the baryonic
  kernel alone leaves it on the isotropic branch; a one-sign pair
  raises ``ValueError``.
- ``assign_alignment_label(..., pair_vertex=True)``: the mesonic label
  while the mesonic kernel is active, the card-4 placeholder otherwise,
  whatever the daughter's own baryon number; the default keeps the
  Round 4 behavior.
- The attached vertex of the ablation variants on the committed
  ``p pbar`` and ``n nbar`` channels: 1 + 2 draws, exact ``(B, Q, S)``
  conservation, opposite baryon numbers on the daughters, both
  daughters on the mesonic label at the nominal kappas and on the
  placeholder at the isotropic point.
- ``decay_variant`` at a neutral mesonic root below every continuum
  threshold (the direct pool holds the pair channels) runs every
  policy over a pinned event range without a refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from na49_mixing.cascade_ablation import (
    ABLATION_POLICIES,
    _committed_catalog,
    _committed_fams,
    _draw_attached_vertex,
    decay_variant,
    exchange_root_state,
    variant_stream,
)
from na49_mixing.cascade_alignment import (
    BARYONIC_ALIGNMENT_LABEL,
    MESONIC_ALIGNMENT_LABEL,
    assign_alignment_label,
    draw_mesonic_alignment,
    draw_vertex_cos_theta,
)
from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL, enumerate_channels

BARYONIC_QN = (1, 1, 0)
ANTIBARYONIC_QN = (-1, -1, 0)
NEUTRON_QN = (1, 0, 0)
ANTINEUTRON_QN = (-1, 0, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)

#: A neutral mesonic root below every continuum threshold of the
#: committed configuration: the direct pool carries the pair channels.
ROOT_MASS = 2.4
MASTER_SEED = 20260928


def _event(mass: float = ROOT_MASS) -> dict:
    return {
        "topology": "pair_probe",
        "momentum": (float(mass), 0.0, 0.0, 0.0),
        "family_id": "mesonic_neutral",
        "mass": float(mass),
        "qn": (0, 0, 0),
        "axis": (0.0, 0.0, 1.0),
        "ancestry": ("exchange_probe",),
    }


class _Counting:
    def __init__(self, source) -> None:
        self._source = source
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._source.random())


def test_pair_draw_follows_the_symmetric_mesonic_dispatch():
    no_baryon = (MESONIC_QN, MESONIC_QN)
    for pair in ((BARYONIC_QN, ANTIBARYONIC_QN),
                 (ANTIBARYONIC_QN, BARYONIC_QN),
                 (NEUTRON_QN, ANTINEUTRON_QN),
                 (STRANGE_BARYONIC_QN, ANTIBARYONIC_QN)):
        for u in (0.0, 0.1, 0.3, 0.5, 0.77, 1.0):
            for kappa_m in (0.5, 1.0, 4.0, -2.0):
                assert draw_vertex_cos_theta(
                    u, pair, kappa_mesonic=kappa_m) == draw_vertex_cos_theta(
                        u, no_baryon, kappa_mesonic=kappa_m)
                assert draw_vertex_cos_theta(
                    u, pair, kappa_mesonic=kappa_m) == draw_mesonic_alignment(
                        u, kappa_m)
                # the baryonic kernel does not act on the pair
                assert draw_vertex_cos_theta(
                    u, pair, kappa_baryonic=3.0, kappa_mesonic=kappa_m
                ) == draw_mesonic_alignment(u, kappa_m)
            # the vertex default and the baryonic kernel alone: isotropic
            assert draw_vertex_cos_theta(u, pair) == 2.0 * u - 1.0
            assert draw_vertex_cos_theta(
                u, pair, kappa_baryonic=1.0) == 2.0 * u - 1.0


def test_one_sign_pair_still_raises():
    for pair in ((BARYONIC_QN, STRANGE_BARYONIC_QN),
                 (BARYONIC_QN, NEUTRON_QN),
                 (ANTIBARYONIC_QN, ANTINEUTRON_QN)):
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair)
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair, kappa_mesonic=1.0)


def test_pair_vertex_label():
    for qn in (BARYONIC_QN, ANTIBARYONIC_QN, NEUTRON_QN, MESONIC_QN):
        assert assign_alignment_label(
            qn, 1.0, 1.0, pair_vertex=True) == MESONIC_ALIGNMENT_LABEL
        assert assign_alignment_label(
            qn, 0.0, 1.0, pair_vertex=True) == MESONIC_ALIGNMENT_LABEL
        assert assign_alignment_label(
            qn, 1.0, 0.0, pair_vertex=True) == ISOTROPIC_ALIGNMENT_LABEL
        assert assign_alignment_label(
            qn, 0.0, 0.0, pair_vertex=True) == ISOTROPIC_ALIGNMENT_LABEL
    # the default keeps the Round 4 bookkeeping
    assert assign_alignment_label(BARYONIC_QN, 1.0, 0.0) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(MESONIC_QN, 0.0, 1.0) == MESONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(BARYONIC_QN, 0.0, 1.0) == ISOTROPIC_ALIGNMENT_LABEL


def _pair_channels(state, catalog, fams):
    channels = enumerate_channels(state, catalog, fams)
    assert not any(channel.kind == "continuum" for channel in channels)
    pairs = [
        channel for channel in channels
        if channel.daughters in (("catalog:2212", "catalog:-2212"),
                                 ("catalog:2112", "catalog:-2112"))
    ]
    assert len(pairs) == 2, [channel.daughters for channel in channels]
    return pairs


def test_attached_pair_vertex_budget_conservation_and_labels():
    catalog = _committed_catalog()
    fams = _committed_fams()
    state = exchange_root_state(_event())
    for channel in _pair_channels(state, catalog, fams):
        for kappa_b, kappa_m, expected in ((1.0, 1.0, MESONIC_ALIGNMENT_LABEL),
                                           (0.0, 1.0, MESONIC_ALIGNMENT_LABEL),
                                           (1.0, 0.0, ISOTROPIC_ALIGNMENT_LABEL),
                                           (0.0, 0.0, ISOTROPIC_ALIGNMENT_LABEL)):
            rng_c = _Counting(np.random.default_rng(MASTER_SEED))
            rng_k = _Counting(np.random.default_rng(MASTER_SEED + 1))
            selected, (d0, d1) = _draw_attached_vertex(
                state, [channel], catalog, fams, kappa_b, kappa_m, rng_c, rng_k)
            assert selected is channel
            assert (rng_c.calls, rng_k.calls) == (1, 2)
            assert d0.qn[0] + d1.qn[0] == 0 and d0.qn[0] != 0
            assert tuple(a + b for a, b in zip(d0.qn, d1.qn)) == (0, 0, 0)
            assert d0.alignment_label == d1.alignment_label == expected
            total = np.array(d0.momentum) + np.array(d1.momentum)
            assert np.allclose(total, np.array(state.momentum), atol=1e-9)


def test_decay_variant_runs_the_pair_pool_without_refusal():
    event = _event()
    for name in ("isotropic_sequential", "aligned_sequential"):
        policy = ABLATION_POLICIES[name]
        for event_index in range(64):
            streams = variant_stream(MASTER_SEED, event_index, 1)
            trace = decay_variant(event, policy, streams[0], streams[1])
            assert trace and trace[0].qn == (0, 0, 0)
