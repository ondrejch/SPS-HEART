"""Importance weights of the leading splits (2026-11-03, the fourteen-parameter fit).

At an eligible baryonic root the cascade draws the branch first: the leading-meson split with probability F_M, else
the leading-baryon split with probability F_B, else the ordinary channel draw. Given the branch, the channel and the
kinematics do not depend on the two shares. On a split branch the direction cosine c of the daughter the law directs
follows

    p_kappa(c) = kappa exp(kappa (c - 1)) / (1 - exp(-2 kappa)),   -1 <= c <= 1,

with kappa = kappa_B fixed on the baryon branch and kappa = 2 b_M q*^2 on the meson branch, q* the decay momentum. The
recorder of :mod:`na49_mixing.cascade_decay` stores per root the eligibility of both splits, the branch, the root's
baryon share, c and q*. The weight of a root from the generation values to trial values is the ratio of the branch
probabilities times, on a split branch, the ratio of the direction densities; an event's weight is the product over
its roots. The weights are exact: the support of every branch is unchanged for shares inside (0, 1).
"""

from __future__ import annotations

import numpy as np

#: the parameter names, in the order of a split point (F_B, F_M, kappa_B, b_M)
SPLIT_NAMES = ("leading_split_fraction", "meson_split_fraction", "leading_split_kappa", "meson_split_slope")
#: the number of root slots per event in the arrays
SPLIT_SLOTS = 3
#: the branch codes of the arrays
BRANCH_NONE, BRANCH_ORDINARY, BRANCH_BARYON, BRANCH_MESON = -1, 0, 1, 2


def records_to_row(records: list) -> dict:
    """The per-event array values of a root list: ``split_br{k}``, ``split_me{k}``, ``split_be{k}``,
    ``split_fb{k}``, ``split_c{k}``, ``split_q{k}`` for k < SPLIT_SLOTS (branch -1 for an empty slot)."""
    if len(records) > SPLIT_SLOTS:
        raise ValueError(f"split records: {len(records)} eligible roots in one event, more than {SPLIT_SLOTS} slots")
    row = {}
    for k in range(SPLIT_SLOTS):
        r = records[k] if k < len(records) else None
        row[f"split_br{k}"] = float(BRANCH_NONE if r is None else r["branch"])
        row[f"split_me{k}"] = float(bool(r and r["me"]))
        row[f"split_be{k}"] = float(bool(r and r["be"]))
        row[f"split_fb{k}"] = float(r["f_b"]) if (r is not None and r.get("f_b") is not None) else np.nan
        row[f"split_c{k}"] = float(r["c"]) if (r is not None and "c" in r) else np.nan
        row[f"split_q{k}"] = float(r["q"]) if (r is not None and "q" in r) else np.nan
    return row


def _log_direction(kappa, c):
    """log p_kappa(c) for kappa > 0 (elementwise)."""
    kappa = np.asarray(kappa, dtype=float)
    return np.log(kappa) + kappa * (np.asarray(c, dtype=float) - 1.0) - np.log1p(-np.exp(-2.0 * kappa))


def split_log_ratio(shape: dict, trial, generation) -> np.ndarray:
    """The per-event log weight from the generation split point to the trial one, both (F_B, F_M, kappa_B, b_M)."""
    fb1, fm1, kb1, bm1 = (float(v) for v in trial)
    fb0, fm0, kb0, bm0 = (float(v) for v in generation)
    n = len(shape["split_br0"])
    out = np.zeros(n)
    for k in range(SPLIT_SLOTS):
        br = np.asarray(shape[f"split_br{k}"], dtype=float)
        me = np.asarray(shape[f"split_me{k}"], dtype=float) > 0.5
        be = np.asarray(shape[f"split_be{k}"], dtype=float) > 0.5
        c = np.asarray(shape[f"split_c{k}"], dtype=float)
        q = np.asarray(shape[f"split_q{k}"], dtype=float)
        # the root's baryon share scales with the global one (they differ only under a renormalized split)
        fb_root0 = np.where(np.isfinite(shape[f"split_fb{k}"]), shape[f"split_fb{k}"], fb0)
        fb_root1 = fb_root0 * (fb1 / fb0)
        not_meson = np.where(me, np.log1p(-fm1) - np.log1p(-fm0), 0.0)
        term = np.zeros(n)
        meson = br == BRANCH_MESON
        if np.any(meson):
            term[meson] = (np.log(fm1 / fm0) + _log_direction(2.0 * bm1 * q[meson] ** 2, c[meson])
                           - _log_direction(2.0 * bm0 * q[meson] ** 2, c[meson]))
        baryon = br == BRANCH_BARYON
        if np.any(baryon):
            term[baryon] = (not_meson[baryon] + np.log(fb_root1[baryon] / fb_root0[baryon])
                            + _log_direction(kb1, c[baryon]) - _log_direction(kb0, c[baryon]))
        ordinary = br == BRANCH_ORDINARY
        if np.any(ordinary):
            term[ordinary] = not_meson[ordinary] + np.where(
                be[ordinary], np.log1p(-fb_root1[ordinary]) - np.log1p(-fb_root0[ordinary]), 0.0)
        out += term
    return out
