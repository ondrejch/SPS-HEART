"""Importance-ensemble discipline for finite-Monte-Carlo fits (2026-09-28).

The Round 4 and Round 5 fits reweighted a sample generated at one
parameter point to trial points far away and read the objective off a
few dozen effective events: the R51 fit report records an effective
sample size fraction of 0.00024 at its optimum, and the R52 comparison
records 37 effective events out of 800,000 on its one-model side
(review record ``.collab/reviews/fable-speedup-2026-09-28.md``,
FAB-02).  This module makes the trust region a refusal instead of a
post-hoc note:

- :func:`ess_fraction` is the effective-sample-size fraction
  ``(sum w)^2 / sum w^2 / n`` of a log-weight vector after one global
  shift (the shift leaves it unchanged);
- :func:`check_trust_region` returns that fraction and raises
  :class:`RegenerateAt` when it falls below the floor, carrying the
  trial point so the caller regenerates there (plan section 5.3:
  regenerate when the reweighting leaves the region the sample
  supports);
- :func:`simplex_from_logits` and :func:`logits_from_simplex` place
  class fractions on the simplex through a softmax, so an optimizer
  moves in an unconstrained space and never clips at a vertex.

Hermetic: NumPy only.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_ESS_FLOOR",
    "RegenerateAt",
    "check_trust_region",
    "ess_fraction",
    "logits_from_simplex",
    "simplex_from_logits",
]

#: The default effective-sample-size floor of the trust region: a trial
#: point whose weights keep fewer than one event in ten effective is
#: outside the region the generated sample supports.
DEFAULT_ESS_FLOOR: float = 0.1


class RegenerateAt(Exception):
    """A trial point lies outside the importance trust region.

    ``point`` is the trial point the caller passed to
    :func:`check_trust_region`; ``ess`` the effective-sample-size
    fraction measured there.  The caller regenerates the sample at the
    point (or refuses the trial) instead of reading an objective off a
    degenerate ensemble.
    """

    def __init__(self, ess: float, point=None, floor: float = DEFAULT_ESS_FLOOR):
        self.ess = float(ess)
        self.point = point
        self.floor = float(floor)
        super().__init__(
            f"importance trust region left: effective sample size fraction "
            f"{self.ess:.3e} below the floor {self.floor:g} at point {point!r}"
        )


def _weights_from_log(log_w) -> np.ndarray:
    lw = np.asarray(log_w, dtype=float).ravel()
    if lw.size == 0:
        return lw
    finite = np.isfinite(lw)
    if not finite.any():
        return np.zeros(lw.size, dtype=float)
    shift = float(np.max(lw[finite]))
    with np.errstate(over="ignore", under="ignore"):
        w = np.where(finite, np.exp(lw - shift), 0.0)
    return w


def ess_fraction(log_w) -> float:
    """The effective-sample-size fraction of a log-weight vector.

    ``(sum w)^2 / sum w^2 / n`` after one global log shift; ``-inf``
    entries are zero weights and an all-non-finite vector has fraction
    0.  Equal weights give exactly 1; one dominant weight gives ``1/n``.
    """
    w = _weights_from_log(log_w)
    if w.size == 0:
        return 1.0
    total = float(np.sum(w))
    squares = float(np.sum(w * w))
    if squares <= 0.0 or total <= 0.0:
        return 0.0
    return total * total / squares / float(w.size)


def check_trust_region(log_w, *, floor: float = DEFAULT_ESS_FLOOR, point=None) -> float:
    """Return the effective-sample-size fraction or raise :class:`RegenerateAt`.

    ``floor`` must lie in ``(0, 1]``.  A degenerate ensemble is never
    silently accepted: the exception carries the trial ``point`` so the
    caller can regenerate there.
    """
    floor = float(floor)
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"the effective-sample-size floor must be in (0, 1], got {floor!r}")
    ess = ess_fraction(log_w)
    if ess < floor:
        raise RegenerateAt(ess, point=point, floor=floor)
    return ess


def simplex_from_logits(logits: Sequence[float]) -> np.ndarray:
    """Softmax: unconstrained logits to fractions on the simplex.

    The fractions are positive and sum to 1 to float rounding; no
    vertex clipping is ever needed.  A shift of every logit by the same
    constant leaves the fractions unchanged, so the parametrization has
    one redundant direction; :func:`logits_from_simplex` fixes it by
    returning logits with zero mean.
    """
    z = np.asarray(logits, dtype=float).ravel()
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError(f"logits must be a non-empty finite vector, got {logits!r}")
    e = np.exp(z - z.max())
    return e / e.sum()


def logits_from_simplex(fractions: Sequence[float], *, floor: float = 1e-300) -> np.ndarray:
    """Inverse of :func:`simplex_from_logits` with zero-mean logits.

    Fractions must be non-negative and sum to 1 within ``1e-9``; a zero
    fraction is floored at ``floor`` before the logarithm.
    """
    f = np.asarray(fractions, dtype=float).ravel()
    if f.size == 0 or not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError(f"fractions must be non-negative and finite, got {fractions!r}")
    if abs(float(f.sum()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {float(f.sum())!r}")
    z = np.log(np.maximum(f, floor))
    return z - z.mean()
