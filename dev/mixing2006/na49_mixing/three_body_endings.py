"""The three-body endings of the continuum cascade (2026-10-20; owner:
"implement 3body vtx").

``KernelConfig.three_body_endings`` (the strength kappa_3 [GeV^-2]) gives
every continuum family three-body direct channels of stable hadrons beside
its two-body direct pairs.  They enter the direct pool, so a line ends in
one of them only at a vertex with no open continuation channel, at the
floor of the cascade: the continuation through the continuum stays a chain
of two-body steps.

The register's open question 9 records the physics: the soft pion spectra
prefer the omega's three pions to the two of a rho at every kind of
mesonic vertex, and near the floor a mesonic line of the two-body cascade
ends only in two-body pairs, where the omega's three-body decay is the one
route to three soft pions.  The owner's point: "the cascading mechanism
could include n_body decays ... the de-excitation is so rapid (~10^-24 s)
that it is effectively off the shell".

The weight of a three-body channel is put on the footing of the kernel's
two-body direct weight ``(2J_0+1)(2J_1+1) q* C_iso``, which is proportional
to ``M R_2(M) / pi`` with the Lorentz-invariant two-body phase space
``R_2 = pi q* / M``:

    w_3 = kappa_3 (2J_0+1)(2J_1+1)(2J_2+1) M R_3(M) / pi * C_iso,
    M R_3(M) / pi = 2 pi * integral q*(m; m_0, m_1) q(M; m, m_2) dm,

the integral over the (0, 1) invariant mass m on [m_0 + m_1, M - m_2]
(the Dalitz strip at fixed m^2 has the length 4 q* q M / m).  kappa_3 is
the one parameter: the ratio of the three-body to the two-body coupling at
a common phase-space normalization, the statistical model's interaction
volume up to convention.  ``C_iso`` is the probability that the charge
state carries the parent's isospin, summed over the intermediate couplings
of (0, 1) and over the distinct orderings of the charges, the two-body
convention of the kernel (an I = 0 parent gives pi+ pi- : pi0 pi0 = 2 : 1);
a vertex of the non-diffractive isospin mixture counts each charge
combination once, as its two-body pairs do.

The kinematics is flat three-body phase space
(:func:`na49_mixing.resonance_decay.three_body_phase_space`, the
``phase_space`` law) with one more uniform for the azimuth about the
sampler's axis, so the point is isotropic; the daughters are stable
hadrons.  The flavor ledger counts the created strange pairs as at a
two-body vertex: (Xi, K, K) at a non-strange baryonic line pays gamma_s^2
and assoc_norm.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

from .hadron_catalog import HadronCatalog

__all__ = ["THREE_BODY_SETS", "three_body_classes_for", "three_body_families", "three_body_weight",
           "three_body_isospin_share", "phase_space_factor", "parse_triple"]

_PI = (211, 111, -211)
_K, _KBAR = (321, 311), (-321, -311)
_N, _NBAR = (2212, 2112), (-2212, -2112)
_XI, _XIBAR = (3322, 3312), (-3322, -3312)

#: The three-body sets: name -> the multiplet triples whose charge members
#: form the classes (each set with its charge conjugates).
THREE_BODY_SETS: dict[str, tuple] = {
    "pipipi": ((_PI, _PI, _PI),),
    "Kpipi": ((_K, _PI, _PI), (_KBAR, _PI, _PI)),
    "Npipi": ((_N, _PI, _PI), (_NBAR, _PI, _PI)),
    "XiKK": ((_XI, _K, _K), (_XIBAR, _KBAR, _KBAR)),
}

_PHASE_SPACE_POINTS = 48


def parse_triple(class_id: str) -> tuple[int, int, int]:
    """The daughters of a three-body class identifier ``"(d0,d1,d2)"``."""
    text = str(class_id).strip()
    parts = text[1:-1].split(",") if text.startswith("(") and text.endswith(")") else []
    if len(parts) != 3:
        raise ValueError(f"malformed three-body class identifier {class_id!r}: expected '(d0,d1,d2)'")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def three_body_classes_for(qn, catalog: HadronCatalog, sets) -> tuple[str, ...]:
    """The three-body classes of a family of ``(B, Q, S)`` *qn*: every
    charge combination of the named sets' multiplet triples whose daughters
    sum to *qn* and are catalog states, each unordered combination once, in
    a fixed order."""
    from .resonance_decay import _daughter_bqs
    want = tuple(int(v) for v in qn)
    have = {int(r.pdg_id) for r in catalog.records()} | {int(p) for p in catalog.stable_states()}
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for name in sets:
        for multiplets in THREE_BODY_SETS[name]:
            for combo in itertools.product(*multiplets):
                if not all(d in have for d in combo):
                    continue
                bqs = tuple(sum(v) for v in zip(*(_daughter_bqs(d, catalog) for d in combo)))
                key = tuple(sorted(combo))
                if bqs == want and key not in seen:
                    seen.add(key)
                    out.append(tuple(combo))
    return tuple(f"({a},{b},{c})" for a, b, c in out)


def three_body_families(fams, catalog: HadronCatalog, sets, strength: float) -> list:
    """The families with their three-body classes and the strength kappa_3;
    a family without a matching combination is unchanged."""
    if not (math.isfinite(float(strength)) and float(strength) > 0.0):
        raise ValueError(f"three_body_endings: the strength must be a finite positive kappa_3, got {strength!r}")
    out = []
    for f in fams:
        classes = three_body_classes_for(f.qn, catalog, sets)
        out.append(dataclasses.replace(f, three_body_classes=classes, three_body_strength=float(strength))
                   if classes else f)
    return out


def phase_space_factor(M: float, m0: float, m1: float, m2: float) -> float:
    """``M R_3(M) / pi = 2 pi * integral q*(m; m0, m1) q(M; m, m2) dm`` on
    ``[m0 + m1, M - m2]`` [GeV^3], by the trapezoid on a grid uniform in
    ``phi`` with ``m = lo + (hi - lo) sin^2(phi)`` (the integrand vanishes at
    both ends); 0 below the threshold."""
    from .resonance_widths import decay_momentum
    lo, hi = float(m0) + float(m1), float(M) - float(m2)
    if not hi > lo:
        return 0.0
    n = _PHASE_SPACE_POINTS
    half_pi = 0.5 * math.pi
    total, prev = 0.0, 0.0
    for j in range(n):
        phi = half_pi * j / (n - 1)
        s, c = math.sin(phi), math.cos(phi)
        m = lo + (hi - lo) * s * s
        g = decay_momentum(m, m0, m1) * decay_momentum(M, m, m2) * 2.0 * (hi - lo) * s * c
        if j:
            total += 0.5 * (g + prev) * (half_pi / (n - 1))
        prev = g
    return 2.0 * math.pi * total


def three_body_isospin_share(parent_i: float, parent_i3x2: int, triple, catalog: HadronCatalog) -> float:
    """The probability that the charge state *triple* carries the parent's
    isospin, summed over the intermediate isospin of the first two daughters
    and over the distinct orderings of the charges among daughters of one
    multiplet (the kernel's two-body convention: pi+ pi- sums both orderings,
    N pi one)."""
    from .cascade_decay import _flavor_i3x2, _isospin_of, _multiplet_key
    from .resonance_decay import clebsch_gordan_squared
    base = tuple(int(d) for d in triple)
    keys = [_multiplet_key(d, catalog) for d in base]
    orders = {tuple(base[k] for k in perm) for perm in itertools.permutations(range(3))
              if all(keys[perm[j]] == keys[j] for j in range(3))}
    total = 0.0
    for order in orders:
        (i0, m0), (i1, m1), (i2, m2) = ((_isospin_of(d, catalog), _flavor_i3x2(d, catalog)) for d in order)
        m01 = m0 + m1
        k = abs(i0 - i1)
        while k <= i0 + i1 + 1e-9:
            if abs(m01) <= 2 * k + 1e-9:
                total += (clebsch_gordan_squared(k, m01, i0, m0, i1, m1)
                          * clebsch_gordan_squared(parent_i, parent_i3x2, k, m01, i2, m2))
            k += 1.0
    return total


def three_body_weight(strength: float, parent_mass: float, triple, masses, spins, iso: float) -> float:
    """``kappa_3 * prod(2J+1) * M R_3(M) / pi * C_iso`` (0 below threshold)."""
    phase = phase_space_factor(parent_mass, masses[0], masses[1], masses[2])
    return float(strength) * spins[0] * spins[1] * spins[2] * phase * float(iso)
