"""The floor pools of the charged and strange mesonic continuum (2026-10-09).

A continuum state with no open continuation channel (below the
continuation floor) decays directly into two established hadrons of its
family's allowed decay classes.  The I = 1 charged mesonic families carry
light pairs only (pi pi, pi omega, pi rho, pi eta, K Kbar, K* Kbar), while
the neutral family's pool also holds the heavier pairs (rho omega, rho rho,
f2 pi, a2 pi, a1 pi, b1 pi, rho(1450) pi, phi pi, K* Kbar*).  At the R85
optimum the floor states of 2-3 GeV decaying into the light pairs make the
hard central tail (the pi+- at x_F 0-0.25 above 0.8 GeV/c 1.22-1.30 of the
data, the K- there 1.40-1.70; the fixed-kernel note, "The floor pools at the
R85 optimum").

A floor pool lists, per family, the heavier pairs that join the family's
direct pairs at a vertex with no open continuation channel
(``ContinuumFamily.floor_decay_classes``; the kernel weight law of the
direct pairs, ``cascade_decay._enumerate_continuum_channels``).  The
continuation steps keep the committed classes and their siblings: the
class list feeds the continuation siblings as well, and the same pairs
there dilute the kaon shares of every step (+1,457 at the R85 point with
the yields fixed, against -2,743 for the floor alone).

- ``"charged"``: the I = 1 partners of the neutral pool's heavier pairs in
  the charged and anticharged families;
- ``"strange"``: the K*(892) rho, K*(892) omega, K eta, K* eta, phi K,
  K1(1270) pi, K2*(1430) pi and K f2 pairs in the four strange mesonic
  families (the Q = -1 and Q = +1 families come from the strange mesonic
  isospin completion).
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _parse_class, _qn_of
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["SECTORS", "FLOOR_CLASSES", "floor_pool_families"]

#: The heavier pairs of the Q = +1 charged mesonic family (the I = 1
#: partners of the neutral pool's pairs; the anticharged family takes the
#: conjugates).
_CHARGED_Q1 = ("(213,113)", "(213,223)", "(213,221)", "(333,211)", "(225,211)", "(215,111)", "(115,211)", "(20213,111)",
               "(20113,211)", "(10213,111)", "(10113,211)", "(100213,111)", "(100113,211)", "(100223,211)", "(323,-313)",
               "(213,225)")
#: The heavier pairs of the S = -1 strange mesonic families at Q = 0 and
#: Q = -1 (the S = +1 families take the conjugates).
_STRANGE_Q0 = ("(-323,213)", "(-313,113)", "(-313,223)", "(-311,221)", "(-313,221)", "(333,-311)", "(-10323,211)",
               "(-10313,111)", "(-325,211)", "(-315,111)", "(-311,225)")
_STRANGE_QM1 = ("(-323,113)", "(-313,-213)", "(-323,223)", "(-321,221)", "(-323,221)", "(333,-321)", "(-10323,111)",
                "(-10313,-211)", "(-325,111)", "(-315,-211)", "(-321,225)")
#: The neutral non-strange mesons of the pools, their own antiparticles.
_SELF_CONJUGATE = frozenset({111, 113, 115, 221, 223, 225, 333, 20113, 10113, 100113, 100223})


def _conjugate(classes: tuple[str, ...]) -> tuple[str, ...]:
    out = []
    for class_id in classes:
        d0, d1 = _parse_class(class_id)
        out.append(f"({d0 if d0 in _SELF_CONJUGATE else -d0},{d1 if d1 in _SELF_CONJUGATE else -d1})")
    return tuple(out)


#: The floor classes per sector and family id.
FLOOR_CLASSES: dict[str, dict[str, tuple[str, ...]]] = {
    "charged": {"mesonic_charged": _CHARGED_Q1, "mesonic_anticharged": _conjugate(_CHARGED_Q1)},
    "strange": {"mesonic_strange": _STRANGE_Q0, "mesonic_strange_qm1": _STRANGE_QM1,
                "mesonic_antistrange": _conjugate(_STRANGE_Q0), "mesonic_antistrange_q1": _conjugate(_STRANGE_QM1)},
}
SECTORS = tuple(FLOOR_CLASSES)


def floor_pool_families(fams: list[ContinuumFamily], catalog: HadronCatalog,
                        sectors: tuple[str, ...] | list[str]) -> list[ContinuumFamily]:
    """The families with the selected sectors' floor classes set
    (order and the other families unchanged; a pair already among the
    family's allowed decay classes is not added).  A floor class whose
    daughters do not sum to the family's (B, Q, S) is refused; a sector
    family that is absent is refused too."""
    unknown = [s for s in sectors if s not in FLOOR_CLASSES]
    if unknown:
        raise ValueError(f"floor_pool_families: unknown sectors {unknown}; known {list(SECTORS)}")
    wanted: dict[str, tuple[str, ...]] = {}
    for sector in sectors:
        wanted.update(FLOOR_CLASSES[sector])
    present = {str(f.family_id) for f in fams}
    missing = sorted(set(wanted) - present)
    if missing:
        raise ValueError(f"floor_pool_families: the floor pools name absent families {missing}")
    out = []
    for family in fams:
        classes = wanted.get(str(family.family_id))
        if classes:
            target = tuple(int(v) for v in family.qn)
            for class_id in classes:
                q0, q1 = (_qn_of(catalog, d) for d in _parse_class(class_id))
                if tuple(int(a) + int(b) for a, b in zip(q0, q1)) != target:
                    raise ValueError(f"floor_pool_families: class {class_id} does not carry the (B, Q, S) {target} "
                                     f"of {family.family_id}")
            extra = tuple(c for c in classes if c not in family.allowed_decay_classes)
            family = dataclasses.replace(
                family, floor_decay_classes=extra,
                provenance=tuple(family.provenance) + (
                    "floor pool: the direct pairs " + ", ".join(extra) + " at a vertex with no open continuation "
                    "channel (na49_mixing.floor_pools, 2026-10-09)",))
        out.append(family)
    return out
