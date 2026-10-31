"""The baryon-resonance pairs of the neutral mesonic continuum (2026-10-10).

The neutral mesonic family pairs a baryon with an antibaryon only as p pbar
and n nbar.  A neutral mesonic state therefore emits every antibaryon of a
direct pair bare, and a bare antibaryon of a 3-4 GeV state takes a decay
momentum of 1.2-1.8 GeV/c.  With the charge-conjugate baryonic towers at
the R87 optimum the hard central pbar still run 1.19-1.28 of the data above
0.8 GeV/c at x_F < 0.25, and the soft central pbar 0.66.  The pairs of equal
isospin among N, N*(1440), N(1520), N(1535) (X Ybar and Y Xbar, the charge
states matched) and Delta(1232) Deltabar lower the pbar chi2 by 585 there
with the yields fixed (the fixed-kernel note, "The charge-conjugate
baryonic towers at the R87 optimum").

At the 2.0 GeV floor with ``continuation_q_min`` = 0.25 GeV/c the direct
offers never act: the neutral mesonic pion continuation opens at 2.30 GeV,
below it the lightest pair, N Nbar(1440), is closed at its pole-mass sum
of 2.378 GeV, and above it the topology cascades draw among the
continuation channels only.  The pairs act through their continuation
channels, which open at 2.99 GeV (the review of 2026-10-22).

``KernelConfig.baryon_resonance_pairs`` sets these pairs as the neutral
mesonic family's ``ContinuumFamily.resonance_pair_classes``; the
enumeration (``cascade_decay._enumerate_continuum_channels``) offers them
with the allowed decay classes' weight law, as direct pairs and as
continuation channels, and holds the baryon-pair share of the continuation
channels and of the direct pairs each at its value without them
(``cascade_decay._hold_pair_shares``), so the pair strength r_BBbar keeps
its meaning.  The pairs need the charge-conjugate towers: the antibaryon
resonance records and the antibaryonic continuum families come from
``KernelConfig.conjugate_towers``.  Off, the families are the committed
ones bit for bit.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _parse_class, _qn_of
from .conjugate_towers import charge_conjugate
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["MULTIPLETS", "GROUPS", "PAIR_FAMILY", "resonance_pair_classes", "resonance_pair_families"]

#: The baryon multiplets of the pairs (pdg ids by charge state).
MULTIPLETS: dict[str, tuple[int, ...]] = {
    "N": (2212, 2112),
    "N(1440)": (12212, 12112),
    "N(1520)": (2216, 2116),
    "N(1535)": (102212, 102112),
    "Delta(1232)": (2224, 2214, 2114, 1114),
}
#: The isospin groups: a pair takes a baryon and an antibaryon of one group.
GROUPS: tuple[tuple[str, ...], ...] = (("N", "N(1440)", "N(1520)", "N(1535)"), ("Delta(1232)",))
#: The family that carries the pairs.
PAIR_FAMILY = "mesonic_neutral"


def resonance_pair_classes(catalog: HadronCatalog) -> tuple[str, ...]:
    """Every class ``(X, Ybar)`` with X and Y of one isospin group and of
    equal charge, in the order of the groups and multiplets (the catalog
    must hold the antibaryon records, ``conjugate_towers``)."""
    out = []
    for group in GROUPS:
        for a in group:
            for b in group:
                for x in MULTIPLETS[a]:
                    for y in MULTIPLETS[b]:
                        if int(catalog.get(x).Q) == int(catalog.get(y).Q):
                            out.append(f"({x},{charge_conjugate(y, catalog)})")
    return tuple(out)


def resonance_pair_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the neutral mesonic family's resonance-pair
    classes set (a class already among its allowed decay classes is not
    repeated; order and the other families unchanged).  A class whose
    daughters do not sum to (0, 0, 0) is refused, and so is a kernel
    without the neutral mesonic family or without the antibaryon records."""
    if not any(str(f.family_id) == PAIR_FAMILY for f in fams):
        raise ValueError(f"resonance_pair_families: the kernel has no {PAIR_FAMILY!r} family")
    for name, ids in MULTIPLETS.items():
        for pdg in ids:
            try:
                catalog.get(-pdg)
            except KeyError:
                raise ValueError(f"resonance_pair_families: the catalog has no antibaryon of {name} ({-pdg}); "
                                 "the resonance pairs need the charge-conjugate towers") from None
    classes = resonance_pair_classes(catalog)
    out = []
    for family in fams:
        if str(family.family_id) == PAIR_FAMILY:
            for class_id in classes:
                q0, q1 = (_qn_of(catalog, d) for d in _parse_class(class_id))
                if tuple(int(a) + int(b) for a, b in zip(q0, q1)) != tuple(int(v) for v in family.qn):
                    raise ValueError(f"resonance_pair_families: class {class_id} does not carry the (B, Q, S) "
                                     f"{tuple(family.qn)} of {family.family_id}")
            extra = tuple(c for c in classes if c not in family.allowed_decay_classes)
            family = dataclasses.replace(
                family, resonance_pair_classes=extra,
                provenance=tuple(family.provenance) + (
                    "baryon-resonance pairs: " + ", ".join(extra) + " with the allowed classes' weight law, the "
                    "baryon-pair share of each channel set held (na49_mixing.resonance_pairs, 2026-10-10)",))
        out.append(family)
    return out
