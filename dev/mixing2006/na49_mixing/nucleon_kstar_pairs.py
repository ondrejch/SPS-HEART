"""The (N, Kbar*(892)) pairs of the strange baryonic lines (2026-10-12).

``KernelConfig.nucleon_kstar_pairs`` adds the class (N, Kbar*(892)) beside
every (N, Kbar) class of a strange baryonic family (B != 0, S != 0), the
isospin-completion families and the conjugate towers included; the
conjugate families take (Nbar, K*(892)) beside (Nbar, K).

The non-strange baryonic families carry (N, rho) beside (N, pi) and
(Delta, rho) beside (Delta, pi), while a strange baryonic vertex hands its
strangeness to a meson only as a K-bar.  The PDG 2025 tables list
N Kbar*(892) for Lambda(1810), Sigma(1750) and Sigma(1915).  At the R92
optimum the soft K- at x_F 0.25-0.55 stand at 0.67-0.77 of the data, and
the class lifts them to 0.71-0.93 (the fixed-kernel note,
``_knowledge/evidence/fixed-kernel-residual-mechanisms-2026-10-03.md``, the
probes at the R92 optimum).

Each class enters with the weight law of the family's other classes; the
isospin factor of the vertex sets its charge shares, as for (N, Kbar).
The class is written with the nucleon first, and a family that already
holds it keeps it once.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _parse_class
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["KSTAR_OF_KAON", "NUCLEONS", "nucleon_kstar_classes", "nucleon_kstar_families"]

#: The K*(892) of each kaon: K+ -> K*+, K0 -> K*0 and the conjugates.
KSTAR_OF_KAON = {321: 323, 311: 313, -321: -323, -311: -313}
#: The nucleons and antinucleons.
NUCLEONS = frozenset({2212, 2112, -2212, -2112})


def nucleon_kstar_classes(family: ContinuumFamily) -> tuple[str, ...]:
    """The (N, K*) classes the option adds to *family*: one beside each
    (N, K) class of a strange baryonic family, in class order, those the
    family already holds left out; empty for any other family."""
    if int(family.qn[0]) == 0 or int(family.qn[2]) == 0:
        return ()
    held = set(family.allowed_decay_classes)
    add = []
    for class_id in family.allowed_decay_classes:
        d0, d1 = _parse_class(class_id)
        for nucleon, kaon in ((d0, d1), (d1, d0)):
            if nucleon in NUCLEONS and kaon in KSTAR_OF_KAON:
                add.append(f"({nucleon},{KSTAR_OF_KAON[kaon]})")
    return tuple(c for c in dict.fromkeys(add) if c not in held)


def nucleon_kstar_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the (N, K*) classes appended.  The catalog must
    hold the four K*(892) records; a kernel whose strange baryonic
    families carry no (N, K) class is refused."""
    ids = {int(r.pdg_id) for r in catalog.records()}
    missing = sorted(set(KSTAR_OF_KAON.values()) - ids)
    if missing:
        raise ValueError(f"nucleon_kstar_pairs: the catalog lacks the K*(892) records {missing}")
    out, added = [], 0
    for family in fams:
        add = nucleon_kstar_classes(family)
        if add:
            added += len(add)
            family = dataclasses.replace(family, allowed_decay_classes=tuple(family.allowed_decay_classes) + add)
        out.append(family)
    if not added:
        raise ValueError("nucleon_kstar_pairs: no strange baryonic family carries an (N, Kbar) class")
    return out
