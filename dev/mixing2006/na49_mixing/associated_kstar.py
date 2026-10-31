"""The vector partners of the associated pairs, (Y, K*(892)) beside (Y, K)
(2026-10-21).

``KernelConfig.associated_kstar_pairs`` adds, beside every (hyperon, kaon)
class of a non-strange baryonic family (B != 0, S = 0; the isospin partners,
the Delta-like families and the conjugate towers), the class with the kaon's
vector partner: 321 -> 323, 311 -> 313 and the conjugates.

The non-strange baryonic lines carry (N, pi) beside (N, rho) and (Delta, pi)
beside (Delta, rho), and since R93 the strange baryonic lines carry (N, Kbar*)
beside (N, Kbar); their associated strangeness alone left as (Y, K). The
vector-to-pseudoscalar weight r_V of the flavor ledger applies to the new
classes as to every vector. At the R104 optimum the classes are flat with the
normalizations fitted (+15 +- 12 and -10 +- 15, two seed blocks, paired) and
raise the kaon yields (+148 and +103 with the yields fixed); the Gemini review
of 2026-10-21 (`.collab/`) named the omission.
"""

from __future__ import annotations

import dataclasses

from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["KAON_VECTOR_PARTNER", "associated_kstar_families"]

#: The kaon and its vector partner.
KAON_VECTOR_PARTNER = {321: 323, 311: 313, -321: -323, -311: -313}


def associated_kstar_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the (Y, K*) classes appended after their own: beside
    every (hyperon, kaon) class of a family with B != 0 and S = 0."""
    from .cascade_decay import _parse_class
    recs = {int(r.pdg_id): r for r in catalog.records()}
    missing = sorted(set(KAON_VECTOR_PARTNER.values()) - set(recs))
    if missing:
        raise ValueError(f"associated_kstar_pairs: the K*(892) records {missing} are absent")
    out = []
    for family in fams:
        if int(family.qn[0]) == 0 or int(family.qn[2]) != 0:
            out.append(family)
            continue
        add = []
        for c in family.allowed_decay_classes:
            d0, d1 = _parse_class(c)
            for y, k in ((d0, d1), (d1, d0)):
                if k in KAON_VECTOR_PARTNER and y in recs and int(recs[y].B) != 0 and int(recs[y].S) != 0:
                    add.append(f"({y},{KAON_VECTOR_PARTNER[k]})")
        add = [c for c in dict.fromkeys(add) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + add)) if add else family)
    return out
