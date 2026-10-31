"""The S = -2 excited baryonic continuum (2026-10-30).

``KernelConfig.xi_tower`` adds the step of the cascade N* -> K Y*,
Y* -> K Xi*, Xi* -> Xi pi that the kernel lacks: two continuum families,
an isospin doublet at (B, Q, S) = (1, 0, -2) and (1, -1, -2), with the
strange baryonic family's spectral law, floor and options, decaying through
(Xi, pi), (Xi(1530), pi), (Lambda, Kbar), (Sigma, Kbar) and
(Sigma(1385), Kbar), and their antibaryonic conjugates for the anti-Xi.
The families are not roots.  The (Xi, K) classes the strange baryonic
lines carry under ``xi_baryons`` then open (Xi_cont, K) continuation
channels by the families' quantum numbers, at the flavor ledger's gamma_s
per created s-sbar pair, so the step adds no parameter.

At the R128 point the step raises the Xi- from 0.14 to 0.25 of the
NA61/SHINE yield (EPJC 80 (2020) 833) and the anti-Xi+ from 0.02 to 0.03,
and costs the six spectra -17 +- 17 and -2 +- 15 in chi2 on two paired
seed blocks (inside the noise); it is adopted on physics.  The families
are built as the probe hook PROBE_XI_TOWER of 2026-10-18 built them, at
the end of the kernel load, so the option reproduces the probe's events.
The decision is the model decision register
(``_knowledge/paper/model-decision-register.md``, section 4, item 6).
"""

from __future__ import annotations

import dataclasses

from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["XI_TOWER_FAMILIES", "xi_tower_classes", "xi_tower_families"]

#: The family ids, the baryonic doublet and its conjugates.
XI_TOWER_FAMILIES = ("baryonic_xi_q0", "baryonic_xi_qm1", "anti_baryonic_xi_q0", "anti_baryonic_xi_qm1")
#: The template family whose spectral law, floor and options the S = -2 families take.
_TEMPLATE = "baryonic_strange"


def xi_tower_classes() -> dict[str, tuple[tuple[int, ...], list[str]]]:
    """The baryonic families' quantum numbers and direct classes (the
    conjugates follow by :func:`_conjugate`)."""
    q0 = ["(3322,111)", "(3312,211)", "(3324,111)", "(3314,211)", "(3122,-311)", "(3212,-311)", "(3222,-321)",
          "(3214,-311)", "(3224,-321)"]
    qm1 = ["(3312,111)", "(3322,-211)", "(3314,111)", "(3324,-211)", "(3122,-321)", "(3212,-321)", "(3112,-311)",
           "(3114,-311)", "(3214,-321)"]
    return {"baryonic_xi_q0": ((1, 0, -2), q0), "baryonic_xi_qm1": ((1, -1, -2), qm1)}


def _conjugate(cls: str) -> str:
    return "(" + ",".join(t if int(t) == 111 else str(-int(t)) for t in cls.strip("()").split(",")) + ")"


def xi_tower_families(catalog: HadronCatalog, fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The families with the S = -2 doublet and its conjugates appended.
    A kernel without the strange baryonic family, without a record a class
    names (the Xi baryons and the antibaryons of the conjugate towers), or
    already holding one of the ids is refused."""
    base = next((f for f in fams if str(f.family_id) == _TEMPLATE), None)
    if base is None:
        raise ValueError(f"xi_tower: the template family {_TEMPLATE!r} is absent")
    held = {str(f.family_id) for f in fams} & set(XI_TOWER_FAMILIES)
    if held:
        raise ValueError(f"xi_tower: the kernel already holds {sorted(held)}")
    have = {int(r.pdg_id) for r in catalog.records()} | {int(p) for p in catalog.stable_states()}
    new = []
    for fid, (qn, classes) in xi_tower_classes().items():
        conj = [_conjugate(c) for c in classes]
        missing = sorted({int(t) for c in classes + conj for t in c.strip("()").split(",")} - have)
        if missing:
            raise ValueError(f"xi_tower: the catalog lacks {missing} (the option needs xi_baryons and the conjugate towers)")
        new.append(dataclasses.replace(base, family_id=fid, qn=qn, allowed_decay_classes=tuple(classes), active=False,
                                       isospin=0.5))
        new.append(dataclasses.replace(base, family_id="anti_" + fid, qn=tuple(-v for v in qn),
                                       allowed_decay_classes=tuple(conj), active=False, isospin=0.5))
    return list(fams) + new
