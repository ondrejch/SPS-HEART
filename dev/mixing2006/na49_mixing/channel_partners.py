"""The isospin-allowed partner channels of the baryonic lines (2026-10-15).

Two kernel options complete the baryonic families' classes with channels
that isospin allows and the committed classes leave out:

- ``KernelConfig.hyperon_meson_partners``: beside every (Y, pi0) class of a
  strange baryonic family the classes (Y, rho0), (Y, omega) and (Y, eta),
  and beside every (Y, pi+-) class (Y, rho+-), Y a hyperon record, both
  towers; the enumeration's isospin factor removes the forbidden ones.
  The non-strange baryonic families carry the vector and eta partners of
  their pions, (N, rho), (N, omega), (N, eta), and the strange ones carried
  none.
- ``KernelConfig.delta_isoscalar_pairs``: the (Delta(1232), omega),
  (Delta, eta), (Delta, f0(500)) and (Delta, f0(980)) pairs at the four
  I = 3/2 families and their conjugates (3/2 x 0), the analogues of the
  (N, omega), (N, eta), (N, f0) classes of the I = 1/2 families.

- ``KernelConfig.charged_scalar_pairs``: the isospin partners of the light
  scalar mesons' neutral-pool pairs in the charged mesonic pools, (f0(980),
  pi+-), (a0(980)+-, pi0) and (a0(980)0, pi+-), the light scalar mesons'
  classes rotated to charge +-1.

The decision is the model decision register
(``_knowledge/paper/model-decision-register.md``, the omissions decided
inside the noise) and the probes the fixed-kernel note
(``PROBE_STRANGE_VECTORS = yv`` and ``PROBE_DELTA_I0 = all``, the probes
at the R92 optimum and the completion probes at the R97 optimum), of which
these options are the kernel forms: the classes join at the end of the
kernel load, after the (N, Kbar*) pairs, as the probes add them.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _parse_class
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["PION_PARTNERS", "DELTA_LIKE", "ISOSCALAR_MESONS", "CHARGED_SCALAR_PAIRS", "hyperon_meson_partner_families",
           "delta_isoscalar_pair_families", "charged_scalar_pair_families"]

#: The charged-pool classes of the light scalar mesons (f0(980) 9010221, a0(980) 9000111 and +-9000211).
CHARGED_SCALAR_PAIRS = {"mesonic_charged": ("(9010221,211)", "(9000211,111)", "(9000111,211)"),
                        "mesonic_anticharged": ("(9010221,-211)", "(-9000211,111)", "(9000111,-211)")}

#: The partners of each pion: rho0, omega, eta beside pi0; rho+- beside pi+-.
PION_PARTNERS = {111: (113, 223, 221), 211: (213,), -211: (-213,)}
#: The Delta(1232) of each I = 3/2 baryonic family.
DELTA_LIKE = {"baryonic_np_q2": 2224, "baryonic_np_delta_q1": 2214, "baryonic_np_delta_q0": 2114, "baryonic_np_qm1": 1114}
DELTA_LIKE.update({f"anti_{k}": -v for k, v in list(DELTA_LIKE.items())})
#: The isoscalar mesons of the (Delta, I = 0) pairs: omega, eta, f0(500), f0(980).
ISOSCALAR_MESONS = (223, 221, 9000221, 9010221)


def hyperon_meson_partner_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the (Y, rho/omega/eta) partners appended at every
    strange baryonic family (a class already held is not repeated)."""
    recs = {int(r.pdg_id): r for r in catalog.records()}
    out, found = [], 0
    for family in fams:
        b, s = int(family.qn[0]), int(family.qn[2])
        add = []
        if b != 0 and s != 0:
            for c in family.allowed_decay_classes:
                d0, d1 = _parse_class(c)
                for y, k in ((d0, d1), (d1, d0)):
                    if k in PION_PARTNERS and y in recs and int(recs[y].B) != 0 and int(recs[y].S) != 0:
                        add.extend(f"({y},{v})" for v in PION_PARTNERS[k])
        found += len(add)
        add = [c for c in dict.fromkeys(add) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + add)) if add else family)
    if not found:
        raise ValueError("hyperon_meson_partners: no (Y, pi) class in the kernel's strange baryonic families")
    return out


def delta_isoscalar_pair_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The four I = 3/2 families and their conjugates with the (Delta,
    I = 0 meson) pairs appended.  A kernel without the isoscalar mesons (the
    isoscalar dipion and the light scalar mesons) or without the I = 3/2
    families is refused."""
    have = {int(r.pdg_id) for r in catalog.records()}
    missing = [m for m in ISOSCALAR_MESONS if m not in have]
    if missing:
        raise ValueError(f"delta_isoscalar_pairs: the catalog lacks the isoscalar mesons {missing}")
    ids = {str(f.family_id) for f in fams}
    absent = sorted(set(DELTA_LIKE) - ids)
    if absent:
        raise ValueError(f"delta_isoscalar_pairs: the I = 3/2 families {absent} are absent")
    out = []
    for family in fams:
        delta = DELTA_LIKE.get(str(family.family_id))
        if delta is not None:
            extra = [f"({delta},{m})" for m in ISOSCALAR_MESONS]
            held = sorted(set(extra) & set(family.allowed_decay_classes))
            if held:
                raise ValueError(f"delta_isoscalar_pairs: family {family.family_id!r} already holds {held}")
            family = dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
                list(family.allowed_decay_classes) + extra))
        out.append(family)
    return out


def charged_scalar_pair_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The charged mesonic families with the scalar partners appended.  A
    kernel without the light scalar mesons is refused."""
    have = {int(r.pdg_id) for r in catalog.records()}
    if not {9010221, 9000111, 9000211, -9000211} <= have:
        raise ValueError("charged_scalar_pairs: the catalog lacks the light scalar mesons (KernelConfig.light_scalar_mesons)")
    out = []
    for family in fams:
        extra = [c for c in CHARGED_SCALAR_PAIRS.get(str(family.family_id), ()) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + extra)) if extra else family)
    return out
