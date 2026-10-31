"""The isospin completion of the continuum towers (Round 8, the prerequisite of
tasks 45 to 47; ``TASK-20261004-r8-isospin-complete-towers``, 2026-10-04).

The committed configuration declares one continuum family per (B, S)
charge sector for the baryons and the strange mesons: ``baryonic_np``
(1, 1, 0), ``baryonic_strange`` (1, 0, -1), ``mesonic_strange`` (0, 0, -1)
and ``mesonic_antistrange`` (0, 0, 1) (the non-strange mesons carry the
complete triplet).  A continuum line therefore never changes its charge
state within its sector: a p* line sheds only neutral mesons while it stays
a baryonic continuum, so a fast pi- needs two-step chains.  At the R74
optimum the completed non-strange baryonic tower raised the forward pi- by
a factor 1.7 to 2.8 (``_knowledge/evidence/fixed-kernel-residual-
mechanisms-2026-10-03.md``, the forward isospin section).

:func:`isospin_partner_families` adds, per selected sector, one family for
every charge state the quark model allows and the configuration lacks:

==================  ========  ==============================  =============
sector              (B, S)    added charges (isospin)          source family
==================  ========  ==============================  =============
baryonic            (1, 0)    -1 (3/2), 0 (1/2), +2 (3/2)      baryonic_np
strange_baryonic    (1, -1)   -1 (1), +1 (1)                   baryonic_strange
strange_mesonic     (0, -1)   -1 (1/2)                         mesonic_strange
strange_mesonic     (0, +1)   +1 (1/2)                         mesonic_antistrange
==================  ========  ==============================  =============

The isospin is the configuration's minimal-|I3| rule (I = |I3| with
2 I3 = 2Q - B - S).  Each added family copies its source family (the
spectral shape, the kernel options of the run) and takes the source's
decay classes rotated within the isospin multiplets of their daughters
(the ``resonance_decay`` multiplet key), keeping the classes whose
(B, Q, S) sum equals the new family's.  The added families are decay
daughters only (``active = False``): the source sampling is unchanged.
The committed configuration is unchanged; a run selects the sectors through
``na49_mixing.kernel_config.KernelConfig.isospin_completion``.

The Delta-like lines (2026-10-04)
---------------------------------

The isovector exchange (``na49_mixing.isovector_exchange``) leaves a line at
I = 3/2 with charge +1 or 0 in two thirds of its t = 0 and in one third of
its t = -1 lines.  The minimal-|I3| rule gives those charges I = 1/2 (the
``baryonic_np`` and ``baryonic_np_q0`` families), so
:func:`delta_line_families` adds the two I = 3/2 families
``baryonic_np_delta_q1`` and ``baryonic_np_delta_q0`` as cascade roots:
the ``baryonic_np`` classes rotated into the line's charge, the isospin 3/2
(the Clebsch-Gordan shares then select the channels of an I = 3/2 parent),
``active = False``.  As continuation daughters the first family of the
quantum numbers stays in use (the I = 1/2 family), so a Delta-like state
reaches the I = 3/2 tower only at charges +2 and -1.  A run selects them
through ``KernelConfig.delta_lines``.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _parse_class, _qn_of
from .hadron_catalog import HadronCatalog
from .resonance_decay import _multiplet_key
from .spectral_density import ContinuumFamily

__all__ = ["DELTA_LINE_CHARGES", "ISOVECTOR_NEUTRAL_MESONIC_ID", "ISOVECTOR_PAIR_CLASSES", "SECTORS", "delta_line_families",
           "isospin_partner_families", "isovector_meson_pair_families", "isovector_neutral_mesonic_families"]

#: The sectors: (source family id, (B, S), the added charges).
SECTORS: dict[str, tuple[tuple[str, tuple[int, int], tuple[int, ...]], ...]] = {
    "baryonic": (("baryonic_np", (1, 0), (-1, 0, 2)),),
    "strange_baryonic": (("baryonic_strange", (1, -1), (-1, 1)),),
    "strange_mesonic": (("mesonic_strange", (0, -1), (-1,)), ("mesonic_antistrange", (0, 1), (1,))),
}

#: The charges of the Delta-like (I = 3/2) line families.
DELTA_LINE_CHARGES: tuple[int, ...] = (1, 0)

#: The baryon-pair classes of the I = 1 charged mesonic continuum (2026-10-05):
#: a M+ decays to p nbar, a M- to n pbar.  NA49 (Fischer 2002, sect. 4) found
#: the pairs produced dominantly in I = 1 combinations, p nbar preferred over
#: p pbar for a proton projectile; the committed configuration carries pair
#: classes only in the I = 0 neutral family, so a proton line cannot prefer
#: nbar.  With these classes a p* emits n M+ (Clebsch-Gordan share 2/3) and
#: the M+ decays to the charge-correlated pair.
ISOVECTOR_PAIR_CLASSES: dict[str, tuple[str, ...]] = {
    "mesonic_charged": ("(2212,-2112)",),
    "mesonic_anticharged": ("(2112,-2212)",),
}

#: The weak neutral kaon eigenstates stay out of the rotation pools (the
#: classes carry the flavor states 311 and -311).
_WEAK_EIGENSTATES = frozenset({310, 130})


def _family_id(source: str, charge: int) -> str:
    return f"{source}_q{charge}" if charge >= 0 else f"{source}_qm{-charge}"


def _members(catalog: HadronCatalog) -> dict:
    """The isospin multiplets of the catalog and the stable states:
    multiplet key -> [(pdg, (B, Q, S))] in pdg order."""
    pool = set(int(r.pdg_id) for r in catalog) | set(int(p) for p in catalog.stable_states())
    pool |= {-p for p in list(pool)}
    members: dict = {}
    for pdg in sorted(pool):
        if pdg in _WEAK_EIGENSTATES:
            continue
        try:
            key, qn = _multiplet_key(pdg, catalog), _qn_of(catalog, pdg)
        except (KeyError, ValueError):
            continue
        members.setdefault(key, []).append((pdg, tuple(int(v) for v in qn)))
    return members


def _rotated_classes(source: ContinuumFamily, target_qn: tuple, members: dict, catalog: HadronCatalog) -> tuple[str, ...]:
    classes: list[str] = []
    for class_id in source.allowed_decay_classes:
        d0, d1 = _parse_class(class_id)
        for p0, q0 in members.get(_multiplet_key(d0, catalog), [(d0, _qn_of(catalog, d0))]):
            for p1, q1 in members.get(_multiplet_key(d1, catalog), [(d1, _qn_of(catalog, d1))]):
                if (q0[0] + q1[0], q0[1] + q1[1], q0[2] + q1[2]) == target_qn:
                    rotated = f"({p0},{p1})"
                    if rotated not in classes:
                        classes.append(rotated)
    return tuple(classes)


def isospin_partner_families(fams: list[ContinuumFamily], catalog: HadronCatalog,
                             sectors: tuple[str, ...] | list[str]) -> list[ContinuumFamily]:
    """The families that complete the selected sectors (see the module
    docstring), in sector and charge order.  A sector whose source family
    is absent, or a charge state already declared, adds nothing."""
    unknown = [s for s in sectors if s not in SECTORS]
    if unknown:
        raise ValueError(f"isospin_partner_families: unknown sectors {unknown}; known {sorted(SECTORS)}")
    declared = {tuple(int(v) for v in f.qn) for f in fams}
    by_id = {str(f.family_id): f for f in fams}
    members = _members(catalog)
    added: list[ContinuumFamily] = []
    for sector in sectors:
        for source_id, (b, s), charges in SECTORS[sector]:
            source = by_id.get(source_id)
            if source is None:
                continue
            for q in charges:
                qn = (b, q, s)
                if qn in declared:
                    continue
                classes = _rotated_classes(source, qn, members, catalog)
                if not classes:
                    raise ValueError(f"isospin_partner_families: no rotated class for {source_id} at charge {q}")
                isospin = abs(2 * q - b - s) / 2.0
                provenance = tuple(source.provenance) + (
                    f"isospin completion: charge {q} partner of {source_id}, classes rotated within the isospin "
                    f"multiplets, I = {isospin} (minimal |I3|) (na49_mixing.continuum_isospin, 2026-10-04)",)
                added.append(dataclasses.replace(source, family_id=_family_id(source_id, q), qn=qn, allowed_decay_classes=classes,
                                                 isospin=isospin, active=False, provenance=provenance))
                declared.add(qn)
    return added


def delta_line_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The I = 3/2 root families of the isovector exchange's lines at the
    charges :data:`DELTA_LINE_CHARGES` (see the module docstring), from the
    ``baryonic_np`` family.  A charge with an I = 3/2 family already
    declared adds nothing."""
    by_id = {str(f.family_id): f for f in fams}
    source = by_id.get("baryonic_np")
    if source is None:
        raise ValueError("delta_line_families: the baryonic_np family is absent")
    members = _members(catalog)
    added: list[ContinuumFamily] = []
    for q in DELTA_LINE_CHARGES:
        qn = (1, q, 0)
        if any(tuple(int(v) for v in f.qn) == qn and f.isospin == 1.5 for f in fams):
            continue
        classes = _rotated_classes(source, qn, members, catalog)
        provenance = tuple(source.provenance) + (
            f"Delta-like line: the charge {q} I = 3/2 root of the isovector exchange, the baryonic_np classes "
            "rotated within the isospin multiplets (na49_mixing.continuum_isospin, 2026-10-04)",)
        added.append(dataclasses.replace(source, family_id=f"baryonic_np_delta_q{q}", qn=qn,
                                         allowed_decay_classes=classes, isospin=1.5, active=False,
                                         provenance=provenance))
    return added


#: The family id of the I = 1 neutral mesonic continuum (2026-10-06).
ISOVECTOR_NEUTRAL_MESONIC_ID = "mesonic_neutral_i1"


def isovector_neutral_mesonic_families(fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The I = 1 neutral mesonic continuum (2026-10-06; card
    TASK-20261005-r8-isovector-mesonic-completion, step 2): the neutral
    mesonic family at I = 1, I3 = 0, with the neutral family's concrete
    pair classes, whose Clebsch-Gordan shares at I = 1 select the allowed
    ones (omega pi0 and rho+- pi-+ open, pi0 pi0 and rho0 pi0 closed).
    Without it an I = 0 neutral state cannot emit a rho0, because isospin
    forbids rho0 plus an I = 0 daughter.  The family is reached as a
    continuation daughter only under the isospin-resolved daughter rule
    (``continuation_daughters = "isospin"``); like the partner and
    Delta-line families it is declared inactive, so the sampled root
    surfaces (``active_families``, ``continuum_density``, the blend
    routing) keep one neutral mesonic family (2026-10-07; review
    2026-10-06, Grok H1)."""
    source = next((f for f in fams if str(f.family_id) == "mesonic_neutral"), None)
    if source is None or source.isospin is None:
        raise ValueError("isovector_neutral_mesonic_families: the neutral mesonic family with an isospin rule is required")
    if any(str(f.family_id) == ISOVECTOR_NEUTRAL_MESONIC_ID for f in fams):
        raise ValueError("isovector_neutral_mesonic_families: the I = 1 neutral family is already present")
    return [dataclasses.replace(
        source, family_id=ISOVECTOR_NEUTRAL_MESONIC_ID, isospin=1.0, active=False,
        provenance=tuple(source.provenance) + (
            "isovector neutral mesonic continuum: the neutral family at I = 1, I3 = 0 (na49_mixing.continuum_isospin, "
            "2026-10-06; the rho0/omega ratio of Fischer 2022 Table 9)",))]


def isovector_meson_pair_families(fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The families with the baryon-pair classes of
    :data:`ISOVECTOR_PAIR_CLASSES` appended to the I = 1 charged mesonic
    families (order and the other families unchanged; a class already
    present is not added twice)."""
    out = []
    for family in fams:
        extra = tuple(c for c in ISOVECTOR_PAIR_CLASSES.get(str(family.family_id), ())
                      if c not in family.allowed_decay_classes)
        if extra:
            family = dataclasses.replace(
                family, allowed_decay_classes=tuple(family.allowed_decay_classes) + extra,
                provenance=tuple(family.provenance) + (
                    "isovector meson pairs: the I = 1 baryon-pair classes " + ", ".join(extra)
                    + " (na49_mixing.continuum_isospin, 2026-10-05; Fischer 2002 sect. 4)",))
        out.append(family)
    return out

