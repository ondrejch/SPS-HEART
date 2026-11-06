"""The baryon-antibaryon pairs of the mesonic continuum completed (2026-11-04).

The mesonic continuum creates a baryon-antibaryon pair only at the neutral
mesonic family, and only from the non-strange baryons: the allowed classes
p pbar and n nbar and the resonance pairs of ``na49_mixing.resonance_pairs``
(N, N(1440), N(1520), N(1535), Delta(1232)).  The charged, strange and
antistrange mesonic families carry no baryon pair.  A mesonic state therefore
never emits a hyperon-antihyperon pair, and every antihyperon of the cascade
comes from a second step (an antinucleon line that turns into an antihyperon
and a kaon).  At the R137 point the model gives 0.03 of the NA61/SHINE
anti-Xi+ yield and 0.015 of the anti-Xi(1530)0, while the Xi(1530)0, which the
leading baryon's strange lines feed, is at 0.89 (EPJC 80 (2020) 833).

``KernelConfig.baryon_pair_completion`` completes the pairs with one rule:
every pair (X, Ybar) of the ground-state octet (N, Lambda, Sigma, Xi) and
decuplet (Delta(1232), Sigma(1385), Xi(1530)) whose (B, Q, S) sums to the
mesonic family's.  The family's isospin factor then selects the couplings: at
the neutral family (I = 0) Lambda Lambdabar, Sigma Sigmabar and Xi Xibar; at
the strange family (I = 1/2) Lambda nbar, Sigma+ pbar, Sigma0 nbar, Xi0
Lambdabar and Xi- Sigmabar-, in which the cluster's strange quark passes to
the hyperon; at the charged family (I = 1) Sigma+ Lambdabar and Xi0 Xibar-
(and p nbar in the "all" scope).  With ``KernelConfig.omega_baryon`` the
Omega- joins the decuplet: Omega Omegabar at the neutral family, Omega-
Xibar+ at the strange family.  Each class pays the flavor ledger's gamma_s
per created s-sbar pair and r_BBbar per created baryon pair, so the
completion adds no parameter.

The weights follow the committed pairs' convention.  The octet-octet classes
(``completion_pair_classes``) enter with the weight law and set the
completion's share of the vertex, as p pbar and n nbar set the committed
share.  The classes with a decuplet member (``completion_resonance_classes``)
are held at that share, as the committed Delta(1232) and N* pairs are held at
the p pbar and n nbar share (``cascade_decay._hold_completion_share``).  Both
stay outside the committed pairs' hold, which works on the raw weights before
gamma_s: placed among the allowed classes, the hyperon pairs raised the held
N* and Delta pairs with them (a 4 GeV neutral state's non-strange pair share
went from 0.155 to 0.247 in the first build).  At a completed family each
baryon-pair continuation channel takes half its weight, so a pair class
carries one unit split over its two continuations (the X* Ybar and X Ybar*
steps), as the charge-conjugate halving gives at the neutral family
(``cascade_decay._halve_conjugate_pairs``).

The members "octet" (``KernelConfig.baryon_pair_completion_members``) leave the decuplet classes out, so the
excited hyperons of a pair come from the continuum partner's decays alone.  The hold of the decuplet classes moves
share between the strangeness sectors: the Lambda has no I = 0 decuplet partner and the Xi has, so at a 4 GeV neutral
state the S = -2 pairs take 0.29 of the S = -1 pairs with the decuplet held and 0.18 with the octet alone (R137 flavor
point; with the decuplet the probe gave anti-Xi(1530)0 4.5 times and Xibar+ 2.2 times NA61/SHINE).

The pair share ``KernelConfig.baryon_pair_share`` = "universal" states that a mesonic cluster's chance to make a
baryon pair depends on its mass and not on its charge or strangeness.  At every completed family but the neutral one,
the completion's pairs that create no s-sbar (p nbar at the charged family, Y Nbar at the strange family) take the
neutral family's committed pair share at the same mass, and the pairs that create one keep their weight-law ratio to
them; at the neutral family the rule is the treatment above.  The thin meson pools of the charged and strange
families (seven and fourteen classes against 52) otherwise hand their pairs 0.2-0.6 of the vertex.  The hold works
on the raw weights, before the flavor factors, as the importance reweighting needs.

The pair share "line" (2026-11-04) is the universal share on the clusters a baryonic line emits directly.  At the
completed families but the neutral one the completion's classes are offered only at a mesonic state whose parent is a
baryonic state (``CascadeState.parent``, |B| = 1), at the universal share; every other state of those families keeps
the committed vertex.  The neutral family's completion stays at every state.  Fischer's n+p pbar correlation (NPA 715
(2003) 118c, Fig. 8a) needs the pairs of the charged clusters a proton or neutron line emits (p* -> n M+ -> p nbar).
At the R138 point the universal share also gave the central charged clusters' pairs.  Their protons raised the
central protons 8-16%, while the forward pbar needed the committed pairs at their fitted rate (the probe bpr138).

The scopes: "strange" adds the classes with a strange member at every
mesonic family; "strange_neutral" adds them at the neutral family only (the
family with the full meson pool; the strange families carry seven meson
classes, so at a 4 GeV strange state the octet Y Nbar pairs alone take 0.59
of the continuation weight); "charged_nucleon" adds the neutral family's
classes of "strange_neutral" and the charged families' non-strange classes
(p nbar at the positive family, n pbar at the negative one, the I = 1 pairs
of Fischer's n+p pbar correlation), and nothing at the strange families;
"all" adds the non-strange classes the charged families lack (p nbar, the
Delta pairs) beside the strange classes everywhere.  Without the Omega- record
no pair carries S = -3.  Off, the families are the committed ones
bit for bit.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import _qn_of
from .conjugate_towers import charge_conjugate
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["OCTET", "DECUPLET", "OMEGA", "SCOPES", "MEMBERS", "SHARES", "NEUTRAL_FAMILY", "completion_classes", "baryon_pair_completion_families"]

#: The ground-state octet baryons (pdg ids by charge state).
OCTET: dict[str, tuple[int, ...]] = {
    "N": (2212, 2112),
    "Lambda": (3122,),
    "Sigma": (3222, 3212, 3112),
    "Xi": (3322, 3312),
}
#: The ground-state decuplet baryons in the catalog (the Omega- is absent).
DECUPLET: dict[str, tuple[int, ...]] = {
    "Delta(1232)": (2224, 2214, 2114, 1114),
    "Sigma(1385)": (3224, 3214, 3114),
    "Xi(1530)": (3324, 3314),
}
#: The S = -3 member, offered when the catalog holds it (``KernelConfig.omega_baryon``).
OMEGA = 3334
#: The scopes: the classes with a strange member at every mesonic family or
#: at the neutral family only, or every class at every mesonic family.
SCOPES = ("strange", "strange_neutral", "charged_nucleon", "all")
#: The members: the octet pairs alone, or the octet pairs and the pairs with a decuplet member (held at the octet
#: pairs' share).
MEMBERS = ("octet", "octet_decuplet")
#: The pair-share laws: None (the weight law), "universal" (at every completed family but the neutral one, the
#: completion's pairs that create no s-sbar take the neutral family's committed pair share at the same mass) or "line"
#: (the universal share, the completion offered at those families only on a state a baryonic line emits directly).
SHARES = (None, "universal", "line")
#: The family the "strange_neutral" scope completes.
NEUTRAL_FAMILY = "mesonic_neutral"


def _members(table: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(pdg for ids in table.values() for pdg in ids)


def _omega(catalog: HadronCatalog) -> tuple[int, ...]:
    """The Omega- as a decuplet member when the catalog holds it and its antibaryon."""
    try:
        catalog.get(OMEGA)
        catalog.get(-OMEGA)
    except KeyError:
        return ()
    return (OMEGA,)


def completion_classes(family: ContinuumFamily, catalog: HadronCatalog,
                       scope: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The completion's classes ``(X, Ybar)`` of one mesonic family, as
    (octet-octet classes, classes with a decuplet member) in the order of
    the members; a class the family already offers (allowed or resonance
    pair) is left out, and so is every class of a family the scope leaves
    out."""
    if scope not in SCOPES:
        raise ValueError(f"baryon_pair_completion: the scope must be one of {SCOPES}, got {scope!r}")
    if scope == "strange_neutral" and str(family.family_id) != NEUTRAL_FAMILY:
        return (), ()
    # "charged_nucleon" (2026-11-04): the neutral family as "strange_neutral", the charged families' non-strange
    # pairs alone (p nbar, n pbar), no strange family
    neutral = str(family.family_id) == NEUTRAL_FAMILY
    if scope == "charged_nucleon" and not neutral and int(family.qn[2]) != 0:
        return (), ()
    octet = set(_members(OCTET))
    members = _members(OCTET) + _members(DECUPLET) + _omega(catalog)
    have = set(family.allowed_decay_classes) | set(getattr(family, "resonance_pair_classes", ()) or ())
    want = tuple(int(v) for v in family.qn)
    ground, resonance = [], []
    for x in members:
        qx = _qn_of(catalog, x)
        for y in members:
            ybar = charge_conjugate(y, catalog)
            qy = _qn_of(catalog, ybar)
            if tuple(int(a) + int(b) for a, b in zip(qx, qy)) != want:
                continue
            non_strange = int(qx[2]) == 0 and int(qy[2]) == 0
            if scope == "charged_nucleon":
                if non_strange == neutral:
                    continue
            elif scope != "all" and non_strange:
                continue
            class_id = f"({x},{ybar})"
            if class_id in have:
                continue
            (ground if x in octet and y in octet else resonance).append(class_id)
    return tuple(ground), tuple(resonance)


def baryon_pair_completion_families(fams: list[ContinuumFamily], catalog: HadronCatalog,
                                    scope: str, members: str = "octet_decuplet",
                                    share: str | None = None) -> list[ContinuumFamily]:
    """The families with the mesonic families' pairs completed: the
    octet-octet classes set as a family's ``completion_pair_classes``, the
    classes with a decuplet member as its ``completion_resonance_classes``,
    and its ``baryon_pair_completion`` set to *scope* where it gains a class
    (order and the other families unchanged); *members* "octet" leaves the
    decuplet classes out; *share* "universal" sets every completed family's
    ``pair_share_reference`` to the neutral family but on the neutral
    family itself, and "line" also sets those families'
    ``completion_line_only``.  A catalog without a member or
    its antibaryon (the Xi baryons, the charge-conjugate towers) is
    refused."""
    if share not in SHARES:
        raise ValueError(f"baryon_pair_completion: the share must be one of {SHARES}, got {share!r}")
    if members not in MEMBERS:
        raise ValueError(f"baryon_pair_completion: the members must be one of {MEMBERS}, got {members!r}")
    for pdg in _members(OCTET) + _members(DECUPLET):
        for pid in (pdg, -pdg):
            try:
                catalog.get(pid)
            except KeyError:
                raise ValueError(f"baryon_pair_completion: the catalog has no record {pid}; the completion needs the "
                                 "Xi baryons and the charge-conjugate towers") from None
    out = []
    for family in fams:
        if int(family.qn[0]) == 0:
            ground, resonance = completion_classes(family, catalog, scope)
            if members == "octet":
                resonance = ()
            if ground or resonance:
                family = dataclasses.replace(
                    family, completion_pair_classes=ground, completion_resonance_classes=resonance,
                    baryon_pair_completion=scope,
                    pair_share_reference=(NEUTRAL_FAMILY if share in ("universal", "line")
                                          and str(family.family_id) != NEUTRAL_FAMILY else None),
                    completion_line_only=(share == "line" and str(family.family_id) != NEUTRAL_FAMILY),
                    provenance=tuple(family.provenance) + (
                        f"baryon-pair completion ({scope}): " + ", ".join(ground) + " with the weight law; "
                        + ", ".join(resonance) + " held at their share (na49_mixing.baryon_pair_completion, "
                        "2026-11-04)",))
        out.append(family)
    return out
