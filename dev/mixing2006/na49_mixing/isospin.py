"""Isospin reflection utilities for neutron-projectile reactions.

Beam substitution p -> n at fixed target corresponds to a full isospin
reflection, I3 -> -I3, applied to the projectile-side parent content:
p<->n, pi+<->pi-, rho+<->rho-, K+<->K0, Delta++<->Delta-,
Delta+<->Delta0, N*+<->N*0, with isoscalar states fixed. A one-unit
shift-down is the wrong rule: it would map N*0 onto a nonexistent N*-,
whereas reflection closes inside the existing multiplets.

The operation is exact for this framework because every supported
charge state carries its own Clebsch-weighted decay table
(``decays.py``), so reflecting the primary parent id automatically
selects the isospin-conjugate cascade. Combined with the generator's
projectile-to-target CMS mirroring, reflection reproduces the NA49
hemisphere identity pi+/pi-(p fragmentation) = pi-/pi+(n fragmentation).

Charge bookkeeping follows baryon number B and strangeness S of each
sector (Q = I3 + (B + S)/2 under I3 -> -I3):

- non-strange mesons (B=0, S=0):   Q(reflect(x)) = -Q(x)
- strange mesons (B=0, S=+1):      Q(reflect(x)) = 1 - Q(x)
- anti-strange mesons (B=0, S=-1): Q(reflect(x)) = -1 - Q(x)
- matter baryons, S=0 (N, Delta, N*, towers):
                                   Q(reflect(x)) = 1 - Q(x)
- matter baryons, S=-1 (Lambda, Sigma):
                                   Q(reflect(x)) = -Q(x)
- antibaryons (B=-1):              Q(reflect(x)) = -1 - Q(x)

Only primary parent ids are ever reflected here. Charged resonance
partners reached as cascade daughters need no mapping: their decays are
already registered per charge state.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

# Canonical I3 -> -I3 reflection over every supported primary identity.
# Involution is mandatory: reflect(reflect(x)) == x for every entry.
REFLECTION_MAP: Dict[int, int] = {
    # nucleons
    2212: 2112, 2112: 2212,
    -2212: -2112, -2112: -2212,
    # pions
    211: -211, -211: 211,
    111: 111,
    # rho(770) triplet
    213: -213, -213: 213,
    113: 113,
    # a2(1320)
    215: -215, -215: 215,
    115: 115,
    # rho3(1690)
    217: -217, -217: 217,
    117: 117,
    # b1(1235)
    10213: -10213, -10213: 10213,
    10113: 10113,
    # a1(1260)
    20213: -20213, -20213: 20213,
    20113: 20113,
    # isoscalar mesons (fixed)
    221: 221,    # eta
    223: 223,    # omega
    225: 225,    # f2(1270)
    229: 229,    # f4(2050)
    333: 333,    # phi(1020)
    30223: 30223,   # omega(1650)
    100223: 100223,  # omega(1420)
    9900113: 9900113,  # M_eff(450)0
    9900213: 9900213,  # M_eff(850)0
    9900313: 9900313,  # M_eff(1250)0
    # kaons: K+ <-> K0 (neutral K represented by K0S, pdg 310)
    321: 310, 310: 321,
    323: 313, 313: 323,     # K*(892)+ <-> K*(892)0
    325: 315, 315: 325,     # K2*(1430)+ <-> K2*(1430)0
    10323: 10313, 10313: 10323,  # K1(1270)+ <-> K1(1270)0
    -323: -313, -313: -323,   # K*(892)- <-> aK*(892)0
    -325: -315, -315: -325,   # K2*(1430)- <-> aK2*(1430)0
    # Delta(1232)
    2224: 1114, 1114: 2224,
    2214: 2114, 2114: 2214,
    # anti-Delta(1232) (anti rule: Q -> -1 - Q)
    -2224: -1114, -1114: -2224,
    -2214: -2114, -2114: -2214,
    # N*(1440), N*(1520), N(1535), N*(1680), N(1650), N(1720)
    12212: 12112, 12112: 12212,
    2216: 2116, 2116: 2216,
    102212: 102112, 102112: 102212,
    22212: 22112, 22112: 22212,
    122212: 122112, 122112: 122212,
    212214: 212114, 212114: 212214,
    # Delta(1600)
    202224: 201114, 201114: 202224,
    202214: 202114, 202114: 202214,
    # Delta(1700)
    122224: 121114, 121114: 122224,
    122214: 122114, 122114: 122214,
    # Sigma / Sigma(1385) / Lambda
    3222: 3112, 3112: 3222,
    3212: 3212,
    3224: 3114, 3114: 3224,
    3214: 3214,
    3122: 3122,
    102134: 102134,
    # effective N* towers (matter)
    9911201: 9912201, 9912201: 9911201,
    9921201: 9922201, 9922201: 9921201,
    9931201: 9932201, 9932201: 9931201,
    # effective anti-N* towers (anti rule: Q -> -1 - Q)
    -9911201: -9912201, -9912201: -9911201,
    -9921201: -9922201, -9922201: -9921201,
    -9931201: -9932201, -9932201: -9931201,
}

# Sector membership for the charge-rule validation. Every mapped id must
# belong to exactly one sector.
_MESONS = frozenset({
    111, 211, -211, 113, 213, -213, 115, 215, -215, 117, 217, -217,
    10113, 10213, -10213, 20113, 20213, -20213,
    221, 223, 225, 229, 333, 30223, 100223,
    9900113, 9900213, 9900313,
})
# Strange mesons carry charge offset S/2 with S = +1 (K, K*, K1, K2*)
# or S = -1 (anti-kaon partners).
_STRANGE_S1 = frozenset({310, 321, 313, 323, 315, 325, 10313, 10323})
_STRANGE_SBAR1 = frozenset({-313, -323, -315, -325})
_MATTER_BARYONS_S0 = frozenset({
    2212, 2112,
    1114, 2114, 2214, 2224,
    12112, 12212, 2116, 2216, 102112, 102212, 22112, 22212,
    122112, 122212, 212114, 212214,
    201114, 202114, 202214, 202224,
    121114, 122114, 122214, 122224,
    9911201, 9912201, 9921201, 9922201, 9931201, 9932201,
})
# Lambda and Sigma states carry S = -1: Q = I3 exactly.
_MATTER_BARYONS_SBAR1 = frozenset({
    3112, 3212, 3222, 3114, 3214, 3224, 3122, 102134,
})
_ANTIBARYONS = frozenset({
    -2212, -2112,
    -1114, -2114, -2214, -2224,
    -12112, -12212, -2116, -2216, -102112, -102212, -22112, -22212,
    -122112, -122212, -212114, -212214,
    -9911201, -9912201, -9921201, -9922201, -9931201, -9932201,
})

# Neutral anti-N* primaries without a registered charged anti-partner.
# These identities are proton-projectile-only: a neutron beam (or neutron
# target side) combined with such a primary must be rejected by config
# validation in parallel.py, because their reflected partners would need
# negatively charged anti-N* decay tables that do not exist.
ANTINEUTRON_PROHIBITED_PRIMARIES: frozenset[int] = frozenset({
    -12112, -22112, -102112, -122112, -212114, -2116,
})


def has_reflection(pdg_id: int) -> bool:
    """Return ``True`` if *pdg_id* has a defined I3 reflection."""
    return int(pdg_id) in REFLECTION_MAP


def reflect_pdg(pdg_id: int) -> int:
    """Return the I3-reflected partner of *pdg_id*.

    Raises ``KeyError`` with the particle name when the id is outside
    the curated map, so gaps surface loudly instead of silently
    generating the wrong charge content.
    """
    pid = int(pdg_id)
    try:
        return REFLECTION_MAP[pid]
    except KeyError:
        from .pdg_database import get_particle
        p = get_particle(pid)
        name = p.name if p is not None else "unknown"
        raise KeyError(
            f"No isospin reflection defined for PDG {pid} ({name}); "
            "extend isospin.REFLECTION_MAP and its sector sets."
        ) from None


def reflect_generator_identity(gen) -> int:
    """Reflect the loaded primary identity of *gen* in place.

    The mass law (Breit-Wigner parameters, tabulated histogram, or
    fixed-PDG mass law shape) comes from the generator card body and is
    untouched; only the parent id and its cached mass/charge are
    replaced. Returns the new id.
    """
    new_id = reflect_pdg(int(getattr(gen, "_id", 0)))
    gen._id = int(new_id)
    gen._refresh_id_cache()
    return int(new_id)


def primary_ids_from_cards(generators_dir: Optional[str] = None) -> set:
    """Collect the first-line PDG id of every bundled ``.gen`` card."""
    if generators_dir is None:
        generators_dir = str(Path(__file__).resolve().parent / "generators")
    out: set = set()
    for card in Path(generators_dir).rglob("*.gen"):
        token = card.read_text(errors="replace").split()[0]
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def validate_reflection_domain(extra_primary_ids: Iterable[int] = ()) -> None:
    """Fail loudly if any primary-reachable id lacks a reflection entry.

    Covers the bundled generator cards plus any caller-supplied extra
    primaries (benchmark selector maps pass their values here). Also
    asserts involution and the per-sector charge rules against the PDG
    table, so a hand-edit that breaks the physics fails immediately.
    """
    from .pdg_database import get_particle

    domain = set(primary_ids_from_cards()) | {int(i) for i in extra_primary_ids}
    missing = sorted(i for i in domain if i not in REFLECTION_MAP)
    uncurated = [i for i in missing if i not in ANTINEUTRON_PROHIBITED_PRIMARIES]
    if uncurated:
        raise AssertionError(
            f"Reflection map incomplete; missing primaries: {uncurated}"
        )

    for pid, rid in REFLECTION_MAP.items():
        assert REFLECTION_MAP[rid] == pid, f"reflection not involutive at {pid}"
        if pid in _MESONS:
            sector_rule = lambda q, r: r == -q
        elif pid in _STRANGE_S1:
            sector_rule = lambda q, r: r == 1 - q
        elif pid in _STRANGE_SBAR1:
            sector_rule = lambda q, r: r == -1 - q
        elif pid in _MATTER_BARYONS_S0:
            sector_rule = lambda q, r: r == 1 - q
        elif pid in _MATTER_BARYONS_SBAR1:
            sector_rule = lambda q, r: r == -q
        elif pid in _ANTIBARYONS:
            sector_rule = lambda q, r: r == -1 - q
        else:
            raise AssertionError(
                f"PDG {pid} mapped but not assigned to a sector set"
            )
        p, rp = get_particle(pid), get_particle(rid)
        assert p is not None and rp is not None, f"untabled pair {pid}->{rid}"
        assert sector_rule(float(p.charge), float(rp.charge)), (
            f"charge rule violated for {pid} (Q={p.charge}) -> "
            f"{rid} (Q={rp.charge})"
        )

    overlap = (
        (_MESONS & _MATTER_BARYONS_S0) | (_MESONS & _ANTIBARYONS)
        | (_MESONS & _STRANGE_S1) | (_MESONS & _STRANGE_SBAR1)
        | (_STRANGE_S1 & _STRANGE_SBAR1)
        | (_STRANGE_S1 & _MATTER_BARYONS_S0)
        | (_STRANGE_SBAR1 & _ANTIBARYONS)
        | (_MATTER_BARYONS_S0 & _ANTIBARYONS)
        | (_MATTER_BARYONS_SBAR1 & _MESONS) | (_MATTER_BARYONS_SBAR1 & _STRANGE_S1)
        | (_MATTER_BARYONS_SBAR1 & _MATTER_BARYONS_S0)
        | (_MATTER_BARYONS_SBAR1 & _ANTIBARYONS)
    )
    assert not overlap, f"sector sets overlap: {sorted(overlap)}"
