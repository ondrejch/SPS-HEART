"""The PDG 2025 N and Delta resonances between N(1720) and the 2.0 GeV
continuation floor (2026-10-21).

``KernelConfig.pdg_nucleon_resonances`` adds N(1675), N(1700), N(1710),
N(1875), N(1880), N(1895), N(1900), Delta(1620), Delta(1900), Delta(1905),
Delta(1910), Delta(1920) and Delta(1950) at their PDG 2025 Breit-Wigner masses
and widths and their listed modes
(``config/v2_singlet/pdg_nucleon_resonances_pdg2025.json``), and beside every
(N*, pi) class of a non-strange baryonic line the classes of the new states
with a pion.

The committed catalog stops at N(1720) and Delta(1700), so every
non-diffractive line drawn between 1.72 and 2.0 GeV, which decays through the
duality pick of the low-mass records, falls on the tails of the lower states,
whose N pi fractions reach 51-65%. The PDG states of that band decay far less
to N pi and much more to N omega, N eta, N eta', N sigma, Lambda K and Sigma K.
The completion follows the rule of the PDG hyperons of R98 (the model decision
register, section 3.3): established states at their measured values, no
parameter. The Gemini review of 2026-10-21 (`.collab/`) named the gap.

The fractions follow the N* range rule (one common position inside the PDG
ranges, :func:`na49_mixing.isoscalar_dipion.range_position_fractions`); each
mode's charge split is the isospin Clebsch-Gordan share of the parent's charge
state over the daughters' multiplets, and the three-body N pi pi of
Delta(1950) the share of :func:`na49_mixing.three_body_endings.
three_body_isospin_share`. The records take an N* or Delta record of the
catalog as their template. The ids are private: the catalog's legacy
numbering already holds some of the PDG codes. The option enters before the
conjugate towers, which conjugate its records and classes.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["PDG_NUCLEON_RESONANCES_REL", "load_pdg_nucleon_resonances", "pdg_nucleon_resonances_catalog",
           "pdg_nucleon_resonances_families", "new_resonance_ids", "MODE_KINDS"]

#: The configuration, relative to the nested repository root.
PDG_NUCLEON_RESONANCES_REL = Path("config") / "v2_singlet" / "pdg_nucleon_resonances_pdg2025.json"

#: Each mode kind: the representative daughters whose isospin multiplets the
#: charge split runs over (two for a two-body mode, three for a three-body one).
MODE_KINDS = {
    "Npi": (2212, 211), "Neta": (2212, 221), "Netap": (2212, 331), "Nomega": (2212, 223), "Nrho": (2212, 213),
    "Nsigma": (2212, 9000221), "Na0": (2212, 9000211), "Dpi": (2224, 211), "Deta": (2224, 221), "LK": (3122, 321),
    "SK": (3222, 321), "LKstar": (3122, 323), "N1440pi": (12212, 211), "N1520pi": (2216, 211),
    "N1535pi": (102212, 211), "N1680pi": (22212, 211), "Npipi": (2212, 211, 211),
}
_TEMPLATES = {0.5: 22212, 1.5: 122224}  # N*(1680)+, Delta(1700)++
_PIONS = {1: 211, 0: 111, -1: -211}
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0


def load_pdg_nucleon_resonances(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / PDG_NUCLEON_RESONANCES_REL).read_text(encoding="utf-8"))


def new_resonance_ids(config: dict) -> tuple[int, ...]:
    """Every charge state's id of the configuration."""
    return tuple(int(p) for mult in config["multiplets"] for p in mult["pdg_ids"].values())


def _tag(name: str, field: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(PDG_NUCLEON_RESONANCES_REL), section_or_table=name, field=field,
                         accessed_on="2026-10-21", transcribed_value=name, transcriber="na49-res PDG N and Delta option")


#: The isosinglets that join the catalog after this option (the eta'(958) at
#: the end of the kernel load): a singlet's split is trivial.
_LATE_SINGLETS = (331,)


def _members(pdg: int, catalog: HadronCatalog) -> tuple[int, ...]:
    from .resonance_decay import multiplet_charge_states
    if int(pdg) in _LATE_SINGLETS:
        return (int(pdg),)
    return tuple(int(m) for m in multiplet_charge_states(int(pdg), catalog))


def _iso(pdg: int, catalog: HadronCatalog) -> tuple[float, int]:
    from .cascade_decay import _flavor_i3x2, _isospin_of
    if int(pdg) in _LATE_SINGLETS:
        return 0.0, 0
    return float(_isospin_of(int(pdg), catalog)), int(_flavor_i3x2(int(pdg), catalog))


def _split(parent_i: float, parent_i3x2: int, kind: str, catalog: HadronCatalog) -> list[tuple[tuple[int, ...], float]]:
    """The charge states of a mode and their shares for one parent charge state
    (normalized; an isospin-forbidden mode raises)."""
    from .resonance_decay import clebsch_gordan_squared
    reps = MODE_KINDS[kind]
    out = []
    if len(reps) == 2:
        for a, b in itertools.product(_members(reps[0], catalog), _members(reps[1], catalog)):
            (ia, ma), (ib, mb) = _iso(a, catalog), _iso(b, catalog)
            if ma + mb != parent_i3x2:
                continue
            w = clebsch_gordan_squared(parent_i, parent_i3x2, ia, ma, ib, mb)
            if w > 0.0:
                out.append(((a, b), float(w)))
    else:
        from .three_body_endings import three_body_isospin_share
        seen = set()
        for combo in itertools.product(*(_members(r, catalog) for r in reps)):
            key = tuple(sorted(combo))
            if key in seen or sum(_iso(d, catalog)[1] for d in combo) != parent_i3x2:
                continue
            seen.add(key)
            w = three_body_isospin_share(parent_i, parent_i3x2, combo, catalog)
            if w > 0.0:
                out.append((tuple(combo), float(w)))
    total = math.fsum(w for _, w in out)
    if not total > 0.0:
        raise ValueError(f"pdg_nucleon_resonances: the mode {kind} is isospin-forbidden for I = {parent_i}")
    return [(d, w / total) for d, w in out]


def _pole(catalog: HadronCatalog, pdg: int) -> float:
    from .cascade_decay import _daughter_pole_mass
    if int(pdg) in _LATE_SINGLETS and int(pdg) not in {int(r.pdg_id) for r in catalog.records()}:
        # the eta'(958) before the eta' option appends it: its mass from that option's configuration
        from .eta_prime import ETA_PRIME_REL
        root = Path(__file__).resolve().parents[1]
        return float(json.loads((root / ETA_PRIME_REL).read_text(encoding="utf-8"))["record"]["pole_mass"])
    return float(_daughter_pole_mass(catalog, int(pdg)))


def pdg_nucleon_resonances_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with every charge state of the configuration's multiplets
    appended (the three-body law carried over); each record's support runs
    from its lowest channel threshold at the daughters' pole masses to five
    widths above its pole. A catalog that already holds one of the ids is
    refused."""
    from .isoscalar_dipion import range_position_fractions
    have = {int(r.pdg_id) for r in catalog.records()} | {int(p) for p in catalog.stable_states()}
    clash = sorted(set(new_resonance_ids(config)) & have)
    if clash:
        raise ValueError(f"pdg_nucleon_resonances: the ids {clash} are already catalog states")
    new = []
    for mult in config["multiplets"]:
        iso = float(mult["I"])
        template = catalog.get(_TEMPLATES[iso])
        fractions = range_position_fractions(mult["modes"])
        for charge_text, pdg in mult["pdg_ids"].items():
            charge = int(charge_text)
            i3x2 = 2 * charge - 1  # B = 1, S = 0: I3 = Q - 1/2
            name = f"{mult['name']}{ {2: '++', 1: '+', 0: '0', -1: '-'}[charge] }".replace(" ", "")
            channels = []
            for mode, fraction in zip(mult["modes"], fractions):
                three = len(MODE_KINDS[mode["kind"]]) == 3
                for daughters, share in _split(iso, i3x2, mode["kind"], catalog):
                    channels.append(DecayChannel(int(pdg), tuple(daughters), fraction * share, int(mode["L"]), three,
                                                 _tag(mult["name"], "branching_fraction")))
            pole, width = float(mult["pole_mass"]), float(mult["nominal_width"])
            lo = min(math.fsum(_pole(catalog, d) for d in c.daughters) for c in channels)
            new.append(dataclasses.replace(template, pdg_id=int(pdg), name=name, Q=charge, isospin=iso, isospin3=i3x2,
                                           spin=float(mult["J"]), parity=str(mult["P"]), pole_mass=pole,
                                           nominal_width=width, mass_range=(lo, pole + M_UP_WIDTHS * width),
                                           decay_channels=tuple(channels), stable_for_decay=False,
                                           provenance=_tag(mult["name"], "record")))
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def _is_nstar_pi_class(class_id: str, catalog: HadronCatalog) -> bool:
    toks = [int(t) for t in str(class_id).strip("()").split(",")]
    if len(toks) != 2:
        return False
    a, b = toks
    for baryon, pion in ((a, b), (b, a)):
        if abs(pion) in (211, 111):
            try:
                rec = catalog.get(int(baryon))
            except KeyError:
                continue
            if int(rec.B) == 1 and int(rec.S) == 0 and float(rec.nominal_width) > 0.0 and abs(int(baryon)) not in (2212, 2112):
                return True
    return False


def pdg_nucleon_resonances_families(fams: list[ContinuumFamily], catalog: HadronCatalog, config: dict) -> list[ContinuumFamily]:
    """The families with, beside their (N*, pi) classes, the classes of the new
    states with a pion at every charge the line allows (appended after their
    own); a family without an (N*, pi) class is unchanged."""
    ids = new_resonance_ids(config)
    out = []
    for family in fams:
        qn = tuple(int(v) for v in family.qn)
        if qn[0] != 1 or qn[2] != 0 or not any(_is_nstar_pi_class(c, catalog) for c in family.allowed_decay_classes):
            out.append(family)
            continue
        have = {tuple(sorted(int(t) for t in str(c).strip("()").split(","))) for c in family.allowed_decay_classes}
        extra = []
        for pdg in ids:
            q = int(catalog.get(pdg).Q)
            pion = _PIONS.get(qn[1] - q)
            if pion is None:
                continue
            key = tuple(sorted((pdg, pion)))
            if key not in have:
                have.add(key)
                extra.append(f"({pdg},{pion})")
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + extra)) if extra else family)
    return out
