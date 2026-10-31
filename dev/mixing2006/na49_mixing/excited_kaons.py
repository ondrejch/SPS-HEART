"""The excited kaons between the K1(1270)/K2*(1430) and the continuum floor:
K*(1410), K0*(1430), K1(1400) and K*(1680) (2026-10-28).

``KernelConfig.excited_kaons`` adds the sixteen records (four states, two
charges, the antiparticles) at their PDG 2024 values
(``config/v2_singlet/excited_kaons_pdg2024.json``) and, beside every class of
a baryonic family that pairs a baryon with a kaon (K+-, K0, Kbar0: the
associated (Y, K) pairs of the non-strange lines, the (N, Kbar) pairs of the
strange lines, the conjugate towers), the classes with each excited kaon of
the kaon's charge and strangeness in the kaon's place.

The committed catalog ends the strange meson spectrum at the K1(1270) and
K2*(1430), while the non-strange one runs to 2 GeV (f2, a2, rho(1450),
omega(1420), pi(1300), f0(1370) and beyond; the register's item 1 lists the
omission). At a baryonic vertex the strangeness therefore leaves either as a
light kaon or kaon vector in a direct pair or as a K-like continuum state
above the 2 GeV floor, with nothing between 1.43 and 2 GeV, where the
non-strange side has its states. The excited kaons fill that gap with
established states: a (Y, K*(1680)) pair gives the kaon's momentum to a
1.7 GeV state whose decays keep it forward, where the K-like continuum
hadronizes at the center (the forward K+ deficit and the K*0 excess,
``_knowledge/evidence/systematics-r108-2026-10-23.md``, R125). The vector
members (K*(1410), K*(1680)) carry the ledger's r_V as every vector; the
direct pairs' weight law applies as to every class.

The mesonic families' classes (the strange floor pools' (K1(1270), pi) and
(K2*(1430), pi), the neutral lines' (K, Kbar) pairs) are left as committed:
this option concerns the baryon-kaon vertex.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["EXCITED_KAONS_REL", "EXCITED_KAON_PARTNERS", "load_excited_kaons", "excited_kaons_catalog",
           "excited_kaons_families", "excited_kaon_classes"]

#: The configuration, relative to the nested repository root.
EXCITED_KAONS_REL = Path("config") / "v2_singlet" / "excited_kaons_pdg2024.json"
#: The kaon and its excited partners of the same charge and strangeness.
EXCITED_KAON_PARTNERS = {321: (100323, 10321, 20323, 30323), 311: (100313, 10311, 20313, 30313),
                         -321: (-100323, -10321, -20323, -30323), -311: (-100313, -10311, -20313, -30313)}
_TEMPLATE_PDG = 10323  # K1(1270)+
#: The line shape's upper edge in widths above the pole (as the catalog's K2*(1430)).
M_UP_WIDTHS = 5.0


def load_excited_kaons(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / EXCITED_KAONS_REL).read_text(encoding="utf-8"))


def _tag(name: str, field: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(EXCITED_KAONS_REL), section_or_table=name, field=field,
                         accessed_on="2026-10-28", transcribed_value=name, transcriber="na49-res excited kaons option")


def _pole(catalog: HadronCatalog, pdg: int) -> float:
    from .cascade_decay import _daughter_pole_mass
    return float(_daughter_pole_mass(catalog, int(pdg)))


def excited_kaons_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the excited kaon records appended (the three-body law
    carried over); each record's support runs from its lowest channel
    threshold at the daughters' pole masses to five widths above its pole. A
    catalog that already holds one is refused."""
    have = {int(r.pdg_id) for r in catalog.records()}
    template = catalog.get(_TEMPLATE_PDG)
    new = []
    for spec in config["records"]:
        pdg, name = int(spec["pdg_id"]), str(spec["name"])
        if pdg in have:
            raise ValueError(f"excited_kaons: the catalog already holds {name}; the option adds it")
        total = math.fsum(float(m["fraction"]) for m in spec["modes"])
        channels = tuple(DecayChannel(pdg, tuple(int(d) for d in m["daughters"]), float(m["fraction"]) / total, int(m["L"]),
                                      bool(m["three_body"]), _tag(name, "branching_fraction")) for m in spec["modes"])
        pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
        lo = min(math.fsum(_pole(catalog, d) for d in m["daughters"]) for m in spec["modes"])
        new.append(dataclasses.replace(template, pdg_id=pdg, name=name, Q=int(spec["Q"]), S=int(spec["S"]), isospin=0.5,
                                       isospin3=int(spec["isospin3"]), spin=float(spec["spin"]), parity=str(spec["parity"]),
                                       pole_mass=pole, nominal_width=width, mass_range=(lo, pole + M_UP_WIDTHS * width),
                                       decay_channels=channels, stable_for_decay=False, provenance=_tag(name, "record")))
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def excited_kaon_classes(classes, catalog: HadronCatalog) -> list[str]:
    """The excited-kaon analogs of a baryonic family's classes: beside every
    class of a baryon record and a kaon, the classes with each excited kaon
    of the kaon's charge in the kaon's place, in the order of the classes."""
    from .cascade_decay import _parse_class
    recs = {int(r.pdg_id): r for r in catalog.records()}
    seen = {tuple(sorted(_parse_class(c))) for c in classes}
    out = []
    for c in classes:
        d0, d1 = _parse_class(c)
        for baryon, kaon in ((d0, d1), (d1, d0)):
            if kaon in EXCITED_KAON_PARTNERS and baryon in recs and int(recs[baryon].B) != 0:
                for heavy in EXCITED_KAON_PARTNERS[kaon]:
                    key = tuple(sorted((baryon, heavy)))
                    if key not in seen:
                        seen.add(key)
                        out.append(f"({baryon},{heavy})")
    return out


def excited_kaons_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the excited-kaon classes appended after their own, in
    the allowed decay classes of every baryonic family (B != 0); the mesonic
    families are returned unchanged."""
    recs = {int(r.pdg_id) for r in catalog.records()}
    missing = sorted({p for ps in EXCITED_KAON_PARTNERS.values() for p in ps} - recs)
    if missing:
        raise ValueError(f"excited_kaons: the records {missing} are absent")
    out = []
    for family in fams:
        if int(family.qn[0]) == 0:
            out.append(family)
            continue
        extra = excited_kaon_classes(list(family.allowed_decay_classes), catalog)
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + extra)) if extra else family)
    return out
