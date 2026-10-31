"""The I = 0 members of the axial-vector nonets, f1(1285) and h1(1170)
(2026-10-21).

``KernelConfig.axial_isoscalars`` adds the f1(1285) and h1(1170) records at
their PDG 2024 values (``config/v2_singlet/axial_isoscalars_pdg2024.json``)
and, beside every continuum class of an omega and a pion, the classes with
each of them in the omega's place.

The committed catalog carries the I = 1 members of the 1++ and 1+- nonets,
a1(1260) and b1(1235), and the strange K1(1270), without their isoscalar
partners: the same incompleteness the eta'(958) closed for the pseudoscalar
nonet in R98 (the model decision register, section 3.2). The f1(1285) is
narrow (23 MeV) and decays to four pions through rho pi pi (Q = 0.23 GeV),
to eta pi pi through a0(980) pi, to K Kbar pi and to gamma rho0; the
h1(1170) is broad (375 MeV) and decays to rho pi. The external reviews of
2026-10-21 (`.collab/`) named both as the untested isoscalar candidates for
the soft pions of the register's open question 1.

The classes follow the omega, the I = 0 member of the vector nonet, wherever
it pairs with a pion; the kernel's isospin factors decide which charge states
a line of given isospin reaches (an I = 1 line, not an I = 0 one). The
records take the b1(1235)0 record as their template (J = 1, P = +) at
I = 0; the flavor ledger counts them with the other non-vector mesons.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["AXIAL_ISOSCALARS_REL", "AXIAL_ISOSCALAR_PDGS", "load_axial_isoscalars", "axial_isoscalars_catalog",
           "axial_isoscalars_families", "axial_isoscalar_classes"]

#: The configuration, relative to the nested repository root.
AXIAL_ISOSCALARS_REL = Path("config") / "v2_singlet" / "axial_isoscalars_pdg2024.json"
AXIAL_ISOSCALAR_PDGS = (20223, 10223)
_TEMPLATE_PDG = 10113  # b1(1235)0
_OMEGA = "223"
_PIONS = ("111", "211", "-211")
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0

#: PROBE_AXIAL_I1_ONLY = 1 (2026-10-21; the card
#: TASK-20261021-r106-pbar-and-forward-piplus-mechanisms, piece P2, set by
#: ``.collab/tooling/probe_forward_r75.py`` before the first kernel load):
#: the (f1, pi0) and (h1, pi0) classes stay out of the I = 0 families.  The
#: direct (f1, pi0) and (h1, pi0) pairs already carry a Clebsch-Gordan
#: weight of zero there; what the classes add at an I = 0 vertex is the
#: isospin-allowed continuation channels (mesonic_neutral, f1) and
#: (mesonic_neutral, h1), the axial analogs of the committed
#: (mesonic_neutral, omega), so the flag removes the f1 and h1 emission of
#: the neutral chain (inside the noise at the R106 optimum, +209 and +40
#: with the yields fixed; the review of 2026-10-22).  ``False`` keeps the
#: committed classes bit for bit.
AXIAL_NEUTRAL_I1_ONLY: bool = False


def load_axial_isoscalars(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / AXIAL_ISOSCALARS_REL).read_text(encoding="utf-8"))


def _tag(name: str, field: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(AXIAL_ISOSCALARS_REL), section_or_table=name, field=field,
                         accessed_on="2026-10-21", transcribed_value=name, transcriber="na49-res axial isoscalars option")


def _pole(catalog: HadronCatalog, pdg: int) -> float:
    if int(pdg) == 22:
        return 0.0
    from .cascade_decay import _daughter_pole_mass
    return float(_daughter_pole_mass(catalog, int(pdg)))


def axial_isoscalars_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the f1(1285) and h1(1170) records appended (the
    three-body law carried over); each record's support runs from its lowest
    channel threshold at the daughters' pole masses to five widths above its
    pole. A catalog that already holds one is refused."""
    have = {int(r.pdg_id) for r in catalog.records()}
    template = catalog.get(_TEMPLATE_PDG)
    new = []
    for spec in config["records"]:
        pdg, name = int(spec["pdg_id"]), str(spec["name"])
        if pdg in have:
            raise ValueError(f"axial_isoscalars: the catalog already holds {name}; the option adds it")
        total = math.fsum(float(m["fraction"]) for m in spec["modes"])
        channels = tuple(DecayChannel(pdg, tuple(int(d) for d in m["daughters"]), float(m["fraction"]) / total, int(m["L"]),
                                      bool(m["three_body"]), _tag(name, "branching_fraction")) for m in spec["modes"])
        pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
        lo = min(math.fsum(_pole(catalog, d) for d in m["daughters"]) for m in spec["modes"])
        new.append(dataclasses.replace(template, pdg_id=pdg, name=name, Q=0, isospin=0.0, isospin3=0, spin=1.0, parity="+",
                                       pole_mass=pole, nominal_width=width, mass_range=(lo, pole + M_UP_WIDTHS * width),
                                       decay_channels=channels, stable_for_decay=False, provenance=_tag(name, "record")))
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def axial_isoscalar_classes(classes, drop_pi0: bool = False) -> list[str]:
    """The axial-isoscalar analogs of a family's classes: beside every class of
    an omega and a pion, the classes with the f1(1285) and the h1(1170) in the
    omega's place, in the order of the classes.  With *drop_pi0* (piece P2,
    ``AXIAL_NEUTRAL_I1_ONLY``) the pi0 classes are left out: an I = 0 family
    cannot carry the isospin-forbidden (I = 0, I = 1) pair."""
    seen = {tuple(sorted(t.strip() for t in str(c).strip("()").split(","))) for c in classes}
    out = []
    for c in classes:
        toks = [t.strip() for t in str(c).strip("()").split(",")]
        if len(toks) != 2 or _OMEGA not in toks:
            continue
        other = toks[1] if toks[0] == _OMEGA else toks[0]
        if other not in _PIONS:
            continue
        if drop_pi0 and other == _PIONS[0]:
            continue
        for pdg in AXIAL_ISOSCALAR_PDGS:
            new_toks = [str(pdg) if t == _OMEGA else t for t in toks]
            key = tuple(sorted(new_toks))
            if key not in seen:
                seen.add(key)
                out.append("(" + ",".join(new_toks) + ")")
    return out


def axial_isoscalars_families(fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The families with the axial-isoscalar classes appended after their own,
    in the allowed decay classes and in the floor classes.  Under
    ``AXIAL_NEUTRAL_I1_ONLY`` (piece P2, 2026-10-21) an I = 0 family
    (``family.isospin == 0.0``) takes no (f1, pi0) or (h1, pi0) class: the
    (I = 0, I = 1) pair is isospin-forbidden there, while the I = 1 charged
    and anticharged families keep their (f1, pi+-) and (h1, pi+-) classes."""
    out = []
    for family in fams:
        drop_pi0 = bool(AXIAL_NEUTRAL_I1_ONLY) and family.isospin is not None and float(family.isospin) == 0.0
        extra = axial_isoscalar_classes(list(family.allowed_decay_classes), drop_pi0=drop_pi0)
        floor = tuple(getattr(family, "floor_decay_classes", ()) or ())
        extra_floor = axial_isoscalar_classes(list(floor), drop_pi0=drop_pi0) if floor else []
        if not extra and not extra_floor:
            out.append(family)
            continue
        changes = {}
        if extra:
            changes["allowed_decay_classes"] = type(family.allowed_decay_classes)(list(family.allowed_decay_classes) + extra)
        if extra_floor:
            changes["floor_decay_classes"] = type(floor)(list(floor) + extra_floor)
        out.append(dataclasses.replace(family, **changes))
    return out
