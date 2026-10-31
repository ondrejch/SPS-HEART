"""The charge-conjugate baryonic towers (2026-10-09).

The committed continuum carries baryonic families (B = +1) and no
antibaryonic ones.  A neutral mesonic continuum state that continues
through a baryon-pair class, (p, pbar) or (n, nbar), therefore replaces
the baryon by a baryonic continuum state and emits the antibaryon as a bare
established sibling; the conjugate channel (the baryon as the sibling, an
antibaryonic continuum state continuing) does not exist.  Every antibaryon
then takes the full decay momentum of a heavy step and none comes from the
decay of an antibaryon resonance, so the cascade breaks charge conjugation
in the baryon sector.  At the R87 optimum the central pbar run 0.63 of the
data below 0.4 GeV/c and 1.34-1.41 above 0.8 GeV/c at x_F < 0.25, and the
conjugate towers lower the pbar chi2 by 310 with the yields fixed (the
fixed-kernel note, "The charge-conjugate baryonic towers at the R87
optimum").

``KernelConfig.conjugate_towers`` applies, after every other option:

- :func:`conjugate_catalog`: every antibaryon record is rebuilt as the
  charge conjugate of its baryon (pdg, B, Q, S and 2 I3 negated, the
  decay channels' daughters conjugated, the mass, width, spin, parity and
  branching fractions kept; the catalog convention gives an antibaryon its
  baryon's parity label).  The committed antibaryon records predate the N*
  revision of the isoscalar dipion and cover the neutral N* only;
- :func:`conjugate_baryon_families`: the conjugate of every B = +1 family
  is appended (family id ``"anti_" + family_id``, the quantum numbers, the
  allowed decay classes and the floor classes conjugated, every other field
  copied);
- ``cascade_decay._enumerate_continuum_channels`` gives each of the two
  conjugate baryon-pair continuations of a mesonic vertex half its weight
  (the vertex's baryon-pair share stays the committed one, so the pair
  strength r_BBbar keeps its meaning);
- ``flavor_weights.flavor_exponents`` reads the strangeness signs of an
  antibaryonic vertex conjugated, so the associated-production exponent
  counts Lambdabar K- at an antibaryonic vertex as it counts Lambda K+ at a
  baryonic one.

Off, the catalog and the families are the committed ones bit for bit.
"""

from __future__ import annotations

import dataclasses

from .cascade_decay import CONJUGATE_FAMILY_PREFIX, _charge_conjugate_pdg, _parse_class
from .hadron_catalog import HadronCatalog
from .spectral_density import ContinuumFamily

__all__ = ["CONJUGATE_FAMILY_PREFIX", "charge_conjugate", "conjugate_catalog", "conjugate_baryon_families"]


def charge_conjugate(pdg_id: int, catalog: HadronCatalog) -> int:
    """The charge conjugate of a catalog pdg id: a record with nonzero
    (B, Q, S), or one whose negated id the catalog also holds, maps to the
    negated id; a neutral non-strange meson (pi0, rho0, omega, eta, the
    weak neutral kaons) and the photon are their own conjugates."""
    return _charge_conjugate_pdg(pdg_id, catalog)


def conjugate_catalog(catalog: HadronCatalog) -> HadronCatalog:
    """The catalog with every antibaryon rebuilt as the conjugate of its
    baryon; the existing antibaryon records keep their names, the new ones
    take ``"a" + name``.  The record order is the committed one, the new
    antibaryons appended."""
    baryons = [record for record in catalog.records() if int(record.B) == 1]
    rebuilt = {-int(record.pdg_id) for record in baryons}
    names = {}
    out = []
    for record in catalog.records():
        if int(record.pdg_id) in rebuilt:
            names[int(record.pdg_id)] = record.name
            out.append(None)  # placeholder: the rebuilt record takes the committed slot
        else:
            out.append(record)
    conjugates = {}
    for record in baryons:
        pdg = -int(record.pdg_id)
        channels = tuple(
            dataclasses.replace(channel, parent_pdg=pdg,
                                daughters=tuple(charge_conjugate(d, catalog) for d in channel.daughters))
            for channel in record.decay_channels)
        conjugates[pdg] = dataclasses.replace(record, pdg_id=pdg, name=names.get(pdg, "a" + record.name), B=-int(record.B),
                                              Q=-int(record.Q), S=-int(record.S), isospin3=-int(record.isospin3),
                                              decay_channels=channels)
    placed = set()
    for i, record in enumerate(catalog.records()):
        if out[i] is None:
            out[i] = conjugates[int(record.pdg_id)]
            placed.add(int(record.pdg_id))
    out.extend(conjugates[pdg] for pdg in (-int(r.pdg_id) for r in baryons) if pdg not in placed)
    return HadronCatalog(out, three_body_law=catalog.three_body_law)


def _conjugate_classes(classes, catalog: HadronCatalog):
    if not classes:
        return classes
    out = []
    for class_id in classes:
        d0, d1 = _parse_class(class_id)
        out.append(f"({charge_conjugate(d0, catalog)},{charge_conjugate(d1, catalog)})")
    return type(classes)(out)


def conjugate_baryon_families(fams: list[ContinuumFamily], catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The conjugates of the B = +1 families, in the families' order
    (``catalog`` names the conjugates of the classes' daughters)."""
    return [
        dataclasses.replace(family, family_id=CONJUGATE_FAMILY_PREFIX + str(family.family_id),
                            qn=type(family.qn)(-int(v) for v in family.qn),
                            allowed_decay_classes=_conjugate_classes(family.allowed_decay_classes, catalog),
                            floor_decay_classes=_conjugate_classes(getattr(family, "floor_decay_classes", ()), catalog))
        for family in fams if int(family.qn[0]) == 1
    ]
