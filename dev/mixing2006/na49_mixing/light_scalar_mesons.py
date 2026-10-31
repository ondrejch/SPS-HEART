"""The light scalar mesons f0(980) and a0(980) (2026-10-11).

``KernelConfig.light_scalar_mesons`` adds the f0(980) record, the three
a0(980) charge states and their classes in the neutral mesonic family and
the non-strange baryonic family, from
``config/v2_singlet/light_scalar_mesons_pdg2025.json``.

The committed catalog carries no light meson below the phi(1020) that
decays into K Kbar, and in the committed kernel a non-strange baryonic
vertex opens no kaon-pair channel.  At the R91 optimum the soft K- at
x_F 0.10-0.55 stand at 0.82, 0.60 and 0.38 of the data, and the model
makes about twice the NA61 phi yield.  The diagnosis is the fixed-kernel
note (``_knowledge/evidence/fixed-kernel-residual-mechanisms-2026-10-03.md``,
the probes at the R91 optimum).

Each record sits at its PDG 2025 estimate; its modes enter at the listed
ratio and are split over the charge states by their isospin
Clebsch-Gordan shares; the neutral kaons are K0 and K0bar, as in the
K*(892) tables.  The line shape runs from the lightest
threshold to the pole plus five widths, as the committed f2(1270) and
a2(1320) records.  The classes join the committed families before the
isospin completion, which rotates the (N, f0(980)) class to the charge
partners; the I = 3/2 families take it at Clebsch-Gordan weight 0, and the
conjugate towers carry its conjugate.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, HadronRecord, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["LIGHT_SCALAR_MESONS_REL", "load_light_scalar_mesons", "light_scalar_mesons_catalog",
           "light_scalar_mesons_families"]

#: The configuration, relative to the nested repository root.
LIGHT_SCALAR_MESONS_REL = Path("config") / "v2_singlet" / "light_scalar_mesons_pdg2025.json"
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0
_PI0, _PIP, _PIM, _ETA, _KP, _KM, _K0, _K0B = 111, 211, -211, 221, 321, -321, 311, -311

#: The charge split of each mode kind, keyed by the record's charge: a list
#: of (daughters, Clebsch-Gordan share).
_SPLITS = {
    "pipi_I0": {0: [((_PIP, _PIM), 2 / 3), ((_PI0, _PI0), 1 / 3)]},
    "KKbar_neutral": {0: [((_KP, _KM), 1 / 2), ((_K0, _K0B), 1 / 2)]},
    "KKbar_charged": {1: [((_KP, _K0B), 1.0)], -1: [((_KM, _K0), 1.0)]},
    "etapi": {0: [((_ETA, _PI0), 1.0)], 1: [((_ETA, _PIP), 1.0)], -1: [((_ETA, _PIM), 1.0)]},
}


def load_light_scalar_mesons(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / LIGHT_SCALAR_MESONS_REL).read_text(encoding="utf-8"))


def _tag(field: str, value, section: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(LIGHT_SCALAR_MESONS_REL), section_or_table=section, field=field,
                         accessed_on="2026-10-11", transcribed_value=value,
                         transcriber="na49-res light scalar mesons option")


def _record(catalog: HadronCatalog, spec: dict) -> HadronRecord:
    pid, name, charge = int(spec["pdg_id"]), str(spec["name"]), int(spec["Q"])
    total = math.fsum(float(m["fraction"]) for m in spec["modes"])
    if not total > 0.0:
        raise ValueError(f"light_scalar_mesons: the {name} fractions sum to {total!r}")
    channels = []
    for mode in spec["modes"]:
        kind = str(mode["kind"])
        if kind not in _SPLITS or charge not in _SPLITS[kind]:
            raise ValueError(f"light_scalar_mesons: no {kind!r} split for {name} at charge {charge}")
        for daughters, share in _SPLITS[kind][charge]:
            fraction = float(mode["fraction"]) / total * share
            channels.append(DecayChannel(pid, daughters, fraction, 0, False,
                                         _tag("branching_fraction", fraction, f"{name} {kind}")))
    threshold = min(math.fsum(catalog._daughter_mass(d) for d in c.daughters) for c in channels)
    pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
    return HadronRecord(pdg_id=pid, name=name, B=0, Q=charge, S=0, isospin=float(spec["isospin"]),
                        isospin3=int(spec["isospin3"]), spin=float(spec["spin"]), parity=str(spec["parity"]),
                        pole_mass=pole, nominal_width=width, mass_range=(float(threshold), pole + M_UP_WIDTHS * width),
                        decay_channels=tuple(channels), stable_for_decay=False, provenance=_tag("record", spec["pdg"], name))


def light_scalar_mesons_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the scalar records appended (the three-body law
    carried over).  A catalog that already holds one of them is refused."""
    ids = {int(r.pdg_id) for r in catalog.records()}
    new = tuple(_record(catalog, spec) for spec in config["records"])
    held = sorted(ids & {int(r.pdg_id) for r in new})
    if held:
        raise ValueError(f"light_scalar_mesons: the catalog already holds {held}; the option adds them")
    known = ids | set(int(p) for p in catalog.stable_states())
    missing = sorted({d for r in new for c in r.decay_channels for d in c.daughters} - known)
    if missing:
        raise ValueError(f"light_scalar_mesons: daughters {missing} are not catalog records")
    return HadronCatalog(tuple(catalog.records()) + new, three_body_law=catalog.three_body_law)


def light_scalar_mesons_families(fams: list[ContinuumFamily], config: dict) -> list[ContinuumFamily]:
    """The committed families with the scalar classes appended to the named
    families' pools (before the isospin completion)."""
    extra = {k: tuple(v) for k, v in config["classes"].items() if k != "rule"}
    present = {str(f.family_id) for f in fams}
    absent = sorted(set(extra) - present)
    if absent:
        raise ValueError(f"light_scalar_mesons: class families {absent} are absent")
    out = []
    for family in fams:
        add = extra.get(str(family.family_id), ())
        dup = sorted(set(add) & set(family.allowed_decay_classes))
        if dup:
            raise ValueError(f"light_scalar_mesons: family {family.family_id!r} already holds {dup}")
        out.append(dataclasses.replace(family, allowed_decay_classes=tuple(family.allowed_decay_classes) + add) if add else family)
    return out
