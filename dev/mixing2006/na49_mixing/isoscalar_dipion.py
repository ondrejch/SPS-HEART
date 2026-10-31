"""The isoscalar S-wave dipion of the nucleon resonances (2026-10-08).

``KernelConfig.isoscalar_dipion`` replaces the catalog's N* decay tables by
the PDG 2025 hadronic decay modes and adds the f0(500) record and the
(N, f0(500)) class of the baryonic continuum's direct pool, from
``config/v2_singlet/isoscalar_dipion_pdg2025.json``.

The committed tables fold the unrepresented part of the PDG N pi pi width
into N pi (``decays.build_soft_resonance_decays``, ``residual_to_n_pi``).
In the PDG listing that part is mostly N sigma, the isoscalar S-wave
dipion, which keeps the nucleon's charge, where N pi turns a proton into a
neutron in two cases of three.  The N(1720) table carried N rho at 35% and
no N omega against PDG's N rho 1-2%, N omega 12-40% and Delta pi 47-89%.
The diagnosis is the forward charge budget of the fixed-kernel note
(``_knowledge/evidence/fixed-kernel-residual-mechanisms-2026-10-03.md``).

The f0(500) is a catalog record at the PDG pole (475 MeV, width 550 MeV),
decaying to pi+ pi- and pi0 pi0 at the I = 0 shares 2 : 1 in an S-wave.
Each N* mode enters at the midpoint of its PDG range and is split over the
charge states by its isospin Clebsch-Gordan share; the fractions of a
multiplet are normalized to unity.  The floor class joins the committed
``baryonic_np`` family before the isospin completion, which rotates it to
the charge partners; the I = 3/2 families take it at Clebsch-Gordan
weight 0 in the enumeration.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["ISOSCALAR_DIPION_REL", "F0_500_PDG", "load_isoscalar_dipion", "isoscalar_dipion_catalog",
           "isoscalar_dipion_families"]

#: The configuration, relative to the nested repository root.
ISOSCALAR_DIPION_REL = Path("config") / "v2_singlet" / "isoscalar_dipion_pdg2025.json"
F0_500_PDG = 9000221
_PI0, _PIP, _PIM, _P, _N = 111, 211, -211, 2212, 2112

#: The charge split of each mode kind: (the I3 = +1/2 parent, the I3 = -1/2
#: parent), each a list of (daughters, Clebsch-Gordan share).
_SPLITS = {
    "Npi": ([((_P, _PI0), 1 / 3), ((_N, _PIP), 2 / 3)], [((_P, _PIM), 2 / 3), ((_N, _PI0), 1 / 3)]),
    "Nrho": ([((_P, 113), 1 / 3), ((_N, 213), 2 / 3)], [((_P, -213), 2 / 3), ((_N, 113), 1 / 3)]),
    "Dpi": ([((2224, _PIM), 1 / 2), ((2214, _PI0), 1 / 3), ((2114, _PIP), 1 / 6)],
            [((2214, _PIM), 1 / 6), ((2114, _PI0), 1 / 3), ((1114, _PIP), 1 / 2)]),
    "Nsigma": ([((_P, F0_500_PDG), 1.0)], [((_N, F0_500_PDG), 1.0)]),
    "Neta": ([((_P, 221), 1.0)], [((_N, 221), 1.0)]),
    "Nomega": ([((_P, 223), 1.0)], [((_N, 223), 1.0)]),
    "LK": ([((3122, 321), 1.0)], [((3122, 311), 1.0)]),
    "N1440pi": ([((12212, _PI0), 1 / 3), ((12112, _PIP), 2 / 3)], [((12212, _PIM), 2 / 3), ((12112, _PI0), 1 / 3)]),
    "N1520pi": ([((2216, _PI0), 1 / 3), ((2116, _PIP), 2 / 3)], [((2216, _PIM), 2 / 3), ((2116, _PI0), 1 / 3)]),
}


def load_isoscalar_dipion(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / ISOSCALAR_DIPION_REL).read_text(encoding="utf-8"))


def _tag(field: str, value, section: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(ISOSCALAR_DIPION_REL), section_or_table=section, field=field,
                         accessed_on="2026-10-08", transcribed_value=value, transcriber="na49-res isoscalar dipion option")


def _f0_record(catalog: HadronCatalog, spec: dict):
    pid = int(spec["pdg_id"])
    channels = (DecayChannel(pid, (_PIP, _PIM), 2.0 / 3.0, 0, False, _tag("branching_fraction", 2.0 / 3.0, "f0_500")),
                DecayChannel(pid, (_PI0, _PI0), 1.0 / 3.0, 0, False, _tag("branching_fraction", 1.0 / 3.0, "f0_500")))
    threshold = min(sum(catalog._daughter_mass(d) for d in c.daughters) for c in channels)
    template = catalog.get(113)
    return dataclasses.replace(
        template, pdg_id=pid, name=str(spec["name"]), B=0, Q=0, S=0, isospin=float(spec["isospin"]), isospin3=0,
        spin=float(spec["spin"]), parity=str(spec["parity"]), pole_mass=float(spec["pole_mass"]),
        nominal_width=float(spec["nominal_width"]), mass_range=(float(threshold), float(spec["m_up"])),
        decay_channels=channels, stable_for_decay=False, provenance=_tag("record", spec["pdg"], "f0_500"))


def _pdg_range(text: str) -> tuple[float, float]:
    """The (low, high) fraction of a stored PDG range: "a-b %" or "<x %"
    (an upper limit, from 0); the qualifier after the percent sign is a
    note."""
    m = re.match(r"\s*<\s*([0-9.]+)\s*%", text)
    if m:
        return 0.0, float(m.group(1)) / 100.0
    m = re.match(r"\s*([0-9.]+)\s*-\s*([0-9.]+)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    raise ValueError(f"isoscalar_dipion: unreadable PDG range {text!r}")


def range_position_fractions(modes) -> list[float]:
    """The fractions of a multiplet's modes at one common position inside
    their PDG ranges, the position set so that the fractions sum to one
    (the review of R99, 2026-10-16, M6): mode i takes m_i - lam h_i, with
    m_i and h_i the midpoint and half-width of its range and lam =
    (sum m - 1) / sum h.  Every mode stays inside its range when |lam| <= 1;
    beyond that the position is held at the range edge and the fractions
    are normalized."""
    ranges = [_pdg_range(str(mode["pdg_range"])) for mode in modes]
    mids = [0.5 * (lo + hi) for lo, hi in ranges]
    halves = [0.5 * (hi - lo) for lo, hi in ranges]
    lam = (math.fsum(mids) - 1.0) / math.fsum(halves) if math.fsum(halves) > 0.0 else 0.0
    lam = max(-1.0, min(1.0, lam))
    values = [m - lam * h for m, h in zip(mids, halves)]
    total = math.fsum(values)
    return [v / total for v in values]


def _nstar_channels(pid: int, positive: bool, multiplet: dict, range_rule: bool = False) -> tuple[DecayChannel, ...]:
    total = math.fsum(float(c["fraction"]) for c in multiplet["channels"])
    if not total > 0.0:
        raise ValueError(f"isoscalar_dipion: the {multiplet['multiplet']} fractions sum to {total!r}")
    fractions = (range_position_fractions(multiplet["channels"]) if range_rule
                 else [float(c["fraction"]) / total for c in multiplet["channels"]])
    out = []
    for mode, mode_fraction in zip(multiplet["channels"], fractions):
        kind = str(mode["kind"])
        if kind not in _SPLITS:
            raise ValueError(f"isoscalar_dipion: unknown mode kind {kind!r} in {multiplet['multiplet']}")
        for daughters, share in _SPLITS[kind][0 if positive else 1]:
            fraction = mode_fraction * share
            out.append(DecayChannel(pid, tuple(int(d) for d in daughters), fraction, int(mode["orbital_angular_momentum"]),
                                    False, _tag("branching_fraction", fraction, f"nstar {multiplet['multiplet']} {kind}")))
    return tuple(out)


def isoscalar_dipion_catalog(catalog: HadronCatalog, config: dict, range_rule: bool = False) -> HadronCatalog:
    """The catalog with the f0(500) record appended and the N* multiplets'
    tables replaced (the three-body law carried over).  *range_rule*
    (``KernelConfig.nstar_range_rule``) sets the fractions by
    :func:`range_position_fractions` instead of normalizing the midpoints.
    A catalog that already holds the f0(500), or misses an N* record or a
    daughter, is refused."""
    ids = {int(r.pdg_id) for r in catalog.records()}
    spec = config["f0_500"]
    if int(spec["pdg_id"]) in ids:
        raise ValueError("isoscalar_dipion: the catalog already holds the f0(500); the option adds it")
    with_f0 = HadronCatalog(tuple(catalog.records()) + (_f0_record(catalog, spec),), three_body_law=catalog.three_body_law)
    known = ids | {int(spec["pdg_id"])} | set(int(p) for p in with_f0.stable_states())
    tables = {}
    for multiplet in config["nstar"]:
        plus, zero = (int(p) for p in multiplet["pdg_ids"])
        for pid, positive in ((plus, True), (zero, False)):
            if pid not in ids:
                raise ValueError(f"isoscalar_dipion: the N* record {pid} ({multiplet['multiplet']}) is absent from the catalog")
            tables[pid] = _nstar_channels(pid, positive, multiplet, range_rule)
            missing = sorted({d for c in tables[pid] for d in c.daughters} - known)
            if missing:
                raise ValueError(f"isoscalar_dipion: daughters {missing} of {pid} are not catalog records")
    records = tuple(dataclasses.replace(r, decay_channels=tables[int(r.pdg_id)]) if int(r.pdg_id) in tables else r
                    for r in with_f0.records())
    return HadronCatalog(records, three_body_law=catalog.three_body_law)


def isoscalar_dipion_families(fams: list[ContinuumFamily], config: dict) -> list[ContinuumFamily]:
    """The committed families with the floor classes appended to the named
    families' direct pools (before the isospin completion)."""
    extra = {k: tuple(v) for k, v in config["floor_classes"].items() if k != "rule"}
    present = {str(f.family_id) for f in fams}
    absent = sorted(set(extra) - present)
    if absent:
        raise ValueError(f"isoscalar_dipion: floor families {absent} are absent")
    out = []
    for family in fams:
        add = extra.get(str(family.family_id), ())
        dup = sorted(set(add) & set(family.allowed_decay_classes))
        if dup:
            raise ValueError(f"isoscalar_dipion: family {family.family_id!r} already holds {dup}")
        out.append(dataclasses.replace(family, allowed_decay_classes=tuple(family.allowed_decay_classes) + add) if add else family)
    return out
