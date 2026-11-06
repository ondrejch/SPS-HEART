"""The PDG hyperon resonances up to the continuation floor (2026-10-15).

``KernelConfig.pdg_hyperons`` adds the Lambda and Sigma resonances of the
PDG 2025 summary tables between the catalog's Sigma(1385) and Lambda(1520)
and the 2.0 GeV continuation floor, from
``config/v2_singlet/pdg_hyperons_pdg2025.json``: Lambda(1405, 1600, 1670,
1690, 1800, 1810, 1820, 1830, 1890) and Sigma(1660, 1670, 1750, 1775, 1910,
1915) in three charge states, with their classes in the baryonic families.

The committed catalog carries Lambda, Sigma, Sigma(1385) and Lambda(1520)
and no hyperon resonance between Lambda(1520) and the floor.  Its
associated production therefore runs through those two states alone, and
the fit's effective factor on it (assoc_norm, 3.86 at the R97 optimum)
stands in for the missing spectrum; the N Kbar decays of the missing
states also feed the K- at the hyperon lines' rapidity.  The decision is
the model decision register (``_knowledge/paper/model-decision-register.md``)
and the probe the fixed-kernel note (the probes at the R91 optimum,
``PROBE_HYPERONS``), of which this option is the kernel form.

Each state sits at its listed mass and width, its modes at the midpoints
of the listed ranges and normalized over the kept ones; a two-body mode's
charge members enter at their isospin Clebsch-Gordan shares.  The classes
are (Lambda*, K) and (Sigma*, K) at the non-strange baryonic families of
the matching charge and (Sigma*, pi) and (Lambda*, pi) at the strange
ones, as the committed Lambda(1520) and Sigma(1385).  They join before the
charge-conjugate towers, which conjugate the records and the classes.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .resonance_decay import clebsch_gordan_squared
from .spectral_density import ContinuumFamily

__all__ = ["PDG_HYPERONS_REL", "load_pdg_hyperons", "pdg_hyperons_catalog", "pdg_hyperons_families"]

#: The configuration, relative to the nested repository root.
PDG_HYPERONS_REL = Path("config") / "v2_singlet" / "pdg_hyperons_pdg2025.json"
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0
#: Each multiplet of a mode: (isospin, [(pdg, 2 I3)]).
MULTIPLETS = {"N": (0.5, [(2212, 1), (2112, -1)]), "Kb": (0.5, [(-311, 1), (-321, -1)]),
              "K": (0.5, [(321, 1), (311, -1)]), "pi": (1.0, [(211, 2), (111, 0), (-211, -2)]),
              "Sig": (1.0, [(3222, 2), (3212, 0), (3112, -2)]), "S1385": (1.0, [(3224, 2), (3214, 0), (3114, -2)]),
              "Lam": (0.0, [(3122, 0)]), "eta": (0.0, [(221, 0)]), "sigma": (0.0, [(9000221, 0)]),
              "L1520": (0.0, [(102134, 0)]), "L1405": (0.0, [(13122, 0)]),
              "Kbs": (0.5, [(-313, 1), (-323, -1)]),
              "Delta": (1.5, [(2224, 3), (2214, 1), (2114, -1), (1114, -3)])}
#: The template record of each kind: Lambda(1520) for a Lambda*, Sigma(1385) of the charge for a Sigma*.
_LAMBDA_TEMPLATE = 102134
_SIGMA_TEMPLATE = {2: 3224, 0: 3214, -2: 3114}


def load_pdg_hyperons(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / PDG_HYPERONS_REL).read_text(encoding="utf-8"))


def _tag(name: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(PDG_HYPERONS_REL), section_or_table=name, field="record",
                         accessed_on="2026-10-15", transcribed_value=name, transcriber="na49-res PDG hyperons option")


def _split(kind: str, isospin: float, i3x2: int):
    """The charge members of a mode and their Clebsch-Gordan shares, with the three-body flag."""
    if kind in ("Lam pi pi", "Sig pi pi") and int(i3x2) != 0:
        # the fixed neutral triples below hold the charge of a neutral parent only (the code reviews muse13f M1 and
        # glm53f F14: a charged parent would break (B, Q, S) through them)
        raise ValueError(f"pdg_hyperons: the mode {kind!r} is tabulated for a neutral parent only (2 I3 = {i3x2})")
    if kind == "Lam pi pi":
        return [((3122, 211, -211), 2 / 3, True), ((3122, 111, 111), 1 / 3, True)]
    if kind == "Sig pi pi":
        return [((3222, -211, 111), 1 / 3, True), ((3112, 211, 111), 1 / 3, True), ((3212, 211, -211), 1 / 3, True)]
    a, b = kind.split()
    (ia, ma), (ib, mb) = MULTIPLETS[a], MULTIPLETS[b]
    out = []
    for da, xa in ma:
        for db, xb in mb:
            if xa + xb == i3x2:
                w = clebsch_gordan_squared(isospin, i3x2, ia, xa, ib, xb)
                if w > 1e-12:
                    out.append(((da, db), w, False))
    return out


def _records(catalog: HadronCatalog, config: dict):
    """The new records, with the pdg ids of the Lambda* and Sigma* multiplets."""
    records, mass = [], {}

    def daughter_mass(pid):
        return mass[pid] if pid in mass else catalog._daughter_mass(pid)
    lam_ids, sig_ids = [], []
    for spec in config["records"]:
        name, ids, iso = str(spec["name"]), tuple(int(v) for v in spec["pdg_ids"]), float(spec["isospin"])
        members = [(ids[0], 0, "")] if iso == 0.0 else list(zip(ids, (2, 0, -2), ("+", "0", "-")))
        (lam_ids if iso == 0.0 else sig_ids).append(ids)
        total = math.fsum(float(m["fraction"]) for m in spec["modes"])
        for pid, i3x2, suffix in members:
            channels = []
            for mode in spec["modes"]:
                for daughters, share, three in _split(str(mode["kind"]), iso, i3x2):
                    channels.append(DecayChannel(pid, tuple(daughters), float(mode["fraction"]) / total * share,
                                                 int(mode["L"]), three, _tag(name)))
            lo = min(math.fsum(daughter_mass(d) for d in c.daughters) for c in channels)
            template = catalog.get(_LAMBDA_TEMPLATE if iso == 0.0 else _SIGMA_TEMPLATE[i3x2])
            pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
            records.append(dataclasses.replace(
                template, pdg_id=pid, name=name + suffix, B=1, Q=i3x2 // 2, S=-1, isospin=iso, isospin3=i3x2,
                spin=float(spec["spin"]), parity=str(spec["parity"]), pole_mass=pole, nominal_width=width,
                mass_range=(float(lo), pole + M_UP_WIDTHS * width), decay_channels=tuple(channels),
                stable_for_decay=False, provenance=_tag(name)))
            mass[pid] = pole
    return records, lam_ids, sig_ids


def pdg_hyperons_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the hyperon records appended (the three-body law
    carried over).  A catalog that already holds one of them is refused."""
    new, _lam, _sig = _records(catalog, config)
    ids = {int(r.pdg_id) for r in catalog.records()}
    held = sorted(ids & {int(r.pdg_id) for r in new})
    if held:
        raise ValueError(f"pdg_hyperons: the catalog already holds {held}; the option adds them")
    known = ids | {int(r.pdg_id) for r in new} | set(int(p) for p in catalog.stable_states())
    missing = sorted({d for r in new for c in r.decay_channels for d in c.daughters} - known)
    if missing:
        raise ValueError(f"pdg_hyperons: daughters {missing} are not catalog records")
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def hyperon_classes(config: dict, catalog: HadronCatalog) -> dict[str, list[str]]:
    """The added classes by family id (the non-strange and strange baryonic
    families and the Delta-like lines of charge +1 and 0)."""
    _new, lam_ids, sig_ids = _records(catalog, config)
    lam = [i[0] for i in lam_ids]
    sp, s0, sm = ([i[k] for i in sig_ids] for k in range(3))
    extra = {
        "baryonic_np": [f"({x},321)" for x in lam] + [f"({x},321)" for x in s0] + [f"({x},311)" for x in sp],
        "baryonic_np_q0": [f"({x},311)" for x in lam] + [f"({x},321)" for x in sm] + [f"({x},311)" for x in s0],
        "baryonic_np_q2": [f"({x},321)" for x in sp],
        "baryonic_np_qm1": [f"({x},311)" for x in sm],
        "baryonic_strange": [f"({x},-211)" for x in sp] + [f"({x},111)" for x in s0] + [f"({x},211)" for x in sm]
                            + [f"({x},111)" for x in lam],
        "baryonic_strange_q1": [f"({x},111)" for x in sp] + [f"({x},211)" for x in s0] + [f"({x},211)" for x in lam],
        "baryonic_strange_qm1": [f"({x},111)" for x in sm] + [f"({x},-211)" for x in s0] + [f"({x},-211)" for x in lam],
    }
    extra["baryonic_np_delta_q1"] = list(extra["baryonic_np"])
    extra["baryonic_np_delta_q0"] = list(extra["baryonic_np_q0"])
    return extra


def pdg_hyperons_families(fams: list[ContinuumFamily], config: dict, catalog: HadronCatalog) -> list[ContinuumFamily]:
    """The families with the hyperon classes appended (a class already held
    is not repeated; the other families unchanged).  A kernel without one of
    the named families is refused."""
    extra = hyperon_classes(config, catalog)
    present = {str(f.family_id) for f in fams}
    absent = sorted(set(extra) - present)
    if absent:
        raise ValueError(f"pdg_hyperons: class families {absent} are absent (the option needs the isospin "
                         "completion and the Delta lines)")
    out = []
    for family in fams:
        add = [c for c in extra.get(str(family.family_id), []) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + add)) if add else family)
    return out
