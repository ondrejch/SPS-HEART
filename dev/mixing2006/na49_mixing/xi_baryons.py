"""The S = -2 baryons (2026-10-16).

``KernelConfig.xi_baryons`` adds the Xi0, the Xi- and the Xi(1530) of the
PDG 2025 summary tables from ``config/v2_singlet/xi_baryons_pdg2025.json``,
with the classes (Xi, K) and (Xi(1530), K) at the strange baryonic families
of the matching charge.

The committed catalog holds no baryon of strangeness -2, so the cascade
makes no Xi, while NA61/SHINE measures 0.0033 Xi- per inelastic event at
158 GeV/c, 2.8% of the Lambda.  A strange baryonic line reaches the Xi by
creating one s-sbar pair beside its own strange quark; the flavor weights
count that pair (gamma_s), so the option predicts the Xi yield without a
new parameter.  The Xi0 and Xi- decay weakly and are stable here, as the
feed-down corrected NA49 tables count the Lambda and Sigma; the Xi(1530)
decays to Xi pi at the isospin Clebsch-Gordan shares.  The Xi resonances
above Xi(1530) list no branching fractions and stay out.  The records and
classes join before the charge-conjugate towers, which conjugate both.
The decision is the model decision register
(``_knowledge/paper/model-decision-register.md``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["XI_BARYONS_REL", "load_xi_baryons", "xi_baryons_catalog", "xi_baryons_families", "xi_classes"]

#: The configuration, relative to the nested repository root.
XI_BARYONS_REL = Path("config") / "v2_singlet" / "xi_baryons_pdg2025.json"
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0
#: The template record of each charge: Sigma(1385) of the charge.
_TEMPLATE = {0: 3214, -1: 3114}


def load_xi_baryons(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / XI_BARYONS_REL).read_text(encoding="utf-8"))


def _tag(name: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(XI_BARYONS_REL), section_or_table=name, field="record",
                         accessed_on="2026-10-16", transcribed_value=name, transcriber="na49-res Xi baryons option")


def _records(catalog: HadronCatalog, config: dict):
    """The new records, the stable ground states first."""
    out, mass = [], {}
    for spec in config["records"]:
        pid, name, i3x2 = int(spec["pdg_id"]), str(spec["name"]), int(spec["isospin3x2"])
        q = (i3x2 + 1 - 2) // 2
        pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
        template = catalog.get(_TEMPLATE[q])
        if spec["stable"]:
            channels, mass_range = (), (pole, pole)
        else:
            channels = tuple(DecayChannel(pid, tuple(int(d) for d in c["daughters"]), float(c["fraction"]), int(c["L"]),
                                          False, _tag(name)) for c in spec["channels"])
            lo = min(sum(mass[d] if d in mass else catalog._daughter_mass(d) for d in c.daughters) for c in channels)
            mass_range = (float(lo), pole + M_UP_WIDTHS * width)
        out.append(dataclasses.replace(
            template, pdg_id=pid, name=name, B=1, Q=q, S=-2, isospin=0.5, isospin3=i3x2, spin=float(spec["spin"]),
            parity=str(spec["parity"]), pole_mass=pole, nominal_width=width, mass_range=mass_range,
            decay_channels=channels, stable_for_decay=bool(spec["stable"]), provenance=_tag(name)))
        mass[pid] = pole
    return out


def xi_baryons_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the Xi records appended (the three-body law carried
    over).  A catalog that already holds one of them is refused."""
    new = _records(catalog, config)
    ids = {int(r.pdg_id) for r in catalog.records()}
    held = sorted(ids & {int(r.pdg_id) for r in new})
    if held:
        raise ValueError(f"xi_baryons: the catalog already holds {held}; the option adds them")
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def xi_classes() -> dict[str, list[str]]:
    """The added classes by family id: (Xi, K) and (Xi(1530), K) at the
    strange baryonic families of charge 0, +1 and -1."""
    return {
        "baryonic_strange": ["(3312,321)", "(3322,311)", "(3314,321)", "(3324,311)"],
        "baryonic_strange_q1": ["(3322,321)", "(3324,321)"],
        "baryonic_strange_qm1": ["(3312,311)", "(3314,311)"],
    }


def xi_baryons_families(fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The families with the Xi classes appended (a class already held is
    not repeated; the other families unchanged).  A kernel without one of
    the strange baryonic families is refused."""
    extra = xi_classes()
    present = {str(f.family_id) for f in fams}
    absent = sorted(set(extra) - present)
    if absent:
        raise ValueError(f"xi_baryons: class families {absent} are absent (the option needs the strange baryonic "
                         "isospin completion)")
    out = []
    for family in fams:
        add = [c for c in extra.get(str(family.family_id), []) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + add)) if add else family)
    return out
