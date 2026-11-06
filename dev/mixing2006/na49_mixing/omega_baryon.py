"""The S = -3 baryon (2026-11-04).

``KernelConfig.omega_baryon`` adds the Omega- of the PDG 2025 summary tables
from ``config/v2_singlet/omega_baryon_pdg2025.json``, stable here as the Xi
are (its decays are weak), and the classes (Omega-, K+) and (Omega-, K0) at
the S = -2 excited baryonic continuum (``na49_mixing.xi_tower``) with their
conjugates.  The record joins before the charge-conjugate towers, which
build the anti-Omega+ from it.

The committed catalog holds no baryon of strangeness -3, so the cascade
makes no Omega.  An S = -2 line reaches the Omega by creating one more
s-sbar pair (gamma_s), and under ``baryon_pair_completion`` a mesonic state
creates Omega Omegabar (three s-sbar pairs) and, at the strange mesonic
families, Omega- Xibar+ (two).  The pairs give Omega and anti-Omega in equal
numbers; the baryon-number-carrying lines give Omega alone.  The ratio
anti-Omega+/Omega- therefore measures how much of the Omega a baryon-free
source makes.  NA49 reports a preliminary 0.67 +- 0.62 at mid-rapidity in
p+p at 158 GeV/c (uncorrected; M. Mitrovski for NA49, SQM 2003,
nucl-ex/0305029).  The option adds no parameter.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .hadron_catalog import HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["OMEGA_BARYON_REL", "OMEGA", "load_omega_baryon", "omega_baryon_catalog", "omega_baryon_families"]

#: The configuration, relative to the nested repository root.
OMEGA_BARYON_REL = Path("config") / "v2_singlet" / "omega_baryon_pdg2025.json"
#: The Omega- pdg id.
OMEGA = 3334
#: The template record: the Xi-, the stable S = -2 baryon of the same charge.
_TEMPLATE = 3312


def load_omega_baryon(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / OMEGA_BARYON_REL).read_text(encoding="utf-8"))


def _tag(name: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(OMEGA_BARYON_REL), section_or_table=name, field="record",
                         accessed_on="2026-11-04", transcribed_value=name, transcriber="na49-res Omega baryon option")


def omega_baryon_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the Omega- appended (the three-body law carried
    over).  A catalog without the Xi- template or already holding the Omega-
    is refused."""
    ids = {int(r.pdg_id) for r in catalog.records()}
    if OMEGA in ids:
        raise ValueError("omega_baryon: the catalog already holds the Omega-; the option adds it")
    if _TEMPLATE not in ids:
        raise ValueError("omega_baryon: the catalog has no Xi- (the option needs xi_baryons)")
    template = catalog.get(_TEMPLATE)
    new = []
    for spec in config["records"]:
        pole = float(spec["pole_mass"])
        if not spec["stable"]:
            raise ValueError("omega_baryon: the records are stable (the Omega- decays weakly)")
        new.append(dataclasses.replace(
            template, pdg_id=int(spec["pdg_id"]), name=str(spec["name"]), B=1, Q=-1, S=-3, isospin=0.0, isospin3=0,
            spin=float(spec["spin"]), parity=str(spec["parity"]), pole_mass=pole,
            nominal_width=float(spec["nominal_width"]), mass_range=(pole, pole), decay_channels=(),
            stable_for_decay=True, provenance=_tag(str(spec["name"]))))
    return HadronCatalog(tuple(catalog.records()) + tuple(new), three_body_law=catalog.three_body_law)


def omega_baryon_families(fams: list[ContinuumFamily], config: dict) -> list[ContinuumFamily]:
    """The families with the Omega classes appended (a class already held is
    not repeated; the other families unchanged).  A kernel without one of
    the S = -2 continuum families is refused."""
    extra = {str(k): list(v) for k, v in config["classes"].items()}
    absent = sorted(set(extra) - {str(f.family_id) for f in fams})
    if absent:
        raise ValueError(f"omega_baryon: class families {absent} are absent (the option needs the xi tower)")
    out = []
    for family in fams:
        add = [c for c in extra.get(str(family.family_id), []) if c not in family.allowed_decay_classes]
        out.append(dataclasses.replace(family, allowed_decay_classes=tuple(family.allowed_decay_classes) + tuple(add))
                   if add else family)
    return out
