"""The eta'(958), the ninth member of the pseudoscalar nonet (2026-10-15).

``KernelConfig.eta_prime`` adds the eta'(958) record at its PDG 2025 values
(``config/v2_singlet/eta_prime_pdg2025.json``) and, beside every continuum
class with an eta, the class with the eta' in its place.

The committed catalog carries the vector nonet in full (rho, omega,
K*(892), phi) and the pseudoscalar nonet without the eta'.  The eta'
decays to eta pi pi in 65% of its decays, at a Q value of 0.13 GeV, and to
rho0 gamma in 29%: a soft multi-pion source and a source of eta.  At the R97
optimum the cascade makes 1.6 times NA27's omega, whose three-body pions
carry the central soft pions, and 0.69 of NA27's eta.  The decision is the
model decision register (``_knowledge/paper/model-decision-register.md``)
and the probe the fixed-kernel note (``PROBE_ETAPRIME``), of which this
option is the kernel form.

The eta' takes the eta's record as its template (I = 0, J^P = 0-).  Its
hidden s-sbar content is the flavor weights' entry for pdg 331
(``na49_mixing.flavor_weights.DEFAULT_HIDDEN_SSBAR_CONTENT``).  The
classes join after the conjugate towers and the (N, Kbar*) pairs, at the
end of the kernel load, as the probe adds them.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag
from .spectral_density import ContinuumFamily

__all__ = ["ETA_PRIME_REL", "ETA_PRIME_PDG", "load_eta_prime", "eta_prime_catalog", "eta_prime_families"]

#: The configuration, relative to the nested repository root.
ETA_PRIME_REL = Path("config") / "v2_singlet" / "eta_prime_pdg2025.json"
ETA_PRIME_PDG = 331
_ETA_PDG = 221
#: The line shape's upper edge in widths above the pole.
M_UP_WIDTHS = 5.0


def load_eta_prime(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / ETA_PRIME_REL).read_text(encoding="utf-8"))


def _tag(field: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(ETA_PRIME_REL), section_or_table="eta'(958)", field=field, accessed_on="2026-10-15",
                         transcribed_value="eta'(958)", transcriber="na49-res eta prime option")


def eta_prime_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the eta' record appended (the three-body law carried
    over).  A catalog that already holds it is refused."""
    spec = config["record"]
    if int(spec["pdg_id"]) in {int(r.pdg_id) for r in catalog.records()}:
        raise ValueError("eta_prime: the catalog already holds the eta'(958); the option adds it")
    total = math.fsum(float(m["fraction"]) for m in spec["modes"])
    channels = tuple(DecayChannel(int(spec["pdg_id"]), tuple(int(d) for d in m["daughters"]), float(m["fraction"]) / total,
                                  int(m["L"]), bool(m["three_body"]), _tag("branching_fraction")) for m in spec["modes"])
    pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
    record = dataclasses.replace(catalog.get(_ETA_PDG), pdg_id=int(spec["pdg_id"]), name=str(spec["name"]), pole_mass=pole,
                                 nominal_width=width, mass_range=(0.0, pole + M_UP_WIDTHS * width), decay_channels=channels,
                                 stable_for_decay=False, provenance=_tag("record"))
    return HadronCatalog(tuple(catalog.records()) + (record,), three_body_law=catalog.three_body_law)


def eta_prime_classes(classes) -> list[str]:
    """The eta' analogs of a family's classes: each eta of a class replaced,
    and both in a class of two, in the order of the classes."""
    out = []
    # a class is an unordered pair: (eta', eta) and (eta, eta') of an (eta, eta) class are one class
    seen = {tuple(sorted(t.strip() for t in str(c).strip("()").split(","))) for c in classes}
    for c in classes:
        toks = [t.strip() for t in str(c).strip("()").split(",")]
        pos = [i for i, t in enumerate(toks) if t == str(_ETA_PDG)]
        for k in range(1, len(pos) + 1):
            for sub in itertools.combinations(pos, k):
                new_toks = [str(ETA_PRIME_PDG) if i in sub else t for i, t in enumerate(toks)]
                key = tuple(sorted(new_toks))
                if key not in seen:
                    seen.add(key)
                    out.append("(" + ",".join(new_toks) + ")")
    return out


def eta_prime_families(fams: list[ContinuumFamily]) -> list[ContinuumFamily]:
    """The families with the eta' classes appended after their own."""
    out = []
    for family in fams:
        extra = eta_prime_classes(list(family.allowed_decay_classes))
        out.append(dataclasses.replace(family, allowed_decay_classes=type(family.allowed_decay_classes)(
            list(family.allowed_decay_classes) + extra)) if extra else family)
    return out
