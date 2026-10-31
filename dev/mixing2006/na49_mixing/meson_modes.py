"""The PDG 2024 decay tables of the cascade's light mesons (2026-10-16).

``KernelConfig.pdg_meson_modes`` replaces the committed records of the
rho0, omega, phi(1020), f2(1270), a2(1320), K1(1270) and K2*(1430) with
their PDG 2024 widths and decay modes, at the masses of the vendored PDG
table that the catalog's validation pins every record to, from
``config/v2_singlet/meson_modes_pdg2024.json``.

The committed catalog cites the PDG 2024 tables, but its meson records
carry an older edition, and they keep only the two-body modes whose
daughters are catalog records, renormalized over them.  The external
review of R99 (2026-10-16; the triage
``_knowledge/operations/physics-review-triage-2026-10-16-r99.md``) found the
f2(1270) without its four-pion modes and the a2(1320) without omega pi pi
and K Kbar; its verification found the K1(1270) at the old mass and without
K0*(1430) pi, and the K2*(1430) without K*(892) pi pi.  The option takes
every mode of at least 1% of the state at its central value, a four-body
mode through its quasi-three-body form and a mode through a state absent
from the catalog through its three-body final state (the file's ``rule``).
The antiparticles of the strange mesons are rebuilt as charge conjugates.
The committed families and classes are unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .conjugate_towers import charge_conjugate
from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag

__all__ = ["MESON_MODES_REL", "load_meson_modes", "meson_modes_catalog"]

#: The configuration, relative to the nested repository root.
MESON_MODES_REL = Path("config") / "v2_singlet" / "meson_modes_pdg2024.json"


def load_meson_modes(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / MESON_MODES_REL).read_text(encoding="utf-8"))


def _tag(name: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(MESON_MODES_REL), section_or_table=name, field="record",
                         accessed_on="2026-10-16", transcribed_value=name, transcriber="na49-res PDG meson modes option")


def _channels(pid: int, spec: dict) -> tuple[DecayChannel, ...]:
    total = math.fsum(float(c["fraction"]) for c in spec["channels"])
    if not total > 0.0:
        raise ValueError(f"pdg_meson_modes: the {spec['name']} fractions sum to {total!r}")
    return tuple(DecayChannel(pid, tuple(int(d) for d in c["daughters"]), float(c["fraction"]) / total, int(c["L"]),
                              len(c["daughters"]) == 3, _tag(spec["name"])) for c in spec["channels"])


def _replaced(record, spec: dict, channels: tuple[DecayChannel, ...], catalog: HadronCatalog):
    pole, width = float(spec["pole_mass"]), float(spec["nominal_width"])
    lo = min(math.fsum(catalog._daughter_mass(d) for d in c.daughters) for c in channels)
    hi = float(record.mass_range[1]) + (pole - float(record.pole_mass))
    return dataclasses.replace(record, pole_mass=pole, nominal_width=width, mass_range=(float(lo), hi),
                               decay_channels=channels, provenance=_tag(spec["name"]))


def meson_modes_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the named meson records replaced (the three-body
    law carried over).  A record or a daughter absent from the catalog is
    refused, and so is a channel that breaks (B, Q, S)."""
    ids = {int(r.pdg_id) for r in catalog.records()}
    known = ids | set(int(p) for p in catalog.stable_states())
    new = {}
    for spec in config["records"]:
        pid = int(spec["pdg_id"])
        if pid not in ids:
            raise ValueError(f"pdg_meson_modes: the record {pid} ({spec['name']}) is absent from the catalog")
        record = catalog.get(pid)
        channels = _channels(pid, spec)
        missing = sorted({d for c in channels for d in c.daughters} - known)
        if missing:
            raise ValueError(f"pdg_meson_modes: daughters {missing} of {spec['name']} are not catalog records")
        new[pid] = _replaced(record, spec, channels, catalog)
        if "conjugate_id" in spec:
            cid = int(spec["conjugate_id"])
            if cid not in ids or cid != -pid:
                raise ValueError(f"pdg_meson_modes: the conjugate {cid} of {spec['name']} is absent or not its antiparticle")
            anti = catalog.get(cid)
            conj = tuple(dataclasses.replace(c, parent_pdg=cid, daughters=tuple(charge_conjugate(d, catalog) for d in c.daughters))
                         for c in channels)
            new[cid] = _replaced(anti, spec, conj, catalog)
    out = HadronCatalog(tuple(new.get(int(r.pdg_id), r) for r in catalog.records()), three_body_law=catalog.three_body_law)
    from .resonance_decay import _daughter_bqs
    for rec in new.values():
        for c in rec.decay_channels:
            bqs = tuple(sum(v) for v in zip(*(_daughter_bqs(int(d), out) for d in c.daughters)))
            if bqs != (int(rec.B), int(rec.Q), int(rec.S)):
                raise ValueError(f"pdg_meson_modes: {rec.name} -> {c.daughters} breaks (B, Q, S)")
    return out
