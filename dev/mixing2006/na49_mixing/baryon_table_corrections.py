"""The PDG 2025 tables of the committed Delta(1600) and Delta(1700), and the
PDG 2025 widths of the committed N* and Delta records (2026-10-21).

``KernelConfig.baryon_table_corrections`` replaces the decay tables of the
four charge states of Delta(1600) and Delta(1700) with their PDG 2025 modes,
from ``config/v2_singlet/baryon_table_corrections_pdg2025.json``.

The PDG updates of R84 and R100 replaced the N* tables and left the two
Delta resonances at the legacy Round 2 transcription: N pi, Delta pi and
N rho only. The PDG 2025 Delta(1600) lists no N rho (the committed table
holds 27%) and an N(1440) pi of 17-27%; the Delta(1700) adds N(1520) pi,
N(1535) pi and Delta eta. The Gemini review of 2026-10-21 (`.collab/`,
gap B3) named the two. Its gap B4 item 3, the Lambda(1520) Lambda pi pi
written as its charged member alone, is a transcription error and a revision
of the committed catalog (``helpers/benchmarking/
catalog_dipion_isospin_revision.py``).

The Delta fractions follow the N* range rule
(:func:`na49_mixing.isoscalar_dipion.range_position_fractions`) and the
charge splits the isospin Clebsch-Gordan shares of
:mod:`na49_mixing.pdg_nucleon_resonances`. The masses and widths stay the
committed ones, as the N* tables keep theirs. The option enters before the
conjugate towers, which conjugate the records.

``KernelConfig.baryon_pdg_widths`` sets the committed N(1440), N(1520),
N(1680), N(1720) and Delta(1600) at the PDG 2025 Breit-Wigner width
estimates (the file's ``widths``), the values the PDG 2025 records of
:mod:`na49_mixing.pdg_nucleon_resonances` carry: 0.30 -> 0.35, 0.115 ->
0.11, 0.13 -> 0.12, 0.20 -> 0.25 and 0.35 -> 0.25 GeV (the last outside the
PDG range of 0.20-0.30). The widths shape the duality picks of the
low-mass lines through the records' line shapes (R102). The masses stay
the vendored table's, which the catalog's validation pins them to.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .hadron_catalog import DecayChannel, HadronCatalog, ProvenanceTag

__all__ = ["BARYON_TABLE_CORRECTIONS_REL", "load_baryon_table_corrections", "baryon_table_corrections_catalog",
           "baryon_pdg_widths_catalog", "M_UP_WIDTHS"]

#: The line shape's upper edge in widths above the pole (the committed N* and Delta records' rule).
M_UP_WIDTHS = 5.0

#: The configuration, relative to the nested repository root.
BARYON_TABLE_CORRECTIONS_REL = Path("config") / "v2_singlet" / "baryon_table_corrections_pdg2025.json"


def load_baryon_table_corrections(nested_root) -> dict:
    """The parsed configuration (``nested_root`` is the ``dev/mixing2006`` root)."""
    return json.loads((Path(nested_root) / BARYON_TABLE_CORRECTIONS_REL).read_text(encoding="utf-8"))


def _tag(name: str) -> ProvenanceTag:
    return ProvenanceTag(document=str(BARYON_TABLE_CORRECTIONS_REL), section_or_table=name, field="branching_fraction",
                         accessed_on="2026-10-21", transcribed_value=name,
                         transcriber="na49-res baryon table corrections option")


def _record(catalog: HadronCatalog, pdg: int, name: str):
    try:
        return catalog.get(int(pdg))
    except KeyError:
        raise ValueError(f"baryon_table_corrections: the record {pdg} ({name}) is absent from the catalog") from None


def _delta_channels(record, mult: dict, catalog: HadronCatalog) -> tuple[DecayChannel, ...]:
    from .isoscalar_dipion import range_position_fractions
    from .pdg_nucleon_resonances import MODE_KINDS, _split
    iso, charge = float(mult["I"]), int(record.Q)
    out = []
    for mode, fraction in zip(mult["modes"], range_position_fractions(mult["modes"])):
        three = len(MODE_KINDS[mode["kind"]]) == 3
        for daughters, share in _split(iso, 2 * charge - 1, mode["kind"], catalog):
            out.append(DecayChannel(int(record.pdg_id), tuple(daughters), fraction * share, int(mode["L"]), three,
                                    _tag(mult["name"])))
    return tuple(out)


def baryon_table_corrections_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the configuration's Delta tables replaced (the
    three-body law carried over). An absent record, a
    record whose isospin or charge disagrees with the configuration, and a
    channel that breaks (B, Q, S) are refused."""
    new = {}
    for mult in config["delta_tables"]:
        for charge_text, pdg in mult["pdg_ids"].items():
            record = _record(catalog, pdg, mult["name"])
            if float(record.isospin) != float(mult["I"]) or int(record.Q) != int(charge_text):
                raise ValueError(f"baryon_table_corrections: the record {pdg} is not the {mult['name']} of charge {charge_text}")
            new[int(pdg)] = dataclasses.replace(record, decay_channels=_delta_channels(record, mult, catalog))
    out = HadronCatalog(tuple(new.get(int(r.pdg_id), r) for r in catalog.records()), three_body_law=catalog.three_body_law)
    from .resonance_decay import _daughter_bqs
    for rec in new.values():
        if not math.isclose(math.fsum(c.branching_fraction for c in rec.decay_channels), 1.0, rel_tol=1e-12):
            raise ValueError(f"baryon_table_corrections: the {rec.name} fractions do not sum to one")
        for c in rec.decay_channels:
            bqs = tuple(sum(v) for v in zip(*(_daughter_bqs(int(d), out) for d in c.daughters)))
            if bqs != (int(rec.B), int(rec.Q), int(rec.S)):
                raise ValueError(f"baryon_table_corrections: {rec.name} -> {c.daughters} breaks (B, Q, S)")
    return out


def baryon_pdg_widths_catalog(catalog: HadronCatalog, config: dict) -> HadronCatalog:
    """The catalog with the configuration's N* and Delta records at their
    PDG 2025 widths, each support's upper edge at :data:`M_UP_WIDTHS` widths
    above the pole (the three-body law carried over). An absent record is
    refused."""
    new = {}
    for spec in config["widths"]:
        width = float(spec["nominal_width"])
        if not width > 0.0:
            raise ValueError(f"baryon_pdg_widths: the {spec['name']} width {width!r} is not positive")
        for pdg in spec["pdg_ids"]:
            record = _record(catalog, pdg, spec["name"])
            lo = float(record.mass_range[0])
            new[int(pdg)] = dataclasses.replace(record, nominal_width=width,
                                                mass_range=(lo, float(record.pole_mass) + M_UP_WIDTHS * width))
    return HadronCatalog(tuple(new.get(int(r.pdg_id), r) for r in catalog.records()), three_body_law=catalog.three_body_law)
