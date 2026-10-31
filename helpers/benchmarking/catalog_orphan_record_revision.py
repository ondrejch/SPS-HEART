"""The orphan-record revision of the committed resonance catalog (2026-10-29).

The Round 2 import (``import_resonance_catalog.py``, card 2) transcribed the
eta'(958) under the card-pinned key 377 (its PDG Monte Carlo id is 331) and
flagged the id for a later revision. No continuum class of the families file
and no decay channel of the catalog names 377, so no kernel ever produced the
record. The eta' of the fitted kernels is the record of
``KernelConfig.eta_prime`` (``na49_mixing/eta_prime.py``: pdg 331 at its PDG
2025 values, adopted in R98), which joins with its classes at the end of the
kernel load. With the option on, the kernel held two eta' records, one of
them unreachable (the catalog readout of 2026-10-29, the probe log's entry on
the radiative decays).

This module is the one definition of the revision, on top of the Round 2
transcription, the card-5 isospin completion and the revisions of 2026-10-02,
2026-10-05 and 2026-10-21. :func:`apply_revision` removes the orphan records
after checking that no channel names them, so the events of every kernel are
unchanged. The radiative revision's table keeps its eta' entry as the record
of what it did; it skips a record the payload does not hold.
"""

from __future__ import annotations

import copy

REVISION_ID = "TASK-20261029-catalog-orphan-etaprime"
ACCESSED_ON = "2026-10-29"

#: The records no class and no channel reaches, by pdg id and name.
ORPHAN_RECORDS = {377: "eta-prime(958)"}


def apply_revision(payload: dict) -> dict:
    """The revised catalog payload (a deep copy): the orphan records removed.
    A channel that names an orphan is an error, and so is an orphan key that
    carries another record's name."""
    new = copy.deepcopy(payload)
    for record in new["records"]:
        for ch in record.get("decay_channels", []):
            named = sorted({int(d) for d in ch["daughters"]} & set(ORPHAN_RECORDS))
            if named:
                raise ValueError(f"catalog orphan revision: record {record['pdg_id']} decays into {named}")
        pdg = int(record["pdg_id"])
        if pdg in ORPHAN_RECORDS and record["name"] != ORPHAN_RECORDS[pdg]:
            raise ValueError(f"catalog orphan revision: record {pdg} is {record['name']!r}, not {ORPHAN_RECORDS[pdg]!r}")
    new["records"] = [r for r in new["records"] if int(r["pdg_id"]) not in ORPHAN_RECORDS]
    return new


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "dev/mixing2006/config/v2_singlet/resonance_catalog.json")
    raw = path.read_text()
    payload = json.loads(raw)
    revised = apply_revision(payload)
    indent = 2 if '\n  "' in raw[:400] else 1
    path.write_text(json.dumps(revised, indent=indent, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print(f"revised {path}: {len(payload['records']) - len(revised['records'])} records removed")
