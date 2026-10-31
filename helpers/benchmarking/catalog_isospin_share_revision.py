"""The isospin-share revision of the committed resonance catalog (2026-10-05).

The code and physics review of 2026-10-05 (``.collab/reviews/fable-code-
physics-review-2026-10-05.md``, finding M-4; card
``TASK-20261005-catalog-delta1700-charge-shares``) found two departures of
the transcribed charge shares from the Clebsch-Gordan values, both flagged
by the catalog's own ``isospin_closure_report`` as warnings:

1. Delta(1700)0 (pdg 122114) -> N pi: the transcription carries p pi- 0.10
   and n pi0 0.05.  For I = 3/2, I3 = -1/2 the shares are 1/3 for p pi- and
   2/3 for n pi0, so the two values are swapped (the Delta(1700)+ record
   122214 carries p pi0 0.10 and n pi+ 0.05, correctly).
2. The N*+ records (12212, 2216, 22212, 122212, 212214) split their Delta pi
   class with the rounded shares 0.33 and 0.17 for Delta+ pi0 and Delta0 pi+
   (and their N*0 partners carry the exact 1/3 and 1/6), so the mirror check
   fails by 1 to 2%.  The revision restores the exact shares at the
   unchanged class total, in the antiparticle record too where the
   catalog carries one (-212214).

This module is the one definition of the revision, on top of the Round 2
transcription, the card-5 isospin completion and the radiative revision.
:func:`apply_revision` turns a catalog payload into the revised one; the
catalog tests compare against :data:`REVISED_SHARES`.  The phi(1020)
K+K- / K0S K0L split is physical (phase space near threshold) and stays.
"""

from __future__ import annotations

import copy
from fractions import Fraction

REVISION_ID = "TASK-20261005-catalog-delta1700-charge-shares"
ACCESSED_ON = "2026-10-05"
TRANSCRIBER = "na49-res code and physics review, card 4 (2026-10-05)"

#: The Delta(1700)0 N pi channels and their Clebsch-Gordan shares of the class.
DELTA1700_0 = 122114
DELTA1700_0_NPI = {(2212, -211): Fraction(1, 3), (2112, 111): Fraction(2, 3)}

#: The N*+ records whose Delta pi class carries rounded shares, and the exact
#: shares of an I = 1/2, I3 = +1/2 parent: Delta++ pi- 1/2, Delta+ pi0 1/3,
#: Delta0 pi+ 1/6.
NSTAR_PLUS = (12212, 2216, 22212, 122212, 212214)
NSTAR_PLUS_DELTA_PI = {(2224, -211): Fraction(1, 2), (2214, 111): Fraction(1, 3), (2114, 211): Fraction(1, 6)}


#: The self-conjugate daughters of the revised classes.
_SELF_CONJUGATE = frozenset({111})


def _conjugate(shares: dict) -> dict:
    """The shares of the charge-conjugate record (the antiparticle of a
    revised parent): each daughter conjugated, the shares unchanged."""
    return {tuple(d if d in _SELF_CONJUGATE else -d for d in key): value for key, value in shares.items()}


def _targets(records: dict) -> list:
    out = [(DELTA1700_0, DELTA1700_0_NPI)] + [(p, NSTAR_PLUS_DELTA_PI) for p in NSTAR_PLUS]
    for pdg, shares in list(out):
        if -pdg in records:
            out.append((-pdg, _conjugate(shares)))
    return out


def _class_channels(record: dict, shares: dict) -> list:
    out = []
    for ch in record.get("decay_channels", []):
        key = tuple(int(d) for d in ch["daughters"])
        for k in (key, key[::-1]):
            if k in shares:
                out.append((ch, shares[k]))
                break
    return out


def _restamp(ch: dict, value: float, note: str) -> None:
    previous = float(ch["branching_fraction"])
    ch["branching_fraction"] = value
    prov = dict(ch.get("provenance", {}))
    prov["section_or_table"] = f"{prov.get('section_or_table', '')}; {note} (was {previous!r})".lstrip("; ")
    prov["accessed_on"] = ACCESSED_ON
    prov["transcribed_value"] = value
    prov["transcriber"] = TRANSCRIBER
    ch["provenance"] = prov


def revised_shares(payload: dict) -> dict:
    """The revised branching fractions {(pdg, daughters): value} that
    :func:`apply_revision` writes (computed from the payload's class totals)."""
    out = {}
    records = {int(r["pdg_id"]): r for r in payload["records"]}
    for pdg, shares in _targets(records):
        chans = _class_channels(records[pdg], shares)
        if len(chans) != len(shares):
            raise ValueError(f"catalog isospin-share revision: record {pdg} lacks the channels {sorted(shares)}")
        total = sum(Fraction(str(ch["branching_fraction"])) for ch, _s in chans)
        for ch, share in chans:
            out[(pdg, tuple(int(d) for d in ch["daughters"]))] = round(float(total * share), 9)
    return out


def apply_revision(payload: dict) -> dict:
    """The revised catalog payload (a deep copy; the class totals are kept)."""
    new = copy.deepcopy(payload)
    values = revised_shares(new)
    for r in new["records"]:
        pdg = int(r["pdg_id"])
        for ch in r.get("decay_channels", []):
            key = (pdg, tuple(int(d) for d in ch["daughters"]))
            if key in values and float(ch["branching_fraction"]) != values[key]:
                note = ("isospin-share revision: the Clebsch-Gordan share of the class" if abs(pdg) == DELTA1700_0
                        else "isospin-share revision: the exact Clebsch-Gordan share of the Delta pi class")
                _restamp(ch, values[key], note)
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
    changed = sum(1 for a, b in zip(payload["records"], revised["records"]) if a != b)
    print(f"revised {path}: {changed} records changed")
