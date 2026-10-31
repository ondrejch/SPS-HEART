"""The dipion isospin revision of the committed resonance catalog (2026-10-21).

The Gemini review of R104 (2026-10-21, ``.collab/``, gap B4 item 3) found
the Lambda(1520) (pdg 102134) carrying its Lambda pi pi mode (10% in the PDG
2025 table) as Lambda pi+ pi- alone. The Lambda(1520) and the Lambda are both
I = 0, so the dipion is I = 0, and an I = 0 pion pair is pi+ pi- : pi0 pi0
= 2 : 1. The Round 2 transcription (``na49_mixing/decays.py``,
``build_soft_resonance_decays``, the Lambda(1520) block) wrote the whole mode
as its charged member, which overstates the Lambda(1520)'s charged pions from
this mode by a factor of 1.5 and leaves out its pi0 pairs. The scan of every
three-body mode of the catalog against the three-body isospin share
(``tests/test_catalog_dipion_isospin_revision.py``) finds no other departure
besides the documented ones.

This module is the one definition of the revision, on top of the Round 2
transcription, the card-5 isospin completion, the radiative revision and the
isospin-share revision of 2026-10-05. :func:`apply_revision` turns a catalog
payload into the revised one: the mode's total fraction and orbital wave are
kept, and the pi0 pi0 member joins after the charged one.
"""

from __future__ import annotations

import copy
from fractions import Fraction

REVISION_ID = "TASK-20261021-catalog-dipion-isospin"
ACCESSED_ON = "2026-10-21"
TRANSCRIBER = "na49-res external review of R104, gap B4 (2026-10-21)"

#: The records whose I = 0 dipion mode carries only its charged member, the
#: charged member's daughters and the dipion's charge shares.
LAMBDA1520 = 102134
DIPION_MODES = {
    LAMBDA1520: ((3122, 211, -211), {(3122, 211, -211): Fraction(2, 3), (3122, 111, 111): Fraction(1, 3)}),
}

#: The self-conjugate daughters of the revised modes.
_SELF_CONJUGATE = frozenset({111})


def _conjugate(key: tuple) -> tuple:
    return tuple(d if d in _SELF_CONJUGATE else -d for d in key)


def _targets(records: dict) -> list:
    out = [(pdg, charged, shares) for pdg, (charged, shares) in DIPION_MODES.items()]
    for pdg, charged, shares in list(out):
        if -pdg in records:
            out.append((-pdg, _conjugate(charged), {_conjugate(k): v for k, v in shares.items()}))
    return out


def _same(a, b) -> bool:
    return sorted(int(d) for d in a) == sorted(int(d) for d in b)


def revised_fractions(payload: dict) -> dict:
    """The revised branching fractions {(pdg, sorted daughters): value} that
    :func:`apply_revision` writes (computed from the payload's mode totals)."""
    records = {int(r["pdg_id"]): r for r in payload["records"]}
    out = {}
    for pdg, charged, shares in _targets(records):
        chans = [ch for ch in records[pdg].get("decay_channels", []) if any(_same(ch["daughters"], k) for k in shares)]
        if not any(_same(ch["daughters"], charged) for ch in chans):
            raise ValueError(f"catalog dipion revision: record {pdg} lacks the channel {charged}")
        total = sum(Fraction(str(ch["branching_fraction"])) for ch in chans)
        for key, share in shares.items():
            out[(pdg, tuple(sorted(key)))] = float(total * share)
    return out


def apply_revision(payload: dict) -> dict:
    """The revised catalog payload (a deep copy; the mode totals are kept)."""
    new = copy.deepcopy(payload)
    values = revised_fractions(new)
    records = {int(r["pdg_id"]): r for r in new["records"]}
    for pdg, charged, shares in _targets(records):
        chans = records[pdg]["decay_channels"]
        template = next(ch for ch in chans if _same(ch["daughters"], charged))
        pristine = copy.deepcopy(template)
        for key in shares:
            value = values[(pdg, tuple(sorted(key)))]
            have = [ch for ch in chans if _same(ch["daughters"], key)]
            if have:
                ch = have[0]
                if float(ch["branching_fraction"]) == value:
                    continue
                previous = float(ch["branching_fraction"])
            else:
                ch = copy.deepcopy(pristine)
                ch["daughters"] = list(key)
                chans.insert(chans.index(template) + 1, ch)
                previous = 0.0
            ch["branching_fraction"] = value
            prov = dict(ch.get("provenance", {}))
            prov["section_or_table"] = (f"{prov.get('section_or_table', '')}; dipion isospin revision: the I = 0 "
                                        f"pi pi share of the mode (was {previous!r})").lstrip("; ")
            prov["accessed_on"] = ACCESSED_ON
            prov["transcribed_value"] = value
            prov["transcriber"] = TRANSCRIBER
            ch["provenance"] = prov
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
