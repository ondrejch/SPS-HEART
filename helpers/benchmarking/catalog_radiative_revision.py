"""The radiative revision of the committed resonance catalog (2026-10-02).

The Round 2 import (``import_resonance_catalog.py``, card 2) represented
only the hadronic channels of eta, eta-prime, omega and phi.  The decay
kernel renormalizes over the represented channels, so the folded-out
radiative modes turned into hadronic ones: 0.41 pi+ pi- pairs per eta
instead of 0.27, 1.00 per omega instead of 0.91, and a K0S K0L share of
0.508 for phi that had absorbed its rho pi and eta gamma modes (the
feed-down and feed-over review,
``_knowledge/evidence/feed-down-feed-over-review-2026-10-02.md``).

This module is the one definition of the revision.  It sits on top of the
Round 2 transcription and the card-5 isospin completion, as card 5 sits on
top of card 2: the import driver and its sealed run record stay as they
are, :func:`apply_revision` turns a catalog payload into the revised one,
and the catalog tests compare the four revised records against
:data:`REVISED_CHANNELS`.  The photon (22) is a stable non-hadronic final
state (``hadron_catalog.NON_HADRONIC_FINAL_STATES``), never a record.

Sources:
- eta and omega: the repository's legacy decay table
  (``na49_mixing/decays.py``, ``build_soft_resonance_decays``; PDG 2025
  summary values), whose radiative entries the import had folded out;
- eta-prime(958) and phi(1020): PDG Live, 2026 release (accessed
  2026-10-02).
"""

from __future__ import annotations

import copy

REVISION_ID = "TASK-20261002-kernel-feedover-threshold-radiative"
ACCESSED_ON = "2026-10-02"
TRANSCRIBER = "na49-res feed-down and feed-over review fix (2026-10-02)"
PHOTON = 22
PI0_MASS = 0.1349766

DOC_LEGACY = ("na49_mixing/decays.py build_soft_resonance_decays (legacy decay table; "
              "PDG 2025 summary values)")
DOC_ETAPRIME = ("PDG Live, eta'(958) decay modes (2026 release), "
                "pdglive.lbl.gov/Particle.action?node=M002")
DOC_PHI = ("PDG Live, phi(1020) decay modes (2026 release), "
           "pdglive.lbl.gov/Particle.action?node=M004")

_OMEGA_NORM = 0.892 + 0.0833 + 0.0153

#: The revised channel tables: ``pdg -> [(daughters, branching_fraction,
#: L, is_three_body, document, section, new)]``.  ``new`` is False for a
#: channel that keeps its committed entry (value and provenance) and True
#: for a channel this revision transcribes.  L follows the catalog's
#: convention (the lowest parity-allowed partial wave; a photon is 1-).
REVISED_CHANNELS: dict[int, list[tuple]] = {
    221: [
        ((211, -211, 111), 0.2302, 0, True, None, None, False),
        ((111, 111, 111), 0.3256, 0, True, None, None, False),
        ((211, -211, 22), 0.0428, 0, True, DOC_LEGACY, "eta(548) block: pi+ pi- gamma", True),
        ((22, 22), 0.3936, 1, False, DOC_LEGACY, "eta(548) block: gamma gamma", True),
    ],
    223: [
        ((211, -211, 111), 0.892 / _OMEGA_NORM, 0, True, DOC_LEGACY,
         "omega(782) block: normalized over pi+ pi- pi0, pi0 gamma and pi+ pi-", True),
        ((111, 22), 0.0833 / _OMEGA_NORM, 1, False, DOC_LEGACY,
         "omega(782) block: normalized over pi+ pi- pi0, pi0 gamma and pi+ pi-", True),
        ((211, -211), 0.0153 / _OMEGA_NORM, 1, False, DOC_LEGACY,
         "omega(782) block: normalized over pi+ pi- pi0, pi0 gamma and pi+ pi-", True),
    ],
    377: [
        ((211, -211, 221), 0.425, 0, True, None, None, False),
        ((111, 111, 221), 0.224, 0, True, None, None, False),
        ((113, 22), 0.2948, 1, False, DOC_ETAPRIME, "rho0 gamma (including non-resonant pi+ pi- gamma)", True),
        ((223, 22), 0.0252, 1, False, DOC_ETAPRIME, "omega gamma", True),
        ((22, 22), 0.02307, 1, False, DOC_ETAPRIME, "gamma gamma", True),
        ((111, 111, 111), 0.0025, 0, True, DOC_ETAPRIME, "3 pi0", True),
        ((211, -211, 111), 0.00361, 0, True, DOC_ETAPRIME, "pi+ pi- pi0", True),
    ],
    333: [
        ((321, -321), 0.501, 1, False, DOC_PHI, "K+ K-", True),
        ((310, 130), 0.336, 1, False, DOC_PHI, "K0L K0S", True),
        ((213, -211), 0.049, 1, False, DOC_PHI,
         "rho pi + pi+ pi- pi0 (14.7%), one third per rho pi charge state (I = 0 parent)", True),
        ((113, 111), 0.049, 1, False, DOC_PHI,
         "rho pi + pi+ pi- pi0 (14.7%), one third per rho pi charge state (I = 0 parent)", True),
        ((-213, 211), 0.049, 1, False, DOC_PHI,
         "rho pi + pi+ pi- pi0 (14.7%), one third per rho pi charge state (I = 0 parent)", True),
        ((221, 22), 0.01306, 1, False, DOC_PHI, "eta gamma", True),
        ((111, 22), 0.00133, 1, False, DOC_PHI, "pi0 gamma", True),
    ],
}

#: The remaining unrepresented budget of the revised records (Dalitz,
#: leptonic and sub-0.3% modes), written to the ``radiative_budget`` field
#: where the record carries one.
def residual_budget(pdg: int) -> float:
    return 1.0 - sum(br for _d, br, *_ in REVISED_CHANNELS[pdg])


def _threshold(daughters, masses: dict[int, float]) -> float:
    return sum(0.0 if int(d) == PHOTON else float(masses[int(d)]) for d in daughters)


def apply_revision(payload: dict) -> dict:
    """The revised catalog payload: the four records' channel tables
    replaced by :data:`REVISED_CHANNELS` (kept channels keep their entries),
    each support's lower edge moved to the lowest channel threshold, and
    the ``radiative_budget`` of eta and eta-prime set to the remaining
    budget.  Every other byte of the payload is unchanged."""
    out = copy.deepcopy(payload)
    masses = {int(r["pdg_id"]): float(r["pole_mass"]) for r in out["records"]}
    for record in out["records"]:
        pdg = int(record["pdg_id"])
        if pdg not in REVISED_CHANNELS:
            continue
        old = {tuple(int(d) for d in c["daughters"]): c for c in record["decay_channels"]}
        channels = []
        for daughters, br, L, three, document, section, new in REVISED_CHANNELS[pdg]:
            if not new:
                kept = old[tuple(daughters)]
                if abs(float(kept["branching_fraction"]) - br) > 1e-15:
                    raise ValueError(f"record {pdg}: the kept channel {daughters} changed value")
                channels.append(kept)
                continue
            channels.append({
                "parent_pdg": pdg, "daughters": list(daughters), "branching_fraction": br,
                "orbital_angular_momentum": L, "is_three_body": three,
                "provenance": {"document": document, "section_or_table": section, "field": "branching_fraction",
                               "accessed_on": ACCESSED_ON, "transcribed_value": br, "transcriber": TRANSCRIBER},
            })
        record["decay_channels"] = channels
        record["mass_range"][0] = min(_threshold(c["daughters"], masses) for c in channels)
        if "radiative_budget" in record:
            record["radiative_budget"] = round(residual_budget(pdg), 10)
    return out
