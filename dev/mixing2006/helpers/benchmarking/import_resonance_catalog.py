#!/usr/bin/env python3
"""Provenance-tagged initial resonance catalog import (Round 2, card 2).

Task ``TASK-20260913-r2-provenance-catalog-import`` (plan section 22,
task 10; plan section 8.1 step 1).  The driver imports the
provenance-tagged initial resonance catalog into
``config/v2_singlet/resonance_catalog.json`` (the plan section 4.4
catalog file) and writes the acceptance record
``catalog_import_report.json``.

The record set (plan section 8.1 as resolved by the round 2 index,
items 2 and 11):

- every unstable established hadron of the vendored PDG table
  (``na49_mixing/pdg_database.py``), excluding the ``N_eff``/``M_eff``
  Regge placeholder towers (PDG 9911201, 9912201, 9921201, 9922201,
  9931201, 9932201 and their anti partners; 9900113, 9900213, 9900313),
  the muons (leptons, not established hadrons), and the NA49 FSB
  ``cluster`` bookkeeping object (id 2219, not an established hadron);
- plus eta-prime, added with PDG provenance (the vendored table does
  not carry it).  The card pins the catalog key PDG 377 for eta-prime;
  the PDG Monte Carlo id of eta'(958) is 331 (PDG 2024 Monte Carlo
  numbering), so the record carries the card-pinned key 377 with the
  name "eta-prime(958)" and the id correction is recorded here and in
  the run record (a later catalog-revision card reconciles the key,
  plan section 4.2);
- plus the pre-registered stable-state termination set (p, n, pbar,
  nbar, pi+-, pi0, K+-, K0S, K0L) as ``stable_for_decay`` records, and
  seven further cascade-terminal ground states (Lambda, aLambda,
  Sigma+, Sigma0, Sigma-, K0, aK0): none of these has an open strong
  channel at its pole mass (their decays are weak or electromagnetic),
  so the card's termination rule ("every unstable record has at least
  one open channel at its pole_mass") forces ``stable_for_decay=True``.
  Their presence as records keeps the Lambda(1520) / Sigma(1385) /
  K-star / N-star channels resolvable (card-1 daughter coverage) while
  their weak/EM decays stay outside the strong cascade, exactly like
  the K0S/K0L weak-decay endpoints of index item 11.

The transcribed physics values are data in this module (the documented
manual step of the card); the driver only reads them, validates, and
writes.  Branching fractions and particle properties are external
physics inputs, not fit parameters (plan section 8.2):

- repository-local sources: the legacy per-charge Clebsch-weighted
  decay tables of ``na49_mixing/decays.py``
  (``build_soft_resonance_decays``, ``build_kaon_decays``,
  ``build_omega_decays``, ``build_rho_decays``), the Breit-Wigner
  upper mass cuts (``dumphigh``) those builders document for the
  rho(770), Delta(1232), and K*(892) spectral daughters, and the
  PDG-2025-anchored subset configuration
  ``helpers/benchmarking/config/soft_resonance_branching_pdg2025.json``;
- external sources: the PDG 2024/2025 summary and listing tables, and
  for eta-prime the PDG 2024 light-unflavored-meson summary table
  (mass 957.78 MeV, full width 0.188 MeV, pi+ pi- eta (42.5 +- 0.5) %,
  pi0 pi0 eta (22.4 +- 0.5) %).

Orbital angular momenta follow one pre-registered rule: the lowest
partial wave L consistent with the parent J^P and the daughter parity
product, (−1)^L = P_parent / prod(P_daughters), within the angular-
momentum range |J − j| <= L <= J + j (j = daughter spin sum); three-body
channels carry the total L of that wave (the card-1 ``DecayChannel``
convention).  The m_up values follow the repository where it documents
an upper mass cut: the Breit-Wigner ``dumphigh`` parameters of the
``decays.py`` builders for the rho(770), Delta(1232) (with the
charge-conjugate partners), and K*(892) states; everywhere else the
pre-registered rule ``m_up = pole_mass + 5 * nominal_width``.

Validation (write-after-validate): the assembled records must pass
``HadronCatalog.validate()`` with zero errors, every physics value must
carry a complete six-field ``ProvenanceTag``, and every unstable record
must keep at least one channel open at its pole mass (the card's
termination rule).  On any violation the driver names every error and
exits with code 2 without writing the catalog.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: standard library plus the vendored ``na49_mixing`` modules
only; no event generation, no external tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the legacy_v1 manifest.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple["Path", "Path"]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        print(
            "import_resonance_catalog: cannot locate the na49_mixing package "
            f"next to {here}; the driver runs from the nested tree "
            "(dev/mixing2006/helpers/benchmarking/) or its monorepo mirror "
            "(helpers/benchmarking/)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from na49_mixing.hadron_catalog import (  # noqa: E402
    STABLE_STATES,
    DecayChannel,
    HadronCatalog,
    HadronRecord,
    ProvenanceTag,
)
from na49_mixing.pdg_database import (  # noqa: E402
    _PDG_TABLE,
    get_charge,
    get_mass,
    get_particle,
)
from na49_mixing.quantum_numbers import QuantumNumbers, quantum_numbers  # noqa: E402

CARD_ID = "TASK-20260913-r2-provenance-catalog-import"
CATALOG_LABEL = "r2_initial_catalog"
RECORDED_ON = "2026-09-13"
ACCESSED_ON = "2026-09-13"
TRANSCRIBER = "na49-res r2 card 2"
PROVENANCE_DISCIPLINE = "plan section 8.2; every physics value carries a ProvenanceTag"

#: Catalog version of the plan section 4.4 catalog file (card schema).
CATALOG_VERSION = 1

#: PDG Monte Carlo id pinned by the card for eta-prime.  The card text
#: (Objective, Public interface, Deterministic tests) pins 377; the PDG
#: 2024 Monte Carlo numbering carries eta'(958) as 331.  The catalog uses
#: the card-pinned key 377 with the name "eta-prime(958)"; the id
#: correction is recorded in the run record and a later catalog-revision
#: card reconciles the key (plan section 4.2).
ETAPRIME_PDG_ID = 377

# The placeholder towers excluded from the established catalog (round 2
# index, item 2): the N_eff / M_eff Regge placeholders and their anti
# partners.  They keep their continuum_resonance-family role (plan
# section 9, Round 3) and are not re-labeled by this card.
PLACEHOLDER_TOWER_IDS = frozenset({
    9911201, 9912201, 9921201, 9922201, 9931201, 9932201,
    -9911201, -9912201, -9921201, -9922201, -9931201, -9932201,
    9900113, 9900213, 9900313,
})

# Vendored-table entries that are not established hadrons: the muons
# (leptons) and the NA49 FSB "cluster" bookkeeping object.
NON_HADRON_IDS = frozenset({13, -13, 2219})

#: Vendored-unstable established hadrons with no open strong channel at
#: their pole mass (weak or electromagnetic decays only).  The card's
#: termination rule makes them cascade-terminal ``stable_for_decay``
#: records; their weak/EM decays stay outside the strong cascade like
#: the K0S/K0L weak-decay endpoints of index item 11.
STRONG_TERMINAL_GROUND_STATES = frozenset({
    311, -311,    # K0, aK0: flavor eigenstates (neutral-kaon endpoints)
    3122, -3122,  # Lambda, aLambda: weak decays only
    3222, 3212, 3112,  # Sigma+, Sigma0 (EM Lambda gamma), Sigma-: no strong channel
})

# Delta(1700) ids are absent from quantum_numbers' baryon registry; the
# catalog record carries (B, Q, S) = (1, Q, 0) directly for them (the
# daughters are all resolvable, so validation is unaffected).
_DELTA1700_IDS = frozenset({121114, 122114, 122214, 122224})

# Source documents named by the provenance tags.
DOC_VENDORED = "na49_mixing/pdg_database.py vendored PDG table (PDG 2024 summary values)"
DOC_PDGLISTINGS = "PDG 2024 particle listings (pdgLive I^G(J^PC) summary values)"
DOC_DECAYS = (
    "na49_mixing/decays.py build_soft_resonance_decays (legacy per-charge "
    "Clebsch-weighted decay table; PDG 2025 summary values, see "
    "helpers/benchmarking/config/soft_resonance_branching_pdg2025.json)"
)
DOC_DECAYS_CONJ = (
    "na49_mixing/decays.py build_soft_resonance_decays, matter charge state, "
    "charge-conjugate channel set (strong-interaction C symmetry); PDG 2025 "
    "summary values"
)
DOC_KAON_DECAYS = (
    "na49_mixing/decays.py build_kaon_decays (legacy decay table; PDG 2025 "
    "summary values)"
)
DOC_KAON_DECAYS_CONJ = (
    "na49_mixing/decays.py build_kaon_decays, matter charge state, "
    "charge-conjugate channel set (strong-interaction C symmetry); PDG 2025 "
    "summary values"
)
DOC_OMEGA_DECAYS = (
    "na49_mixing/decays.py build_omega_decays (legacy decay table; PDG 2025 "
    "summary values)"
)
DOC_RHO_DECAYS = (
    "na49_mixing/decays.py build_rho_decays (legacy decay table; PDG 2025 "
    "summary values)"
)
DOC_RHO_CONJ = (
    "na49_mixing/decays.py build_rho_decays, rho(770) charge-state convention, "
    "isospin-partner channel set; PDG 2025 summary values"
)
DOC_PDG_ETAPRIME = (
    "PDG 2024 review of particle physics, light unflavored mesons summary "
    "table (S. Navas et al., Phys. Rev. D 110, 030001 (2024), "
    "pdg.lbl.gov/2024/tables/rpp2024-tab-mesons-light.pdf)"
)
DOC_DECAYS_BW = (
    "na49_mixing/decays.py build_soft_resonance_decays Breit-Wigner "
    "parameters (BreitWigner(schwel, rmas, brei, pbrei, dumplow, dumphigh); "
    "the legacy .legacy/inc/TBW.h linear damping convention, Fischer 2022 "
    "sect. 16.4.2 upper mass limit m_up)"
)
DOC_DECAYS_BW_CONJ = (
    "na49_mixing/decays.py build_soft_resonance_decays Breit-Wigner "
    "parameters of the matter charge state, charge-conjugate partner "
    "(strong-interaction C symmetry); legacy .legacy/inc/TBW.h damping "
    "convention"
)
DOC_CARD_RULE = (
    "TASK-20260913-r2-provenance-catalog-import pre-registered m_up rule "
    "(plan terminology 'upper mass cut m_up'): m_up = pole_mass + "
    "5 * nominal_width where the repository documents no upper mass cut; "
    "the lower bound is the lowest channel threshold (the card-1 highest-channel-threshold rule was corrected on 2026-09-28, FAB-07)"
)
DOC_CARD_RULE_TERMINAL = (
    "TASK-20260913-r2-provenance-catalog-import terminal-record convention: "
    "m_up = pole_mass for a stable_for_decay record"
)

MASS_TOL = 1e-9          # transcription vs vendored pole-mass cross-check
OPEN_CHANNEL_TOL = 1e-9  # GeV: channel open at the pole mass

#: The pre-registered stable-state termination list (round 2 index, item 11)
#: as written to the catalog header and the acceptance report, in the
#: card-schema order.
STABLE_HEADER_STATES = (2212, 2112, -2212, -2112, 211, -211, 111,
                        321, -321, 310, 130)

#: The card's record-count reconciliation note (see the module docstring).
RECORD_SET_NOTE = (
    "record_count == unstable_record_count + 11: unstable_record_count "
    "counts every record outside the pre-registered 11-state stable_states "
    "termination list (round 2 index, item 11).  It comprises 80 unstable "
    "established hadrons (vendored table minus the N_eff/M_eff placeholder "
    "towers, the muons, and the NA49 FSB cluster object) plus eta-prime "
    "(card-pinned key 377; PDG MC id 331) and 7 cascade-terminal ground "
    "states (K0, aK0, Lambda, aLambda, Sigma+, Sigma0, Sigma-) that have "
    "no open strong channel at their pole mass, so the card's termination "
    "rule makes them stable_for_decay records whose weak/EM decays stay "
    "outside the strong cascade, exactly like the K0S/K0L weak-decay "
    "endpoints of index item 11.  Eta-prime carries the card-pinned PDG "
    "key 377; the PDG Monte Carlo id of eta'(958) is 331 (PDG 2024 "
    "numbering), correction recorded in the run record."
)


def _required_stable_ids() -> frozenset[int]:
    """The stable_for_decay record set: the pre-registered 11-state
    termination set plus the seven cascade-terminal ground states."""
    return STABLE_STATES | STRONG_TERMINAL_GROUND_STATES


def required_unstable_ids() -> frozenset[int]:
    """The required unstable record set, computed from the vendored table.

    Vendored unstable established hadrons, minus the placeholder towers,
    the non-established objects (muons, the NA49 FSB cluster), the
    pre-registered stable states, and the cascade-terminal ground
    states, plus eta-prime (the card-pinned key 377; the PDG MC id of
    eta'(958) is 331, correction recorded in the run record).
    """
    vendored_unstable = {p.pdg_id for p in _PDG_TABLE if not p.stable}
    unstable = (
        vendored_unstable
        - PLACEHOLDER_TOWER_IDS
        - NON_HADRON_IDS
        - set(STABLE_STATES)
        - STRONG_TERMINAL_GROUND_STATES
    )
    return frozenset(unstable | {ETAPRIME_PDG_ID})


def required_stable_ids() -> frozenset[int]:
    """The required stable_for_decay record set (see the module docstring)."""
    return _required_stable_ids()


# ---------------------------------------------------------------------------
# The transcription (the documented manual step of this card).
#
# One entry per record: transcribed isospin/isospin3/spin/parity, the
# transcribed m_up, the transcribed (B, Q, S) bookkeeping, and the
# per-channel (daughters, branching_fraction, orbital_angular_momentum,
# is_three_body, document, table) tuples.  The branching fractions are
# the repository's documented per-channel values (legacy per-charge
# Clebsch-weighted tables; charge conjugates of the matter tables for
# the anti partners; PDG 2024 for eta-prime).  Every entry below was
# transcribed for this card; the driver reads, cross-checks, and
# validates them, and invents no physics value.
# ---------------------------------------------------------------------------

_TR: list[dict] = []


def _tr(pdg_id, B, Q, S, isospin, isospin3, spin, parity, m_up, channels,
        stable=False, name=None, pole_mass=None, nominal_width=None,
        m_up_source=None):
    entry = {
        "pdg_id": int(pdg_id),
        "B": int(B), "Q": int(Q), "S": int(S),
        "isospin": float(isospin), "isospin3": int(isospin3),
        "spin": float(spin), "parity": str(parity),
        "m_up": float(m_up),
        "stable_for_decay": bool(stable),
        "name": name,
        "channels": channels,
        # (document, section_or_table) for the transcribed m_up; the
        # pre-registered rule document when the repository documents no
        # upper mass cut for this state.
        "m_up_source": m_up_source,
    }
    # Records absent from the vendored table carry their transcribed
    # pole mass and nominal width (eta-prime).
    if pole_mass is not None:
        entry["pole_mass"] = float(pole_mass)
    if nominal_width is not None:
        entry["nominal_width"] = float(nominal_width)
    _TR.append(entry)


def _ch(daughters, branching_fraction, L, document, table, is_three_body=False):
    return {
        "daughters": tuple(int(d) for d in daughters),
        "branching_fraction": float(branching_fraction),
        "orbital_angular_momentum": int(L),
        "is_three_body": bool(is_three_body),
        "document": document,
        "table": table,
    }


# Stable-state termination records (the pre-registered 11 plus the seven
# cascade-terminal ground states).  Terminal records carry no channels;
# m_up = pole_mass by the terminal-record convention.
for _pid, _B, _Q, _S, _isospin, _isospin3, _spin, _parity in (
    (2212, 1, 1, 0, 0.5, 1, 0.5, "+"),      # p
    (-2212, -1, -1, 0, 0.5, -1, 0.5, "+"),  # pbar
    (2112, 1, 0, 0, 0.5, -1, 0.5, "+"),     # n
    (-2112, -1, 0, 0, 0.5, 1, 0.5, "+"),    # nbar
    (211, 0, 1, 0, 1.0, 2, 0.0, "-"),       # pi+
    (-211, 0, -1, 0, 1.0, -2, 0.0, "-"),    # pi-
    (111, 0, 0, 0, 1.0, 0, 0.0, "-"),       # pi0
    (321, 0, 1, 1, 0.5, 1, 0.0, "-"),       # K+
    (-321, 0, -1, -1, 0.5, -1, 0.0, "-"),   # K-
    (310, 0, 0, 0, 0.5, 0, 0.0, "-"),       # K0S
    (130, 0, 0, 0, 0.5, 0, 0.0, "-"),       # K0L
    (311, 0, 0, 1, 0.5, -1, 0.0, "-"),      # K0 (cascade-terminal)
    (-311, 0, 0, -1, 0.5, 1, 0.0, "-"),     # aK0 (cascade-terminal)
    (3122, 1, 0, -1, 0.0, 0, 0.5, "+"),     # Lambda (cascade-terminal)
    (-3122, -1, 0, 1, 0.0, 0, 0.5, "+"),    # aLambda (cascade-terminal)
    (3222, 1, 1, -1, 1.0, 2, 0.5, "+"),     # Sigma+ (cascade-terminal)
    (3212, 1, 0, -1, 1.0, 0, 0.5, "+"),     # Sigma0 (cascade-terminal; EM decay)
    (3112, 1, -1, -1, 1.0, -2, 0.5, "+"),   # Sigma- (cascade-terminal)
):
    _mass = get_mass(_pid)
    _tr(_pid, _B, _Q, _S, _isospin, _isospin3, _spin, _parity, _mass, (),
        stable=True)

# ---------------------------------------------------------------- Delta(1232)
# decays.py build_soft_resonance_decays Delta(1232) multiplet block; the
# anti partners beyond the repository's anti(Delta-) entry are the
# charge-conjugate channel sets (C symmetry, identical branching).
# m_up: the repository documents the Delta spectral upper cut in
# delta_bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0); the anti
# partners carry the same cut by C symmetry.
_tbl_delta = "build_soft_resonance_decays: Delta(1232) multiplet block"
_tbl_delta_bw = "delta_bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0): dumphigh = 2.0"
_bw_matter = (DOC_DECAYS_BW, _tbl_delta_bw)
_bw_conj = (DOC_DECAYS_BW_CONJ, _tbl_delta_bw)
_tr(2224, 1, 2, 0, 1.5, 3, 1.5, "+", 2.0,
    [_ch((2212, 211), 1.0, 1, DOC_DECAYS, _tbl_delta)], m_up_source=_bw_matter)
_tr(2214, 1, 1, 0, 1.5, 1, 1.5, "+", 2.0,
    [_ch((2212, 111), 0.6667, 1, DOC_DECAYS, _tbl_delta),
     _ch((2112, 211), 0.3333, 1, DOC_DECAYS, _tbl_delta)], m_up_source=_bw_matter)
_tr(2114, 1, 0, 0, 1.5, -1, 1.5, "+", 2.0,
    [_ch((2212, -211), 0.3333, 1, DOC_DECAYS, _tbl_delta),
     _ch((2112, 111), 0.6667, 1, DOC_DECAYS, _tbl_delta)], m_up_source=_bw_matter)
_tr(1114, 1, -1, 0, 1.5, -3, 1.5, "+", 2.0,
    [_ch((2112, -211), 1.0, 1, DOC_DECAYS, _tbl_delta)], m_up_source=_bw_matter)
_tr(-1114, -1, 1, 0, 1.5, 3, 1.5, "+", 2.0,
    [_ch((-2112, 211), 1.0, 1, DOC_DECAYS,
         "build_soft_resonance_decays: physical anti(Delta-) block")],
    m_up_source=_bw_conj)
_tr(-2114, -1, 0, 0, 1.5, 1, 1.5, "+", 2.0,
    [_ch((-2212, 211), 0.3333, 1, DOC_DECAYS_CONJ, _tbl_delta),
     _ch((-2112, 111), 0.6667, 1, DOC_DECAYS_CONJ, _tbl_delta)],
    m_up_source=_bw_conj)
_tr(-2214, -1, -1, 0, 1.5, -1, 1.5, "+", 2.0,
    [_ch((-2212, 111), 0.6667, 1, DOC_DECAYS_CONJ, _tbl_delta),
     _ch((-2112, -211), 0.3333, 1, DOC_DECAYS_CONJ, _tbl_delta)],
    m_up_source=_bw_conj)
_tr(-2224, -1, -2, 0, 1.5, -3, 1.5, "+", 2.0,
    [_ch((-2212, -211), 1.0, 1, DOC_DECAYS_CONJ, _tbl_delta)],
    m_up_source=_bw_conj)

# ---------------------------------------------------------------- N*(1440)
# decays.py: br_npi 0.65, br_deltapi 0.165, residual 0.185 folded into N pi
# (total N pi 0.835, Delta pi 0.165; exact per-charge Clebsch weights).
# m_up: pre-registered rule pole + 5 * nominal_width = 1.44 + 5 * 0.30
# = 2.94 (nominal_width 0.30 is the vendored PDG 2024 value).
_tbl_n1440 = "build_soft_resonance_decays: N*(1440) block (per-charge Clebsch weights)"
_tr(12112, 1, 0, 0, 0.5, -1, 0.5, "+", 1.44 + 5 * 0.30,
    [_ch((2212, -211), 0.556666667, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2112, 111), 0.278333333, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2214, -211), 0.0275, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2114, 111), 0.055, 1, DOC_DECAYS, _tbl_n1440),
     _ch((1114, 211), 0.0825, 1, DOC_DECAYS, _tbl_n1440)])
_tr(12212, 1, 1, 0, 0.5, 1, 0.5, "+", 1.44 + 5 * 0.30,
    [_ch((2212, 111), 0.278333333, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2112, 211), 0.556666667, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2224, -211), 0.0825, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2214, 111), 0.05445, 1, DOC_DECAYS, _tbl_n1440),
     _ch((2114, 211), 0.02805, 1, DOC_DECAYS, _tbl_n1440)])
_tr(-12112, -1, 0, 0, 0.5, 1, 0.5, "+", 1.44 + 5 * 0.30,
    [_ch((-2212, 211), 0.556666667, 1, DOC_DECAYS_CONJ, _tbl_n1440),
     _ch((-2112, 111), 0.278333333, 1, DOC_DECAYS_CONJ, _tbl_n1440),
     _ch((-2214, 211), 0.0275, 1, DOC_DECAYS_CONJ, _tbl_n1440),
     _ch((-2114, 111), 0.055, 1, DOC_DECAYS_CONJ, _tbl_n1440),
     _ch((-1114, -211), 0.0825, 1, DOC_DECAYS_CONJ, _tbl_n1440)])

# ---------------------------------------------------------------- N*(1520)
# decays.py: br_npi 0.60, br_deltapi 0.28, residual 0.12 -> N pi 0.72.
# isospin: PDG I(J^P) = 1/2(3/2-).
_tbl_n1520 = "build_soft_resonance_decays: N*(1520) block (per-charge Clebsch weights)"
_tr(2116, 1, 0, 0, 0.5, -1, 1.5, "-", 1.52 + 5 * 0.115,
    [_ch((2212, -211), 0.48, 2, DOC_DECAYS, _tbl_n1520),
     _ch((2112, 111), 0.24, 2, DOC_DECAYS, _tbl_n1520),
     _ch((2214, -211), 0.046666667, 0, DOC_DECAYS, _tbl_n1520),
     _ch((2114, 111), 0.093333333, 0, DOC_DECAYS, _tbl_n1520),
     _ch((1114, 211), 0.14, 0, DOC_DECAYS, _tbl_n1520)])
_tr(2216, 1, 1, 0, 0.5, 1, 1.5, "-", 1.52 + 5 * 0.115,
    [_ch((2212, 111), 0.24, 2, DOC_DECAYS, _tbl_n1520),
     _ch((2112, 211), 0.48, 2, DOC_DECAYS, _tbl_n1520),
     _ch((2224, -211), 0.14, 0, DOC_DECAYS, _tbl_n1520),
     _ch((2214, 111), 0.0924, 0, DOC_DECAYS, _tbl_n1520),
     _ch((2114, 211), 0.0476, 0, DOC_DECAYS, _tbl_n1520)])
_tr(-2116, -1, 0, 0, 0.5, 1, 1.5, "-", 1.52 + 5 * 0.115,
    [_ch((-2212, 211), 0.48, 2, DOC_DECAYS_CONJ, _tbl_n1520),
     _ch((-2112, 111), 0.24, 2, DOC_DECAYS_CONJ, _tbl_n1520),
     _ch((-2214, 211), 0.046666667, 0, DOC_DECAYS_CONJ, _tbl_n1520),
     _ch((-2114, 111), 0.093333333, 0, DOC_DECAYS_CONJ, _tbl_n1520),
     _ch((-1114, -211), 0.14, 0, DOC_DECAYS_CONJ, _tbl_n1520)])

# ---------------------------------------------------------------- N*(1680)
# decays.py: br_npi 0.65, br_deltapi 0.17, residual 0.18 -> N pi 0.83.
# isospin: PDG I(J^P) = 1/2(5/2+).
_tbl_n1680 = "build_soft_resonance_decays: N*(1680) block (per-charge Clebsch weights)"
_tr(22112, 1, 0, 0, 0.5, -1, 2.5, "+", 1.685 + 5 * 0.130,
    [_ch((2212, -211), 0.553333333, 3, DOC_DECAYS, _tbl_n1680),
     _ch((2112, 111), 0.276666667, 3, DOC_DECAYS, _tbl_n1680),
     _ch((2214, -211), 0.028333333, 1, DOC_DECAYS, _tbl_n1680),
     _ch((2114, 111), 0.056666667, 1, DOC_DECAYS, _tbl_n1680),
     _ch((1114, 211), 0.085, 1, DOC_DECAYS, _tbl_n1680)])
_tr(22212, 1, 1, 0, 0.5, 1, 2.5, "+", 1.685 + 5 * 0.130,
    [_ch((2212, 111), 0.276666667, 3, DOC_DECAYS, _tbl_n1680),
     _ch((2112, 211), 0.553333333, 3, DOC_DECAYS, _tbl_n1680),
     _ch((2224, -211), 0.085, 1, DOC_DECAYS, _tbl_n1680),
     _ch((2214, 111), 0.0561, 1, DOC_DECAYS, _tbl_n1680),
     _ch((2114, 211), 0.0289, 1, DOC_DECAYS, _tbl_n1680)])
_tr(-22112, -1, 0, 0, 0.5, 1, 2.5, "+", 1.685 + 5 * 0.130,
    [_ch((-2212, 211), 0.553333333, 3, DOC_DECAYS_CONJ, _tbl_n1680),
     _ch((-2112, 111), 0.276666667, 3, DOC_DECAYS_CONJ, _tbl_n1680),
     _ch((-2214, 211), 0.028333333, 1, DOC_DECAYS_CONJ, _tbl_n1680),
     _ch((-2114, 111), 0.056666667, 1, DOC_DECAYS_CONJ, _tbl_n1680),
     _ch((-1114, -211), 0.085, 1, DOC_DECAYS_CONJ, _tbl_n1680)])

# ---------------------------------------------------------------- N(1535)
# decays.py: br_npi 0.42, br_neta 0.45, residual 0.13 -> N pi 0.55.
_tbl_n1535 = "build_soft_resonance_decays: N(1535) block (N pi + N eta)"
_tr(102112, 1, 0, 0, 0.5, -1, 0.5, "-", 1.535 + 5 * 0.150,
    [_ch((2212, -211), 0.366666667, 0, DOC_DECAYS, _tbl_n1535),
     _ch((2112, 111), 0.183333333, 0, DOC_DECAYS, _tbl_n1535),
     _ch((2112, 221), 0.45, 0, DOC_DECAYS, _tbl_n1535)])
_tr(102212, 1, 1, 0, 0.5, 1, 0.5, "-", 1.535 + 5 * 0.150,
    [_ch((2212, 111), 0.183333333, 0, DOC_DECAYS, _tbl_n1535),
     _ch((2112, 211), 0.366666667, 0, DOC_DECAYS, _tbl_n1535),
     _ch((2212, 221), 0.45, 0, DOC_DECAYS, _tbl_n1535)])
_tr(-102112, -1, 0, 0, 0.5, 1, 0.5, "-", 1.535 + 5 * 0.150,
    [_ch((-2212, 211), 0.366666667, 0, DOC_DECAYS_CONJ, _tbl_n1535),
     _ch((-2112, 111), 0.183333333, 0, DOC_DECAYS_CONJ, _tbl_n1535),
     _ch((-2112, 221), 0.45, 0, DOC_DECAYS_CONJ, _tbl_n1535)])

# ---------------------------------------------------------------- N(1650)
# decays.py: N pi 0.65, Delta pi 0.15, N eta 0.08, Lambda K 0.12 (sum 1.0).
# L: the Delta pi channels carry the lowest parity-allowed L = 2
# (Delta(3/2+) pi(0-) at J^P = 1/2-: even L required, L = 1 is the only
# other feasible value); the N pi / N eta / Lambda K channels carry L = 0.
_tbl_n1650 = "build_soft_resonance_decays: N(1650) block (N pi, Delta pi, N eta, Lambda K)"
_tr(122112, 1, 0, 0, 0.5, -1, 0.5, "-", 1.655 + 5 * 0.125,
    [_ch((2212, -211), 0.433333333, 0, DOC_DECAYS, _tbl_n1650),
     _ch((2112, 111), 0.216666667, 0, DOC_DECAYS, _tbl_n1650),
     _ch((2214, -211), 0.025, 2, DOC_DECAYS, _tbl_n1650),
     _ch((2114, 111), 0.05, 2, DOC_DECAYS, _tbl_n1650),
     _ch((1114, 211), 0.075, 2, DOC_DECAYS, _tbl_n1650),
     _ch((2112, 221), 0.08, 0, DOC_DECAYS, _tbl_n1650),
     _ch((3122, 311), 0.12, 0, DOC_DECAYS, _tbl_n1650)])
_tr(122212, 1, 1, 0, 0.5, 1, 0.5, "-", 1.655 + 5 * 0.125,
    [_ch((2212, 111), 0.216666667, 0, DOC_DECAYS, _tbl_n1650),
     _ch((2112, 211), 0.433333333, 0, DOC_DECAYS, _tbl_n1650),
     _ch((2224, -211), 0.075, 2, DOC_DECAYS, _tbl_n1650),
     _ch((2214, 111), 0.0495, 2, DOC_DECAYS, _tbl_n1650),
     _ch((2114, 211), 0.0255, 2, DOC_DECAYS, _tbl_n1650),
     _ch((2212, 221), 0.08, 0, DOC_DECAYS, _tbl_n1650),
     _ch((3122, 321), 0.12, 0, DOC_DECAYS, _tbl_n1650)])
_tr(-122112, -1, 0, 0, 0.5, 1, 0.5, "-", 1.655 + 5 * 0.125,
    [_ch((-2212, 211), 0.433333333, 0, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-2112, 111), 0.216666667, 0, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-2214, 211), 0.025, 2, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-2114, 111), 0.05, 2, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-1114, -211), 0.075, 2, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-2112, 221), 0.08, 0, DOC_DECAYS_CONJ, _tbl_n1650),
     _ch((-3122, -311), 0.12, 0, DOC_DECAYS_CONJ, _tbl_n1650)])

# ---------------------------------------------------------------- N(1720)
# decays.py: N pi 0.15, Delta pi 0.35, N rho 0.35, N eta 0.05, Lambda K 0.10.
_tbl_n1720 = ("build_soft_resonance_decays: N(1720) block "
              "(N pi, Delta pi, N rho, N eta, Lambda K)")
_tr(212114, 1, 0, 0, 0.5, -1, 1.5, "+", 1.720 + 5 * 0.200,
    [_ch((2212, -211), 0.1, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2112, 111), 0.05, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2214, -211), 0.058333333, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2114, 111), 0.116666667, 1, DOC_DECAYS, _tbl_n1720),
     _ch((1114, 211), 0.175, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2212, -213), 0.233333333, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2112, 113), 0.116666667, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2112, 221), 0.05, 1, DOC_DECAYS, _tbl_n1720),
     _ch((3122, 311), 0.10, 1, DOC_DECAYS, _tbl_n1720)])
_tr(212214, 1, 1, 0, 0.5, 1, 1.5, "+", 1.720 + 5 * 0.200,
    [_ch((2212, 111), 0.05, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2112, 211), 0.1, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2224, -211), 0.175, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2214, 111), 0.1155, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2114, 211), 0.0595, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2212, 113), 0.116666667, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2112, 213), 0.233333333, 1, DOC_DECAYS, _tbl_n1720),
     _ch((2212, 221), 0.05, 1, DOC_DECAYS, _tbl_n1720),
     _ch((3122, 321), 0.10, 1, DOC_DECAYS, _tbl_n1720)])
_tr(-212114, -1, 0, 0, 0.5, 1, 1.5, "+", 1.720 + 5 * 0.200,
    [_ch((-2212, 211), 0.1, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2112, 111), 0.05, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2214, 211), 0.058333333, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2114, 111), 0.116666667, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-1114, -211), 0.175, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2212, 213), 0.233333333, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2112, 113), 0.116666667, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2112, 221), 0.05, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-3122, -311), 0.10, 1, DOC_DECAYS_CONJ, _tbl_n1720)])
_tr(-212214, -1, -1, 0, 0.5, -1, 1.5, "+", 1.720 + 5 * 0.200,
    [_ch((-2212, 111), 0.05, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2112, -211), 0.1, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2224, 211), 0.175, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2214, 111), 0.1155, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2114, -211), 0.0595, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2212, 113), 0.116666667, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2112, -213), 0.233333333, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-2212, 221), 0.05, 1, DOC_DECAYS_CONJ, _tbl_n1720),
     _ch((-3122, -321), 0.10, 1, DOC_DECAYS_CONJ, _tbl_n1720)])

# ---------------------------------------------------------------- Delta(1600)
# decays.py _add_delta_excited Delta(1600) block: N pi 0.18, Delta pi 0.55,
# N rho 0.27; exact I = 3/2 Clebsch-Gordan charge weights.
_tbl_d1600 = "build_soft_resonance_decays: Delta(1600) block (exact I=3/2 CG weights)"
_tr(202224, 1, 2, 0, 1.5, 3, 1.5, "+", 1.600 + 5 * 0.350,
    [_ch((2212, 211), 0.18, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2224, 111), 0.33, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2214, 211), 0.22, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2212, 213), 0.27, 1, DOC_DECAYS, _tbl_d1600)])
_tr(202214, 1, 1, 0, 1.5, 1, 1.5, "+", 1.600 + 5 * 0.350,
    [_ch((2212, 111), 0.12, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2112, 211), 0.06, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2224, -211), 0.22, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2214, 111), 0.036666667, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2114, 211), 0.293333333, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2212, 113), 0.18, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2112, 213), 0.09, 1, DOC_DECAYS, _tbl_d1600)])
_tr(202114, 1, 0, 0, 1.5, -1, 1.5, "+", 1.600 + 5 * 0.350,
    [_ch((2212, -211), 0.06, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2112, 111), 0.12, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2214, -211), 0.293333333, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2114, 111), 0.036666667, 1, DOC_DECAYS, _tbl_d1600),
     _ch((1114, 211), 0.22, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2212, -213), 0.09, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2112, 113), 0.18, 1, DOC_DECAYS, _tbl_d1600)])
_tr(201114, 1, -1, 0, 1.5, -3, 1.5, "+", 1.600 + 5 * 0.350,
    [_ch((2112, -211), 0.18, 1, DOC_DECAYS, _tbl_d1600),
     _ch((1114, 111), 0.33, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2114, -211), 0.22, 1, DOC_DECAYS, _tbl_d1600),
     _ch((2112, -213), 0.27, 1, DOC_DECAYS, _tbl_d1600)])

# ---------------------------------------------------------------- Delta(1700)
# decays.py _add_delta_excited Delta(1700) block: N pi 0.15, Delta pi 0.50,
# N rho 0.35; exact I = 3/2 CG charge weights.  The (B, Q, S) = (1, Q, 0)
# bookkeeping is pinned per record (see _DELTA1700_IDS note).
_tbl_d1700 = "build_soft_resonance_decays: Delta(1700) block (exact I=3/2 CG weights)"
_tr(122224, 1, 2, 0, 1.5, 3, 1.5, "-", 1.700 + 5 * 0.300,
    [_ch((2212, 211), 0.15, 2, DOC_DECAYS, _tbl_d1700),
     _ch((2224, 111), 0.3, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2214, 211), 0.2, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2212, 213), 0.35, 0, DOC_DECAYS, _tbl_d1700)])
_tr(122214, 1, 1, 0, 1.5, 1, 1.5, "-", 1.700 + 5 * 0.300,
    [_ch((2212, 111), 0.1, 2, DOC_DECAYS, _tbl_d1700),
     _ch((2112, 211), 0.05, 2, DOC_DECAYS, _tbl_d1700),
     _ch((2224, -211), 0.2, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2214, 111), 0.033333333, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2114, 211), 0.266666667, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2212, 113), 0.233333333, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2112, 213), 0.116666667, 0, DOC_DECAYS, _tbl_d1700)])
_tr(122114, 1, 0, 0, 1.5, -1, 1.5, "-", 1.700 + 5 * 0.300,
    [_ch((2212, -211), 0.1, 2, DOC_DECAYS, _tbl_d1700),
     _ch((2112, 111), 0.05, 2, DOC_DECAYS, _tbl_d1700),
     _ch((2214, -211), 0.266666667, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2114, 111), 0.033333333, 0, DOC_DECAYS, _tbl_d1700),
     _ch((1114, 211), 0.2, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2212, -213), 0.116666667, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2112, 113), 0.233333333, 0, DOC_DECAYS, _tbl_d1700)])
_tr(121114, 1, -1, 0, 1.5, -3, 1.5, "-", 1.700 + 5 * 0.300,
    [_ch((2112, -211), 0.15, 2, DOC_DECAYS, _tbl_d1700),
     _ch((1114, 111), 0.3, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2114, -211), 0.2, 0, DOC_DECAYS, _tbl_d1700),
     _ch((2112, -213), 0.35, 0, DOC_DECAYS, _tbl_d1700)])

# ---------------------------------------------------------------- light mesons
# m_up: the repository documents the rho spectral upper cut in
# rho_bw = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5).
_tbl_rho = "build_soft_resonance_decays: rho(770) block"
_tbl_rho_bw = "rho_bw = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5): dumphigh = 1.5"
_rho_bw = (DOC_DECAYS_BW, _tbl_rho_bw)
_tr(113, 0, 0, 0, 1.0, 0, 1.0, "-", 1.5,
    [_ch((211, -211), 1.0, 1, DOC_DECAYS, _tbl_rho)], m_up_source=_rho_bw)
_tr(213, 0, 1, 0, 1.0, 2, 1.0, "-", 1.5,
    [_ch((211, 111), 1.0, 1, DOC_DECAYS, _tbl_rho)], m_up_source=_rho_bw)
_tr(-213, 0, -1, 0, 1.0, -2, 1.0, "-", 1.5,
    [_ch((-211, 111), 1.0, 1, DOC_DECAYS, _tbl_rho)], m_up_source=_rho_bw)

_tbl_rho1450 = ("build_rho_decays: rho(1450) pi+ pi- convention; the charged "
                "partners carry the isospin-partner channel set")
_tr(100113, 0, 0, 0, 1.0, 0, 1.0, "-", 1.465 + 5 * 0.400,
    [_ch((211, -211), 1.0, 1, DOC_RHO_DECAYS, _tbl_rho1450)])
_tr(100213, 0, 1, 0, 1.0, 2, 1.0, "-", 1.465 + 5 * 0.400,
    [_ch((211, 111), 1.0, 1, DOC_RHO_CONJ, _tbl_rho1450)])
_tr(-100213, 0, -1, 0, 1.0, -2, 1.0, "-", 1.465 + 5 * 0.400,
    [_ch((-211, 111), 1.0, 1, DOC_RHO_CONJ, _tbl_rho1450)])
_tr(30113, 0, 0, 0, 1.0, 0, 1.0, "-", 1.720 + 5 * 0.250,
    [_ch((211, -211), 1.0, 1, DOC_RHO_DECAYS, _tbl_rho1450)])
_tr(30213, 0, 1, 0, 1.0, 2, 1.0, "-", 1.720 + 5 * 0.250,
    [_ch((211, 111), 1.0, 1, DOC_RHO_CONJ, _tbl_rho1450)])
_tr(-30213, 0, -1, 0, 1.0, -2, 1.0, "-", 1.720 + 5 * 0.250,
    [_ch((-211, 111), 1.0, 1, DOC_RHO_CONJ, _tbl_rho1450)])

_tbl_omega = ("build_omega_decays: omega(782) block (represented hadronic "
              "modes, renormalized to unity; the unrepresented radiative "
              "pi0 gamma mode is folded out per the library convention)")
_w3 = 0.892 / 0.9073
_tr(223, 0, 0, 0, 0.0, 0, 1.0, "-", 0.78265 + 5 * 0.00849,
    [_ch((211, -211, 111), _w3, 0, DOC_OMEGA_DECAYS, _tbl_omega, True),
     _ch((211, -211), 1.0 - _w3, 1, DOC_OMEGA_DECAYS, _tbl_omega)])

_tbl_omega_1420 = "build_omega_decays: omega(1420) block (rho0 pi0)"
_tr(100223, 0, 0, 0, 0.0, 0, 1.0, "-", 1.410 + 5 * 0.290,
    [_ch((113, 111), 1.0, 1, DOC_OMEGA_DECAYS, _tbl_omega_1420)])
_tbl_omega_1650 = "build_omega_decays: omega(1650) block (rho0 pi0)"
_tr(30223, 0, 0, 0, 0.0, 0, 1.0, "-", 1.670 + 5 * 0.315,
    [_ch((113, 111), 1.0, 1, DOC_OMEGA_DECAYS, _tbl_omega_1650)])

_tbl_eta = ("build_soft_resonance_decays: eta(548) block; the catalog "
            "represents the hadronic channels; the radiative gamma-gamma "
            "(39.36%) and pi+ pi- gamma (4.28%) modes stay unrepresented")
_tr(221, 0, 0, 0, 0.0, 0, 0.0, "-", 0.547862 + 5 * 1.31e-6,
    [_ch((211, -211, 111), 0.2302, 0, DOC_DECAYS, _tbl_eta, True),
     _ch((111, 111, 111), 0.3256, 0, DOC_DECAYS, _tbl_eta, True)])

_tbl_etaprime = "PDG 2024 summary table, eta'(958) decay modes"
_tr(ETAPRIME_PDG_ID, 0, 0, 0, 0.0, 0, 0.0, "-", 0.95778 + 5 * 0.000188,
    [_ch((211, -211, 221), 0.425, 0, DOC_PDG_ETAPRIME, _tbl_etaprime, True),
     _ch((111, 111, 221), 0.224, 0, DOC_PDG_ETAPRIME, _tbl_etaprime, True)],
    name="eta-prime(958)", pole_mass=0.95778, nominal_width=0.000188)

# ---------------------------------------------------------------- b1 / a1 / a2
_tbl_b1 = "build_soft_resonance_decays: b1(1235) block (omega pi)"
_tr(10113, 0, 0, 0, 1.0, 0, 1.0, "+", 1.2295 + 5 * 0.142,
    [_ch((223, 111), 1.0, 0, DOC_DECAYS, _tbl_b1)])
_tr(10213, 0, 1, 0, 1.0, 2, 1.0, "+", 1.2295 + 5 * 0.142,
    [_ch((223, 211), 1.0, 0, DOC_DECAYS, _tbl_b1)])
_tr(-10213, 0, -1, 0, 1.0, -2, 1.0, "+", 1.2295 + 5 * 0.142,
    [_ch((223, -211), 1.0, 0, DOC_DECAYS_CONJ, _tbl_b1)])

_tbl_a1 = "build_soft_resonance_decays: a1(1260) block (rho pi)"
_tr(20113, 0, 0, 0, 1.0, 0, 1.0, "+", 1.230 + 5 * 0.420,
    [_ch((213, -211), 0.5, 0, DOC_DECAYS, _tbl_a1),
     _ch((-213, 211), 0.5, 0, DOC_DECAYS, _tbl_a1)])
_tr(20213, 0, 1, 0, 1.0, 2, 1.0, "+", 1.230 + 5 * 0.420,
    [_ch((213, 111), 0.5, 0, DOC_DECAYS, _tbl_a1),
     _ch((113, 211), 0.5, 0, DOC_DECAYS, _tbl_a1)])
_tr(-20213, 0, -1, 0, 1.0, -2, 1.0, "+", 1.230 + 5 * 0.420,
    [_ch((-213, 111), 0.5, 0, DOC_DECAYS_CONJ, _tbl_a1),
     _ch((113, -211), 0.5, 0, DOC_DECAYS_CONJ, _tbl_a1)])

_tbl_a2 = "build_soft_resonance_decays: a2(1320) block (rho pi + eta pi, normalized)"
_a2_rhopi = 0.70 / 0.845
_a2_etapi = 0.145 / 0.845
_tr(115, 0, 0, 0, 1.0, 0, 2.0, "+", 1.3183 + 5 * 0.107,
    [_ch((213, -211), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((-213, 211), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((221, 111), _a2_etapi, 2, DOC_DECAYS, _tbl_a2)])
_tr(215, 0, 1, 0, 1.0, 2, 2.0, "+", 1.3183 + 5 * 0.107,
    [_ch((213, 111), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((113, 211), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((221, 211), _a2_etapi, 2, DOC_DECAYS, _tbl_a2)])
_tr(-215, 0, -1, 0, 1.0, -2, 2.0, "+", 1.3183 + 5 * 0.107,
    [_ch((-213, 111), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((113, -211), _a2_rhopi / 2, 2, DOC_DECAYS, _tbl_a2),
     _ch((221, -211), _a2_etapi, 2, DOC_DECAYS, _tbl_a2)])

# ---------------------------------------------------------------- f2 / rho3 / f4
_tbl_f2 = ("build_soft_resonance_decays: f2(1270) block (PDG fit fractions "
           "normalized over the represented channels)")
_f2_pipi = 2.0 / 3.0 * 0.843 / 0.889
_f2_kk = 0.5 * 0.046 / 0.889
_tr(225, 0, 0, 0, 0.0, 0, 2.0, "+", 1.2755 + 5 * 0.187,
    [_ch((211, -211), _f2_pipi, 2, DOC_DECAYS, _tbl_f2),
     _ch((111, 111), 0.843 / 0.889 - _f2_pipi, 2, DOC_DECAYS, _tbl_f2),
     _ch((321, -321), _f2_kk, 2, DOC_DECAYS, _tbl_f2),
     _ch((310, 130), 0.046 / 0.889 - _f2_kk, 2, DOC_DECAYS, _tbl_f2)])

_tbl_rho3 = ("build_soft_resonance_decays: rho3(1690) block (PDG channels "
             "normalized over the represented hadronic modes)")
_r3_pipi = 0.236 / 0.4498
_r3_omegapi = 0.16 / 0.4498
_r3_kkpi = 0.038 / 0.4498
_r3_kk = 0.0158 / 0.4498
_tr(117, 0, 0, 0, 1.0, 0, 3.0, "-", 1.6888 + 5 * 0.161,
    [_ch((211, -211), _r3_pipi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((223, 111), _r3_omegapi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((321, -321, 111), _r3_kkpi, 0, DOC_DECAYS, _tbl_rho3, True),
     _ch((321, -321), _r3_kk, 3, DOC_DECAYS, _tbl_rho3)])
_tr(-217, 0, -1, 0, 1.0, -2, 3.0, "-", 1.6888 + 5 * 0.161,
    [_ch((-211, 111), _r3_pipi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((223, -211), _r3_omegapi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((311, -321, 111), _r3_kkpi, 0, DOC_DECAYS, _tbl_rho3, True),
     _ch((311, -321), _r3_kk, 3, DOC_DECAYS, _tbl_rho3)])
_tr(217, 0, 1, 0, 1.0, 2, 3.0, "-", 1.6888 + 5 * 0.161,
    [_ch((211, 111), _r3_pipi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((223, 211), _r3_omegapi, 3, DOC_DECAYS, _tbl_rho3),
     _ch((-311, 321, 111), _r3_kkpi, 0, DOC_DECAYS, _tbl_rho3, True),
     _ch((-311, 321), _r3_kk, 3, DOC_DECAYS, _tbl_rho3)])

_tbl_f4 = ("build_soft_resonance_decays: f4(2050) block (PDG channels "
           "normalized over the represented hadronic modes)")
_f4_pipi = 2.0 / 3.0 * 0.17 / 0.1789
_f4_kk = 0.5 * 0.0068 / 0.1789
_f4_etaeta = 0.0021 / 0.1789
_tr(229, 0, 0, 0, 0.0, 0, 4.0, "+", 2.018 + 5 * 0.237,
    [_ch((211, -211), _f4_pipi, 4, DOC_DECAYS, _tbl_f4),
     _ch((111, 111), 0.17 / 0.1789 - _f4_pipi, 4, DOC_DECAYS, _tbl_f4),
     _ch((321, -321), _f4_kk, 4, DOC_DECAYS, _tbl_f4),
     _ch((310, 130), 0.0068 / 0.1789 - _f4_kk, 4, DOC_DECAYS, _tbl_f4),
     _ch((221, 221), _f4_etaeta, 4, DOC_DECAYS, _tbl_f4)])

# ---------------------------------------------------------------- K*(892) / K1 / K2* / phi
# m_up: the repository documents the K* spectral upper cut in
# kstar_bw = BreitWigner(0.6337, 0.89166, 0.0508, 1.0, 1.15, 1.4).
_tbl_kstar = "build_kaon_decays: K*(892) block (per-charge K-pi Clebsch weights)"
_tbl_kstar_bw = ("kstar_bw = BreitWigner(0.6337, 0.89166, 0.0508, 1.0, 1.15, 1.4): "
                 "dumphigh = 1.4")
_kstar_bw = (DOC_DECAYS_BW, _tbl_kstar_bw)
_tr(323, 0, 1, 1, 0.5, 1, 1.0, "-", 1.4,
    [_ch((311, 211), 0.667, 1, DOC_KAON_DECAYS, _tbl_kstar),
     _ch((321, 111), 0.333, 1, DOC_KAON_DECAYS, _tbl_kstar)],
    m_up_source=_kstar_bw)
_tr(-323, 0, -1, -1, 0.5, -1, 1.0, "-", 1.4,
    [_ch((-311, -211), 0.667, 1, DOC_KAON_DECAYS, _tbl_kstar),
     _ch((-321, 111), 0.333, 1, DOC_KAON_DECAYS, _tbl_kstar)],
    m_up_source=_kstar_bw)
_tr(313, 0, 0, 1, 0.5, -1, 1.0, "-", 1.4,
    [_ch((321, -211), 0.667, 1, DOC_KAON_DECAYS, _tbl_kstar),
     _ch((311, 111), 0.333, 1, DOC_KAON_DECAYS, _tbl_kstar)],
    m_up_source=_kstar_bw)
_tr(-313, 0, 0, -1, 0.5, 1, 1.0, "-", 1.4,
    [_ch((-321, 211), 0.667, 1, DOC_KAON_DECAYS, _tbl_kstar),
     _ch((-311, 111), 0.333, 1, DOC_KAON_DECAYS, _tbl_kstar)],
    m_up_source=_kstar_bw)

# K1(1270): PDG 2024 I(J^P) = 1/2(1+).  The daughters (rho K, K* pi,
# omega K) are two negative-parity states, so the lowest parity-allowed
# partial wave is L = 0, matching a1(1260) and b1(1235).
_tbl_k1 = "build_kaon_decays: K1(1270) block (rho K, K* pi, omega K, normalized)"
_k1_rhok = 0.42 / 0.69
_k1_kstpi = 0.16 / 0.69
_k1_omegak = 0.11 / 0.69
_tr(10323, 0, 1, 1, 0.5, 1, 1.0, "+", 1.272 + 5 * 0.090,
    [_ch((213, 311), 2.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((113, 321), 1.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((313, 211), 2.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((323, 111), 1.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((223, 321), _k1_omegak, 0, DOC_KAON_DECAYS, _tbl_k1)])
_tr(-10323, 0, -1, -1, 0.5, -1, 1.0, "+", 1.272 + 5 * 0.090,
    [_ch((-213, -311), 2.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((113, -321), 1.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((-313, -211), 2.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((-323, 111), 1.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((223, -321), _k1_omegak, 0, DOC_KAON_DECAYS, _tbl_k1)])
_tr(10313, 0, 0, 1, 0.5, -1, 1.0, "+", 1.272 + 5 * 0.090,
    [_ch((-213, 321), 2.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((113, 311), 1.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((323, -211), 2.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((313, 111), 1.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((223, 311), _k1_omegak, 0, DOC_KAON_DECAYS, _tbl_k1)])
_tr(-10313, 0, 0, -1, 0.5, 1, 1.0, "+", 1.272 + 5 * 0.090,
    [_ch((213, -321), 2.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((113, -311), 1.0 / 3.0 * _k1_rhok, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((-323, 211), 2.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((-313, 111), 1.0 / 3.0 * _k1_kstpi, 0, DOC_KAON_DECAYS, _tbl_k1),
     _ch((223, -311), _k1_omegak, 0, DOC_KAON_DECAYS, _tbl_k1)])

_tbl_k2 = "build_kaon_decays: K2*(1430) block (K pi, K* pi, rho K, omega K; normalized)"
_k2_kpi = 0.499 / 0.862
_k2_kstpi = 0.248 / 0.862
_k2_rhok = 0.087 / 0.862
_k2_omegak = 0.028 / 0.862
_tr(325, 0, 1, 1, 0.5, 1, 2.0, "+", 1.4256 + 5 * 0.0985,
    [_ch((311, 211), 2.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((321, 111), 1.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((313, 211), 2.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((323, 111), 1.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((213, 311), 2.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((113, 321), 1.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((223, 321), _k2_omegak, 2, DOC_KAON_DECAYS, _tbl_k2)])
_tr(-325, 0, -1, -1, 0.5, -1, 2.0, "+", 1.4256 + 5 * 0.0985,
    [_ch((-311, -211), 2.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-321, 111), 1.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-313, -211), 2.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-323, 111), 1.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-213, -311), 2.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((113, -321), 1.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((223, -321), _k2_omegak, 2, DOC_KAON_DECAYS, _tbl_k2)])
_tr(315, 0, 0, 1, 0.5, -1, 2.0, "+", 1.4256 + 5 * 0.0985,
    [_ch((321, -211), 2.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((311, 111), 1.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((323, -211), 2.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((313, 111), 1.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-213, 321), 2.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((113, 311), 1.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((223, 311), _k2_omegak, 2, DOC_KAON_DECAYS, _tbl_k2)])
_tr(-315, 0, 0, -1, 0.5, 1, 2.0, "+", 1.4256 + 5 * 0.0985,
    [_ch((-321, 211), 2.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-311, 111), 1.0 / 3.0 * _k2_kpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-323, 211), 2.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((-313, 111), 1.0 / 3.0 * _k2_kstpi, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((213, -321), 2.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((113, -311), 1.0 / 3.0 * _k2_rhok, 2, DOC_KAON_DECAYS, _tbl_k2),
     _ch((223, -311), _k2_omegak, 2, DOC_KAON_DECAYS, _tbl_k2)])

_tbl_phi = ("build_kaon_decays: phi(1020) block (the K0S K0L pair represents "
            "the neutral-kaon pair with one member observed as a K0S, the "
            "legacy neutral-kaon feed-down convention)")
_tr(333, 0, 0, 0, 0.0, 0, 1.0, "-", 1.019461 + 5 * 0.004249,
    [_ch((321, -321), 0.492, 1, DOC_KAON_DECAYS, _tbl_phi),
     _ch((310, 130), 0.508, 1, DOC_KAON_DECAYS, _tbl_phi)])

# ---------------------------------------------------------------- hyperon resonances
_tbl_l1520 = ("build_soft_resonance_decays: Lambda(1520) block "
              "(N Kbar + Sigma pi + Lambda pi pi, normalized)")
_l_nkbar = 0.5 * 0.45 / 0.97
_l_sp = 0.42 / 0.97 / 3.0
_tr(102134, 1, 0, -1, 0.0, 0, 1.5, "-", 1.5195 + 5 * 0.0156,
    [_ch((2212, -321), _l_nkbar, 2, DOC_DECAYS, _tbl_l1520),
     _ch((2112, -311), _l_nkbar, 2, DOC_DECAYS, _tbl_l1520),
     _ch((3222, -211), _l_sp, 2, DOC_DECAYS, _tbl_l1520),
     _ch((3212, 111), _l_sp, 2, DOC_DECAYS, _tbl_l1520),
     _ch((3112, 211), _l_sp, 2, DOC_DECAYS, _tbl_l1520),
     _ch((3122, 211, -211), 0.10 / 0.97, 1, DOC_DECAYS, _tbl_l1520, True)])

_tbl_s1385 = ("build_soft_resonance_decays: Sigma(1385) block "
              "(Lambda pi dominant with subleading Sigma pi, normalized)")
_s_lpi = 0.87 / 0.987
_s_sp = 0.5 * 0.117 / 0.987
_tr(3224, 1, 1, -1, 1.0, 2, 1.5, "+", 1.3828 + 5 * 0.036,
    [_ch((3122, 211), _s_lpi, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3222, 111), _s_sp, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3212, 211), _s_sp, 1, DOC_DECAYS, _tbl_s1385)])
_tr(3214, 1, 0, -1, 1.0, 0, 1.5, "+", 1.3837 + 5 * 0.036,
    [_ch((3122, 111), _s_lpi, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3222, -211), _s_sp, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3112, 211), _s_sp, 1, DOC_DECAYS, _tbl_s1385)])
_tr(3114, 1, -1, -1, 1.0, -2, 1.5, "+", 1.3872 + 5 * 0.039,
    [_ch((3122, -211), _s_lpi, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3112, 111), _s_sp, 1, DOC_DECAYS, _tbl_s1385),
     _ch((3212, -211), _s_sp, 1, DOC_DECAYS, _tbl_s1385)])

assert len(_TR) == 98, f"transcription carries {len(_TR)} records, expected 98"


# ---------------------------------------------------------------------------
# Tag composition: the transcription's values become one ProvenanceTag
# (six non-null fields) per physics value in the staged catalog_values
# payload.  The composition is deterministic metadata work; the values
# themselves are the transcribed literals above.
# ---------------------------------------------------------------------------

def _tag(document: str, table: str, field: str, value) -> dict:
    return {
        "document": document,
        "section_or_table": table,
        "field": field,
        "accessed_on": ACCESSED_ON,
        "transcribed_value": value,
        "transcriber": TRANSCRIBER,
    }


def compose_catalog_values() -> dict:
    """Compose the machine-readable catalog_values payload from the
    transcription and the vendored table (the driver invents no physics
    value: every transcribed_value below is a vendored-table entry or a
    transcription literal)."""
    records = []
    for entry in _TR:
        pid = int(entry["pdg_id"])
        vendored = get_particle(pid)
        name = entry.get("name") or (vendored.name if vendored else "")
        if vendored is not None:
            pole = float(vendored.mass)
            width = float(vendored.width)
            mass_doc, mass_tbl = DOC_VENDORED, "_PDG_TABLE ParticleData entry"
        else:
            pole = entry.get("pole_mass")
            width_val = entry.get("nominal_width")
            if pole is None or width_val is None:
                raise SystemExit(
                    f"import_resonance_catalog: record pdg {pid} is not in the "
                    "vendored table and its transcription carries no "
                    "pole_mass/nominal_width"
                )
            pole = float(pole)
            width = float(width_val)
            mass_doc = mass_tbl = DOC_PDG_ETAPRIME
        record_doc = DOC_PDG_ETAPRIME if vendored is None else DOC_VENDORED
        if entry["stable_for_decay"]:
            m_up_doc = DOC_CARD_RULE_TERMINAL
            m_up_tbl = "terminal record: no channels, m_up = pole_mass"
        else:
            m_up_doc, m_up_tbl = entry.get("m_up_source") or (
                DOC_CARD_RULE,
                "m_up = pole_mass + 5 * nominal_width (pre-registered rule)")
        values = {
            "pole_mass": _tag(mass_doc, mass_tbl, "pole_mass", pole),
            "nominal_width": _tag(mass_doc, mass_tbl, "nominal_width", width),
            "mass_range": _tag(
                m_up_doc,
                m_up_tbl,
                "mass_range", entry["m_up"],
            ),
            "spin": _tag(DOC_PDGLISTINGS, "I^G(J^PC) summary values", "spin",
                         entry["spin"]),
            "parity": _tag(DOC_PDGLISTINGS, "I^G(J^PC) summary values", "parity",
                           entry["parity"]),
            "isospin": _tag(DOC_PDGLISTINGS, "I^G(J^PC) summary values", "isospin",
                            entry["isospin"]),
            "isospin3": _tag(DOC_PDGLISTINGS, "I^G(J^PC) summary values", "isospin3",
                             entry["isospin3"]),
        }
        channels = [
            {
                "daughters": list(c["daughters"]),
                "branching_fraction": c["branching_fraction"],
                "orbital_angular_momentum": c["orbital_angular_momentum"],
                "is_three_body": c["is_three_body"],
                "provenance": _tag(c["document"], c["table"], "branching_fraction",
                                   c["branching_fraction"]),
            }
            for c in entry["channels"]
        ]
        records.append({
            "pdg_id": pid,
            "name": name,
            "B": entry["B"], "Q": entry["Q"], "S": entry["S"],
            "stable_for_decay": entry["stable_for_decay"],
            "values": values,
            "record_provenance": _tag(mass_doc, f"{name} record", "record", None),
            "channels": channels,
        })
    return {
        "card": CARD_ID,
        "recorded_on": RECORDED_ON,
        "transcriber": TRANSCRIBER,
        "provenance_discipline": PROVENANCE_DISCIPLINE,
        "records": records,
    }


#: The embedded transcription (the documented manual step of this card),
#: materialized as the machine-readable catalog_values structure.
EMBEDDED_CATALOG_VALUES = compose_catalog_values()


# ---------------------------------------------------------------------------
# Validation and record building
# ---------------------------------------------------------------------------

_TAG_FIELDS = ("document", "section_or_table", "field", "accessed_on",
               "transcribed_value", "transcriber")


def _tag_problems(label: str, tag: Any, allow_null_value: bool) -> list[str]:
    """Completeness of one ProvenanceTag dict: all six fields present and
    non-null; the record-level tag may carry a null transcribed_value."""
    problems: list[str] = []
    if not isinstance(tag, dict):
        return [f"{label}: provenance is not a mapping"]
    for field in _TAG_FIELDS:
        if field not in tag:
            problems.append(f"{label}: provenance field {field!r} is missing")
        elif tag[field] is None and not (field == "transcribed_value"
                                         and allow_null_value):
            problems.append(f"{label}: provenance field {field!r} is null")
    if isinstance(tag, dict) and tag.get("accessed_on") is not None:
        text = str(tag["accessed_on"])
        try:
            datetime.date.fromisoformat(text)
        except ValueError:
            problems.append(f"{label}: accessed_on {text!r} is not an ISO date")
    return problems


def build_records(
    values: dict, full_mode: bool
) -> tuple[list[HadronRecord], list[str]]:
    """Cross-check the transcription against the vendored table and build
    the :class:`HadronRecord` set.

    Returns ``(records, problems)``; an empty problem list means every
    cross-check passed.  In full mode the record set must equal the
    required set; in fixture mode every record must be a legitimate
    catalog state.  No physics value is invented here: every number
    comes from the transcription or the vendored table.
    """
    problems: list[str] = []
    required_unstable = required_unstable_ids()
    required_stable = required_stable_ids()
    daughter_mass: dict[int, float] = {}

    # Daughter masses first: record daughters consume their pole mass,
    # stable daughters the vendored table mass.
    for entry in values["records"]:
        pid = int(entry["pdg_id"])
        vendored = get_particle(pid)
        if vendored is not None:
            daughter_mass[pid] = float(vendored.mass)
        else:
            tag = entry.get("values", {}).get("pole_mass", {})
            value = tag.get("transcribed_value") if isinstance(tag, dict) else None
            if value is None:
                daughter_mass[pid] = float(get_mass(pid, default=0.0))
            else:
                daughter_mass[pid] = float(value)

    # -- cross-checks --------------------------------------------------------
    for entry in values["records"]:
        pid = int(entry["pdg_id"])
        label = f"record {entry.get('name') or pid} (pdg {pid})"
        vendored = get_particle(pid)

        if full_mode:
            if entry["stable_for_decay"]:
                if pid not in required_stable:
                    problems.append(f"{label}: not in the required stable record set")
            elif pid not in required_unstable:
                problems.append(f"{label}: not in the required unstable record set")
        else:
            if pid not in required_unstable and pid not in required_stable:
                problems.append(f"{label}: pdg id is not a legitimate catalog state")

        q_vendored = int(round(get_charge(pid))) if vendored is not None else None
        if q_vendored is not None and q_vendored != int(entry["Q"]):
            problems.append(
                f"{label}: transcription Q {entry['Q']} disagrees with the "
                f"vendored table charge {q_vendored}")
        if pid != ETAPRIME_PDG_ID and pid not in _DELTA1700_IDS:
            try:
                resolved = quantum_numbers(pid)
            except (KeyError, ValueError) as exc:
                problems.append(f"{label}: quantum numbers unresolvable: {exc}")
            else:
                expected = QuantumNumbers(int(entry["B"]), int(entry["Q"]),
                                          int(entry["S"]))
                if resolved != expected:
                    problems.append(
                        f"{label}: transcription (B, Q, S) = "
                        f"{(entry['B'], entry['Q'], entry['S'])} disagrees with "
                        f"quantum_numbers() = {resolved.as_tuple()}")

        if vendored is not None:
            for slot, actual in (("pole_mass", vendored.mass),
                                 ("nominal_width", vendored.width)):
                tag = entry["values"].get(slot)
                value = tag.get("transcribed_value") if isinstance(tag, dict) else None
                if value is None:
                    problems.append(f"{label}: {slot} provenance value is null")
                elif abs(float(value) - float(actual)) > MASS_TOL:
                    problems.append(
                        f"{label}: transcription {slot} {value} disagrees with "
                        f"the vendored table {actual}")

        problems.extend(_tag_problems(f"{label}: record provenance",
                                      entry.get("record_provenance"), True))
        for slot, tag in entry.get("values", {}).items():
            problems.extend(_tag_problems(f"{label}: {slot} provenance", tag, False))
        for channel in entry.get("channels", []):
            daughters = tuple(channel.get("daughters", ()))
            problems.extend(_tag_problems(
                f"{label}: channel {daughters} provenance",
                channel.get("provenance"), False))

        expected_stable = pid in required_stable
        if full_mode and bool(entry["stable_for_decay"]) != expected_stable:
            problems.append(
                f"{label}: stable_for_decay {entry['stable_for_decay']} disagrees "
                "with the required record set")

    if problems:
        # Refuse before any record construction: the transcription is
        # incomplete or disagrees with the vendored table.
        return [], problems

    # -- HadronRecord construction -------------------------------------------
    records: list[HadronRecord] = []
    for entry in values["records"]:
        pid = int(entry["pdg_id"])
        vendored = get_particle(pid)
        name = entry.get("name") or (vendored.name if vendored else "")
        pole = (float(vendored.mass) if vendored is not None
                else daughter_mass.get(pid, 0.0))
        width = (float(vendored.width) if vendored is not None
                 else _entry_width(entry))
        channels = []
        for channel in entry.get("channels", []):
            prov = channel["provenance"]
            channels.append(DecayChannel(
                parent_pdg=pid,
                daughters=tuple(int(d) for d in channel["daughters"]),
                branching_fraction=float(channel["branching_fraction"]),
                orbital_angular_momentum=int(channel["orbital_angular_momentum"]),
                is_three_body=bool(channel.get("is_three_body", False)),
                provenance=ProvenanceTag(
                    document=str(prov["document"]),
                    section_or_table=str(prov["section_or_table"]),
                    field=str(prov["field"]),
                    accessed_on=str(prov["accessed_on"]),
                    transcribed_value=prov["transcribed_value"],
                    transcriber=str(prov["transcriber"]),
                ),
            ))
        thresholds = [
            math.fsum(daughter_mass.get(int(d), get_mass(int(d), default=0.0))
                      for d in channel.daughters)
            for channel in channels
        ]
        lowest = min(thresholds) if thresholds else pole
        m_up = float(entry["values"]["mass_range"]["transcribed_value"])
        rec_tag = entry.get("record_provenance") or {}
        records.append(HadronRecord(
            pdg_id=pid,
            name=name,
            B=int(entry["B"]), Q=int(entry["Q"]), S=int(entry["S"]),
            isospin=float(entry["values"]["isospin"]["transcribed_value"]),
            isospin3=int(entry["values"]["isospin3"]["transcribed_value"]),
            spin=float(entry["values"]["spin"]["transcribed_value"]),
            parity=str(entry["values"]["parity"]["transcribed_value"]),
            pole_mass=pole,
            nominal_width=float(entry["values"]["nominal_width"]["transcribed_value"]),
            mass_range=(lowest, m_up),
            decay_channels=tuple(channels),
            stable_for_decay=bool(entry["stable_for_decay"]),
            provenance=ProvenanceTag(
                document=str(rec_tag.get("document", "")),
                section_or_table=str(rec_tag.get("section_or_table", "")),
                field=str(rec_tag.get("field", "record")),
                accessed_on=str(rec_tag.get("accessed_on", "")),
                transcribed_value=rec_tag.get("transcribed_value"),
                transcriber=str(rec_tag.get("transcriber", "")),
            ),
        ))
    return records, problems


def _entry_width(entry: dict) -> float:
    """Nominal width of a record absent from the vendored table, from its
    transcription provenance (eta-prime)."""
    tag = entry.get("values", {}).get("nominal_width")
    if isinstance(tag, dict) and tag.get("transcribed_value") is not None:
        return float(tag["transcribed_value"])
    return 0.0


def _termination_problems(records: list[HadronRecord]) -> list[str]:
    """The card's termination rule: every unstable record has at least one
    open channel at its pole mass, so the recursive algorithm of card 6
    always terminates (the termination set is closed under decay)."""
    problems: list[str] = []
    for record in records:
        if record.stable_for_decay or not record.decay_channels:
            continue
        open_channels = [
            channel
            for channel in record.decay_channels
            if math.fsum(_daughter_mass(int(d)) for d in channel.daughters)
            <= record.pole_mass + OPEN_CHANNEL_TOL
        ]
        if not open_channels:
            problems.append(
                f"record {record.name} (pdg {record.pdg_id}): no decay channel "
                f"is open at its pole mass {record.pole_mass}"
            )
    return problems


def _daughter_mass(pdg_id: int) -> float:
    """Channel-daughter mass for the termination check: the vendored table
    mass (every daughter is a stable-state endpoint or a catalog record
    whose mass equals the vendored table mass within tolerance)."""
    entry = get_particle(int(pdg_id))
    if entry is not None:
        return float(entry.mass)
    return float(get_mass(int(pdg_id), default=0.0))


def run_import(values: dict, out_path: Path, report_dir: Path,
               full_mode: bool, test_pass: bool) -> int:
    """Validate and write the catalog; returns 0 on success, 2 on refusal."""
    records, problems = build_records(values, full_mode)
    if problems:
        print("import_resonance_catalog: refusing the import; "
              f"{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    catalog = HadronCatalog(records)
    report = catalog.validate()
    all_errors = list(report.errors)
    all_errors.extend(_termination_problems(records))
    if all_errors:
        print("import_resonance_catalog: refusing the catalog; "
              f"{len(all_errors)} validation error(s):", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    # The assembled catalog has validated in memory (write-after-validate);
    # the round trip below re-parses the written file and re-validates.
    payload = {
        "catalog_version": CATALOG_VERSION,
        "label": CATALOG_LABEL,
        "recorded_on": RECORDED_ON,
        "provenance_discipline": PROVENANCE_DISCIPLINE,
        "stable_states": list(STABLE_HEADER_STATES),
    }
    payload.update(catalog.to_dict())

    # Write: the staged transcription, the catalog, and the acceptance report.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "catalog_values.json").write_text(
        json.dumps(values, indent=2) + "\n", encoding="utf-8")

    # Post-write verification: the written file must parse into
    # HadronCatalog with zero validation errors.
    reparse = HadronCatalog.from_json(out_path)
    reparse_report = reparse.validate()
    if reparse_report.errors:
        out_path.unlink(missing_ok=True)
        print("import_resonance_catalog: refusing the catalog; the written "
              f"file does not re-validate ({len(reparse_report.errors)} "
              "error(s)):", file=sys.stderr)
        for error in reparse_report.errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    # unstable_record_count: every record outside the pre-registered
    # 11-state stable_states termination list (the card's acceptance
    # assertion record_count == unstable_record_count + 11).  It comprises
    # the decayable resonance records and the cascade-terminal ground
    # states, both reported separately below.
    header_stable = set(STABLE_HEADER_STATES)
    unstable_count = sum(1 for r in records if int(r.pdg_id) not in header_stable)
    decayable_count = sum(1 for r in records if not r.stable_for_decay)
    terminal_ground_count = sum(
        1 for r in records
        if r.stable_for_decay and int(r.pdg_id) not in header_stable)
    stable_count = len(records) - unstable_count
    if full_mode and len(records) != unstable_count + len(STABLE_HEADER_STATES):
        print(
            "import_resonance_catalog: refusing the catalog; the record-"
            "count assertion record_count == unstable_record_count + 11 "
            f"fails ({len(records)} != {unstable_count} + "
            f"{len(STABLE_HEADER_STATES)})",
            file=sys.stderr,
        )
        return 2
    channel_count = sum(len(r.decay_channels) for r in records)
    try:
        manifest = json.loads(
            (NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json")
            .read_text(encoding="utf-8"))
        manifest_repinned = any(
            entry.get("card") == CARD_ID
            for entry in manifest.get("freeze_exceptions", []))
    except (OSError, ValueError):
        manifest_repinned = False

    try:
        rel_catalog = str(out_path.resolve().relative_to(MONO_ROOT))
    except ValueError:
        rel_catalog = str(out_path)

    payload_ids = {int(r.pdg_id) for r in records}
    eta_prime_present = ETAPRIME_PDG_ID in payload_ids
    report_payload = {
        "card": CARD_ID,
        "catalog_path": rel_catalog,
        "catalog_sha256": digest,
        "record_count": len(records),
        "unstable_record_count": unstable_count,
        "stable_record_count": stable_count,
        "decayable_record_count": decayable_count,
        "cascade_terminal_ground_state_count": terminal_ground_count,
        "channel_count": channel_count,
        "stable_states": list(STABLE_HEADER_STATES),
        "placeholder_towers_excluded": not (payload_ids & PLACEHOLDER_TOWER_IDS),
        "eta_prime_added": eta_prime_present,
        "validation_errors": 0,
        "provenance_complete": True,
        "test_pass": bool(test_pass),
        "manifest_repinned": manifest_repinned,
        "record_set_note": RECORD_SET_NOTE,
        "orbital_angular_momentum_convention": (
            "lowest partial wave L consistent with the parent J^P and the "
            "daughter parity product, (−1)^L = P_parent / prod(P_daughters), "
            "within |J − j| <= L <= J + j; three-body channels carry the "
            "total L of that wave"),
    }
    report_path = report_dir / "catalog_import_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n",
                           encoding="utf-8")
    print(f"import_resonance_catalog: wrote {out_path}")
    print(f"import_resonance_catalog: report {report_path}")
    print(
        f"import_resonance_catalog: {len(records)} records "
        f"({unstable_count} outside the stable_states list: "
        f"{decayable_count} decayable + {terminal_ground_count} "
        f"cascade-terminal ground states; {stable_count} stable_states "
        f"records; {channel_count} channels), validation errors 0"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (0 success, 2 refusal)."""
    parser = argparse.ArgumentParser(
        description="Import the provenance-tagged initial resonance catalog "
                    "(Round 2, card 2).")
    parser.add_argument(
        "--out", type=Path,
        default=NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json",
        help="catalog output path (default: the plan section 4.4 catalog file)")
    parser.add_argument(
        "--report", type=Path,
        default=MONO_ROOT / "dev" / "00runs" / "r2_catalog_import_2026-09-13",
        help="run directory for catalog_values.json and "
             "catalog_import_report.json")
    parser.add_argument(
        "--values", type=Path, default=None,
        help="optional transcription fixture (catalog_values.json structure); "
             "default: the embedded full transcription.  Fixture mode writes "
             "the fixture's catalog only and checks that every record is a "
             "legitimate catalog state.")
    parser.add_argument(
        "--test-pass", choices=("true", "false"), default="false",
        help="record the deterministic-test outcome in the acceptance report "
             "(the run record documents the test invocation)")
    args = parser.parse_args(argv)

    values = EMBEDDED_CATALOG_VALUES
    full_mode = True
    if args.values is not None:
        try:
            values = json.loads(args.values.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"import_resonance_catalog: cannot read the transcription "
                  f"fixture {args.values}: {exc}", file=sys.stderr)
            return 2
        full_mode = False

    return run_import(values, args.out, args.report, full_mode,
                      test_pass=args.test_pass == "true")


if __name__ == "__main__":
    raise SystemExit(main())

