#!/usr/bin/env python3
"""Staged R20 driver: the single-resonance unit grid (Round 2, card 10).

Task ``TASK-20260913-r2-run-r20-single-resonance-unit-grid`` (the run
R20 of plan section 8, line 551; the Round 2 card set of plan section
23).  R20 is the per-state verification of the cards 4-7 decay kernel:
for every catalog state, 100 fixed-mass points spanning threshold to
upper support; 10,000 decays per point for the important states and
2,000 for the minor states; the sampled channel, charge combination,
and kinematics are verified against the pre-registered analytic
expectation (the card-3/4 width law, the card-5 charge weights, exact
on-shell two-body kinematics, the spin-averaged angular kernel).  R20
feeds the Gate 2 branching-test and threshold-discontinuity clauses
(plan section 8, line 575) and the local-conservation clause at the
per-vertex level.  Sealed evidence under
``dev/00runs/r20_single_resonance_unit_grid_2026-09-13/``.

The driver implements the full run procedure and executes it in
phases: each invocation performs exactly one step and stops (the run
card dispatches the steps; pre-registration, generation, verification,
seal, and clean-environment validation land in this order, and the run
dir is immutable after the seal).

Modes (exactly one per invocation):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.
- ``--generate [--state PDG]``: generate the pre-registered decays for
  one state (with ``--state``) or for every state (without, catalog
  record order), writing ``state_<pdg>.json`` into the run dir, then
  stop.  Requires the pre-registration; every state row is asserted
  against it before the first decay.
- ``--verify``: run the verification over the generated
  ``state_<pdg>.json`` files, write ``aggregate.json`` and
  ``gate2_r20.json`` into the run dir, then stop.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the verification from
  a clean interpreter over the sealed artifacts, and confirm the
  report hashes match; write ``clean_environment_commands.json`` into
  the run dir, then stop.

Pre-registration contract (``pre_registration.json``; the driver
asserts the per-state rows against it before any generation):

- **State coverage**: every catalog record with
  ``stable_for_decay = false`` (the stable endpoints have no decays to
  verify; they are listed as "no decays").
- **Grid**: per state, ``M_j = mass_range[0] + (mass_range[1] -
  mass_range[0]) * j / 99``, ``j = 0..99`` (100 points, both endpoints
  inclusive; the R20 grid convention shared with the card-8
  open-weight sweep).
- **Counts**: 10,000 decays per point for the important states and
  2,000 for the minor states (round 2 index item 8).  A catalog state
  on neither list is a pre-registration violation: the driver raises
  and stops.
- **Three-body channel set**: the approved card-7 set (the approval
  gate closed 2026-09-14); the executed set is the committed catalog's
  ``is_three_body`` channels, machine-enumerated in the
  pre-registration.
- **Degenerate point rule**: a grid point ``M_j`` at which the total
  open weight (the card-3/4 channel-summed width) is exactly zero is a
  documented degenerate point (0 decays, the skip reason in the
  per-state report); it does not consume the decay budget.  The
  pre-registration records the expected skip points per state.
- **Seeds**: master seed 20260913, ``replica_index = 0``; per (state
  rank ``s``, point rank ``j``, decay ``i``): ``event_index = s *
  1_000_000 + j * 10_000 + i`` with ``i`` below the state's per-point
  count (the closed labels ``decay_channel`` / ``decay_kinematics``,
  re-derived per decay through the card-6 kernel; no extension of
  ``SUBSTREAM_LABELS``).
- **Verification tolerances**: 3-sigma per check with the binomial
  ``1/sqrt(N)`` standard error; ``1e-9`` for the deterministic closure
  and on-shell checks.

The "unit" of the unit grid: the parent mass is fixed at ``M_j`` (not
a spectral draw); the daughter resonances keep their card-3 mass draws
(``draw_mass`` inside the on-shell mass budget window).  No other
kernel parameter is touched.

Deterministic per-vertex statistics (the local-conservation clause):
per vertex the maximum component deviation of the daughter
four-momentum sum from the parent four-momentum, and per daughter the
on-shell deviation against the realized mass in the kernel's
``|m^2 - m^2_ref|`` convention, additionally reported as an
invariant-mass deviation.  A resonance daughter's realized mass is the
parent mass of its own child vertex, recovered from the kernel's
queue-order vertex list (the child mapping is asserted); a stable
daughter's reference mass is its catalog pole mass (the vendored PDG
table otherwise).  The kernel's own ``KinematicsError`` protections
propagate unchanged and stop the run.

Verification (per state, aggregated over open points and decays):

- **Branching fractions**: at the pole-mass point (the open grid point
  closest to the record ``pole_mass``) and at every open point, the
  empirical per-channel fraction matches the normalized open weight
  ``weight(c) / sum_open weight`` (the card-3/4 law) within 3-sigma
  per channel.  For a single-combination class the object weight is
  ``BR_c * Gamma_c(M_j)`` exactly; for a multi-combination class the
  card-6 documented class distribution ``cw(pair) * Gamma_class(M_j) /
  BR_class`` applies.  At the pole point the normalized open weight
  equals the catalog branching fraction per channel whenever the
  represented channels sum to 1 (all records except the documented
  radiative_budget records eta and eta-prime(958), where the kernel's
  renormalized represented-channels convention applies).
- **Charge weights**: at every open point, within each two-body
  multiplet class with at least two open combinations, the empirical
  combination share matches the card-5 share ``cw / sum_open cw``
  within 3-sigma.
- **Mass shells**: deterministic; the maxima over all vertices are
  reported against the ``1e-9`` tolerance.
- **Angular kernels**: the ``cos(theta)`` of the first daughter in the
  parent rest frame (the inverse of the kernel's own boost
  convention), one entry per two-body vertex, pooled over the open
  points, is uniform in 10 bins to 3-sigma per bin.
- **Threshold continuity**: at the two lowest open points, for every
  channel and class combination open at both points, the sampled
  difference of the estimates agrees with the expected difference
  (the card-3/4 normalized open weight evaluated at the two points)
  within the combined 3-sigma.  A combination that opens between the
  two points is the width law's own threshold behavior and is verified
  by the per-point branching check instead.

``aggregate.json`` and ``gate2_r20.json`` carry no wall-clock fields:
their bytes are a pure function of the sealed inputs, which is what
the clean-environment validation hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no external tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the resonance catalog.
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
            "staged_r20_unit_grid: cannot locate the na49_mixing package "
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

import numpy as np  # noqa: E402

from na49_mixing.hadron_catalog import HadronCatalog, HadronRecord  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402

# The private imports below are the committed card-6 kernel's own class
# bookkeeping (multiplet keys, charge-pair keys) and boost convention.
# The R20 expected values must use the exact conventions the kernel
# implements, so the driver reuses them instead of re-deriving private
# variants; the kernel itself is never modified.
from na49_mixing.resonance_decay import (  # noqa: E402
    ResonanceDecayer,
    SourceRecord,
    _boost_lab,
    _channel_pair_key,
    _class_flavor_pair,
    _class_key,
    class_pair_of,
    derive_charge_weights,
)
from na49_mixing.resonance_widths import (  # noqa: E402
    channel_width,
    total_width,
)

RUN_ID = "R20"
CARD_ID = "TASK-20260913-r2-run-r20-single-resonance-unit-grid"
RUN_NAME = "r20_single_resonance_unit_grid_2026-09-13"
DEFAULT_RUN_DIR = "dev/00runs/r20_single_resonance_unit_grid_2026-09-13"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
DRIVER_REL = "helpers/benchmarking/staged_r20_unit_grid.py"

#: The card chain this run sits on (cards 1-9 of the Round 2 set plus
#: the run card itself).
CARD_CHAIN = (
    "TASK-20260913-r2-hadron-resonance-schema",
    "TASK-20260913-r2-provenance-catalog-import",
    "TASK-20260913-r2-mass-dependent-widths",
    "TASK-20260913-r2-barrier-factors",
    "TASK-20260913-r2-isospin-charge-completion",
    "TASK-20260913-r2-recursive-two-body-decays",
    "TASK-20260913-r2-selected-three-body-decays",
    "TASK-20260913-r2-branching-spectral-normalization-tests",
    "TASK-20260913-r2-source-grid-runner",
    "TASK-20260913-r2-run-r20-single-resonance-unit-grid",
)

# ---------------------------------------------------------------------------
# Pre-registered grid, statistics, seeds, tolerances (the run card).
# A change re-opens the run card before the affected phase starts.
# ---------------------------------------------------------------------------
GRID_POINTS = 100
GRID_FORMULA = (
    "M_j = mass_range[0] + (mass_range[1] - mass_range[0]) * j / 99, "
    "j = 0..99 (both endpoints inclusive; the R20 grid convention "
    "shared with the card-8 open-weight sweep)"
)

#: The R20 statistics (round 2 index item 8): 10,000 decays per point
#: for the important states, 2,000 for the minor states.
IMPORTANT_DECAYS_PER_POINT = 10_000
MINOR_DECAYS_PER_POINT = 2_000

#: The pre-registered important/minor split (round 2 index item 8), by
#: PDG id.  "Important" are the states whose decay chains feed at
#: least one of pi+-, K+-, p, pbar; "minor" are all remaining catalog
#: states.  The driver asserts that every catalog record with
#: ``stable_for_decay = false`` is on exactly one list; a state on
#: neither list is a pre-registration violation.
IMPORTANT_STATES = frozenset({
    # Delta(1232) and anti
    1114, 2114, 2214, 2224, -1114, -2114, -2214, -2224,
    # N*(1440) and the available anti partner
    12112, 12212, -12112,
    # N*(1520) and the available anti partner
    2116, 2216, -2116,
    # N(1535) and the available anti partner
    102112, 102212, -102112,
    # N*(1680) and the available anti partner
    22112, 22212, -22112,
    # N(1650) and the available anti partner
    122112, 122212, -122112,
    # N(1720) and the available anti partners
    212114, 212214, -212114, -212214,
    # Delta(1600)
    201114, 202114, 202214, 202224,
    # Delta(1700)
    121114, 122114, 122214, 122224,
    # rho(770)
    113, 213, -213,
    # omega(782)
    223,
    # a2(1320)
    115, 215, -215,
    # K*(892)
    313, 323, -313, -323,
    # K1(1270)
    10313, 10323, -10313, -10323,
    # K2*(1430)
    315, 325, -315, -325,
    # phi(1020)
    333,
    # Lambda(1520)
    102134,
    # Sigma(1385)
    3114, 3214, 3224,
})

MINOR_STATES = frozenset({
    # eta
    221,
    # eta-prime(958)
    377,
    # rho(1450)
    100113, 100213, -100213,
    # rho(1700)
    30113, 30213, -30213,
    # b1(1235)
    10113, 10213, -10213,
    # a1(1260)
    20113, 20213, -20213,
    # rho3(1690)
    117, 217, -217,
    # f2(1270)
    225,
    # f4(2050)
    229,
    # omega(1420)
    100223,
    # omega(1650)
    30223,
})

#: The pre-registered states a later revision of the committed catalog
#: removed: the eta-prime(958) at the card-pinned key 377, which no class
#: and no channel reaches (the orphan-record revision of 2026-10-29,
#: ``catalog_orphan_record_revision.py``).  R20 ran it on the catalog of
#: 2026-09-13; a run on the revised catalog covers the other states.
REMOVED_BY_REVISION = frozenset({377})

#: Master seed and the per-decay event-index map (the run card's seed
#: map).  ``s`` is the state rank (catalog record order over the
#: non-stable records, 0-based), ``j`` the point rank 0..99, ``i`` the
#: decay index within the point (below the state's per-point count).
#: ``replica_index`` is 0 throughout.
MASTER_SEED = 20260913
REPLICA_INDEX = 0
SEED_RULE = "event_index = s * 1_000_000 + j * 10_000 + i"
SEED_LABELS = ("decay_channel", "decay_kinematics")

#: Verification tolerances (the run card): 3-sigma per statistical
#: check with the binomial 1/sqrt(N) standard error; 1e-9 for the
#: deterministic closure and on-shell checks.
N_SIGMA = 3.0
DETERMINISTIC_TOL = 1e-9
ANGULAR_BINS = 10

#: The approved card-7 three-body channel set (the approval gate
#: closed 2026-09-14; round 2 index, scope decision on the task-15
#: channel set).  The committed catalog flags only the omega(782)
#: channel of this proposal; the phi(1020), a1(1260), rho(770), and
#: Delta(1232) rows are the documented catalog-side follow-up and are
#: not executed by R20.
APPROVED_CARD7_THREE_BODY = (
    {
        "parent_pdg": 223,
        "parent": "omega(782)",
        "daughters": (211, -211, 111),
        "proposal": "omega(782) -> pi+ pi- pi0 (BR 0.892)",
        "in_committed_catalog": True,
    },
    {
        "parent_pdg": 333,
        "parent": "phi(1020)",
        "daughters": (211, -211, 111),
        "proposal": "phi(1020) -> pi+ pi- pi0 (0.157)",
        "in_committed_catalog": False,
    },
    {
        "parent_pdg": 20113,
        "parent": "a1(1260)0",
        "daughters": (211, -211, 111),
        "proposal": "a1(1260) -> pi+ pi- pi0 / pi0 pi0 pi0 (0.85 combined)",
        "in_committed_catalog": False,
    },
    {
        "parent_pdg": 113,
        "parent": "rho(770)0",
        "daughters": (211, -211, 111),
        "proposal": "rho(770) -> pi+ pi- pi0 (0.049)",
        "in_committed_catalog": False,
    },
    {
        "parent_pdg": 2224,
        "parent": "Delta(1232)++",
        "daughters": None,
        "proposal": "Delta(1232) -> N pi pi (0.011)",
        "in_committed_catalog": False,
    },
)

#: The recorded skip reason of a degenerate point.
DEGENERATE_SKIP_REASON = (
    "total open weight zero at the grid point (the card-3/4 hard zero "
    "at the channel threshold); 0 decays, the decay budget is not "
    "consumed"
)

#: The pole-point expected-value convention (recorded verbatim in the
#: pre-registration; see the module docstring).
POLE_EXPECTED_NOTE = (
    "the pole-point expected fraction is the normalized open weight at "
    "the pole grid point; it equals the catalog branching fraction per "
    "channel whenever the represented channels sum to 1 (all records "
    "except the documented radiative_budget records eta (pdg 221) and "
    "eta-prime(958) (pdg 377), where the kernel's renormalized "
    "represented-channels convention applies)"
)

#: The threshold-continuity convention (recorded verbatim in the
#: pre-registration; see the module docstring).
CONTINUITY_NOTE = (
    "at the two lowest open points, for every channel and class "
    "combination open at both points, the sampled difference of the "
    "estimates agrees with the expected difference (the card-3/4 "
    "normalized open weight evaluated at the two points) within the "
    "combined 3 sigma; a combination that opens between the two points "
    "is the width law's own threshold behavior and is verified by the "
    "per-point branching check instead"
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as UTF-8 JSON (indent 2, fixed key order, a
    trailing newline, LF endings); identical inputs produce identical
    bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _canonical_json_bytes(payload: dict) -> bytes:
    """The exact byte sequence :func:`_write_json` writes for *payload*."""
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_key(daughters) -> str:
    """The JSON channel key of a daughter tuple (``"211,-211"``)."""
    return ",".join(str(int(d)) for d in daughters)


def environment_fingerprint() -> dict:
    """The environment fingerprint recorded in the provenance and the
    clean-environment validation."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "executable": sys.executable,
    }


def _git_info() -> dict:
    """The recorded git identities (defensive: a checkout without a
    working git records ``None``)."""
    info: dict = {}
    for name, cwd in (("nested_head", NESTED_ROOT), ("outer_head", MONO_ROOT)):
        try:
            head = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(cwd), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            info[name] = None
            info[name.replace("head", "dirty_entries")] = None
        else:
            info[name] = head
            info[name.replace("head", "dirty_entries")] = len(
                dirty.splitlines()
            )
    return info


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)


#: Cache of the on-shell reference masses of stable daughters (the
#: kernel's daughter-mass resolution: the catalog pole mass, else the
#: vendored PDG table's mass).
_STABLE_MASS_CACHE: dict[int, float] = {}


def _stable_reference_mass(catalog: HadronCatalog, pdg: int) -> float:
    """The on-shell reference mass of a stable daughter (the kernel's
    daughter-mass resolution: the catalog record's pole mass, else the
    vendored PDG table's mass)."""
    cached = _STABLE_MASS_CACHE.get(pdg)
    if cached is not None:
        return cached
    try:
        mass = float(catalog.get(pdg).pole_mass)
    except KeyError:
        entry = get_particle(pdg)
        if entry is None:
            raise KeyError(
                f"daughter pdg_id {pdg} is neither a catalog record nor "
                "in the vendored PDG table"
            ) from None
        mass = float(entry.mass)
    _STABLE_MASS_CACHE[pdg] = mass
    return mass


# ---------------------------------------------------------------------------
# Catalog rows, classification, grid, degenerate points
# ---------------------------------------------------------------------------
def _daughter_masses(
    catalog: HadronCatalog, record: HadronRecord
) -> dict[int, float]:
    """Pole masses of every channel daughter of *record* (the card-3
    width law's daughter-mass map; the same construction as the
    kernel's map)."""
    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter in masses:
                continue
            try:
                masses[daughter] = float(catalog.get(daughter).pole_mass)
            except KeyError:
                entry = get_particle(daughter)
                if entry is None:
                    raise KeyError(
                        f"daughter pdg_id {daughter} is neither a catalog "
                        "record nor in the vendored PDG table"
                    ) from None
                masses[daughter] = float(entry.mass)
    return masses


def _grid_masses(record: HadronRecord) -> list[float]:
    """The pre-registered R20 grid ``M_j = lo + (hi - lo) * j / 99``."""
    lo = float(record.mass_range[0])
    hi = float(record.mass_range[1])
    return [lo + (hi - lo) * j / 99 for j in range(GRID_POINTS)]


def _degenerate_points(
    record: HadronRecord, masses: dict[int, float]
) -> list[int]:
    """The degenerate grid points of *record*: the ``j`` whose total
    open weight (the card-3/4 channel-summed width) is exactly zero.
    A negative or non-finite total width is a width-law violation and
    stops the run."""
    out: list[int] = []
    for j, mass in enumerate(_grid_masses(record)):
        total = total_width(mass, record, masses)
        if math.isnan(total) or total < 0.0:
            raise RuntimeError(
                f"staged_r20_unit_grid: record {record.name} (pdg "
                f"{record.pdg_id}) has a negative or non-finite total "
                f"width {total!r} at grid mass {mass!r} GeV (a "
                "width-law violation; the run stops)"
            )
        if total == 0.0:
            out.append(j)
    return out


def _state_rows(catalog: HadronCatalog) -> list[dict]:
    """The pre-registered state rows: every catalog record with
    ``stable_for_decay = false``, in catalog record order, with the
    state rank ``s`` and the pre-registered per-point decay count.

    The important/minor assertion of the run card: every non-stable
    catalog record must be on exactly one of the two pre-registered
    lists; a state not on either list is a pre-registration violation
    (the driver raises and stops).  The reverse check holds too: every
    list entry must exist as a non-stable catalog record, except a state
    a later catalog revision removed (:data:`REMOVED_BY_REVISION`)."""
    if IMPORTANT_STATES & MINOR_STATES:
        raise RuntimeError(
            "staged_r20_unit_grid: the pre-registered important and "
            "minor state lists overlap; the per-state counts are "
            "ambiguous (a pre-registration violation)"
        )
    rows: list[dict] = []
    problems: list[str] = []
    seen_important: set[int] = set()
    seen_minor: set[int] = set()
    rank = 0
    for record in catalog.records():
        if record.stable_for_decay:
            continue
        pdg = int(record.pdg_id)
        if pdg in IMPORTANT_STATES:
            classification = "important"
            decays = IMPORTANT_DECAYS_PER_POINT
            seen_important.add(pdg)
        elif pdg in MINOR_STATES:
            classification = "minor"
            decays = MINOR_DECAYS_PER_POINT
            seen_minor.add(pdg)
        else:
            problems.append(
                f"catalog state {record.name} (pdg {pdg}) is on neither "
                "the important nor the minor pre-registered list; a "
                "state not on either list is a pre-registration violation"
            )
            continue
        rows.append({
            "rank": rank,
            "record": record,
            "classification": classification,
            "decays_per_point": decays,
        })
        rank += 1
    for pdg in sorted((IMPORTANT_STATES | MINOR_STATES)
                      - seen_important - seen_minor - REMOVED_BY_REVISION):
        problems.append(
            f"pre-registered state pdg {pdg} is not a non-stable catalog "
            "record (the state lists must match the committed catalog)"
        )
    if problems:
        raise RuntimeError(
            "staged_r20_unit_grid: the pre-registration state lists "
            "violate the round 2 index item 8 contract:\n  "
            + "\n  ".join(problems)
        )
    return rows


def _state_channel_rows(record: HadronRecord) -> list[dict]:
    """The per-channel rows recorded for *record* (the executed channel
    set the verification compares against)."""
    return [
        {
            "daughters": [int(d) for d in channel.daughters],
            "branching_fraction": float(channel.branching_fraction),
            "is_three_body": bool(channel.is_three_body),
            "orbital_angular_momentum": int(channel.orbital_angular_momentum),
        }
        for channel in record.decay_channels
    ]


# ---------------------------------------------------------------------------
# The pre-registered expected open weights (the card-6 documented
# convention, a read-only mirror for the expected values; the committed
# kernel remains the object under test)
# ---------------------------------------------------------------------------
class _StateContext:
    """Per-state context: daughter masses, the grid, the degenerate
    points, and the card-5 class maps (cached per (record, multiplet))."""

    def __init__(self, catalog: HadronCatalog, record: HadronRecord) -> None:
        self.catalog = catalog
        self.record = record
        self.masses = _daughter_masses(catalog, record)
        self.grid = _grid_masses(record)
        self.degenerate = set(_degenerate_points(record, self.masses))
        self.class_maps: dict[tuple, dict] = {}

    def class_map(self, channel) -> dict:
        multiplet = class_pair_of(channel)
        key = (int(self.record.pdg_id), multiplet)
        cached = self.class_maps.get(key)
        if cached is None:
            cached = derive_charge_weights(
                self.record, multiplet, self.catalog
            )
            if not cached:
                raise RuntimeError(
                    "staged_r20_unit_grid: the multiplet class "
                    f"{multiplet} of {self.record.name} (pdg "
                    f"{self.record.pdg_id}) has an empty card-5 charge "
                    "map; the committed catalog passed the card-5 audit "
                    "so this is a structural violation (the run stops)"
                )
            self.class_maps[key] = cached
        return cached


def _expected_point(
    ctx: _StateContext, mass: float
) -> tuple[dict[str, float], dict[str, dict[str, tuple[str, float]]]]:
    """The pre-registered expected open-object weights at *mass*.

    The documented card-6 convention (module docstring of
    ``na49_mixing.resonance_decay``): the sampled object is one
    concrete charge combination of one two-body multiplet class, with
    the open weight ``weight(pair) = cw(pair) * Gamma_class(M) /
    BR_class`` over the open combinations of the class (which
    reproduces the pre-registered ``BR_c * Gamma_c(M)`` exactly for a
    single-combination class), and every flagged three-body channel is
    one open object at its card-3/4 partial width.  Returns:

    - ``weights``: ``{channel_key: open weight}`` over the open
      objects (``channel_key`` is the daughter-tuple string);
    - ``class_shares``: per two-body class with at least two open
      combinations, ``{class key: {channel_key: (pair key, expected
      share)}}`` with the card-5 shares ``cw / sum_open cw``.
    """
    record = ctx.record
    catalog = ctx.catalog
    weights: dict[str, float] = {}
    classes: dict[tuple, list] = {}
    order: list[tuple] = []
    for channel in record.decay_channels:
        if channel.is_three_body:
            if len(channel.daughters) != 3:
                continue
            if not float(channel.branching_fraction) > 0.0:
                continue
            gamma = channel_width(mass, channel, record, ctx.masses)
            if not gamma > 0.0:
                continue
            key = _channel_key(channel.daughters)
            if key in weights:
                raise RuntimeError(
                    f"staged_r20_unit_grid: record {record.name} (pdg "
                    f"{record.pdg_id}) carries the duplicate channel "
                    f"{key}; the channel set is not enumerable"
                )
            weights[key] = float(gamma)
            continue
        if len(channel.daughters) != 2:
            continue
        key = _class_key(_class_flavor_pair(channel.daughters), catalog)
        if key not in classes:
            classes[key] = []
            order.append(key)
        classes[key].append(channel)

    class_shares: dict[str, dict[str, tuple[str, float]]] = {}
    for class_id in order:
        class_channels = classes[class_id]
        class_map = ctx.class_map(class_channels[0])
        allowed: list[tuple] = []
        for channel in class_channels:
            if not float(channel.branching_fraction) > 0.0:
                continue
            gamma = channel_width(mass, channel, record, ctx.masses)
            if not gamma > 0.0:
                continue
            pair_key = _channel_pair_key(
                channel.daughters, class_pair_of(channel), catalog
            )
            if pair_key not in class_map:
                continue  # isospin-forbidden: zero weight (card 5)
            allowed.append((channel, pair_key, gamma))
        if not allowed:
            continue
        gamma_class = math.fsum(gamma for _, _, gamma in allowed)
        if not gamma_class > 0.0:
            continue
        br_class = math.fsum(
            float(channel.branching_fraction)
            for channel, _, _ in allowed
        )
        if not br_class > 0.0:
            continue
        open_cw: dict[str, tuple[str, float]] = {}
        for channel, pair_key, _ in allowed:
            cw = float(class_map[pair_key])
            if not cw > 0.0:
                continue  # zero tabulated share (the kernel's rule)
            key = _channel_key(channel.daughters)
            if key in open_cw:
                raise RuntimeError(
                    f"staged_r20_unit_grid: record {record.name} (pdg "
                    f"{record.pdg_id}) class {class_id} carries the "
                    f"concrete combination {key} twice"
                )
            open_cw[key] = (str(pair_key), cw)
            weights[key] = cw * gamma_class / br_class
        if len(open_cw) >= 2:
            cw_total = math.fsum(cw for _, cw in open_cw.values())
            class_shares[str(class_id)] = {
                key: (pair_key, cw / cw_total)
                for key, (pair_key, cw) in open_cw.items()
            }
    return weights, class_shares


# ---------------------------------------------------------------------------
# Mode: --pre-registration
# ---------------------------------------------------------------------------
def _build_pre_registration(catalog: HadronCatalog) -> dict:
    rows = _state_rows(catalog)
    states: list[dict] = []
    skip_points_per_state: dict[str, list[int]] = {}
    total_budget = 0
    total_expected_decays = 0
    total_skips = 0
    skips_by_class = {"important": 0, "minor": 0}
    three_body_executed: list[dict] = []
    for record in catalog.records():
        for channel in record.decay_channels:
            if channel.is_three_body:
                three_body_executed.append({
                    "parent_pdg": int(record.pdg_id),
                    "parent": str(record.name),
                    "daughters": [int(d) for d in channel.daughters],
                    "branching_fraction": float(channel.branching_fraction),
                })
    for row in rows:
        record = row["record"]
        ctx = _StateContext(catalog, record)
        degenerate = sorted(ctx.degenerate)
        n_open = GRID_POINTS - len(degenerate)
        skip_points_per_state[str(int(record.pdg_id))] = degenerate
        states.append({
            "rank": row["rank"],
            "pdg_id": int(record.pdg_id),
            "name": str(record.name),
            "classification": row["classification"],
            "decays_per_point": row["decays_per_point"],
            "pole_mass": float(record.pole_mass),
            "nominal_width": float(record.nominal_width),
            "mass_range": [
                float(record.mass_range[0]), float(record.mass_range[1]),
            ],
            "channel_branching_sum": math.fsum(
                float(ch.branching_fraction)
                for ch in record.decay_channels
            ),
            "channels": _state_channel_rows(record),
            "expected_skip_points": degenerate,
            "expected_skip_count": len(degenerate),
            "open_points": n_open,
            "expected_decays": n_open * row["decays_per_point"],
        })
        total_budget += GRID_POINTS * row["decays_per_point"]
        total_expected_decays += n_open * row["decays_per_point"]
        total_skips += len(degenerate)
        skips_by_class[row["classification"]] += len(degenerate)

    no_decay_states = [
        {"pdg_id": int(record.pdg_id), "name": str(record.name)}
        for record in catalog.records()
        if record.stable_for_decay
    ]
    return {
        "schema": "na49_mixing.staged_r20_unit_grid.pre_registration/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "written_before_generation": True,
        "grid": {
            "points_per_state": GRID_POINTS,
            "formula": GRID_FORMULA,
            "endpoints_inclusive": True,
            "note": "the parent mass is fixed at M_j (the unit of the "
                    "unit grid: the parent mass is not a spectral draw); "
                    "the daughter resonances keep their card-3 mass "
                    "draws (draw_mass inside the on-shell mass budget "
                    "window); no other kernel parameter is touched",
        },
        "state_coverage": {
            "rule": "every catalog record with stable_for_decay = false "
                    "(the stable endpoints have no decays to verify; "
                    "they are listed as no-decay states)",
            "n_important": len(IMPORTANT_STATES),
            "n_minor": len(MINOR_STATES),
            "n_decay_states": len(rows),
            "n_no_decay_states": len(no_decay_states),
            "important_pdgs": sorted(IMPORTANT_STATES),
            "minor_pdgs": sorted(MINOR_STATES),
            "no_decay_states": no_decay_states,
            "assertion": "every catalog record with stable_for_decay = "
                         "false is on exactly one list; a state not on "
                         "either list is a pre-registration violation "
                         "(the driver raises)",
            "states": states,
        },
        "counts": {
            "important_decays_per_point": IMPORTANT_DECAYS_PER_POINT,
            "minor_decays_per_point": MINOR_DECAYS_PER_POINT,
            "definition": "round 2 index item 8: important states are "
                          "the states whose decay chains feed at least "
                          "one of pi+-, K+-, p, pbar; minor states are "
                          "all remaining catalog states",
        },
        "three_body_channels": {
            "approved_card7_set": [
                {
                    "parent_pdg": entry["parent_pdg"],
                    "parent": entry["parent"],
                    "daughters": (list(entry["daughters"])
                                  if entry["daughters"] else None),
                    "proposal": entry["proposal"],
                    "in_committed_catalog": entry["in_committed_catalog"],
                }
                for entry in APPROVED_CARD7_THREE_BODY
            ],
            "executed_set": three_body_executed,
            "note": "the executed set is the committed catalog's "
                    "is_three_body channels (machine-enumerated); the "
                    "card-7 approved rows beyond omega(782) -> pi+ pi- "
                    "pi0 are the documented catalog-side follow-up "
                    "(round 2 index scope decision) and are not part of "
                    "the committed catalog",
        },
        "degenerate_point_rule": {
            "definition": "a grid point M_j at which the total open "
                          "weight (the card-3/4 channel-summed width) is "
                          "exactly zero",
            "budget": "0 decays, the skip reason in the per-state "
                      "report; the point does not consume the decay "
                      "budget",
            "checks_note": "the branching verification at the pole and "
                           "the threshold-continuity check use the open "
                           "points only",
            "expected_skip_points_per_state": skip_points_per_state,
            "expected_skip_count_total": total_skips,
        },
        "seeds": {
            "master_seed": MASTER_SEED,
            "replica_index": REPLICA_INDEX,
            "event_index_rule": SEED_RULE,
            "definitions": "s = state rank (catalog record order over "
                           "the non-stable records, 0-based); j = point "
                           "rank 0..99; i = decay index within the "
                           "point, i below the state's per-point count",
            "substream_labels": list(SEED_LABELS),
            "labels_note": "the closed labels decay_channel / "
                           "decay_kinematics; no extension of "
                           "SUBSTREAM_LABELS; the kernel re-derives both "
                           "from the same (master_seed, replica_index, "
                           "event_index) triple",
        },
        "verification_tolerances": {
            "n_sigma": N_SIGMA,
            "sigma_convention": "binomial standard error "
                                "sqrt(p (1 - p) / N), the 1/sqrt(N) "
                                "convention",
            "deterministic_tol": DETERMINISTIC_TOL,
            "deterministic_note": "four-momentum closure in GeV; on "
                                  "shell in the kernel's |m^2 - "
                                  "m^2_ref| convention (GeV^2), "
                                  "additionally reported as an "
                                  "invariant-mass deviation in GeV",
            "angular_bins": ANGULAR_BINS,
            "pole_expected_note": POLE_EXPECTED_NOTE,
            "continuity_note": CONTINUITY_NOTE,
        },
        "generation_plan": {
            "order": "catalog record order (the state rank s)",
            "total_decays_budget_all_points": total_budget,
            "total_expected_decays_open_points": total_expected_decays,
            "expected_skip_points_total": total_skips,
            "expected_skips_by_class": skips_by_class,
        },
    }


def _run_pre_registration(run_dir: Path) -> None:
    catalog = _load_catalog()
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / "pre_registration.json"
    prov_path = run_dir / "provenance.json"
    if pre_path.is_file() and sorted(run_dir.glob("state_*.json")):
        raise RuntimeError(
            f"staged_r20_unit_grid: {pre_path} exists and state files "
            "are present; the pre-registration is immutable after "
            "generation has started (a change invalidates the run; "
            "re-seal under a new date)"
        )
    _write_json(pre_path, _build_pre_registration(catalog))

    catalog_path = NESTED_ROOT / CATALOG_REL
    provenance = {
        "schema": "na49_mixing.staged_r20_unit_grid.provenance/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "card_chain": list(CARD_CHAIN),
        "catalog": {
            "path": CATALOG_REL,
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        },
        "driver": DRIVER_REL,
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_generation": True,
    }
    _write_json(prov_path, provenance)
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    print(
        "staged_r20_unit_grid: pre-registration written to "
        f"{pre_path} ({pre['state_coverage']['n_decay_states']} decay "
        "states = "
        f"{pre['state_coverage']['n_important']} important + "
        f"{pre['state_coverage']['n_minor']} minor, "
        f"{pre['generation_plan']['total_expected_decays_open_points']} "
        "expected decays over the open points, "
        f"{pre['generation_plan']['expected_skip_points_total']} "
        "expected degenerate-point skips); nothing generated"
    )


# ---------------------------------------------------------------------------
# Mode: --generate
# ---------------------------------------------------------------------------
def _load_pre_registration(run_dir: Path) -> dict:
    path = run_dir / "pre_registration.json"
    if not path.is_file():
        raise RuntimeError(
            f"staged_r20_unit_grid: {path} is missing; the run requires "
            "the pre-registration (run --pre-registration first)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_rows_against_pre_registration(
    catalog: HadronCatalog, pre: dict
) -> list[dict]:
    """Assert every state row against the written pre-registration
    (rank, classification, per-point count, expected skip points) and
    the pre-registration's seed/grid conventions against this driver;
    a mismatch invalidates the run (the driver raises)."""
    rows = _state_rows(catalog)
    states = {
        int(entry["pdg_id"]): entry
        for entry in pre["state_coverage"]["states"]
    }
    problems: list[str] = []
    for row in rows:
        record = row["record"]
        pdg = int(record.pdg_id)
        entry = states.get(pdg)
        if entry is None:
            problems.append(
                f"state {record.name} (pdg {pdg}) is missing from the "
                "pre-registration"
            )
            continue
        if int(entry["rank"]) != row["rank"]:
            problems.append(f"pdg {pdg}: rank mismatch")
        if str(entry["classification"]) != row["classification"]:
            problems.append(f"pdg {pdg}: classification mismatch")
        if int(entry["decays_per_point"]) != row["decays_per_point"]:
            problems.append(f"pdg {pdg}: per-point count mismatch")
        degenerate = sorted(_degenerate_points(
            record, _daughter_masses(catalog, record)
        ))
        if [int(j) for j in entry["expected_skip_points"]] != degenerate:
            problems.append(f"pdg {pdg}: expected skip points mismatch")
    if int(pre["seeds"]["master_seed"]) != MASTER_SEED:
        problems.append("master_seed mismatch")
    if int(pre["seeds"]["replica_index"]) != REPLICA_INDEX:
        problems.append("replica_index mismatch")
    if str(pre["seeds"]["event_index_rule"]) != SEED_RULE:
        problems.append("event_index_rule mismatch")
    if int(pre["grid"]["points_per_state"]) != GRID_POINTS:
        problems.append("grid points mismatch")
    if problems:
        raise RuntimeError(
            "staged_r20_unit_grid: the written pre-registration does not "
            "match this driver and catalog (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return rows


def _generate_state(catalog: HadronCatalog, run_dir: Path, row: dict) -> Path:
    """Generate the pre-registered decays of one state over its grid
    and write ``state_<pdg>.json``.  Open points carry the full
    pre-registered decay count; degenerate points carry 0 decays and
    the recorded skip reason."""
    record = row["record"]
    rank = row["rank"]
    n_decays = row["decays_per_point"]
    ctx = _StateContext(catalog, record)
    degenerate = sorted(ctx.degenerate)
    points: dict[str, dict] = {}
    state_max_closure = 0.0
    state_max_os2 = 0.0
    state_max_os = 0.0
    state_vertices = 0
    state_two = 0
    state_three = 0
    total_decays = 0
    for j, mass in enumerate(ctx.grid):
        if j in ctx.degenerate:
            points[str(j)] = {
                "j": j,
                "mass_gev": mass,
                "degenerate": True,
                "n_decays": 0,
                "skip_reason": DEGENERATE_SKIP_REASON,
            }
            continue
        counts: dict[str, int] = {}
        hist = [0] * ANGULAR_BINS
        n_two = 0
        n_three = 0
        max_closure = 0.0
        max_os2 = 0.0
        max_os = 0.0
        for i in range(n_decays):
            event_index = rank * 1_000_000 + j * 10_000 + i
            decayer = ResonanceDecayer(
                catalog, MASTER_SEED, REPLICA_INDEX, event_index
            )
            vertices, _finals = decayer.decay_source(SourceRecord(
                pdg_id=int(record.pdg_id),
                four_momentum=(float(mass), 0.0, 0.0, 0.0),
                B=int(record.B),
                Q=int(record.Q),
                S=int(record.S),
                mass=float(mass),
                source_id=int(event_index),
            ))
            if not vertices:
                raise RuntimeError(
                    f"staged_r20_unit_grid: state {record.name} (pdg "
                    f"{record.pdg_id}) at grid point j = {j} "
                    f"(M = {mass!r} GeV) produced no vertices; a "
                    "non-stable record must decay"
                )
            root = vertices[0]
            key = _channel_key(root.charge_combination)
            counts[key] = counts.get(key, 0) + 1
            # Deterministic per-vertex statistics over the kernel's
            # queue-order vertex list; a resonance daughter's realized
            # mass is the parent mass of its own child vertex (the
            # child mapping below is asserted).
            child_ptr = 1
            n_vertices = len(vertices)
            for vertex in vertices:
                pe, px, py, pz = (
                    float(x) for x in vertex.parent_four_momentum
                )
                for k, parent_comp in enumerate((pe, px, py, pz)):
                    total_k = 0.0
                    for daughter in vertex.daughters:
                        total_k += float(daughter.four_momentum[k])
                    worst = abs(total_k - parent_comp)
                    if worst > max_closure:
                        max_closure = worst
                for daughter in vertex.daughters:
                    de, dpx, dpy, dpz = (
                        float(x) for x in daughter.four_momentum
                    )
                    m2 = de * de - dpx * dpx - dpy * dpy - dpz * dpz
                    if daughter.is_resonance:
                        if child_ptr >= n_vertices:
                            raise RuntimeError(
                                "staged_r20_unit_grid: the vertex list "
                                f"of {record.name} (pdg {record.pdg_id}) "
                                "at grid point j = "
                                f"{j} has more resonance daughters than "
                                "child vertices (queue-order "
                                "reconstruction failed)"
                            )
                        child = vertices[child_ptr]
                        child_ptr += 1
                        if int(child.parent_pdg) != int(daughter.pdg_id):
                            raise RuntimeError(
                                "staged_r20_unit_grid: the child vertex "
                                f"of daughter pdg {daughter.pdg_id} does "
                                "not match the queue order (expected "
                                f"pdg {int(daughter.pdg_id)}, found "
                                f"{int(child.parent_pdg)})"
                            )
                        mass_ref = float(child.parent_mass)
                    else:
                        mass_ref = _stable_reference_mass(
                            catalog, int(daughter.pdg_id)
                        )
                    dev2 = abs(m2 - mass_ref * mass_ref)
                    dev = abs(math.sqrt(m2) - mass_ref)
                    if dev2 > max_os2:
                        max_os2 = dev2
                    if dev > max_os:
                        max_os = dev
                if (len(vertex.daughters) == 2
                        and not vertex.channel.is_three_body):
                    n_two += 1
                    beta = (px / pe, py / pe, pz / pe)
                    rest = _boost_lab(
                        vertex.daughters[0].four_momentum,
                        (-beta[0], -beta[1], -beta[2]),
                    )
                    p_rest = math.sqrt(
                        rest[1] * rest[1]
                        + rest[2] * rest[2]
                        + rest[3] * rest[3]
                    )
                    if not p_rest > 0.0:
                        raise RuntimeError(
                            "staged_r20_unit_grid: a two-body daughter "
                            f"of {record.name} (pdg {record.pdg_id}) at "
                            f"grid point j = {j} is at rest in the "
                            "parent rest frame (the sampled channel is "
                            "not open); the run stops"
                        )
                    cos_theta = rest[3] / p_rest
                    index = int((cos_theta + 1.0) * 0.5 * ANGULAR_BINS)
                    index = min(max(index, 0), ANGULAR_BINS - 1)
                    hist[index] += 1
                elif (len(vertex.daughters) == 3
                        and vertex.channel.is_three_body):
                    n_three += 1
                else:
                    raise RuntimeError(
                        "staged_r20_unit_grid: a vertex of "
                        f"{record.name} (pdg {record.pdg_id}) at grid "
                        f"point j = {j} carries "
                        f"{len(vertex.daughters)} daughters with "
                        f"is_three_body = "
                        f"{vertex.channel.is_three_body}; the vertex "
                        "form is not recognizable"
                    )
            if child_ptr != n_vertices:
                raise RuntimeError(
                    "staged_r20_unit_grid: the queue-order child "
                    f"reconstruction of {record.name} (pdg "
                    f"{record.pdg_id}) at grid point j = {j} consumed "
                    f"{child_ptr} of {n_vertices} vertices"
                )
            state_max_closure = max(state_max_closure, max_closure)
            state_max_os2 = max(state_max_os2, max_os2)
            state_max_os = max(state_max_os, max_os)
            state_vertices += n_vertices
            state_two += n_two
            state_three += n_three
        points[str(j)] = {
            "j": j,
            "mass_gev": mass,
            "degenerate": False,
            "n_decays": n_decays,
            "channel_counts": counts,
            "n_two_body_vertices": n_two,
            "n_three_body_vertices": n_three,
            "cos_theta_hist": hist,
            "max_four_momentum_closure_deviation_gev": max_closure,
            "max_on_shell_deviation_gev2": max_os2,
            "max_on_shell_deviation_gev": max_os,
        }
        total_decays += n_decays
        print(
            f"staged_r20_unit_grid: {record.name} (pdg {record.pdg_id}) "
            f"point {j + 1}/{GRID_POINTS} done"
        )
    payload = {
        "schema": "na49_mixing.staged_r20_unit_grid.state/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "state": {
            "pdg_id": int(record.pdg_id),
            "name": str(record.name),
            "rank": rank,
            "classification": row["classification"],
            "decays_per_point": n_decays,
            "B": int(record.B),
            "Q": int(record.Q),
            "S": int(record.S),
            "pole_mass": float(record.pole_mass),
            "nominal_width": float(record.nominal_width),
            "mass_range": [
                float(record.mass_range[0]), float(record.mass_range[1]),
            ],
            "channels": _state_channel_rows(record),
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "replica_index": REPLICA_INDEX,
            "event_index_rule": SEED_RULE,
            "substream_labels": list(SEED_LABELS),
        },
        "grid": {
            "points": GRID_POINTS,
            "formula": GRID_FORMULA,
            "masses_gev": ctx.grid,
        },
        "degenerate_points": [
            {
                "j": j,
                "mass_gev": ctx.grid[j],
                "reason": DEGENERATE_SKIP_REASON,
            }
            for j in degenerate
        ],
        "points": points,
        "deterministic": {
            "max_four_momentum_closure_deviation_gev": state_max_closure,
            "max_on_shell_deviation_gev2": state_max_os2,
            "max_on_shell_deviation_gev": state_max_os,
            "n_vertices_checked": state_vertices,
            "n_two_body_vertices": state_two,
            "n_three_body_vertices": state_three,
            "tolerance": DETERMINISTIC_TOL,
        },
        "totals": {
            "n_decays": total_decays,
            "n_open_points": GRID_POINTS - len(degenerate),
            "n_degenerate_points": len(degenerate),
        },
        "written_utc": _now_utc(),
    }
    path = run_dir / f"state_{int(record.pdg_id)}.json"
    _write_json(path, payload)
    return path


def _run_generation(run_dir: Path, state_pdg: int | None) -> None:
    catalog = _load_catalog()
    pre = _load_pre_registration(run_dir)
    rows = _assert_rows_against_pre_registration(catalog, pre)
    if state_pdg is not None:
        selected = [
            row for row in rows
            if int(row["record"].pdg_id) == int(state_pdg)
        ]
        if not selected:
            raise SystemExit(
                f"staged_r20_unit_grid: --state {state_pdg} is not a "
                "pre-registered decayable catalog state "
                "(stable_for_decay = false on the pre-registered lists)"
            )
        rows = selected
    run_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        record = row["record"]
        path = _generate_state(catalog, run_dir, row)
        print(
            "staged_r20_unit_grid: wrote "
            f"{path} ({record.name}, pdg {record.pdg_id}, "
            f"{row['classification']}, "
            f"{row['decays_per_point']} decays per point)"
        )


# ---------------------------------------------------------------------------
# Mode: --verify (and the clean-interpreter probe's re-run)
# ---------------------------------------------------------------------------
def _se2(p: float, n: int) -> float:
    """Squared binomial standard error of a fraction at *n* trials."""
    if n <= 0:
        return 0.0
    return p * (1.0 - p) / n


def _zscore(frac: float, prob: float, n: int) -> float:
    """``|frac - prob|`` in units of the binomial 1/sqrt(N) standard
    error of the expected fraction; 0.0 when both agree exactly and
    the error is 0 (a deterministic outcome), infinity on a deviation
    against a zero-error expectation."""
    se = math.sqrt(_se2(prob, n))
    dev = abs(frac - prob)
    if not se > 0.0:
        return 0.0 if dev <= 0.0 else math.inf
    return dev / se


def _zdiff(diff: float, se: float) -> float:
    """``|diff|`` in units of the (combined) standard error *se*."""
    if not se > 0.0:
        return 0.0 if abs(diff) <= 0.0 else math.inf
    return abs(diff) / se


def _pole_point(ctx: _StateContext) -> int:
    """The open grid point closest to the record pole mass (ties to
    the lowest j)."""
    pole = float(ctx.record.pole_mass)
    best_j = None
    best_d = None
    for j in range(GRID_POINTS):
        if j in ctx.degenerate:
            continue
        d = abs(ctx.grid[j] - pole)
        if best_d is None or d < best_d:
            best_j, best_d = j, d
    if best_j is None:
        raise RuntimeError(
            f"staged_r20_unit_grid: {ctx.record.name} (pdg "
            f"{ctx.record.pdg_id}) has no open grid point; the grid "
            "convention is violated"
        )
    return int(best_j)


def _verify_state(
    catalog: HadronCatalog, row: dict, pre_by_pdg: dict, run_dir: Path
) -> dict:
    """Verify one state over its generated ``state_<pdg>.json`` and
    return the per-state aggregate entry (all numbers are plain
    floats/ints; no wall-clock)."""
    record = row["record"]
    pdg = int(record.pdg_id)
    path = run_dir / f"state_{pdg}.json"
    if not path.is_file():
        raise RuntimeError(
            f"staged_r20_unit_grid: the state file {path} is missing; "
            "the verification requires every pre-registered state"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ctx = _StateContext(catalog, record)

    # -- integrity against the pre-registration and this driver --------
    state_block = payload.get("state") or {}
    pre_entry = pre_by_pdg[pdg]
    if int(state_block.get("pdg_id", -1)) != pdg:
        raise RuntimeError(f"staged_r20_unit_grid: {path}: pdg mismatch")
    if int(state_block.get("rank", -1)) != row["rank"]:
        raise RuntimeError(f"staged_r20_unit_grid: {path}: rank mismatch")
    if str(state_block.get("classification")) != row["classification"]:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: classification mismatch"
        )
    if (int(state_block.get("decays_per_point", -1))
            != row["decays_per_point"]):
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: per-point count mismatch"
        )
    seeds = payload.get("seed_map") or {}
    if int(seeds.get("master_seed", -1)) != MASTER_SEED:
        raise RuntimeError(f"staged_r20_unit_grid: {path}: seed mismatch")
    if int(seeds.get("replica_index", -2)) != REPLICA_INDEX:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: replica index mismatch"
        )
    if str(seeds.get("event_index_rule")) != SEED_RULE:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: event_index rule mismatch"
        )
    grid_block = payload.get("grid") or {}
    if [float(x) for x in grid_block.get("masses_gev", [])] != ctx.grid:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: the grid masses do not "
            "match the pre-registered grid"
        )
    file_degenerate = sorted(
        int(entry["j"]) for entry in payload.get("degenerate_points", [])
    )
    if file_degenerate != sorted(ctx.degenerate):
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: the degenerate points do "
            "not match the pre-registered inventory"
        )
    points = payload.get("points") or {}
    if sorted(int(k) for k in points) != list(range(GRID_POINTS)):
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: the per-point records do "
            "not cover the grid"
        )

    pole_j = _pole_point(ctx)
    open_js = [j for j in range(GRID_POINTS) if j not in ctx.degenerate]
    if len(open_js) < 2:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: fewer than two open points; "
            "the threshold-continuity check is undefined"
        )

    branch_pole = 0.0
    branch_open = 0.0
    charge_max = 0.0
    continuity = 0.0
    pooled_hist = [0] * ANGULAR_BINS
    pooled_two = 0
    max_closure = 0.0
    max_os2 = 0.0
    max_os = 0.0
    n_decays_total = 0
    first = None
    second = None
    for j in open_js:
        point = points[str(j)]
        if bool(point.get("degenerate", False)):
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} is recorded "
                "degenerate but the pre-registered inventory has it open"
            )
        n = int(point["n_decays"])
        if n != row["decays_per_point"]:
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} carries "
                f"{n} decays against the pre-registered "
                f"{row['decays_per_point']}"
            )
        counts = {
            str(k): int(v) for k, v in point["channel_counts"].items()
        }
        if sum(counts.values()) != n:
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} channel "
                "counts do not sum to the decay count"
            )
        weights, class_shares = _expected_point(ctx, ctx.grid[j])
        w_total = math.fsum(weights.values())
        if not w_total > 0.0:
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} has no open "
                "weight (it should have been a degenerate point)"
            )
        unexplained = sorted(set(counts) - set(weights))
        if unexplained:
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} sampled the "
                f"channel(s) {unexplained} outside the expected open "
                "set (kernel/mirror divergence)"
            )
        for key, w in weights.items():
            z = _zscore(counts.get(key, 0) / n, w / w_total, n)
            if z > branch_open:
                branch_open = z
            if j == pole_j and z > branch_pole:
                branch_pole = z
        for shares in class_shares.values():
            n_cls = sum(counts.get(key, 0) for key in shares)
            if n_cls <= 0:
                continue
            for key, (_pair_key, share) in shares.items():
                z = _zscore(counts.get(key, 0) / n_cls, share, n_cls)
                if z > charge_max:
                    charge_max = z
        hist = [int(x) for x in point["cos_theta_hist"]]
        n_two = int(point["n_two_body_vertices"])
        n_three = int(point["n_three_body_vertices"])
        if len(hist) != ANGULAR_BINS or sum(hist) != n_two:
            raise RuntimeError(
                f"staged_r20_unit_grid: {path}: point {j} angular "
                "histogram is inconsistent"
            )
        for b in range(ANGULAR_BINS):
            pooled_hist[b] += hist[b]
        pooled_two += n_two
        max_closure = max(
            max_closure,
            float(point["max_four_momentum_closure_deviation_gev"]),
        )
        max_os2 = max(max_os2, float(point["max_on_shell_deviation_gev2"]))
        max_os = max(max_os, float(point["max_on_shell_deviation_gev"]))
        n_decays_total += n
        if first is None:
            first = (counts, weights, class_shares)
        elif second is None:
            second = (counts, weights, class_shares)

    # -- angular kernel (pooled over the open points) ------------------
    angular_max = 0.0
    if pooled_two > 0:
        for b in range(ANGULAR_BINS):
            z = _zscore(
                pooled_hist[b] / pooled_two, 1.0 / ANGULAR_BINS, pooled_two
            )
            if z > angular_max:
                angular_max = z

    # -- threshold continuity at the two lowest open points ------------
    if first is None or second is None:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: fewer than two open points "
            "were aggregated (the threshold-continuity check is "
            "undefined)"
        )
    c1, w1, s1 = first
    c2, w2, s2 = second
    n1 = n2 = row["decays_per_point"]
    t1 = math.fsum(w1.values())
    t2 = math.fsum(w2.values())
    for key, w in w1.items():
        if key not in w2:
            continue
        p1 = w / t1
        p2 = w2[key] / t2
        se = math.sqrt(_se2(p1, n1) + _se2(p2, n2))
        z = _zdiff((c1.get(key, 0) / n1 - c2.get(key, 0) / n2)
                   - (p1 - p2), se)
        if z > continuity:
            continuity = z
    for cls, shares1 in s1.items():
        shares2 = s2.get(cls)
        if shares2 is None:
            continue
        n_cls1 = sum(c1.get(k, 0) for k in shares1)
        n_cls2 = sum(c2.get(k, 0) for k in shares2)
        if n_cls1 <= 0 or n_cls2 <= 0:
            continue
        for key, (_pair_key, sh1) in shares1.items():
            if key not in shares2:
                continue
            sh2 = shares2[key][1]
            se = math.sqrt(_se2(sh1, n_cls1) + _se2(sh2, n_cls2))
            z = _zdiff((c1.get(key, 0) / n_cls1 - c2.get(key, 0) / n_cls2)
                       - (sh1 - sh2), se)
            if z > continuity:
                continuity = z

    # -- deterministic maxima cross-check (state level == point max) ---
    det = payload.get("deterministic") or {}
    state_closure = float(det["max_four_momentum_closure_deviation_gev"])
    state_os2 = float(det["max_on_shell_deviation_gev2"])
    state_os = float(det["max_on_shell_deviation_gev"])
    if (state_closure != max_closure or state_os2 != max_os2
            or state_os != max_os):
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: the state-level deterministic "
            "maxima do not match the maxima over the per-point records"
        )
    totals = payload.get("totals") or {}
    if int(totals.get("n_decays", -1)) != n_decays_total:
        raise RuntimeError(
            f"staged_r20_unit_grid: {path}: the totals do not match the "
            "per-point decay counts"
        )

    branch_all = max(branch_pole, branch_open)
    checks_pass = (
        branch_pole <= N_SIGMA
        and branch_open <= N_SIGMA
        and charge_max <= N_SIGMA
        and angular_max <= N_SIGMA
        and continuity <= N_SIGMA
        and max_closure <= DETERMINISTIC_TOL
        and max_os2 <= DETERMINISTIC_TOL
        and max_os <= DETERMINISTIC_TOL
    )
    return {
        "name": str(record.name),
        "rank": row["rank"],
        "classification": row["classification"],
        "pole_point": pole_j,
        "n_open_points": len(open_js),
        "n_degenerate_points": len(ctx.degenerate),
        "n_decays": n_decays_total,
        "branching_pole_max_abs_sigma": branch_pole,
        "branching_open_max_abs_sigma": branch_open,
        "charge_max_abs_sigma": charge_max,
        "angular_max_abs_sigma": angular_max,
        "threshold_continuity_max_abs_sigma": continuity,
        "degenerate_points": len(ctx.degenerate),
        "max_four_momentum_closure_deviation_gev": max_closure,
        "max_on_shell_deviation_gev2": max_os2,
        "max_on_shell_deviation_gev": max_os,
        "pass": bool(checks_pass),
    }


def compute_verification_payloads(run_dir: Path) -> tuple[dict, dict]:
    """Run the verification over the generated state files and return
    the ``(aggregate, gate2)`` payloads without writing anything (the
    clean-interpreter probe re-uses this entry point; the payloads
    carry no wall-clock fields, so equal inputs hash equal)."""
    run_dir = Path(run_dir)
    catalog = _load_catalog()
    pre = _load_pre_registration(run_dir)
    rows = _state_rows(catalog)
    pre_by_pdg = {
        int(entry["pdg_id"]): entry
        for entry in pre["state_coverage"]["states"]
    }
    missing = [
        str(run_dir / f"state_{int(row['record'].pdg_id)}.json")
        for row in rows
        if not (run_dir / f"state_{int(row['record'].pdg_id)}.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            "staged_r20_unit_grid: the verification requires every "
            "pre-registered state; missing: " + ", ".join(missing)
        )
    per_state = {
        str(int(row["record"].pdg_id)): _verify_state(
            catalog, row, pre_by_pdg, run_dir
        )
        for row in rows
    }

    gate_per_state = {}
    for str_pdg, entry in per_state.items():
        branch_all = max(
            entry["branching_pole_max_abs_sigma"],
            entry["branching_open_max_abs_sigma"],
        )
        gate_per_state[str_pdg] = {
            "branching_max_abs_sigma": branch_all,
            "charge_max_abs_sigma": entry["charge_max_abs_sigma"],
            "angular_max_abs_sigma": entry["angular_max_abs_sigma"],
            "degenerate_points": entry["degenerate_points"],
            "pass": bool(
                branch_all <= N_SIGMA
                and entry["charge_max_abs_sigma"] <= N_SIGMA
                and entry["angular_max_abs_sigma"] <= N_SIGMA
            ),
        }
    max_closure = max(
        entry["max_four_momentum_closure_deviation_gev"]
        for entry in per_state.values()
    )
    max_os2 = max(
        entry["max_on_shell_deviation_gev2"]
        for entry in per_state.values()
    )
    max_os = max(
        entry["max_on_shell_deviation_gev"] for entry in per_state.values()
    )
    max_continuity = max(
        entry["threshold_continuity_max_abs_sigma"]
        for entry in per_state.values()
    )
    aggregate_pass = all(entry["pass"] for entry in per_state.values())
    aggregate = {
        "schema": "na49_mixing.staged_r20_unit_grid.aggregate/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "pre_registration_sha256": _sha256_file(
            run_dir / "pre_registration.json"
        ),
        "provenance_sha256": _sha256_file(run_dir / "provenance.json"),
        "tolerances": {
            "n_sigma": N_SIGMA,
            "deterministic_tol": DETERMINISTIC_TOL,
            "angular_bins": ANGULAR_BINS,
        },
        "notes": {
            "pole_expected": POLE_EXPECTED_NOTE,
            "continuity": CONTINUITY_NOTE,
        },
        "per_state": per_state,
        "totals": {
            "n_states": len(per_state),
            "n_states_pass": sum(
                1 for entry in per_state.values() if entry["pass"]
            ),
            "n_decays": sum(
                int(entry["n_decays"]) for entry in per_state.values()
            ),
            "max_four_momentum_closure_deviation_gev": max_closure,
            "max_on_shell_deviation_gev2": max_os2,
            "max_on_shell_deviation_gev": max_os,
            "branching_max_abs_sigma": max(
                max(entry["branching_pole_max_abs_sigma"],
                    entry["branching_open_max_abs_sigma"])
                for entry in per_state.values()
            ),
            "charge_max_abs_sigma": max(
                entry["charge_max_abs_sigma"]
                for entry in per_state.values()
            ),
            "angular_max_abs_sigma": max(
                entry["angular_max_abs_sigma"]
                for entry in per_state.values()
            ),
            "threshold_continuity_max_abs_sigma": max_continuity,
        },
        "pass": bool(aggregate_pass),
    }
    branching_pass = all(
        entry["pass"] for entry in gate_per_state.values()
    )
    local_pass = (
        max_closure <= DETERMINISTIC_TOL
        and max_os <= DETERMINISTIC_TOL
        and max_os2 <= DETERMINISTIC_TOL
    )
    gate = {
        "gate": "gate 2 (plan section 8, line 575)",
        "run": RUN_ID,
        "clauses": {
            "local_conservation": {
                "max_four_momentum_closure_deviation_gev": max_closure,
                "max_on_shell_deviation_gev": max_os,
                "max_on_shell_deviation_gev2": max_os2,
                "pass": bool(local_pass),
            },
            "branching_test": {
                "per_state": gate_per_state,
                "pass": bool(branching_pass),
            },
            "threshold_discontinuity": {
                "two_lowest_points_max_abs_sigma": max_continuity,
                "pass": bool(max_continuity <= N_SIGMA),
            },
            "observables": None,
            "no_branching_fit": {"no_fitting": True},
        },
        "aggregate_sha256": hashlib.sha256(
            _canonical_json_bytes(aggregate)
        ).hexdigest(),
        "sealed": True,
        "note": "sealed marks the gate evidence as the sealed form: the "
                "step-5 seal covers this file byte-for-byte (written "
                "before sealing, never edited after)",
    }
    return aggregate, gate


def _run_verify(run_dir: Path) -> None:
    aggregate, gate = compute_verification_payloads(run_dir)
    _write_json(run_dir / "aggregate.json", aggregate)
    _write_json(run_dir / "gate2_r20.json", gate)
    totals = aggregate["totals"]
    print(
        "staged_r20_unit_grid: verification written to "
        f"{run_dir / 'aggregate.json'} and {run_dir / 'gate2_r20.json'} "
        f"({totals['n_states']} states, {totals['n_states_pass']} pass, "
        f"branching max |sigma| "
        f"{totals['branching_max_abs_sigma']:.3f}, charge "
        f"{totals['charge_max_abs_sigma']:.3f}, angular "
        f"{totals['angular_max_abs_sigma']:.3f}, continuity "
        f"{totals['threshold_continuity_max_abs_sigma']:.3f}, closure "
        f"{totals['max_four_momentum_closure_deviation_gev']:.3e} GeV, "
        f"pass = {aggregate['pass']})"
    )


# ---------------------------------------------------------------------------
# Mode: --seal
# ---------------------------------------------------------------------------
def _run_seal(run_dir: Path) -> None:
    for required in (
        "pre_registration.json",
        "provenance.json",
        "aggregate.json",
        "gate2_r20.json",
    ):
        if not (run_dir / required).is_file():
            raise RuntimeError(
                f"staged_r20_unit_grid: {run_dir / required} is missing; "
                "the seal requires the full evidence chain (the run "
                "procedure steps 1-4)"
            )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if sums_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            f"staged_r20_unit_grid: {run_dir} is already sealed; the "
            "run dir is immutable after the seal"
        )
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel in ("seal_manifest.json", "SHA256SUMS.txt"):
            continue
        entries.append((rel, _sha256_file(path)))
        total_bytes += path.stat().st_size
    with open(sums_path, "w", encoding="utf-8", newline="\n") as handle:
        for rel, digest in entries:
            handle.write(f"{digest}  {rel}\n")
    manifest = {
        "schema": "na49_mixing.staged_r20_unit_grid.seal_manifest/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256_sums_file": "SHA256SUMS.txt",
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": ["seal_manifest.json", "SHA256SUMS.txt"],
        "note": "the run dir is immutable after the seal (the Round 1 "
                "sealing discipline); clean_environment_commands.json "
                "is the step-6 post-seal validation record, written "
                "after this manifest and never edited",
    }
    _write_json(manifest_path, manifest)
    print(
        "staged_r20_unit_grid: sealed "
        f"{run_dir} ({len(entries)} files, {total_bytes} bytes, "
        f"SHA256SUMS.txt sha256 {manifest['sha256_sums_sha256']})"
    )


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------
#: The clean-interpreter probe: imports this driver module fresh, runs
#: the verification computation over the sealed run dir, and prints the
#: canonical report hashes (it writes nothing).
_CLEAN_PROBE_SOURCE = """\
import hashlib
import importlib.util
import json
import sys

driver_path, run_dir = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "staged_r20_unit_grid_clean_probe", driver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
aggregate, gate = module.compute_verification_payloads(run_dir)


def _canon(payload):
    return (json.dumps(payload, indent=2) + "\\n").encode("utf-8")


print(json.dumps({
    "aggregate_sha256":
        hashlib.sha256(_canon(aggregate)).hexdigest(),
    "gate2_sha256":
        hashlib.sha256(_canon(gate)).hexdigest(),
    "environment": module.environment_fingerprint(),
}))
"""


def _run_clean_validate(run_dir: Path) -> None:
    out_path = run_dir / "clean_environment_commands.json"
    if out_path.is_file():
        raise RuntimeError(
            f"staged_r20_unit_grid: {out_path} exists; the "
            "clean-environment validation is recorded once per seal"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"staged_r20_unit_grid: {run_dir} is not sealed; run --seal "
            "before the clean-environment validation"
        )
    checked = 0
    mismatches: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = run_dir / rel
        if not path.is_file() or _sha256_file(path) != digest:
            mismatches.append(rel)
        checked += 1
    if mismatches:
        raise RuntimeError(
            "staged_r20_unit_grid: the seal verification failed for: "
            + ", ".join(mismatches)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "staged_r20_unit_grid: the seal manifest's SHA256SUMS.txt "
            "digest does not match the sums file"
        )
    sealed = {
        "aggregate.json": _sha256_file(run_dir / "aggregate.json"),
        "gate2_r20.json": _sha256_file(run_dir / "gate2_r20.json"),
    }
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-c", _CLEAN_PROBE_SOURCE,
        str(driver_path), str(run_dir),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "staged_r20_unit_grid: the clean-interpreter probe failed:\n"
            + completed.stderr
        )
    probe_line = completed.stdout.strip().splitlines()[-1]
    probe = json.loads(probe_line)
    recomputed = {
        "aggregate.json": str(probe["aggregate_sha256"]),
        "gate2_r20.json": str(probe["gate2_sha256"]),
    }
    hashes_match = sealed == recomputed
    payload = {
        "schema":
            "na49_mixing.staged_r20_unit_grid.clean_environment/1",
        "run": RUN_ID,
        "card": CARD_ID,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": checked,
            "mismatches": [],
        },
        "commands": [
            {
                "purpose": "clean-interpreter re-run of the "
                           "verification over the sealed artifacts "
                           "(computes the report payloads and prints "
                           "their hashes; writes nothing)",
                "argv": argv,
                "cwd": str(Path.cwd()),
            }
        ],
        "environment": environment_fingerprint(),
        "probe_environment": probe.get("environment", {}),
        "sealed_report_sha256": sealed,
        "recomputed_report_sha256": recomputed,
        "hashes_match": bool(hashes_match),
    }
    _write_json(out_path, payload)
    if not hashes_match:
        raise RuntimeError(
            "staged_r20_unit_grid: the clean-interpreter re-run does "
            "not reproduce the sealed report hashes (sealed "
            f"{sealed}, recomputed {recomputed}); the record is in "
            f"{out_path}"
        )
    print(
        "staged_r20_unit_grid: clean-environment validation passed "
        f"({checked} sealed files verified; the re-run from a clean "
        "interpreter reproduces the sealed aggregate.json and "
        f"gate2_r20.json hashes); record at {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r20_unit_grid.py",
        description=(
            "Run R20, the single-resonance unit grid (Round 2, "
            "card 10): every catalog state, 100 fixed-mass points, "
            "10,000/2,000 decays per point, the branching / charge / "
            "mass-shell / angular verification, the seal, and the "
            "clean-environment validation.  Each invocation performs "
            "exactly one step and stops."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json into the "
             "run dir, then stop (nothing is generated)",
    )
    group.add_argument(
        "--generate", action="store_true",
        help="generate the pre-registered decays (all states, or one "
             "with --state), writing state_<pdg>.json, then stop",
    )
    group.add_argument(
        "--verify", action="store_true",
        help="run the verification over the state files, write "
             "aggregate.json and gate2_r20.json, then stop",
    )
    group.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt over the run "
             "dir, then stop (the run dir is immutable after the seal)",
    )
    group.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal, re-run the verification from a clean "
             "interpreter, and confirm the report hashes match; write "
             "clean_environment_commands.json, then stop",
    )
    parser.add_argument(
        "--state", type=int, default=None,
        help="generate only this PDG id (together with --generate)",
    )
    parser.add_argument(
        "--run-dir", default=DEFAULT_RUN_DIR,
        help=f"the run directory (default: {DEFAULT_RUN_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.state is not None and not args.generate:
        raise SystemExit(
            "staged_r20_unit_grid: --state is only valid together with "
            "--generate"
        )
    run_dir = Path(args.run_dir)
    if args.pre_registration:
        _run_pre_registration(run_dir)
    elif args.generate:
        _run_generation(run_dir, args.state)
    elif args.verify:
        _run_verify(run_dir)
    elif args.seal:
        _run_seal(run_dir)
    else:
        _run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
