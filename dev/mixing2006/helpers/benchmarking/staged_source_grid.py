#!/usr/bin/env python3
"""Staged source-grid driver for the Round 2 resonance network (card 9).

Task ``TASK-20260913-r2-source-grid-runner`` (plan section 22, task 17;
the run definitions R21 and R22 of plan section 8, lines 559-565; the
round 2 index items 6, 9 and 10).  The driver generates the isolated
fixed-quantum-number source grids of runs R21 and R22 through the
card-6/7 decay kernel (``na49_mixing.resonance_decay``) with the
pre-registered grids, statistics, seeds, source mixture model, and
per-point output schema.  The runs themselves (the pre-registration
assertion against the run cards, generation at scale, sealing, gate
JSONs) are cards 11 and 12; this driver writes no gate JSON and no
seal files.

Pre-registered grids (round 2 index item 6; the driver enforces them):

- Baryonic (R21): source ``(B, Q, S) = (1, 1, 0)``, at rest; ``M_k =
  1.07324891 + 0.1 k`` for ``k = 0..19`` (20 points, 1.07324891 to
  2.97324891 GeV).
- Mesonic (R22): source ``(B, Q, S) = (0, 0, 0)``, at rest; ``M_k =
  0.2699532 + 0.1 k`` for ``k = 0..27`` (28 points, 0.2699532 to
  2.9699532 GeV).

Statistics (T1, plan section 5.1): 4 replicas x 25,000 decays per grid
point.  Seeds: master seed 20260913; per (point ``k``, replica ``r``,
decay ``i < 25,000``): ``derive_substreams(20260913, r, k * 1_000_000 +
i)`` with the closed labels ``decay_channel`` / ``decay_kinematics``
(no extension of ``SUBSTREAM_LABELS``).  The angular kernels are the
spin-averaged (unpolarized) kernels of round 2 index item 9.

Source mixture model (the card's pre-registered flat record mixture):

1. The open set at point ``k`` is the set of catalog records with the
   source's ``(B, Q, S)`` whose truncated spectral window
   ``[mass_range[0], min(mass_range[1], M_k)]`` is non-empty.  The
   driver asserts the open set is non-empty at every grid point; a
   violation raises and stops the run (no fallback decay path is
   implemented).
2. A record is drawn uniformly over the open set (equal record weights;
   the isospin charge weights of card 5 act inside each record, so
   "equal weights" applies to records, never to charge states).
3. The record's mass is drawn from its card-3 spectral function
   restricted to the window (``resonance_widths.draw_mass``); a
   degenerate window (width <= 1e-12) fixes the mass at ``M_k``.
4. The record decays through the card-6/7 kernel; the source
   four-momentum is the at-rest four-vector of the realized mass.

Committed-catalog reading of the mixture model (documented scan
conventions; the full rationale lives in the run record):

- The pre-registered grid starts are the class-minimum channel
  thresholds of the two source quantum numbers (``m_p + m_pi0`` and
  ``2 m_pi0``).  On the committed catalog the first resonance of each
  class opens at its fully-open threshold ``mass_range[0]`` (Delta+
  1.07913531 GeV, rho0 0.2791398 GeV), slightly above the grid start;
  the first resonances join the open set at point ``k = 1`` of both
  grids.  The open set is nonetheless non-empty at every grid point
  through the stable termination records that carry the source quantum
  numbers (the proton at ``(1, 1, 0)``, ``pi0`` at ``(0, 0, 0)``): a
  stable record's support is a single point (the pole mass), so its
  truncated window ``[pole mass, min(pole mass, M_k)]`` is degenerate
  but non-empty at every grid mass above the pole.  The card's
  non-emptiness assert holds on every point of both grids.
- A drawn record that is already in the stable set has no decay (the
  kernel's ``decay_source`` returns none; plan section 8.4 step 7 stops
  when all remaining particles are configured stable states, and the
  source itself is one).  Such an event contributes the source itself
  as the single final particle at the realized four-momentum, with no
  producing vertex: a direct source product, never a resonance
  feed-down.  The flat mixture therefore carries the stable identity of
  the source class at the open-set weight -- a scan convention, not a
  physics claim (the flat mixture is what the Gate 2
  threshold-discontinuity clause is measured against).
- The degenerate-window rule fixes the record's mass at the grid point
  ``M_k`` for every degenerate window (the card's rule; the card-3
  spectral draw is undefined there).  The stable records' windows are
  degenerate at every grid point, so their identity events carry the
  at-rest four-momentum ``(M_k, 0, 0, 0)`` of the card's item 4
  exactly: the source is realized at the grid mass and does not decay
  there (no open channel exists at the grid start, where the width law
  is the exact hard zero at the class threshold).

Draw budget per decay: the mixture layer draws the record with one
uniform and realizes its mass with the card-3 ``draw_mass`` (two
uniforms per non-degenerate draw, none for a degenerate window); the
kernel's vertex draws re-derive the two labeled substream ids from the
same ``(master_seed, replica, event_index)`` triple (the card-6 decayer
serves one decay).  Both layers draw from the two closed labels
``decay_channel`` / ``decay_kinematics`` (no extension of
``SUBSTREAM_LABELS``): the mixture layer consumes one-step spawned
children of the labeled stream ids (``SeedSequence(stream_id).spawn(1)``
-- the record draw from the ``decay_channel`` child, the card-3
``draw_mass`` from the ``decay_kinematics`` child), while the kernel
keeps the labeled ids themselves.  Sharing the labeled ids directly
between the two layers would make the record-draw uniform and the
kernel's first vertex-channel uniform the same value and would correlate
the record selection with the root vertex's channel selection (a
measurable within-class charge-split bias); the spawned-child split
keeps the layers on disjoint uniform sequences under the same labeled
anchors and is deterministic and reproducible.

Per-point output schema (one ``point_<k:02d>.json`` per point, plus one
``replica_<r>_point_<k:02d>.json`` per replica/point with the raw
per-replica diagnostics)::

    {
      "point": <k>,
      "source_mass_gev": <M_k>,
      "source_qn": [B, Q, S],
      "open_records": [<pdg ids in catalog record order>],
      "n_decays": <replicas * decays_per_replica>,
      "multiplicities": {"<stable species>": {"mean": ..., "se": ...}, ...},
      "momentum_moments": {"<species>": {"mean_p_gev": ...,
                          "mean_pt_y_gev": ..., "mean_e_gev": ...}, ...},
      "feed_down": {"<species>": {"fraction_resonant": ...,
                    "by_parent": {"<resonance pdg>": ...}}, ...},
      "charge_balance": {"total_Q_exact": <Q>, "total_B_exact": <B>,
                          "events_checked": <n>},
      "pairs": {"pi_plus_pi_minus_per_event": {"mean": ..., "se": ...},
                 "p_pbar_per_event": {...}, "n_nbar_per_event": {...}}
    }

The multiplicity species set is the catalog's stable termination set
(every ``catalog.stable_states()`` member, in catalog record order; all
listed, zero where absent).  ``mean_pt_y_gev`` is the density-``p_T f``
convention of the house terminology (``<p_T>_y``): the energy-weighted
track mean ``sum(E * pT) / sum(E)`` over the species' particles.
``mean_p_gev`` and ``mean_e_gev`` are the unweighted track means.  The
``pairs`` block counts the charge-conjugate co-occurrence per event
(``min(n, nbar)`` per event, mean and standard error over the point's
events).  Species keys transliterate the record name's charge symbol
(``+`` -> ``_plus``, ``-`` -> ``_minus``, a trailing ``0`` -> ``_0``),
so ``pi+`` -> ``pi_plus``, ``pi0`` -> ``pi_0``, ``proton`` verbatim.

Per-event ``(B, Q, S)`` balance against the source is asserted on every
event (a check, never a repair); the card-6/7 kernel's own per-vertex
conservation, closure, and on-shell assertions propagate unchanged and
stop the run.

``pre_registration.json`` (the grids, the mixture model, the seed map,
the statistics, the output schema, and the per-point open sets) and
``provenance.json`` (the catalog file with its SHA-256, the card chain)
are written before any generation.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: standard library plus the vendored ``na49_mixing`` modules
only; no event generation at scale, no external tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
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
            "staged_source_grid: cannot locate the na49_mixing package "
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

from na49_mixing.excitation_kinematics import FourVector  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog, HadronRecord  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.quantum_numbers import quantum_numbers  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer, SourceRecord  # noqa: E402
from na49_mixing.resonance_widths import draw_mass  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.source_event import FinalParticle  # noqa: E402
from na49_mixing.threshold_metric import (  # noqa: E402
    SIGMA_TOLERANCE,
    THRESHOLD_METRIC_ID,
    RecordSet,
    full_analytic_difference,
)

CARD_ID = "TASK-20260913-r2-source-grid-runner"
CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
DRIVER_REL = "helpers/benchmarking/staged_source_grid.py"

#: The card chain this driver sits on (cards 1-9 of the Round 2 set).
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
)

# ---------------------------------------------------------------------------
# Pre-registered grids, statistics, seeds (the contract for cards 11/12;
# a change re-opens this card before either run starts).
# ---------------------------------------------------------------------------
GRIDS: dict[str, dict] = {
    "baryonic": {
        "source_qn": (1, 1, 0),
        "start_gev": 1.07324891,
        "step_gev": 0.1,
        "points": 20,
        "label": "20 points, 1.07324891 + 0.1 k, (1,1,0) at rest",
        "threshold_note": "the threshold min(m_p + m_pi0, m_n + m_pi+) "
                          "= 1.07324891 GeV is the grid start",
    },
    "mesonic": {
        "source_qn": (0, 0, 0),
        "start_gev": 0.2699532,
        "step_gev": 0.1,
        "points": 28,
        "label": "28 points, 0.2699532 + 0.1 k, (0,0,0) at rest",
        "threshold_note": "the threshold 2 m_pi0 = 0.2699532 GeV is the "
                          "grid start",
    },
}

#: T1 statistics (plan section 5.1): 4 replicas x 25,000 decays per grid
#: point, the "mechanism scans and template generation" standard.
T1_REPLICAS = 4
T1_DECAYS_PER_REPLICA = 25_000
T1_LABEL = "T1: 4 replicas x 25,000 decays per grid point (plan section 5.1)"

#: Master seed and the per-decay event-index map (the card's seed map).
MASTER_SEED = 20260913
SEED_RULE = "derive_substreams(20260913, r, k * 1_000_000 + i)"
SEED_LABELS = ("decay_channel", "decay_kinematics")

#: Smoke mode: 2 points (k = 0, 1) x 200 decays x 1 replica, hermetic.
SMOKE_POINTS = 2
SMOKE_DECAYS = 200
SMOKE_REPLICAS = 1

#: Degenerate-window width bound (the card's mixture item 3).
DEGENERATE_WINDOW_TOL = 1e-12

#: The committed-catalog note recorded in the pre-registration: the
#: stable termination records carrying the source quantum numbers keep
#: the card's open-set non-emptiness assert true at every grid point.
STABLE_IDENTITY_NOTE = (
    "the stable termination records carrying the source (B, Q, S) keep "
    "the open set non-empty at every grid point: their spectral support "
    "is a single point (the pole mass), so their truncated window "
    "[mass_range[0], min(mass_range[1], M_k)] is degenerate but "
    "non-empty at every grid mass above the pole"
)

#: The pair diagnostics (the plan section 8 R22 "pair production"
#: verification set of the card's output schema).  Each pair is the
#: charge-conjugate co-occurrence count ``min(n, nbar)`` per event.
PAIR_DEFINITIONS = (
    ("pi_plus_pi_minus_per_event", 211, -211),
    ("p_pbar_per_event", 2212, -2212),
    ("n_nbar_per_event", 2112, -2112),
)

#: The new-run threshold-discontinuity metric (the Round 3 card 2
#: re-registration; plan section 8, Gate 2 threshold-discontinuity
#: clause, line 580; the R31 card 8 dependency).  New source-grid runs
#: compare every adjacent pair and species against the FULL analytic
#: difference on both pair classes (open-set-change and open-set-
#: unchanged), superseding the R21/R22 pre-registered convention that
#: applied ``delta_analytic = 0`` at open-set-unchanged pairs.  The
#: sealed R21/R22 gate reports keep their original verdicts (read-only
#: archives; the revised verdict lives in the card 2 run dir).
DISCONTINUITY_METRIC = {
    "id": THRESHOLD_METRIC_ID,
    "engine": "na49_mixing.threshold_metric.full_analytic_difference",
    "record_set": "na49_mixing.threshold_metric.RecordSet",
    "evaluate": "na49_mixing.threshold_metric.evaluate_pair",
    "statement": (
        "for every adjacent pair (k, k+1) and every stable species, "
        "with SE_combined = sqrt(SE_k^2 + SE_{k+1}^2): "
        "|mu(k+1) - mu(k) - delta_analytic_full| <= 3 SE_combined on "
        "BOTH pair classes (open-set-change and open-set-unchanged), "
        "for the species multiplicity and the per-species momentum "
        "moments <p> and <p_T>_y; delta_analytic_full is the full "
        "analytic difference of the flat-mixture expectation (the "
        "open-set-change contribution plus the analytic momentum "
        "drift), with no Monte Carlo"
    ),
    "tolerance_sigma": SIGMA_TOLERANCE,
    "card": "TASK-20260917-r3-gate2-followup-threshold-metric-"
            "reregistration",
    "supersedes": (
        "the R21/R22 pre-registered metric applied delta_analytic = 0 "
        "at open-set-unchanged pairs; the sealed gate2_r21.json / "
        "gate2_r22.json verdicts are untouched"
    ),
}


def _grid_masses(grid: dict) -> list[float]:
    """The grid masses ``M_k = start + step * k`` of *grid*."""
    return [
        float(grid["start_gev"]) + float(grid["step_gev"]) * k
        for k in range(int(grid["points"]))
    ]


def _open_set(
    catalog: HadronCatalog, source_qn: tuple[int, int, int], mass: float
) -> list[HadronRecord]:
    """The card's open set at grid mass *mass*: the catalog records with
    the source's ``(B, Q, S)`` whose truncated spectral window
    ``[mass_range[0], min(mass_range[1], M_k)]`` is non-empty, in
    catalog record order."""
    out: list[HadronRecord] = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != source_qn:
            continue
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        if min(hi, float(mass)) >= lo:
            out.append(record)
    return out


def _assert_open_set(
    records: list[HadronRecord],
    source_qn: tuple[int, int, int],
    mass: float,
) -> None:
    """The card's open-set assert: non-empty at every grid point; a
    violation raises and stops the run (no fallback decay path)."""
    if records:
        return
    raise RuntimeError(
        "staged_source_grid: the open set at source quantum numbers "
        f"{tuple(source_qn)} and grid mass {mass!r} GeV is empty; the "
        "pre-registered mixture model has no record to draw and no "
        "fallback decay path is implemented (the run stops)"
    )


def _stable_species_keys(catalog: HadronCatalog) -> tuple[str, ...]:
    """The per-species key set of the diagnostics: the catalog's stable
    termination set, in catalog record order, with the charge-symbol
    transliteration (the card's multiplicity species set)."""
    stable = catalog.stable_states()
    return tuple(
        _species_key(record.name)
        for record in catalog.records()
        if int(record.pdg_id) in stable
    )


def threshold_metric_table(
    catalog: HadronCatalog,
    source_type: str,
    point_indices: "list[int] | None" = None,
) -> dict:
    """The re-registered discontinuity-metric table of one grid (pure
    function of the catalog and the grid; no draws).

    For every adjacent pair of *point_indices* and every stable species
    the payload carries :func:`full_analytic_difference`'s full analytic
    difference with its open-set-change and momentum-drift decomposition
    (the new-run metric of :data:`DISCONTINUITY_METRIC`).  The record
    set of a pair is the union of the card open sets at the two point
    masses (the card-9 truncated-spectral-window rule)."""
    if source_type not in GRIDS:
        raise ValueError(
            f"staged_source_grid: unknown source type {source_type!r}"
        )
    grid = GRIDS[source_type]
    source_qn = (int(grid["source_qn"][0]), int(grid["source_qn"][1]),
                 int(grid["source_qn"][2]))
    masses = _grid_masses(grid)
    if point_indices is None:
        point_indices = list(range(int(grid["points"])))
    species_keys = _stable_species_keys(catalog)
    pairs: dict[str, dict] = {}
    for k, kp1 in zip(point_indices, point_indices[1:]):
        open_k = _open_set(catalog, source_qn, masses[k])
        open_kp1 = _open_set(catalog, source_qn, masses[kp1])
        _assert_open_set(open_k, source_qn, masses[k])
        _assert_open_set(open_kp1, source_qn, masses[kp1])
        union: dict[int, HadronRecord] = {}
        for record in open_k + open_kp1:
            union.setdefault(int(record.pdg_id), record)
        record_set = RecordSet(
            catalog, [union[pdg] for pdg in sorted(union)])
        per_species: dict[str, dict] = {}
        for key in species_keys:
            per_species[key] = full_analytic_difference(
                record_set, masses[k], masses[kp1], key)
        pairs[f"{k},{kp1}"] = {
            "pair": [int(k), int(kp1)],
            "masses_gev": [float(masses[k]), float(masses[kp1])],
            "open_records_at_M_k": [int(r.pdg_id) for r in open_k],
            "open_records_at_M_kp1": [int(r.pdg_id) for r in open_kp1],
            "species": per_species,
        }
    return {
        "schema": "na49_mixing.staged_source_grid.threshold_metric/2",
        "metric": DISCONTINUITY_METRIC,
        "source_type": source_type,
        "source_qn": [int(q) for q in source_qn],
        "grid_masses_gev": [float(masses[k]) for k in point_indices],
        "grid_points": [int(k) for k in point_indices],
        "species_keys": list(species_keys),
        "pairs": pairs,
    }


def _species_key(name: str) -> str:
    """The JSON species key of a catalog record name: the charge symbol
    transliterated (``+`` -> ``_plus``, ``-`` -> ``_minus``, a trailing
    ``0`` -> ``_0``), matching the card's schema example
    (``pi_plus``, ``pi_minus``, ``pi_0``, ``proton``, ``neutron``)."""
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _bqs(catalog: HadronCatalog, pdg_id: int) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` of a final particle: the catalog record fields
    for records, the exact PDG quantum numbers otherwise (the card-6
    kernel's ``_bqs_of`` convention)."""
    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        qn = quantum_numbers(int(pdg_id))
        return (int(qn.B), int(qn.Q), int(qn.S))
    return (int(record.B), int(record.Q), int(record.S))


def _daughter_masses(
    catalog: HadronCatalog, record: HadronRecord
) -> dict[int, float]:
    """Pole masses of every channel daughter of *record* (the card-3
    width law's daughter-mass map; the card-1 catalog threshold
    convention; the same construction as the kernel's map)."""
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


class _Accumulator:
    """Per-event diagnostic accumulators of one (point, replica) sample.

    All sums are accumulated in generation order, so the aggregated
    payload is bit-reproducible for the same event sequence."""

    def __init__(self, species_keys: tuple[str, ...]) -> None:
        self.n_events = 0
        self.count = {key: 0 for key in species_keys}
        self.count_sq = {key: 0 for key in species_keys}
        self.p_sum = {key: 0.0 for key in species_keys}
        self.e_sum = {key: 0.0 for key in species_keys}
        self.ept_sum = {key: 0.0 for key in species_keys}
        self.n_species = {key: 0 for key in species_keys}
        self.n_resonant = {key: 0 for key in species_keys}
        self.parent_counts: dict[str, dict[int, int]] = {
            key: {} for key in species_keys
        }
        self.pair_keys = tuple(name for name, _, _ in PAIR_DEFINITIONS)
        self.pair = {key: 0 for key in self.pair_keys}
        self.pair_sq = {key: 0 for key in self.pair_keys}

    def add_event(self, counts: dict[str, int]) -> None:
        self.n_events += 1
        for key, c in counts.items():
            self.count[key] += c
            self.count_sq[key] += c * c

    def add_particle(
        self,
        key: str,
        p: float,
        e: float,
        pt: float,
        resonant: bool,
        parent_pdg: int | None,
    ) -> None:
        self.n_species[key] += 1
        self.p_sum[key] += p
        self.e_sum[key] += e
        self.ept_sum[key] += e * pt
        if resonant and parent_pdg is not None:
            self.n_resonant[key] += 1
            bucket = self.parent_counts[key]
            bucket[int(parent_pdg)] = bucket.get(int(parent_pdg), 0) + 1

    def add_pairs(self, values: dict[str, int]) -> None:
        for key in self.pair_keys:
            v = values[key]
            self.pair[key] += v
            self.pair_sq[key] += v * v

    def merge(self, other: "_Accumulator") -> None:
        """Merge *other* into ``self`` (the per-point aggregation over
        the replicas; exact integer counts, float sums in replica
        order)."""
        self.n_events += other.n_events
        for key in self.count:
            self.count[key] += other.count[key]
            self.count_sq[key] += other.count_sq[key]
            self.p_sum[key] += other.p_sum[key]
            self.e_sum[key] += other.e_sum[key]
            self.ept_sum[key] += other.ept_sum[key]
            self.n_species[key] += other.n_species[key]
            self.n_resonant[key] += other.n_resonant[key]
            for pdg, cnt in other.parent_counts[key].items():
                bucket = self.parent_counts[key]
                bucket[pdg] = bucket.get(pdg, 0) + cnt
        for key in self.pair:
            self.pair[key] += other.pair[key]
            self.pair_sq[key] += other.pair_sq[key]


def _mean_se(total: int, total_sq: int, n: int) -> tuple[float, float]:
    """Mean and standard error of the mean over *n* events (the sample
    standard deviation with the ``n - 1`` convention; 0.0 for fewer
    than two events)."""
    if n <= 0:
        return 0.0, 0.0
    mean = total / n
    if n < 2:
        return mean, 0.0
    variance = (total_sq - total * total / n) / (n - 1)
    return mean, math.sqrt(max(variance / n, 0.0))


def _diagnostics_payload(
    acc: _Accumulator,
    point: int,
    mass: float,
    source_qn: tuple[int, int, int],
    open_records: list[int],
) -> dict:
    """The card's per-point output schema for one accumulated sample."""
    n = acc.n_events
    if n <= 0:
        raise RuntimeError(
            "staged_source_grid: an empty event sample cannot produce "
            "the per-point diagnostics"
        )

    multiplicities: dict[str, dict[str, float]] = {}
    for key in acc.count:
        mean, se = _mean_se(acc.count[key], acc.count_sq[key], n)
        multiplicities[key] = {"mean": mean, "se": se}

    momentum_moments: dict[str, dict[str, float]] = {}
    feed_down: dict[str, dict] = {}
    for key in acc.count:
        n_species = acc.n_species[key]
        if n_species <= 0:
            continue
        e_sum = acc.e_sum[key]
        momentum_moments[key] = {
            "mean_p_gev": acc.p_sum[key] / n_species,
            "mean_pt_y_gev": (
                acc.ept_sum[key] / e_sum if e_sum > 0.0 else 0.0
            ),
            "mean_e_gev": e_sum / n_species,
        }
        feed_down[key] = {
            "fraction_resonant": acc.n_resonant[key] / n_species,
            "by_parent": {
                str(pdg): cnt / n_species
                for pdg, cnt in sorted(acc.parent_counts[key].items())
            },
        }

    pairs: dict[str, dict[str, float]] = {}
    for name in acc.pair_keys:
        mean, se = _mean_se(acc.pair[name], acc.pair_sq[name], n)
        pairs[name] = {"mean": mean, "se": se}

    return {
        "point": int(point),
        "source_mass_gev": float(mass),
        "source_qn": [int(q) for q in source_qn],
        "open_records": [int(p) for p in open_records],
        "n_decays": int(n),
        "multiplicities": multiplicities,
        "momentum_moments": momentum_moments,
        "feed_down": feed_down,
        "charge_balance": {
            "total_Q_exact": int(source_qn[1]),
            "total_B_exact": int(source_qn[0]),
            "events_checked": int(n),
        },
        "pairs": pairs,
    }


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as UTF-8 JSON (indent 2, fixed key order, a
    trailing newline); identical inputs produce identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _realize_record(
    catalog: HadronCatalog,
    records: list[HadronRecord],
    mass: float,
    rng_record: np.random.Generator,
    rng_mass: np.random.Generator,
    daughter_maps: dict[int, dict[int, float]],
) -> tuple[HadronRecord, float]:
    """Mixture steps 2-3: draw one record uniformly over the open set
    (one uniform on the ``decay_channel`` substream) and realize its
    mass from the card-3 spectral function restricted to the truncated
    window (the card-3 ``draw_mass`` on the ``decay_kinematics``
    substream); a degenerate window (width <= 1e-12) fixes the mass at
    the grid point (the card's rule)."""
    index = min(int(rng_record.random() * len(records)), len(records) - 1)
    record = records[int(index)]
    lo = float(record.mass_range[0])
    hi = min(float(record.mass_range[1]), float(mass))
    if hi - lo <= DEGENERATE_WINDOW_TOL:
        return record, float(mass)
    dmap = daughter_maps.get(int(record.pdg_id))
    if dmap is None:
        dmap = _daughter_masses(catalog, record)
        daughter_maps[int(record.pdg_id)] = dmap
    drawn = draw_mass((lo, hi), record, dmap, rng_mass)
    if drawn is None:
        raise RuntimeError(
            f"staged_source_grid: record {record.name} (pdg "
            f"{record.pdg_id}) has no line-shape mass in the window "
            f"{(lo, hi)!r} at grid mass {mass!r} GeV; no fallback mass "
            "is implemented"
        )
    return record, float(drawn)


def _record_name(catalog: HadronCatalog, pdg_id: int) -> str:
    """The species name of a final particle: the catalog record's name
    for records, the vendored PDG table's name otherwise, the bare id
    when neither resolves (defensive; every termination state of the
    committed catalog is a record)."""
    try:
        return str(catalog.get(int(pdg_id)).name)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            return str(int(pdg_id))
        return str(entry.name)


def _accumulate_event(
    acc: _Accumulator,
    catalog: HadronCatalog,
    vertices: list,
    finals: list[FinalParticle],
    source_pdg: int,
    source_qn: tuple[int, int, int],
) -> None:
    """One event into *acc*: the per-event ``(B, Q, S)`` balance against
    the source (checked, never repaired), the species counts, the
    momentum sums, the feed-down lineage, and the pair co-occurrence."""
    event_pdg_counts: dict[int, int] = {}
    for final in finals:
        pdg = int(final.pdg_id)
        event_pdg_counts[pdg] = event_pdg_counts.get(pdg, 0) + 1

    # -- per-event (B, Q, S) balance against the source -----------------
    total_b = total_q = total_s = 0
    for pdg, cnt in event_pdg_counts.items():
        b, q, s = _bqs(catalog, pdg)
        total_b += b * cnt
        total_q += q * cnt
        total_s += s * cnt
    if (total_b, total_q, total_s) != tuple(source_qn):
        raise RuntimeError(
            "staged_source_grid: an event's final particles sum to "
            f"(B, Q, S) = {(total_b, total_q, total_s)} but the source "
            f"carries {tuple(source_qn)} (pdg {source_pdg}); conservation "
            "is checked, never repaired (the run stops)"
        )

    event_counts: dict[str, int] = {}
    for pdg, cnt in event_pdg_counts.items():
        key = _species_key(_record_name(catalog, pdg))
        event_counts[key] = event_counts.get(key, 0) + cnt
    acc.add_event(event_counts)

    # -- per-particle sums and feed-down lineage -------------------------
    for final in finals:
        pdg = int(final.pdg_id)
        key = _species_key(_record_name(catalog, pdg))
        p4 = final.p4
        resonant = False
        parent_pdg: int | None = None
        if final.parent_id is not None:
            parent_pdg = int(vertices[int(final.parent_id)].parent_pdg)
            resonant = parent_pdg != int(source_pdg)
        acc.add_particle(
            key,
            float(p4.momentum()),
            float(p4.e),
            float(math.hypot(p4.px, p4.py)),
            resonant,
            parent_pdg if resonant else None,
        )

    # -- charge-conjugate co-occurrence (pair production) ----------------
    pair_counts = {
        name: min(event_pdg_counts.get(pdga, 0),
                  event_pdg_counts.get(pdgb, 0))
        for name, pdga, pdgb in PAIR_DEFINITIONS
    }
    acc.add_pairs(pair_counts)


def _run_grid(
    catalog: HadronCatalog,
    source_type: str,
    point_indices: list[int],
    decays_per_replica: int,
    replicas: int,
    master_seed: int,
    run_dir: Path,
    mode: str,
) -> None:
    """Generate the requested grid slice and write the run artifacts.

    ``pre_registration.json`` and ``provenance.json`` are written before
    any generation.  Per point, one ``replica_<r>_point_<k:02d>.json``
    per replica with the raw per-replica diagnostics and one
    ``point_<k:02d>.json`` with the replica-merged aggregates."""
    grid = GRIDS[source_type]
    source_qn = (int(grid["source_qn"][0]), int(grid["source_qn"][1]),
                 int(grid["source_qn"][2]))
    masses = _grid_masses(grid)
    stable = catalog.stable_states()

    # The per-point open sets (the card's pre-registered open sets,
    # computed from the committed catalog) and the non-emptiness assert.
    open_sets: dict[int, list[HadronRecord]] = {}
    for k in point_indices:
        records = _open_set(catalog, source_qn, masses[k])
        _assert_open_set(records, source_qn, masses[k])
        open_sets[k] = records

    species_keys = tuple(
        _species_key(record.name)
        for record in catalog.records()
        if int(record.pdg_id) in stable
    )

    # -- pre-registration and provenance, before any generation ---------
    pre_registration = {
        "schema": "na49_mixing.staged_source_grid.pre_registration/1",
        "card": CARD_ID,
        "mode": mode,
        "source_type": source_type,
        "source_qn": [int(q) for q in source_qn],
        "grids": {
            name: {
                "label": spec["label"],
                "source_qn": [int(q) for q in spec["source_qn"]],
                "start_gev": float(spec["start_gev"]),
                "step_gev": float(spec["step_gev"]),
                "points": int(spec["points"]),
                "threshold_note": spec["threshold_note"],
            }
            for name, spec in GRIDS.items()
        },
        "grid_points": [int(k) for k in point_indices],
        "grid_masses_gev": [float(masses[k]) for k in point_indices],
        "statistics": {
            "standard": T1_LABEL,
            "replicas": int(replicas),
            "decays_per_replica": int(decays_per_replica),
        },
        "seed_map": {
            "master_seed": int(master_seed),
            "rule": SEED_RULE,
            "substream_labels": list(SEED_LABELS),
            "note": "the mixture record draw consumes one uniform on the "
                    "decay_channel layer and the card-3 draw_mass consumes "
                    "two uniforms per non-degenerate draw on the "
                    "decay_kinematics layer; both layers draw from spawned "
                    "children of the two labeled stream ids "
                    "(SeedSequence(stream_id).spawn(1), no extension of "
                    "SUBSTREAM_LABELS) while the card-6/7 kernel keeps the "
                    "labeled ids themselves re-derived from the same "
                    "(master_seed, replica, event_index) triple, so the "
                    "record selection stays uncorrelated with the root "
                    "vertex's channel selection",
        },
        "mixture_model": {
            "open_set_rule": "the catalog records with the source "
                             "(B, Q, S) whose truncated spectral window "
                             "[mass_range[0], min(mass_range[1], M_k)] is "
                             "non-empty; asserted non-empty at every grid "
                             "point (a violation raises and stops the run)",
            "open_set_nonempty_note": STABLE_IDENTITY_NOTE,
            "record_draw": "uniform over the open set (equal record "
                           "weights; the card-5 isospin charge weights act "
                           "inside each record)",
            "mass_rule": "the card-3 spectral function restricted to the "
                         "truncated window (resonance_widths.draw_mass); a "
                         "degenerate window (width <= 1e-12) fixes the mass "
                         "at the grid point M_k",
            "decay": "the card-6/7 kernel through the resolved "
                     "SourceRecord (the at-rest four-momentum of the "
                     "realized mass); a drawn record already in the stable "
                     "set contributes itself as the single final particle "
                     "at the realized four-momentum (plan section 8.4 "
                     "step 7; a direct source product, never a resonance "
                     "feed-down)",
            "kernels": "spin-averaged (unpolarized; round 2 index item 9)",
        },
        "threshold_metric": DISCONTINUITY_METRIC,
        "open_sets": {
            str(k): [int(record.pdg_id) for record in open_sets[k]]
            for k in point_indices
        },
        "output_schema": {
            "point_file": "point_<k:02d>.json",
            "replica_file": "replica_<r>_point_<k:02d>.json",
            "keys": [
                "point", "source_mass_gev", "source_qn", "open_records",
                "n_decays", "multiplicities", "momentum_moments",
                "feed_down", "charge_balance", "pairs",
            ],
            "multiplicity_species": "the catalog stable termination set, "
                                    "all listed, in catalog record order",
            "mean_pt_y_convention": "the density-p_T f convention "
                                    "(<p_T>_y): sum(E * pT) / sum(E) over "
                                    "the species' particles",
            "pairs_convention": "the charge-conjugate co-occurrence "
                                "min(n, nbar) per event, mean and standard "
                                "error over the point's events",
        },
        "written_before_generation": True,
    }
    _write_json(run_dir / "pre_registration.json", pre_registration)

    catalog_path = NESTED_ROOT / CATALOG_REL
    provenance = {
        "schema": "na49_mixing.staged_source_grid.provenance/1",
        "card": CARD_ID,
        "card_chain": list(CARD_CHAIN),
        "catalog": {
            "path": CATALOG_REL,
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        },
        "driver": DRIVER_REL,
        "written_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "written_before_generation": True,
    }
    _write_json(run_dir / "provenance.json", provenance)

    # -- generation -----------------------------------------------------
    daughter_maps: dict[int, dict[int, float]] = {}
    for k in point_indices:
        mass = masses[k]
        records = open_sets[k]
        record_pdgs = [int(record.pdg_id) for record in records]
        point_acc = _Accumulator(species_keys)
        for replica in range(replicas):
            acc = _Accumulator(species_keys)
            for i in range(decays_per_replica):
                event_index = k * 1_000_000 + i
                event_index = k * 1_000_000 + i
                subs = derive_substreams(master_seed, replica, event_index)
                # The mixture layer consumes one-step spawned children of
                # the two labeled stream ids (no extension of
                # SUBSTREAM_LABELS); the kernel keeps the labeled ids
                # themselves, so the two layers never share a uniform
                # value and the record selection stays uncorrelated with
                # the root vertex's channel selection.
                rng_record = np.random.default_rng(
                    np.random.SeedSequence(
                        subs.stream_for("decay_channel")
                    ).spawn(1)[0]
                )
                rng_mass = np.random.default_rng(
                    np.random.SeedSequence(
                        subs.stream_for("decay_kinematics")
                    ).spawn(1)[0]
                )
                record, realized = _realize_record(
                    catalog, records, mass, rng_record, rng_mass,
                    daughter_maps,
                )
                decayer = ResonanceDecayer(
                    catalog, master_seed, replica, event_index
                )
                vertices, finals = decayer.decay_source(SourceRecord(
                    pdg_id=int(record.pdg_id),
                    four_momentum=(realized, 0.0, 0.0, 0.0),
                    B=int(source_qn[0]),
                    Q=int(source_qn[1]),
                    S=int(source_qn[2]),
                    mass=float(realized),
                    source_id=int(event_index),
                ))
                if not finals:
                    # The drawn record is already in the stable set (plan
                    # section 8.4 step 7): the source contributes itself as
                    # the single final particle at the realized
                    # four-momentum, with no producing vertex.
                    finals = [FinalParticle(
                        particle_id=0,
                        pdg_id=int(record.pdg_id),
                        p4=FourVector(float(realized), 0.0, 0.0, 0.0),
                        source_id=int(event_index),
                        parent_id=None,
                        channel_family="source_stable",
                    )]
                _accumulate_event(
                    acc, catalog, vertices, finals,
                    int(record.pdg_id), source_qn,
                )
            _write_json(
                run_dir / f"replica_{replica}_point_{k:02d}.json",
                _diagnostics_payload(acc, k, mass, source_qn, record_pdgs),
            )
            point_acc.merge(acc)
        _write_json(
            run_dir / f"point_{k:02d}.json",
            _diagnostics_payload(point_acc, k, mass, source_qn, record_pdgs),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_source_grid.py",
        description=(
            "Generate the isolated fixed-quantum-number source grids of "
            "runs R21 (baryonic) and R22 (mesonic) through the card-6/7 "
            "decay kernel (Round 2, card 9)."
        ),
    )
    parser.add_argument(
        "--source-type", choices=sorted(GRIDS), required=True,
        help="the pre-registered source type (baryonic = R21, mesonic = R22)",
    )
    parser.add_argument(
        "--points", type=int, default=None,
        help="the number of grid points; must match the source type "
             "(20 baryonic, 28 mesonic); the smoke mode fixes 2 (k = 0, 1)",
    )
    parser.add_argument(
        "--decays-per-replica", type=int, default=T1_DECAYS_PER_REPLICA,
        help=f"decays per replica per point (default "
             f"{T1_DECAYS_PER_REPLICA}); the smoke mode fixes {SMOKE_DECAYS}",
    )
    parser.add_argument(
        "--replicas", type=int, default=T1_REPLICAS,
        help=f"replicas per point (default {T1_REPLICAS}); the smoke mode "
             f"fixes {SMOKE_REPLICAS}",
    )
    parser.add_argument(
        "--master-seed", type=int, default=MASTER_SEED,
        help=f"master seed (default {MASTER_SEED})",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="the run directory for pre_registration.json, provenance.json, "
             "and the per-point/per-replica artifacts",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"hermetic smoke run: {SMOKE_POINTS} points (k = 0, 1) x "
             f"{SMOKE_DECAYS} decays x {SMOKE_REPLICAS} replica into "
             "--run-dir",
    )
    parser.add_argument(
        "--threshold-metric", action="store_true",
        help="write threshold_metric_<source-type>.json (the "
             "re-registered discontinuity-metric table over the "
             "pre-registered grid, the new-run metric of "
             "DISCONTINUITY_METRIC) into --run-dir, then stop (no "
             "generation)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_type = str(args.source_type)
    grid = GRIDS[source_type]
    if args.threshold_metric:
        # The re-registered discontinuity-metric table of the
        # pre-registered grid (the new-run metric; no generation).
        run_dir = Path(str(args.run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
        table = threshold_metric_table(catalog, source_type)
        out_path = run_dir / f"threshold_metric_{source_type}.json"
        _write_json(out_path, table)
        print(
            f"staged_source_grid: wrote {out_path} "
            f"({len(table['pairs'])} adjacent pairs x "
            f"{len(table['species_keys'])} species under the "
            f"re-registered metric {THRESHOLD_METRIC_ID})"
        )
        return 0
    if args.smoke:
        point_indices = list(range(SMOKE_POINTS))
        decays = SMOKE_DECAYS
        replicas = SMOKE_REPLICAS
        mode = "smoke"
    else:
        points = (int(args.points) if args.points is not None
                  else int(grid["points"]))
        if points != int(grid["points"]):
            raise SystemExit(
                f"staged_source_grid: --points {points} does not match the "
                f"{source_type} source type ({int(grid['points'])} "
                "pre-registered grid points)"
            )
        point_indices = list(range(points))
        decays = int(args.decays_per_replica)
        replicas = int(args.replicas)
        mode = "production"
        if decays <= 0 or replicas <= 0:
            raise SystemExit(
                "staged_source_grid: --decays-per-replica and --replicas "
                "must be positive"
            )

    run_dir = Path(str(args.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)

    _run_grid(
        catalog=catalog,
        source_type=source_type,
        point_indices=point_indices,
        decays_per_replica=decays,
        replicas=replicas,
        master_seed=int(args.master_seed),
        run_dir=run_dir,
        mode=mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
