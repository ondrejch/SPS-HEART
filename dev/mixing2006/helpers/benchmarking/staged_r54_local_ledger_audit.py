#!/usr/bin/env python3
"""R54 driver: the local-ledger audit (Round 5, card 14).

Task ``TASK-20260925-r5-run-r54-local-ledger-audit`` (plan section 11
run R54, lines 928-930; plan section 23 lines 1825-1827, "Round 5 --
exchange topology").  The card executes run R54, the local-ledger
audit: exactly one million events across all topologies, asserting that
every nonzero quantum-number transfer is attached to an allowed
exchange record (plan line 930).  This is a generation audit: it draws
no optimizer (index item 10), no parameter is fit, and the assertion is
zero violations across all events on card 2's five ledger rules (plan
lines 930 and 875-879, plan section 11.4).

The five ledger rules (the card's pre-registered assertion block)
---------------------------------------------------------------
The audit mechanism is card 2's ``na49_mixing.exchange_ledger.
audit_event`` (the :class:`~na49_mixing.exchange_ledger.LedgerAuditReport`
with an empty violation list on a compliant event), consumed over the
class records produced by the card-9 runner:

- (a) ``closed_set_membership``: every exchange record is in the closed
  allowed-exchange set (:data:`~na49_mixing.exchange_ledger.
  ALLOWED_EXCHANGES`, exactly the vacuum exchange at this round);
- (b) ``vacuum_zero_transfer``: the vacuum exchange transfers zero;
- (c) ``inheritance_after_declared_transfer``: each outgoing excitation
  equals the line quantum numbers after the declared transfer (outgoing
  = incoming + transfer, exact integers);
- (d) ``local_decay_closure``: every decay vertex closes locally;
- (e) ``central_source_neutrality``: the central source carries
  B = Q = S = 0 (plan line 849).

The audit inputs are built from every landed class record uniformly:
the two incoming proton lines (the pinned ``(B, Q, S, 2I3) =
(1, 1, 0, 1)`` pair of the M8 p+p configuration), the event's exchange
records (name, declared transfer), the two outgoing baryonic lines, the
central source where the class carries one (the ND family), and the
decay vertices.  The decay vertices are realized at the line level:
each baryonic line's outgoing ledger state paired with the final-state
hadrons attributed to that line (the class records' ``line`` labels:
``"excitation"``/``"remnant"`` on the single-diffraction class,
``"projectile"``/``"target"`` on the double-diffraction and ND-family
classes), plus the central source with its ``"central"`` products on
the ND family.  The frozen simplified cascade's internal vertices close
locally by the cascade module's own DECAY_CLOSURE_TOL check consumed at
generation; the line-level vertex is the full-chain closure against the
line's ledger state, and the nd_multi class's two central sources (each
carrying ``(0, 0, 0, 0)`` exactly) are audited as the combined neutral
vertex.  The audit compares quantum numbers only (the ledger performs
no kinematics); the per-event four-momentum closure is the house
conservation contract's separate check.

Topology sampling (index item 6; the recorded source)
-----------------------------------------------------
The mixed draw runs at the R51 frozen fractions when available (the
sealed R51 ``fit_report.json`` fitted values, asserted byte-exactly at
pre-registration and re-asserted at generation), otherwise at the equal
nominal fractions of index item 2 (0.25 each over SD_p, SD_t, DD, ND).
At the sealed R51 point the four fitted inelastic fractions sum to one
exactly in float arithmetic, so the elastic fraction (the complement of
the inelastic mass, the runner's ``pinned_trace`` default rule computed
from the same source) is exactly 0.0 and no elastic event realizes in
production; the elastic class is nevertheless included in the
generation surface: it is dispatchable through the card-9
``run_topology``, its events are audited by the same five-rule
machinery (the smoke pins the elastic dispatch and audit explicitly),
and the gate evidence records its fraction.  The nd_multi class is
sampled only when the multiple-excitation class is retained (the R53
outcome, if run): the drawn ``"nd"`` labels dispatch the card-8
nd_multi generation at the sealed fitted P(n = 2), exactly the R53
mixed-draw convention; otherwise P(n = 2) = 0.0 (the unretained default)
and the drawn ``"nd"`` labels run card 7's ND generation directly.

The R53 outcome (the R53-PENDING default and the re-audit obligation)
---------------------------------------------------------------------
The R53 outcome is read from the sealed R53 archive
(``gate5_r53.json``: the ``status``, the ``fitted`` P(n = 2), and the
``retained`` boolean; the seal verifies).  When the sealed archive is
absent or incomplete (R53 still generating), the pre-registration
records the R53-PENDING default: retained false, P(n = 2) = 0.0 (the
card-8 unretained default), with the re-audit obligation flag: if R53
later seals with ``retained: true``, this pre-registration is
invalidated (the audit re-registers under a new date with the sealed
fitted P(n = 2) and re-executes); a sealed ``retained: false`` confirms
the recorded 0.0 and the run stands.  The generation re-reads the
outcome and refuses any drift against the pre-registered block (a
pending-to-retained flip stops the run with the obligation message; it
is never silently absorbed).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the M8 frozen
  parameter map, the sealed R50/R51 seals (and the sealed R53 seal when
  the archive is present), the frozen fractions against the sealed R51
  ``fit_report.json``, the nominal-parameter reference against the
  sealed R50 pre-registration, and the seed-block disjointness before
  anything is written.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one mixed-draw replica x 2,000
  events at the smoke seed 20312925 (raw offset +12000 from master
  20260925) on the event-index partition 1_600_000_000 + i (i < 2,000),
  plus the per-class pinned dispatch exercise (every topology class,
  including elastic and the nd_multi class at the pinned smoke exercise
  P(n = 2), through the same audit machinery on the dedicated smoke
  class-exercise partition), and the smoke audit evaluation, into an
  explicit ``--run-dir`` (the default production run dir is refused).
  Two identical smoke invocations produce byte-identical artifacts.  No
  sealed archive is read and no SPS table is read (the smoke asserts
  the frozen-value constants against no sealed archive; the production
  assertion runs in --pre-registration and --evaluate).  ``--gate`` /
  ``--seal`` / ``--clean-validate`` refuse smoke artifacts (the R31/8c
  convention).
- ``--generate [--replica R]``: generate the requested production
  replica records (any subset; the default is every replica), one
  record file per replica, then stop.  Requires the pre-registration
  and re-asserts the frozen fractions and the R53-outcome consistency
  before generating.  A completed record that matches the
  pre-registered identity is reused unchanged, so the declared tier can
  be completed replica by replica (the pre-registered gateway fallback
  runs one job per replica).
- ``--evaluate``: the ledger audit and evaluation over the generated
  artifacts (the stored records are re-audited from their stored audit
  inputs, the per-rule violation counts pool, the per-class draw counts
  are checked against the recorded fractions at the recorded MC
  tolerance, the conservation aggregate pools),
  ``audit_report.json``, then stop.  Requires all 20 production
  records.
- ``--gate``: write ``gate5_r54.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the audit
  report.
- ``--clean-validate``: re-run the audit evaluation from a clean
  interpreter over the generated artifacts with a byte-identical
  report, re-run the pre-registered probe replica (replica 0) in memory
  against its sealed record, write ``clean_environment_commands.json``,
  then stop.  Exactly one pass; executed before the seal (the card-19
  seal-ordering discipline; a second invocation refuses).
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  Requires the clean-environment validation
  record (exactly one pass, before the seal).  The run dir is immutable
  after the seal.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic in ``--smoke``; the production modes read the committed
catalog and continuum config, the sealed R50 archive (the nominal-
parameter reference), the sealed R51 archive (the frozen fractions),
and the sealed R53 archive when present (the nd_multi retention
outcome).  The frozen simplified cascade at the sealed R45 point
(kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal
width 0.3 GeV) is consumed without refit (the R46 ``no_refit`` clause);
the exchange slopes act on the exchange kinematics only, never on any
daughter-level transverse dynamics; no Round 5 record claims a
separation of exchange recoil from decay transverse dynamics (the R46
outcome, separation_claimed false).

The sibling R51 driver module is imported and its record/conservation
machinery consumed verbatim (``event_conservation``, ``_line_qn``,
``verify_sealed_dir``, ``_load_catalog``, ``_load_families``, the
``ConservationViolation`` contract), so the conservation contract and
the sealed-archive discipline are the R51 machinery bit for bit.
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
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the resonance catalog, the continuum config, and
# the sealed archives.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        _refuse(
            "cannot locate the na49_mixing package next to "
            f"{here}; the driver runs from the nested tree "
            "(dev/mixing2006/helpers/benchmarking/) or its monorepo mirror "
            "(helpers/benchmarking/)"
        )
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


def _refuse(message: str) -> NoReturn:
    """Print a refusal to stderr and exit with code 2 before the step."""
    print(
        f"staged_r54_local_ledger_audit: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# The sibling R51 driver module: the consumed read-only machinery whose
# exact functions carry the conservation contract
# (event_conservation, ConservationViolation), the ledger-line reader
# (_line_qn), the sealed-archive discipline (verify_sealed_dir), and
# the configuration loaders (_load_catalog, _load_families).  Consuming
# the module verbatim keeps the conservation contract and the seal
# discipline the R51 machinery bit for bit (the R53 consumption
# pattern).
import staged_r51_topology_fraction_fit as r51  # noqa: E402

from na49_mixing.exchange_ledger import (  # noqa: E402
    ALLOWED_EXCHANGES,
    Exchange,
    LineState,
    VACUUM,
    ZERO_TRANSFER,
    audit_event,
)
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.rng_substreams import (  # noqa: E402
    SUBSTREAM_LABELS,
    derive_substreams,
)
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.topology_double_diffraction import DDEvent  # noqa: E402
from na49_mixing.topology_elastic import (  # noqa: E402
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
    check_zero_transfer,
)
from na49_mixing.topology_multi_neutral import (  # noqa: E402
    COUNT_BND,
    DEFAULT_P2,
    NDMultiEvent,
    P_N2_TOLERANCE,
)
from na49_mixing.topology_non_diffractive import (  # noqa: E402
    CENTRAL_LINE_LABEL,
    NDEvent,
)
from na49_mixing.topology_runner import (  # noqa: E402
    INELASTIC_ORDER,
    INELASTIC_SET,
    NOMINAL_PARAMS,
    TOPOLOGY_CLASSES,
    TopologyNominalParams,
    _baryonic_lines,
    draw_topology,
    extract_observables,
    run_topology,
)
from na49_mixing.topology_single_diffraction import SDEvent  # noqa: E402

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the imports).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260925-r5-run-r54-local-
# ledger-audit; the collision, statistics, seeds, sampling, and
# assertion blocks are verbatim from the card and the round 5 index
# items 6, 8, 9, and 10).
# ---------------------------------------------------------------------------
RUN_ID = "R54"
CARD = "TASK-20260925-r5-run-r54-local-ledger-audit"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r54_local_ledger_audit_2026-09-25 re-pins to the execution
#: date (the R50/R51/R52/R53 sealed archives moved the same way, to
#: 2026-09-27); this driver's execution pins 2026-09-28.
RUN_NAME = "r54_local_ledger_audit_2026-09-28"
DEFAULT_RUN_DIR = "dev/00runs/r54_local_ledger_audit_2026-09-28"
GATE_RECORD_NAME = "gate5_r54.json"
AUDIT_REPORT_NAME = "audit_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_EXERCISE_NAME = "smoke_class_exercise.json"
SMOKE_REPORT_NAME = "smoke_audit_report.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r54_local_ledger_audit.py")
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r54_local_ledger_audit.py")

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9, cards 10-13 (the sealed R50/R51/R52 runs
#: and the R53 conditional run), and this card.
CARD_CHAIN = (
    "TASK-20260920-r4-gate4-decision",
    "TASK-20260925-r5-elastic-reference",
    "TASK-20260925-r5-exchange-quantum-number-ledger",
    "TASK-20260925-r5-intact-low-mass-remnants",
    "TASK-20260925-r5-topology-mass-laws",
    "TASK-20260925-r5-single-diffraction",
    "TASK-20260925-r5-double-diffraction",
    "TASK-20260925-r5-non-diffrative-central-excitation",
    "TASK-20260925-r5-multiple-neutral-excitations",
    "TASK-20260925-r5-topology-template-runner",
    "TASK-20260925-r5-run-r50-pure-topology-templates",
    "TASK-20260925-r5-run-r51-topology-fraction-fit",
    "TASK-20260925-r5-run-r52-central-excitation-necessity",
    "TASK-20260925-r5-run-r53-multiple-neutral-excitation",
    "TASK-20260925-r5-run-r54-local-ledger-audit",
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests,
#: including the R45/R46/R50/R51/R52/R53 driver pins).
FROZEN_SUBSTRATE: dict[str, tuple[str, str]] = {
    "builder_blend_path": (
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py",
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    ),
    "card6_routing": (
        "dev/mixing2006/na49_mixing/spectral_density.py",
        "6f889f6d72968de53e7b27e32ba64d124d1b4e782a73606e6f6794eef9f156a9",
    ),
    "card2_engine": (
        "dev/mixing2006/na49_mixing/threshold_metric.py",
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    ),
    "catalog": (
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
        "e42a5da7137d789ba90221478c9bb404898f4fe8a7ee4604cd85b5623b751752",
    ),
    "continuum_config": (
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
        "6da57d5709b94061bc178fe94781dfe5eddcc597c5dde46bfcd5db56985cde7d",
    ),
    "r45_driver": (
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py",
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    ),
    "r46_driver": (
        "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py",
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    ),
    "r50_driver": (
        "helpers/benchmarking/staged_r50_pure_topology_templates.py",
        "b74e5408195ca6a3a53caf495da0c3f1206a9b43a73f336ba24b5cd6eb936fa3",
    ),
    "r51_driver": (
        "helpers/benchmarking/staged_r51_topology_fraction_fit.py",
        "73bd74ab72270ae47b900b4da171b88db606234b1cbd4345c8135f48911f40a1",
    ),
    "r52_driver": (
        "helpers/benchmarking/staged_r52_central_excitation_necessity.py",
        "889131347693b3e3338ececeb63ac883993b1ed04f83fde5449471f898469fa7",
    ),
    # NOTE: the R53 driver file is deliberately NOT pinned by digest
    # here: the card's provenance list records the sealed R50, R51, and
    # (if run) R53 RUN-DIR SEAL STATES, not the R53 driver's bytes, and
    # the R53 landing (card 13) was still in flight at this driver's
    # authoring (the R53 generation running; the driver file untracked
    # and owned by the card-13 executor).  The R53 consumption in this
    # driver rides on the sealed R53 archive's seal (the gate file plus
    # its SHA256SUMS.txt verification), which is the R53 integrity
    # surface the card names.
}

#: The Round 5 landed modules recorded in the provenance (read-only
#: inputs; cards 1, 5, 6, 7 are the class modules, cards 2, 3, 4 the
#: ledger/remnant/mass-law modules, card 9 the runner; card 8's
#: nd_multi module is recorded as the registry's unretained extension).
ROUND5_MODULES: dict[str, str] = {
    "card1_elastic": "dev/mixing2006/na49_mixing/topology_elastic.py",
    "card2_ledger": "dev/mixing2006/na49_mixing/exchange_ledger.py",
    "card3_remnants": "dev/mixing2006/na49_mixing/topology_remnants.py",
    "card4_mass_laws": "dev/mixing2006/na49_mixing/topology_mass_laws.py",
    "card5_single_diffraction": (
        "dev/mixing2006/na49_mixing/topology_single_diffraction.py"),
    "card6_double_diffraction": (
        "dev/mixing2006/na49_mixing/topology_double_diffraction.py"),
    "card7_non_diffractive": (
        "dev/mixing2006/na49_mixing/topology_non_diffractive.py"),
    "card8_multi_neutral": (
        "dev/mixing2006/na49_mixing/topology_multi_neutral.py"),
    "card9_runner": "dev/mixing2006/na49_mixing/topology_runner.py",
}

#: The M8 frozen ten-parameter set (the sha256 of the canonical JSON of
#: the manifest's frozen map).
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
BEAM_MOMENTUM_GEV_C = 158.0
BLEND_RAMP_START_GEV = 2.680194110727528
#: The Feynman-x / extraction energy: the card-1 module constant
#: SQRT_S = 17.270 GeV (the recorded M8 quote; the runner's pinned
#: extraction default).
SQRT_S_EXTRACTION_GEV = 17.270

#: The R45 frozen simplified-cascade point (consumed without refit; the
#: R46 no_refit clause).
R45_FROZEN_CASCADE = {
    "kappa_B": 0.0,
    "kappa_M": 4.0,
    "b_dec_gev2": 2.0,
    "baryonic_nominal_width_gev": 0.3,
    "W_flavor": 1.0,
}

#: The standing model-uncertainty carry (verbatim from the card).
MODEL_UNCERTAINTY_CARRY = {
    "r32_spread": {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
        "note": (
            "the R32 transition-window spread at 4,800,000 events; the "
            "model-uncertainty term recorded in the run record (the audit "
            "asserts the ledger rules, not the spectral agreement, so the "
            "spread is carried, not consumed)"
        ),
    },
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_seagull_note": (
        "the b_transport_seagull beyond_isotropic_not_positive residual "
        "is a carry, not a pass criterion (index item 6)"
    ),
    "engine_moment_floor_note": (
        "the calculation-engine moment-approximation floor (~1e-3 GeV/c) "
        "is tracked on its separate track "
        "(_knowledge/evidence/engine-moment-approximation-floor.md) and "
        "is not conflated with the R32 spread"
    ),
}

# ---------------------------------------------------------------------------
# Statistics, seeds, and the pre-registered replica map
# ---------------------------------------------------------------------------
#: T3 replica count over the audit's own per-replica depth (index item
#: 6): exactly 1,000,000 events = 20 replicas x 50,000 p+p events (plan
#: section 5.1 line 255's replica count; plan line 930 requires at
#: least one million and the card fixes the count with equality).
N_REPLICAS = 20
EVENTS_PER_REPLICA = 50_000
N_EVENTS_TOTAL = N_REPLICAS * EVENTS_PER_REPLICA   # 1,000,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
SEED_SLOT_OFFSET = 44
#: The raw replica seeds 20300969-20300988 (k = 0..19; index item 9;
#: the block is disjoint from every previously used raw block and from
#: every other reserved Round 5 block).
REPLICA_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + SEED_OFFSET + SEED_SLOT_OFFSET + k
    for k in range(N_REPLICAS))
EVENT_INDEX_BASE = 1_400_000_000
EVENT_INDEX_RULE = "1_400_000_000 + i, i < 50,000 (per replica)"

SMOKE_SEED = MASTER_SEED + 40_000 + 12_000              # 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"
#: The smoke class-exercise partition (the per-class pinned dispatch of
#: the smoke mode; disjoint from every production partition, from the
#: round-5 smoke partition, and from every previously used partition;
#: recorded in the smoke pre-registration's disjointness report).
SMOKE_EXERCISE_INDEX_BASE = 1_610_000_000
SMOKE_EXERCISE_EVENTS_PER_CLASS = 8
SMOKE_EXERCISE_INDEX_RULE = (
    "1_610_000_000 + j (j < 48; six classes x 8 events, the fixed class "
    "order elastic, sd_projectile, sd_target, dd, nd, nd_multi)")
#: The pinned smoke exercise P(n = 2) (a smoke-only fixture value, the
#: R53 drawn value, recorded as such; it exercises both count strata of
#: the card-8 generation and is never a production value).
SMOKE_EXERCISE_P_N2 = 0.25

#: The reserved round-5 raw seed blocks (index item 9; recorded so the
#: disjointness check covers the reserved neighbors explicitly; the
#: R50-R53 blocks are now used and recorded below).
RESERVED_ROUND5_BLOCKS = {
    "r50": [20300925, 20300944],
    "r51": [20300945, 20300952],
    "r52": [20300953, 20300960],
    "r53": [20300961, 20300968],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9, plus this
#: driver's smoke class-exercise partition).
RESERVED_ROUND5_PARTITIONS = {
    "r50": [1_000_000_000, 1_000_025_000],
    "r51": [1_100_000_000, 1_100_100_000],
    "r52": [1_200_000_000, 1_200_100_000],
    "r53": [1_300_000_000, 1_300_100_000],
    "r54": [1_400_000_000, 1_400_050_000],
    "r55": [1_500_000_000, 1_500_250_000],
    "round5_smoke": [1_600_000_000, 1_600_002_000],
    "round5_smoke_class_exercise": [1_610_000_000, 1_610_000_100],
}

#: The previously-used seed blocks and event-index partitions, recorded
#: from the sealed pre-registrations and the round 5 index item 9 (the
#: disjointness evidence of the card; any (seed, event-index) collision
#: stops the run).
PREVIOUS_USED_BLOCKS = {
    "r23": {
        "master_seed": 20260913,
        "raw_seeds": [20300913, 20300920],
        "event_index_range": None,
        "source": "the sealed R23-era raw block (the R41 disjointness record)",
    },
    "r32": {
        "master_seed": 20260918,
        "raw_seeds": [20301519, 20301826],
        "event_index_range": None,
        "source": (
            "the sealed R40 pre-registration's disjointness record (the "
            "R32 production blocks and confirmation block under the "
            "20260918 master)"
        ),
    },
    "round4_production": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301083],
        "event_index_range": None,
        "source": (
            "the round 4 index card: R40 20301020-20301023, R41/R42/R43 "
            "20301020-20301027, R44 20301020-20301039, R45 "
            "20301056-20301075, R46 20301076-20301083"
        ),
    },
    "round4_smoke": {
        "master_seed": 20260920,
        "raw_seeds": [20307920, 20311920],
        "event_index_range": None,
        "source": (
            "the round 4 index card: the smoke blocks 20307920-20311920 "
            "(raw offsets +7000 through +11000)"
        ),
    },
    "r46_smoke": {
        "master_seed": 20260920,
        "raw_seeds": [20312920, 20312920],
        "event_index_range": None,
        "source": "the sealed R46 pre-registration smoke block",
    },
    "r40_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301023],
        "event_index_range": [300_000_000, 389_025_000],
        "event_index_partition": (
            "300_000_000 + s*10_000_000 + p*1_000_000 + i, s < 9, p < 10, "
            "i < 25,000 (stays below 389,025,000)"),
        "source": ("sealed dev/00runs/"
                   "r40_fixed_source_alignment_scan_2026-09-21/"
                   "pre_registration.json"),
    },
    "r41_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_range": [400_000_000, 400_100_000],
        "event_index_partition": "400_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r41_nonstrange_pilot_fit_2026-09-23/"
                   "pre_registration.json"),
    },
    "r42_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_range": [500_000_000, 500_100_000],
        "event_index_partition": "500_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r42_recoil_decay_separation_2026-09-23/"
                   "pre_registration.json"),
    },
    "r43_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_range": [600_000_000, 600_100_000],
        "event_index_partition": "600_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r43_cascade_ablation_2026-09-23/"
                   "pre_registration.json"),
    },
    "r44_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301039],
        "event_index_range": [700_000_000, 700_250_000],
        "event_index_partition": "700_000_000 + i, i < 250,000",
        "source": ("sealed dev/00runs/"
                   "r44_frozen_nonstrange_validation_2026-09-23/"
                   "pre_registration.json"),
    },
    "r45_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301056, 20301075],
        "event_index_range": [800_000_000, 800_250_000],
        "event_index_partition": "800_000_000 + i, i < 250,000",
        "source": ("sealed dev/00runs/"
                   "r45_simplified_alignment_freeze_2026-09-24/"
                   "pre_registration.json"),
    },
    "r46_partition": {
        "master_seed": 20260920,
        "raw_seeds": [20301076, 20301083],
        "event_index_range": [900_000_000, 900_100_000],
        "event_index_partition": "900_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r46_simplified_recoil_decay_profile_2026-09-24/"
                   "pre_registration.json"),
    },
    "round4_smoke_partitions": {
        "master_seed": 20260920,
        "raw_seeds": None,
        "event_index_range": [400_000_000, 400_100_000],
        "event_index_partition": (
            "the Round 4 smoke partitions: R41 400_000_000 + i (slot "
            "8000), R42 500_000_000 + i (slot 9000), R43 690_000_000 + i, "
            "R44 790_000_000 + i, R45 890_000_000 + i, R46 990_000_000 + "
            "i (each i < 2,000; the recorded union with the R41 range is "
            "the loosest bound the record carries)"),
        "source": ("the sealed R41-R46 pre-registrations (the round 5 "
                   "index item 9)"),
    },
    "r50_partition": {
        "master_seed": 20260925,
        "raw_seeds": [20300925, 20300944],
        "event_index_range": [1_000_000_000, 1_000_025_000],
        "event_index_partition": "1_000_000_000 + i, i < 25,000 (per slot)",
        "source": ("sealed dev/00runs/"
                   "r50_pure_topology_templates_2026-09-27/"
                   "pre_registration.json"),
    },
    "r51_partition": {
        "master_seed": 20260925,
        "raw_seeds": [20300945, 20300952],
        "event_index_range": [1_100_000_000, 1_100_100_000],
        "event_index_partition": "1_100_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r51_topology_fraction_fit_2026-09-27/"
                   "pre_registration.json"),
    },
    "r52_partition": {
        "master_seed": 20260925,
        "raw_seeds": [20300953, 20300960],
        "event_index_range": [1_200_000_000, 1_200_100_000],
        "event_index_partition": "1_200_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r52_central_excitation_necessity_2026-09-27/"
                   "pre_registration.json"),
    },
    "r53_partition": {
        "master_seed": 20260925,
        "raw_seeds": [20300961, 20300968],
        "event_index_range": [1_300_000_000, 1_300_100_000],
        "event_index_partition": "1_300_000_000 + i, i < 100,000",
        "source": ("sealed dev/00runs/"
                   "r53_multiple_neutral_excitation_2026-09-27/"
                   "pre_registration.json"),
    },
}

#: The clean-environment validation's pre-registered probe replica (the
#: replica 0 mixed draw; the probe re-run is the determinism evidence of
#: acceptance (v)).
PROBE_REPLICA = 0

# ---------------------------------------------------------------------------
# The topology sampling (index item 6; the recorded source)
# ---------------------------------------------------------------------------
R50_RUN_DIR = "dev/00runs/r50_pure_topology_templates_2026-09-27"
R51_RUN_DIR = "dev/00runs/r51_topology_fraction_fit_2026-09-27"
R53_RUN_DIR = "dev/00runs/r53_multiple_neutral_excitation_2026-09-27"
R51_FIT_REPORT_NAME = "fit_report.json"
R53_GATE_NAME = "gate5_r53.json"
R50_PREREGISTRATION_NAME = "pre_registration.json"

#: The frozen R51 fitted point (asserted byte-exactly against the sealed
#: R51 fit_report.json before any generation; identical to the R52/R53
#: drivers' FROZEN_FIT pin).
FROZEN_FIT: dict[str, float] = {
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
    "dd_fraction": 0.9999997010778905,
    "nd_fraction": 3.1345895878942542e-09,
    "epsilon": 0.0033443027832747944,
    "b_sd": 1.8457918026393214,
}

#: The R51 frozen inelastic fractions (the draw surface; keyed by the
#: runner's class labels; asserted byte-exactly at generation).
R51_FROZEN_FRACTIONS: dict[str, float] = {
    "sd_projectile": FROZEN_FIT["sd_projectile_fraction"],
    "sd_target": FROZEN_FIT["sd_target_fraction"],
    "dd": FROZEN_FIT["dd_fraction"],
    "nd": FROZEN_FIT["nd_fraction"],
}

#: The elastic fraction of the mixed draw: the complement of the R51
#: frozen inelastic mass (the runner's ``pinned_trace`` default rule,
#: computed from the same sealed source).  At the sealed R51 point the
#: four fitted fractions sum to one exactly in float arithmetic, so the
#: complement is exactly 0.0 -- the same value the sealed R51 sample's
#: elastic convention records (``elastic_events`` 0 at the elastic
#: fraction exactly 0.0).  The elastic class is included in the
#: generation surface (dispatchable, audited, gate-recorded); at this
#: recorded fraction no elastic event realizes in production and the
#: draw-count check requires the realized elastic count to be exactly
#: zero.
ELASTIC_FRACTION_GENERATION = 1.0 - math.fsum(R51_FROZEN_FRACTIONS.values())

#: The unretained default P(n = 2) (the R53-PENDING default; the card-8
#: DEFAULT_P2, asserted equal in the surface check).
R53_P_N2_DEFAULT = 0.0

#: The per-class draw-count MC tolerance (the recorded MC tolerance of
#: acceptance (i): the realized count within 5.0 binomial standard
#: deviations of the expected count n * f; a class with expected count
#: exactly 0 must realize exactly 0; a divergence is a named deviation).
MC_TOLERANCE_SIGMA = 5.0

#: The event four-momentum closure tolerance (the house conservation
#: contract).
EVENT_CLOSURE_TOL = 1e-9

#: The five ledger rules (the card's assertion block; the gate record's
#: rule list) and their audit_event message tags.
AUDIT_RULES: tuple[str, ...] = (
    "closed_set_membership",
    "vacuum_zero_transfer",
    "inheritance_after_declared_transfer",
    "local_decay_closure",
    "central_source_neutrality",
)
AUDIT_RULE_TAGS: dict[str, str] = {
    "(a)": "closed_set_membership",
    "(b)": "vacuum_zero_transfer",
    "(c)": "inheritance_after_declared_transfer",
    "(d)": "local_decay_closure",
    "(e)": "central_source_neutrality",
}

#: The pinned incoming ledger lines of the M8 p+p configuration: the two
#: proton lines (B, Q, S, 2I3) = (1, 1, 0, 1); the total (2, 2, 0, 2)
#: (the card-2 pre-registered instantiation).  The smoke class exercise
#: cross-checks this pair against the elastic module's own incoming
#: record.
INCOMING_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

#: The class order of the inelastic walk (the runner's fixed
#: INELASTIC_ORDER) and the smoke exercise's fixed class order (the
#: registry's naming order with the elastic class first).
CLASS_ORDER: tuple[str, ...] = tuple(INELASTIC_ORDER)
SMOKE_EXERCISE_CLASSES: tuple[str, ...] = (
    "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi",
)

#: The sealed R52 trigger and the R51 named-residual kinds (the
#: inherited model-uncertainty caveat block of the frozen-point caveat).
R51_NAMED_RESIDUAL_KINDS = (
    "optimizer_non_convergence",
    "replica_block_minima_disagree",
    "block_curvature_failure",
    "weight_degeneracy",
)

INCOMING_REFERENCE_NOTE = (
    "the incoming sum of the pinned M8 p+p configuration: "
    "(E_beam + m_p, 0, 0, p_beam) with E_beam = sqrt(m_p^2 + p_beam^2), "
    "m_p = 0.93827231 GeV, p_beam = 158.0 GeV/c; every event's final "
    "state closes against this reference at tolerance 1e-9 relative"
)

SUBSTREAM_COLLISION_RULE = (
    "if any raw (seed, event-index) substream collides with a "
    "pre-registered block, the run stops and reports; a silent re-block "
    "never happens (index item 9)")


class LedgerAuditViolation(RuntimeError):
    """A nonzero ledger-audit violation of the card's assertion (the
    five ledger rules of plan section 11.4; checked, never repaired: a
    violation is a named finding and fails the run)."""


ConservationViolation = r51.ConservationViolation


def replica_seed(replica: int) -> int:
    """The raw seed of replica *replica* (the pre-registered R54
    block)."""
    k = int(replica)
    if not 0 <= k < N_REPLICAS:
        _refuse(f"replica index {k} outside the R54 block "
                f"0..{N_REPLICAS - 1}")
    return REPLICA_SEEDS[k]


def replica_name(replica: int) -> str:
    return RECORD_FMT.format(int(replica))


def event_id(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_id(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


# ---------------------------------------------------------------------------
# The interpretation notes consumed at module level (the recorded
# instantiation of the card's elastic-inclusion and R53-PENDING clauses)
# ---------------------------------------------------------------------------
ELASTIC_INCLUSION_NOTE = (
    "the elastic class is included in the generation (the 'all "
    "topologies' of plan line 930 includes the elastic reference): the "
    "elastic class is dispatchable through the card-9 run_topology, its "
    "events are audited by the same five-rule machinery (the smoke pins "
    "the elastic dispatch and audit explicitly, and the elastic "
    "zero-transfer code check runs on every generated elastic event), "
    "and the gate evidence records its fraction.  The elastic fraction "
    "is recorded from the same source as the inelastic fractions: it is "
    "the complement of the R51 frozen inelastic mass (the runner's "
    "pinned_trace default rule, computed from the same sealed source), "
    "which at the sealed R51 point is exactly 0.0 because the four "
    "fitted inelastic fractions sum to one exactly in float arithmetic "
    "(the same value the sealed R51 sample's elastic convention "
    "records).  At the recorded fractions no elastic event realizes in "
    "production and the draw-count check requires the realized elastic "
    "count to be exactly zero; a positive elastic fraction would "
    "renormalize the asserted R51 values and is a named deviation the "
    "card does not sanction.")

REAUDIT_OBLIGATION_NOTE = (
    "if R53 later seals with retained: true, this pre-registration is "
    "invalidated: the audit re-registers under a new date with the "
    "sealed fitted P(n = 2) and re-executes (the re-audit obligation; "
    "the run identity re-pins; the generation refuses the "
    "pending-to-retained flip against a standing pre-registration).  A "
    "sealed retained: false confirms the recorded unretained default "
    "0.0 and the run stands.")


# ---------------------------------------------------------------------------
# The audit-surface invariants (module level; in-memory only)
# ---------------------------------------------------------------------------
def assert_audit_surface() -> dict:
    """The audit-surface invariants (module level): the closed
    allowed-exchange set is exactly the vacuum exchange; the six-entry
    class registry with the elastic class non-inelastic and the
    nd_multi class unretained in code; the R51 frozen fractions
    non-negative and summing to one exactly in float arithmetic; the
    elastic fraction the exact complement 0.0; the R53-PENDING default
    equal to the card-8 unretained default; the nominal-parameter
    reference intact."""
    if ALLOWED_EXCHANGES != frozenset({VACUUM}):
        _refuse("the closed allowed-exchange set drifted off {vacuum}")
    if tuple(VACUUM.transfer) != tuple(ZERO_TRANSFER) or \
            tuple(ZERO_TRANSFER) != (0, 0, 0, 0):
        _refuse("the vacuum exchange's zero transfer drifted")
    if set(TOPOLOGY_CLASSES) != {
            "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}:
        _refuse(f"the class registry drifted: {sorted(TOPOLOGY_CLASSES)!r}")
    if TOPOLOGY_CLASSES["elastic"]["inelastic"]:
        _refuse("the elastic registry entry must carry inelastic: False")
    if TOPOLOGY_CLASSES["nd_multi"]["retained"]:
        _refuse("the nd_multi registry entry must carry retained: False "
                "(the code surface stays unretained; the R53 outcome "
                "carries the retention)")
    if set(INELASTIC_SET) != {"sd_projectile", "sd_target", "dd", "nd"}:
        _refuse(f"the inelastic set drifted: {sorted(INELASTIC_SET)!r}")
    if dict(NOMINAL_PARAMS.class_fractions) != {
            "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}:
        _refuse("the runner's nominal inelastic fractions drifted (the "
                "equal-nominal fallback reference)")
    if float(NOMINAL_PARAMS.p_n2) != 0.0:
        _refuse("the runner's nominal p_n2 drifted off the unretained 0.0")
    for label, value in R51_FROZEN_FRACTIONS.items():
        if not math.isfinite(value) or value < 0.0:
            _refuse(f"the R51 frozen fraction {label!r} is not a finite "
                    f"non-negative value: {value!r}")
    inelastic_mass = math.fsum(R51_FROZEN_FRACTIONS.values())
    if inelastic_mass != 1.0:
        _refuse("the R51 frozen inelastic fractions do not sum to one "
                f"exactly in float arithmetic: {inelastic_mass!r}; the "
                "elastic-complement rule refuses an inexact mass")
    if ELASTIC_FRACTION_GENERATION != 0.0:
        _refuse("the elastic fraction (the complement of the R51 frozen "
                f"inelastic mass) is {ELASTIC_FRACTION_GENERATION!r}, "
                "expected exactly 0.0")
    if R53_P_N2_DEFAULT != DEFAULT_P2 or R53_P_N2_DEFAULT != 0.0:
        _refuse("the R53-PENDING default P(n = 2) drifted off the card-8 "
                f"unretained default: {R53_P_N2_DEFAULT!r} vs "
                f"{DEFAULT_P2!r}")
    if COUNT_BND != 0.5 or P_N2_TOLERANCE <= 0.0:
        _refuse("the card-8 nd_multi bound surface drifted")
    if not 0.0 <= SMOKE_EXERCISE_P_N2 <= COUNT_BND:
        _refuse("the pinned smoke exercise P(n = 2) is outside the "
                "card-8 bound")
    if tuple(AUDIT_RULES) != (
            "closed_set_membership", "vacuum_zero_transfer",
            "inheritance_after_declared_transfer", "local_decay_closure",
            "central_source_neutrality"):
        _refuse(f"the five ledger rule names drifted: {AUDIT_RULES!r}")
    if len(AUDIT_RULE_TAGS) != 5 or sorted(AUDIT_RULE_TAGS.values()) != \
            sorted(AUDIT_RULES):
        _refuse("the audit rule tags drifted off the rule names")
    if tuple(INCOMING_QN) != (1, 1, 0, 1):
        _refuse("the pinned incoming proton line drifted off (1, 1, 0, 1)")
    if FROZEN_FIT["b_sd"] != 1.8457918026393214 or \
            FROZEN_FIT["epsilon"] != 0.0033443027832747944:
        _refuse("the frozen mass-law point drifted off the sealed R51 "
                "fitted values")
    return {
        "allowed_exchanges": sorted(e.name for e in ALLOWED_EXCHANGES),
        "vacuum_transfer": list(ZERO_TRANSFER),
        "class_registry": sorted(TOPOLOGY_CLASSES),
        "elastic_inelastic": TOPOLOGY_CLASSES["elastic"]["inelastic"],
        "nd_multi_retained_in_code": TOPOLOGY_CLASSES["nd_multi"]["retained"],
        "inelastic_set": sorted(INELASTIC_SET),
        "class_order": list(CLASS_ORDER),
        "r51_frozen_fractions": dict(R51_FROZEN_FRACTIONS),
        "r51_inelastic_mass": inelastic_mass,
        "elastic_fraction": ELASTIC_FRACTION_GENERATION,
        "elastic_inclusion_note": ELASTIC_INCLUSION_NOTE,
        "r53_p_n2_default": R53_P_N2_DEFAULT,
        "card8_bound": COUNT_BND,
        "incoming_qn": list(INCOMING_QN),
        "audit_rules": list(AUDIT_RULES),
        "mc_tolerance_sigma": MC_TOLERANCE_SIGMA,
        "pass": True,
    }


_FIT_SURFACE_STATUS = assert_audit_surface()


# ---------------------------------------------------------------------------
# Small deterministic JSON / hash helpers
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(root: Path) -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root),
            capture_output=True, text=True, timeout=30,
        ).stdout
        return {"head": head, "branch": branch, "dirty": bool(dirty.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "branch": None, "dirty": None}


def environment_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "machine": platform.machine(),
    }


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)


def _load_families() -> list:
    return families_from_json(str(NESTED_ROOT / FAMILIES_REL))


def verify_frozen_substrate() -> dict:
    """Verify the frozen substrate digests (byte-unchanged inputs; the
    card-pinned surfaces and the R45-R53 driver pins)."""
    status: dict[str, dict] = {}
    for key, (rel, expected) in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        if not path.is_file():
            _refuse(f"frozen substrate file missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            _refuse(
                f"frozen substrate digest drift for {key} ({rel}): "
                f"expected {expected}, actual {actual}; the run refuses "
                "to touch a drifted surface"
            )
        status[key] = {"sha256": actual, "verified": True}
    return status


def round5_module_digests() -> dict:
    """The landed Round 5 module digests (read-only inputs, recorded in
    the provenance)."""
    return {
        name: _sha256_file(MONO_ROOT / rel)
        for name, rel in ROUND5_MODULES.items()
    }


def m8_frozen_set_status() -> dict:
    """The M8 frozen ten-parameter set: read from the legacy_v1
    manifest's frozen map (the freeze test's canonical form), verified
    against the pre-registered hash, and asserted against the frozen
    mass-law point and the fixed blend ramp start."""
    manifest_path = (NESTED_ROOT / "config" / "v2_singlet"
                     / "legacy_v1_manifest.json")
    if not manifest_path.is_file():
        _refuse(f"the legacy_v1 manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    computed_hash = hashlib.sha256(
        json.dumps(m8, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()
    if computed_hash != M8_PARAMETER_HASH:
        _refuse(
            "the M8 frozen parameter map hash drifted: expected "
            f"{M8_PARAMETER_HASH}, actual {computed_hash}"
        )
    slope = float(m8["regge.effective_transfer_slope"])
    if slope != ELASTIC_TRANSFER_SLOPE:
        _refuse(
            "the M8-frozen effective_transfer_slope "
            f"{slope!r} is not byte-identical to the card-1 module "
            f"constant {ELASTIC_TRANSFER_SLOPE!r}"
        )
    boundary = float(m8["continuum.match_mass"])
    if boundary != BLEND_RAMP_START_GEV:
        _refuse(
            "the M8-frozen kernel boundary "
            f"{boundary!r} is not the fixed blend ramp start "
            f"{BLEND_RAMP_START_GEV!r} (the blend ramp start is never a "
            "free parameter)"
        )
    return {
        "hash": computed_hash,
        "verified": True,
        "effective_transfer_slope_gev2": slope,
        "blend_ramp_start_gev": boundary,
        "source": ("config/v2_singlet/legacy_v1_manifest.json "
                   "(frozen_fit_outputs.m8_frozen_pp_parameters)"),
    }


def verify_sealed_dir(rel: str) -> dict:
    """Verify a sealed run dir's SHA256SUMS.txt entry by entry (the
    verify-sealed discipline; the archive must be intact before it is
    consumed read-only)."""
    return r51.verify_sealed_dir(rel)


def verify_sealed_inputs() -> dict:
    """The sealed-input verification: the R50 archive (the nominal-
    parameter reference), the R51 archive (the frozen fractions), and
    the R53 archive when its sealed gate exists (the nd_multi retention
    outcome).  The R53-PENDING state (the archive absent or not yet
    sealed) is recorded, never refused: the card consumes the R53
    outcome when run, else the unretained default."""
    out = {
        "r50": verify_sealed_dir(R50_RUN_DIR),
        "r51": verify_sealed_dir(R51_RUN_DIR),
    }
    gate_path = MONO_ROOT / R53_RUN_DIR / R53_GATE_NAME
    sums_path = MONO_ROOT / R53_RUN_DIR / SHA256SUMS_NAME
    if gate_path.is_file() and sums_path.is_file():
        out["r53"] = verify_sealed_dir(R53_RUN_DIR)
    else:
        out["r53"] = {
            "run_dir": R53_RUN_DIR,
            "state": "R53-PENDING",
            "note": ("the sealed R53 archive is absent or not yet sealed "
                     "(the R53 generation is in flight); the unretained "
                     "default P(n = 2) = 0.0 applies with the re-audit "
                     "obligation"),
        }
    return out


# ---------------------------------------------------------------------------
# The seed-block disjointness (the card's stop condition)
# ---------------------------------------------------------------------------
def seed_disjointness_report() -> dict:
    """The seed-block disjointness check: the exact previously-used
    blocks recorded from the sealed pre-registrations and the round 5
    index item 9, and the event-index partition disjointness.  Any
    (seed, event-index) substream collision refuses the run (never a
    silent re-block)."""
    problems: list[str] = []
    r54_lo = EVENT_INDEX_BASE
    r54_hi = EVENT_INDEX_BASE + EVENTS_PER_REPLICA
    shared: dict[str, list[int]] = {}
    for name, block in PREVIOUS_USED_BLOCKS.items():
        seeds = block.get("raw_seeds")
        if seeds is not None:
            lo, hi = int(seeds[0]), int(seeds[-1])
            overlap = sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS))
            if overlap:
                shared[name] = [int(overlap[0]), int(overlap[-1])]
        p_range = block.get("event_index_range")
        if p_range is not None:
            p_lo, p_hi = p_range
            if r54_lo < p_hi and p_lo < r54_hi:
                problems.append(
                    f"the R54 event-index partition [{r54_lo}, {r54_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if name == "r54":
            continue
        if sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS)):
            problems.append(
                f"the R54 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if name == "r54":
            continue
        if r54_lo < hi and lo < r54_hi:
            problems.append(
                f"the R54 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r54] seed disjointness: the raw seeds {lo}..{hi} are "
                f"shared with {name} by the established pre-registration "
                "convention; the (seed, event-index) substreams stay "
                "disjoint because the event-index partitions are disjoint",
                flush=True,
            )
    if problems:
        _refuse(
            "the seed-block disjointness check failed: "
            + "; ".join(problems)
            + " (a raw (seed, event-index) substream collision stops the "
            "run; never a silent re-block)"
        )
    return {
        "r54_raw_seed_block": [min(REPLICA_SEEDS), max(REPLICA_SEEDS)],
        "r54_raw_seed_rule": (
            "raw replica seed = 20260925 + 40000 + 44 + k = 20300969 + k "
            f"per replica k < {N_REPLICAS} (index item 9); the block "
            "20300969-20300988 is disjoint from every previously used raw "
            "block and from every other reserved Round 5 block"),
        "r54_replica_seeds": list(REPLICA_SEEDS),
        "r54_event_index_partition": EVENT_INDEX_RULE,
        "r54_event_index_range": [r54_lo, r54_hi],
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R54 raw block 20300969-20300988 shares no raw seed with "
            "any previously used block (the R50 block 20300925-20300944, "
            "the R51 block 20300945-20300952, the R52 block 20300953-"
            "20300960, and the R53 block 20300961-20300968 all end below "
            "it; the R55 reserved block 20300989-20301008, the Round 4 "
            "blocks 20301020-20301083, and the smoke blocks 20307920-"
            "20311920 sit above it; the R32 blocks 20301519-20301826 "
            "under the 20260918 master are disjoint); the substreams are "
            "disjoint by construction (both the numpy.random.SeedSequence "
            "spawn tree and derive_substreams(seed, 0, event_id) key on "
            "the raw seed and the event index) and the event-index "
            "partitions are disjoint"),
        "reserved_round5_blocks": {
            name: list(block)
            for name, block in RESERVED_ROUND5_BLOCKS.items()},
        "reserved_round5_partitions": {
            name: list(block)
            for name, block in RESERVED_ROUND5_PARTITIONS.items()},
        "previously_used_blocks": PREVIOUS_USED_BLOCKS,
        "disjoint": True,
    }


def disjointness_report_for_smoke() -> dict:
    """The smoke pre-registration's disjointness record (identical to
    the production check; the smoke seed, partition, and class-exercise
    partition are additionally disjoint from the R54 production block
    and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(REPLICA_SEEDS):
        _refuse("the smoke seed collides with the R54 production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_REPLICA):
        _refuse("the smoke event-index partition overlaps the R54 "
                "production partition")
    exercise_lo = SMOKE_EXERCISE_INDEX_BASE
    exercise_hi = exercise_lo + SMOKE_EXERCISE_EVENTS_PER_CLASS * len(
        SMOKE_EXERCISE_CLASSES)
    for name, (lo, hi) in (
            ("r54 production", (EVENT_INDEX_BASE,
                                EVENT_INDEX_BASE + EVENTS_PER_REPLICA)),
            ("round5 smoke", (smoke_lo, smoke_hi))):
        if exercise_lo < hi and lo < exercise_hi:
            _refuse(
                f"the smoke class-exercise partition overlaps the {name} "
                "partition")
    report["smoke_raw_seed"] = SMOKE_SEED
    report["smoke_event_index_range"] = [smoke_lo, smoke_hi]
    report["smoke_class_exercise_index_rule"] = SMOKE_EXERCISE_INDEX_RULE
    report["smoke_class_exercise_index_range"] = [exercise_lo, exercise_hi]
    report["smoke_disjoint_from_production"] = True
    return report


# ---------------------------------------------------------------------------
# The frozen-fraction assertion against the sealed R51 (before any
# generation)
# ---------------------------------------------------------------------------
def assert_fractions_against_r51() -> dict:
    """The topology sampling fractions are the sealed R51 fitted point
    exactly (the card: "the sealed R51 ``fit_report.json`` values,
    asserted at generation"; the fit report's seal verifies first)."""
    run_dir = MONO_ROOT / R51_RUN_DIR
    fit_report_path = run_dir / R51_FIT_REPORT_NAME
    if not fit_report_path.is_file():
        _refuse(f"the sealed R51 fit report is missing: {fit_report_path}")
    seal = verify_sealed_dir(R51_RUN_DIR)
    fit_report = _read_json(fit_report_path)
    fitted = fit_report.get("fitted") or {}
    best = (fit_report.get("optimizer") or {}).get("best_point") or {}
    for name in ("sd_projectile_fraction", "sd_target_fraction",
                 "dd_fraction", "epsilon", "b_sd"):
        if float(fitted.get(name, float("nan"))) != FROZEN_FIT[name]:
            _refuse(
                f"the frozen {name} {FROZEN_FIT[name]!r} is not "
                f"byte-identical to the sealed R51 fitted value "
                f"{fitted.get(name)!r}")
        if float(best.get(name, float("nan"))) != FROZEN_FIT[name]:
            _refuse(
                f"the sealed R51 best_point {name} {best.get(name)!r} "
                f"differs from the fitted block {fitted.get(name)!r}")
    if float(fitted.get("nd_fraction", float("nan"))) != \
            FROZEN_FIT["nd_fraction"]:
        _refuse("the frozen f_nd is not byte-identical to the sealed R51 "
                f"nd_fraction: {fitted.get('nd_fraction')!r}")
    if fit_report.get("run") != "R51" or fit_report.get("mode") != \
            "production":
        _refuse("the R51 fit report is not the sealed production record "
                f"(run {fit_report.get('run')!r}, "
                f"mode {fit_report.get('mode')!r})")
    return {
        "source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
        "fit_report_sha256": _sha256_file(fit_report_path),
        "seal": seal,
        "r51_fitted": {name: float(fitted[name]) for name in
                       ("sd_projectile_fraction", "sd_target_fraction",
                        "dd_fraction", "epsilon", "b_sd")},
        "r51_nd_fraction": float(fitted["nd_fraction"]),
        "r51_optimizer_converged": bool(
            (fit_report.get("optimizer") or {}).get("converged")),
        "r51_frozen_caveat": (
            "the frozen point is the R51 optimum, which did not converge "
            "and sits at the dd boundary with f_nd ~ 3e-09; the R51 named "
            "residuals (optimizer_non_convergence, "
            "replica_block_minima_disagree, block_curvature_failure, "
            "weight_degeneracy) are carried into this audit as the "
            "inherited model-uncertainty caveats"),
        "r51_named_residual_kinds": list(R51_NAMED_RESIDUAL_KINDS),
        "match": True,
    }


def assert_nominal_reference_against_r50() -> dict:
    """The nominal-parameter reference (the equal-nominal fallback's
    anchor): the sealed R50 pre-registration's nominal parameters (the
    0.25-each inelastic fractions, the nominal epsilon 0.0, the
    M8-frozen B_SD, the elastic fraction 0.0, the unretained p_n2 0.0)
    against the runner's NOMINAL_PARAMS."""
    pre_path = MONO_ROOT / R50_RUN_DIR / R50_PREREGISTRATION_NAME
    if not pre_path.is_file():
        _refuse(f"the sealed R50 pre-registration is missing: {pre_path}")
    pre = _read_json(pre_path)
    nominal = pre.get("nominal_parameters") or {}
    if dict(nominal.get("class_fractions") or {}) != {
            "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}:
        _refuse("the sealed R50 nominal inelastic fractions drifted: "
                f"{nominal.get('class_fractions')!r}")
    if float(nominal.get("epsilon", float("nan"))) != \
            float(NOMINAL_PARAMS.epsilon):
        _refuse("the sealed R50 nominal epsilon is not the runner's "
                f"nominal: {nominal.get('epsilon')!r}")
    if float(nominal.get("b_sd_gev2", float("nan"))) != \
            float(NOMINAL_PARAMS.b_sd):
        _refuse("the sealed R50 nominal b_sd is not byte-identical to the "
                f"runner's nominal: {nominal.get('b_sd_gev2')!r}")
    if float(nominal.get("elastic_fraction", float("nan"))) != 0.0:
        _refuse("the sealed R50 nominal elastic fraction is not 0.0")
    if float(nominal.get("p_n2", float("nan"))) != 0.0:
        _refuse("the sealed R50 nominal p_n2 is not 0.0")
    if pre.get("run") != "R50" or pre.get("mode") != "production":
        _refuse("the R50 pre-registration is not the sealed production "
                f"record (run {pre.get('run')!r}, mode {pre.get('mode')!r})")
    return {
        "source": f"{R50_RUN_DIR}/{R50_PREREGISTRATION_NAME}",
        "r50_nominal": {
            "fractions": {"sd_projectile": 0.25, "sd_target": 0.25,
                          "dd": 0.25, "nd": 0.25},
            "epsilon": float(nominal["epsilon"]),
            "b_sd_gev2": float(nominal["b_sd_gev2"]),
            "elastic_fraction": 0.0,
            "p_n2": 0.0,
        },
        "role": ("the nominal-parameter reference (the equal-nominal "
                 "fallback's anchor; the R54 sampling runs at the R51 "
                 "frozen fractions, so the reference is recorded, not "
                 "drawn)"),
        "match": True,
    }


# ---------------------------------------------------------------------------
# The R53 outcome (the R53-PENDING default and the re-audit obligation)
# ---------------------------------------------------------------------------
def read_r53_outcome() -> dict:
    """The R53 outcome (the nd_multi retention decision): the sealed
    R53 ``gate5_r53.json`` entry (status, fitted P(n = 2), retained)
    when the sealed archive exists (its seal verifies); the R53-PENDING
    default (retained false, P(n = 2) = 0.0, the unretained default)
    otherwise, with the re-audit obligation."""
    run_dir = MONO_ROOT / R53_RUN_DIR
    gate_path = run_dir / R53_GATE_NAME
    sums_path = run_dir / SHA256SUMS_NAME
    if not (run_dir.is_dir() and gate_path.is_file()
            and sums_path.is_file()):
        return {
            "status": "R53-PENDING",
            "pending": True,
            "retained": False,
            "p_n2": R53_P_N2_DEFAULT,
            "source": None,
            "reaudit_obligation": REAUDIT_OBLIGATION_NOTE,
            "note": (
                "the sealed R53 archive is absent or not yet sealed (the "
                "R53 generation is in flight); the card consumes the R53 "
                "outcome when run, else the unretained default P(n = 2) = "
                "0.0 and nd_multi events are absent"),
        }
    seal = verify_sealed_dir(R53_RUN_DIR)
    gate = _read_json(gate_path)
    if gate.get("run") != "R53" or not gate.get("sealed"):
        _refuse("the sealed R53 gate record is not the sealed R53 "
                f"production evidence (run {gate.get('run')!r}, sealed "
                f"{gate.get('sealed')!r})")
    status = gate.get("status")
    if status not in ("TRIGGERED", "NOT_TRIGGERED"):
        _refuse(f"the sealed R53 gate status drifted: {status!r}")
    retained = bool(gate.get("retained"))
    fitted = gate.get("fitted") or {}
    if retained:
        if "p_n2" not in fitted:
            _refuse("the sealed R53 gate retains nd_multi without a "
                    "fitted P(n = 2)")
        p_n2 = float(fitted["p_n2"])
        if not 0.0 <= p_n2 <= COUNT_BND + P_N2_TOLERANCE:
            _refuse("the sealed R53 fitted P(n = 2) is outside the "
                    f"card-8 bound: {p_n2!r}")
    else:
        p_n2 = R53_P_N2_DEFAULT
    return {
        "status": status,
        "pending": False,
        "retained": retained,
        "p_n2": p_n2,
        "source": f"{R53_RUN_DIR}/{R53_GATE_NAME}",
        "gate_file_sha256": _sha256_file(gate_path),
        "seal": seal,
        "reaudit_obligation": REAUDIT_OBLIGATION_NOTE,
        "note": (
            "the sealed R53 outcome is consumed verbatim: the drawn nd "
            "labels dispatch the card-8 nd_multi generation at the "
            "sealed fitted P(n = 2) exactly when the sealed retained "
            "boolean is true; otherwise P(n = 2) = 0.0 and card 7's ND "
            "generation runs"),
    }


def assert_outcome_against_preregistration(outcome: dict, pre: dict) -> dict:
    """The generation-time consistency of the live R53 outcome against
    the pre-registered block: the pending-to-retained flip fires the
    re-audit obligation and refuses (never silently absorbed); a sealed
    retained-false resolution confirms the recorded 0.0; any other
    drift refuses."""
    pre_outcome = (pre.get("topology_sampling") or {}).get("nd_multi") or {}
    pre_pending = bool(pre_outcome.get("pending"))
    pre_retained = bool(pre_outcome.get("retained"))
    pre_p_n2 = float(pre_outcome.get("p_n2", float("nan")))
    now_pending = bool(outcome.get("pending"))
    now_retained = bool(outcome.get("retained"))
    now_p_n2 = float(outcome.get("p_n2", float("nan")))
    if pre_pending and now_pending:
        resolution = "still pending (the R53 archive is not sealed)"
    elif pre_pending and not now_pending and not now_retained:
        resolution = ("resolved: the sealed R53 outcome retains nothing; "
                      "the recorded unretained default 0.0 stands")
    elif pre_pending and now_retained:
        _refuse(
            "the re-audit obligation fires: the pre-registration recorded "
            "the R53-PENDING default P(n = 2) = 0.0 and the sealed R53 "
            f"outcome now retains nd_multi at P(n = 2) = {now_p_n2!r}; "
            "this pre-registration is invalidated (the audit re-registers "
            "under a new date with the sealed fitted P(n = 2) and "
            "re-executes; the run identity re-pins)")
    elif not pre_pending and (pre_retained != now_retained
                              or pre_p_n2 != now_p_n2):
        _refuse(
            "the live R53 outcome drifted against the pre-registered "
            f"block (pre retained {pre_retained!r} / P(n = 2) "
            f"{pre_p_n2!r}, now retained {now_retained!r} / P(n = 2) "
            f"{now_p_n2!r}); the run refuses")
    else:
        resolution = "matches the pre-registered sealed outcome"
    return {
        "pre_registered_pending": pre_pending,
        "pre_registered_retained": pre_retained,
        "pre_registered_p_n2": pre_p_n2,
        "live_pending": now_pending,
        "live_retained": now_retained,
        "live_p_n2": now_p_n2,
        "resolution": resolution,
        "consistent": True,
    }


# ---------------------------------------------------------------------------
# The draw parameters and the sampling surface
# ---------------------------------------------------------------------------
def frozen_draw_params(p_n2: float) -> TopologyNominalParams:
    """The frozen draw parameters: the R51 fitted fractions, epsilon,
    and B_SD, the nominal remnant weights, and P(n = 2) at the R53
    outcome's value (0.0 at the R53-PENDING default)."""
    return TopologyNominalParams(
        epsilon=FROZEN_FIT["epsilon"],
        b_sd=FROZEN_FIT["b_sd"],
        class_fractions=dict(R51_FROZEN_FRACTIONS),
        remnant_weights=dict(NOMINAL_PARAMS.remnant_weights),
        p_n2=float(p_n2),
    )


def sampling_from_outcome(outcome: dict) -> dict:
    """The production sampling surface from the recorded R53 outcome:
    the frozen draw parameters, the recorded elastic fraction, the
    nd_multi dispatch flag (sampled only when retained), and the
    six-entry recorded fractions block (the gate record's shape)."""
    p_n2 = float(outcome["p_n2"])
    nd_multi_active = bool(outcome["retained"]) and p_n2 > 0.0
    params = frozen_draw_params(p_n2 if nd_multi_active else 0.0)
    fractions = {
        "elastic": ELASTIC_FRACTION_GENERATION,
        "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
        "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
        "dd": R51_FROZEN_FRACTIONS["dd"],
        "nd": R51_FROZEN_FRACTIONS["nd"],
        "nd_multi": (p_n2 if nd_multi_active else R53_P_N2_DEFAULT),
    }
    return {
        "params": params,
        "elastic_fraction": ELASTIC_FRACTION_GENERATION,
        "nd_multi_active": nd_multi_active,
        "p_n2": float(params.p_n2),
        "fractions": fractions,
        "outcome": dict(outcome),
    }


# ---------------------------------------------------------------------------
# The per-event audit machinery (card 2's audit_event over the class
# records)
# ---------------------------------------------------------------------------
def exchange_records_of(event) -> list:
    """The event's exchange records as ``(name, (B, Q, S, 2I3))`` pairs,
    read uniformly from the landed class records (the elastic module's
    ExchangeRecord, the SD class's ``(name, transfer)`` tuple, the DD
    class's ``(name, transfer)`` pair, the ND family's ExchangeRecord
    pair)."""
    if isinstance(event, ElasticEvent):
        return [(str(event.exchange.name),
                 tuple(int(v) for v in event.exchange.transfer))]
    if isinstance(event, SDEvent):
        return [(str(event.exchange[0]),
                 tuple(int(v) for v in event.exchange[1]))]
    if isinstance(event, DDEvent):
        return [(str(name), tuple(int(v) for v in transfer))
                for name, transfer in event.exchanges]
    # NDEvent (and the registered-but-unretained NDMultiEvent)
    return [(str(exchange.name), tuple(int(v) for v in exchange.transfer))
            for exchange in event.exchanges]


def _as_ledger_line(line) -> LineState:
    """The card-2 :class:`~na49_mixing.exchange_ledger.LineState` of any
    record line state (the elastic module's qn-tuple form or the
    ledger's named-field form), the four-momentum carried (the ledger
    performs no kinematics; the field exists for event-level audits)."""
    b, q, s, i3x2 = r51._line_qn(line)
    fm = getattr(line, "four_momentum", None)
    four_momentum = None
    if fm is not None:
        e, px, py, pz = (float(fm[0]), float(fm[1]), float(fm[2]),
                         float(fm[3]))
        four_momentum = (e, px, py, pz)
    return LineState(
        B=int(b), Q=int(q), S=int(s), I3x2=int(i3x2),
        four_momentum=four_momentum)


def _central_sources_of(event) -> list:
    """The event's central-source ledger states (the ND class's single
    source; the nd_multi class's one-or-two sources; the other classes
    carry none)."""
    if isinstance(event, NDEvent):
        return [event.central_source]
    if isinstance(event, NDMultiEvent):
        return list(event.central_sources)
    return []


def _as_exchange(name, transfer) -> Exchange:
    """The card-2 :class:`~na49_mixing.exchange_ledger.Exchange` of one
    record exchange (name, transfer) pair (the closed-set membership
    audit consumes the declared records as card-2 objects)."""
    b, q, s, i3x2 = (int(transfer[0]), int(transfer[1]), int(transfer[2]),
                     int(transfer[3]))
    return Exchange(name=str(name), transfer=(b, q, s, i3x2))


def _decay_vertices_of(event, outgoing) -> tuple:
    """The event's decay vertices, realized at the line level: each
    baryonic line's outgoing ledger state paired with the final-state
    hadrons attributed to that line (the class records' ``line``
    labels), plus the central source with its ``"central"`` products on
    the ND family (the nd_multi class's two central sources, each
    carrying ``(0, 0, 0, 0)`` exactly, audited as the combined neutral
    vertex).  The elastic class carries no decay vertex (no excitation,
    no decay exists)."""
    if isinstance(event, ElasticEvent):
        return ()
    by_label: dict[str, list] = {}
    for hadron in event.final_state:
        by_label.setdefault(str(hadron.line), []).append(
            _as_ledger_line(hadron))
    vertices: list = []
    if isinstance(event, SDEvent):
        lines = tuple(outgoing)
        if event.side == "projectile":
            excitation, remnant = lines[0], lines[1]
        else:
            remnant, excitation = lines[0], lines[1]
        if "excitation" in by_label:
            vertices.append((excitation, tuple(by_label["excitation"])))
        if "remnant" in by_label:
            vertices.append((remnant, tuple(by_label["remnant"])))
        missing = {"excitation", "remnant"} - set(by_label)
        if missing:
            raise LedgerAuditViolation(
                f"the SD event record carries no final-state hadrons for "
                f"the line label(s) {sorted(missing)} (checked, never "
                "repaired)")
    else:
        if "projectile" in by_label:
            vertices.append((outgoing[0], tuple(by_label["projectile"])))
        if "target" in by_label:
            vertices.append((outgoing[1], tuple(by_label["target"])))
        if "central" in by_label:
            sources = _central_sources_of(event)
            if not sources:
                raise LedgerAuditViolation(
                    "the event record carries central final-state hadrons "
                    "with no central source (checked, never repaired)")
            total = [0, 0, 0, 0]
            for source in sources:
                for k, value in enumerate(source.qn):
                    total[k] += int(value)
            combined = LineState(B=total[0], Q=total[1], S=total[2],
                                 I3x2=total[3])
            vertices.append((combined, tuple(by_label["central"])))
    return tuple(vertices)


def audit_inputs_from_event(event) -> tuple:
    """The card-2 audit inputs of one live class record: ``(incoming,
    exchanges, outgoing, central_source, decay_vertices)``.  The
    incoming lines are the pinned proton pair (the elastic record's own
    incoming pair, asserted against the pin); the exchanges are built
    as card-2 :class:`~na49_mixing.exchange_ledger.Exchange` objects
    from the records' (name, transfer) pairs (the closed-set membership
    audit consumes the declared records)."""
    if isinstance(event, ElasticEvent):
        first, second = tuple(_as_ledger_line(line)
                              for line in event.incoming)
        incoming = (first, second)
        for line in incoming:
            if line.qn != INCOMING_QN:
                raise LedgerAuditViolation(
                    "the elastic record's incoming line carries "
                    f"{line.qn}, expected the pinned {INCOMING_QN}")
    else:
        incoming = (
            LineState(B=INCOMING_QN[0], Q=INCOMING_QN[1], S=INCOMING_QN[2],
                      I3x2=INCOMING_QN[3]),
            LineState(B=INCOMING_QN[0], Q=INCOMING_QN[1], S=INCOMING_QN[2],
                      I3x2=INCOMING_QN[3]),
        )
    exchanges = tuple(
        _as_exchange(name, transfer)
        for name, transfer in exchange_records_of(event))
    outgoing_list = [_as_ledger_line(line)
                     for line in _baryonic_lines(event)]
    outgoing = (outgoing_list[0], outgoing_list[1])
    sources = _central_sources_of(event)
    central = None
    if sources:
        total = [0, 0, 0, 0]
        for source in sources:
            for k, value in enumerate(source.qn):
                total[k] += int(value)
        central = LineState(B=total[0], Q=total[1], S=total[2],
                            I3x2=total[3])
    vertices = _decay_vertices_of(event, outgoing)
    return incoming, exchanges, outgoing, central, vertices


def audit_block(report) -> dict:
    """The stored audit block of one
    :class:`~na49_mixing.exchange_ledger.LedgerAuditReport`: the
    per-rule violation counts (by the report messages' ``(x)`` tags),
    the violation list, and the pass flag."""
    rules = {name: 0 for name in AUDIT_RULES}
    for violation in report.violations:
        for tag, name in AUDIT_RULE_TAGS.items():
            if violation.startswith(tag):
                rules[name] += 1
                break
        else:
            raise LedgerAuditViolation(
                f"the audit report carries an untagged violation: "
                f"{violation!r} (the five-rule tags are the pre-registered "
                "record surface; checked, never repaired)")
    return {
        "rules": rules,
        "violations": list(report.violations),
        "pass": not report.violations,
    }


def run_ledger_audit(event) -> dict:
    """The card-2 five-rule audit of one live class record
    (:func:`~na49_mixing.exchange_ledger.audit_event` over the record's
    ledger surfaces); returns the stored audit block."""
    incoming, exchanges, outgoing, central, vertices = \
        audit_inputs_from_event(event)
    report = audit_event(incoming, exchanges, outgoing, central, vertices)
    return audit_block(report)


def audit_from_record(record: dict) -> dict:
    """The re-audit from the stored per-event record (the evaluation's
    recheck and the clean-environment validation's mechanism): the
    stored audit inputs (the incoming pair, the exchange records, the
    outgoing lines, the central source, the line-attributed final state)
    rebuild the audit inputs exactly and the five-rule audit re-runs.
    Returns the recomputed audit block; the caller compares it against
    the stored ``au`` block (a mismatch is a named finding)."""
    incoming_list = [
        LineState(B=int(qn[0]), Q=int(qn[1]), S=int(qn[2]), I3x2=int(qn[3]))
        for qn in record["iq"]]
    incoming = (incoming_list[0], incoming_list[1])
    exchanges = tuple(
        _as_exchange(name, transfer) for name, transfer in record["xt"])
    outgoing_list = [
        LineState(B=int(line[4]), Q=int(line[5]), S=int(line[6]),
                  I3x2=int(line[7]))
        for line in record["ln"]]
    outgoing = (outgoing_list[0], outgoing_list[1])
    cq = record.get("cq") or []
    central = None
    if cq:
        total = [0, 0, 0, 0]
        for qn in cq:
            for k in range(4):
                total[k] += int(qn[k])
        central = LineState(B=total[0], Q=total[1], S=total[2],
                            I3x2=total[3])
    vertices: list = []
    if record["cl"] != "elastic":
        by_label: dict[str, list] = {}
        for entry in record["fs"]:
            by_label.setdefault(str(entry[10]), []).append(
                LineState(B=int(entry[1]), Q=int(entry[2]), S=int(entry[3]),
                          I3x2=int(entry[4])))
        lines = [
            LineState(B=int(line[4]), Q=int(line[5]), S=int(line[6]),
                      I3x2=int(line[7]))
            for line in record["ln"]]
        if record["cl"] in ("sd_projectile", "sd_target"):
            side = str(record["side"])
            if side == "projectile":
                excitation, remnant = lines[0], lines[1]
            else:
                remnant, excitation = lines[0], lines[1]
            if "excitation" in by_label:
                vertices.append((excitation, tuple(by_label["excitation"])))
            if "remnant" in by_label:
                vertices.append((remnant, tuple(by_label["remnant"])))
        else:
            if "projectile" in by_label:
                vertices.append((lines[0], tuple(by_label["projectile"])))
            if "target" in by_label:
                vertices.append((lines[1], tuple(by_label["target"])))
            if "central" in by_label:
                if central is None:
                    raise LedgerAuditViolation(
                        "the stored record carries central final-state "
                        "hadrons with no central source (checked, never "
                        "repaired)")
                vertices.append((central, tuple(by_label["central"])))
    report = audit_event(incoming, exchanges, outgoing, central,
                         tuple(vertices))
    return audit_block(report)


# ---------------------------------------------------------------------------
# The per-event record (the full auditable final state; the R50
# full-record convention extended with the line attribution and the
# audit block)
# ---------------------------------------------------------------------------
EVENT_SCHEMA = "na49_mixing.staged_r54_local_ledger_audit.event/1"


def serialize_audit_event(event, obs, conservation, audit) -> dict:
    """The per-event record of the replica file (without its hash): the
    full stored event in the R50 convention (every ledger state and
    every four-momentum as an exact float) extended with the line
    attribution of every final-state hadron, the central-source ledger
    states, the named exchange records, and the generation-time audit
    block, so every stored event is independently re-auditable (the
    evaluation's recheck and the clean-environment validation consume
    exactly these fields)."""
    if isinstance(event, ElasticEvent):
        fs_records: list = []
        lines = list(event.outgoing)
    else:
        fs_records = [
            [
                (None if h.pdg_id is None else int(h.pdg_id)),
                int(h.qn[0]), int(h.qn[1]), int(h.qn[2]), int(h.qn[3]),
                float(h.mass),
                float(h.four_momentum[0]), float(h.four_momentum[1]),
                float(h.four_momentum[2]), float(h.four_momentum[3]),
                str(h.line),
            ]
            for h in event.final_state
        ]
        lines = list(event.lines)
    record = {
        "au": dict(audit),
        "bqs": conservation["bqs"],
        "cl": event.class_label,
        "clo": conservation["closure"],
        "cq": [list(source.qn) for source in _central_sources_of(event)],
        "en": conservation["d_energy_gev"],
        "ex": [float(m) for m in obs.excitation_masses],
        "fs": fs_records,
        "gp": (None if obs.gap_size is None else float(obs.gap_size)),
        "iq": [list(INCOMING_QN), list(INCOMING_QN)],
        "ln": [
            [
                float(line.four_momentum[0]), float(line.four_momentum[1]),
                float(line.four_momentum[2]), float(line.four_momentum[3]),
            ] + list(r51._line_qn(line))
            for line in lines
        ],
        "mo": conservation["d_momentum_gev"],
        "ob": [
            float(obs.x_f_projectile), float(obs.x_f_target),
            int(obs.multiplicity_total), int(obs.multiplicity_central),
            int(obs.central_pions), int(obs.central_protons),
            int(obs.forward_pions), int(obs.forward_protons),
            float(obs.seagull_moment),
        ],
        "sh": conservation["mass_shell_worst"],
        "tr": [float(t) for t in obs.transfers],
        "xt": [[name, list(transfer)]
               for name, transfer in exchange_records_of(event)],
    }
    if isinstance(event, SDEvent):
        record["side"] = str(event.side)
    return record


# ---------------------------------------------------------------------------
# The generation (the mixed-draw replica records at the R51 frozen
# fractions)
# ---------------------------------------------------------------------------
def generate_replica_events(
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
    sampling: dict,
):
    """Generate *n_events* mixed-draw events of one replica at the
    recorded sampling surface and return the per-event records, the
    class census, the count census, the audit aggregate, the
    conservation aggregate, and the event-hash set digest.

    Every event consumes the closed five-label substream quintuple at
    replica slot 0 (the R50/R51/R52/R53 convention: the raw seed carries
    the replica identity), draws its class on the closed
    ``topology_selection`` substream at the recorded fractions (the R51
    frozen fractions; the elastic class included in the draw at its
    recorded fraction), dispatches the card-9 ``run_topology`` (a drawn
    ``nd`` label dispatches the card-8 nd_multi generation exactly when
    the R53 outcome retains the class, else card 7's ND generation),
    extracts the observables, runs the card-2 five-rule audit (a
    violation raises and fails the run: the assertion is zero
    violations, checked, never repaired) and the house conservation
    contract (a violation raises and stops the run)."""
    params = sampling["params"]
    elastic_fraction = float(sampling["elastic_fraction"])
    nd_multi_active = bool(sampling["nd_multi_active"])
    records = []
    census: dict[str, int] = {}
    count_census: dict[str, int] = {"n1": 0, "n2": 0}
    bqs_census: dict[tuple, int] = {}
    worst_closure = 0.0
    worst_shell = 0.0
    worst_energy_gev = 0.0
    worst_momentum_gev = 0.0
    for i in range(int(n_events)):
        subs = derive_substreams(int(seed), 0, event_index_base + i)
        rng_topology_selection = np.random.default_rng(
            subs.stream_for("topology_selection"))
        rng_spectral_state = np.random.default_rng(
            subs.stream_for("spectral_state"))
        rng_exchange_kinematics = np.random.default_rng(
            subs.stream_for("exchange_kinematics"))
        rng_decay_channel = np.random.default_rng(
            subs.stream_for("decay_channel"))
        rng_decay_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics"))
        label = draw_topology(
            R51_FROZEN_FRACTIONS, elastic_fraction, rng_topology_selection)
        if label not in INELASTIC_SET and label != "elastic":
            raise LedgerAuditViolation(
                f"event {event_index_base + i}: the class draw returned "
                f"{label!r} outside the sampled set (the four inelastic "
                f"classes and the elastic class; checked, never repaired)")
        dispatch_label = label
        if label == "nd" and nd_multi_active:
            dispatch_label = "nd_multi"
        event = run_topology(
            dispatch_label, params, catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=params.p_n2)
        obs = extract_observables(event, SQRT_S)
        if isinstance(event, ElasticEvent):
            # the elastic class's pre-registered code check (plan line
            # 824): the zero-transfer ledger audit of the normalization
            # reference; a violation raises and stops the run
            check_zero_transfer(event)
        conservation = r51.event_conservation(event)
        audit = run_ledger_audit(event)
        if not audit["pass"]:
            raise LedgerAuditViolation(
                f"event {event_index_base + i} ({label}): the ledger "
                "audit recorded "
                f"{len(audit['violations'])} violation(s): "
                + "; ".join(audit["violations"]))
        record = serialize_audit_event(event, obs, conservation, audit)
        record["i"] = int(i)
        record["id"] = int(event_index_base + i)
        if isinstance(event, NDMultiEvent):
            record["mu"] = {
                "count": int(event.count),
                "p_n2": float(event.p_n2),
            }
            if event.count == 2:
                count_census["n2"] += 1
            else:
                count_census["n1"] += 1
        record.pop("h", None)
        record["h"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8")).hexdigest()
        census[label] = census.get(label, 0) + 1
        bqs_census[tuple(conservation["bqs"])] = bqs_census.get(
            tuple(conservation["bqs"]), 0) + 1
        worst_closure = max(worst_closure, conservation["closure_worst"])
        worst_shell = max(worst_shell, conservation["mass_shell_worst"])
        worst_energy_gev = max(worst_energy_gev,
                               conservation["d_energy_gev"])
        worst_momentum_gev = max(worst_momentum_gev,
                                 conservation["d_momentum_gev"])
        records.append(record)
    hash_set = hashlib.sha256(
        "\n".join(r["h"] for r in records).encode("utf-8")).hexdigest()
    audit_agg = {
        "events": len(records),
        "violations": 0,
        "rules": {name: 0 for name in AUDIT_RULES},
        "pass": True,
    }
    conservation_agg = {
        "events": len(records),
        "violations": 0,
        "b_q_s_census": {
            ",".join(str(v) for v in k): v
            for k, v in sorted(bqs_census.items())
        },
        "worst_closure_relative": worst_closure,
        "worst_mass_shell_relative": worst_shell,
        "worst_energy_closure_gev": worst_energy_gev,
        "worst_momentum_closure_gev": worst_momentum_gev,
        "tolerance": EVENT_CLOSURE_TOL,
    }
    return records, census, count_census, audit_agg, conservation_agg, \
        hash_set


REPLICA_SCHEMA = "na49_mixing.staged_r54_local_ledger_audit.replica/1"


def _identity_block(run_dir: Path) -> dict:
    return {
        "driver_sha256": DRIVER_SHA256,
        "pre_registration_sha256": _sha256_file(
            run_dir / PREREGISTRATION_NAME),
    }


def build_replica_record(
    replica: int,
    records: list,
    census: dict,
    count_census: dict,
    audit_agg: dict,
    conservation: dict,
    hash_set: str,
    run_dir: Path,
    sampling: dict,
    outcome_resolution: dict | None = None,
    mode: str = "production",
) -> dict:
    """The replica record: the identity block, the seed map, the
    generation fractions, the per-event records, the class and count
    censuses, the audit and conservation aggregates, and the event-hash
    set digest (the generation-time determinism evidence)."""
    return {
        "schema": REPLICA_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "replica": int(replica),
        "seed": replica_seed(replica),
        "n_events": len(records),
        "identity": _identity_block(run_dir),
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 44 + k = 20300969 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300969-20300988 is disjoint from every "
                "previously used raw block and from every other reserved "
                "Round 5 block)"),
            "event_index_rule": EVENT_INDEX_RULE,
            "substream_labels": sorted(SUBSTREAM_LABELS),
            "consumed_labels": [
                "topology_selection", "spectral_state",
                "exchange_kinematics", "decay_channel", "decay_kinematics",
            ],
        },
        "generation_fractions": {
            "note": GENERATION_FRACTIONS_NOTE,
            "class_fractions": dict(R51_FROZEN_FRACTIONS),
            "elastic_fraction": float(sampling["elastic_fraction"]),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": float(sampling["p_n2"]),
            "nd_multi_active": bool(sampling["nd_multi_active"]),
        },
        "r53_outcome_resolution": outcome_resolution,
        "class_census": dict(census),
        "count_census": dict(count_census),
        "audit_aggregate": audit_agg,
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }


def load_pre_registration(run_dir: Path, *, smoke: bool = False) -> dict:
    name = ("smoke_" + PREREGISTRATION_NAME) if smoke else PREREGISTRATION_NAME
    path = Path(run_dir) / name
    if not path.is_file():
        _refuse(f"the run dir carries no pre-registration: {path}")
    return _read_json(path)


def _verify_replica_identity(record: dict, replica: int, run_dir: Path) -> None:
    """A completed production replica record must match the
    pre-registered identity (the seed, the event count, the driver
    digest, and the pre-registration digest); any drift refuses the run
    (a completed record is reused unchanged or refused, never silently
    overwritten)."""
    expected_seed = replica_seed(replica)
    if record.get("mode") != "production":
        _refuse(
            f"the replica record {replica_name(replica)} is not a "
            f"production record (mode {record.get('mode')!r})")
    if record.get("replica") != int(replica):
        _refuse(
            f"the replica record {replica_name(replica)} carries replica "
            f"{record.get('replica')!r}, expected {int(replica)}")
    if record.get("seed") != expected_seed:
        _refuse(
            f"the replica record {replica_name(replica)} carries seed "
            f"{record.get('seed')!r}, expected {expected_seed}")
    if record.get("n_events") != EVENTS_PER_REPLICA or len(
            record.get("events", [])) != EVENTS_PER_REPLICA:
        _refuse(
            f"the replica record {replica_name(replica)} carries "
            f"{record.get('n_events')} events, expected "
            f"{EVENTS_PER_REPLICA}")
    identity = record.get("identity") or {}
    current = _identity_block(run_dir)
    if identity.get("pre_registration_sha256") != current[
            "pre_registration_sha256"]:
        _refuse(
            f"the replica record {replica_name(replica)} was generated "
            "against a different pre-registration digest "
            f"({identity.get('pre_registration_sha256')!r} vs "
            f"{current['pre_registration_sha256']!r}); the run refuses "
            "to mix pre-registrations")
    if identity.get("driver_sha256") != current["driver_sha256"]:
        _refuse(
            f"the replica record {replica_name(replica)} was generated "
            "by a different driver digest "
            f"({identity.get('driver_sha256')!r} vs "
            f"{current['driver_sha256']!r})")
    recomputed = hashlib.sha256(
        "\n".join(r["h"] for r in record["events"]).encode(
            "utf-8")).hexdigest()
    if recomputed != record.get("event_hash_set_sha256"):
        _refuse(
            f"the replica record {replica_name(replica)} carries a stale "
            "event-hash set digest (the stored events do not reproduce "
            "it)")


def run_generation(run_dir: Path, replicas: list[int] | None = None) -> list:
    """Generate the requested production replica records (the default is
    every replica), one record file per replica, then stop.  Requires
    the pre-registration; re-asserts the frozen fractions against the
    sealed R51 archive and the R53 outcome against the pre-registered
    block before generating.  A completed record that matches the
    pre-registered identity is reused unchanged (the pre-registered
    gateway fallback runs one job per replica; the local fallback is one
    single-threaded process per --replica invocation)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    pre = load_pre_registration(run_dir)
    verify_frozen_substrate()
    seed_disjointness_report()
    fraction_assertion = assert_fractions_against_r51()
    outcome = read_r53_outcome()
    resolution = assert_outcome_against_preregistration(outcome, pre)
    sampling = sampling_from_outcome(outcome)
    if replicas is None:
        replicas = list(range(N_REPLICAS))
    written = []
    catalog = _load_catalog()
    fams = _load_families()
    for replica in replicas:
        name = replica_name(replica)
        path = run_dir / name
        if path.is_file():
            existing = _read_json(path)
            _verify_replica_identity(existing, replica, run_dir)
            print(f"[r54] replica {int(replica)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        seed = replica_seed(replica)
        print(
            f"[r54] generating replica {int(replica)} ({name}): seed "
            f"{seed}, {EVENTS_PER_REPLICA} mixed-draw events",
            flush=True,
        )
        (records, census, count_census, audit_agg, conservation,
         hash_set) = generate_replica_events(
            seed, EVENTS_PER_REPLICA, EVENT_INDEX_BASE, catalog, fams,
            sampling)
        record = build_replica_record(
            replica, records, census, count_census, audit_agg, conservation,
            hash_set, run_dir, sampling, outcome_resolution=resolution)
        record["reused"] = False
        _write_json(path, record)
        print(
            f"[r54] replica {int(replica)} written: {path} "
            f"(census {census}, audit pass {audit_agg['pass']}, worst "
            f"closure {conservation['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# The evaluation (the ledger audit and evaluation over the generated
# artifacts; the stored records are re-audited from their stored audit
# inputs)
# ---------------------------------------------------------------------------
def _empty_pooled() -> dict:
    return {
        "events": 0,
        "audit_rules": {name: 0 for name in AUDIT_RULES},
        "violations": 0,
        "stored_audit_mismatches": 0,
        "b_q_s_census": {},
        "class_census": {},
        "count_census": {"n1": 0, "n2": 0},
        "worst_closure_relative": 0.0,
        "worst_mass_shell_relative": 0.0,
        "worst_energy_closure_gev": 0.0,
        "worst_momentum_closure_gev": 0.0,
    }


def _absorb_event(pooled: dict, ev: dict) -> None:
    """Absorb one stored event into the pooled aggregates (the re-audit
    included: the stored audit block must reproduce exactly)."""
    block = audit_from_record(ev)
    stored = ev["au"]
    if (block["rules"] != stored["rules"]
            or block["violations"] != stored["violations"]
            or block["pass"] != stored["pass"]):
        pooled["stored_audit_mismatches"] += 1
    for name, count in block["rules"].items():
        pooled["audit_rules"][name] += int(count)
    pooled["violations"] += int(sum(block["rules"].values()))
    pooled["events"] += 1
    label = str(ev["cl"])
    pooled["class_census"][label] = pooled["class_census"].get(label, 0) + 1
    mu = ev.get("mu")
    if mu is not None:
        key = "n2" if int(mu["count"]) == 2 else "n1"
        pooled["count_census"][key] += 1
    bqs = ",".join(str(int(v)) for v in ev["bqs"])
    pooled["b_q_s_census"][bqs] = pooled["b_q_s_census"].get(bqs, 0) + 1
    pooled["worst_closure_relative"] = max(
        pooled["worst_closure_relative"], max(float(x) for x in ev["clo"]))
    pooled["worst_mass_shell_relative"] = max(
        pooled["worst_mass_shell_relative"], float(ev["sh"]))
    pooled["worst_energy_closure_gev"] = max(
        pooled["worst_energy_closure_gev"], float(ev["en"]))
    pooled["worst_momentum_closure_gev"] = max(
        pooled["worst_momentum_closure_gev"], float(ev["mo"]))


def _absorb_replica_record(pooled: dict, record: dict) -> dict:
    """Absorb one replica record's events into the pooled aggregates and
    return the per-replica summary."""
    for ev in record["events"]:
        _absorb_event(pooled, ev)
    return {
        "replica": record.get("replica"),
        "seed": record.get("seed"),
        "n_events": int(record.get("n_events",
                                   len(record.get("events", [])))),
        "class_census": dict(record.get("class_census") or {}),
        "count_census": dict(record.get("count_census") or {}),
        "event_hash_set_sha256": record.get("event_hash_set_sha256"),
        "audit_aggregate": dict(record.get("audit_aggregate") or {}),
    }


def _draw_count_check(census: dict, fractions: dict, n_events: int) -> dict:
    """The per-class draw-count check at the recorded MC tolerance: the
    realized count within ``MC_TOLERANCE_SIGMA`` binomial standard
    deviations of the expected count ``n * f``; a class with expected
    count exactly 0 must realize exactly 0."""
    per_class = {}
    ok = True
    for label in ("elastic",) + tuple(CLASS_ORDER):
        fraction = float(fractions[label])
        realized = int(census.get(label, 0))
        expected = fraction * float(n_events)
        entry = {"fraction": fraction, "realized": realized,
                 "expected": expected}
        if expected == 0.0:
            entry["sigma"] = None
            entry["pull"] = None
            entry["within_tolerance"] = (realized == 0)
        else:
            variance = float(n_events) * fraction * (1.0 - fraction)
            sigma = math.sqrt(variance)
            pull = (float(realized) - expected) / sigma
            entry["sigma"] = sigma
            entry["pull"] = pull
            entry["within_tolerance"] = abs(pull) <= MC_TOLERANCE_SIGMA
        ok = ok and entry["within_tolerance"]
        per_class[label] = entry
    return {
        "per_class": per_class,
        "mc_tolerance_sigma": MC_TOLERANCE_SIGMA,
        "note": (
            "the per-class marginal binomial check of acceptance (i): the "
            "realized draw count within the recorded MC tolerance of the "
            "expected count n * f at the recorded fractions; a class with "
            "expected count exactly 0 must realize exactly 0; a "
            "divergence is a named deviation (acceptance (iv))"),
        "pass": ok,
    }


def _topology_sampling_block(pre: dict | None, smoke: bool) -> dict:
    """The recorded topology-sampling block of the evaluation report:
    the source, the archive pin, the six fractions, and (production) the
    pre-registered assertion."""
    fractions = {
        "elastic": ELASTIC_FRACTION_GENERATION,
        "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
        "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
        "dd": R51_FROZEN_FRACTIONS["dd"],
        "nd": R51_FROZEN_FRACTIONS["nd"],
        "nd_multi": R53_P_N2_DEFAULT,
    }
    return {
        "source": ("R51 frozen fractions (the sealed "
                   f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME} fitted values; "
                   "asserted at generation)"),
        "archive_pin": R51_RUN_DIR,
        "fractions": fractions,
        "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
        "smoke": smoke,
        "pre_registered_assertion": (
            None if pre is None
            else str((pre.get("topology_sampling") or {}).get("assertion"))),
    }


def run_audit_evaluation(run_dir: Path, *, smoke: bool = False,
                         pre: dict | None = None) -> dict:
    """The ledger audit and evaluation over the generated artifacts:
    every stored event is re-audited from its stored audit inputs (the
    stored audit block must reproduce exactly), the per-rule violation
    counts pool (zero on a passing run), the per-class draw counts are
    checked against the recorded fractions at the recorded MC tolerance,
    and the conservation aggregate pools.  Returns the report dict with
    its digest (no wall-clock field; the bytes are a pure function of
    the sealed artifacts, the pre-registered seeds, and the committed
    code)."""
    run_dir = Path(run_dir)
    pooled = _empty_pooled()
    per_replica = []
    if smoke:
        names = [SMOKE_REPLICA_NAME]
    else:
        names = [replica_name(r) for r in range(N_REPLICAS)]
        for name in names:
            if not (run_dir / name).is_file():
                _refuse(f"the production record is missing: "
                        f"{run_dir / name}")
    for index, name in enumerate(names):
        record = _read_json(run_dir / name)
        if not smoke:
            _verify_replica_identity(record, index, run_dir)
        per_replica.append(_absorb_replica_record(pooled, record))
    if smoke:
        exercise_path = run_dir / SMOKE_EXERCISE_NAME
        if not exercise_path.is_file():
            _refuse(f"the smoke class exercise record is missing: "
                    f"{exercise_path}")
        exercise = _read_json(exercise_path)
    else:
        exercise = None
    if pooled["stored_audit_mismatches"]:
        _refuse(
            "the stored audit blocks do not reproduce under the re-audit "
            f"({pooled['stored_audit_mismatches']} mismatches): the "
            "stored records and their audit blocks disagree (checked, "
            "never repaired)")
    if pooled["violations"]:
        _refuse(
            "the ledger audit recorded "
            f"{pooled['violations']} violation(s) across "
            f"{pooled['events']} events (the assertion is zero "
            "violations; a violation is a named finding and fails the "
            "run)")
    if pooled["b_q_s_census"] and set(pooled["b_q_s_census"]) != {"2,2,0"}:
        _refuse(
            "the conservation census carries a non-(2, 2, 0) event class: "
            f"{sorted(pooled['b_q_s_census'])} (the house conservation "
            "contract)")
    sampling_block = _topology_sampling_block(pre, smoke)
    draw_check = _draw_count_check(
        pooled["class_census"], sampling_block["fractions"], pooled["events"])
    named_residuals: list[dict] = []
    if not draw_check["pass"]:
        named_residuals.append({
            "kind": "topology_sampling_divergence",
            "diagnosis": (
                "the per-class draw counts diverge from the recorded "
                "topology sampling beyond the recorded MC tolerance "
                "(acceptance (i)/(iv); a named deviation, recorded)"),
            "per_class": draw_check["per_class"],
        })
    if pooled["worst_closure_relative"] > EVENT_CLOSURE_TOL or \
            pooled["worst_mass_shell_relative"] > EVENT_CLOSURE_TOL:
        named_residuals.append({
            "kind": "conservation_closure_beyond_tolerance",
            "diagnosis": (
                "a stored event's recorded closure exceeds the tolerance "
                "(the generation raises live; a stored excess is a record "
                "integrity failure)"),
            "worst_closure_relative": pooled["worst_closure_relative"],
            "worst_mass_shell_relative": pooled["worst_mass_shell_relative"],
        })
    report = {
        "schema": "na49_mixing.staged_r54_local_ledger_audit.report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "objective": OBJECTIVE_NOTE,
        "statistics": {
            "tier": ("exactly 1,000,000 events = 20 replicas x 50,000 "
                     "p+p events (index item 6; the T3 replica count, "
                     "plan section 5.1 line 255; plan line 930 requires "
                     "at least one million, met with equality)" if not smoke
                     else "the smoke statistics: one mixed-draw replica x "
                          "2,000 events at the smoke seed 20312925 plus "
                          "the per-class pinned dispatch exercise"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
            "smoke_events": SMOKE_EVENTS if smoke else None,
        },
        "topology_sampling": sampling_block,
        "draw_count_check": draw_check,
        "ledger_audit": {
            "events_checked": pooled["events"],
            "violations": pooled["violations"],
            "pass": pooled["violations"] == 0
                    and pooled["stored_audit_mismatches"] == 0,
            "rules": list(AUDIT_RULES),
            "per_rule_counts": dict(pooled["audit_rules"]),
            "stored_audit_mismatches": pooled["stored_audit_mismatches"],
            "reaudit_note": (
                "every stored event is re-audited from its stored audit "
                "inputs (the incoming pair, the exchange records, the "
                "outgoing lines, the central source, the line-attributed "
                "final state) through card 2's audit_event; the stored "
                "generation-time audit block must reproduce exactly"),
        },
        "conservation": {
            "events_checked": pooled["events"],
            "violations": 0,
            "pass": True,
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_closure_relative": pooled["worst_closure_relative"],
            "worst_mass_shell_relative": pooled["worst_mass_shell_relative"],
            "worst_energy_closure_gev": pooled["worst_energy_closure_gev"],
            "worst_momentum_closure_gev": pooled["worst_momentum_closure_gev"],
            "b_q_s_census": dict(pooled["b_q_s_census"]),
        },
        "per_replica": per_replica,
        "class_exercise": (None if exercise is None else {
            "partition": SMOKE_EXERCISE_INDEX_RULE,
            "p_n2_fixture": SMOKE_EXERCISE_P_N2,
            "classes": exercise.get("classes"),
        }),
        "r53_outcome": (None if smoke else dict(
            (pre or {}).get("topology_sampling", {}).get("nd_multi")
            or {})),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": named_residuals,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "no_optimizer_note": (
            "index item 10: R54 draws no optimizer and fits no parameter "
            "(the generation audit); optimizer_invoked false, "
            "optimizer_uses 0"),
        "no_separation_claim": (
            "no Round 5 record claims a separation of exchange recoil "
            "from decay transverse dynamics (the R46 outcome, "
            "separation_claimed false); the exchange slopes act on the "
            "exchange kinematics only (plan section 24, line 1866)"),
        "determinism": DETERMINISM_NOTE,
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "acceptance": ACCEPTANCE_NOTE,
        "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """--gate: write gate5_r54.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the audit report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / AUDIT_REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {AUDIT_REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    cons = report["conservation"]
    audit = report["ledger_audit"]
    sampling = report["topology_sampling"]
    gate = {
        "schema": "na49_mixing.staged_r54_local_ledger_audit.gate/1",
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R54",
        "card": CARD,
        "clause": ("color-singlet audit (plan section 5.5 item 7, "
                   "line 323)"),
        "events_checked": int(report["ledger_audit"]["events_checked"]),
        "topology_sampling": {
            "source": sampling["source"],
            "fractions": dict(sampling["fractions"]),
            "archive_pin": sampling["archive_pin"],
        },
        "ledger_audit": {
            "violations": int(audit["violations"]),
            "pass": bool(audit["pass"]),
            "rules": list(AUDIT_RULES),
            "per_rule_counts": dict(audit["per_rule_counts"]),
        },
        "conservation": {
            "violations": int(cons["violations"]),
            "pass": bool(cons["pass"]),
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_energy_closure_gev": cons["worst_energy_closure_gev"],
        },
        "r53_outcome": (report.get("r53_outcome") or {}),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": list(report["named_residual_register"]),
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "report_digest": report.get("digest"),
        "sealed": True,
        "sealed_note": (
            "the run seals after this record (the seal manifest and "
            "SHA256SUMS.txt cover this file; the run dir is immutable "
            "after the seal)"),
        "gate_note": (
            "evidence only; the written retain / revise / reject decision "
            "on the Gate 5 clauses is the round's Gate 5 follow-up "
            "obligation"),
    }
    _write_json(run_dir / GATE_RECORD_NAME, gate)
    print(f"[r54] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


# ---------------------------------------------------------------------------
# The clean-environment validation (exactly one pass, before the seal)
# ---------------------------------------------------------------------------
def run_clean_validate(run_dir: Path) -> dict:
    """--clean-validate: re-run the audit evaluation from a clean
    interpreter over the generated artifacts with a byte-identical
    report, re-run the pre-registered probe replica (replica 0) in
    memory against its sealed record, write
    clean_environment_commands.json, then stop.  Exactly one pass;
    executed before the seal (a second invocation refuses; the sealed
    dir refuses -- the card-19 seal-ordering discipline)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    if (run_dir / SEAL_NAME).is_file():
        _refuse(
            f"the run dir is already sealed: {run_dir / SEAL_NAME}; "
            "clean-validate is never invoked on the sealed dir (the "
            "card-19 seal-ordering discipline)")
    clean_path = run_dir / CLEAN_ENV_NAME
    if clean_path.is_file():
        _refuse(
            f"the clean-environment validation already ran: {clean_path}; "
            "exactly one pass executes before the seal (the card-19 "
            "seal-ordering discipline)")
    for name in (AUDIT_REPORT_NAME, GATE_RECORD_NAME):
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    pre = load_pre_registration(run_dir)
    verify_frozen_substrate()
    fraction_assertion = assert_fractions_against_r51()
    outcome = read_r53_outcome()
    resolution = assert_outcome_against_preregistration(outcome, pre)
    sealed_report = _read_json(run_dir / AUDIT_REPORT_NAME)
    rerun_report = run_audit_evaluation(run_dir, smoke=False, pre=pre)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the probe replica re-run in memory (the determinism evidence of
    # acceptance (v)): the pre-registered probe replica (replica 0)
    # regenerates from its pre-registered seed and must reproduce its
    # sealed event-hash set, its class and count censuses, its audit
    # aggregate, and its conservation aggregate exactly
    probe = PROBE_REPLICA
    catalog = _load_catalog()
    fams = _load_families()
    sampling = sampling_from_outcome(outcome)
    (records, census, count_census, audit_agg, conservation,
     hash_set) = generate_replica_events(
        replica_seed(probe), EVENTS_PER_REPLICA, EVENT_INDEX_BASE,
        catalog, fams, sampling)
    sealed_probe = _read_json(run_dir / replica_name(probe))
    probe_matches = bool(
        hash_set == sealed_probe.get("event_hash_set_sha256")
        and census == sealed_probe.get("class_census")
        and count_census == sealed_probe.get("count_census")
        and audit_agg == sealed_probe.get("audit_aggregate")
        and conservation == sealed_probe.get("conservation")
    )
    payload = {
        "schema": ("na49_mixing.staged_r54_local_ledger_audit."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r54_local_ledger_audit.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            (f"python helpers/benchmarking/"
             f"staged_r54_local_ledger_audit.py "
             f"--clean-validate --run-dir {run_dir}"),
        ],
        "seal": {
            "entries": 0,
            "mismatches": [],
            "state": ("pending (the first clean validation runs before "
                      "the seal so the validation record is sealed "
                      "content; the card's step order lands the seal "
                      "over it)"),
        },
        "report_digest": sealed_report.get("digest"),
        "report_rerun_digest": rerun_report.get("digest"),
        "report_hash_match": bool(report_matches),
        "probe_replica": {
            "replica": probe,
            "file": replica_name(probe),
            "seed": replica_seed(probe),
            "event_hash_set_match": bool(
                hash_set == sealed_probe.get("event_hash_set_sha256")),
            "census_matches": bool(census == sealed_probe.get(
                "class_census")),
            "count_census_matches": bool(count_census == sealed_probe.get(
                "count_census")),
            "audit_aggregate_matches": bool(audit_agg == sealed_probe.get(
                "audit_aggregate")),
            "conservation_matches": bool(conservation == sealed_probe.get(
                "conservation")),
            "matches_sealed_record": probe_matches,
        },
        "verified_utc": _now_utc(),
    }
    _write_json(clean_path, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter audit evaluation does not reproduce "
            "the report digest (sealed "
            f"{sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(
        f"[r54] clean validation written: {clean_path} (report hash "
        f"match: {report_matches}, probe replica {probe} match: "
        f"{probe_matches})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# The seal (the run dir is immutable afterwards)
# ---------------------------------------------------------------------------
def write_sha256sums(run_dir: Path) -> int:
    """The strict path-sorted checksum inventory (the
    make_archive_checksums convention, run-dir-relative; LF endings)."""
    run_dir = Path(run_dir)
    names = sorted(
        str(path.relative_to(run_dir).as_posix())
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != SHA256SUMS_NAME
    )
    lines = []
    for name in names:
        digest = _sha256_file(run_dir / name)
        lines.append(f"{digest}  {name}")
    Path(run_dir / SHA256SUMS_NAME).write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8"))
    return len(names)


def write_seal_manifest(run_dir: Path, n_entries: int) -> dict:
    """The seal manifest: the inventory, the identity digests, and the
    environment (the R40 convention)."""
    run_dir = Path(run_dir)
    files = sorted(
        str(path.relative_to(run_dir).as_posix())
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != SEAL_NAME
        and path.name != SHA256SUMS_NAME
    )
    manifest = {
        "schema": ("na49_mixing.staged_r54_local_ledger_audit."
                   "seal/1"),
        "run": RUN_ID,
        "card": CARD,
        "run_dir": str(run_dir),
        "files": files,
        "sha256sums_entries": n_entries,
        "record_count": sum(
            1 for name in files if name.startswith("replica_")
        ),
        "sealed_utc": _now_utc(),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "clean_validation_record": CLEAN_ENV_NAME,
    }
    _write_json(run_dir / SEAL_NAME, manifest)
    return manifest


def run_seal(run_dir: Path) -> dict:
    """--seal: write seal_manifest.json and SHA256SUMS.txt over the run
    dir, then stop.  Requires the clean-environment validation record
    (exactly one pass, before the seal).  The run dir is immutable after
    the seal."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    if (run_dir / SEAL_NAME).is_file():
        _refuse(f"the run dir is already sealed: {run_dir / SEAL_NAME}")
    required = (
        PREREGISTRATION_NAME, PROVENANCE_NAME, AUDIT_REPORT_NAME,
        GATE_RECORD_NAME, CLEAN_ENV_NAME,
    )
    for name in required:
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    missing = [
        replica_name(r) for r in range(N_REPLICAS)
        if not (run_dir / replica_name(r)).is_file()
    ]
    if missing:
        _refuse(f"the production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r54] sealed {run_dir} ({n_entries} entries)", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# The smoke-artifact refusal (the R31/8c convention)
# ---------------------------------------------------------------------------
def _refuse_smoke_artifacts(run_dir: Path) -> None:
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention): a run dir whose pre-registration is a smoke
    pre-registration, or that carries any ``smoke_``-prefixed file, is
    refused; a run dir without a pre-registration is refused too."""
    run_dir = Path(run_dir)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file():
        pre = _read_json(pre_path)
        if pre.get("mode") == "smoke":
            _refuse(
                f"the run dir carries a smoke pre-registration: "
                f"{pre_path}; smoke artifacts are never sealed "
                "(the R31/8c convention)")
    else:
        _refuse(f"the run dir carries no pre-registration: {pre_path}")
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name.startswith(SMOKE_PREFIX):
            _refuse(
                f"the run dir carries a smoke artifact: {path}; smoke "
                "artifacts are refused by --gate / --seal / "
                "--clean-validate (the R31/8c convention)")


# ---------------------------------------------------------------------------
# The per-class pinned dispatch exercise (the smoke mode's hermetic
# 'across all topologies' evidence)
# ---------------------------------------------------------------------------
EXERCISE_SCHEMA = ("na49_mixing.staged_r54_local_ledger_audit."
                   "class_exercise/1")


def run_smoke_class_exercise(catalog, fams) -> dict:
    """The per-class pinned dispatch: every topology class (elastic,
    sd_projectile, sd_target, dd, nd, nd_multi) through the same
    generation, audit, and conservation machinery on the dedicated smoke
    class-exercise partition (the hermetic evidence that the five-rule
    audit covers every landed class; 'across all topologies' and
    'elastic included' exercised end to end).  The nd_multi class runs
    at the pinned smoke exercise P(n = 2) (a smoke fixture value, the
    R53 drawn value, recorded as such; never a production value).  The
    elastic dispatch cross-checks the pinned incoming pair against the
    elastic module's own incoming record."""
    params = frozen_draw_params(SMOKE_EXERCISE_P_N2)
    per_class: dict[str, dict] = {}
    records: list = []
    slot = 0
    for cls in SMOKE_EXERCISE_CLASSES:
        class_records = []
        count_census = {"n1": 0, "n2": 0}
        violations = 0
        for j in range(SMOKE_EXERCISE_EVENTS_PER_CLASS):
            subs = derive_substreams(
                SMOKE_SEED, 0, SMOKE_EXERCISE_INDEX_BASE + slot)
            rng_topology_selection = np.random.default_rng(
                subs.stream_for("topology_selection"))
            rng_spectral_state = np.random.default_rng(
                subs.stream_for("spectral_state"))
            rng_exchange_kinematics = np.random.default_rng(
                subs.stream_for("exchange_kinematics"))
            rng_decay_channel = np.random.default_rng(
                subs.stream_for("decay_channel"))
            rng_decay_kinematics = np.random.default_rng(
                subs.stream_for("decay_kinematics"))
            event = run_topology(
                cls, params, catalog, fams,
                rng_topology_selection=rng_topology_selection,
                rng_spectral_state=rng_spectral_state,
                rng_exchange_kinematics=rng_exchange_kinematics,
                rng_decay_channel=rng_decay_channel,
                rng_decay_kinematics=rng_decay_kinematics,
                p_n2=params.p_n2)
            if isinstance(event, ElasticEvent):
                check_zero_transfer(event)
                incoming = tuple(r51._line_qn(line)
                                 for line in event.incoming)
                if incoming != (INCOMING_QN, INCOMING_QN):
                    raise LedgerAuditViolation(
                        "the elastic module's incoming pair drifted off "
                        f"the pinned {INCOMING_QN}: {incoming!r}")
            obs = extract_observables(event, SQRT_S)
            conservation = r51.event_conservation(event)
            audit = run_ledger_audit(event)
            if not audit["pass"]:
                violations += len(audit["violations"])
                raise LedgerAuditViolation(
                    f"the smoke class exercise ({cls}, event {j}): the "
                    "ledger audit recorded "
                    f"{len(audit['violations'])} violation(s): "
                    + "; ".join(audit["violations"]))
            record = serialize_audit_event(event, obs, conservation, audit)
            record["i"] = int(j)
            record["id"] = int(SMOKE_EXERCISE_INDEX_BASE + slot)
            if isinstance(event, NDMultiEvent):
                record["mu"] = {
                    "count": int(event.count),
                    "p_n2": float(event.p_n2),
                }
                if event.count == 2:
                    count_census["n2"] += 1
                else:
                    count_census["n1"] += 1
            record.pop("h", None)
            record["h"] = hashlib.sha256(
                json.dumps(record, sort_keys=True,
                           separators=(",", ":")).encode(
                               "utf-8")).hexdigest()
            class_records.append(record)
            slot += 1
        per_class[cls] = {
            "events": len(class_records),
            "violations": violations,
            "pass": violations == 0,
            "count_census": dict(count_census),
        }
        records.extend(class_records)
    hash_set = hashlib.sha256(
        "\n".join(r["h"] for r in records).encode("utf-8")).hexdigest()
    return {
        "schema": EXERCISE_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "partition": SMOKE_EXERCISE_INDEX_RULE,
        "p_n2_fixture": SMOKE_EXERCISE_P_N2,
        "p_n2_fixture_note": (
            "the pinned smoke exercise P(n = 2) is a smoke-only fixture "
            "value (the R53 drawn value) exercising both count strata of "
            "the card-8 generation; it is never a production value and "
            "the production dispatch follows the recorded R53 outcome"),
        "classes": per_class,
        "event_hash_set_sha256": hash_set,
        "events": records,
    }


# ---------------------------------------------------------------------------
# The hermetic smoke mode (driver validation; never sealed)
# ---------------------------------------------------------------------------
def run_smoke(run_dir: Path) -> dict:
    """The hermetic end-to-end driver validation: a smoke
    pre-registration, one mixed-draw replica x 2,000 events at the smoke
    seed 20312925 on the event-index partition 1_600_000_000 + i (i <
    2,000) at the R51 frozen fractions (the recorded sampling surface at
    the R53-PENDING default: the nd dispatch at P(n = 2) = 0.0), the
    per-class pinned dispatch exercise (every topology class through the
    same audit machinery), and the smoke audit evaluation, into an
    explicit --run-dir (the default production run dir is refused).  No
    sealed archive is read and no SPS table is read (the smoke asserts
    the frozen-value constants against no sealed archive; the production
    assertion runs in --pre-registration and --evaluate).  Two identical
    smoke invocations produce byte-identical artifacts (no wall-clock
    and no git state in the smoke artifacts)."""
    run_dir = Path(run_dir)
    if run_dir.resolve() == (MONO_ROOT / DEFAULT_RUN_DIR).resolve():
        _refuse(
            "--smoke refuses the default production run dir (the smoke "
            f"artifacts are never sealed into {DEFAULT_RUN_DIR}); pass "
            "an explicit --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_substrate()
    m8_frozen_set_status()
    seed_disjointness_report()
    smoke_pre = {
        "schema": ("na49_mixing.staged_r54_local_ledger_audit."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "audit_surface": dict(_FIT_SURFACE_STATUS),
        "assertion": ASSERTION_NOTE,
        "statistics": {
            "note": (
                "the smoke statistics: one mixed-draw replica x 2,000 "
                "events at the smoke seed 20312925 plus the per-class "
                "pinned dispatch exercise (six classes x 8 events on the "
                "dedicated smoke class-exercise partition); the smoke "
                "exercises pre-registration, generation, the ledger "
                "audit, and evaluation end to end; the production "
                "statistics are the 20 pre-registered replicas of 50,000 "
                "events"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
            "smoke_events": SMOKE_EVENTS,
            "smoke_class_exercise": {
                "partition": SMOKE_EXERCISE_INDEX_RULE,
                "events_per_class": SMOKE_EXERCISE_EVENTS_PER_CLASS,
                "classes": list(SMOKE_EXERCISE_CLASSES),
                "p_n2_fixture": SMOKE_EXERCISE_P_N2,
            },
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "smoke_raw_seed": SMOKE_SEED,
            "smoke_event_index_rule": SMOKE_EVENT_INDEX_RULE,
            "smoke_class_exercise_index_rule": SMOKE_EXERCISE_INDEX_RULE,
            "production_raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 44 + k = 20300969 "
                f"+ k per replica k < {N_REPLICAS}"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "topology_sampling": {
            "source": (
                "the R51 frozen fractions (the driver's pinned constants "
                "in smoke mode; the production assertion against the "
                "sealed R51 fit report runs in --pre-registration and "
                "--evaluate)"),
            "fractions": {
                "elastic": ELASTIC_FRACTION_GENERATION,
                "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
                "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
                "dd": R51_FROZEN_FRACTIONS["dd"],
                "nd": R51_FROZEN_FRACTIONS["nd"],
                "nd_multi": R53_P_N2_DEFAULT,
            },
            "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
            "nd_multi": {
                "status": "R53-PENDING",
                "pending": True,
                "retained": False,
                "p_n2": R53_P_N2_DEFAULT,
                "reaudit_obligation": REAUDIT_OBLIGATION_NOTE,
                "smoke_note": (
                    "the smoke's mixed draw runs the nd dispatch at the "
                    "R53-PENDING default 0.0 (the pinned smoke constant; "
                    "the production dispatch follows the recorded R53 "
                    "outcome); the nd_multi generation path itself is "
                    "exercised by the smoke class exercise at the pinned "
                    "fixture P(n = 2)"),
            },
        },
        "conservation": {
            "contract": (
                "every event conserves total B, Q, and S exactly "
                "(integer arithmetic) and closes the total "
                "four-momentum at tolerance 1e-09 relative against the "
                "incoming sum; a violation raises and stops the run"),
            "tolerance": EVENT_CLOSURE_TOL,
            "expected_b_q_s": [2, 2, 0],
            "incoming_reference": INCOMING_REFERENCE_NOTE,
        },
        "determinism": DETERMINISM_NOTE,
        "no_separation_claim": (
            "no Round 5 record claims a separation of exchange recoil "
            "from decay transverse dynamics (the R46 outcome, "
            "separation_claimed false)"),
        "no_optimizer": (
            "index item 10: R54 draws no optimizer and fits no parameter; "
            "the smoke exercises the audit machinery only"),
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "m8_frozen_set": m8_frozen_set_status(),
        "frozen_substrate": {
            name: {"sha256": _sha256_file(MONO_ROOT / rel_path),
                   "expected": expected}
            for name, (rel_path, expected) in FROZEN_SUBSTRATE.items()
        },
        "smoke_note": (
            "the smoke artifacts are never sealed: --gate / --seal / "
            "--clean-validate refuse smoke artifacts (the R31/8c "
            "convention); two identical smoke invocations produce "
            "byte-identical artifacts (no wall-clock and no git state "
            "in the smoke artifacts); no sealed archive and no SPS table "
            "is read"),
    }
    _write_json(run_dir / ("smoke_" + PREREGISTRATION_NAME), smoke_pre)
    catalog = _load_catalog()
    fams = _load_families()
    sampling = sampling_from_outcome({
        "status": "R53-PENDING",
        "pending": True,
        "retained": False,
        "p_n2": R53_P_N2_DEFAULT,
    })
    (records, census, count_census, audit_agg, conservation,
     hash_set) = generate_replica_events(
        SMOKE_SEED, SMOKE_EVENTS, SMOKE_EVENT_INDEX_BASE, catalog, fams,
        sampling)
    replica_record = {
        "schema": REPLICA_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "replica": "smoke",
        "seed": SMOKE_SEED,
        "n_events": len(records),
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed": SMOKE_SEED,
            "event_index_rule": SMOKE_EVENT_INDEX_RULE,
            "substream_labels": list(SUBSTREAM_LABELS),
            "consumed_labels": [
                "topology_selection", "spectral_state",
                "exchange_kinematics", "decay_channel", "decay_kinematics",
            ],
            "mixed_draw_note": (
                "the smoke exercises the card-9 per-event class draw on "
                "the closed topology_selection substream at the recorded "
                "R51 frozen fractions (the recorded sampling surface)"),
        },
        "generation_fractions": {
            "note": GENERATION_FRACTIONS_NOTE,
            "class_fractions": dict(R51_FROZEN_FRACTIONS),
            "elastic_fraction": float(sampling["elastic_fraction"]),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": float(sampling["p_n2"]),
            "nd_multi_active": bool(sampling["nd_multi_active"]),
        },
        "class_census": dict(census),
        "count_census": dict(count_census),
        "audit_aggregate": audit_agg,
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }
    _write_json(run_dir / SMOKE_REPLICA_NAME, replica_record)
    print(
        f"[r54] smoke generation complete: {len(records)} events, census "
        f"{census}, audit pass {audit_agg['pass']}, worst closure "
        f"{conservation['worst_closure_relative']:.3e}",
        flush=True,
    )
    exercise = run_smoke_class_exercise(catalog, fams)
    _write_json(run_dir / SMOKE_EXERCISE_NAME, exercise)
    print(
        f"[r54] smoke class exercise complete: "
        f"{sum(entry['events'] for entry in exercise['classes'].values())} "
        "events across "
        f"{len(exercise['classes'])} classes, all audits pass",
        flush=True,
    )
    report = run_audit_evaluation(run_dir, smoke=True)
    _write_json(run_dir / SMOKE_REPORT_NAME, report)
    print(
        f"[r54] smoke evaluation complete: ledger violations "
        f"{report['ledger_audit']['violations']}, worst closure "
        f"{report['conservation']['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {
        "events": len(records),
        "census": census,
        "count_census": count_census,
        "exercise_classes": {
            cls: entry["events"]
            for cls, entry in exercise["classes"].items()},
        "ledger_violations": report["ledger_audit"]["violations"],
        "worst_closure_relative":
            report["conservation"]["worst_closure_relative"],
    }


# ---------------------------------------------------------------------------
# The pre-registration notes
# ---------------------------------------------------------------------------
OBJECTIVE_NOTE = (
    "plan section 11 run R54 (lines 928-930): 'Generate at least one "
    "million events across all topologies and assert that every nonzero "
    "quantum-number transfer is attached to an allowed exchange record.' "
    "The pre-registered instantiation (index items 6, 8, and 10): "
    "exactly 1,000,000 events = 20 replicas x 50,000 (the T3 replica "
    "count, plan section 5.1 line 255; plan line 930 requires at least "
    "one million, met with equality).  The assertion is zero violations "
    "across all events on card 2's five ledger rules (plan lines 930 "
    "and 875-879): (a) every exchange record is in the closed allowed "
    "set, (b) the vacuum exchange transfers zero, (c) each outgoing "
    "excitation equals the line quantum numbers after the declared "
    "transfer, (d) every decay vertex closes locally, (e) the central "
    "source carries B = Q = S = 0.  The audit mechanism is card 2's "
    "audit_event (the LedgerAuditReport with an empty violation list on "
    "a compliant event), consumed over the class records produced by "
    "the card-9 runner.  No optimizer (index item 10).")

ASSERTION_NOTE = (
    "zero violations across all 1,000,000 events on card 2's five ledger "
    "rules (plan lines 930 and 875-879, plan section 11.4): "
    "(a) closed_set_membership -- every exchange record is in the closed "
    "allowed-exchange set (exactly the vacuum exchange at this round); "
    "(b) vacuum_zero_transfer -- the vacuum exchange transfers zero; "
    "(c) inheritance_after_declared_transfer -- each outgoing excitation "
    "equals the line quantum numbers after the declared transfer "
    "(outgoing = incoming + transfer, exact integers); (d) "
    "local_decay_closure -- every decay vertex closes locally (realized "
    "at the line level: each baryonic line's outgoing ledger state "
    "against its attributed final-state products, the central source "
    "against its products; the frozen cascade's internal vertices close "
    "locally by the cascade module's DECAY_CLOSURE_TOL check); (e) "
    "central_source_neutrality -- the central source carries "
    "B = Q = S = 0 (plan line 849).  The audit runs at generation on "
    "every event (a violation raises and fails the run: a violation is a "
    "named finding), and the evaluation re-audits every stored event "
    "from its stored audit inputs (the stored audit block must reproduce "
    "exactly).  This is the color-singlet audit of plan section 5.5 item "
    "7 (line 323).")

RECORDING_CONVENTION_NOTE = (
    "the per-event record is the R50 full-final-state convention (every "
    "ledger state and every four-momentum as an exact float) extended "
    "with the line attribution of every final-state hadron, the "
    "central-source ledger states, the named exchange records, and the "
    "generation-time audit block: every stored event is independently "
    "re-auditable (the evaluation's recheck and the clean-environment "
    "validation consume exactly the stored fields).  The per-event hash "
    "is the sha256 of the record's canonical JSON without the ``h`` key "
    "and the event_hash_set_sha256 is the sha256 over the joined "
    "per-event hashes, so the stored bytes themselves carry the "
    "bit-level identity evidence.  The dense-lattice published-cell "
    "recording of R51/R52 does not apply here: R54 asserts the ledger "
    "rules, not the published-cell agreement, and stores the full "
    "audit-observable event record instead")

DETERMINISM_NOTE = (
    "a per-replica re-run from a clean interpreter reproduces the report "
    "hashes; audit_report.json and gate5_r54.json carry no wall-clock "
    "fields (their bytes are a pure function of the sealed artifacts, "
    "the pre-registered seeds, and the committed code); the per-event "
    "hash is the sha256 of the canonical JSON of the stored event record "
    "and the event_hash_set_sha256 is the sha256 over the joined "
    "per-event hashes, so the stored events carry the bit-level identity "
    "evidence and the probe replica's clean re-run must reproduce them")

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(viii) of the card: (i) statistics: "
    "exactly 1,000,000 events generated (20 replicas x 50,000); the "
    "per-class draw counts are recorded and match the pre-registered "
    "topology sampling at the recorded MC tolerance (the per-class "
    "marginal binomial check at 5.0 sigma; a class with expected count "
    "exactly 0 must realize exactly 0); (ii) ledger assertion: zero "
    "violations across all 1,000,000 events on card 2's five ledger "
    "rules (closed-set membership, vacuum zero transfer, inheritance "
    "after the declared transfer, local decay closure, central-source "
    "neutrality); the color-singlet audit of plan section 5.5 item 7; "
    "(iii) conservation: zero B, Q, S violations and zero four-momentum "
    "closure failures at tolerance 1e-09 on all 1,000,000 events; the "
    "worst closure residual recorded in GeV; (iv) topology sampling "
    "record: the source of the fractions is recorded (the R51 frozen "
    "values with the sealed archive pin) and asserted at generation; a "
    "divergence is a named deviation; (v) determinism: the "
    "clean-environment re-run reproduces the per-replica report hashes; "
    "(vi) no optimizer: optimizer_invoked false, optimizer_uses 0, no "
    "parameter fit; (vii) seal and clean validation: exactly one "
    "clean-environment validation pass executes before the seal (the "
    "card-19 seal-ordering discipline) and the seal is intact "
    "afterwards; (viii) pins and freeze: the driver and test file land "
    "(both trees byte-identical), the freeze re-pin is recorded, and the "
    "full nested suite is green (the land gate's obligation)")

GENERATION_FRACTIONS_NOTE = (
    "the R54 mixed draw runs at the R51 frozen fractions (the sealed "
    "fit_report.json fitted values, asserted byte-exactly at "
    "pre-registration and re-asserted at generation) with the elastic "
    "fraction the exact complement 0.0 (the recorded source convention; "
    "see the elastic-inclusion note) and the nd dispatch governed by the "
    "recorded R53 outcome (the card-8 nd_multi generation at the sealed "
    "fitted P(n = 2) exactly when the sealed retained boolean is true, "
    "else card 7's ND generation at the unretained default 0.0); the "
    "mass-law parameters are frozen at the R51 fitted epsilon and B_SD "
    "(the round's frozen point, the R52/R53 convention)")

GENERATION_GATEWAY_NOTE = (
    "gateway-primary generation (host cl, workers c0801-c0809 x 32 task "
    "slots; one job per replica, --tasks 1, single math thread "
    "(NA49_CLUSTER_GW_TASKS forces thread counts to 1), submission "
    "paced by the capacity check, no local process pool inside the "
    "chunk command, chunk outputs appended only to the new run dir); "
    "the driver's local fallback is one single-threaded process per "
    "--replica invocation (a completed record that matches the "
    "pre-registered identity is reused unchanged)")

COLLISION_NOTE = (
    "the R32 p+p configuration (index item 8): p+p at sqrt(s) = 17.270 "
    "GeV (beam 158 GeV/c), the M8-frozen ten-parameter set (hash "
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1) "
    "through the ResonancePPBuilder surface with the card-6 routing at "
    "the fixed blend ramp start B = 2.680194110727528 GeV and the "
    "card-9 feed-down lineage convention; the blend ramp start is never "
    "a free parameter.  The topology layer replaces the single "
    "two-source inelastic topology: the M8 set's two (B, Q, S) = (1, 1, "
    "0) sources are the incoming lines, the class selection decides "
    "which line is excited and what is produced centrally, and the "
    "excited states decay through the frozen simplified cascade at the "
    "sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 "
    "GeV^-2, baryonic nominal width 0.3 GeV) consumed without refit "
    "(the R46 no_refit clause); the exchange slopes act on the exchange "
    "kinematics only, not on the daughter transverse dynamics (plan "
    "section 24, line 1866); no Round 5 record claims a separation of "
    "exchange recoil from decay transverse dynamics (the R46 outcome, "
    "separation_claimed false).")


# ---------------------------------------------------------------------------
# The pre-registration (written once, before any generation)
# ---------------------------------------------------------------------------
def run_pre_registration(run_dir: Path, *, smoke: bool = False) -> dict:
    """Write pre_registration.json (and, in production, provenance.json)
    into the run dir, then stop.  Nothing is generated.  A pre-existing
    pre-registration refuses the run (the pre-registration is written
    once, before any generation; a change after generation invalidates
    the run and re-seals under a new date)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file():
        _refuse(
            f"the pre-registration already exists: {pre_path}; the "
            "pre-registration is written once, before any generation (a "
            "change after generation invalidates the run and re-seals "
            "under a new date)"
        )
    substrate = verify_frozen_substrate()
    m8_status = m8_frozen_set_status()
    disjointness = seed_disjointness_report()
    if not smoke:
        sealed_inputs = verify_sealed_inputs()
        fraction_assertion = assert_fractions_against_r51()
        nominal_reference = assert_nominal_reference_against_r50()
        outcome = read_r53_outcome()
    else:
        sealed_inputs = None
        fraction_assertion = None
        nominal_reference = None
        outcome = {
            "status": "R53-PENDING",
            "pending": True,
            "retained": False,
            "p_n2": R53_P_N2_DEFAULT,
            "source": None,
            "reaudit_obligation": REAUDIT_OBLIGATION_NOTE,
            "note": ("the smoke pre-registration records the pinned "
                     "R53-PENDING constants against no sealed archive; "
                     "the production read runs in --pre-registration"),
        }
    fractions = {
        "elastic": ELASTIC_FRACTION_GENERATION,
        "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
        "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
        "dd": R51_FROZEN_FRACTIONS["dd"],
        "nd": R51_FROZEN_FRACTIONS["nd"],
        "nd_multi": (float(outcome["p_n2"])
                     if outcome["retained"] else R53_P_N2_DEFAULT),
    }
    payload = {
        "schema": ("na49_mixing.staged_r54_local_ledger_audit."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
        "mode": "smoke" if smoke else "production",
        "recorded_utc": None if smoke else _now_utc(),
        "written_before_generation": True,
        "objective": OBJECTIVE_NOTE,
        "collision": COLLISION_NOTE,
        "audit_surface": dict(_FIT_SURFACE_STATUS),
        "assertion": ASSERTION_NOTE,
        "statistics": {
            "tier": (
                "exactly 1,000,000 events = 20 replicas x 50,000 p+p "
                "events (index item 6; the T3 replica count, plan "
                "section 5.1 line 255; plan line 930 requires at least "
                "one million, met with equality; a larger count is a "
                "named deviation)"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 44 + k = 20300969 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300969-20300988 is disjoint from every "
                "previously used raw block and from every other reserved "
                "Round 5 block)"),
            "raw_replica_seeds": list(REPLICA_SEEDS),
            "event_index_rule": EVENT_INDEX_RULE,
            "event_index_base": EVENT_INDEX_BASE,
            "substream_labels": list(SUBSTREAM_LABELS),
            "disjointness": disjointness,
            "smoke": {
                "raw_seed": SMOKE_SEED,
                "raw_seed_rule": (
                    "raw smoke seed = 20260925 + 40000 + 12000 = "
                    "20312925 (the round-5 +12000 smoke block, disjoint "
                    "from every production replica and from every "
                    "previously used smoke block)"),
                "events": SMOKE_EVENTS,
                "event_index_rule": SMOKE_EVENT_INDEX_RULE,
                "class_exercise_index_rule": SMOKE_EXERCISE_INDEX_RULE,
            },
            "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        },
        "topology_sampling": {
            "source": (
                "the R51 frozen fractions (the sealed "
                f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME} fitted values, "
                "asserted at generation)" if not smoke else
                "the driver's pinned R51 frozen-fraction constants (the "
                "production assertion against the sealed R51 fit report "
                "runs in --pre-registration and --evaluate)"),
            "fallback": (
                "the equal nominal fractions of index item 2 (0.25 each "
                "over sd_projectile, sd_target, dd, nd) -- NOT in force: "
                "the sealed R51 fractions are available"),
            "fallback_source": (
                f"the sealed R50 nominal reference "
                f"({R50_RUN_DIR}/{R50_PREREGISTRATION_NAME}, its nominal "
                "inelastic fractions 0.25 each)"),
            "fractions": fractions,
            "assertion": (
                "the fractions are asserted byte-exactly against the "
                "sealed R51 fit report at pre-registration and "
                "re-asserted at generation (the driver re-reads the "
                "sealed archive before every generate/evaluate pass); a "
                "divergence is a named deviation (acceptance (iv)); a "
                "silent switch never happens"),
            "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
            "elastic_included": True,
            "nd_multi": {
                "status": (None if smoke else outcome["status"]),
                "pending": bool(outcome["pending"]),
                "retained": bool(outcome["retained"]),
                "p_n2": float(outcome["p_n2"]),
                "source": outcome.get("source"),
                "reaudit_obligation": REAUDIT_OBLIGATION_NOTE,
                "dispatch": (
                    "the drawn 'nd' labels dispatch the card-8 nd_multi "
                    "generation at the sealed fitted P(n = 2) exactly "
                    "when the recorded R53 outcome retains the class; "
                    "otherwise card 7's ND generation runs at the "
                    "unretained default 0.0 and nd_multi events are "
                    "absent"),
            },
            "draw": (
                "the per-event class draw on the closed "
                "topology_selection substream through the card-9 "
                "draw_topology at the recorded fractions; each inelastic "
                "class is returned with exactly its recorded fraction "
                "and the elastic class occupies the lower edge at its "
                "recorded fraction (exactly 0.0 at the sealed R51 "
                "point)"),
            "mc_tolerance": {
                "sigma": MC_TOLERANCE_SIGMA,
                "rule": (
                    "the per-class marginal binomial check: the realized "
                    "draw count within 5.0 binomial standard deviations "
                    "of the expected count n * f; a class with expected "
                    "count exactly 0 must realize exactly 0; a "
                    "divergence is a named deviation (acceptance (iv))"),
            },
        },
        "frozen_fractions_assertion": fraction_assertion,
        "nominal_reference": nominal_reference,
        "r53_outcome_resolution": (None if smoke else {
            "status": outcome["status"],
            "pending": bool(outcome["pending"]),
            "retained": bool(outcome["retained"]),
            "p_n2": float(outcome["p_n2"]),
            "source": outcome.get("source"),
            "gate_file_sha256": outcome.get("gate_file_sha256"),
            "note": outcome.get("note"),
        }),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "conservation": {
            "contract": (
                "every event conserves total B, Q, and S exactly "
                "(integer arithmetic) and closes the total "
                "four-momentum at tolerance 1e-09 relative against the "
                "incoming sum; a violation raises and stops the run"),
            "tolerance": EVENT_CLOSURE_TOL,
            "expected_b_q_s": [2, 2, 0],
            "incoming_reference": INCOMING_REFERENCE_NOTE,
        },
        "determinism": DETERMINISM_NOTE,
        "no_separation_claim": (
            "no Round 5 record claims a separation of exchange recoil "
            "from decay transverse dynamics (the R46 outcome, "
            "separation_claimed false)"),
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "m8_frozen_set": m8_status,
        "frozen_substrate": {
            name: {"sha256": _sha256_file(MONO_ROOT / rel_path),
                   "expected": expected}
            for name, (rel_path, expected) in FROZEN_SUBSTRATE.items()
        },
        "sealed_inputs": sealed_inputs,
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "generation": GENERATION_GATEWAY_NOTE,
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r54_local_ledger_audit."
                       "provenance/1"),
            "run": RUN_ID,
            "card": CARD,
            "recorded_utc": _now_utc(),
            "driver": {
                "nested": {
                    "path": DRIVER_REL_NESTED,
                    "sha256": _sha256_file(MONO_ROOT / DRIVER_REL_NESTED),
                },
                "outer": {
                    "path": DRIVER_REL_OUTER,
                    "sha256": _sha256_file(MONO_ROOT / DRIVER_REL_OUTER),
                },
                "byte_identical": (
                    _sha256_file(MONO_ROOT / DRIVER_REL_NESTED)
                    == _sha256_file(MONO_ROOT / DRIVER_REL_OUTER)),
            },
            "round5_modules": round5_module_digests(),
            "frozen_substrate": {
                name: {"sha256": digest, "verified": True}
                for name, digest in substrate.items()
            },
            "m8_frozen_set": m8_status,
            "sealed_inputs": sealed_inputs,
            "r53_outcome": {
                "status": outcome["status"],
                "pending": bool(outcome["pending"]),
                "retained": bool(outcome["retained"]),
                "p_n2": float(outcome["p_n2"]),
                "source": outcome.get("source"),
            },
            "r45_r46_driver_pins": {
                "r45": FROZEN_SUBSTRATE["r45_driver"][1],
                "r46": FROZEN_SUBSTRATE["r46_driver"][1],
            },
            "git_state": _git_info(MONO_ROOT),
            "environment": environment_fingerprint(),
            "card_chain": list(CARD_CHAIN),
        }
        if not provenance["driver"]["byte_identical"]:
            _refuse(
                "the dual-tree driver copies are not byte-identical: "
                f"nested {provenance['driver']['nested']['sha256']} vs "
                f"outer {provenance['driver']['outer']['sha256']}")
        _write_json(run_dir / PROVENANCE_NAME, provenance)
        print(f"[r54] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r54] pre-registration written: {pre_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R54 driver: the local-ledger audit "
                    "(TASK-20260925-r5-run-r54-local-ledger-audit)")
    parser.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop")
    parser.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (never sealed)")
    parser.add_argument(
        "--generate", action="store_true",
        help="generate the production replica records, then stop")
    parser.add_argument(
        "--replica", type=int, action="append", default=None,
        help="generate only the given replica index (repeatable)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="the ledger audit and evaluation (audit_report.json), then "
             "stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r54.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the audit evaluation from a clean interpreter and "
             "the probe replica, then stop (exactly one pass, before the "
             "seal)")
    parser.add_argument(
        "--run-dir", default=None,
        help="the run dir (the default production run dir for the "
             "production modes; --smoke requires an explicit one)")
    return parser


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    modes = [
        bool(args.pre_registration), bool(args.smoke), bool(args.generate),
        bool(args.evaluate), bool(args.gate), bool(args.seal),
        bool(args.clean_validate),
    ]
    if sum(modes) != 1:
        parser.error("exactly one of --pre-registration / --smoke / "
                     "--generate / --evaluate / --gate / --seal / "
                     "--clean-validate is required")
    run_dir = Path(args.run_dir) if args.run_dir else (
        MONO_ROOT / DEFAULT_RUN_DIR)
    if args.smoke:
        if args.run_dir is None:
            _refuse("--smoke requires an explicit --run-dir")
        result = run_smoke(run_dir)
        print(f"[r54] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, replicas=args.replica)
        print(f"[r54] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        pre = load_pre_registration(run_dir)
        report = run_audit_evaluation(run_dir, smoke=False, pre=pre)
        _write_json(run_dir / AUDIT_REPORT_NAME, report)
        print(
            f"[r54] evaluation written: {run_dir / AUDIT_REPORT_NAME} "
            f"(digest {report['digest'][:16]}...)",
            flush=True,
        )
        return 0
    if args.gate:
        run_gate(run_dir)
        return 0
    if args.seal:
        run_seal(run_dir)
        return 0
    if args.clean_validate:
        run_clean_validate(run_dir)
        return 0
    parser.error("unreachable mode dispatch")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
