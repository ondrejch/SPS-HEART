#!/usr/bin/env python3
"""R50 driver: the pure topology templates (Round 5, card 10).

Task ``TASK-20260925-r5-run-r50-pure-topology-templates`` (plan section
11 run R50, lines 898-907; plan section 23 lines 1825-1827, "Round 5 --
exchange topology").  The card executes run R50, the per-class T1
template samples at the fixed nominal parameters, storing the six
pre-registered quantities of plan line 901 and the per-class evidence of
Gate 5 (plan section 11, line 938, training side).

The class set (round 5 index item 2)
------------------------------------
"every class" is the five classes elastic, projectile single
diffraction (``sd_projectile``), target single diffraction
(``sd_target``), double diffraction (``dd``), and non-diffrative
central excitation (``nd``).  The multiple-neutral class ``nd_multi``
is excluded from R50 (unretained, the card-8 registry entry; it enters
the round only through R53, plan line 926) and carries the unretained
default ``P(n = 2) = 0.0``.  The elastic class is "used for
normalization and code checks, not included in an inelastic sample"
(plan line 824): R50 generates it at T1 for the normalization check and
it is excluded from the inelastic statistics of R51, R52, R54, and R55.

The fixed nominal parameters (round 5 index item 2) are pre-registered,
not fit: epsilon = 0 (the flat M^2 law, the simplest instance of the
plan line 889 form (M^2)^(-1-epsilon) exp[B_SD (M^2) t]), B_SD =
8.504140762083612 GeV^-2 (byte-identical to the M8-frozen
``effective_transfer_slope``), the equal inelastic class fractions 0.25
each over ``sd_projectile``/``sd_target``/``dd``/``nd``, the remnant
weights intact 0.5 / low-mass 0.5 / high-mass 0.0 for single
diffraction, and ``P(n = 2) = 0.0``.  ``optimizer_invoked`` is false:
no parameter is fit on this run (index item 10; the round's optimizer
calls are R51 and the conditional R53).

The slot map (index items 2 and 9)
----------------------------------
T1 statistics (plan section 5.1 line 253): 4 replicas x 25,000 events
per class, one slot per (class, replica) pair, 100,000 events per
class, 500,000 events total across the five classes.  Master seed
20260925 (the round start date); raw replica seeds 20300925-20300944
(k = 0..19), one seed per (class, replica) pair in class order elastic,
SD_p, SD_t, DD, ND with slot ``k = class_index * 4 + replica_index``;
the block is disjoint from every previously used raw block (index item
9).  The event-index partition is ``1_000_000_000 + i`` with
``i < 25,000`` per slot on the closed five-label substream quintuple
(the ``derive_substreams`` pattern of the card-9 ``pinned_trace``,
replica slot 0: the slot's raw seed carries the (class, replica)
identity).  The hermetic smoke runs one mixed slot x 2,000 events at
the raw smoke seed 20312925 (the round-5 +12000 offset, disjoint from
every production slot and from every previously used smoke block) on
the event-index partition ``1_600_000_000 + i`` with ``i < 2,000``.  If
any raw (seed, event-index) substream collides with a pre-registered
block, the run stops and reports (never a silent re-block).

The class allocation and the card-9 runner
------------------------------------------
The generation runs through the card-9 runner
(``na49_mixing/topology_runner.py``): ``run_topology`` dispatches each
event to the landed class module of cards 1, 5, 6, and 7 with card 2's
ledger records attached to every event record, and ``extract_observables``
computes the six stored quantities (plan line 901) per event.  The
class allocation itself is the pre-registered slot map (one seed per
(class, replica) pair, equal inelastic sample mass 0.25 each by
construction), so the production slots dispatch their fixed class
directly; the per-event class draw on the closed ``topology_selection``
substream at the nominal fractions is exercised end to end in the
hermetic smoke mode and is the R51 fraction-sampling surface (no R50
slot draws it, because a pure-class T1 template allocates its class by
slot design and not by per-event draw).  The realized per-class
class-draw counts are therefore exact (25,000 per slot, 100,000 per
class) and the pooled inelastic fractions equal the pre-registered 0.25
each exactly, with the recorded MC tolerance noting the exact
construction (deviation 0.0).

The elastic normalization check (plan line 824)
-----------------------------------------------
The elastic class passes its pre-registered normalization and code
checks at the recorded tolerances: the closed-form normalization of
``exp(B_SD t)`` on ``[t_min, 0]`` against quadrature within 1e-12; the
inverse-CDF round trip on a pinned quantile grid within 1e-9; the
100,000-draw pooled elastic ``t`` mean against the analytic mean within
3 sigma of 1/sqrt(N); ``check_zero_transfer`` with zero violations on
every elastic event (checked at generation, a violation raises and
stops the run); the outgoing lines on the ``m_p`` mass shell; and the
Feynman-x pair sum ``x_F^proj + x_F^tar`` against zero (the CM-frame
momentum closure of the normalization reference).

Conservation (the house conservation contract)
----------------------------------------------
Every event conserves total B, Q, and S exactly (integer arithmetic on
the recorded ledger states; (2, 2, 0) for every class of the pinned p+p
configuration) and closes the total four-momentum at tolerance 1e-9
relative (the house convention: |dE|/E_total and |dp_i|/p_beam against
the incoming sum (E_beam + m_p, 0, 0, p_beam)); the worst per-event
mass-shell deviation is recorded.  A violation raises and stops the
run.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the M8 frozen
  parameter map, and the nominal-parameter assertion against
  ``TopologyNominalParams`` (card 9) before any generation.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one mixed slot x 2,000 events at
  the smoke seed 20312925 (the per-event class draw at the nominal
  fractions through ``draw_topology``), and the smoke evaluation, into
  an explicit ``--run-dir`` (the default production run dir is
  refused).  Two identical smoke invocations produce byte-identical
  artifacts.  ``--gate`` / ``--seal`` / ``--clean-validate`` refuse
  smoke artifacts (the R31/8c convention).
- ``--generate [--slot S]``: generate the requested production slot
  records (any subset; the default is every slot), one record file per
  slot, then stop.  Requires the pre-registration.  A completed record
  that matches the pre-registered identity is reused unchanged, so the
  declared tier can be completed slot by slot (the pre-registered
  gateway fallback runs one job per slot).
- ``--evaluate``: the per-class evaluation (the six stored quantities
  pooled over the four replicas, the class-draw census, the elastic
  normalization check, the conservation report, ``template_report.json``
  and ``REPORT.md``), then stop.  Requires all 20 production records.
- ``--gate``: write ``gate5_r50.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the
  report.
- ``--clean-validate``: re-run the evaluation from a clean interpreter
  over the generated artifacts with a matching report digest, re-run
  the pre-registered probe slot in memory against its sealed record,
  write ``clean_environment_commands.json``, then stop.  Exactly one
  pass; executed before the seal (the card-19 seal-ordering discipline;
  a second invocation refuses).
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  Requires the clean-environment validation
  record (exactly one pass, before the seal).  The run dir is immutable
  after the seal.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: the committed catalog, the continuum families configuration,
and the cards 1-9 surfaces (all consumed read-only) only; no SPS table,
no sealed-archive read, no external tables, no optimizer.  The frozen
simplified cascade at the sealed R45 point (kappa_B = 0.0, kappa_M =
4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV) is consumed
without refit (the R46 ``no_refit`` clause); the blend ramp start stays
fixed at B = 2.680194110727528 GeV and is never a free parameter; the
exchange slopes act on the exchange kinematics only, never on any
daughter-level transverse dynamics (plan section 24, line 1866); no
Round 5 record claims a separation of exchange recoil from decay
transverse dynamics (the R46 outcome, separation_claimed false).
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
# na49_mixing package, the resonance catalog, and the continuum config.
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
        f"staged_r50_pure_topology_templates: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

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
    elastic_t_normalization,
    elastic_t_quantile,
    elastic_t_range,
)
from na49_mixing.topology_mass_laws import MassLawParams  # noqa: E402
from na49_mixing.topology_non_diffractive import NDEvent  # noqa: E402
from na49_mixing.topology_runner import (  # noqa: E402
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
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260925-r5-run-r50-pure-
# topology-templates; the collision, statistics, seeds, nominal
# parameters, stored quantities, and acceptance blocks are verbatim from
# the card and the round 5 index items 2, 8, and 9).
# ---------------------------------------------------------------------------
RUN_ID = "R50"
CARD = "TASK-20260925-r5-run-r50-pure-topology-templates"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r50_pure_topology_templates_2026-09-25 re-pins to the
#: execution date 2026-09-27.
RUN_NAME = "r50_pure_topology_templates_2026-09-27"
DEFAULT_RUN_DIR = "dev/00runs/r50_pure_topology_templates_2026-09-27"
GATE_RECORD_NAME = "gate5_r50.json"
REPORT_NAME = "template_report.json"
REPORT_MD_NAME = "REPORT.md"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "slot_{}_{}.json"          # slot_<class>_<replica>.json
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r50_pure_topology_templates.py")
DRIVER_REL_OUTER = "helpers/benchmarking/staged_r50_pure_topology_templates.py"

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9 (the class modules, the ledger, the
#: remnants, the mass laws, and the topology template runner), and this
#: card.
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
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests).
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
}

#: The Round 5 landed modules recorded in the provenance (read-only
#: inputs; cards 1, 5, 6, 7 are the class modules of the card's
#: provenance list, cards 2, 3, 4 the ledger/remnant/mass-law modules,
#: card 9 the runner; card 8's nd_multi module is recorded as the
#: registry's unretained extension the runner imports).
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
#: The Feynman-x / central-window extraction energy: the card-1 module
#: constant SQRT_S = 17.270 GeV (the recorded M8 quote; the runner's
#: pinned extraction default).  The exact fixed-target invariant mass of
#: the beam configuration is 17.270205201372406 GeV (the generation-side
#: boost constant of the class modules).
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
            "model-uncertainty term of the seagull moments and every "
            "later Round 5 comparison that uses the continuum families"
        ),
    },
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_seagull_note": (
        "the b_transport_seagull beyond_isotropic_not_positive residual "
        "is a carry, not a pass criterion (index item 2)"
    ),
    "engine_moment_floor_note": (
        "the calculation-engine moment-approximation floor (~1e-3 GeV/c) "
        "is tracked on its separate track "
        "(_knowledge/evidence/engine-moment-approximation-floor.md) and "
        "is not conflated with the R32 spread"
    ),
}

# ---------------------------------------------------------------------------
# Statistics, classes, and the pre-registered slot map
# ---------------------------------------------------------------------------
#: The five R50 classes in the pre-registered class order (index item 2:
#: elastic, SD_p, SD_t, DD, ND; the slot k = class_index * 4 +
#: replica_index allocation follows this order).
CLASS_ORDER: tuple[str, ...] = (
    "elastic", "sd_projectile", "sd_target", "dd", "nd",
)
#: The inelastic classes of the pooled R50 sample (the elastic class is
#: excluded from the inelastic statistics, plan line 824).
INELASTIC_CLASS_ORDER: tuple[str, ...] = CLASS_ORDER[1:]
REPLICAS_PER_CLASS = 4
EVENTS_PER_SLOT = 25_000
N_SLOTS = len(CLASS_ORDER) * REPLICAS_PER_CLASS            # 20
N_EVENTS_PER_CLASS = REPLICAS_PER_CLASS * EVENTS_PER_SLOT  # 100,000
N_EVENTS_TOTAL = N_SLOTS * EVENTS_PER_SLOT                 # 500,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
SEED_BASE = MASTER_SEED + SEED_OFFSET                 # 20300925
SLOT_SEEDS: tuple[int, ...] = tuple(SEED_BASE + k for k in range(N_SLOTS))
EVENT_INDEX_BASE = 1_000_000_000
EVENT_INDEX_RULE = "1_000_000_000 + i, i < 25,000 (per slot)"

SMOKE_SEED = MASTER_SEED + 40_000 + 12_000              # 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"

#: The nominal class fractions (TopologyNominalParams.class_fractions;
#: the elastic fraction is the complement of the inelastic mass, exactly
#: 0.0 at the pre-registered 0.25-each split, so no per-event uniform
#: returns "elastic").
NOMINAL_ELASTIC_FRACTION = 0.0

#: The reserved round-5 raw seed blocks (index item 9; recorded so the
#: disjointness check covers the reserved neighbors explicitly).
RESERVED_ROUND5_BLOCKS = {
    "r51": [20300945, 20300952],
    "r52": [20300953, 20300960],
    "r53": [20300961, 20300968],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9).
RESERVED_ROUND5_PARTITIONS = {
    "r51": [1_100_000_000, 1_100_100_000],
    "r52": [1_200_000_000, 1_200_100_000],
    "r53": [1_300_000_000, 1_300_100_000],
    "r54": [1_400_000_000, 1_400_050_000],
    "r55": [1_500_000_000, 1_500_250_000],
    "round5_smoke": [1_600_000_000, 1_600_002_000],
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
}

#: The clean-environment validation's pre-registered probe slot (the nd
#: class, replica 0: the richest single-slot exercise of the dispatch,
#: the spectral draws, the central-source cascade, and the observable
#: extraction; the probe re-run is the determinism evidence of
#: acceptance (vi)).
PROBE_SLOT = 16


def slot_class(slot: int) -> str:
    """The class label of slot *slot* (the pre-registered order)."""
    k = int(slot)
    if not 0 <= k < N_SLOTS:
        _refuse(f"slot index {k} outside the R50 block 0..{N_SLOTS - 1}")
    return CLASS_ORDER[k // REPLICAS_PER_CLASS]


def slot_replica(slot: int) -> int:
    k = int(slot)
    if not 0 <= k < N_SLOTS:
        _refuse(f"slot index {k} outside the R50 block 0..{N_SLOTS - 1}")
    return k % REPLICAS_PER_CLASS


def slot_seed(slot: int) -> int:
    k = int(slot)
    if not 0 <= k < N_SLOTS:
        _refuse(f"slot index {k} outside the R50 block 0..{N_SLOTS - 1}")
    return SLOT_SEEDS[k]


def slot_name(slot: int) -> str:
    return RECORD_FMT.format(slot_class(slot), slot_replica(slot))


def event_id(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_id(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


def seed_disjointness_report() -> dict:
    """The seed-block disjointness check (the card's stop condition): the
    exact previously-used blocks recorded from the sealed
    pre-registrations and the round 5 index item 9, the shared raw seeds
    named, and the event-index partition disjointness verified.  Any
    (seed, event-index) substream collision refuses the run (never a
    silent re-block)."""
    problems: list[str] = []
    r50_lo = EVENT_INDEX_BASE
    r50_hi = EVENT_INDEX_BASE + EVENTS_PER_SLOT
    shared: dict[str, list[int]] = {}
    for name, block in PREVIOUS_USED_BLOCKS.items():
        seeds = block.get("raw_seeds")
        if seeds is not None:
            lo, hi = int(seeds[0]), int(seeds[-1])
            overlap = sorted(set(range(lo, hi + 1)) & set(SLOT_SEEDS))
            if overlap:
                shared[name] = [int(overlap[0]), int(overlap[-1])]
        p_range = block.get("event_index_range")
        if p_range is not None:
            p_lo, p_hi = p_range
            if r50_lo < p_hi and p_lo < r50_hi:
                problems.append(
                    f"the R50 event-index partition [{r50_lo}, {r50_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if sorted(set(range(lo, hi + 1)) & set(SLOT_SEEDS)):
            problems.append(
                f"the R50 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if r50_lo < hi and lo < r50_hi:
            problems.append(
                f"the R50 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r50] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r50_raw_seed_block": [min(SLOT_SEEDS), max(SLOT_SEEDS)],
        "r50_event_index_partition": EVENT_INDEX_RULE,
        "r50_event_index_range": [r50_lo, r50_hi],
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R50 raw block 20300925-20300944 shares no raw seed with "
            "any previously used block (R23 20300913-20300920 under the "
            "20260913 master ends below it; the Round 4 blocks "
            "20301020-20301083 and smoke blocks 20307920-20311920 under "
            "the 20260920 master sit above it; the R32 blocks "
            "20301519-20301826 under the 20260918 master are disjoint); "
            "the substreams are disjoint by construction (both the "
            "numpy.random.SeedSequence spawn tree and "
            "derive_substreams(20260925, 0, event_id) key on the raw "
            "seed and the event index) and the event-index partitions "
            "are disjoint"
        ),
        "reserved_round5_blocks": {
            name: list(block)
            for name, block in RESERVED_ROUND5_BLOCKS.items()},
        "reserved_round5_partitions": {
            name: list(block)
            for name, block in RESERVED_ROUND5_PARTITIONS.items()},
        "previously_used_blocks": PREVIOUS_USED_BLOCKS,
        "disjoint": True,
    }


# ---------------------------------------------------------------------------
# The nominal-parameter assertion (the card: the generation parameters
# equal TopologyNominalParams exactly; asserted before any generation)
# ---------------------------------------------------------------------------
def assert_nominal_parameters() -> dict:
    """The nominal-parameter assertion against the card-9 runner: the
    module-level ``NOMINAL_PARAMS`` equals the pre-registered R50 values
    exactly (epsilon = 0, B_SD byte-identical to the M8-frozen
    effective_transfer_slope, the inelastic fractions 0.25 each, the
    remnant weights intact 0.5 / low-mass 0.5 / high-mass 0.0, and the
    unretained default P(n = 2) = 0.0); the class set is exactly the six
    registry entries with nd_multi unretained and the elastic class
    outside the inelastic set."""
    params = NOMINAL_PARAMS
    if not isinstance(params, TopologyNominalParams):
        _refuse(
            "the runner's NOMINAL_PARAMS is not a TopologyNominalParams: "
            f"{type(params)!r}"
        )
    if params.epsilon != 0.0:
        _refuse(f"the nominal epsilon drifted: {params.epsilon!r}")
    if params.b_sd != ELASTIC_TRANSFER_SLOPE:
        _refuse(
            "the nominal B_SD is not byte-identical to the M8-frozen "
            f"effective_transfer_slope: {params.b_sd!r} vs "
            f"{ELASTIC_TRANSFER_SLOPE!r}"
        )
    expected_fractions = {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25,
    }
    if dict(params.class_fractions) != expected_fractions:
        _refuse(
            "the nominal inelastic fractions drifted: "
            f"{dict(params.class_fractions)!r}"
        )
    expected_weights = {
        "intact_proton": 0.5,
        "low_mass_baryonic_resonance": 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }
    if dict(params.remnant_weights) != expected_weights:
        _refuse(
            "the nominal remnant weights drifted: "
            f"{dict(params.remnant_weights)!r}"
        )
    if params.p_n2 != 0.0:
        _refuse(f"the unretained P(n = 2) default drifted: {params.p_n2!r}")
    if set(TOPOLOGY_CLASSES) != {
            "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}:
        _refuse(
            "the class registry drifted: " f"{sorted(TOPOLOGY_CLASSES)!r}")
    if TOPOLOGY_CLASSES["nd_multi"]["retained"]:
        _refuse("the nd_multi registry entry must carry retained: False")
    if TOPOLOGY_CLASSES["elastic"]["inelastic"]:
        _refuse("the elastic registry entry must carry inelastic: False")
    if set(INELASTIC_SET) != {"sd_projectile", "sd_target", "dd", "nd"}:
        _refuse(f"the inelastic set drifted: {sorted(INELASTIC_SET)!r}")
    if len(CLASS_ORDER) != 5 or len(set(CLASS_ORDER)) != 5:
        _refuse("the R50 class order is not the five distinct classes")
    mass_params = MassLawParams(epsilon=params.epsilon, b_sd=params.b_sd)
    if (mass_params.epsilon, mass_params.b_sd) != (params.epsilon,
                                                   params.b_sd):
        _refuse("the MassLawParams construction identity failed")
    elastic_fraction = 1.0 - sum(
        float(v) for v in params.class_fractions.values())
    if elastic_fraction != NOMINAL_ELASTIC_FRACTION:
        _refuse(
            "the nominal elastic fraction drifted: "
            f"{elastic_fraction!r} (expected {NOMINAL_ELASTIC_FRACTION!r})"
        )
    return {
        "epsilon": params.epsilon,
        "b_sd_gev2": params.b_sd,
        "class_fractions": dict(params.class_fractions),
        "remnant_weights": dict(params.remnant_weights),
        "p_n2": params.p_n2,
        "elastic_fraction": elastic_fraction,
        "nd_multi_status": (
            "registered but unretained (retained: False); excluded from "
            "R50 (it enters the round only through R53, plan line 926)"),
        "class_set": list(CLASS_ORDER),
        "inelastic_set": sorted(INELASTIC_SET),
        "mass_law_construction": {
            "epsilon": mass_params.epsilon, "b_sd": mass_params.b_sd,
        },
        "pass": True,
    }


_NOMINAL_STATUS = assert_nominal_parameters()


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
    card-pinned surfaces and the R45/R46 driver pins)."""
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
    against the pre-registered hash, and asserted against the nominal
    B_SD and the fixed blend ramp start."""
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
            f"{slope!r} is not byte-identical to the nominal B_SD "
            f"{ELASTIC_TRANSFER_SLOPE!r}"
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


# ---------------------------------------------------------------------------
# The per-event physics layer
# ---------------------------------------------------------------------------
EVENT_CLOSURE_TOL = 1e-9
INCOMING_REFERENCE_NOTE = (
    "the incoming sum of the pinned M8 p+p configuration: "
    "(E_beam + m_p, 0, 0, p_beam) with E_beam = sqrt(m_p^2 + p_beam^2), "
    "m_p = 0.93827231 GeV, p_beam = 158.0 GeV/c; every event's final "
    "state (the elastic class: its two outgoing lines) closes against "
    "this reference at tolerance 1e-9 relative"
)


class ConservationViolation(RuntimeError):
    """The pre-registered conservation check of the card fired (checked,
    never repaired; a violation raises and stops the run)."""


def _incoming_reference() -> tuple[float, float]:
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    return e_total, BEAM_MOMENTUM


def _line_qn(line) -> tuple[int, int, int, int]:
    """The doubled-I3 integer quantum numbers ``(B, Q, S, 2I3)`` of one
    ledger line state (the elastic module's ``qn`` tuple or the card-2
    ledger ``LineState`` fields, read uniformly)."""
    qn = getattr(line, "qn", None)
    if qn is not None:
        return (int(qn[0]), int(qn[1]), int(qn[2]), int(qn[3]))
    return (int(line.B), int(line.Q), int(line.S), int(line.I3x2))


def event_conservation(event) -> dict:
    """The per-event conservation record (the house conservation
    contract): the exact integer (B, Q, S) of the summed ledger states
    and the relative four-momentum closure components against the
    incoming reference.  Raises :class:`ConservationViolation` on any
    violation (checked, never repaired)."""
    e_total, p_beam = _incoming_reference()
    if isinstance(event, ElasticEvent):
        summed = list(event.outgoing)
    else:
        summed = list(event.final_state)
    if not summed:
        raise ConservationViolation(
            "the event carries no final state to sum "
            f"(class {event.class_label!r})")
    e_sum = math.fsum(float(h.four_momentum[0]) for h in summed)
    px_sum = math.fsum(float(h.four_momentum[1]) for h in summed)
    py_sum = math.fsum(float(h.four_momentum[2]) for h in summed)
    pz_sum = math.fsum(float(h.four_momentum[3]) for h in summed)
    b_total = sum(_line_qn(h)[0] for h in summed)
    q_total = sum(_line_qn(h)[1] for h in summed)
    s_total = sum(_line_qn(h)[2] for h in summed)
    if (b_total, q_total, s_total) != (2, 2, 0):
        raise ConservationViolation(
            f"event {event.class_label!r}: the summed (B, Q, S) = "
            f"{(b_total, q_total, s_total)}, expected (2, 2, 0) exactly"
        )
    d_e = abs(e_sum - e_total) / e_total
    d_px = abs(px_sum) / p_beam
    d_py = abs(py_sum) / p_beam
    d_pz = abs(pz_sum - p_beam) / p_beam
    worst = max(d_e, d_px, d_py, d_pz)
    if worst > EVENT_CLOSURE_TOL:
        raise ConservationViolation(
            f"event {event.class_label!r}: the four-momentum closure "
            f"residual {worst!r} exceeds the tolerance "
            f"{EVENT_CLOSURE_TOL!r} (dE {d_e!r}, dpx {d_px!r}, "
            f"dpy {d_py!r}, dpz {d_pz!r})"
        )
    worst_shell = 0.0
    for hadron in summed:
        e, px, py, pz = (float(x) for x in hadron.four_momentum)
        mass = float(getattr(hadron, "mass", PROTON_MASS))
        m2 = e * e - px * px - py * py - pz * pz
        scale = max(mass * mass, 1e-9)
        worst_shell = max(worst_shell, abs(m2 - mass * mass) / scale)
    if worst_shell > EVENT_CLOSURE_TOL:
        raise ConservationViolation(
            f"event {event.class_label!r}: the worst mass-shell "
            f"deviation {worst_shell!r} exceeds {EVENT_CLOSURE_TOL!r}"
        )
    return {
        "bqs": [int(b_total), int(q_total), int(s_total)],
        "closure": [d_e, d_px, d_py, d_pz],
        "closure_worst": worst,
        "mass_shell_worst": worst_shell,
        "d_energy_gev": abs(e_sum - e_total),
        "d_momentum_gev": max(abs(px_sum), abs(py_sum),
                              abs(pz_sum - p_beam)),
    }


def exchange_transfers(event) -> list:
    """The declared ``(B, Q, S, 2I3)`` transfer tuples of the event's
    exchange records (the card-2 ledger records attached to every event
    record; all vacuum transfers are zero at this substrate)."""
    if isinstance(event, ElasticEvent):
        return [tuple(int(v) for v in event.exchange.transfer)]
    if isinstance(event, SDEvent):
        return [tuple(int(v) for v in event.exchange[1])]
    if isinstance(event, DDEvent):
        return [tuple(int(v) for v in transfer)
                for _name, transfer in event.exchanges]
    # NDEvent (and the registered-but-unretained NDMultiEvent)
    return [tuple(int(v) for v in exchange.transfer)
            for exchange in event.exchanges]


def serialize_event(event, obs, conservation) -> dict:
    """The per-event record of the slot file (without its hash): the
    conservation block, the raw outgoing lines and final states (the
    full stored event: every ledger state and every four-momentum as an
    exact float), the excitation masses, the transfers, the gap, and
    the extracted observables.  The caller adds the event identity
    fields and the per-event hash: the sha256 of the record's canonical
    JSON without the ``h`` key, so the stored bytes themselves carry
    the bit-level identity evidence."""
    if isinstance(event, ElasticEvent):
        fs_records = []
    else:
        fs_records = [
            [
                (None if h.pdg_id is None else int(h.pdg_id)),
                int(h.qn[0]), int(h.qn[1]), int(h.qn[2]), int(h.qn[3]),
                float(h.mass),
                float(h.four_momentum[0]), float(h.four_momentum[1]),
                float(h.four_momentum[2]), float(h.four_momentum[3]),
            ]
            for h in event.final_state
        ]
    lines = [
        [
            float(line.four_momentum[0]), float(line.four_momentum[1]),
            float(line.four_momentum[2]), float(line.four_momentum[3]),
        ] + list(_line_qn(line))
        for line in _baryonic_lines(event)
    ]
    record = {
        "bqs": conservation["bqs"],
        "clo": conservation["closure"],
        "ex": [float(m) for m in obs.excitation_masses],
        "fs": fs_records,
        "gp": (None if obs.gap_size is None else float(obs.gap_size)),
        "ln": lines,
        "ob": [
            float(obs.x_f_projectile), float(obs.x_f_target),
            int(obs.multiplicity_total), int(obs.multiplicity_central),
            int(obs.central_pions), int(obs.central_protons),
            int(obs.forward_pions), int(obs.forward_protons),
            float(obs.seagull_moment),
        ],
        "sh": conservation["mass_shell_worst"],
        "tr": [float(t) for t in obs.transfers],
        "xt": exchange_transfers(event),
    }
    return record


def generate_slot_events(
    class_label: str,
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
    *,
    mixed_draw: bool = False,
    params: TopologyNominalParams | None = None,
):
    """Generate *n_events* events of one slot and return the per-event
    records, the class census, the conservation aggregate, and the
    event-hash set digest.

    The production slots generate their fixed class directly through
    :func:`run_topology` (the pre-registered one-class-per-slot T1
    template design).  The smoke slot sets *mixed_draw* to exercise the
    card-9 per-event class draw on the closed ``topology_selection``
    substream at the nominal fractions end to end.  Every event runs the
    conservation contract (a violation raises and stops the run) and the
    elastic class runs its zero-transfer code check.
    """
    if params is None:
        params = NOMINAL_PARAMS
    records = []
    census: dict[str, int] = {}
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
        if mixed_draw:
            label = draw_topology(
                params.class_fractions, NOMINAL_ELASTIC_FRACTION,
                rng_topology_selection)
        else:
            label = class_label
        event = run_topology(
            label, params, catalog, fams,
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
        conservation = event_conservation(event)
        record = serialize_event(event, obs, conservation)
        record["i"] = int(i)
        record["id"] = int(event_index_base + i)
        if mixed_draw:
            record["class_label"] = label
        # the per-event hash covers the full stored record including the
        # event index and id, over the canonical JSON without the h key
        # itself (so the stored bytes reproduce the hash)
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
    conservation = {
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
    return records, census, conservation, hash_set


# ---------------------------------------------------------------------------
# The elastic normalization check (plan line 824; the card-1 acceptance)
# ---------------------------------------------------------------------------
def elastic_normalization_check(elastic_records: list[dict]) -> dict:
    """The elastic class's pre-registered normalization and code checks
    at the recorded tolerances: the closed-form normalization against
    quadrature within 1e-12; the inverse-CDF round trip on the pinned
    257-point quantile grid within 1e-9; the pooled elastic ``t`` mean
    against the analytic mean within 3 sigma of 1/sqrt(N); zero
    zero-transfer violations (checked at generation); the Feynman-x
    pair sum against zero; and the mass shell of the outgoing lines."""
    t_min, _t_max = elastic_t_range(SQRT_S)
    b_slope = ELASTIC_TRANSFER_SLOPE
    closed = elastic_t_normalization(t_min)
    quadrature = _quadrature_normalization(t_min, b_slope)
    normalization_dev = abs(closed - quadrature)
    roundtrip_dev = 0.0
    for j in range(1, 258):
        u = j / 257.0
        t_u = elastic_t_quantile(u, t_min)
        cdf = (math.exp(b_slope * t_u) - math.exp(b_slope * t_min)) / (
            1.0 - math.exp(b_slope * t_min))
        roundtrip_dev = max(roundtrip_dev, abs(cdf - u))
    t_values = [float(r["tr"][0]) for r in elastic_records]
    n = len(t_values)
    t_mean = math.fsum(t_values) / n
    t_analytic_mean = _analytic_t_mean(t_min, b_slope)
    t_var = math.fsum((t - t_mean) ** 2 for t in t_values) / n
    t_sd = math.sqrt(max(0.0, t_var))
    sigma_mean = t_sd / math.sqrt(n)
    z_mean = ((t_mean - t_analytic_mean) / sigma_mean
              if sigma_mean > 0.0 else float("inf"))
    xf_pair_sum_max = max(
        abs(float(r["ob"][0]) + float(r["ob"][1]))
        for r in elastic_records)
    xf_range_ok = all(
        -1.0 <= float(r["ob"][0]) <= 1.0
        and -1.0 <= float(r["ob"][1]) <= 1.0
        for r in elastic_records)
    mass_shell_worst = max(float(r["sh"]) for r in elastic_records)
    checks = {
        "normalization_identity": {
            "closed_form": closed,
            "quadrature": quadrature,
            "quadrature_convention": (
                "scipy.integrate.quad, epsabs = epsrel = 1e-13, limit "
                "400 (the landed card-1 test_t_normalization convention)"),
            "max_dev": normalization_dev,
            "tolerance": 1e-12,
            "pass": bool(normalization_dev <= 1e-12),
        },
        "quantile_roundtrip": {
            "max_dev": roundtrip_dev,
            "tolerance": 1e-9,
            "grid": ("u_j = j / 257, j = 1..257 (the pinned 257-point "
                     "grid; u = 0 is excluded because exp(B_SD t_min) "
                     "underflows and no float64 uniform reaches the "
                     "range edge, the card-1 documented domain fact)"),
            "pass": bool(roundtrip_dev <= 1e-9),
        },
        "t_sample_mean": {
            "sample_mean_gev2": t_mean,
            "analytic_mean_gev2": t_analytic_mean,
            "sample_sd_gev2": t_sd,
            "sigma_of_mean_gev2": sigma_mean,
            "z": z_mean,
            "tolerance_sigma": 3.0,
            "n": n,
            "pass": bool(abs(z_mean) <= 3.0),
        },
        "zero_transfer_violations": 0,
        "zero_transfer_note": (
            "check_zero_transfer ran on every elastic event at "
            "generation; a violation raises and stops the run"),
        "xf_pair_sum_max_abs": xf_pair_sum_max,
        "xf_pair_sum_tolerance": 1e-9,
        "xf_in_unit_range": bool(xf_range_ok),
        "mass_shell_worst_relative": mass_shell_worst,
    }
    verdict = bool(
        checks["normalization_identity"]["pass"]
        and checks["quantile_roundtrip"]["pass"]
        and checks["t_sample_mean"]["pass"]
        and xf_pair_sum_max <= 1e-9
        and xf_range_ok
        and mass_shell_worst <= EVENT_CLOSURE_TOL
    )
    return {
        "verdict": "pass" if verdict else "fail",
        "tolerance_note": (
            "the recorded tolerances: normalization identity 1e-12, "
            "quantile round trip 1e-9, t mean 3 sigma of 1/sqrt(N), "
            "x_F pair sum 1e-9, mass shell 1e-9 relative"),
        "checks": checks,
    }


def _quadrature_normalization(t_min: float, b_slope: float,
                              _n_nodes: int = 0) -> float:
    """The adaptive quadrature of exp(B t) on [t_min, 0] (the
    normalization identity check's quadrature reference; the landed
    card-1 convention: scipy.integrate.quad at epsabs = epsrel = 1e-13,
    limit 400, warnings ignored around the underflowed lower edge)."""
    import warnings

    from scipy.integrate import quad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quad_value, _err = quad(
            lambda t: math.exp(b_slope * t), t_min, 0.0,
            epsabs=1e-13, epsrel=1e-13, limit=400)
    return quad_value


def _analytic_t_mean(t_min: float, b_slope: float) -> float:
    """The analytic mean of exp(B t) on [t_min, 0]:
    (-1/B + exp(B t_min) (1/B - t_min)) / (1 - exp(B t_min)); at the
    pinned configuration exp(B t_min) underflows to 0 and the value is
    exactly -1/B."""
    e_lo = math.exp(b_slope * t_min)
    return (-1.0 / b_slope + e_lo * (1.0 / b_slope - t_min)) / (1.0 - e_lo)


# ---------------------------------------------------------------------------
# The slot records (the deterministic generation records)
# ---------------------------------------------------------------------------
SLOT_SCHEMA = "na49_mixing.staged_r50_pure_topology_templates.slot/1"


def _identity_block(run_dir: Path) -> dict:
    return {
        "driver_sha256": DRIVER_SHA256,
        "pre_registration_sha256": _sha256_file(
            run_dir / PREREGISTRATION_NAME),
    }


def build_slot_record(
    slot: int,
    records: list,
    census: dict,
    conservation: dict,
    hash_set: str,
    run_dir: Path,
    mode: str = "production",
    mixed_note: str | None = None,
) -> dict:
    """The slot record: the identity block, the seed map, the per-event
    records, the class census, the conservation aggregate, and the
    event-hash set digest (the generation-time determinism evidence)."""
    return {
        "schema": SLOT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "slot": int(slot),
        "class_label": slot_class(slot),
        "replica": slot_replica(slot),
        "seed": slot_seed(slot),
        "n_events": len(records),
        "identity": _identity_block(run_dir),
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                f"raw slot seed = 20260925 + k = 20300925 + k per slot "
                f"k < {N_SLOTS} (slot k = class_index * 4 + "
                "replica_index in the class order elastic, sd_projectile, "
                "sd_target, dd, nd; the block 20300925-20300944 is "
                "disjoint from every previously used raw block)"),
            "event_index_rule": EVENT_INDEX_RULE,
            "substream_labels": sorted(SUBSTREAM_LABELS),
            "consumed_labels": [
                "topology_selection", "spectral_state",
                "exchange_kinematics", "decay_channel", "decay_kinematics",
            ],
        },
        "class_census": dict(census),
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
        "mixed_draw_note": mixed_note,
    }


def load_pre_registration(run_dir: Path, *, smoke: bool = False) -> dict:
    name = ("smoke_" + PREREGISTRATION_NAME) if smoke else PREREGISTRATION_NAME
    path = Path(run_dir) / name
    if not path.is_file():
        _refuse(f"the run dir carries no pre-registration: {path}")
    return _read_json(path)


def _verify_slot_identity(record: dict, slot: int, run_dir: Path) -> None:
    """A completed production slot record must match the pre-registered
    identity (the seed, the class, the event count, the driver digest,
    and the pre-registration digest); any drift refuses the run (a
    completed record is reused unchanged or refused, never silently
    overwritten)."""
    expected_seed = slot_seed(slot)
    expected_class = slot_class(slot)
    expected_replica = slot_replica(slot)
    if record.get("mode") != "production":
        _refuse(
            f"the slot record {slot_name(slot)} is not a production "
            f"record (mode {record.get('mode')!r})")
    if record.get("slot") != int(slot):
        _refuse(
            f"the slot record {slot_name(slot)} carries slot "
            f"{record.get('slot')!r}, expected {int(slot)}")
    if record.get("seed") != expected_seed:
        _refuse(
            f"the slot record {slot_name(slot)} carries seed "
            f"{record.get('seed')!r}, expected {expected_seed}")
    if record.get("class_label") != expected_class:
        _refuse(
            f"the slot record {slot_name(slot)} carries class "
            f"{record.get('class_label')!r}, expected {expected_class!r}")
    if record.get("replica") != expected_replica:
        _refuse(
            f"the slot record {slot_name(slot)} carries replica "
            f"{record.get('replica')!r}, expected {expected_replica}")
    if record.get("n_events") != EVENTS_PER_SLOT or len(
            record.get("events", [])) != EVENTS_PER_SLOT:
        _refuse(
            f"the slot record {slot_name(slot)} carries "
            f"{record.get('n_events')} events, expected "
            f"{EVENTS_PER_SLOT}")
    identity = record.get("identity") or {}
    current = _identity_block(run_dir)
    if identity.get("pre_registration_sha256") != current[
            "pre_registration_sha256"]:
        _refuse(
            f"the slot record {slot_name(slot)} was generated against a "
            "different pre-registration digest "
            f"({identity.get('pre_registration_sha256')!r} vs "
            f"{current['pre_registration_sha256']!r}); the run refuses "
            "to mix pre-registrations")
    if identity.get("driver_sha256") != current["driver_sha256"]:
        _refuse(
            f"the slot record {slot_name(slot)} was generated by a "
            "different driver digest "
            f"({identity.get('driver_sha256')!r} vs "
            f"{current['driver_sha256']!r})")
    # the stored event-hash set must match the stored events
    recomputed = hashlib.sha256(
        "\n".join(r["h"] for r in record["events"]).encode(
            "utf-8")).hexdigest()
    if recomputed != record.get("event_hash_set_sha256"):
        _refuse(
            f"the slot record {slot_name(slot)} carries a stale "
            "event-hash set digest (the stored events do not reproduce "
            "it)")


def expected_class_slot(slot: int):
    return slot_class(slot)


def expected_class_label(label: str):
    return label


def run_generation(
    run_dir: Path,
    slots: list[int] | None = None,
) -> list:
    """Generate the requested production slot records (the default is
    every slot), one record file per slot, then stop.  Requires the
    pre-registration.  A completed record that matches the
    pre-registered identity is reused unchanged."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    load_pre_registration(run_dir)
    if slots is None:
        slots = list(range(N_SLOTS))
    written = []
    catalog = _load_catalog()
    fams = _load_families()
    for slot in slots:
        name = slot_name(slot)
        path = run_dir / name
        if path.is_file():
            existing = _read_json(path)
            _verify_slot_identity(existing, slot, run_dir)
            print(f"[r50] slot {int(slot)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        label = slot_class(slot)
        seed = slot_seed(slot)
        print(
            f"[r50] generating slot {int(slot)} ({name}): class "
            f"{label}, seed {seed}, {EVENTS_PER_SLOT} events",
            flush=True,
        )
        records, census, conservation, hash_set = generate_slot_events(
            label, seed, EVENTS_PER_SLOT, EVENT_INDEX_BASE, catalog, fams)
        record = build_slot_record(
            slot, records, census, conservation, hash_set, run_dir)
        record["reused"] = False
        _write_json(path, record)
        print(
            f"[r50] slot {int(slot)} written: {path} "
            f"(worst closure {conservation['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# The pre-registration (written before any generation)
# ---------------------------------------------------------------------------
PREREGISTRATION_SCHEMA = (
    "na49_mixing.staged_r50_pure_topology_templates.pre_registration/1")

OBJECTIVE_NOTE = (
    "plan section 11 run R50 (lines 898-907): 'Generate T1 samples for "
    "every class with fixed nominal parameters. Store: leading-baryon "
    "x_F; gap-size distribution; excitation masses; multiplicity "
    "distribution; central and forward species yields; seagull "
    "moments.'  The pre-registered instantiation (index items 2 and 8): "
    "the five classes elastic, sd_projectile, sd_target, dd, nd; the "
    "nd_multi class excluded (unretained, P(n = 2) = 0.0); the elastic "
    "class generated at T1 for the normalization check and excluded "
    "from the inelastic statistics of R51, R52, R54, and R55 (plan line "
    "824).  The generation runs through the card-9 runner "
    "(draw_topology, run_topology, extract_observables) on the landed "
    "class modules of cards 1, 5, 6, and 7, with card 2's ledger "
    "records attached to every event record.  No optimizer on this run."
)

COLLISION_NOTE = (
    "the R32 p+p configuration (index item 8): p+p at sqrt(s) = 17.270 "
    "GeV (beam 158 GeV/c), the M8-frozen ten-parameter set (hash "
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1) "
    "through the ResonancePPBuilder surface with the card-6 routing at "
    "the fixed blend ramp start B = 2.680194110727528 GeV and the "
    "card-9 feed-down lineage convention; the blend ramp start is never "
    "a free parameter.  The topology layer replaces the single "
    "two-source inelastic topology: the M8 set's two (B, Q, S) = (1, 1, "
    "0) sources are the incoming lines.  The frozen simplified cascade "
    "at the sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = "
    "2.0 GeV^-2, baryonic nominal width 0.3 GeV) is consumed without "
    "refit (the R46 no_refit clause); the exchange slopes act on the "
    "exchange kinematics only, not on the daughter transverse dynamics "
    "(plan section 24, line 1866); no Round 5 record claims a "
    "separation of exchange recoil from decay transverse dynamics (the "
    "R46 outcome, separation_claimed false)."
)

STATISTICS_NOTE = (
    "T1 (plan section 5.1 line 253): 4 replicas x 25,000 events per "
    "class across the five classes = 100,000 events per class, 500,000 "
    "events total; one slot per (class, replica) pair."
)

CLASS_SET_NOTE = (
    "the class set (index item 2): the five classes elastic, "
    "sd_projectile, sd_target, dd, nd; nd_multi excluded (registered "
    "via the card-9 registry entry, unretained, P(n = 2) = 0.0; it "
    "enters the round only through R53, plan line 926); the elastic "
    "class excluded from the inelastic statistics of R51, R52, R54, and "
    "R55 (plan line 824)."
)

STORED_QUANTITIES = (
    "leading_baryon_xf",
    "gap_size",
    "excitation_masses",
    "multiplicity",
    "central_forward_yields",
    "seagull_moments",
)
STORED_QUANTITIES_NOTE = (
    "the six stored quantities of plan line 901, computed per event by "
    "the card-9 extract_observables at the pinned CM-frame convention "
    "(x_F = 2 p_L^CM / sqrt(s) with the runner's SQRT_S = 17.270 GeV "
    "denominator and the central window |y^CM| < 0.5, the card-9 review "
    "fix); the seagull moment is the arithmetic mean transverse "
    "momentum of the final state (the plan section 7.3 m_T-slope "
    "moment building block, lines 420-431 and 442)"
)

DETERMINISM_NOTE = (
    "a per-slot re-run from a clean interpreter reproduces the report "
    "hashes; the report and the gate record carry no wall-clock fields "
    "(their bytes are a pure function of the sealed artifacts, the "
    "pre-registered seeds, and the committed code); the per-event hash "
    "is the sha256 of the canonical JSON of the stored event record and "
    "the event_hash_set_sha256 is the sha256 over the joined per-event "
    "hashes, so the stored events carry the bit-level identity evidence "
    "and the probe slot's clean re-run must reproduce them"
)

SUBSTREAM_COLLISION_RULE = (
    "if any raw (seed, event-index) substream collides with a "
    "pre-registered block, the run stops and reports; a silent re-block "
    "never happens (index item 9)"
)

GENERATION_NOTE = (
    "gateway-primary generation (one job per (class, replica) slot, "
    "25,000 events per slot, per the pre-registered seeds, with "
    "per-slot checkpoint files; one job per slot, --tasks 1, single "
    "math thread (NA49_CLUSTER_GW_TASKS forces thread counts to 1), no "
    "local process pool inside the chunk command, submission paced by "
    "the capacity check, chunk outputs appended only to the new run "
    "dir); the driver's local fallback is one single-threaded process "
    "per --slot invocation (the pre-registered one-job-per-slot "
    "granularity; a completed record that matches the pre-registered "
    "identity is reused unchanged)"
)

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(ix) of the card: (i) sample "
    "completeness: 20 slots, exactly 25,000 events per slot, 100,000 "
    "per class, 500,000 total, the per-class class-draw counts at the "
    "pre-registered nominal fractions; (ii) the six stored quantities "
    "recorded per class, pooled over the four replicas; (iii) "
    "nominal-parameter fidelity: the generation parameters equal "
    "TopologyNominalParams exactly and nd_multi events are absent from "
    "the inelastic sample; (iv) the elastic normalization check passes "
    "at the recorded tolerance and the elastic class is excluded from "
    "the inelastic statistics; (v) zero B, Q, S violations and zero "
    "four-momentum closure failures at tolerance 1e-09 on all 500,000 "
    "events with the worst closure residual recorded in GeV; (vi) the "
    "clean-environment re-run reproduces the per-slot report hashes; "
    "(vii) exactly one clean-environment validation pass executes "
    "before the seal and the seal is intact afterwards; (viii) "
    "optimizer_invoked is false, no parameter is fit; (ix) the driver "
    "and test file land byte-identical in both trees, the freeze re-pin "
    "is recorded, and the full nested suite is green"
)


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
    payload = {
        "schema": ("na49_mixing.staged_r50_pure_topology_templates."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
        "mode": "smoke" if smoke else "production",
        "recorded_utc": None if smoke else _now_utc(),
        "objective": OBJECTIVE_NOTE,
        "collision": COLLISION_NOTE,
        "statistics": {
            "tier": STATISTICS_NOTE,
            "replicas_per_class": REPLICAS_PER_CLASS,
            "events_per_slot": EVENTS_PER_SLOT,
            "events_per_class": N_EVENTS_PER_CLASS,
            "events_total": N_EVENTS_TOTAL,
            "slots": N_SLOTS,
            "class_order": list(CLASS_ORDER),
        },
        "class_set": CLASS_SET_NOTE,
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw slot seed = 20260925 + k per slot k < 20 (slot k = "
                "class_index * 4 + replica_index; class order elastic, "
                "sd_projectile, sd_target, dd, nd); the block "
                "20300925-20300944 is disjoint from every previously "
                "used raw block (index item 9)"),
            "raw_slot_seeds": list(SLOT_SEEDS),
            "per_class_seed_blocks": {
                label: [
                    SLOT_SEEDS[class_index * REPLICAS_PER_CLASS],
                    SLOT_SEEDS[class_index * REPLICAS_PER_CLASS
                               + REPLICAS_PER_CLASS - 1],
                ]
                for class_index, label in enumerate(CLASS_ORDER)
            },
            "event_index_rule": EVENT_INDEX_RULE,
            "event_index_base": EVENT_INDEX_BASE,
            "substream_labels": list(SUBSTREAM_LABELS),
            "disjointness": disjointness,
            "smoke": {
                "raw_seed": SMOKE_SEED,
                "raw_seed_rule": (
                    "raw smoke seed = 20260925 + 40000 + 12000 = "
                    "20312925 (the round-5 +12000 smoke block, disjoint "
                    "from every production slot and from every "
                    "previously used smoke block)"),
                "events": SMOKE_EVENTS,
                "event_index_rule": SMOKE_EVENT_INDEX_RULE,
            },
            "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        },
        "nominal_parameters": dict(_NOMINAL_STATUS),
        "stored_quantities": list(STORED_QUANTITIES),
        "stored_quantities_note": STORED_QUANTITIES_NOTE,
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
        "generation": GENERATION_NOTE,
        "no_optimizer": True,
        "optimizer_invoked": False,
        "optimizer_note": (
            "no optimizer is invoked on R50 (index item 10); the "
            "round's optimizer calls are R51 (card 11, the primary "
            "fit) and R53 (card 13, conditional); nothing on this run "
            "is fit"),
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
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
        "elastic_normalization_check": {
            "definition": (
                "the elastic class's pre-registered normalization and "
                "code checks (plan line 824): the closed-form "
                "normalization of exp(B_SD t) on [t_min, 0] against "
                "quadrature within 1e-12; the inverse-CDF round trip on "
                "the pinned 257-point quantile grid within 1e-9; the "
                "pooled elastic t mean against the analytic mean within "
                "3 sigma of 1/sqrt(N); check_zero_transfer with zero "
                "violations on every elastic event; the outgoing lines "
                "on the m_p mass shell; and the Feynman-x pair sum "
                "against zero"),
            "excluded_from_inelastic_statistics": True,
        },
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r50_pure_topology_templates."
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
        print(f"[r50] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r50] pre-registration written: {pre_path}", flush=True)
    return payload


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
# The hermetic smoke mode (driver validation; never sealed)
# ---------------------------------------------------------------------------
def run_smoke(run_dir: Path) -> dict:
    """The hermetic end-to-end driver validation: a smoke
    pre-registration, one mixed slot x 2,000 events at the smoke seed
    20312925 (the per-event class draw at the nominal fractions through
    ``draw_topology``), and the smoke evaluation, into an explicit
    ``--run-dir`` (the default production run dir is refused).  Two
    identical smoke invocations produce byte-identical artifacts (no
    wall-clock and no git state in the smoke artifacts)."""
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
        "schema": ("na49_mixing.staged_r50_pure_topology_templates."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "statistics": {
            "note": (
                "the smoke statistics: one mixed slot x 2,000 events at "
                "the smoke seed 20312925; the smoke exercises "
                "pre-registration, generation (through the card-9 "
                "draw_topology at the nominal fractions), recording, and "
                "evaluation end to end; the production statistics are "
                "the 20 pre-registered slots of 25,000 events"),
            "replicas_per_class": REPLICAS_PER_CLASS,
            "events_per_slot": EVENTS_PER_SLOT,
            "events_per_class": N_EVENTS_PER_CLASS,
            "events_total": N_EVENTS_TOTAL,
            "slots": N_SLOTS,
            "class_order": list(CLASS_ORDER),
            "smoke_events": SMOKE_EVENTS,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "smoke_raw_seed": SMOKE_SEED,
            "smoke_event_index_rule": SMOKE_EVENT_INDEX_RULE,
            "production_raw_seed_rule": (
                "raw slot seed = 20260925 + k per slot k < 20"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "nominal_parameters": dict(_NOMINAL_STATUS),
        "stored_quantities": list(STORED_QUANTITIES),
        "stored_quantities_note": STORED_QUANTITIES_NOTE,
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
        "no_optimizer": True,
        "optimizer_invoked": False,
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
            "in the smoke artifacts)"),
    }
    _write_json(run_dir / ("smoke_" + PREREGISTRATION_NAME), smoke_pre)
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set = generate_slot_events(
        "elastic",  # the fixed-class argument is unused by the mixed draw
        SMOKE_SEED, SMOKE_EVENTS, SMOKE_EVENT_INDEX_BASE, catalog, fams,
        mixed_draw=True)
    slot_record = {
        "schema": SLOT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "slot": "smoke_mixed",
        "class_label": (
            "mixed (the per-event class draw at the nominal fractions; "
            "no elastic event is drawn at the nominal elastic fraction "
            "0.0)"),
        "replica": None,
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
                "the closed topology_selection substream at the nominal "
                "fractions (the R51 fraction-sampling surface); the "
                "production slots dispatch their fixed class through "
                "the pre-registered slot map"),
        },
        "class_census": dict(census),
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }
    _write_json(run_dir / "smoke_slot.json", slot_record)
    report = run_evaluate_machinery(run_dir, smoke=True)
    _write_json(run_dir / ("smoke_" + REPORT_NAME), report)
    print(
        f"[r50] smoke complete: {len(records)} events, census {census}, "
        f"worst closure {conservation['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {"events": len(records), "census": census}


def disjointness_report_for_smoke() -> dict:
    """The smoke pre-registration's disjointness record (identical to
    the production check; the smoke seed and partition are additionally
    disjoint from the R50 production block and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(SLOT_SEEDS):
        _refuse("the smoke seed collides with the R50 production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_SLOT):
        _refuse("the smoke event-index partition overlaps the R50 "
                "production partition")
    report["smoke_raw_seed"] = SMOKE_SEED
    report["smoke_event_index_range"] = [smoke_lo, smoke_hi]
    report["smoke_disjoint_from_production"] = True
    return report


# ---------------------------------------------------------------------------
# The evaluation (the six stored quantities; the report machinery)
# ---------------------------------------------------------------------------
REPORT_SCHEMA = "na49_mixing.staged_r50_pure_topology_templates.report/1"
XF_HIST_BINS = 42
XF_HIST_LO, XF_HIST_HI = -1.05, 1.05
GAP_HIST_BINS = 50
GAP_HIST_LO, GAP_HIST_HI = 0.0, 17.28
MASS_HIST_BINS = 90
MASS_HIST_LO, MASS_HIST_HI = 0.0, 18.0


def _histogram(values: list, lo: float, hi: float, n_bins: int) -> dict:
    """The deterministic histogram of *values* over [lo, hi] with
    *n_bins* equal-width bins; out-of-range values are counted in the
    out_of_range block, never bled into the end bins."""
    edges = [lo + (hi - lo) * j / n_bins for j in range(n_bins + 1)]
    counts = [0] * n_bins
    out_lo = 0
    out_hi = 0
    for v in values:
        if v < lo:
            out_lo += 1
        elif v > hi:
            out_hi += 1
        else:
            idx = int((v - lo) / (hi - lo) * n_bins)
            if idx >= n_bins:
                idx = n_bins - 1
            counts[idx] += 1
    return {
        "edges": edges,
        "counts": counts,
        "out_of_range_low": out_lo,
        "out_of_range_high": out_hi,
    }


def _stats(values: list) -> dict:
    """The deterministic summary statistics (population convention;
    exact ``math.fsum`` accumulation)."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "convention": "population (ddof = 0), math.fsum accumulation"}
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(max(0.0, var)),
        "min": min(values),
        "max": max(values),
        "convention": "population (ddof = 0), math.fsum accumulation",
    }


def _aggregate_class(class_label: str, records: list[dict]) -> dict:
    """The per-class pooled aggregates of the six stored quantities
    (plan line 901) from the stored per-event records."""
    xf_p = [float(r["ob"][0]) for r in records]
    xf_t = [float(r["ob"][1]) for r in records]
    n_total = [int(r["ob"][2]) for r in records]
    n_cen = [int(r["ob"][3]) for r in records]
    c_pi = [int(r["ob"][4]) for r in records]
    c_pr = [int(r["ob"][5]) for r in records]
    f_pi = [int(r["ob"][6]) for r in records]
    f_pr = [int(r["ob"][7]) for r in records]
    sg = [float(r["ob"][8]) for r in records]
    gap = [float(r["gp"]) for r in records if r["gp"] is not None]
    gap_absent = sum(1 for r in records if r["gp"] is None)
    masses = [float(m) for r in records for m in r["ex"]]
    multiplicity_counts: dict[int, int] = {}
    for n in n_total:
        multiplicity_counts[n] = multiplicity_counts.get(n, 0) + 1
    transfers = [float(t) for r in records for t in r["tr"]]
    n_events = len(records)
    return {
        "events": n_events,
        "leading_baryon_xf": {
            "projectile": {**_stats(xf_p),
                           "histogram": _histogram(
                               xf_p, XF_HIST_LO, XF_HIST_HI,
                               XF_HIST_BINS)},
            "target": {**_stats(xf_t),
                       "histogram": _histogram(
                           xf_t, XF_HIST_LO, XF_HIST_HI, XF_HIST_BINS)},
        },
        "gap_size": {
            "present": len(gap),
            "absent_no_gap_field": gap_absent,
            "gap_field_note": (
                "the gap size is recorded on the sd_projectile, "
                "sd_target, and dd classes (the record's gap_size); the "
                "elastic and nd classes carry no gap field (None)"),
            **_stats(gap),
            "histogram": _histogram(gap, GAP_HIST_LO, GAP_HIST_HI,
                                    GAP_HIST_BINS),
        },
        "excitation_masses": {
            "count": len(masses),
            "per_event_note": (
                "the excitation masses per event: one for sd_projectile/"
                "sd_target (the SD excitation) and nd (the central "
                "mass), two for dd (the pair); the elastic class has "
                "none"),
            **_stats(masses),
            "histogram": _histogram(masses, MASS_HIST_LO, MASS_HIST_HI,
                                    MASS_HIST_BINS),
        },
        "transfers": _stats(transfers),
        "multiplicity": {
            **_stats([float(n) for n in n_total]),
            "distribution": {
                str(n): c for n, c in sorted(multiplicity_counts.items())
            },
            "central_activity_mean": (
                math.fsum(n_cen) / n_events if n_events else None),
        },
        "central_forward_yields": {
            "central_pions_mean": (math.fsum(c_pi) / n_events
                                   if n_events else None),
            "central_protons_mean": (math.fsum(c_pr) / n_events
                                     if n_events else None),
            "forward_pions_mean": (math.fsum(f_pi) / n_events
                                   if n_events else None),
            "forward_protons_mean": (math.fsum(f_pr) / n_events
                                     if n_events else None),
            "central_pions_total": sum(c_pi),
            "central_protons_total": sum(c_pr),
            "forward_pions_total": sum(f_pi),
            "forward_protons_total": sum(f_pr),
        },
        "seagull_moments": _stats(sg),
    }


def run_evaluate_machinery(run_dir: Path, *, smoke: bool = False) -> dict:
    """The evaluation: load the slot records, pool the per-class
    aggregates of the six stored quantities (plan line 901), record the
    class-draw census, the elastic normalization check, and the
    conservation report, and return the report dict with its digest
    (no wall-clock: the bytes are a pure function of the stored
    artifacts, the pre-registered seeds, and the committed code)."""
    run_dir = Path(run_dir)
    if smoke:
        record_files = ["smoke_slot.json"]
    else:
        record_files = [slot_name(k) for k in range(N_SLOTS)]
        for name in record_files:
            if not (run_dir / name).is_file():
                _refuse(f"the production record is missing: "
                        f"{run_dir / name}")
    pre = load_pre_registration(run_dir, smoke=smoke)
    slots = []
    class_records: dict[str, list[dict]] = {}
    for name in record_files:
        record = _read_json(run_dir / name)
        slots.append(record)
        for event in record["events"]:
            label = (event["class_label"] if smoke
                     else record["class_label"])
            class_records.setdefault(label, []).append(event)
    # the pooled conservation report over the full sample
    violations = 0
    events_checked = 0
    worst_closure = 0.0
    worst_shell = 0.0
    worst_energy_gev = 0.0
    worst_momentum_gev = 0.0
    bqs_census: dict[str, int] = {}
    for record in slots:
        cons = record["conservation"]
        violations += int(cons["violations"])
        events_checked += int(cons["events"])
        worst_closure = max(worst_closure,
                            float(cons["worst_closure_relative"]))
        worst_shell = max(worst_shell,
                          float(cons["worst_mass_shell_relative"]))
        worst_energy_gev = max(worst_energy_gev,
                               float(cons["worst_energy_closure_gev"]))
        worst_momentum_gev = max(worst_momentum_gev,
                                 float(cons["worst_momentum_closure_gev"]))
        for key, count in cons["b_q_s_census"].items():
            bqs_census[key] = bqs_census.get(key, 0) + int(count)
    conservation_report = {
        "events_checked": events_checked,
        "violations": violations,
        "tolerance": EVENT_CLOSURE_TOL,
        "worst_closure_relative": worst_closure,
        "worst_mass_shell_relative": worst_shell,
        "worst_energy_closure_gev": worst_energy_gev,
        "worst_momentum_closure_gev": worst_momentum_gev,
        "b_q_s_census": bqs_census,
        "pass": bool(violations == 0
                     and events_checked == sum(
                         len(r["events"]) for r in slots)),
    }
    # the elastic normalization check (production only; the mixed smoke
    # draw never returns "elastic" at the nominal elastic fraction 0.0)
    elastic_check = None
    if not smoke:
        elastic_check = elastic_normalization_check(
            class_records.get("elastic", []))
        if elastic_check["verdict"] != "pass":
            _refuse(
                "the elastic normalization check failed: "
                f"{elastic_check}")
    # the class-draw census (production: the exact slot allocation)
    if smoke:
        class_draw_census = {
            label: {
                "events": len(events),
                "note": ("the mixed smoke draw at the nominal "
                         "fractions; the per-event class draw on the "
                         "closed topology_selection substream"),
            }
            for label, events in sorted(class_records.items())
        }
    else:
        class_draw_census = {}
        for class_index, label in enumerate(CLASS_ORDER):
            events = len(class_records.get(label, []))
            inelastic = label in INELASTIC_CLASS_ORDER
            expected = (N_EVENTS_PER_CLASS if inelastic
                        else N_EVENTS_PER_CLASS)
            if events != expected:
                _refuse(
                    f"the class {label} carries {events} events, "
                    f"expected {expected}")
            class_draw_census[label] = {
                "events": events,
                "replicas": REPLICAS_PER_CLASS,
                "seed_block": [
                    SLOT_SEEDS[class_index * REPLICAS_PER_CLASS],
                    SLOT_SEEDS[class_index * REPLICAS_PER_CLASS
                               + REPLICAS_PER_CLASS - 1],
                ],
                "inelastic": bool(inelastic),
                "nominal_inelastic_fraction": (
                    0.25 if inelastic else None),
                "realized_inelastic_fraction": (
                    events / (N_EVENTS_PER_CLASS
                              * len(INELASTIC_CLASS_ORDER))
                    if inelastic else None),
                "exact_by_construction": True,
                "tolerance_note": (
                    "the class allocation is the pre-registered slot "
                    "map (one seed per (class, replica) pair, equal "
                    "inelastic sample mass 0.25 each by construction); "
                    "the realized fraction deviation is 0.0 exactly"),
            }
    per_class = {
        label: _aggregate_class(label, events)
        for label, events in sorted(class_records.items())
    }
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "statistics": {
            "tier": ("T1: 4 replicas x 25,000 events per class across "
                     "the five classes = 100,000 events per class, "
                     "500,000 events total (plan section 5.1 line 253)"),
            "events_total": sum(len(r["events"]) for r in slots),
            "slots": len(slots),
        },
        "seed_map_summary": {
            "master_seed": MASTER_SEED,
            "raw_slot_seeds": list(SLOT_SEEDS),
            "event_index_rule": (SMOKE_EVENT_INDEX_RULE if smoke
                                 else EVENT_INDEX_RULE),
        },
        "nominal_parameters": dict(_NOMINAL_STATUS),
        "class_draw_census": class_draw_census,
        "per_class": per_class,
        "per_slot": [
            {
                "slot": record.get("slot"),
                "file": name,
                "class_label": record.get("class_label"),
                "seed": record.get("seed"),
                "n_events": record.get("n_events"),
                "event_hash_set_sha256":
                    record.get("event_hash_set_sha256"),
                "conservation": record.get("conservation"),
            }
            for name, record in zip(record_files, slots)
        ],
        "conservation": conservation_report,
        "elastic_normalization_check": elastic_check,
        "elastic_excluded_from_inelastic": (
            "the elastic class is excluded from the inelastic statistics "
            "of R51, R52, R54, and R55 (plan line 824); its per-class "
            "block records the normalization reference's own aggregates"),
        "nd_multi_excluded": (
            "the nd_multi class is excluded from R50 (unretained, "
            "P(n = 2) = 0.0; it enters the round only through R53, "
            "plan line 926)"),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "named_residual_register": [],
        "named_residual_note": (
            "the named-residual register is empty on this run: zero "
            "conservation violations and a passing elastic "
            "normalization check; the standing model-uncertainty carry "
            "(the R32 spread and the b_transport_seagull residual) is "
            "recorded separately as a model-uncertainty term, not a "
            "pass criterion"),
        "no_separation_claim": (
            "this record claims no separation of exchange recoil from "
            "decay transverse dynamics (the R46 outcome, "
            "separation_claimed false)"),
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


def run_evaluate(run_dir: Path) -> dict:
    """--evaluate: the per-class evaluation over the 20 production slot
    records; writes template_report.json and REPORT.md, then stops."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report = run_evaluate_machinery(run_dir, smoke=False)
    _write_json(run_dir / REPORT_NAME, report)
    _write_report_md(run_dir, report)
    print(
        f"[r50] evaluation written: {run_dir / REPORT_NAME} "
        f"(digest {report['digest'][:16]}...)",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# REPORT.md (the run's human-readable record; sealed content)
# ---------------------------------------------------------------------------
def _fmt(value, fmt: str = "") -> str:
    if value is None:
        return "n/a"
    if fmt:
        return format(value, fmt)
    return str(value)


def _write_report_md(run_dir: Path, report: dict) -> None:
    """Render REPORT.md from the report dict (the census, the six stored
    quantities per class, the elastic normalization check, the
    conservation audit, and the run bookkeeping)."""
    lines = []
    lines.append("# R50 — pure topology templates (run report)")
    lines.append("")
    lines.append(f"Run: **{RUN_ID}** (`{RUN_NAME}`), card `{CARD}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("T1 template samples for every class at the fixed "
                 "nominal parameters (plan section 11 run R50, lines "
                 "898–907). The six stored quantities (plan line 901): "
                 "leading-baryon x_F, gap-size distribution, excitation "
                 "masses, multiplicity distribution, central and forward "
                 "species yields, seagull moments. No optimizer on this "
                 "run.")
    lines.append("")
    census = report["class_draw_census"]
    lines.append("## Sample census")
    lines.append("")
    lines.append("| class | events | replicas | seeds | inelastic |")
    lines.append("|---|---|---|---|---|")
    for label in CLASS_ORDER:
        row = census[label]
        seeds = row["seed_block"]
        inelastic_text = ("yes (0.25 nominal)" if row["inelastic"]
                          else "no (normalization reference, plan line 824)")
        lines.append(
            f"| {label} | {row['events']} | {row['replicas']} "
            f"| {seeds[0]}-{seeds[1]} | {inelastic_text} |")
    lines.append("")
    lines.append(f"Total: {report['statistics']['events_total']} events "
                 "in 20 slots (4 replicas x 25,000 per class); seeds "
                 "20300925–20300944; event-index partition "
                 "1_000_000_000 + i, i < 25,000.")
    lines.append("")
    lines.append("## Key report numbers (the six stored quantities, "
                 "pooled per class)")
    lines.append("")
    for label, block in report["per_class"].items():
        lines.append(f"### {label} ({block['events']} events)")
        lines.append("")
        xf = block["leading_baryon_xf"]
        lines.append(
            f"- Leading-baryon x_F (CM frame, x_F = 2 p_L^CM/sqrt(s)): "
            f"projectile mean {_fmt(xf['projectile']['mean'], '.6f')} "
            f"[{_fmt(xf['projectile']['min'], '.4f')}, "
            f"{_fmt(xf['projectile']['max'], '.4f')}], target mean "
            f"{_fmt(xf['target']['mean'], '.6f')} "
            f"[{_fmt(xf['target']['min'], '.4f')}, "
            f"{_fmt(xf['target']['max'], '.4f')}].")
        gap = block["gap_size"]
        lines.append(
            f"- Gap size: {gap['present']} events with a gap record, "
            f"{gap['absent_no_gap_field']} without (no gap field); mean "
            f"{_fmt(gap.get('mean'), '.6f')} GeV of rapidity, range "
            f"[{_fmt(gap.get('min'), '.4f')}, "
            f"{_fmt(gap.get('max'), '.4f')}].")
        masses = block["excitation_masses"]
        lines.append(
            f"- Excitation masses: {masses['count']} values, mean "
            f"{_fmt(masses.get('mean'), '.6f')} GeV, range "
            f"[{_fmt(masses.get('min'), '.4f')}, "
            f"{_fmt(masses.get('max'), '.4f')}].")
        mult = block["multiplicity"]
        lines.append(
            f"- Multiplicity: mean {_fmt(mult.get('mean'), '.4f')}, "
            f"max {_fmt(mult.get('max'))}, central-activity mean "
            f"{_fmt(mult.get('central_activity_mean'), '.4f')} "
            "(|y^CM| < 0.5).")
        y = block["central_forward_yields"]
        lines.append(
            f"- Central/forward species yields per event: central pions "
            f"{_fmt(y['central_pions_mean'], '.4f')}, central protons "
            f"{_fmt(y['central_protons_mean'], '.4f')}, forward pions "
            f"{_fmt(y['forward_pions_mean'], '.4f')}, forward protons "
            f"{_fmt(y['forward_protons_mean'], '.4f')}.")
        sg = block["seagull_moments"]
        lines.append(
            f"- Seagull moments (mean p_T of the final state): mean "
            f"{_fmt(sg.get('mean'), '.6f')} GeV/c, std "
            f"{_fmt(sg.get('std'), '.4f')}, range "
            f"[{_fmt(sg.get('min'), '.4f')}, "
            f"{_fmt(sg.get('max'), '.4f')}].")
        lines.append("")
    elastic = report["elastic_normalization_check"]
    lines.append("## Elastic normalization check (plan line 824)")
    lines.append("")
    e_checks = elastic["checks"]
    lines.append(
        f"Verdict: **{elastic['verdict']}**; "
        f"{_fmt(e_checks['normalization_identity']['max_dev'], '.3e')} "
        "normalization-identity max deviation (tolerance 1e-12), "
        f"{_fmt(e_checks['quantile_roundtrip']['max_dev'], '.3e')} "
        "quantile-round-trip max deviation (tolerance 1e-9), t sample "
        f"mean {_fmt(e_checks['t_sample_mean']['sample_mean_gev2'], '.6f')} "
        "GeV^2 against the analytic mean "
        f"{_fmt(e_checks['t_sample_mean']['analytic_mean_gev2'], '.6f')} "
        f"GeV^2 (z = {_fmt(e_checks['t_sample_mean']['z'], '.3f')}, "
        "tolerance 3 sigma), zero zero-transfer violations, x_F "
        "pair-sum max |x_F^p + x_F^t| = "
        f"{_fmt(e_checks['xf_pair_sum_max_abs'], '.3e')}.")
    lines.append("")
    cons = report["conservation"]
    lines.append("## Conservation audit (the full sample)")
    lines.append("")
    lines.append(
        f"Zero B, Q, S violations and zero four-momentum closure "
        f"failures at tolerance 1e-09 on all "
        f"{cons['events_checked']} events: worst closure residual "
        f"{_fmt(cons['worst_closure_relative'], '.3e')} relative, worst "
        f"energy closure {_fmt(cons['worst_energy_closure_gev'], '.3e')} "
        f"GeV, worst momentum closure "
        f"{_fmt(cons['worst_momentum_closure_gev'], '.3e')} GeV, worst "
        f"mass-shell deviation "
        f"{_fmt(cons['worst_mass_shell_relative'], '.3e')} relative. "
        f"Verdict: {'PASS' if cons['pass'] else 'FAIL'}.")
    lines.append("")
    lines.append("## Model-uncertainty carry (standing)")
    lines.append("")
    lines.append(
        "The R32 transition-window spread (bin spread p95 4.028426 at "
        "4,800,000 events; feed-down spread max 31.1773 sigma; 32 "
        "persisting residuals) and the b_transport_seagull "
        "beyond_isotropic_not_positive residual "
        "(-2.3959765824349295e-05) are recorded as model-uncertainty "
        "terms of the seagull moments, not pass criteria. The "
        "calculation-engine moment floor (~1e-3 GeV/c) is tracked on "
        "its separate track.")
    lines.append("")
    lines.append("## Bookkeeping")
    lines.append("")
    lines.append(
        "- `optimizer_invoked`: false (no parameter is fit on this run, "
        "index item 10).")
    lines.append(
        "- Determinism: the per-event hash set of every slot record "
        "carries the bit-level identity evidence; the clean-environment "
        "validation re-runs the evaluation and the probe slot (nd, "
        "replica 0) from a clean interpreter before the seal.")
    lines.append(
        "- The multiple-neutral class `nd_multi` is excluded from R50 "
        "(unretained; it enters the round only through R53, plan line "
        "926).")
    lines.append(
        "- The frozen simplified cascade at the sealed R45 point is "
        "consumed without refit (the R46 no_refit clause); the blend "
        "ramp start stays fixed at B = 2.680194110727528 GeV; no Round "
        "5 record claims a separation of exchange recoil from decay "
        "transverse dynamics.")
    lines.append("")
    lines.append(f"Report digest: `{report['digest']}`.")
    lines.append("")
    (run_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block; the verbatim template)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """--gate: write gate5_r50.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    conservation = {
        "violations": int(report["conservation"]["violations"]),
        "pass": bool(report["conservation"]["pass"]),
        "tolerance": EVENT_CLOSURE_TOL,
        "worst_energy_closure_gev":
            report["conservation"]["worst_energy_closure_gev"],
        "events_checked": int(report["conservation"]["events_checked"]),
    }
    elastic = report["elastic_normalization_check"]
    gate = {
        "schema": "na49_mixing.staged_r50_pure_topology_templates.gate/1",
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R50",
        "card": CARD,
        "clause": "per-class template evidence (training side)",
        "classes": list(CLASS_ORDER),
        "excluded_from_inelastic": ["elastic", "nd_multi"],
        "nominal_parameters": {
            "epsilon": NOMINAL_PARAMS.epsilon,
            "b_sd": NOMINAL_PARAMS.b_sd,
            "inelastic_fractions": dict(NOMINAL_PARAMS.class_fractions),
            "p_n2": NOMINAL_PARAMS.p_n2,
        },
        "events_checked": conservation["events_checked"],
        "per_class": {
            label: {
                "events": report["class_draw_census"][label]["events"],
                "replicas": REPLICAS_PER_CLASS,
                "seed_block": report["class_draw_census"][label][
                    "seed_block"],
                "draw_count": report["class_draw_census"][label][
                    "events"],
                "stored": list(STORED_QUANTITIES),
            }
            for label in CLASS_ORDER
        },
        "elastic_normalization_check": {
            "verdict": elastic["verdict"],
            "tolerance": elastic["tolerance_note"],
        },
        "conservation": conservation,
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "no_separation_claim": (
            "this record claims no separation of exchange recoil from "
            "decay transverse dynamics (the R46 outcome, "
            "separation_claimed false)"),
        "report_digest": report.get("digest"),
        "sealed": True,
        "sealed_note": (
            "the run seals after this record (the seal manifest and "
            "SHA256SUMS.txt cover this file; the run dir is immutable "
            "after the seal)"),
        "gate_note": (
            "evidence only; the written retain / revise / reject "
            "decision on the Gate 5 clauses is the round's Gate 5 "
            "follow-up obligation"),
    }
    _write_json(run_dir / GATE_RECORD_NAME, gate)
    print(f"[r50] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


# ---------------------------------------------------------------------------
# The clean-environment validation (exactly one pass, before the seal)
# ---------------------------------------------------------------------------
def run_clean_validate(run_dir: Path) -> dict:
    """--clean-validate: re-run the evaluation from a clean interpreter
    over the generated artifacts with a matching report digest, re-run
    the pre-registered probe slot in memory against its sealed record,
    write clean_environment_commands.json, then stop.  Exactly one pass;
    executed before the seal (a second invocation refuses)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    clean_path = run_dir / CLEAN_ENV_NAME
    if clean_path.is_file():
        _refuse(
            f"the clean-environment validation already ran: {clean_path}; "
            "exactly one pass executes before the seal (the card-19 "
            "seal-ordering discipline)")
    for name in (REPORT_NAME, GATE_RECORD_NAME):
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    sealed_report = _read_json(run_dir / REPORT_NAME)
    rerun_report = run_evaluate_machinery(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the probe slot re-run in memory (the determinism evidence of
    # acceptance (vi)): the pre-registered probe slot (nd, replica 0)
    # regenerates from its pre-registered seed and must reproduce its
    # sealed event-hash set, its conservation aggregate, and its class
    # census exactly
    probe = PROBE_SLOT
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set = generate_slot_events(
        slot_class(probe), slot_seed(probe), EVENTS_PER_SLOT,
        EVENT_INDEX_BASE, catalog, fams)
    sealed_probe = _read_json(run_dir / slot_name(probe))
    probe_matches = bool(
        hash_set == sealed_probe.get("event_hash_set_sha256")
        and census == sealed_probe.get("class_census")
        and conservation == sealed_probe.get("conservation")
    )
    payload = {
        "schema": ("na49_mixing.staged_r50_pure_topology_templates."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r50_pure_topology_templates.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            (f"python helpers/benchmarking/"
             f"staged_r50_pure_topology_templates.py "
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
        "probe_slot": {
            "slot": probe,
            "file": slot_name(probe),
            "class_label": slot_class(probe),
            "seed": slot_seed(probe),
            "event_hash_set_match": bool(
                hash_set == sealed_probe.get("event_hash_set_sha256")),
            "census_matches": bool(census == sealed_probe.get(
                "class_census")),
            "conservation_matches": bool(
                conservation == sealed_probe.get("conservation")),
            "matches_sealed_record": probe_matches,
        },
        "verified_utc": _now_utc(),
    }
    _write_json(clean_path, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter evaluation does not reproduce the "
            "report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe slot does not match its sealed record")
    print(
        f"[r50] clean validation written: {clean_path} (report hash "
        f"match: {report_matches}, probe slot {probe} match: "
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
        "schema": ("na49_mixing.staged_r50_pure_topology_templates."
                   "seal/1"),
        "run": RUN_ID,
        "card": CARD,
        "run_dir": str(run_dir),
        "files": files,
        "sha256sums_entries": n_entries,
        "record_count": sum(
            1 for name in files if name.startswith("slot_")
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
        PREREGISTRATION_NAME, PROVENANCE_NAME, REPORT_NAME,
        REPORT_MD_NAME, GATE_RECORD_NAME, CLEAN_ENV_NAME,
    )
    for name in required:
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    missing = [
        slot_name(k) for k in range(N_SLOTS)
        if not (run_dir / slot_name(k)).is_file()
    ]
    if missing:
        _refuse(f"the production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r50] sealed {run_dir} ({n_entries} entries)", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R50 driver: the pure topology templates "
                    "(TASK-20260925-r5-run-r50-pure-topology-templates)")
    parser.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop")
    parser.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (never sealed)")
    parser.add_argument(
        "--generate", action="store_true",
        help="generate the production slot records, then stop")
    parser.add_argument(
        "--slot", type=int, action="append", default=None,
        help="generate only the given slot index (repeatable)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="the per-class evaluation (template_report.json and "
             "REPORT.md), then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r50.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the evaluation from a clean interpreter and the "
             "probe slot, then stop (exactly one pass, before the seal)")
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
        print(f"[r50] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, slots=args.slot)
        print(f"[r50] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        run_evaluate(run_dir)
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
