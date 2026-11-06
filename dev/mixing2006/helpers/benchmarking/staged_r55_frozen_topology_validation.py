#!/usr/bin/env python3
"""R55 driver: the frozen topology validation (Round 5, card 15).

Task ``TASK-20260925-r5-run-r55-frozen-topology-validation`` (plan
section 11 run R55, lines 932-934; plan section 23 lines 1825-1827,
"Round 5 -- exchange topology").  The card executes run R55, the frozen
topology validation: a T3 run with the topology fractions frozen at the
round's frozen point (the R51 fit values, updated by the R52 outcome
and, if run, the R53 outcome), validating the leading baryons, gaps,
multiplicities, and nonstrange spectra, sealed evidence under
``dev/00runs/r55_frozen_topology_validation_<date>/``.

The round's frozen point (index item 7)
---------------------------------------
The frozen topology fractions are the R51 fit values, updated by the
R52 outcome and, if run, the R53 outcome:

- R52 (the sealed ``gate5_r52.json``): the zero model (the ND fraction
  zero, the remaining three inelastic fractions at the R52 renormalized
  values) is the challenger; the card's update rule keeps the ND
  fraction at the R51 fit value exactly when the zero model is retained
  by no held-out family, and moves to the renormalized zero-model
  fractions exactly when the zero model is retained by every family; a
  mixed per-family verdict is outside the card's two-case rule and
  refuses the run.  At the sealed R52 outcome the zero model is
  retained by no family (every family's recorded retention verdict is
  "not retained"; 157 zero-model degradation named residuals), so the
  one-excitation topology is retained and the ND fraction stays at the
  R51 fit value.
- R53 (the sealed ``gate5_r53.json``): ran and recorded ``retained:
  false`` (status TRIGGERED; the fitted P(n >= 2) sits on the 0.5
  boundary).  The nd_multi class therefore stays unretained: P(n = 2) =
  0.0 (the card-8 unretained default) and nd_multi events are absent
  from the frozen sample; the drawn ``nd`` labels run card 7's ND
  generation directly.
- epsilon and B_SD are frozen at the R51 fit values; the blend ramp
  start stays fixed at B = 2.680194110727528 GeV (Round 4 obligation 4;
  never a free parameter); the elastic fraction is the exact complement
  0.0 of the frozen inelastic mass.

Statistics (T3, plan section 5.1 line 255)
------------------------------------------
20 replicas x 250,000 p+p events = 5,000,000 events.  Master seed
20260925; raw replica seeds 20300989-20301008 (k = 0..19); per-replica
event index ``1_500_000_000 + i`` with i < 250,000, the
derive_substreams pattern on the closed decay_channel/decay_kinematics
labels (and the closed five-label quintuple).  The block and partition
are disjoint from every previously used raw block and partition (index
item 9); a colliding raw (seed, event-index) substream stops the run.

Validation observables (index item 7; four families)
----------------------------------------------------
- the leading baryon x_F spectra of p and p-bar (x_F = 2 p_L^CM /
  sqrt(s), Feynman x, plan section 7.1; the count-based 1D x_F
  histograms of the final-state protons and antiprotons on the R50
  pinned binning; the data side is the sealed R50 nominal-point
  reference reweighted to the frozen fractions -- the recorded R51
  convention, the sealed substrate carrying no published 1D x_F table);
- the gap-size distribution (the event gap size of the
  gap-carrying classes sd_projectile, sd_target, dd, on the R50 pinned
  binning; the data side is the sealed R50 reference reweighted to the
  frozen fractions and renormalized over the gap-carrying classes);
- the charged and species-separated multiplicity distributions (the
  charged multiplicity n_ch over the support 2..20, the R51 informative
  subset 2..14; the species-separated per-event counts of the five
  nonstrange species over the support 0..20; the data side is the
  sealed R50 reference reweighted to the frozen fractions);
- the nonstrange spectra of pi+, pi-, pi0, p, p-bar against the sealed
  R01 published tables, recorded with the dense-accumulation/then-pick
  convention (inherited state of the index card): the finals accumulate
  into the species' dense pT-major lattices with the invariant measure
  weight e_CM / p_T (the R01 accumulation measure; the stored lab
  four-momenta inverted through the pinned boost at the extraction
  energy SQRT_S = 17.270 GeV), the published rows are picked from the
  dense totals at the pick map ``bin_index_pt * n_x + bin_index_xf``,
  hole cells stay unpicked and are dropped.  pi+, pi-, p, and p-bar
  carry the published data deviance; pi0 carries no published table
  (the recorded R52 convention) and is accumulated model-side on the
  pi- published geometry at the pi0 mass 0.1349766 GeV with no data
  deviance.

Evaluation (plan section 5.4; no optimizer, index item 10)
----------------------------------------------------------
Per-bin deviance on the pre-registered observables with the
sigma^2 = v_ref/v_data + se^2 form (the published/reference variance
plus the between-replica MC variance, the std ddof=1 over the 20
replica blocks divided by sqrt(20), floored at the engine's
DETERMINISTIC_SE_FLOOR); data and MC contributions reported separately;
the named-residual register assembled on any bound exceedance (the 2.0
pooled-SE bound of the per-bin difference for a named finding; the
|residual| > 3.0 sigma entries carry the house named-residual form,
SIGMA_TOLERANCE = 3.0); the five-percent sector-degradation bound of
the acceptance (iv) evaluated per paper-window sector against the
sealed R51 training baseline (the pre-registered reference; the
recorded caveat: the R51 training baseline carries the profiled
fit-time normalization nuisances, the frozen evaluation runs
un-profiled; a violating sector is a named finding with the recorded
caveat as its stated reason).  The sealed R54 ledger audit must pass on
the frozen topology (zero violations on card 2's five ledger rules)
before the validation is interpreted; the per-event conservation check
runs on all 5,000,000 events of this run.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the M8 frozen
  parameter map, the sealed R50/R51/R52 seals and the sealed R53/R54
  seals, the frozen fractions against the sealed R51
  ``fit_report.json``, the R52-outcome retention semantics against the
  sealed ``gate5_r52.json``, the R53 outcome against the sealed
  ``gate5_r53.json``, the R54 audit precondition (the sealed
  ``gate5_r54.json`` passing on the frozen topology), the R50-pinned
  reference recomputation (cross-checked against the sealed R51 fit
  report's reference data bit-exactly), and the seed-block
  disjointness before anything is written.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one mixed-draw replica x 2,000
  events at the smoke seed 20312925 (raw offset +12000 from master
  20260925) on the event-index partition 1_600_000_000 + i (i < 2,000)
  at the frozen fractions (the recorded sampling surface at the R53
  unretained default), plus the per-class pinned dispatch exercise
  (every topology class, including elastic and the nd_multi class at
  the pinned smoke exercise P(n = 2), through the same accumulation
  machinery on the dedicated smoke class-exercise partition), and the
  smoke validation evaluation on the synthetic hermetic grid, into an
  explicit ``--run-dir`` (the default production run dir is refused).
  Two identical smoke invocations produce byte-identical artifacts.  No
  sealed archive and no SPS table is read (the smoke asserts the
  frozen-value constants against no sealed archive; the production
  assertion runs in --pre-registration and --evaluate).  ``--gate`` /
  ``--seal`` / ``--clean-validate`` refuse smoke artifacts (the R31/8c
  convention).
- ``--generate [--replica R]``: generate the requested production
  replica records (any subset; the default is every replica), one
  record file per replica, then stop.  Requires the pre-registration
  and re-asserts the frozen fractions, the R52/R53 outcomes, and the
  R54 audit precondition before generating.  A completed record that
  matches the pre-registered identity is reused unchanged, so the
  declared tier can be completed replica by replica (the pre-registered
  gateway fallback runs one job per replica, ``--tasks 1``, single math
  thread).
- ``--evaluate``: the validation evaluation over the generated
  artifacts (the pooled accumulations, the four observable families
  with their per-bin residual tables, the draw-count check, the
  conservation aggregate, the sector-degradation table),
  ``validation_report.json``, then stop.  Requires all 20 production
  records.
- ``--gate``: write ``gate5_r55.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the
  validation report.
- ``--clean-validate``: re-run the validation evaluation from a clean
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
catalog and continuum config, the sealed R01 archive (the published
data side), the SPS tables (the published binning), the sealed R50
archive (the nominal-parameter reference and the pinned reference
events), the sealed R51 archive (the frozen fractions, epsilon, B_SD),
the sealed R52 archive (the retention outcome), the sealed R53 archive
(the nd_multi outcome), and the sealed R54 archive (the ledger-audit
precondition).  The frozen simplified cascade at the sealed R45 point
(kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal
width 0.3 GeV) is consumed without refit (the R46 ``no_refit`` clause);
the exchange slopes act on the exchange kinematics only, never on any
daughter-level transverse dynamics; no Round 5 record claims a
separation of exchange recoil from decay transverse dynamics (the R46
outcome, separation_claimed false).

The sibling R51 driver module is imported and its
record/conservation machinery consumed verbatim (``event_conservation``,
``_line_qn``, ``_cm_boost_parameters``, ``verify_sealed_dir``,
``_load_catalog``, ``_load_families``, the ``ConservationViolation``
contract), so the conservation contract and the sealed-archive
discipline are the R51 machinery bit for bit.
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
        f"staged_r55_frozen_topology_validation: {message}",
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
# (_line_qn), the CM boost parameters (_cm_boost_parameters), the
# sealed-archive discipline (verify_sealed_dir), and the configuration
# loaders (_load_catalog, _load_families).  Consuming the module
# verbatim keeps the conservation contract and the seal discipline the
# R51 machinery bit for bit (the R54 consumption pattern).
import staged_r51_topology_fraction_fit as r51  # noqa: E402

from helpers.benchmarking import sps_core  # noqa: E402
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402

from na49_mixing.exchange_ledger import (  # noqa: E402
    ALLOWED_EXCHANGES,
    VACUUM,
    ZERO_TRANSFER,
)
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.rng_substreams import (  # noqa: E402
    SUBSTREAM_LABELS,
    derive_substreams,
)
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402
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
# Pre-registered run identity (card TASK-20260925-r5-run-r55-frozen-
# topology-validation; the collision, statistics, seeds, sampling, and
# assertion blocks are verbatim from the card and the round 5 index
# items 7, 8, 9, and 10).
# ---------------------------------------------------------------------------
RUN_ID = "R55"
CARD = "TASK-20260925-r5-run-r55-frozen-topology-validation"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r55_frozen_topology_validation_2026-09-25 re-pins to the
#: execution date (the R50/R51/R52/R53 sealed archives moved the same
#: way, to 2026-09-27, and the R54 archive to 2026-09-28); this
#: driver's execution pins 2026-09-28.
RUN_NAME = "r55_frozen_topology_validation_2026-09-28"

DEFAULT_RUN_DIR = "dev/00runs/r55_frozen_topology_validation_2026-09-28"
GATE_RECORD_NAME = "gate5_r55.json"
REPORT_NAME = "validation_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_EXERCISE_NAME = "smoke_class_exercise.json"
SMOKE_REPORT_NAME = "smoke_validation_report.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r55_frozen_topology_validation.py")
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r55_frozen_topology_validation.py")

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9, cards 10-14 (the sealed R50/R51/R52/R53
#: runs and the R54 audit), and this card.
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
    "TASK-20260925-r5-run-r55-frozen-topology-validation",
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests,
#: including the R45/R46/R50/R51/R52/R53/R54 driver pins; the R53 and
#: R54 drivers landed before this driver's authoring, so their file
#: digests are pinned here).
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
    # NOTE: the R53 and R54 driver files are deliberately NOT pinned by
    # digest here: this card's provenance list records the sealed R51,
    # R52, R54, and R53 RUN-DIR SEAL STATES, not the sibling drivers'
    # bytes, and the R53/R54 driver files were still untracked at this
    # driver's authoring (their landing commits are the orchestrator's
    # follow-up).  The R53/R54 consumption in this driver rides on the
    # sealed archives' seals (the gate files plus their SHA256SUMS.txt
    # verification), which is the integrity surface the card names.
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
            "model-uncertainty term of the validation (recorded, not "
            "folded into the frozen comparison's arithmetic)"
        ),
    },
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_seagull_note": (
        "the b_transport_seagull beyond_isotropic_not_positive residual "
        "is a carry, not a pass criterion (index item 7)"
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
#: T3 statistics (plan section 5.1 line 255): exactly 5,000,000 events
#: = 20 replicas x 250,000 p+p events (the card's pre-registered
#: instantiation).
N_REPLICAS = 20
EVENTS_PER_REPLICA = 250_000
N_EVENTS_TOTAL = N_REPLICAS * EVENTS_PER_REPLICA   # 5,000,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
SEED_SLOT_OFFSET = 64
#: The raw replica seeds 20300989-20301008 (k = 0..19; index item 9;
#: the block is disjoint from every previously used raw block and from
#: every other reserved Round 5 block).
REPLICA_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + SEED_OFFSET + SEED_SLOT_OFFSET + k
    for k in range(N_REPLICAS))
EVENT_INDEX_BASE = 1_500_000_000
EVENT_INDEX_RULE = "1_500_000_000 + i, i < 250,000 (per replica)"

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
#: R50-R54 blocks are used and this run's block is the r55 entry).
RESERVED_ROUND5_BLOCKS = {
    "r50": [20300925, 20300944],
    "r51": [20300945, 20300952],
    "r52": [20300953, 20300960],
    "r53": [20300961, 20300968],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9, plus the
#: round-5 smoke partitions).
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
                   "r43_cascade_ablation_2026-09-22/"
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
    "r54_partition": {
        "master_seed": 20260925,
        "raw_seeds": [20300969, 20300988],
        "event_index_range": [1_400_000_000, 1_400_050_000],
        "event_index_partition": "1_400_000_000 + i, i < 50,000",
        "source": ("sealed dev/00runs/"
                   "r54_local_ledger_audit_2026-09-28/"
                   "pre_registration.json"),
    },
}

#: The clean-environment validation's pre-registered probe replica (the
#: replica 0 mixed draw; the probe re-run is the determinism evidence of
#: acceptance (ix)).
PROBE_REPLICA = 0

# ---------------------------------------------------------------------------
# The topology sampling (index items 7 and 8; the recorded source)
# ---------------------------------------------------------------------------
R50_RUN_DIR = "dev/00runs/r50_pure_topology_templates_2026-09-27"
R51_RUN_DIR = "dev/00runs/r51_topology_fraction_fit_2026-09-27"
R52_RUN_DIR = "dev/00runs/r52_central_excitation_necessity_2026-09-27"
R53_RUN_DIR = "dev/00runs/r53_multiple_neutral_excitation_2026-09-27"
R54_RUN_DIR = "dev/00runs/r54_local_ledger_audit_2026-09-28"
R51_FIT_REPORT_NAME = "fit_report.json"
R52_GATE_NAME = "gate5_r52.json"
R53_GATE_NAME = "gate5_r53.json"
R54_GATE_NAME = "gate5_r54.json"
R50_PREREGISTRATION_NAME = "pre_registration.json"
R50_TEMPLATE_REPORT_NAME = "template_report.json"
R50_SLOT_FMT = "slot_{}_{}.json"
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"

#: The frozen R51 fitted point (asserted byte-exactly against the sealed
#: R51 fit_report.json before any generation; identical to the R52/R53/
#: R54 drivers' FROZEN_FIT pin).
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
#: generation surface (dispatchable, exercised by the smoke class
#: exercise, gate-recorded); at this recorded fraction no elastic event
#: realizes in production and the draw-count check requires the
#: realized elastic count to be exactly zero.
ELASTIC_FRACTION_GENERATION = 1.0 - math.fsum(R51_FROZEN_FRACTIONS.values())

#: The unretained default P(n = 2) (the R53-PENDING default; the card-8
#: DEFAULT_P2, asserted equal in the surface check).
R53_P_N2_DEFAULT = 0.0

#: The per-class draw-count MC tolerance (the recorded MC tolerance of
#: the draw-count check: the realized count within 5.0 binomial standard
#: deviations of the expected count n * f; a class with expected count
#: exactly 0 must realize exactly 0; a divergence is a named deviation).
MC_TOLERANCE_SIGMA = 5.0

#: The event four-momentum closure tolerance (the house conservation
#: contract).
EVENT_CLOSURE_TOL = 1e-9

#: The named-residual bounds (the card's evaluation block, never
#: relaxed): a bin difference beyond 2.0 pooled standard deviations is a
#: named finding; the |residual| > 3.0 sigma entries carry the house
#: named-residual form.
NAMED_FINDING_SE_FACTOR = 2.0
SIGMA_TOLERANCE = 3.0

#: The five-percent sector-degradation bound (the acceptance (iv);
#: never relaxed).
SECTOR_DEGRADATION_FRACTION = 0.05

#: The class order of the inelastic walk (the runner's fixed
#: INELASTIC_ORDER) and the smoke exercise's fixed class order (the
#: registry's naming order with the elastic class first).
CLASS_ORDER: tuple[str, ...] = tuple(INELASTIC_ORDER)
SMOKE_EXERCISE_CLASSES: tuple[str, ...] = (
    "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi",
)

#: The gap-carrying classes (the event gap size is recorded on the
#: single-diffraction and double-diffraction classes; the elastic and
#: nd classes carry no gap field -- the R50 stored-quantity convention).
GAP_CARRYING_CLASSES: tuple[str, ...] = (
    "sd_projectile", "sd_target", "dd",
)

#: The R51 named-residual kinds (the inherited model-uncertainty caveat
#: block of the frozen-point caveat).
R51_NAMED_RESIDUAL_KINDS = (
    "optimizer_non_convergence",
    "replica_block_minima_disagree",
    "block_curvature_failure",
    "weight_degeneracy",
)

INCOMING_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

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


class ValidationRefusal(RuntimeError):
    """A pre-registered validation bound fired (checked, never repaired;
    the finding is recorded and the step refuses)."""


ConservationViolation = r51.ConservationViolation


def replica_seed(replica: int) -> int:
    """The raw seed of replica *replica* (the pre-registered R55
    block)."""
    k = int(replica)
    if not 0 <= k < N_REPLICAS:
        _refuse(f"replica index {k} outside the R55 block "
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
# instantiation of the card's elastic-inclusion and R53-outcome clauses)
# ---------------------------------------------------------------------------
ELASTIC_INCLUSION_NOTE = (
    "the elastic class is included in the generation surface (its "
    "events are exercisable through the card-9 run_topology and the "
    "smoke class exercise pins the dispatch); the R50 template report "
    "records the elastic class as excluded from the inelastic "
    "statistics of R51, R52, R54, and R55 (plan line 824).  The elastic "
    "fraction is recorded from the same source as the inelastic "
    "fractions: it is the complement of the R51 frozen inelastic mass "
    "(the runner's pinned_trace default rule, computed from the same "
    "sealed source), which at the sealed R51 point is exactly 0.0 "
    "because the four fitted inelastic fractions sum to one exactly in "
    "float arithmetic (the same value the sealed R51 sample's elastic "
    "convention records).  At the recorded fractions no elastic event "
    "realizes in production and the draw-count check requires the "
    "realized elastic count to be exactly zero; a positive elastic "
    "fraction would renormalize the asserted R51 values and is a named "
    "deviation the card does not sanction.")

REAUDIT_OBLIGATION_NOTE = (
    "the re-audit obligation: if the R53 outcome changed after a "
    "pre-registration recorded a pending or different value, the "
    "pre-registration is invalidated -- the run re-registers under a "
    "new date with the sealed outcome and re-executes (the run "
    "identity re-pins; the generation refuses the flip against a "
    "standing pre-registration).  A sealed retained: false confirms the "
    "recorded unretained default 0.0 and the run stands.")


# ---------------------------------------------------------------------------
# The validation-surface invariants (module level; in-memory only)
# ---------------------------------------------------------------------------
def assert_validation_surface() -> dict:
    """The validation-surface invariants (module level): the closed
    allowed-exchange set is exactly the vacuum exchange; the six-entry
    class registry with the elastic class non-inelastic and the
    nd_multi class unretained in code; the R51 frozen fractions
    non-negative and summing to one exactly in float arithmetic; the
    elastic fraction the exact complement 0.0; the R53-PENDING default
    equal to the card-8 unretained default; the nominal-parameter
    reference intact; the validation-observable register's pinned
    binning supports sane; the named-residual bounds the card's
    literals."""
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
    if FROZEN_FIT["b_sd"] != 1.8457918026393214 or \
            FROZEN_FIT["epsilon"] != 0.0033443027832747944:
        _refuse("the frozen mass-law point drifted off the sealed R51 "
                "fitted values")
    if SQRT_S != SQRT_S_EXTRACTION_GEV:
        _refuse("the card-1 module constant SQRT_S drifted off the "
                f"pinned extraction energy {SQRT_S_EXTRACTION_GEV!r}")
    if BEAM_MOMENTUM != BEAM_MOMENTUM_GEV_C:
        _refuse("the card-1 module constant BEAM_MOMENTUM drifted off "
                f"the pinned beam momentum {BEAM_MOMENTUM_GEV_C!r}")
    if ELASTIC_TRANSFER_SLOPE != 8.504140762083612:
        _refuse("the card-1 module constant ELASTIC_TRANSFER_SLOPE "
                "drifted off the M8-frozen effective_transfer_slope "
                "8.504140762083612 (the elastic class's normalization "
                "reference; the frozen draw's B_SD is the R51-fitted "
                f"mass-law slope {FROZEN_FIT['b_sd']!r}, a different "
                "parameter)")
    if NAMED_FINDING_SE_FACTOR != 2.0 or SIGMA_TOLERANCE != 3.0:
        _refuse("the named-residual bounds drifted off the card's "
                "pre-registered literals (2.0 pooled SE; 3.0 sigma)")
    if SECTOR_DEGRADATION_FRACTION != 0.05:
        _refuse("the sector-degradation bound drifted off the 5% "
                "pre-registered literal")
    # the pinned reference binnings (the R50 stored-quantity convention)
    if len(GAP_HISTOGRAM_EDGES) != 51 or \
            any(b <= a for a, b in zip(GAP_HISTOGRAM_EDGES,
                                       GAP_HISTOGRAM_EDGES[1:])):
        _refuse("the pinned gap-size histogram edges drifted")
    if len(LEAD_XF_HISTOGRAM_EDGES) != 43 or \
            any(b <= a for a, b in zip(LEAD_XF_HISTOGRAM_EDGES,
                                       LEAD_XF_HISTOGRAM_EDGES[1:])):
        _refuse("the pinned leading-baryon x_F histogram edges drifted")
    if NCH_SUPPORT_LO != 2 or NCH_SUPPORT_HI != 20:
        _refuse("the charged-multiplicity support drifted off the "
                "pre-registered 2..20")
    if SPECIES_MULT_SUPPORT_HI != 20:
        _refuse("the species-multiplicity support drifted off the "
                "pre-registered 0..20")
    if set(SPECIES_MULT_ORDER) != {
            "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton"}:
        _refuse("the species-separated multiplicity register drifted "
                f"off the five nonstrange species: {SPECIES_MULT_ORDER!r}")
    if tuple(SPECTRUM_SPECIES_ORDER) != (
            "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton"):
        _refuse("the nonstrange-spectra register drifted off the card's "
                f"pre-registered species order: "
                f"{SPECTRUM_SPECIES_ORDER!r}")
    if set(DATA_SPECIES) != {"pi_plus", "pi_minus", "proton", "antiproton"}:
        _refuse("the data-backed species register drifted off the four "
                f"R01-published nonstrange species: {DATA_SPECIES!r}")
    if MODEL_ONLY_SPECIES != ("pi_zero",):
        _refuse("the model-only species register drifted off pi_zero")
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
        "mc_tolerance_sigma": MC_TOLERANCE_SIGMA,
        "named_finding_se_factor": NAMED_FINDING_SE_FACTOR,
        "sigma_tolerance": SIGMA_TOLERANCE,
        "sector_degradation_fraction": SECTOR_DEGRADATION_FRACTION,
        "validation_register": {
            "leading_baryon_xf": ["proton", "antiproton"],
            "gap_size_distribution": True,
            "multiplicity": "charged and species-separated",
            "nonstrange_spectra": list(SPECTRUM_SPECIES_ORDER),
        },
        "pass": True,
    }


# ---------------------------------------------------------------------------
# The pinned reference binnings (the R50 stored-quantity convention)
# ---------------------------------------------------------------------------
#: The gap-size histogram binning of the sealed R50 stored quantity
#: (the ``gap_size.histogram.edges`` of the sealed R50 template report,
#: 50 bins of width 0.3456 over [0, 17.28]; verified against the sealed
#: report at pre-registration).
GAP_HISTOGRAM_EDGES: tuple[float, ...] = (
    0.0, 0.3456, 0.6912, 1.0368000000000002, 1.3824, 1.7280000000000002,
    2.0736000000000003, 2.4192, 2.7648, 3.1104000000000003,
    3.4560000000000004, 3.8016, 4.147200000000001, 4.4928, 4.8384,
    5.184000000000001, 5.5296, 5.8751999999999995, 6.2208000000000006,
    6.566400000000001, 6.912000000000001, 7.2576, 7.6032,
    7.948800000000001, 8.294400000000001, 8.64, 8.9856, 9.3312, 9.6768,
    10.0224, 10.368000000000002, 10.713600000000001, 11.0592, 11.4048,
    11.750399999999999, 12.096000000000002, 12.441600000000001, 12.7872,
    13.132800000000001, 13.4784, 13.824000000000002, 14.1696, 14.5152,
    14.860800000000001, 15.2064, 15.552, 15.897600000000002, 16.2432,
    16.588800000000003, 16.9344, 17.28,
)
#: The leading-baryon x_F histogram binning of the sealed R50 stored
#: quantity (the ``leading_baryon_xf.*.histogram.edges`` of the sealed
#: R50 template report, 42 bins of width 0.05 over [-1.05, 1.05];
#: verified against the sealed report at pre-registration).
LEAD_XF_HISTOGRAM_EDGES: tuple[float, ...] = (
    -1.05, -1.0, -0.9500000000000001, -0.9, -0.8500000000000001, -0.8,
    -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.44999999999999996, -0.4,
    -0.35, -0.30000000000000004, -0.25, -0.19999999999999996,
    -0.1499999999999999, -0.10000000000000009, -0.050000000000000044,
    0.0, 0.050000000000000044, 0.10000000000000009, 0.15000000000000013,
    0.19999999999999996, 0.25, 0.30000000000000004, 0.3500000000000001,
    0.40000000000000013, 0.44999999999999996, 0.5000000000000002, 0.55,
    0.5999999999999999, 0.6500000000000001, 0.7, 0.7500000000000002,
    0.8, 0.8499999999999999, 0.9000000000000001, 0.95,
    1.0000000000000002, 1.05,
)
#: The charged-multiplicity support: n_ch in 2..20 (the R51 training
#: statistic's informative subset is 2..14; the support is extended to
#: 20 so a tail event refuses nothing silently -- bins above the
#: reference's structural upper edge carry zero-reference entries that
#: are recorded, not bled).
NCH_SUPPORT_LO = 2
NCH_SUPPORT_HI = 20
NCH_BINS: tuple[int, ...] = tuple(range(NCH_SUPPORT_LO, NCH_SUPPORT_HI + 1))
#: The species-separated multiplicity support: the per-event count of
#: each species in 0..20 (the sealed R50 inelastic pools' realized
#: per-species maxima are 6 (pi_plus), 5 (pi_minus), 10 (pi_zero),
#: 2 (proton), 0 (antiproton); the support carries margin; an
#: out-of-support count refuses the run).
SPECIES_MULT_SUPPORT_HI = 20
#: The species-separated multiplicity register (the card's five
#: nonstrange species).
SPECIES_MULT_ORDER: tuple[str, ...] = (
    "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton",
)

#: The nonstrange-spectra register (the card's species order) and its
#: data-backed / model-only split (the R01 published tables carry
#: pi_plus, pi_minus, proton, antiproton; pi0 has no published table --
#: the recorded R52 convention -- and is accumulated model-side on the
#: pi- published geometry).
SPECTRUM_SPECIES_ORDER: tuple[str, ...] = (
    "pi_plus", "pi_minus", "pi_zero", "proton", "antiproton",
)
DATA_SPECIES: tuple[str, ...] = ("pi_plus", "pi_minus", "proton",
                                 "antiproton")
MODEL_ONLY_SPECIES: tuple[str, ...] = ("pi_zero",)

#: The species' catalog identities (the paper-window pdgs; the R44
#: species register restricted to the pre-registered nonstrange set).
SPECTRUM_PDGS: dict[str, int] = {
    "pi_plus": 211, "pi_minus": -211, "pi_zero": 111,
    "proton": 2212, "antiproton": -2212,
}
PDG_TO_SPECIES: dict[int, str] = {v: k for k, v in SPECTRUM_PDGS.items()}

PI_ZERO_MASS_GEV = 0.1349766
PI_ZERO_TABLE_PATH = "model://pi_zero_held_out_diagnostic"
PI_ZERO_GEOMETRY_NOTE = (
    "the sealed R01 archive and the committed SPS data carry no "
    "published pi0 table (the R01 tables are pi+, pi-, K+, K-, proton, "
    "antiproton), so the pi0 spectrum is accumulated model-side on the "
    "pi- published geometry (the row window at the pi0 mass "
    "0.1349766 GeV; the recorded R52 pi0 convention extended to the "
    "published lattice) and carries no data deviance; the "
    "deviance-based validation covers the four published nonstrange "
    "species")

R51_TRAINING_NOTE = (
    "the R51 training observables (the pre-registered disjointness "
    "reference): the leading-proton spectrum on the proton published "
    "cells, the B_transport combination on the aligned cells, the "
    "central-window gap fraction, and the charged-multiplicity "
    "distribution over n_ch = 2..14; the R55 validation register's "
    "families are the topology-sensitive extensions (the 1D "
    "leading-baryon x_F spectra, the gap-size distribution, the "
    "species-separated multiplicities) plus the nonstrange published-"
    "cell spectra; the charged-multiplicity family shares the R51 "
    "training statistic's bins 2..14 and its R50-pinned reference "
    "(the recorded overlap; the R51 fit consumed the same reference as "
    "its training data side)")

#: The fit-side species labels of the sealed R01 tables (the loader's
#: species keys) and the validation register's mapping.
R01_SPECIES_LABELS: dict[str, str] = {
    "pi_plus": "pi_plus", "pi_minus": "pi_minus", "proton": "proton",
    "antiproton": "antiproton",
}


_FIT_SURFACE_STATUS = assert_validation_surface()


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
    card-pinned surfaces and the R45-R52 driver pins)."""
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
    """The sealed-input verification: the R01 archive (the published
    data side), the R50 archive (the nominal reference events), the R51
    archive (the frozen point), the R52 archive (the retention
    outcome), the R53 archive (the nd_multi outcome), and the R54
    archive (the ledger-audit precondition).  Every archive is sealed
    on this run's substrate; a missing or unsealed one refuses (the
    card's dependency list is closed)."""
    out = {
        "r01": verify_sealed_dir(R01_RUN_DIR),
        "r50": verify_sealed_dir(R50_RUN_DIR),
        "r51": verify_sealed_dir(R51_RUN_DIR),
        "r52": verify_sealed_dir(R52_RUN_DIR),
        "r53": verify_sealed_dir(R53_RUN_DIR),
        "r54": verify_sealed_dir(R54_RUN_DIR),
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
    r55_lo = EVENT_INDEX_BASE
    r55_hi = EVENT_INDEX_BASE + EVENTS_PER_REPLICA
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
            if r55_lo < p_hi and p_lo < r55_hi:
                problems.append(
                    f"the R55 event-index partition [{r55_lo}, {r55_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if name == "r55":
            continue
        if sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS)):
            problems.append(
                f"the R55 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if name == "r55":
            continue
        if r55_lo < hi and lo < r55_hi:
            problems.append(
                f"the R55 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r55] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r55_raw_seed_block": [min(REPLICA_SEEDS), max(REPLICA_SEEDS)],
        "r55_raw_seed_rule": (
            "raw replica seed = 20260925 + 40000 + 64 + k = 20300989 + k "
            f"per replica k < {N_REPLICAS} (index item 9); the block "
            "20300989-20301008 is disjoint from every previously used raw "
            "block and from every other reserved Round 5 block"),
        "r55_replica_seeds": list(REPLICA_SEEDS),
        "r55_event_index_partition": EVENT_INDEX_RULE,
        "r55_event_index_range": [r55_lo, r55_hi],
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R55 raw block 20300989-20301008 shares no raw seed with "
            "any previously used block (the R50 block 20300925-20300944, "
            "the R51 block 20300945-20300952, the R52 block 20300953-"
            "20300960, the R53 block 20300961-20300968, and the R54 "
            "block 20300969-20300988 all end below it; the Round 4 "
            "blocks 20301020-20301083 and the smoke blocks 20307920-"
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
    partition are additionally disjoint from the R55 production block
    and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(REPLICA_SEEDS):
        _refuse("the smoke seed collides with the R55 production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_REPLICA):
        _refuse("the smoke event-index partition overlaps the R55 "
                "production partition")
    exercise_lo = SMOKE_EXERCISE_INDEX_BASE
    exercise_hi = exercise_lo + SMOKE_EXERCISE_EVENTS_PER_CLASS * len(
        SMOKE_EXERCISE_CLASSES)
    for name, (lo, hi) in (
            ("r55 production", (EVENT_INDEX_BASE,
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
    report["smoke_block_note"] = (
        "the smoke seed 20312925 and the partition 1_600_000_000 + i are "
        "the round-5 shared smoke block (the recorded R50-R54 smoke "
        "convention: the smoke is a hermetic driver validation, never "
        "sealed, and its substream reuse across the round's smoke runs "
        "is the recorded convention; the smoke block is disjoint from "
        "every production block and partition)")
    return report


# ---------------------------------------------------------------------------
# The frozen-fraction assertion against the sealed R51 (before any
# generation)
# ---------------------------------------------------------------------------
def assert_fractions_against_r51() -> dict:
    """The topology sampling fractions are the sealed R51 fitted point
    exactly (the card: "the R51 fit values"; the fit report's seal
    verifies first)."""
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
            "weight_degeneracy) are carried into this validation as the "
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
                 "fallback's anchor; the R55 sampling runs at the R51 "
                 "frozen fractions, so the reference is recorded, not "
                 "drawn)"),
        "match": True,
    }


# ---------------------------------------------------------------------------
# The R52 outcome (the retention semantics of the frozen point)
# ---------------------------------------------------------------------------
def r52_update_rule(verdicts: dict, renorm: dict) -> tuple[dict, bool, str]:
    """The card's two-case frozen-point update rule (the pure form the
    tests exercise): when the zero model is retained by no held-out
    family, the one-excitation topology is retained and the ND fraction
    stays at the R51 fit value; when the zero model is retained by
    every family, the ND fraction is zero and the remaining three
    inelastic fractions are the R52 renormalized zero-model values; a
    mixed per-family verdict is outside the card's two-case rule and
    refuses the run.  Returns ``(fractions, zero_model_retained,
    update_note)``."""
    retained_families = [f for f, v in verdicts.items() if v == "retained"]
    if retained_families and len(retained_families) != len(verdicts):
        _refuse(
            "the sealed R52 retention verdict is mixed across families "
            f"({sorted(verdicts.items())!r}); the card's two-case update "
            "rule covers only the all-retained and none-retained "
            "outcomes; the run refuses")
    zero_retained = bool(retained_families)
    if zero_retained:
        fractions = {
            "sd_projectile": float(renorm["sd_projectile"]),
            "sd_target": float(renorm["sd_target"]),
            "dd": float(renorm["dd"]),
            "nd": 0.0,
        }
        update_note = (
            "the zero model is retained by every held-out family (the "
            "sealed R52 retention verdicts); the ND fraction is zero and "
            "the remaining three inelastic fractions are the R52 "
            "renormalized zero-model values (the card's update rule)")
    else:
        fractions = dict(R51_FROZEN_FRACTIONS)
        update_note = (
            "the zero model is retained by no held-out family (every "
            "family's sealed R52 retention verdict is 'not retained'; "
            "the zero model's held-out deviance degrades beyond the 2.0 "
            "pooled-SE bound); the one-excitation topology is retained "
            "and the ND fraction stays at the R51 fit value (the card's "
            "update rule)")
    return fractions, zero_retained, update_note


def read_r52_outcome() -> dict:
    """The R52 outcome (the central-excitation necessity verdict): the
    sealed R52 ``gate5_r52.json`` (its seal verifies first).  The card's
    update rule: when the zero model (the ND fraction zero, the
    remaining three inelastic fractions at the R52 renormalized values)
    is retained by no held-out family, the one-excitation topology is
    retained and the ND fraction stays at the R51 fit value; when the
    zero model is retained by every family, the ND fraction is zero and
    the remaining three inelastic fractions are the R52 renormalized
    values; a mixed per-family verdict is outside the card's two-case
    rule and refuses the run."""
    run_dir = MONO_ROOT / R52_RUN_DIR
    gate_path = run_dir / R52_GATE_NAME
    sums_path = run_dir / SHA256SUMS_NAME
    if not (run_dir.is_dir() and gate_path.is_file()
            and sums_path.is_file()):
        _refuse(
            "the sealed R52 archive is absent or not yet sealed (the "
            f"card's dependency list is closed): {gate_path}")
    seal = verify_sealed_dir(R52_RUN_DIR)
    gate = _read_json(gate_path)
    if gate.get("run") != "R52" or not gate.get("sealed"):
        _refuse("the sealed R52 gate record is not the sealed R52 "
                f"production evidence (run {gate.get('run')!r}, sealed "
                f"{gate.get('sealed')!r})")
    retention = gate.get("retention") or {}
    if not retention:
        _refuse("the sealed R52 gate carries no retention block")
    verdicts = {}
    for family, verdict in sorted(retention.items()):
        if verdict not in ("retained", "not retained"):
            _refuse(f"the sealed R52 retention verdict for {family!r} is "
                    f"outside the recorded vocabulary: {verdict!r}")
        verdicts[family] = verdict
    zero_model = gate.get("zero_model") or {}
    renorm = zero_model.get("renormalized_fractions") or {}
    if float(zero_model.get("nd_fraction", float("nan"))) != 0.0:
        _refuse("the sealed R52 zero model's nd_fraction drifted off 0.0: "
                f"{zero_model.get('nd_fraction')!r}")
    if set(renorm) != {"dd", "nd", "sd_projectile", "sd_target"}:
        _refuse("the sealed R52 renormalized-fractions block drifted: "
                f"{sorted(renorm)!r}")
    if float(renorm.get("nd", float("nan"))) != 0.0:
        _refuse("the sealed R52 renormalized nd fraction is not 0.0")
    trigger = gate.get("trigger_for_r53") or {}
    if not trigger.get("zero_model_triggered"):
        _refuse("the sealed R52 zero-model trigger flag drifted off true "
                "(the zero model's held-out degradation is the recorded "
                "verdict surface)")
    fractions, zero_retained, update_note = r52_update_rule(verdicts, renorm)
    inelastic_mass = math.fsum(fractions.values())
    if inelastic_mass != 1.0:
        _refuse("the frozen inelastic fractions after the R52 update do "
                f"not sum to one exactly in float arithmetic: "
                f"{inelastic_mass!r}")
    return {
        "status": "sealed",
        "zero_model_retained": zero_retained,
        "retention_verdicts": verdicts,
        "zero_model_renormalized_fractions": {
            name: float(renorm[name]) for name in sorted(renorm)},
        "zero_model_nd_fraction": 0.0,
        "zero_model_triggered": True,
        "trigger_sigma": float(trigger.get("sigma", float("nan"))),
        "frozen_fractions_after_r52": dict(fractions),
        "update_note": update_note,
        "source": f"{R52_RUN_DIR}/{R52_GATE_NAME}",
        "gate_file_sha256": _sha256_file(gate_path),
        "seal": seal,
    }


def assert_r52_outcome_against_preregistration(outcome: dict,
                                               pre: dict) -> dict:
    """The generation-time consistency of the live R52 outcome against
    the pre-registered block: the zero-model retention flag and the
    frozen fractions must match the pre-registered values exactly; any
    drift refuses (never silently absorbed)."""
    pre_block = (pre.get("topology_sampling") or {}).get("r52_outcome") or {}
    pre_retained = bool(pre_block.get("zero_model_retained"))
    pre_fractions = dict(pre_block.get("frozen_fractions_after_r52") or {})
    now_retained = bool(outcome.get("zero_model_retained"))
    now_fractions = dict(outcome.get("frozen_fractions_after_r52") or {})
    if pre_retained != now_retained:
        _refuse(
            "the live R52 outcome drifted against the pre-registered "
            f"block (pre zero-model-retained {pre_retained!r}, now "
            f"{now_retained!r}); the run refuses")
    if pre_fractions != now_fractions:
        _refuse(
            "the live R52 outcome's frozen fractions drifted against the "
            f"pre-registered block (pre {pre_fractions!r}, now "
            f"{now_fractions!r}); the run refuses")
    return {
        "pre_registered_zero_model_retained": pre_retained,
        "live_zero_model_retained": now_retained,
        "fractions_match": pre_fractions == now_fractions,
        "resolution": "matches the pre-registered sealed outcome",
        "consistent": True,
    }


# ---------------------------------------------------------------------------
# The R53 outcome (the nd_multi retention decision; the R54 machinery
# verbatim)
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
                "the sealed R53 archive is absent or not yet sealed; the "
                "card consumes the R53 outcome when run, else the "
                "unretained default P(n = 2) = 0.0 and nd_multi events "
                "are absent"),
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
            "this pre-registration is invalidated (the run re-registers "
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
# The R54 audit precondition (the card's dependency; asserted before
# any generation and before the validation is interpreted)
# ---------------------------------------------------------------------------
def assert_r54_audit_precondition() -> dict:
    """The sealed R54 ledger audit must pass on the frozen topology
    (zero violations on card 2's five ledger rules) before the
    validation is interpreted (the card's dependency clause; index
    card, Dependencies).  The precondition asserts: the R54 seal is
    intact; the sealed gate's ledger audit passed with zero violations
    on exactly 1,000,000 events; the audit's recorded topology sampling
    fractions equal the frozen fractions block exactly (the audit ran
    on the frozen topology); the recorded R53 outcome matches the live
    read (retained false, P(n = 2) = 0.0)."""
    seal = verify_sealed_dir(R54_RUN_DIR)
    gate = _read_json(MONO_ROOT / R54_RUN_DIR / R54_GATE_NAME)
    if gate.get("run") != "R54" or not gate.get("sealed"):
        _refuse("the sealed R54 gate record is not the sealed R54 "
                f"production evidence (run {gate.get('run')!r}, sealed "
                f"{gate.get('sealed')!r})")
    audit = gate.get("ledger_audit") or {}
    if not audit.get("pass"):
        _refuse("the sealed R54 ledger audit did not pass (the card's "
                "precondition: the audit must pass on the frozen "
                "topology before the validation is interpreted)")
    if int(audit.get("violations", -1)) != 0:
        _refuse("the sealed R54 ledger audit carries "
                f"{audit.get('violations')!r} violations (expected 0)")
    gate_events = int(gate.get("events_checked", -1))
    if gate_events != 1_000_000:
        _refuse("the sealed R54 gate's events_checked drifted off "
                f"1,000,000: {gate.get('events_checked')!r}")
    sampling = gate.get("topology_sampling") or {}
    live_fractions = {
        "elastic": ELASTIC_FRACTION_GENERATION,
        "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
        "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
        "dd": R51_FROZEN_FRACTIONS["dd"],
        "nd": R51_FROZEN_FRACTIONS["nd"],
        "nd_multi": R53_P_N2_DEFAULT,
    }
    if dict(sampling.get("fractions") or {}) != live_fractions:
        _refuse(
            "the sealed R54 audit's recorded topology sampling fractions "
            "differ from the frozen fractions block (the audit must have "
            f"run on the frozen topology): recorded "
            f"{sampling.get('fractions')!r} vs frozen {live_fractions!r}")
    r53_live = read_r53_outcome()
    r54_r53 = gate.get("r53_outcome") or {}
    if bool(r54_r53.get("retained")) != bool(r53_live["retained"]) or \
            float(r54_r53.get("p_n2", float("nan"))) != \
            float(r53_live["p_n2"]):
        _refuse(
            "the sealed R54 gate's recorded R53 outcome differs from the "
            f"live read (recorded retained "
            f"{r54_r53.get('retained')!r} / P(n = 2) "
            f"{r54_r53.get('p_n2')!r}, live retained "
            f"{r53_live['retained']!r} / P(n = 2) {r53_live['p_n2']!r})")
    return {
        "source": f"{R54_RUN_DIR}/{R54_GATE_NAME}",
        "gate_file_sha256": _sha256_file(MONO_ROOT / R54_RUN_DIR
                                         / R54_GATE_NAME),
        "seal": seal,
        "ledger_audit_pass": True,
        "violations": 0,
        "events_checked": gate_events,
        "rules": list(audit.get("rules") or []),
        "per_rule_counts": dict(audit.get("per_rule_counts") or {}),
        "topology_sampling_matches_frozen_fractions": True,
        "r53_outcome_consistent": True,
        "pass": True,
    }
# ---------------------------------------------------------------------------
# The draw parameters and the sampling surface
# ---------------------------------------------------------------------------
def frozen_draw_params(p_n2: float,
                       class_fractions: dict | None = None
                       ) -> TopologyNominalParams:
    """The frozen draw parameters: the frozen inelastic fractions (the
    R51 fitted fractions as updated by the R52 outcome; the R51 values
    when no update mapping is passed), epsilon and B_SD at the R51 fit
    values, the nominal remnant weights, and P(n = 2) at the R53
    outcome's value (0.0 at the unretained default)."""
    return TopologyNominalParams(
        epsilon=FROZEN_FIT["epsilon"],
        b_sd=FROZEN_FIT["b_sd"],
        class_fractions=dict(class_fractions or R51_FROZEN_FRACTIONS),
        remnant_weights=dict(NOMINAL_PARAMS.remnant_weights),
        p_n2=float(p_n2),
    )


def sampling_from_outcomes(r52_outcome: dict, r53_outcome: dict) -> dict:
    """The production sampling surface from the recorded R52 and R53
    outcomes: the frozen draw parameters (the R51 fitted fractions as
    updated by the R52 outcome), the recorded elastic fraction, the
    nd_multi dispatch flag (sampled only when the R53 outcome retains
    the class), and the six-entry recorded fractions block (the gate
    record's shape)."""
    p_n2 = float(r53_outcome["p_n2"])
    nd_multi_active = bool(r53_outcome["retained"]) and p_n2 > 0.0
    r52_fractions = dict(r52_outcome["frozen_fractions_after_r52"])
    params = frozen_draw_params(
        p_n2 if nd_multi_active else 0.0, class_fractions=r52_fractions)
    fractions = {
        "elastic": ELASTIC_FRACTION_GENERATION,
        "sd_projectile": float(
            r52_outcome["frozen_fractions_after_r52"]["sd_projectile"]),
        "sd_target": float(
            r52_outcome["frozen_fractions_after_r52"]["sd_target"]),
        "dd": float(r52_outcome["frozen_fractions_after_r52"]["dd"]),
        "nd": float(r52_outcome["frozen_fractions_after_r52"]["nd"]),
        "nd_multi": (p_n2 if nd_multi_active else R53_P_N2_DEFAULT),
    }
    return {
        "params": params,
        "elastic_fraction": ELASTIC_FRACTION_GENERATION,
        "nd_multi_active": nd_multi_active,
        "p_n2": float(params.p_n2),
        "fractions": fractions,
        "r52_outcome": dict(r52_outcome),
        "r53_outcome": dict(r53_outcome),
    }


def frozen_fractions_block(sampling: dict) -> dict:
    """The six-entry frozen fractions block (the gate record's shape)
    of a sampling surface."""
    return dict(sampling["fractions"])


# ---------------------------------------------------------------------------
# The published-lattice geometry (the dense-accumulation/then-pick
# convention; the R01/R44/R51 recorded instantiation)
# ---------------------------------------------------------------------------
def _unique_edges(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    keep = [float(values[0])]
    for value in values[1:]:
        if float(value) != keep[-1]:
            keep.append(float(value))
    return np.asarray(keep, dtype=float)


def _cell_area_from_table(table, i: int) -> float:
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[i])
    ip = int(table.bin_index_pt[i])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


def _pi_zero_model_table(data_tables: list):
    """The model-only pi0 table: the published pi- table's grid geometry
    at the pi0 mass and pdg (the card's pre-registered pi0 territory;
    the recorded R52 pi0 convention extended to the published lattice;
    no data vector, no deviance)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    pi_minus = next(t for t in data_tables if t.species == "pi_minus")
    n_bins = int(pi_minus.data.size)
    xf_pts = np.asarray(pi_minus.xf_centers, dtype=float)[
        np.asarray(pi_minus.bin_index_xf, dtype=int)]
    pt_pts = np.asarray(pi_minus.pt_centers, dtype=float)[
        np.asarray(pi_minus.bin_index_pt, dtype=int)]
    x_edges = sps_core.edges_from_centers(
        np.asarray(pi_minus.xf_centers, dtype=float))
    p_edges = sps_core.edges_from_centers(
        np.asarray(pi_minus.pt_centers, dtype=float))
    ones = np.ones(n_bins, dtype=float)
    table = BinnedFitTable.from_arrays(
        species="pi_zero",
        pdg_id=111,
        mass_gev=PI_ZERO_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=31.78,
        path=PI_ZERO_TABLE_PATH,
        xf=xf_pts,
        pt=pt_pts,
        value=ones,
        error=ones,
        window={
            "xF": [float(x_edges[0]), float(x_edges[-1])],
            "pT": [float(p_edges[0]), float(p_edges[-1])],
        },
    )
    if int(table.data.size) != n_bins:
        _refuse(
            "the model-only pi0 table's bin count "
            f"{int(table.data.size)} does not match the pi- grid's "
            f"{n_bins}"
        )
    return table


def _load_raw_config() -> dict:
    path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    if not path.is_file():
        _refuse(f"production config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_production_tables() -> list:
    """The SPS tables (the published binning; the R01 loader) plus the
    model-only pi0 table on the pi- geometry (the card's pre-registered
    pi0 territory; no data vector).  The production pre-registration and
    evaluation read the SPS tables; the smoke mode builds the synthetic
    hermetic grid."""
    tables = r01.load_binned_tables(_load_raw_config())
    tables.append(_pi_zero_model_table(tables))
    return tables


def _load_smoke_tables() -> list:
    """The synthetic hermetic binning grid of the smoke mode: the four
    data-backed nonstrange tables built from the committed table
    declarations (species, pdg, mass, normalization) only; no SPS table
    is read (the R32 smoke-grid pattern), plus the pi0 model-only table
    on the pi- geometry."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    raw_config = _load_raw_config()
    binned = (raw_config.get("fit") or {}).get("binned") or {}
    if not binned or not binned.get("tables"):
        _refuse("config carries no fit.binned tables block; the smoke "
                "grid needs the committed table declarations")
    n_x = 10
    n_p = 10
    window = dict(binned.get("window") or {})
    xf_lo = float(window.get("xf_min", r01.PAPER_WINDOW["xF"][0]))
    xf_hi = float(window.get("xf_max", r01.PAPER_WINDOW["xF"][1]))
    pt_lo = float(window.get("pt_min", r01.PAPER_WINDOW["pT"][0]))
    pt_hi = float(window.get("pt_max", r01.PAPER_WINDOW["pT"][1]))
    xf_centers = np.linspace(xf_lo, xf_hi, n_x + 2)[1:-1]
    pt_centers = np.linspace(pt_lo, pt_hi, n_p + 2)[1:-1]
    tables: list = []
    for entry in binned["tables"]:
        if str(entry["species"]) not in DATA_SPECIES:
            continue
        xf_pts = np.tile(xf_centers, n_p)
        pt_pts = np.repeat(pt_centers, n_x)
        ones = np.ones(xf_pts.size, dtype=float)
        tables.append(BinnedFitTable.from_arrays(
            species=str(entry["species"]),
            pdg_id=int(entry["pdg"]),
            mass_gev=float(entry["mass_gev"]),
            normalization=str(entry["normalization"]),
            sigma_inel_mb=float(binned.get("sigma_inel_mb", 31.78)),
            path="smoke://synthetic-grid",
            xf=xf_pts,
            pt=pt_pts,
            value=ones,
            error=ones,
            window=window,
        ))
    if len(tables) != len(DATA_SPECIES):
        _refuse("the smoke grid built "
                f"{len(tables)} tables, expected the four data-backed "
                f"nonstrange species {DATA_SPECIES}")
    tables.append(_pi_zero_model_table(tables))
    return tables


def _load_sealed_r01(tables: list) -> dict:
    """The sealed R01 data vector and published variances of the
    data-backed species, cross-checked bit for bit against the locally
    loaded SPS tables (the R11 parse_sealed_binned discipline; the
    R51/R44 loader verbatim)."""
    archive = MONO_ROOT / R01_RUN_DIR
    residuals = _read_json(archive / R01_RESIDUALS_NAME)
    covariance = _read_json(archive / R01_COVARIANCE_NAME)
    sealed: dict[str, dict[str, list[float]]] = {}
    for table in tables:
        if table.species not in DATA_SPECIES:
            continue
        sealed[table.species] = {
            "d": [], "v_data": [], "xF_mid": [], "pT_mid": [],
        }
    for entry in residuals:
        if entry["species"] not in sealed:
            # the kaon residuals are present in the sealed archive and
            # are not consumed (the kaon sectors are not in the R55
            # pre-registered validation register)
            continue
        slot = sealed[entry["species"]]
        slot["d"].append(float(entry["d"]))
    for entry in covariance["V_data"]:
        if entry["species"] not in sealed:
            continue
        sealed[entry["species"]]["v_data"].append(
            float(entry["variance"]))
    for entry in _read_json(archive / R01_PREDICTIONS_NAME):
        if entry["species"] not in sealed:
            continue
        slot = sealed[entry["species"]]
        slot["xF_mid"].append(float(entry["xF_mid"]))
        slot["pT_mid"].append(float(entry["pT_mid"]))
    for table in tables:
        if table.species not in sealed:
            continue
        n_bins = int(table.data.size)
        slot = sealed[table.species]
        for field, values in slot.items():
            if len(values) != n_bins:
                _refuse(
                    f"sealed {table.species} {field} carries "
                    f"{len(values)} entries for {n_bins} bins"
                )
        for i in range(n_bins):
            if slot["d"][i] != float(table.data[i]):
                _refuse(
                    f"sealed {table.species} bin {i}: the sealed d "
                    "differs from the loaded SPS table value; the "
                    "sealed archive and the local tables disagree"
                )
    return sealed


def build_geometry(tables: list) -> dict:
    """The dense-lattice geometry of the five nonstrange species (the
    dense-accumulation/then-pick convention): per species the (x_F,
    p_T) cell edges, the pT-major dense lattice size, the dense offset
    in the flat accumulation array, the published-row pick map
    ``bin_index_pt * n_x + bin_index_xf``, and the published rows' cell
    areas.  A hole cell (a lattice cell with no published row) stays
    unpicked; its weight is dropped, never bled into a neighboring row
    or species."""
    geometry: dict = {
        "edges": {}, "n_x": {}, "n_p": {}, "dense_sizes": {},
        "dense_offsets": {}, "pick": {}, "areas": {}, "row_count": {},
        "pi_zero_geometry_note": PI_ZERO_GEOMETRY_NOTE,
    }
    dense_offset = 0
    for table in tables:
        species = str(table.species)
        if species not in SPECTRUM_SPECIES_ORDER:
            # the card's pre-registered validation register: the kaon
            # published tables are present in the R01 loader's output
            # and are not accumulated (recorded, not bled)
            continue
        x_edges = _unique_edges(sps_core.edges_from_centers(
            table.xf_centers))
        p_edges = _unique_edges(sps_core.edges_from_centers(
            table.pt_centers))
        n_x = int(np.asarray(table.bin_index_xf).max()) + 1
        n_p = int(np.asarray(table.bin_index_pt).max()) + 1
        pick = (np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
                + np.asarray(table.bin_index_xf, dtype=np.int64))
        if pick.size and not (
                int(pick.min()) >= 0 and int(pick.max()) < n_x * n_p):
            _refuse(
                f"species {species!r}: a published bin_index pair maps "
                f"outside the dense lattice ({n_x} x {n_p}); the pick "
                "map cannot be built")
        if len(set(pick.tolist())) != int(pick.size):
            _refuse(
                f"species {species!r}: two published rows map to one "
                "dense lattice cell; the pick map is not one-to-one")
        areas = [_cell_area_from_table(table, i)
                 for i in range(int(table.data.size))]
        geometry["edges"][species] = (
            [float(v) for v in x_edges], [float(v) for v in p_edges])
        geometry["n_x"][species] = n_x
        geometry["n_p"][species] = n_p
        geometry["dense_sizes"][species] = n_x * n_p
        geometry["dense_offsets"][species] = dense_offset
        geometry["pick"][species] = [int(v) for v in pick]
        geometry["areas"][species] = [float(a) for a in areas]
        geometry["row_count"][species] = int(pick.size)
        dense_offset += n_x * n_p
    geometry["dense_total"] = int(dense_offset)
    if sorted(geometry["edges"]) != sorted(SPECTRUM_SPECIES_ORDER):
        _refuse(f"the geometry's species set drifted: "
                f"{sorted(geometry['edges'])!r}")
    return geometry


def _assign_cells(xf, pt, x_edges, p_edges, n_x):
    """The dense-lattice cell assignment of (x_F, p_T) pairs (the
    published geometry; out-of-window pairs are dropped and counted,
    never bled into an end cell)."""
    xf = np.asarray(xf, dtype=float)
    pt = np.asarray(pt, dtype=float)
    if xf.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    x_edges = np.asarray(x_edges, dtype=float)
    p_edges = np.asarray(p_edges, dtype=float)
    ix = np.searchsorted(x_edges, xf, side="right") - 1
    ip = np.searchsorted(p_edges, pt, side="right") - 1
    inside = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < p_edges.size - 1)
    dropped = int((~inside).sum())
    flat = (ip[inside] * n_x + ix[inside]).astype(np.int64)
    return flat, dropped


# ---------------------------------------------------------------------------
# The per-event physics layer (the R51 conservation contract consumed
# verbatim) and the CM-frame extraction
# ---------------------------------------------------------------------------
_CM_BETA, _CM_GAMMA = r51._cm_boost_parameters(SQRT_S)


def live_finals_of(event) -> tuple:
    """The event's final-state records (the class records' final_state;
    the elastic event's two outgoing lines)."""
    if isinstance(event, ElasticEvent):
        return tuple(event.outgoing)
    return tuple(event.final_state)


def live_finals_as_records(event) -> list:
    """The event's final states in the stored-record 10-field form
    (pdg, B, Q, S, 2I3, mass, E, px, py, pz) so the live accumulation
    and the sealed-R50 reference recomputation share one record shape."""
    records = []
    for hadron in live_finals_of(event):
        b, q, s, i3x2 = r51._line_qn(hadron)
        pdg = getattr(hadron, "pdg_id", None)
        records.append([
            (None if pdg is None else int(pdg)),
            int(b), int(q), int(s), int(i3x2),
            float(getattr(hadron, "mass", PROTON_MASS)),
            float(hadron.four_momentum[0]), float(hadron.four_momentum[1]),
            float(hadron.four_momentum[2]), float(hadron.four_momentum[3]),
        ])
    return records


def lead_xf_records_of_finals(finals) -> tuple[list, list]:
    """The final-state proton and antiproton x_F values (the CM-frame
    Feynman x at the extraction energy, x_F = 2 p_L^CM / sqrt(s); the
    R51 spectrum-record convention)."""
    protons: list[float] = []
    pbars: list[float] = []
    for fsr in finals:
        pdg = fsr[0] if isinstance(fsr, (list, tuple)) else \
            getattr(fsr, "pdg_id", None)
        if pdg is None or int(pdg) not in (2212, -2212):
            continue
        if isinstance(fsr, (list, tuple)):
            e, _px, _py, pz = (float(fsr[6]), float(fsr[7]),
                               float(fsr[8]), float(fsr[9]))
        else:
            e, _px, _py, pz = (float(x) for x in fsr.four_momentum)
        pz_cm = _CM_GAMMA * (pz - _CM_BETA * e)
        x_f = 2.0 * pz_cm / SQRT_S
        if int(pdg) == 2212:
            protons.append(x_f)
        else:
            pbars.append(x_f)
    return protons, pbars


def charged_multiplicity_of_records(finals) -> int:
    """The charged multiplicity of final-state records (the count of
    records with nonzero charge Q; the below-threshold baryonic system
    carries Q = 0 and does not count; the elastic pair's two lines
    count)."""
    n = 0
    for fsr in finals:
        if isinstance(fsr, (list, tuple)):
            if int(fsr[2]) != 0:
                n += 1
        else:
            if r51._line_qn(fsr)[1] != 0:
                n += 1
    return n


class _ReplicaAccumulator:
    """One replica's validation accumulation (the R44 sparse-row
    convention adapted to the topology events): the finals' dense
    lattice rows (flat index, invariant measure weight), the
    leading-baryon x_F histograms, the gap-size histogram, the
    charged-multiplicity histogram, the species-separated multiplicity
    histograms, and the rolling per-event digest."""

    __slots__ = (
        "rows_flat", "rows_weight", "lead_xf", "gap", "nch", "spec_mult",
        "rolling", "_geometry",
    )

    def __init__(self, geometry: dict) -> None:
        self.rows_flat: list[int] = []
        self.rows_weight: list[float] = []
        self.lead_xf = {
            "proton": [0] * (len(LEAD_XF_HISTOGRAM_EDGES) - 1),
            "antiproton": [0] * (len(LEAD_XF_HISTOGRAM_EDGES) - 1),
            "dropped": {"proton": 0, "antiproton": 0},
            "n_protons": 0,
            "n_antiprotons": 0,
        }
        self.gap = {
            "hist": [0] * (len(GAP_HISTOGRAM_EDGES) - 1),
            "events_with_field": 0,
            "events_without_field": 0,
            "dropped_out_of_range": 0,
        }
        self.nch = [0] * len(NCH_BINS)
        self.spec_mult = {
            sp: [0] * (SPECIES_MULT_SUPPORT_HI + 1)
            for sp in SPECIES_MULT_ORDER}
        self.rolling = hashlib.sha256()
        self._geometry = geometry

    # -- the spectra accumulation (the dense-accumulation/then-pick
    #    convention; the finals' CM-frame (x_F, p_T) cells with the
    #    invariant measure weight e_CM / p_T) --
    def add_spectrum_final(self, species: str,
                           p4: tuple[float, float, float, float]) -> None:
        e, px, py, pz = (float(v) for v in p4)
        pz_cm = _CM_GAMMA * (pz - _CM_BETA * e)
        e_cm = _CM_GAMMA * (e - _CM_BETA * pz)
        p_t = math.sqrt(px * px + py * py)
        if p_t <= 0.0:
            return
        x_f = 2.0 * pz_cm / SQRT_S
        x_edges, p_edges = self._geometry["edges"][species]
        flat, _dropped = _assign_cells(
            np.array([x_f]), np.array([p_t]), x_edges, p_edges,
            self._geometry["n_x"][species])
        if flat.size:
            self.rows_flat.append(
                int(self._geometry["dense_offsets"][species])
                + int(flat[0]))
            self.rows_weight.append(e_cm / p_t)

    # -- the 1D count histograms (np.histogram semantics: the last bin
    #    closed on the right; out-of-range values dropped and counted) --
    @staticmethod
    def _bin_index(value: float, edges: tuple[float, ...]) -> int | None:
        if value < edges[0] or value > edges[-1]:
            return None
        if value == edges[-1]:
            return len(edges) - 2
        lo, hi = 0, len(edges) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if value < edges[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    def add_lead_xf(self, species: str, x_f: float) -> None:
        idx = self._bin_index(x_f, LEAD_XF_HISTOGRAM_EDGES)
        if idx is None:
            self.lead_xf["dropped"][species] += 1
            return
        self.lead_xf[species][idx] += 1
        self.lead_xf[f"n_{species}s"] += 1

    def add_gap(self, gap_size) -> None:
        if gap_size is None:
            self.gap["events_without_field"] += 1
            return
        idx = self._bin_index(float(gap_size), GAP_HISTOGRAM_EDGES)
        if idx is None:
            self.gap["dropped_out_of_range"] += 1
            self.gap["events_with_field"] += 1
            return
        self.gap["hist"][idx] += 1
        self.gap["events_with_field"] += 1

    def add_nch(self, n_ch: int) -> None:
        if not NCH_SUPPORT_LO <= n_ch <= NCH_SUPPORT_HI:
            raise ValidationRefusal(
                f"the event carries charged multiplicity {n_ch} outside "
                f"the pre-registered support {NCH_BINS[0]}.."
                f"{NCH_BINS[-1]}; the accumulation refuses (checked, "
                "never bled)")
        self.nch[n_ch - NCH_SUPPORT_LO] += 1

    def add_species_count(self, species: str, count: int) -> None:
        if not 0 <= count <= SPECIES_MULT_SUPPORT_HI:
            raise ValidationRefusal(
                f"the event carries a {species} multiplicity {count} "
                "outside the pre-registered support 0.."
                f"{SPECIES_MULT_SUPPORT_HI}; the accumulation refuses "
                "(checked, never bled)")
        self.spec_mult[species][count] += 1

    def update_rolling(self, line: str) -> None:
        self.rolling.update(line.encode("utf-8"))
        self.rolling.update(b"\n")

    def event_rolling_digest(self) -> str:
        return self.rolling.hexdigest()


def accumulate_live_event(acc: _ReplicaAccumulator, event, obs,
                          event_index: int) -> None:
    """Accumulate one live class event into the replica accumulator:
    the final-state species' spectra rows and multiplicity counts, the
    charged multiplicity, the gap size, and the rolling per-event
    digest line (the class label, the event index, and every final's
    pdg with exact hex four-momentum components)."""
    fs_records = live_finals_as_records(event)
    species_counts = {sp: 0 for sp in SPECIES_MULT_ORDER}
    fs_lines: list[str] = []
    for fsr in fs_records:
        pdg = fsr[0]
        e, px, py, pz = fsr[6], fsr[7], fsr[8], fsr[9]
        if pdg is None:
            fs_lines.append(
                f"none:{float(e).hex()}:{float(px).hex()}:"
                f"{float(py).hex()}:{float(pz).hex()}")
        else:
            fs_lines.append(
                f"{int(pdg)}:{float(e).hex()}:{float(px).hex()}:"
                f"{float(py).hex()}:{float(pz).hex()}")
        if pdg is None:
            continue
        species = PDG_TO_SPECIES.get(int(pdg))
        if species is None:
            continue
        species_counts[species] += 1
        acc.add_spectrum_final(species, (e, px, py, pz))
    protons, pbars = lead_xf_records_of_finals(fs_records)
    for x_f in protons:
        acc.add_lead_xf("proton", x_f)
    for x_f in pbars:
        acc.add_lead_xf("antiproton", x_f)
    n_ch = charged_multiplicity_of_records(fs_records)
    acc.add_nch(n_ch)
    for sp in SPECIES_MULT_ORDER:
        acc.add_species_count(sp, species_counts[sp])
    acc.add_gap(obs.gap_size)
    acc.update_rolling(
        f"{event.class_label}|{int(event_index)}|{n_ch}|"
        + ("none" if obs.gap_size is None
           else float(obs.gap_size).hex())
        + "|" + ";".join(fs_lines))


# ---------------------------------------------------------------------------
# The R50-pinned reference recomputation (the pre-registration step)
# ---------------------------------------------------------------------------
def r50_reference_payload(geometry: dict) -> dict:
    """The sealed R50 nominal-point reference, recomputed read-only from
    the sealed R50 slot records with the R55 validation register's
    accumulation (the per-class histograms at the register's pinned
    binnings) and cross-checked against the sealed R51 fit report's
    reference data bit-exactly (the 0.25-each pooled gap fraction and
    charged-multiplicity distribution the R51 fit consumed as its
    training data side).  The reweighted-to-frozen-fractions mixtures
    are computed at evaluation from the pinned per-class histograms."""
    seal = verify_sealed_dir(R50_RUN_DIR)
    template = _read_json(MONO_ROOT / R50_RUN_DIR
                          / R50_TEMPLATE_REPORT_NAME)
    # the pinned binnings are verified against the sealed R50 template
    # report's stored histograms (the pin-and-verify discipline)
    stored_gap = template["per_class"]["dd"]["gap_size"]["histogram"]
    if [float(v) for v in stored_gap["edges"]] != \
            [float(v) for v in GAP_HISTOGRAM_EDGES]:
        _refuse("the pinned gap-size edges differ from the sealed R50 "
                "template report's stored histogram edges")
    stored_xf = template["per_class"]["dd"]["leading_baryon_xf"][
        "projectile"]["histogram"]
    if [float(v) for v in stored_xf["edges"]] != \
            [float(v) for v in LEAD_XF_HISTOGRAM_EDGES]:
        _refuse("the pinned leading-baryon x_F edges differ from the "
                "sealed R50 template report's stored histogram edges")
    per_class: dict[str, dict] = {}
    for cls in CLASS_ORDER:
        nch = [0] * len(NCH_BINS)
        spec = {sp: [0] * (SPECIES_MULT_SUPPORT_HI + 1)
                for sp in SPECIES_MULT_ORDER}
        gap = {"hist": [0] * (len(GAP_HISTOGRAM_EDGES) - 1),
               "events_with_field": 0, "events_without_field": 0,
               "dropped_out_of_range": 0}
        lead = {"proton": [0] * (len(LEAD_XF_HISTOGRAM_EDGES) - 1),
                "antiproton": [0] * (len(LEAD_XF_HISTOGRAM_EDGES) - 1),
                "dropped": {"proton": 0, "antiproton": 0},
                "n_protons": 0, "n_antiprotons": 0}
        gap_flags = 0
        n_events = 0
        for rep in range(4):
            path = MONO_ROOT / R50_RUN_DIR / R50_SLOT_FMT.format(cls, rep)
            if not path.is_file():
                _refuse(f"the sealed R50 slot record is missing: {path}")
            record = _read_json(path)
            if record.get("class_label") != cls:
                _refuse(f"the R50 slot record {path.name} carries class "
                        f"{record.get('class_label')!r}")
            if int(record.get("n_events")) != 25_000:
                _refuse(f"the R50 slot record {path.name} carries "
                        f"{record.get('n_events')} events, expected "
                        "25,000")
            acc = _ReplicaAccumulator(geometry)
            for event in record["events"]:
                n_events += 1
                n_ch = charged_multiplicity_of_records(event["fs"])
                acc.add_nch(n_ch)
                counts = {sp: 0 for sp in SPECIES_MULT_ORDER}
                for fsr in event["fs"]:
                    pdg = fsr[0]
                    if pdg is None:
                        continue
                    species = PDG_TO_SPECIES.get(int(pdg))
                    if species is None:
                        continue
                    counts[species] += 1
                    acc.add_spectrum_final(
                        species,
                        (float(fsr[6]), float(fsr[7]), float(fsr[8]),
                         float(fsr[9])))
                for sp in SPECIES_MULT_ORDER:
                    acc.add_species_count(sp, counts[sp])
                protons, pbars = lead_xf_records_of_finals(event["fs"])
                for x_f in protons:
                    acc.add_lead_xf("proton", x_f)
                for x_f in pbars:
                    acc.add_lead_xf("antiproton", x_f)
                acc.add_gap(event.get("gp"))
                if int(event["ob"][3]) == 0:
                    gap_flags += 1
            # the accumulated per-class histograms (exact integer counts)
            for k in range(len(nch)):
                nch[k] += acc.nch[k]
            for sp in SPECIES_MULT_ORDER:
                for k in range(SPECIES_MULT_SUPPORT_HI + 1):
                    spec[sp][k] += acc.spec_mult[sp][k]
            for k in range(len(gap["hist"])):
                gap["hist"][k] += acc.gap["hist"][k]
            gap["events_with_field"] += acc.gap["events_with_field"]
            gap["events_without_field"] += acc.gap["events_without_field"]
            gap["dropped_out_of_range"] += acc.gap["dropped_out_of_range"]
            for sp in ("proton", "antiproton"):
                for k in range(len(lead[sp])):
                    lead[sp][k] += acc.lead_xf[sp][k]
                lead["dropped"][sp] += acc.lead_xf["dropped"][sp]
                lead[f"n_{sp}s"] += acc.lead_xf[f"n_{sp}s"]
        per_class[cls] = {
            "n_events": n_events,
            "n_ch_hist": nch,
            "species_counts": spec,
            "gap": gap,
            "lead_xf": lead,
            "gap_flag_count": gap_flags,
        }
    # -- the 0.25-each pooled cross-check against the sealed R51 fit
    #    report's reference data (bit-exact; the R51 fit consumed the
    #    same reference as its training data side) --
    fractions = {cls: 0.25 for cls in CLASS_ORDER}
    gap_fraction_r51 = math.fsum(
        fractions[cls] * per_class[cls]["gap_flag_count"]
        / per_class[cls]["n_events"]
        for cls in CLASS_ORDER)
    gap_variance_r51 = math.fsum(
        fractions[cls] ** 2
        * (per_class[cls]["gap_flag_count"] / per_class[cls]["n_events"])
        * (1.0 - per_class[cls]["gap_flag_count"]
           / per_class[cls]["n_events"])
        / per_class[cls]["n_events"]
        for cls in CLASS_ORDER)
    mult_distribution = {}
    mult_variance = {}
    for n in range(2, 15):
        p = math.fsum(
            fractions[cls] * per_class[cls]["n_ch_hist"][n - 2]
            / per_class[cls]["n_events"]
            for cls in CLASS_ORDER)
        v = math.fsum(
            fractions[cls] ** 2
            * (per_class[cls]["n_ch_hist"][n - 2]
               / per_class[cls]["n_events"])
            * (1.0 - per_class[cls]["n_ch_hist"][n - 2]
               / per_class[cls]["n_events"])
            / per_class[cls]["n_events"]
            for cls in CLASS_ORDER)
        mult_distribution[str(n)] = p
        mult_variance[str(n)] = v
    # the cross-check target is the sealed R51 pre-registration's pinned
    # R50 reference (the pre-registration precedence; the R51 fit
    # report's per-statistic ``mc_se`` values are the MODEL side's
    # between-replica standard errors, not the reference variances)
    r51_pre_path = MONO_ROOT / R51_RUN_DIR / PREREGISTRATION_NAME
    if not r51_pre_path.is_file():
        _refuse(f"the sealed R51 pre-registration is missing: "
                f"{r51_pre_path}")
    r51_pre = _read_json(r51_pre_path)
    r51_ref = r51_pre.get("r50_reference") or {}
    if not r51_ref:
        _refuse("the sealed R51 pre-registration carries no r50_reference "
                "block (the cross-check cannot run)")
    r51_gap_value = float(r51_ref["gap_fraction"]["value"])
    r51_gap_variance = float(r51_ref["gap_fraction"]["variance"])
    r51_mult_data = [float(r51_ref["charged_multiplicity"]["distribution"][
        str(n)]) for n in range(2, 15)]
    r51_mult_variance = [float(r51_ref["charged_multiplicity"]["variance"][
        str(n)]) for n in range(2, 15)]
    checks = {
        "gap_fraction_value": {
            "recomputed": gap_fraction_r51,
            "r51": r51_gap_value,
            "match": gap_fraction_r51 == r51_gap_value,
        },
        "gap_fraction_variance": {
            "recomputed": gap_variance_r51,
            "r51": r51_gap_variance,
            "match": gap_variance_r51 == r51_gap_variance,
        },
        "charged_distribution": {
            "recomputed": [mult_distribution[str(n)]
                           for n in range(2, 15)],
            "r51": r51_mult_data,
            "match": [mult_distribution[str(n)] for n in range(2, 15)]
                     == r51_mult_data,
        },
        "charged_variance": {
            "recomputed": [mult_variance[str(n)] for n in range(2, 15)],
            "r51": r51_mult_variance,
            "match": [mult_variance[str(n)] for n in range(2, 15)]
                     == r51_mult_variance,
        },
    }
    failures = [name for name, entry in checks.items()
                if not entry["match"]]
    if failures:
        _refuse(f"the recomputed R50 reference disagrees with the sealed "
                f"R51 fit report's reference data on {failures}; the "
                "pre-registration precedence refuses (the run will not "
                "build on a drifted reference)")
    return {
        "source": R50_RUN_DIR,
        "seal": seal,
        "construction": (
            "the per-class histograms of the sealed R50 T1 slots "
            "(100,000 events per inelastic class) recomputed read-only "
            "with the R55 validation register's accumulation (the "
            "pinned binnings; the CM-frame Feynman-x extraction at the "
            "extraction energy); the 0.25-each pooled mixtures "
            "cross-check bit-exactly against the sealed R51 "
            "pre-registration's pinned r50_reference block (the "
            "pre-registration precedence)"),
        "events_per_class": {cls: per_class[cls]["n_events"]
                             for cls in CLASS_ORDER},
        "per_class": {
            cls: {
                "n_events": per_class[cls]["n_events"],
                "n_ch_hist": per_class[cls]["n_ch_hist"],
                "species_counts": per_class[cls]["species_counts"],
                "gap": per_class[cls]["gap"],
                "lead_xf": per_class[cls]["lead_xf"],
                "gap_flag_count": per_class[cls]["gap_flag_count"],
            }
            for cls in CLASS_ORDER},
        "cross_check": checks,
        "r51_reference_overlap_note": R51_TRAINING_NOTE,
    }
# ---------------------------------------------------------------------------
# The generation (the mixed-draw replica records at the frozen point)
# ---------------------------------------------------------------------------
REPLICA_SCHEMA = "na49_mixing.staged_r55_frozen_topology_validation.replica/1"

GENERATION_FRACTIONS_NOTE = (
    "the R55 mixed draw runs at the round's frozen point (the R51 fitted "
    "fractions as updated by the R52 outcome -- the sealed R52 retention "
    "verdicts keep the one-excitation topology, so the ND fraction stays "
    "at the R51 fit value -- and the R53 outcome's unretained nd_multi "
    "default, P(n = 2) = 0.0) with the elastic fraction the exact "
    "complement 0.0 (the recorded source convention; see the "
    "elastic-inclusion note); the mass-law parameters are frozen at the "
    "R51 fitted epsilon and B_SD; the blend ramp start stays fixed at "
    "B = 2.680194110727528 GeV (Round 4 obligation 4).  No refit: "
    "no_fitting is true, optimizer_invoked is false (index item 10).")

GENERATION_GATEWAY_NOTE = (
    "gateway-primary generation (host cl, workers c0801-c0809 x 32 task "
    "slots; one job per replica, --tasks 1, single math thread "
    "(NA49_CLUSTER_GW_TASKS forces thread counts to 1), submission "
    "paced by the capacity check, no local process pool inside the "
    "chunk command, chunk outputs appended only to the new run dir); "
    "the driver's local fallback is one single-threaded process per "
    "--replica invocation (a completed record that matches the "
    "pre-registered identity is reused unchanged)")


def generate_replica_events(
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
    sampling: dict,
    geometry: dict,
):
    """Generate *n_events* mixed-draw events of one replica at the
    recorded sampling surface and return the accumulator block, the
    class census, the count census, and the conservation aggregate.

    Every event consumes the closed five-label substream quintuple at
    replica slot 0 (the R50/R51/R52/R53/R54 convention: the raw seed
    carries the replica identity), draws its class on the closed
    ``topology_selection`` substream at the frozen fractions (the
    elastic class included in the draw at its recorded fraction), and
    dispatches the card-9 ``run_topology`` (a drawn ``nd`` label
    dispatches the card-8 nd_multi generation exactly when the R53
    outcome retains the class, else card 7's ND generation), extracts
    the observables, runs the house conservation contract (a violation
    raises and stops the run), and accumulates the validation register
    (the dense-accumulation/then-pick spectra rows, the leading-baryon
    x_F histograms, the gap-size histogram, the charged and
    species-separated multiplicity histograms, and the rolling
    per-event digest).  The elastic class's pre-registered zero-
    transfer code check runs on every generated elastic event (plan
    line 824)."""
    params = sampling["params"]
    elastic_fraction = float(sampling["elastic_fraction"])
    nd_multi_active = bool(sampling["nd_multi_active"])
    acc = _ReplicaAccumulator(geometry)
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
            params.class_fractions, elastic_fraction, rng_topology_selection)
        if label not in INELASTIC_SET and label != "elastic":
            raise ValidationRefusal(
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
        if isinstance(event, ElasticEvent):
            # the elastic class's pre-registered code check (plan line
            # 824): the zero-transfer ledger audit of the normalization
            # reference; a violation raises and stops the run
            check_zero_transfer(event)
        obs = extract_observables(event, SQRT_S)
        conservation = r51.event_conservation(event)
        accumulate_live_event(acc, event, obs, event_index_base + i)
        if isinstance(event, NDMultiEvent):
            if event.count == 2:
                count_census["n2"] += 1
            else:
                count_census["n1"] += 1
        census[label] = census.get(label, 0) + 1
        bqs_census[tuple(conservation["bqs"])] = bqs_census.get(
            tuple(conservation["bqs"]), 0) + 1
        worst_closure = max(worst_closure, conservation["closure_worst"])
        worst_shell = max(worst_shell, conservation["mass_shell_worst"])
        worst_energy_gev = max(worst_energy_gev,
                               conservation["d_energy_gev"])
        worst_momentum_gev = max(worst_momentum_gev,
                                 conservation["d_momentum_gev"])
    conservation_agg = {
        "events": int(n_events),
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
    return {
        "accumulation": {
            "rows_flat": acc.rows_flat,
            "rows_weight": acc.rows_weight,
            "lead_xf": acc.lead_xf,
            "gap": acc.gap,
            "nch": acc.nch,
            "spec_mult": acc.spec_mult,
            "event_rolling_sha256": acc.event_rolling_digest(),
        },
        "class_census": census,
        "count_census": count_census,
        "conservation": conservation_agg,
    }


def _identity_block(run_dir: Path, *, smoke: bool = False) -> dict:
    return {
        "driver_sha256": DRIVER_SHA256,
        "pre_registration_sha256": _sha256_file(
            run_dir / (("smoke_" + PREREGISTRATION_NAME) if smoke
                       else PREREGISTRATION_NAME)),
    }


def build_replica_record(
    replica: int | str,
    generation: dict,
    run_dir: Path,
    sampling: dict,
    outcome_resolutions: dict,
    mode: str = "production",
) -> dict:
    """The replica record: the identity block, the seed map, the
    generation fractions, the accumulated validation register, the
    class and count censuses, the conservation aggregate, and the
    rolling per-event digest (the generation-time determinism
    evidence)."""
    record = {
        "schema": REPLICA_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "replica": (replica if isinstance(replica, str)
                    else int(replica)),
        "seed": (SMOKE_SEED if isinstance(replica, str)
                 else replica_seed(replica)),
        "n_events": generation["conservation"]["events"],
        "identity": _identity_block(run_dir, smoke=(mode == "smoke")),
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 64 + k = 20300989 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300989-20301008 is disjoint from every "
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
            "class_fractions": dict(sampling["fractions"]),
            "elastic_fraction": float(sampling["elastic_fraction"]),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": float(sampling["p_n2"]),
            "nd_multi_active": bool(sampling["nd_multi_active"]),
        },
        "outcome_resolutions": outcome_resolutions,
        "class_census": dict(generation["class_census"]),
        "count_census": dict(generation["count_census"]),
        "conservation": generation["conservation"],
        "accumulation": generation["accumulation"],
        "reused": False,
    }
    record.pop("digest", None)
    record["digest"] = hashlib.sha256(
        json.dumps(record, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return record


def load_pre_registration(run_dir: Path, *, smoke: bool = False) -> dict:
    name = ("smoke_" + PREREGISTRATION_NAME) if smoke else PREREGISTRATION_NAME
    path = Path(run_dir) / name
    if not path.is_file():
        _refuse(f"the run dir carries no pre-registration: {path}")
    return _read_json(path)


def _verify_replica_identity(record: dict, replica: int, run_dir: Path) -> None:
    """A completed production replica record must match the
    pre-registered identity (the seed, the event count, the driver
    digest, the pre-registration digest, and the record's own digest);
    any drift refuses the run (a completed record is reused unchanged
    or refused, never silently overwritten)."""
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
    if record.get("n_events") != EVENTS_PER_REPLICA or \
            int((record.get("conservation") or {}).get("events", -1)) != \
            EVENTS_PER_REPLICA:
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
    stored_digest = record.get("digest")
    recomputed = dict(record)
    recomputed.pop("digest", None)
    digest = hashlib.sha256(
        json.dumps(recomputed, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    if stored_digest != digest:
        _refuse(
            f"the replica record {replica_name(replica)} carries a stale "
            "digest (the stored accumulation does not reproduce it)")


def run_generation(run_dir: Path, replicas: list[int] | None = None) -> list:
    """Generate the requested production replica records (the default is
    every replica), one record file per replica, then stop.  Requires
    the pre-registration; re-asserts the frozen fractions against the
    sealed R51 archive, the R52/R53 outcomes against the pre-registered
    blocks, and the R54 audit precondition before generating.  A
    completed record that matches the pre-registered identity is reused
    unchanged (the pre-registered gateway fallback runs one job per
    replica; the local fallback is one single-threaded process per
    --replica invocation)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    pre = load_pre_registration(run_dir)
    verify_frozen_substrate()
    seed_disjointness_report()
    fraction_assertion = assert_fractions_against_r51()
    r52_outcome = read_r52_outcome()
    r52_resolution = assert_r52_outcome_against_preregistration(
        r52_outcome, pre)
    r53_outcome = read_r53_outcome()
    r53_resolution = assert_outcome_against_preregistration(r53_outcome, pre)
    r54_precondition = assert_r54_audit_precondition()
    sampling = sampling_from_outcomes(r52_outcome, r53_outcome)
    # the frozen fractions must match the pre-registered block exactly
    pre_fractions = dict((pre.get("topology_sampling") or {}).get(
        "fractions") or {})
    if pre_fractions != sampling["fractions"]:
        _refuse(
            "the live frozen fractions drifted against the "
            f"pre-registered block (pre {pre_fractions!r}, now "
            f"{sampling['fractions']!r}); the run refuses")
    if replicas is None:
        replicas = list(range(N_REPLICAS))
    written = []
    catalog = _load_catalog()
    fams = _load_families()
    geometry = build_geometry(_load_production_tables())
    for replica in replicas:
        name = replica_name(replica)
        path = run_dir / name
        if path.is_file():
            existing = _read_json(path)
            _verify_replica_identity(existing, replica, run_dir)
            print(f"[r55] replica {int(replica)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        seed = replica_seed(replica)
        print(
            f"[r55] generating replica {int(replica)} ({name}): seed "
            f"{seed}, {EVENTS_PER_REPLICA} mixed-draw events",
            flush=True,
        )
        generation = generate_replica_events(
            seed, EVENTS_PER_REPLICA, EVENT_INDEX_BASE, catalog, fams,
            sampling, geometry)
        record = build_replica_record(
            replica, generation, run_dir, sampling,
            {"r52": r52_resolution, "r53": r53_resolution,
             "r54_precondition_pass": r54_precondition["pass"]})
        _write_json(path, record)
        print(
            f"[r55] replica {int(replica)} written: {path} "
            f"(census {generation['class_census']}, worst closure "
            f"{generation['conservation']['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written
# ---------------------------------------------------------------------------
# The evaluation (the pooled accumulation and the four validation
# families; the frozen-parameter evaluation, no optimizer)
# ---------------------------------------------------------------------------
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)

SE_CONVENTION = (
    "the between-replica standard error (std ddof=1 over the 20 T3 "
    f"replica blocks, divided by sqrt(20)), floored at the engine's "
    f"DETERMINISTIC_SE_FLOOR = {float(DETERMINISTIC_SE_FLOOR)!r}; the "
    "pre-registered bounds (the 2.0 pooled-SE named-finding bound, the "
    f"{SIGMA_TOLERANCE!r}-sigma house form, the "
    f"{SECTOR_DEGRADATION_FRACTION!r} sector bound) are never relaxed "
    "(the acceptance precedent of TASK-20260918-r3-acceptance-rescope-"
    "card8a-8b)")


def se_from(stack: np.ndarray) -> np.ndarray:
    """The between-replica standard error (std ddof=1 over the blocks,
    divided by sqrt(k); the house convention, floored at the engine's
    deterministic SE floor; fewer than two blocks carry no between-
    replica estimate and the floor stands)."""
    if stack.shape[0] < 2:
        return np.full(stack.shape[1:], float(SE_FLOOR))
    with np.errstate(invalid="ignore"):
        out = np.std(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])
    out = np.where(np.isfinite(out), out, float("inf"))
    return np.maximum(out, float(SE_FLOOR))


def _deviance_core(d: np.ndarray, v_data: np.ndarray, m: np.ndarray,
                   se: np.ndarray) -> tuple:
    """The finite-bin deviance core: the summed squared normalized
    residual and the finite-bin mask (the structural drop rule; the
    caller records the drops once from the pooled leg)."""
    d = np.asarray(d, dtype=float)
    v_data = np.asarray(v_data, dtype=float)
    m = np.asarray(m, dtype=float)
    sigma2 = v_data + np.maximum(se, SE_FLOOR) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (d - m) / np.sqrt(np.where(sigma2 > 0.0, sigma2, 1.0))
    mask = (
        np.isfinite(d) & np.isfinite(v_data) & (v_data >= 0.0)
        & np.isfinite(m) & (sigma2 > 0.0) & np.isfinite(residual)
    )
    value = float(np.sum((residual[mask]) ** 2)) if bool(mask.any()) else 0.0
    return value, mask


def sigma_components(d: np.ndarray, v_data: np.ndarray, m: np.ndarray,
                     se: np.ndarray) -> dict:
    """The data and MC contributions of one observable's deviance (the
    card's 'data and MC contributions are reported separately' clause):
    the total form sigma^2 = v_data + se^2, the data-only form
    sigma^2 = v_data, and the MC-only form sigma^2 = se^2 (floored),
    each the summed squared normalized residual over the same finite
    bins as the total."""
    d = np.asarray(d, dtype=float)
    v_data = np.asarray(v_data, dtype=float)
    m = np.asarray(m, dtype=float)
    se = np.maximum(np.asarray(se, dtype=float), SE_FLOOR)
    sigma2 = v_data + se ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (d - m) / np.sqrt(np.where(sigma2 > 0.0, sigma2, 1.0))
        residual_data = (d - m) / np.sqrt(np.where(v_data > 0.0, v_data, 1.0))
        residual_mc = (d - m) / np.sqrt(np.where(se ** 2 > 0.0, se ** 2, 1.0))
    mask = (
        np.isfinite(d) & np.isfinite(v_data) & (v_data >= 0.0)
        & np.isfinite(m) & (sigma2 > 0.0)
        & np.isfinite(residual) & np.isfinite(residual_data)
        & np.isfinite(residual_mc)
    )
    return {
        "deviance": float(np.sum(residual[mask] ** 2)),
        "deviance_data_only": float(np.sum(residual_data[mask] ** 2)),
        "deviance_mc_only": float(np.sum(residual_mc[mask] ** 2)),
        "n_kept": int(mask.sum()),
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
            "the per-class marginal binomial check: the realized draw "
            "count within the recorded MC tolerance of the expected "
            "count n * f at the recorded frozen fractions; a class with "
            "expected count exactly 0 must realize exactly 0; a "
            "divergence is a named deviation"),
        "pass": ok,
    }


def _mixture_reference(per_class: dict, classes: tuple, weights_in: dict,
                       hist_key: str, n_key: str = "n_events") -> tuple:
    """The reference mixture of the sealed R50 per-class histograms at
    the given class weights: the values ``sum_cls w_cls p_cls(bin)``
    and the diagonal variances ``sum_cls w_cls^2 p(1-p)/n_cls`` (the
    R51 reference convention's weights moved from the 0.25-each nominal
    to the caller's weights; the per-class binomial variance, diagonal)."""
    weights = {cls: float(weights_in[cls]) for cls in classes}
    mass = math.fsum(weights.values())
    if mass <= 0.0:
        _refuse(f"the reference mixture weights over {classes!r} carry a "
                f"non-positive mass: {mass!r}")
    weights = {cls: w / mass for cls, w in weights.items()}
    n_bins: int | None = None
    for cls in classes:
        hist = per_class[cls][hist_key]
        n_bins = len(hist) if n_bins is None else n_bins
        if len(hist) != n_bins:
            _refuse(f"the reference {hist_key} histograms disagree on "
                    "their bin count across the classes")
    if n_bins is None:
        _refuse(f"the reference mixture over {classes!r} carries no "
                "class histogram")
    values = np.zeros(n_bins, dtype=float)
    variances = np.zeros(n_bins, dtype=float)
    for cls in classes:
        hist = per_class[cls][hist_key]
        n_cls = float(per_class[cls][n_key])
        w = weights[cls]
        for k in range(n_bins):
            p = hist[k] / n_cls
            values[k] += w * p
            variances[k] += w * w * p * (1.0 - p) / n_cls
    return values, variances, weights


def _family_block(key: str, kind: str, data_source: str, d, v_data,
                  m: np.ndarray, se: np.ndarray, note: str,
                  coords: dict | None = None,
                  model_only: bool = False) -> dict:
    """One validation observable's report block: the model vector, the
    between-replica SE, the deviance with its data and MC contributions
    (None for a model-only observable), the per-bin residual table with
    the beyond-bound flags, and the dropped-bin list (the structural
    drop rule; recorded, never silently dropped)."""
    m = np.asarray(m, dtype=float)
    se = np.asarray(se, dtype=float)
    block: dict = {
        "key": key,
        "kind": kind,
        "data_source": data_source,
        "note": note,
        "m": [float(v) for v in m],
        "se": [float(v) for v in se],
        "n_bins": int(m.size),
    }
    if model_only:
        block["d"] = None
        block["v_data"] = None
        block["deviance"] = None
        block["deviance_data_only"] = None
        block["deviance_mc_only"] = None
        block["n_kept"] = int(m.size)
        block["dropped_bins"] = []
        block["beyond_2x_se_count"] = 0
        block["per_bin"] = [
            {
                "bin": i,
                "d": None,
                "v_data": None,
                "m": float(m[i]),
                "se": float(se[i]),
                "pooled_se": None,
                "difference": None,
                "sigma": None,
                "beyond_2x_se": False,
                "house_form": False,
            }
            for i in range(int(m.size))
        ]
        if coords is not None:
            for i in range(int(m.size)):
                block["per_bin"][i]["xF_mid"] = coords["xF_mid"][i]
                block["per_bin"][i]["pT_mid"] = coords["pT_mid"][i]
        return block
    d = np.asarray(d, dtype=float)
    v_data = np.asarray(v_data, dtype=float)
    components = sigma_components(d, v_data, m, se)
    _value, mask = _deviance_core(d, v_data, m, se)
    sigma2 = v_data + np.maximum(se, SE_FLOOR) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        difference = d - m
        pooled_se = np.sqrt(np.where(sigma2 > 0.0, sigma2, float("nan")))
        sig = difference / pooled_se
    beyond = np.abs(difference) > NAMED_FINDING_SE_FACTOR * pooled_se
    house = np.abs(sig) > SIGMA_TOLERANCE
    dropped_bins = []
    per_bin = []
    for i in range(int(m.size)):
        if not bool(mask[i]):
            reason = "non-finite data or published variance"
            if not (math.isfinite(float(d[i]))
                    and math.isfinite(float(v_data[i]))
                    and float(v_data[i]) >= 0.0):
                reason = "non-finite data or published variance"
            elif not (sigma2[i] > 0.0):
                reason = ("the pooled variance is zero (a structurally "
                          "undefined statistic bin; the R11 convention)")
            dropped_bins.append({
                "statistic": key, "index": int(i), "reason": reason})
            continue
        row = {
            "bin": i,
            "d": float(d[i]),
            "v_data": float(v_data[i]),
            "m": float(m[i]),
            "se": float(se[i]),
            "pooled_se": float(pooled_se[i]),
            "difference": float(difference[i]),
            "sigma": (None if not math.isfinite(float(sig[i]))
                      else float(sig[i])),
            "beyond_2x_se": bool(beyond[i]),
            "house_form": bool(house[i] and math.isfinite(float(sig[i]))),
        }
        if coords is not None:
            row["xF_mid"] = coords["xF_mid"][i]
            row["pT_mid"] = coords["pT_mid"][i]
        per_bin.append(row)
    block["d"] = [float(v) for v in d]
    block["v_data"] = [float(v) for v in v_data]
    block["deviance"] = components["deviance"]
    block["deviance_data_only"] = components["deviance_data_only"]
    block["deviance_mc_only"] = components["deviance_mc_only"]
    block["n_kept"] = components["n_kept"]
    block["dropped_bins"] = dropped_bins
    block["beyond_2x_se_count"] = int(sum(
        1 for row in per_bin if row["beyond_2x_se"]))
    block["house_form_count"] = int(sum(
        1 for row in per_bin if row["house_form"]))
    block["per_bin"] = per_bin
    return block


def _empty_pooled(geometry: dict) -> dict:
    return {
        "events": 0,
        "class_census": {},
        "count_census": {"n1": 0, "n2": 0},
        "b_q_s_census": {},
        "worst_closure_relative": 0.0,
        "worst_mass_shell_relative": 0.0,
        "worst_energy_closure_gev": 0.0,
        "worst_momentum_closure_gev": 0.0,
        "dense": np.zeros(int(geometry["dense_total"]), dtype=float),
        "stack_dense": np.zeros((0, int(geometry["dense_total"]))),
        "lead_xf_pooled": {
            sp: np.zeros(len(LEAD_XF_HISTOGRAM_EDGES) - 1, dtype=float)
            for sp in ("proton", "antiproton")},
        "lead_xf_dropped": {sp: 0 for sp in ("proton", "antiproton")},
        "lead_xf_n": {sp: 0 for sp in ("proton", "antiproton")},
        "stack_lead_xf": {
            sp: np.zeros((0, len(LEAD_XF_HISTOGRAM_EDGES) - 1))
            for sp in ("proton", "antiproton")},
        "gap_pooled": np.zeros(len(GAP_HISTOGRAM_EDGES) - 1, dtype=float),
        "gap_events_with_field": 0,
        "gap_events_without_field": 0,
        "gap_dropped_out_of_range": 0,
        "stack_gap": np.zeros((0, len(GAP_HISTOGRAM_EDGES) - 1)),
        "stack_gap_n": np.zeros(0),
        "nch_pooled": np.zeros(len(NCH_BINS), dtype=float),
        "stack_nch": np.zeros((0, len(NCH_BINS))),
        "spec_pooled": {
            sp: np.zeros(SPECIES_MULT_SUPPORT_HI + 1, dtype=float)
            for sp in SPECIES_MULT_ORDER},
        "stack_spec": {
            sp: np.zeros((0, SPECIES_MULT_SUPPORT_HI + 1))
            for sp in SPECIES_MULT_ORDER},
        "per_replica": [],
    }


def _absorb_replica_record(pooled: dict, record: dict) -> dict:
    """Absorb one replica record's accumulation into the pooled
    aggregates and the per-replica stacks."""
    acc = record["accumulation"]
    flat = np.asarray(acc["rows_flat"], dtype=np.int64)
    weight = np.asarray(acc["rows_weight"], dtype=float)
    dense_r = np.zeros(pooled["dense"].size, dtype=float)
    if flat.size:
        np.add.at(dense_r, flat, weight)
        np.add.at(pooled["dense"], flat, weight)
    n_events = int(record["conservation"]["events"])
    lead = record["accumulation"]["lead_xf"]
    gap = record["accumulation"]["gap"]
    nch = np.asarray(record["accumulation"]["nch"], dtype=float)
    for sp in ("proton", "antiproton"):
        arr = np.asarray(lead[sp], dtype=float)
        pooled["lead_xf_pooled"][sp] += arr
        pooled["lead_xf_dropped"][sp] += int(lead["dropped"][sp])
        pooled["lead_xf_n"][sp] += int(lead[f"n_{sp}s"])
        pooled["stack_lead_xf"][sp] = np.vstack(
            [pooled["stack_lead_xf"][sp], arr[None, :]])
    pooled["gap_pooled"] += np.asarray(gap["hist"], dtype=float)
    pooled["gap_events_with_field"] += int(gap["events_with_field"])
    pooled["gap_events_without_field"] += int(gap["events_without_field"])
    pooled["gap_dropped_out_of_range"] += int(gap["dropped_out_of_range"])
    pooled["stack_gap"] = np.vstack(
        [pooled["stack_gap"], np.asarray(gap["hist"], dtype=float)[None, :]])
    pooled["stack_gap_n"] = np.append(
        pooled["stack_gap_n"], float(gap["events_with_field"]))
    pooled["nch_pooled"] += nch
    pooled["stack_nch"] = np.vstack([pooled["stack_nch"], nch[None, :]])
    for sp in SPECIES_MULT_ORDER:
        arr = np.asarray(acc["spec_mult"][sp], dtype=float)
        pooled["spec_pooled"][sp] += arr
        pooled["stack_spec"][sp] = np.vstack(
            [pooled["stack_spec"][sp], arr[None, :]])
    pooled["events"] += n_events
    for label, count in (record.get("class_census") or {}).items():
        pooled["class_census"][label] = \
            pooled["class_census"].get(label, 0) + int(count)
    for key in ("n1", "n2"):
        pooled["count_census"][key] += int(
            (record.get("count_census") or {}).get(key, 0))
    cons = record["conservation"]
    for bqs, count in (cons.get("b_q_s_census") or {}).items():
        pooled["b_q_s_census"][bqs] = \
            pooled["b_q_s_census"].get(bqs, 0) + int(count)
    pooled["worst_closure_relative"] = max(
        pooled["worst_closure_relative"],
        float(cons["worst_closure_relative"]))
    pooled["worst_mass_shell_relative"] = max(
        pooled["worst_mass_shell_relative"],
        float(cons["worst_mass_shell_relative"]))
    pooled["worst_energy_closure_gev"] = max(
        pooled["worst_energy_closure_gev"],
        float(cons["worst_energy_closure_gev"]))
    pooled["worst_momentum_closure_gev"] = max(
        pooled["worst_momentum_closure_gev"],
        float(cons["worst_momentum_closure_gev"]))
    return {
        "replica": record.get("replica"),
        "seed": record.get("seed"),
        "n_events": n_events,
        "class_census": dict(record.get("class_census") or {}),
        "count_census": dict(record.get("count_census") or {}),
        "event_rolling_sha256": acc.get("event_rolling_sha256"),
        "conservation": dict(cons),
        "rows": int(flat.size),
    }


def _load_blocks(run_dir: Path, *, smoke: bool) -> list:
    """Load the replica records (the production records from the run
    dir; the smoke from the smoke replica file), verifying every
    production record's identity."""
    run_dir = Path(run_dir)
    blocks = []
    names = ([SMOKE_REPLICA_NAME] if smoke
             else [replica_name(r) for r in range(N_REPLICAS)])
    for index, name in enumerate(names):
        path = run_dir / name
        if not path.is_file():
            _refuse(f"the production record is missing: {path}")
        record = _read_json(path)
        if not smoke:
            _verify_replica_identity(record, index, run_dir)
        blocks.append(record)
    return blocks


def _r51_training_baseline() -> dict:
    """The sealed R51 training baseline of the acceptance (iv) sector
    bound: the per-bin mean training deviance of the sealed fit at the
    frozen point (the ``deviance_contribution_fit / n_kept`` of the
    sealed fit report's agreement blocks).  The proton sector's baseline
    is the leading-proton spectrum; the antiproton sector's baseline is
    the B_transport combination (the R51 training set carries no
    single-species antiproton statistic; the combination caveat is
    recorded).  The pi+ / pi- sectors carry no R51 training baseline
    (the row records that fact)."""
    fit_report = _read_json(MONO_ROOT / R51_RUN_DIR / R51_FIT_REPORT_NAME)
    by_key = {agr["key"]: agr for agr in fit_report["agreement"]}
    baseline = {}
    proton = by_key.get("leading_proton_spectrum")
    if proton is not None:
        baseline["proton"] = {
            "statistic": "leading_proton_spectrum",
            "deviance_per_bin": float(
                proton["deviance_contribution_fit"]) / float(proton[
                    "n_kept"]),
        }
    b_transport = by_key.get("b_transport_spectrum")
    if b_transport is not None:
        baseline["antiproton"] = {
            "statistic": "b_transport_spectrum",
            "deviance_per_bin": float(
                b_transport["deviance_contribution_fit"]) / float(
                    b_transport["n_kept"]),
            "caveat": (
                "the R51 training set carries no single-species "
                "antiproton statistic; the baseline is the B_transport "
                "combination (p - p-bar) training deviance, recorded "
                "with that combination caveat"),
        }
    for sp in ("pi_plus", "pi_minus"):
        baseline[sp] = {
            "statistic": None,
            "deviance_per_bin": None,
            "caveat": ("the R51 training set carries no pion statistic "
                       "(the training observables were the proton and "
                       "B_transport published cells and the R50-pinned "
                       "gap and multiplicity references); the sector's "
                       "deviance is recorded for the Gate 5 written "
                       "decision, the 5% bound is not evaluable"),
        }
    return {
        "source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
        "fit_report_sha256": _sha256_file(
            MONO_ROOT / R51_RUN_DIR / R51_FIT_REPORT_NAME),
        "per_sector": baseline,
        "caveat": (
            "the R51 training baseline carries the profiled fit-time "
            "normalization nuisances (the fit's norm nuisance columns), "
            "the frozen evaluation runs un-profiled; a sector exceeding "
            "the 5% bound is a named finding whose stated reason is this "
            "recorded caveat (the acceptance (iv) form)"),
    }


def _sector_degradation_table(spectra_blocks: dict, baseline: dict) -> dict:
    """The acceptance (iv) sector table: each data-backed paper-window
    sector's validation deviance per bin against its sealed R51
    training baseline at the 5% bound (never relaxed); a violating
    sector is a named finding with the recorded caveat as its stated
    reason.  The kaon sectors are not in the R55 pre-registered
    validation register and are recorded as absent."""
    rows = []
    named = []
    for sp in ("pi_plus", "pi_minus", "proton", "antiproton"):
        block = spectra_blocks[sp]
        r55_per_bin = (float(block["deviance"]) / float(block["n_kept"])
                       if block["deviance"] is not None and block["n_kept"]
                       else None)
        base = baseline["per_sector"].get(sp) or {}
        base_per_bin = base.get("deviance_per_bin")
        row = {
            "sector": sp,
            "r55_deviance_per_bin": r55_per_bin,
            "n_bins": int(block["n_kept"]),
            "r51_baseline_statistic": base.get("statistic"),
            "r51_baseline_deviance_per_bin": base_per_bin,
        }
        if r55_per_bin is None or base_per_bin is None:
            row["relative_change"] = None
            row["degrades_beyond_5pct"] = None
            row["reason"] = base.get("caveat")
        else:
            relative = (r55_per_bin - float(base_per_bin)) / float(
                base_per_bin)
            row["relative_change"] = relative
            row["degrades_beyond_5pct"] = bool(
                relative > SECTOR_DEGRADATION_FRACTION)
            if row["degrades_beyond_5pct"]:
                row["reason"] = base.get("caveat") or (
                    "the sector's validation deviance per bin exceeds the "
                    "sealed R51 training baseline by more than the 5% "
                    "bound with no stated physics reason (a Gate 5 "
                    "finding)")
                named.append({
                    "kind": "sector_degradation_beyond_5pct",
                    "sector": sp,
                    "r55_deviance_per_bin": r55_per_bin,
                    "r51_baseline_deviance_per_bin": float(base_per_bin),
                    "relative_change": relative,
                    "bound": SECTOR_DEGRADATION_FRACTION,
                    "reason": row["reason"],
                })
        rows.append(row)
    for sp in ("k_plus", "k_minus"):
        rows.append({
            "sector": sp,
            "r55_deviance_per_bin": None,
            "n_bins": 0,
            "r51_baseline_statistic": None,
            "r51_baseline_deviance_per_bin": None,
            "relative_change": None,
            "degrades_beyond_5pct": None,
            "reason": ("the kaon sectors are not in the R55 "
                       "pre-registered validation register (the card's "
                       "nonstrange species set)"),
        })
    return {"rows": rows, "bound": SECTOR_DEGRADATION_FRACTION,
            "named_findings": named}


def _assemble_named_residuals(families: dict, sector: dict,
                              draw_check: dict) -> list:
    """The named-residual register (the card's evaluation block): the
    per-bin bound exceedances (the 2.0 pooled-SE named-finding bound;
    the |residual| > 3.0 sigma entries carrying the house form), the
    sector findings, and the draw-count divergences."""
    register: list[dict] = []
    for family_name, family in families.items():
        blocks = family if isinstance(family, dict) and \
            all(isinstance(v, dict) and "per_bin" in v
                for v in family.values()) else {"": family}
        for stat_name, block in blocks.items():
            if not isinstance(block, dict) or block.get("d") is None:
                continue
            for row in block["per_bin"]:
                if not row["beyond_2x_se"]:
                    continue
                register.append({
                    "kind": "validation_bin_residual_beyond_2x_pooled_se",
                    "family": family_name,
                    "statistic": block["key"],
                    "bin": row["bin"],
                    "difference": row["difference"],
                    "pooled_se": row["pooled_se"],
                    "sigma": row["sigma"],
                    "house_form": bool(row["house_form"]),
                    "diagnosis": (
                        "the frozen-topology prediction differs from the "
                        "recorded data side by 2.0 pooled standard "
                        "deviations or more (the pre-registered bound; a "
                        "named finding, recorded)"),
                })
    register.extend(sector["named_findings"])
    if not draw_check["pass"]:
        register.append({
            "kind": "topology_sampling_divergence",
            "diagnosis": (
                "the per-class draw counts diverge from the recorded "
                "frozen fractions beyond the recorded MC tolerance (a "
                "named deviation, recorded)"),
            "per_class": draw_check["per_class"],
        })
    return register


def _conditions_block(report_parts: dict) -> dict:
    """The pre-registered acceptance conditions block (the seven
    plan-section-5.5 conditions with their cross-references)."""
    return {
        "1_conservation": {
            "events": report_parts["events"],
            "violations": report_parts["violations"],
            "pass": report_parts["conservation_pass"],
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_energy_closure_gev":
                report_parts["worst_energy_closure_gev"],
        },
        "2_training_cross_reference": {
            "source": f"{R51_RUN_DIR}/{GATE_RECORD_NAME_R51}",
            "fit_report": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            "fit_report_sha256": report_parts["r51_fit_report_sha256"],
            "note": (
                "this run does not re-evaluate the training set; the "
                "training improvement is the sealed R51 fit evidence "
                "(the fit's profiled -2lnL at the optimum against the "
                "initial point, the sealed fit report's "
                "objective_at_optimum and objective_at_initial blocks)"),
        },
        "3_validation": {
            family: {
                stat: {
                    "deviance": block.get("deviance"),
                    "deviance_data_only": block.get("deviance_data_only"),
                    "deviance_mc_only": block.get("deviance_mc_only"),
                    "n_bins": block.get("n_bins"),
                    "n_kept": block.get("n_kept"),
                    "beyond_2x_se_count": block.get("beyond_2x_se_count"),
                    "house_form_count": block.get("house_form_count"),
                }
                for stat, block in (stats.items() if
                                    isinstance(stats, dict) and
                                    all(isinstance(v, dict)
                                        for v in stats.values())
                                    else [("",
                                           stats)])
            }
            for family, stats in report_parts["families"].items()
        },
        "4_sector_degradation_5pct": {
            "rows": report_parts["sector"]["rows"],
            "bound": SECTOR_DEGRADATION_FRACTION,
            "named_findings": report_parts["sector"]["named_findings"],
        },
        "5_replica_stability_cross_reference": {
            "source": f"{R51_RUN_DIR}/{GATE_RECORD_NAME_R51}",
            "note": (
                "the sealed R51 replica-block stability table (all 8 "
                "blocks) is cross-referenced; every fitted parameter's "
                "stability entry failed the 20%-of-pooled-error bound on "
                "the sealed R51 record (sd_projectile_fraction, "
                "sd_target_fraction, dd_fraction, epsilon, b_sd); each "
                "failing parameter is a named finding feeding the Gate 5 "
                "unconstrained clause (recorded in the named-residual "
                "register)"),
            "failing_parameters": report_parts["r51_failing_parameters"],
        },
        "6_information_criterion_cross_reference": {
            "source": f"{R52_RUN_DIR}/{R52_GATE_NAME}",
            "note": (
                "not applicable to a single frozen configuration; the "
                "R52 zero/one comparison is the information-criterion "
                "evidence of the round, cross-referenced from the sealed "
                "R52 gate evidence"),
        },
        "7_color_singlet_audit": {
            "source": f"{R54_RUN_DIR}/{R54_GATE_NAME}",
            "ledger_audit_pass": True,
            "violations": 0,
            "note": (
                "cross-referenced from the sealed R54 audit (zero "
                "violations on the five ledger rules over 1,000,000 "
                "events) and the per-event conservation evidence of this "
                "run; the ledger audit passed on the frozen topology "
                "before the validation was interpreted (the "
                "precondition asserted at pre-registration and "
                "generation)"),
        },
    }


GATE_RECORD_NAME_R51 = "gate5_r51.json"
# ---------------------------------------------------------------------------
# The validation evaluation (the pooled families over the generated
# artifacts)
# ---------------------------------------------------------------------------
RECORDING_CONVENTION_NOTE = (
    "the dense-lattice accumulation per species on the published "
    "evaluation lattice, published rows selected with the pick map "
    "``bin_index_pt * n_x + bin_index_xf``, hole cells dropped (the "
    "inherited state of the index card; the R01 ``_bin_track_sums`` "
    "convention).  The finals accumulate with the invariant measure "
    "weight e_CM / p_T (the R01 accumulation measure; the stored lab "
    "four-momenta inverted through the pinned boost at the extraction "
    "energy SQRT_S = 17.270 GeV, the R51 spectrum-record convention) "
    "and the model value is the self-normalized estimator "
    "m = sums / (pi sqrt(s) N_eff dxF dpT).  The replica records carry "
    "the accumulated register (the sparse dense rows, the leading-baryon "
    "x_F histograms, the gap-size histogram, the charged and "
    "species-separated multiplicity histograms, the class and count "
    "censuses, the conservation aggregate, and the rolling per-event "
    "digest), not the per-event records; the per-event hash evidence is "
    "the rolling digest, and the accumulated register is the evaluation's "
    "consumed surface")

DETERMINISM_NOTE = (
    "a per-replica re-run from a clean interpreter reproduces the report "
    "hashes; validation_report.json and gate5_r55.json carry no "
    "wall-clock fields (their bytes are a pure function of the sealed "
    "artifacts, the pre-registered seeds, and the committed code); the "
    "rolling per-event digest folds every event's class label, event "
    "index, charged multiplicity, gap size, and every final's pdg with "
    "exact hex four-momentum components, and the accumulated register "
    "carries the bit-level identity evidence; the probe replica's clean "
    "re-run must reproduce them")

OBJECTIVE_NOTE = (
    "plan section 11 run R55 (lines 932-934): 'T3 run with topology "
    "fractions frozen. Validate leading baryons, gaps, multiplicities, "
    "and nonstrange spectra.'  The pre-registered instantiation (index "
    "items 7, 8, and 10): the frozen topology fractions are the R51 fit "
    "values, updated by the R52 outcome (the sealed R52 retention "
    "verdicts keep the one-excitation topology: the ND fraction stays "
    "at the R51 fit value) and by the R53 outcome (the sealed R53 "
    "retained flag is false: the nd_multi class stays absent, P(n = 2) "
    "= 0.0); the mass-law parameters are frozen at the R51 fit values; "
    "the blend ramp start stays fixed at B = 2.680194110727528 GeV.  "
    "Statistics T3 (plan section 5.1 line 255): 20 replicas x 250,000 "
    "p+p events = 5,000,000 events.  The validation observables are the "
    "leading baryon x_F spectra of p and p-bar, the gap-size "
    "distribution, the charged and species-separated multiplicity "
    "distributions, and the nonstrange spectra of pi+, pi-, pi0, p, "
    "p-bar against the sealed R01 published tables (the "
    "dense-accumulation/then-pick convention).  No optimizer (index "
    "item 10): the run is a frozen-parameter evaluation.")

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(xi) of the card: (i) conservation "
    "and on-shell: 5,000,000 events, zero B, Q, S violations and zero "
    "four-momentum closure failures at tolerance 1e-09; the worst "
    "closure residual recorded in GeV; (ii) training improvement: "
    "cross-referenced from the sealed R51 fit evidence (this run does "
    "not re-evaluate the training set); (iii) held-out improvement or "
    "preservation: per validation observable, the frozen-topology "
    "predictions are recorded against the sealed R01 published rows "
    "with the pre-registered residual bounds; a failing observable is a "
    "named finding (the 2.0 pooled-SE bound for a named finding; the "
    "|residual| > 3.0 sigma entries carry the house named-residual "
    "form); (iv) no sector degradation: no previously acceptable sector "
    "(the six paper-window species under the card-9 feed-down lineage) "
    "degrades by more than 5% in validation deviance without a stated "
    "physics reason (a violating sector is a named finding with the "
    "reason, or a Gate 5 finding if no reason exists); (v) parameter "
    "stability: cross-referenced from the sealed R51 replica-block "
    "stability table (all 8 blocks); a failing parameter is a named "
    "finding feeding the Gate 5 unconstrained clause; (vi) held-out "
    "likelihood / information criterion: not applicable to a single "
    "frozen configuration; the R52 zero/one comparison is the "
    "information-criterion evidence of the round, cross-referenced from "
    "the sealed R52 gate evidence; (vii) color-singlet audit: "
    "cross-referenced from the sealed R54 audit (zero violations on the "
    "five ledger rules) and the per-event conservation evidence of this "
    "run; the ledger audit must pass on the frozen topology before the "
    "validation is interpreted; (viii) frozen-set fidelity: the frozen "
    "fractions, epsilon, B_SD, and P(n = 2) equal the pre-registered "
    "frozen point exactly (asserted against the sealed R51/R52/R53 "
    "archives); no refit; no_fitting is true, optimizer_invoked is "
    "false; (ix) determinism: the clean-environment re-run reproduces "
    "the per-replica report hashes; (x) seal and clean validation: "
    "exactly one clean-environment validation pass executes before the "
    "seal (the card-19 seal-ordering discipline); the seal is intact "
    "afterwards; (xi) pins and freeze: the driver and test file land "
    "(both trees byte-identical), the freeze re-pin is recorded, and "
    "the full nested suite is green (the land gate's obligation)")

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

DATA_SOURCES_NOTE = {
    "nonstrange_spectra": (
        "the sealed R01 published tables (the R01 archive's residuals, "
        "covariance, and bin-prediction records, cross-checked bit for "
        "bit against the locally loaded SPS tables); pi+, pi-, p, and "
        "p-bar carry the published data deviance; pi0 carries no "
        "published table (the recorded R52 convention) and is "
        "accumulated model-side on the pi- published geometry"),
    "leading_baryon_xf": (
        "the sealed R50 nominal-point reference (the 0.25-each pooled "
        "per-class histograms recomputed read-only from the sealed R50 "
        "slot records, reweighted to the frozen fractions at "
        "evaluation; the sealed substrate carries no published 1D x_F "
        "table -- the recorded R51 reference convention)"),
    "gap_size_distribution": (
        "the sealed R50 nominal-point reference (the gap-carrying "
        "classes' per-class gap-size histograms reweighted to the "
        "frozen fractions and renormalized over the gap-carrying "
        "classes; the sealed substrate carries no published gap table "
        "-- the recorded R51 reference convention)"),
    "multiplicity": (
        "the sealed R50 nominal-point reference (the per-class "
        "charged-multiplicity and species-separated multiplicity "
        "histograms reweighted to the frozen fractions; the sealed "
        "substrate carries no published multiplicity table -- the "
        "recorded R51 reference convention; the charged-multiplicity "
        "family's bins 2..14 are the R51 training statistic's bins and "
        "the recomputed 0.25-each pooled reference cross-checks "
        "bit-exactly against the sealed R51 fit report's reference "
        "data)"),
}


def run_validation_evaluation(run_dir: Path, *, smoke: bool = False,
                              pre: dict | None = None) -> dict:
    """The validation evaluation over the generated artifacts: the
    pooled accumulations, the four observable families with their
    per-bin residual tables, the draw-count check, the conservation
    aggregate, the sector-degradation table, and the named-residual
    register.  Returns the report dict with its digest (no wall-clock
    field; the bytes are a pure function of the sealed artifacts, the
    pre-registered seeds, and the committed code)."""
    run_dir = Path(run_dir)
    blocks = _load_blocks(run_dir, smoke=smoke)
    tables = _load_smoke_tables() if smoke else _load_production_tables()
    geometry = build_geometry(tables)
    pooled = _empty_pooled(geometry)
    for record in blocks:
        _absorb_replica_record(pooled, record)
    events = pooled["events"]
    expected_events = SMOKE_EVENTS if smoke else N_EVENTS_TOTAL
    if events != expected_events:
        _refuse(
            f"the pooled events {events} do not match the pre-registered "
            f"statistics {expected_events}")
    if pooled["b_q_s_census"] and set(pooled["b_q_s_census"]) != {"2,2,0"}:
        _refuse(
            "the conservation census carries a non-(2, 2, 0) event class: "
            f"{sorted(pooled['b_q_s_census'])} (the house conservation "
            "contract)")
    if pooled["worst_closure_relative"] > EVENT_CLOSURE_TOL or \
            pooled["worst_mass_shell_relative"] > EVENT_CLOSURE_TOL:
        _refuse(
            "a stored event's recorded closure exceeds the tolerance "
            f"(worst closure {pooled['worst_closure_relative']!r}, worst "
            f"mass shell {pooled['worst_mass_shell_relative']!r}; the "
            "generation raises live; a stored excess is a record "
            "integrity failure)")
    # -- the sampling block and the draw-count check --
    if smoke:
        sampling_fractions = {
            "elastic": ELASTIC_FRACTION_GENERATION,
            "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
            "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
            "dd": R51_FROZEN_FRACTIONS["dd"],
            "nd": R51_FROZEN_FRACTIONS["nd"],
            "nd_multi": R53_P_N2_DEFAULT,
        }
    else:
        assert pre is not None
        sampling_fractions = dict(
            (pre.get("topology_sampling") or {}).get("fractions") or {})
        if not sampling_fractions:
            _refuse("the pre-registration carries no topology-sampling "
                    "fractions block")
    draw_check = _draw_count_check(
        pooled["class_census"], sampling_fractions, events)
    # -- the spectra family (the dense-accumulation/then-pick convention)
    n_eff = float(events)
    sealed = None
    if not smoke:
        sealed = _load_sealed_r01(tables)
    spectra_blocks: dict[str, dict] = {}
    for sp in SPECTRUM_SPECIES_ORDER:
        offs = int(geometry["dense_offsets"][sp])
        size = int(geometry["dense_sizes"][sp])
        pick = np.asarray(geometry["pick"][sp], dtype=np.int64)
        areas = np.asarray(geometry["areas"][sp], dtype=float)
        dense_slice = pooled["dense"][offs:offs + size]
        picked_pooled = dense_slice[pick]
        m_pooled = picked_pooled / (math.pi * SQRT_S * n_eff * areas)
        rows = []
        for record in blocks:
            flat = np.asarray(record["accumulation"]["rows_flat"],
                              dtype=np.int64)
            weight = np.asarray(record["accumulation"]["rows_weight"],
                                dtype=float)
            dense_r = np.zeros(pooled["dense"].size, dtype=float)
            if flat.size:
                np.add.at(dense_r, flat, weight)
            n_r = float(record["conservation"]["events"])
            m_r = dense_r[offs:offs + size][pick] / (
                math.pi * SQRT_S * n_r * areas)
            rows.append(m_r)
        stack = (np.vstack(rows) if rows
                 else np.zeros((0, int(m_pooled.size))))
        se = se_from(stack)
        if sp == "pi_zero":
            coords = {
                "xF_mid": [float(sealed["pi_minus"]["xF_mid"][i])
                           for i in range(len(geometry["pick"][sp]))],
                "pT_mid": [float(sealed["pi_minus"]["pT_mid"][i])
                           for i in range(len(geometry["pick"][sp]))],
            } if sealed is not None else None
            block = _family_block(
                f"{sp}_spectrum", "spectrum_model_only",
                PI_ZERO_GEOMETRY_NOTE, None, None, m_pooled, se,
                ("the pi0 spectrum is accumulated model-side on the pi- "
                 "published geometry (the recorded R52 pi0 convention) "
                 "and carries no data deviance"),
                coords=coords, model_only=True)
        else:
            if sealed is not None:
                d = np.asarray(sealed[sp]["d"], dtype=float)
                v_data = np.asarray(sealed[sp]["v_data"], dtype=float)
                coords = {
                    "xF_mid": [float(v) for v in sealed[sp]["xF_mid"]],
                    "pT_mid": [float(v) for v in sealed[sp]["pT_mid"]],
                }
                data_source = DATA_SOURCES_NOTE["nonstrange_spectra"]
            else:
                d = np.ones(int(m_pooled.size), dtype=float)
                v_data = np.ones(int(m_pooled.size), dtype=float)
                coords = None
                data_source = ("the synthetic hermetic grid (the smoke's "
                               "unit data side; the production data side "
                               "is the sealed R01 published tables)")
            block = _family_block(
                f"{sp}_spectrum", "spectrum", data_source, d, v_data,
                m_pooled, se,
                ("the species' published-cell spectrum on the R01 "
                 "published rows (the dense-accumulation/then-pick "
                 "convention; the invariant measure weight e_CM / p_T; "
                 "the self-normalized estimator)"),
                coords=coords)
        spectra_blocks[sp] = block
    # -- the reference-side families (the sealed R50 reference reweighted
    #    to the frozen fractions; the smoke's self-consistency side) --
    if not smoke:
        assert pre is not None
        r50_ref = (pre.get("r50_reference") or {}).get("per_class") or {}
        if not r50_ref:
            _refuse("the pre-registration carries no R50-pinned reference")
        frozen_weights = {
            cls: float(sampling_fractions[cls]) for cls in CLASS_ORDER}
    else:
        r50_ref = None
        frozen_weights = {cls: 0.25 for cls in CLASS_ORDER}
    lead_blocks: dict[str, dict] = {}
    for sp in ("proton", "antiproton"):
        per_class = {
            cls: {"hist": r50_ref[cls]["lead_xf"][sp],
                  "n_events": r50_ref[cls]["n_events"]}
            for cls in CLASS_ORDER} if r50_ref is not None else None
        n_sp = float(pooled["lead_xf_n"][sp])
        m_pooled = pooled["lead_xf_pooled"][sp] / n_eff
        rows = [pooled["stack_lead_xf"][sp][j] / float(
            blocks[j]["conservation"]["events"])
            for j in range(pooled["stack_lead_xf"][sp].shape[0])]
        stack = (np.vstack(rows) if rows
                 else np.zeros((0, int(m_pooled.size))))
        se = se_from(stack)
        if per_class is not None:
            values, variances, _weights = _mixture_reference(
                per_class, CLASS_ORDER, frozen_weights, "hist")
            d, v_data = values, variances
            data_source = DATA_SOURCES_NOTE["leading_baryon_xf"]
        else:
            d, v_data = m_pooled.copy(), np.full(int(m_pooled.size),
                                                 float(SE_FLOOR))
            data_source = ("the smoke's self-consistency side (the "
                           "pooled model values; a wiring check; the "
                           "production data side is the sealed R50 "
                           "reference reweighted to the frozen "
                           "fractions)")
        lead_blocks[sp] = _family_block(
            f"leading_baryon_xf_{sp}", "distribution", data_source,
            d, v_data, m_pooled, se,
            (f"the final-state {sp} x_F spectrum (x_F = 2 p_L^CM / "
             "sqrt(s), Feynman x, plan section 7.1) on the R50 pinned "
             "binning; the count-based histogram of every final-state "
             f"{sp} (the leading-baryon structure's observable, plan "
             "line 913); out-of-range values dropped and counted"))
    # -- the gap-size family --
    per_class_gap = {
        cls: {"hist": r50_ref[cls]["gap"]["hist"],
              "n_events": r50_ref[cls]["gap"]["events_with_field"]}
        for cls in GAP_CARRYING_CLASSES} if r50_ref is not None else None
    gap_total = float(pooled["gap_events_with_field"])
    m_gap = pooled["gap_pooled"] / gap_total
    rows = [pooled["stack_gap"][j] / float(pooled["stack_gap_n"][j])
            for j in range(pooled["stack_gap"].shape[0])]
    stack = (np.vstack(rows) if rows else np.zeros((0, int(m_gap.size))))
    se_gap = se_from(stack)
    if per_class_gap is not None:
        d_gap, v_gap, _weights = _mixture_reference(
            per_class_gap, GAP_CARRYING_CLASSES, frozen_weights, "hist",
            n_key="n_events")
        gap_data_source = DATA_SOURCES_NOTE["gap_size_distribution"]
    else:
        d_gap, v_gap = m_gap.copy(), np.full(int(m_gap.size), float(SE_FLOOR))
        gap_data_source = ("the smoke's self-consistency side (the pooled "
                           "model values; a wiring check)")
    gap_block = _family_block(
        "gap_size_distribution", "distribution", gap_data_source,
        d_gap, v_gap, m_gap, se_gap,
        ("the event gap-size distribution over the gap-carrying classes "
         "(sd_projectile, sd_target, dd; the record's gap_size; the "
         "elastic and nd classes carry no gap field -- the R50 "
         "stored-quantity convention), on the R50 pinned binning; the "
         "fraction per bin over the events carrying the field"))
    # -- the multiplicity families --
    per_class_nch = {
        cls: {"hist": r50_ref[cls]["n_ch_hist"],
              "n_events": r50_ref[cls]["n_events"]}
        for cls in CLASS_ORDER} if r50_ref is not None else None
    m_nch = pooled["nch_pooled"] / n_eff
    rows = [pooled["stack_nch"][j] / float(
        blocks[j]["conservation"]["events"])
        for j in range(pooled["stack_nch"].shape[0])]
    stack = (np.vstack(rows) if rows else np.zeros((0, int(m_nch.size))))
    se_nch = se_from(stack)
    if per_class_nch is not None:
        d_nch, v_nch, _weights = _mixture_reference(
            per_class_nch, CLASS_ORDER, frozen_weights, "hist")
        nch_data_source = DATA_SOURCES_NOTE["multiplicity"]
    else:
        d_nch, v_nch = m_nch.copy(), np.full(int(m_nch.size), float(SE_FLOOR))
        nch_data_source = ("the smoke's self-consistency side (the pooled "
                           "model values; a wiring check)")
    nch_block = _family_block(
        "charged_multiplicity_distribution", "distribution",
        nch_data_source, d_nch, v_nch, m_nch, se_nch,
        ("the charged-multiplicity distribution over the support "
         "n_ch = 2..20 (the R51 training statistic's informative subset "
         "is 2..14; the support is extended so a tail event is recorded, "
         "never bled); n_ch is the count of final-state records with "
         "nonzero charge Q (the below-threshold baryonic system carries "
         "Q = 0 and does not count)"))
    species_blocks: dict[str, dict] = {}
    for sp in SPECIES_MULT_ORDER:
        per_class_sp = {
            cls: {"hist": r50_ref[cls]["species_counts"][sp],
                  "n_events": r50_ref[cls]["n_events"]}
            for cls in CLASS_ORDER} if r50_ref is not None else None
        m_sp = pooled["spec_pooled"][sp] / n_eff
        rows = [pooled["stack_spec"][sp][j] / float(
            blocks[j]["conservation"]["events"])
            for j in range(pooled["stack_spec"][sp].shape[0])]
        stack = (np.vstack(rows) if rows
                 else np.zeros((0, int(m_sp.size))))
        se_sp = se_from(stack)
        if per_class_sp is not None:
            d_sp, v_sp, _weights = _mixture_reference(
                per_class_sp, CLASS_ORDER, frozen_weights, "hist")
            data_source = DATA_SOURCES_NOTE["multiplicity"]
        else:
            d_sp, v_sp = m_sp.copy(), np.full(int(m_sp.size), float(SE_FLOOR))
            data_source = ("the smoke's self-consistency side (the pooled "
                           "model values; a wiring check)")
        species_blocks[sp] = _family_block(
            f"species_multiplicity_{sp}", "distribution", data_source,
            d_sp, v_sp, m_sp, se_sp,
            (f"the per-event {sp} multiplicity distribution over the "
             "support 0..20 (the count of final-state records of the "
             "species; the R50 inelastic pools' realized maxima are the "
             "recorded calibration)"))
    # -- the sector table and the named-residual register (production) --
    baseline = None
    sector = {"rows": [], "named_findings": [], "bound":
              SECTOR_DEGRADATION_FRACTION}
    named_residuals: list[dict] = []
    r51_failing_parameters: list[str] = []
    r51_fit_report_sha256 = None
    if not smoke:
        baseline = _r51_training_baseline()
        sector = _sector_degradation_table(spectra_blocks, baseline)
        families = {
            "nonstrange_spectra": spectra_blocks,
            "leading_baryon_xf": lead_blocks,
            "gap_size_distribution": gap_block,
            "multiplicity_charged": nch_block,
            "multiplicity_species": species_blocks,
        }
        named_residuals = _assemble_named_residuals(
            families, sector, draw_check)
        # the condition (v) cross-reference: the sealed R51 replica-block
        # stability failures feed the Gate 5 unconstrained clause
        r51_gate = _read_json(MONO_ROOT / R51_RUN_DIR / GATE_RECORD_NAME_R51)
        stability = r51_gate.get("replica_block_stability") or {}
        for parameter, entry in sorted(stability.items()):
            if isinstance(entry, dict) and entry.get("pass") is False:
                r51_failing_parameters.append(parameter)
                named_residuals.append({
                    "kind": "r51_replica_block_stability_failure",
                    "parameter": parameter,
                    "diagnosis": (
                        "the sealed R51 replica-block stability table "
                        "carries a failing entry for this parameter (the "
                        "acceptance (v) cross-reference); the parameter "
                        "is weakly constrained by the training "
                        "observables and the Gate 5 unconstrained clause "
                        "consumes it directly"),
                })
        r51_fit_report_sha256 = _sha256_file(
            MONO_ROOT / R51_RUN_DIR / R51_FIT_REPORT_NAME)
        named_residuals.append({
            "kind": "r51_inherited_non_convergence",
            "diagnosis": (
                "the frozen point is the R51 optimum, which did not "
                "converge (the R51 named residuals "
                "optimizer_non_convergence, "
                "replica_block_minima_disagree, block_curvature_failure, "
                "weight_degeneracy); this validation inherits that state "
                "as a model-uncertainty caveat on the frozen parameters"),
        })
    conservation_pass = (
        pooled["worst_closure_relative"] <= EVENT_CLOSURE_TOL
        and pooled["worst_mass_shell_relative"] <= EVENT_CLOSURE_TOL
        and set(pooled["b_q_s_census"] or {"2,2,0": 0}) == {"2,2,0"})
    report_parts = {
        "events": events,
        "violations": 0,
        "conservation_pass": bool(conservation_pass),
        "worst_energy_closure_gev": pooled["worst_energy_closure_gev"],
        "families": {
            "nonstrange_spectra": spectra_blocks,
            "leading_baryon_xf": lead_blocks,
            "gap_size_distribution": gap_block,
            "multiplicity_charged": nch_block,
            "multiplicity_species": species_blocks,
        },
        "sector": sector,
        "r51_failing_parameters": r51_failing_parameters,
        "r51_fit_report_sha256": r51_fit_report_sha256,
    }
    frozen_point = {
        "fractions": dict(sampling_fractions),
        "epsilon": FROZEN_FIT["epsilon"],
        "b_sd": FROZEN_FIT["b_sd"],
        "p_n2": (float(pre["topology_sampling"]["nd_multi"]["p_n2"])
                 if not smoke else R53_P_N2_DEFAULT),
        "nd_multi": ("absent (the sealed R53 outcome retains nothing; "
                     "P(n = 2) = 0.0 the unretained default)"),
        "blend_ramp_start_gev": BLEND_RAMP_START_GEV,
        "sources": [
            f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            f"{R52_RUN_DIR}/{R52_GATE_NAME}",
            f"{R53_RUN_DIR}/{R53_GATE_NAME}",
        ],
        "refit": False,
    }
    report = {
        "schema": "na49_mixing.staged_r55_frozen_topology_validation."
                  "report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "objective": OBJECTIVE_NOTE,
        "statistics": {
            "tier": ("exactly 5,000,000 events = 20 replicas x 250,000 "
                     "p+p events (T3, plan section 5.1 line 255)" if not
                     smoke else
                     "the smoke statistics: one mixed-draw replica x "
                     "2,000 events at the smoke seed 20312925 plus the "
                     "per-class pinned dispatch exercise"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
            "smoke_events": SMOKE_EVENTS if smoke else None,
        },
        "frozen_point": frozen_point,
        "topology_sampling": {
            "source": ("the round's frozen point (the sealed R51 fitted "
                       "fractions as updated by the sealed R52 outcome "
                       "and the sealed R53 outcome; asserted at "
                       "pre-registration and re-asserted at generation)"),
            "archive_pins": [R51_RUN_DIR, R52_RUN_DIR, R53_RUN_DIR],
            "fractions": dict(sampling_fractions),
            "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
            "smoke": smoke,
        },
        "draw_count_check": draw_check,
        "conservation": {
            "events_checked": events,
            "violations": 0,
            "pass": bool(conservation_pass),
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_closure_relative": pooled["worst_closure_relative"],
            "worst_mass_shell_relative":
                pooled["worst_mass_shell_relative"],
            "worst_energy_closure_gev": pooled["worst_energy_closure_gev"],
            "worst_momentum_closure_gev":
                pooled["worst_momentum_closure_gev"],
            "b_q_s_census": dict(pooled["b_q_s_census"]),
        },
        "validation_observables": {
            "leading_baryon_xf": lead_blocks,
            "gap_size_distribution": gap_block,
            "multiplicity_charged": nch_block,
            "multiplicity_species": species_blocks,
            "nonstrange_spectra": spectra_blocks,
            "data_sources": dict(DATA_SOURCES_NOTE),
            "register_note": R51_TRAINING_NOTE,
            "pi_zero_note": PI_ZERO_GEOMETRY_NOTE,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "note": RECORDING_CONVENTION_NOTE,
            "out_of_window_drops": {
                "lead_xf": {sp: int(pooled["lead_xf_dropped"][sp])
                            for sp in ("proton", "antiproton")},
                "lead_xf_n": {sp: int(pooled["lead_xf_n"][sp])
                              for sp in ("proton", "antiproton")},
                "gap_dropped_out_of_range":
                    int(pooled["gap_dropped_out_of_range"]),
                "gap_events_without_field":
                    int(pooled["gap_events_without_field"]),
            },
        },
        "sector_degradation": (None if smoke else sector),
        "conditions": ("smoke (the acceptance conditions are "
                       "production-mode artifacts)" if smoke else
                       _conditions_block(report_parts)),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": named_residuals,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "no_optimizer_note": (
            "index item 10: R55 draws no optimizer and fits no parameter "
            "(the frozen-parameter validation); optimizer_invoked false, "
            "optimizer_uses 0; no_fitting true"),
        "no_separation_claim": (
            "no Round 5 record claims a separation of exchange recoil "
            "from decay transverse dynamics (the R46 outcome, "
            "separation_claimed false); the exchange slopes act on the "
            "exchange kinematics only (plan section 24, line 1866)"),
        "determinism": DETERMINISM_NOTE,
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "acceptance": ACCEPTANCE_NOTE,
        "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        "per_replica": [
            {
                "replica": record.get("replica"),
                "seed": record.get("seed"),
                "n_events": int(record["conservation"]["events"]),
                "class_census": dict(record.get("class_census") or {}),
                "event_rolling_sha256":
                    record["accumulation"]["event_rolling_sha256"],
            }
            for record in blocks
        ],
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
    """--gate: write gate5_r55.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the validation
    report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    cons = report["conservation"]
    sampling = report["topology_sampling"]
    frozen_point = report["frozen_point"]
    families = report["validation_observables"]
    conditions = report["conditions"]
    gate = {
        "schema": "na49_mixing.staged_r55_frozen_topology_validation."
                  "gate/1",
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R55",
        "card": CARD,
        "clause": ("frozen fractions reproduce topology-sensitive "
                   "observables"),
        "frozen_point": {
            "fractions": dict(frozen_point["fractions"]),
            "epsilon": frozen_point["epsilon"],
            "b_sd": frozen_point["b_sd"],
            "p_n2": frozen_point["p_n2"],
            "nd_multi": frozen_point["nd_multi"],
            "blend_ramp_start_gev": frozen_point["blend_ramp_start_gev"],
            "source": list(frozen_point["sources"]),
            "refit": False,
        },
        "events_checked": int(cons["events_checked"]),
        "conservation": {
            "violations": int(cons["violations"]),
            "pass": bool(cons["pass"]),
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_energy_closure_gev": cons["worst_energy_closure_gev"],
        },
        "validation_observables": {
            "leading_baryon_xf": ["proton", "antiproton"],
            "gap_size_distribution": True,
            "multiplicity": "charged and species-separated",
            "nonstrange_spectra": list(SPECTRUM_SPECIES_ORDER),
            "data_source": "sealed R01 published tables",
            "per_family_blocks": {
                name: {
                    stat: {
                        "deviance": block.get("deviance"),
                        "deviance_data_only":
                            block.get("deviance_data_only"),
                        "deviance_mc_only": block.get("deviance_mc_only"),
                        "n_bins": block.get("n_bins"),
                        "n_kept": block.get("n_kept"),
                        "beyond_2x_se_count":
                            block.get("beyond_2x_se_count"),
                        "house_form_count":
                            block.get("house_form_count"),
                    }
                    for stat, block in (stats.items() if
                                        isinstance(stats, dict) and
                                        all(isinstance(v, dict) for
                                            v in stats.values())
                                        else [("", stats)])
                }
                for name, stats in (
                    ("nonstrange_spectra", families["nonstrange_spectra"]),
                    ("leading_baryon_xf", families["leading_baryon_xf"]),
                    ("gap_size_distribution",
                     families["gap_size_distribution"]),
                    ("multiplicity_charged",
                     families["multiplicity_charged"]),
                    ("multiplicity_species",
                     families["multiplicity_species"]),
                )
            },
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "conditions": conditions,
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register":
            list(report["named_residual_register"]),
        "no_fitting": True,
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
    print(f"[r55] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


# ---------------------------------------------------------------------------
# The clean-environment validation (exactly one pass, before the seal)
# ---------------------------------------------------------------------------
def run_clean_validate(run_dir: Path) -> dict:
    """--clean-validate: re-run the validation evaluation from a clean
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
    for name in (REPORT_NAME, GATE_RECORD_NAME):
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    pre = load_pre_registration(run_dir)
    verify_frozen_substrate()
    assert_fractions_against_r51()
    r52_outcome = read_r52_outcome()
    assert_r52_outcome_against_preregistration(r52_outcome, pre)
    r53_outcome = read_r53_outcome()
    assert_outcome_against_preregistration(r53_outcome, pre)
    assert_r54_audit_precondition()
    sealed_report = _read_json(run_dir / REPORT_NAME)
    rerun_report = run_validation_evaluation(run_dir, smoke=False, pre=pre)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the probe replica re-run in memory (the determinism evidence of
    # acceptance (ix)): the pre-registered probe replica (replica 0)
    # regenerates from its pre-registered seed and must reproduce its
    # sealed accumulation and aggregates exactly
    probe = PROBE_REPLICA
    catalog = _load_catalog()
    fams = _load_families()
    sampling = sampling_from_outcomes(r52_outcome, r53_outcome)
    geometry = build_geometry(_load_production_tables())
    generation = generate_replica_events(
        replica_seed(probe), EVENTS_PER_REPLICA, EVENT_INDEX_BASE,
        catalog, fams, sampling, geometry)
    sealed_probe = _read_json(run_dir / replica_name(probe))
    live_acc = generation["accumulation"]
    sealed_acc = sealed_probe["accumulation"]
    acc_matches = (
        live_acc["rows_flat"] == sealed_acc["rows_flat"]
        and live_acc["rows_weight"] == sealed_acc["rows_weight"]
        and live_acc["lead_xf"] == sealed_acc["lead_xf"]
        and live_acc["gap"] == sealed_acc["gap"]
        and live_acc["nch"] == sealed_acc["nch"]
        and live_acc["spec_mult"] == sealed_acc["spec_mult"]
        and live_acc["event_rolling_sha256"] ==
        sealed_acc["event_rolling_sha256"])
    probe_matches = bool(
        acc_matches
        and generation["class_census"] == sealed_probe.get("class_census")
        and generation["count_census"] == sealed_probe.get("count_census")
        and generation["conservation"] == sealed_probe.get("conservation"))
    payload = {
        "schema": ("na49_mixing.staged_r55_frozen_topology_validation."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r55_frozen_topology_validation.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            (f"python helpers/benchmarking/"
             f"staged_r55_frozen_topology_validation.py "
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
            "accumulation_match": bool(acc_matches),
            "census_matches": bool(
                generation["class_census"] == sealed_probe.get(
                    "class_census")),
            "count_census_matches": bool(
                generation["count_census"] == sealed_probe.get(
                    "count_census")),
            "conservation_matches": bool(
                generation["conservation"] == sealed_probe.get(
                    "conservation")),
            "matches_sealed_record": probe_matches,
        },
        "verified_utc": _now_utc(),
    }
    _write_json(clean_path, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter validation evaluation does not "
            "reproduce the report digest (sealed "
            f"{sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(
        f"[r55] clean validation written: {clean_path} (report hash "
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
        "schema": ("na49_mixing.staged_r55_frozen_topology_validation."
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
        PREREGISTRATION_NAME, PROVENANCE_NAME, REPORT_NAME,
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
    print(f"[r55] sealed {run_dir} ({n_entries} entries)", flush=True)
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
# evidence that the accumulation covers every landed class)
# ---------------------------------------------------------------------------
EXERCISE_SCHEMA = ("na49_mixing.staged_r55_frozen_topology_validation."
                   "class_exercise/1")


def run_smoke_class_exercise(catalog, fams, sampling, geometry) -> dict:
    """The per-class pinned dispatch: every topology class (elastic,
    sd_projectile, sd_target, dd, nd, nd_multi) through the same
    generation and accumulation machinery on the dedicated smoke
    class-exercise partition (the hermetic evidence that the validation
    register covers every landed class).  The nd_multi class runs at the
    pinned smoke exercise P(n = 2) (a smoke fixture value, the R53 drawn
    value, recorded as such; never a production value).  The elastic
    dispatch cross-checks the pinned incoming pair against the elastic
    module's own incoming record and runs the zero-transfer code
    check."""
    params = frozen_draw_params(SMOKE_EXERCISE_P_N2)
    per_class: dict[str, dict] = {}
    slot = 0
    for cls in SMOKE_EXERCISE_CLASSES:
        acc = _ReplicaAccumulator(geometry)
        count_census = {"n1": 0, "n2": 0}
        violations = 0
        worst_closure = 0.0
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
                    raise ValidationRefusal(
                        "the elastic module's incoming pair drifted off "
                        f"the pinned {INCOMING_QN}: {incoming!r}")
            obs = extract_observables(event, SQRT_S)
            conservation = r51.event_conservation(event)
            worst_closure = max(worst_closure,
                                conservation["closure_worst"])
            accumulate_live_event(acc, event, obs,
                                  SMOKE_EXERCISE_INDEX_BASE + slot)
            if isinstance(event, NDMultiEvent):
                if event.count == 2:
                    count_census["n2"] += 1
                else:
                    count_census["n1"] += 1
            slot += 1
        per_class[cls] = {
            "events": SMOKE_EXERCISE_EVENTS_PER_CLASS,
            "violations": violations,
            "pass": violations == 0,
            "count_census": dict(count_census),
            "event_rolling_sha256": acc.event_rolling_digest(),
            "rows": len(acc.rows_flat),
            "conservation": {
                "worst_closure_relative": worst_closure,
                "pass": worst_closure <= EVENT_CLOSURE_TOL,
            },
        }
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
    }


# ---------------------------------------------------------------------------
# The hermetic smoke mode (driver validation; never sealed)
# ---------------------------------------------------------------------------
def run_smoke(run_dir: Path) -> dict:
    """The hermetic end-to-end driver validation: a smoke
    pre-registration, one mixed-draw replica x 2,000 events at the smoke
    seed 20312925 on the event-index partition 1_600_000_000 + i (i <
    2,000) at the frozen fractions (the recorded sampling surface at the
    R53 unretained default: the nd dispatch at P(n = 2) = 0.0), the
    per-class pinned dispatch exercise (every topology class through the
    same accumulation machinery), and the smoke validation evaluation on
    the synthetic hermetic grid, into an explicit --run-dir (the default
    production run dir is refused).  No sealed archive is read and no
    SPS table is read (the smoke asserts the frozen-value constants
    against no sealed archive; the production assertion runs in
    --pre-registration and --evaluate).  Two identical smoke invocations
    produce byte-identical artifacts (no wall-clock and no git state in
    the smoke artifacts)."""
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
        "schema": ("na49_mixing.staged_r55_frozen_topology_validation."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "validation_surface": dict(_FIT_SURFACE_STATUS),
        "assertion": ASSERTION_NOTE,
        "statistics": {
            "note": (
                "the smoke statistics: one mixed-draw replica x 2,000 "
                "events at the smoke seed 20312925 plus the per-class "
                "pinned dispatch exercise (six classes x 8 events on the "
                "dedicated smoke class-exercise partition); the smoke "
                "exercises pre-registration, frozen-set assertion "
                "(constants only), generation, accumulation, and the "
                "validation evaluation end to end; the production "
                "statistics are the 20 pre-registered replicas of "
                "250,000 events"),
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
                "raw replica seed = 20260925 + 40000 + 64 + k = 20300989 "
                f"+ k per replica k < {N_REPLICAS}"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "topology_sampling": {
            "source": (
                "the round's frozen point (the driver's pinned constants "
                "in smoke mode; the production assertion against the "
                "sealed R51/R52/R53 archives runs in --pre-registration "
                "and --evaluate)"),
            "fractions": {
                "elastic": ELASTIC_FRACTION_GENERATION,
                "sd_projectile": R51_FROZEN_FRACTIONS["sd_projectile"],
                "sd_target": R51_FROZEN_FRACTIONS["sd_target"],
                "dd": R51_FROZEN_FRACTIONS["dd"],
                "nd": R51_FROZEN_FRACTIONS["nd"],
                "nd_multi": R53_P_N2_DEFAULT,
            },
            "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
            "r52_outcome": {
                "note": ("the smoke pre-registration records the pinned "
                         "R52 constants against no sealed archive; the "
                         "production read runs in --pre-registration"),
                "zero_model_retained": False,
                "frozen_fractions_after_r52": dict(R51_FROZEN_FRACTIONS),
            },
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
        "validation_observables": {
            "leading_baryon_xf": ["proton", "antiproton"],
            "gap_size_distribution": True,
            "multiplicity": "charged and species-separated",
            "nonstrange_spectra": list(SPECTRUM_SPECIES_ORDER),
            "pinned_binnings": {
                "gap_edges_count": len(GAP_HISTOGRAM_EDGES),
                "lead_xf_edges_count": len(LEAD_XF_HISTOGRAM_EDGES),
                "nch_support": [NCH_SUPPORT_LO, NCH_SUPPORT_HI],
                "species_mult_support_hi": SPECIES_MULT_SUPPORT_HI,
            },
            "pi_zero_note": PI_ZERO_GEOMETRY_NOTE,
            "register_note": R51_TRAINING_NOTE,
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
            "index item 10: R55 draws no optimizer and fits no parameter "
            "(the frozen-parameter validation); the smoke exercises the "
            "accumulation machinery only"),
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
    sampling = sampling_from_outcomes(
        {"frozen_fractions_after_r52": dict(R51_FROZEN_FRACTIONS)},
        {"status": "R53-PENDING", "pending": True, "retained": False,
         "p_n2": R53_P_N2_DEFAULT})
    geometry = build_geometry(_load_smoke_tables())
    generation = generate_replica_events(
        SMOKE_SEED, SMOKE_EVENTS, SMOKE_EVENT_INDEX_BASE, catalog, fams,
        sampling, geometry)
    replica_record = build_replica_record(
        "smoke", generation, run_dir, sampling, {}, mode="smoke")
    _write_json(run_dir / SMOKE_REPLICA_NAME, replica_record)
    print(
        f"[r55] smoke generation complete: {SMOKE_EVENTS} events, census "
        f"{generation['class_census']}, worst closure "
        f"{generation['conservation']['worst_closure_relative']:.3e}",
        flush=True,
    )
    exercise = run_smoke_class_exercise(catalog, fams, sampling, geometry)
    _write_json(run_dir / SMOKE_EXERCISE_NAME, exercise)
    print(
        f"[r55] smoke class exercise complete: "
        f"{sum(entry['events'] for entry in exercise['classes'].values())} "
        "events across "
        f"{len(exercise['classes'])} classes, all accumulations pass",
        flush=True,
    )
    report = run_validation_evaluation(run_dir, smoke=True)
    _write_json(run_dir / SMOKE_REPORT_NAME, report)
    print(
        f"[r55] smoke evaluation complete: conservation violations "
        f"{report['conservation']['violations']}, worst closure "
        f"{report['conservation']['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {
        "events": SMOKE_EVENTS,
        "census": generation["class_census"],
        "count_census": generation["count_census"],
        "exercise_classes": {
            cls: entry["events"]
            for cls, entry in exercise["classes"].items()},
        "worst_closure_relative":
            report["conservation"]["worst_closure_relative"],
    }
# ---------------------------------------------------------------------------
# The pre-registration (written once, before any generation)
# ---------------------------------------------------------------------------
ASSERTION_NOTE = (
    "the frozen-set fidelity (the acceptance (viii)): the frozen "
    "fractions, epsilon, B_SD, and P(n = 2) equal the pre-registered "
    "frozen point exactly, asserted against the sealed R51, R52, and R53 "
    "archives before any generation (the R51 fit report's fitted values "
    "byte-exactly; the R52 gate's retention verdicts through the card's "
    "two-case update rule; the R53 gate's retained boolean false with "
    "P(n = 2) = 0.0); no refit (plan line 934; index item 10); "
    "no_fitting is true, optimizer_invoked is false.  The R54 audit "
    "precondition: the sealed R54 ledger audit passes on the frozen "
    "topology (zero violations on card 2's five ledger rules over "
    "1,000,000 events; the audit's recorded topology sampling fractions "
    "equal the frozen fractions block) before any generation and before "
    "the validation is interpreted.  The conservation contract runs on "
    "every event of the 5,000,000 (a violation raises and stops the "
    "run).  The substream collision rule: a colliding raw (seed, "
    "event-index) substream stops the run with a report (never a silent "
    "re-block).")


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
        r52_outcome = read_r52_outcome()
        r53_outcome = read_r53_outcome()
        r54_precondition = assert_r54_audit_precondition()
        # the pinned R50 reference (recomputed read-only from the sealed
        # R50 slots; cross-checked against the sealed R51 fit report)
        geometry = build_geometry(_load_production_tables())
        r50_reference = r50_reference_payload(geometry)
        sampling = sampling_from_outcomes(r52_outcome, r53_outcome)
    else:
        sealed_inputs = None
        fraction_assertion = None
        nominal_reference = None
        r52_outcome = {
            "zero_model_retained": False,
            "frozen_fractions_after_r52": dict(R51_FROZEN_FRACTIONS),
            "note": ("the smoke pre-registration records the pinned R52 "
                     "constants against no sealed archive; the "
                     "production read runs in --pre-registration"),
        }
        r53_outcome = {
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
        r54_precondition = None
        r50_reference = None
        sampling = sampling_from_outcomes(r52_outcome, r53_outcome)
    fractions = frozen_fractions_block(sampling)
    payload = {
        "schema": ("na49_mixing.staged_r55_frozen_topology_validation."
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
        "validation_surface": dict(_FIT_SURFACE_STATUS),
        "assertion": ASSERTION_NOTE,
        "statistics": {
            "tier": (
                "exactly 5,000,000 events = 20 replicas x 250,000 p+p "
                "events (T3, plan section 5.1 line 255; the card's "
                "pre-registered instantiation)"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 64 + k = 20300989 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300989-20301008 is disjoint from every "
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
                    "20312925 (the round-5 shared smoke block, disjoint "
                    "from every production replica and from every "
                    "previously used production block)"),
                "events": SMOKE_EVENTS,
                "event_index_rule": SMOKE_EVENT_INDEX_RULE,
                "class_exercise_index_rule": SMOKE_EXERCISE_INDEX_RULE,
            },
            "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        },
        "topology_sampling": {
            "source": (
                "the round's frozen point (the sealed "
                f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME} fitted values, "
                "updated by the sealed R52 outcome and the sealed R53 "
                "outcome; asserted at pre-registration and re-asserted "
                "at generation)" if not smoke else
                "the driver's pinned frozen-point constants (the "
                "production assertion against the sealed archives runs "
                "in --pre-registration and --evaluate)"),
            "fractions": fractions,
            "assertion": (
                "the fractions are asserted byte-exactly against the "
                "sealed R51 fit report and the sealed R52/R53 gate "
                "records at pre-registration and re-asserted at "
                "generation (the driver re-reads the sealed archives "
                "before every generate/evaluate pass); a divergence is "
                "a named deviation; a silent switch never happens"),
            "elastic_fraction_note": ELASTIC_INCLUSION_NOTE,
            "elastic_included": True,
            "r52_outcome": (None if smoke else {
                "status": r52_outcome["status"],
                "zero_model_retained":
                    bool(r52_outcome["zero_model_retained"]),
                "retention_verdicts": dict(
                    r52_outcome["retention_verdicts"]),
                "zero_model_renormalized_fractions": dict(
                    r52_outcome["zero_model_renormalized_fractions"]),
                "zero_model_triggered":
                    bool(r52_outcome["zero_model_triggered"]),
                "trigger_sigma": r52_outcome["trigger_sigma"],
                "frozen_fractions_after_r52": dict(
                    r52_outcome["frozen_fractions_after_r52"]),
                "update_note": r52_outcome["update_note"],
                "source": r52_outcome["source"],
                "gate_file_sha256": r52_outcome["gate_file_sha256"],
            }) or {
                "note": r52_outcome["note"],
                "zero_model_retained": False,
                "frozen_fractions_after_r52": dict(
                    R51_FROZEN_FRACTIONS),
            },
            "nd_multi": {
                "status": (None if smoke else r53_outcome["status"]),
                "pending": bool(r53_outcome["pending"]),
                "retained": bool(r53_outcome["retained"]),
                "p_n2": float(r53_outcome["p_n2"]),
                "source": r53_outcome.get("source"),
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
                "draw_topology at the recorded frozen fractions; each "
                "inelastic class is returned with exactly its recorded "
                "fraction and the elastic class occupies the lower edge "
                "at its recorded fraction (exactly 0.0 at the sealed "
                "R51 point)"),
            "mc_tolerance": {
                "sigma": MC_TOLERANCE_SIGMA,
                "rule": (
                    "the per-class marginal binomial check: the realized "
                    "draw count within 5.0 binomial standard deviations "
                    "of the expected count n * f; a class with expected "
                    "count exactly 0 must realize exactly 0; a "
                    "divergence is a named deviation"),
            },
        },
        "r54_audit_precondition": (None if smoke else r54_precondition),
        "frozen_fractions_assertion": fraction_assertion,
        "nominal_reference": nominal_reference,
        "r53_outcome_resolution": (None if smoke else {
            "status": r53_outcome["status"],
            "pending": bool(r53_outcome["pending"]),
            "retained": bool(r53_outcome["retained"]),
            "p_n2": float(r53_outcome["p_n2"]),
            "source": r53_outcome.get("source"),
            "gate_file_sha256": r53_outcome.get("gate_file_sha256"),
            "note": r53_outcome.get("note"),
        }),
        "validation_observables": {
            "leading_baryon_xf": ["proton", "antiproton"],
            "gap_size_distribution": True,
            "multiplicity": "charged and species-separated",
            "nonstrange_spectra": list(SPECTRUM_SPECIES_ORDER),
            "data_sources": dict(DATA_SOURCES_NOTE),
            "pinned_binnings": {
                "gap_edges": list(GAP_HISTOGRAM_EDGES),
                "lead_xf_edges": list(LEAD_XF_HISTOGRAM_EDGES),
                "nch_support": [NCH_SUPPORT_LO, NCH_SUPPORT_HI],
                "nch_r51_informative_subset": [2, 14],
                "species_mult_support_hi": SPECIES_MULT_SUPPORT_HI,
                "species_mult_order": list(SPECIES_MULT_ORDER),
            },
            "pi_zero_note": PI_ZERO_GEOMETRY_NOTE,
            "register_note": R51_TRAINING_NOTE,
            "evaluation": (
                "per-bin deviance on the pre-registered observables with "
                "the sigma^2 = v_ref/v_data + se^2 form (the "
                "published/reference variance plus the between-replica "
                "MC variance, std ddof=1 over the 20 replica blocks "
                "divided by sqrt(20), floored at the engine's "
                "DETERMINISTIC_SE_FLOOR); the published binning is "
                "preserved; data and MC contributions are reported "
                "separately; the named-residual register is assembled on "
                "any bound exceedance (the 2.0 pooled-SE bound of the "
                "per-bin difference for a named finding; the |residual| "
                f"> {SIGMA_TOLERANCE!r} sigma entries carry the house "
                "named-residual form); the sector-degradation bound is "
                f"{SECTOR_DEGRADATION_FRACTION!r} against the sealed R51 "
                "training baseline (the recorded nuisance caveat)"),
        },
        "r50_reference": r50_reference,
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
        "recording_convention": {
            "note": RECORDING_CONVENTION_NOTE,
            "generation": GENERATION_GATEWAY_NOTE,
        },
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r55_frozen_topology_validation."
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
            "r51_fit_report_digest":
                fraction_assertion["fit_report_sha256"],
            "r52_outcome": {
                "source": r52_outcome["source"],
                "zero_model_retained":
                    bool(r52_outcome["zero_model_retained"]),
                "gate_file_sha256": r52_outcome["gate_file_sha256"],
            },
            "r53_outcome": {
                "status": r53_outcome["status"],
                "pending": bool(r53_outcome["pending"]),
                "retained": bool(r53_outcome["retained"]),
                "p_n2": float(r53_outcome["p_n2"]),
                "source": r53_outcome.get("source"),
            },
            "r54_precondition": {
                "source": r54_precondition["source"],
                "ledger_audit_pass": True,
                "violations": 0,
                "gate_file_sha256": r54_precondition["gate_file_sha256"],
            },
            "r01_table_digests": {
                name: _sha256_file(MONO_ROOT / R01_RUN_DIR / name)
                for name in (R01_PREDICTIONS_NAME, R01_COVARIANCE_NAME,
                             R01_RESIDUALS_NAME)
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
        print(f"[r55] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r55] pre-registration written: {pre_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R55 driver: the frozen topology validation "
                    "(TASK-20260925-r5-run-r55-frozen-topology-validation)")
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
        help="the validation evaluation (validation_report.json), then "
             "stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r55.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the validation evaluation from a clean interpreter "
             "and the probe replica, then stop (exactly one pass, before "
             "the seal)")
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
        print(f"[r55] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, replicas=args.replica)
        print(f"[r55] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        pre = load_pre_registration(run_dir)
        report = run_validation_evaluation(run_dir, smoke=False, pre=pre)
        _write_json(run_dir / REPORT_NAME, report)
        print(
            f"[r55] evaluation written: {run_dir / REPORT_NAME} "
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