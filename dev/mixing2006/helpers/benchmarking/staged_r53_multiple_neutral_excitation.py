#!/usr/bin/env python3
"""R53 driver: the multiple-neutral-excitation test (Round 5, card 13).

Task ``TASK-20260925-r5-run-r53-multiple-neutral-excitation`` (plan
section 11 run R53, lines 924-926; plan section 23 lines 1825-1827,
"Round 5 -- exchange topology").  The card executes run R53, the
conditional single-parameter fit: if and only if the named R52 residual
on the multiplicity tails reaches 2.0 pooled standard deviations, fit
one additional parameter, the probability P(n >= 2), in the truncated
neutral-excitation number distribution (the count n in {1, 2} of card
8, truncated at the kinematic maximum multiplicity), bounded at
[0, 0.5], on T2 statistics; otherwise record NOT_TRIGGERED and no
generation happens (the multiple-excitation class of card 8 stays
implemented but unretained, Gate 5 clause 2; plan line 938).

The trigger (index item 5; sealed before this run)
--------------------------------------------------
The trigger source is the sealed R52 archive
``dev/00runs/r52_central_excitation_necessity_2026-09-27/``: the
``gate5_r52.json`` ``trigger_for_r53`` entry, the named residual
``multiplicity_tail_residual_one_model_beyond_2x_pooled_se`` at
n_ch = 8 with value 0.030148582276380863 against the pooled SE
0.002738966420355409 (sigma 11.007284372792265, triggered True at the
2.0 pooled-SE bound).  The trigger entry is read from the sealed gate
evidence and asserted verbatim before any generation (the driver
pinned-literal assertion; the trigger is never re-derivable from a
regenerated R52 and the driver never attempts it).  The trigger value,
the pooled SE, the sigma, and the boolean are recorded verbatim in the
pre-registration.

The fit (the triggered run's single free parameter)
---------------------------------------------------
The R51 sealed mixed draw ran at the nominal fractions (0.25 each over
sd_projectile, sd_target, dd, nd), so the ND-class events in the sealed
sample are the nd_multi extension's training events at the n = 1
records.  The fit set is exactly one free parameter:

- ``p_n2`` -- the probability of the second neutral excitation, in the
  bounded interval [0, 0.5] (the card-8 ``COUNT_BND``; the truncated
  count n in {1, 2}; the truncated-at-the-kinematic-maximum-multiplicity
  convention: the nd_multi count draw is consumed on the closed
  ``topology_selection`` substream before card 7's order and the frozen
  simplified cascade is bounded by construction, so the truncation is
  realized through the bounded count n in {1, 2} -- a count above the
  kinematic maximum multiplicity is not representable in the
  pre-registered model).

The topology fractions, epsilon, and B_SD stay frozen at the sealed R51
fit values (the sealed ``fit_report.json`` ``fitted`` block, asserted
byte-exactly before any generation); no strange-sector parameter is fit
(plan line 918); the blend ramp start stays fixed at
B = 2.680194110727528 GeV (Round 4 obligation 4).

The nd_multi class registry entry (``retained: False``) flips to
retained in the gate evidence of a triggered run; the class becomes the
fit's degree of freedom (the draw P(n = 2) of card 8).

The likelihood (plan section 5.4; the R51 pattern at the frozen point)
----------------------------------------------------------------------
The importance-weighted -2lnL over the pre-registered bins with the
per-bin covariance: the fractions, epsilon, and B_SD enter through the
exact per-event importance weights of the R51 machinery at the frozen
fitted point (the R51 module's ``event_log_weights`` consumed verbatim
at the frozen values, so the weight ensemble is THE R51 frozen-point
ensemble bit for bit), and the single free parameter p_n2 enters as a
posterior count-draw probability over the ND-class events.  The
estimator is the documented two-component mixture identity: at count
probability p_n2, the ND-class contribution to a multiplicity-bin (or
any ND-sensitive) count is the weighted sum over the n = 1 records
scaled by (1 - p_n2) plus the weighted sum over the n = 2 records
scaled by p_n2.  The estimator is exact on the sealed sample: the n = 1
events of the sealed ND records are the exact n = 1 draws of the
nd_multi generation at the same p_n2 (the count draw is consumed
before card 7's order, so an n = 1 nd_multi event is bit-identical to
card 7's ND event on the same seed trace; the card-8 bit-identity
contract), and the n = 2 records are generated fresh under the
pre-registered R53 seed block at the drawn p_n2 with their exact
importance weights (the R51 weight machinery applied to the card-7
fields the n = 2 records share; the second central source carries no
drawn exchange-kinematics variable into the R51 weight identity beyond
the card-4 mass-law record fields, so the R51 ND weight rows apply
verbatim).

The n = 2 generation (the fresh replica events; pre-registered)
---------------------------------------------------------------
Per production replica, the fresh n = 2 events are generated at the
pre-registered raw seed (the R53 block 20300961-20300968,
k = 0..7) on the event-index partition 1_300_000_000 + i (i < 100,000)
under the pre-registered drawn p_n2: every event consumes the closed
five-label substream quintuple at replica slot 0 (the R50/R51/R52
convention; the raw seed carries the replica identity), the count draw
resolves against the drawn p_n2, and an n = 1 resolution runs card 7's
ND generation on the remaining streams (bit-identical to the sealed R51
records on the same trace -- recorded, and exercised by the smoke
probe); an n = 2 resolution runs the card-8 four-system generation (the
second central source on the residual phase space with the exact
four-system closure ``M_1 + M_2 + M_c1 + M_c2 <= sqrt(s)`` by
rejection).  Every event conserves total B, Q, and S exactly (integer
arithmetic) and closes the total four-momentum at tolerance 1e-9
relative against the incoming sum; a violation raises and stops the run
(the house conservation contract).  The n = 2 events carry the
four-system solve (central pair at rest, t_c = t_1 + t_2 exactly on the
record; the operative solve realizes the projectile-side transfer
through the documented invariant).

The fit observables (the card's pre-registered sets)
----------------------------------------------------
- The multiplicity tails (the fit target): the multiplicity bins at
  n_ch at or above the 90th percentile of the R51 training
  distribution (the sealed R50-pinned reference; the threshold is the
  smallest bin whose cumulative probability reaches 0.90, bins
  n_ch >= 6 exactly as in R52).
- The held-out central spectra of R52 (pi+, pi-, K+, K- in |y^CM| < 0.5
  from the sealed R01 published binned tables, dense-accumulation/
  then-pick) as the preservation check, cross-referenced against the
  sealed R52 one-model values (the no-degradation check against the
  5% validation-deviance bound, plan section 5.5 condition 4, line
  320).
- The R51 training observables (the leading-proton spectrum, the gap
  fraction, the charged multiplicity, B_transport) as the
  no-degradation check (the R51 sealed model vectors cross-referenced).

The R51 training observables and the multiplicity tails are ND-insitive
at p_n2 (the multiplicity vectors enter through the ND-class mixture
term only), so the fit's response to p_n2 is exactly the ND-class
mixture identity; the held-out central spectra respond through the same
term.

The Hessian discipline (plan line 1871; the R46 pattern)
--------------------------------------------------------
No Hessian-based covariance is quoted from a non-converged or noisy
finite-MC objective.  The single-parameter Hessian is quoted only after
the optimizer converged, the 8 replica-block minima agree (the per-
block spread below 20% of the pooled error), and the pooled curvature
is positive definite and stable across all 8 replica blocks.  A
boundary minimum, disagreeing block minima, or a curvature failure are
recorded as the named residuals (the R46 forms ``boundary_minimum``,
``replica_block_minima_disagree``, ``block_curvature_failure``); a
non-convergence records the non-convergence and emits no Hessian-based
covariance (the R51/R46 precedent).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: read the sealed R52 trigger entry, verify the
  frozen substrate digests, the sealed R51/R52/R50/R01 seals, the
  frozen values against the sealed R51 ``fit_report.json``, the
  multiplicity tail cut, and the seed-block disjointness; write
  ``pre_registration.json`` and ``provenance.json`` into the run dir,
  then stop.  Nothing is generated; the pre-registration is written
  once, before any generation.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one mixed-draw replica x 2,000
  events at the smoke seed 20312925 on the event-index partition
  1_600_000_000 + i (i < 2,000), the fit machinery on synthetic data,
  into an explicit ``--run-dir`` (the default production run dir is
  refused).  ``--gate`` / ``--seal`` / ``--clean-validate`` refuse
  smoke artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested production
  replica records (any subset; the default is every replica) at the
  drawn p_n2, one record file per replica, then stop.  Requires the
  pre-registration and the triggered boolean.  A completed record that
  matches the pre-registered identity is reused unchanged (the
  pre-registered gateway fallback runs one job per replica; the local
  fallback is one single-threaded process per --replica invocation).
- ``--evaluate``: the single-parameter fit: the per-replica-block
  likelihoods, the pooled fit, the replica-block stability table, the
  tail-improvement and no-degradation evaluations, the Hessian (only
  after the replica minima agree and the curvature is stable),
  ``fit_report.json`` and ``REPORT.md``, then stop.  Requires all 8
  production records.
- ``--gate``: write ``gate5_r53.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the fit
  report.
- ``--clean-validate``: re-run the fit evaluation from a clean
  interpreter over the generated artifacts with a byte-identical
  report, re-run the pre-registered probe replica (replica 0) in
  memory against its sealed record, write
  ``clean_environment_commands.json``, then stop.  Exactly one pass;
  executed before the seal (the card-19 seal-ordering discipline; a
  second invocation refuses).
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  Requires the clean-environment validation
  record (exactly one pass, before the seal).  The run dir is
  immutable after the seal.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

The frozen simplified cascade at the sealed R45 point (kappa_B = 0.0,
kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV) is
consumed without refit (the R46 ``no_refit`` clause); the exchange
slopes act on the exchange kinematics only, never on any daughter-level
transverse dynamics; no Round 5 record claims a separation of exchange
recoil from decay transverse dynamics (the R46 outcome,
separation_claimed false).
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
# na49_mixing package, the resonance catalog, the continuum config, the
# sealed archives, the SPS tables, and the sibling R51/R52 driver
# modules.
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
        f"staged_r53_multiple_neutral_excitation: {message}",
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
# exact functions carry the importance weights (event_log_weights and
# the class law ratios), the record/conservation extraction
# (serialize_event, event_conservation, spectrum_records,
# charged_multiplicity, remnant_masses, exchange_transfers), the
# sealed-archive loaders (load_blocks, verify_sealed_dir,
# r50_reference_payload, _load_production_tables, _load_sealed_r01,
# build_geometry, _assign_cells, se_from, _sqrt_s_conversion,
# _species_pick_values, ess_fraction, build_statistics), and the fit
# statistics (Statistic, build_statistics, the nuisance-profile
# algebra's constants).  Consuming the module verbatim guarantees the
# frozen-point weight ensemble is the R51 ensemble bit for bit.
import staged_r51_topology_fraction_fit as r51  # noqa: E402

# The sibling R52 driver module: the central-spectrum extraction
# (central_spectrum_records, build_heldout_geometry,
# multiplicity_tail_cut) and the sealed-R52 trigger-entry surface.
import staged_r52_central_excitation_necessity as r52  # noqa: E402

from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.rng_substreams import (  # noqa: E402
    SUBSTREAM_LABELS,
    derive_substreams,
)
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.topology_multi_neutral import (  # noqa: E402
    COUNT_BND,
    DEFAULT_P2,
    NDMultiEvent,
    P_N2_TOLERANCE,
    generate_nd_multi,
)
from na49_mixing.topology_runner import (  # noqa: E402
    INELASTIC_ORDER,
    INELASTIC_SET,
    NOMINAL_PARAMS,
    TOPOLOGY_CLASSES,
    TopologyNominalParams,
    draw_topology,
    extract_observables,
    run_topology,
)
from na49_mixing.topology_single_diffraction import FinalHadron  # noqa: E402

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260925-r5-run-r53-multiple-
# neutral-excitation; the collision, statistics, seeds, observables,
# likelihood, and Hessian-discipline blocks are verbatim from the card
# and the round 5 index items 5, 8, 9, and 10).
# ---------------------------------------------------------------------------
RUN_ID = "R53"
CARD = "TASK-20260925-r5-run-r53-multiple-neutral-excitation"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r53_multiple_neutral_excitation_2026-09-25 re-pins to the
#: execution date 2026-09-27 (the R50/R51/R52 sealed archives moved the
#: same way).
RUN_NAME = "r53_multiple_neutral_excitation_2026-09-27"
DEFAULT_RUN_DIR = "dev/00runs/r53_multiple_neutral_excitation_2026-09-27"
GATE_RECORD_NAME = "gate5_r53.json"
REPORT_NAME = "fit_report.json"
REPORT_MD_NAME = "REPORT.md"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_REPORT_NAME = "smoke_fit_report.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r53_multiple_neutral_excitation.py")
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r53_multiple_neutral_excitation.py")
R52_DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r52_central_excitation_necessity.py")
R52_DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r52_central_excitation_necessity.py")

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9, cards 10-12 (the sealed R50/R51/R52 runs),
#: and this card.
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
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests,
#: including the R45/R46/R50/R51/R52 driver pins).
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
}

#: The M8 frozen ten-parameter set (the sha256 of the canonical JSON of
#: the manifest's frozen map).
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
BEAM_MOMENTUM_GEV_C = 158.0
BLEND_RAMP_START_GEV = 2.680194110727528
SQRT_S_EXTRACTION_GEV = 17.270
CM_BETA_PINNED = 0.9940791999921171
CM_GAMMA_PINNED = 9.203303892226666

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
            "model-uncertainty term of the fit (the carry on the "
            "continuum families; recorded, not folded into the fit's "
            "arithmetic)"
        ),
    },
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_seagull_note": (
        "the b_transport_seagull beyond_isotropic_not_positive residual "
        "is a carry, not a pass criterion (index item 4)"
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
#: T2 (plan section 5.1 line 254, the upper T2 bound; the R23/R32/R41
#: convention): 8 replicas x 100,000 p+p events = 800,000 events; each
#: replica is one mixed draw at the drawn p_n2 (the count draw on the
#: closed topology_selection substream; the n = 1 and n = 2 events both
#: stored).
N_REPLICAS = 8
EVENTS_PER_REPLICA = 100_000
N_EVENTS_TOTAL = N_REPLICAS * EVENTS_PER_REPLICA   # 800,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
#: The R53 raw replica seeds 20300961-20300968 (k = 0..7; index item 9;
#: the block is disjoint from every previously used raw block and from
#: every other reserved Round 5 block).
REPLICA_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + SEED_OFFSET + 36 + k for k in range(N_REPLICAS))
EVENT_INDEX_BASE = 1_300_000_000
EVENT_INDEX_RULE = "1_300_000_000 + i, i < 100,000 (per replica)"

SMOKE_SEED = MASTER_SEED + 40_000 + 12_000              # 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"

#: The reserved round-5 raw seed blocks (index item 9; the R50, R51,
#: and R52 blocks are used and recorded in PREVIOUS_USED_BLOCKS below).
RESERVED_ROUND5_BLOCKS = {
    "r50_used": [20300925, 20300944],
    "r51_used": [20300945, 20300952],
    "r52_used": [20300953, 20300960],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9).
RESERVED_ROUND5_PARTITIONS = {
    "r50_used": [1_000_000_000, 1_000_025_000],
    "r51_used": [1_100_000_000, 1_100_100_000],
    "r52_used": [1_200_000_000, 1_200_100_000],
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
}

#: The clean-environment validation's pre-registered probe replica
#: (the replica 0 mixed draw; the probe re-run is the determinism
#: evidence of acceptance (vii)).
PROBE_REPLICA = 0

# ---------------------------------------------------------------------------
# The sealed R52 trigger entry (consumed verbatim; asserted before any
# generation)
# ---------------------------------------------------------------------------
R52_RUN_DIR = "dev/00runs/r52_central_excitation_necessity_2026-09-27"
R52_GATE_NAME = "gate5_r52.json"
R52_TRIGGER_ENTRY: dict[str, Any] = {
    "named_residual": "multiplicity_tail_residual_one_model_beyond_"
                      "2x_pooled_se",
    "value": 0.030148582276380863,
    "pooled_se": 0.002738966420355409,
    "sigma": 11.007284372792265,
    "triggered": True,
    "argmax_bin_n_ch": 8,
}
R52_REPORT_DIGEST = "37ffc0bcdeaf5728d09032d34ff43ffb359c8aafce9257d8aed0e241a9971460"

#: The frozen R51 fitted values (the one-model frozen point; asserted
#: byte-exactly against the sealed R51 fit_report.json before any
#: generation; identical to the R52 driver's FROZEN_FIT).
FROZEN_FIT: dict[str, float] = {
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
    "dd_fraction": 0.9999997010778905,
    "nd_fraction": 3.1345895878942542e-09,
    "epsilon": 0.0033443027832747944,
    "b_sd": 1.8457918026393214,
}

#: The drawn P(n = 2) of the fit sample (the generation point of the
#: fresh n = 2 events; pre-registered inside the bound; the fit's
#: initial point).
DRAWN_P_N2 = 0.25

#: The fit set (exactly one free parameter) and its bound.
FIT_PARAMETERS: tuple[str, ...] = ("p_n2",)
PARAM_BOUNDS: dict[str, tuple[float, float]] = {"p_n2": (0.0, COUNT_BND)}
INITIAL_VALUES: dict[str, float] = {"p_n2": DRAWN_P_N2}
BOUNDS_NOTE = (
    "the single free parameter P(n >= 2) bounded at [0, 0.5] (the "
    "card-8 COUNT_BND; plan line 829 'bounded'; the R53 step fits "
    "inside this bound, never above); the truncated count n in {1, 2} "
    "is truncated at the kinematic maximum multiplicity (the frozen "
    "simplified cascade is bounded by construction and no count above "
    "2 is representable in the pre-registered model); the topology "
    "fractions, epsilon, and B_SD stay frozen at the R51 fit values; no "
    "strange-sector parameter (plan line 918); the blend ramp start is "
    "fixed at B = 2.680194110727528 GeV (Round 4 obligation 4)"
)

#: The optimizer settings (pre-registered; deterministic; the R41/R51
#: convention with the one-parameter surface).
OPTIMIZER_SETTINGS = {
    "algorithm": ("scipy.optimize.minimize, method Nelder-Mead, bounded "
                  "(scipy vertex clipping into PARAM_BOUNDS), adaptive "
                  "True (the R41 convention)"),
    "xatol": 1e-3,
    "fatol": 1e-6,
    "maxiter": 200,
    "maxfev": 400,
    "initial_point": [INITIAL_VALUES[name] for name in FIT_PARAMETERS],
    "determinism_note": (
        "the objective is a deterministic function of the parameter "
        "vector on the fixed generated sample (no sampler runs inside "
        "the objective), so the bounded Nelder-Mead walk from the fixed "
        "initial point is deterministic and a clean-environment re-run "
        "reproduces the fit report bytes"),
    "invalid_trial_note": (
        "a trial outside the bound [0, 0.5] or with a non-finite "
        "objective value returns +inf (the R41 target convention); the "
        "simplex avoids it"),
}

#: The Hessian step (pre-registered central-difference step on the
#: single parameter) and the stability bound.
HESSIAN_STEPS: dict[str, float] = {"p_n2": 0.005}
STABILITY_SPREAD_BOUND = 0.20
SE_FLOOR = float(r51.SE_FLOOR)
ESS_FLOOR = r51.ESS_FLOOR

#: The tail-improvement and no-degradation bounds (acceptance (iv)).
TAIL_IMPROVEMENT_BOUND = 2.0          # 2.0 pooled SE (the card)
NO_DEGRADATION_BOUND = 0.05           # 5% validation deviance

#: The charged-multiplicity bins (the R51 training support; an event
#: outside the support refuses the run).
MULTIPLICITY_BINS: tuple[int, ...] = tuple(range(2, 15))

#: The central-activity rapidity window (the runner's CENTRAL_WINDOW).
CENTRAL_WINDOW = 0.5

#: The event four-momentum closure tolerance (the house conservation
#: contract).
EVENT_CLOSURE_TOL = 1e-9

#: The held-out central-spectrum species and their pinned masses (the
#: R52 driver's constants re-exported; the deviance statistics cover
#: the four published species).
PUBLISHED_SPECIES: tuple[str, ...] = r52.PUBLISHED_SPECIES
PI0_SPECIES = r52.PI0_SPECIES
PI0_ROW_MASS_GEV = r52.PI0_ROW_MASS_GEV
PI0_GEOMETRY_SPECIES = r52.PI0_GEOMETRY_SPECIES
SPECIES_PDGS = dict(r52.SPECIES_PDGS)
SPECIES_MASSES = dict(r52.SPECIES_MASSES)

#: The sealed input archives.
R01_ARCHIVE = r51.R01_ARCHIVE
R50_RUN_DIR = r51.R50_RUN_DIR
R51_RUN_DIR = r52.R51_RUN_DIR
R51_FIT_REPORT_NAME = r52.R51_FIT_REPORT_NAME

#: The class order of the inelastic sample (the runner's fixed
#: INELASTIC_ORDER; the mixed draw walks this order).
CLASS_ORDER: tuple[str, ...] = tuple(INELASTIC_ORDER)

INCOMING_REFERENCE_NOTE = (
    "the incoming sum of the pinned M8 p+p configuration: "
    "(E_beam + m_p, 0, 0, p_beam) with E_beam = sqrt(m_p^2 + p_beam^2), "
    "m_p = 0.93827231 GeV, p_beam = 158.0 GeV/c; every event's final "
    "state closes against this reference at tolerance 1e-9 relative"
)


def replica_seed(replica: int) -> int:
    """The raw seed of replica *replica* (the pre-registered R53
    block)."""
    k = int(replica)
    if not 0 <= k < N_REPLICAS:
        _refuse(f"replica index {k} outside the R53 block "
                f"0..{N_REPLICAS - 1}")
    return REPLICA_SEEDS[k]


def replica_name(replica: int) -> str:
    return RECORD_FMT.format(int(replica))


def event_id(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_id(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


# ---------------------------------------------------------------------------
# The sealed-R52 trigger entry (read and asserted before any generation)
# ---------------------------------------------------------------------------
def read_r52_trigger() -> dict:
    """The sealed R52 ``gate5_r52.json`` ``trigger_for_r53`` entry, read
    from the sealed archive (its seal verifies), asserted verbatim
    against the pinned literals, and returned with the seal state and
    the file digest (the pre-registration's trigger block)."""
    seal = verify_sealed_dir(R52_RUN_DIR)
    gate = _read_json(MONO_ROOT / R52_RUN_DIR / R52_GATE_NAME)
    trigger = gate.get("trigger_for_r53") or {}
    for key, expected in R52_TRIGGER_ENTRY.items():
        actual = trigger.get(key)
        if isinstance(expected, bool):
            ok = bool(actual) == expected
        else:
            ok = actual == expected
        if not ok:
            _refuse(
                f"the sealed R52 trigger entry drifted: {key} is "
                f"{actual!r}, expected the pinned {expected!r} (the "
                "trigger is consumed verbatim from the sealed gate "
                "evidence; a drift refuses the run)")
    if gate.get("run") != "R52" or not gate.get("sealed"):
        _refuse("the sealed R52 gate record is not the sealed R52 "
                f"production evidence (run {gate.get('run')!r}, sealed "
                f"{gate.get('sealed')!r})")
    return {
        "source": f"{R52_RUN_DIR}/{R52_GATE_NAME}",
        "gate_file_sha256": _sha256_file(
            MONO_ROOT / R52_RUN_DIR / R52_GATE_NAME),
        "seal": seal,
        "entry": {key: trigger.get(key) for key in R52_TRIGGER_ENTRY},
        "report_digest": gate.get("report_digest"),
        "match": True,
    }


def assert_trigger_fired() -> dict:
    """The trigger boolean is asserted True before any generation (the
    NOT_TRIGGERED path of the card writes the record and seals without
    generation; this driver's production modes refuse the run when the
    sealed boolean is False -- the pre-registration records the boolean
    and the run identity, and the fired path is the only path this
    driver executes)."""
    trigger = read_r52_trigger()
    if not trigger["entry"]["triggered"]:
        _refuse(
            "the sealed R52 trigger boolean is False: the card's "
            "NOT_TRIGGERED path applies (no generation, no optimizer; "
            "the multiple-excitation class stays implemented but "
            "unretained, Gate 5 clause 2).  Re-run the driver only "
            "after the trigger re-pins under a new sealed boolean.")
    return trigger


# ---------------------------------------------------------------------------
# The frozen-value assertion against the sealed R51 (before any
# generation)
# ---------------------------------------------------------------------------
def assert_frozen_values_against_r51() -> dict:
    """The frozen values are the sealed R51 fitted point exactly (the
    card: "the frozen values ... are asserted against the sealed R51
    fit_report.json before any generation")."""
    run_dir = MONO_ROOT / R51_RUN_DIR
    fit_report = _read_json(run_dir / R51_FIT_REPORT_NAME)
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
        "fit_report_sha256": _sha256_file(
            run_dir / R51_FIT_REPORT_NAME),
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
            "weight_degeneracy) are carried into this fit as the "
            "inherited model-uncertainty caveats"),
        "match": True,
    }


# ---------------------------------------------------------------------------
# The nd_multi registry and the frozen-value invariants (module level)
# ---------------------------------------------------------------------------
def assert_fit_set() -> dict:
    """The fit-set invariants (module level; acceptance (iii)): exactly
    one free parameter P(n >= 2) bounded at [0, 0.5]; the topology
    fractions, epsilon, and B_SD frozen at the R51 fit values; no
    strange-sector parameter; the blend ramp start fixed at B; the
    nd_multi registry entry intact (the class label, the bounded count,
    the bound, and the unretained default; the gate evidence flips the
    retained flag on the triggered path)."""
    if len(FIT_PARAMETERS) != 1 or FIT_PARAMETERS[0] != "p_n2":
        _refuse("the R53 fit set is not exactly one free parameter "
                f"{FIT_PARAMETERS!r}")
    lo, hi = PARAM_BOUNDS["p_n2"]
    if (lo, hi) != (0.0, COUNT_BND):
        _refuse(f"the p_n2 bound drifted: {(lo, hi)!r} vs "
                f"(0.0, {COUNT_BND!r})")
    if not DRAWN_P_N2 <= COUNT_BND:
        _refuse("the drawn P(n = 2) exceeds the bound")
    if dict(NOMINAL_PARAMS.class_fractions) != {
            "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}:
        _refuse("the runner's nominal inelastic fractions drifted")
    if set(TOPOLOGY_CLASSES) != {
            "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}:
        _refuse(f"the class registry drifted: {sorted(TOPOLOGY_CLASSES)!r}")
    entry = TOPOLOGY_CLASSES["nd_multi"]
    if entry.get("retained"):
        _refuse("the nd_multi registry entry must carry retained: False "
                "(the code surface stays unretained; the gate evidence "
                "records the flip)")
    if not COUNT_BND > 0.0 or DEFAULT_P2 != 0.0 or P_N2_TOLERANCE <= 0.0:
        _refuse("the card-8 nd_multi surface drifted")
    if TOPOLOGY_CLASSES["elastic"]["inelastic"]:
        _refuse("the elastic registry entry must carry inelastic: False")
    if set(INELASTIC_SET) != {"sd_projectile", "sd_target", "dd", "nd"}:
        _refuse(f"the inelastic set drifted: {sorted(INELASTIC_SET)!r}")
    return {
        "fit_set": list(FIT_PARAMETERS),
        "count": len(FIT_PARAMETERS),
        "bound": [lo, hi],
        "initial_values": dict(INITIAL_VALUES),
        "drawn_p_n2": DRAWN_P_N2,
        "frozen": {
            "fractions": {name: FROZEN_FIT[name] for name in
                          ("sd_projectile_fraction", "sd_target_fraction",
                           "dd_fraction", "nd_fraction")},
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            "refit": False,
        },
        "excluded": {
            "blend_ramp_start": (
                "Round 4 obligation 4 (fixed at B = "
                "2.680194110727528 GeV); never a free parameter"),
            "elastic": "plan line 824 (excluded from the inelastic sample)",
            "strange_sector": "plan line 918",
            "exchange_topology_flavor_branching_catalog": "plan section 24",
        },
        "truncation": (
            "the truncated count n in {1, 2} of card 8, truncated at "
            "the kinematic maximum multiplicity: the frozen simplified "
            "cascade is bounded by construction and no count above 2 is "
            "representable in the pre-registered model"),
        "registry_entry_flipped_in_gate": (
            "the nd_multi registry entry (retained: False in code) "
            "flips to retained in the gate evidence of this triggered "
            "run; the class is the fit's degree of freedom"),
        "pass": True,
    }


_FIT_SET_STATUS = assert_fit_set()


def frozen_params() -> TopologyNominalParams:
    """The frozen draw parameters: the R51 fitted fractions, epsilon,
    and B_SD, the nominal remnant weights, and P(n = 2) at the caller's
    value (the fit sample's drawn p_n2 at generation)."""
    return TopologyNominalParams(
        epsilon=FROZEN_FIT["epsilon"],
        b_sd=FROZEN_FIT["b_sd"],
        class_fractions=dict(NOMINAL_PARAMS.class_fractions),
        remnant_weights=dict(NOMINAL_PARAMS.remnant_weights),
        p_n2=DRAWN_P_N2,
    )


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
    the provenance; card 8's nd_multi module is the fit's machinery)."""
    modules = dict(r52.ROUND5_MODULES)
    return {
        name: _sha256_file(MONO_ROOT / rel)
        for name, rel in modules.items()
    }


def m8_frozen_set_status() -> dict:
    """The M8 frozen ten-parameter set: read from the legacy_v1
    manifest's frozen map (the freeze test's canonical form), verified
    against the pre-registered hash, and asserted against the frozen
    B_SD lineage and the fixed blend ramp start."""
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
        "effective_transfer_slope_gev2": float(
            m8["regge.effective_transfer_slope"]),
        "blend_ramp_start_gev": boundary,
        "note": (
            "the M8-frozen effective_transfer_slope is the R50/R51 "
            "generation nominal B_SD; this fit runs at the frozen R51 "
            "fitted B_SD asserted against the sealed R51 fit report"),
        "source": ("config/v2_singlet/legacy_v1_manifest.json "
                   "(frozen_fit_outputs.m8_frozen_pp_parameters)"),
    }


def verify_sealed_dir(rel: str) -> dict:
    """Verify a sealed run dir's SHA256SUMS.txt entry by entry (the R41
    verify-sealed discipline; the archive must be intact before it is
    consumed read-only)."""
    return r51.verify_sealed_dir(rel)


def verify_sealed_inputs() -> dict:
    """The sealed-input verification: the R51 archive (the frozen
    values and the n = 1 training events), the R52 archive (the trigger
    source), the R50 archive (the multiplicity-tail data anchor), and
    the R01 archive (the data side)."""
    return {
        "r52": verify_sealed_dir(R52_RUN_DIR),
        "r51": verify_sealed_dir(R51_RUN_DIR),
        "r50": verify_sealed_dir(R50_RUN_DIR),
        "r01": verify_sealed_dir(R01_ARCHIVE),
    }


# ---------------------------------------------------------------------------
# The seed-block disjointness (the card's stop condition)
# ---------------------------------------------------------------------------
def seed_disjointness_report() -> dict:
    """The seed-block disjointness check for the R53 block: the exact
    previously-used blocks recorded from the sealed pre-registrations
    and the round 5 index item 9, and the event-index partition
    disjointness.  Any (seed, event-index) substream collision refuses
    the run (never a silent re-block)."""
    problems: list[str] = []
    r53_lo = EVENT_INDEX_BASE
    r53_hi = EVENT_INDEX_BASE + EVENTS_PER_REPLICA
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
            if r53_lo < p_hi and p_lo < r53_hi:
                problems.append(
                    f"the R53 event-index partition [{r53_lo}, {r53_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS)):
            problems.append(
                f"the R53 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if name in ("r50_used", "r51_used", "r52_used"):
            continue
        if r53_lo < hi and lo < r53_hi:
            problems.append(
                f"the R53 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r53] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r53_raw_seed_block": [min(REPLICA_SEEDS), max(REPLICA_SEEDS)],
        "r53_raw_seed_rule": (
            "raw replica seed = 20260925 + 40000 + 36 + k = 20300961 + k "
            f"per replica k < {N_REPLICAS} (index item 9); the block "
            "20300961-20300968 is disjoint from every previously used "
            "raw block and from every other reserved Round 5 block"),
        "r53_replica_seeds": list(REPLICA_SEEDS),
        "r53_event_index_partition": EVENT_INDEX_RULE,
        "r53_event_index_range": [r53_lo, r53_hi],
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R53 raw block 20300961-20300968 shares no raw seed with "
            "any previously used block (R23 20300913-20300920 under the "
            "20260913 master ends below it; the R50 20300925-20300944, "
            "R51 20300945-20300952, and R52 20300953-20300960 blocks end "
            "below it; the Round 4 blocks 20301020-20301083 and smoke "
            "blocks 20307920-20311920 under the 20260920 master sit "
            "above it; the R32 blocks 20301519-20301826 under the "
            "20260918 master are disjoint); the substreams are disjoint "
            "by construction (both the numpy.random.SeedSequence spawn "
            "tree and derive_substreams(seed, 0, event_id) key on the "
            "raw seed and the event index) and the event-index "
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


# ---------------------------------------------------------------------------
# The R52 held-out cross-reference constants (sealed; asserted)
# ---------------------------------------------------------------------------
#: The sealed R52 one-model multiplicity-tail values (the comparison
#: baseline of the tail-improvement acceptance (iv); recorded in the
#: pre-registration and asserted against the sealed R52 held-out report
#: at evaluation).
R52_TAIL_BASELINE: dict[str, dict[str, float]] = {
    # tail bin n_ch (the kept set of the sealed R52 comparison): the
    # one-model value, its pull, and the bin's pooled SE
    "6": {"m_one": 0.24383649773433855, "one_pull": -3.479702664276799,
          "pooled_se": 128.55690472037142},
    "7": {"m_one": 0.015599463071232755, "one_pull": -6.789241677451963,
          "pooled_se": 129.75666411557367},
    "8": {"m_one": 0.05378608227638086, "one_pull": -11.007284372792265,
          "pooled_se": 163.98934345672424},
    "9": {"m_one": 0.0012107748834836796, "one_pull": -3.2275330182175592,
          "pooled_se": 35.86129429703968},
    "10": {"m_one": 0.011255884218804553, "one_pull": -8.64851490276768,
           "pooled_se": 106.57975451423248},
    "11": {"m_one": 7.3417919550140085e-06,
           "one_pull": -0.40329516314767483,
           "pooled_se": 1.2386810572171372},
    "12": {"m_one": 0.0010849913710599985,
           "one_pull": -4.111869873367374,
           "pooled_se": 35.73599771985144},
    "14": {"m_one": 0.00013140447700381452,
           "one_pull": -1.9087579562443522,
           "pooled_se": 7.765964468771355},
}
R52_TAIL_KEPT_BINS = (6, 7, 8, 9, 10, 11, 12, 14)


# ---------------------------------------------------------------------------
# The per-event physics layer (the house conservation contract; the
# R51 machinery consumed verbatim)
# ---------------------------------------------------------------------------
EVENT_SCHEMA = "na49_mixing.staged_r53_multiple_neutral_excitation.event/1"
ConservationViolation = r51.ConservationViolation
event_conservation = r51.event_conservation
serialize_event = r51.serialize_event
spectrum_records = r51.spectrum_records
charged_multiplicity = r51.charged_multiplicity
exchange_transfers = r51.exchange_transfers
remnant_masses = r51.remnant_masses
se_from = r51.se_from
ess_fraction = r51.ess_fraction
_assign_cells = r51._assign_cells
_class_log_law_weights = r51.class_log_law_weights
_event_log_weights = r51.event_log_weights


def multi_remnant_masses(event) -> list:
    """The per-event remnant masses the mass-law ratios consume (the
    R51 ``remnant_masses`` convention at the nd_multi extension): the
    ND-family classes carry the two remnant masses (projectile side
    first); the n = 2 record carries the same two baryonic remnant
    masses (the card-8 record's own ``remnants`` field; the baryonic
    lines and the remnant selections are card 7's)."""
    if isinstance(event, NDMultiEvent):
        return [float(event.remnants[0].mass), float(event.remnants[1].mass)]
    return remnant_masses(event)


def _serialize_event_multi_fields(event, obs, conservation, protons,
                                  pbars, n_ch) -> dict:
    """The R51 ``serialize_event`` body at the nd_multi record (the
    compact fit-relevant payload; the ``rm`` field is the nd_multi
    extension of the R51 convention)."""
    record = {
        "bqs": conservation["bqs"],
        "cl": event.class_label,
        "clo": conservation["closure"],
        "ex": [float(m) for m in obs.excitation_masses],
        "gp": (None if obs.gap_size is None else float(obs.gap_size)),
        "mc": int(obs.multiplicity_central),
        "mt": int(obs.multiplicity_total),
        "nc": int(n_ch),
        "pb": [[float(x), float(p), float(w)] for x, p, w in pbars],
        "pp": [[float(x), float(p), float(w)] for x, p, w in protons],
        "rm": multi_remnant_masses(event),
        "sg": float(obs.seagull_moment),
        "sh": conservation["mass_shell_worst"],
        "tr": [float(t) for t in obs.transfers],
        "xt": exchange_transfers(event),
        "xp": [float(obs.x_f_projectile), float(obs.x_f_target)],
    }
    return record


def serialize_multi_event(event, obs, conservation, pik, n_ch) -> dict:
    """The per-event record of the replica file (without its hash): the
    R52 record convention (the R51 compact record plus the ``pk`` block
    of the central pi/K spectrum records) extended with the multi block
    ``mu``: the drawn count (1 or 2), the P(n = 2) of the draw, the
    per-source central masses, and the second source's drawn transfer
    (0.0 at n = 1).  The caller adds the event identity fields and the
    per-event hash: the sha256 of the record's canonical JSON without
    the ``h`` key."""
    protons, pbars = spectrum_records(event)
    # the r51 serialize_event's remnant_masses does not carry the
    # nd_multi record; the rm field is built here (the nd_multi
    # extension of the R51 convention) and the rest of the record
    # follows the R51 serialize_event body verbatim
    if isinstance(event, NDMultiEvent):
        record = _serialize_event_multi_fields(
            event, obs, conservation, protons, pbars, n_ch)
        # the stored class label is the drawn class "nd" (the R51
        # weight machinery's class surface; the count stratum lives in
        # the mu block and the class_label field of the nd_multi record
        # is the card-8 class label)
        record["cl"] = "nd"
    else:
        record = serialize_event(
            event, obs, conservation, protons, pbars, n_ch)
    record["pk"] = {
        species: [[float(x), float(p), float(w)] for x, p, w in records]
        for species, records in pik.items()
    }
    if isinstance(event, NDMultiEvent):
        record["mu"] = {
            "count": int(event.count),
            "p_n2": float(event.p_n2),
            "central_masses": [float(m) for m in event.central_masses],
            "t_2c": float(event.t_2c),
        }
    else:
        record["mu"] = None
    return record


# ---------------------------------------------------------------------------
# The mixed-draw generation at the drawn p_n2 (the fresh replica events)
# ---------------------------------------------------------------------------
GENERATION_NOTE_TEXT = (
    "the R53 mixed draw runs at the R51-fitted fractions, epsilon, and "
    "B_SD with the nd_multi machinery active at the drawn P(n = 2) = "
    f"{DRAWN_P_N2!r} (the unretained-at-R50/R51 default lifted for this "
    "fit): the per-event class draw on the closed topology_selection "
    "substream runs the R51 nominal fractions (0.25 each over "
    "sd_projectile, sd_target, dd, nd; the elastic fraction exactly "
    "0.0), an nd resolution dispatches to the card-8 nd_multi "
    "generation (the count draw against the drawn P(n = 2) is consumed "
    "on topology_selection before card 7's order; an n = 1 resolution "
    "is bit-identical to card 7's ND record), and the remaining three "
    "classes run their committed cards-5/6/7 generations unchanged.  "
    "Every event conserves total B, Q, and S exactly (integer "
    "arithmetic) and closes the total four-momentum at tolerance 1e-9 "
    "relative against the incoming sum; a violation raises and stops "
    "the run.  The n = 2 events carry the four-system solve (central "
    "pair at rest, t_c = t_1 + t_2 exactly on the record; the exact "
    "four-system closure M_1 + M_2 + M_c1 + M_c2 <= sqrt(s) by "
    "rejection)")


def generate_replica_events(
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
):
    """Generate *n_events* mixed-draw events of one replica at the
    drawn P(n = 2) and return the per-event records, the class and
    count censuses, the conservation aggregate, and the event-hash set
    digest.

    Every event consumes the closed five-label substream quintuple at
    replica slot 0 (the R50/R51/R52 convention: the raw seed carries
    the replica identity), draws its class on the closed
    ``topology_selection`` substream at the R51-fitted fractions (the
    elastic fraction exactly 0.0; a drawn elastic label refuses the
    run), dispatches an nd resolution through the card-8 nd_multi
    generation (the count draw against the drawn P(n = 2) precedes
    card 7's order; an n = 1 resolution is bit-identical to card 7's
    ND record on the same trace) and the remaining three classes
    through their committed generations, extracts the observables and
    the central pi/K spectrum records, and runs the house conservation
    contract (a violation raises and stops the run)."""
    params = frozen_params()
    if params.p_n2 != DRAWN_P_N2 or not 0.0 < DRAWN_P_N2 <= COUNT_BND:
        _refuse("the draw parameters carry a P(n = 2) outside the "
                "pre-registered drawn value and bound")
    inelastic_ok = set(INELASTIC_SET) | {"nd_multi"}
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
            NOMINAL_PARAMS.class_fractions, 0.0, rng_topology_selection)
        if label == "elastic" or label not in inelastic_ok:
            raise ConservationViolation(
                f"event {event_index_base + i}: the class draw returned "
                f"{label!r} outside the inelastic set "
                f"{sorted(inelastic_ok)} (the R53 mixed draw runs at the "
                "elastic fraction 0.0; checked, never repaired)")
        # the nd resolution dispatches through the card-8 nd_multi
        # generation at the drawn P(n = 2) (the unretained-at-R50/R51
        # default lifted for this fit): the count draw on the closed
        # topology_selection substream precedes card 7's draw order, so
        # an n = 1 resolution is bit-identical to card 7's ND record on
        # the same trace (the card-8 bit-identity contract) and the n = 2
        # resolution carries the four-system extension
        dispatch_label = "nd_multi" if label == "nd" else label
        event = run_topology(
            dispatch_label, params, catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=params.p_n2)
        obs = extract_observables(event, r51.SQRT_S)
        conservation = event_conservation(event)
        pik = r52.central_spectrum_records(event)
        n_ch = charged_multiplicity(event)
        if n_ch < MULTIPLICITY_BINS[0] or n_ch > MULTIPLICITY_BINS[-1]:
            raise ConservationViolation(
                f"event {event_index_base + i}: the charged multiplicity "
                f"{n_ch} is outside the pre-registered bins "
                f"{MULTIPLICITY_BINS}; the generation refuses (checked, "
                "never bled into an end bin; the truncated-count "
                "convention is bounded by the model, a live refusal is "
                "a model failure and stops the run)")
        record = serialize_multi_event(event, obs, conservation, pik,
                                       n_ch)
        record["i"] = int(i)
        record["id"] = int(event_index_base + i)
        record.pop("h", None)
        record["h"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8")).hexdigest()
        census[label] = census.get(label, 0) + 1
        if isinstance(event, NDMultiEvent) and event.count == 2:
            count_census["n2"] += 1
        elif isinstance(event, NDMultiEvent):
            count_census["n1"] += 1
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
        "n1_n2_census": dict(count_census),
    }
    return records, census, conservation_agg, hash_set


REPLICA_SCHEMA = "na49_mixing.staged_r53_multiple_neutral_excitation.replica/1"


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
    conservation: dict,
    hash_set: str,
    run_dir: Path,
    mode: str = "production",
) -> dict:
    """The replica record: the identity block, the seed map, the
    per-event records, the class and count censuses, the conservation
    aggregate, and the event-hash set digest (the generation-time
    determinism evidence)."""
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
                "raw replica seed = 20260925 + 40000 + 36 + k = 20300961 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300961-20300968 is disjoint from every "
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
            "note": GENERATION_NOTE_TEXT,
            "class_fractions": dict(NOMINAL_PARAMS.class_fractions),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": DRAWN_P_N2,
        },
        "class_census": dict(census),
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


def _verify_replica_identity(record: dict, replica: int,
                             run_dir: Path) -> None:
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
    """Generate the requested production replica records (the default
    is every replica), one record file per replica, then stop.
    Requires the pre-registration and the fired trigger.  A completed
    record that matches the pre-registered identity is reused unchanged
    (the pre-registered gateway fallback runs one job per replica; the
    local fallback is one single-threaded process per --replica
    invocation)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    assert_trigger_fired()
    pre = load_pre_registration(run_dir)
    registered_p_n2 = float((pre.get("fit_set") or {}).get(
        "drawn_p_n2", float("nan")))
    if registered_p_n2 != DRAWN_P_N2:
        _refuse(
            "the pre-registration's drawn P(n = 2) drifted: "
            f"{registered_p_n2!r} vs {DRAWN_P_N2!r}")
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
            print(f"[r53] replica {int(replica)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        seed = replica_seed(replica)
        print(
            f"[r53] generating replica {int(replica)} ({name}): seed "
            f"{seed}, {EVENTS_PER_REPLICA} mixed-draw events at the drawn "
            f"P(n = 2) = {DRAWN_P_N2!r}",
            flush=True,
        )
        records, census, conservation, hash_set = generate_replica_events(
            seed, EVENTS_PER_REPLICA, EVENT_INDEX_BASE, catalog, fams)
        record = build_replica_record(
            replica, records, census, conservation, hash_set, run_dir)
        record["reused"] = False
        _write_json(path, record)
        print(
            f"[r53] replica {int(replica)} written: {path} "
            f"(census {census}, count census "
            f"{conservation['n1_n2_census']}, worst closure "
            f"{conservation['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# The held-out geometry and the block loading (the R52 machinery)
# ---------------------------------------------------------------------------
class HeldoutBlock:
    """One replica's held-out evaluation arrays: the charged-multiplicity
    bin indices, the ND-class event indices (the p_n2 mixture term's
    support), and the per-species dense-lattice entries of the central
    pi/K records (the flat cell, the parallel event index, the invariant
    measure, and the out-of-window drop count)."""

    __slots__ = ("replica", "seed", "n_events", "nch_idx", "nd_idx",
                 "species_flat", "species_evt", "species_w",
                 "species_dropped", "species_total",
                 "prot_flat", "prot_evt", "prot_w",
                 "pbar_flat", "pbar_evt", "pbar_w",
                 "log_w", "log_shift", "weight_rows")


def load_heldout_blocks(run_dir: Path, geometry, held, *, smoke: bool,
                        verify: bool = True) -> list:
    """Load the replica records into the held-out evaluation arrays
    (identity verified by default; the smoke mode loads the single
    smoke replica record)."""
    blocks = []
    if smoke:
        records = [(SMOKE_REPLICA_NAME,
                    _read_json(Path(run_dir) / SMOKE_REPLICA_NAME))]
    else:
        records = [(replica_name(rep),
                    _read_json(Path(run_dir) / replica_name(rep)))
                   for rep in range(N_REPLICAS)]
    for rep, (name, record) in enumerate(records):
        if verify and not smoke:
            _verify_replica_identity(record, rep, Path(run_dir))
        block = HeldoutBlock()
        block.replica = record.get("replica", "smoke")
        block.seed = int(record["seed"])
        block.n_events = int(record["n_events"])
        nch = np.empty(block.n_events, dtype=np.int64)
        nd_idx = np.empty(block.n_events, dtype=np.int64)
        pk_lists = {species: ([], [], [], []) for species in SPECIES_PDGS}
        for i, ev in enumerate(record["events"]):
            nch[i] = int(ev["nc"]) - MULTIPLICITY_BINS[0]
            nd_idx[i] = 1 if ev.get("mu") is not None else 0
            pk = ev.get("pk") or {}
            for species in SPECIES_PDGS:
                for x_f, p_t, w_meas in pk.get(species, []):
                    pk_lists[species][0].append(float(x_f))
                    pk_lists[species][1].append(float(p_t))
                    pk_lists[species][2].append(float(w_meas))
                    pk_lists[species][3].append(int(i))
        block.nch_idx = nch
        block.nd_idx = nd_idx
        block.species_flat, block.species_evt, block.species_w = {}, {}, {}
        block.species_dropped, block.species_total = {}, {}
        # the full published-row geometry entries of the training
        # spectra (the proton and antiproton records; the R51 loader's
        # convention over the full pick geometry)
        prot_evt, prot_xf, prot_pt, prot_w = [], [], [], []
        pbar_evt, pbar_xf, pbar_pt, pbar_w = [], [], [], []
        for i, ev in enumerate(record["events"]):
            for x_f, p_t, w_meas in ev.get("pp", []):
                prot_evt.append(i)
                prot_xf.append(float(x_f))
                prot_pt.append(float(p_t))
                prot_w.append(float(w_meas))
            for x_f, p_t, w_meas in ev.get("pb", []):
                pbar_evt.append(i)
                pbar_xf.append(float(x_f))
                pbar_pt.append(float(p_t))
                pbar_w.append(float(w_meas))
        block.prot_flat, block.prot_evt, block.prot_w, _ = _assign_cells(
            np.asarray(prot_xf, dtype=float),
            np.asarray(prot_pt, dtype=float),
            geometry["edges"]["proton"][0], geometry["edges"]["proton"][1],
            geometry["n_x"]["proton"], np.asarray(prot_evt, dtype=np.int64),
            np.asarray(prot_w, dtype=float))
        block.pbar_flat, block.pbar_evt, block.pbar_w, _ = _assign_cells(
            np.asarray(pbar_xf, dtype=float),
            np.asarray(pbar_pt, dtype=float),
            geometry["edges"]["antiproton"][0],
            geometry["edges"]["antiproton"][1],
            geometry["n_x"]["antiproton"],
            np.asarray(pbar_evt, dtype=np.int64),
            np.asarray(pbar_w, dtype=float))
        for species in SPECIES_PDGS:
            xf_p, pt_p, w_p, evt_p = pk_lists[species]
            if species == PI0_SPECIES:
                edges = geometry["edges"][PI0_GEOMETRY_SPECIES]
                n_x = geometry["n_x"][PI0_GEOMETRY_SPECIES]
            else:
                edges = geometry["edges"][species]
                n_x = geometry["n_x"][species]
            flat, kept_evt, kept_w, dropped = _assign_species_cells(
                np.asarray(xf_p, dtype=float), np.asarray(pt_p, dtype=float),
                np.asarray(w_p, dtype=float),
                np.asarray(evt_p, dtype=np.int64), edges, n_x)
            block.species_flat[species] = flat
            block.species_evt[species] = kept_evt
            block.species_w[species] = kept_w
            block.species_dropped[species] = dropped
            block.species_total[species] = len(xf_p)
        blocks.append(block)
    return blocks


def _assign_species_cells(xf, pt, measures, evt, edges, n_x):
    """The dense-lattice cell assignment of one species' (x_F, p_T)
    pairs (the R52 machinery; out-of-window pairs are dropped and
    counted, never bled into an end cell)."""
    flat, kept_evt, kept_w, dropped = _assign_cells(
        xf, pt, edges[0], edges[1], n_x, evt, measures)
    return flat, kept_evt, kept_w, dropped


def _species_model_vectors(blocks, weights, species, held, sqrt_s_data,
                           dense_size):
    """The weighted per-species model vectors on a species' central
    published rows (the R52 conversion
    ``m = sums / (pi sqrt(s) N_eff dxF dpT)``; the between-replica SE is
    the between-block standard error over the 8 blocks, floored)."""
    held_sp = held["species"][species] if species in held["species"] \
        else held["pi0"]
    pick = held_sp["central_pick"]
    areas = held_sp["central_areas"]
    sums_total = np.zeros(dense_size)
    per_block_sums = []
    n_eff_per_block = []
    for block, w in zip(blocks, weights):
        flat = block.species_flat[species]
        if flat.size:
            sw = w[block.species_evt[species]] * block.species_w[species]
            sums = np.bincount(flat, weights=sw, minlength=dense_size)
        else:
            sums = np.zeros(dense_size)
        per_block_sums.append(sums)
        sums_total = sums_total + sums
        n_eff_per_block.append(float(np.sum(w)))
    n_eff_total = math.fsum(n_eff_per_block)
    pooled = r51._species_pick_values(
        sums_total, pick, areas, sqrt_s_data, n_eff_total)
    block_values = [
        r51._species_pick_values(s, pick, areas, sqrt_s_data, n)
        for s, n in zip(per_block_sums, n_eff_per_block)]
    se = se_from(np.asarray(block_values, dtype=float))
    return {
        "pooled": pooled,
        "block_values": block_values,
        "se": se,
        "n_eff_total": n_eff_total,
        "n_eff_per_block": n_eff_per_block,
        "records_total": int(sum(b.species_total[species]
                                 for b in blocks)),
        "dropped_out_of_window": int(sum(b.species_dropped[species]
                                         for b in blocks)),
    }


def _pick_population(blocks, species, held, dense_size) -> np.ndarray:
    """The population flags of a species' central rows under the model:
    a row is populated when at least one block's (unit-weight) record
    contributes to its dense cell."""
    held_sp = held["species"][species] if species in held["species"] \
        else held["pi0"]
    pop = np.zeros(max(int(dense_size), 1), dtype=bool)
    for block in blocks:
        flat = block.species_flat[species]
        if flat.size:
            pop[flat] = True
    return pop[held_sp["central_pick"]] if held_sp["central_pick"].size \
        else np.zeros(0, dtype=bool)


# ---------------------------------------------------------------------------
# The single-parameter fit machinery (the p_n2 mixture identity)
# ---------------------------------------------------------------------------
class Pn2Objective:
    """The -2lnL of the single-parameter fit set on the pre-registered
    bins (plan section 5.4): the fractions, epsilon, and B_SD enter
    through the exact per-event importance weights of the R51 machinery
    at the frozen fitted point (``event_log_weights`` at the frozen
    values, one global log shift), and the single free parameter p_n2
    enters through the two-component ND-class mixture identity: the
    ND-class contribution to a multiplicity count is the weighted sum
    over the n = 1 records scaled by (1 - p_n2) plus the weighted sum
    over the n = 2 records scaled by p_n2.  The central-spectrum
    families respond through the same term.  The nuisance profile (the
    two spectrum normalization nuisances plus the model-uncertainty
    carry) is solved exactly per trial by the generalized-least-squares
    profile (the R51 algebra; the nuisance columns and priors are the
    R51 constants verbatim).  A trial outside the bound [0, 0.5] or
    with a non-finite objective value returns +inf (the R41 target
    convention)."""

    def __init__(self, heldout_blocks: list, stats: list, geometry: dict,
                 sqrt_s: float) -> None:
        self.blocks = heldout_blocks
        self.stats = stats
        self.geometry = geometry
        self.sqrt_s = float(sqrt_s)
        self.kept: dict[str, np.ndarray] = {}
        self.dropped_bins: list = []

    def build_bin_structure(self) -> dict:
        """The fixed kept/dropped bin structure (computed once at the
        unit weights): a bin is kept when its data side is finite with
        a non-negative published variance AND at least one stored event
        of any block contributes to it (the R11 convention)."""
        self.kept = {}
        self.dropped_bins = []
        for stat in self.stats:
            d = np.asarray(stat.d, dtype=float)
            v = np.asarray(stat.v_data, dtype=float)
            data_ok = (np.isfinite(d) & np.isfinite(v) & (v >= 0.0))
            if stat.kind == "distribution":
                pop = np.zeros(len(MULTIPLICITY_BINS), dtype=bool)
                for block in self.blocks:
                    pop |= np.bincount(
                        block.nch_idx, minlength=len(MULTIPLICITY_BINS)
                    ).astype(bool)
                keep = data_ok & pop
            elif stat.kind == "spectrum":
                # the training spectra's population over the FULL
                # published rows (the R51 build_unit_structure
                # convention at the generation point)
                prot_pop = self.geometry["_population"]["proton"]
                pbar_pop = self.geometry["_population"]["antiproton"]
                if stat.key == "leading_proton_spectrum":
                    keep = data_ok & prot_pop[
                        self.geometry["pick_full"]["proton"]]
                elif stat.key == "b_transport_spectrum":
                    keep = data_ok & np.asarray([
                        bool(prot_pop[e["bin_a"]] or pbar_pop[e["bin_b"]])
                        for e in stat.cells], dtype=bool)
                else:
                    keep = data_ok
            else:
                keep = data_ok
            self.kept[stat.key] = keep
            for i in np.flatnonzero(~keep):
                self.dropped_bins.append({
                    "statistic": stat.key, "index": int(i),
                    "reason": ("structurally empty at the generation point "
                               "(no stored event contributes at unit "
                               "weights; the R11 convention)"
                               if stat.kind == "distribution" else
                               "non-finite data or published variance"),
                })
        return {
            "kept_per_statistic": {
                key: int(mask.sum()) for key, mask in self.kept.items()},
            "dropped": len(self.dropped_bins),
        }

    def evaluate(self, p_n2: float) -> dict:
        """The profiled -2lnL at the trial p_n2 (the single-parameter
        objective; the frozen-point weight ensemble is precomputed once
        and the p_n2 response is the exact ND-class mixture identity).
        Returns the objective value with the pooled and per-block model
        vectors, the between-replica SEs, and the profiled nuisances."""
        p = float(p_n2)
        if not (math.isfinite(p) and 0.0 <= p <= COUNT_BND + P_N2_TOLERANCE):
            return {"value": float("inf")}
        if p < 0.0:
            p = 0.0
        geometry = self.geometry
        # -- the multiplicity vectors (the p_n2 mixture identity) -------
        # The ND-class events of each block carry their class weight w_nd
        # (the frozen-point R51 weight row); the n = 1 records scale by
        # (1 - p) and the n = 2 records by p.  The non-ND classes carry
        # their class weights unchanged.  The p_n2 response of every
        # statistic is carried by the ND-class term alone.
        mult_total = np.zeros(len(MULTIPLICITY_BINS))
        prot_total = np.zeros(geometry["dense_size"]["proton"])
        pbar_total = np.zeros(geometry["dense_size"]["antiproton"])
        per_block = []
        n_eff_total = 0.0
        for block in self.blocks:
            idx_by_nd = {
                1: np.flatnonzero(block.nd_idx == 1),
                2: np.flatnonzero(block.nd_idx == 2),
            }
            non_nd = np.flatnonzero(block.nd_idx == 0)
            scale = np.ones(block.n_events, dtype=float)
            # the ND-class scale: the count-draw posterior identity.
            # The mixture term applies to the ND-class weight rows; the
            # exact per-event identity is the count posterior
            # p(count | class = nd, p_n2) over the generated sample,
            # normalized per class inside the estimator (the
            # self-normalized convention), so the ND-class rows enter
            # as the relative (1 - p) / p mixture of the two count
            # strata against their own pooled self-normalizer.  The
            # implementation applies the scale to the ND rows and lets
            # the self-normalization absorb the common factor.
            scale[idx_by_nd[1]] = 1.0 - p
            scale[idx_by_nd[2]] = p
            w_full = np.exp(block.log_w - block.log_shift) * scale
            n_eff = float(np.sum(w_full))
            if not math.isfinite(n_eff) or n_eff <= 0.0:
                return {"value": float("inf")}
            mult_counts = np.bincount(
                block.nch_idx, weights=w_full, minlength=len(MULTIPLICITY_BINS))
            prot_w = (w_full[block.prot_evt] * block.prot_w
                      if block.prot_evt.size else np.zeros(0, dtype=float))
            pbar_w = (w_full[block.pbar_evt] * block.pbar_w
                      if block.pbar_evt.size else np.zeros(0, dtype=float))
            prot_sums = np.bincount(
                block.prot_flat, weights=prot_w,
                minlength=geometry["dense_size"]["proton"])
            pbar_sums = np.bincount(
                block.pbar_flat, weights=pbar_w,
                minlength=geometry["dense_size"]["antiproton"])
            per_block.append({
                "n_eff": n_eff, "mult_counts": mult_counts,
                "prot_sums": prot_sums, "pbar_sums": pbar_sums,
                "ess": ess_fraction(w_full),
            })
            prot_total += prot_sums
            pbar_total += pbar_sums
            mult_total += mult_counts
            n_eff_total += n_eff
        if not (math.isfinite(n_eff_total) and n_eff_total > 0.0):
            return {"value": float("inf")}
        # -- the per-block and pooled species values (the published rows;
        #    the training observables live on the full published rows)
        prot_pick_full = geometry["pick_full"]["proton"]
        pbar_pick_full = geometry["pick_full"]["antiproton"]
        prot_areas_full = geometry["areas_full"]["proton"]
        pbar_areas_full = geometry["areas_full"]["antiproton"]
        prot_pooled = r51._species_pick_values(
            prot_total, prot_pick_full, prot_areas_full, self.sqrt_s,
            n_eff_total)
        pbar_pooled = r51._species_pick_values(
            pbar_total, pbar_pick_full, pbar_areas_full, self.sqrt_s,
            n_eff_total)
        prot_block = [r51._species_pick_values(
            e["prot_sums"], prot_pick_full, prot_areas_full, self.sqrt_s,
            e["n_eff"]) for e in per_block]
        pbar_block = [r51._species_pick_values(
            e["pbar_sums"], pbar_pick_full, pbar_areas_full, self.sqrt_s,
            e["n_eff"]) for e in per_block]
        # -- the statistic vectors (per block and pooled) --
        pooled_vectors = []
        block_vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum" and stat.key == \
                    "leading_proton_spectrum":
                pooled_vectors.append(np.asarray(prot_pooled, dtype=float))
                block_vectors.append(
                    [np.asarray(v, dtype=float) for v in prot_block])
            elif stat.kind == "spectrum":
                pooled_vectors.append(np.asarray([
                    prot_pooled[e["bin_a"]] - pbar_pooled[e["bin_b"]]
                    for e in stat.cells
                ], dtype=float))
                block_vectors.append([np.asarray([
                    vb[e["bin_a"]] - bb[e["bin_b"]]
                    for e in stat.cells
                ], dtype=float) for vb, bb in zip(prot_block, pbar_block)])
            else:
                pooled_vectors.append(
                    np.asarray(mult_total / n_eff_total, dtype=float))
                block_vectors.append([
                    np.asarray(e["mult_counts"] / e["n_eff"], dtype=float)
                    for e in per_block])
        # -- the between-replica SEs --
        se_vectors = [
            se_from(np.asarray(vectors, dtype=float))
            for vectors in block_vectors
        ]
        # -- the profiled -2lnL on the fixed kept set --
        value, eta_hat = self._profiled_minus2lnl(
            pooled_vectors, se_vectors)
        if not math.isfinite(value):
            return {"value": float("inf")}
        return {
            "value": float(value),
            "pooled_vectors": pooled_vectors,
            "block_vectors": block_vectors,
            "se": se_vectors,
            "n_eff_total": float(n_eff_total),
            "log_shift": 0.0,
            "eta_hat": [float(v) for v in eta_hat],
            "ess": {
                "per_block": [e["ess"] for e in per_block],
                "min": min(e["ess"] for e in per_block),
            },
        }

    # -- the nuisance profile (the R51 algebra verbatim) ------------------
    def _profiled_minus2lnl(self, pooled_vectors, se_vectors):
        rows: list[float] = []
        columns: list[list[float]] = []
        for stat_index, stat in enumerate(self.stats):
            keep = self.kept[stat.key]
            m = np.asarray(pooled_vectors[stat_index], dtype=float)
            se = np.asarray(se_vectors[stat_index], dtype=float)
            d = np.asarray(stat.d, dtype=float)
            v_data = np.asarray(stat.v_data, dtype=float)
            indices = np.flatnonzero(keep)
            if indices.size == 0:
                continue
            mc_var = np.where(
                np.isfinite(se), np.maximum(se, SE_FLOOR) ** 2, float("inf"))
            total_var = v_data[indices] + mc_var[indices]
            finite = np.isfinite(total_var) & (total_var > 0.0)
            for pos, i in enumerate(indices):
                if not finite[pos]:
                    self.dropped_bins.append({
                        "statistic": stat.key, "index": int(i),
                        "reason": ("zero or non-finite total variance at "
                                   "the trial (the R11 convention)"),
                    })
                    continue
                weight = 1.0 / math.sqrt(float(total_var[pos]))
                rows.append((float(d[i]) - float(m[i])) * weight)
                columns.append([float(v) * weight
                                for v in self._nuisance_design_row(
                                    stat, i, float(m[i]))])
        if not rows:
            _refuse(
                "r53 fit: no finite objective bin (the statistic set is "
                "empty after the structure drops); the run stops"
            )
        r = np.asarray(rows, dtype=float)
        design = np.asarray(columns, dtype=float)
        prior_diag = np.asarray([
            r51.NORM_NUISANCE_PRIOR ** 2, r51.NORM_NUISANCE_PRIOR ** 2,
            r51.CARRY_NUISANCE_PRIOR ** 2,
        ], dtype=float)
        normal = design.T @ design + np.diag(1.0 / prior_diag)
        rhs = design.T @ r
        try:
            eta_hat = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            eta_hat = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        profiled = float(r @ r - rhs @ eta_hat)
        return profiled, eta_hat

    def _nuisance_design_row(self, stat, index: int,
                             model_value: float = 1.0) -> list:
        """The multiplicative nuisance design row (the R51 algebra
        verbatim: the derivative of the model value with respect to the
        nuisance is the model value itself)."""
        row = [0.0] * len(r51.NUISANCE_COLUMNS)
        column = r51.SPECTRUM_NORM_COLUMN.get(stat.key)
        if column is not None:
            row[column] = float(model_value)
        if stat.key == "b_transport_spectrum" and stat.carry_mask is not None:
            row[r51.CARRY_COLUMN_INDEX] = (
                float(model_value) * float(stat.carry_mask[index]))
        return row


# ---------------------------------------------------------------------------
# The block weights (the exact product law on the stored n = 2 records)
# ---------------------------------------------------------------------------
def block_weights(block, geometry) -> tuple[np.ndarray, float]:
    """One block's exact per-event log importance weights at the frozen
    point (the R51 machinery verbatim) extended with the exact product
    mass-law row of the stored n = 2 records: the frozen-point R51 rows
    for the four drawn classes (the R51 weight machinery at the frozen
    fractions, epsilon, and B_SD) and, for the n = 2 records, the
    second central source's exact card-4 mass-law ratio appended to the
    ND row (the drawn ``t_2c`` transfer through the exact transfer-law
    ratio, the drawn ``M_2c^2`` through the exact central law ratio on
    the residual limits, and the second source's log central
    normalization).  Returns ``(log_w, log_shift)`` with one global log
    shift across the blocks (the R51 evaluate convention)."""
    log_all = []
    for blk in (block,):
        rows = blk.weight_rows
        if rows is None:
            _refuse("the block carries no precomputed weight rows")
        log_all.append(np.asarray(rows, dtype=float))
    return log_all[0], float(block.log_shift)


def _minimize_pooled(objective: Pn2Objective):
    """The pre-registered one-parameter optimizer (scipy Nelder-Mead,
    bounded, deterministic; the settings are the pre-registered
    OPTIMIZER_SETTINGS)."""
    from scipy.optimize import minimize

    bounds = [tuple(PARAM_BOUNDS[name]) for name in FIT_PARAMETERS]
    x0 = list(OPTIMIZER_SETTINGS["initial_point"])

    def target(x):
        (p_n2,) = (float(v) for v in x)
        value = objective.evaluate(p_n2)["value"]
        if not math.isfinite(value):
            return float("inf")
        return float(value)

    return minimize(
        target, x0, method="Nelder-Mead", bounds=bounds,
        options={
            "xatol": OPTIMIZER_SETTINGS["xatol"],
            "fatol": OPTIMIZER_SETTINGS["fatol"],
            "maxiter": OPTIMIZER_SETTINGS["maxiter"],
            "maxfev": OPTIMIZER_SETTINGS["maxfev"],
            "adaptive": True,
        },
    )


def _fit_blocks_individually(heldout_blocks: list, stats: list,
                             geometry: dict, sqrt_s: float) -> list:
    """The 8 per-block fits (the replica-block stability table): the
    same pre-registered procedure on each block's own data."""
    block_fits = []
    for block in heldout_blocks:
        objective = Pn2Objective([block], stats, geometry, sqrt_s)
        objective.kept = dict(_POOLED_OBJECTIVE_KEPT)
        result = _minimize_pooled(objective)
        block_fits.append({
            "replica": block.replica,
            "minus2lnl": float(result.fun),
            **{name: float(result.x[i])
               for i, name in enumerate(FIT_PARAMETERS)},
            "converged": bool(result.success),
            "nfev": int(result.nfev),
        })
    return block_fits


def _pooled_curvature(objective: Pn2Objective, best_x) -> np.ndarray:
    """The 1x1 numerical Hessian of the pooled objective at the optimum
    (central differences with the pre-registered step)."""
    n = len(FIT_PARAMETERS)
    steps = [HESSIAN_STEPS[name] for name in FIT_PARAMETERS]
    hessian = np.zeros((n, n), dtype=float)

    def value_at(x):
        value = objective.evaluate(*x)["value"]
        return float(value) if math.isfinite(value) else float("inf")

    for i in range(n):
        si = steps[i]
        base = list(best_x)
        xpp = list(base); xpp[i] += si
        xmm = list(base); xmm[i] -= si
        entry = (value_at(xpp) - 2.0 * value_at(base) + value_at(xmm)) \
            / (si * si)
        hessian[i][i] = entry
    return hessian


def _pooled_errors(objective: Pn2Objective, best_x) -> dict:
    """The pooled per-parameter errors: the diagonal of the numerical
    Hessian of the pooled objective (the pre-registered steps; a
    non-positive curvature quotes an infinite error, which fails the
    stability bound and is named)."""
    curvature = _pooled_curvature(objective, best_x)
    errors = {}
    for index, name in enumerate(FIT_PARAMETERS):
        diagonal = float(curvature[index][index])
        errors[name] = (
            1.0 / math.sqrt(diagonal)
            if diagonal > 0.0 and math.isfinite(diagonal) else float("inf"))
    return errors


def _block_curvatures(heldout_blocks: list, stats: list, geometry: dict,
                      sqrt_s: float, best_x) -> list:
    """The per-block curvature of the single parameter at the pooled
    optimum (the Hessian-discipline stability check across all 8
    replica blocks; plan line 1871: the local curvature is stable)."""
    curvatures = []
    for block in heldout_blocks:
        objective = Pn2Objective([block], stats, geometry, sqrt_s)
        objective.kept = dict(_POOLED_OBJECTIVE_KEPT)
        curvature = _pooled_curvature(objective, best_x)
        value = float(curvature[0][0])
        curvatures.append(value if math.isfinite(value) else None)
    return curvatures


def _stability_table(best_x, block_fits, pooled_errors: dict) -> dict:
    """The acceptance (v) stability table: per parameter the 8 block
    values, the spread, and the 20%-of-pooled-error bound."""
    entries = {}
    for name in FIT_PARAMETERS:
        values = [fit[name] for fit in block_fits]
        spread = (
            float(max(values) - min(values)) if len(values) >= 2
            else float("nan")
        )
        pooled_error = float(pooled_errors[name])
        bound = STABILITY_SPREAD_BOUND * pooled_error
        entries[name] = {
            "per_block": [float(v) for v in values],
            "spread": spread,
            "pooled_error": pooled_error,
            "bound": bound,
            "pass": bool(spread < bound) if math.isfinite(bound) else False,
        }
    return entries


def _boundary_parameters(best_x) -> list:
    """The parameters sitting on a pre-registered bound (the tolerance
    1e-9; the named residual ``boundary_minimum``)."""
    out = []
    for index, name in enumerate(FIT_PARAMETERS):
        lo, hi = PARAM_BOUNDS[name]
        value = float(best_x[index])
        if abs(value - lo) <= 1e-9 or abs(value - hi) <= 1e-9:
            out.append({"parameter": name, "value": value,
                        "bound": [lo, hi]})
    return out


def _parameter_profiles(objective: Pn2Objective, best_x) -> dict:
    """The finite-MC profile (the fallback's recorded profile): the 11
    equally spaced points over the pre-registered bound [0, 0.5]."""
    profiles = {}
    for index, name in enumerate(FIT_PARAMETERS):
        lo, hi = PARAM_BOUNDS[name]
        points = []
        for g in range(11):
            value = lo + (hi - lo) * g / 10.0
            result = objective.evaluate(value)
            points.append({
                "value": value,
                "minus2lnl": (float(result["value"])
                              if math.isfinite(result["value"]) else None),
            })
        profiles[name] = points
    return profiles


def quoted_flag(minima_agree: bool, positive: bool, converged: bool,
                curvature_stable: bool, smoke: bool) -> bool:
    """The Hessian is quoted only after the optimizer converged, the
    replica-block minima agree (the stability table passes on every
    parameter), the pooled curvature is positive definite, and the
    per-block curvatures are stable (all positive and agreeing in sign
    and scale; plan line 1871; the R46 pattern; never in smoke)."""
    return bool((not smoke) and converged and minima_agree and positive
                and curvature_stable)


# ---------------------------------------------------------------------------
# The tail-improvement and no-degradation evaluations (acceptance (iv))
# ---------------------------------------------------------------------------
def _tail_improvement(tail_bins: list, report_bins: list,
                      r52_baseline: dict) -> dict:
    """The tail-improvement record: per kept tail bin the R52 one-model
    deviance (the trigger baseline), the R53 fit's one-model deviance at
    the fitted p_n2, the difference, the pooled SE of the deviance
    difference (propagated from the R53 fit's MC error), and the
    beyond-bound flag at the 2.0 pooled-SE bound (the card's retention
    bound; a positive difference is an improvement)."""
    entries = []
    for entry in report_bins:
        if not entry.get("kept"):
            continue
        n_ch = entry["bin"]
        baseline = r52_baseline.get(str(n_ch))
        if baseline is None:
            # the R52 comparison kept this bin only when both its
            # models populated it; a bin outside the sealed baseline's
            # kept set carries no R52 deviance baseline and is recorded
            # as skipped (the R53 improvement table covers the sealed
            # baseline's kept bins)
            entries.append({
                "bin": int(n_ch),
                "r52_one_deviance": None,
                "r53_one_deviance": float(entry["one_deviance"]),
                "difference": None,
                "pooled_se": None,
                "sigma": None,
                "improved_beyond_2x_se": False,
                "note": ("no sealed R52 baseline on this bin (the R52 "
                         "comparison's kept set does not carry it); the "
                         "bin is skipped in the improvement table"),
            })
            continue
        # the R52 one-model deviance (the trigger baseline) and the R53
        # fit's one-model deviance at the fitted p_n2
        r52_deviance = baseline["one_pull"] ** 2
        r53_deviance = entry["one_deviance"]
        difference = r52_deviance - r53_deviance
        # the pooled SE of the deviance difference: propagated from the
        # R53 fit's MC error (the R52 side is the sealed baseline and
        # carries no new MC error into the difference)
        sigma_one = math.sqrt(float(entry["v_data"])
                              + float(entry["se_one"]) ** 2)
        deriv = 2.0 * (float(entry["d"]) - float(entry["m_one"])) \
            / (sigma_one ** 2) * float(entry["se_one"])
        pooled_se = abs(deriv)
        if not math.isfinite(pooled_se) or pooled_se < SE_FLOOR:
            pooled_se = SE_FLOOR
        sigma = difference / pooled_se if pooled_se > 0.0 else 0.0
        entries.append({
            "bin": int(n_ch),
            "r52_one_deviance": r52_deviance,
            "r53_one_deviance": r53_deviance,
            "difference": difference,
            "pooled_se": pooled_se,
            "sigma": sigma,
            "improved_beyond_2x_se": bool(
                difference >= TAIL_IMPROVEMENT_BOUND * pooled_se
                and difference > 0.0),
        })
    degradations = [e["difference"] for e in entries
                    if e.get("difference") is not None]
    max_sigma = max((e["sigma"] for e in entries
                     if e.get("sigma") is not None), default=0.0)
    return {
        "per_bin": entries,
        "max_improvement_sigma": max_sigma,
        "any_improvement_beyond_2x_se": bool(
            max_sigma >= TAIL_IMPROVEMENT_BOUND),
        "bound": TAIL_IMPROVEMENT_BOUND,
        "note": (
            "the tail-improvement evaluation (acceptance (iv)): the R53 "
            "fit's multiplicity-tail deviance against the R52 one-model "
            "baseline (the sealed trigger evidence), the difference with "
            "its pooled SE propagated from the R53 fit's MC error, and "
            "the beyond-bound flag at the 2.0 pooled-SE bound"),
    }


def _no_degradation(r52_report: dict, r53_families: dict) -> list:
    """The no-degradation check (plan section 5.5 condition 4, line
    320): per previously acceptable sector the R53 fit's deviance
    against the R52 one-model deviance; a sector degrading by more than
    5% in validation deviance without a stated physics reason is
    named."""
    findings = []
    # The R52 one-model per-family deviance totals from the sealed
    # report (the cross-referenced held-out evidence)
    r52_per_bin = r52_report.get("held_out", {}).get("per_bin") or {}
    for family, r53_family in r53_families.items():
        bins_r52 = r52_per_bin.get(family) or []
        r52_total = math.fsum(
            float(b["one_deviance"]) for b in bins_r52
            if b.get("kept") and b.get("one_deviance") is not None)
        bins_r53 = r53_family.get("per_bin") or []
        r53_total = math.fsum(
            float(b["one_deviance"]) for b in bins_r53
            if b.get("kept") and b.get("one_deviance") is not None)
        if r52_total <= 0.0:
            continue
        change = (r53_total - r52_total) / r52_total
        findings.append({
            "family": family,
            "r52_one_deviance_total": r52_total,
            "r53_one_deviance_total": r53_total,
            "relative_change": change,
            "degrades_beyond_5pct": bool(change > NO_DEGRADATION_BOUND),
        })
    return findings


# ---------------------------------------------------------------------------
# The named-residual register (the R46/R51/R52 forms)
# ---------------------------------------------------------------------------
def _assemble_named_residuals(report: dict) -> list:
    """The named-residual register (the house forms): every failing
    acceptance verdict named with the numbers and a one-line
    diagnosis."""
    residuals = []
    optimizer = report["optimizer"]
    if not optimizer.get("converged"):
        residuals.append({
            "kind": "optimizer_non_convergence",
            "diagnosis": (
                "the pre-registered optimizer did not reach the "
                "convergence tolerances within the pre-registered "
                "evaluation budget on the finite-MC objective; no "
                "Hessian-based covariance is quoted from a non-converged "
                "finite-MC objective (plan line 1871) and the finite-MC "
                "profile is recorded"),
            "message": optimizer.get("message"),
            "nfev": optimizer.get("nfev"),
            "nit": optimizer.get("nit"),
        })
    for entry in report.get("boundary_parameters", []):
        residuals.append({
            "kind": "boundary_minimum",
            "parameter": entry["parameter"],
            "value": entry["value"],
            "bound": entry["bound"],
            "diagnosis": (
                "the pooled optimum sits on the parameter's "
                "pre-registered bound; the fit range does not extend "
                "(the R46 boundary-minimum pattern)"),
        })
    stability = report.get("replica_block_stability") or {}
    for name, entry in stability.items():
        if not entry.get("pass"):
            residuals.append({
                "kind": "replica_block_minima_disagree",
                "parameter": name,
                "per_block": entry["per_block"],
                "spread": entry["spread"],
                "pooled_error": entry["pooled_error"],
                "bound": entry["bound"],
                "sigma": (entry["spread"] / entry["pooled_error"]
                          if math.isfinite(entry["pooled_error"])
                          and entry["pooled_error"] > 0.0 else None),
                "diagnosis": (
                    "the 8-block spread exceeds 20% of the pooled error "
                    "(the pre-registered acceptance (v) bound); the "
                    "parameter is weakly constrained by the multiplicity "
                    "tails and the Hessian discipline withholds the "
                    "covariance (the R46 pattern)"),
            })
    hessian = report.get("hessian") or {}
    if hessian and not hessian.get("positive_definite", True):
        residuals.append({
            "kind": "block_curvature_failure",
            "eigenvalues": hessian.get("eigenvalues"),
            "block_curvatures": hessian.get("block_curvatures"),
            "diagnosis": (
                "the pooled numerical Hessian at the optimum is not "
                "positive definite or the per-block curvatures are "
                "unstable; no Hessian-based covariance is quoted (the "
                "R46 block-curvature pattern)"),
        })
    ess = (report.get("objective_at_optimum") or {}).get("ess") or {}
    if ess.get("min") is not None and float(ess["min"]) < ESS_FLOOR:
        residuals.append({
            "kind": "weight_degeneracy",
            "ess_min": ess["min"],
            "per_block": ess.get("per_block"),
            "diagnosis": (
                "the effective sample size fraction at the pooled "
                "optimum fell below the pre-registered 0.01 bound (the "
                "weight-degeneracy diagnosis); the trial's importance "
                "ensemble is degenerate and the optimum's location is "
                "recorded with that caveat"),
        })
    for finding in report.get("no_degradation_5pct") or []:
        if finding.get("degrades_beyond_5pct"):
            residuals.append({
                "kind": "sector_degradation_beyond_5pct",
                "family": finding["family"],
                "relative_change": finding["relative_change"],
                "diagnosis": (
                    "the previously acceptable sector degrades by more "
                    "than 5% in validation deviance at the fitted p_n2 "
                    "(plan section 5.5 condition 4, line 320); the "
                    "degradation is named and the Gate 5 follow-up "
                    "consumes it"),
            })
    # the inherited caveats (the instantiation notes carried verbatim)
    residuals.append({
        "kind": "r51_inherited_non_convergence",
        "diagnosis": (
            "the frozen point is the R51 optimum, which did not converge "
            "(the R51 named residuals optimizer_non_convergence, "
            "replica_block_minima_disagree, block_curvature_failure, "
            "weight_degeneracy); this fit inherits that state as a "
            "model-uncertainty caveat on the frozen parameters"),
    })
    residuals.append({
        "kind": "pi0_published_table_absent",
        "diagnosis": r52.PI0_NOTE,
    })
    residuals.append({
        "kind": "multiplicity_tail_training_overlap",
        "diagnosis": r52.MULTIPLICITY_TAIL_NOTE,
    })
    return residuals


# ---------------------------------------------------------------------------
# The main fit evaluation (the run's --evaluate; the report machinery)
# ---------------------------------------------------------------------------
REPORT_SCHEMA = "na49_mixing.staged_r53_multiple_neutral_excitation.report/1"

DETERMINISM_NOTE = (
    "a per-replica re-run from a clean interpreter reproduces the report "
    "hashes; fit_report.json and gate5_r53.json carry no wall-clock "
    "fields (their bytes are a pure function of the sealed artifacts, "
    "the pre-registered seeds, and the committed code); the per-event "
    "hash is the sha256 of the canonical JSON of the stored event record "
    "and the event_hash_set_sha256 is the sha256 over the joined "
    "per-event hashes, so the stored events carry the bit-level identity "
    "evidence and the probe replica's clean re-run must reproduce them")

RECORDING_CONVENTION_NOTE = (
    "dense-lattice accumulation per species on the published evaluation "
    "lattice, published rows selected with the pick map "
    "bin_index_pt * n_x + bin_index_xf, hole cells dropped (the "
    "corrected card-17 relitigation convention, the inherited state of "
    "the index card); the held-out central rows are the published rows "
    "whose row-mid y^CM at the species mass falls inside |y^CM| < 0.5 "
    "(the data-side window convention; the model side accumulates every "
    "particle record on the dense lattice and picks the same central "
    "rows; no particle-level y cut is applied)")

LIKELIHOOD_NOTE = (
    "-2lnL per pre-registered bin = ((d - m)/sqrt(V))^2 with "
    "V = V_data + V_mc (reported separately), the published binning "
    "preserved, and the nuisance profile (the two spectrum "
    "normalization nuisances plus the model-uncertainty carry) solved "
    "exactly per trial by the generalized-least-squares profile (the "
    "R51 algebra verbatim).  The single free parameter P(n >= 2) enters "
    "through the exact two-component ND-class mixture identity: the "
    "ND-class contribution to a count is the weighted n = 1 sum scaled "
    "by (1 - p_n2) plus the weighted n = 2 sum scaled by p_n2 (the "
    "self-normalized estimator absorbs the common factor).  The per-bin "
    "MC error is the between-replica standard error (std ddof=1 over "
    "the 8 replica blocks, divided by sqrt(8)), floored at the engine's "
    "DETERMINISTIC_SE_FLOOR; a bin where the sample is structurally "
    "zero in every block is recorded as dropped with the reason (the "
    "R11 convention; checked, never bled into a kept bin).  Data and "
    "MC contributions are reported separately (the V_data and V_mc "
    "blocks of the per-bin records)")

N2_EXTENSION_NOTE = (
    "the n = 2 records are the card-8 four-system generation (the two "
    "neutral central sources at (B, Q, S) = (0, 0, 0), each decaying "
    "through the frozen simplified cascade; the exact four-system "
    "closure M_1 + M_2 + M_c1 + M_c2 <= sqrt(s) by rejection; the "
    "central pair at rest in the overall CM with the two baryonic "
    "systems recoiling back-to-back and the production angle from the "
    "drawn projectile-side transfer through the documented invariant).  "
    "The n = 1 records are bit-identical to card 7's ND records on the "
    "same seed trace (the card-8 bit-identity contract; the count draw "
    "precedes card 7's order).  The weight of a stored n = 2 record "
    "carries the second source's exact card-4 mass-law ratio (the drawn "
    "t_2c transfer through the exact transfer-law ratio, the drawn "
    "M_2c^2 through the exact central law ratio on the residual limits, "
    "and the second source's log central normalization) appended to the "
    "R51 ND row; the recorded t_c = t_1 + t_2 exactly (the "
    "central-production idealization)")

TRIGGER_NOTE = (
    "the trigger (index item 5) is consumed verbatim from the sealed "
    "R52 gate5_r52.json trigger_for_r53 entry (the named residual "
    "multiplicity_tail_residual_one_model_beyond_2x_pooled_se at "
    "n_ch = 8, value 0.030148582276380863, pooled SE "
    "0.002738966420355409, sigma 11.007284372792265, triggered True at "
    "the 2.0 pooled-SE bound), asserted before any generation; it is "
    "never re-derivable from a regenerated R52 and the driver never "
    "attempts it")

GATE5_CLAUSE_NOTE = (
    "the nd_multi class is retained only if the multiplicity-tail "
    "improvement is real and stable (the acceptance (iv)/(v) evidence); "
    "the written retain / revise / reject decision on the Gate 5 "
    "clauses is the round's Gate 5 follow-up obligation (plan line 938: "
    "'Do not retain extra central excitations merely because they "
    "improve a global inclusive fit')")

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(xi) of the card: (i) the sealed R52 "
    "trigger entry read and asserted verbatim before any generation; "
    "(ii) not applicable (the triggered path; the NOT_TRIGGERED path "
    "generates nothing); (iii) the single-parameter fit on P(n >= 2) "
    "bounded at [0, 0.5], the fitted value and the per-replica-block "
    "table in fit_report.json; (iv) the multiplicity-tail improvement "
    "against the R52 baseline with the pooled SE against the 2.0 bound, "
    "the held-out central spectra preserved, and no previously "
    "acceptable sector degrading by more than 5% in validation deviance "
    "without a stated physics reason; (v) P(n >= 2) stable across all 8 "
    "independent replica blocks, the per-block spread below 20% of the "
    "pooled error; (vi) conservation on all 800,000 events; (vii) the "
    "clean-environment re-run reproduces the per-replica report hashes; "
    "(viii) the Hessian discipline (quoted only after replica-minimum "
    "agreement and stable curvature, or quoted: false with the named "
    "residuals); (ix) optimizer_invoked true (the round's second and "
    "final optimizer call; R51 is the first); (x) exactly one "
    "clean-environment validation pass before the seal; (xi) the driver "
    "and test file land byte-identical in both trees and the freeze "
    "re-pin is recorded")

SUBSTREAM_COLLISION_RULE = (
    "if any raw (seed, event-index) substream collides with a "
    "pre-registered block, the run stops and reports; a silent re-block "
    "never happens (index item 9)")

GENERATION_GATEWAY_NOTE = (
    "gateway-primary generation on the triggered path (host cl, workers "
    "c0801-c0809 x 32 task slots; one job per replica, --tasks 1, single "
    "math thread (NA49_CLUSTER_GW_TASKS forces thread counts to 1), "
    "submission paced by the capacity check, no local process pool "
    "inside the chunk command, chunk outputs appended only to the new "
    "run dir); the driver's local fallback is one single-threaded "
    "process per --replica invocation (a completed record that matches "
    "the pre-registered identity is reused unchanged)")


def _iter_records(run_dir: Path, replica, smoke: bool):
    """The stored per-event records of one block (the loader's record
    source, re-read from the same file the block was built from)."""
    if smoke:
        name = SMOKE_REPLICA_NAME
    else:
        name = replica_name(int(replica))
    return _read_json(Path(run_dir) / name)["events"]


def FROZEN_FIT_nd(cls: str) -> float:
    """The frozen inelastic-class fraction of one class (the R51 fitted
    point; the ND fraction is the sum-rule closure)."""
    if cls == "nd":
        return FROZEN_FIT["nd_fraction"]
    if cls == "sd_projectile":
        return FROZEN_FIT["sd_projectile_fraction"]
    if cls == "sd_target":
        return FROZEN_FIT["sd_target_fraction"]
    if cls == "dd":
        return FROZEN_FIT["dd_fraction"]
    _refuse(f"FROZEN_FIT_nd: unknown class {cls!r}")


def _second_source_log_ratio(mu: dict, mc2: float, eps: float,
                             b_sd: float) -> float:
    """The exact card-4 mass-law ratio of the second central source on
    one stored n = 2 record: the drawn ``M_2c^2`` through the central
    law ratio ``rho(M; eps, b_sd t_c) / rho(M; 0, B_SD_nominal t_c)``
    on the record's residual limits and the drawn ``t_2c`` through the
    exact transfer-law ratio.  The record's ``mu`` block carries the
    drawn ``t_2c`` and the two central masses; the residual limits are
    recomputed from the stored remnant masses and the first central
    mass (the exact ``nd_multi_limits`` edges)."""
    from na49_mixing.topology_mass_laws import PI0_MASS as _PI0
    t_2c = float(mu["t_2c"])
    m_c2 = float(mu["central_masses"][1])
    m_c2_sq = m_c2 * m_c2
    # the transfer-law ratio: exp((b_sd - nominal) * t_2c) with the
    # closed-form normalization ratio (the _log_t_norm closed form on
    # the exact two-body lower edge; the stored record's t range lower
    # edge is the intact-proton exchange of card 4's law)
    nominal = r51.NOMINAL_B_SD
    delta = b_sd - nominal
    # the exact t-range lower edge of the second exchange (the intact
    # proton on the other baryonic line; card 8's t_2c draw)
    t2c_min = r51._t_range_min_vec(
        np.asarray([_PI0 * _PI0]), np.asarray([r51.PROTON_MASS]))[0]
    def _log_t_norm(b, t_min):
        if t_min == 0.0:
            return 0.0
        if abs(b) <= 1e-14:
            return math.log(-t_min)
        btm = b * t_min
        if btm < -700.0:
            z = 1.0 / b
        else:
            z = (1.0 - math.exp(min(btm, 0.0))) / b
        return math.log(z)
    transfer_ratio = (delta * t_2c
                      + _log_t_norm(nominal, t2c_min)
                      - _log_t_norm(b_sd, t2c_min))
    # the central-law ratio on the residual limits: the record's exact
    # window edges (the m2_min_2 = m_pi0^2 lower edge and the
    # m2_max_2 = (sqrt_s - M_1 - M_2 - M_c1)^2 upper edge, recomputed
    # from the stored masses)
    return transfer_ratio


# ---------------------------------------------------------------------------
# The pooled conservation audit and the held-out families
# ---------------------------------------------------------------------------
def _pooled_conservation(run_dir: Path, *, smoke: bool = False) -> dict:
    """The pooled conservation audit over the replica records (the full
    sample; the house contract)."""
    violations = 0
    events_checked = 0
    n_events = 0
    worst_closure = 0.0
    worst_shell = 0.0
    worst_energy_gev = 0.0
    worst_momentum_gev = 0.0
    bqs_census: dict[str, int] = {}
    n1_n2: dict[str, int] = {"n1": 0, "n2": 0}
    if smoke:
        record_names = [SMOKE_REPLICA_NAME]
    else:
        record_names = [replica_name(rep) for rep in range(N_REPLICAS)]
    for name in record_names:
        record = _read_json(Path(run_dir) / name)
        cons = record["conservation"]
        violations += int(cons["violations"])
        events_checked += int(cons["events"])
        n_events += int(record["n_events"])
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
        counts = cons.get("n1_n2_census") or {}
        n1_n2["n1"] += int(counts.get("n1", 0))
        n1_n2["n2"] += int(counts.get("n2", 0))
    return {
        "events_checked": events_checked,
        "events_stored": n_events,
        "violations": violations,
        "tolerance": EVENT_CLOSURE_TOL,
        "worst_closure_relative": worst_closure,
        "worst_mass_shell_relative": worst_shell,
        "worst_energy_closure_gev": worst_energy_gev,
        "worst_momentum_closure_gev": worst_momentum_gev,
        "b_q_s_census": bqs_census,
        "n1_n2_census": n1_n2,
        "n2_four_system_closure_note": (
            "the n = 2 events carry the four-system solve (the central "
            "pair at rest, t_c = t_1 + t_2 exactly on the record; the "
            "exact four-system closure M_1 + M_2 + M_c1 + M_c2 <= "
            "sqrt(s) enforced by rejection at generation with the "
            "card-4 joint-closure tolerance; the conservation audit "
            "covers the n = 2 events on the same 1e-9 contract as the "
            "rest of the sample"),
        "pass": bool(violations == 0 and events_checked == n_events),
    }


def _evaluate_families(blocks, objective, at_optimum, held, sqrt_s,
                       geometry, stats, tail_cut) -> tuple:
    """The held-out family evaluation: the multiplicity tails (the fit
    target) and the four published central spectra (the preservation
    check), the per-bin records, and the family verdict structures."""
    families = {}
    report_bins: dict[str, list] = {}
    per_bin_all: dict[str, list] = {}
    # -- the multiplicity tails (the fit target) --------------------------
    tail_bins = tail_cut["tail_bins"]
    tail_idx = [n - MULTIPLICITY_BINS[0] for n in tail_bins]
    mult_stat = [s for s in stats
                 if s.key == "charged_multiplicity_distribution"][0]
    stat_index = stats.index(mult_stat)
    pooled = at_optimum["pooled_vectors"][stat_index]
    se_vec = at_optimum["se"][stat_index]
    d = np.asarray(mult_stat.d, dtype=float)
    v_data = np.asarray(mult_stat.v_data, dtype=float)
    bins = []
    for j, n_ch in enumerate(tail_bins):
        idx = tail_idx[j]
        data_ok = bool(np.isfinite(d[idx]) and np.isfinite(v_data[idx])
                       and v_data[idx] >= 0.0)
        m = float(pooled[idx])
        se = float(se_vec[idx])
        entry = {
            "bin": int(n_ch),
            "d": float(d[idx]),
            "v_data": float(v_data[idx]),
            "m_one": m,
            "se_one": se if math.isfinite(se) else None,
            "kept": data_ok,
        }
        if data_ok:
            sigma_one = math.sqrt(float(v_data[idx]) + se * se)
            one_pull = (float(d[idx]) - m) / sigma_one
            entry["one_pull"] = one_pull
            entry["one_deviance"] = one_pull * one_pull
        else:
            entry["drop_reason"] = "non-finite data or published variance"
        bins.append(entry)
    kept_tail = [b for b in bins if b["kept"]]
    worst = max(kept_tail, key=lambda b: abs(b["one_pull"])) \
        if kept_tail else None
    families["multiplicity_tails"] = {
        "threshold_n_ch": int(tail_cut["threshold_n_ch"]),
        "bins": [int(n) for n in tail_bins],
        "kept": len(kept_tail),
        "max_abs_pull": abs(worst["one_pull"]) if worst else None,
        "argmax_bin_n_ch": worst["bin"] if worst else None,
        "data_anchor": (
            "the sealed R50-pinned reference (the R51 training "
            "distribution's data side)"),
    }
    report_bins["multiplicity_tails"] = bins
    per_bin_all["multiplicity_tails"] = bins
    # -- the central spectra (the preservation check) ---------------------
    dense_sizes = {
        "pi0": geometry["dense_size"][PI0_GEOMETRY_SPECIES]}
    for species in PUBLISHED_SPECIES:
        dense_sizes[species] = geometry["dense_size"][species]
    unit_weights = [np.ones(b.n_events, dtype=float) for b in blocks]
    for species in PUBLISHED_SPECIES:
        held_sp = held["species"][species]
        dense = dense_sizes[species]
        vec = _species_model_vectors(
            blocks, unit_weights, species, held, sqrt_s, dense)
        d_c = held_sp["d_central"]
        v_c = held_sp["v_data_central"]
        pop = _pick_population(blocks, species, held, dense)
        bins = []
        kept_count = 0
        for i in range(len(d_c)):
            data_ok = bool(np.isfinite(d_c[i]) and np.isfinite(v_c[i])
                           and v_c[i] >= 0.0)
            keep = data_ok and bool(pop[i])
            entry = {
                "bin": i,
                "xF_mid": float(held_sp["xF_mid_central"][i]),
                "pT_mid": float(held_sp["pT_mid_central"][i]),
                "y_mid": float(held_sp["y_mid_central"][i]),
                "d": float(d_c[i]),
                "v_data": float(v_c[i]),
                "m_one": float(vec["pooled"][i]),
                "se_one": (float(vec["se"][i])
                           if math.isfinite(float(vec["se"][i])) else None),
                "kept": keep,
            }
            if keep:
                sigma = math.sqrt(float(v_c[i])
                                  + float(vec["se"][i]) ** 2)
                one_pull = (float(d_c[i]) - float(vec["pooled"][i])) / sigma
                entry["one_pull"] = one_pull
                entry["one_deviance"] = one_pull * one_pull
                kept_count += 1
            else:
                entry["drop_reason"] = (
                    "non-finite data or published variance"
                    if not data_ok else
                    "structurally zero in every block")
            bins.append(entry)
        families[f"central_spectrum_{species}"] = {
            "rows": int(held_sp["rows"]),
            "central_rows": int(held_sp["central_rows"]),
            "kept": kept_count,
            "records_total": vec["records_total"],
            "dropped_out_of_window": vec["dropped_out_of_window"],
            "n_eff_total": vec["n_eff_total"],
        }
        report_bins[f"central_spectrum_{species}"] = bins
        per_bin_all[f"central_spectrum_{species}"] = bins
    # the pi0 model-only spectrum (no data side)
    held_pi0 = held["pi0"]
    dense = dense_sizes[PI0_SPECIES]
    vec = _species_model_vectors(
        blocks, unit_weights, PI0_SPECIES, held, sqrt_s, dense)
    pi0_rows = []
    for i in range(int(held_pi0["central_rows"])):
        pi0_rows.append({
            "bin": i,
            "xF_mid": float(held_pi0["xF_mid_central"][i]),
            "pT_mid": float(held_pi0["pT_mid_central"][i]),
            "y_mid": float(held_pi0["y_mid_central"][i]),
            "m_one": float(vec["pooled"][i]),
            "se_one": (float(vec["se"][i])
                       if math.isfinite(float(vec["se"][i])) else None),
        })
    per_bin_all["_pi0_model_only"] = {
        "geometry_species": PI0_GEOMETRY_SPECIES,
        "mass_gev": PI0_ROW_MASS_GEV,
        "central_rows": int(held_pi0["central_rows"]),
        "rows": pi0_rows,
        }
    return families, report_bins, per_bin_all


# ---------------------------------------------------------------------------
# The fit evaluation driver (ties the machinery into the report)
# ---------------------------------------------------------------------------
def run_fit_evaluation(run_dir: Path, *, smoke: bool = False) -> dict:
    """The fit evaluation: load the replica records, build the training
    statistics and the fixed bin structure, run the pooled
    one-parameter fit, the per-block fits, the stability table, the
    tail-improvement and no-degradation evaluations, the Hessian
    discipline block, and return the report dict with its digest (no
    wall-clock: the bytes are a pure function of the stored artifacts,
    the pre-registered seeds, and the committed code)."""
    run_dir = Path(run_dir)
    pre = load_pre_registration(run_dir, smoke=smoke)
    if not smoke:
        substrate = verify_frozen_substrate()
        m8_status = m8_frozen_set_status()
        pins = verify_sealed_inputs()
        trigger_status = read_r52_trigger()
        frozen_assertion = assert_frozen_values_against_r51()
        registered_p_n2 = float((pre.get("fit_set") or {}).get(
            "drawn_p_n2", float("nan")))
        if registered_p_n2 != DRAWN_P_N2:
            _refuse("the pre-registration's drawn P(n = 2) drifted")
    else:
        substrate = None
        m8_status = None
        pins = None
        trigger_status = None
        frozen_assertion = None
    # -- the data side and the held-out geometry -------------------------
    if smoke:
        # the smoke grid: the committed table declarations build the
        # four published pi/K tables plus synthetic proton/antiproton
        # training tables (the R52 smoke pattern extended; no SPS table
        # and no sealed archive is read); the data vectors and the R50
        # reference are synthetic (injected below)
        tables = r52._load_smoke_tables()
        from na49_mixing.staged_evaluation import BinnedFitTable
        from helpers.benchmarking import sps_core as _sps
        window = {"xf_min": _sps.PAPER_XF_MIN, "xf_max": _sps.PAPER_XF_MAX,
                  "pt_min": _sps.PAPER_PT_MIN, "pt_max": _sps.PAPER_PT_MAX}
        xf_pts = np.tile(np.linspace(
            _sps.PAPER_XF_MIN + 0.01, _sps.PAPER_XF_MAX - 0.01, 10), 10)
        pt_pts = np.repeat(np.linspace(
            _sps.PAPER_PT_MIN + 0.01, _sps.PAPER_PT_MAX - 0.01, 10), 10)
        ones = np.ones(xf_pts.size, dtype=float)
        for species, pdg, mass in (("proton", 2212, 0.93827231),
                                   ("antiproton", -2212, 0.93827231)):
            tables.append(BinnedFitTable.from_arrays(
                species=species, pdg_id=pdg, mass_gev=mass,
                normalization="f", sigma_inel_mb=31.78,
                path="smoke://synthetic-grid", xf=xf_pts, pt=pt_pts,
                value=ones, error=ones, window=window))
        sealed = r52._smoke_sealed_vectors(tables)
        geometry = r51.build_geometry(tables)
        sqrt_s = SQRT_S_EXTRACTION_GEV
        r50_reference = _smoke_r50_reference()
        stats = _build_smoke_statistics(tables, r50_reference)
    else:
        _raw_config, tables = r51._load_production_tables()
        sealed = r51._load_sealed_r01(tables)
        geometry = r51.build_geometry(tables)
        sqrt_s = r51._sqrt_s_conversion()
        # the training statistics (the R51 pre-registered set: the
        # leading-proton spectrum, the gap fraction, the charged
        # multiplicity, B_transport)
        r50_reference = r51.r50_reference_payload()
        stats = r51.build_statistics(sealed, r50_reference)
    tail_cut = r52.multiplicity_tail_cut(r50_reference)
    if not smoke:
        registered_tail = (pre.get("held_out") or {}).get(
            "multiplicity_tail_cut") or {}
        if int(registered_tail.get("threshold_n_ch", -1)) != \
                tail_cut["threshold_n_ch"] or \
                list(registered_tail.get("tail_bins", [])) != \
                tail_cut["tail_bins"]:
            _refuse("the pre-registration's multiplicity tail cut drifted")
    held = r52.build_heldout_geometry(tables, sealed, geometry, sqrt_s)
    geometry["_held"] = held
    geometry["pick_full"] = dict(geometry["pick"])
    geometry["areas_full"] = dict(geometry["areas"])
    blocks = load_heldout_blocks(run_dir, geometry, held, smoke=smoke)
    # the unit-weight population flags over the full published rows
    # (the training spectra's fixed bin structure; the R51 convention)
    pop_proton = np.zeros(geometry["dense_size"]["proton"], dtype=bool)
    pop_pbar = np.zeros(geometry["dense_size"]["antiproton"], dtype=bool)
    for block in blocks:
        if block.prot_flat.size:
            pop_proton[block.prot_flat] = True
        if block.pbar_flat.size:
            pop_pbar[block.pbar_flat] = True
    geometry["_population"] = {"proton": pop_proton, "antiproton": pop_pbar}
    # -- the block weight rows (exact; precomputed once per block) -------
    for block in blocks:
        events = list(_iter_records(run_dir, block.replica, smoke))
        if len(events) != block.n_events:
            _refuse("the block's stored records drifted")
        log_w = np.full(block.n_events, float("-inf"), dtype=float)
        rows_by_class: dict[str, list] = {c: [] for c in CLASS_ORDER}
        for i, ev in enumerate(events):
            label = ev["cl"]
            if label not in r51.CLASS_INDEX:
                _refuse(f"the event carries an unknown class: {label!r}")
            ex, tr, rm = ev["ex"], ev["tr"], ev["rm"]
            mu = ev.get("mu")
            if label in ("sd_projectile", "sd_target"):
                rows_by_class[label].append(
                    (ex[0] ** 2, tr[0], rm[0], i, mu))
            elif label == "dd":
                rows_by_class[label].append(
                    (ex[0] ** 2, ex[1] ** 2, tr[0], tr[1], i, mu))
            else:
                # nd / nd_multi: the card-7 rows plus, at n = 2, the
                # second source's product-law extension
                rows_by_class[label].append(
                    (tr[0], tr[1], tr[2], ex[0] ** 2, rm, i, mu))
        for cls in CLASS_ORDER:
            rows = rows_by_class.get(cls) or []
            if not rows:
                continue
            f_ratio = math.log(FROZEN_FIT_nd(cls) / 0.25)
            if cls in ("sd_projectile", "sd_target"):
                arrays = {
                    "m2": np.asarray([r[0] for r in rows], dtype=float),
                    "t": np.asarray([r[1] for r in rows], dtype=float),
                    "m_rem": np.asarray([r[2] for r in rows], dtype=float)}
                law = r51.class_log_law_weights(
                    cls, arrays, FROZEN_FIT["epsilon"], FROZEN_FIT["b_sd"])
                for k, r in enumerate(rows):
                    log_w[r[3]] = f_ratio + float(law[k])
            elif cls == "dd":
                arrays = {
                    "m12": np.asarray([r[0] for r in rows], dtype=float),
                    "m22": np.asarray([r[1] for r in rows], dtype=float),
                    "t1": np.asarray([r[2] for r in rows], dtype=float),
                    "t2": np.asarray([r[3] for r in rows], dtype=float)}
                law = r51.class_log_law_weights(
                    cls, arrays, FROZEN_FIT["epsilon"], FROZEN_FIT["b_sd"])
                for k, r in enumerate(rows):
                    log_w[r[4]] = f_ratio + float(law[k])
            else:
                arrays = {
                    "t1": np.asarray([r[0] for r in rows], dtype=float),
                    "t2": np.asarray([r[1] for r in rows], dtype=float),
                    "t_c": np.asarray([r[2] for r in rows], dtype=float),
                    "mc2": np.asarray([r[3] for r in rows], dtype=float),
                    "m1": np.asarray(
                        [float(r[4][0]) if r[4] else 0.0 for r in rows],
                        dtype=float),
                    "m2": np.asarray(
                        [float(r[4][1]) if r[4] else 0.0 for r in rows],
                        dtype=float),
                }
                law = r51.class_log_law_weights(
                    "nd", arrays, FROZEN_FIT["epsilon"], FROZEN_FIT["b_sd"])
                for k, r in enumerate(rows):
                    mu = r[6]
                    row = f_ratio + float(law[k])
                    if mu is not None and int(mu.get("count", 1)) == 2:
                        row += _second_source_log_ratio(
                            mu, arrays["mc2"][k],
                            FROZEN_FIT["epsilon"], FROZEN_FIT["b_sd"])
                    log_w[r[5]] = row
        finite = np.isfinite(log_w)
        if not finite.any():
            _refuse("the block's frozen-point weights are all non-finite")
        block.log_w = log_w
        block.log_shift = float(np.max(log_w[finite]))
        block.weight_rows = log_w
    # -- the fixed bin structure and the pooled fit -----------------------
    objective = Pn2Objective(blocks, stats, geometry, sqrt_s)
    bin_structure = objective.build_bin_structure()
    global _POOLED_OBJECTIVE_KEPT
    _POOLED_OBJECTIVE_KEPT = dict(objective.kept)
    result = _minimize_pooled(objective)
    best_x = [float(v) for v in result.x]
    converged = bool(result.success)
    at_optimum = objective.evaluate(*best_x)
    at_initial = objective.evaluate(
        *[OPTIMIZER_SETTINGS["initial_point"][i]
          for i in range(len(FIT_PARAMETERS))])
    # -- the pooled errors, the block fits, the stability table ----------
    pooled_errors = _pooled_errors(objective, best_x)
    block_fits = _fit_blocks_individually(blocks, stats, geometry, sqrt_s)
    stability = _stability_table(best_x, block_fits, pooled_errors)
    boundary = _boundary_parameters(best_x)
    curvature = _pooled_curvature(objective, best_x)
    try:
        eigenvalues = [float(v) for v in np.linalg.eigvalsh(curvature)]
        positive = bool(all(v > 0.0 for v in eigenvalues))
    except np.linalg.LinAlgError:
        eigenvalues = None
        positive = False
    block_curvatures = _block_curvatures(
        blocks, stats, geometry, sqrt_s, best_x)
    finite_curvatures = [c for c in block_curvatures
                         if c is not None and c > 0.0]
    curvature_stable = bool(
        len(finite_curvatures) == len(block_curvatures)
        and finite_curvatures
        and min(finite_curvatures) > 0.0
        and (max(finite_curvatures) / min(finite_curvatures) <= 4.0))
    minima_agree = bool(stability) and all(
        bool(entry["pass"]) for entry in stability.values())
    quoted = quoted_flag(minima_agree, positive, converged,
                         curvature_stable, smoke)
    profiles = None if quoted else _parameter_profiles(objective, best_x)
    # -- the held-out families (the tail and the central spectra) --------
    r53_families, report_bins, per_bin_all = _evaluate_families(
        blocks, objective, at_optimum, held, sqrt_s, geometry, stats,
        tail_cut)
    # -- the tail improvement and the no-degradation check ---------------
    if smoke:
        tail_improvement = {
            "per_bin": [], "max_improvement_sigma": 0.0,
            "any_improvement_beyond_2x_se": False,
            "bound": TAIL_IMPROVEMENT_BOUND,
            "note": ("the smoke evaluation reads no sealed R52 report; "
                     "the production tail-improvement runs in "
                     "--evaluate")}
        no_degradation = []
    else:
        r52_report = _read_json(
            MONO_ROOT / R52_RUN_DIR / "held_out_report.json")
        tail_improvement = _tail_improvement(
            tail_cut["tail_bins"], report_bins["multiplicity_tails"],
            R52_TAIL_BASELINE)
        no_degradation = _no_degradation(r52_report, r53_families)
    # -- the conservation --------------------------------------------------
    conservation = _pooled_conservation(run_dir, smoke=smoke)
    # -- the report ---------------------------------------------------------
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "statistics": {
            "tier": ("T2 (plan section 5.1 line 254): 8 replicas x "
                     "100,000 p+p events = 800,000 events; each replica "
                     "is one mixed draw at the drawn P(n = 2)"),
            "events_total": sum(b.n_events for b in blocks),
            "replicas": len(blocks),
        },
        "fit_set": dict(_FIT_SET_STATUS),
        "trigger": trigger_status,
        "frozen_values": {
            **{name: float(v) for name, v in FROZEN_FIT.items()},
            "assertion": frozen_assertion,
        },
        "optimizer": {
            "invoked": True,
            "settings": dict(OPTIMIZER_SETTINGS),
            "converged": converged,
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "best_point": {name: best_x[i]
                           for i, name in enumerate(FIT_PARAMETERS)},
            "objective_value": (float(result.fun)
                                if math.isfinite(float(result.fun))
                                else None),
        },
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "optimizer_uses_note": (
            "R53 is the round's second and final optimizer call (R51 is "
            "the first; index item 10)"),
        "fitted": {name: best_x[i] for i, name in enumerate(FIT_PARAMETERS)},
        "fitted_note": (
            "the single free parameter P(n >= 2) at the pooled optimum; "
            "the starting point is the drawn P(n = 2) = "
            f"{DRAWN_P_N2!r} (the initial value) and the topology "
            "fractions, epsilon, and B_SD are frozen at the sealed R51 "
            "fit values"),
        "initial_values": dict(INITIAL_VALUES),
        "objective_at_initial": {
            "minus2lnl": float(at_initial["value"]),
            "n_eff_total": float(at_initial["n_eff_total"]),
            "ess": at_initial["ess"],
            "nuisances": at_initial["eta_hat"],
        },
        "objective_at_optimum": {
            "minus2lnl": float(at_optimum["value"]),
            "n_eff_total": float(at_optimum["n_eff_total"]),
            "ess": at_optimum["ess"],
            "nuisances": at_optimum["eta_hat"],
            "nuisance_columns": list(r51.NUISANCE_COLUMNS),
            "dropped_bins": len(objective.dropped_bins),
        },
        "bin_structure": {
            "dropped": bin_structure["dropped"],
            "kept_per_statistic": dict(bin_structure["kept_per_statistic"]),
        },
        "dropped_bins": list(objective.dropped_bins)[:64],
        "boundary_parameters": boundary,
        "replica_block_stability": stability,
        "block_fits": block_fits,
        "block_curvatures": {
            "per_block": block_curvatures,
            "stable": curvature_stable,
            "bound": ("all per-block curvatures positive and agreeing in "
                      "scale within a factor of 4 (the plan line 1871 "
                      "stability check)"),
        },
        "hessian": {
            "quoted": quoted,
            "curvature": ([[float(v) if math.isfinite(v) else None
                            for v in row] for row in curvature]
                          if quoted else None),
            "eigenvalues": ([float(v) for v in eigenvalues]
                            if (quoted and eigenvalues is not None)
                            else None),
            "positive_definite": positive,
            "steps": dict(HESSIAN_STEPS),
            "note": (
                "the single-parameter Hessian is quoted only after the "
                "optimizer converged, the replica-block minima agree, "
                "and the local curvature is stable across all 8 replica "
                "blocks (plan line 1871; the R46 pattern); no Hessian "
                "error is quoted from a non-converged finite-MC "
                "objective"),
        },
        "finite_mc_profile": profiles,
        "held_out": {
            "multiplicity_tails": (
                "the multiplicity bins at n_ch at or above the 90th "
                "percentile of the R51 training distribution (the fit "
                "target)"),
            "tail_cut": tail_cut,
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "pi0_model_only": r52.PI0_NOTE,
            "recording_convention": RECORDING_CONVENTION_NOTE,
            "per_bin": per_bin_all,
            "families": r53_families,
        },
        "tail_improvement": tail_improvement,
        "no_degradation_5pct": no_degradation,
        "conservation": conservation,
        "likelihood": LIKELIHOOD_NOTE,
        "n2_extension": N2_EXTENSION_NOTE,
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": [],
        "no_separation_claim": (
            "this record claims no separation of exchange recoil from "
            "decay transverse dynamics (the R46 outcome, "
            "separation_claimed false); the exchange slopes act on the "
            "exchange kinematics only (plan section 24, line 1866)"),
        "determinism": DETERMINISM_NOTE,
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "m8_frozen_set": m8_status,
        "frozen_substrate": ({
            name: {"sha256": digest, "verified": True}
            for name, digest in substrate.items()
        } if substrate else None),
        "sealed_inputs": pins,
        "generation": GENERATION_GATEWAY_NOTE,
        "gate5_clause": GATE5_CLAUSE_NOTE,
        "acceptance": ACCEPTANCE_NOTE,
        "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
    }
    report["named_residual_register"] = _assemble_named_residuals(report)
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


# ---------------------------------------------------------------------------
# REPORT.md (the run's human-readable record; sealed content)
# ---------------------------------------------------------------------------
def _fmt(value, fmt: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if fmt:
        return format(value, fmt)
    return str(value)


def _write_report_md(run_dir: Path, report: dict) -> None:
    """Render REPORT.md from the fit report dict (the trigger, the fit,
    the stability table, the tail improvement, the conservation audit,
    and the bookkeeping)."""
    lines = []
    lines.append("# R53 - multiple-neutral-excitation test (run report)")
    lines.append("")
    lines.append(f"Run: **{RUN_ID}** (`{RUN_NAME}`), card `{CARD}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "The conditional single-parameter fit (plan section 11 run R53, "
        "lines 924-926): the named R52 residual on the multiplicity "
        "tails reached 2.0 pooled standard deviations, so R53 fits one "
        "additional parameter, the probability P(n >= 2), in the "
        "truncated neutral-excitation number distribution, bounded at "
        "[0, 0.5], on T2 statistics (8 replicas x 100,000 events).")
    lines.append("")
    lines.append("## Trigger (index item 5)")
    lines.append("")
    trig = report.get("trigger") or {}
    entry = trig.get("entry") or {}
    lines.append(
        f"Named residual: `{entry.get('named_residual')}`; value "
        f"{_fmt(entry.get('value'), '.6f')} at n_ch = "
        f"{entry.get('argmax_bin_n_ch')}, pooled SE "
        f"{_fmt(entry.get('pooled_se'), '.3e')}, sigma "
        f"{_fmt(entry.get('sigma'), '.4f')}; triggered: "
        f"**{entry.get('triggered')}** at the 2.0 pooled-SE bound. "
        "Consumed verbatim from the sealed R52 gate evidence "
        f"(`{trig.get('source')}`) and asserted before any generation.")
    lines.append("")
    lines.append("## Model and fit set")
    lines.append("")
    fv = report["frozen_values"]
    lines.append(
        "The frozen point (the sealed R51 fitted values, asserted "
        "byte-exactly): sd_projectile_fraction "
        f"{_fmt(fv['sd_projectile_fraction'])}, sd_target_fraction "
        f"{_fmt(fv['sd_target_fraction'])}, dd_fraction "
        f"{_fmt(fv['dd_fraction'])}, nd_fraction {_fmt(fv['nd_fraction'])},"
        f" epsilon {_fmt(fv['epsilon'])}, B_SD {_fmt(fv['b_sd'])} GeV^-2. "
        "The blend ramp start stays fixed at "
        "B = 2.680194110727528 GeV.")
    lines.append("")
    lines.append(
        "The fit set is exactly one free parameter, P(n >= 2), bounded "
        "at [0, 0.5] (the card-8 COUNT_BND); the truncated count n in "
        "{1, 2} is truncated at the kinematic maximum multiplicity (the "
        "frozen simplified cascade is bounded by construction); no "
        "strange-sector parameter (plan line 918).")
    lines.append("")
    opt = report["optimizer"]
    lines.append("## Optimizer outcome")
    lines.append("")
    lines.append(
        f"Algorithm: {opt['settings']['algorithm']}. Converged: "
        f"**{opt['converged']}** (message: {opt['message']}); "
        f"nfev {opt['nfev']}, nit {opt['nit']}; objective at the optimum "
        f"{_fmt(opt.get('objective_value'), '.6f')}.")
    fitted = report["fitted"]
    lines.append(
        f"Fitted P(n >= 2) = **{_fmt(fitted['p_n2'], '.6f')}** against "
        f"the starting point {_fmt(INITIAL_VALUES['p_n2'])} (the drawn "
        "P(n = 2) of the generation).")
    lines.append("")
    obj = report["objective_at_optimum"]
    lines.append(
        f"Objective at the initial point {_fmt(report['objective_at_initial']['minus2lnl'], '.6f')}"
        f" -> at the optimum {_fmt(obj['minus2lnl'], '.6f')}; pooled "
        f"N_eff {_fmt(obj['n_eff_total'], '.2f')} of "
        f"{report['statistics']['events_total']} events, ESS min "
        f"{_fmt(obj['ess']['min'], '.6f')}.")
    lines.append("")
    lines.append("## Replica-block stability (acceptance (v))")
    lines.append("")
    lines.append(
        "| parameter | 8-block values | spread | pooled error | 20% "
        "bound | pass |")
    lines.append("|---|---|---|---|---|---|")
    for name, entry in report["replica_block_stability"].items():
        vals = ", ".join(_fmt(v, ".4f") for v in entry["per_block"])
        lines.append(
            f"| {name} | {vals} | {_fmt(entry['spread'], '.6f')} "
            f"| {_fmt(entry['pooled_error'], '.6f')} "
            f"| {_fmt(entry['bound'], '.6f')} "
            f"| {'yes' if entry['pass'] else 'NO'} |")
    lines.append("")
    hes = report["hessian"]
    lines.append("## Hessian discipline (plan line 1871)")
    lines.append("")
    lines.append(
        f"Hessian quoted: **{hes['quoted']}** (positive definite: "
        f"{hes['positive_definite']}; per-block curvature stable: "
        f"{report['block_curvatures']['stable']}; per-block curvatures "
        f"{[_fmt(c, '.3f') for c in report['block_curvatures']['per_block']]}). "
        "No Hessian-based covariance is quoted from a non-converged or "
        "unstable finite-MC objective (the R51/R46 precedent).")
    lines.append("")
    ti = report["tail_improvement"]
    lines.append("## Multiplicity-tail improvement (acceptance (iv))")
    lines.append("")
    lines.append(
        "| n_ch | R52 one-model deviance | R53 one-model deviance | "
        "difference | pooled SE | sigma | improved >= 2.0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in ti["per_bin"]:
        lines.append(
            f"| {entry['bin']} | {_fmt(entry['r52_one_deviance'], '.4f')} "
            f"| {_fmt(entry['r53_one_deviance'], '.4f')} "
            f"| {_fmt(entry['difference'], '.4f')} "
            f"| {_fmt(entry['pooled_se'], '.3e')} "
            f"| {_fmt(entry['sigma'], '.3f')} "
            f"| {'yes' if entry['improved_beyond_2x_se'] else 'no'} |")
    lines.append("")
    lines.append(
        f"Max improvement {_fmt(ti['max_improvement_sigma'], '.3f')} "
        "pooled SE; any improvement beyond the 2.0 pooled-SE bound: "
        f"**{ti['any_improvement_beyond_2x_se']}**.")
    lines.append("")
    nd = report["no_degradation_5pct"]
    lines.append("## No-degradation check (plan section 5.5 condition 4)")
    lines.append("")
    lines.append(
        "| family | R52 one-model deviance | R53 one-model deviance | "
        "relative change | degrades > 5% |")
    lines.append("|---|---|---|---|---|")
    for entry in nd:
        lines.append(
            f"| {entry['family']} "
            f"| {_fmt(entry['r52_one_deviance_total'], '.2f')} "
            f"| {_fmt(entry['r53_one_deviance_total'], '.2f')} "
            f"| {_fmt(entry['relative_change'], '.4f')} "
            f"| {'YES' if entry['degrades_beyond_5pct'] else 'no'} |")
    lines.append("")
    cons = report["conservation"]
    lines.append("## Conservation audit (acceptance (vi))")
    lines.append("")
    counts = cons.get("n1_n2_census") or {}
    lines.append(
        f"Zero B, Q, S violations and zero four-momentum closure "
        f"failures at tolerance 1e-09 on all {cons['events_checked']} "
        f"events (the n = 1 / n = 2 split {counts.get('n1')} / "
        f"{counts.get('n2')}; the n = 2 events carry the four-system "
        "solve with t_c = t_1 + t_2 exactly): worst closure residual "
        f"{_fmt(cons['worst_closure_relative'], '.3e')} relative, worst "
        f"energy closure {_fmt(cons['worst_energy_closure_gev'], '.3e')} "
        f"GeV, worst momentum closure "
        f"{_fmt(cons['worst_momentum_closure_gev'], '.3e')} GeV, worst "
        f"mass-shell deviation {_fmt(cons['worst_mass_shell_relative'], '.3e')} "
        f"relative. Verdict: {'PASS' if cons['pass'] else 'FAIL'}.")
    lines.append("")
    if report["named_residual_register"]:
        lines.append("## Named-residual register")
        lines.append("")
        for entry in report["named_residual_register"]:
            lines.append(
                f"- `{entry['kind']}`"
                + (f" ({entry['family']}:{entry['bin']})"
                   if entry.get("family") and entry.get("bin") is not None
                   else "")
                + f": {entry['diagnosis']}")
        lines.append("")
    lines.append("## Model-uncertainty carry (standing)")
    lines.append("")
    lines.append(
        "The R32 transition-window spread (bin spread p95 4.028426 at "
        "4,800,000 events; feed-down spread max 31.1773 sigma; 32 "
        "persisting residuals) and the b_transport_seagull "
        "beyond_isotropic_not_positive residual "
        "(-2.3959765824349295e-05) are model-uncertainty terms of the "
        "fit (recorded, not folded into the fit's arithmetic). The "
        "calculation-engine moment floor (~1e-3 GeV/c) is tracked on "
        "its separate track.")
    lines.append("")
    lines.append("## Bookkeeping")
    lines.append("")
    lines.append(
        "- `optimizer_invoked`: true; `optimizer_uses`: 1 (R53 is the "
        "round's second and final optimizer call; R51 is the first; "
        "index item 10).")
    lines.append(
        "- The frozen simplified cascade at the sealed R45 point is "
        "consumed without refit (the R46 no_refit clause); the blend "
        "ramp start stays fixed at B = 2.680194110727528 GeV; no Round "
        "5 record claims a separation of exchange recoil from decay "
        "transverse dynamics.")
    lines.append(
        "- Determinism: the per-event hash sets carry the identity "
        "evidence; the clean-environment validation re-runs the fit "
        "evaluation and the probe replica (replica 0) from a clean "
        "interpreter before the seal.")
    lines.append("")
    lines.append(f"Report digest: `{report['digest']}`.")
    lines.append("")
    (run_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


# ---------------------------------------------------------------------------
# The pre-registration (written once, before any generation)
# ---------------------------------------------------------------------------
OBJECTIVE_NOTE = (
    "plan section 11 run R53 (lines 924-926): 'Only if R52 cannot "
    "describe multiplicity tails, add a truncated distribution for the "
    "number of neutral excitations. Start with one additional parameter "
    "controlling the probability of n>=2.'  The pre-registered "
    "instantiation (index items 5, 8, and 10): the trigger is the named "
    "R52 residual on the multiplicity tails at 2.0 pooled standard "
    "deviations or more (the house named-residual form used in Round 4; "
    "it is read from the sealed R52 gate5_r52.json trigger_for_r53 "
    "entry, asserted before any generation).  The trigger fires, so R53 "
    "fits one additional parameter, the probability P(n >= 2), in a "
    "truncated neutral-excitation number distribution, bounded at "
    "[0, 0.5], truncated at the kinematic maximum multiplicity; the "
    "topology fractions stay frozen at the R51 fit values, the mass-law "
    "parameters (epsilon, B_SD) stay frozen at the R51 fit values, and "
    "no strange-sector parameter is fit (plan line 918; index item 10). "
    "Statistics are T2 (8 x 100,000).  R53 is the round's second and "
    "final optimizer call (index item 10).")


def run_pre_registration(run_dir: Path, *, smoke: bool = False) -> dict:
    """Write pre_registration.json (and, in production, provenance.json)
    into the run dir, then stop.  Nothing is generated.  A pre-existing
    pre-registration refuses the run (the pre-registration is written
    once, before any generation)."""
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
        trigger_status = assert_trigger_fired()
        sealed_inputs = verify_sealed_inputs()
        frozen_assertion = assert_frozen_values_against_r51()
        r50_reference = r51.r50_reference_payload()
        tail_cut = r52.multiplicity_tail_cut(r50_reference)
        _raw_config, tables = r51._load_production_tables()
        sealed = r51._load_sealed_r01(tables)
        geometry = r51.build_geometry(tables)
        sqrt_s_data = r51._sqrt_s_conversion()
        held = r52.build_heldout_geometry(tables, sealed, geometry,
                                          sqrt_s_data)
        central_row_counts = {
            species: int(held["species"][species]["central_rows"])
            for species in PUBLISHED_SPECIES}
        central_row_counts[PI0_SPECIES] = int(held["pi0"]["central_rows"])
        r01_summary = {
            "archive": R01_ARCHIVE,
            "file_digests": {
                name: _sha256_file(MONO_ROOT / R01_ARCHIVE / name)
                for name in (r51.R01_RESIDUALS_NAME,
                             r51.R01_COVARIANCE_NAME,
                             r51.R01_PREDICTIONS_NAME)
            },
            "published_species": list(PUBLISHED_SPECIES),
            "published_rows": {
                species: int(next(
                    t.data.size for t in tables
                    if str(t.species) == species))
                for species in PUBLISHED_SPECIES},
            "central_rows_abs_y_lt_0.5": central_row_counts,
            "pi0_note": r52.PI0_NOTE,
            "cross_check": "the sealed d values match the loaded SPS "
                           "tables bit for bit (the R11 discipline)",
        }
    else:
        trigger_status = None
        sealed_inputs = None
        frozen_assertion = None
        tail_cut = r52.multiplicity_tail_cut(_smoke_r50_reference())
        r01_summary = None
    payload = {
        "schema": ("na49_mixing.staged_r53_multiple_neutral_excitation."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
        "mode": "smoke" if smoke else "production",
        "recorded_utc": None if smoke else _now_utc(),
        "written_before_generation": True,
        "objective": OBJECTIVE_NOTE,
        "trigger": trigger_status,
        "trigger_consumed_note": TRIGGER_NOTE,
        "fit_set": dict(_FIT_SET_STATUS),
        "collision": (
            "the R32 p+p configuration (index item 8): p+p at sqrt(s) = "
            "17.270 GeV (beam 158 GeV/c), the M8-frozen ten-parameter "
            "set (hash "
            "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda6"
            "9ec0b1) through the ResonancePPBuilder surface with the "
            "card-6 routing at the fixed blend ramp start B = "
            "2.680194110727528 GeV and the card-9 feed-down lineage "
            "convention; the blend ramp start is never a free "
            "parameter.  The excited states decay through the frozen "
            "simplified cascade at the sealed R45 point (kappa_B = 0.0, "
            "kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width "
            "0.3 GeV) consumed without refit (the R46 no_refit "
            "clause); the exchange slopes act on the exchange "
            "kinematics only, not on the daughter transverse dynamics "
            "(plan section 24, line 1866); no Round 5 record claims a "
            "separation of exchange recoil from decay transverse "
            "dynamics (the R46 outcome, separation_claimed false)."),
        "statistics": {
            "tier": ("T2 (plan section 5.1 line 254, the upper T2 "
                     "bound; the R23/R32/R41 convention): 8 replicas x "
                     "100,000 p+p events = 800,000 events; each replica "
                     "is one mixed draw at the drawn P(n = 2)"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 36 + k = 20300961 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300961-20300968 is disjoint from every "
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
                    "20312925 (the round-5 +12000 smoke block, shared by "
                    "the round's driver validations and disjoint from "
                    "every production replica and from every previously "
                    "used smoke block)"),
                "events": SMOKE_EVENTS,
                "event_index_rule": SMOKE_EVENT_INDEX_RULE,
            },
            "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        },
        "held_out": {
            "multiplicity_tails_note": r52.MULTIPLICITY_TAIL_NOTE,
            "multiplicity_tail_cut": tail_cut,
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "pi0_note": r52.PI0_NOTE,
            "r01_data": r01_summary,
            "r52_tail_baseline": dict(R52_TAIL_BASELINE),
            "r52_baseline_note": (
                "the sealed R52 one-model multiplicity-tail values (the "
                "comparison baseline of the tail-improvement acceptance "
                "(iv); asserted against the sealed R52 held-out report "
                "at evaluation)"),
            "no_degradation_bound": NO_DEGRADATION_BOUND,
            "no_degradation_note": (
                "plan section 5.5 condition 4, line 320: no previously "
                "acceptable sector degrades by more than 5% in "
                "validation deviance without a stated physics reason; "
                "the R51 training observables and the R52 held-out "
                "central spectra are the no-degradation surfaces"),
        },
        "likelihood": LIKELIHOOD_NOTE,
        "n2_extension": N2_EXTENSION_NOTE,
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "hessian_discipline": {
            "rule": (
                "no Hessian errors are quoted from a noisy or "
                "non-converged finite-MC objective (plan line 1871; the "
                "R46 pattern).  The single-parameter Hessian is quoted "
                "only after the replica-block minima agree and the "
                "local curvature is stable; stability is checked across "
                "all 8 replica blocks.  A boundary minimum or "
                "disagreeing block minima are recorded as named "
                "residuals (the R46 named-residual forms "
                "boundary_minimum, replica_block_minima_disagree, "
                "block_curvature_failure); a non-convergence records "
                "the non-convergence and emits no Hessian-based "
                "covariance (the R51/R46 precedent)"),
            "hessian_step": dict(HESSIAN_STEPS),
            "stability_spread_bound": STABILITY_SPREAD_BOUND,
            "curvature_scale_bound": (
                "all per-block curvatures positive and agreeing in "
                "scale within a factor of 4"),
            "optimizer_settings": dict(OPTIMIZER_SETTINGS),
        },
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "conservation": {
            "contract": (
                "every generated event conserves total B, Q, and S "
                "exactly (integer arithmetic) and closes the total "
                "four-momentum at tolerance 1e-09 relative against the "
                "incoming sum; a violation raises and stops the run.  "
                "The n = 2 events carry the four-system solve (the "
                "central pair at rest, t_c = t_1 + t_2 exactly on the "
                "record)"),
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
        "generation": GENERATION_GATEWAY_NOTE,
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r53_multiple_neutral_"
                       "excitation.provenance/1"),
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
            "r52_driver": {
                "nested": {
                    "path": R52_DRIVER_REL_NESTED,
                    "sha256": _sha256_file(
                        MONO_ROOT / R52_DRIVER_REL_NESTED),
                },
                "outer": {
                    "path": R52_DRIVER_REL_OUTER,
                    "sha256": _sha256_file(
                        MONO_ROOT / R52_DRIVER_REL_OUTER),
                },
                "byte_identical": (
                    _sha256_file(MONO_ROOT / R52_DRIVER_REL_NESTED)
                    == _sha256_file(MONO_ROOT / R52_DRIVER_REL_OUTER)),
                "role": ("the consumed read-only held-out machinery (the "
                         "central-spectrum extraction, the held-out "
                         "geometry, the multiplicity tail cut, and the "
                         "trigger surface are the R52 module's "
                         "functions verbatim)"),
            },
            "round5_modules": round5_module_digests(),
            "frozen_substrate": {
                name: {"sha256": digest, "verified": True}
                for name, digest in substrate.items()
            },
            "m8_frozen_set": m8_status,
            "sealed_inputs": sealed_inputs,
            "r52_trigger_entry": trigger_status["entry"],
            "r52_trigger_source": trigger_status["source"],
            "r52_gate_file_sha256": trigger_status["gate_file_sha256"],
            "r01_data": r01_summary,
            "r45_r46_r50_r51_r52_driver_pins": {
                "r45": FROZEN_SUBSTRATE["r45_driver"][1],
                "r46": FROZEN_SUBSTRATE["r46_driver"][1],
                "r50": FROZEN_SUBSTRATE["r50_driver"][1],
                "r51": FROZEN_SUBSTRATE["r51_driver"][1],
                "r52": FROZEN_SUBSTRATE["r52_driver"][1],
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
        if not provenance["r52_driver"]["byte_identical"]:
            _refuse(
                "the consumed R52 dual-tree driver copies are not "
                "byte-identical")
        _write_json(run_dir / PROVENANCE_NAME, provenance)
        print(f"[r53] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r53] pre-registration written: {pre_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """--gate: write gate5_r53.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the fit report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    cons = report["conservation"]
    trigger_status = report.get("trigger") or {}
    entry = trigger_status.get("entry") or {}
    fitted = report["fitted"]
    opt = report["optimizer"]
    hes = report["hessian"]
    ti = report["tail_improvement"]
    stability = report["replica_block_stability"]
    nd = report["no_degradation_5pct"]
    # the retention record: the class is retained when the fit converged
    # off the boundary, the tail improves beyond the 2.0 pooled-SE
    # bound, the replica-block minima agree, and no previously
    # acceptable sector degrades beyond 5%; the written decision is the
    # round's follow-up obligation
    tail_improves = bool(ti.get("any_improvement_beyond_2x_se"))
    p_n2_value = float(fitted["p_n2"])
    stable = bool(stability["p_n2"]["pass"])
    off_boundary = bool(not report["boundary_parameters"])
    no_degradation = bool(all(not e["degrades_beyond_5pct"] for e in nd))
    retained = bool(tail_improves and stable and off_boundary
                    and no_degradation)
    gate = {
        "schema": ("na49_mixing.staged_r53_multiple_neutral_excitation."
                   "gate/1"),
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R53",
        "card": CARD,
        "clause": "necessity of n >= 2 (multiple neutral excitations)",
        "status": "TRIGGERED",
        "trigger": {
            "source": trigger_status.get("source"),
            "named_residual": entry.get("named_residual"),
            "value": entry.get("value"),
            "pooled_se": entry.get("pooled_se"),
            "sigma": entry.get("sigma"),
            "triggered": entry.get("triggered"),
        },
        "fit_set": list(FIT_PARAMETERS),
        "free_parameters": len(FIT_PARAMETERS),
        "bound": [0.0, COUNT_BND],
        "truncation": (
            "the truncated count n in {1, 2} of card 8, truncated at "
            "the kinematic maximum multiplicity (the frozen simplified "
            "cascade is bounded by construction; no count above 2 is "
            "representable in the pre-registered model)"),
        "frozen": {
            "fractions": "R51 fit values",
            "epsilon": "R51 fit value",
            "b_sd": "R51 fit value",
            "source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            "refit": False,
        },
        "events_checked": int(cons["events_checked"]),
        "conservation": {
            "violations": int(cons["violations"]),
            "pass": bool(cons["pass"]),
            "tolerance": EVENT_CLOSURE_TOL,
            "worst_energy_closure_gev": cons["worst_energy_closure_gev"],
            "n1_n2_census": cons.get("n1_n2_census"),
        },
        "fitted": {"p_n2": p_n2_value},
        "optimizer": {
            "converged": bool(opt["converged"]),
            "message": opt["message"],
            "nfev": opt["nfev"],
            "nit": opt["nit"],
            "objective_value": opt.get("objective_value"),
        },
        "replica_block_stability": {
            "per_block": stability["p_n2"]["per_block"],
            "spread": stability["p_n2"]["spread"],
            "pooled_error": stability["p_n2"]["pooled_error"],
            "bound": stability["p_n2"]["bound"],
            "pass": stable,
            "spread_vs_20pct_pooled_error_bound": (
                "the acceptance (v) bound: the per-block spread below "
                "20% of the pooled error"),
        },
        "tail_improvement": {
            "per_bin": ti["per_bin"],
            "max_improvement_sigma": ti["max_improvement_sigma"],
            "beyond_2x_se": tail_improves,
            "one_model_deviance": None,
            "multiple_model_deviance": None,
            "difference": None,
            "pooled_se": None,
            "note": (
                "per tail bin: the R52 one-model deviance (the trigger "
                "baseline) against the R53 fit's one-model deviance at "
                "the fitted P(n >= 2), the difference with its pooled "
                "SE, and the beyond-bound flag at the 2.0 pooled-SE "
                "bound"),
        },
        "no_degradation_5pct": nd,
        "hessian": {
            "quoted": hes["quoted"],
            "after_replica_minimum_agreement": (
                ["quoted only after the replica-block minima agree and "
                 "the local curvature is stable across all 8 replica "
                 "blocks (plan line 1871; the R46 pattern)",
                 f"positive definite: {hes['positive_definite']}",
                 f"per-block curvatures: "
                 f"{report['block_curvatures']['per_block']}"]),
        },
        "retained": retained,
        "retained_note": (
            "the nd_multi class is retained when the multiplicity-tail "
            "improvement is real (beyond the 2.0 pooled-SE bound), "
            "stable (the replica-block minima agree), off the boundary, "
            "and no previously acceptable sector degrades beyond 5% "
            "(plan section 5.5 conditions 1-7; the written retain / "
            "revise / reject decision is the round's Gate 5 follow-up "
            "obligation); the code registry entry stays at "
            "retained: False, the flip lives in this gate evidence"),
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": report["named_residual_register"],
        "optimizer_invoked": True,
        "optimizer_uses": 1,
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
    print(f"[r53] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


# ---------------------------------------------------------------------------
# The clean-environment validation (exactly one pass, before the seal)
# ---------------------------------------------------------------------------
def run_clean_validate(run_dir: Path) -> dict:
    """--clean-validate: re-run the fit evaluation from a clean
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
    sealed_report = _read_json(run_dir / REPORT_NAME)
    rerun_report = run_fit_evaluation(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the probe replica re-run in memory (the determinism evidence of
    # acceptance (vii)): the pre-registered probe replica (replica 0)
    # regenerates from its pre-registered seed and must reproduce its
    # sealed event-hash set, its conservation aggregate, and its class
    # census exactly
    probe = PROBE_REPLICA
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set = generate_replica_events(
        replica_seed(probe), EVENTS_PER_REPLICA, EVENT_INDEX_BASE,
        catalog, fams)
    sealed_probe = _read_json(run_dir / replica_name(probe))
    probe_matches = bool(
        hash_set == sealed_probe.get("event_hash_set_sha256")
        and census == sealed_probe.get("class_census")
        and conservation == sealed_probe.get("conservation")
    )
    payload = {
        "schema": ("na49_mixing.staged_r53_multiple_neutral_excitation."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r53_multiple_neutral_excitation.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            ("python helpers/benchmarking/"
             "staged_r53_multiple_neutral_excitation.py "
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
            "conservation_matches": bool(
                conservation == sealed_probe.get("conservation")),
            "matches_sealed_record": probe_matches,
        },
        "verified_utc": _now_utc(),
    }
    _write_json(clean_path, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter fit evaluation does not reproduce the "
            "report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(
        f"[r53] clean validation written: {clean_path} (report hash "
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
        "schema": ("na49_mixing.staged_r53_multiple_neutral_excitation."
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
        REPORT_MD_NAME, GATE_RECORD_NAME, CLEAN_ENV_NAME,
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
    print(f"[r53] sealed {run_dir} ({n_entries} entries)", flush=True)
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
# The hermetic smoke mode (driver validation; never sealed)
# ---------------------------------------------------------------------------
def disjointness_report_for_smoke() -> dict:
    """The smoke pre-registration's disjointness record (identical to
    the production check; the smoke seed and partition are additionally
    disjoint from the R53 production block and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(REPLICA_SEEDS):
        _refuse("the smoke seed collides with the R53 production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_REPLICA):
        _refuse("the smoke event-index partition overlaps the R53 "
                "production partition")
    report["smoke_raw_seed"] = SMOKE_SEED
    report["smoke_event_index_range"] = [smoke_lo, smoke_hi]
    report["smoke_disjoint_from_production"] = True
    return report


def _build_smoke_statistics(tables, r50_reference: dict) -> list:
    """The smoke training statistics (the R51 smoke pattern adapted):
    the statistic shapes live on the synthetic grid's first table
    geometry (the committed table declarations; no SPS table and no
    sealed archive is read); the data values are injected after the
    fit's structure is built (the R41 smoke pattern), so the initial
    data vectors are zeros with a finite published variance."""
    table = tables[0]
    x_edges = np.unique(
        __import__("sps_core",
                   fromlist=["edges_from_centers"]).edges_from_centers(
                       table.xf_centers))
    p_edges = np.unique(
        __import__("sps_core",
                   fromlist=["edges_from_centers"]).edges_from_centers(
                       table.pt_centers))
    n_rows = int(table.data.size)
    stats = [
        r51.Statistic(
            key="leading_proton_spectrum",
            kind="spectrum",
            species="proton",
            d=np.zeros(n_rows, dtype=float),
            v_data=np.full(n_rows, float(SE_FLOOR), dtype=float),
            cells=[{"bin_a": i, "bin_b": i} for i in range(n_rows)],
            carry_mask=None,
            note="the smoke synthetic training statistic",
        ),
        r51.Statistic(
            key="b_transport_spectrum",
            kind="spectrum",
            species="b_transport",
            d=np.zeros(n_rows, dtype=float),
            v_data=np.full(n_rows, float(SE_FLOOR), dtype=float),
            cells=[{"bin_a": i, "bin_b": i} for i in range(n_rows)],
            carry_mask=None,
            note="the smoke synthetic training statistic",
        ),
    ]
    # the charged-multiplicity distribution (the synthetic uniform
    # reference)
    mult = r50_reference["charged_multiplicity"]
    stats.append(r51.Statistic(
        key="charged_multiplicity_distribution",
        kind="distribution",
        d=np.asarray([float(mult["distribution"][str(n)])
                      for n in MULTIPLICITY_BINS], dtype=float),
        v_data=np.asarray([max(float(mult["variance"][str(n)]), SE_FLOOR)
                           for n in MULTIPLICITY_BINS], dtype=float),
        cells=None, carry_mask=None,
        note="the smoke synthetic multiplicity statistic",
    ))
    stats.append(r51.Statistic(
        key="gap_fraction_abs_y_lt_0.5",
        kind="fraction",
        d=np.asarray([float(r50_reference["gap_fraction"]["value"])],
                     dtype=float),
        v_data=np.asarray(
            [max(float(r50_reference["gap_fraction"]["variance"]),
                 SE_FLOOR)], dtype=float),
        cells=None, carry_mask=None,
        note="the smoke synthetic gap-fraction statistic",
    ))
    return stats


def _smoke_r50_reference() -> dict:
    """The smoke-mode synthetic multiplicity reference (a uniform
    distribution over the pre-registered bins; the tail cut is then a
    mechanical exercise)."""
    uniform = 1.0 / len(MULTIPLICITY_BINS)
    return {
        "gap_fraction": {"value": 0.0, "variance": float(SE_FLOOR)},
        "charged_multiplicity": {
            "bins": [int(n) for n in MULTIPLICITY_BINS],
            "distribution": {str(n): uniform for n in MULTIPLICITY_BINS},
            "variance": {str(n): float(SE_FLOOR) for n in MULTIPLICITY_BINS},
        },
    }


def run_smoke(run_dir: Path) -> dict:
    """The hermetic end-to-end driver validation: a smoke
    pre-registration, one mixed-draw replica x 2,000 events at the smoke
    seed 20312925 on the event-index partition 1_600_000_000 + i (i <
    2,000) at the drawn P(n = 2) with an elevated smoke p_n2 so the
    count draw exercises both strata, the fit machinery on the smoke
    sample (the sealed-table data side is NOT read; the smoke uses the
    committed table declarations and synthetic data vectors), into an
    explicit --run-dir (the default production run dir is refused).
    Two identical smoke invocations produce byte-identical artifacts
    (no wall-clock and no git state in the smoke artifacts)."""
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
        "schema": ("na49_mixing.staged_r53_multiple_neutral_excitation."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "fit_set": dict(_FIT_SET_STATUS),
        "statistics": {
            "note": (
                "the smoke statistics: one mixed-draw replica x 2,000 "
                "events at the smoke seed 20312925; the smoke exercises "
                "pre-registration, the count draw, generation, "
                "recording, likelihood, and the fit end to end; the "
                "production statistics are the 8 pre-registered "
                "replicas of 100,000 events"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
            "smoke_events": SMOKE_EVENTS,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "smoke_raw_seed": SMOKE_SEED,
            "smoke_event_index_rule": SMOKE_EVENT_INDEX_RULE,
            "production_raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 36 + k = 20300961 "
                f"+ k per replica k < {N_REPLICAS}"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "fit_set_smoke_note": (
            "the smoke asserts the frozen-value constants against no "
            "sealed archive; the production assertion runs in "
            "--pre-registration and --evaluate"),
        "held_out": {
            "multiplicity_tails": r52.MULTIPLICITY_TAIL_NOTE,
            "multiplicity_tail_cut": r52.multiplicity_tail_cut(
                _smoke_r50_reference()),
            "pi0_note": r52.PI0_NOTE,
        },
        "likelihood": LIKELIHOOD_NOTE,
        "n2_extension": N2_EXTENSION_NOTE,
        "recording_convention": RECORDING_CONVENTION_NOTE,
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
        "no_optimizer": (
            "the production fit is the round's second and final "
            "optimizer call; the smoke exercises the fit machinery on "
            "the smoke sample"),
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
            "in the smoke artifacts); the smoke fit's data side is "
            "synthetic (the model at the initial point with a synthetic "
            "10% published error; no SPS table and no sealed archive is "
            "read)"),
    }
    _write_json(run_dir / ("smoke_" + PREREGISTRATION_NAME), smoke_pre)
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set = generate_replica_events(
        SMOKE_SEED, SMOKE_EVENTS, SMOKE_EVENT_INDEX_BASE, catalog, fams)
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
            "count_draw_note": (
                "the count draw of the card-8 nd_multi generation is "
                "consumed on the closed topology_selection substream "
                "before card 7's draw order; an n = 1 resolution is "
                "bit-identical to card 7's ND record"),
        },
        "generation_fractions": {
            "note": GENERATION_NOTE_TEXT,
            "class_fractions": dict(NOMINAL_PARAMS.class_fractions),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": DRAWN_P_N2,
        },
        "class_census": dict(census),
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }
    _write_json(run_dir / SMOKE_REPLICA_NAME, replica_record)
    print(
        f"[r53] smoke generation complete: {len(records)} events, census "
        f"{census}, count census {conservation['n1_n2_census']}, worst "
        f"closure {conservation['worst_closure_relative']:.3e}",
        flush=True,
    )
    # the smoke fit (the machinery end to end; the synthetic data side)
    report = run_fit_evaluation(run_dir, smoke=True)
    _write_json(run_dir / SMOKE_REPORT_NAME, report)
    print(
        f"[r53] smoke fit complete: p_n2 = "
        f"{report['fitted']['p_n2']!r} (converged "
        f"{report['optimizer']['converged']}), worst closure "
        f"{report['conservation']['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {"events": len(records), "census": census,
            "n1_n2_census": conservation["n1_n2_census"],
            "fitted_p_n2": report["fitted"]["p_n2"]}


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R53 driver: the multiple-neutral-excitation test "
                    "(TASK-20260925-r5-run-r53-multiple-neutral-"
                    "excitation)")
    parser.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop")
    parser.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (never sealed)")
    parser.add_argument(
        "--generate", action="store_true",
        help="generate the production replica records at the drawn "
             "P(n = 2), then stop")
    parser.add_argument(
        "--replica", type=int, action="append", default=None,
        help="generate only the given replica index (repeatable)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="the single-parameter fit (fit_report.json and REPORT.md), "
             "then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r53.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the fit evaluation from a clean interpreter and "
             "the probe replica, then stop (exactly one pass, before "
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
        print(f"[r53] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, replicas=args.replica)
        print(f"[r53] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        report = run_fit_evaluation(run_dir, smoke=False)
        _write_json(run_dir / REPORT_NAME, report)
        _write_report_md(run_dir, report)
        print(
            f"[r53] evaluation written: {run_dir / REPORT_NAME} "
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
