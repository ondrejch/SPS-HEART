#!/usr/bin/env python3
"""R52 driver: the central-excitation necessity test (Round 5, card 12).

Task ``TASK-20260925-r5-run-r52-central-excitation-necessity`` (plan
section 11 run R52, lines 920-922; plan section 23 lines 1825-1827,
"Round 5 -- exchange topology").  The card executes run R52, the
frozen-parameter deviance comparison of the one-neutral-central-
excitation model (the R51 sealed events, reused) against the
zero-excitation model (the R51-frozen topology with the ND fraction
zeroed and the remaining three inelastic fractions renormalized,
regenerated at T2), on the held-out central pion and kaon spectra and
the multiplicity tails.  No optimizer: R52 is a frozen-parameter
deviance comparison, not a refit (index item 10); ``optimizer_invoked``
is false and no parameter of any model is fit here.

The two models (index item 4)
-----------------------------
- The ONE model is the R51-frozen topology at the sealed R51 fitted
  point (the fractions, epsilon, B_SD of the sealed
  ``r51_topology_fraction_fit_2026-09-27/fit_report.json``, asserted
  byte-exactly before any evaluation), represented by the R51 sealed
  events consumed read-only.  The R51 sample was generated at the
  nominal 0.25-each point, so the frozen-point model side carries the
  exact per-event importance weights of the R51 machinery (the mixture
  fraction ratio times the class mass-law density ratio; the R51
  module's weight functions are consumed verbatim so the one-side
  weights are THE R51 weights at the frozen point).
- The ZERO model is the same frozen topology with the ND
  central-excitation class fraction set to exactly 0.0 and the
  remaining three inelastic fractions (SD_p, SD_t, DD) renormalized to
  sum to one, regenerated at T2 (8 x 100,000) under the R52 zero-model
  seed block, with epsilon and B_SD at the frozen values and
  P(n >= 2) = 0.0 (the unretained card-8 default; the multi-neutral
  machinery stays off exactly as in R51).  The zero sample is generated
  at its own evaluation point, so its model side carries unit weights
  (the generation-point identity).

The one-side reuse mechanism (the card's "no regeneration" contract)
--------------------------------------------------------------------
The card reuses the R51 sealed events (no new generation; the sealed
R51 archive is the event source, read-only; its seal verifies).  The
sealed R51 replica records store the compact R51 record convention,
which carries the proton and antiproton spectrum records but not the
central pion/kaon records the R52 held-out set needs, so the one-side
held-out extraction re-derives the identical events from the sealed
R51 identity (the sealed per-replica seeds and event-index partition,
the unchanged committed class modules) and verifies bit-identity
against the sealed archive: every per-event record hash and every
replica ``event_hash_set_sha256`` must match the sealed archive
exactly, or the run refuses.  No new randomness enters: the
re-derivation reproduces the sealed draws bit-exactly (the same
evidence the R51 clean-environment validation recorded for its probe
replica), the consumed events are the sealed events, and the frozen
values are asserted before any evaluation.  The multiplicity tails and
the one-side weights are computed from the sealed records directly.

The held-out set (index item 4; plan section 7.4 data splits)
-------------------------------------------------------------
- The central pion and kaon spectra of pi+, pi-, K+, K- in the central
  rapidity window |y^CM| < 0.5, from the sealed R01 published binned
  tables (the published rows whose row-mid y^CM at the species mass
  falls inside the window), recorded with the dense-accumulation/
  then-pick convention (inherited state): every model-side particle
  record accumulates on the dense published evaluation lattice and the
  central published rows are picked with the pick map
  ``bin_index_pt * n_x + bin_index_xf``; hole cells dropped.
- The pi0 central spectrum: the sealed R01 archive and the committed
  SPS data carry no published pi0 table (the R01 tables are pi+, pi-,
  K+, K-, proton, antiproton), so the pi0 spectrum is accumulated and
  recorded model-side on the pi- published geometry (the row window at
  the pi0 mass 0.1349766 GeV) and carries no data deviance; the
  deviance-based statistics cover the four published species.  This
  instantiation is recorded in the pre-registration and named in the
  register.
- The charged multiplicity tails: the multiplicity bins at n_ch >= the
  90th percentile of the R51 training distribution (the R51 training
  statistic's data side, the sealed R50-pinned reference over the
  pre-registered bins n_ch = 2..14; the threshold is the smallest bin
  whose cumulative probability reaches 0.90).  The sealed substrate
  carries no published multiplicity table, so the tail data side is
  the same R50-pinned reference the R51 training multiplicity
  distribution used; the tail bins are therefore a subset of the R51
  training bins and this overlap with the card's held-out-disjointness
  clause is recorded explicitly in the pre-registration and in the
  register.  The central spectra carry the full held-out role; no
  R52 decision moves a held-out observable into training (plan section
  7.4, line 453).

The retention rule (pre-registered; never relaxed)
--------------------------------------------------
Per held-out observable family (the four published central spectra and
the multiplicity tails), with V = V_data + V_mc per model (the plan
section 5.4 convention; the MC error is the between-replica standard
error over the 8 replica blocks, floored at the engine's
DETERMINISTIC_SE_FLOOR):

- per bin: one_deviance = ((d - m_one)/sqrt(V_one))^2,
  zero_deviance = ((d - m_zero)/sqrt(V_zero))^2,
  difference = zero_deviance - one_deviance,
  pooled_se = the pooled standard error of the deviance difference,
  propagated from both models' MC errors,
  pooled_se = sqrt( (2 (d - m_zero)/V_zero se_zero)^2
                    + (2 (d - m_one)/V_one se_one)^2 ),
  floored at the deterministic SE floor; beyond_2x_se = the bin
  degrades iff difference >= 2.0 pooled_se and difference > 0.
- family degradation: the largest per-bin degradation
  G = max over kept bins of difference / pooled_se (0.0 when no bin
  degrades); the zero model degrades the held-out deviance by 2.0
  pooled standard deviations or more on those bins iff G >= 2.0.
- family preservation: the one model preserves the bins iff its
  largest per-bin pull against the data, P = max |(d - m_one) /
  sqrt(V_one)| over the kept bins, is at most 2.0.
- family verdict: "retained" iff the zero model degrades by >= 2.0
  pooled SE AND the one model preserves the bins; otherwise "not
  retained".  The per-family verdicts are the recorded retention
  evidence; the written Gate 5 retain / revise / reject decision is
  the round's follow-up obligation.
- kept bins: the bins whose data side is finite with a non-negative
  published variance and that BOTH models populate (a bin where
  either model is structurally zero in every block is recorded as
  dropped with the reason; checked, never bled into a kept bin).

The R53 trigger (index item 5): the named R52 residual on the
multiplicity tails at 2.0 pooled standard deviations or more -- the
ONE model's largest multiplicity-tail pull (the full per-bin
covariance; the R53 extension parameter P(n >= 2) lives in the ND
class, which exists only in the one model, so the one model is the
extension target) is >= 2.0.  The value, the pooled SE, the sigma, and
the trigger boolean are recorded in named form in ``gate5_r52.json``;
card 13 consumes that entry.  The zero model's tail residual is
recorded beside it.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the M8 frozen
  parameter map, the sealed R51/R50/R01 seals, the R51 archive digest
  pin (the one-side source), the frozen values against the sealed R51
  ``fit_report.json``, the renormalization algebra, the multiplicity
  tail cut, and the seed-block disjointness before anything is
  written; the pre-registration is written once, before any
  generation.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one zero-model replica x 2,000
  events at the smoke seed 20312925 (raw offset +12000 from master
  20260925) on the event-index partition 1_600_000_000 + i (i < 2,000),
  the one-side reuse-machinery exercise (the smoke events re-derived
  from the smoke seed and verified bit-identical against the smoke
  record -- the production reuse path exercised hermetically), and the
  smoke held-out evaluation (synthetic data; no SPS table and no
  sealed archive read), into an explicit ``--run-dir`` (the default
  production run dir is refused).  Two identical smoke invocations
  produce byte-identical artifacts.  ``--gate`` / ``--seal`` /
  ``--clean-validate`` refuse smoke artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested ZERO-model
  production replica records (any subset; the default is every
  replica), one record file per replica, then stop.  Requires the
  pre-registration.  A completed record that matches the
  pre-registered identity is reused unchanged, so the declared tier
  can be completed replica by replica (the pre-registered gateway
  fallback runs one job per replica; the gateway agent command is not
  configured on this checkout, so the local fallback applies and is
  recorded).  ``--generate`` covers the zero model only; the one model
  is never generated here.
- ``--evaluate``: the held-out evaluation: the frozen-value and
  renormalization assertions, the sealed R51 archive consumption (the
  one-side weights and the bit-identity re-derivation with the pi/K
  extraction), the zero-model record loading, the per-bin deviances,
  the zero/one differences with pooled SE, the retention verdicts, the
  R53 trigger, ``held_out_report.json`` and ``REPORT.md``, then stop.
  Requires all 8 zero-model production records and the sealed R51
  archive.
- ``--gate``: write ``gate5_r52.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the
  held-out report.
- ``--clean-validate``: re-run the held-out evaluation from a clean
  interpreter over the generated artifacts with a byte-identical
  report, re-run the pre-registered zero-model probe replica (replica
  0) in memory against its sealed record, and re-derive the one-side
  probe replica (R51 replica 0) against the sealed R51 archive, write
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

Hermetic in ``--smoke``; the production modes read the committed
catalog and continuum config, the sealed R01 archive, the SPS tables
(the published data the held-out spectra are defined against), the
sealed R50 archive (the multiplicity-tail data anchor), and the sealed
R51 archive (the one-model event source and the frozen values).  The
frozen simplified cascade at the sealed R45 point (kappa_B = 0.0,
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
# sealed archives, the SPS tables, and the sibling R51 driver module.
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
        f"staged_r52_central_excitation_necessity: {message}",
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
# exact functions carry the one-side weights (event_log_weights), the
# record/conservation extraction (serialize_event, event_conservation,
# spectrum_records, charged_multiplicity), the sealed-archive loaders
# (load_blocks, verify_sealed_dir, r50_reference_payload,
# _load_production_tables, _load_sealed_r01, build_geometry,
# _assign_cells, se_from, _sqrt_s_conversion), and the R51 constants.
# Consuming the module verbatim guarantees the one-side weighted
# evaluation is the R51 algebra bit for bit.
import staged_r51_topology_fraction_fit as r51  # noqa: E402

from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.rng_substreams import (  # noqa: E402
    SUBSTREAM_LABELS,
    derive_substreams,
)
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402
from na49_mixing.topology_elastic import (  # noqa: E402
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
)
from na49_mixing.topology_mass_laws import PI0_MASS  # noqa: E402
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
# Pre-registered run identity (card TASK-20260925-r5-run-r52-central-
# excitation-necessity; the collision, statistics, seeds, held-out set,
# likelihood, and acceptance blocks are verbatim from the card and the
# round 5 index items 4, 8, 9, and 10).
# ---------------------------------------------------------------------------
RUN_ID = "R52"
CARD = "TASK-20260925-r5-run-r52-central-excitation-necessity"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r52_central_excitation_necessity_2026-09-25 re-pins to the
#: execution date 2026-09-27 (the R50/R51 sealed archives moved the
#: same way).
RUN_NAME = "r52_central_excitation_necessity_2026-09-27"
DEFAULT_RUN_DIR = "dev/00runs/r52_central_excitation_necessity_2026-09-27"
GATE_RECORD_NAME = "gate5_r52.json"
REPORT_NAME = "held_out_report.json"
REPORT_MD_NAME = "REPORT.md"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_REPORT_NAME = "smoke_held_out_report.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r52_central_excitation_necessity.py")
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r52_central_excitation_necessity.py")
R51_DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r51_topology_fraction_fit.py")
R51_DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r51_topology_fraction_fit.py")

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9, card 10 (the sealed R50 templates), card 11
#: (the sealed R51 fraction fit), and this card.
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
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests,
#: including the R45/R46/R50/R51 driver pins).
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
#: The Feynman-x / central-window extraction energy: the card-1 module
#: constant SQRT_S = 17.270 GeV (the recorded M8 quote; the CM-frame
#: convention's beta/gamma at this energy are beta = 0.9940791999921171
#: and gamma = 9.203303892226666).
SQRT_S_EXTRACTION_GEV = 17.270
#: The pinned CM-frame boost parameters at the extraction energy
#: (asserted against the computed values at import).
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
            "model-uncertainty term of the comparison (the carry on the "
            "continuum families; recorded, not folded into the frozen "
            "comparison's arithmetic)"
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
#: convention): the zero model is 8 replicas x 100,000 p+p events =
#: 800,000 regenerated events; the one model reuses the 800,000 sealed
#: R51 events.
N_REPLICAS = 8
EVENTS_PER_REPLICA = 100_000
N_EVENTS_TOTAL = N_REPLICAS * EVENTS_PER_REPLICA   # 800,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
#: The zero-model raw replica seeds 20300953-20300960 (k = 0..7; index
#: item 9; the block is disjoint from every previously used raw block
#: and from every other reserved Round 5 block).
REPLICA_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + SEED_OFFSET + 28 + k for k in range(N_REPLICAS))
EVENT_INDEX_BASE = 1_200_000_000
EVENT_INDEX_RULE = "1_200_000_000 + i, i < 100,000 (per replica)"

SMOKE_SEED = MASTER_SEED + 40_000 + 12_000              # 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"

#: The reserved round-5 raw seed blocks (index item 9; the R50 and R51
#: blocks are now used and recorded in PREVIOUS_USED_BLOCKS below).
RESERVED_ROUND5_BLOCKS = {
    "r50_used": [20300925, 20300944],
    "r51_used": [20300945, 20300952],
    "r53": [20300961, 20300968],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9).
RESERVED_ROUND5_PARTITIONS = {
    "r50_used": [1_000_000_000, 1_000_025_000],
    "r51_used": [1_100_000_000, 1_100_100_000],
    "r53": [1_300_000_000, 1_300_100_000],
    "r54": [1_400_000_000, 1_400_050_000],
    "r55": [1_500_000_000, 1_500_250_000],
    "round5_smoke": [1_600_000_000, 1_600_002_000],
}

#: The previously-used seed blocks and event-index partitions, recorded
#: from the sealed pre-registrations and the round 5 index item 9 (the
#: disjointness evidence of the card; any (seed, event-index) collision
#: stops the run).  The R51 entries are the one-side source's identity:
#: the reuse re-consumes the sealed R51 raw block and partition
#: bit-exactly by design (the reuse contract), which is not a new draw
#: and is recorded as such; the disjointness check below covers the NEW
#: zero-model block only.
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
}

#: The clean-environment validation's pre-registered probe replicas:
#: the zero-model replica 0 (regenerated in memory against its sealed
#: record) and the one-side R51 replica 0 (re-derived against the
#: sealed R51 archive).
PROBE_REPLICA = 0

# ---------------------------------------------------------------------------
# The one-side frozen values (the sealed R51 fitted point; asserted
# against the sealed R51 fit_report.json before any evaluation)
# ---------------------------------------------------------------------------
FROZEN_FIT: dict[str, float] = {
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
    "dd_fraction": 0.9999997010778905,
    "nd_fraction": 3.1345895878942542e-09,
    "epsilon": 0.0033443027832747944,
    "b_sd": 1.8457918026393214,
}

#: The held-out retention bound (the card: "2.0 pooled standard
#: deviations or more"; never relaxed).
RETENTION_SE_BOUND = 2.0
#: The multiplicity-tail quantile of the R51 training distribution
#: (index item 4: "the multiplicity bins at n_ch greater than or equal
#: to the 90th percentile of the R51 training distribution").
TAIL_QUANTILE = 0.90

#: The charged-multiplicity bins (the R51 training support; an event
#: outside the support refuses the run).
MULTIPLICITY_BINS: tuple[int, ...] = tuple(range(2, 15))

#: The central-activity rapidity window (the runner's CENTRAL_WINDOW;
#: plan index item 4: |y| < 0.5).
CENTRAL_WINDOW = 0.5

#: The event four-momentum closure tolerance (the house conservation
#: contract).
EVENT_CLOSURE_TOL = 1e-9

#: The held-out central-spectrum species: the published four and the
#: model-only pi0 (no published pi0 table exists in the sealed R01
#: archive or the committed SPS data; recorded instantiation).
PUBLISHED_SPECIES: tuple[str, ...] = ("pi_plus", "pi_minus", "k_plus",
                                      "k_minus")
PI0_SPECIES = "pi0"
#: The pi0 row-window mass (the pinned pi0 mass; the pi0 spectrum is
#: accumulated on the pi- published geometry).
PI0_ROW_MASS_GEV = 0.1349766
#: The pi0 model-side accumulation geometry (the pi- published table's
#: geometry; recorded instantiation).
PI0_GEOMETRY_SPECIES = "pi_minus"

#: The held-out species' particle ids (the FinalHadron pdg_id values)
#: and their pinned masses (the pi-/K masses are asserted against the
#: loaded R01 tables' declared masses at the evaluation).
SPECIES_PDGS: dict[str, int] = {
    "pi_plus": 211,
    "pi_minus": -211,
    "pi0": 111,
    "k_plus": 321,
    "k_minus": -321,
}
SPECIES_MASSES: dict[str, float] = {
    "pi_plus": 0.1395699,
    "pi_minus": 0.1395699,
    "pi0": PI0_ROW_MASS_GEV,
    "k_plus": 0.493677,
    "k_minus": 0.493677,
}

#: The sealed input archives.
R01_ARCHIVE = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R50_RUN_DIR = "dev/00runs/r50_pure_topology_templates_2026-09-27"
R51_RUN_DIR = "dev/00runs/r51_topology_fraction_fit_2026-09-27"
R51_FIT_REPORT_NAME = "fit_report.json"

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
    """The raw seed of zero-model replica *replica* (the pre-registered
    R52 block)."""
    k = int(replica)
    if not 0 <= k < N_REPLICAS:
        _refuse(f"replica index {k} outside the R52 block "
                f"0..{N_REPLICAS - 1}")
    return REPLICA_SEEDS[k]


def replica_name(replica: int) -> str:
    return RECORD_FMT.format(int(replica))


def event_id(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_id(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


# ---------------------------------------------------------------------------
# The frozen-point and renormalization assertions (module level)
# ---------------------------------------------------------------------------
def assert_frozen_values() -> dict:
    """The frozen-value invariants (module level): the six frozen
    numbers form the R51 fitted point; the fraction closure holds; the
    renormalized zero-model fractions are non-negative, carry the ND
    fraction exactly 0.0, and sum to one within the runner's fraction
    tolerance; the frozen epsilon/B_SD are inside the R51 pre-registered
    brackets; the runner's registry is intact."""
    f_sdp = FROZEN_FIT["sd_projectile_fraction"]
    f_sdt = FROZEN_FIT["sd_target_fraction"]
    f_dd = FROZEN_FIT["dd_fraction"]
    f_nd = FROZEN_FIT["nd_fraction"]
    closure = 1.0 - (f_sdp + f_sdt + f_dd)
    if closure != f_nd:
        _refuse(
            "the frozen R51 fraction closure drifted: 1 - (f_sdp + f_sdt "
            f"+ f_dd) = {closure!r} vs the frozen f_nd {f_nd!r}")
    if not (0.0 <= f_nd <= 1.0) or not (0.0 <= f_sdp <= 1.0) \
            or not (0.0 <= f_sdt <= 1.0) or not (0.0 <= f_dd <= 1.0):
        _refuse("a frozen R51 fraction is outside [0, 1]")
    norm = 1.0 - f_nd
    if not (norm > 0.0):
        _refuse("the frozen f_nd leaves an empty renormalized inelastic "
                "mass")
    renorm = {
        "sd_projectile": f_sdp / norm,
        "sd_target": f_sdt / norm,
        "dd": f_dd / norm,
        "nd": 0.0,
    }
    renorm_sum = math.fsum(renorm.values())
    if abs(renorm_sum - 1.0) > 1e-12:
        _refuse(
            "the renormalized zero-model fractions do not sum to one "
            f"within 1e-12: {renorm_sum!r}")
    if renorm["nd"] != 0.0:
        _refuse("the zero-model ND fraction is not exactly 0.0")
    if not -0.5 <= FROZEN_FIT["epsilon"] <= 0.5:
        _refuse("the frozen epsilon is outside the R51 pre-registered "
                "bracket [-0.5, 0.5]")
    if not 1.0 <= FROZEN_FIT["b_sd"] <= 20.0:
        _refuse("the frozen B_SD is outside the R51 pre-registered "
                "bracket [1.0, 20.0]")
    if dict(NOMINAL_PARAMS.class_fractions) != {
            "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}:
        _refuse("the runner's nominal inelastic fractions drifted")
    if set(TOPOLOGY_CLASSES) != {
            "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}:
        _refuse(f"the class registry drifted: {sorted(TOPOLOGY_CLASSES)!r}")
    if TOPOLOGY_CLASSES["nd_multi"]["retained"]:
        _refuse("the nd_multi registry entry must carry retained: False")
    if TOPOLOGY_CLASSES["elastic"]["inelastic"]:
        _refuse("the elastic registry entry must carry inelastic: False")
    if set(INELASTIC_SET) != {"sd_projectile", "sd_target", "dd", "nd"}:
        _refuse(f"the inelastic set drifted: {sorted(INELASTIC_SET)!r}")
    return {
        "frozen_values": dict(FROZEN_FIT),
        "fraction_closure": (
            "f_nd = 1 - f_sd_p - f_sd_t - f_dd exactly at the frozen R51 "
            "point (the sum rule over the inelastic sample)"),
        "renormalization_rule": (
            "f'_c = f_c / (1 - f_nd) for c in {sd_projectile, sd_target, "
            "dd}; the zero-model ND fraction is exactly 0.0"),
        "renormalized_fractions": dict(renorm),
        "renormalized_sum": renorm_sum,
        "zero_model_params": {
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": 0.0,
            "remnant_weights": dict(NOMINAL_PARAMS.remnant_weights),
            "note": ("the R51-frozen topology with the ND fraction zeroed; "
                     "epsilon and B_SD stay at the frozen R51 values; "
                     "P(n >= 2) = 0.0 (the unretained card-8 default)"),
        },
        "class_set": list(CLASS_ORDER),
        "inelastic_set": sorted(INELASTIC_SET),
        "pass": True,
    }


_FROZEN_STATUS = assert_frozen_values()
RENORMALIZED_FRACTIONS: dict[str, float] = dict(
    _FROZEN_STATUS["renormalized_fractions"])


def zero_model_params() -> TopologyNominalParams:
    """The zero-model generation parameters: the R51-frozen topology
    with the ND fraction zeroed (the frozen epsilon/B_SD; the nominal
    remnant weights; P(n >= 2) = 0.0)."""
    return TopologyNominalParams(
        epsilon=FROZEN_FIT["epsilon"],
        b_sd=FROZEN_FIT["b_sd"],
        class_fractions=dict(RENORMALIZED_FRACTIONS),
        remnant_weights=dict(NOMINAL_PARAMS.remnant_weights),
        p_n2=0.0,
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
    card-pinned surfaces and the R45/R46/R50/R51 driver pins)."""
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
    slope = float(m8["regge.effective_transfer_slope"])
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
        "note": (
            "the M8-frozen effective_transfer_slope is the R50/R51 "
            "generation nominal B_SD "
            f"({ELASTIC_TRANSFER_SLOPE!r}); the R52 models run at the "
            "frozen R51 fitted B_SD asserted against the sealed R51 "
            "fit report"),
        "source": ("config/v2_singlet/legacy_v1_manifest.json "
                   "(frozen_fit_outputs.m8_frozen_pp_parameters)"),
    }


def verify_sealed_dir(rel: str) -> dict:
    """Verify a sealed run dir's SHA256SUMS.txt entry by entry (the R41
    verify-sealed discipline; the archive must be intact before it is
    consumed read-only)."""
    return r51.verify_sealed_dir(rel)


def verify_sealed_inputs() -> dict:
    """The sealed-input verification: the R51 archive (the one-side
    event source and the frozen values), the R50 archive (the
    multiplicity-tail data anchor), and the R01 archive (the data
    side)."""
    return {
        "r51": verify_sealed_dir(R51_RUN_DIR),
        "r50": verify_sealed_dir(R50_RUN_DIR),
        "r01": verify_sealed_dir(R01_ARCHIVE),
    }


# ---------------------------------------------------------------------------
# The R51 archive digest pin (the one-side source; recorded in the
# pre-registration before any generation)
# ---------------------------------------------------------------------------
def r51_archive_pin() -> dict:
    """The sealed R51 archive's digest pin: the fit report's digest and
    the per-replica record digests with their event-hash-set digests
    (the one-side bit-identity targets)."""
    run_dir = MONO_ROOT / R51_RUN_DIR
    seal = verify_sealed_dir(R51_RUN_DIR)
    fit_report = _read_json(run_dir / R51_FIT_REPORT_NAME)
    replicas = []
    for rep in range(N_REPLICAS):
        record = _read_json(run_dir / replica_name(rep))
        replicas.append({
            "replica": rep,
            "seed": int(record["seed"]),
            "n_events": int(record["n_events"]),
            "file_sha256": _sha256_file(run_dir / replica_name(rep)),
            "event_hash_set_sha256": record.get("event_hash_set_sha256"),
            "class_census": dict(record.get("class_census") or {}),
        })
    return {
        "run_dir": R51_RUN_DIR,
        "seal": seal,
        "fit_report_name": R51_FIT_REPORT_NAME,
        "fit_report_sha256": _sha256_file(
            run_dir / R51_FIT_REPORT_NAME),
        "fit_report_digest_field": fit_report.get("digest"),
        "replicas": replicas,
        "role": (
            "the one-model event source (read-only; reused, never "
            "regenerated); the per-replica digests and event-hash-set "
            "digests are the bit-identity targets of the one-side "
            "consumption"),
    }


# ---------------------------------------------------------------------------
# The frozen-value assertion against the sealed R51 (before any
# evaluation)
# ---------------------------------------------------------------------------
def assert_frozen_values_against_r51(pin: dict | None = None) -> dict:
    """The frozen values are the sealed R51 fitted point exactly (the
    card: "the frozen values ... are asserted against the sealed R51
    fit_report.json before any evaluation")."""
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
    r51_named = [entry.get("kind") for entry in
                 (fit_report.get("named_residual_register") or [])]
    return {
        "source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
        "fit_report_sha256": (
            pin["fit_report_sha256"] if pin else _sha256_file(
                run_dir / R51_FIT_REPORT_NAME)),
        "r51_fitted": {name: float(fitted[name]) for name in
                       ("sd_projectile_fraction", "sd_target_fraction",
                        "dd_fraction", "epsilon", "b_sd")},
        "r51_nd_fraction": float(fitted["nd_fraction"]),
        "r51_optimizer_converged": bool(
            (fit_report.get("optimizer") or {}).get("converged")),
        "r51_named_residual_kinds": r51_named,
        "r51_frozen_caveat": (
            "the frozen point is the R51 optimum, which did not converge "
            "and sits at the dd boundary with f_nd ~ 3e-09; the R51 named "
            "residuals (optimizer_non_convergence, "
            "replica_block_minima_disagree, block_curvature_failure, "
            "weight_degeneracy) are carried into this comparison as the "
            "inherited model-uncertainty caveats"),
        "match": True,
    }


# ---------------------------------------------------------------------------
# The seed-block disjointness (the card's stop condition)
# ---------------------------------------------------------------------------
def seed_disjointness_report() -> dict:
    """The seed-block disjointness check for the NEW zero-model block:
    the exact previously-used blocks recorded from the sealed
    pre-registrations and the round 5 index item 9, and the event-index
    partition disjointness.  Any (seed, event-index) substream collision
    refuses the run (never a silent re-block).  The one-side reuse
    re-consumes the sealed R51 raw block and partition bit-exactly by
    design (the reuse contract; not a new draw) and is recorded here as
    such."""
    problems: list[str] = []
    r52_lo = EVENT_INDEX_BASE
    r52_hi = EVENT_INDEX_BASE + EVENTS_PER_REPLICA
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
            if r52_lo < p_hi and p_lo < r52_hi:
                problems.append(
                    f"the R52 event-index partition [{r52_lo}, {r52_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS)):
            problems.append(
                f"the R52 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if name in ("r50_used", "r51_used"):
            continue
        if r52_lo < hi and lo < r52_hi:
            problems.append(
                f"the R52 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r52] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r52_raw_seed_block": [min(REPLICA_SEEDS), max(REPLICA_SEEDS)],
        "r52_raw_seed_rule": (
            "raw zero-model replica seed = 20260925 + 40000 + 28 + k = "
            "20300953 + k per replica k < 8 (index item 9); the block "
            "20300953-20300960 is disjoint from every previously used "
            "raw block and from every other reserved Round 5 block"),
        "r52_replica_seeds": list(REPLICA_SEEDS),
        "r52_event_index_partition": EVENT_INDEX_RULE,
        "r52_event_index_range": [r52_lo, r52_hi],
        "one_side_reuse": (
            "the one model re-consumes the sealed R51 raw block "
            "20300945-20300952 on the partition 1_100_000_000 + i "
            "bit-exactly (the reuse contract; the sealed draws are "
            "reproduced, no new draw happens; recorded, not a collision"),
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R52 raw block 20300953-20300960 shares no raw seed with "
            "any previously used block (R23 20300913-20300920 under the "
            "20260913 master ends below it; the R50 block 20300925-"
            "20300944 and the R51 block 20300945-20300952 end below it; "
            "the Round 4 blocks 20301020-20301083 and smoke blocks "
            "20307920-20311920 under the 20260920 master sit above it; "
            "the R32 blocks 20301519-20301826 under the 20260918 master "
            "are disjoint); the substreams are disjoint by construction "
            "(both the numpy.random.SeedSequence spawn tree and "
            "derive_substreams(seed, 0, event_id) key on the raw seed "
            "and the event index) and the event-index partitions are "
            "disjoint"),
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
# The multiplicity-tail cut (the 90th percentile of the R51 training
# distribution; computed from the sealed R50-pinned reference)
# ---------------------------------------------------------------------------
def multiplicity_tail_cut(r50_reference: dict) -> dict:
    """The tail cut: the smallest pre-registered bin whose cumulative
    probability in the R51 training distribution (the R50-pinned
    reference the R51 training multiplicity statistic used) reaches
    TAIL_QUANTILE; the tail bins are the bins at or above it."""
    mult = r50_reference["charged_multiplicity"]
    bins = [int(n) for n in mult["bins"]]
    dist = mult["distribution"]
    cumulative = 0.0
    threshold = None
    cumulative_at = {}
    for n in bins:
        cumulative = math.fsum([cumulative, float(dist[str(n)])])
        cumulative_at[str(n)] = cumulative
        if threshold is None and cumulative >= TAIL_QUANTILE:
            threshold = n
    if threshold is None:
        threshold = bins[-1]
    tail_bins = [n for n in bins if n >= threshold]
    return {
        "quantile": TAIL_QUANTILE,
        "definition": (
            "the smallest pre-registered bin whose cumulative "
            "probability in the R51 training distribution (the sealed "
            "R50-pinned reference the R51 training multiplicity "
            "statistic used) reaches the quantile; the tail bins are "
            "the bins at or above it"),
        "threshold_n_ch": int(threshold),
        "cumulative_at_threshold": cumulative_at[str(threshold)],
        "tail_bins": tail_bins,
        "source": (
            "the R51 training distribution's data side (the sealed "
            "R50-pinned reference; the sealed substrate carries no "
            "published multiplicity table)"),
        "training_overlap_note": (
            "the tail bins are a subset of the R51 training "
            "multiplicity bins (2..14) and share the R50-pinned "
            "reference anchor; this overlap with the card's "
            "held-out-disjointness clause is recorded explicitly (the "
            "central spectra carry the full held-out role; no R52 "
            "decision moves a held-out observable into training, plan "
            "section 7.4 line 453)"),
    }


# ---------------------------------------------------------------------------
# The held-out geometry (the central published rows)
# ---------------------------------------------------------------------------
def _row_y_cm(xf_mid: float, pt_mid: float, mass: float,
              sqrt_s: float) -> float:
    """The row-mid CM rapidity of one published (x_F, p_T) cell at the
    species mass (the data-side y convention at the exact fixed-target
    invariant mass; the cell-mid y selects the held-out central rows)."""
    p_l = 0.5 * float(xf_mid) * float(sqrt_s)
    m_t = math.sqrt(float(pt_mid) * float(pt_mid) + float(mass) * float(mass))
    return 0.5 * math.log(
        (p_l + math.hypot(p_l, m_t)) / (-p_l + math.hypot(p_l, m_t)))


def build_heldout_geometry(tables: list, sealed: dict, geometry: dict,
                           sqrt_s_data: float) -> dict:
    """The held-out central-row geometry: per published species the
    central mask (|y^CM(row)| < 0.5 at the species mass), the central
    pick (the dense indices of the central rows), the central areas,
    and the central data vectors; the pi0 geometry on the pi- published
    table (the row window at the pi0 mass; model-side only)."""
    tables_by_species = {str(t.species): t for t in tables}
    held = {"species": {}, "pi0": None}
    for species in PUBLISHED_SPECIES:
        if species not in tables_by_species:
            _refuse(f"the binning tables are missing the held-out "
                    f"species: {species}")
        table = tables_by_species[species]
        mass = float(table.mass_gev)
        if mass != SPECIES_MASSES[species]:
            _refuse(
                f"the {species} table mass {mass!r} drifted off the "
                f"pinned {SPECIES_MASSES[species]!r}")
        n_rows = int(table.data.size)
        mask = np.empty(n_rows, dtype=bool)
        ys = np.empty(n_rows, dtype=float)
        for i in range(n_rows):
            y = _row_y_cm(
                float(table.xf_centers[int(table.bin_index_xf[i])]),
                float(table.pt_centers[int(table.bin_index_pt[i])]),
                mass, sqrt_s_data)
            ys[i] = y
            mask[i] = abs(y) < CENTRAL_WINDOW
        n_central = int(mask.sum())
        if n_central == 0:
            _refuse(f"the {species} table carries no central |y| < 0.5 "
                    "row; the held-out set cannot be built")
        pick = geometry["pick"][species]
        areas = geometry["areas"][species]
        central_pick = np.asarray(pick[mask], dtype=np.int64)
        if len(set(central_pick.tolist())) != n_central:
            _refuse(f"the {species} central pick is not one-to-one")
        held["species"][species] = {
            "mass_gev": mass,
            "rows": n_rows,
            "central_rows": n_central,
            "central_pick": central_pick,
            "central_areas": np.asarray(areas[mask], dtype=float),
            "d_central": np.asarray(
                sealed[species]["d"], dtype=float)[mask],
            "v_data_central": np.asarray(
                sealed[species]["v_data"], dtype=float)[mask],
            "xF_mid_central": np.asarray(
                sealed[species]["xF_mid"], dtype=float)[mask],
            "pT_mid_central": np.asarray(
                sealed[species]["pT_mid"], dtype=float)[mask],
            "y_mid_central": ys[mask],
            "window": CENTRAL_WINDOW,
        }
    # the pi0 geometry: the pi- published table's lattice, the row
    # window at the pi0 mass, model-side only (no published pi0 table)
    table = tables_by_species[PI0_GEOMETRY_SPECIES]
    n_rows = int(table.data.size)
    mask = np.empty(n_rows, dtype=bool)
    ys = np.empty(n_rows, dtype=float)
    for i in range(n_rows):
        y = _row_y_cm(
            float(table.xf_centers[int(table.bin_index_xf[i])]),
            float(table.pt_centers[int(table.bin_index_pt[i])]),
            PI0_ROW_MASS_GEV, sqrt_s_data)
        ys[i] = y
        mask[i] = abs(y) < CENTRAL_WINDOW
    pick = geometry["pick"][PI0_GEOMETRY_SPECIES]
    areas = geometry["areas"][PI0_GEOMETRY_SPECIES]
    held["pi0"] = {
        "geometry_species": PI0_GEOMETRY_SPECIES,
        "mass_gev": PI0_ROW_MASS_GEV,
        "rows": n_rows,
        "central_rows": int(mask.sum()),
        "central_pick": np.asarray(pick[mask], dtype=np.int64),
        "central_areas": np.asarray(areas[mask], dtype=float),
        "xF_mid_central": np.asarray(
            sealed[PI0_GEOMETRY_SPECIES]["xF_mid"], dtype=float)[mask],
        "pT_mid_central": np.asarray(
            sealed[PI0_GEOMETRY_SPECIES]["pT_mid"], dtype=float)[mask],
        "y_mid_central": ys[mask],
        "window": CENTRAL_WINDOW,
        "data_side": None,
        "data_side_note": (
            "no published pi0 table exists in the sealed R01 archive or "
            "the committed SPS data (the R01 tables are pi+, pi-, K+, "
            "K-, proton, antiproton); the pi0 central spectrum is "
            "accumulated and recorded model-side and carries no data "
            "deviance (the recorded instantiation)"),
    }
    return held


# ---------------------------------------------------------------------------
# The CM-frame extraction of the central pi/K records (the held-out
# species' final-state (x_F, p_T) pairs; the runner's pinned CM-frame
# convention at the extraction energy SQRT_S = 17.270 GeV)
# ---------------------------------------------------------------------------
def _cm_boost_checked() -> tuple[float, float]:
    """The lab to overall-CM boost parameters at the extraction energy,
    asserted against the pinned literals."""
    beta, gamma = r51._cm_boost_parameters(SQRT_S)
    if beta != CM_BETA_PINNED or gamma != CM_GAMMA_PINNED:
        _refuse(
            "the CM boost parameters drifted off the pinned values: "
            f"(beta, gamma) = ({beta!r}, {gamma!r}) vs the pinned "
            f"({CM_BETA_PINNED!r}, {CM_GAMMA_PINNED!r})")
    return beta, gamma


_CM_BETA, _CM_GAMMA = _cm_boost_checked()


def central_spectrum_records(event) -> dict:
    """The event's central-spectrum species records: per held-out
    species the final-state hadrons' ``(x_F, p_T, e/p_T)`` triples in
    the CM frame (the R01 accumulation measure ``e_CM / p_T``; the
    stored lab four-momentum is inverted through the pinned boost,
    exactly the inverse of the runner's forward boost; x_F = 2 p_L^CM /
    sqrt(s) at the extraction energy)."""
    out = {species: [] for species in SPECIES_PDGS}
    if isinstance(event, ElasticEvent):
        finals: tuple = tuple(event.outgoing)
    else:
        finals = tuple(event.final_state)
    for hadron in finals:
        pdg = getattr(hadron, "pdg_id", None)
        if pdg is None:
            continue
        species = None
        for name, pid in SPECIES_PDGS.items():
            if int(pdg) == pid:
                species = name
                break
        if species is None:
            continue
        e, px, py, pz = (float(x) for x in hadron.four_momentum)
        pz_cm = _CM_GAMMA * (pz - _CM_BETA * e)
        e_cm = _CM_GAMMA * (e - _CM_BETA * pz)
        x_f = 2.0 * pz_cm / SQRT_S
        p_t = math.sqrt(px * px + py * py)
        measure = e_cm / p_t if p_t > 0.0 else 0.0
        out[species].append((x_f, p_t, measure))
    return out


# ---------------------------------------------------------------------------
# The per-event physics layer (the house conservation contract; the
# R51 machinery consumed verbatim)
# ---------------------------------------------------------------------------
EVENT_SCHEMA = "na49_mixing.staged_r52_central_excitation_necessity.event/1"
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
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)


def serialize_zero_event(event, obs, conservation, pik, n_ch) -> dict:
    """The per-event record of the zero-model replica file (without its
    hash): the R51 compact record convention plus the ``pk`` block (the
    central pi/K spectrum records the held-out set consumes; the R51
    record convention does not carry them).  The caller adds the event
    identity fields and the per-event hash: the sha256 of the record's
    canonical JSON without the ``h`` key."""
    protons, pbars = spectrum_records(event)
    record = serialize_event(event, obs, conservation, protons, pbars, n_ch)
    record["pk"] = {
        species: [[float(x), float(p), float(w)] for x, p, w in records]
        for species, records in pik.items()
    }
    return record


# ---------------------------------------------------------------------------
# The zero-model generation (the R52 mixed draw at the renormalized
# frozen fractions)
# ---------------------------------------------------------------------------
GENERATION_FRACTIONS_NOTE = (
    "the R52 zero-model mixed draw runs at the renormalized frozen "
    "fractions (the R51 fitted fractions with the ND fraction set to "
    "exactly 0.0 and the remaining three renormalized to sum to one) "
    "with epsilon and B_SD at the frozen R51 values and P(n >= 2) = 0.0 "
    "(the unretained card-8 default; the multi-neutral machinery stays "
    "off exactly as in R51): the per-event class draw on the closed "
    "topology_selection substream is the zero-model generation surface, "
    "and the zero sample is generated at its own evaluation point (its "
    "model side carries unit weights, the generation-point identity)")


def generate_zero_replica_events(
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
):
    """Generate *n_events* zero-model mixed-draw inelastic events of one
    replica and return the per-event records, the class census, the
    conservation aggregate, and the event-hash set digest.

    Every event consumes the closed five-label substream quintuple at
    replica slot 0 (the R50/R51 convention: the raw seed carries the
    replica identity), draws its class on the closed
    ``topology_selection`` substream at the renormalized frozen
    fractions (the elastic fraction exactly 0.0 and the ND fraction
    exactly 0.0; a drawn elastic or nd label refuses the run -- the
    zero sample carries no central excitation by construction, checked,
    never repaired), runs the card-9 ``run_topology`` dispatch at the
    zero-model parameters, extracts the observables and the central
    pi/K spectrum records, and runs the house conservation contract (a
    violation raises and stops the run)."""
    params = zero_model_params()
    if params.class_fractions["nd"] != 0.0:
        _refuse("the zero-model draw parameters carry a nonzero ND "
                "fraction")
    if params.p_n2 != 0.0:
        _refuse("the zero-model draw parameters carry a nonzero P(n >= 2)")
    if params.epsilon != FROZEN_FIT["epsilon"] or \
            params.b_sd != FROZEN_FIT["b_sd"]:
        _refuse("the zero-model draw parameters drifted off the frozen "
                "R51 epsilon/B_SD")
    inelastic_ok = {"sd_projectile", "sd_target", "dd"}
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
        label = draw_topology(
            params.class_fractions, 0.0, rng_topology_selection)
        if label == "elastic" or label == "nd" or label not in inelastic_ok:
            raise ConservationViolation(
                f"event {event_index_base + i}: the zero-model class draw "
                f"returned {label!r} outside the zero-model set "
                f"{sorted(inelastic_ok)} (the ND fraction is exactly 0.0 "
                "and the elastic fraction is 0.0; checked, never repaired)")
        event = run_topology(
            label, params, catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=params.p_n2)
        obs = extract_observables(event, SQRT_S)
        conservation = event_conservation(event)
        pik = central_spectrum_records(event)
        n_ch = charged_multiplicity(event)
        if n_ch < MULTIPLICITY_BINS[0] or n_ch > MULTIPLICITY_BINS[-1]:
            raise ConservationViolation(
                f"event {event_index_base + i}: the charged multiplicity "
                f"{n_ch} is outside the pre-registered bins "
                f"{MULTIPLICITY_BINS}; the generation refuses (checked, "
                "never bled into an end bin)")
        record = serialize_zero_event(event, obs, conservation, pik, n_ch)
        record["i"] = int(i)
        record["id"] = int(event_index_base + i)
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
    return records, census, conservation_agg, hash_set


REPLICA_SCHEMA = "na49_mixing.staged_r52_central_excitation_necessity.replica/1"


def _identity_block(run_dir: Path) -> dict:
    return {
        "driver_sha256": DRIVER_SHA256,
        "pre_registration_sha256": _sha256_file(
            run_dir / PREREGISTRATION_NAME),
    }


def build_zero_replica_record(
    replica: int,
    records: list,
    census: dict,
    conservation: dict,
    hash_set: str,
    run_dir: Path,
    mode: str = "production",
) -> dict:
    """The zero-model replica record: the identity block, the seed map,
    the per-event records, the class census, the conservation aggregate,
    and the event-hash set digest (the generation-time determinism
    evidence)."""
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
                "raw zero-model replica seed = 20260925 + 40000 + 28 + k = "
                f"20300953 + k per replica k < {N_REPLICAS} (index item 9; "
                "the block 20300953-20300960 is disjoint from every "
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
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": 0.0,
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


def _verify_zero_replica_identity(record: dict, replica: int,
                                  run_dir: Path) -> None:
    """A completed zero-model production replica record must match the
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
    """Generate the requested ZERO-model production replica records (the
    default is every replica), one record file per replica, then stop.
    Requires the pre-registration.  A completed record that matches the
    pre-registered identity is reused unchanged (the pre-registered
    gateway fallback runs one job per replica; the local fallback is one
    single-threaded process per --replica invocation).  The one model is
    never generated here (it reuses the sealed R51 events)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    pre = load_pre_registration(run_dir)
    zero = pre.get("zero_model") or {}
    if float(zero.get("nd_fraction", -1.0)) != 0.0:
        _refuse("the pre-registration's zero model does not carry the ND "
                f"fraction 0.0: {zero.get('nd_fraction')!r}")
    registered = zero.get("renormalized_fractions") or {}
    if {k: float(v) for k, v in registered.items()} != \
            {k: float(v) for k, v in RENORMALIZED_FRACTIONS.items()}:
        _refuse(
            "the pre-registration's renormalized fractions drifted off "
            f"the frozen-point algebra: {registered!r} vs "
            f"{RENORMALIZED_FRACTIONS!r}")
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
            _verify_zero_replica_identity(existing, replica, run_dir)
            print(f"[r52] replica {int(replica)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        seed = replica_seed(replica)
        print(
            f"[r52] generating zero-model replica {int(replica)} ({name}): "
            f"seed {seed}, {EVENTS_PER_REPLICA} mixed-draw events at the "
            "renormalized frozen fractions",
            flush=True,
        )
        records, census, conservation, hash_set = \
            generate_zero_replica_events(
                seed, EVENTS_PER_REPLICA, EVENT_INDEX_BASE, catalog, fams)
        record = build_zero_replica_record(
            replica, records, census, conservation, hash_set, run_dir)
        record["reused"] = False
        _write_json(path, record)
        print(
            f"[r52] replica {int(replica)} written: {path} "
            f"(census {census}, worst closure "
            f"{conservation['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# The one-side consumption (the sealed R51 archive; the bit-identity
# re-derivation with the pi/K extraction)
# ---------------------------------------------------------------------------
def rederive_r51_replica(replica: int, sealed_record: dict, catalog, fams):
    """Re-derive R51 replica *replica* from the sealed identity (the
    sealed seed and the R51 event-index partition, the unchanged
    committed class modules at the R51 generation point) and verify
    bit-identity against the sealed record: every per-event record hash
    and the replica's ``event_hash_set_sha256`` must match exactly, or
    the run refuses (the reuse contract; no new draw happens -- the
    re-derivation reproduces the sealed draws bit-exactly).  Returns
    ``(per_species_records, identity)`` where per_species_records maps
    each held-out species to ``(xf, pt, measure, event_index)``
    concatenated arrays of the extracted central pi/K records."""
    seed = int(sealed_record["seed"])
    base = int(r51.EVENT_INDEX_BASE)
    n_events = len(sealed_record["events"])
    if n_events != EVENTS_PER_REPLICA:
        _refuse(f"the sealed R51 replica {replica} carries {n_events} "
                f"events, expected {EVENTS_PER_REPLICA}")
    per_species = {species: ([], [], [], []) for species in SPECIES_PDGS}
    hashes = []
    mismatches = []
    worst_closure = 0.0
    for i in range(n_events):
        subs = derive_substreams(seed, 0, base + i)
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
        if label == "elastic" or label not in INELASTIC_SET:
            raise ConservationViolation(
                f"event {base + i}: the re-derived class draw returned "
                f"{label!r} outside the inelastic set "
                f"{sorted(INELASTIC_SET)} (the R51 mixed draw runs at the "
                "elastic fraction 0.0; checked, never repaired)")
        event = run_topology(
            label, NOMINAL_PARAMS, catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=NOMINAL_PARAMS.p_n2)
        obs = extract_observables(event, SQRT_S)
        conservation = event_conservation(event)
        protons, pbars = spectrum_records(event)
        n_ch = charged_multiplicity(event)
        record = serialize_event(
            event, obs, conservation, protons, pbars, n_ch)
        record["i"] = int(i)
        record["id"] = int(base + i)
        record.pop("h", None)
        record["h"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8")).hexdigest()
        hashes.append(record["h"])
        if record["h"] != sealed_record["events"][i].get("h"):
            mismatches.append(int(i))
        worst_closure = max(worst_closure, conservation["closure_worst"])
        # the central pi/K extraction on the identical event
        pik = central_spectrum_records(event)
        for species, triples in pik.items():
            for x_f, p_t, measure in triples:
                per_species[species][0].append(x_f)
                per_species[species][1].append(p_t)
                per_species[species][2].append(measure)
                per_species[species][3].append(i)
    hash_set = hashlib.sha256(
        "\n".join(hashes).encode("utf-8")).hexdigest()
    identity = {
        "replica": int(replica),
        "seed": seed,
        "events": n_events,
        "per_event_hash_mismatches": len(mismatches),
        "event_hash_set_match": bool(
            hash_set == sealed_record.get("event_hash_set_sha256")),
        "worst_closure_relative": worst_closure,
        "bit_identical": bool(not mismatches and hash_set ==
                              sealed_record.get("event_hash_set_sha256")),
    }
    if not identity["bit_identical"]:
        _refuse(
            f"the one-side re-derivation of the sealed R51 replica "
            f"{replica} is not bit-identical to the sealed archive "
            f"({len(mismatches)} per-event hash mismatches, hash-set "
            f"match {identity['event_hash_set_match']}); the reuse "
            "contract refuses (the consumed events must be the sealed "
            "events)")
    lattice = {}
    for species, (xf, pt, w, evt) in per_species.items():
        lattice[species] = (
            np.asarray(xf, dtype=float), np.asarray(pt, dtype=float),
            np.asarray(w, dtype=float), np.asarray(evt, dtype=np.int64))
    return lattice, identity


# ---------------------------------------------------------------------------
# The held-out evaluation blocks (one side: the sealed R51 archive + the
# re-derivation; zero side: the R52 replica records)
# ---------------------------------------------------------------------------
class HeldoutBlock:
    """One replica's held-out evaluation arrays: the charged-multiplicity
    bin indices and the per-species dense-lattice entries of the central
    pi/K records (the flat cell, the parallel event index, the invariant
    measure, and the out-of-window drop count)."""

    __slots__ = ("replica", "seed", "n_events", "nch_idx",
                 "species_flat", "species_evt", "species_w",
                 "species_dropped", "species_total")


def _empty_species_arrays(species_iter):
    return {species: (np.zeros(0, dtype=np.int64),
                      np.zeros(0, dtype=np.int64),
                      np.zeros(0, dtype=float), 0)
            for species in species_iter}


def _assign_species_cells(xf, pt, measures, evt, edges, n_x):
    flat, kept_evt, kept_w, dropped = _assign_cells(
        xf, pt, edges[0], edges[1], n_x, evt, measures)
    return flat, kept_evt, kept_w, dropped


def one_side_blocks(run_dir: Path, geometry, held, catalog, fams) -> tuple:
    """The ONE side: the sealed R51 archive consumed read-only.

    (a) the sealed replica records load through the R51 loader (the
    replica identity and the event-hash sets verified against the sealed
    archive) and carry the frozen-point weight inputs and the charged
    multiplicities;
    (b) each replica is re-derived from the sealed identity and verified
    bit-identical (every per-event record hash and every replica
    ``event_hash_set_sha256``), and the central pi/K records are
    extracted on the identical events (the compact R51 record convention
    does not carry them);
    (c) the pi/K records are assigned to the dense published lattices.

    Returns ``(r51_blocks, heldout_blocks, reuse_identities)``."""
    r51_blocks = r51.load_blocks(MONO_ROOT / R51_RUN_DIR, smoke=False,
                                 geometry=geometry)
    heldout_blocks = []
    identities = []
    for rep in range(N_REPLICAS):
        sealed_record = _read_json(MONO_ROOT / R51_RUN_DIR
                                   / replica_name(rep))
        lattice, identity = rederive_r51_replica(
            rep, sealed_record, catalog, fams)
        identities.append(identity)
        block = HeldoutBlock()
        block.replica = rep
        block.seed = int(sealed_record["seed"])
        block.n_events = int(sealed_record["n_events"])
        nch = np.empty(block.n_events, dtype=np.int64)
        for i, ev in enumerate(sealed_record["events"]):
            nch[i] = int(ev["nc"]) - MULTIPLICITY_BINS[0]
        block.nch_idx = nch
        block.species_flat, block.species_evt, block.species_w = {}, {}, {}
        block.species_dropped, block.species_total = {}, {}
        for species in SPECIES_PDGS:
            xf, pt, w, evt = lattice[species]
            if species == PI0_SPECIES:
                edges = geometry["edges"][PI0_GEOMETRY_SPECIES]
                n_x = geometry["n_x"][PI0_GEOMETRY_SPECIES]
            else:
                edges = geometry["edges"][species]
                n_x = geometry["n_x"][species]
            flat, kept_evt, kept_w, dropped = _assign_species_cells(
                xf, pt, w, evt, edges, n_x)
            block.species_flat[species] = flat
            block.species_evt[species] = kept_evt
            block.species_w[species] = kept_w
            block.species_dropped[species] = dropped
            block.species_total[species] = int(xf.size)
        heldout_blocks.append(block)
    return r51_blocks, heldout_blocks, identities


def zero_side_blocks(run_dir: Path, geometry, *, smoke: bool = False) -> tuple:
    """The ZERO side: the 8 regenerated replica records (identity
    verified) loaded into the held-out evaluation arrays (the charged
    multiplicities and the per-species ``pk`` lattice entries).  The
    smoke mode loads the single smoke replica record instead."""
    blocks = []
    if smoke:
        records = [(SMOKE_REPLICA_NAME,
                    _read_json(Path(run_dir) / SMOKE_REPLICA_NAME))]
    else:
        records = [(replica_name(rep),
                    _read_json(Path(run_dir) / replica_name(rep)))
                   for rep in range(N_REPLICAS)]
    for rep, (name, record) in enumerate(records):
        if not smoke:
            _verify_zero_replica_identity(record, rep, Path(run_dir))
        block = HeldoutBlock()
        block.replica = record.get("replica", "smoke")
        block.seed = int(record["seed"])
        block.n_events = int(record["n_events"])
        nch = np.empty(block.n_events, dtype=np.int64)
        pk_lists = {species: ([], [], [], []) for species in SPECIES_PDGS}
        for i, ev in enumerate(record["events"]):
            nch[i] = int(ev["nc"]) - MULTIPLICITY_BINS[0]
            pk = ev.get("pk") or {}
            for species in SPECIES_PDGS:
                for x_f, p_t, w_meas in pk.get(species, []):
                    pk_lists[species][0].append(float(x_f))
                    pk_lists[species][1].append(float(p_t))
                    pk_lists[species][2].append(float(w_meas))
                    pk_lists[species][3].append(int(i))
        block.nch_idx = nch
        block.species_flat, block.species_evt, block.species_w = {}, {}, {}
        block.species_dropped, block.species_total = {}, {}
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


def _weights_one_side(blocks_one, stored_by_class: dict) -> dict:
    """The ONE side's frozen-point importance weights: the R51 weight
    machinery (``event_log_weights``) at the frozen fractions, epsilon,
    and B_SD, with ONE global log shift across the blocks (the R51
    evaluate convention; the self-normalized estimator and the ESS are
    shift-invariant)."""
    log_all = []
    fractions = None
    for block in blocks_one:
        lw, fractions = r51.event_log_weights(
            [block], FROZEN_FIT["sd_projectile_fraction"],
            FROZEN_FIT["sd_target_fraction"], FROZEN_FIT["dd_fraction"],
            FROZEN_FIT["epsilon"], FROZEN_FIT["b_sd"])
        if lw is None:
            _refuse("the frozen fractions are outside the importance "
                    "domain (a negative closure)")
        log_all.append(lw[0])
    stacked = np.concatenate(log_all) if log_all else np.zeros(0)
    finite = np.isfinite(stacked)
    if not finite.any():
        _refuse("the one-side frozen-point weights are all non-finite")
    shift = float(np.max(stacked[finite]))
    with np.errstate(over="ignore", under="ignore"):
        weights = [np.exp(lw - shift) for lw in log_all]
    ess = [ess_fraction(w) for w in weights]
    n_eff = [float(np.sum(w)) for w in weights]
    # the pooled sum accumulates sequentially in block order exactly as
    # the R51 evaluate's accumulation does (the bit-identity of the
    # multiplicity cross-check depends on the same operation order)
    n_eff_total = 0.0
    for value in n_eff:
        n_eff_total += value
    if not (math.isfinite(n_eff_total) and n_eff_total > 0.0):
        _refuse("the one-side pooled effective sample size is non-finite")
    return {
        "weights": weights,
        "log_shift": shift,
        "n_eff_per_block": n_eff,
        "n_eff_total": n_eff_total,
        "ess_per_block": ess,
        "ess_min": min(ess),
        "fractions": fractions,
        "stored_by_class": stored_by_class,
    }


def _multiplicity_vectors(blocks, weights) -> tuple:
    """The weighted charged-multiplicity vectors: the pooled counts and
    the per-block fractions (the R51 evaluate convention)."""
    mult_total = np.zeros(len(MULTIPLICITY_BINS))
    per_block = []
    for block, w in zip(blocks, weights):
        counts = np.bincount(
            block.nch_idx, weights=w, minlength=len(MULTIPLICITY_BINS))
        per_block.append(counts)
        mult_total = mult_total + counts
    return mult_total, per_block


def _species_model_vectors(blocks, weights, species, held, sqrt_s_data,
                           dense_size):
    """The weighted per-species model vectors on a species' central
    published rows: the pooled values, the per-block values, and the
    between-replica SEs (the R51 conversion
    ``m = sums / (pi sqrt(s) N_eff dxF dpT)``)."""
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
    """The population flags of a species' central rows under one model:
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
# The retention algebra (pre-registered; never relaxed)
# ---------------------------------------------------------------------------
def bin_comparison(d, v_data, m_one, m_zero, se_one, se_zero) -> dict:
    """One held-out bin's comparison record (the pre-registered algebra):
    the per-model pulls and deviances with V = V_data + V_mc, the
    deviance difference, the pooled SE of the deviance difference
    (propagated from both models' MC errors), the residual shift and the
    model gap in pooled SE units, and the beyond-bound flag."""
    v_one = float(v_data) + float(se_one) ** 2
    v_zero = float(v_data) + float(se_zero) ** 2
    sigma_one = math.sqrt(v_one) if math.isfinite(v_one) else float("inf")
    sigma_zero = math.sqrt(v_zero) if math.isfinite(v_zero) else float("inf")
    one_pull = (float(d) - float(m_one)) / sigma_one
    zero_pull = (float(d) - float(m_zero)) / sigma_zero
    one_deviance = one_pull * one_pull
    zero_deviance = zero_pull * zero_pull
    difference = zero_deviance - one_deviance
    deriv_zero = (2.0 * (float(d) - float(m_zero)) / v_zero
                  * float(se_zero)) if math.isfinite(v_zero) \
        and v_zero > 0.0 else 0.0
    deriv_one = (2.0 * (float(d) - float(m_one)) / v_one
                 * float(se_one)) if math.isfinite(v_one) \
        and v_one > 0.0 else 0.0
    se_diff = math.sqrt(deriv_zero ** 2 + deriv_one ** 2)
    if not math.isfinite(se_diff) or se_diff < SE_FLOOR:
        se_diff = SE_FLOOR
    pooled_model_gap_se = math.sqrt(float(se_one) ** 2
                                    + float(se_zero) ** 2)
    if not math.isfinite(pooled_model_gap_se) or pooled_model_gap_se < \
            SE_FLOOR:
        pooled_model_gap_se = SE_FLOOR
    residual_shift = abs(float(d) - float(m_zero)) - abs(float(d)
                                                         - float(m_one))
    model_gap = (float(m_zero) - float(m_one)) / pooled_model_gap_se
    beyond = bool(difference >= RETENTION_SE_BOUND * se_diff
                  and difference > 0.0)
    return {
        "one_pull": one_pull,
        "zero_pull": zero_pull,
        "one_deviance": one_deviance,
        "zero_deviance": zero_deviance,
        "difference": difference,
        "pooled_se": se_diff,
        "pooled_se_model_gap": pooled_model_gap_se,
        "residual_shift": residual_shift,
        "model_gap": model_gap,
        "beyond_2x_se": beyond,
    }


def family_verdict(bins: list) -> dict:
    """One held-out family's retention record (the pre-registered
    algebra): the largest per-bin deviance degradation in pooled SE
    units, the zero-model degradation flag at the 2.0 pooled-SE bound,
    the one model's largest pull, the preservation flag, and the
    verdict."""
    degradations = [b["difference"] / b["pooled_se"] for b in bins
                    if b["difference"] > 0.0]
    g = max(degradations) if degradations else 0.0
    p_one = max((abs(b["one_pull"]) for b in bins), default=0.0)
    zero_degrades = bool(g >= RETENTION_SE_BOUND)
    one_preserves = bool(p_one <= RETENTION_SE_BOUND)
    retained = bool(zero_degrades and one_preserves)
    return {
        "max_degradation_pooled_se": g,
        "zero_degrades_at_2x_pooled_se": zero_degrades,
        "one_model_max_abs_pull": p_one,
        "one_model_preserves": one_preserves,
        "verdict": "retained" if retained else "not retained",
        "bound": RETENTION_SE_BOUND,
    }


# ---------------------------------------------------------------------------
# The held-out evaluation (the run's --evaluate; the report machinery)
# ---------------------------------------------------------------------------
REPORT_SCHEMA = "na49_mixing.staged_r52_central_excitation_necessity.report/1"

DETERMINISM_NOTE = (
    "a per-replica re-run from a clean interpreter reproduces the report "
    "hashes; held_out_report.json and gate5_r52.json carry no wall-clock "
    "fields (their bytes are a pure function of the sealed artifacts, "
    "the pre-registered seeds, and the committed code); the per-event "
    "hash is the sha256 of the canonical JSON of the stored event record "
    "and the event_hash_set_sha256 is the sha256 over the joined "
    "per-event hashes, so the zero-model records and the one-side "
    "bit-identity re-derivation carry the identity evidence")

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
    "-2lnL per held-out bin = ((d - m)/sqrt(V))^2 with V = V_data + V_mc "
    "(reported separately), the published binning preserved, and no "
    "nuisance profiling (R52 is a frozen-parameter comparison, not a "
    "fit; the R51 fit's normalization and carry nuisances are not "
    "re-profiled here).  The per-bin MC error is the between-replica "
    "standard error (std ddof=1 over the 8 replica blocks, divided by "
    "sqrt(8)), floored at the engine's DETERMINISTIC_SE_FLOOR; a bin "
    "where either model is structurally zero in every block is recorded "
    "as dropped with the reason (the R11 convention; checked, never "
    "bled into a kept bin).  The retention bound is the pre-registered "
    "2.0 pooled standard deviations of the deviance difference; the "
    "one-side preservation bound is the same 2.0 on the largest per-bin "
    "pull.  Data and MC contributions are reported separately (the "
    "V_data and V_mc blocks of the per-bin records)")

ONE_SIDE_REUSE_NOTE = (
    "the one model reuses the sealed R51 events (no new generation; the "
    "sealed R51 archive is the event source, read-only; its seal "
    "verifies).  The sealed R51 replica records store the compact R51 "
    "record convention, which carries the proton and antiproton "
    "spectrum records but not the central pion/kaon records the R52 "
    "held-out set needs, so the one-side held-out extraction re-derives "
    "the identical events from the sealed identity (the sealed "
    "per-replica seeds and event-index partition, the unchanged "
    "committed class modules at the R51 generation point) and verifies "
    "bit-identity against the sealed archive: every per-event record "
    "hash and every replica event_hash_set_sha256 must match exactly, "
    "or the run refuses.  No new randomness enters (the re-derivation "
    "reproduces the sealed draws bit-exactly, the same evidence the R51 "
    "clean-environment validation recorded for its probe replica); the "
    "multiplicity tails and the one-side frozen-point importance "
    "weights are computed from the sealed records directly")

PI0_NOTE = (
    "the sealed R01 archive and the committed SPS data carry no "
    "published pi0 table (the R01 tables are pi+, pi-, K+, K-, proton, "
    "antiproton), so the pi0 central spectrum is accumulated and "
    "recorded model-side on the pi- published geometry (the row window "
    "at the pi0 mass 0.1349766 GeV) and carries no data deviance; the "
    "deviance-based retention and trigger statistics cover the four "
    "published species (the recorded instantiation of the card's "
    "held-out set)")

MULTIPLICITY_TAIL_NOTE = (
    "the multiplicity tails are the bins at n_ch greater than or equal "
    "to the 90th percentile of the R51 training distribution (the R51 "
    "training statistic's data side, the sealed R50-pinned reference "
    "over the pre-registered bins n_ch = 2..14; the threshold is the "
    "smallest bin whose cumulative probability reaches 0.90).  The "
    "sealed substrate carries no published multiplicity table, so the "
    "tail data side is the same R50-pinned reference the R51 training "
    "multiplicity distribution used; the tail bins are therefore a "
    "subset of the R51 training bins and this overlap with the card's "
    "held-out-disjointness clause is recorded explicitly.  The central "
    "spectra carry the full held-out role; no R52 decision moves a "
    "held-out observable into training (plan section 7.4, line 453)")

RETENTION_ALGEBRA_NOTE = (
    "per held-out observable family (the four published central spectra "
    "and the multiplicity tails), with V = V_data + V_mc per model: "
    "per bin, one_deviance = ((d - m_one)/sqrt(V_one))^2, zero_deviance "
    "= ((d - m_zero)/sqrt(V_zero))^2, difference = zero_deviance - "
    "one_deviance, pooled_se = the pooled standard error of the "
    "deviance difference propagated from both models' MC errors, "
    "pooled_se = sqrt((2 (d - m_zero)/V_zero se_zero)^2 + (2 (d - "
    "m_one)/V_one se_one)^2) floored at the deterministic SE floor, "
    "and beyond_2x_se = difference >= 2.0 pooled_se and difference > 0. "
    "Family degradation: the largest per-bin difference/pooled_se over "
    "the kept bins (0.0 when no bin degrades); the zero model degrades "
    "the held-out deviance by 2.0 pooled standard deviations or more on "
    "those bins iff it is >= 2.0.  Family preservation: the one model's "
    "largest per-bin pull against the data over the kept bins is at "
    "most 2.0.  Family verdict: retained iff both; otherwise not "
    "retained.  The per-bin residual shift |d - m_zero| - |d - m_one| "
    "and the standardized model gap (m_zero - m_one)/sqrt(se_one^2 + "
    "se_zero^2) are recorded beside the deviance algebra (the two "
    "models' V differ by construction -- the reused one side carries "
    "its frozen-point importance-ensemble MC error, the regenerated "
    "zero side its generation-point MC error -- so both unit systems "
    "are recorded transparently).  The written Gate 5 retain / revise / "
    "reject decision is the round's follow-up obligation")

R53_TRIGGER_NOTE = (
    "the named R52 residual on the multiplicity tails at 2.0 pooled "
    "standard deviations or more (index item 5; the house named-"
    "residual form used in Round 4): the ONE model's largest "
    "multiplicity-tail pull against the data (the full per-bin "
    "covariance sqrt(V_data + V_mc)) is >= 2.0.  The R53 extension "
    "parameter P(n >= 2) lives in the ND class, which exists only in "
    "the one model, so the one model is the extension target and its "
    "tail residual drives the trigger; the zero model's tail residual "
    "is recorded beside it and a disagreement between the two trigger "
    "booleans is named in the register.  Card 13 consumes the recorded "
    "entry; if the trigger does not fire, card 13 records NOT_TRIGGERED "
    "and the multiple-excitation class remains implemented but "
    "unretained (Gate 5 clause 2)")

COLLISION_NOTE = (
    "the R32 p+p configuration (index item 8): p+p at sqrt(s) = 17.270 "
    "GeV (beam 158 GeV/c), the M8-frozen ten-parameter set (hash "
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1) "
    "through the ResonancePPBuilder surface with the card-6 routing at "
    "the fixed blend ramp start B = 2.680194110727528 GeV and the "
    "card-9 feed-down lineage convention; the blend ramp start is never "
    "a free parameter.  The zero/one pair shares the feed-down lineage "
    "and the normalization (plan line 1872): both models are the same "
    "M8 p+p configuration through the same card-9 topology runner, the "
    "same frozen simplified cascade, and the same R01 data-units "
    "conversion; only the ND class fraction differs (0.25-at-generation "
    "importance-reweighted to the frozen fitted value on the one side, "
    "exactly 0.0 on the zero side) and the class draws consume the same "
    "closed substream pattern.  The excited states decay through the "
    "frozen simplified cascade at the sealed R45 point (kappa_B = 0.0, "
    "kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV) "
    "consumed without refit (the R46 no_refit clause); the exchange "
    "slopes act on the exchange kinematics only, not on the daughter "
    "transverse dynamics (plan section 24, line 1866); no Round 5 "
    "record claims a separation of exchange recoil from decay "
    "transverse dynamics (the R46 outcome, separation_claimed false).")

GENERATION_NOTE = (
    "gateway-primary generation for the zero model (host cl, workers "
    "c0801-c0809 x 32 task slots; one job per replica, --tasks 1, single "
    "math thread (NA49_CLUSTER_GW_TASKS forces thread counts to 1), "
    "submission paced by the capacity check, no local process pool "
    "inside the chunk command, chunk outputs appended only to the new "
    "run dir); the driver's local fallback is one single-threaded "
    "process per --replica invocation (a completed record that matches "
    "the pre-registered identity is reused unchanged).  The one model "
    "is never generated: it reuses the sealed R51 events")

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(ix) of the card: (i) zero-model "
    "regeneration: 8 replicas x 100,000 events, the ND fraction exactly "
    "0, the remaining three inelastic fractions equal the R51 fit "
    "values renormalized to sum to one (recorded in the pre-registration "
    "and asserted at generation), the R51 sealed archive seal verifies; "
    "(ii) one-model reuse: the R51 sealed events consumed read-only (no "
    "regeneration; the bit-identity re-derivation reproduces the sealed "
    "events exactly and every digest is verified), the frozen values "
    "match the sealed R51 fit report exactly; (iii) held-out "
    "comparison: per-bin deviance, the zero/one difference, and the "
    "pooled SE recorded for the central pion and kaon spectra and the "
    "multiplicity tails; the 2.0 pooled-SE retention bound and the "
    "per-family verdict recorded; (iv) named residuals: every bin "
    "beyond the bound named in the house form, the R53 trigger entry "
    "recorded; (v) conservation: zero B, Q, S violations and zero "
    "four-momentum closure failures at tolerance 1e-09 on all 800,000 "
    "regenerated events, the worst closure residual recorded in GeV, "
    "the R51 conservation record cross-referenced; (vi) determinism: "
    "the clean-environment re-run reproduces the per-replica report "
    "hashes; (vii) no optimizer: optimizer_invoked false, "
    "optimizer_uses 0; (viii) exactly one clean-environment validation "
    "pass executes before the seal and the seal is intact afterwards; "
    "(ix) the driver and test file land (both trees byte-identical), "
    "the freeze re-pin is recorded, and the full nested suite is green "
    "(the land gate's obligation)")

SUBSTREAM_COLLISION_RULE = (
    "if any raw (seed, event-index) substream collides with a "
    "pre-registered block, the run stops and reports; a silent re-block "
    "never happens (index item 9)")


def run_heldout_evaluation(run_dir: Path, *, smoke: bool = False) -> dict:
    """The held-out evaluation: the frozen-value assertions, the sealed
    R51 archive consumption (the one-side weights and the bit-identity
    re-derivation with the pi/K extraction), the zero-model record
    loading, the per-bin deviances, the zero/one differences with pooled
    SE, the retention verdicts, the R53 trigger, and the report dict
    with its digest (no wall-clock: the bytes are a pure function of the
    stored artifacts, the pre-registered seeds, and the committed
    code)."""
    run_dir = Path(run_dir)
    pre = load_pre_registration(run_dir, smoke=smoke)
    if smoke:
        return _smoke_evaluation(run_dir, pre)
    # -- the substrate and the assertions --------------------------------
    substrate = verify_frozen_substrate()
    m8_status = m8_frozen_set_status()
    pins = verify_sealed_inputs()
    r51_pin = r51_archive_pin()
    frozen_assertion = assert_frozen_values_against_r51(r51_pin)
    registered = (pre.get("zero_model") or {}).get(
        "renormalized_fractions") or {}
    if {k: float(v) for k, v in registered.items()} != \
            {k: float(v) for k, v in RENORMALIZED_FRACTIONS.items()}:
        _refuse(
            "the pre-registration's renormalized fractions drifted off "
            f"the frozen-point algebra: {registered!r} vs "
            f"{RENORMALIZED_FRACTIONS!r}")
    registered_frozen = (pre.get("one_model") or {}).get("frozen_values") \
        or {}
    if {k: float(v) for k, v in registered_frozen.items()} != \
            {k: float(v) for k, v in FROZEN_FIT.items()}:
        _refuse("the pre-registration's frozen values drifted off the "
                "driver constants")
    # -- the data side and the held-out geometry -------------------------
    _raw_config, tables = r51._load_production_tables()
    sealed = r51._load_sealed_r01(tables)
    geometry = r51.build_geometry(tables)
    sqrt_s_data = r51._sqrt_s_conversion()
    held = build_heldout_geometry(tables, sealed, geometry, sqrt_s_data)
    registered_counts = ((pre.get("held_out") or {}).get(
        "central_row_counts") or {})
    for species in PUBLISHED_SPECIES:
        if int(registered_counts.get(species, -1)) != \
                held["species"][species]["central_rows"]:
            _refuse(
                f"the pre-registration's central-row count for {species} "
                f"({registered_counts.get(species)!r}) differs from the "
                f"recomputed {held['species'][species]['central_rows']}")
    if int(registered_counts.get(PI0_SPECIES, -1)) != held["pi0"][
            "central_rows"]:
        _refuse("the pre-registration's pi0 central-row count differs "
                "from the recomputed count")
    # -- the multiplicity-tail cut (asserted against the
    #    pre-registration) ------------------------------------------------
    r50_reference = r51.r50_reference_payload()
    tail_cut = multiplicity_tail_cut(r50_reference)
    registered_tail = (pre.get("held_out") or {}).get(
        "multiplicity_tail_cut") or {}
    if int(registered_tail.get("threshold_n_ch", -1)) != \
            tail_cut["threshold_n_ch"] or \
            list(registered_tail.get("tail_bins", [])) != \
            tail_cut["tail_bins"]:
        _refuse(
            "the pre-registration's multiplicity tail cut drifted: "
            f"{registered_tail!r} vs {tail_cut!r}")
    tail_bins = tail_cut["tail_bins"]
    tail_idx = [n - MULTIPLICITY_BINS[0] for n in tail_bins]
    mult_dist = r50_reference["charged_multiplicity"]["distribution"]
    mult_var = r50_reference["charged_multiplicity"]["variance"]
    d_tail = np.asarray([float(mult_dist[str(n)]) for n in tail_bins],
                        dtype=float)
    v_tail = np.asarray([max(float(mult_var[str(n)]), SE_FLOOR)
                         for n in tail_bins], dtype=float)
    # -- the ONE side (the sealed R51 archive; the reuse) ----------------
    catalog = _load_catalog()
    fams = _load_families()
    blocks_one_raw, blocks_one, reuse_identities = one_side_blocks(
        run_dir, geometry, held, catalog, fams)
    weights_one = _weights_one_side(blocks_one_raw, {})
    # the multiplicity cross-check against the sealed R51 fit report
    mult_one_total, mult_one_blocks = _multiplicity_vectors(
        blocks_one_raw, weights_one["weights"])
    n_eff_total = weights_one["n_eff_total"]
    mult_one_pooled = mult_one_total / n_eff_total
    sealed_report = _read_json(MONO_ROOT / R51_RUN_DIR
                               / R51_FIT_REPORT_NAME)
    ag = [a for a in sealed_report["agreement"]
          if a["key"] == "charged_multiplicity_distribution"][0]
    sealed_model = np.asarray(ag["model_fit"], dtype=float)
    if not np.array_equal(mult_one_pooled, sealed_model):
        _refuse(
            "the one-side frozen-point multiplicity vector does not "
            "reproduce the sealed R51 fit report's model vector bit for "
            "bit (the weighted-evaluation identity check)")
    se_one_mult = se_from(np.asarray(
        [counts / n for counts, n in zip(
            mult_one_blocks, weights_one["n_eff_per_block"])],
        dtype=float))
    sealed_se = [None if v is None else float(v) for v in ag["mc_se"]]
    recomputed_se = [None if not math.isfinite(float(v)) else float(v)
                     for v in se_one_mult]
    mult_se_match = all(
        (a is None and b is None) or (a is not None and b is not None
                                      and a == b)
        for a, b in zip(sealed_se, recomputed_se))
    # -- the ZERO side (the regenerated records) -------------------------
    blocks_zero = zero_side_blocks(run_dir, geometry, smoke=False)
    weights_zero = [np.ones(block.n_events, dtype=float)
                    for block in blocks_zero]
    for block in blocks_zero:
        if block.n_events != EVENTS_PER_REPLICA:
            _refuse(f"the zero-model replica {block.replica} carries "
                    f"{block.n_events} events")
    mult_zero_total, mult_zero_blocks = _multiplicity_vectors(
        blocks_zero, weights_zero)
    n_eff_zero_total = float(sum(block.n_events for block in blocks_zero))
    mult_zero_pooled = mult_zero_total / n_eff_zero_total
    se_zero_mult = se_from(np.asarray(
        [counts / float(block.n_events) for counts, block in zip(
            mult_zero_blocks, blocks_zero)], dtype=float))
    # -- the central spectra comparison ----------------------------------
    dense_sizes = {
        "pi0": geometry["dense_size"][PI0_GEOMETRY_SPECIES]}
    for species in PUBLISHED_SPECIES:
        dense_sizes[species] = geometry["dense_size"][species]
    families = {}
    per_bin_all = {}
    for species in PUBLISHED_SPECIES:
        held_sp = held["species"][species]
        dense = dense_sizes[species]
        one_vec = _species_model_vectors(
            blocks_one, weights_one["weights"], species, held,
            sqrt_s_data, dense)
        zero_vec = _species_model_vectors(
            blocks_zero, weights_zero, species, held, sqrt_s_data, dense)
        pop_one = _pick_population(blocks_one, species, held, dense)
        pop_zero = _pick_population(blocks_zero, species, held, dense)
        d = held_sp["d_central"]
        v_data = held_sp["v_data_central"]
        bins = []
        kept = np.zeros(len(d), dtype=bool)
        for i in range(len(d)):
            data_ok = bool(np.isfinite(d[i]) and np.isfinite(v_data[i])
                           and v_data[i] >= 0.0)
            populated = bool(pop_one[i] and pop_zero[i])
            keep = data_ok and populated
            kept[i] = keep
            entry = {
                "bin": i,
                "xF_mid": float(held_sp["xF_mid_central"][i]),
                "pT_mid": float(held_sp["pT_mid_central"][i]),
                "y_mid": float(held_sp["y_mid_central"][i]),
                "d": float(d[i]),
                "v_data": float(v_data[i]),
                "m_one": float(one_vec["pooled"][i]),
                "m_zero": float(zero_vec["pooled"][i]),
                "se_one": (float(one_vec["se"][i])
                           if math.isfinite(float(one_vec["se"][i]))
                           else None),
                "se_zero": (float(zero_vec["se"][i])
                            if math.isfinite(float(zero_vec["se"][i]))
                            else None),
                "populated_one": bool(pop_one[i]),
                "populated_zero": bool(pop_zero[i]),
                "kept": keep,
            }
            if keep:
                entry.update(bin_comparison(
                    d[i], v_data[i], one_vec["pooled"][i],
                    zero_vec["pooled"][i],
                    one_vec["se"][i], zero_vec["se"][i]))
            else:
                entry["drop_reason"] = (
                    "non-finite data or published variance"
                    if not data_ok else
                    "structurally zero in every block (model: "
                    + ("one" if not pop_one[i] else "zero") + ")")
            bins.append(entry)
        kept_bins = [b for b in bins if b["kept"]]
        verdict = family_verdict(kept_bins)
        families[f"central_spectrum_{species}"] = {
            **verdict,
            "rows": int(held_sp["rows"]),
            "central_rows": int(held_sp["central_rows"]),
            "kept": int(kept.sum()),
            "one_records": one_vec["records_total"],
            "zero_records": zero_vec["records_total"],
            "one_dropped_out_of_window": one_vec["dropped_out_of_window"],
            "zero_dropped_out_of_window": zero_vec["dropped_out_of_window"],
            "one_n_eff_total": one_vec["n_eff_total"],
            "zero_n_eff_total": zero_vec["n_eff_total"],
        }
        per_bin_all[f"central_spectrum_{species}"] = bins
    # the pi0 model-only spectrum (no data side)
    held_pi0 = held["pi0"]
    dense = dense_sizes[PI0_SPECIES]
    one_vec = _species_model_vectors(
        blocks_one, weights_one["weights"], PI0_SPECIES, held,
        sqrt_s_data, dense)
    zero_vec = _species_model_vectors(
        blocks_zero, weights_zero, PI0_SPECIES, held, sqrt_s_data, dense)
    pi0_rows = []
    for i in range(int(held_pi0["central_rows"])):
        pi0_rows.append({
            "bin": i,
            "xF_mid": float(held_pi0["xF_mid_central"][i]),
            "pT_mid": float(held_pi0["pT_mid_central"][i]),
            "y_mid": float(held_pi0["y_mid_central"][i]),
            "m_one": float(one_vec["pooled"][i]),
            "m_zero": float(zero_vec["pooled"][i]),
            "se_one": (float(one_vec["se"][i])
                       if math.isfinite(float(one_vec["se"][i])) else None),
            "se_zero": (float(zero_vec["se"][i])
                        if math.isfinite(float(zero_vec["se"][i])) else None),
        })
    # -- the multiplicity tails ------------------------------------------
    bins = []
    kept = np.zeros(len(tail_bins), dtype=bool)
    for j, n_ch in enumerate(tail_bins):
        idx = tail_idx[j]
        data_ok = bool(np.isfinite(d_tail[j]) and np.isfinite(v_tail[j])
                       and v_tail[j] >= 0.0)
        populated = bool(mult_one_total[idx] > 0.0
                         and mult_zero_total[idx] > 0.0)
        keep = data_ok and populated
        kept[j] = keep
        m_one = float(mult_one_pooled[idx])
        m_zero = float(mult_zero_pooled[idx])
        se_one = float(se_one_mult[idx])
        se_zero = float(se_zero_mult[idx])
        entry = {
            "bin": int(n_ch),
            "d": float(d_tail[j]),
            "v_data": float(v_tail[j]),
            "m_one": m_one,
            "m_zero": m_zero,
            "se_one": se_one if math.isfinite(se_one) else None,
            "se_zero": se_zero if math.isfinite(se_zero) else None,
            "kept": keep,
        }
        if keep:
            entry.update(bin_comparison(
                d_tail[j], v_tail[j], m_one, m_zero, se_one, se_zero))
        else:
            entry["drop_reason"] = (
                "non-finite data or published variance"
                if not data_ok else
                "structurally zero in every block (model: "
                + ("one" if mult_one_total[idx] > 0.0 else "one and zero"
                   if mult_zero_total[idx] > 0.0 else "one") + ")")
        bins.append(entry)
    kept_tail = [b for b in bins if b["kept"]]
    tail_verdict = family_verdict(kept_tail)
    families["multiplicity_tails"] = {
        **tail_verdict,
        "threshold_n_ch": int(tail_cut["threshold_n_ch"]),
        "bins": [int(n) for n in tail_bins],
        "kept": int(kept.sum()),
        "data_anchor": (
            "the sealed R50-pinned reference (the R51 training "
            "distribution's data side)"),
    }
    per_bin_all["multiplicity_tails"] = bins
    # -- the R53 trigger ---------------------------------------------------
    trigger = _r53_trigger(kept_tail, kept_bins_all=kept_tail)
    # -- the conservation --------------------------------------------------
    conservation_zero = _pooled_zero_conservation(run_dir)
    r51_gate = _read_json(MONO_ROOT / R51_RUN_DIR / "gate5_r51.json")
    # -- the named residuals ------------------------------------------------
    named = _assemble_named_residuals(
        families, per_bin_all, weights_one, trigger,
        r51_inherited=sealed_report.get("named_residual_register") or [])
    # -- the report ----------------------------------------------------------
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "production",
        "statistics": {
            "tier": ("T2 (plan section 5.1 line 254): the zero model is "
                     "8 replicas x 100,000 p+p events = 800,000 "
                     "regenerated events; the one model reuses the "
                     "800,000 sealed R51 events"),
            "one_model_events": sum(b.n_events for b in blocks_one),
            "zero_model_events": sum(b.n_events for b in blocks_zero),
            "replicas": N_REPLICAS,
        },
        "frozen_values": {
            **{name: float(v) for name, v in FROZEN_FIT.items()},
            "assertion": frozen_assertion,
        },
        "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": 0.0,
            "regenerated": True,
            "events": sum(b.n_events for b in blocks_zero),
            "conservation": conservation_zero,
        },
        "one_model": {
            "source": "sealed R51 events",
            "frozen_source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            "refit": False,
            "events": sum(b.n_events for b in blocks_one),
            "reuse": {
                "mechanism": ONE_SIDE_REUSE_NOTE,
                "per_replica": reuse_identities,
                "all_bit_identical": bool(all(
                    e["bit_identical"] for e in reuse_identities)),
                "r51_archive_pin": {
                    "fit_report_sha256": r51_pin["fit_report_sha256"],
                    "replica_file_sha256": [
                        e["file_sha256"] for e in r51_pin["replicas"]],
                },
            },
            "weights": {
                "log_shift": weights_one["log_shift"],
                "n_eff_per_block": weights_one["n_eff_per_block"],
                "n_eff_total": weights_one["n_eff_total"],
                "ess_per_block": weights_one["ess_per_block"],
                "ess_min": weights_one["ess_min"],
                "note": (
                    "the exact per-event importance weights of the R51 "
                    "machinery at the frozen fitted point (the R51 "
                    "module's event_log_weights consumed verbatim; one "
                    "global log shift); the frozen-point ensemble "
                    "inherits the R51 weight degeneracy (the R51 named "
                    "residual)"),
            },
            "multiplicity_crosscheck": {
                "model_fit_matches_sealed_report": True,
                "mc_se_matches_sealed_report": bool(mult_se_match),
                "note": ("the one-side frozen-point multiplicity vector "
                         "reproduces the sealed R51 fit report's "
                         "charged-multiplicity model vector bit for bit "
                         "(the weighted-evaluation identity)"),
            },
        },
        "held_out": {
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "pi0_model_only": PI0_NOTE,
            "multiplicity_tails": MULTIPLICITY_TAIL_NOTE,
            "tail_cut": tail_cut,
            "recording_convention": RECORDING_CONVENTION_NOTE,
            "per_bin": per_bin_all,
            "pi0_spectrum_model_only": {
                "geometry_species": PI0_GEOMETRY_SPECIES,
                "mass_gev": PI0_ROW_MASS_GEV,
                "central_rows": int(held_pi0["central_rows"]),
                "rows": pi0_rows,
                "one_n_eff_total": one_vec["n_eff_total"],
                "zero_n_eff_total": zero_vec["n_eff_total"],
            },
        },
        "retention": {
            **{name: {key: value for key, value in family.items()
                      if key != "bound"}
               for name, family in families.items()},
            "bound": RETENTION_SE_BOUND,
            "algebra": RETENTION_ALGEBRA_NOTE,
            "all_families_retained": bool(all(
                family["verdict"] == "retained"
                for family in families.values())),
            "all_families_retained_note": (
                "informational rollup; the pre-registered retention "
                "evidence is the per-family verdicts"),
        },
        "trigger_for_r53": trigger,
        "likelihood": LIKELIHOOD_NOTE,
        "conservation": {
            "zero_model": conservation_zero,
            "one_model_cross_reference": {
                "source": f"{R51_RUN_DIR}/gate5_r51.json",
                "violations": int(r51_gate["conservation"]["violations"]),
                "pass": bool(r51_gate["conservation"]["pass"]),
                "worst_energy_closure_gev": float(
                    r51_gate["conservation"]["worst_energy_closure_gev"]),
                "note": ("the reused R51 events inherit the R51 "
                         "conservation record (cross-referenced from the "
                         "sealed R51 gate evidence); the re-derivation's "
                         "per-event conservation fields are hash-"
                         "verified against the sealed records"),
            },
            "tolerance": EVENT_CLOSURE_TOL,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "one_side_reuse": ONE_SIDE_REUSE_NOTE,
        "r53_trigger_definition": R53_TRIGGER_NOTE,
        "named_residual_register": named,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "no_separation_claim": (
            "this record claims no separation of exchange recoil from "
            "decay transverse dynamics (the R46 outcome, "
            "separation_claimed false); the exchange slopes act on the "
            "exchange kinematics only (plan section 24, line 1866)"),
        "determinism": DETERMINISM_NOTE,
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "m8_frozen_set": m8_status,
        "frozen_substrate": {
            name: {"sha256": digest, "verified": True}
            for name, digest in substrate.items()
        },
        "sealed_inputs": pins,
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


def _r53_trigger(kept_tail: list, kept_bins_all: list) -> dict:
    """The R53 trigger record (index item 5): the ONE model's largest
    multiplicity-tail pull against the data at the 2.0 pooled-SE bound;
    the zero model's tail residual recorded beside it."""
    if not kept_tail:
        return {
            "named_residual": None,
            "value": None,
            "pooled_se": None,
            "sigma": None,
            "triggered": False,
            "note": ("no kept multiplicity-tail bin; the trigger "
                     "condition cannot be demonstrated"),
            "zero_model_max_abs_pull": None,
            "booleans_agree": True,
        }
    worst_one = max(kept_tail, key=lambda b: abs(b["one_pull"]))
    worst_zero = max(kept_tail, key=lambda b: abs(b["zero_pull"]))
    sigma_one = abs(worst_one["one_pull"])
    sigma_zero = abs(worst_zero["zero_pull"])
    triggered = bool(sigma_one >= RETENTION_SE_BOUND)
    zero_triggered = bool(sigma_zero >= RETENTION_SE_BOUND)
    entry = {
        "named_residual": "multiplicity_tail_residual_one_model_beyond_"
                          "2x_pooled_se",
        "value": abs(worst_one["d"] - worst_one["m_one"]),
        "pooled_se": math.sqrt(
            float(worst_one["v_data"]) + float(worst_one["se_one"]) ** 2),
        "sigma": sigma_one,
        "triggered": triggered,
        "argmax_bin_n_ch": worst_one["bin"],
        "zero_model_max_abs_pull": sigma_zero,
        "zero_model_triggered": zero_triggered,
        "booleans_agree": bool(triggered == zero_triggered),
        "definition": R53_TRIGGER_NOTE,
    }
    if not entry["booleans_agree"]:
        entry["disagreement_note"] = (
            "the one-model and zero-model tail trigger booleans "
            "disagree; both are recorded and the gate consumes the "
            "one-model boolean (the R53 extension target)")
    return entry


def _pooled_zero_conservation(run_dir: Path) -> dict:
    """The pooled conservation audit over the 8 regenerated zero-model
    records (the full sample; the house contract)."""
    violations = 0
    events_checked = 0
    n_events = 0
    worst_closure = 0.0
    worst_shell = 0.0
    worst_energy_gev = 0.0
    worst_momentum_gev = 0.0
    bqs_census: dict[str, int] = {}
    for rep in range(N_REPLICAS):
        record = _read_json(Path(run_dir) / replica_name(rep))
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
        "pass": bool(violations == 0 and events_checked == n_events),
    }


def _assemble_named_residuals(families: dict, per_bin_all: dict,
                              weights_one: dict, trigger: dict,
                              r51_inherited: list) -> list:
    """The named-residual register (the house forms): the beyond-bound
    held-out bins, the weight degeneracy, the inherited R51 caveats, the
    instantiation notes, and the trigger entry."""
    residuals = []
    for family, bins in per_bin_all.items():
        for entry in bins:
            if entry.get("beyond_2x_se"):
                residuals.append({
                    "kind": "heldout_bin_degradation_beyond_2x_pooled_se",
                    "family": family,
                    "bin": entry["bin"],
                    "difference": entry["difference"],
                    "pooled_se": entry["pooled_se"],
                    "sigma": entry["difference"] / entry["pooled_se"],
                    "diagnosis": (
                        "the zero model's held-out deviance degrades this "
                        "bin by 2.0 pooled standard deviations or more "
                        "over the one model (the pre-registered bound)"),
                })
    ess_min = weights_one.get("ess_min")
    if ess_min is not None and math.isfinite(ess_min) and \
            ess_min < r51.ESS_FLOOR:
        residuals.append({
            "kind": "weight_degeneracy",
            "ess_min": ess_min,
            "per_block": weights_one.get("ess_per_block"),
            "diagnosis": (
                "the one model's frozen-point importance ensemble is "
                "degenerate (the effective sample size fraction below "
                "the pre-registered 0.01 bound; the R51 named residual "
                "carried into this comparison); the one-side MC errors "
                "and the preservation pulls carry that caveat"),
        })
    residuals.append({
        "kind": "r51_inherited_non_convergence",
        "diagnosis": (
            "the frozen point is the R51 optimum, which did not converge "
            "(the R51 named residuals optimizer_non_convergence, "
            "replica_block_minima_disagree, block_curvature_failure, "
            "weight_degeneracy); this frozen-parameter comparison "
            "inherits that state as a model-uncertainty caveat and fits "
            "nothing"),
        "r51_residual_kinds": [entry.get("kind")
                               for entry in r51_inherited],
    })
    residuals.append({
        "kind": "pi0_published_table_absent",
        "diagnosis": PI0_NOTE,
    })
    residuals.append({
        "kind": "multiplicity_tail_training_overlap",
        "diagnosis": MULTIPLICITY_TAIL_NOTE,
    })
    if trigger.get("triggered"):
        residuals.append({
            "kind": str(trigger["named_residual"]),
            "value": trigger["value"],
            "pooled_se": trigger["pooled_se"],
            "sigma": trigger["sigma"],
            "argmax_bin_n_ch": trigger.get("argmax_bin_n_ch"),
            "diagnosis": (
                "the one model's multiplicity-tail residual is at or "
                "above 2.0 pooled standard deviations; the named R52 "
                "residual that drives the R53 conditional trigger "
                "(index item 5); card 13 consumes this entry"),
        })
    return residuals


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
    """Render REPORT.md from the held-out report dict (the models, the
    reuse evidence, the comparison tables, the retention verdicts, the
    R53 trigger, the conservation audit, and the bookkeeping)."""
    lines = []
    lines.append("# R52 — central-excitation necessity test (run report)")
    lines.append("")
    lines.append(f"Run: **{RUN_ID}** (`{RUN_NAME}`), card `{CARD}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "The frozen-parameter deviance comparison (plan section 11 run "
        "R52, lines 920–922): the one-neutral-central-excitation model "
        "(the R51 sealed events, reused) against the zero-excitation "
        "model (the R51-frozen topology with the ND fraction zeroed and "
        "the remaining three inelastic fractions renormalized, "
        "regenerated at T2), on the held-out central pion and kaon "
        "spectra and the multiplicity tails. No optimizer: this is a "
        "frozen-parameter deviance comparison, not a refit.")
    lines.append("")
    stats = report["statistics"]
    lines.append(
        f"Sample: the one model reuses {stats['one_model_events']} sealed "
        f"R51 events (seeds 20300945–20300952, partition 1_100_000_000 + "
        f"i); the zero model regenerates {stats['zero_model_events']} "
        f"events in {stats['replicas']} replicas (T2, 8 x 100,000); "
        "seeds 20300953–20300960; event-index partition 1_200_000_000 + "
        "i, i < 100,000.")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    fv = report["frozen_values"]
    lines.append(
        "The frozen point (the sealed R51 fitted values, asserted "
        f"byte-exactly): sd_projectile_fraction "
        f"{_fmt(fv['sd_projectile_fraction'])}, sd_target_fraction "
        f"{_fmt(fv['sd_target_fraction'])}, dd_fraction "
        f"{_fmt(fv['dd_fraction'])}, nd_fraction {_fmt(fv['nd_fraction'])},"
        f" epsilon {_fmt(fv['epsilon'])}, B_SD {_fmt(fv['b_sd'])} GeV^-2.")
    lines.append("")
    zm = report["zero_model"]
    lines.append(
        "The zero model: the ND fraction exactly 0.0; the renormalized "
        f"fractions sd_projectile {_fmt(zm['renormalized_fractions']['sd_projectile'])}"
        f", sd_target {_fmt(zm['renormalized_fractions']['sd_target'])}, "
        f"dd {_fmt(zm['renormalized_fractions']['dd'])}; epsilon and "
        "B_SD at the frozen values; P(n >= 2) = 0.0 (the unretained "
        "card-8 default).")
    lines.append("")
    one = report["one_model"]
    lines.append(
        "The one model: the R51 sealed events consumed read-only. The "
        "sealed R51 records carry the proton and antiproton spectrum "
        "records but not the central pi/K records, so the held-out "
        "extraction re-derives the identical events from the sealed "
        "identity and verifies bit-identity against the sealed archive "
        "(every per-event record hash and every replica event-hash-set "
        "digest). All 8 replicas bit-identical: "
        f"**{one['reuse']['all_bit_identical']}**. The frozen-point "
        f"importance weights: pooled N_eff {_fmt(one['weights']['n_eff_total'], '.4f')}"
        f" of {stats['one_model_events']} events, ESS min "
        f"{_fmt(one['weights']['ess_min'], '.6f')} (the R51 weight "
        "degeneracy carried); the one-side frozen-point multiplicity "
        "vector reproduces the sealed R51 fit report's model vector "
        "bit for bit.")
    lines.append("")
    lines.append("## Held-out comparison")
    lines.append("")
    lines.append(
        "| family | bins | kept | max degradation (pooled SE) | zero "
        "degrades >= 2.0 | one max |pull| | one preserves | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, family in report["retention"].items():
        if name in ("bound", "algebra", "all_families_retained",
                    "all_families_retained_note"):
            continue
        lines.append(
            f"| {name} | {family.get('central_rows', family.get('bins'))} "
            f"| {family['kept']} "
            f"| {_fmt(family['max_degradation_pooled_se'], '.3f')} "
            f"| {'yes' if family['zero_degrades_at_2x_pooled_se'] else 'no'} "
            f"| {_fmt(family['one_model_max_abs_pull'], '.3f')} "
            f"| {'yes' if family['one_model_preserves'] else 'NO'} "
            f"| **{family['verdict']}** |")
    lines.append("")
    lines.append(
        "Per-bin records (data, both models' values with their "
        "between-replica MC errors, pulls, deviances, the difference, "
        "the pooled SE, and the beyond-bound flags) are in "
        "`held_out_report.json`. The pi0 central spectrum is recorded "
        "model-side only (no published pi0 table exists in the sealed "
        "R01 archive or the committed SPS data). The multiplicity-tail "
        "data side is the sealed R50-pinned reference; the tail bins "
        "are a subset of the R51 training bins and that overlap is "
        "recorded in the pre-registration and in the register.")
    lines.append("")
    trig = report["trigger_for_r53"]
    lines.append("## R53 trigger (index item 5)")
    lines.append("")
    lines.append(
        f"Named residual: `{trig['named_residual']}`; the one model's "
        f"largest multiplicity-tail pull {_fmt(trig['sigma'], '.4f')} "
        f"(value {_fmt(trig['value'], '.6f')} at n_ch = "
        f"{trig.get('argmax_bin_n_ch')}, pooled SE "
        f"{_fmt(trig['pooled_se'], '.3e')}); triggered at the 2.0 "
        f"pooled-SE bound: **{trig['triggered']}**. The zero model's "
        f"largest tail pull {_fmt(trig.get('zero_model_max_abs_pull'), '.4f')}"
        f" (triggered: {trig.get('zero_model_triggered')}; booleans "
        f"agree: {trig.get('booleans_agree')}). Card 13 consumes this "
        "entry; if it does not fire, card 13 records NOT_TRIGGERED and "
        "the multiple-excitation class remains implemented but "
        "unretained.")
    lines.append("")
    cons = report["conservation"]
    zc = cons["zero_model"]
    lines.append("## Conservation audit (the regenerated zero sample)")
    lines.append("")
    lines.append(
        f"Zero B, Q, S violations and zero four-momentum closure "
        f"failures at tolerance 1e-09 on all {zc['events_checked']} "
        f"regenerated events: worst closure residual "
        f"{_fmt(zc['worst_closure_relative'], '.3e')} relative, worst "
        f"energy closure {_fmt(zc['worst_energy_closure_gev'], '.3e')} "
        f"GeV, worst momentum closure "
        f"{_fmt(zc['worst_momentum_closure_gev'], '.3e')} GeV, worst "
        f"mass-shell deviation {_fmt(zc['worst_mass_shell_relative'], '.3e')} "
        f"relative. Verdict: {'PASS' if zc['pass'] else 'FAIL'}. The "
        "reused R51 events inherit the R51 conservation record "
        f"({cons['one_model_cross_reference']['source']}: "
        f"{cons['one_model_cross_reference']['violations']} violations, "
        "worst energy closure "
        f"{_fmt(cons['one_model_cross_reference']['worst_energy_closure_gev'], '.3e')} GeV).")
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
        "comparison (recorded, not folded into the frozen comparison's "
        "arithmetic). The calculation-engine moment floor (~1e-3 GeV/c) "
        "is tracked on its separate track.")
    lines.append("")
    lines.append("## Bookkeeping")
    lines.append("")
    lines.append(
        "- `optimizer_invoked`: false; `optimizer_uses`: 0 (R52 is a "
        "frozen-parameter deviance comparison, not a refit; index item "
        "10).")
    lines.append(
        "- The frozen simplified cascade at the sealed R45 point is "
        "consumed without refit (the R46 no_refit clause); the blend "
        "ramp start stays fixed at B = 2.680194110727528 GeV; no Round "
        "5 record claims a separation of exchange recoil from decay "
        "transverse dynamics.")
    lines.append(
        "- The zero/one pair shares the feed-down lineage and the "
        "normalization (plan line 1872): both models run the same M8 "
        "p+p configuration, the same card-9 topology runner, the same "
        "frozen cascade, and the same R01 data-units conversion.")
    lines.append(
        "- Determinism: the per-event hash sets and the one-side "
        "bit-identity re-derivation carry the identity evidence; the "
        "clean-environment validation re-runs the held-out evaluation "
        "and both probe replicas (zero-model replica 0 and R51 replica "
        "0) from a clean interpreter before the seal.")
    lines.append("")
    lines.append(f"Report digest: `{report['digest']}`.")
    lines.append("")
    (run_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


# ---------------------------------------------------------------------------
# The hermetic smoke mode (driver validation; never sealed)
# ---------------------------------------------------------------------------
def _load_smoke_tables():
    """The synthetic hermetic binning grid of the smoke mode: the four
    published pi/K tables built from the committed table declarations
    (species, pdg, mass, normalization) only; no SPS table is read (the
    R32 smoke-grid pattern)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    raw_config = json.loads((NESTED_ROOT / "config"
                             / r51.r01.PRODUCTION_CONFIG_NAME).read_text(
                                 encoding="utf-8"))
    binned = (raw_config.get("fit") or {}).get("binned") or {}
    if not binned or not binned.get("tables"):
        _refuse("config carries no fit.binned tables block; the smoke "
                "grid needs the committed table declarations")
    n_x = 10
    n_p = 10
    window = dict(binned.get("window") or {})
    xf_lo = float(window.get("xf_min", r51.r01.PAPER_WINDOW["xF"][0]))
    xf_hi = float(window.get("xf_max", r51.r01.PAPER_WINDOW["xF"][1]))
    pt_lo = float(window.get("pt_min", r51.r01.PAPER_WINDOW["pT"][0]))
    pt_hi = float(window.get("pt_max", r51.r01.PAPER_WINDOW["pT"][1]))
    xf_centers = np.linspace(xf_lo, xf_hi, n_x + 2)[1:-1]
    pt_centers = np.linspace(pt_lo, pt_hi, n_p + 2)[1:-1]
    tables = []
    for entry in binned["tables"]:
        if str(entry["species"]) not in PUBLISHED_SPECIES:
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
    if len(tables) != len(PUBLISHED_SPECIES):
        _refuse("the smoke grid built "
                f"{len(tables)} tables, expected the four published "
                f"species {PUBLISHED_SPECIES}")
    return tables


def _smoke_sealed_vectors(tables) -> dict:
    """The smoke-mode synthetic sealed vectors (row coordinates from the
    synthetic grid; the data values are injected later)."""
    sealed = {}
    for table in tables:
        sealed[str(table.species)] = {
            "d": [0.0] * int(table.data.size),
            "v_data": [float(SE_FLOOR)] * int(table.data.size),
            "xF_mid": [float(table.xf_centers[int(table.bin_index_xf[i])])
                       for i in range(int(table.data.size))],
            "pT_mid": [float(table.pt_centers[int(table.bin_index_pt[i])])
                       for i in range(int(table.data.size))],
        }
    return sealed


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


def _smoke_evaluation(run_dir: Path, pre: dict) -> dict:
    """The smoke held-out evaluation: the zero-model smoke records, the
    one-side reuse-machinery exercise (the smoke events re-derived from
    the smoke seed and verified bit-identical against the smoke
    record), and the comparison against synthetic data (the zero side's
    pooled vectors with a synthetic 10% published error; no SPS table
    and no sealed archive read; the R41 synthetic-data pattern)."""
    catalog = _load_catalog()
    fams = _load_families()
    tables = _load_smoke_tables()
    sealed = _smoke_sealed_vectors(tables)
    geometry = r51.build_geometry(tables)
    sqrt_s_data = SQRT_S_EXTRACTION_GEV
    held = build_heldout_geometry(tables, sealed, geometry, sqrt_s_data)
    tail_cut = multiplicity_tail_cut(_smoke_r50_reference())
    tail_bins = tail_cut["tail_bins"]
    tail_idx = [n - MULTIPLICITY_BINS[0] for n in tail_bins]
    # -- the ZERO side from the smoke records -----------------------------
    blocks_zero = zero_side_blocks(run_dir, geometry, smoke=True)
    weights_zero = [np.ones(block.n_events, dtype=float)
                    for block in blocks_zero]
    mult_zero_total, mult_zero_blocks = _multiplicity_vectors(
        blocks_zero, weights_zero)
    n_eff_zero_total = float(sum(b.n_events for b in blocks_zero))
    mult_zero_pooled = mult_zero_total / n_eff_zero_total
    se_zero_mult = se_from(np.asarray(
        [c / float(b.n_events) for c, b in zip(mult_zero_blocks, blocks_zero)],
        dtype=float))
    # -- the ONE side: the reuse-machinery exercise (the smoke events
    #    re-derived from the smoke seed and verified bit-identical) ------
    sealed_record = _read_json(run_dir / SMOKE_REPLICA_NAME)
    lattice, identity = rederive_smoke_replica(
        sealed_record, catalog, fams)
    r51_blocks = _smoke_r51_style_blocks(sealed_record, geometry)
    weights_one = _weights_one_side(r51_blocks, {})
    mult_one_total, mult_one_blocks = _multiplicity_vectors(
        r51_blocks, weights_one["weights"])
    n_eff_one_total = weights_one["n_eff_total"]
    mult_one_pooled = mult_one_total / n_eff_one_total
    se_one_mult = se_from(np.asarray(
        [c / n for c, n in zip(mult_one_blocks,
                               weights_one["n_eff_per_block"])],
        dtype=float))
    one_side_block = _smoke_heldout_block(lattice, sealed_record, geometry)
    blocks_one = [one_side_block]
    # -- the synthetic data (the zero side's pooled vectors, 10% error) ---
    dense_sizes = {species: geometry["dense_size"][
        (PI0_GEOMETRY_SPECIES if species == PI0_SPECIES else species)]
        for species in SPECIES_PDGS}
    zero_pooled = {}
    for species in PUBLISHED_SPECIES:
        zero_pooled[species] = _species_model_vectors(
            blocks_zero, weights_zero, species, held, sqrt_s_data,
            dense_sizes[species])
    for species in PUBLISHED_SPECIES:
        held_sp = held["species"][species]
        values = np.asarray(zero_pooled[species]["pooled"], dtype=float)
        finite = np.isfinite(values)
        d = np.where(finite, values, 0.0)
        v = np.where(finite, (0.1 * np.abs(d)) ** 2, float("inf"))
        v = np.maximum(v, float(SE_FLOOR))
        held_sp["d_central"] = d
        held_sp["v_data_central"] = v
    d_tail = np.asarray(
        [float(mult_zero_pooled[j]) if mult_zero_pooled[j] > 0.0 else 0.0
         for j in tail_idx], dtype=float)
    v_tail = np.maximum((0.1 * np.abs(d_tail)) ** 2, float(SE_FLOOR))
    # -- the comparison ----------------------------------------------------
    families = {}
    per_bin_all = {}
    for species in PUBLISHED_SPECIES:
        held_sp = held["species"][species]
        dense = dense_sizes[species]
        one_vec = _species_model_vectors(
            blocks_one, weights_one["weights"], species, held,
            sqrt_s_data, dense_sizes[species])
        zero_vec = zero_pooled[species]
        pop_one = _pick_population(blocks_one, species, held, dense)
        pop_zero = _pick_population(blocks_zero, species, held, dense)
        d = held_sp["d_central"]
        v_data = held_sp["v_data_central"]
        bins = []
        for i in range(len(d)):
            data_ok = bool(np.isfinite(d[i]) and np.isfinite(v_data[i])
                           and v_data[i] >= 0.0)
            populated = bool(pop_one[i] and pop_zero[i])
            keep = data_ok and populated
            entry = {
                "bin": i,
                "xF_mid": float(held_sp["xF_mid_central"][i]),
                "pT_mid": float(held_sp["pT_mid_central"][i]),
                "d": float(d[i]),
                "v_data": float(v_data[i]),
                "m_one": float(one_vec["pooled"][i]),
                "m_zero": float(zero_vec["pooled"][i]),
                "se_one": (float(one_vec["se"][i])
                           if math.isfinite(float(one_vec["se"][i]))
                           else None),
                "se_zero": (float(zero_vec["se"][i])
                            if math.isfinite(float(zero_vec["se"][i]))
                            else None),
                "kept": keep,
            }
            if keep:
                entry.update(bin_comparison(
                    d[i], v_data[i], one_vec["pooled"][i],
                    zero_vec["pooled"][i], one_vec["se"][i],
                    zero_vec["se"][i]))
            else:
                entry["drop_reason"] = (
                    "non-finite data or published variance"
                    if not data_ok else
                    "structurally zero in every block (model: "
                    + ("one" if not pop_one[i] else "zero") + ")")
            bins.append(entry)
        kept_bins = [b for b in bins if b["kept"]]
        verdict = family_verdict(kept_bins)
        families[f"central_spectrum_{species}"] = {
            **verdict,
            "central_rows": int(held_sp["central_rows"]),
            "kept": len(kept_bins),
        }
        per_bin_all[f"central_spectrum_{species}"] = bins
    bins = []
    for j, n_ch in enumerate(tail_bins):
        idx = tail_idx[j]
        populated = bool(mult_one_total[idx] > 0.0
                         and mult_zero_total[idx] > 0.0)
        keep = bool(np.isfinite(v_tail[j]) and populated)
        m_one = float(mult_one_pooled[idx])
        m_zero = float(mult_zero_pooled[idx])
        se_one = float(se_one_mult[idx])
        se_zero = float(se_zero_mult[idx])
        entry = {
            "bin": int(n_ch),
            "d": float(d_tail[j]),
            "v_data": float(v_tail[j]),
            "m_one": m_one,
            "m_zero": m_zero,
            "se_one": se_one if math.isfinite(se_one) else None,
            "se_zero": se_zero if math.isfinite(se_zero) else None,
            "kept": keep,
        }
        if keep:
            entry.update(bin_comparison(d_tail[j], v_tail[j], m_one,
                                        m_zero, se_one, se_zero))
        else:
            entry["drop_reason"] = (
                "structurally zero in every block (model: "
                + ("one" if mult_one_total[idx] > 0.0 else
                   "one and zero" if mult_zero_total[idx] > 0.0
                   else "one") + ")")
        bins.append(entry)
    kept_tail = [b for b in bins if b["kept"]]
    tail_verdict = family_verdict(kept_tail)
    families["multiplicity_tails"] = {
        **tail_verdict,
        "threshold_n_ch": int(tail_cut["threshold_n_ch"]),
        "bins": [int(n) for n in tail_bins],
        "kept": len(kept_tail),
    }
    per_bin_all["multiplicity_tails"] = bins
    trigger = _r53_trigger(kept_tail, kept_tail)
    named = _assemble_named_residuals(
        families, per_bin_all, weights_one, trigger, r51_inherited=[])
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "statistics": {
            "tier": ("the smoke statistics: one zero-model replica x "
                     "2,000 events at the smoke seed 20312925; the smoke "
                     "exercises pre-registration, zero-model generation, "
                     "the one-side reuse-machinery exercise, recording, "
                     "and the held-out evaluation end to end"),
            "one_model_events": sum(b.n_events for b in blocks_one),
            "zero_model_events": sum(b.n_events for b in blocks_zero),
            "replicas": 1,
        },
        "frozen_values": {
            **{name: float(v) for name, v in FROZEN_FIT.items()},
            "assertion": None,
            "smoke_note": ("the smoke asserts the frozen-value constants "
                           "against no sealed archive; the production "
                           "assertion runs in --pre-registration and "
                           "--evaluate"),
        },
        "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "regenerated": True,
            "events": sum(b.n_events for b in blocks_zero),
            "conservation": _read_json(
                run_dir / SMOKE_REPLICA_NAME)["conservation"],
        },
        "one_model": {
            "source": "the smoke's own re-derived events (the "
                      "reuse-machinery exercise; no sealed archive is "
                      "read)",
            "refit": False,
            "events": sum(b.n_events for b in blocks_one),
            "reuse": {
                "mechanism": ONE_SIDE_REUSE_NOTE,
                "smoke_exercise": identity,
                "all_bit_identical": bool(identity["bit_identical"]),
            },
            "weights": {
                "log_shift": weights_one["log_shift"],
                "n_eff_per_block": weights_one["n_eff_per_block"],
                "n_eff_total": weights_one["n_eff_total"],
                "ess_per_block": weights_one["ess_per_block"],
                "ess_min": weights_one["ess_min"],
            },
            "multiplicity_crosscheck": {
                "model_fit_matches_sealed_report": None,
                "note": ("the smoke reads no sealed archive; the "
                         "cross-check runs in production"),
            },
        },
        "held_out": {
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "pi0_model_only": PI0_NOTE,
            "multiplicity_tails": MULTIPLICITY_TAIL_NOTE,
            "tail_cut": tail_cut,
            "recording_convention": RECORDING_CONVENTION_NOTE,
            "per_bin": per_bin_all,
            "pi0_spectrum_model_only": {
                "central_rows": int(held["pi0"]["central_rows"]),
                "rows": [],
            },
        },
        "retention": {
            **{name: {key: value for key, value in family.items()
                      if key != "bound"}
               for name, family in families.items()},
            "bound": RETENTION_SE_BOUND,
            "algebra": RETENTION_ALGEBRA_NOTE,
            "all_families_retained": bool(all(
                family["verdict"] == "retained"
                for family in families.values())),
            "all_families_retained_note": (
                "informational rollup; the pre-registered retention "
                "evidence is the per-family verdicts"),
        },
        "trigger_for_r53": trigger,
        "likelihood": LIKELIHOOD_NOTE,
        "conservation": {
            "zero_model": _read_json(
                run_dir / SMOKE_REPLICA_NAME)["conservation"],
            "one_model_cross_reference": {
                "source": "none in smoke (no sealed archive read)",
                "violations": 0,
                "pass": True,
                "worst_energy_closure_gev": None,
            },
            "tolerance": EVENT_CLOSURE_TOL,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "one_side_reuse": ONE_SIDE_REUSE_NOTE,
        "r53_trigger_definition": R53_TRIGGER_NOTE,
        "named_residual_register": named,
        "optimizer_invoked": False,
        "optimizer_uses": 0,
        "no_separation_claim": (
            "no Round 5 record claims a separation of exchange recoil "
            "from decay transverse dynamics (the R46 outcome, "
            "separation_claimed false)"),
        "determinism": DETERMINISM_NOTE,
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "smoke_note": (
            "the smoke artifacts are never sealed: --gate / --seal / "
            "--clean-validate refuse smoke artifacts (the R31/8c "
            "convention); two identical smoke invocations produce "
            "byte-identical artifacts (no wall-clock and no git state "
            "in the smoke artifacts); the smoke evaluation's data side "
            "is synthetic (the zero side's pooled vectors with a "
            "synthetic 10% published error; no SPS table and no sealed "
            "archive is read); the smoke's one-side weights are the "
            "frozen-point weights relative to the 0.25 nominal "
            "reference applied to the smoke's renormalized-fraction "
            "sample, a machinery exercise whose estimator is not the "
            "production one-side estimator"),
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


def rederive_smoke_replica(sealed_record: dict, catalog, fams):
    """The smoke-mode reuse-machinery exercise: the smoke zero-model
    events re-derived from the smoke seed and verified bit-identical
    against the smoke record (the production reuse path exercised
    hermetically; the compact zero-format records carry the same
    weight-input fields, so the recomputed per-event hashes cover the
    ``pk`` block too)."""
    seed = int(sealed_record["seed"])
    base = int(SMOKE_EVENT_INDEX_BASE)
    n_events = len(sealed_record["events"])
    per_species = {species: ([], [], [], []) for species in SPECIES_PDGS}
    hashes = []
    mismatches = []
    worst_closure = 0.0
    for i in range(n_events):
        subs = derive_substreams(seed, 0, base + i)
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
            zero_model_params().class_fractions, 0.0, rng_topology_selection)
        if label not in ("sd_projectile", "sd_target", "dd"):
            raise ConservationViolation(
                f"event {base + i}: the smoke re-derivation's class draw "
                f"returned {label!r} outside the zero-model set")
        event = run_topology(
            label, zero_model_params(), catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=0.0)
        obs = extract_observables(event, SQRT_S)
        conservation = event_conservation(event)
        pik = central_spectrum_records(event)
        n_ch = charged_multiplicity(event)
        record = serialize_zero_event(event, obs, conservation, pik, n_ch)
        record["i"] = int(i)
        record["id"] = int(base + i)
        record.pop("h", None)
        record["h"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8")).hexdigest()
        hashes.append(record["h"])
        if record["h"] != sealed_record["events"][i].get("h"):
            mismatches.append(int(i))
        worst_closure = max(worst_closure, conservation["closure_worst"])
        for species, triples in pik.items():
            for x_f, p_t, measure in triples:
                per_species[species][0].append(x_f)
                per_species[species][1].append(p_t)
                per_species[species][2].append(measure)
                per_species[species][3].append(i)
    hash_set = hashlib.sha256(
        "\n".join(hashes).encode("utf-8")).hexdigest()
    identity = {
        "replica": "smoke",
        "seed": seed,
        "events": n_events,
        "per_event_hash_mismatches": len(mismatches),
        "event_hash_set_match": bool(
            hash_set == sealed_record.get("event_hash_set_sha256")),
        "worst_closure_relative": worst_closure,
        "bit_identical": bool(not mismatches and hash_set ==
                              sealed_record.get("event_hash_set_sha256")),
    }
    if not identity["bit_identical"]:
        _refuse("the smoke re-derivation is not bit-identical to the "
                "smoke record (the reuse-machinery exercise refuses)")
    lattice = {}
    for species, (xf, pt, w, evt) in per_species.items():
        lattice[species] = (
            np.asarray(xf, dtype=float), np.asarray(pt, dtype=float),
            np.asarray(w, dtype=float), np.asarray(evt, dtype=np.int64))
    return lattice, identity


def _smoke_heldout_block(lattice, sealed_record, geometry) -> HeldoutBlock:
    """The smoke one-side held-out block from the re-derived lattice."""
    block = HeldoutBlock()
    block.replica = "smoke"
    block.seed = int(sealed_record["seed"])
    block.n_events = int(sealed_record["n_events"])
    block.nch_idx = np.asarray(
        [int(ev["nc"]) - MULTIPLICITY_BINS[0]
         for ev in sealed_record["events"]], dtype=np.int64)
    block.species_flat, block.species_evt, block.species_w = {}, {}, {}
    block.species_dropped, block.species_total = {}, {}
    for species in SPECIES_PDGS:
        xf, pt, w, evt = lattice[species]
        if species == PI0_SPECIES:
            edges = geometry["edges"][PI0_GEOMETRY_SPECIES]
            n_x = geometry["n_x"][PI0_GEOMETRY_SPECIES]
        else:
            edges = geometry["edges"][species]
            n_x = geometry["n_x"][species]
        flat, kept_evt, kept_w, dropped = _assign_species_cells(
            xf, pt, w, evt, edges, n_x)
        block.species_flat[species] = flat
        block.species_evt[species] = kept_evt
        block.species_w[species] = kept_w
        block.species_dropped[species] = dropped
        block.species_total[species] = int(xf.size)
    return block


def _smoke_r51_style_blocks(sealed_record, geometry) -> list:
    """The smoke stand-in for the R51 loader's blocks: the weight-input
    arrays built from the smoke record's fields (the class rows the R51
    weight machinery consumes)."""
    events = sealed_record["events"]
    n_events = len(events)
    cls = np.empty(n_events, dtype=np.int64)
    rows_by_class: dict[str, list] = {c: [] for c in CLASS_ORDER}
    for i, ev in enumerate(events):
        label = ev["cl"]
        if label not in r51.CLASS_INDEX:
            _refuse(f"the smoke event carries an unknown class: {label!r}")
        cls[i] = r51.CLASS_INDEX[label]
        ex, tr, rm = ev["ex"], ev["tr"], ev["rm"]
        if label in ("sd_projectile", "sd_target"):
            rows_by_class[label].append((ex[0] ** 2, tr[0], rm[0]))
        elif label == "dd":
            rows_by_class[label].append(
                (ex[0] ** 2, ex[1] ** 2, tr[0], tr[1]))
        else:
            rows_by_class[label].append(
                (tr[0], tr[1], tr[2], ex[0] ** 2, rm[0], rm[1]))
    block = r51.Block()
    block.replica = "smoke"
    block.seed = sealed_record.get("seed")
    block.n_events = n_events
    block.cls = cls
    block.idx_by_class = {
        cls_label: np.flatnonzero(cls == r51.CLASS_INDEX[cls_label])
        for cls_label in CLASS_ORDER}
    builders = {
        "sd_projectile": lambda rows: {
            "m2": np.asarray([r[0] for r in rows], dtype=float),
            "t": np.asarray([r[1] for r in rows], dtype=float),
            "m_rem": np.asarray([r[2] for r in rows], dtype=float)},
        "sd_target": lambda rows: {
            "m2": np.asarray([r[0] for r in rows], dtype=float),
            "t": np.asarray([r[1] for r in rows], dtype=float),
            "m_rem": np.asarray([r[2] for r in rows], dtype=float)},
        "dd": lambda rows: {
            "m12": np.asarray([r[0] for r in rows], dtype=float),
            "m22": np.asarray([r[1] for r in rows], dtype=float),
            "t1": np.asarray([r[2] for r in rows], dtype=float),
            "t2": np.asarray([r[3] for r in rows], dtype=float)},
        "nd": lambda rows: {
            "t1": np.asarray([r[0] for r in rows], dtype=float),
            "t2": np.asarray([r[1] for r in rows], dtype=float),
            "t_c": np.asarray([r[2] for r in rows], dtype=float),
            "mc2": np.asarray([r[3] for r in rows], dtype=float),
            "m1": np.asarray([r[4] for r in rows], dtype=float),
            "m2": np.asarray([r[5] for r in rows], dtype=float)},
    }
    block.arrays_by_class = {
        cls_label: builders[cls_label](rows_by_class[cls_label])
        for cls_label in CLASS_ORDER}
    block.gap = np.zeros(n_events, dtype=bool)
    block.nch_idx = np.asarray(
        [int(ev["nc"]) - MULTIPLICITY_BINS[0] for ev in events],
        dtype=np.int64)
    block.prot_flat = np.zeros(0, dtype=np.int64)
    block.pbar_flat = np.zeros(0, dtype=np.int64)
    return [block]

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
        r51_pin = r51_archive_pin()
        frozen_assertion = assert_frozen_values_against_r51(r51_pin)
        r50_reference = r51.r50_reference_payload()
        tail_cut = multiplicity_tail_cut(r50_reference)
        _raw_config, tables = r51._load_production_tables()
        sealed = r51._load_sealed_r01(tables)
        geometry = r51.build_geometry(tables)
        sqrt_s_data = r51._sqrt_s_conversion()
        held = build_heldout_geometry(tables, sealed, geometry, sqrt_s_data)
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
            "pi0_note": PI0_NOTE,
            "cross_check": "the sealed d values match the loaded SPS "
                           "tables bit for bit (the R11 discipline)",
        }
    else:
        sealed_inputs = None
        r51_pin = None
        frozen_assertion = None
        r50_reference = None
        tail_cut = multiplicity_tail_cut(_smoke_r50_reference())
        central_row_counts = None
        r01_summary = None
    payload = {
        "schema": ("na49_mixing.staged_r52_central_excitation_necessity."
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
        "one_model": {
            "source": "sealed R51 events",
            "run_dir": R51_RUN_DIR,
            "frozen_source": f"{R51_RUN_DIR}/{R51_FIT_REPORT_NAME}",
            "frozen_values": dict(FROZEN_FIT),
            "assertion_against_sealed_r51": frozen_assertion,
            "regenerated": False,
            "events": N_EVENTS_TOTAL,
            "reuse_mechanism": ONE_SIDE_REUSE_NOTE,
            "archive_digest_pin": r51_pin,
        },
        "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "renormalization_rule": _FROZEN_STATUS["renormalization_rule"],
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": 0.0,
            "regenerated": True,
            "events": N_EVENTS_TOTAL,
            "assertion_at_generation": (
                "the draw parameters are asserted at generation: the ND "
                "fraction exactly 0.0, the renormalized fractions "
                "byte-identical to this pre-registration, epsilon and "
                "B_SD at the frozen values, P(n >= 2) = 0.0; a drawn "
                "elastic or nd label refuses the run"),
        },
        "statistics": {
            "tier": ("T2 (plan section 5.1 line 254, the upper T2 bound; "
                     "the R23/R32/R41 convention): the zero model is 8 "
                     "replicas x 100,000 p+p events = 800,000 "
                     "regenerated events; the one model reuses the "
                     "800,000 sealed R51 events"),
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw zero-model replica seed = 20260925 + 40000 + 28 + k "
                f"= 20300953 + k per replica k < {N_REPLICAS} (index item "
                "9; the block 20300953-20300960 is disjoint from every "
                "previously used raw block and from every other reserved "
                "Round 5 block)"),
            "raw_replica_seeds": list(REPLICA_SEEDS),
            "event_index_rule": EVENT_INDEX_RULE,
            "event_index_base": EVENT_INDEX_BASE,
            "substream_labels": list(SUBSTREAM_LABELS),
            "disjointness": disjointness,
            "one_side_reuse": (
                "the one model re-consumes the sealed R51 raw block "
                "20300945-20300952 on the partition 1_100_000_000 + i "
                "bit-exactly (the reuse contract; the sealed draws are "
                "reproduced, no new draw happens)"),
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
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "central_species_note": (
                "the central pion and kaon spectra of pi+, pi-, K+, K- "
                "in the central rapidity window |y^CM| < 0.5 from the "
                "sealed R01 published binned tables (the published rows "
                "whose row-mid y^CM at the species mass falls inside the "
                "window), recorded with the dense-accumulation/then-pick "
                "convention"),
            "pi0_note": PI0_NOTE,
            "multiplicity_tails_note": MULTIPLICITY_TAIL_NOTE,
            "multiplicity_tail_cut": tail_cut,
            "central_row_counts": central_row_counts,
            "r01_data": r01_summary,
            "disjointness": (
                "the central pi/K spectra are disjoint from every R51 "
                "training observable (the R51 training set is the "
                "leading-proton spectrum, the B_transport spectrum, the "
                "gap fraction, and the charged-multiplicity "
                "distribution); the multiplicity-tail bins are a subset "
                "of the R51 training multiplicity bins and share the "
                "R50-pinned reference anchor (recorded above; the "
                "sealed substrate carries no published multiplicity "
                "table); plan section 7.4 line 453 stands: a failed "
                "held-out observable is never moved into training"),
        },
        "retention_rule": {
            "bound": RETENTION_SE_BOUND,
            "statement": (
                "the one-excitation topology is retained only if the "
                "zero model degrades the held-out deviance by 2.0 pooled "
                "standard deviations or more on the pre-registered bins "
                "and the one model preserves them (plan section 5.5 "
                "items 2-3, lines 318-319)"),
            "algebra": RETENTION_ALGEBRA_NOTE,
            "recorded_per_family": True,
        },
        "r53_trigger": {
            "definition": R53_TRIGGER_NOTE,
            "bound": RETENTION_SE_BOUND,
            "recorded_in": GATE_RECORD_NAME,
        },
        "likelihood": LIKELIHOOD_NOTE,
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "conservation": {
            "contract": (
                "every regenerated zero-model event conserves total B, "
                "Q, and S exactly (integer arithmetic) and closes the "
                "total four-momentum at tolerance 1e-09 relative against "
                "the incoming sum; a violation raises and stops the "
                "run.  The reused R51 events inherit the R51 "
                "conservation record (cross-referenced from the sealed "
                "R51 gate evidence)"),
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
            "R52 is a frozen-parameter deviance comparison, not a refit "
            "(index item 10); optimizer_invoked is false, optimizer_uses "
            "is 0, and the driver carries no optimizer"),
        "frozen_cascade": dict(R45_FROZEN_CASCADE),
        "m8_frozen_set": m8_status,
        "frozen_substrate": {
            name: {"sha256": _sha256_file(MONO_ROOT / rel_path),
                   "expected": expected}
            for name, (rel_path, expected) in FROZEN_SUBSTRATE.items()
        },
        "sealed_inputs": sealed_inputs,
        "generation": GENERATION_NOTE,
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r52_central_excitation_"
                       "necessity.provenance/1"),
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
            "r51_driver": {
                "nested": {
                    "path": R51_DRIVER_REL_NESTED,
                    "sha256": _sha256_file(
                        MONO_ROOT / R51_DRIVER_REL_NESTED),
                },
                "outer": {
                    "path": R51_DRIVER_REL_OUTER,
                    "sha256": _sha256_file(
                        MONO_ROOT / R51_DRIVER_REL_OUTER),
                },
                "byte_identical": (
                    _sha256_file(MONO_ROOT / R51_DRIVER_REL_NESTED)
                    == _sha256_file(MONO_ROOT / R51_DRIVER_REL_OUTER)),
                "role": ("the consumed read-only weight/record machinery "
                         "(the one-side frozen-point importance weights "
                         "and the sealed-archive loaders are the R51 "
                         "module's functions verbatim)"),
            },
            "round5_modules": round5_module_digests(),
            "frozen_substrate": {
                name: {"sha256": digest, "verified": True}
                for name, digest in substrate.items()
            },
            "m8_frozen_set": m8_status,
            "sealed_inputs": sealed_inputs,
            "r51_archive_pin": r51_pin,
            "r01_data": r01_summary,
            "r45_r46_r50_r51_driver_pins": {
                "r45": FROZEN_SUBSTRATE["r45_driver"][1],
                "r46": FROZEN_SUBSTRATE["r46_driver"][1],
                "r50": FROZEN_SUBSTRATE["r50_driver"][1],
                "r51": FROZEN_SUBSTRATE["r51_driver"][1],
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
        if not provenance["r51_driver"]["byte_identical"]:
            _refuse(
                "the consumed R51 dual-tree driver copies are not "
                "byte-identical")
        _write_json(run_dir / PROVENANCE_NAME, provenance)
        print(f"[r52] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r52] pre-registration written: {pre_path}", flush=True)
    return payload


OBJECTIVE_NOTE = (
    "plan section 11 run R52 (lines 920-922): 'Compare models with zero "
    "and one neutral central mesonic excitation using held-out central "
    "pion and kaon spectra and multiplicity tails.'  The pre-registered "
    "instantiation (index items 4, 8, and 10): the one model is the "
    "R51-frozen topology and reuses the R51 sealed events (no new "
    "generation); the zero model is the R51-frozen topology with the ND "
    "central-excitation class fraction set to zero and the remaining "
    "three inelastic fractions (SD_p, SD_t, DD) renormalized to sum to "
    "one, regenerated at T2 (8 x 100,000) under the R52 zero-model seed "
    "block.  The comparison observables are held out and "
    "pre-registered: the central pion and kaon spectra of pi+, pi-, "
    "pi0, K+, K- in the central rapidity window |y| < 0.5 from the "
    "sealed R01 published binned tables, recorded with the "
    "dense-accumulation/then-pick convention, and the charged "
    "multiplicity tails, the multiplicity bins at n_ch greater than or "
    "equal to the 90th percentile of the R51 training distribution.  "
    "R52 is a frozen-parameter deviance comparison, not a refit (index "
    "item 10): no optimizer.  The zero/one pair shares the feed-down "
    "lineage and the normalization (plan line 1872; the R32 p+p "
    "configuration, index item 8).")


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block; the verbatim template)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """--gate: write gate5_r52.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the held-out report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    zc = report["conservation"]["zero_model"]
    conservation = {
        "violations": int(zc["violations"]),
        "pass": bool(zc["pass"]),
        "tolerance": EVENT_CLOSURE_TOL,
        "worst_energy_closure_gev": zc["worst_energy_closure_gev"],
        "events_checked": int(zc["events_checked"]),
        "r51_cross_reference": (
            f"{R51_RUN_DIR}/gate5_r51.json"),
    }
    one = report["one_model"]
    zero = report["zero_model"]
    per_bin_summary = {}
    for family, bins in report["held_out"]["per_bin"].items():
        per_bin_summary[family] = [
            {
                "bin": entry["bin"],
                "one_deviance": entry.get("one_deviance"),
                "zero_deviance": entry.get("zero_deviance"),
                "difference": entry.get("difference"),
                "pooled_se": entry.get("pooled_se"),
                "beyond_2x_se": entry.get("beyond_2x_se", False),
            }
            for entry in bins if entry.get("kept")
        ]
    gate = {
        "schema": ("na49_mixing.staged_r52_central_excitation_necessity."
                   "gate/1"),
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R52",
        "card": CARD,
        "clause": "necessity of the neutral central excitation",
        "one_model": {
            "source": "sealed R51 events",
            "frozen_source": one["frozen_source"],
            "refit": False,
            "events": int(one["events"]),
            "weights": {
                "n_eff_total": one["weights"]["n_eff_total"],
                "ess_min": one["weights"]["ess_min"],
            },
        },
        "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": zero["renormalized_fractions"],
            "regenerated": True,
            "events": int(zero["events"]),
        },
        "held_out": {
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "multiplicity_tails": (
                "n_ch >= 90th percentile of R51 training"),
            "pi0_note": PI0_NOTE,
            "multiplicity_tail_training_overlap": MULTIPLICITY_TAIL_NOTE,
            "per_bin": per_bin_summary,
        },
        "retention": {
            name: family["verdict"]
            for name, family in report["retention"].items()
            if name not in ("bound", "algebra", "all_families_retained",
                            "all_families_retained_note")
        },
        "trigger_for_r53": {
            "named_residual": report["trigger_for_r53"].get(
                "named_residual"),
            "value": report["trigger_for_r53"].get("value"),
            "pooled_se": report["trigger_for_r53"].get("pooled_se"),
            "sigma": report["trigger_for_r53"].get("sigma"),
            "triggered": report["trigger_for_r53"].get("triggered"),
            "argmax_bin_n_ch": report["trigger_for_r53"].get(
                "argmax_bin_n_ch"),
            "zero_model_max_abs_pull": report["trigger_for_r53"].get(
                "zero_model_max_abs_pull"),
            "zero_model_triggered": report["trigger_for_r53"].get(
                "zero_model_triggered"),
        },
        "conservation": conservation,
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "named_residual_register": report["named_residual_register"],
        "optimizer_invoked": False,
        "optimizer_uses": 0,
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
    print(f"[r52] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


# ---------------------------------------------------------------------------
# The clean-environment validation (exactly one pass, before the seal)
# ---------------------------------------------------------------------------
def run_clean_validate(run_dir: Path) -> dict:
    """--clean-validate: re-run the held-out evaluation from a clean
    interpreter over the generated artifacts with a byte-identical
    report, re-run the pre-registered zero-model probe replica (replica
    0) in memory against its sealed record, and re-derive the one-side
    probe replica (R51 replica 0) against the sealed R51 archive, write
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
    rerun_report = run_heldout_evaluation(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the zero-model probe replica re-run in memory (the determinism
    # evidence of acceptance (vi)): the pre-registered probe replica
    # (replica 0) regenerates from its pre-registered seed and must
    # reproduce its sealed event-hash set, its conservation aggregate,
    # and its class census exactly
    probe = PROBE_REPLICA
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set =         generate_zero_replica_events(
            replica_seed(probe), EVENTS_PER_REPLICA, EVENT_INDEX_BASE,
            catalog, fams)
    sealed_probe = _read_json(run_dir / replica_name(probe))
    probe_matches = bool(
        hash_set == sealed_probe.get("event_hash_set_sha256")
        and census == sealed_probe.get("class_census")
        and conservation == sealed_probe.get("conservation")
    )
    # the one-side probe: the sealed R51 replica 0 re-derived against
    # the sealed archive (the reuse contract's bit-identity, re-verified
    # from a clean interpreter)
    sealed_r51_probe = _read_json(MONO_ROOT / R51_RUN_DIR
                                  / replica_name(probe))
    _lattice, one_identity = rederive_r51_replica(
        probe, sealed_r51_probe, catalog, fams)
    payload = {
        "schema": ("na49_mixing.staged_r52_central_excitation_necessity."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r52_central_excitation_necessity.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            ("python helpers/benchmarking/"
             "staged_r52_central_excitation_necessity.py "
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
        "zero_probe_replica": {
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
        "one_side_probe_replica": {
            "replica": probe,
            "source": f"{R51_RUN_DIR}/{replica_name(probe)}",
            "bit_identical": bool(one_identity["bit_identical"]),
            "per_event_hash_mismatches":
                one_identity["per_event_hash_mismatches"],
            "event_hash_set_match":
                one_identity["event_hash_set_match"],
        },
        "verified_utc": _now_utc(),
    }
    _write_json(clean_path, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter held-out evaluation does not "
            "reproduce the report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the zero-model probe replica does not match its sealed "
                "record")
    if not one_identity["bit_identical"]:
        _refuse("the one-side probe replica is not bit-identical to the "
                "sealed R51 archive")
    print(
        f"[r52] clean validation written: {clean_path} (report hash "
        f"match: {report_matches}, zero probe replica {probe} match: "
        f"{probe_matches}, one-side probe bit-identical: "
        f"{one_identity['bit_identical']})",
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
        "schema": ("na49_mixing.staged_r52_central_excitation_necessity."
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
        _refuse(f"the zero-model production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r52] sealed {run_dir} ({n_entries} entries)", flush=True)
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
def run_smoke(run_dir: Path) -> dict:
    """The hermetic end-to-end driver validation: a smoke
    pre-registration, one zero-model replica x 2,000 events at the smoke
    seed 20312925 on the event-index partition 1_600_000_000 + i (i <
    2,000), the one-side reuse-machinery exercise (the smoke events
    re-derived from the smoke seed and verified bit-identical against
    the smoke record), and the smoke held-out evaluation (synthetic
    data; no SPS table and no sealed archive read), into an explicit
    --run-dir (the default production run dir is refused).  Two
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
        "schema": ("na49_mixing.staged_r52_central_excitation_necessity."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "statistics": {
            "note": (
                "the smoke statistics: one zero-model replica x 2,000 "
                "events at the smoke seed 20312925; the smoke exercises "
                "pre-registration, zero-model generation, the one-side "
                "reuse-machinery exercise, recording, and the held-out "
                "evaluation end to end; the production statistics are "
                "the 8 pre-registered replicas of 100,000 events"),
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
                "raw zero-model replica seed = 20260925 + 40000 + 28 + k "
                f"= 20300953 + k per replica k < {N_REPLICAS}"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "one_model": {
            "frozen_values": dict(FROZEN_FIT),
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "reuse_mechanism": ONE_SIDE_REUSE_NOTE,
        },
        "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
        },
        "held_out": {
            "central_spectra_abs_y_lt_0.5": list(PUBLISHED_SPECIES),
            "multiplicity_tails": MULTIPLICITY_TAIL_NOTE,
            "multiplicity_tail_cut": multiplicity_tail_cut(
                _smoke_r50_reference()),
            "pi0_note": PI0_NOTE,
        },
        "retention_rule": {
            "bound": RETENTION_SE_BOUND,
            "algebra": RETENTION_ALGEBRA_NOTE,
        },
        "r53_trigger": {"definition": R53_TRIGGER_NOTE},
        "likelihood": LIKELIHOOD_NOTE,
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
            "R52 is a frozen-parameter deviance comparison, not a refit "
            "(index item 10); optimizer_invoked is false"),
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
            "in the smoke artifacts); the smoke evaluation's data side "
            "is synthetic (the zero side's pooled vectors with a "
            "synthetic 10% published error; no SPS table and no sealed "
            "archive is read)"),
    }
    _write_json(run_dir / ("smoke_" + PREREGISTRATION_NAME), smoke_pre)
    catalog = _load_catalog()
    fams = _load_families()
    records, census, conservation, hash_set =         generate_zero_replica_events(
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
            "zero_draw_note": (
                "the smoke exercises the card-9 per-event class draw on "
                "the closed topology_selection substream at the "
                "renormalized frozen fractions (the ND fraction exactly "
                "0.0; the zero-model generation surface)"),
        },
        "generation_fractions": {
            "note": GENERATION_FRACTIONS_NOTE,
            "renormalized_fractions": dict(RENORMALIZED_FRACTIONS),
            "epsilon": FROZEN_FIT["epsilon"],
            "b_sd": FROZEN_FIT["b_sd"],
            "p_n2": 0.0,
        },
        "class_census": dict(census),
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }
    _write_json(run_dir / SMOKE_REPLICA_NAME, replica_record)
    report = run_heldout_evaluation(run_dir, smoke=True)
    _write_json(run_dir / SMOKE_REPORT_NAME, report)
    print(
        f"[r52] smoke complete: {len(records)} events, census {census}, "
        f"worst closure {conservation['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {"events": len(records), "census": census}


def disjointness_report_for_smoke() -> dict:
    """The smoke pre-registration's disjointness record (identical to
    the production check; the smoke seed and partition are additionally
    disjoint from the R52 zero-model production block and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(REPLICA_SEEDS):
        _refuse("the smoke seed collides with the R52 zero-model "
                "production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_REPLICA):
        _refuse("the smoke event-index partition overlaps the R52 "
                "zero-model production partition")
    report["smoke_raw_seed"] = SMOKE_SEED
    report["smoke_event_index_range"] = [smoke_lo, smoke_hi]
    report["smoke_disjoint_from_production"] = True
    return report


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R52 driver: the central-excitation necessity test "
                    "(TASK-20260925-r5-run-r52-central-excitation-"
                    "necessity)")
    parser.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop")
    parser.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (never sealed)")
    parser.add_argument(
        "--generate", action="store_true",
        help="generate the ZERO-model production replica records, then "
             "stop")
    parser.add_argument(
        "--replica", type=int, action="append", default=None,
        help="generate only the given zero-model replica index "
             "(repeatable)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="the held-out evaluation (held_out_report.json and "
             "REPORT.md), then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r52.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the held-out evaluation from a clean interpreter and "
             "both probe replicas, then stop (exactly one pass, before "
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
        print(f"[r52] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, replicas=args.replica)
        print(f"[r52] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        report = run_heldout_evaluation(run_dir, smoke=False)
        _write_json(run_dir / REPORT_NAME, report)
        _write_report_md(run_dir, report)
        print(
            f"[r52] evaluation written: {run_dir / REPORT_NAME} "
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