#!/usr/bin/env python3
"""R51 driver: the topology fraction fit (Round 5, card 11).

Task ``TASK-20260925-r5-run-r51-topology-fraction-fit`` (plan section 11
run R51, lines 909-918; plan section 23 lines 1825-1827, "Round 5 --
exchange topology").  The card executes run R51, the topology fraction
fit: the round's primary optimizer call over the five free parameters
(the three free inelastic class fractions SD_p / SD_t / DD with ND
determined by the sum rule over the inelastic sample, the mass-law
exponent epsilon, and the effective transfer slope B_SD), fit on the
pre-registered training observables, with no strange-sector parameters
(plan line 918) and the blend ramp start fixed at B (Round 4 obligation
4).  The elastic class is excluded from the inelastic sample (plan line
824); the elastic fraction of the mixed draw is exactly 0.0 and no
event of the run is elastic.

The fit set (index item 3; the gate template's five entries)
------------------------------------------------------------
The free parameter vector is exactly ``[sd_projectile_fraction,
sd_target_fraction, dd_fraction, epsilon, b_sd]`` with the closure
``f_nd = 1 - f_sd_p - f_sd_t - f_dd`` (the inelastic fractions sum to
one over the inelastic sample; a trial with ``f_nd < 0`` or a negative
fraction is outside the domain and carries an infinite objective).  The
initial values are the sealed R50 nominal point: fractions 0.25 each
over SD_p, SD_t, DD, ND, epsilon = 0, B_SD = 8.504140762083612 GeV^-2
(byte-identical to the M8-frozen ``effective_transfer_slope``), asserted
against the sealed R50 pre-registration before any generation.  No
strange-sector parameter, no exchange/topology/flavor/branching
parameter, and never the blend ramp start is fit (plan section 24).

The training observables (plan lines 911-916; plan section 7.3)
---------------------------------------------------------------
The pre-registered instantiation maps the plan's four named observables
onto the sealed data surfaces (the observable instantiation table of the
pre-registration):

- ``leading_proton_spectrum``: the proton species' published-cell
  spectrum.  Under the topology model every final-state proton
  originates in a baryonic leading line (the central source carries
  B = Q = S = 0), so the proton spectrum on the published (x_F, p_T)
  cells is the leading-proton structure's observable.  Data side: the
  sealed R01 proton table (d and the per-bin published variance, the
  dense-accumulation/then-pick convention, holes dropped).  Model side:
  the weighted per-cell invariant-measure sums (each final-state
  proton contributes its CM energy over its transverse momentum, the
  R01 accumulation measure ``e/p_T``) in the data units ``m = sums /
  (pi sqrt(s) N_eff dxF dpT)`` (the R01 convention).
- ``b_transport_spectrum``: the net-proton combination
  ``B_transport = p - p-bar`` (plan section 7.3 line 431) over the
  aligned published proton/antiproton cells.  Data side: the sealed R01
  proton and antiproton tables' aligned cells (d = d_p - d_pbar,
  v_data = v_p + v_pbar).  Model side: m_p - m_pbar per aligned cell.
  The standing model-uncertainty carry applies to this observable: one
  correlated multiplicative nuisance on the bins whose proton or
  antiproton cell is in the sealed 32 persisting-bin set (prior 0.25,
  the R41 convention).
- ``gap_fraction_abs_y_lt_0.5``: the fraction of inelastic events with
  no hadronic final-state particle in the central window |y^CM| < 0.5
  (the event's central multiplicity zero; the runner's CENTRAL_WINDOW).
  Data side: the sealed R50 nominal-point reference (the 0.25-each
  pooled mixture of the four inelastic class pools' realized
  central-gap fractions, with its binomial MC variance as V_data; the
  sealed substrate carries no published gap-fraction table, so the
  R50-pinned reference is the pre-registered data anchor of this
  statistic).
- ``charged_multiplicity_distribution``: the distribution of the
  charged multiplicity n_ch (the count of final-state records with
  nonzero charge Q; the below-threshold baryonic system carries Q = 1
  and counts) over the pre-registered bins n_ch = 2 .. 14.  Data side:
  the sealed R50 nominal-point reference (the 0.25-each pooled mixture
  per bin, with its per-class binomial variance, diagonal; the sealed
  substrate carries no published multiplicity table, so the R50-pinned
  reference is the pre-registered data anchor).  The bins cover the
  realized charged-multiplicity support of the R50 reference and the
  R51 sample; an event outside the support refuses the run (checked,
  never bled into an end bin).

The R50-pinned references are extracted read-only from the sealed R50
archive at pre-registration time (after the R50 seal verification),
recorded in the pre-registration, and re-verified bit-exactly at the fit
evaluation.  The fraction identification rides on the R01-anchored
spectra; the R50-anchored statistics are the nominal-point shape anchors
(their data side was constructed at the nominal fractions, which is
recorded as the reference's construction note).

The importance-weighting fit (plan section 5.3 line 288; the R41
pattern)
---------------------------------------------------------------------
The 8 production replicas are generated once, at the pre-registered
initial point (the per-event class draw at the nominal fractions on the
closed ``topology_selection`` substream, the R51 fraction-sampling
surface; the class generation through the card-9 ``run_topology`` on the
closed five-label substream quintuple).  The objective at a trial
parameter point is the exact per-event importance weight

    w_i(theta) = (f_c / 0.25) * rho_c(event_i; epsilon, B_SD) /
                 rho_c(event_i; 0, B_SD_nominal),

the mixture fraction ratio times the class's mass-law density ratio
evaluated from the event's stored excitation masses, transfers, and
remnant masses:

- single diffraction: the M^2 marginal on ``[m_p^2, (sqrt_s -
  m_remnant)^2]`` at t = 0 (the power integral ratio) and the t draw of
  ``exp(B_SD t)`` on ``[t_min(M^2, m_remnant), 0]`` (the exact closed
  form; the t factor acts on the exchange kinematics only, plan section
  24 line 1866);
- double diffraction: the joint law's accepted-pair density (the product
  of the two marginals restricted by the exact closure ``M_1 + M_2 <=
  sqrt_s``; the closure-restricted joint mass of the unnormalized
  marginals, evaluated by the pre-registered Gauss-Legendre quadrature
  of the reduced one-dimensional form) and the two exact t draws;
- non-diffractive central excitation: the two exact t draws and the
  central mass ``M_c^2`` drawn at the total transfer ``t_c = t_1 + t_2``
  on ``[m_pi0^2, (sqrt_s - M_1 - M_2)^2]`` (the density ratio with the
  per-event total transfer; the normalization ratio is the incomplete-
  gamma closed form of the continuum integral, evaluated per event).

The weights are computed in the log domain and exponentiated after a
single global log-shift (the self-normalized estimator and the ESS are
shift-invariant; the recorded shift is a diagnostic).  At the initial
point every weight is exactly 1.  The mass-law normalization ratios use
the smooth continuum normalizations on the exact draw windows; the
implemented M^2 draws normalize on the documented card-4 4096-point
grid (the sub-tolerance convention, order 1e-3), and the weight
convention's smooth normalizations stand at that documented
sub-tolerance from the implemented grid normalization (recorded as the
weight-convention note; the t draws carry no grid and are exact).  A
trial whose pooled weights or sums are non-finite returns an infinite
objective (the R41 target convention); the optimizer avoids it.

The likelihood (plan section 5.4) is the profiled -2lnL on the training
bins with the per-bin covariance V = V_data + V_mc (reported
separately), the published binning preserved, and the nuisance vector
profiled exactly per trial: one multiplicative normalization nuisance
per spectrum training block (prior 0.35, the R41 standing convention)
plus the model-uncertainty carry on the B_transport bins (prior 0.25).
A bin whose model value is structurally zero in every replica block
carries an infinite sigma and drops out of the objective (the R11
convention; the dropped bins are listed).  The per-bin MC error is the
between-replica standard error (std ddof=1 over the 8 blocks, divided
by sqrt(8)), floored at the engine's DETERMINISTIC_SE_FLOOR.

Hessian discipline (plan line 1871; the R46 pattern)
----------------------------------------------------
The pooled Hessian is quoted only after the optimizer converged, the
replica-block minima agree (the acceptance (ii) stability table passes
on every parameter: the 8-block spread below 20% of the pooled error),
and the pooled curvature is positive definite.  Otherwise ``quoted:
false`` with the named residuals (``boundary_minimum``,
``replica_block_minima_disagree``, ``block_curvature_failure``,
``optimizer_non_convergence``, ``weight_degeneracy``) and the finite-MC
profile (the pre-registered 9-point per-parameter profile through the
best point); no Hessian-based covariance is ever emitted from a
non-converged finite-MC objective.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the M8 frozen
  parameter map, the sealed R50 seal, and the initial-value assertion
  against the sealed R50 nominal point before any generation; records
  the optimizer settings, the training-observable instantiation, the
  R50-pinned reference values, and the seed-block disjointness.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, one mixed replica x 2,000 events
  at the smoke seed 20312925 (raw offset +12000 from master 20260925)
  on the event-index partition ``1_600_000_000 + i`` (i < 2,000), and
  the smoke fit (synthetic data: the model at the pre-registered
  synthetic-truth point with a synthetic 10% published error; no SPS
  table and no sealed archive read), into an explicit ``--run-dir``
  (the default production run dir is refused).  Two identical smoke
  invocations produce byte-identical artifacts.  ``--gate`` / ``--seal``
  / ``--clean-validate`` refuse smoke artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested production
  replica records (any subset; the default is every replica), one
  record file per replica, then stop.  Requires the pre-registration.
  A completed record that matches the pre-registered identity is reused
  unchanged, so the declared tier can be completed replica by replica
  (the pre-registered gateway fallback runs one job per replica).
- ``--evaluate``: the fit evaluation (the per-replica-block
  likelihoods, the pooled fit, the per-block fits, the replica-block
  stability table, the Hessian discipline block or the named residuals
  with the finite-MC profile, ``fit_report.json`` and ``REPORT.md``),
  then stop.  Requires all 8 production records.
- ``--gate``: write ``gate5_r51.json`` (evidence only; the written
  retain / revise / reject decision on the Gate 5 clauses is the
  round's Gate 5 follow-up obligation), then stop.  Requires the fit
  report.
- ``--clean-validate``: re-run the fit evaluation from a clean
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
catalog and continuum config, the sealed R01 archive, the SPS tables
(the published data the likelihood is defined against), and the sealed
R50 archive (the nominal starting values and the R50-pinned training
references).  The frozen simplified cascade at the sealed R45 point
(kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal
width 0.3 GeV) is consumed without refit (the R46 ``no_refit`` clause);
the exchange slopes act on the exchange kinematics only, never on any
daughter-level transverse dynamics; no Round 5 record claims a
separation of exchange recoil from decay transverse dynamics (the R46
outcome, separation_claimed false).
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
# sealed archives, and the SPS tables.
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
        f"staged_r51_topology_fraction_fit: {message}",
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
from na49_mixing.topology_mass_laws import (  # noqa: E402
    PI0_MASS,
    MassLawParams,
)
from na49_mixing.topology_non_diffractive import NDEvent  # noqa: E402
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

from helpers.benchmarking import sps_core  # noqa: E402
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260925-r5-run-r51-topology-
# fraction-fit; the collision, statistics, seeds, fit set, likelihood,
# and acceptance blocks are verbatim from the card and the round 5 index
# items 3, 8, 9, and 10).
# ---------------------------------------------------------------------------
RUN_ID = "R51"
CARD = "TASK-20260925-r5-run-r51-topology-fraction-fit"
#: The run date moves with the actual execution (the card: "the date
#: moves with the actual execution; a late execution re-seals under a
#: new date and re-pins this card's run identity"); the pre-registered
#: identity r51_topology_fraction_fit_2026-09-25 re-pins to the
#: execution date 2026-09-27.
RUN_NAME = "r51_topology_fraction_fit_2026-09-27"
DEFAULT_RUN_DIR = "dev/00runs/r51_topology_fraction_fit_2026-09-27"
GATE_RECORD_NAME = "gate5_r51.json"
REPORT_NAME = "fit_report.json"
REPORT_MD_NAME = "REPORT.md"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r51_topology_fraction_fit.py")
DRIVER_REL_OUTER = "helpers/benchmarking/staged_r51_topology_fraction_fit.py"

#: The card chain this run sits on: the Round 4 closure, the Round 5
#: implementation cards 1-9, card 10 (the sealed R50 templates), and
#: this card.
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
)

#: The frozen substrate digests (byte-unchanged inputs, verified before
#: any step; the card-pinned surfaces with their expected digests,
#: including the R50 driver pin).
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
#: constant SQRT_S = 17.270 GeV (the recorded M8 quote; the runner's
#: pinned extraction default; the CM-frame convention's beta/gamma at
#: this energy are beta = 0.9940791999921171 and gamma =
#: 9.203303892226666).
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
            "model-uncertainty term of the fit (the carry on the "
            "B_transport observable)"
        ),
    },
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_seagull_note": (
        "the b_transport_seagull beyond_isotropic_not_positive residual "
        "is a carry, not a pass criterion (index item 3)"
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
#: convention): 8 replicas x 100,000 p+p events = 800,000 events.
N_REPLICAS = 8
EVENTS_PER_REPLICA = 100_000
N_EVENTS_TOTAL = N_REPLICAS * EVENTS_PER_REPLICA   # 800,000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
#: The raw replica seeds 20300945-20300952 (k = 0..7; index item 9; the
#: block is disjoint from every previously used raw block and from every
#: other reserved Round 5 block).
REPLICA_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + SEED_OFFSET + 20 + k for k in range(N_REPLICAS))
EVENT_INDEX_BASE = 1_100_000_000
EVENT_INDEX_RULE = "1_100_000_000 + i, i < 100,000 (per replica)"

SMOKE_SEED = MASTER_SEED + 40_000 + 12_000              # 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"

#: The reserved round-5 raw seed blocks (index item 9; recorded so the
#: disjointness check covers the reserved neighbors explicitly; the R50
#: block 20300925-20300944 is now used and recorded below).
RESERVED_ROUND5_BLOCKS = {
    "r50": [20300925, 20300944],
    "r52": [20300953, 20300960],
    "r53": [20300961, 20300968],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
#: The reserved round-5 event-index partitions (index item 9).
RESERVED_ROUND5_PARTITIONS = {
    "r50": [1_000_000_000, 1_000_025_000],
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

#: The clean-environment validation's pre-registered probe replica (the
#: replica 0 mixed draw; the probe re-run is the determinism evidence of
#: acceptance (v)).
PROBE_REPLICA = 0

# ---------------------------------------------------------------------------
# The fit set (index item 3; the gate template's five entries)
# ---------------------------------------------------------------------------
FIT_PARAMETERS: tuple[str, ...] = (
    "sd_projectile_fraction",
    "sd_target_fraction",
    "dd_fraction",
    "epsilon",
    "b_sd",
)
#: The pre-registered initial values (the sealed R50 nominal point).
INITIAL_VALUES: dict[str, float] = {
    "sd_projectile_fraction": 0.25,
    "sd_target_fraction": 0.25,
    "dd_fraction": 0.25,
    "epsilon": 0.0,
    "b_sd": 8.504140762083612,
}
#: The parameter bounds (pre-registered brackets; an optimum on any
#: bound is the named residual ``boundary_minimum``).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "sd_projectile_fraction": (0.0, 1.0),
    "sd_target_fraction": (0.0, 1.0),
    "dd_fraction": (0.0, 1.0),
    "epsilon": (-0.5, 0.5),
    "b_sd": (1.0, 20.0),
}
BOUNDS_NOTE = (
    "the fractions within [0, 1] with the sum rule f_nd = 1 - f_sd_p - "
    "f_sd_t - f_dd >= 0 (the inelastic fractions sum to one over the "
    "inelastic sample; a trial violating the sum rule or a fraction "
    "bound carries an infinite objective); epsilon within [-0.5, 0.5] "
    "(the mass-law exponent bracket around the flat nominal, the plan "
    "line 889 form (M^2)^(-1-epsilon) exp[B_SD (M^2) t]); b_sd within "
    "[1.0, 20.0] GeV^-2 (the identifiable effective transfer slope's "
    "pre-registered bracket around the M8-frozen nominal 8.504140762083612)"
)
#: The optimizer settings (pre-registered; deterministic; recorded in
#: the pre-registration so the run is reproducible).
OPTIMIZER_SETTINGS = {
    "algorithm": ("scipy.optimize.minimize, method Nelder-Mead, bounded "
                  "(scipy vertex clipping into PARAM_BOUNDS), adaptive "
                  "True (the R41 convention)"),
    "xatol": 1e-3,
    "fatol": 1e-6,
    "maxiter": 600,
    "maxfev": 1200,
    "initial_point": [INITIAL_VALUES[name] for name in FIT_PARAMETERS],
    "determinism_note": (
        "the objective is a deterministic function of the parameter "
        "vector on the fixed generated sample (no sampler runs inside "
        "the objective), so the bounded Nelder-Mead walk from the fixed "
        "initial point is deterministic and a clean-environment re-run "
        "reproduces the fit report bytes"),
    "invalid_trial_note": (
        "a trial outside the domain (a negative fraction or f_nd < 0) "
        "or with a non-finite objective value returns +inf (the R41 "
        "target convention); the simplex avoids it"),
}
#: The Hessian steps (pre-registered central-difference steps).
HESSIAN_STEPS: dict[str, float] = {
    "sd_projectile_fraction": 0.005,
    "sd_target_fraction": 0.005,
    "dd_fraction": 0.005,
    "epsilon": 0.005,
    "b_sd": 0.05,
}
STABILITY_SPREAD_BOUND = 0.20
NORM_NUISANCE_PRIOR = 0.35
CARRY_NUISANCE_PRIOR = 0.25
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)
ESS_FLOOR = 0.01
#: The finite-MC profile's pre-registered grid (the fallback's recorded
#: profile: 9 equally spaced points over each parameter's bounds, the
#: other four parameters at the best point).
PROFILE_GRID_POINTS = 9

#: The charged-multiplicity distribution's pre-registered bins (the
#: realized charged-multiplicity support n_ch = 2..14 of the sealed R50
#: reference; an event outside the support refuses the run).
MULTIPLICITY_BINS: tuple[int, ...] = tuple(range(2, 15))

#: The gap-fraction observable's central window (the runner's
#: CENTRAL_WINDOW; plan index item 3: |y| < 0.5).
CENTRAL_WINDOW = 0.5

#: The event four-momentum closure tolerance (the house conservation
#: contract).
EVENT_CLOSURE_TOL = 1e-9

INCOMING_REFERENCE_NOTE = (
    "the incoming sum of the pinned M8 p+p configuration: "
    "(E_beam + m_p, 0, 0, p_beam) with E_beam = sqrt(m_p^2 + p_beam^2), "
    "m_p = 0.93827231 GeV, p_beam = 158.0 GeV/c; every event's final "
    "state closes against this reference at tolerance 1e-9 relative"
)

#: The sealed input archives.
R01_ARCHIVE = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"
R50_RUN_DIR = "dev/00runs/r50_pure_topology_templates_2026-09-27"
R50_PREREGISTRATION_NAME = "pre_registration.json"
R50_SEAL_SUMS_NAME = "SHA256SUMS.txt"
R50_SLOT_FMT = "slot_{}_{}.json"
R4_AUDIT_CARD1_DIR = "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20"
PERSISTING_BIN_ATTRIBUTION = "persisting_bin_attribution.json"

#: The spectrum species of the fit (the proton and antiproton tables;
#: the plan's leading-proton and net-proton observables).
SPECTRUM_SPECIES: tuple[str, str] = ("proton", "antiproton")
#: The b_transport combination (plan section 7.3 line 431:
#: B_transport = p - p-bar; the antiproton carries the minus sign).
B_TRANSPORT_PAIR = ("proton", "antiproton", -1.0)

#: The class order of the inelastic sample (the runner's fixed
#: INELASTIC_ORDER; the mixed draw walks this order).
CLASS_ORDER: tuple[str, ...] = tuple(INELASTIC_ORDER)


def replica_seed(replica: int) -> int:
    """The raw seed of replica *replica* (the pre-registered block)."""
    k = int(replica)
    if not 0 <= k < N_REPLICAS:
        _refuse(f"replica index {k} outside the R51 block 0..{N_REPLICAS - 1}")
    return REPLICA_SEEDS[k]


def replica_name(replica: int) -> str:
    return RECORD_FMT.format(int(replica))


def event_id(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_id(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


# ---------------------------------------------------------------------------
# The fit-set assertion (module level; in-memory only)
# ---------------------------------------------------------------------------
def assert_fit_set() -> dict:
    """The fit-set invariants (acceptance (iii); index item 3; plan
    section 24): exactly five free parameters; the three free inelastic
    class fractions with the sum-rule closure over the inelastic sample
    (the elastic class excluded per plan line 824); no strange-sector
    parameter; the blend ramp start fixed at B; no exchange, topology,
    flavor, branching, or catalog parameter is fit."""
    if len(FIT_PARAMETERS) != 5 or len(set(FIT_PARAMETERS)) != 5:
        _refuse(f"the fit set is not five distinct parameters: {FIT_PARAMETERS!r}")
    if tuple(FIT_PARAMETERS) != (
            "sd_projectile_fraction", "sd_target_fraction",
            "dd_fraction", "epsilon", "b_sd"):
        _refuse(f"the fit-set names drifted: {FIT_PARAMETERS!r}")
    for name in ("epsilon", "b_sd"):
        if name not in INITIAL_VALUES or name not in PARAM_BOUNDS:
            _refuse(f"the fit set is missing {name}")
    if set(INITIAL_VALUES) != set(FIT_PARAMETERS) or set(
            PARAM_BOUNDS) != set(FIT_PARAMETERS):
        _refuse("the initial values or the bounds drifted off the fit set")
    for name, value in INITIAL_VALUES.items():
        lo, hi = PARAM_BOUNDS[name]
        if not lo <= value <= hi:
            _refuse(f"the initial value of {name} is outside its bounds")
    f_nd = 1.0 - (INITIAL_VALUES["sd_projectile_fraction"]
                  + INITIAL_VALUES["sd_target_fraction"]
                  + INITIAL_VALUES["dd_fraction"])
    if f_nd != 0.25:
        _refuse(f"the initial f_nd drifted: {f_nd!r} (expected 0.25)")
    if INITIAL_VALUES["b_sd"] != ELASTIC_TRANSFER_SLOPE:
        _refuse(
            "the initial B_SD is not byte-identical to the M8-frozen "
            f"effective_transfer_slope: {INITIAL_VALUES['b_sd']!r} vs "
            f"{ELASTIC_TRANSFER_SLOPE!r}")
    if INITIAL_VALUES["epsilon"] != 0.0:
        _refuse("the initial epsilon drifted off the R50 nominal 0.0")
    if NOMINAL_PARAMS.b_sd != INITIAL_VALUES["b_sd"] or \
            NOMINAL_PARAMS.epsilon != INITIAL_VALUES["epsilon"]:
        _refuse("the runner's NOMINAL_PARAMS drifted off the R50 nominal "
                "mass-law point")
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
    if tuple(CLASS_ORDER) != (
            "sd_projectile", "sd_target", "dd", "nd"):
        _refuse(f"the class order drifted: {CLASS_ORDER!r}")
    return {
        "parameters": list(FIT_PARAMETERS),
        "count": len(FIT_PARAMETERS),
        "closure": "f_nd = 1 - f_sd_p - f_sd_t - f_dd (the inelastic "
                   "fractions sum to one over the inelastic sample)",
        "initial_values": dict(INITIAL_VALUES),
        "initial_f_nd": f_nd,
        "bounds": {name: list(bound)
                   for name, bound in PARAM_BOUNDS.items()},
        "excluded": {
            "blend_ramp_start": (
                "Round 4 obligation 4 (fixed at B = "
                f"{BLEND_RAMP_START_GEV!r} GeV); never a free parameter"),
            "strange_sector": "plan line 918",
            "elastic": "plan line 824 (excluded from the inelastic sample)",
            "exchange_topology_flavor_branching_catalog": "plan section 24",
        },
        "class_set": list(CLASS_ORDER),
        "inelastic_set": sorted(INELASTIC_SET),
        "nd_multi_status": (
            "registered but unretained (retained: False); excluded from "
            "the R51 draw (it enters the round only through R53, plan "
            "line 926)"),
        "pass": True,
    }


_FIT_SET_STATUS = assert_fit_set()


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
    card-pinned surfaces and the R45/R46/R50 driver pins)."""
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
    against the pre-registered hash, and asserted against the initial
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
    if slope != INITIAL_VALUES["b_sd"]:
        _refuse(
            "the M8-frozen effective_transfer_slope "
            f"{slope!r} is not byte-identical to the initial B_SD "
            f"{INITIAL_VALUES['b_sd']!r}"
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
    """Verify a sealed run dir's SHA256SUMS.txt entry by entry (the R41
    verify-sealed discipline; the archive must be intact before it is
    consumed read-only)."""
    run_dir = MONO_ROOT / rel
    sums = run_dir / SHA256SUMS_NAME
    if not run_dir.is_dir() or not sums.is_file():
        _refuse(f"the sealed run dir or its checksum inventory is missing: "
                f"{run_dir}")
    entries = 0
    mismatches = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        path = run_dir / name
        entries += 1
        if not path.is_file() or _sha256_file(path) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(f"the sealed run dir {rel} carries {len(mismatches)} "
                f"checksum mismatches: {mismatches[:8]}")
    return {"run_dir": rel, "entries": entries, "mismatches": [],
            "intact": True}


def verify_sealed_inputs() -> dict:
    """The sealed-input verification: the R50 archive (the nominal
    starting values and the R50-pinned training references) and the R01
    archive (the data side)."""
    return {
        "r50": verify_sealed_dir(R50_RUN_DIR),
        "r01": verify_sealed_dir(R01_ARCHIVE),
    }


# ---------------------------------------------------------------------------
# The initial-value assertion against the sealed R50 (before any
# generation)
# ---------------------------------------------------------------------------
def assert_initial_values_against_r50() -> dict:
    """The initial values are the sealed R50 nominal point exactly (the
    card: "the initial values are asserted against the sealed R50
    nominal point before any generation")."""
    pre_path = MONO_ROOT / R50_RUN_DIR / R50_PREREGISTRATION_NAME
    if not pre_path.is_file():
        _refuse(f"the sealed R50 pre-registration is missing: {pre_path}")
    pre = _read_json(pre_path)
    nominal = pre.get("nominal_parameters") or {}
    expected_fractions = {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25,
    }
    if dict(nominal.get("class_fractions") or {}) != expected_fractions:
        _refuse(
            "the sealed R50 nominal inelastic fractions drifted: "
            f"{nominal.get('class_fractions')!r}")
    if float(nominal.get("epsilon")) != INITIAL_VALUES["epsilon"]:
        _refuse("the sealed R50 nominal epsilon is not the R51 initial "
                f"epsilon: {nominal.get('epsilon')!r}")
    if float(nominal.get("b_sd_gev2")) != INITIAL_VALUES["b_sd"]:
        _refuse("the sealed R50 nominal b_sd is not byte-identical to the "
                f"R51 initial B_SD: {nominal.get('b_sd_gev2')!r}")
    if float(nominal.get("elastic_fraction")) != 0.0:
        _refuse("the sealed R50 nominal elastic fraction is not 0.0")
    if float(nominal.get("p_n2")) != 0.0:
        _refuse("the sealed R50 nominal p_n2 is not 0.0")
    if pre.get("run") != "R50" or pre.get("mode") != "production":
        _refuse("the R50 pre-registration is not the sealed production "
                f"record (run {pre.get('run')!r}, mode {pre.get('mode')!r})")
    return {
        "source": f"{R50_RUN_DIR}/{R50_PREREGISTRATION_NAME}",
        "r50_nominal": {
            "fractions": dict(expected_fractions),
            "epsilon": float(nominal["epsilon"]),
            "b_sd_gev2": float(nominal["b_sd_gev2"]),
        },
        "r51_initial_values": dict(INITIAL_VALUES),
        "match": True,
    }


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
    r51_lo = EVENT_INDEX_BASE
    r51_hi = EVENT_INDEX_BASE + EVENTS_PER_REPLICA
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
            if r51_lo < p_hi and p_lo < r51_hi:
                problems.append(
                    f"the R51 event-index partition [{r51_lo}, {r51_hi}) "
                    f"overlaps the {name} partition [{p_lo}, {p_hi})")
    for name, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        if sorted(set(range(lo, hi + 1)) & set(REPLICA_SEEDS)):
            problems.append(
                f"the R51 raw block overlaps the reserved {name} block "
                f"{lo}..{hi}")
    for name, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        if r51_lo < hi and lo < r51_hi:
            problems.append(
                f"the R51 event-index partition overlaps the reserved "
                f"{name} partition [{lo}, {hi})")
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r51] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r51_raw_seed_block": [min(REPLICA_SEEDS), max(REPLICA_SEEDS)],
        "r51_raw_seed_rule": (
            "raw replica seed = 20260925 + 40000 + 20 + k = 20300945 + k "
            "per replica k < 8 (index item 9); the block 20300945-20300952 "
            "is disjoint from every previously used raw block and from "
            "every other reserved Round 5 block"),
        "r51_replica_seeds": list(REPLICA_SEEDS),
        "r51_event_index_partition": EVENT_INDEX_RULE,
        "r51_event_index_range": [r51_lo, r51_hi],
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R51 raw block 20300945-20300952 shares no raw seed with "
            "any previously used block (R23 20300913-20300920 under the "
            "20260913 master ends below it; the R50 block 20300925-"
            "20300944 ends below it; the Round 4 blocks 20301020-20301083 "
            "and smoke blocks 20307920-20311920 under the 20260920 master "
            "sit above it; the R32 blocks 20301519-20301826 under the "
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
# The R50-pinned training references (the data anchors of the gap-fraction
# and charged-multiplicity statistics; extracted read-only from the sealed
# R50 archive)
# ---------------------------------------------------------------------------
def _r50_inelastic_class_of(record: dict) -> str:
    label = record.get("class_label")
    if label not in INELASTIC_SET:
        _refuse(f"the R50 record carries a non-inelastic class: {label!r}")
    return str(label)


def r50_reference_payload() -> dict:
    """The sealed R50 nominal-point training references: the 0.25-each
    pooled mixture of the four inelastic class pools' realized
    central-window gap fraction and charged-multiplicity distribution,
    with their per-class binomial variances (diagonal).  Read-only from
    the sealed R50 slot records (the seal is verified first).  The
    charged multiplicity is the count of final-state records with
    nonzero charge Q (the below-threshold baryonic system carries Q = 1
    and counts); the gap flag is the event's central multiplicity zero
    (no hadronic final-state particle in |y^CM| < 0.5, the runner's
    CENTRAL_WINDOW)."""
    seal = verify_sealed_dir(R50_RUN_DIR)
    gap_counts: dict[str, int] = {}
    mult_counts: dict[str, dict[int, int]] = {}
    class_events: dict[str, int] = {}
    for cls in CLASS_ORDER:
        gap_counts[cls] = 0
        mult_counts[cls] = {n: 0 for n in MULTIPLICITY_BINS}
        class_events[cls] = 0
    for cls in CLASS_ORDER:
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
                        f"{record.get('n_events')} events, expected 25,000")
            for event in record["events"]:
                label = _r50_inelastic_class_of({"class_label": cls})
                n_ch = 0
                for fsr in event["fs"]:
                    if int(fsr[2]) != 0:
                        n_ch += 1
                if n_ch not in mult_counts[label]:
                    _refuse(
                        "the sealed R50 event carries a charged "
                        f"multiplicity {n_ch} outside the pre-registered "
                        f"bins {MULTIPLICITY_BINS}; the reference "
                        "extraction refuses (checked, never bled)")
                mult_counts[label][n_ch] += 1
                if int(event["ob"][3]) == 0:
                    gap_counts[label] += 1
                class_events[label] += 1
    fractions = {cls: 0.25 for cls in CLASS_ORDER}
    gap_fraction = math.fsum(
        fractions[cls] * gap_counts[cls] / class_events[cls]
        for cls in CLASS_ORDER)
    gap_variance = math.fsum(
        fractions[cls] ** 2
        * (gap_counts[cls] / class_events[cls])
        * (1.0 - gap_counts[cls] / class_events[cls])
        / class_events[cls]
        for cls in CLASS_ORDER)
    mult_distribution = {}
    mult_variance = {}
    for n in MULTIPLICITY_BINS:
        p = math.fsum(
            fractions[cls] * mult_counts[cls][n] / class_events[cls]
            for cls in CLASS_ORDER)
        v = math.fsum(
            fractions[cls] ** 2
            * (mult_counts[cls][n] / class_events[cls])
            * (1.0 - mult_counts[cls][n] / class_events[cls])
            / class_events[cls]
            for cls in CLASS_ORDER)
        mult_distribution[str(n)] = p
        mult_variance[str(n)] = v
    total = math.fsum(mult_distribution.values())
    if abs(total - 1.0) > 1e-12:
        _refuse("the R50 reference multiplicity distribution does not sum "
                f"to one: {total!r}")
    return {
        "source": R50_RUN_DIR,
        "seal": seal,
        "construction": (
            "the 0.25-each pooled mixture of the four inelastic class "
            "pools' realized statistics (100,000 events per class, the "
            "sealed R50 T1 slots); the reference was constructed at the "
            "nominal fractions, which is its recorded construction note"),
        "events_per_class": class_events,
        "gap_fraction": {
            "value": gap_fraction,
            "variance": gap_variance,
            "per_class": {
                cls: gap_counts[cls] / class_events[cls]
                for cls in CLASS_ORDER},
            "definition": (
                "the fraction of inelastic events with no hadronic "
                "final-state particle in the central window |y^CM| < 0.5 "
                "(the event's central multiplicity zero)"),
        },
        "charged_multiplicity": {
            "bins": [int(n) for n in MULTIPLICITY_BINS],
            "distribution": mult_distribution,
            "variance": mult_variance,
            "per_class_counts": {
                cls: {str(n): mult_counts[cls][n]
                      for n in MULTIPLICITY_BINS}
                for cls in CLASS_ORDER},
            "definition": (
                "the charged multiplicity n_ch is the count of "
                "final-state records with nonzero charge Q; the "
                "below-threshold baryonic system carries Q = 1 and "
                "counts; the bins are the pre-registered support "
                "n_ch = 2..14"),
        },
    }


def _verify_r50_reference(payload: dict, reference: dict) -> None:
    """The pre-registration precedence check: the recomputed R50-pinned
    reference matches the pre-registered values bit-exactly."""
    if payload.get("gap_fraction", {}).get("value") != \
            reference["gap_fraction"]["value"]:
        _refuse("the recomputed R50 gap-fraction reference differs from "
                "the pre-registered value (the pre-registration "
                "precedence; the run refuses)")
    if payload.get("gap_fraction", {}).get("variance") != \
            reference["gap_fraction"]["variance"]:
        _refuse("the recomputed R50 gap-fraction variance differs from "
                "the pre-registered value")
    if payload.get("charged_multiplicity", {}).get("distribution") != \
            reference["charged_multiplicity"]["distribution"]:
        _refuse("the recomputed R50 multiplicity distribution differs "
                "from the pre-registered values")
    if payload.get("charged_multiplicity", {}).get("variance") != \
            reference["charged_multiplicity"]["variance"]:
        _refuse("the recomputed R50 multiplicity variances differ from "
                "the pre-registered values")


# ---------------------------------------------------------------------------
# The R01 data side (the sealed published tables; the R41 load discipline)
# ---------------------------------------------------------------------------
def _load_production_tables():
    """The six-species SPS tables (the published binning; the R01
    loader).  The production evaluation reads the SPS tables; the smoke
    mode builds the synthetic hermetic grid."""
    raw_config = json.loads((NESTED_ROOT / "config"
                             / r01.PRODUCTION_CONFIG_NAME).read_text(
                                 encoding="utf-8"))
    return raw_config, r01.load_binned_tables(raw_config)


def _load_smoke_tables():
    """The synthetic hermetic binning grid of the smoke mode: the proton
    and antiproton tables built from the committed table declarations
    (species, pdg, mass, normalization) only; no SPS table is read (the
    R32 smoke-grid pattern)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    raw_config = json.loads((NESTED_ROOT / "config"
                             / r01.PRODUCTION_CONFIG_NAME).read_text(
                                 encoding="utf-8"))
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
    tables = []
    for entry in binned["tables"]:
        if str(entry["species"]) not in SPECTRUM_SPECIES:
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
    if len(tables) != 2:
        _refuse("the smoke grid built "
                f"{len(tables)} tables, expected the two spectrum "
                f"species {SPECTRUM_SPECIES}")
    return tables


def _load_sealed_r01(tables) -> dict:
    """The sealed R01 data vector and published variances, cross-checked
    bit for bit against the locally loaded SPS tables (the R11
    parse_sealed_binned discipline; the R41 loader verbatim)."""
    archive = MONO_ROOT / R01_ARCHIVE
    residuals = _read_json(archive / R01_RESIDUALS_NAME)
    covariance = _read_json(archive / R01_COVARIANCE_NAME)
    sealed: dict[str, dict[str, list[float]]] = {}
    for table in tables:
        sealed[table.species] = {
            "d": [], "v_data": [], "xF_mid": [], "pT_mid": [],
        }
    for entry in residuals:
        slot = sealed[entry["species"]]
        slot["d"].append(float(entry["d"]))
    for entry in covariance["V_data"]:
        sealed[entry["species"]]["v_data"].append(
            float(entry["variance"]))
    for entry in _read_json(archive / R01_PREDICTIONS_NAME):
        slot = sealed[entry["species"]]
        slot["xF_mid"].append(float(entry["xF_mid"]))
        slot["pT_mid"].append(float(entry["pT_mid"]))
    for table in tables:
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


def _aligned_cells(sealed: dict) -> list:
    """The intersection of the proton and antiproton tabulated cells,
    sorted by (xF, pT) (the R11 aligned-cells convention).  Each entry
    carries the cell's (bin_proton, bin_antiproton, xF_mid, pT_mid)."""
    species_a, species_b, _sign = B_TRANSPORT_PAIR
    coords_a = [
        (round(float(x), 12), round(float(p), 12), int(i))
        for i, (x, p) in enumerate(zip(
            sealed[species_a]["xF_mid"], sealed[species_a]["pT_mid"]))
    ]
    index_a = {(x, p): i for i, (x, p, _i) in enumerate(coords_a)}
    out = []
    for i_b, (x, p) in enumerate(zip(
            sealed[species_b]["xF_mid"], sealed[species_b]["pT_mid"])):
        key = (round(float(x), 12), round(float(p), 12))
        if key in index_a:
            out.append({
                "bin_a": int(index_a[key]),
                "bin_b": int(i_b),
                "xF_mid": key[0],
                "pT_mid": key[1],
            })
    out.sort(key=lambda e: (e["xF_mid"], e["pT_mid"]))
    return out


def _cell_area_from_table(table, i: int) -> float:
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[i])
    ip = int(table.bin_index_pt[i])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


def _load_carry_bins() -> list:
    """The sealed 32 persisting bins (the blend-band-sensitive set),
    read read-only from the sealed card-1 attribution (the R41 loader)."""
    path = MONO_ROOT / R4_AUDIT_CARD1_DIR / PERSISTING_BIN_ATTRIBUTION
    if not path.is_file():
        _refuse(f"the sealed card-1 attribution is missing: {path}")
    attribution = _read_json(path)
    rows = attribution.get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        _refuse(
            "the sealed card-1 attribution does not carry the 32 "
            "persisting rows (found "
            f"{len(rows) if isinstance(rows, list) else type(rows)!r})"
        )
    return [{"species": str(row["species"]), "bin_index": int(row["bin_index"])}
            for row in rows]


# ---------------------------------------------------------------------------
# The per-event physics layer (the house conservation contract)
# ---------------------------------------------------------------------------
EVENT_SCHEMA = "na49_mixing.staged_r51_topology_fraction_fit.event/1"


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


# ---------------------------------------------------------------------------
# The CM-frame extraction of the spectrum-species records (the proton and
# antiproton final-state hadrons' (x_F, p_T) pairs; the runner's pinned
# CM-frame convention at the extraction energy SQRT_S = 17.270 GeV)
# ---------------------------------------------------------------------------
def _cm_boost_parameters(sqrt_s: float) -> tuple[float, float]:
    """The lab to overall-CM boost parameters ``(beta, gamma)`` of the
    pinned M8 p+p beam configuration at *sqrt_s* (the runner's
    convention; the pinned values at the extraction energy are beta =
    0.9940791999921171 and gamma = 9.203303892226666)."""
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    beta = BEAM_MOMENTUM / e_total
    gamma = e_total / sqrt_s
    return beta, gamma


_CM_BETA, _CM_GAMMA = _cm_boost_parameters(SQRT_S)


def spectrum_records(event) -> tuple[list, list]:
    """The event's proton and antiproton ``(x_F, p_T, e/p_T)`` records
    in the CM frame (the fit's spectrum-species extraction, the R01
    model convention): the final-state records with ``pdg_id`` 2212 /
    -2212, x_F = 2 p_L^CM / sqrt(s) at the extraction energy, p_T the
    invariant transverse momentum, and the invariant measure weight
    ``e_CM / p_T`` (the R01 accumulation measure: the published f is
    accumulated with the particle's CM energy over its transverse
    momentum; the stored lab four-momentum is inverted through the
    pinned boost, exactly the inverse of the runner's forward boost)."""
    protons: list = []
    pbars: list = []
    if isinstance(event, ElasticEvent):
        finals: tuple = tuple(event.outgoing)
    else:
        finals = tuple(event.final_state)
    for hadron in finals:
        pdg = getattr(hadron, "pdg_id", None)
        if pdg is None or int(pdg) not in (2212, -2212):
            continue
        e, px, py, pz = (float(x) for x in hadron.four_momentum)
        pz_cm = _CM_GAMMA * (pz - _CM_BETA * e)
        e_cm = _CM_GAMMA * (e - _CM_BETA * pz)
        x_f = 2.0 * pz_cm / SQRT_S
        p_t = math.sqrt(px * px + py * py)
        measure = e_cm / p_t if p_t > 0.0 else 0.0
        if int(pdg) == 2212:
            protons.append((x_f, p_t, measure))
        else:
            pbars.append((x_f, p_t, measure))
    return protons, pbars


def charged_multiplicity(event) -> int:
    """The event's charged multiplicity n_ch: the count of final-state
    records with nonzero charge Q (the below-threshold baryonic system
    carries Q = 1 and counts; the elastic pair's two lines count)."""
    if isinstance(event, ElasticEvent):
        finals: tuple = tuple(event.outgoing)
    else:
        finals = tuple(event.final_state)
    return sum(1 for hadron in finals if _line_qn(hadron)[1] != 0)


def remnant_masses(event) -> list:
    """The per-event remnant masses the mass-law ratios consume: the
    single-diffraction class the non-excited line's remnant mass; the
    non-diffractive class the two remnant masses (projectile side
    first); the double-diffraction class none (both lines excited; the
    joint law's windows place the other line at the intact proton
    mass)."""
    if isinstance(event, SDEvent):
        return [float(event.remnant.mass)]
    if isinstance(event, NDEvent):
        return [float(event.remnants[0].mass), float(event.remnants[1].mass)]
    if isinstance(event, DDEvent):
        return []
    raise ValueError(
        f"remnant_masses: unsupported event record {type(event)!r}")


def serialize_event(event, obs, conservation, protons, pbars, n_ch) -> dict:
    """The per-event record of the replica file (without its hash): the
    compact fit-relevant payload (the R51 record convention; the R50
    template archive remains the full-final-state evidence base).  The
    caller adds the event identity fields and the per-event hash: the
    sha256 of the record's canonical JSON without the ``h`` key."""
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
        "rm": remnant_masses(event),
        "sg": float(obs.seagull_moment),
        "sh": conservation["mass_shell_worst"],
        "tr": [float(t) for t in obs.transfers],
        "xt": exchange_transfers(event),
        "xp": [float(obs.x_f_projectile), float(obs.x_f_target)],
    }
    return record


# ---------------------------------------------------------------------------
# The generation (the mixed-draw replica records)
# ---------------------------------------------------------------------------
ELASTIC_FRACTION_GENERATION = 0.0
GENERATION_FRACTIONS_NOTE = (
    "the R51 mixed draw runs at the pre-registered initial point (the "
    "nominal fractions 0.25 each over sd_projectile, sd_target, dd, nd "
    "with the elastic fraction exactly 0.0, plan line 824): the "
    "per-event class draw on the closed topology_selection substream is "
    "the R51 fraction-sampling surface, and the fit's importance "
    "weights carry the trial fractions (the R41 pattern: one "
    "generation at the initial point, exact per-event path ratios per "
    "trial)")


def generate_replica_events(
    seed: int,
    n_events: int,
    event_index_base: int,
    catalog,
    fams,
):
    """Generate *n_events* mixed-draw inelastic events of one replica
    and return the per-event records, the class census, the
    conservation aggregate, and the event-hash set digest.

    Every event consumes the closed five-label substream quintuple at
    replica slot 0 (the R50 convention: the raw seed carries the
    replica identity), draws its class on the closed
    ``topology_selection`` substream at the nominal fractions (the
    elastic fraction exactly 0.0, so no uniform returns "elastic"; a
    drawn elastic label refuses the run -- the fit sample is inelastic
    only), runs the card-9 ``run_topology`` dispatch, extracts the
    observables and the spectrum-species records, and runs the house
    conservation contract (a violation raises and stops the run)."""
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
            NOMINAL_PARAMS.class_fractions, ELASTIC_FRACTION_GENERATION,
            rng_topology_selection)
        if label == "elastic" or label not in INELASTIC_SET:
            raise ConservationViolation(
                f"event {event_index_base + i}: the class draw returned "
                f"{label!r} outside the inelastic set {sorted(INELASTIC_SET)}"
                " (the R51 mixed draw runs at the elastic fraction 0.0; "
                "checked, never repaired)")
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


REPLICA_SCHEMA = "na49_mixing.staged_r51_topology_fraction_fit.replica/1"


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
    per-event records, the class census, the conservation aggregate, and
    the event-hash set digest (the generation-time determinism
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
                "raw replica seed = 20260925 + 40000 + 20 + k = 20300945 + "
                f"k per replica k < {N_REPLICAS} (index item 9; the block "
                "20300945-20300952 is disjoint from every previously used "
                "raw block and from every other reserved Round 5 block)"),
            "event_index_rule": EVENT_INDEX_RULE,
            "substream_labels": sorted(SUBSTREAM_LABELS),
            "consumed_labels": [
                "topology_selection", "spectral_state",
                "exchange_kinematics", "decay_channel", "decay_kinematics",
            ],
        },
        "generation_fractions": GENERATION_FRACTIONS_NOTE,
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
    the pre-registration.  A completed record that matches the
    pre-registered identity is reused unchanged (the pre-registered
    gateway fallback runs one job per replica; the local fallback is
    one single-threaded process per --replica invocation)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    load_pre_registration(run_dir)
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
            print(f"[r51] replica {int(replica)} ({name}) reused unchanged",
                  flush=True)
            written.append(name)
            continue
        seed = replica_seed(replica)
        print(
            f"[r51] generating replica {int(replica)} ({name}): seed "
            f"{seed}, {EVENTS_PER_REPLICA} mixed-draw events",
            flush=True,
        )
        records, census, conservation, hash_set = generate_replica_events(
            seed, EVENTS_PER_REPLICA, EVENT_INDEX_BASE, catalog, fams)
        record = build_replica_record(
            replica, records, census, conservation, hash_set, run_dir)
        record["reused"] = False
        _write_json(path, record)
        print(
            f"[r51] replica {int(replica)} written: {path} "
            f"(census {census}, worst closure "
            f"{conservation['worst_closure_relative']:.3e})",
            flush=True,
        )
        written.append(name)
    return written


# ---------------------------------------------------------------------------
# The importance-weight machinery (plan section 5.3 line 288; the R41
# pattern: one generation at the initial point, exact per-event ratios
# per trial, computed in the log domain)
# ---------------------------------------------------------------------------
#: The nominal mixture fraction of every inelastic class (the generation
#: point; the weight's fraction ratio is f_c / 0.25).
NOMINAL_CLASS_FRACTION = 0.25
#: The flat-slope threshold of the t normalization (the mass-law
#: module's convention).
_FLAT_SLOPE_EPS = 1e-14
#: The exact-branch threshold of the power integral (below it the
#: integral is the logarithmic limit).
_POWER_EPS_BRANCH = 1e-13
#: The exp1 branch of the incomplete-gamma evaluation (the s -> 0
#: recurrence's cancellation bound).
_GAMMA_S_BRANCH = 1e-8

_SQRT_S_SQUARE = SQRT_S * SQRT_S
_PROTON_MASS_SQUARE = PROTON_MASS * PROTON_MASS


def _log_power_integral(eps: float, a: float, c: float) -> float:
    """``log`` of the smooth power integral ``A(eps; a, c) =
    int_a^c x^(-1-eps) dx`` (the single-diffraction marginal's smooth
    normalization; a < c, both positive).  ``eps = 0`` is the
    logarithmic limit ``log(c/a)``."""
    if not (math.isfinite(a) and math.isfinite(c)) or not c > a > 0.0:
        _refuse(f"log_power_integral: the window ({a!r}, {c!r}) is invalid")
    eps = float(eps)
    if abs(eps) < _POWER_EPS_BRANCH:
        return math.log(math.log(c / a))
    num = math.expm1(-eps * math.log(c)) - math.expm1(-eps * math.log(a))
    value = num / (-eps)
    if not (math.isfinite(value) and value > 0.0):
        _refuse(
            "log_power_integral: the power integral is non-finite or "
            f"non-positive at (eps={eps!r}, a={a!r}, c={c!r})")
    return math.log(value)


def _log_t_norm(b: float, t_min: float) -> float:
    """``log`` of the t draw's normalization ``Z_t(b, t_min) =
    int_{t_min}^0 exp(b t) dt = (1 - exp(b t_min)) / b`` (the exact
    closed form; the flat-slope limit ``-t_min`` at |b| below the
    threshold).  A degenerate range ``t_min = 0`` contributes a zero
    log factor (the point-mass convention: both densities are point
    masses at t = 0 and their ratio is 1)."""
    b = float(b)
    t_min = float(t_min)
    if not math.isfinite(b) or not math.isfinite(t_min) or t_min > 0.0:
        _refuse(
            f"log_t_norm: the arguments (b={b!r}, t_min={t_min!r}) are "
            "outside the domain")
    if t_min == 0.0:
        return 0.0
    if abs(b) <= _FLAT_SLOPE_EPS:
        return math.log(-t_min)
    btm = b * t_min
    if btm < -700.0:
        z = 1.0 / b
    else:
        z = (1.0 - math.exp(btm)) / b
    if not (math.isfinite(z) and z > 0.0):
        _refuse(
            f"log_t_norm: the normalization is non-finite or "
            f"non-positive at (b={b!r}, t_min={t_min!r})")
    return math.log(z)


def _t_range_min_vec(m2, m_rem):
    """The vectorized mirror of ``topology_mass_laws.t_range``'s lower
    edge (the same arithmetic in the same order, elementwise; the
    inputs are the drawn masses of stored events, always inside the
    domain)."""
    m2 = np.asarray(m2, dtype=float)
    m_rem = np.asarray(m_rem, dtype=float)
    s = _SQRT_S_SQUARE
    m2_rem = m_rem * m_rem
    e_p = (s + _PROTON_MASS_SQUARE - m2_rem) / (2.0 * SQRT_S)
    e_m = (s + m2 - m2_rem) / (2.0 * SQRT_S)
    p_star = np.sqrt(np.maximum(
        _kallen(s, _PROTON_MASS_SQUARE, m2_rem), 0.0)) / (2.0 * SQRT_S)
    q = np.sqrt(np.maximum(_kallen(s, m2, m2_rem), 0.0)) / (2.0 * SQRT_S)
    return _PROTON_MASS_SQUARE + m2 - 2.0 * (e_p * e_m + p_star * q)


def _kallen(a, b, c):
    """The Kallen triangle (the excitation-kinematics module's formula,
    elementwise)."""
    return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c


# -- the double-diffraction joint mass (the smooth continuum
#    normalization of the closure-restricted product; cached per eps) --
_DD_GL_ORDER = 256
_DD_GL_NODES, _DD_GL_WEIGHTS = np.polynomial.legendre.leggauss(_DD_GL_ORDER)
_DD_NORM_CACHE: dict[float, float] = {}


def _log_dd_joint_mass(eps: float) -> float:
    """``log`` of the double-diffraction joint mass ``A_DD(eps)``: the
    closure-restricted integral of the two unnormalized marginals,

        A_DD(eps) = 4 int_{m_p}^{sqrt_s - m_p} u^(-1-2 eps) J(u) du,

    with ``J(u) = ((sqrt_s - u)^(-2 eps) - m_p^(-2 eps)) / (-2 eps)``
    (the logarithmic limit ``log((sqrt_s - u)/m_p)`` at eps = 0), the
    exact reduction of ``int int_{M_1 + M_2 <= sqrt_s} x^(-1-eps)
    y^(-1-eps) dx dy`` under ``u = sqrt(x), v = sqrt(y)``.  The
    integrand is smooth on the closed domain; the quadrature is the
    pre-registered 256-point Gauss-Legendre rule (cached per eps)."""
    key = float(eps)
    cached = _DD_NORM_CACHE.get(key)
    if cached is not None:
        return cached
    lo = PROTON_MASS
    hi = SQRT_S - PROTON_MASS
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    u = mid + half * _DD_GL_NODES
    eps = float(eps)
    if abs(eps) < _POWER_EPS_BRANCH:
        j_values = np.log((SQRT_S - u) / PROTON_MASS)
        integrand = j_values / u
    else:
        j_values = (
            np.power(SQRT_S - u, -2.0 * eps) - PROTON_MASS ** (-2.0 * eps)
        ) / (-2.0 * eps)
        integrand = np.power(u, -1.0 - 2.0 * eps) * j_values
    mass = 4.0 * half * float(np.dot(_DD_GL_WEIGHTS, integrand))
    if not (math.isfinite(mass) and mass > 0.0):
        _refuse(
            "log_dd_joint_mass: the joint mass is non-finite or "
            f"non-positive at eps={eps!r}")
    log_mass = math.log(mass)
    _DD_NORM_CACHE[key] = log_mass
    return log_mass


# -- the non-diffractive central mass integral (the smooth continuum
#    normalization at the per-event total transfer; incomplete-gamma
#    closed form, vectorized) --
def _gamma_upper_vec(s, z):
    """The upper incomplete gamma ``Gamma(s, z) = int_z^inf t^(s-1)
    e^(-t) dt`` for ``s in (-1, inf)``, vectorized: the direct regularized
    product for ``s > 0``, the ``s + 1`` recurrence for ``s < 0`` (the
    identity ``Gamma(s+1, z) = s Gamma(s, z) + z^s e^(-z)``), and the
    ``E1`` branch at ``|s|`` below the recorded threshold (the
    recurrence's cancellation bound; the ``s -> 0`` limit is
    ``E1(z)``)."""
    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    s, z = np.broadcast_arrays(s, z)
    out = np.empty(s.shape, dtype=float)
    small = np.abs(s) < _GAMMA_S_BRANCH
    pos = (~small) & (s > 0.0)
    neg = (~small) & (s < 0.0)
    if small.any():
        import scipy.special as _sp

        out[small] = _sp.exp1(z[small])
    if pos.any():
        import scipy.special as _sp

        out[pos] = _sp.gamma(s[pos]) * _sp.gammaincc(s[pos], z[pos])
    if neg.any():
        import scipy.special as _sp

        s1 = s + 1.0
        out[neg] = (
            _sp.gamma(s1[neg]) * _sp.gammaincc(s1[neg], z[neg])
            - np.power(z[neg], s[neg]) * np.exp(-z[neg])
        ) / s[neg]
    return out


def _central_mass_log_integral(eps, beta, a, c):
    """``log`` of the central law's smooth normalization ``A_c(eps,
    beta; a, c) = int_a^c x^(-1-eps) exp(beta x) dx``, vectorized over
    the events (``beta = b_sd * t_c <= 0``; ``a`` the scalar lower edge
    ``m_pi0^2``; ``c`` the per-event central window edge).  The closed
    form is the incomplete gamma ``A = (-beta)^eps [Gamma(-eps, -beta a)
    - Gamma(-eps, -beta c)]`` for ``beta < 0`` and the power integral
    for ``beta = 0``; the E1 branch covers ``eps -> 0``."""
    eps = np.asarray(eps, dtype=float)
    beta = np.asarray(beta, dtype=float)
    c = np.asarray(c, dtype=float)
    if not (math.isfinite(a) and a > 0.0):
        _refuse(f"central_mass_log_integral: the lower edge {a!r} is invalid")
    eps, beta, c = np.broadcast_arrays(eps, beta, c)
    flat = np.abs(beta) < _FLAT_SLOPE_EPS
    out = np.empty(beta.shape, dtype=float)
    if flat.any():
        eps_f = eps[flat]
        c_f = c[flat]
        aa = math.log(a)
        cc_f = np.log(c_f)
        zero_eps = np.abs(eps_f) < _POWER_EPS_BRANCH
        vals = np.empty(eps_f.shape, dtype=float)
        num = np.expm1(-eps_f * cc_f) - np.expm1(-eps_f * aa)
        with np.errstate(divide="ignore", invalid="ignore"):
            power_vals = num / (-eps_f)
        good = (~zero_eps) & np.isfinite(power_vals) & (power_vals > 0.0)
        vals[good] = np.log(power_vals[good])
        if zero_eps.any():
            vals[zero_eps] = np.log(np.log(c_f[zero_eps] / a))
        out[flat] = vals
    live = ~flat
    if live.any():
        s = -eps[live]
        beta_f = beta[live]
        ta = -beta_f * a
        tc = -beta_f * c[live]
        g_ta = _gamma_upper_vec(s, ta)
        g_tc = _gamma_upper_vec(s, tc)
        diff = g_ta - g_tc
        with np.errstate(divide="ignore", invalid="ignore"):
            log_a = eps[live] * np.log(-beta_f) + np.log(diff)
        out[live] = log_a
    return out


#: The cached per-window SD normalization ratios (keyed by the window's
#: upper edge; the remnant masses take a discrete set).
_SD_NORM_CACHE: dict[tuple[float, float], tuple[float, float]] = {}


def _sd_log_norm_pair(eps: float, m2_max: float) -> tuple[float, float]:
    """The pair ``(log A(0; win), log A(eps; win))`` of the
    single-diffraction marginal's smooth normalizations on the window
    ``[m_p^2, m2_max]`` (cached per (eps, m2_max))."""
    key = (float(eps), float(m2_max))
    cached = _SD_NORM_CACHE.get(key)
    if cached is None:
        cached = (_log_power_integral(0.0, _PROTON_MASS_SQUARE, m2_max),
                  _log_power_integral(eps, _PROTON_MASS_SQUARE, m2_max))
        _SD_NORM_CACHE[key] = cached
    return cached


#: The nominal law point (the generation point of the importance
#: ensemble).
NOMINAL_EPSILON = 0.0
NOMINAL_B_SD = ELASTIC_TRANSFER_SLOPE


def class_log_law_weights(cls: str, arrays: dict, eps: float,
                          b_sd: float) -> np.ndarray:
    """The per-event log mass-law ratio ``log rho_c(event; eps, b_sd) -
    log rho_c(event; 0, B_SD_nominal)`` for one class's stored events
    (the exact per-event density ratios of the card-4 draws; see the
    module docstring's weight-convention block)."""
    eps = float(eps)
    b_sd = float(b_sd)
    delta_b = b_sd - NOMINAL_B_SD
    if cls in ("sd_projectile", "sd_target"):
        m2 = np.asarray(arrays["m2"], dtype=float)
        t = np.asarray(arrays["t"], dtype=float)
        m_rem = np.asarray(arrays["m_rem"], dtype=float)
        m2_max = (SQRT_S - m_rem) ** 2
        t_min = _t_range_min_vec(m2, m_rem)
        # the marginal normalizations, cached per window
        log_a0 = np.empty(m2.shape, dtype=float)
        log_ae = np.empty(m2.shape, dtype=float)
        for window in np.unique(m2_max):
            a0, ae = _sd_log_norm_pair(eps, float(window))
            log_a0[m2_max == window] = a0
            log_ae[m2_max == window] = ae
        # the nominal-side pair is cached per window as well
        log_a0_nom = np.empty(m2.shape, dtype=float)
        log_ae_nom = np.empty(m2.shape, dtype=float)
        for window in np.unique(m2_max):
            a0n, aen = _sd_log_norm_pair(NOMINAL_EPSILON, float(window))
            log_a0_nom[m2_max == window] = a0n
            log_ae_nom[m2_max == window] = aen
        log_zt = _log_t_norm_vec(b_sd, t_min)
        log_zt_nom = _log_t_norm_vec(NOMINAL_B_SD, t_min)
        return (
            -eps * np.log(m2)
            + (log_a0 - log_ae) - (log_a0_nom - log_ae_nom)
            + delta_b * t
            + (log_zt_nom - log_zt)
        )
    if cls == "dd":
        m12 = np.asarray(arrays["m12"], dtype=float)
        m22 = np.asarray(arrays["m22"], dtype=float)
        t1 = np.asarray(arrays["t1"], dtype=float)
        t2 = np.asarray(arrays["t2"], dtype=float)
        m2_max = (SQRT_S - PROTON_MASS) ** 2
        log_a0, log_ae = _sd_log_norm_pair(eps, m2_max)
        log_a0n, log_aen = _sd_log_norm_pair(NOMINAL_EPSILON, m2_max)
        log_dd = _log_dd_joint_mass(eps)
        log_dd_nom = _log_dd_joint_mass(NOMINAL_EPSILON)
        t_min1 = _t_range_min_vec(m12, PROTON_MASS)
        t_min2 = _t_range_min_vec(m22, PROTON_MASS)
        log_zt1 = _log_t_norm_vec(b_sd, t_min1)
        log_zt1_nom = _log_t_norm_vec(NOMINAL_B_SD, t_min1)
        log_zt2 = _log_t_norm_vec(b_sd, t_min2)
        log_zt2_nom = _log_t_norm_vec(NOMINAL_B_SD, t_min2)
        return (
            -eps * (np.log(m12) + np.log(m22))
            + ((log_a0 - log_ae) - (log_a0n - log_aen))
            + (log_dd_nom - log_dd)
            + delta_b * (t1 + t2)
            + (log_zt1_nom - log_zt1) + (log_zt2_nom - log_zt2)
        )
    if cls == "nd":
        t1 = np.asarray(arrays["t1"], dtype=float)
        t2 = np.asarray(arrays["t2"], dtype=float)
        t_c = np.asarray(arrays["t_c"], dtype=float)
        mc2 = np.asarray(arrays["mc2"], dtype=float)
        m1 = np.asarray(arrays["m1"], dtype=float)
        m2 = np.asarray(arrays["m2"], dtype=float)
        m2_max = (SQRT_S - m1 - m2) ** 2
        t_min1 = _t_range_min_vec(m1 * m1, PROTON_MASS)
        t_min2 = _t_range_min_vec(m2 * m2, PROTON_MASS)
        log_zt1 = _log_t_norm_vec(b_sd, t_min1)
        log_zt1_nom = _log_t_norm_vec(NOMINAL_B_SD, t_min1)
        log_zt2 = _log_t_norm_vec(b_sd, t_min2)
        log_zt2_nom = _log_t_norm_vec(NOMINAL_B_SD, t_min2)
        beta_trial = b_sd * t_c
        beta_nom = NOMINAL_B_SD * t_c
        log_ac_trial = _central_mass_log_integral(
            np.full(t_c.shape, eps), beta_trial, PI0_MASS * PI0_MASS,
            np.maximum(m2_max, PI0_MASS * PI0_MASS * (1.0 + 1e-12)))
        log_ac_nom = _central_mass_log_integral(
            np.full(t_c.shape, NOMINAL_EPSILON), beta_nom,
            PI0_MASS * PI0_MASS,
            np.maximum(m2_max, PI0_MASS * PI0_MASS * (1.0 + 1e-12)))
        return (
            delta_b * (t1 + t2)
            + (log_zt1_nom - log_zt1) + (log_zt2_nom - log_zt2)
            - eps * np.log(mc2)
            + (b_sd - NOMINAL_B_SD) * t_c * mc2
            + (log_ac_nom - log_ac_trial)
        )
    _refuse(f"class_log_law_weights: unknown class {cls!r}")


def _log_t_norm_vec(b: float, t_min) -> np.ndarray:
    """The vectorized :func:`_log_t_norm` (the same closed form,
    elementwise)."""
    t_min = np.asarray(t_min, dtype=float)
    out = np.zeros(t_min.shape, dtype=float)
    degenerate = t_min == 0.0
    live = ~degenerate
    b_float = float(b)
    if abs(b_float) <= _FLAT_SLOPE_EPS:
        out[live] = np.log(-t_min[live])
        return out
    btm = b_float * t_min
    z = np.where(btm < -700.0, 1.0 / b_float,
                 (1.0 - np.exp(np.minimum(btm, 0.0))) / b_float)
    if not np.all(np.isfinite(z[live])) or not np.all(z[live] > 0.0):
        _refuse(
            "log_t_norm_vec: a normalization is non-finite or "
            f"non-positive at b={b_float!r}")
    out[live] = np.log(z[live])
    return out


def event_log_weights(blocks: list, f_sdp: float, f_sdt: float,
                      f_dd: float, eps: float, b_sd: float):
    """The per-event log importance weights over all blocks: the
    mixture fraction ratio ``log(f_c / 0.25)`` plus the class's log
    mass-law ratio.  Returns ``(log_w per block, fractions dict)``; the
    fraction closure ``f_nd = 1 - sum`` is validated (a negative
    closure or a negative fraction refuses)."""
    f_nd = 1.0 - float(f_sdp) - float(f_sdt) - float(f_dd)
    fractions = {
        "sd_projectile": float(f_sdp),
        "sd_target": float(f_sdt),
        "dd": float(f_dd),
        "nd": f_nd,
    }
    if f_nd < 0.0 or any(v < 0.0 for v in fractions.values()):
        return None, fractions
    log_ratio = {
        cls: math.log(fractions[cls] / NOMINAL_CLASS_FRACTION)
        if fractions[cls] > 0.0 else float("-inf")
        for cls in CLASS_ORDER
    }
    per_block = []
    for block in blocks:
        log_w = np.full(block.n_events, float("-inf"), dtype=float)
        for cls in CLASS_ORDER:
            idx = block.idx_by_class[cls]
            if idx.size == 0:
                continue
            law = class_log_law_weights(cls, block.arrays_by_class[cls],
                                        eps, b_sd)
            log_w[idx] = log_ratio[cls] + law
        per_block.append(log_w)
    return per_block, fractions


def ess_fraction(w: np.ndarray) -> float:
    """``(sum w)^2 / sum w^2 / n`` over the weighted ensemble (the
    weight-degeneracy diagnostic; the house convention)."""
    if w.size == 0:
        return 1.0
    total = float(np.sum(w))
    squares = float(np.sum(w * w))
    if squares <= 0.0 or total <= 0.0:
        return 0.0
    return (total * total) / squares / float(w.size)


# ---------------------------------------------------------------------------
# The block loader (the sealed replica records -> the evaluation arrays)
# ---------------------------------------------------------------------------
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASS_ORDER)}


class Block:
    """One replica block's evaluation arrays: the class codes, the
    per-class weight inputs, the charged-multiplicity bins, the central-
    gap flags, and the spectrum-species lattice entries (the dense-cell
    assignment of the stored (x_F, p_T) pairs against the loaded
    published geometry)."""

    __slots__ = (
        "replica", "seed", "n_events", "cls", "idx_by_class",
        "arrays_by_class", "gap", "nch_idx", "prot_flat", "prot_evt",
        "prot_w", "pbar_flat", "pbar_evt", "pbar_w",
        "proton_out_of_window",
        "antiproton_out_of_window", "n_protons", "n_antiprotons",
    )


def _assign_cells(xf, pt, x_edges, p_edges, n_x, event_index=None,
                  measures=None):
    """The dense-lattice cell assignment of (x_F, p_T) pairs (the
    published geometry; out-of-window pairs are dropped and counted,
    never bled into an end cell).  The parallel event indices and the
    invariant measures are filtered to the kept pairs so the lattice
    entries carry their event map and measure."""
    xf = np.asarray(xf, dtype=float)
    pt = np.asarray(pt, dtype=float)
    if xf.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        kept = (np.zeros(0, dtype=np.int64)
                if event_index is not None else None)
        kept_w = (np.zeros(0, dtype=float) if measures is not None else None)
        return empty_i, kept, kept_w, 0
    ix = np.searchsorted(x_edges, xf, side="right") - 1
    ip = np.searchsorted(p_edges, pt, side="right") - 1
    inside = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < p_edges.size - 1)
    dropped = int((~inside).sum())
    flat = (ip[inside] * n_x + ix[inside]).astype(np.int64)
    kept_events = (np.asarray(event_index, dtype=np.int64)[inside]
                   if event_index is not None else None)
    kept_w = (np.asarray(measures, dtype=float)[inside]
              if measures is not None else None)
    return flat, kept_events, kept_w, dropped


def load_blocks(run_dir: Path, *, smoke: bool, geometry) -> list:
    """Load the sealed replica records into the evaluation arrays (the
    production replicas from the run dir's replica files; the smoke
    from the smoke replica file)."""
    run_dir = Path(run_dir)
    blocks = []
    names = ([SMOKE_REPLICA_NAME] if smoke
             else [replica_name(r) for r in range(N_REPLICAS)])
    if not smoke:
        for name in names:
            if not (run_dir / name).is_file():
                _refuse(f"the production record is missing: {run_dir / name}")
    for index, name in enumerate(names):
        record = _read_json(run_dir / name)
        events = record["events"]
        n_events = len(events)
        if not smoke:
            _verify_replica_identity(record, index, run_dir)
        cls = np.empty(n_events, dtype=np.int64)
        gap = np.empty(n_events, dtype=bool)
        nch_idx = np.empty(n_events, dtype=np.int64)
        rows_by_class: dict[str, list] = {c: [] for c in CLASS_ORDER}
        prot_evt, prot_xf, prot_pt, prot_w = [], [], [], []
        pbar_evt, pbar_xf, pbar_pt, pbar_w = [], [], [], []
        proton_out = 0
        antiproton_out = 0
        for i, ev in enumerate(events):
            label = ev["cl"]
            if label not in CLASS_INDEX:
                _refuse(f"the event carries an unknown class: {label!r}")
            ci = CLASS_INDEX[label]
            cls[i] = ci
            gap[i] = int(ev["mc"]) == 0
            nch = int(ev["nc"])
            if nch < MULTIPLICITY_BINS[0] or nch > MULTIPLICITY_BINS[-1]:
                _refuse(
                    f"the event carries charged multiplicity {nch} outside "
                    f"the pre-registered bins {MULTIPLICITY_BINS}; the "
                    "evaluation refuses (checked, never bled)")
            nch_idx[i] = nch - MULTIPLICITY_BINS[0]
            ex = ev["ex"]
            tr = ev["tr"]
            rm = ev["rm"]
            if label in ("sd_projectile", "sd_target"):
                if len(ex) != 1 or len(tr) != 1 or len(rm) != 1:
                    _refuse(f"the sd event {i} carries a malformed payload")
                rows_by_class[label].append((ex[0] ** 2, tr[0], rm[0]))
            elif label == "dd":
                if len(ex) != 2 or len(tr) != 2 or rm:
                    _refuse(f"the dd event {i} carries a malformed payload")
                rows_by_class[label].append(
                    (ex[0] ** 2, ex[1] ** 2, tr[0], tr[1]))
            else:
                if len(ex) != 1 or len(tr) != 3 or len(rm) != 2:
                    _refuse(f"the nd event {i} carries a malformed payload")
                rows_by_class[label].append(
                    (tr[0], tr[1], tr[2], ex[0] ** 2, rm[0], rm[1]))
            for x_f, p_t, w_meas in ev["pp"]:
                prot_evt.append(i)
                prot_xf.append(x_f)
                prot_pt.append(p_t)
                prot_w.append(w_meas)
            for x_f, p_t, w_meas in ev["pb"]:
                pbar_evt.append(i)
                pbar_xf.append(x_f)
                pbar_pt.append(p_t)
                pbar_w.append(w_meas)
        def _sd_arrays(rows):
            return {
                "m2": np.asarray([r[0] for r in rows], dtype=float),
                "t": np.asarray([r[1] for r in rows], dtype=float),
                "m_rem": np.asarray([r[2] for r in rows], dtype=float),
            }

        def _dd_arrays(rows):
            return {
                "m12": np.asarray([r[0] for r in rows], dtype=float),
                "m22": np.asarray([r[1] for r in rows], dtype=float),
                "t1": np.asarray([r[2] for r in rows], dtype=float),
                "t2": np.asarray([r[3] for r in rows], dtype=float),
            }

        def _nd_arrays(rows):
            return {
                "t1": np.asarray([r[0] for r in rows], dtype=float),
                "t2": np.asarray([r[1] for r in rows], dtype=float),
                "t_c": np.asarray([r[2] for r in rows], dtype=float),
                "mc2": np.asarray([r[3] for r in rows], dtype=float),
                "m1": np.asarray([r[4] for r in rows], dtype=float),
                "m2": np.asarray([r[5] for r in rows], dtype=float),
            }

        idx_by_class = {
            cls_label: np.flatnonzero(cls == CLASS_INDEX[cls_label])
            for cls_label in CLASS_ORDER
        }
        builders = {
            "sd_projectile": _sd_arrays, "sd_target": _sd_arrays,
            "dd": _dd_arrays, "nd": _nd_arrays,
        }
        arrays_by_class = {
            cls_label: builders[cls_label](rows_by_class[cls_label])
            for cls_label in CLASS_ORDER
        }
        block = Block()
        block.replica = (record.get("replica") if not smoke else "smoke")
        block.seed = record.get("seed")
        block.n_events = n_events
        block.cls = cls
        block.idx_by_class = idx_by_class
        block.arrays_by_class = arrays_by_class
        block.gap = gap
        block.nch_idx = nch_idx
        # the spectrum-species lattice entries (the pick geometry)
        prot_flat, prot_kept_evt, prot_kept_w, prot_dropped = _assign_cells(
            np.asarray(prot_xf, dtype=float), np.asarray(prot_pt, dtype=float),
            geometry["edges"]["proton"][0], geometry["edges"]["proton"][1],
            geometry["n_x"]["proton"], np.asarray(prot_evt, dtype=np.int64),
            np.asarray(prot_w, dtype=float))
        pbar_flat, pbar_kept_evt, pbar_kept_w, pbar_dropped = _assign_cells(
            np.asarray(pbar_xf, dtype=float), np.asarray(pbar_pt, dtype=float),
            geometry["edges"]["antiproton"][0],
            geometry["edges"]["antiproton"][1], geometry["n_x"]["antiproton"],
            np.asarray(pbar_evt, dtype=np.int64),
            np.asarray(pbar_w, dtype=float))
        block.prot_flat = prot_flat
        block.prot_evt = prot_kept_evt
        block.prot_w = prot_kept_w
        block.pbar_flat = pbar_flat
        block.pbar_evt = pbar_kept_evt
        block.pbar_w = pbar_kept_w
        block.proton_out_of_window = prot_dropped
        block.antiproton_out_of_window = pbar_dropped
        block.n_protons = len(prot_evt)
        block.n_antiprotons = len(pbar_evt)
        blocks.append(block)
    return blocks



# ---------------------------------------------------------------------------
# The training statistics (the pre-registered instantiation)
# ---------------------------------------------------------------------------
class Statistic:
    """One training-observable statistic (d, V_data, and the model
    evaluator kind).  Kinds: 'spectrum' (the binned published cells),
    'fraction' (the single central-gap bin), 'distribution' (the
    charged-multiplicity bins)."""

    __slots__ = ("key", "kind", "d", "v_data", "cells", "carry_mask",
                 "species", "note")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def build_statistics(sealed: dict, r50_reference: dict) -> list:
    """The pre-registered training statistic set: the leading-proton
    spectrum (the proton species' published cells), the B_transport
    spectrum (the aligned proton/antiproton cells, plan section 7.3
    line 431), the central-window gap fraction, and the charged-
    multiplicity distribution."""
    stats = []
    species_a, species_b, sign = B_TRANSPORT_PAIR
    # S1: the leading-proton spectrum on the proton table's published rows
    stats.append(Statistic(
        key="leading_proton_spectrum", kind="spectrum", species="proton",
        d=np.asarray(sealed["proton"]["d"], dtype=float),
        v_data=np.asarray(sealed["proton"]["v_data"], dtype=float),
        cells=None, carry_mask=None,
        note=("the proton species' published-cell spectrum; under the "
              "topology model every final-state proton originates in a "
              "baryonic leading line (the central source carries "
              "B = Q = S = 0), so the proton spectrum on the published "
              "(x_F, p_T) cells is the leading-proton structure's "
              "observable (plan line 913)")))
    # S2: the B_transport combination over the aligned cells
    cells = _aligned_cells(sealed)
    if not cells:
        _refuse("no aligned proton/antiproton cells exist; the b_transport "
                "statistic cannot be built")
    d = np.asarray([
        sealed[species_a]["d"][e["bin_a"]]
        + sign * sealed[species_b]["d"][e["bin_b"]]
        for e in cells
    ], dtype=float)
    v_data = np.asarray([
        sealed[species_a]["v_data"][e["bin_a"]]
        + sealed[species_b]["v_data"][e["bin_b"]]
        for e in cells
    ], dtype=float)
    carry_set = {(entry["species"], int(entry["bin_index"]))
                 for entry in _load_carry_bins()}
    carry = np.asarray([
        1.0 if ((species_a, e["bin_a"]) in carry_set
                or (species_b, e["bin_b"]) in carry_set) else 0.0
        for e in cells
    ], dtype=float)
    stats.append(Statistic(
        key="b_transport_spectrum", kind="spectrum", species=None,
        d=d, v_data=v_data, cells=cells, carry_mask=carry,
        note=("the net-proton combination B_transport = p - p-bar (plan "
              "section 7.3 line 431) over the aligned published cells; "
              "the standing model-uncertainty carry applies to the bins "
              "whose proton or antiproton cell is in the sealed 32 "
              "persisting-bin set (the R41 carry convention)")))
    # S3: the central-window gap fraction (the R50-pinned reference)
    gap = r50_reference["gap_fraction"]
    stats.append(Statistic(
        key="gap_fraction_abs_y_lt_0.5", kind="fraction", species=None,
        d=np.asarray([gap["value"]], dtype=float),
        v_data=np.asarray([max(gap["variance"], SE_FLOOR)], dtype=float),
        cells=None, carry_mask=None,
        note=("the fraction of inelastic events with no hadronic "
              "final-state particle in |y^CM| < 0.5; the data side is "
              "the sealed R50 nominal-point reference (the 0.25-each "
              "pooled mixture, its binomial MC variance as V_data; the "
              "sealed substrate carries no published gap-fraction "
              "table)")))
    # S4: the charged-multiplicity distribution (the R50-pinned reference)
    mult = r50_reference["charged_multiplicity"]
    bins = mult["bins"]
    d = np.asarray([mult["distribution"][str(n)] for n in bins], dtype=float)
    v_data = np.asarray([max(mult["variance"][str(n)], SE_FLOOR)
                         for n in bins], dtype=float)
    stats.append(Statistic(
        key="charged_multiplicity_distribution", kind="distribution",
        species=None, d=d, v_data=v_data, cells=None, carry_mask=None,
        note=("the charged-multiplicity distribution over the "
              "pre-registered bins n_ch = 2..14; the data side is the "
              "sealed R50 nominal-point reference (the 0.25-each pooled "
              "mixture, its per-class binomial variance, diagonal; the "
              "sealed substrate carries no published multiplicity "
              "table)")))
    return stats
# ---------------------------------------------------------------------------
# The profiled objective (plan section 5.4; the R41 pattern)
# ---------------------------------------------------------------------------
NUISANCE_COLUMNS = (
    "norm_leading_proton_spectrum", "norm_b_transport_spectrum", "carry")
SPECTRUM_NORM_COLUMN = {
    "leading_proton_spectrum": 0,
    "b_transport_spectrum": 1,
}
CARRY_COLUMN_INDEX = 2


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


def _species_pick_values(sums: np.ndarray, pick: np.ndarray,
                         areas: np.ndarray, sqrt_s: float,
                         n_eff: float) -> np.ndarray:
    """The weighted per-species binned prediction in the data units (the
    R01 convention ``m = sums / (pi sqrt(s) N_eff dxF dpT)``): the dense
    lattice slice is picked at the published rows, so the returned
    values align 1:1 with the published rows and a hole cell's weight
    stays unpicked and is dropped."""
    dense_slice = sums[:int(pick.max()) + 1] if pick.size else sums[:0]
    picked = dense_slice[pick]
    return picked / (math.pi * float(sqrt_s) * float(n_eff) * areas)


class FitObjective:
    """The profiled -2lnL of the pre-registered likelihood (plan section
    5.4): the five-parameter fit set enters through the exact per-event
    importance weights (the mixture fraction ratio and the class mass-law
    ratios); the nuisance profile (the two spectrum normalization
    nuisances plus the model-uncertainty carry) is solved exactly per
    trial by the generalized-least-squares profile.  The dropped-bin set
    is fixed at the unit-weight sample structure (the bins structurally
    empty at the generation point and the bins with non-finite data or
    published variance; the R11 convention, listed, never silently
    dropped), so the objective is a continuous function of the fit set
    on a fixed bin set; a bin populated only at the initial point and
    emptied by a trial's reweighting keeps its data residual with the
    trial's honest zero prediction."""

    def __init__(self, stats: list, blocks: list, geometry: dict,
                 sqrt_s: float) -> None:
        self.stats = stats
        self.blocks = blocks
        self.geometry = geometry
        self.sqrt_s = float(sqrt_s)
        self.kept: dict[str, np.ndarray] = {}
        self.dropped_bins: list = []

    # -- the fixed bin structure (computed once, at the unit weights) ----
    def build_bin_structure(self) -> dict:
        """The fixed kept/dropped bin structure: a bin is kept when its
        data side is finite with a non-negative published variance AND
        at least one stored event of any block contributes to it at
        unit weights (the generation-point structure)."""
        self.kept = {}
        self.dropped_bins = []
        structure = self.geometry["unit_structure"]
        for stat in self.stats:
            d = np.asarray(stat.d, dtype=float)
            v = np.asarray(stat.v_data, dtype=float)
            data_ok = (np.isfinite(d) & np.isfinite(v) & (v >= 0.0))
            populated = np.asarray(
                structure[stat.key], dtype=bool)
            keep = data_ok & populated
            self.kept[stat.key] = keep
            for i in np.flatnonzero(~keep):
                self.dropped_bins.append({
                    "statistic": stat.key, "index": int(i),
                    "reason": ("structurally empty at the generation point "
                               "(no stored event contributes at unit "
                               "weights; the R11 convention)"
                               if not bool(populated[i]) else
                               "non-finite data or published variance"),
                })
        return {
            "kept_per_statistic": {
                key: int(mask.sum()) for key, mask in self.kept.items()},
            "dropped": len(self.dropped_bins),
        }

    # -- the trial evaluation --------------------------------------------
    def evaluate(self, f_sdp: float, f_sdt: float, f_dd: float,
                 eps: float, b_sd: float) -> dict:
        f_nd = 1.0 - float(f_sdp) - float(f_sdt) - float(f_dd)
        fractions = {
            "sd_projectile": float(f_sdp), "sd_target": float(f_sdt),
            "dd": float(f_dd), "nd": f_nd,
        }
        log_all = []
        for block in self.blocks:
            lw, _ = event_log_weights(
                [block], f_sdp, f_sdt, f_dd, eps, b_sd)
            if lw is None:
                return {"value": float("inf"), "fractions": fractions}
            log_all.append(lw[0])
        stacked = np.concatenate(log_all) if log_all else np.zeros(0)
        finite = np.isfinite(stacked)
        if not finite.any():
            return {"value": float("inf"), "fractions": fractions}
        # one GLOBAL log shift across the blocks, so the pooled sums and
        # the per-block sums are mutually consistent (the self-normalized
        # estimator and the ESS are shift-invariant)
        shift = float(np.max(stacked[finite]))
        with np.errstate(over="ignore", under="ignore"):
            weights = [np.exp(lw - shift) for lw in log_all]
        per_block = []
        prot_total = np.zeros(self.geometry["dense_size"]["proton"])
        pbar_total = np.zeros(self.geometry["dense_size"]["antiproton"])
        mult_total = np.zeros(len(MULTIPLICITY_BINS))
        gap_total = 0.0
        n_eff_total = 0.0
        for block, w in zip(self.blocks, weights):
            n_eff = float(np.sum(w))
            if not math.isfinite(n_eff) or n_eff <= 0.0:
                return {"value": float("inf"), "fractions": fractions}
            gap_num = float(np.sum(w[block.gap]))
            mult_counts = np.bincount(
                block.nch_idx, weights=w, minlength=len(MULTIPLICITY_BINS))
            prot_w = (w[block.prot_evt] * block.prot_w
                      if block.prot_evt.size else np.zeros(0, dtype=float))
            pbar_w = (w[block.pbar_evt] * block.pbar_w
                      if block.pbar_evt.size else np.zeros(0, dtype=float))
            prot_sums = np.bincount(
                block.prot_flat, weights=prot_w,
                minlength=self.geometry["dense_size"]["proton"])
            pbar_sums = np.bincount(
                block.pbar_flat, weights=pbar_w,
                minlength=self.geometry["dense_size"]["antiproton"])
            per_block.append({
                "n_eff": n_eff, "gap_num": gap_num,
                "mult_counts": mult_counts, "prot_sums": prot_sums,
                "pbar_sums": pbar_sums, "ess": ess_fraction(w),
            })
            prot_total += prot_sums
            pbar_total += pbar_sums
            mult_total += mult_counts
            gap_total += gap_num
            n_eff_total += n_eff
        if not (math.isfinite(n_eff_total) and n_eff_total > 0.0):
            return {"value": float("inf"), "fractions": fractions}
        # -- the per-block and pooled species values (the published rows) --
        geometry = self.geometry
        prot_pick = geometry["pick"]["proton"]
        pbar_pick = geometry["pick"]["antiproton"]
        prot_areas = geometry["areas"]["proton"]
        pbar_areas = geometry["areas"]["antiproton"]
        prot_pooled = _species_pick_values(
            prot_total, prot_pick, prot_areas, self.sqrt_s, n_eff_total)
        pbar_pooled = _species_pick_values(
            pbar_total, pbar_pick, pbar_areas, self.sqrt_s, n_eff_total)
        prot_block = [_species_pick_values(
            entry["prot_sums"], prot_pick, prot_areas, self.sqrt_s,
            entry["n_eff"]) for entry in per_block]
        pbar_block = [_species_pick_values(
            entry["pbar_sums"], pbar_pick, pbar_areas, self.sqrt_s,
            entry["n_eff"]) for entry in per_block]
        # -- the statistic vectors (per block and pooled) --
        pooled_vectors = []
        block_vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum" and stat.species == "proton":
                pooled_vectors.append(np.asarray(prot_pooled, dtype=float))
                block_vectors.append(
                    [np.asarray(v, dtype=float) for v in prot_block])
            elif stat.kind == "spectrum":
                species_a, species_b, sign = B_TRANSPORT_PAIR
                pooled_vectors.append(np.asarray([
                    prot_pooled[e["bin_a"]] + sign * pbar_pooled[e["bin_b"]]
                    for e in stat.cells
                ], dtype=float))
                block_vectors.append([np.asarray([
                    vb[e["bin_a"]] + sign * bb[e["bin_b"]]
                    for e in stat.cells
                ], dtype=float) for vb, bb in zip(prot_block, pbar_block)])
            elif stat.kind == "fraction":
                pooled_vectors.append(np.asarray(
                    [gap_total / n_eff_total], dtype=float))
                block_vectors.append([np.asarray(
                    [entry["gap_num"] / entry["n_eff"]], dtype=float)
                    for entry in per_block])
            else:
                pooled_vectors.append(
                    np.asarray(mult_total / n_eff_total, dtype=float))
                block_vectors.append([
                    np.asarray(entry["mult_counts"] / entry["n_eff"],
                               dtype=float) for entry in per_block])
        # -- the between-replica SEs --
        se_vectors = [
            se_from(np.asarray(vectors, dtype=float))
            for vectors in block_vectors
        ]
        # -- the profiled -2lnL on the fixed kept set --
        value, eta_hat = self._profiled_minus2lnl(
            pooled_vectors, se_vectors)
        if not math.isfinite(value):
            return {"value": float("inf"), "fractions": fractions}
        return {
            "value": float(value),
            "pooled_vectors": pooled_vectors,
            "block_vectors": block_vectors,
            "se": se_vectors,
            "per_block": per_block,
            "n_eff_total": float(n_eff_total),
            "log_shift": float(shift),
            "eta_hat": [float(v) for v in eta_hat],
            "ess": {
                "per_block": [entry["ess"] for entry in per_block],
                "min": min(entry["ess"] for entry in per_block),
            },
            "fractions": fractions,
        }

    # -- the nuisance profile --------------------------------------------
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
                "r51 fit: no finite objective bin (the statistic set is "
                "empty after the structure drops); the run stops"
            )
        r = np.asarray(rows, dtype=float)
        design = np.asarray(columns, dtype=float)
        prior_diag = np.asarray([
            NORM_NUISANCE_PRIOR ** 2, NORM_NUISANCE_PRIOR ** 2,
            CARRY_NUISANCE_PRIOR ** 2,
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
        """The multiplicative nuisance design row (the profile of the
        documented convention ``m -> m (1 + eta)``): the derivative of
        the model value with respect to the nuisance is the model value
        itself, so the row entry is ``m_i`` (a zero model bin carries a
        zero row: a normalization cannot move a structurally empty
        bin).  The carry column is the bin's carry mask times the model
        value."""
        row = [0.0] * len(NUISANCE_COLUMNS)
        column = SPECTRUM_NORM_COLUMN.get(stat.key)
        if column is not None:
            row[column] = float(model_value)
        if stat.key == "b_transport_spectrum" and stat.carry_mask is not None:
            row[CARRY_COLUMN_INDEX] = (
                float(model_value) * float(stat.carry_mask[index]))
        return row


# ---------------------------------------------------------------------------
# The geometry (the published pick geometry and the unit-weight structure)
# ---------------------------------------------------------------------------
def build_geometry(tables: list) -> dict:
    """The published pick geometry of the two spectrum species (the
    dense pT-major lattices, the pick maps, the cell areas) with the
    per-species edges (the R41 layout conventions)."""
    geometry = {"edges": {}, "n_x": {}, "dense_size": {}, "pick": {},
                "areas": {}, "unit_structure": {}}
    for table in tables:
        species = str(table.species)
        x_edges = np.unique(sps_core.edges_from_centers(table.xf_centers))
        p_edges = np.unique(sps_core.edges_from_centers(table.pt_centers))
        n_x = int(table.bin_index_xf.max()) + 1
        n_p = int(table.bin_index_pt.max()) + 1
        pick = (np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
                + np.asarray(table.bin_index_xf, dtype=np.int64))
        if pick.size and not (
                int(pick.min()) >= 0 and int(pick.max()) < n_x * n_p):
            _refuse(
                f"species {species!r}: a published bin_index pair maps "
                f"outside the dense lattice ({n_x} x {n_p}); the pick map "
                "cannot be built")
        if len(set(pick.tolist())) != int(pick.size):
            _refuse(
                f"species {species!r}: two published rows map to one "
                "dense lattice cell; the pick map is not one-to-one")
        areas = np.asarray([
            (float(x_edges[int(table.bin_index_xf[i]) + 1])
             - float(x_edges[int(table.bin_index_xf[i])]))
            * (float(p_edges[int(table.bin_index_pt[i]) + 1])
               - float(p_edges[int(table.bin_index_pt[i])]))
            for i in range(int(table.data.size))
        ], dtype=float)
        geometry["edges"][species] = (x_edges, p_edges)
        geometry["n_x"][species] = n_x
        geometry["dense_size"][species] = n_x * n_p
        geometry["pick"][species] = pick
        geometry["areas"][species] = areas
    return geometry


def build_unit_structure(stats: list, blocks: list, geometry: dict) -> dict:
    """The unit-weight population structure per statistic (computed
    once, at the generation point): which bins at least one stored
    event of any block contributes to."""
    prot_pop = np.zeros(geometry["dense_size"]["proton"], dtype=bool)
    pbar_pop = np.zeros(geometry["dense_size"]["antiproton"], dtype=bool)
    gap_pop = False
    mult_pop = np.zeros(len(MULTIPLICITY_BINS), dtype=bool)
    n_protons = 0
    n_antiprotons = 0
    for block in blocks:
        if block.prot_flat.size:
            prot_pop[block.prot_flat] = True
            n_protons += int(block.prot_flat.size)
        if block.pbar_flat.size:
            pbar_pop[block.pbar_flat] = True
            n_antiprotons += int(block.pbar_flat.size)
        gap_pop = bool(gap_pop or block.gap.any())
        mult_pop |= np.bincount(
            block.nch_idx, minlength=len(MULTIPLICITY_BINS)).astype(bool)
    structure = {}
    for stat in stats:
        if stat.kind == "spectrum" and stat.species == "proton":
            structure[stat.key] = prot_pop[geometry["pick"]["proton"]]
        elif stat.kind == "spectrum":
            species_a, species_b, _sign = B_TRANSPORT_PAIR
            structure[stat.key] = np.asarray([
                bool(prot_pop[e["bin_a"]] or pbar_pop[e["bin_b"]])
                for e in stat.cells
            ], dtype=bool)
        elif stat.kind == "fraction":
            structure[stat.key] = np.asarray([gap_pop], dtype=bool)
        else:
            structure[stat.key] = mult_pop
    structure["_counts"] = {
        "protons_stored": n_protons, "antiprotons_stored": n_antiprotons,
        "proton_bins_populated": int(prot_pop.sum()),
        "antiproton_bins_populated": int(pbar_pop.sum()),
        "multiplicity_bins_populated": int(mult_pop.sum()),
        "gap_flag_any": bool(gap_pop),
    }
    return structure


# ---------------------------------------------------------------------------
# The optimizer, the per-block fits, the stability table, the Hessian
# discipline, and the finite-MC profile (the R41/R46 pattern)
# ---------------------------------------------------------------------------
def _domain_ok(f_sdp: float, f_sdt: float, f_dd: float) -> bool:
    f_nd = 1.0 - float(f_sdp) - float(f_sdt) - float(f_dd)
    return (f_nd >= 0.0 and float(f_sdp) >= 0.0 and float(f_sdt) >= 0.0
            and float(f_dd) >= 0.0)


def _minimize_pooled(objective: FitObjective):
    """The pre-registered 5-parameter optimizer (scipy Nelder-Mead,
    bounded, deterministic; the settings are the pre-registered
    OPTIMIZER_SETTINGS)."""
    from scipy.optimize import minimize

    bounds = [tuple(PARAM_BOUNDS[name]) for name in FIT_PARAMETERS]
    x0 = list(OPTIMIZER_SETTINGS["initial_point"])

    def target(x):
        f_sdp, f_sdt, f_dd, eps, b_sd = (float(v) for v in x)
        if not _domain_ok(f_sdp, f_sdt, f_dd):
            return float("inf")
        value = objective.evaluate(f_sdp, f_sdt, f_dd, eps, b_sd)["value"]
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


def _fit_blocks_individually(stats, blocks, geometry, sqrt_s) -> list:
    """The 8 per-block fits (the replica-block stability table): the
    same pre-registered procedure on each block's own data."""
    block_fits = []
    for block in blocks:
        objective = FitObjective(stats, [block], geometry, sqrt_s)
        objective.build_bin_structure()
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


def _pooled_curvature(objective: FitObjective, best_x) -> np.ndarray:
    """The 5x5 numerical Hessian of the pooled objective at the optimum
    (central differences with the pre-registered steps)."""
    n = len(FIT_PARAMETERS)
    steps = [HESSIAN_STEPS[name] for name in FIT_PARAMETERS]
    hessian = np.zeros((n, n), dtype=float)
    cache: dict[tuple, float] = {}

    def value_at(x):
        key = tuple(round(float(v), 15) for v in x)
        if key not in cache:
            if not _domain_ok(*x[:3]):
                cache[key] = float("inf")
            else:
                value = objective.evaluate(*x)["value"]
                cache[key] = float(value) if math.isfinite(value) else \
                    float("inf")
        return cache[key]

    for i in range(n):
        for j in range(i, n):
            si, sj = steps[i], steps[j]
            base = list(best_x)
            xpp = list(base); xpp[i] += si; xpp[j] += sj
            xpm = list(base); xpm[i] += si; xpm[j] -= sj
            xmp = list(base); xmp[i] -= si; xmp[j] += sj
            xmm = list(base); xmm[i] -= si; xmm[j] -= sj
            entry = (value_at(xpp) - value_at(xmp)
                     - value_at(xpm) + value_at(xmm)) / (4.0 * si * sj)
            hessian[i][j] = entry
            hessian[j][i] = entry
    return hessian


def _pooled_errors(objective: FitObjective, best_x) -> dict:
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


def _stability_table(best_x, block_fits, pooled_errors: dict) -> dict:
    """The acceptance (ii) stability table: per parameter the 8 block
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


def _parameter_profiles(objective: FitObjective, best_x) -> dict:
    """The finite-MC profile (the fallback's recorded profile): per
    parameter the 9-point equally spaced grid over the pre-registered
    bounds, the other four parameters at the best point."""
    profiles = {}
    for index, name in enumerate(FIT_PARAMETERS):
        lo, hi = PARAM_BOUNDS[name]
        points = []
        for g in range(PROFILE_GRID_POINTS):
            value = lo + (hi - lo) * g / (PROFILE_GRID_POINTS - 1)
            x = list(best_x)
            x[index] = value
            if not _domain_ok(*x[:3]):
                points.append({"value": value, "minus2lnl": None})
                continue
            result = objective.evaluate(*x)
            points.append({
                "value": value,
                "minus2lnl": (float(result["value"])
                              if math.isfinite(result["value"]) else None),
            })
        profiles[name] = points
    return profiles


def quoted_flag(minima_agree: bool, positive: bool, converged: bool,
                smoke: bool) -> bool:
    """The Hessian is quoted only after the optimizer converged, the
    replica-block minima agree (the stability table passes on every
    parameter), and the pooled curvature is positive definite (plan
    line 1871; the R46 pattern; never in smoke)."""
    return bool((not smoke) and converged and minima_agree and positive)


def _assemble_named_residuals(report: dict) -> list:
    """The named-residual register (the R46 forms): every failing
    acceptance verdict named with the numbers and a one-line diagnosis."""
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
                    "(the pre-registered acceptance (ii) bound); the "
                    "parameter is weakly constrained by the training "
                    "observables and the Gate 5 clause 1 unconstrained "
                    "clause consumes it directly"),
            })
    hessian = report.get("hessian") or {}
    if hessian and not hessian.get("positive_definite", True):
        residuals.append({
            "kind": "block_curvature_failure",
            "eigenvalues": hessian.get("eigenvalues"),
            "diagnosis": (
                "the pooled numerical Hessian at the optimum is not "
                "positive definite; no Hessian-based covariance is "
                "quoted (the R46 block-curvature pattern)"),
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
    return residuals


# ---------------------------------------------------------------------------
# The fit evaluation (the run's --evaluate; the report machinery)
# ---------------------------------------------------------------------------
REPORT_SCHEMA = "na49_mixing.staged_r51_topology_fraction_fit.report/1"

DETERMINISM_NOTE = (
    "a per-replica re-run from a clean interpreter reproduces the report "
    "hashes; fit_report.json and gate5_r51.json carry no wall-clock "
    "fields (their bytes are a pure function of the sealed artifacts, "
    "the pre-registered seeds, and the committed code); the per-event "
    "hash is the sha256 of the canonical JSON of the stored event record "
    "and the event_hash_set_sha256 is the sha256 over the joined "
    "per-event hashes, so the stored events carry the bit-level identity "
    "evidence and the probe replica's clean re-run must reproduce them")

SUBSTREAM_COLLISION_RULE = (
    "if any raw (seed, event-index) substream collides with a "
    "pre-registered block, the run stops and reports; a silent re-block "
    "never happens (index item 9)")

WEIGHT_CONVENTION_NOTE = (
    "the importance weights are the exact per-event density ratios of "
    "the pre-registered fit set: the mixture fraction ratio f_c / 0.25 "
    "(the drawn class is part of the event's identity; the reweighted "
    "estimator targets the joint (class, kinematics) density at the "
    "trial fractions) times the class's mass-law density ratio "
    "rho_c(event; epsilon, B_SD) / rho_c(event; 0, B_SD_nominal), "
    "evaluated from the event's stored excitation masses, transfers, and "
    "remnant masses: the single-diffraction marginal (the smooth power "
    "integral on the exact window [m_p^2, (sqrt_s - m_remnant)^2] at "
    "t = 0) and the exact t draw; the double-diffraction joint law (the "
    "closure-restricted product of the two marginals; the reduced "
    "one-dimensional form evaluated by the pre-registered 256-point "
    "Gauss-Legendre rule, cached per epsilon) and the two exact t draws; "
    "the non-diffractive central law (the incomplete-gamma closed form "
    "of the continuum integral at the per-event total transfer t_c = "
    "t_1 + t_2, with the s + 1 recurrence and the E1 branch at |s| < "
    "1e-8) and the two exact t draws.  The weights are computed in the "
    "log domain and exponentiated after one global log-shift per trial "
    "(the self-normalized estimator and the ESS are shift-invariant; "
    "the shift is recorded as a diagnostic).  At the pre-registered "
    "initial point every weight is exactly 1 (the nominal identity, "
    "checked at the evaluation).  The implemented M^2 draws normalize "
    "on the documented card-4 4096-point grid (the sub-tolerance "
    "convention, order 1e-3); the weight convention's smooth continuum "
    "normalizations stand at that documented sub-tolerance from the "
    "implemented grid normalization (recorded here as the weight-"
    "convention note); the t draws carry no grid and are exact")

SQRT_S_CONVERSION_NOTE = (
    "the data-units conversion uses the exact fixed-target invariant "
    "mass sqrt(s) = 17.270205201372406 GeV (the R01 convention's "
    "Reaction-based value; the CM-frame extraction of the stored "
    "(x_F, p_T) pairs uses the runner's extraction energy SQRT_S = "
    "17.270 GeV with beta = 0.9940791999921171 and gamma = "
    "9.203303892226666, the pinned convention of the R50 stored "
    "quantities)")


def _sqrt_s_conversion() -> float:
    """The R01 data-units conversion's sqrt(s) (the exact invariant mass
    of the pinned beam configuration; the R41 convention)."""
    from na49_mixing.reaction import Reaction

    raw = json.loads((NESTED_ROOT / "config"
                      / r01.PRODUCTION_CONFIG_NAME).read_text(
                          encoding="utf-8"))
    return float(Reaction(float(raw.get("beam_momentum", 158.0)),
                          2212, 2212).sqrt_s())


def _conservation_report(blocks: list, run_dir: Path, *, smoke: bool) -> dict:
    """The pooled conservation report over the full sample (the replica
    records' conservation aggregates)."""
    violations = 0
    events_checked = 0
    worst_closure = 0.0
    worst_shell = 0.0
    worst_energy_gev = 0.0
    worst_momentum_gev = 0.0
    bqs_census: dict[str, int] = {}
    n_events = 0
    for block in blocks:
        run_dir_name = (SMOKE_REPLICA_NAME if smoke
                        else replica_name(int(block.replica)))
        record = _read_json(Path(run_dir) / run_dir_name)
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


def _analytic_checks(objective: FitObjective, initial: dict) -> dict:
    """The analytic identity checks (the agreement table's 'analytic'
    column where applicable): the fraction-sum closure at the fitted
    point, the nominal-point weight identity (the log shift zero and
    the ESS exactly 1 at the initial point), the two-body Feynman-x
    pair sums on the pooled sample, and the elastic-class absence."""
    xf_sum_max = 0.0
    for block in objective.blocks:
        run_dir_name = (SMOKE_REPLICA_NAME
                        if getattr(block, "replica", None) == "smoke"
                        else replica_name(int(block.replica)))
        record = _read_json(objective.geometry["_run_dir"] / run_dir_name)
        for event in record["events"]:
            if event["cl"] in ("sd_projectile", "sd_target", "dd"):
                xf_sum_max = max(
                    xf_sum_max,
                    abs(float(event["xp"][0]) + float(event["xp"][1])))
    return {
        "fraction_closure": (
            "f_nd = 1 - f_sd_p - f_sd_t - f_dd exactly at the fitted "
            "point (the sum rule over the inelastic sample)"),
        "fitted_f_nd": (objective.last_fractions or {}).get("nd"),
        "nominal_weight_identity": {
            "log_shift_at_initial": initial.get("log_shift"),
            "ess_at_initial": initial.get("ess", {}).get("per_block"),
            "note": ("at the pre-registered initial point every "
                     "importance weight is exactly 1: the log shift is "
                     "0.0 and every block's ESS is exactly 1"),
            "pass": bool(initial.get("log_shift") == 0.0 and all(
                v == 1.0 for v in (initial.get("ess", {}).get("per_block")
                                   or []))),
        },
        "xf_pair_sum_max_abs_two_body": xf_sum_max,
        "xf_pair_sum_tolerance": 1e-9,
        "xf_pair_sum_note": (
            "the two-body classes (sd_projectile, sd_target, dd) carry "
            "the exact back-to-back CM pair sum x_F^p + x_F^t = 0 (the "
            "closed kinematics; the elastic normalization reference "
            "carries the same identity)"),
        "elastic_events": 0,
        "elastic_note": (
            "the mixed draw runs at the elastic fraction exactly 0.0 "
            "(plan line 824); a drawn elastic label refuses the run"),
    }


def _statistic_agreement(stat, pooled_fit, pooled_initial, se,
                         objective: FitObjective) -> dict:
    """One statistic's agreement block (the fit vs the R50-pinned
    initial-point model, with the kept-bin pulls)."""
    keep = objective.kept[stat.key]
    d = np.asarray(stat.d, dtype=float)
    m_fit = np.asarray(pooled_fit, dtype=float)
    m_ini = np.asarray(pooled_initial, dtype=float)
    se_v = np.asarray(se, dtype=float)
    idx = np.flatnonzero(keep)
    if idx.size:
        pulls_fit = (d[idx] - m_fit[idx]) / np.maximum(se_v[idx], SE_FLOOR)
        pulls_ini = (d[idx] - m_ini[idx]) / np.maximum(se_v[idx], SE_FLOOR)
        dev_fit = float(np.sum(pulls_fit ** 2))
        dev_ini = float(np.sum(pulls_ini ** 2))
        max_pull_fit = float(np.max(np.abs(pulls_fit)))
        max_pull_ini = float(np.max(np.abs(pulls_ini)))
    else:
        dev_fit = dev_ini = 0.0
        max_pull_fit = max_pull_ini = 0.0
    return {
        "key": stat.key,
        "kind": stat.kind,
        "note": stat.note,
        "n_bins": int(d.size),
        "n_kept": int(idx.size),
        "deviance_contribution_fit": dev_fit,
        "deviance_contribution_initial": dev_ini,
        "max_abs_pull_fit": max_pull_fit,
        "max_abs_pull_initial": max_pull_ini,
        "data": [float(v) for v in d],
        "model_fit": [float(v) for v in m_fit],
        "model_initial_r50_pinned": [float(v) for v in m_ini],
        "mc_se": [float(v) if math.isfinite(float(v)) else None
                  for v in se_v],
        "kept": [bool(v) for v in keep],
    }


def run_fit_machinery(run_dir: Path, *, smoke: bool = False) -> dict:
    """The fit evaluation: load the sealed inputs and the replica
    records, build the training statistics and the fixed bin structure,
    run the pooled optimizer, the per-block fits, the stability table,
    the Hessian discipline block (or the named residuals with the
    finite-MC profile), assemble the agreement table and the
    conservation report, and return the report dict with its digest (no
    wall-clock: the bytes are a pure function of the stored artifacts,
    the pre-registered seeds, and the committed code)."""
    run_dir = Path(run_dir)
    pre = load_pre_registration(run_dir, smoke=smoke)
    if smoke:
        tables = _load_smoke_tables()
        sealed = None
        r50_reference = None
    else:
        _raw_config, tables = _load_production_tables()
        sealed = _load_sealed_r01(tables)
        r50_reference = r50_reference_payload()
        registered = pre.get("r50_reference") or {}
        _verify_r50_reference(registered, r50_reference)
    tables_by_species = {str(t.species): t for t in tables}
    missing = [s for s in SPECTRUM_SPECIES if s not in tables_by_species]
    if missing:
        _refuse(f"the binning tables are missing the spectrum species: "
                f"{missing}")
    geometry = build_geometry(tables)
    geometry["_run_dir"] = run_dir
    sqrt_s = (_sqrt_s_conversion() if not smoke else SQRT_S)
    blocks = load_blocks(run_dir, smoke=smoke, geometry=geometry)
    if smoke:
        stats = _build_smoke_statistics(tables)
    else:
        stats = build_statistics(sealed, r50_reference)
    structure = build_unit_structure(stats, blocks, geometry)
    geometry["unit_structure"] = {
        key: value for key, value in structure.items()
        if not key.startswith("_")}
    objective = FitObjective(stats, blocks, geometry, sqrt_s)
    bin_structure = objective.build_bin_structure()
    if smoke:
        # the smoke fit's synthetic data: the model at the pre-registered
        # synthetic-truth point (the initial values) with a synthetic 10%
        # published error (the R41 pattern; no sealed data is read)
        truth = objective.evaluate(
            *[OPTIMIZER_SETTINGS["initial_point"][i]
              for i in range(len(FIT_PARAMETERS))])
        _inject_smoke_data(
            stats, {stat.key: truth["pooled_vectors"][i]
                    for i, stat in enumerate(stats)})
    # -- the pooled fit --
    result = _minimize_pooled(objective)
    best_x = [float(v) for v in result.x]
    converged = bool(result.success)
    at_optimum = objective.evaluate(*best_x)
    at_initial = objective.evaluate(
        *[OPTIMIZER_SETTINGS["initial_point"][i]
          for i in range(len(FIT_PARAMETERS))])
    objective.last_fractions = at_optimum.get("fractions")
    # -- the pooled errors, the block fits, the stability table --
    pooled_errors = _pooled_errors(objective, best_x)
    block_fits = _fit_blocks_individually(
        stats, blocks, geometry, sqrt_s)
    stability = _stability_table(best_x, block_fits, pooled_errors)
    boundary = _boundary_parameters(best_x)
    curvature = _pooled_curvature(objective, best_x)
    try:
        eigenvalues = [float(v) for v in np.linalg.eigvalsh(curvature)]
        positive = bool(all(v > 0.0 for v in eigenvalues))
    except np.linalg.LinAlgError:
        # a non-finite curvature entry (a curvature probe outside the
        # sum-rule domain) is the curvature failure itself: the
        # eigenvalues are not defined, the Hessian is not quoted, and
        # the named residual carries the diagnosis
        eigenvalues = None
        positive = False
    minima_agree = bool(stability) and all(
        bool(entry["pass"]) for entry in stability.values())
    quoted = quoted_flag(minima_agree, positive, converged, smoke)
    profiles = None if quoted else _parameter_profiles(objective, best_x)
    # -- the report --
    conservation = _conservation_report(blocks, run_dir, smoke=smoke)
    agreement = [
        _statistic_agreement(
            stat, at_optimum["pooled_vectors"][i],
            at_initial["pooled_vectors"][i], at_optimum["se"][i], objective)
        for i, stat in enumerate(stats)
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "statistics": {
            "tier": ("T2: 8 replicas x 100,000 p+p events = 800,000 "
                     "events (plan section 5.1 line 254, the upper T2 "
                     "bound; the R23/R32/R41 convention)"),
            "events_total": sum(block.n_events for block in blocks),
            "replicas": len(blocks),
        },
        "fit_set": dict(_FIT_SET_STATUS),
        "initial_values": dict(INITIAL_VALUES),
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
        "fitted": {
            **{name: best_x[i] for i, name in enumerate(FIT_PARAMETERS)},
            "nd_fraction": 1.0 - best_x[0] - best_x[1] - best_x[2],
            "fractions": at_optimum.get("fractions"),
        },
        "objective_at_optimum": {
            "minus2lnl_profiled": at_optimum["value"],
            "n_eff_total": at_optimum["n_eff_total"],
            "log_shift": at_optimum["log_shift"],
            "nuisance_columns": list(NUISANCE_COLUMNS),
            "nuisances": at_optimum["eta_hat"],
            "ess": at_optimum["ess"],
            "dropped_bins": len(objective.dropped_bins),
        },
        "objective_at_initial": {
            "minus2lnl_profiled": at_initial["value"],
            "n_eff_total": at_initial["n_eff_total"],
            "log_shift": at_initial["log_shift"],
            "nuisances": at_initial["eta_hat"],
            "ess": at_initial["ess"],
        },
        "block_fits": block_fits,
        "replica_block_stability": stability,
        "boundary_parameters": boundary,
        "hessian": {
            "quoted": quoted,
            "minima_agree": minima_agree,
            "curvature": [[float(v) for v in row] for row in curvature],
            "eigenvalues": eigenvalues,
            "positive_definite": positive,
            "steps": {name: float(HESSIAN_STEPS[name])
                      for name in FIT_PARAMETERS},
            "pooled_errors": {name: pooled_errors[name]
                              for name in FIT_PARAMETERS},
            "note": ("the Hessian is quoted only after the optimizer "
                     "converged, the replica-block minima agree (the "
                     "stability table passes on every parameter), and "
                     "the pooled curvature is positive definite (plan "
                     "line 1871; the R46 pattern); no Hessian-based "
                     "covariance is emitted otherwise"),
        },
        "parameter_profiles": profiles,
        "agreement": agreement,
        "analytic_checks": _analytic_checks(objective, at_initial),
        "bin_structure": bin_structure,
        "dropped_bins": objective.dropped_bins,
        "conservation": conservation,
        "model_uncertainty_carry": MODEL_UNCERTAINTY_CARRY,
        "weight_convention": WEIGHT_CONVENTION_NOTE,
        "sqrt_s_conversion": SQRT_S_CONVERSION_NOTE,
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "named_residual_register": [],
        "no_separation_claim": (
            "this record claims no separation of exchange recoil from "
            "decay transverse dynamics (the R46 outcome, "
            "separation_claimed false); the exchange slopes act on the "
            "exchange kinematics only (plan section 24, line 1866)"),
        "determinism": DETERMINISM_NOTE,
    }
    report["named_residual_register"] = _assemble_named_residuals(report)
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    report["digest"] = digest
    return report


def _build_smoke_statistics(tables: list) -> list:
    """The smoke statistic set with placeholder data (the synthetic data
    is injected after the first truth-point evaluation; the R41
    synthetic-data pattern)."""
    sealed = {
        str(t.species): {"d": [0.0] * int(t.data.size),
                         "v_data": [float(SE_FLOOR)] * int(t.data.size),
                         "xF_mid": [float(x) for x in t.xf_centers[:int(t.data.size)]],
                         "pT_mid": [float(p) for p in t.pt_centers[:int(t.data.size)]]}
        for t in tables
    }
    r50_reference = {
        "gap_fraction": {"value": 0.0, "variance": float(SE_FLOOR)},
        "charged_multiplicity": {
            "bins": [int(n) for n in MULTIPLICITY_BINS],
            "distribution": {str(n): 0.0 for n in MULTIPLICITY_BINS},
            "variance": {str(n): float(SE_FLOOR) for n in MULTIPLICITY_BINS},
        },
    }
    return build_statistics(sealed, r50_reference)


def _inject_smoke_data(stats: list, truth_vectors: dict) -> None:
    """The smoke fit's synthetic data: the model at the pre-registered
    synthetic-truth point (the initial values), with a synthetic 10%
    published error; the R41 pattern."""
    for stat in stats:
        stat.d = np.asarray(truth_vectors[stat.key], dtype=float).copy()
        finite = np.isfinite(stat.d)
        stat.d = np.where(finite, stat.d, 0.0)
        stat.v_data = np.where(finite, (0.1 * np.abs(stat.d)) ** 2,
                               float("inf"))
        stat.v_data = np.maximum(stat.v_data, float(SE_FLOOR))


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
    """Render REPORT.md from the fit report dict (the objective, the
    fitted parameters, the agreement table, the stability table, the
    Hessian discipline outcome, the conservation audit, and the
    bookkeeping)."""
    lines = []
    lines.append("# R51 — topology fraction fit (run report)")
    lines.append("")
    lines.append(f"Run: **{RUN_ID}** (`{RUN_NAME}`), card `{CARD}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "The round's primary optimizer call (plan section 11 run R51, "
        "lines 909–918): the five free parameters (the three free "
        "inelastic class fractions SD_p, SD_t, DD with ND determined by "
        "the sum rule over the inelastic sample, the mass-law exponent "
        "epsilon, and the effective transfer slope B_SD) fit on the "
        "pre-registered training observables (the leading-proton "
        "spectrum, the central-window gap fraction, the charged "
        "multiplicity distribution, and the net-proton distribution "
        "B_transport = p − p-bar), no strange-sector parameters, the "
        "blend ramp start fixed at B. The elastic class is excluded "
        "from the inelastic sample (plan line 824).")
    lines.append("")
    census_note = report["statistics"]
    lines.append(
        f"Sample: {census_note['events_total']} events in "
        f"{census_note['replicas']} replicas (T2, 8 x 100,000); seeds "
        "20300945–20300952; event-index partition 1_100_000_000 + i, "
        "i < 100,000; the per-event class draw at the nominal fractions "
        "with the importance-weighted fit (one generation at the "
        "initial point, exact per-event ratios per trial).")
    lines.append("")
    lines.append("## Optimizer outcome")
    lines.append("")
    opt = report["optimizer"]
    fitted = report["fitted"]
    lines.append(
        f"Algorithm: {opt['settings']['algorithm']}; xatol "
        f"{_fmt(opt['settings']['xatol'])}, fatol "
        f"{_fmt(opt['settings']['fatol'])}, maxiter "
        f"{_fmt(opt['settings']['maxiter'])}, maxfev "
        f"{_fmt(opt['settings']['maxfev'])}.")
    lines.append("")
    lines.append(
        f"Converged: **{opt['converged']}** (\"{opt['message']}\"; "
        f"{opt['nfev']} evaluations, {opt['nit']} iterations); pooled "
        f"-2lnL at the optimum {_fmt(opt['objective_value'], '.6f')}, "
        f"at the initial (R50 nominal) point "
        f"{_fmt(report['objective_at_initial']['minus2lnl_profiled'], '.6f')}.")
    lines.append("")
    lines.append("| parameter | initial (R50 nominal) | fitted | bounds |")
    lines.append("|---|---|---|---|")
    for i, name in enumerate(FIT_PARAMETERS):
        lo, hi = PARAM_BOUNDS[name]
        lines.append(
            f"| {name} | {_fmt(OPTIMIZER_SETTINGS['initial_point'][i])} "
            f"| {_fmt(best_value(report, name))} | [{lo}, {hi}] |")
    lines.append(
        f"| nd_fraction (sum rule) | 0.25 | "
        f"{_fmt(fitted.get('nd_fraction'))} | determined (1 - sum) |")
    lines.append("")
    lines.append("## Training-observable agreement (fit vs R50-pinned "
                 "initial-point model)")
    lines.append("")
    lines.append(
        "| statistic | bins | kept | deviance at fit | deviance at "
        "initial | max |pull| fit | max |pull| initial |")
    lines.append("|---|---|---|---|---|---|---|")
    for block in report["agreement"]:
        lines.append(
            f"| {block['key']} | {block['n_bins']} | {block['n_kept']} "
            f"| {_fmt(block['deviance_contribution_fit'], '.4f')} "
            f"| {_fmt(block['deviance_contribution_initial'], '.4f')} "
            f"| {_fmt(block['max_abs_pull_fit'], '.3f')} "
            f"| {_fmt(block['max_abs_pull_initial'], '.3f')} |")
    lines.append("")
    lines.append(
        "Data sides: the leading-proton and B_transport spectra are "
        "anchored on the sealed R01 published tables; the gap fraction "
        "and the charged-multiplicity distribution are anchored on the "
        "sealed R50 nominal-point references (the sealed substrate "
        "carries no published gap or multiplicity table; the reference "
        "construction at the nominal fractions is recorded in the "
        "pre-registration). The full per-bin vectors are in "
        "`fit_report.json` (data, model at the fit, model at the "
        "initial point, MC standard errors, kept masks).")
    lines.append("")
    stab = report["replica_block_stability"]
    lines.append("## Replica-block stability (acceptance (ii))")
    lines.append("")
    lines.append("| parameter | 8-block spread | pooled error | 20% bound | pass |")
    lines.append("|---|---|---|---|---|")
    for name, entry in stab.items():
        lines.append(
            f"| {name} | {_fmt(entry['spread'], '.6f')} "
            f"| {_fmt(entry['pooled_error'], '.6f')} "
            f"| {_fmt(entry['bound'], '.6f')} "
            f"| {'yes' if entry['pass'] else 'NO'} |")
    lines.append("")
    hess = report["hessian"]
    lines.append("## Hessian discipline (plan line 1871)")
    lines.append("")
    if hess["quoted"]:
        lines.append(
            "The pooled Hessian is **quoted** (the optimizer converged, "
            "the 8 replica-block minima agree, and the pooled curvature "
            "is positive definite).")
    else:
        lines.append(
            "The pooled Hessian is **not quoted** (quoted: false). No "
            "Hessian-based covariance is emitted from a non-converged "
            "or unstable finite-MC objective; the finite-MC profile "
            "(the 9-point per-parameter grid through the best point) "
            "and the named-residual register carry the outcome.")
    lines.append("")
    if report.get("named_residual_register"):
        lines.append("## Named-residual register")
        lines.append("")
        for entry in report["named_residual_register"]:
            lines.append(
                f"- `{entry['kind']}`"
                + (f" ({entry['parameter']})" if entry.get("parameter")
                   else "") + f": {entry['diagnosis']}")
        lines.append("")
    cons = report["conservation"]
    lines.append("## Conservation audit (the full sample)")
    lines.append("")
    lines.append(
        f"Zero B, Q, S violations and zero four-momentum closure "
        f"failures at tolerance 1e-09 on all {cons['events_checked']} "
        f"events: worst closure residual "
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
        "(-2.3959765824349295e-05) are model-uncertainty terms of the "
        "fit (the carry nuisance on the B_transport bins, prior 0.25). "
        "The calculation-engine moment floor (~1e-3 GeV/c) is tracked "
        "on its separate track.")
    lines.append("")
    lines.append("## Bookkeeping")
    lines.append("")
    lines.append(
        "- `optimizer_invoked`: true; `optimizer_uses`: 1 (the round's "
        "primary optimizer call; R53 is the only other optimizer call "
        "of the round and is conditional).")
    lines.append(
        "- The frozen simplified cascade at the sealed R45 point is "
        "consumed without refit (the R46 no_refit clause); the blend "
        "ramp start stays fixed at B = 2.680194110727528 GeV; no Round "
        "5 record claims a separation of exchange recoil from decay "
        "transverse dynamics.")
    lines.append(
        "- The nd_multi class is excluded from the R51 draw (unretained; "
        "it enters the round only through R53, plan line 926).")
    lines.append(
        "- Determinism: the per-event hash set of every replica record "
        "carries the bit-level identity evidence; the clean-"
        "environment validation re-runs the fit evaluation and the "
        "probe replica (replica 0) from a clean interpreter before the "
        "seal.")
    lines.append("")
    lines.append(f"Report digest: `{report['digest']}`.")
    lines.append("")
    (run_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


def best_value(report: dict, name: str) -> float:
    """The fitted value of one fit parameter from the report."""
    return report["optimizer"]["best_point"][name]


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block; the verbatim template)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """--gate: write gate5_r51.json (evidence only; the written retain /
    revise / reject decision on the Gate 5 clauses is the round's Gate 5
    follow-up obligation), then stop.  Requires the fit report."""
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
    fitted = report["fitted"]
    stability = report["replica_block_stability"]
    gate = {
        "schema": "na49_mixing.staged_r51_topology_fraction_fit.gate/1",
        "gate": "gate 5 (plan section 11, line 938)",
        "run": "R51",
        "card": CARD,
        "clause": "fractions required by topology-sensitive data",
        "fit_set": list(FIT_PARAMETERS),
        "free_parameters": 5,
        "constraint": "inelastic fractions sum to 1 (3 free)",
        "excluded": {
            "blend_ramp_start": "round 4 obligation 4 (fixed at B)",
            "strange_sector": "plan line 918",
            "elastic": "plan line 824",
        },
        "initial_values": {
            "fractions": [0.25, 0.25, 0.25, 0.25],
            "epsilon": 0.0,
            "b_sd": 8.504140762083612,
            "source": "sealed R50 nominal point",
        },
        "observables": [
            "leading_proton_spectrum", "gap_fraction_abs_y_lt_0.5",
            "charged_multiplicity", "b_transport",
        ],
        "events_checked": conservation["events_checked"],
        "conservation": conservation,
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
        },
        "fitted": {
            "fractions": [
                fitted["fractions"]["sd_projectile"],
                fitted["fractions"]["sd_target"],
                fitted["fractions"]["dd"],
                fitted["fractions"]["nd"],
            ],
            "epsilon": fitted["epsilon"],
            "b_sd": fitted["b_sd"],
        },
        "replica_block_stability": {
            name: {
                "per_block": entry["per_block"],
                "spread": entry["spread"],
                "pooled_error": entry["pooled_error"],
                "bound": entry["bound"],
                "pass": entry["pass"],
            }
            for name, entry in stability.items()
        },
        "hessian": {
            "quoted": report["hessian"]["quoted"],
            **({name: report["hessian"]["pooled_errors"][name]
                for name in FIT_PARAMETERS}
               if report["hessian"]["quoted"] else {}),
        },
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
    print(f"[r51] gate record written: {run_dir / GATE_RECORD_NAME}",
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
    dir refuses — the card-19 seal-ordering discipline)."""
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
    rerun_report = run_fit_machinery(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # the probe replica re-run in memory (the determinism evidence of
    # acceptance (v)): the pre-registered probe replica (replica 0)
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
        "schema": ("na49_mixing.staged_r51_topology_fraction_fit."
                   "clean_env/1"),
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            ("cd dev/mixing2006 && python -m pytest "
             "tests/test_staged_r51_topology_fraction_fit.py -q"),
            ("python -m pytest tests/test_legacy_v1_freeze.py "
             "-W error::UserWarning -q"),
            (f"python helpers/benchmarking/"
             f"staged_r51_topology_fraction_fit.py "
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
        f"[r51] clean validation written: {clean_path} (report hash "
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
        "schema": ("na49_mixing.staged_r51_topology_fraction_fit."
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
    print(f"[r51] sealed {run_dir} ({n_entries} entries)", flush=True)
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
    pre-registration, one mixed replica x 2,000 events at the smoke seed
    20312925 on the event-index partition 1_600_000_000 + i (i < 2,000),
    and the smoke fit (synthetic data; no SPS table and no sealed
    archive read), into an explicit --run-dir (the default production
    run dir is refused).  Two identical smoke invocations produce
    byte-identical artifacts (no wall-clock and no git state in the
    smoke artifacts)."""
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
        "schema": ("na49_mixing.staged_r51_topology_fraction_fit."
                   "pre_registration/1"),
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke",
        "objective": OBJECTIVE_NOTE,
        "statistics": {
            "note": (
                "the smoke statistics: one mixed replica x 2,000 events "
                "at the smoke seed 20312925; the smoke exercises "
                "pre-registration, generation (through the card-9 "
                "draw_topology at the nominal fractions), recording, "
                "likelihood, and fit end to end; the production "
                "statistics are the 8 pre-registered replicas of "
                "100,000 events"),
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
                "raw replica seed = 20260925 + 40000 + 20 + k = 20300945 "
                f"+ k per replica k < {N_REPLICAS}"),
            "disjointness": disjointness_report_for_smoke(),
        },
        "fit_set": dict(_FIT_SET_STATUS),
        "initial_values": dict(INITIAL_VALUES),
        "optimizer": dict(OPTIMIZER_SETTINGS),
        "observables": OBSERVABLE_NOTE,
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "weight_convention": WEIGHT_CONVENTION_NOTE,
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
            "synthetic (the model at the pre-registered synthetic-truth "
            "point with a synthetic 10% published error; no SPS table "
            "and no sealed archive is read)"),
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
            "mixed_draw_note": (
                "the smoke exercises the card-9 per-event class draw on "
                "the closed topology_selection substream at the nominal "
                "fractions (the R51 fraction-sampling surface)"),
        },
        "generation_fractions": GENERATION_FRACTIONS_NOTE,
        "class_census": dict(census),
        "conservation": conservation,
        "event_hash_set_sha256": hash_set,
        "events": records,
        "reused": False,
    }
    _write_json(run_dir / SMOKE_REPLICA_NAME, replica_record)
    report = run_fit_machinery(run_dir, smoke=True)
    _write_json(run_dir / ("smoke_" + REPORT_NAME), report)
    print(
        f"[r51] smoke complete: {len(records)} events, census {census}, "
        f"worst closure {conservation['worst_closure_relative']:.3e}",
        flush=True,
    )
    return {"events": len(records), "census": census}


def disjointness_report_for_smoke() -> dict:
    """The smoke pre-registration's disjointness record (identical to
    the production check; the smoke seed and partition are additionally
    disjoint from the R51 production block and partition)."""
    report = seed_disjointness_report()
    if SMOKE_SEED in set(REPLICA_SEEDS):
        _refuse("the smoke seed collides with the R51 production block")
    smoke_lo = SMOKE_EVENT_INDEX_BASE
    smoke_hi = SMOKE_EVENT_INDEX_BASE + SMOKE_EVENTS
    if (EVENT_INDEX_BASE < smoke_hi and smoke_lo < EVENT_INDEX_BASE
            + EVENTS_PER_REPLICA):
        _refuse("the smoke event-index partition overlaps the R51 "
                "production partition")
    report["smoke_raw_seed"] = SMOKE_SEED
    report["smoke_event_index_range"] = [smoke_lo, smoke_hi]
    report["smoke_disjoint_from_production"] = True
    return report


# ---------------------------------------------------------------------------
# The pre-registration notes
# ---------------------------------------------------------------------------
OBJECTIVE_NOTE = (
    "plan section 11 run R51 (lines 909-918): 'Fit class fractions "
    "using: leading-proton spectrum; diffractive or gap fractions; "
    "charged multiplicity distribution; net-proton distribution.  Do "
    "not include strange-sector parameters.'  The pre-registered "
    "instantiation (index items 3, 8, and 10): the five free "
    "parameters (the three free inelastic class fractions SD_p, SD_t, "
    "DD with ND determined by the sum rule over the inelastic sample; "
    "the elastic class excluded per plan line 824) plus the two "
    "effective mass-law parameters (epsilon and B_SD); the round's "
    "primary optimizer call.  The fit is the importance-weighted "
    "profiled -2lnL on the pre-registered training observables over "
    "the 8 generated replica blocks (the R41 pattern: one generation "
    "at the initial point, exact per-event path ratios per trial).")

OBSERVABLE_NOTE = {
    "instantiation_table": (
        "the plan's four named observables map onto the sealed data "
        "surfaces as follows (the pre-registered instantiation)"),
    "leading_proton_spectrum": (
        "the proton species' published-cell spectrum (the R01 proton "
        "table, d and v_data; the model side is the weighted per-cell "
        "proton count in the data units m = sums / (pi sqrt(s) N_eff "
        "dxF dpT), the R01 conversion; under the topology model every "
        "final-state proton originates in a baryonic leading line, so "
        "the proton spectrum on the published cells is the "
        "leading-proton structure's observable; plan line 913)"),
    "b_transport_spectrum": (
        "the net-proton combination B_transport = p - p-bar (plan "
        "section 7.3 line 431) over the aligned R01 proton/antiproton "
        "cells (d = d_p - d_pbar, v_data = v_p + v_pbar); the model "
        "side is m_p - m_pbar per aligned cell; the standing "
        "model-uncertainty carry applies to this observable (the "
        "correlated multiplicative nuisance on the sealed 32 "
        "persisting-bin cells, prior 0.25; plan line 916)"),
    "gap_fraction_abs_y_lt_0.5": (
        "the fraction of inelastic events with no hadronic final-state "
        "particle in the central window |y^CM| < 0.5 (the event's "
        "central multiplicity zero; plan line 914 'diffractive or gap "
        "fractions'); the data side is the sealed R50 nominal-point "
        "reference (the 0.25-each pooled mixture of the four inelastic "
        "class pools' realized central-gap fractions, its binomial MC "
        "variance as V_data; the sealed substrate carries no published "
        "gap-fraction table, so the R50-pinned reference is the "
        "pre-registered data anchor, and its construction at the "
        "nominal fractions is recorded as the reference's construction "
        "note)"),
    "charged_multiplicity_distribution": (
        "the charged-multiplicity distribution over the pre-registered "
        "bins n_ch = 2..14 (the count of final-state records with "
        "nonzero charge Q; the below-threshold baryonic system carries "
        "Q = 1 and counts; plan line 915); the data side is the sealed "
        "R50 nominal-point reference (the 0.25-each pooled mixture per "
        "bin, its per-class binomial variance, diagonal; the sealed "
        "substrate carries no published multiplicity table, so the "
        "R50-pinned reference is the pre-registered data anchor)"),
    "excluded": (
        "the pion and kaon spectra are R52's held-out observables, not "
        "R51 training statistics (the R52 contract, index item 4); no "
        "strange-sector observable enters the fit (plan line 918)"),
}

RECORDING_CONVENTION_NOTE = (
    "dense-lattice accumulation per species on the published evaluation "
    "lattice, published rows selected with the pick map "
    "bin_index_pt * n_x + bin_index_xf, hole cells dropped (the "
    "corrected card-17 relitigation convention, the inherited state of "
    "the index card); the per-event records store the compact "
    "fit-relevant payload (the class label, the observables block, the "
    "excitation masses, the transfers, the remnant masses, the "
    "spectrum-species (x_F, p_T) pairs, the conservation block, and the "
    "per-event hash) instead of the R50 full-final-state records (the "
    "R50 template archive remains the full-final-state evidence base; "
    "this recording scope is the pre-registered R51 record convention)")

STATISTICS_NOTE = (
    "T2 (plan section 5.1 line 254, the upper T2 bound; the R23/R32/R41 "
    "convention): 8 replicas x 100,000 p+p events = 800,000 events, "
    "mixed inelastic draws at the initial point (the per-event class "
    "draw at the nominal fractions; the elastic fraction exactly 0.0)")

LIKELIHOOD_NOTE = (
    "-2lnL = (d - m)' V^-1 (d - m) + eta' P^-1 eta, with V = V_data + "
    "V_mc (reported separately), the published binning preserved, and "
    "the nuisance vector profiled exactly per trial: one multiplicative "
    "normalization nuisance per spectrum training block (prior 0.35, "
    "the R41 standing convention) plus the model-uncertainty carry as "
    "one correlated multiplicative nuisance on the b_transport bins "
    "whose proton or antiproton cell is in the sealed 32 "
    "persisting-bin set (prior 0.25).  The per-bin MC error is the "
    "between-replica standard error (std ddof=1 over the 8 replica "
    "blocks, divided by sqrt(8)), floored at the engine's "
    "DETERMINISTIC_SE_FLOOR; a bin structurally empty at the generation "
    "point carries an infinite sigma and drops out of the objective "
    "(the R11 convention; the dropped bins are listed).  The gap-"
    "fraction and multiplicity-distribution statistics are normalized "
    "and carry no normalization nuisance.  Data and MC contributions "
    "are reported separately (the V_data and V_mc blocks of the "
    "agreement table)")

HESSIAN_DISCIPLINE_NOTE = (
    "no Hessian errors are quoted from a noisy or non-converged "
    "finite-MC objective (plan line 1871; the R46 pattern).  The pooled "
    "Hessian is quoted only after the optimizer converged on the "
    "five-parameter fit set, the replica-block minima agree (the "
    "acceptance (ii) stability table passes on every parameter: the "
    "8-block spread below 20% of the pooled error), and the pooled "
    "curvature is positive definite; stability is checked across all 8 "
    "replica blocks.  A boundary minimum, disagreeing block minima, a "
    "curvature failure, a convergence failure, or a weight-degeneracy "
    "ESS are recorded as named residuals (the R46 named-residual forms "
    "boundary_minimum, replica_block_minima_disagree, "
    "block_curvature_failure, plus optimizer_non_convergence and "
    "weight_degeneracy) and the finite-MC profile (the pre-registered "
    "9-point per-parameter grid through the best point) is recorded; "
    "no Hessian-based covariance is emitted")

ACCEPTANCE_NOTE = (
    "pre-registered acceptance (i)-(ix) of the card: (i) fit "
    "convergence: the optimizer converges on the five-parameter fit "
    "set and the fitted values and the per-replica-block table are "
    "recorded in fit_report.json (a non-convergence is the named "
    "residual with the finite-MC profile and no Hessian covariance, "
    "the card's fallback); (ii) replica-block stability: the fitted "
    "parameters stable across all 8 independent replica blocks, the "
    "per-block spread below 20% of the pooled error per parameter; a "
    "failing parameter is a named finding feeding the Gate 5 clause 1 "
    "unconstrained clause; (iii) fit-set restriction: exactly five "
    "free parameters, no strange-sector parameters, the blend ramp "
    "start stays fixed at B, no exchange/topology/flavor/branching/"
    "catalog parameter is fit; (iv) conservation: 800,000 events, "
    "zero B, Q, S violations and zero four-momentum closure failures "
    "at tolerance 1e-09, the worst closure residual recorded in GeV; "
    "(v) determinism: the clean-environment re-run reproduces the "
    "per-replica report hashes; (vi) Hessian discipline: the Hessian "
    "quoted only after replica-minimum agreement and stable curvature, "
    "or quoted: false with the named residuals; (vii) the primary "
    "optimizer call: optimizer_invoked true, optimizer_uses 1; (viii) "
    "exactly one clean-environment validation pass executes before the "
    "seal and the seal is intact afterwards; (ix) the driver and test "
    "file land (both trees byte-identical), the freeze re-pin is "
    "recorded, and the full nested suite is green (the land gate's "
    "obligation)")

GENERATION_NOTE = (
    "gateway-primary generation (host cl, workers c0801-c0809 x 32 task "
    "slots; one job per replica, --tasks 1, single math thread "
    "(NA49_CLUSTER_GW_TASKS forces thread counts to 1), submission "
    "paced by the capacity check, no local process pool inside the "
    "chunk command, chunk outputs appended only to the new run dir); "
    "the driver's local fallback is one single-threaded process per "
    "--replica invocation (a completed record that matches the "
    "pre-registered identity is reused unchanged)")


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
        initial_assertion = assert_initial_values_against_r50()
        r50_reference = r50_reference_payload()
        _raw_config, tables = _load_production_tables()
        sealed = _load_sealed_r01(tables)
        geometry = build_geometry(tables)
        r01_summary = {
            "archive": R01_ARCHIVE,
            "file_digests": {
                name: _sha256_file(MONO_ROOT / R01_ARCHIVE / name)
                for name in (R01_RESIDUALS_NAME, R01_COVARIANCE_NAME,
                             R01_PREDICTIONS_NAME)
            },
            "spectrum_species": list(SPECTRUM_SPECIES),
            "published_rows": {
                str(t.species): int(t.data.size) for t in tables
                if str(t.species) in SPECTRUM_SPECIES},
            "lattice_geometry": {
                species: {"n_x": geometry["n_x"][species],
                          "dense_size": geometry["dense_size"][species],
                          "pick_rows": int(geometry["pick"][species].size)}
                for species in SPECTRUM_SPECIES},
            "aligned_cells_b_transport": len(_aligned_cells(sealed)),
            "carry_bins": len(_load_carry_bins()),
            "cross_check": "the sealed d values match the loaded SPS "
                           "tables bit for bit (the R11 discipline)",
        }
    else:
        sealed_inputs = None
        initial_assertion = None
        r50_reference = None
        r01_summary = None
    payload = {
        "schema": ("na49_mixing.staged_r51_topology_fraction_fit."
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
        "fit_set": dict(_FIT_SET_STATUS),
        "initial_values": {
            "values": dict(INITIAL_VALUES),
            "assertion_against_sealed_r50": initial_assertion,
        },
        "observables": OBSERVABLE_NOTE,
        "recording_convention": RECORDING_CONVENTION_NOTE,
        "statistics": {
            "tier": STATISTICS_NOTE,
            "replicas": N_REPLICAS,
            "events_per_replica": EVENTS_PER_REPLICA,
            "events_total": N_EVENTS_TOTAL,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": (
                "raw replica seed = 20260925 + 40000 + 20 + k = 20300945 "
                f"+ k per replica k < {N_REPLICAS} (index item 9; the "
                "block 20300945-20300952 is disjoint from every "
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
            },
            "substream_collision_rule": SUBSTREAM_COLLISION_RULE,
        },
        "likelihood": LIKELIHOOD_NOTE,
        "optimizer": dict(OPTIMIZER_SETTINGS),
        "hessian_discipline": HESSIAN_DISCIPLINE_NOTE,
        "weight_convention": WEIGHT_CONVENTION_NOTE,
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
        "r50_reference": r50_reference,
        "r01_data": r01_summary,
        "multiplicity_bins": [int(n) for n in MULTIPLICITY_BINS],
        "generation": GENERATION_NOTE,
        "card_chain": list(CARD_CHAIN),
        "acceptance": ACCEPTANCE_NOTE,
    }
    _write_json(pre_path, payload)
    if not smoke:
        provenance = {
            "schema": ("na49_mixing.staged_r51_topology_fraction_fit."
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
            "r01_data": r01_summary,
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
        print(f"[r51] provenance written: "
              f"{run_dir / PROVENANCE_NAME}", flush=True)
    print(f"[r51] pre-registration written: {pre_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# The collision note (the R32 p+p configuration; the R50 note adapted)
# ---------------------------------------------------------------------------
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
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R51 driver: the topology fraction fit "
                    "(TASK-20260925-r5-run-r51-topology-fraction-fit)")
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
        help="the fit evaluation (fit_report.json and REPORT.md), then "
             "stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate5_r51.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="re-run the fit evaluation from a clean interpreter and the "
             "probe replica, then stop (exactly one pass, before the "
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
        print(f"[r51] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(run_dir, replicas=args.replica)
        print(f"[r51] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        report = run_fit_machinery(run_dir, smoke=False)
        _write_json(run_dir / REPORT_NAME, report)
        _write_report_md(run_dir, report)
        print(
            f"[r51] evaluation written: {run_dir / REPORT_NAME} "
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
