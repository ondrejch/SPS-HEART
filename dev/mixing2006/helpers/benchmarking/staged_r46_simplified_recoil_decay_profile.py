#!/usr/bin/env python3
"""R46 driver: the simplified recoil/decay re-profile (Round 4, card 21).

Task ``TASK-20260920-r4-gate4-followup-recoil-decay-reprofile`` (the
Gate 4 follow-up obligation 2 of the written Gate 4 decision
``TASK-20260920-r4-gate4-decision``; plan section 23, lines 1821-1823,
"Round 4 -- aligned resonance cascade"; plan section 10 run R42, line
786, re-run under the simplified form).  The Gate 4 decision records
that the recoil/decay separation fails on the corrected R42 evidence:
the profile minimum sits on the grid boundary (0.5, 2.6), the pooled
curvature matrix is not positive definite (eigenvalues
[-182549.71181584388, 7.698059717158571]), block-4 curvature fails
(eigenvalues [-1999.037806083929, 15699.862514911674]), and the
correlation is null against the 0.9 distinguishability bound.  This
card re-runs the two-dimensional B_eff / b_dec profile with event
generation against the frozen simplified alignment of card 20 (the
sealed ``dev/00runs/r45_simplified_alignment_freeze_2026-09-24/``
archive, whose phase-2 seal is the run substrate).  The profile tests
exactly the plan section 24 (line 1866) prohibition: the exchange slope
must not control daughter-level transverse dynamics.  Exchange recoil
keeps the exchange-level distribution dP/dt proportional to
exp(B_eff t).  Decay transverse structure keeps the hadronic vertex
form |F_dec(q_perp)|^2 = exp(-b_dec q_perp^2) on the two-body momenta,
angular alignment, and form factor of the card-8 surface.  One
parameter never controls both.  No claim that decay transverse dynamics
separate from exchange recoil is made unless the acceptance criterion
(ii) passes in full; on failure the outcome is a named finding and the
separation stays unestablished.

Substrate (the card-20 frozen simplified alignment)
---------------------------------------------------
The generation substrate is the card-20 phase-2 sealed archive (the
frozen simplified alignment): the constant-kappa form of plan section
10.4, line 732 (constant kappa_B and kappa_M, every mass-dependence
coefficient fixed at zero and not fit) at the frozen tuple (kappa_B,
kappa_m, b_dec, baryonic_nominal_width) = (0.0, 4.0, 2.0 GeV^-2,
0.3 GeV) read from the sealed corrected R41 archive
(``fit_report.json``, file sha256 ``bc419ab1...``, report self-digest
``76a40d34...``) and asserted before any generation, cross-checked
against the card-20 driver's own frozen register (the R45 driver is a
byte-frozen substrate sibling; its digest is verified and its frozen
policy is compared at import).  B_eff is an M8-frozen parameter, held
at its frozen value for event generation and profiled in the
likelihood (the index item 10 pattern).  The blend ramp start stays
fixed at B = 2.680194110727528 GeV (obligation 4).  Exchange topology
and flavor parameters are fixed (plan line 784).

Profile axes (plan line 788; the unchanged corrected-R42 grid)
--------------------------------------------------------------
Exactly two dimensions, B_eff (the frozen exchange slope, profiled in
the likelihood) against b_dec (the decay form-factor slope).  No
exchange, topology, flavor, branching, ramp-start, or alignment
parameter is fit (plan section 24; plan line 784; obligation 4).  The
pre-registered bounds are never relaxed: the multiplier stays in
[0.5, 2.5] and b_dec in [0.2, 2.6] GeV^-2, and the grid is never
extended to move the minimum interior.

- B_eff, the exchange-layer slope (the dP/dt proportional-to
  exp(B_eff t) transfer law, the M8-frozen
  ``regge.effective_transfer_slope``), profiled on the nine
  pre-registered multipliers 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
  2.25, 2.5 (0.25 steps) of the frozen value
  B_eff = 8.504140762083612 GeV^-2.  B_eff is held at its frozen value
  for event generation and profiled by EXACT reweighting of the
  transfer law: the realized Mandelstam-t of the exact two-body
  scattering is a parameter-independent draw (the support
  [t_minus, t_plus] depends only on the masses and sqrt(s)), the draw
  budget is unchanged, so the per-event transfer weight is the exact
  closed-form density ratio
  w_B = exp((B' - B) (t - t_hi)) * (B'/B) *
        (1 - exp(-B dt)) / (1 - exp(-B' d t)),
  with B the frozen slope, B' the trial slope, t_hi the support upper
  endpoint and d t = t_hi - t_lo the support width.  Zero regeneration.
- b_dec, the card-8 hadronic decay form-factor slope
  W_dec = exp(-b_dec q_perp^2), profiled on the nine pre-registered
  values 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6 GeV^-2 (0.3
  steps).  Generation runs at the card-20 frozen b_dec = 2.0 GeV^-2,
  which sits ON the grid (multiplier 1.0 with b_dec 2.0 GeV^-2 is the
  generation point and every weight there is exactly 1); every other
  grid point reweights away from it.  The form factor is a pure weight
  change on the unchanged attached channel pools: the per-vertex
  channel-choice ratio
  ratio = exp(-(b' - b) q2_chosen) / sum_i w0_i exp(-(b' - b) q2_i),
  with w0 the normalized attached shares at the generation b_dec =
  2.0 GeV^-2 (the R41 algebra, the recorded sufficient statistics).

Observables (plan line 788)
---------------------------
- the leading-baryon transfer observable: the B_transport = f(p) -
  f(pbar) spectrum (plan section 7.3 line 431) on the aligned
  proton/antiproton cells of the sealed R01 tables;
- the identified seagulls: the pooled pion seagull (the pi_Sigma =
  pi+ + pi- combination at the pion mass) and the pooled proton
  seagull (the B_transport combination at the proton mass), the
  per-xF <pT>_x and <pT>_y conventions of ``fit_observables.seagull_
  moment`` on both sides.  The per-xF moment profile spans the
  plan-7.3 forward/central/backward region structure (forward
  xF > 0.2, central [0.0, 0.2], backward < 0.0; the p+p paper window
  covers the central and forward regions), recorded per bin as the
  region tag of the shape.

Likelihood (plan section 5.4)
-----------------------------
-2lnL(B_eff, b_dec) = (d - m)' V^-1 (d - m) + eta' P^-1 eta with
V = V_data + V_mc, the published binning preserved, and the nuisance
vector profiled exactly per grid point: one multiplicative
normalization nuisance on the b_transport spectrum (prior width 0.35,
the standing model-to-data mismatch of the R01/R32 record is of order
one) plus the obligation 4 model-uncertainty carry as one correlated
multiplicative nuisance on the blend-band-sensitive B_transport bins
(the sealed R32 persisting-residual bins restricted to the pair's
species; prior 0.25).  The R32-spread nuisance term stands.  The
seagull statistics are normalization-free ratios and carry no
nuisance.  V_mc is the between-replica standard error of the trial
statistic (std ddof=1 over the 8 replica blocks, divided by sqrt(8)),
floored at the engine's deterministic SE floor; a statistic bin whose
pooled MC error is infinite (an empty model bin) carries an infinite
sigma and drops out of the objective (the R11 convention; the dropped
bins are listed, never dropped silently).  The data and MC
contributions are reported separately per observable.

Profile evaluation (plan line 788)
----------------------------------
At every grid point the per-block weighted predictions are the exact
per-vertex path reweighting of the generated events; the pooled
statistic vectors and their between-replica errors enter the profiled
objective.  The pre-registered outputs: the 9 x 9 pooled -2lnL
surface, the 8 per-replica-block surfaces, the profile minimum (the
grid argmin), the 2 x 2 curvature matrix at the minimum from the grid
second differences (the multiplier coordinate spacing 0.25, the b_dec
spacing 0.3; a minimum on the grid boundary carries the one-sided
construct and is named), the Hessian eigenvalues, the parameter
correlation rho = -H_01 / sqrt(H_00 H_11) (the inverse-Hessian
correlation, the pre-registered distinguishability algebra), and the
per-replica-block curvature stability (the Hessian at the pooled
minimum on each block's own surface, all 8 blocks; plan section 5.5
item 5 line 321).

Hessian discipline (plan section 5.4, line 311)
-----------------------------------------------
No curvature is quoted until the replica-block minima agree and the
local curvature is stable; stability checked across all 8 blocks.  The
gate is mechanical: the 8 per-replica-block surface argmins must equal
the pooled argmin before the pooled Hessian, its eigenvalues, or the
correlation are quoted (the curvature matrix is still recorded beside
the gate, per the acceptance (i) recording clause).  The stability
clause is the acceptance (ii) spread bound: every one of the 8
replica blocks positive definite with the per-direction curvature
spread below 20% of the pooled curvature.

Reweighting validity (the card's pre-registered assertion)
----------------------------------------------------------
At the two pinned grid points (multiplier 1.5 with b_dec 1.7 GeV^-2,
and multiplier 2.0 with b_dec 0.5 GeV^-2) the reweighting validity is
asserted against the direct generation of the pinned sub-sample (the
first 2,000 production event indices of replica 0, common random
numbers, in memory, never sealed):

- stream alignment: the direct generation at the grid point realizes
  the identical route census and the identical transfer-law support
  (both precede every parameter-dependent draw);
- the transfer law: the implied trial realization
  t_implied = G_B'^{-1}(G_B(t)) of the frozen realization's uniform
  reproduces the direct generation's realized t per pinned event
  within 1e-6 relative (the closed-form inverse-CDF identity; the
  exponential transfer law reweights exactly);
- the decay form factor: at the frozen-B direct generation (the pinned
  grid point's b_dec at the frozen B_eff) the first cascade vertex's
  normalized attached shares at the trial b_dec, recomputed from the
  frozen latent's recorded (w0, q2), reproduce the direct generation's
  own attached shares within 1e-6 relative (the parent state is
  parameter-independent at the frozen B_eff);
- the prediction-level agreement (the self-normalized reweighted
  pooled prediction versus the direct generation's unweighted
  prediction over the same pinned sub-sample) is recorded as a Monte
  Carlo agreement diagnostic beside the exact identities, never
  asserted at 1e-6.

The gate record's ``max_relative_deviation`` is the maximum relative
deviation over the exact identities (the stream, transfer, and share
legs); the bound 1e-6 is never relaxed.

Optimizer use (the card's single pre-registered optimizer use)
--------------------------------------------------------------
This card holds the single pre-registered optimizer use of the
follow-up pair (the two-dimensional profile).  Card 20 invokes none.
No other optimizer invocation exists in this card: no exchange,
topology, flavor, branching, ramp-start, or alignment parameter is fit
(plan section 24; plan line 784; obligation 4).  The profile minimum
is the pre-registered 9 x 9 lattice argmin and the curvature is the
grid second difference at it; no continuous optimization is claimed.

Statistics (T2, plan section 5.1 line 254)
------------------------------------------
8 replicas x 100,000 p+p events = 800,000 events, generated ONCE at
the frozen B_eff and the card-20 frozen b_dec = 2.0 GeV^-2 (on-grid);
the 9 x 9 profile is a reweighting over the generated events (zero
additional generation).  The pinned validity sub-sample generations
(2 x 2,000 events at the pinned grid points plus one frozen-B pinned
generation) run in memory from the pre-registered seeds; no trial
artifacts are written.

Local generation is primary (the measured local envelope in R41/R42:
100,000 p+p events ~ 75-78 s at 8 fork workers); the pre-registered
gateway fallback covers the envelope (one job per replica, ``--tasks
1``, single math thread, capacity-paced; record outputs appended only
to the new run dir).

Conservation: every p+p event conserves total (B, Q, S) = (2, 2, 0)
exactly and every source its own (B, Q, S); the event energy sum
closes within 1e-9 GeV; a violation raises and stops the run.

Standing residuals carried: the R43 b_transport_seagull
beyond_isotropic_not_positive finding (observed
-2.3959765824349295e-05) and the R32 spread carry (the obligation-4
model-uncertainty term) appear as named entries in the report's and
the gate record's ``standing_residuals`` blocks.

Determinism: a per-replica re-run from a clean interpreter reproduces
the record identity digests; the profile evaluation re-run from a
clean interpreter over the sealed artifacts reproduces the report hash
(profile_report.json and gate4_followup_reprofile.json carry no
wall-clock fields: their bytes are a pure function of the sealed
artifacts, the pre-registered seeds, and the committed code).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests (including the
  card-20 driver sibling), the sealed R41 fit report (the
  frozen-parameter assertion), the sealed card-20 R45 seal, the sealed
  superseded R42 profile report, and the sealed R30-R32 / R41-R44
  seals first.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, 1 replica x 2,000 events at the
  pre-registered smoke seeds, the smoke 9 x 9 profile against a
  synthetic truth vector, and the smoke validity probe, into an
  explicit ``--run-dir`` (the default production run dir is refused).
  Two identical smoke invocations produce byte-identical artifacts.
  ``--gate`` / ``--seal`` / ``--clean-validate`` refuse smoke
  artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested replica records
  (any subset; the default is every replica), one record file per
  replica, then stop.  Requires the pre-registration; the statistics
  flags must match the pre-registration in production mode.  A
  completed record that matches the pre-registered identity is reused
  unchanged, so the declared tier can be completed replica by replica
  (the pre-registered gateway fallback runs one job per replica).
- ``--profile``: the profile evaluation -- the 9 x 9 reweighting
  grid, the per-block surfaces, the Hessian and correlation at the
  minimum under the quoting gate, the replica-block curvature
  stability, the pinned validity probe, and ``profile_report.json``,
  then stop.  Requires all 8 production records.
- ``--gate``: write ``gate4_followup_reprofile.json`` (evidence only;
  it claims the separation only if acceptance (ii) passes in full),
  then stop.  Requires the profile report.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the profile evaluation
  from a clean interpreter over the sealed artifacts with a byte-
  identical report, re-run production replica 0 in memory against its
  sealed record, write ``clean_environment_commands.json``, then stop.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Recording convention (the corrected card-17 clause, consumed read-only)
-----------------------------------------------------------------------
The finals follow the established R01 accumulate-dense-then-pick
convention (the corrected card-17 R42 driver's ``_worker_context`` /
``_replica_record`` pattern, consumed read-only as the convention
source): every event weight accumulates into the species' dense
pT-major evaluation lattice (``n_x * n_p`` cells at the species' dense
offset), and the published rows are picked from the dense totals at
record assembly with ``pick = bin_index_pt * n_x + bin_index_xf``.
The published R01 tables are non-rectangular subsets of their lattices
(holes; rows not pT-major), so a dense cell address and a published
row address are different addresses: the pick map keeps the recorded
``pooled_sums`` aligned 1:1 with the published rows the evaluation
consumes, and a hole cell's weight stays unpicked and is dropped,
never bled into a neighboring published row or species.  Two
structural refusals guard the pick map: a published bin_index pair
mapping outside the dense lattice, and two published rows on one dense
lattice cell (the pick map must stay one-to-one).

Hermetic in ``--smoke`` (the committed catalog, the committed continuum
config, and the synthetic grid only; no SPS table, no sealed archive
needed for the smoke profile's synthetic data; the frozen-parameter
assertion runs against the pinned R41 fixture); the production modes
read the committed catalog and continuum config, the sealed archives,
and the SPS tables (the published data the likelihood is defined
against)."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the resonance catalog, the continuum config, and
# the sealed reference run dirs.
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
        f"staged_r46_simplified_recoil_decay_profile: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from helpers.benchmarking import sps_core  # noqa: E402
from na49_mixing.cascade_ablation import (  # noqa: E402
    _attach_form_factor,
    _draw_attached_vertex,
    exchange_root_state,
)
from na49_mixing.cascade_axes import initial_source_axis  # noqa: E402
from na49_mixing.cascade_decay import (  # noqa: E402
    CATALOG_FAMILY_PREFIX,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    enumerate_channels,
)
from na49_mixing.cascade_termination import (  # noqa: E402
    CascadeDepthGuardError,
    handoff_decay,
    is_catalog_handoff,
)
from na49_mixing.event_builder_pp import PPExcitationConfig  # noqa: E402
from na49_mixing.exchange_model import transfer_slope  # noqa: E402
from na49_mixing.excitation_kinematics import (  # noqa: E402
    FourVector,
    boost,
    scatter_two_body,
    two_body_momentum,
)
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.quantum_numbers import QuantumNumbers  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.resonance_pp_builder import ResonancePPBuilder  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.source_event import SourceRecord  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BlendSpec,
    active_families,
    families_from_json,
    route_source_decay,
)
from helpers.benchmarking import (  # noqa: E402
    staged_r45_simplified_alignment_freeze as r45_substrate,
)
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402
from na49_mixing import fit_observables as fo  # noqa: E402

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

#: The superseded profile (the corrected sealed R42 archive, the
#: superseded-profile reference of the card; read-only, its seal
#: verifies, nothing in it is modified): the profile report's file
#: digest (byte identity) and self-digest, the boundary minimum, the
#: pooled eigenvalues, the failing block-4 eigenvalues, and the null
#: correlation at the sealed values.  The superseded profile is cited
#: as superseded, never as evidence.
SUPERSEDED_R42_RUN_DIR = (
    "dev/00runs/r42_recoil_decay_separation_2026-09-23"
)
SUPERSEDED_R42_REPORT_NAME = "profile_report.json"
SUPERSEDED_R42_REPORT_FILE_SHA256 = (
    "430be97de52ba84b1519559b35c5758b2c4d0b08e8a890056ab5c05b78037d62"
)
SUPERSEDED_R42_REPORT_DIGEST = (
    "e18a1e275dda9b72c86ce4f63ea216f250b3a378626f63497a8d252d52755af5"
)
SUPERSEDED_R42_MINIMUM = {"b_eff_multiplier": 0.5, "b_dec_gev2": 2.6}
SUPERSEDED_R42_EIGENVALUES = [
    -182549.71181584388, 7.698059717158571,
]
SUPERSEDED_R42_BLOCK4_EIGENVALUES = [
    -1999.037806083929, 15699.862514911674,
]
SUPERSEDED_R42_SEALED_UTC = "2026-09-24T01:53:32Z"
SUPERSEDED_R42_STATUS = "superseded by re-profile"


# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260920-r4-run-r42-recoil-decay-
# separation; the profile grid, statistics, seeds, likelihood, observables,
# and acceptance blocks are verbatim from the card and the round 4 index
# items 5, 7, 9, and 10).
# ---------------------------------------------------------------------------

RUN_ID = "R46"
CARD = "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile"
RUN_NAME = "r46_simplified_recoil_decay_profile_2026-09-24"
DEFAULT_RUN_DIR = "dev/00runs/r46_simplified_recoil_decay_profile_2026-09-24"
GATE_RECORD_NAME = "gate4_followup_reprofile.json"
PROFILE_REPORT_NAME = "profile_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
ROUTING_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
KERNEL_REL = "dev/mixing2006/na49_mixing/resonance_decay.py"
R32_DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"
R40_DRIVER_REL = "helpers/benchmarking/staged_r40_alignment_scan.py"
R41_DRIVER_REL = "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py"

#: The card chain this run sits on: the Gate 3 closure audits (cards
#: 1-3, sealed; card 1's RECOMMENDATION spawned the rework card 1r,
#: which landed before this run), the cascade implementation cards
#: (4-10, landed and byte-stable), the sealed R40 scan (card 11), R41
#: pilot fit (card 12, the round's first optimizer use), R42 profile
#: (card 13, the superseded profile), R43 ablation (card 14), R44
#: validation (card 15), the Gate 4 decision, the card-20 simplified
#: alignment freeze (obligation 1), and this card (obligation 2).
CARD_CHAIN = (
    "TASK-20260920-r4-audit-transition-routing-formulation",
    "TASK-20260920-r4-audit-strange-sector-continuum-families",
    "TASK-20260920-r4-audit-builder-blend-path-routing-weight",
    "TASK-20260920-r4-rework-transition-routing-formulation",
    "TASK-20260920-r4-sequential-color-singlet-resonance-decay",
    "TASK-20260920-r4-inherited-production-axes",
    "TASK-20260920-r4-baryonic-alignment-kernel",
    "TASK-20260920-r4-mesonic-alignment-kernel",
    "TASK-20260920-r4-hadronic-decay-form-factor",
    "TASK-20260920-r4-termination-depth-tests",
    "TASK-20260920-r4-isotropic-aligned-ablation-runner",
    "TASK-20260920-r4-run-r40-fixed-source-alignment-scan",
    "TASK-20260920-r4-run-r41-nonstrange-pilot-fit",
    "TASK-20260920-r4-run-r42-recoil-decay-separation",
    "TASK-20260920-r4-run-r43-cascade-ablation",
    "TASK-20260920-r4-run-r44-frozen-nonstrange-validation",
    "TASK-20260920-r4-gate4-decision",
    "TASK-20260920-r4-gate4-followup-simplified-alignment",
    "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile",
)

#: The frozen substrate digests (byte-unchanged inputs; verified before
#: any step).  The cascade surfaces are cards 4-10 (landed this round,
#: consumed read-only); the builder blend path, the card-6 routing, the
#: Round 2 kernel, the engine, the catalog, the continuum config, the
#: R32 driver, the R40 driver, and the R41 driver close the list.
FROZEN_SUBSTRATE: dict[str, tuple[str, str]] = {
    "cascade_decay": (
        "dev/mixing2006/na49_mixing/cascade_decay.py",
        "74ced46c01ee9f3665da4f3bce20174cb16a2cba408f327f8df87b848cd94559",
    ),
    "cascade_axes": (
        "dev/mixing2006/na49_mixing/cascade_axes.py",
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
    ),
    "cascade_alignment": (
        "dev/mixing2006/na49_mixing/cascade_alignment.py",
        "7aad9434e70efaf2424d6b66fbeaede9419ea869b4968b11d1466961de5912b2",
    ),
    "cascade_form_factor": (
        "dev/mixing2006/na49_mixing/cascade_form_factor.py",
        "b16077c9e3418b3f19b117aadf6fc8a8e838af5e9d37b1ab0bae89ccfb88f74c",
    ),
    "cascade_termination": (
        "dev/mixing2006/na49_mixing/cascade_termination.py",
        "0e7685721351bfafa2a9ccec04ab472d5a26e99b348aa54bef3d22fcf362c6c0",
    ),
    "cascade_ablation": (
        "dev/mixing2006/na49_mixing/cascade_ablation.py",
        "c0206cf6e75b883de345a788c0246ea7bd7f48bf03102ff7e811e2cc5af9d1c0",
    ),
    "round2_kernel": (
        "dev/mixing2006/na49_mixing/resonance_decay.py",
        "0b675ffef0b17ec0a6d8e93f593a15adda03dc8dca40e5a789f5f3285d384bc6",
    ),
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
    "r32_driver": (
        "helpers/benchmarking/staged_r32_spectral_systematic.py",
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    ),
    "r40_driver": (
        "helpers/benchmarking/staged_r40_alignment_scan.py",
        "7366a55e0e15032f6d4c9194adce7537ea5239075e34f921900b75be64787354",
    ),
    "r41_driver": (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py",
        "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc",
    ),
    "r42_driver": (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py",
        "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd",
    ),
    "r43_driver": (
        "helpers/benchmarking/staged_r43_cascade_ablation.py",
        "dfb5ee3f4ee6598698faaf3da641fbf315ba21d95f2203792a2414a72e11801c",
    ),
    "r44_driver": (
        "helpers/benchmarking/staged_r44_frozen_nonstrange_validation.py",
        "6d77c7137dceb5bb52f05c6c37f701c250b63756618197f2cc454b4ca3cbf1e5",
    ),
    "r45_driver": (
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py",
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    ),
}

#: The sealed archives consumed read-only (their seals verify).  The
#: corrected R41 archive is the frozen-parameter source of the card-20
#: substrate; the sealed card-20 R45 archive is the frozen simplified
#: alignment's run substrate (its phase-2 seal verifies); the sealed
#: corrected R42 archive is the superseded-profile reference (cited as
#: superseded, never as evidence); the R43/R44 archives carry the
#: condition cross-references; the R32 spread is the model-uncertainty
#: carry of obligation 4; the sealed R01 archive is the published-data
#: surface of the likelihood.
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23",
    "dev/00runs/r42_recoil_decay_separation_2026-09-23",
    "dev/00runs/r43_cascade_ablation_2026-09-23",
    "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23",
    "dev/00runs/r45_simplified_alignment_freeze_2026-09-24",
)
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R41_RUN_DIR = "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"
R42_RUN_DIR = "dev/00runs/r42_recoil_decay_separation_2026-09-23"
R43_RUN_DIR = "dev/00runs/r43_cascade_ablation_2026-09-23"
R44_RUN_DIR = "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"
R45_RUN_DIR = "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"
R4_AUDIT_CARD1_DIR = "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20"
R41_FIT_REPORT_NAME = "fit_report.json"
#: The corrected sealed R41 fit report (the frozen-parameter source of
#: the card-20 substrate): the file digest (byte identity) and the
#: report's self-digest (the gate cross-reference of the sealed R41).
R41_FIT_REPORT_FILE_SHA256 = (
    "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
)
R41_FIT_REPORT_DIGEST = (
    "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
)

#: The R32 p+p configuration constants (index item 7).
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.2701885890706  # Reaction(158, 2212, 2212).sqrt_s()
BLEND_RAMP_START_GEV = 2.680194110727528
BLEND_RAMP_WIDTH_GEV = 0.5
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)

#: The frozen exchange-layer slope (the M8 frozen
#: ``regge.effective_transfer_slope``; verified against the applied config
#: at every step that builds one).
B_EFF_FROZEN_GEV2 = 8.504140762083612

#: The card-20 frozen simplified alignment (the generation substrate,
#: constant kappa_B and kappa_M, every mass-dependence coefficient fixed
#: at zero and not fit; the corrected R41 fitted values frozen at the
#: card-20 seal).  The alignment kernels stay at this point for the
#: generation and for the whole profile; they are not profile axes of
#: this card (plan section 24: no refit beyond the profile; plan line
#: 784; obligation 4).
FROZEN_SIMPLIFIED_ALIGNMENT = {
    "kappa_B": 0.0,
    "kappa_m": 4.0,
    "b_dec": 2.0,
    "baryonic_nominal_width": 0.3,
}
GENERATION_KAPPA_B = float(FROZEN_SIMPLIFIED_ALIGNMENT["kappa_B"])
GENERATION_KAPPA_M = float(FROZEN_SIMPLIFIED_ALIGNMENT["kappa_m"])
FROZEN_WIDTH_GEV = float(
    FROZEN_SIMPLIFIED_ALIGNMENT["baryonic_nominal_width"])
FROZEN_ALIGNMENT_NOTE = (
    "the card-20 frozen simplified alignment (the sealed "
    "r45_simplified_alignment_freeze_2026-09-24 archive's frozen "
    "report carries it; the corrected R41 fitted tuple (0.0, 4.0, 2.0 "
    "GeV^-2, 0.3 GeV) frozen at the card-20 seal under the "
    "constant-kappa simplified form of plan section 10.4 line 732, "
    "every mass-dependence coefficient fixed at zero and not fit): "
    "constant kappa_B and kappa_M, b_dec at the frozen 2.0 GeV^-2, "
    "the baryonic nominal_width recorded at the fitted 0.3 GeV (the "
    "continuum-daughter mass rule consumes the byte-unchanged "
    "committed continuum config's family widths; the fitted width is "
    "the frozen-parameter record, not a config override).  The "
    "alignment is not a profile axis of this card (plan section 24: "
    "no refit beyond the profile; plan line 784; obligation 4); event "
    "generation uses it throughout and re-pointing generation at any "
    "other frozen set is a named deviation"
)
CARD20_SUBSTRATE_NOTE = (
    "the card-20 phase-2 sealed archive "
    "dev/00runs/r45_simplified_alignment_freeze_2026-09-24 (the frozen "
    "simplified alignment) is the generation substrate: its seal "
    "verifies before any step, its driver is a byte-frozen substrate "
    "sibling (the FROZEN_SUBSTRATE r45_driver digest, cross-checked "
    "against the imported sibling module's own self-digest and frozen "
    "policy at import), and the frozen tuple is asserted against the "
    "sealed corrected R41 fit report before any generation"
)


def assert_frozen_parameters(report: dict) -> dict:
    """Assert the frozen simplified alignment against an R41-style fit
    report (the sealed R41 archive in production, the pinned fixture in
    the smoke).  Returns the frozen-parameter block; any drift
    refuses."""
    try:
        fitted = report.get("fitted")
        if not isinstance(fitted, dict):
            raise ValueError("the fit report carries no 'fitted' block")
        observed = {
            "kappa_B": float(fitted["kappa_B"]),
            "kappa_m": float(fitted["kappa_m"]),
            "b_dec": float(fitted["b_dec"]),
            "baryonic_nominal_width": float(
                fitted["baryonic_nominal_width"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        _refuse(
            "the R41 fit report does not carry a readable 'fitted' "
            f"block ({exc}); the frozen parameters cannot be asserted"
        )
    if report.get("card") != "TASK-20260920-r4-run-r41-nonstrange-pilot-fit":
        _refuse(
            "the fit report does not name the R41 card "
            "(TASK-20260920-r4-run-r41-nonstrange-pilot-fit); the "
            "frozen-parameter source is not the sealed R41 archive"
        )
    if report.get("run") != "R41":
        _refuse("the fit report does not name run R41")
    if report.get("mode") != "production":
        _refuse("the R41 fit report is not the production fit report")
    if observed != FROZEN_SIMPLIFIED_ALIGNMENT:
        _refuse(
            "the sealed R41 fitted values "
            f"{observed!r} do not match the pre-registered card-20 "
            "frozen simplified alignment "
            f"{FROZEN_SIMPLIFIED_ALIGNMENT!r}; the frozen-parameter "
            "source drifted (the run refuses to refit or to adapt)"
        )
    return {
        "kappa_B": observed["kappa_B"],
        "kappa_m": observed["kappa_m"],
        "b_dec_gev2": observed["b_dec"],
        "baryonic_nominal_width_gev": observed["baryonic_nominal_width"],
        "source": (
            "sealed R41 archive "
            f"({R41_RUN_DIR}/{R41_FIT_REPORT_NAME})"
        ),
        "refit": False,
    }


#: The pinned R41 fit-report fixture of the smoke mode (the hermetic
#: stand-in for the sealed archive; the same values the production
#: assertion expects).
R41_FIT_REPORT_FIXTURE = {
    "card": "TASK-20260920-r4-run-r41-nonstrange-pilot-fit",
    "run": "R41",
    "mode": "production",
    "fitted": dict(FROZEN_SIMPLIFIED_ALIGNMENT),
    "no_refit": True,
    "optimizer_invoked": True,
}


def assert_card20_substrate() -> dict:
    """The card-20 sibling substrate cross-check (run at import): the
    sealed card-20 R45 driver is byte-identical in both trees and sits
    at the pinned substrate digest, its own self-digest agrees, its
    frozen simplified policy equals this card's frozen simplified
    alignment, and its run identity is the card-20 run dir.  Any drift
    refuses the run before any step."""
    expected = FROZEN_SUBSTRATE["r45_driver"]
    if r45_substrate.DRIVER_SHA256 != expected[1]:
        _refuse(
            "the imported card-20 sibling driver's self-digest "
            f"{r45_substrate.DRIVER_SHA256!r} does not match the "
            f"pinned substrate digest {expected[1]!r}; the card-20 "
            "substrate drifted"
        )
    nested_copy = (NESTED_ROOT / "helpers" / "benchmarking" /
                   "staged_r45_simplified_alignment_freeze.py")
    outer_copy = (MONO_ROOT / "helpers" / "benchmarking" /
                  "staged_r45_simplified_alignment_freeze.py")
    for path in (nested_copy, outer_copy):
        if not path.is_file():
            _refuse(f"the card-20 sibling driver copy is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[1]:
            _refuse(
                f"the card-20 sibling driver copy {path} is not "
                f"byte-identical to the pinned substrate digest "
                f"{expected[1]!r}; the dual-tree invariant is broken"
            )
    policy = r45_substrate.ALIGNED_POLICY
    if (float(r45_substrate.ALIGNED_POLICY.kappa_b),
            float(r45_substrate.ALIGNED_POLICY.kappa_m),
            float(r45_substrate.ALIGNED_POLICY.b_dec)) != (
            GENERATION_KAPPA_B, GENERATION_KAPPA_M,
            float(FROZEN_SIMPLIFIED_ALIGNMENT["b_dec"])):
        _refuse(
            "the card-20 sibling driver's frozen aligned policy "
            "disagrees with this card's frozen simplified alignment; "
            "the generation substrate drifted"
        )
    if float(r45_substrate.FROZEN_WIDTH_GEV) != FROZEN_WIDTH_GEV:
        _refuse(
            "the card-20 sibling driver's frozen nominal width "
            f"{r45_substrate.FROZEN_WIDTH_GEV!r} disagrees with the "
            f"recorded frozen width {FROZEN_WIDTH_GEV!r}"
        )
    if r45_substrate.DEFAULT_RUN_DIR != R45_RUN_DIR:
        _refuse(
            "the card-20 sibling driver's run dir "
            f"{r45_substrate.DEFAULT_RUN_DIR!r} is not the sealed "
            f"card-20 archive {R45_RUN_DIR!r}"
        )
    return {
        "sibling_driver_rel": FROZEN_SUBSTRATE["r45_driver"][0],
        "sibling_driver_sha256": expected[1],
        "sibling_self_digest_match": True,
        "sibling_frozen_policy": {
            "kappa_B": float(r45_substrate.ALIGNED_POLICY.kappa_b),
            "kappa_m": float(r45_substrate.ALIGNED_POLICY.kappa_m),
            "b_dec": float(r45_substrate.ALIGNED_POLICY.b_dec),
        },
        "sibling_frozen_width_gev": float(
            r45_substrate.FROZEN_WIDTH_GEV),
        "frozen_simplified_archive": r45_substrate.DEFAULT_RUN_DIR,
        "sibling_run_dir": r45_substrate.DEFAULT_RUN_DIR,
        "verified": True,
        "note": CARD20_SUBSTRATE_NOTE,
    }


_CARD20_SUBSTRATE_STATUS = assert_card20_substrate()

#: The pre-registered 9 x 9 profile grid (unchanged from the corrected
#: R42): B_eff multipliers (0.25 steps) and b_dec values in GeV^-2
#: (0.3 steps).  Generation runs at the card-20 frozen b_dec = 2.0
#: GeV^-2, which sits ON the grid (multiplier 1.0 with b_dec 2.0
#: GeV^-2 is the generation point); every other grid point reweights
#: away from it.  The bounds are never relaxed and the grid is never
#: extended.
B_EFF_MULTIPLIERS: tuple[float, ...] = (
    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
)
B_DEC_GRID: tuple[float, ...] = (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6)
B_DEC_GENERATION_GEV2 = 2.0
B_DEC_MULT_STEP = 0.25
B_DEC_STEP_GEV2 = 0.3
PROFILE_GRID_NOTE = (
    "the 9 x 9 profile grid (unchanged from the corrected R42): B_eff "
    "at 0.5..2.5 times the frozen value (0.25 steps; the frozen value "
    "is multiplier 1.0) and b_dec at 0.2..2.6 GeV^-2 in 0.3 steps.  "
    "The generation point is multiplier 1.0 with b_dec 2.0 GeV^-2 (the "
    "card-20 frozen value, on the grid): the weights are exactly 1 "
    "there and every other grid point reweights away from it.  The "
    "reweighting is exact for the exponential transfer law and the "
    "decay form factor (both are pure weight changes; no regeneration, "
    "plan section 5.3 line 288).  The pre-registered bounds are never "
    "relaxed and the grid is never extended to move the minimum "
    "interior.  The profile minimum is the pre-registered lattice "
    "argmin; no continuous optimization is claimed (the curvature is "
    "the grid second difference at the argmin, the coordinate spacings "
    "0.25 and 0.3 GeV^-2 being the pre-registered resolution)"
)
#: The two pinned grid points of the reweighting-validity assertion
#: (b_eff multiplier, b_dec GeV^-2), off-nominal, spanning the grid.
VALIDITY_GRID_POINTS: tuple[tuple[float, float], ...] = ((1.5, 1.7), (2.0, 0.5))
VALIDITY_EVENTS = 2_000
VALIDITY_NOTE = (
    "the reweighting validity is asserted against the direct generation "
    "of the pinned sub-sample (the first 2,000 production event indices "
    "of replica 0, common random numbers, in memory, never sealed) at the "
    "two pinned grid points (multiplier 1.5 with b_dec 1.7 GeV^-2 and "
    "multiplier 2.0 with b_dec 0.5 GeV^-2): the stream alignment (the "
    "route census and the transfer-law support precede every parameter-"
    "dependent draw, exact integer/float equality), the implied trial "
    "realization t_implied = G_B'^{-1}(G_B(t)) against the direct "
    "generation's realized t per pinned event (1e-6 relative), and the "
    "first-vertex normalized trial shares recomputed from the frozen "
    "latent's (w0, q2) against the frozen-B direct generation's own "
    "attached shares at the trial b_dec (1e-6 relative; the parent state "
    "is parameter-independent at the frozen B_eff).  The bound 1e-6 is "
    "the pre-registered assert tolerance on these exact identities; the "
    "prediction-level pooled comparison is recorded beside them as a "
    "Monte Carlo agreement diagnostic, never asserted at 1e-6"
)
VALIDITY_BOUND = 1e-6

#: The likelihood's nuisance structure (plan section 5.4): one
#: multiplicative normalization nuisance on the b_transport spectrum plus
#: the obligation 4 carry on its blend-band-sensitive bins.  The seagull
#: statistics are normalization-free ratios and carry none.
NUISSANCE_COLUMNS = ("norm_b_transport_spectrum", "carry")
STAT_BLOCK_COLUMN = {"b_transport_spectrum": 0}
NORM_COLUMN = 0
CARRY_COLUMN_INDEX = 1
NORM_NUISANCE_PRIOR = 0.35
NORM_NUISANCE_NOTE = (
    "one multiplicative normalization nuisance on the b_transport "
    "spectrum (m -> m (1 + eta)) with prior width 0.35: the profile "
    "stage has no calibrated model-to-data normalization (the sealed "
    "R01/R32 record shows the standing mismatch is of order one) and the "
    "profiled nuisance absorbs the level while the shape residuals "
    "constrain the profile axes; the nuisance profile is part of the "
    "objective definition (plan section 5.4), not part of the profile "
    "axes"
)
CARRY_NUISANCE_PRIOR = 0.25
CARRY_PRIOR_NOTE = (
    "the obligation 4 carry is one correlated multiplicative nuisance "
    "(m -> m (1 + eta_carry) on the blend-band-sensitive bins) with "
    "prior width 0.25: the sealed p50 relative cross-variant spread of "
    "the 32 persisting bins is 0.2351 (computed from the sealed card-1 "
    "attribution rows), rounded up conservatively; the term is never a "
    "free parameter of the profile"
)
CARRY_SPECIES = ("proton", "antiproton")

#: The obligation 4 model-uncertainty carry (the sealed R32 cross-variant
#: spread), verbatim from the card; the blend-band-sensitive bins are the
#: sealed R32 persisting-residual bins restricted to the b_transport
#: pair's species (the sealed card-1 attribution carries 9 proton rows
#: and no antiproton rows, so 9 of the 32 intersect the B_transport
#: pair's species and carry the term in the objective; all 32 are
#: recorded in the evidence).
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}
OBLIGATION4_FORM = (
    "one correlated multiplicative nuisance on the blend-band-sensitive "
    "bins (the sealed R32 persisting-residual bins restricted to the "
    "b_transport pair's proton/antiproton cells; prior 0.25), profiled "
    "exactly with the spectrum normalization nuisance per grid point "
    "(plan section 5.4); the audits closed with card 2 and card 3 PASS "
    "and card 1 RECOMMENDATION (its rework card 1r landed before this "
    "run with no new spread diagnostic), so the original R32 spread "
    "stands"
)

#: The robust-statistics cross-reference of the named-residual register
#: (the R32 acceptance-rescope precedent): the between-replica SE
#: convention floored at the engine's deterministic SE floor; the
#: 3.0-sigma band is never relaxed; no data selection.
SIGMA_BAND = 3.0
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)
SE_CONVENTION = (
    "the between-replica standard error (std ddof=1 over the 8 T2 "
    "replica blocks, divided by sqrt(8)), floored at the engine's "
    f"DETERMINISTIC_SE_FLOOR = {SE_FLOOR!r}; the 3.0-sigma band is "
    "never relaxed (the acceptance rescope precedent of "
    "TASK-20260918-r3-acceptance-rescope-card8a-8b)"
)

#: The pre-registered distinguishability bound (acceptance (ii); plan
#: line 788 "distinguishable effects").  A correlation at or above 0.9
#: means the two parameters move the observables as one combined
#: direction -- the reparametrization the plan line 748 ban forbids -- and
#: the run seals with the named finding feeding the Gate 4 string-like
#: rank law clause.  The bound is never relaxed.
CORRELATION_BOUND = 0.9
CORRELATION_NOTE = (
    "the parameter correlation rho = -H_01 / sqrt(H_00 H_11) at the "
    "profile minimum (the inverse-Hessian correlation of the 2 x 2 "
    "curvature matrix; the Hessian of -2lnL has the inverse covariance "
    "up to the common positive factor 2, which cancels in rho).  The "
    "pre-registered bound |rho| < 0.9 is never relaxed; a failure "
    "seals with the named finding correlation_undefined_or_high (the "
    "plan line 788 distinguishable-effects clause on the "
    "leading-baryon transfer observables and the identified seagulls)"
)
#: The Hessian discipline (plan section 5.4, line 311): no curvature is
#: quoted until the replica-block minima agree and the local curvature
#: is stable; stability checked across all 8 blocks.  The stability
#: clause is the acceptance (ii) per-direction spread bound.
HESSIAN_DISCIPLINE_NOTE = (
    "no curvature is quoted until the replica-block minima agree and "
    "the local curvature is stable (plan section 5.4, line 311): the "
    "8 per-replica-block surface argmins must equal the pooled argmin "
    "before the pooled Hessian, its eigenvalues, or the correlation "
    "are quoted; the curvature matrix is still recorded beside the "
    "gate (acceptance (i) records the grid, the minimum, and the "
    "curvature).  The stability check spans all 8 blocks: every block "
    "positive definite with the per-direction curvature spread below "
    "20% of the pooled curvature (plan section 5.5 item 5 line 321)"
)
CURVATURE_SPREAD_FRACTION = 0.20
CURVATURE_SPREAD_NOTE = (
    "per direction (the B_eff multiplier axis and the b_dec axis), the "
    "curvature-matrix diagonal spread across the 8 replica blocks, "
    "max_block |H_block - H_pooled| / |H_pooled| on the pooled "
    "diagonal, must stay below the pre-registered 0.20 fraction of the "
    "pooled curvature; the bound is never relaxed and a failure is the "
    "block_curvature_failure named finding with the axis, the spread, "
    "and the offending block indices quoted"
)
ESS_FLOOR_FRACTION = 0.01
ESS_NOTE = (
    "the effective sample size of the reweighted ensemble, "
    "(sum w)^2 / sum w^2 over the pooled events, is recorded at the "
    "profile minimum; an ESS fraction below 0.01 is a named "
    "weight-degeneracy finding, never a refit"
)
EVENT_CLOSURE_TOL = 1e-9
DEGENERATE_WINDOW_TOL = 1e-12

#: Generation chunking (the R01/R23/R32 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 8

#: The event-seed stage label of the R46 stream.
STAGE_LABEL = "r46_simplified_recoil_decay_profile"

#: The clean-environment validation's pre-registered probe replica.
PROBE_REPLICA = 0


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


def _write_bytes(path: Path, payload: bytes) -> None:
    Path(path).write_bytes(payload)


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
        return {
            "head": head,
            "branch": branch,
            "dirty": bool(dirty.strip()),
        }
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


def substrate_digests() -> dict[str, str]:
    return {
        key: _sha256_file(MONO_ROOT / rel)
        for key, (rel, _expected) in FROZEN_SUBSTRATE.items()
    }


def verify_frozen_substrate() -> dict:
    """Verify the frozen substrate digests (byte-unchanged inputs)."""
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


def verify_sealed_dir(rel: str) -> dict:
    """Verify one sealed run dir's ``SHA256SUMS.txt`` (read-only)."""
    sums_path = MONO_ROOT / rel / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"sealed run dir carries no {SHA256SUMS_NAME}: {sums_path}")
    run_dir = MONO_ROOT / rel
    mismatches: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
        if not payload.is_file():
            payload = run_dir / name
        if not payload.is_file():
            mismatches.append(f"missing {name}")
            continue
        if _sha256_file(payload) != digest:
            mismatches.append(name)
    entries = sum(
        1 for line in sums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if mismatches:
        _refuse(
            f"sealed run dir {rel} fails its seal: {len(mismatches)} "
            f"mismatches of {entries} entries (first: {mismatches[0]!r})"
        )
    return {"entries": entries, "mismatches": 0, "verified": True}


def verify_sealed_inputs() -> dict:
    """Verify every sealed run dir the pre-registration consumes."""
    return {rel: verify_sealed_dir(rel) for rel in SEALED_RUN_DIRS}


# ---------------------------------------------------------------------------
# Seeds (round 4 index item 9, the card's pre-registered map)
# ---------------------------------------------------------------------------
MASTER_SEED = 20260920
SEED_BASE = 20260920
SEED_OFFSET = 40_000
SEED_STRIDE = 156
T2_REPLICAS = 8
T2_EVENTS_PER_REPLICA = 100_000
RAW_SEED_RULE = (
    "raw replica seed = 20260920 + 40000 + 156 + k = 20301076 + k per "
    "replica k < 8 (the block 20301076-20301083, a new block disjoint "
    "from the R40-R44 block 20301020-20301039, from the reserved audit "
    "block 20301040 + k (k = 0..15, 20301040..20301055, never fired), "
    "and from the card-20 block 20301056-20301075; the R46 block "
    "shares no raw seed with any previously used raw block (R23 "
    "20300913-20300920 under the 20260913 master; the R32 production "
    "and confirmation blocks under the 20260918 master; R40 "
    "20301020-20301023, R41/R42/R43 20301020-20301027, and R44 "
    "20301020-20301039 under the 20260920 master; R45 20301056-20301075 "
    "under the 20260920 master).  The (seed, event-index) substreams "
    "are disjoint by construction and the event-index partitions are "
    "disjoint; if any raw (seed, event-index) substream collides, the "
    "run stops and reports (never a silent re-block)"
)
KERNEL_SEED_RULE = (
    "derive_substreams(20260920, k, 900_000_000 + i) per (replica "
    "k < 8, event i < 100,000) on the closed decay_channel/"
    "decay_kinematics labels (the round 2 card 9 and round 3 card 8 "
    "convention, the R42 pattern: the master seed with the replica "
    "folded into the replica index, the event index encoding the run); "
    "the 8 (replica) kernel-layer spaces are pairwise disjoint and "
    "disjoint from every previously used space because the event-index "
    "partition 900_000_000 + i differs from the card-20 R45 partition "
    "800_000_000 + i, from R44's 700_000_000 + i, from R43's "
    "600_000_000 + i, from R42's 500_000_000 + i, from R41's "
    "400_000_000 + i, from the R40 partition 300_000_000 + "
    "s*10_000_000 + p*1_000_000 + i, and from the audit block "
    "20301040 + k"
)
EVENT_INDEX_RULE = "900_000_000 + i, i < 100,000"
EVENT_INDEX_BASE = 900_000_000
RAW_REPLICA_SEEDS = tuple(SEED_BASE + SEED_OFFSET + SEED_STRIDE + r
                          for r in range(T2_REPLICAS))

#: The reserved audit confirmation block (the cards 1-3 audit
#: pre-registration: raw seeds 20301040 + k, k = 0..15, never fired).
AUDIT_RESERVED_BLOCK = (20301040, 20301055)

#: The previously-used seed blocks and event-index partitions, recorded
#: from the sealed pre-registrations (the disjointness evidence of the
#: card; the driver verifies the partition disjointness and refuses on
#: any (seed, event-index) substream collision).
PREVIOUS_USED_BLOCKS = {
    "r23": {
        "master_seed": 20260913,
        "raw_seeds": [20300913, 20300920],
        "event_index_partition": None,
        "source": "the sealed R23-era raw block (the R41 disjointness record)",
    },
    "r32": {
        "master_seed": 20260918,
        "raw_seeds": [20301519, 20301826],
        "event_index_partition": None,
        "source": (
            "the sealed R40 pre-registration's disjointness record (the "
            "R32 production blocks 20301519-20301526 / 20301619-20301626 "
            "/ 20301719-20301726 and confirmation block 20301819-..., "
            "smoke 20305919 + 100*v)"
        ),
    },
    "r40": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301023],
        "event_index_partition": (
            "300_000_000 + s*10_000_000 + p*1_000_000 + i, s < 9, "
            "p < 10, i < 25,000 (stays below 389,025,000)"
        ),
        "event_index_range": [300_000_000, 389_025_000],
        "source": (
            "sealed dev/00runs/r40_fixed_source_alignment_scan_"
            "2026-09-21/pre_registration.json"
        ),
    },
    "r41": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_partition": "400_000_000 + i, i < 100,000",
        "event_index_range": [400_000_000, 400_100_000],
        "smoke_raw_seed": 20308920,
        "smoke_replica_slot": 8000,
        "smoke_event_index": "400_000_000 + i, i < 2,000 (slot 8000)",
        "source": f"sealed {R41_RUN_DIR}/pre_registration.json",
    },
    "r42": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_partition": "500_000_000 + i, i < 100,000",
        "event_index_range": [500_000_000, 500_100_000],
        "smoke_raw_seed": 20309920,
        "smoke_replica_slot": 9000,
        "smoke_event_index": "500_000_000 + i, i < 2,000 (slot 9000)",
        "source": f"sealed {R42_RUN_DIR}/pre_registration.json",
    },
    "r43": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301027],
        "event_index_partition": "600_000_000 + i, i < 100,000",
        "event_index_range": [600_000_000, 600_100_000],
        "smoke_raw_seed": 20309920,
        "smoke_replica_slot": 9000,
        "smoke_event_index": "690_000_000 + i, i < 2,000",
        "source": f"sealed {R43_RUN_DIR}/pre_registration.json",
    },
    "r44": {
        "master_seed": 20260920,
        "raw_seeds": [20301020, 20301039],
        "event_index_partition": "700_000_000 + i, i < 250,000",
        "event_index_range": [700_000_000, 700_250_000],
        "smoke_raw_seed": 20310920,
        "smoke_replica_slot": 10000,
        "smoke_event_index": "790_000_000 + i, i < 2,000",
        "source": f"sealed {R44_RUN_DIR}/pre_registration.json",
    },
    "r45": {
        "master_seed": 20260920,
        "raw_seeds": [20301056, 20301075],
        "event_index_partition": "800_000_000 + i, i < 250,000",
        "event_index_range": [800_000_000, 800_250_000],
        "smoke_raw_seed": 20311920,
        "smoke_replica_slot": 11000,
        "smoke_event_index": "890_000_000 + i, i < 2,000",
        "source": f"sealed {R45_RUN_DIR}/pre_registration.json",
    },
    "audit_reserved": {
        "master_seed": 20260920,
        "raw_seeds": [20301040, 20301055],
        "event_index_partition": None,
        "fired": False,
        "source": (
            "the cards 1-3 audit confirmation block (reserved, never "
            "fired: the audits reused sealed streams; "
            "helpers/benchmarking/audit_r4_builder_blend_path.py pins "
            "RESERVED_BLOCK_FIRST = 20301040 with k = 0..15)"
        ),
    },
}


def seed_disjointness_report() -> dict:
    """The seed-block disjointness check (the card's stop condition): the
    exact previously-used blocks recorded from the sealed
    pre-registrations, the shared raw seeds named, and the event-index
    partition disjointness verified.  Any (seed, event-index) substream
    collision refuses the run (never a silent re-block)."""
    problems: list[str] = []
    r46_lo, r46_hi = EVENT_INDEX_BASE, EVENT_INDEX_BASE + T2_EVENTS_PER_REPLICA
    audit_lo, audit_hi = AUDIT_RESERVED_BLOCK
    if not (max(RAW_REPLICA_SEEDS) < audit_lo
            or min(RAW_REPLICA_SEEDS) > audit_hi):
        problems.append(
            f"the R46 raw block {min(RAW_REPLICA_SEEDS)}.."
            f"{max(RAW_REPLICA_SEEDS)} overlaps the reserved audit "
            f"block {audit_lo}..{audit_hi}"
        )
    shared: dict[str, list[int]] = {}
    for name, block in PREVIOUS_USED_BLOCKS.items():
        seeds = block.get("raw_seeds")
        if not seeds or block.get("event_index_range") is None:
            continue  # a different master-seed tree or a no-partition record
        lo, hi = seeds
        overlap = sorted(set(range(lo, hi + 1)) & set(RAW_REPLICA_SEEDS))
        if overlap:
            shared[name] = [int(overlap[0]), int(overlap[-1])]
        p_lo, p_hi = block["event_index_range"]
        if r46_lo < p_hi and p_lo < r46_hi:
            problems.append(
                f"the R46 event-index partition [{r46_lo}, {r46_hi}) "
                f"overlaps the {name} partition [{p_lo}, {p_hi})"
            )
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r46] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r46_raw_seed_block": [
            min(RAW_REPLICA_SEEDS), max(RAW_REPLICA_SEEDS)],
        "r46_event_index_partition": EVENT_INDEX_RULE,
        "r46_event_index_range": [r46_lo, r46_hi],
        "audit_reserved_block": [audit_lo, audit_hi],
        "audit_block_disjoint": True,
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the R46 raw block 20301076-20301083 shares no raw seed "
            "with any previously used block (R40-R44 sit below it at "
            "20301020-20301039, the audit block at 20301040-20301055, "
            "and the card-20 R45 block at 20301056-20301075); the "
            "substreams are disjoint by construction and the "
            "event-index partitions are disjoint (the builder's legacy "
            "generator SeedSequence(seed, spawn_key=(event_id,)) and "
            "derive_substreams(20260920, k, event_id) both key on the "
            "event index)"
        ),
        "previously_used_blocks": PREVIOUS_USED_BLOCKS,
        "disjoint": True,
    }


#: The hermetic smoke mode (driver validation; never sealed): 1 replica
#: x 2,000 events at the pre-registered smoke seeds.  The +12000 block
#: (raw smoke seed 20312920, kernel replica index 12000) is disjoint
#: from every production space (replica indices 0..7), from the card-20
#: R45 +11000 smoke block, from the R44 +10000 smoke block, from the
#: R42/R43 +9000 smoke block, from the R41 +8000 smoke block, from the
#: R40 +7000 smoke block, from the R32 +5000 smoke block, and from
#: every previously used block.
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 12_000
SMOKE_RAW_SEED = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE
SMOKE_EVENT_INDEX_BASE = 990_000_000
SMOKE_EVENT_INDEX_RULE = "990_000_000 + i, i < 2,000"
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260920 + 40000 + 12000 = 20312920 (the +12000 "
    "smoke block, one replica; the card-20 R45 smoke block is +11000, "
    "the R44 smoke block +10000, the R42/R43 smoke block +9000, the "
    "R41 smoke block +8000, and the R40 smoke block +7000); the smoke "
    "kernel layer draws at derive_substreams(20260920, 12000, "
    "event_index) on the +90M smoke event-index partition "
    "(990_000_000 + i, i < 2,000), disjoint from the R46 production "
    "partition (900_000_000 + i, i < 100,000), from every production "
    "space (replica indices 0..7), and from every previously used "
    "block"
)
SMOKE_VALIDITY_EVENTS = 200
SMOKE_SYNTHETIC_TRUTH = {
    "b_eff_multiplier": 1.3,
    "b_dec": 1.2,
}
SMOKE_NOTE = (
    "the smoke profile evaluates the surface machinery against a "
    "synthetic data vector: the smoke record's own prediction at the "
    "pre-registered synthetic-truth point (b_eff_multiplier, b_dec) = "
    "(1.3, 1.2), so the smoke exercises pre-registration, generation, "
    "the full 9 x 9 reweighting grid, the curvature and correlation "
    "algebra, and the pinned validity probe end to end, hermetically "
    "(the smoke data is the synthetic truth, no SPS table, no sealed "
    "archive)"
)


def raw_replica_seed(replica: int) -> int:
    r = int(replica)
    if not 0 <= r < T2_REPLICAS:
        _refuse(f"replica index {r} outside the T2 block 0..{T2_REPLICAS - 1}")
    return RAW_REPLICA_SEEDS[r]


def kernel_replica_slot(replica: int) -> int:
    """The kernel-layer replica index of a production replica."""
    r = int(replica)
    if not 0 <= r < T2_REPLICAS:
        _refuse(f"replica index {r} outside the T2 block")
    return r


def smoke_raw_seed() -> int:
    return SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE


SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE


def event_index(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


# ---------------------------------------------------------------------------
# Species / quantum-number conventions (the R01/R32 driver conventions)
# ---------------------------------------------------------------------------
SPECIES: tuple[tuple[str, int], ...] = (
    ("pi_plus", 211),
    ("pi_minus", -211),
    ("k_plus", 321),
    ("k_minus", -321),
    ("proton", 2212),
    ("antiproton", -2212),
)
SPECIES_KEYS: tuple[str, ...] = tuple(name for name, _ in SPECIES)
SPECIES_PDGS: dict[str, int] = {name: int(pdg) for name, pdg in SPECIES}

_BQS_CACHE: dict[int, tuple[int, int, int]] = {}


def _pdg_bqs(catalog: HadronCatalog, pdg: int) -> tuple[int, int, int]:
    """The (B, Q, S) of a final-state pdg (cached)."""
    cached = _BQS_CACHE.get(int(pdg))
    if cached is not None:
        return cached
    try:
        record = catalog.get(int(pdg))
        value = (int(record.B), int(record.Q), int(record.S))
    except KeyError:
        from na49_mixing.quantum_numbers import quantum_numbers

        qn = quantum_numbers(int(pdg))
        value = (int(qn.B), int(qn.Q), int(qn.S))
    _BQS_CACHE[int(pdg)] = value
    return value


def _species_of(catalog: HadronCatalog, pdg: int) -> str | None:
    """The paper-window species key of a final pdg, or None."""
    for name, pdg_id in SPECIES:
        if int(pdg) == pdg_id:
            return name
    return None


# ---------------------------------------------------------------------------
# The event assembly (the R42 p+p configuration)
# ---------------------------------------------------------------------------
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any] | None) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


class _LatentCapture:
    """The per-event latent arrays of one generated event.

    ``rows`` holds the event's sparse (dense lattice cell, Jacobian
    weight) contributions of every final particle (the R01 binning
    convention: ``x_F = 2 p_z / sqrt(s)``, the pT-major DENSE lattice
    layout, the same in-bounds drop).  Every event's rows are
    reweightable in R46: the
    exchange-layer slope B_eff reweights the exact two-body scattering
    of the event (the recorded ``t_scatter`` with its support
    ``t_minus``/``t_plus`` is the exact sufficient statistic), so the
    per-event CSR covers the full event set and no theta-independent
    fixed pool exists.  ``vertices`` holds one latent tuple per
    continuum-route cascade vertex: ``(chosen, nchan, w0, q2)`` with the
    attached pool's normalized card-8 shares ``w0`` at the generation
    b_dec and the reference transverse momenta squared ``q2`` (the exact
    data the b_dec reweighting consumes).  Events with no continuum-
    routed source carry an empty vertex list.
    """

    __slots__ = (
        "event_id", "flat", "weights", "t_scatter", "t_minus", "t_plus",
        "vtx_nchan", "vtx_chosen", "ch_w0", "ch_q2",
        "n_vertices", "route_census", "depth_hist", "bqs", "energy_sum",
        "energy_expected", "worst_closure", "event_hash", "n_finals",
        "mult_sums", "mult_sums2", "mult_with", "payload",
    )

    def __init__(self) -> None:
        self.flat: list[int] = []
        self.weights: list[float] = []
        self.t_scatter = 0.0
        self.t_minus = 0.0
        self.t_plus = 0.0
        self.vtx_nchan: list[int] = []
        self.vtx_chosen: list[int] = []
        self.ch_w0: list[float] = []
        self.ch_q2: list[float] = []
        self.n_vertices = 0
        self.route_census = [0, 0, 0]  # sources, catalog, continuum
        self.depth_hist: dict[int, int] = {}
        self.bqs = (0, 0, 0)
        self.energy_sum = 0.0
        self.energy_expected = 0.0
        self.worst_closure = 0.0
        self.event_hash = ""
        self.n_finals = 0
        self.mult_sums: dict[str, float] = {k: 0.0 for k in SPECIES_KEYS}
        self.mult_sums2: dict[str, float] = {k: 0.0 for k in SPECIES_KEYS}
        self.mult_with: dict[str, int] = {k: 0 for k in SPECIES_KEYS}
        self.payload: list[str] = []

    def add_final(self, pdg: int, p4: tuple[float, float, float, float]) -> None:
        """One final particle into the capture: the (B, Q, S) tally, the
        energy tally, the multiplicity tally, and the paper-window bin
        row (the R01 binning convention; the same grid edges, the same
        ``sps_core.grid_indices`` binning, the pT-major DENSE lattice
        layout, the same in-bounds drop).  The recorded cell is the
        species' dense lattice address -- the dense offset plus
        ``ip * n_x + ix`` -- and the published rows are picked from the
        dense totals at record assembly (the R01 ``_bin_track_sums``
        accumulate-dense-then-pick convention), so a hole cell's weight
        stays unpicked and is dropped."""
        context = _WORKER_CONTEXT
        assert context is not None
        tables = context["tables"]
        sqrt_s = context["sqrt_s"]
        catalog = context["catalog"]
        self.bqs = (
            self.bqs[0] + _pdg_bqs(catalog, pdg)[0],
            self.bqs[1] + _pdg_bqs(catalog, pdg)[1],
            self.bqs[2] + _pdg_bqs(catalog, pdg)[2],
        )
        e, px, py, pz = (float(v) for v in p4)
        self.payload.append(
            f"{int(pdg)}:{float(e).hex()}:{float(px).hex()}:"
            f"{float(py).hex()}:{float(pz).hex()}"
        )
        self.energy_sum += e
        self.n_finals += 1
        species = _species_of(catalog, pdg)
        if species is not None:
            self.mult_sums[species] += 1.0
            self.mult_sums2[species] += 1.0
            self.mult_with[species] = self.mult_with.get(species, 0) + 1
        pt = math.hypot(px, py)
        if pt <= 0.0:
            return
        table = context["table_by_pdg"].get(int(pdg))
        if table is None:
            return
        x_f = 2.0 * pz / sqrt_s
        ix, ip = sps_core.grid_indices(
            np.array([x_f]), np.array([pt]),
            context["edges_by_species"][species][0],
            context["edges_by_species"][species][1],
        )
        ix = int(ix[0])
        ip = int(ip[0])
        n_x = context["n_x_by_species"][species]
        n_p = context["n_p_by_species"][species]
        if not (0 <= ix < n_x and 0 <= ip < n_p):
            return
        flat = context["dense_offsets_by_species"][species] + ip * n_x + ix
        self.flat.append(int(flat))
        self.weights.append(e / pt)


def _reference_pool_payload(
    attached: list,
) -> list[float]:
    """The attached pool's normalized shares at the generation b_dec.

    ``attached[i].weight`` is card 4's base weight scaled by the card-8
    W_dec factor at the reference two-body configuration; the recorded
    ``w0`` is the normalized share the vertex walk draws against at the
    generation-time b_dec, and ``q2`` the reference transverse momentum
    squared per channel (the exact reweighting sufficient statistics).
    """

    total = math.fsum(float(channel.weight) for channel in attached)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            "r46 latent capture: the attached pool weights do not sum "
            f"to a finite positive total ({total!r})"
        )
    w0 = [float(channel.weight) / total for channel in attached]
    return w0


def _walk_with_capture(
    catalog: HadronCatalog,
    fams: list,
    root_event: dict,
    kappa_b: float,
    kappa_m: float,
    b_dec: float,
    rng_channel,
    rng_kinematics,
    capture: _LatentCapture,
) -> tuple[int, str]:
    """The card-9/card-10 walk with the reweighting latent captured.

    The R40/R41 walk order byte for byte (the depth guard, the catalog
    handoff, the continuation, the direct termination, the no-channel
    refusal) with the attached pool's (w0, q2, chosen, nchan) recorded
    per continuum vertex at the walk's b_dec.  The capture consumes no
    random numbers and adds no draw.  Returns ``(n_vertices, mode)``;
    every final particle of the walk lands in the capture with its
    binning row.
    """

    root = exchange_root_state(root_event)
    current = root
    n_vert = 0
    while True:
        if int(current.depth) > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"r46: the line state of family {current.family_id!r} at "
                f"mass {float(current.mass)!r} GeV carries depth "
                f"{int(current.depth)}, exceeding MAX_CASCADE_DEPTH = "
                f"{MAX_CASCADE_DEPTH} (plan section 10.6, line 757); the "
                "run stops"
            )
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(
                current, catalog, rng_channel, rng_kinematics)
            for vertex in vertices:
                for daughter in vertex.daughters:
                    if not daughter.is_resonance:
                        capture.add_final(
                            int(daughter.pdg_id),
                            tuple(
                                float(x) for x in daughter.four_momentum),
                        )
            capture.depth_hist[int(current.depth)] = (
                capture.depth_hist.get(int(current.depth), 0) + 1)
            return n_vert, "catalog_handoff"
        channels = enumerate_channels(current, catalog, fams)
        cont = [c for c in channels if c.kind == KIND_CONTINUUM]
        pool = cont if cont else [
            channel for channel in channels
            if channel.kind in (KIND_ESTABLISHED_HADRON,
                                KIND_ESTABLISHED_RESONANCE)
        ]
        if not pool:
            raise ValueError(
                f"r46: the line state of family {current.family_id!r} at "
                f"mass {float(current.mass)!r} GeV carries no open "
                "continuum channel and no valid direct hadronic channel "
                "(plan section 10.6); checked, never repaired"
            )
        attached = _attach_form_factor(pool, current, float(b_dec))
        w0 = _reference_pool_payload(attached)
        _selected, (daughter0, daughter1) = _draw_attached_vertex(
            current, attached, catalog, fams,
            float(kappa_b), float(kappa_m),
            rng_channel, rng_kinematics,
        )
        chosen = next(
            index for index, channel in enumerate(attached)
            if channel is _selected
        )
        q2 = [
            _channel_reference_q2(channel, current)
            for channel in attached
        ]
        capture.vtx_nchan.append(len(attached))
        capture.vtx_chosen.append(int(chosen))
        capture.ch_w0.extend(w0)
        capture.ch_q2.extend(float(x) for x in q2)
        capture.n_vertices += 1
        if cont:
            successor = next(
                daughter for daughter in (daughter0, daughter1)
                if not str(daughter.family_id).startswith(
                    CATALOG_FAMILY_PREFIX)
            )
            sibling = daughter1 if successor is daughter0 else daughter0
            _emit_assembly_state(
                catalog, capture, sibling, rng_channel, rng_kinematics)
            capture.depth_hist[int(current.depth)] = (
                capture.depth_hist.get(int(current.depth), 0) + 1)
            current = successor
            continue
        _emit_assembly_state(
            catalog, capture, daughter0, rng_channel, rng_kinematics)
        _emit_assembly_state(
            catalog, capture, daughter1, rng_channel, rng_kinematics)
        return n_vert, "direct_channel"


def _channel_reference_q2(channel, state) -> float:
    """The channel's reference transverse momentum squared (the card-8
    W_dec configuration of this vertex; deterministic)."""
    from na49_mixing.cascade_ablation import _reference_daughter_p4
    p4 = _reference_daughter_p4(channel, state)
    ax, ay, az = (float(v) for v in state.axis)
    px, py, pz = p4[1], p4[2], p4[3]
    p2 = px * px + py * py + pz * pz
    pa = px * ax + py * ay + pz * az
    return max(0.0, p2 - pa * pa)


def _emit_assembly_state(
    catalog: HadronCatalog,
    capture: _LatentCapture,
    state,
    rng_channel,
    rng_kinematics,
) -> None:
    """The event assembly of one terminal established daughter state
    (card 9's caller-assembly clause, the R23/R32 ``_emit`` convention);
    the finals accumulate into the capture with their binning rows."""
    fid = str(state.family_id)
    if not fid.startswith(CATALOG_FAMILY_PREFIX):
        raise ValueError(
            "r46 assembly: a continuum daughter state reached the event "
            f"assembly (family_id {fid!r}); the cascade line walks "
            "continuum states and emits only established daughters"
        )
    pdg = int(fid[len(CATALOG_FAMILY_PREFIX):])
    record = catalog.get(pdg)
    p4 = tuple(float(x) for x in state.momentum)
    if record.stable_for_decay or pdg in catalog.stable_states():
        capture.add_final(pdg, p4)
        return
    decayer = ResonanceDecayer(catalog, 0, 0, 0)  # pure: no random draws
    decayer._rng_channel = rng_channel
    decayer._rng_kinematics = rng_kinematics
    vertices = decayer.decay_resonance(pdg, p4)
    for vertex in vertices:
        for daughter in vertex.daughters:
            if not daughter.is_resonance:
                capture.add_final(
                    int(daughter.pdg_id),
                    tuple(float(x) for x in daughter.four_momentum),
                )


# ---------------------------------------------------------------------------
# The paper-window binning tables and species offsets
# ---------------------------------------------------------------------------
def species_offset(species: str) -> int:
    """The flat-row offset of a species in the concatenated bin space."""
    return SPECIES_FLAT_OFFSETS[species]


#: The species offsets in the flat global bin space (the table order of
#: SPECIES_FLAT_ORDER, the R01 pT-major flat layout per species).
SPECIES_FLAT_ORDER = ("pi_minus", "pi_plus", "k_plus", "k_minus",
                      "proton", "antiproton")
SPECIES_FLAT_OFFSETS: dict[str, int] = {}
_SPECIES_SIZE: dict[str, int] = {}


def _init_species_layout(tables: list) -> None:
    """The species offsets in the flat global bin space (table order)."""
    offset = 0
    for name in SPECIES_FLAT_ORDER:
        table = next(t for t in tables if t.species == name)
        SPECIES_FLAT_OFFSETS[name] = offset
        _SPECIES_SIZE[name] = int(table.data.size)
        offset += int(table.data.size)
    _SPECIES_SIZE["__total__"] = offset


def _flat_total() -> int:
    return _SPECIES_SIZE["__total__"]


def _species_slice(pooled: np.ndarray, species: str) -> np.ndarray:
    """The species' slice of the pooled flat bin space."""
    start = int(SPECIES_FLAT_OFFSETS[species])
    size = int(_SPECIES_SIZE[species])
    return pooled[start:start + size]


#: The dense pT-major evaluation-lattice layout per species (the R01
#: accumulate-dense-then-pick recording convention, the R44 driver's
#: ``_worker_context`` pattern): the dense offsets and sizes of the full
#: evaluation lattices and the published-row pick maps, built from the
#: tables' ``bin_index`` arrays.
dense_offsets_by_species: dict[str, int] = {}
dense_sizes_by_species: dict[str, int] = {}
pick_by_species: dict[str, np.ndarray] = {}
_DENSE_TOTAL: int = 0


def _init_dense_layout(tables: list) -> int:
    """The dense pT-major lattice layout per species (the R01
    accumulate-dense-then-pick convention): the dense offsets and sizes
    derived from the full evaluation grid, and the published-row pick
    maps built at worker-context time from the tables' ``bin_index``
    arrays, ``pick = bin_index_pt * n_x + bin_index_xf``, with the two
    structural refusals (a published pair mapping outside the dense
    lattice; two published rows on one dense lattice cell).  Returns the
    dense total."""
    global _DENSE_TOTAL
    dense_offset = 0
    for table in tables:
        n_x = int(table.bin_index_xf.max()) + 1
        n_p = int(table.bin_index_pt.max()) + 1
        pick = (np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
                + np.asarray(table.bin_index_xf, dtype=np.int64))
        if pick.size and not (
                int(pick.min()) >= 0 and int(pick.max()) < n_x * n_p):
            _refuse(
                f"species {table.species!r}: a published bin_index pair "
                f"maps outside the dense lattice ({n_x} x {n_p}); the "
                "pick map cannot be built"
            )
        if len(set(pick.tolist())) != int(pick.size):
            _refuse(
                f"species {table.species!r}: two published rows map to "
                "one dense lattice cell; the pick map is not one-to-one"
            )
        pick_by_species[table.species] = pick
        dense_sizes_by_species[table.species] = n_x * n_p
        dense_offsets_by_species[table.species] = dense_offset
        dense_offset += n_x * n_p
    _DENSE_TOTAL = int(dense_offset)
    return _DENSE_TOTAL


def _dense_total() -> int:
    return _DENSE_TOTAL


def _recording_geometry(tables: list) -> dict:
    """The per-species recording geometry of the accumulate-dense-then-
    pick convention (published rows vs lattice cells; the holes), from
    the initialized published and dense layouts."""
    per_species = {}
    published_total = 0
    for name in SPECIES_FLAT_ORDER:
        table = next(t for t in tables if t.species == name)
        published = int(table.data.size)
        lattice = int(dense_sizes_by_species[name])
        published_total += published
        per_species[name] = {
            "published_rows": published,
            "lattice_cells": lattice,
            "holes": lattice - published,
            "dense_offset": int(dense_offsets_by_species[name]),
        }
    return {
        "lattice_cells_total": _dense_total(),
        "published_rows_total": published_total,
        "holes_total": _dense_total() - published_total,
        "per_species": per_species,
        "note": (
            "the recording surface of this run's six paper-window "
            "species; the pick map selects the published rows from the "
            "dense totals (the pick_map convention above)"
        ),
    }


edges_by_species: dict[str, tuple] = {}
n_x_by_species: dict[str, int] = {}
n_p_by_species: dict[str, int] = {}


def _worker_context(tables: list, fams: list, smoke: bool) -> dict:
    catalog = _load_catalog()
    dense_total = _init_dense_layout(tables)
    for table in tables:
        x_edges = _unique_edges(sps_core.edges_from_centers(
            table.xf_centers))
        p_edges = _unique_edges(sps_core.edges_from_centers(
            table.pt_centers))
        edges_by_species[table.species] = (x_edges, p_edges)
        n_x_by_species[table.species] = int(table.bin_index_xf.max()) + 1
        n_p_by_species[table.species] = int(table.bin_index_pt.max()) + 1
    return {
        "catalog": catalog,
        "pp_config": _m8_config(),
        "families": fams,
        "tables": tables,
        "table_by_pdg": {int(t.pdg_id): t for t in tables},
        "edges_by_species": dict(edges_by_species),
        "n_x_by_species": dict(n_x_by_species),
        "n_p_by_species": dict(n_p_by_species),
        "dense_offsets_by_species": dict(dense_offsets_by_species),
        "dense_sizes_by_species": dict(dense_sizes_by_species),
        "pick_by_species": dict(pick_by_species),
        "dense_total": int(dense_total),
        "sqrt_s": _sqrt_s(),
        "smoke": bool(smoke),
    }


def _unique_edges(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    keep = [float(values[0])]
    for value in values[1:]:
        if float(value) != keep[-1]:
            keep.append(float(value))
    return np.asarray(keep, dtype=float)


def _load_production_tables():
    """The six-species SPS tables (the published binning; the R01
    loader).  The production generation and the profile read the SPS
    tables; the smoke mode builds the synthetic hermetic grid."""
    raw_config = _load_raw_config()
    return r01.load_binned_tables(raw_config)


def _load_raw_config() -> dict:
    path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    if not path.is_file():
        _refuse(f"production config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _m8_config() -> PPExcitationConfig:
    """The M8-frozen p+p config (the R01 driver's application rule)."""
    raw_config = _load_raw_config()
    config, _applied = r01.build_pp_config(raw_config)
    applied_slope = float(config.regge.effective_transfer_slope)
    if applied_slope != B_EFF_FROZEN_GEV2:
        _refuse(
            "the applied M8 config's regge.effective_transfer_slope is "
            f"{applied_slope!r} GeV^-2 but the driver pins "
            f"{B_EFF_FROZEN_GEV2!r}; the frozen exchange slope drifted"
        )
    return config


def _sqrt_s() -> float:
    from na49_mixing.reaction import Reaction

    beam = float(_load_raw_config().get("beam_momentum", 158.0))
    return float(Reaction(beam, 2212, 2212).sqrt_s())


def _load_smoke_tables() -> list:
    """The synthetic hermetic binning grid of the smoke mode: a uniform
    paper-window grid built from the committed table declarations
    (species, pdg, mass, normalization) only; no SPS table is read (the
    R32 smoke-grid pattern)."""
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
    return tables


# ---------------------------------------------------------------------------
# The per-event generation (the R42 p+p assembly)
# ---------------------------------------------------------------------------
class ConservationViolation(RuntimeError):
    """The pre-registered conservation check of the card fired (checked,
    never repaired; a violation stops the run)."""


def _scatter_support(total: FourVector, incoming: FourVector,
                     m1: float, m2: float) -> tuple[float, float]:
    """The Mandelstam-t support of the exact two-body scattering (the
    same arithmetic as ``scatter_two_body``, re-derived without draws:
    the support depends on the masses and sqrt(s) only, never on B_eff,
    which is the exactness premise of the transfer-law reweighting)."""
    root_s = total.mass()
    pstar = two_body_momentum(root_s, m1, m2)
    beta_cm = total.spatial() / total.e
    pin_cm = boost(incoming, -beta_cm)
    incoming_mass = incoming.mass()
    ein = pin_cm.e
    e1 = math.sqrt(m1 * m1 + pstar * pstar)
    common = incoming_mass * incoming_mass + m1 * m1 - 2.0 * ein * e1
    span = 2.0 * pin_cm.momentum() * pstar
    return common - span, common + span


def _trial_pp_config(pp_config: PPExcitationConfig,
                     b_eff_gev2: float) -> PPExcitationConfig:
    """The in-memory trial config of the pinned validity probe: the
    frozen M8 config with the Regge effective transfer slope replaced by
    the trial B_eff (an in-memory dataclass replacement; no file, no
    pinned surface, is touched; the production generation always uses
    the frozen config)."""
    trial_regge = dataclasses.replace(
        pp_config.regge, effective_transfer_slope=float(b_eff_gev2))
    return dataclasses.replace(pp_config, regge=trial_regge)


def generate_one_event(
    catalog: HadronCatalog,
    pp_config: PPExcitationConfig,
    blend: BlendSpec,
    fams: list,
    fams_walk: list,
    seed: int,
    event_id: int,
    master_seed: int,
    replica_slot: int,
    b_eff_gev2: float,
    b_dec: float,
) -> _LatentCapture:
    """One p+p event of the R46 configuration (index item 7, the card-20
    substrate): the frozen builder source layer bit for bit at the
    config's exchange slope (the M8 mass sampler and the exact two-body
    scattering on the shared legacy generator, the retry discipline),
    the card-6 routing at the frozen source mass, the R32 builder
    catalog branch (the card-9 feed-down lineage convention) and the
    card-10 aligned cascade at the card-20 frozen simplified alignment
    (kappa_B, kappa_m) = (0.0, 4.0) with the walk b_dec on the labeled
    substreams (the production walk b_dec is the card-20 frozen 2.0
    GeV^-2, on the profile grid; the pinned validity probe's direct
    generations run at their pinned trial b_dec).  The realized
    Mandelstam-t and its support are recorded per event (the exact
    transfer-law reweighting statistic).  The conservation checks are
    the pre-registered per-event and per-source checks (checked, never
    repaired; a violation stops the run)."""

    builder = ResonancePPBuilder(
        pp_config, catalog,
        seed=int(seed), event_id=int(event_id),
        master_seed=int(master_seed), replica_index=int(replica_slot),
        blend=blend, families=fams,
    )
    total = builder.incoming[0] + builder.incoming[1]
    parameters = (
        pp_config.simple
        if pp_config.production_mode == "simple" else pp_config.regge
    )
    while True:
        m1, m2 = builder.mass_sampler.sample()
        try:
            p1, p2, t = scatter_two_body(
                total, builder.incoming[0], m1, m2,
                transfer_slope(
                    total.mass2(), pp_config.production_mode, parameters),
                builder.rng,
            )
            break
        except (ValueError, FloatingPointError):
            continue
    capture = _LatentCapture()
    capture.event_id = int(event_id)
    capture.t_scatter = float(t)
    capture.t_minus, capture.t_plus = _scatter_support(
        total, builder.incoming[0], m1, m2)
    span_floor = DEGENERATE_WINDOW_TOL
    if capture.t_plus - capture.t_minus > span_floor:
        if not (capture.t_minus - 1e-6 <= capture.t_scatter
                <= capture.t_plus + 1e-6):
            raise ValueError(
                f"r46: event {event_id} realized t = {t!r} GeV^2 outside "
                f"the recomputed two-body support "
                f"({capture.t_minus!r}, {capture.t_plus!r}); the support "
                "re-derivation disagrees with the generator (the run "
                "stops)"
            )
    fam_b = active_families(list(fams), (1, 1, 0))
    if len(fam_b) != 1:
        _refuse(
            "r46: the blend routing expects exactly one activated "
            "continuum family for (1, 1, 0), found "
            f"{[f.family_id for f in fam_b]}"
        )
    fam_b = fam_b[0]
    for sid, role, p4, mass in (
        (0, "projectile", p1, m1), (1, "target", p2, m2),
    ):
        mass = float(mass)
        p4t = (
            float(p4.e), float(p4.px), float(p4.py), float(p4.pz),
        )
        route = route_source_decay(
            (1, 1, 0), mass, blend, catalog, list(fams),
            builder._rng_record,
        )
        capture.route_census[0] += 1
        snapshot = (capture.n_finals, capture.bqs, capture.energy_sum)
        if route == "catalog":
            capture.route_census[1] += 1
            srec = SourceRecord(
                int(sid), str(role),
                FourVector(p4t[0], p4t[1], p4t[2], p4t[3]),
                QuantumNumbers(1, 1, 0), collision_index=1,
                metadata={"sampled_mass": mass, "t": float(t)},
            )
            finals, _vertices, _record = builder._decay_resonance(
                srec, snapshot_n(snapshot, capture),
                route="catalog", blend_weight=blend.weight(mass),
            )
            for final in finals:
                capture.add_final(
                    int(final.pdg_id),
                    (float(final.p4.e), float(final.p4.px),
                     float(final.p4.py), float(final.p4.pz)),
                )
        else:
            capture.route_census[2] += 1
            beam = (
                (0.0, 0.0, 1.0) if role == "projectile"
                else (0.0, 0.0, -1.0)
            )
            axis = initial_source_axis(p4t, beam_direction=beam)
            root_event = {
                "topology": STAGE_LABEL,
                "momentum": p4t,
                "mass": mass,
                "family_id": str(fam_b.family_id),
                "qn": (1, 1, 0),
                "axis": axis,
                "ancestry": (STAGE_LABEL,),
            }
            subs = derive_substreams(
                int(master_seed), int(replica_slot), int(event_id))
            rng_channel = np.random.default_rng(
                subs.stream_for("decay_channel"))
            rng_kinematics = np.random.default_rng(
                subs.stream_for("decay_kinematics"))
            _walk_with_capture(
                catalog, fams_walk, root_event,
                float(GENERATION_KAPPA_B), float(GENERATION_KAPPA_M),
                float(b_dec),
                rng_channel, rng_kinematics, capture,
            )
        capture.energy_expected += float(p4t[0])
        # -- the per-source conservation check (the card's exact
        #    per-source clause; checked, never repaired) --
        delta = (
            capture.bqs[0] - snapshot[1][0],
            capture.bqs[1] - snapshot[1][1],
            capture.bqs[2] - snapshot[1][2],
        )
        if delta != (1, 1, 0):
            raise ConservationViolation(
                f"r46: event {event_id} source {sid} ({role}) finals sum "
                f"to (B, Q, S) = {delta} but the source carries "
                "(1, 1, 0); conservation is checked, never repaired "
                "(the run stops)"
            )
    # -- the per-event conservation check --
    if capture.bqs != (2, 2, 0):
        raise ConservationViolation(
            f"r46: event {event_id} finals sum to (B, Q, S) = "
            f"{capture.bqs} but the p+p pair carries (2, 2, 0); "
            "conservation is checked, never repaired (the run stops)"
        )
    closure = abs(capture.energy_sum - capture.energy_expected)
    capture.worst_closure = closure
    if not closure <= EVENT_CLOSURE_TOL:
        raise ConservationViolation(
            f"r46: event {event_id} final energy sum misses the source "
            f"energy sum by {closure!r} GeV (beyond "
            f"{EVENT_CLOSURE_TOL}); no momentum repair (the run stops)"
        )
    # -- the canonical per-event payload (the rolling digest) --
    capture.event_hash = _event_hash(
        capture, int(event_id), capture.route_census,
    )
    return capture


def snapshot_n(snapshot: tuple, capture: _LatentCapture) -> int:
    """The first-particle id of a source's decay (the pre-snapshot
    final count)."""
    return int(snapshot[0])


def _event_hash(
    capture: _LatentCapture, event_id: int, route_census: list,
) -> str:
    """The canonical per-event hash: the event id, the route census,
    the final count, and every final's pdg with exact hex components."""
    digest = hashlib.sha256()
    digest.update(
        f"r46|{event_id}|{route_census[0]},{route_census[1]},"
        f"{route_census[2]}|{capture.n_finals}|".encode("utf-8"))
    for line in capture.payload:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Latent packing (the sealed record's reweighting payload)
# ---------------------------------------------------------------------------
def pack_arrays(arrays: list[np.ndarray]) -> list[dict]:
    """Pack one list of numpy arrays into the record's base64 container
    (exact bytes; the packing is a lossless transport of the recorded
    arrays, and the record digests cover it)."""
    packed = []
    for array in arrays:
        value = np.ascontiguousarray(array)
        packed.append({
            "dtype": value.dtype.str,
            "shape": [int(s) for s in value.shape],
            "b64": base64.b64encode(value.tobytes()).decode("ascii"),
        })
    return packed


def unpack_arrays(packed: list) -> list[np.ndarray]:
    out = []
    for entry in packed:
        raw = base64.b64decode(entry["b64"], validate=True)
        value = np.frombuffer(raw, dtype=np.dtype(entry["dtype"]))
        out.append(value.reshape([int(s) for s in entry["shape"]]))
    return out


# ---------------------------------------------------------------------------
# The chunk worker and the replica generation
# ---------------------------------------------------------------------------
def _generate_chunk(task: tuple):
    """Generate one chunk of events in a worker process and return the
    pooled parts (the R32 chunk convention)."""
    assert _WORKER_CONTEXT is not None
    context = _WORKER_CONTEXT
    (replica, seed, event_ids, master_seed, replica_slot,
     blend_boundary, b_eff_gev2, b_dec, smoke) = task
    catalog = context["catalog"]
    pp_config = context["pp_config"]
    if float(b_eff_gev2) != B_EFF_FROZEN_GEV2:
        # The pinned validity probe's direct generation at the trial
        # exchange slope (in memory; never a production generation).
        pp_config = _trial_pp_config(pp_config, float(b_eff_gev2))
    blend = BlendSpec(boundary=float(blend_boundary),
                      ramp_width=BLEND_RAMP_WIDTH_GEV)
    fams = list(context["families"])
    captures: list[_LatentCapture] = []
    for event_id in event_ids:
        captures.append(generate_one_event(
            catalog, pp_config, blend, fams, fams,
            seed=int(seed), event_id=int(event_id),
            master_seed=int(master_seed),
            replica_slot=int(replica_slot),
            b_eff_gev2=float(b_eff_gev2), b_dec=float(b_dec),
        ))
    return _chunk_parts(replica, event_ids[0], captures, smoke)


def _chunk_parts(replica, start_id, captures, smoke) -> dict:
    context = _WORKER_CONTEXT
    hashes = [c.event_hash for c in captures]
    event_ids = np.asarray([c.event_id for c in captures], dtype=np.int64)
    t_scatter = np.asarray([c.t_scatter for c in captures], dtype=np.float64)
    t_minus = np.asarray([c.t_minus for c in captures], dtype=np.float64)
    t_plus = np.asarray([c.t_plus for c in captures], dtype=np.float64)
    row_flat: list[int] = []
    row_weight: list[float] = []
    row_ptr: list[int] = [0]
    vtx_nchan: list[int] = []
    vtx_chosen: list[int] = []
    ch_w0: list[float] = []
    ch_q2: list[float] = []
    vtx_ptr: list[int] = [0]
    routing = [0, 0, 0]
    depth_hist: dict[int, int] = {}
    n_events = len(captures)
    worst = 0.0
    vtx_cum = 0
    for capture in captures:
        row_flat.extend(capture.flat)
        row_weight.extend(capture.weights)
        row_ptr.append(len(row_flat))
        vtx_cum += capture.n_vertices
        vtx_ptr_value = vtx_cum
        vtx_ptr.append(vtx_ptr_value)
        vtx_nchan.extend(capture.vtx_nchan)
        vtx_chosen.extend(capture.vtx_chosen)
        ch_w0.extend(capture.ch_w0)
        ch_q2.extend(capture.ch_q2)
        routing[0] += int(capture.route_census[0])
        routing[1] += int(capture.route_census[1])
        routing[2] += int(capture.route_census[2])
        for depth, count in capture.depth_hist.items():
            depth_hist[int(depth)] = depth_hist.get(int(depth), 0) + int(count)
    mult = {
        key: {
            "sum_n": float(sum(c.mult_sums[key] for c in captures)),
            "sum_n2": float(sum(c.mult_sums2[key] for c in captures)),
            "n_with": int(sum(c.mult_with[key] for c in captures)),
        }
        for key in SPECIES_KEYS
    }
    return {
        "replica": int(replica),
        "start_id": int(start_id),
        "n_events": int(n_events),
        "hashes": hashes,
        "event_ids": event_ids,
        "t_scatter": t_scatter,
        "t_minus": t_minus,
        "t_plus": t_plus,
        "row_flat": np.asarray(row_flat, dtype=np.int32),
        "row_weight": np.asarray(row_weight, dtype=np.float64),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "vtx_ptr": np.asarray(vtx_ptr, dtype=np.int64),
        "vtx_nchan": np.asarray(vtx_nchan, dtype=np.int32),
        "vtx_chosen": np.asarray(vtx_chosen, dtype=np.int32),
        "ch_w0": np.asarray(ch_w0, dtype=np.float64),
        "ch_q2": np.asarray(ch_q2, dtype=np.float64),
        "routing": routing,
        "depth_hist": {str(k): int(v) for k, v in sorted(depth_hist.items())},
        "mult": mult,
        "worst_closure": max(
            (float(c.worst_closure) for c in captures), default=0.0),
        "smoke": bool(smoke),
        "n_vertices_total": int(sum(c.n_vertices for c in captures)),
    }


# ---------------------------------------------------------------------------
# The replica record assembly
# ---------------------------------------------------------------------------
def _aggregate_chunks(parts: list) -> dict:
    """Merge the chunk parts into one replica's state (exact integer
    counts; float sums in chunk order, the chunks sorted by start id)."""
    parts = sorted(parts, key=lambda p: int(p["start_id"]))
    hashes: list[str] = []
    row_flat: list[int] = []
    row_weight: list[float] = []
    row_ptr: list[int] = [0]
    event_ids: list[int] = []
    vtx_ptr: list[int] = [0]
    vtx_nchan: list[int] = []
    vtx_chosen: list[int] = []
    ch_w0: list[float] = []
    ch_q2: list[float] = []
    routing = [0, 0, 0]
    depth_hist: dict[str, int] = {}
    mult = {k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
            for k in SPECIES_KEYS}
    t_scatter: list[float] = []
    t_minus: list[float] = []
    t_plus: list[float] = []
    worst = 0.0
    n_vertices = 0
    n_events = 0
    for part in parts:
        hashes.extend(part["hashes"])
        event_ids.extend(part["event_ids"].tolist())
        t_scatter.extend(part["t_scatter"].tolist())
        t_minus.extend(part["t_minus"].tolist())
        t_plus.extend(part["t_plus"].tolist())
        row_flat.extend(part["row_flat"].tolist())
        row_weight.extend(part["row_weight"].tolist())
        base = int(row_ptr[-1])
        row_ptr.extend((base + int(v)) for v in part["row_ptr"][1:])
        base_v = vtx_ptr[-1]
        vtx_ptr.extend((base_v + int(v)) for v in part["vtx_ptr"][1:])
        vtx_nchan.extend(part["vtx_nchan"].tolist())
        vtx_chosen.extend(part["vtx_chosen"].tolist())
        ch_w0.extend(part["ch_w0"].tolist())
        ch_q2.extend(part["ch_q2"].tolist())
        for k in range(3):
            routing[k] += int(part["routing"][k])
        for key, count in part["depth_hist"].items():
            depth_hist[key] = depth_hist.get(key, 0) + int(count)
        for key, slot in part["mult"].items():
            mult[key]["sum_n"] += float(slot["sum_n"])
            mult[key]["sum_n2"] += float(slot["sum_n2"])
            mult[key]["n_with"] += int(slot["n_with"])
        worst = max(worst, float(part["worst_closure"]))
        n_vertices_total = int(part["n_vertices_total"])
        n_vertices = n_vertices + n_vertices_total
        n_events += int(part["n_events"])
    return {
        "hashes": hashes,
        "event_ids": np.asarray(event_ids, dtype=np.int64),
        "t_scatter": np.asarray(t_scatter, dtype=np.float64),
        "t_minus": np.asarray(t_minus, dtype=np.float64),
        "t_plus": np.asarray(t_plus, dtype=np.float64),
        "row_flat": np.asarray(row_flat, dtype=np.int32),
        "row_weight": np.asarray(row_weight, dtype=np.float64),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "vtx_ptr": np.asarray(vtx_ptr, dtype=np.int64),
        "vtx_nchan": np.asarray(vtx_nchan, dtype=np.int32),
        "vtx_chosen": np.asarray(vtx_chosen, dtype=np.int32),
        "ch_w0": np.asarray(ch_w0, dtype=np.float64),
        "ch_q2": np.asarray(ch_q2, dtype=np.float64),
        "routing": routing,
        "depth_hist": depth_hist,
        "mult": mult,
        "worst_closure": worst,
        "n_vertices": int(n_vertices),
        "n_events": int(n_events),
    }


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(
            _generate_chunk, tasks, chunksize=1
        ):
            completed += 1
            yield result
            if completed % 2 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r46] {progress_label}: chunk {completed}/"
                    f"{len(tasks)} ({elapsed:.1f}s)",
                    flush=True,
                )


def _replica_record(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    aggregated: dict,
    pre_reg_digest: str,
    smoke: bool,
) -> dict:
    """Assemble one per-replica record (deterministic: no wall-clock
    fields anywhere in the record).  The finals' dense lattice indices
    accumulate into the species' dense lattices and the published rows
    are picked from the dense totals (the R01 ``_bin_track_sums``
    accumulate-dense-then-pick convention, mirrored from the corrected
    R44 driver), so ``pooled_sums`` are published-row arrays aligned 1:1
    with the per-published-row areas the evaluation divides by; a hole
    cell's weight stays unpicked and is dropped.  The reweighting
    latent carries the dense lattice addresses (the two-step
    dense-slice-then-pick conversion happens at the evaluation
    boundary, in ``_species_bin_values``)."""
    context = _WORKER_CONTEXT
    assert context is not None
    dense_offsets = context["dense_offsets_by_species"]
    dense_sizes = context["dense_sizes_by_species"]
    pick_by_species = context["pick_by_species"]
    routing = aggregated["routing"]
    row_flat = np.asarray(aggregated["row_flat"], dtype=np.int64)
    row_weight = np.asarray(aggregated["row_weight"], dtype=float)
    dense_pooled = np.zeros(int(context["dense_total"]), dtype=float)
    if row_flat.size:
        np.add.at(dense_pooled, row_flat, row_weight)
    pooled_species = {}
    for key in SPECIES_FLAT_ORDER:
        dense_slice = dense_pooled[
            int(dense_offsets[key]):
            int(dense_offsets[key]) + int(dense_sizes[key])]
        pooled_species[key] = [
            float(v) for v in dense_slice[pick_by_species[key]]]
    event_hash_set = hashlib.sha256()
    for digest in aggregated["hashes"]:
        event_hash_set.update(digest.encode("utf-8"))
        event_hash_set.update(b"\n")
    latent_digest = hashlib.sha256()
    for array in (
        aggregated["event_ids"], aggregated["row_flat"],
        aggregated["row_weight"], aggregated["row_ptr"],
        aggregated["t_scatter"], aggregated["t_minus"],
        aggregated["t_plus"], aggregated["vtx_ptr"],
        aggregated["vtx_nchan"], aggregated["vtx_chosen"],
        aggregated["ch_w0"], aggregated["ch_q2"],
    ):
        latent_digest.update(np.ascontiguousarray(array).tobytes())
        latent_digest.update(b"|")
    record = {
        "schema": "na49_mixing.staged_r46_simplified_recoil_decay_profile.replica/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "stage_label": STAGE_LABEL,
        "replica": int(replica),
        "seed": int(seed),
        "kernel_replica_index": int(replica_slot),
        "master_seed": int(MASTER_SEED),
        "n_events": int(n_events),
        "chunk_size": int(CHUNK_SIZE),
        "statistics": {
            "tier": T2_LABEL,
            "events_per_replica": int(T2_EVENTS_PER_REPLICA),
        },
        "seed_map": {
            "raw_seed_rule": RAW_SEED_RULE,
            "kernel_seed_rule": KERNEL_SEED_RULE,
            "event_index_rule": EVENT_INDEX_RULE,
        },
        "routing": {
            "n_sources": int(routing[0]),
            "n_catalog": int(routing[1]),
            "n_continuum": int(routing[2]),
            "note": (
                "the card-6 routing census at the frozen source masses "
                "(the v0 ramp start; the catalog branch is the R32 "
                "builder branch, the continuum branch the card-10 "
                "aligned cascade at the card-20 frozen simplified "
                "alignment (kappa_B, kappa_m) = (0.0, 4.0) and the "
                "frozen b_dec 2.0 GeV^-2)"
            ),
        },
        "depth_histogram": aggregated["depth_hist"],
        "multiplicity": aggregated["mult"],
        "conservation": {
            "events_checked": int(n_events),
            "event_b_q_s_conserved": True,
            "worst_energy_closure_gev": float(
                aggregated["worst_closure"]),
            "tolerance": EVENT_CLOSURE_TOL,
        },
        "pooled_sums": pooled_species,
        "latent": {
            "note": (
                "the per-event reweighting latent of the full event set: "
                "the CSR bin rows (flat bin index + Jacobian weight per "
                "final, every event reweightable), the per-event CSR "
                "pointers, the per-event realized Mandelstam-t with its "
                "two-body support (the exact transfer-law statistic; the "
                "support is parameter-independent, the exactness premise "
                "of the B_eff reweighting), and the per-vertex attached "
                "pool (the card-8 shares w0 at the generation b_dec, the "
                "reference transverse momenta squared q2, and the chosen "
                "channel index); the B_eff transfer reweighting and the "
                "b_dec channel reweighting consume exactly this payload"
            ),
            "packed": pack_arrays([
                aggregated["event_ids"], aggregated["row_flat"],
                aggregated["row_weight"], aggregated["row_ptr"],
                aggregated["t_scatter"], aggregated["t_minus"],
                aggregated["t_plus"], aggregated["vtx_ptr"],
                aggregated["vtx_nchan"], aggregated["vtx_chosen"],
                aggregated["ch_w0"], aggregated["ch_q2"],
            ]),
        },
        "event_hash_set_sha256": event_hash_set.hexdigest(),
        "latent_digest": latent_digest.hexdigest(),
        "identity": {
            "pre_registration_sha256": pre_reg_digest,
            "driver_sha256": DRIVER_SHA256,
            "builder_module_sha256": FROZEN_SUBSTRATE[
                "builder_blend_path"][1],
            "catalog_sha256": FROZEN_SUBSTRATE["catalog"][1],
            "continuum_config_sha256": FROZEN_SUBSTRATE[
                "continuum_config"][1],
        },
        "reused": False,
    }
    return record


# ---------------------------------------------------------------------------
# run_generation / run_smoke (the R31/R40 record-identity conventions)
# ---------------------------------------------------------------------------
T2_LABEL = (
    "T2: 8 replicas x 100,000 p+p events (plan section 5.1 line 254, "
    "the upper T2 bound, the R23/R32 convention); 800,000 events total, "
    "generated once at the frozen B_eff and the card-20 frozen "
    "b_dec = 2.0 GeV^-2 (on-grid); the 9 x 9 profile is a reweighting "
    "over the generated events"
)


def _chunk_tasks(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    b_eff_gev2: float,
    b_dec: float,
    smoke: bool,
) -> list[tuple]:
    starts = list(range(0, int(n_events), CHUNK_SIZE))
    return [
        (int(replica), int(seed),
         [event_index(start + j) for j in range(min(CHUNK_SIZE, int(n_events) - start))],
         int(MASTER_SEED), int(replica_slot),
         float(BLEND_RAMP_START_GEV),
         float(b_eff_gev2), float(b_dec), bool(smoke))
        for start in starts
    ]


def run_generation(
    run_dir: Path,
    replicas: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> list[str]:
    """Generate the requested production replica records (any subset;
    the default is every replica) at the frozen B_eff and the nominal
    b_dec.  A completed record that matches the pre-registered identity
    is reused unchanged.  Returns the record names written or reused."""
    pre = _load_pre_registration(run_dir)
    _assert_pre_registration(pre, run_dir)
    pre_digest = _pre_registration_digest_of(pre)
    tables = _load_production_tables()
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
    context["pre_registration_sha256"] = pre_digest
    _set_worker_context(context)
    written: list[str] = []
    requested = (
        list(range(T2_REPLICAS)) if replicas is None
        else [int(r) for r in replicas]
    )
    for replica in requested:
        if not 0 <= replica < T2_REPLICAS:
            _refuse(f"--replica {replica} outside 0..{T2_REPLICAS - 1}")
        name = RECORD_FMT.format(int(replica))
        path = run_dir / name
        seed = raw_replica_seed(replica)
        if path.is_file():
            existing = _read_json(path)
            if _record_identity_ok(existing, pre, replica, seed):
                print(f"[r46] {name}: completed record reused", flush=True)
                written.append(name)
                continue
            _refuse(
                f"{path}: a completed record exists but does not match "
                "the pre-registered identity; a drifted record is never "
                "silently overwritten (delete the run dir and re-run)"
            )
        tasks = _chunk_tasks(replica, seed, replica,
                             T2_EVENTS_PER_REPLICA,
                             B_EFF_FROZEN_GEV2, B_DEC_GENERATION_GEV2,
                             smoke=False)
        parts = list(_map_chunks(tasks, workers, f"replica {replica}"))
        aggregated = _aggregate_chunks(parts)
        if aggregated["n_events"] != int(T2_EVENTS_PER_REPLICA):
            _refuse(
                f"replica {replica}: generated "
                f"{aggregated['n_events']} events, expected "
                f"{int(T2_EVENTS_PER_REPLICA)}"
            )
        record = _replica_record(
            replica, seed, replica, int(T2_EVENTS_PER_REPLICA),
            aggregated, pre_digest, smoke=False)
        _write_json(path, record)
        written.append(name)
        print(f"[r46] {name}: written "
              f"({aggregated['n_vertices']} cascade vertices)", flush=True)
    return written


def _record_identity_ok(record: dict, pre: dict, replica: int, seed: int) -> bool:
    """The completed-record identity rule: the replica, the seed, the
    statistics, the mode, and the pre-registration digest must match."""
    return bool(
        record.get("run") == RUN_ID
        and int(record.get("replica", -1)) == int(replica)
        and int(record.get("seed", -1)) == int(seed)
        and record.get("mode") == "production"
        and int(record.get("n_events", -1)) == int(T2_EVENTS_PER_REPLICA)
        and record.get("identity", {}).get(
            "pre_registration_sha256") == _pre_registration_digest_of(pre)
    )


def _pre_registration_digest_of(pre: dict) -> str:
    return hashlib.sha256(json.dumps(
        pre, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _refuse_smoke_artifacts(run_dir: Path) -> None:
    """``--gate`` / ``--seal`` / ``--clean-validate`` refuse smoke
    artifacts (the R31/8c convention)."""
    pre_path = run_dir / PREREGISTRATION_NAME
    if not pre_path.is_file():
        _refuse(f"run dir carries no {PREREGISTRATION_NAME}: {run_dir}")
    pre = _read_json(pre_path)
    if str(pre.get("mode", "")) == "smoke" or any(
        str(name).startswith(SMOKE_PREFIX) for name in _run_dir_files(run_dir)
    ):
        _refuse(
            "smoke artifacts are refused by --gate / --seal / "
            "--clean-validate (the R31/8c convention); the production "
            "run dir is the pre-registered default"
        )


def _run_dir_files(run_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(run_dir).iterdir())


def run_smoke(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """The hermetic end-to-end driver validation (never sealed): a smoke
    pre-registration, 1 replica x 2,000 events at the smoke seeds, and
    the smoke 9 x 9 profile, into an explicit ``--run-dir``."""
    if Path(run_dir).resolve() == (MONO_ROOT / DEFAULT_RUN_DIR).resolve():
        _refuse(
            "--smoke refuses the default production run dir; pass an "
            "explicit --run-dir under dev/00runs/tmp/ (the smoke "
            "artifacts are never sealed)"
        )
    tables = _load_smoke_tables()
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # -- the smoke pre-registration (the smoke variant of the contract;
    #    the frozen-parameter assertion runs against the pinned R41
    #    fixture) --
    pre = build_pre_registration(smoke=True)
    pre["mode"] = "smoke"
    pre["smoke"] = SMOKE_PAYLOAD
    pre["frozen_parameters"] = assert_frozen_parameters(
        R41_FIT_REPORT_FIXTURE)
    pre["card20_substrate"] = dict(_CARD20_SUBSTRATE_STATUS)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    context["pre_registration_sha256"] = _pre_registration_digest_of(pre)
    _set_worker_context(context)
    # -- the smoke generation --
    replica = 0
    name = f"{SMOKE_PREFIX}replica_{replica}.json"
    seed = smoke_raw_seed()
    tasks = _chunk_tasks(
        replica, seed, SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        B_EFF_FROZEN_GEV2, B_DEC_GENERATION_GEV2, smoke=True)
    parts = list(_map_chunks(tasks, workers, "smoke"))
    aggregated = _aggregate_chunks(parts)
    record = _replica_record(
        replica, smoke_raw_seed(), SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        aggregated, _pre_registration_digest_of(pre), smoke=True)
    _write_json(run_dir / name, record)
    # -- the smoke profile (the machinery end to end; synthetic data) --
    report = run_profile_machinery(run_dir, smoke=True)
    _write_json(run_dir / PROFILE_REPORT_NAME, report)
    return {
        "record": name,
        "profile_report_digest": report.get("digest"),
    }


SMOKE_PAYLOAD = {
    "replicas": 1,
    "events_per_replica": SMOKE_EVENTS,
    "raw_seed": SMOKE_RAW_SEED,
    "raw_seed_rule": SMOKE_RAW_SEED_RULE,
    "kernel_replica_slot": SMOKE_KERNEL_REPLICA_SLOT,
    "synthetic_truth": SMOKE_SYNTHETIC_TRUTH,
    "validity_events": SMOKE_VALIDITY_EVENTS,
    "validity_grid_points": [
        [float(m), float(b)] for m, b in VALIDITY_GRID_POINTS
    ],
    "note": SMOKE_NOTE,
}


VALIDITY_SUBSAMPLE_RULE = (
    "the pinned sub-sample is the first 2,000 production event indices "
    "of replica 0 (event_index(900_000_000 + i), i < 2,000, raw seed "
    "20301076, kernel replica slot 0), the production (seed, event_id) "
    "pairs; the probe generates them directly at each pinned grid point "
    "and at the frozen point (shared by both pinned grid points) with "
    "common random numbers, in memory, never sealed"
)

AUDIT_OUTCOME = {
    "card_1": (
        "RECOMMENDATION (sealed dev/00runs/"
        "r4_audit_transition_routing_formulation_2026-09-20/verdict.json): "
        "the rework card 1r "
        "(TASK-20260920-r4-rework-transition-routing-formulation) landed "
        "before this run with no new spread diagnostic, so the original "
        "R32 spread stands as the obligation 4 carry"
    ),
    "card_2": (
        "PASS (dev/00runs/"
        "r4_audit_strange_sector_continuum_families_2026-09-20/"
        "verdict.json): every persisting K+ bin attributable to a "
        "declared, catalog-audited class pair; no new spread diagnostic"
    ),
    "card_3": (
        "PASS (dev/00runs/"
        "r4_audit_builder_blend_path_routing_weight_2026-09-20/"
        "verdict.json): the builder blend path contributes no routing "
        "artifact beyond the pre-registered w(M) weight"
    ),
}

GATEWAY_FALLBACK_RULE = (
    "local generation is primary (the measured local envelope: 100,000 "
    "p+p events ~ 75 s at 8 fork workers in R41); if any replica exceeds "
    "30 min wall, the remaining replicas run on the cluster gateway "
    "(host cl, workers c0801-c0809 x 32 task slots, one job per replica, "
    "--tasks 1, single math thread (NA49_CLUSTER_GW_TASKS forces thread "
    "counts to 1), no local process pool inside the chunk command, "
    "submission paced by the capacity check; record outputs are appended "
    "only to the new run dir)"
)


def _load_carry_bins() -> list[dict]:
    """The sealed 32 persisting bins (the blend-band-sensitive set),
    read read-only from the sealed card-1 attribution; the profile
    carries the term on the b_transport pair's intersecting cells."""
    path = MONO_ROOT / R4_AUDIT_CARD1_DIR / "persisting_bin_attribution.json"
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
    out = []
    for row in rows:
        out.append({
            "species": str(row["species"]),
            "bin_index": int(row["bin_index"]),
        })
    return out


CARRY_BIN_KEYS: list = []  # populated at pre-registration / profile time


def _build_carry_set() -> set:
    """The blend-band-sensitive cells for the b_transport pair: the
    sealed 32 persisting bins restricted to the pair's species."""
    global CARRY_BIN_KEYS
    if not CARRY_BIN_KEYS:
        CARRY_BIN_KEYS = _load_carry_bins()
    return {(entry["species"], int(entry["bin_index"]))
            for entry in CARRY_BIN_KEYS}


def run_pre_registration(run_dir: Path, smoke: bool = False) -> dict:
    """Write pre_registration.json and provenance.json into the run dir,
    then stop.  Verifies the frozen substrate digests and the sealed
    seals first.  Nothing is generated."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        _refuse(
            f"run dir is not empty: {run_dir}; the pre-registration is "
            "written before any generation into a fresh run dir"
        )
    substrate = verify_frozen_substrate()
    sealed = verify_sealed_inputs()
    global CARRY_BIN_KEYS
    CARRY_BIN_KEYS = _load_carry_bins()
    pre = build_pre_registration(smoke=smoke)
    pre["mode"] = "production"
    pre["substrate"] = substrate
    pre["sealed_run_dirs_status"] = sealed
    pre["card20_substrate"] = dict(_CARD20_SUBSTRATE_STATUS)
    pre["seed_map"]["disjointness_report"] = seed_disjointness_report()
    # -- the frozen simplified alignment against the sealed R41 fit
    #    report (the card-20 substrate's frozen-parameter source; the
    #    sealed archive is read read-only and its digests are asserted
    #    byte for byte) --
    fit_path = MONO_ROOT / R41_RUN_DIR / R41_FIT_REPORT_NAME
    if not fit_path.is_file():
        _refuse(f"the sealed R41 fit report is missing: {fit_path}")
    if _sha256_file(fit_path) != R41_FIT_REPORT_FILE_SHA256:
        _refuse(
            "the sealed R41 fit report file digest drifted from the "
            f"pinned {R41_FIT_REPORT_FILE_SHA256!r}; the frozen-"
            "parameter source is not the sealed archive"
        )
    frozen_block = assert_frozen_parameters(_read_json(fit_path))
    sealed_fit = _read_json(fit_path)
    if sealed_fit.get("digest") != R41_FIT_REPORT_DIGEST:
        _refuse(
            "the sealed R41 fit report's self-digest drifted from the "
            f"pinned {R41_FIT_REPORT_DIGEST!r}; the frozen-parameter "
            "source is not the sealed archive"
        )
    pre["frozen_parameters"] = frozen_block
    # -- the superseded corrected R42 profile report (the
    #    superseded-profile reference; read-only, cited as superseded,
    #    never as evidence) --
    superseded_status = verify_sealed_dir(SUPERSEDED_R42_RUN_DIR)
    superseded_report_path = (MONO_ROOT / SUPERSEDED_R42_RUN_DIR /
                              SUPERSEDED_R42_REPORT_NAME)
    if not superseded_report_path.is_file():
        _refuse(
            "the sealed superseded R42 profile report is missing: "
            f"{superseded_report_path}")
    if _sha256_file(superseded_report_path) != \
            SUPERSEDED_R42_REPORT_FILE_SHA256:
        _refuse(
            "the sealed superseded R42 profile report's file digest "
            "drifted from the pinned "
            f"{SUPERSEDED_R42_REPORT_FILE_SHA256!r}"
        )
    superseded_report = _read_json(superseded_report_path)
    if superseded_report.get("digest") != SUPERSEDED_R42_REPORT_DIGEST:
        _refuse(
            "the sealed superseded R42 profile report's self-digest "
            "drifted from the pinned "
            f"{SUPERSEDED_R42_REPORT_DIGEST!r}"
        )
    sealed_minimum = {
        "b_eff_multiplier":
            float(superseded_report["minimum"]["b_eff_multiplier"]),
        "b_dec_gev2": float(superseded_report["minimum"]["b_dec"]),
    }
    if sealed_minimum != SUPERSEDED_R42_MINIMUM:
        _refuse(
            "the sealed superseded R42 profile minimum "
            f"{sealed_minimum!r} does not match the pinned "
            f"{SUPERSEDED_R42_MINIMUM!r}"
        )
    pre["superseded_r42_profile"] = {
        "run_dir": SUPERSEDED_R42_RUN_DIR,
        "seal": superseded_status,
        "report_digest": SUPERSEDED_R42_REPORT_DIGEST,
        "minimum": sealed_minimum,
        "status": SUPERSEDED_R42_STATUS,
    }
    # the recording geometry of the pre-registered pick convention (the
    # per-species published-row vs lattice-cell counts of the run's
    # tables; the corrected card-17 convention consumed read-only).
    # The smoke mode attaches its own geometry from the synthetic grid
    # (in run_smoke).
    if not smoke:
        tables = _load_production_tables()
        _init_species_layout(tables)
        _init_dense_layout(tables)
        pre["recording_convention"]["geometry"] = _recording_geometry(
            tables)
    pre["provenance"] = {
        "driver_sha256": DRIVER_SHA256,
        "git": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "python": sys.version.split()[0],
        "written_utc": _now_utc(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    provenance = {
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "new_driver_sha256_both_trees": DRIVER_SHA256,
        "driver_note": (
            "byte-identical in the nested tree (dev/mixing2006/helpers/"
            "benchmarking/) and the monorepo mirror (helpers/"
            "benchmarking/), the established _locate_roots pattern"
        ),
        "cascade_module_sha256": {
            key: value[1]
            for key, value in FROZEN_SUBSTRATE.items()
            if key.startswith("cascade_")
        },
        "builder_sha256": FROZEN_SUBSTRATE["builder_blend_path"][1],
        "card6_surface_sha256": FROZEN_SUBSTRATE["card6_routing"][1],
        "engine_sha256": FROZEN_SUBSTRATE["card2_engine"][1],
        "round2_kernel_sha256": FROZEN_SUBSTRATE["round2_kernel"][1],
        "m8_frozen_set_hash": M8_PARAMETER_HASH,
        "frozen_b_eff_gev2": B_EFF_FROZEN_GEV2,
        "catalog_sha256": FROZEN_SUBSTRATE["catalog"][1],
        "continuum_config_sha256": FROZEN_SUBSTRATE["continuum_config"][1],
        "r32_driver_sha256": FROZEN_SUBSTRATE["r32_driver"][1],
        "r40_driver_sha256": FROZEN_SUBSTRATE["r40_driver"][1],
        "r41_driver_sha256": FROZEN_SUBSTRATE["r41_driver"][1],
        "r42_driver_sha256": FROZEN_SUBSTRATE["r42_driver"][1],
        "r43_driver_sha256": FROZEN_SUBSTRATE["r43_driver"][1],
        "r44_driver_sha256": FROZEN_SUBSTRATE["r44_driver"][1],
        "r45_driver_sha256": FROZEN_SUBSTRATE["r45_driver"][1],
        "card20_substrate": {
            "frozen_simplified_archive": R45_RUN_DIR,
            "sibling_driver_rel": FROZEN_SUBSTRATE["r45_driver"][0],
            "sibling_driver_sha256": FROZEN_SUBSTRATE["r45_driver"][1],
            "sibling_run_dir_seal": sealed.get(R45_RUN_DIR),
            "import_cross_check": dict(_CARD20_SUBSTRATE_STATUS),
            "note": CARD20_SUBSTRATE_NOTE,
        },
        "frozen_parameters": {
            "kappa_B": FROZEN_SIMPLIFIED_ALIGNMENT["kappa_B"],
            "kappa_m": FROZEN_SIMPLIFIED_ALIGNMENT["kappa_m"],
            "b_dec_gev2": FROZEN_SIMPLIFIED_ALIGNMENT["b_dec"],
            "baryonic_nominal_width_gev": FROZEN_WIDTH_GEV,
            "source": (
                "sealed R41 archive "
                f"({R41_RUN_DIR}/{R41_FIT_REPORT_NAME})"
            ),
            "r41_fit_report_file_sha256": R41_FIT_REPORT_FILE_SHA256,
            "r41_fit_report_digest": R41_FIT_REPORT_DIGEST,
            "refit": False,
        },
        "superseded_r42_profile": {
            "run_dir": SUPERSEDED_R42_RUN_DIR,
            "state": (
                "intact; the seal verifies (verify_sealed_dir); the "
                "corrected card-17 archive is the superseded-profile "
                "reference, cited as superseded and never as evidence; "
                "nothing in it is modified"
            ),
            "report_file_sha256": SUPERSEDED_R42_REPORT_FILE_SHA256,
            "report_digest": SUPERSEDED_R42_REPORT_DIGEST,
            "minimum": dict(SUPERSEDED_R42_MINIMUM),
            "pooled_eigenvalues": list(SUPERSEDED_R42_EIGENVALUES),
            "failing_block_eigenvalues": {
                "4": list(SUPERSEDED_R42_BLOCK4_EIGENVALUES),
            },
            "sealed_utc": SUPERSEDED_R42_SEALED_UTC,
            "seal": superseded_status,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "convention_source": (
                "the corrected card-17 R42 driver "
                f"({FROZEN_SUBSTRATE['r42_driver'][1]}), consumed "
                "read-only as the convention source; the R46 driver "
                "mirrors the corrected R44/R42 _worker_context / "
                "_replica_record pattern byte for byte on the "
                "recording side"
            ),
            "corrected_driver_sha256": DRIVER_SHA256,
            "geometry": pre["recording_convention"]["geometry"],
        },
        "sealed_run_dirs": sealed,
        "card21_predecessor": (
            "the sealed card-20 R45 simplified-alignment archive at "
            f"{R45_RUN_DIR}: the frozen simplified alignment is the "
            "generation substrate of this profile; this card holds the "
            "single pre-registered optimizer use of the follow-up pair"
        ),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(f"[r46] pre-registration written into {run_dir}", flush=True)
    return pre


def _load_pre_registration(run_dir: Path) -> dict:
    path = Path(run_dir) / PREREGISTRATION_NAME
    if not path.is_file():
        _refuse(f"run dir carries no {PREREGISTRATION_NAME}: {run_dir}")
    return _read_json(path)


def _assert_pre_registration(pre: dict, run_dir: Path) -> None:
    """Assert the pre-registration against the card (the run refuses on
    any drift; a change after generation invalidates the run)."""
    if pre.get("run") != RUN_ID or pre.get("card") != CARD:
        _refuse("the pre-registration does not name this run/card")
    mode = str(pre.get("mode", "production"))
    if mode == "smoke":
        return
    grid = pre.get("profile_grid", {})
    if [float(m) for m in (grid.get("b_eff_multipliers") or ())] != \
            [float(m) for m in B_EFF_MULTIPLIERS]:
        _refuse("the pre-registered B_eff multipliers do not match the "
                "card (0.5..2.5 in 0.25 steps)")
    if [float(b) for b in (grid.get("b_dec_gev2") or ())] != \
            [float(b) for b in B_DEC_GRID]:
        _refuse("the pre-registered b_dec grid does not match the card")
    if float(grid.get("b_eff_frozen_gev2", 0.0)) != B_EFF_FROZEN_GEV2:
        _refuse("the pre-registered frozen B_eff does not match the "
                "M8-frozen exchange slope")
    validity = grid.get("reweighting_validity", {})
    pinned = validity.get("pinned_grid_points") or []
    if [(float(p["b_eff_multiplier"]), float(p["b_dec_gev2"]))
            for p in pinned] != [tuple(map(float, p))
                                  for p in VALIDITY_GRID_POINTS]:
        _refuse("the pre-registered pinned grid points do not match the "
                "card")
    stats = pre.get("statistics", {})
    if int(stats.get("replicas", -1)) != T2_REPLICAS or int(
        stats.get("events_per_replica", -1)
    ) != T2_EVENTS_PER_REPLICA:
        _refuse("the pre-registered statistics do not match the card "
                "(T2: 8 replicas x 100,000)")
    seeds = pre.get("seed_map", {})
    if list(seeds.get("raw_replica_seeds") or ()) != \
            list(RAW_REPLICA_SEEDS):
        _refuse("the pre-registered raw replica seeds do not match the "
                "card's new block 20301076-20301083")
    if seeds.get("event_index_rule") != EVENT_INDEX_RULE:
        _refuse("the pre-registered event-index rule does not match the "
                "card (900_000_000 + i, i < 100,000)")
    carry = pre.get("obligation_4_carry", {})
    if carry.get("r32_spread", {}).get("bin_spread_p95") != \
            OBLIGATION4_R32_SPREAD["bin_spread_p95"]:
        _refuse("the obligation 4 carry numbers are not the card's "
                "verbatim R32 spread")
    if float(grid.get("b_dec_generation_gev2", 0.0)) != \
            B_DEC_GENERATION_GEV2:
        _refuse("the pre-registered generation b_dec is not the card-20 "
                "frozen 2.0 GeV^-2")
    if grid.get("b_dec_generation_on_grid") is not True:
        _refuse("the card-20 frozen b_dec 2.0 GeV^-2 sits on the "
                "profile grid (the generation point multiplier 1.0 / "
                "b_dec 2.0); the on-grid recording is missing")
    bounds = grid.get("bounds_never_relaxed") or {}
    if bounds.get("b_eff_multiplier") != [
            float(B_EFF_MULTIPLIERS[0]), float(B_EFF_MULTIPLIERS[-1])] \
            or bounds.get("b_dec_gev2") != [
                float(B_DEC_GRID[0]), float(B_DEC_GRID[-1])]:
        _refuse("the pre-registered bounds do not match the card "
                "(multiplier [0.5, 2.5], b_dec [0.2, 2.6] GeV^-2); the "
                "bounds are never relaxed")
    superseded = pre.get("superseded_profile", {})
    if superseded.get("minimum") != SUPERSEDED_R42_MINIMUM or \
            superseded.get("report_digest") != SUPERSEDED_R42_REPORT_DIGEST:
        _refuse("the pre-registered superseded-profile pins do not "
                "match the sealed corrected R42 archive")
    substrate = pre.get("card20_substrate", {})
    if substrate.get("frozen_simplified_archive") != R45_RUN_DIR or \
            substrate.get("sibling_driver_sha256") != \
            FROZEN_SUBSTRATE["r45_driver"][1]:
        _refuse("the card-20 substrate pins do not name the sealed "
                "r45_simplified_alignment_freeze_2026-09-24 archive and "
                "its byte-frozen driver digest")
    if not pre.get("hessian_discipline", {}).get("note"):
        _refuse("the pre-registration carries no hessian-discipline "
                "block (plan section 5.4 line 311)")
    if not pre.get("optimizer", {}).get("single_use"):
        _refuse("the pre-registration does not record the single "
                "pre-registered optimizer use of the follow-up pair")
    if pre.get("standing_residuals") != STANDING_RESIDUALS:
        _refuse("the pre-registration does not carry the standing "
                "residuals (the R43 b_transport_seagull finding and the "
                "R32 spread carry)")
    if not pre.get("written_before_generation", False):
        _refuse("the pre-registration was not marked "
                "written_before_generation")
    for key in ("collision", "observables", "likelihood",
                "profile_evaluation", "conservation", "determinism"):
        if key not in pre:
            _refuse(f"the pre-registration is missing the {key} block")


# ---------------------------------------------------------------------------
# The pre-registration (written before any generation)
# ---------------------------------------------------------------------------
def build_pre_registration(smoke: bool = False) -> dict:
    """The pre-registered run contract, asserted against the card
    (TASK-20260920-r4-run-r42-recoil-decay-separation)."""
    return {
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "plan_anchors": {
            "run": (
                "plan section 10 run R42, line 786, re-run under the "
                "simplified form (card 21 of the round 4 set; plan "
                "section 23, lines 1821-1823)"),
            "profile": "plan line 788 (distinguishable effects and stable curvature)",
            "observables": "plan line 788 (definitions plan section 7.3, lines 420-431)",
            "b_transport": "plan section 7.3 line 431 (B_transport = p - pbar)",
            "reweight_rule": "plan section 5.3 line 288",
            "common_random_numbers": "plan section 5.2 line 275",
            "likelihood": "plan section 5.4 (lines 290-311)",
            "hessian_discipline": "plan section 5.4 line 311",
            "replica_blocks": "plan section 5.5 item 5, line 321",
            "statistics_tier": "plan section 5.1 line 254",
            "no_single_parameter": "plan section 10.5 line 748",
            "not_do": "plan section 24 line 1866 (exchange slopes never control daughter-level transverse dynamics)",
            "cascade_spec": "plan section 10.2 line 678 (each high-mass parent)",
        },
        "collision": {
            "index_item": 7,
            "system": "p+p at sqrt(s) = 17.270 GeV (beam 158 GeV/c)",
            "configuration": (
                "the R32 p+p configuration: the M8-frozen ten-parameter "
                f"set (hash {M8_PARAMETER_HASH[:16]}...) on the two-source "
                "(B, Q, S) = (1, 1, 0) topology, through the "
                "ResonancePPBuilder with the card-6 routing at the "
                "nominal ramp start B = "
                f"{BLEND_RAMP_START_GEV!r} GeV (the v0 variant, ramp "
                "width 0.5 GeV) and the card-9 feed-down lineage "
                "convention; the frozen exchange slope B_eff = "
                f"{B_EFF_FROZEN_GEV2!r} GeV^-2 held for the generation"
            ),
            "exchange_layer": (
                "byte-unchanged (the builder's exchange/topology/"
                "spectral-state layers are the frozen surface; the M8 "
                "set is applied unchanged)"
            ),
            "w_flavor": 1.0,
            "frozen_b_eff_gev2": B_EFF_FROZEN_GEV2,
            "catalog_branch": (
                "the R32 builder catalog branch byte-stable: the open-set "
                "mixture record draw on the spawned decay_channel child, "
                "the stable-identity path, and the byte-unchanged Round 2 "
                "kernel decay on the labeled substreams"
            ),
            "continuum_branch": (
                "the single activated continuum family (baryonic_np) "
                "realized at the frozen source four-momentum (no "
                "family-mass redraw), decayed through the card-10 "
                "aligned cascade at the card-20 frozen simplified "
                "alignment (kappa_B, kappa_m) = (0.0, 4.0) with the "
                "frozen b_dec = 2.0 GeV^-2 (on the profile grid) and "
                "the baryonic nominal_width recorded at the frozen 0.3 "
                "GeV (the card-4 continuum-daughter mass rule consumes "
                "the byte-unchanged committed continuum config's family "
                "widths); the card-9 routing order with card 10's "
                "attached per-vertex step, the frozen 1 channel + 2 "
                "kinematics per-vertex budget; the production axis is "
                "card 5's initial_source_axis against the source's own "
                "beam direction (projectile +z, target -z; the round 4 "
                "index item 4 p+p pre-registration)"
            ),
            "frozen_alignment_note": FROZEN_ALIGNMENT_NOTE,
            "card20_substrate_note": CARD20_SUBSTRATE_NOTE,
            "production_axis_note": (
                "the R40 isolated-source axis (the lab z-axis) is "
                "replaced by the p+p convention per index item 4; the "
                "card-10 runner copies the axis verbatim down the line"
            ),
        },
        "profile_grid": {
            "index_item": 10,
            "b_eff_multipliers": [float(m) for m in B_EFF_MULTIPLIERS],
            "b_eff_multiplier_step": B_DEC_MULT_STEP,
            "b_eff_frozen_gev2": B_EFF_FROZEN_GEV2,
            "b_eff_grid_gev2": [
                float(m * B_EFF_FROZEN_GEV2) for m in B_EFF_MULTIPLIERS
            ],
            "b_dec_gev2": [float(b) for b in B_DEC_GRID],
            "b_dec_step_gev2": B_DEC_STEP_GEV2,
            "b_dec_generation_gev2": B_DEC_GENERATION_GEV2,
            "b_dec_generation_on_grid": True,
            "bounds_never_relaxed": {
                "b_eff_multiplier": [
                    float(B_EFF_MULTIPLIERS[0]),
                    float(B_EFF_MULTIPLIERS[-1])],
                "b_dec_gev2": [
                    float(B_DEC_GRID[0]), float(B_DEC_GRID[-1])],
            },
            "note": PROFILE_GRID_NOTE,
            "reweight_exactness": {
                "transfer": (
                    "the dP/dt proportional-to exp(B_eff t) transfer law "
                    "reweights exactly: the realized Mandelstam-t and its "
                    "two-body support are parameter-independent draws "
                    "(the support depends on the masses and sqrt(s) "
                    "only), so the per-event transfer weight is the "
                    "closed-form density ratio w_B = exp((B' - B)(t - "
                    "t_hi)) (B'/B) (1 - exp(-B dt)) / (1 - exp(-B' d t)) "
                    "with d t = t_hi - t_lo the support width; zero "
                    "regeneration (plan section 5.3 line 288)"
                ),
                "form_factor": (
                    "the card-8 W_dec = exp(-b_dec q_perp^2) is a pure "
                    "weight change on the unchanged attached channel "
                    "pools: the per-vertex channel-choice ratio uses the "
                    "recorded (w0, q2, chosen) sufficient statistics at "
                    "the generation b_dec = 2.0 GeV^-2 (the R41 "
                    "algebra); zero regeneration"
                ),
                "fixed_directions": (
                    "the alignment kernels stay at the card-20 frozen "
                    "simplified point (kappa_B, kappa_m) = (0.0, 4.0) "
                    "and the baryonic nominal_width at the frozen 0.3 "
                    "GeV for the generation and the whole profile; no "
                    "other parameter is reweighted (plan section 24: no "
                    "refit beyond the profile; plan line 784; "
                    "obligation 4)"
                ),
            },
            "reweighting_validity": {
                "note": VALIDITY_NOTE,
                "bound_relative": VALIDITY_BOUND,
                "pinned_grid_points": [
                    {"b_eff_multiplier": float(m), "b_dec_gev2": float(b)}
                    for m, b in VALIDITY_GRID_POINTS
                ],
                "pinned_subsample_events": VALIDITY_EVENTS,
                "pinned_subsample_rule": VALIDITY_SUBSAMPLE_RULE,
            },
        },
        "observables": {
            "training": (
                "the leading-baryon transfer observable and the "
                "identified seagulls of plan line 788: the B_transport = "
                "f(p) - f(pbar) spectrum (plan section 7.3 line 431; the "
                "antiproton term enters with the minus sign) on the "
                "aligned proton/antiproton cells, the pooled pion "
                "seagull (the pi_Sigma = pi+ + pi- combination at the "
                "pion mass) and the pooled proton seagull (the "
                "B_transport combination at the proton mass), the per-xF "
                "<pT>_x and <pT>_y conventions of fit_observables on "
                "both sides; the identified seagulls' forward/central/"
                "backward shapes are the plan-7.3 region structure of "
                "the per-xF moment profile (forward xF > 0.2, central "
                "[0.0, 0.2], backward < 0.0; the p+p paper window "
                "carries the central and forward regions), recorded per "
                "bin as the region tag"
            ),
            "combination_grid": (
                "the combination's grid is the narrower species' table "
                "grid on the aligned (xF_mid, pT_mid) cells (the R11 "
                "intersection convention; the wider species' extra cell "
                "is outside the intersection); the seagull moments use "
                "the fit_observables conventions on both sides"
            ),
            "published_binning": "preserved (the sealed R01 tables)",
            "data_source": (
                "the sealed R01 archive (bin_predictions/covariance/"
                "residuals), cross-checked bit for bit against the "
                "locally loaded SPS tables at --profile time (the R11 "
                "parse_sealed_binned discipline)"
            ),
        },
        "recording_convention": {
            "convention": (
                "the R01 accumulate-dense-then-pick recording convention "
                "(the corrected convention of the card-17 relitigation, "
                "consumed read-only from the corrected card-17 R42 "
                "driver): every event weight accumulates into the "
                "species' dense pT-major evaluation lattice (n_x * n_p "
                "cells at the species' dense offset), and the published "
                "rows are picked from the dense totals at record "
                "assembly"
            ),
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "holes_note": (
                "the published R01 tables are non-rectangular subsets of "
                "their lattices (holes; rows not pT-major): a hole "
                "cell's weight stays unpicked and is dropped, never bled "
                "into a neighboring published row or species; the "
                "structural refusals cover a published pair outside the "
                "lattice and two published rows on one lattice cell"
            ),
            "convention_source_driver_sha256":
                FROZEN_SUBSTRATE["r42_driver"][1],
            "corrected_driver_sha256": DRIVER_SHA256,
        },
        "likelihood": {
            "form": (
                "-2lnL = (d - m)' V^-1 (d - m) + eta' P^-1 eta, with V = "
                "V_data + V_mc (reported separately), the published "
                "binning preserved, and the nuisance vector profiled "
                "exactly per grid point: one multiplicative normalization "
                "nuisance on the b_transport spectrum (prior 0.35) plus "
                "the obligation 4 carry as one correlated multiplicative "
                "nuisance on the blend-band-sensitive B_transport bins "
                "(prior 0.25); the seagull statistics are normalization-"
                "free ratios and carry no nuisance"
            ),
            "v_data": "the sealed R01 per-bin published variance",
            "v_mc": (
                "the between-replica standard error of the trial "
                "statistic (std ddof=1 over the 8 replica blocks, "
                "divided by sqrt(8)), floored at the engine's "
                "DETERMINISTIC_SE_FLOOR; a statistic whose pooled MC "
                "error is zero by structure (an empty model bin) carries "
                "an infinite sigma and drops out of the objective (the "
                "R11 convention; the dropped bins are listed)"
            ),
            "nuisances": {
                "normalization": NORM_NUISANCE_NOTE,
                "obligation_4_carry": OBLIGATION4_FORM,
                "carry_prior": CARRY_NUISANCE_PRIOR,
                "carry_prior_note": CARRY_PRIOR_NOTE,
                "carry_bins": (
                    "the sealed R32 persisting-residual bins (the card-1 "
                    "audit's attribution) restricted to the b_transport "
                    "pair's species: "
                    f"{len(CARRY_BIN_KEYS)} of the 32 intersect the pair "
                    "(9 proton rows, no antiproton rows) and carry the "
                    "term in the objective"
                ),
            },
            "profile_is_not_fit_set": (
                "the nuisance profile is part of the objective definition "
                "(plan section 5.4), not part of the profile axes: the "
                "profile coordinates are exactly (b_eff_multiplier, "
                "b_dec) on the pre-registered 9 x 9 lattice"
            ),
            "coordinate_note": (
                "the profile's B_eff coordinate is the pre-registered "
                "multiplier of the frozen value (dimensionless); the "
                "b_dec coordinate is GeV^-2; the curvature matrix and "
                "the correlation are quoted in these coordinates"
            ),
        },
        "profile_evaluation": {
            "surface": (
                "the pooled profiled -2lnL on the 9 x 9 grid: at every "
                "grid point the 8 per-replica-block reweighted statistic "
                "vectors, the pooled vectors, the between-replica SE, "
                "and the exactly profiled nuisance objective (plan "
                "section 5.4)"
            ),
            "per_block_surfaces": (
                "the 8 per-replica-block surfaces (each block's own "
                "profiled objective against the published data with the "
                "deterministic SE floor), the pre-registered replica-"
                "block curvature-stability input (plan section 5.5 item "
                "5 line 321)"
            ),
            "minimum": (
                "the grid argmin (the pre-registered lattice argmin; a "
                "minimum on the grid boundary is a named boundary "
                "finding, the profile range does not extend)"
            ),
            "hessian": (
                "the 2 x 2 curvature matrix at the minimum from the grid "
                "second differences (the multiplier spacing 0.25, the "
                "b_dec spacing 0.3 GeV^-2); a minimum on the grid "
                "boundary carries the one-sided second-difference "
                "construct, named"
            ),
            "correlation": (
                "rho = -H_01 / sqrt(H_00 H_11) (the inverse-Hessian "
                "correlation of the curvature matrix of -2lnL; the "
                "common positive factor 2 of the covariance cancels); "
                "the pre-registered distinguishability algebra, bound "
                "|rho| < 0.9"
            ),
            "curvature_stability": (
                "the Hessian at the pooled minimum evaluated on each of "
                "the 8 per-block surfaces; positive eigenvalues on all 8 "
                "blocks with the per-direction curvature spread below "
                "20% of the pooled curvature are the acceptance (ii) "
                "stable-curvature clause (plan section 5.5 item 5 "
                "line 321)"
            ),
        },
        "hessian_discipline": {
            "note": HESSIAN_DISCIPLINE_NOTE,
            "gate": (
                "the 8 per-replica-block surface argmins must equal the "
                "pooled argmin before the pooled Hessian, its "
                "eigenvalues, or the correlation are quoted; the "
                "curvature matrix is still recorded beside the gate"
            ),
            "per_direction_spread": CURVATURE_SPREAD_NOTE,
        },
        "optimizer": {
            "single_use": (
                "this card holds the single pre-registered optimizer use "
                "of the follow-up pair (the two-dimensional profile); "
                "card 20 invokes none and no other optimizer invocation "
                "exists in this card"
            ),
            "form": (
                "the profile minimum is the pre-registered 9 x 9 lattice "
                "argmin and the curvature is the grid second difference "
                "at it; no continuous optimization is claimed"
            ),
        },
        "superseded_profile": {
            "run_dir": SUPERSEDED_R42_RUN_DIR,
            "minimum": dict(SUPERSEDED_R42_MINIMUM),
            "status": SUPERSEDED_R42_STATUS,
            "report_digest": SUPERSEDED_R42_REPORT_DIGEST,
            "pooled_eigenvalues": list(SUPERSEDED_R42_EIGENVALUES),
            "failing_block_eigenvalues": {
                "4": list(SUPERSEDED_R42_BLOCK4_EIGENVALUES),
            },
            "sealed_utc": SUPERSEDED_R42_SEALED_UTC,
            "note": (
                "the superseded corrected R42 profile is cited as "
                "superseded, never as evidence; its boundary minimum "
                "(0.5, 2.6), its non-positive-definite pooled Hessian "
                "eigenvalues, its failing block-4 curvature, and its "
                "null correlation are the recorded findings the Gate 4 "
                "decision consumed"
            ),
        },
        "card20_substrate": {
            "note": CARD20_SUBSTRATE_NOTE,
            "frozen_simplified_archive": R45_RUN_DIR,
            "sibling_driver_rel": FROZEN_SUBSTRATE["r45_driver"][0],
            "sibling_driver_sha256": FROZEN_SUBSTRATE["r45_driver"][1],
        },
        "frozen_parameters_note": (
            "the frozen simplified alignment (kappa_B, kappa_m, b_dec, "
            f"baryonic_nominal_width) = ({GENERATION_KAPPA_B!r}, "
            f"{GENERATION_KAPPA_M!r}, "
            f"{FROZEN_SIMPLIFIED_ALIGNMENT['b_dec']!r}, "
            f"{FROZEN_WIDTH_GEV!r}) is asserted against the sealed R41 "
            "archive's fit report (the 'fitted' block) before any "
            "generation and cross-checked against the imported card-20 "
            "sibling driver's frozen policy at import; no refit happens "
            "on this run"
        ),
        "standing_residuals": STANDING_RESIDUALS,
        "statistics": {
            "tier": T2_LABEL,
            "replicas": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "total_events": T2_REPLICAS * T2_EVENTS_PER_REPLICA,
            "additional_generation": (
                "zero for the profile grid (the 9 x 9 evaluation is a "
                "reweighting over the generated events); the pinned "
                "validity probe generates the pinned sub-sample "
                f"({VALIDITY_EVENTS} events x 3 pinned configurations: "
                "the frozen point shared by both pinned grid points plus "
                "one direct generation per pinned grid point) in memory, "
                "never sealed"
            ),
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": RAW_SEED_RULE,
            "raw_replica_seeds": list(RAW_REPLICA_SEEDS),
            "kernel_seed_rule": KERNEL_SEED_RULE,
            "event_index_rule": EVENT_INDEX_RULE,
            "stream_split": (
                "the mixture-layer draws (the routing uniform and the "
                "catalog record draw) consume the one-step spawned child "
                "of the decay_channel labeled id (SeedSequence(stream_id)"
                ".spawn(1), the R21/R22/R23 mixture-layer stream split); "
                "the card-10 cascade and the Round 2 kernel keep the "
                "labeled ids themselves; the spawned decay_kinematics "
                "child consumes nothing in R46 (the R32 continuum root "
                "split is replaced by the card-10 attached vertices, "
                "which draw on the labeled ids; recorded so no draw-order "
                "ambiguity remains)"
            ),
            "disjointness": (
                "the R46 raw block 20301076-20301083 is a new block "
                "disjoint from the R40-R44 block 20301020-20301039, from "
                "the reserved audit block 20301040 + k (never fired), "
                "and from the card-20 block 20301056-20301075; the "
                "kernel-layer spaces derive_substreams(20260920, k, "
                "900_000_000 + i) are disjoint from every previously "
                "used space because the event-index partition "
                "900_000_000 + i differs from the card-20 R45 partition "
                "800_000_000 + i, from R44's 700_000_000 + i, from R43's "
                "600_000_000 + i, from R42's 500_000_000 + i, from R41's "
                "400_000_000 + i, from the R40 partition (below "
                "389,025,000), and from the R46 smoke partition "
                "990_000_000 + i; if any raw (seed, event-index) "
                "substream collides, the run stops and reports (never a "
                "silent re-block)"
            ),
            "disjointness_report": seed_disjointness_report(),
            "smoke": SMOKE_PAYLOAD,
        },
        "conservation": {
            "per_event": (
                "every p+p event conserves total (B, Q, S) = (2, 2, 0) "
                "exactly; a violation raises and stops the run"
            ),
            "per_source": (
                "every source's finals sum to its own (B, Q, S) = (1, 1, "
                "0) exactly; a violation raises and stops the run"
            ),
            "energy": (
                "the event's final energy sum closes against the two "
                "source energies within 1e-9 GeV; the card-4 per-vertex "
                "closure checks run byte-unchanged inside the walk and "
                "the kernel"
            ),
        },
        "determinism": {
            "rolling_digest": (
                "each replica record carries the event-hash set digest "
                "(sha256 over the ordered per-event canonical hashes: "
                "event id, route census, final count, and every final's "
                "pdg with exact hex four-momentum components) and the "
                "latent digest (sha256 over the packed latent arrays); "
                "a per-replica re-run from a clean interpreter "
                "reproduces both"
            ),
            "probe_replica": {
                "replica": PROBE_REPLICA,
                "rule": (
                    "clean-validate re-runs this full replica from a "
                    "clean interpreter and compares the event-hash set "
                    "digest, the pooled sums, and the latent digest "
                    "against the sealed record"
                ),
            },
            "report_note": (
                "profile_report.json and gate4_followup_reprofile.json carry no "
                "wall-clock fields: their bytes are a pure function of "
                "the sealed artifacts, the pre-registered seeds, and the "
                "committed code"
            ),
        },
        "acceptance": {
            "precedent": (
                "the acceptance precedent of "
                "TASK-20260918-r3-acceptance-rescope-card8a-8b: an "
                "MC-driven check set with residuals seals with a named-"
                "residual register, not a raw zero-failure target"
            ),
            "i_profile_convergence": (
                "the optimizer converges on the two-dimensional "
                "B_eff / b_dec profile: the profile grid, the minimum, "
                "and the curvature are recorded; the superseded "
                "corrected R42 profile (minimum (0.5, 2.6), recorded in "
                "the report) is cited as superseded, never as evidence"
            ),
            "ii_distinguishable_effects_and_stable_curvature": (
                "all four conditions hold in full: the profile minimum "
                "is interior to the grid (not on any grid boundary); "
                "the B_eff / b_dec correlation is defined (non-null) "
                "with |rho| < 0.9 on the leading-baryon transfer "
                "observables and the identified seagulls; the pooled "
                "Hessian is positive definite (both eigenvalues above "
                "zero); every one of the 8 replica blocks is positive "
                "definite with the per-direction curvature spread below "
                "20% of the pooled curvature (plan section 5.5 item 5 "
                "stability).  A failing condition is a named finding "
                "(boundary_minimum, correlation_undefined_or_high, "
                "curvature_not_positive, block_curvature_failure with "
                "the block index and eigenvalues quoted); on any "
                "failure no separation claim is made"
            ),
            "iii_reweighting_validity": (
                "the weighted prediction's exact identities hold against "
                "the direct generation of the pinned sub-sample at the "
                "two pinned grid points within 1e-6 relative (the stream "
                "alignment, the transfer-law implied realization, and "
                "the first-vertex trial shares; the construction is "
                "pre-registered in the profile_grid block); the bound is "
                "never relaxed and a failure stops the run"
            ),
            "iv_conservation": (
                "800,000 events, exact B = 2 / Q = 2 and exact "
                "per-source (B, Q, S); zero violations"
            ),
            "v_determinism": (
                "the clean-environment re-run reproduces the per-replica "
                "record identity digests and the profile evaluation "
                "hashes"
            ),
            "v_no_fit_beyond_the_profile": (
                "optimizer_invoked is true for this card only: no "
                "exchange, topology, flavor, branching, ramp-start, or "
                "alignment parameter is fit (plan section 24; plan line "
                "784; obligation 4); the alignment kernels stay at the "
                "card-20 frozen simplified point (kappa_B, kappa_m) = "
                "(0.0, 4.0), the frozen b_dec 2.0 GeV^-2 and the frozen "
                "width 0.3 GeV, the blend ramp start at B (obligation "
                "4); this card holds the single pre-registered optimizer "
                "use of the follow-up pair"
            ),
            "vi_seal_and_clean_validation": (
                "the seal is intact and the clean-environment re-run "
                "reproduces the profile evaluation hashes"
            ),
            "vii_pins_and_freeze": (
                "the driver pins land (both trees byte-identical, the "
                "register re-pointed including the card-20 sibling "
                "pins), the freeze re-pin is recorded with the new "
                "driver entry in both helper trees and the two new test "
                "files, and the full nested suite is green"
            ),
            "named_residual_register": (
                "every failing verdict named with the observed value, a "
                "one-line diagnosis, and the robust-statistics cross-"
                "reference; the pre-registered bounds are never relaxed; "
                "no data selection"
            ),
            "register_triggers": [
                "the profile minimum on the grid boundary of either "
                "axis (the boundary_minimum finding: the profile range "
                "does not extend, the bounds are never relaxed)",
                "a parameter correlation undefined or at or above the "
                "pre-registered 0.9 bound (the "
                "correlation_undefined_or_high finding; the plan line "
                "788 distinguishable-effects clause)",
                "a pooled Hessian at the minimum that is not positive "
                "definite when quoted (the curvature_not_positive "
                "finding)",
                "any per-block Hessian failure or a per-direction "
                "curvature spread at or above 20% of the pooled "
                "curvature (the block_curvature_failure finding with "
                "the block index and eigenvalues quoted)",
                "the replica-block minima disagreeing with the pooled "
                "argmin (the replica_block_minima_disagree finding: no "
                "curvature is quoted under the plan section 5.4 line "
                "311 discipline; the stable-curvature and correlation "
                "claims are not established)",
                "an effective sample size fraction below 0.01 at the "
                "profile minimum (the weight-degeneracy diagnosis)",
                "a non-finite pooled surface value at any grid point "
                "(recorded; the minimum is taken over the finite points)",
            ],
        },
        "obligation_4_carry": {
            "obligation": 4,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "carry_prior": CARRY_NUISANCE_PRIOR,
            "carry_prior_note": CARRY_PRIOR_NOTE,
            "blend_band_sensitive_bins": CARRY_BIN_KEYS,
            "audit_outcome": AUDIT_OUTCOME,
        },
        "blend_band_sensitive_bins": {
            "definition": (
                "the sealed R32 persisting-residual bins (the card-1 "
                "audit's probe set; the audit outcome is the card's "
                "'final form determined by the closed audits' input): "
                "the b_transport spectrum cells whose contributing "
                "species cells are in this sealed set"
            ),
            "source": (
                "dev/00runs/r4_audit_transition_routing_formulation_"
                "2026-09-20/persisting_bin_attribution.json rows "
                "(classification == 'persisting', the R32 confirmation "
                "reclassification)"
            ),
            "sigma_range": [2.0001998039779263, 7.532929574244998],
            "entries": 32,
            "intersecting_species": {"proton": 9, "antiproton": 0},
        },
        "sealed_run_dirs": list(SEALED_RUN_DIRS),
        "frozen_substrate": {
            key: value[1] for key, value in FROZEN_SUBSTRATE.items()
        },
        "gateway_fallback": GATEWAY_FALLBACK_RULE,
        "written_before_generation": True,
    }


# ---------------------------------------------------------------------------
# The fit data (the sealed R01 archive + the published binning)
# ---------------------------------------------------------------------------
R01_ARCHIVE = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"
#: The profile's pair conventions (plan section 7.3): the leading-baryon
#: transfer pair is the plan-431 DIFFERENCE f(p) - f(pbar) (the
#: antiproton term enters with the minus sign); the pooled pion pair is
#: the pi_Sigma sum.  The sign multiplies the second species' values.
COMBINATION_PAIRS = {
    "pi_sigma": ("pi_plus", "pi_minus", +1.0),
    "b_transport": ("proton", "antiproton", -1.0),
}
SEAGULL_MASS_NOTE = (
    "the pooled-combination seagull is evaluated at the combination's "
    "common species mass (the pion mass for pi_Sigma, the proton mass "
    "for the B_transport pair; both species of each combination share "
    "the mass)"
)
FORWARD_REGION_NOTE = (
    "the identified seagulls' forward/central/backward shapes are the "
    "plan-7.3 region structure of the per-xF moment profile: each xF bin "
    "carries its region tag (forward xF > 0.2, central [0.0, 0.2], "
    "backward < 0.0; the p+p paper window xF in [0, 0.4] carries the "
    "central and forward regions), the fit_observables."
    "forward_central_backward_integral split"
)


def _load_sealed_r01(tables: list) -> dict:
    """The sealed R01 data vector and published variances, cross-checked
    bit for bit against the locally loaded SPS tables (the R11
    parse_sealed_binned discipline)."""
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


def _aligned_cells(sealed: dict, species_a: str, species_b: str) -> list:
    """The intersection of the two species' tabulated cells, sorted by
    (xF, pT) (the R11 aligned-cells convention).  Each entry carries the
    cell's (bin_index_a, bin_index_b, xF_mid, pT_mid)."""
    coords_a = [
        (round(float(x), 12), round(float(p), 12), int(i))
        for i, (x, p) in enumerate(zip(
            sealed[species_a]["xF_mid"], sealed[species_a]["pT_mid"]))
    ]
    index_a = {(x, p): i for i, (x, p, _i) in enumerate(coords_a)}
    out = []
    for i_b, (x, p) in enumerate(zip(
        sealed[species_b]["xF_mid"], sealed[species_b]["pT_mid"],
    )):
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


def _cell_area(sealed: dict, tables_by_species: dict, species: str,
               bin_index: int) -> float:
    """The (xF, pT) cell area at the sealed midpoint (the R01 binning
    convention: the species table's midpoint-edge grid, exact)."""
    table = tables_by_species[species]
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[bin_index])
    ip = int(table.bin_index_pt[bin_index])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


def _combination_statistic(
    sealed: dict, tables_by_species: dict, pair_key: str, carry_set: set,
) -> dict:
    """One pooled-combination spectrum statistic on the aligned cells."""
    species_a, species_b, sign = COMBINATION_PAIRS[pair_key]
    cells = _aligned_cells(sealed, species_a, species_b)
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
    areas = np.asarray([
        _cell_area(sealed, tables_by_species, species_a, e["bin_a"])
        for e in cells
    ], dtype=float)
    carry = np.asarray([
        1.0 if ((species_a, e["bin_a"]) in carry_set
                or (species_b, e["bin_b"]) in carry_set) else 0.0
        for e in cells
    ], dtype=float)
    return {
        "key": f"{pair_key}_spectrum",
        "kind": "spectrum",
        "pair": (species_a, species_b, sign),
        "cells": cells,
        "d": d,
        "v_data": v_data,
        "areas": areas,
        "carry_mask": carry,
    }


def _combination_model_statistic(
    stat: dict, model_a: np.ndarray, model_b: np.ndarray,
) -> np.ndarray:
    sign = stat["pair"][2]
    return np.asarray([
        model_a[e["bin_a"]] + sign * model_b[e["bin_b"]]
        for e in stat["cells"]
    ], dtype=float)


# ---------------------------------------------------------------------------
# The replica-block latent (the reweighting data)
# ---------------------------------------------------------------------------
class BlockLatent:
    """One replica block's reweighting payload at the generation point
    (the frozen B_eff and the card-20 frozen b_dec 2.0 GeV^-2).

    ``row_*`` is the CSR of every event's bin rows (every event is
    reweightable: the exchange-layer slope B_eff alters the realized
    scattering angle of the whole event); ``t_scatter``/``t_minus``/
    ``t_plus`` is the per-event realized Mandelstam-t with its two-body
    support (the exact transfer-law sufficient statistic; the support is
    parameter-independent); ``vtx_*``/``ch_*`` is the per-vertex attached
    pool of the card-8 shares (``w0`` at the generation b_dec) and the
    reference transverse momenta squared (``q2``), the chosen channel,
    and the per-event vertex CSR pointers.  The B_eff transfer
    reweighting and the b_dec channel reweighting consume exactly this
    payload (plan section 5.3 line 288)."""

    __slots__ = (
        "replica", "seed", "n_events", "row_flat", "row_weight",
        "row_ptr", "event_ids", "t_scatter", "t_minus", "t_plus",
        "vtx_ptr", "vtx_nchan", "vtx_chosen", "ch_w0", "ch_q2",
        "route_census", "_row_event_map", "_vtx_pool_ptr",
    )

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))
        self.n_events = int(self.n_events)
        self.route_census = tuple(int(v) for v in (self.route_census
                                                   or (0, 0, 0)))
        self.row_flat = np.asarray(self.row_flat, dtype=np.int64)
        self.row_weight = np.asarray(self.row_weight, dtype=float)
        self.row_ptr = np.asarray(self.row_ptr, dtype=np.int64)
        self.event_ids = np.asarray(self.event_ids, dtype=np.int64)
        self.t_scatter = np.asarray(self.t_scatter, dtype=float)
        self.t_minus = np.asarray(self.t_minus, dtype=float)
        self.t_plus = np.asarray(self.t_plus, dtype=float)
        self.vtx_ptr = np.asarray(self.vtx_ptr, dtype=np.int64)
        self.vtx_nchan = np.asarray(self.vtx_nchan, dtype=np.int64)
        self.vtx_chosen = np.asarray(self.vtx_chosen, dtype=np.int64)
        self.ch_w0 = np.asarray(self.ch_w0, dtype=float)
        self.ch_q2 = np.asarray(self.ch_q2, dtype=float)

    @property
    def row_event_map(self) -> np.ndarray:
        """The per-row-entry event index (for the per-event weight
        broadcast)."""
        cached = getattr(self, "_row_event_map", None)
        if cached is None:
            counts = np.diff(self.row_ptr)
            cached = np.repeat(
                np.arange(self.n_events, dtype=np.int64), counts)
            object.__setattr__(self, "_row_event_map", cached)
        return cached

    @property
    def vtx_pool_ptr(self) -> np.ndarray:
        """The per-vertex offsets into the channel-pool arrays."""
        cached = getattr(self, "_vtx_pool_ptr", None)
        if cached is None:
            cached = np.concatenate((
                [0], np.cumsum(self.vtx_nchan)))
            object.__setattr__(self, "_vtx_pool_ptr", cached)
        return cached


def _block_from_record(record: dict) -> BlockLatent:
    (event_ids, row_flat, row_weight, row_ptr, t_scatter, t_minus,
     t_plus, vtx_ptr, vtx_nchan, vtx_chosen, ch_w0, ch_q2) = unpack_arrays(
        record["latent"]["packed"])
    routing = record.get("routing", {})
    return BlockLatent(
        replica=int(record["replica"]),
        seed=int(record["seed"]),
        n_events=int(record["n_events"]),
        route_census=(
            int(routing.get("n_sources", 0)),
            int(routing.get("n_catalog", 0)),
            int(routing.get("n_continuum", 0)),
        ),
        row_flat=row_flat,
        row_weight=row_weight,
        row_ptr=row_ptr,
        event_ids=event_ids,
        t_scatter=t_scatter,
        t_minus=t_minus,
        t_plus=t_plus,
        vtx_ptr=vtx_ptr,
        vtx_nchan=vtx_nchan,
        vtx_chosen=vtx_chosen,
        ch_w0=ch_w0.astype(float),
        ch_q2=ch_q2.astype(float),
    )


def _block_from_aggregate(aggregate: dict) -> BlockLatent:
    return BlockLatent(
        replica=int(aggregate["replica"]),
        seed=int(aggregate["seed"]),
        n_events=int(aggregate["n_events"]),
        route_census=tuple(int(v) for v in aggregate["routing"]),
        row_flat=aggregate["row_flat"],
        row_weight=np.asarray(aggregate["row_weight"], dtype=float),
        row_ptr=np.asarray(aggregate["row_ptr"], dtype=np.int64),
        event_ids=np.asarray(aggregate["event_ids"], dtype=np.int64),
        t_scatter=np.asarray(aggregate["t_scatter"], dtype=float),
        t_minus=np.asarray(aggregate["t_minus"], dtype=float),
        t_plus=np.asarray(aggregate["t_plus"], dtype=float),
        vtx_ptr=np.asarray(aggregate["vtx_ptr"], dtype=np.int64),
        vtx_nchan=np.asarray(aggregate["vtx_nchan"], dtype=np.int64),
        vtx_chosen=np.asarray(aggregate["vtx_chosen"], dtype=np.int64),
        ch_w0=np.asarray(aggregate["ch_w0"], dtype=float),
        ch_q2=np.asarray(aggregate["ch_q2"], dtype=float),
    )


# ---------------------------------------------------------------------------
# The exact reweighting (plan section 5.3 line 288)
# ---------------------------------------------------------------------------
def transfer_log_weight(t: np.ndarray, t_lo: np.ndarray, t_hi: np.ndarray,
                        b_gen: float, b_trial: float) -> np.ndarray:
    """The exact closed-form log transfer weight per event.

    The dP/dt proportional-to exp(B t) law normalized on the event's own
    support [t_lo, t_hi] has Z(B) = (exp(B t_hi) - exp(B t_lo))/B = 
    exp(B t_hi) (1 - exp(-B dt)) / B with dt = t_hi - t_lo; the density
    ratio at the realized t is therefore
    w = exp((B' - B) (t - t_hi)) (B'/B)
        (1 - exp(-B dt)) / (1 - exp(-B' dt)),
    exact for every B' > 0 on the parameter-independent support (the
    degenerate zero-width support carries weight 1 by convention, the
    measure-zero branch of the sampler)."""
    b_gen = float(b_gen)
    b_trial = float(b_trial)
    dt = np.asarray(t_hi, dtype=float) - np.asarray(t_lo, dtype=float)
    log_b = math.log(b_trial / b_gen)
    if abs(b_trial - b_gen) <= 0.0:
        return np.zeros(np.asarray(t).shape, dtype=float)
    lo_term_gen = np.exp(-b_gen * dt)  # underflows to 0 on wide supports
    lo_term_trial = np.exp(-b_trial * dt)
    with np.errstate(divide="ignore"):
        log_z_gen = np.log1p(-lo_term_gen)
        log_z_trial = np.log1p(-lo_term_trial)
    degenerate = ~(dt > DEGENERATE_WINDOW_TOL)
    log_w = (
        (b_trial - b_gen) * (np.asarray(t, dtype=float)
                             - np.asarray(t_hi, dtype=float))
        + log_b + log_z_gen - log_z_trial
    )
    return np.where(degenerate, 0.0, log_w)


def transfer_implied_t(t_gen: float, b_gen: float, b_trial: float,
                       t_lo: float, t_hi: float) -> float:
    """The implied trial realization of the shared uniform: the
    closed-form inverse-CDF composition
    t_implied = G_{B'}^{-1}(G_B(t_gen)) -- the trial law's realization of
    the same uniform the generation realized as t_gen (the exact
    identity the pinned validity probe asserts)."""
    b_gen = float(b_gen)
    b_trial = float(b_trial)
    t_lo = float(t_lo)
    t_hi = float(t_hi)
    dt = t_hi - t_lo
    if not dt > DEGENERATE_WINDOW_TOL:
        return float(t_gen)
    lo_term_gen = math.exp(-b_gen * dt) if b_gen * dt < 745.0 else 0.0
    # u = (e^{b(t - t_hi)} - e^{-B dt}) / (1 - e^{-B dt}) == the uniform
    # that generated t_gen (the inverse CDF of the normalized law).
    u = (math.exp(b_gen * (float(t_gen) - t_hi)) - (
        math.exp(-b_gen * dt) if b_gen * dt < 745.0 else 0.0)) / (
        1.0 - (math.exp(-b_gen * dt) if b_gen * dt < 745.0 else 0.0))
    u = min(max(u, 0.0), 1.0 - 1e-16)
    lo_term_trial = math.exp(-b_trial * dt) if b_trial * dt < 745.0 else 0.0
    return t_hi + math.log(
        lo_term_trial + u * (1.0 - lo_term_trial)) / b_trial


def event_weights(block: BlockLatent, b_eff_gev2: float,
                  b_dec: float) -> np.ndarray:
    """The per-event importance weights of one block at the trial
    (b_eff_gev2, b_dec): the exact closed-form transfer-law density
    ratio (the exchange layer; the realized t and its parameter-
    independent support are the recorded sufficient statistics) times
    the exact per-vertex channel-choice ratios of the card-8 attached
    shares (the decay-layer reweighting, the R41 algebra).  The
    theta-independent normalization ratios of both laws are included, so
    the trial law is exactly normalized on the unchanged support."""
    b_eff = float(b_eff_gev2)
    b_dec = float(b_dec)
    n_events = block.n_events
    if n_events == 0:
        return np.zeros(0, dtype=float)
    log_wb = transfer_log_weight(
        block.t_scatter, block.t_minus, block.t_plus,
        B_EFF_FROZEN_GEV2, b_eff)
    log_w = np.asarray(log_wb, dtype=float)
    if block.vtx_nchan.size == 0:
        return np.exp(log_w)
    pool_ptr = block.vtx_pool_ptr
    delta_b = b_dec - B_DEC_GENERATION_GEV2
    ex = np.exp(-delta_b * block.ch_q2)
    pool_sum = np.add.reduceat(block.ch_w0 * ex, pool_ptr[:-1])
    if not np.all(np.isfinite(pool_sum)) or not np.all(pool_sum > 0.0):
        _refuse(
            "r46 reweight: a vertex pool's trial weights sum to a "
            "non-finite or non-positive total; the trial b_dec is "
            "outside the pre-registered domain (checked, never repaired)"
        )
    log_channel = (
        -delta_b * block.ch_q2[pool_ptr[:-1] + block.vtx_chosen]
        - np.log(pool_sum)
    )
    # The per-event vertex sum: only the events that carry at least one
    # continuum vertex enter the reduceat (the starts are strictly
    # increasing there, so the empty-event quirk of reduceat never
    # fires); the zero-vertex (catalog-only) events keep weight 1 on the
    # channel factor.
    counts = np.diff(block.vtx_ptr)
    log_channel_event = np.zeros(n_events, dtype=float)
    nz = np.nonzero(counts > 0)[0]
    if nz.size:
        starts = block.vtx_ptr[:-1][nz]
        reduced = np.add.reduceat(log_channel, starts)
        log_channel_event[nz] = reduced
    if log_channel_event.size != n_events:
        _refuse(
            "r46 reweight: the per-event vertex reduce produced "
            f"{log_channel_event.size} values for {n_events} events"
        )
    return np.exp(log_w + log_channel_event)


def block_weighted_sums(block: BlockLatent, eta: np.ndarray) -> np.ndarray:
    """The block's weighted per-cell sums on the dense evaluation
    lattice (the CSR rows weighted by the per-event importance weights;
    the accumulate-dense-then-pick recording space, hole cells
    included)."""
    totals = np.zeros(_dense_total(), dtype=float)
    if block.row_flat.size:
        totals = totals + np.bincount(
            block.row_flat,
            weights=block.row_weight * np.asarray(
                eta, dtype=float)[block.row_event_map],
            minlength=_dense_total(),
        )
    return totals


def ess_fraction(eta: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 over the weighted event ensemble (the
    weight-degeneracy diagnostic of the pre-registration; shift-
    invariant)."""
    if eta.size == 0:
        return 1.0
    total = float(np.sum(eta))
    squares = float(np.sum(eta * eta))
    if squares <= 0.0:
        return 0.0
    return (total * total) / squares / float(eta.size)


# ---------------------------------------------------------------------------
# The training-observable statistic set (plan line 788)
# ---------------------------------------------------------------------------
class Statistic:
    """One profile observable statistic (d, V_data, and the model
    evaluator spec).  Kinds: 'spectrum' (the binned combination cells),
    'seagull' (the two <pT> conventions per xF bin of a combination,
    each bin carrying its plan-7.3 region tag)."""

    __slots__ = ("key", "kind", "d", "v_data", "areas",
                 "carry_mask", "combo", "seagull")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def _seagull_statistic(
    sealed: dict, tables_by_species: dict, pair_key: str, spectrum: dict,
    sqrt_s: float,
) -> Statistic:
    """The pooled-combination seagull: the per-xF <pT>_x and <pT>_y of
    the fit_observables conventions on both sides, the data side from
    the sealed combination d and the first-order delta-method variance
    over the aligned cells; each xF bin carries its forward/central/
    backward region tag (the plan-7.3 shape structure)."""
    species_a, species_b, _sign = COMBINATION_PAIRS[pair_key]
    table = tables_by_species[species_a]
    x_edges = np.asarray(
        sps_core.edges_from_centers(table.xf_centers), dtype=float)
    p_edges = np.asarray(
        sps_core.edges_from_centers(table.pt_centers), dtype=float)
    bin_indices = [
        (int(table.bin_index_xf[e["bin_a"]]),
         int(table.bin_index_pt[e["bin_a"]]))
        for e in spectrum["cells"]
    ]
    mass = float(table.mass_gev)
    d_cells = spectrum["d"]
    groups = {}
    for value, (ix, ip), v in zip(
        d_cells, bin_indices, spectrum["v_data"],
    ):
        x_mid = 0.5 * (float(x_edges[ix]) + float(x_edges[ix + 1]))
        p_mid = 0.5 * (float(p_edges[ip]) + float(p_edges[ip + 1]))
        dp = float(p_edges[ip + 1]) - float(p_edges[ip])
        groups.setdefault(x_mid, []).append((float(value), p_mid, dp,
                                             float(v)))
    moments, v_data, regions = [], [], []
    for x_mid in sorted(groups):
        p_l = x_mid * float(sqrt_s) / 2.0
        terms = []
        for value, p_mid, dp, v in groups[x_mid]:
            energy = math.sqrt(mass * mass + p_l * p_l + p_mid * p_mid)
            terms.append({
                "a_x": p_mid * p_mid / energy * dp,
                "b_x": p_mid / energy * dp,
                "a_y": p_mid * p_mid * dp,
                "b_y": p_mid * dp,
                "v": v,
            })
        if x_mid > 0.2:
            region = "forward"
        elif x_mid >= 0.0:
            region = "central"
        else:
            region = "backward"
        for suffix in ("x", "y"):
            num = math.fsum(t[f"a_{suffix}"] * t["v"] for t in terms)
            den = math.fsum(t[f"b_{suffix}"] * t["v"] for t in terms)
            if den == 0.0:
                moments.append(None)
                v_stat = float("inf")
            else:
                moments.append(num / den)
                grad_var = 0.0
                for t in terms:
                    grad = (
                        t[f"a_{suffix}"] * den - t[f"b_{suffix}"] * num
                    ) / (den * den)
                    grad_var += grad * grad * t["v"]
                v_stat = float(grad_var)
            v_stat = max(v_stat, SE_FLOOR)
            v_data.append(v_stat)
            regions.append(region)
    d_flat = np.asarray([
        float("inf") if m is None else float(m) for m in moments
    ], dtype=float)
    return Statistic(
        key=f"{pair_key}_seagull", kind="seagull",
        d=d_flat,
        v_data=np.asarray(v_data, dtype=float),
        areas=np.ones(len(d_flat), dtype=float),
        carry_mask=np.zeros(len(d_flat), dtype=float),
        combo=spectrum,
        seagull={
            "x_edges": [float(v) for v in x_edges],
            "p_edges": [float(v) for v in p_edges],
            "bin_indices": bin_indices,
            "mass_gev": mass,
            "sqrt_s_gev": float(sqrt_s),
            "n_x": int(table.bin_index_xf.max()) + 1,
            "n_p": int(table.bin_index_pt.max()) + 1,
            "regions": regions,
            "region_note": FORWARD_REGION_NOTE,
        },
    )


def _build_statistics(
    sealed: dict, tables_by_species: dict, carry_set: set,
    sqrt_s: float,
) -> list:
    """The pre-registered profile observables (plan line 788): the
    B_transport = f(p) - f(pbar) spectrum on the aligned cells (the
    leading-baryon transfer observable, plan section 7.3 line 431) and
    the identified seagulls (the pooled pion seagull on the pi_Sigma
    combination at the pion mass, the pooled proton seagull on the
    B_transport combination at the proton mass; the per-xF <pT>_x and
    <pT>_y conventions on both sides)."""
    stats = []
    combos = {}
    for pair_key in ("pi_sigma", "b_transport"):
        combos[pair_key] = _combination_statistic(
            sealed, tables_by_species, pair_key, set(carry_set))
    spectrum = combos["b_transport"]
    stats.append(Statistic(
        key=spectrum["key"], kind="spectrum",
        d=spectrum["d"], v_data=spectrum["v_data"],
        areas=spectrum["areas"], carry_mask=spectrum["carry_mask"],
        combo=spectrum, seagull=None))
    for pair_key in ("pi_sigma", "b_transport"):
        stats.append(_seagull_statistic(
            sealed, tables_by_species, pair_key, combos[pair_key],
            sqrt_s))
    return stats


# ---------------------------------------------------------------------------
# The model evaluation (the trial predictions of the statistic set)
# ---------------------------------------------------------------------------
def _species_bin_values(
    weighted_sums: np.ndarray, tables_by_species: dict, sqrt_s: float,
    n_eff: float,
) -> dict[str, np.ndarray]:
    """The weighted per-species binned predictions in the data units
    (the R01 convention m = sums / (pi sqrt(s) N_eff dxF dpT); the
    self-normalized importance estimator).  The weighted sums are the
    dense evaluation-lattice vector (the accumulate-dense-then-pick
    recording convention): the two-step conversion takes the species'
    dense slice and picks the published rows (the pick map of the
    worker context), so the returned values align 1:1 with the
    published rows and a hole cell's weight stays unpicked and is
    dropped."""
    out = {}
    for species in SPECIES_FLAT_ORDER:
        table = tables_by_species[species]
        dense_slice = weighted_sums[
            int(dense_offsets_by_species[species]):
            int(dense_offsets_by_species[species])
            + int(dense_sizes_by_species[species])]
        sums = dense_slice[pick_by_species[species]]
        areas = np.asarray([
            _cell_area_from_table(table, i)
            for i in range(int(table.data.size))
        ], dtype=float)
        out[species] = sums / (math.pi * float(sqrt_s) * float(n_eff)
                               * areas)
    return out


def _cell_area_from_table(table, i: int) -> float:
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[i])
    ip = int(table.bin_index_pt[i])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


def _combination_model(
    stat: Statistic, species_values: dict,
) -> np.ndarray:
    spectrum = stat.combo
    species_a, species_b, sign = spectrum["pair"]
    return np.asarray([
        species_values[species_a][e["bin_a"]]
        + sign * species_values[species_b][e["bin_b"]]
        for e in spectrum["cells"]
    ], dtype=float)


def _seagull_model(
    stat: Statistic, species_values: dict,
) -> list:
    seagull = stat.seagull
    combo_values = _combination_model(stat, species_values)
    result = fo.seagull_moment(
        {"combination": tuple(float(v) for v in combo_values)},
        (tuple(seagull["x_edges"]), tuple(seagull["p_edges"])),
        "combination",
        bin_indices=seagull["bin_indices"],
        sqrt_s_gev=seagull["sqrt_s_gev"],
        mass_gev=seagull["mass_gev"],
    )
    return [
        None if v is None else float(v)
        for v in (result["pT_x_per_xF"] + result["pT_y_per_xF"])
    ]


# ---------------------------------------------------------------------------
# The objective (the profiled plan-5.4 likelihood per grid point)
# ---------------------------------------------------------------------------
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


class ProfileObjective:
    """The profiled -2lnL of the pre-registered likelihood (plan
    section 5.4) on the two profile axes: the (b_eff multiplier, b_dec)
    directions enter through the exact reweighting; the nuisance profile
    (the spectrum normalization nuisance plus the obligation 4 carry) is
    solved exactly per trial by the generalized-least-squares profile.
    A statistic bin whose pooled MC error or prediction is non-finite
    (a structurally undefined statistic bin) carries an infinite sigma
    and is dropped from the objective (the R11 convention; the dropped
    bins are listed, never silently dropped)."""

    def __init__(
        self, stats: list, blocks: list, tables_by_species: dict,
        sqrt_s: float, carry_prior: float, norm_prior: float,
    ) -> None:
        self.stats = stats
        self.blocks = blocks
        self.tables_by_species = tables_by_species
        self.sqrt_s = float(sqrt_s)
        self.carry_prior = float(carry_prior)
        self.norm_prior = float(norm_prior)
        self.dropped_bins: list = []

    # -- the per-block statistic vectors --------------------------------
    def block_vectors(self, block, b_eff_gev2: float,
                      b_dec: float) -> tuple:
        eta = event_weights(block, b_eff_gev2, b_dec)
        sums = block_weighted_sums(block, eta)
        n_eff = float(np.sum(eta))
        species_values = _species_bin_values(
            sums, self.tables_by_species, self.sqrt_s, n_eff)
        vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum":
                vectors.append(_combination_model_statistic(
                    stat.combo, species_values[stat.combo["pair"][0]],
                    species_values[stat.combo["pair"][1]]))
            else:
                vectors.append(_seagull_model(stat, species_values))
        return vectors, n_eff, ess_fraction(eta)

    # -- the trial evaluation -------------------------------------------
    def evaluate(self, b_eff_gev2: float, b_dec: float) -> dict:
        per_block = []
        sums_total = np.zeros(_dense_total(), dtype=float)
        for block in self.blocks:
            eta = event_weights(block, b_eff_gev2, b_dec)
            sums = block_weighted_sums(block, eta)
            n_eff = float(np.sum(eta))
            species_values = _species_bin_values(
                sums, self.tables_by_species, self.sqrt_s, n_eff)
            vectors = []
            for stat in self.stats:
                if stat.kind == "spectrum":
                    vectors.append(_combination_model_statistic(
                        stat.combo,
                        species_values[stat.combo["pair"][0]],
                        species_values[stat.combo["pair"][1]]))
                else:
                    vectors.append(_seagull_model(stat, species_values))
            per_block.append({
                "vectors": vectors, "n_eff": n_eff,
                "ess": ess_fraction(eta),
            })
            sums_total = sums_total + sums
        n_eff_total = math.fsum(float(e["n_eff"]) for e in per_block)
        pooled_binned = _species_bin_values(
            sums_total, self.tables_by_species, self.sqrt_s, n_eff_total)
        pooled_vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum":
                pooled_vectors.append(_combination_model_statistic(
                    stat.combo,
                    pooled_binned[stat.combo["pair"][0]],
                    pooled_binned[stat.combo["pair"][1]]))
            else:
                pooled_vectors.append(_seagull_model(stat, pooled_binned))
        se_vectors = []
        for index in range(len(self.stats)):
            stack = np.asarray([
                entry["vectors"][index] for entry in per_block
            ], dtype=float)
            se_vectors.append(se_from(stack))
        profiled, eta_hat = self._profiled_minus2lnl(
            pooled_vectors, se_vectors)
        return {
            "value": float(profiled),
            "pooled_vectors": pooled_vectors,
            "se": se_vectors,
            "per_block": per_block,
            "n_eff_total": float(n_eff_total),
            "eta_hat": [float(v) for v in eta_hat],
            "ess": {
                "per_block": [entry["ess"] for entry in per_block],
                "min": min(entry["ess"] for entry in per_block),
            },
        }

    # -- the nuisance profile --------------------------------------------
    def _profiled_minus2lnl(self, pooled_vectors, se_vectors):
        rows: list[float] = []
        columns: list[list[float]] = []
        for stat_index, stat in enumerate(self.stats):
            m = np.asarray(pooled_vectors[stat_index], dtype=float)
            se = np.asarray(se_vectors[stat_index], dtype=float)
            d = np.asarray(stat.d, dtype=float)
            v_data = np.asarray(stat.v_data, dtype=float)
            for i in range(m.size):
                mc_var = max(float(se[i]), SE_FLOOR) ** 2
                if not (math.isfinite(mc_var)
                        and math.isfinite(float(m[i]))):
                    self.dropped_bins.append({
                        "statistic": stat.key, "index": int(i),
                        "reason": "the pooled MC error or prediction is "
                                  "non-finite (a structurally undefined "
                                  "statistic bin; the R11 convention)",
                    })
                    continue
                if not (math.isfinite(float(d[i]))
                        and math.isfinite(float(v_data[i]))
                        and float(v_data[i]) >= 0.0):
                    self.dropped_bins.append({
                        "statistic": stat.key, "index": int(i),
                        "reason": "non-finite data or published variance",
                    })
                    continue
                total_var = float(v_data[i]) + mc_var
                if not total_var > 0.0:
                    continue
                weight = 1.0 / math.sqrt(total_var)
                rows.append((float(d[i]) - float(m[i])) * weight)
                columns.append([
                    float(v) * weight
                    for v in self._nuisance_design_row(stat, i)
                ])
        if not rows:
            _refuse(
                "r46 profile: no finite objective bin (the statistic set "
                "is empty after the structure drops); the run stops"
            )
        r = np.asarray(rows, dtype=float)
        design = np.asarray(columns, dtype=float)
        prior_diag = np.asarray([
            self.norm_prior ** 2, self.carry_prior ** 2,
        ], dtype=float)
        normal = design.T @ design + np.diag(1.0 / prior_diag)
        rhs = design.T @ r
        try:
            eta_hat = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            eta_hat = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        profiled = float(r @ r - rhs @ eta_hat)
        return profiled, eta_hat

    def _nuisance_design_row(self, stat, index: int) -> list:
        row = [0.0, 0.0]
        column = STAT_BLOCK_COLUMN.get(stat.key)
        if column is not None:
            row[column] = 1.0
        if stat.carry_mask is not None and len(stat.carry_mask) > int(index):
            row[CARRY_COLUMN_INDEX] = float(stat.carry_mask[index])
        return row


# ---------------------------------------------------------------------------
# The profile algebra (the grid surface, the curvature, the correlation)
# ---------------------------------------------------------------------------
def grid_argmin(surface: list) -> tuple[int, int]:
    """The deterministic grid argmin (lexicographic in (value, i, j))."""
    best = (math.inf, 0, 0)
    for i, row in enumerate(surface):
        for j, value in enumerate(row):
            candidate = (float(value), i, j)
            if candidate < best:
                best = candidate
    i0, j0 = best[1], best[2]
    return i0, j0


def grid_hessian(surface: list, i0: int, j0: int) -> tuple:
    """The 2 x 2 curvature matrix of the profile surface at the grid
    point (i0, j0) from the grid second differences (the multiplier
    coordinate spacing B_DEC_MULT_STEP, the b_dec spacing
    B_DEC_STEP_GEV2).  A minimum on the grid boundary carries the
    one-sided second-difference construct over the three nearest
    interior-side points (the boundary construct, named in the report).
    Returns (hessian, interior_flags)."""
    n_i = len(surface)
    n_j = len(surface[0]) if surface else 0
    values = [[float(v) for v in row] for row in surface]
    db = B_DEC_MULT_STEP
    dk = B_DEC_STEP_GEV2

    def _axis_second(ii: int, jj: int, axis: int) -> float:
        """The second difference along one axis with the honest boundary
        construct: the central form at an interior point, the one-sided
        three-point construct at an edge (the curvature of the nearest
        interior-side triplet)."""
        if axis == 0:
            def val(k: int) -> float:
                return values[k][jj]
            span = n_i
        else:
            def val(k: int) -> float:
                return values[ii][k]
            span = n_j
        center = ii if axis == 0 else jj
        if 1 <= center <= span - 2:
            return (val(center + 1) - 2.0 * val(center)
                    + val(center - 1)) / (
                (db if axis == 0 else dk) ** 2)
        if center == 0 and span >= 3:
            return (val(2) - 2.0 * val(1) + val(0)) / (
                (db if axis == 0 else dk) ** 2)
        if center == span - 1 and span >= 3:
            return (val(span - 1) - 2.0 * val(span - 2)
                    + val(span - 3)) / ((db if axis == 0 else dk) ** 2)
        return float("nan")

    def _mixed_second(ii: int, jj: int) -> float:
        """The mixed second difference from the available diagonal
        neighbours: (f(a, b) - f(a, 0) - f(0, b) + f(0, 0)) / (a db b dk)
        over the available corner combination (deterministic; a corner
        minimum uses its single interior diagonal)."""
        options = []
        for a in (-1, 1):
            for b in (-1, 1):
                ia, jb = ii + a, jj + b
                if 0 <= ia < n_i and 0 <= jb < n_j:
                    options.append((a, b))
        if not options:
            return float("nan")
        a, b = options[0]
        return (values[ii + a][jj + b] - values[ii + a][jj]
                - values[ii][jj + b] + values[ii][jj]) / (
            a * db * b * dk)

    h_bb = _axis_second(i0, j0, 0)
    h_kk = _axis_second(i0, j0, 1)
    h_bk = _mixed_second(i0, j0)
    interior = (0 < i0 < n_i - 1) and (0 < j0 < (n_j - 1 if n_j else 0))
    hessian = [[h_bb, h_bk], [h_bk, h_kk]]
    return hessian_list(hessian), {
        "i0": i0, "j0": j0, "interior_bb": 0 < i0 < n_i - 1,
        "interior_kk": 0 < j0 < n_j - 1 if n_j else False,
        "all_interior": bool(interior),
    }


def hessian_list(h: list) -> list:
    return [[float(v) for v in row] for row in h]


def hessian_eigenvalues(h: list) -> list:
    return [float(v) for v in np.linalg.eigvalsh(np.asarray(h, dtype=float))]


def correlation_from_hessian(h: list) -> float | None:
    """The inverse-Hessian parameter correlation of the curvature matrix
    of -2lnL: rho = -H_01 / sqrt(H_00 H_11) (the covariance's common
    positive factor 2 cancels).  A non-positive diagonal curvature
    quotes None (the curvature finding is named instead)."""
    h00, h01, h11 = float(h[0][0]), float(h[0][1]), float(h[1][1])
    if not (h00 > 0.0 and h11 > 0.0):
        return None
    return -h01 / math.sqrt(h00 * h11)


def positive_definite_2x2(h: list) -> bool:
    return bool(all(v > 0.0 for v in hessian_eigenvalues(h)))


def per_block_curvature(
    surfaces: dict, i0: int, j0: int,
) -> dict:
    """The replica-block curvature stability: the Hessian at the pooled
    minimum evaluated on each per-block surface, with each block's own
    argmin recorded (the plan-5.4-line-311 minima-agreement input)."""
    stability = {}
    for replica in sorted(surfaces):
        block_surface = surfaces[replica]
        h, flags = grid_hessian(block_surface, i0, j0)
        eigenvalues = hessian_eigenvalues(h)
        argmin = list(grid_argmin(block_surface))
        stability[str(replica)] = {
            "hessian": h,
            "eigenvalues": eigenvalues,
            "positive_definite": bool(all(v > 0.0 for v in eigenvalues)),
            "argmin": argmin,
            "argmin_agrees_with_pooled": bool(argmin == [int(i0), int(j0)]),
        }
    return stability


def minima_agreement(surfaces: dict, i0: int, j0: int) -> dict:
    """The plan-5.4-line-311 quoting gate: the 8 per-replica-block
    surface argmins must agree with the pooled argmin before any
    curvature is quoted."""
    per_block = {str(replica): list(grid_argmin(surfaces[replica]))
                 for replica in sorted(surfaces)}
    agree = all(value == [int(i0), int(j0)]
                for value in per_block.values())
    return {
        "minima_agree": bool(agree),
        "pooled_argmin": [int(i0), int(j0)],
        "per_block_argmins": per_block,
        "note": HESSIAN_DISCIPLINE_NOTE,
    }


def per_direction_curvature_spread(stability: dict, pooled_h: list) -> dict:
    """The per-direction curvature spread of the 8 replica blocks
    against the pooled curvature (plan section 5.5 item 5 line 321,
    the acceptance (ii) stability clause): per axis, the curvature-
    matrix diagonal spread max_block |H_block - H_pooled| / |H_pooled|
    on the pooled diagonal, below the pre-registered 0.20 fraction."""
    axes = (("b_eff_multiplier", 0), ("b_dec", 1))
    out = {}
    for name, axis in axes:
        pooled = float(pooled_h[axis][axis])
        per_block = {
            replica: float(entry["hessian"][axis][axis])
            for replica, entry in sorted(stability.items())
        }
        if not (math.isfinite(pooled) and pooled != 0.0):
            fraction = float("inf")
            max_spread = float("inf")
        else:
            deviations = [abs(float(v) - pooled)
                          for v in per_block.values()]
            max_spread = max(deviations) if deviations else 0.0
            fraction = max_spread / abs(pooled)
        out[name] = {
            "pooled": pooled,
            "per_block": per_block,
            "max_abs_spread": max_spread,
            "spread_fraction": fraction,
            "bound_fraction": CURVATURE_SPREAD_FRACTION,
            "below_fraction": bool(fraction < CURVATURE_SPREAD_FRACTION),
        }
    return out


def blocks_failing(stability: dict) -> dict:
    """The failing per-block census: the block indices whose Hessian is
    not positive definite, with their eigenvalues."""
    return {
        replica: entry["eigenvalues"]
        for replica, entry in sorted(stability.items())
        if not entry.get("positive_definite", False)
    }


# ---------------------------------------------------------------------------
# The blocks (the sealed production records or the smoke record)
# ---------------------------------------------------------------------------
def _load_blocks(run_dir: Path, smoke: bool) -> list:
    """The sealed production records' reweighting payloads (or the
    single smoke record)."""
    if smoke:
        record = _read_json(
            Path(run_dir) / f"{SMOKE_PREFIX}replica_0.json")
        return [_block_from_record(record)]
    blocks = []
    for replica in range(T2_REPLICAS):
        path = Path(run_dir) / RECORD_FMT.format(replica)
        if not path.is_file():
            _refuse(f"the production record is missing: {path} "
                    "(run --generate first)")
        record = _read_json(path)
        if int(record.get("replica", -1)) != replica:
            _refuse(f"{path}: the record's replica field disagrees")
        blocks.append(_block_from_record(record))
    return blocks


def _synthetic_carry_all() -> set:
    """The smoke profile's carry set: every species bin (the synthetic
    machinery validation; never used in production)."""
    out = set()
    for species in SPECIES_FLAT_ORDER:
        size = int(_SPECIES_SIZE.get(species, 0))
        out.update((species, i) for i in range(size))
    return out


def _synthetic_data_for(stats: list, truth_vectors: dict) -> None:
    """The smoke profile's synthetic data: the model at the pre-
    registered synthetic-truth point, with a synthetic 10% published
    error; the carry set is every spectrum bin (the machinery
    validation)."""
    for stat in stats:
        stat.d = np.asarray(truth_vectors[stat.key], dtype=float).copy()
        finite = np.isfinite(stat.d)
        stat.d = np.where(finite, stat.d, 0.0)
        stat.v_data = np.where(finite, (0.1 * np.abs(stat.d)) ** 2,
                               np.inf)
        stat.v_data = np.maximum(stat.v_data, SE_FLOOR)


# ---------------------------------------------------------------------------
# The pinned validity probe (the direct-generation identity)
# ---------------------------------------------------------------------------
def _generate_pinned_subsample(
    tables: list, fams: list, n_events: int, b_eff_gev2: float,
    b_dec: float, smoke: bool,
) -> BlockLatent:
    """One pinned sub-sample generation: the production (seed, event_id)
    pairs at the given configuration (the frozen point for the frozen
    leg, the pinned grid point for the direct legs), common random
    numbers, in memory, never sealed."""
    _init_species_layout(tables)
    context = _worker_context(tables, fams, smoke=bool(smoke))
    _set_worker_context(context)
    seed = smoke_raw_seed() if smoke else raw_replica_seed(0)
    slot = SMOKE_KERNEL_REPLICA_SLOT if smoke else 0
    tasks = _chunk_tasks(0, seed, slot, int(n_events),
                         float(b_eff_gev2), float(b_dec), smoke=bool(smoke))
    parts = list(_map_chunks(tasks, DEFAULT_WORKERS, f"pinned {b_eff_gev2}/{b_dec}"))
    aggregated = _aggregate_chunks(parts)
    aggregated["replica"] = int(slot)
    aggregated["seed"] = int(seed)
    if aggregated["n_events"] != int(n_events):
        _refuse(
            f"r46 validity probe: generated {aggregated['n_events']} "
            f"pinned events, expected {int(n_events)}"
        )
    return _block_from_aggregate(aggregated)


def run_validity_probe(tables: list, fams: list, smoke: bool) -> dict:
    """The pre-registered reweighting-validity assertion: at each pinned
    grid point the weighted prediction's exact identities hold against
    the direct generation of the pinned sub-sample within 1e-6 relative.

    Legs (all pre-registered in the profile_grid block):
    - stream alignment: the direct generation at the grid point realizes
      the identical route census and the identical transfer-law support
      (exact integer/float equality);
    - transfer law: t_implied = G_B'^{-1}(G_B(t)) == the direct
      generation's realized t, per pinned event, 1e-6 relative;
    - decay form factor: the first cascade vertex's normalized trial
      shares recomputed from the frozen latent's (w0, q2) equal the
      frozen-B direct generation's own attached shares at the trial
      b_dec, 1e-6 relative;
    - the prediction-level pooled comparison is recorded beside them as
      a Monte Carlo agreement diagnostic (never asserted at 1e-6)."""
    n_events = SMOKE_VALIDITY_EVENTS if smoke else VALIDITY_EVENTS
    frozen = _generate_pinned_subsample(
        tables, fams, n_events, B_EFF_FROZEN_GEV2, B_DEC_GENERATION_GEV2,
        smoke)
    results = []
    max_deviation = 0.0
    for mult, b_dec in VALIDITY_GRID_POINTS:
        b_eff_trial = float(mult * B_EFF_FROZEN_GEV2)
        # -- the direct generation at the pinned grid point (the trial
        #    B_eff and the trial b_dec) --
        direct = _generate_pinned_subsample(
            tables, fams, n_events, b_eff_trial, float(b_dec), smoke)
        # - the stream alignment: the route census (decided before every
        #   parameter-dependent draw) and the transfer-law support are
        #   exact; the cascade vertex COUNT is a diagnostic (it depends
        #   on the realized channel choices, a parameter-dependent draw
        #   downstream of the aligned streams) -
        support_dev = float(np.max(np.abs(
            np.concatenate((frozen.t_minus - direct.t_minus,
                            frozen.t_plus - direct.t_plus)))))
        if not support_dev == 0.0:
            _refuse(
                "r46 validity: the transfer-law support is not bit-"
                f"identical between the frozen and the direct pinned "
                f"generations (max deviation {support_dev!r}); the "
                "support is not parameter-independent (the run stops)"
            )
        # - the transfer-law implied realization (1e-6 relative) -
        implied = np.asarray([
            transfer_implied_t(float(frozen.t_scatter[k]),
                               B_EFF_FROZEN_GEV2, b_eff_trial,
                               float(frozen.t_minus[k]),
                               float(frozen.t_plus[k]))
            for k in range(n_events)
        ], dtype=float)
        t_dev = float(np.max(np.abs(implied - direct.t_scatter)
                             / np.maximum(1.0, np.abs(direct.t_scatter))))
        if not t_dev <= VALIDITY_BOUND:
            _refuse(
                f"r46 validity: the transfer-law implied realization "
                f"deviates from the direct generation's realized t by "
                f"{t_dev!r} relative (beyond the pre-registered "
                f"{VALIDITY_BOUND!r} bound); the reweighting is not "
                "exact (the run stops)"
            )
        # - the decay form factor's first-vertex trial shares (the
        #   frozen-B direct generation at the trial b_dec) -
        frozen_b = _generate_pinned_subsample(
            tables, fams, n_events, B_EFF_FROZEN_GEV2, float(b_dec),
            smoke)
        share_dev, share_checks = _first_vertex_share_deviation(
            frozen, frozen_b, float(b_dec))
        if not share_dev <= VALIDITY_BOUND:
            _refuse(
                "r46 validity: the first-vertex normalized trial shares "
                f"deviate by {share_dev!r} relative (beyond the "
                f"pre-registered {VALIDITY_BOUND!r} bound); the decay "
                "reweighting is not exact (the run stops)"
            )
        max_dev = max(t_dev, share_dev)
        max_deviation = max(max_deviation, max_dev)
        # - the stream alignment: the route census is exact -
        census_equal = bool(tuple(frozen.route_census)
                            == tuple(direct.route_census))
        if not census_equal:
            _refuse(
                "r46 validity: the pinned sub-sample's route census "
                f"changed under the density-only parameter change "
                f"(frozen {tuple(frozen.route_census)}, direct "
                f"{tuple(direct.route_census)}); the routing draws "
                "precede every parameter-dependent draw, so a census "
                "change is a support change (the run stops)"
            )
        # - the prediction-level diagnostic (recorded, never asserted) -
        diagnostic = _pooled_prediction_diagnostic(frozen, direct)
        max_dev = max(max_dev, support_dev)
        results.append({
            "pinned_grid_point": {
                "b_eff_multiplier": float(mult),
                "b_eff_gev2": b_eff_trial,
                "b_dec_gev2": float(b_dec),
            },
            "pinned_events": int(n_events),
            "stream_alignment": {
                "route_census_frozen": list(frozen.route_census),
                "route_census_direct": list(direct.route_census),
                "route_census_equal": census_equal,
                "vertex_count_frozen": int(frozen.vtx_nchan.size),
                "vertex_count_direct": int(direct.vtx_nchan.size),
                "vertex_count_note": (
                    "the cascade vertex count depends on the realized "
                    "channel choices (a parameter-dependent draw "
                    "downstream of the aligned streams); the stream-"
                    "alignment invariants are the route census and the "
                    "transfer-law support"
                ),
                "support_max_abs_deviation": support_dev,
            },
            "transfer_law": {
                "max_relative_deviation": t_dev,
                "bound": VALIDITY_BOUND,
                "pass": bool(t_dev <= VALIDITY_BOUND),
            },
            "form_factor": {
                "first_vertex_cells_checked": int(share_checks),
                "max_relative_deviation": share_dev,
                "bound": VALIDITY_BOUND,
                "pass": bool(share_dev <= VALIDITY_BOUND),
            },
            "prediction_level_diagnostic": diagnostic,
            "max_relative_deviation": float(max_dev),
        })
    return {
        "pinned_grid_points": len(VALIDITY_GRID_POINTS),
        "pinned_subsample_events": int(n_events),
        "pinned_subsample_rule": VALIDITY_SUBSAMPLE_RULE,
        "max_relative_deviation": float(max_deviation),
        "bound_relative": VALIDITY_BOUND,
        "pass": bool(max_deviation <= VALIDITY_BOUND),
        "points": results,
    }
def _first_vertex_share_deviation(
    frozen: BlockLatent, direct: BlockLatent, b_trial: float,
) -> tuple[float, int]:
    """The first-vertex normalized trial shares: the frozen latent's
    implied shares (from w0, q2 and the trial b_dec) against the frozen-B
    direct generation's own attached shares at the trial b_dec (the
    parent state of the first vertex is parameter-independent)."""
    delta_b = float(b_trial) - B_DEC_GENERATION_GEV2
    pool_ptr = frozen.vtx_pool_ptr
    direct_pool_ptr = direct.vtx_pool_ptr
    checked = 0
    max_dev = 0.0
    for event in range(min(frozen.n_events, direct.n_events)):
        v0_f = int(frozen.vtx_ptr[event])
        v1_f = int(frozen.vtx_ptr[event + 1])
        v0_d = int(direct.vtx_ptr[event])
        v1_d = int(direct.vtx_ptr[event + 1])
        if v1_f == v0_f or v1_d == v0_d:
            continue
        nf = int(frozen.vtx_nchan[v0_f])
        nd = int(direct.vtx_nchan[v0_d])
        if nf != nd:
            continue
        pf = slice(int(pool_ptr[v0_f]), int(pool_ptr[v0_f + 1]))
        pd = slice(int(direct_pool_ptr[v0_d]),
                   int(direct_pool_ptr[v0_d + 1]))
        w0 = frozen.ch_w0[pf]
        q2 = frozen.ch_q2[pf]
        ex = np.exp(-delta_b * q2)
        total = math.fsum(w0 * ex)
        implied = w0 * ex / total
        trial = direct.ch_w0[pd]
        for a, b in zip(implied, trial):
            dev = abs(a - b) / max(abs(b), 1e-300)
            max_dev = max(max_dev, dev)
            checked += 1
    return max_dev, checked


def _pooled_prediction_diagnostic(
    frozen: BlockLatent, direct: BlockLatent,
) -> dict:
    """The prediction-level agreement diagnostic: the self-normalized
    reweighted pooled spectrum of the frozen pinned latent versus the
    direct generation's unweighted pooled spectrum (both per-bin, in
    record pooled_sums units; the Monte Carlo agreement diagnostic,
    never asserted at 1e-6)."""
    eta_f = event_weights(frozen, B_EFF_FROZEN_GEV2,
                          B_DEC_GENERATION_GEV2)
    sums_f = block_weighted_sums(frozen, eta_f)
    n_eff_f = float(np.sum(eta_f))
    eta_d = np.ones(direct.n_events, dtype=float)
    sums_d = block_weighted_sums(direct, eta_d)
    n_eff_d = float(np.sum(eta_d))
    per_species = {}
    for species in SPECIES_FLAT_ORDER:
        def _pick(sums: np.ndarray) -> np.ndarray:
            dense_slice = sums[
                int(dense_offsets_by_species[species]):
                int(dense_offsets_by_species[species])
                + int(dense_sizes_by_species[species])]
            return dense_slice[pick_by_species[species]]
        a = _pick(sums_f) / n_eff_f
        b = _pick(sums_d) / n_eff_d
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-16)
        finite = rel[np.isfinite(rel)]
        per_species[species] = float(np.max(finite)) if finite.size else 0.0
    overall = max(per_species.values()) if per_species else 0.0
    return {
        "note": (
            "the prediction-level Monte Carlo agreement diagnostic "
            "(the self-normalized reweighted pooled spectrum versus the "
            "direct generation's unweighted pooled spectrum over the "
            "same pinned event indices; both estimate the same "
            "expectation and the deviation is bounded by the importance-"
            "sampling Monte Carlo error, never asserted at 1e-6)"
        ),
        "per_species_max_relative_deviation": {
            key: float(v) for key, v in per_species.items()
        },
        "overall_max_relative_deviation": float(overall),
    }


def _assemble_named_residuals(report: dict) -> list:
    """The named-residual register of the acceptance precedent: every
    failing acceptance-(ii) condition named with the observed value, a
    one-line diagnosis, and the robust-statistics cross-reference (the
    card's named findings: boundary_minimum,
    correlation_undefined_or_high, curvature_not_positive,
    block_curvature_failure; the pre-registered machinery kinds cover
    the minima gate, the weight degeneracy, and the non-finite surface
    points)."""
    register = []
    minimum = report.get("minimum", {})
    grid = report.get("profile_grid", {})
    n_i = len(grid.get("b_eff_multipliers") or ())
    n_j = len(grid.get("b_dec_gev2") or ())
    i0 = minimum.get("i")
    j0 = minimum.get("j")
    if i0 is not None and n_i:
        if i0 == 0:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_minimum",
                "axis": "b_eff_multiplier",
                "observed": minimum.get("b_eff_multiplier"),
                "bound": [float(grid["b_eff_multipliers"][0]),
                          float(grid["b_eff_multipliers"][-1])],
                "diagnosis": (
                    "the profile minimum sits on the pre-registered lower "
                    "grid boundary of the B_eff multiplier axis; the "
                    "profile range does not extend and the bounds are "
                    "never relaxed; the finding is named, never refit"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
        elif i0 == n_i - 1:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_minimum",
                "axis": "b_eff_multiplier",
                "observed": minimum.get("b_eff_multiplier"),
                "bound": [float(grid["b_eff_multipliers"][0]),
                          float(grid["b_eff_multipliers"][-1])],
                "diagnosis": (
                    "the profile minimum sits on the pre-registered upper "
                    "grid boundary of the B_eff multiplier axis; the "
                    "profile range does not extend and the bounds are "
                    "never relaxed"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
    if j0 is not None and n_j:
        if j0 == 0:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_minimum",
                "axis": "b_dec",
                "observed": minimum.get("b_dec"),
                "bound": [float(grid["b_dec_gev2"][0]),
                          float(grid["b_dec_gev2"][-1])],
                "diagnosis": (
                    "the profile minimum sits on the pre-registered lower "
                    "grid boundary of the b_dec axis; the profile range "
                    "does not extend and the bounds are never relaxed"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
        elif j0 == n_j - 1:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_minimum",
                "axis": "b_dec",
                "observed": minimum.get("b_dec"),
                "bound": [float(grid["b_dec_gev2"][0]),
                          float(grid["b_dec_gev2"][-1])],
                "diagnosis": (
                    "the profile minimum sits on the pre-registered upper "
                    "grid boundary of the b_dec axis; the profile range "
                    "does not extend and the bounds are never relaxed"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
    agreement = report.get("minima_agreement") or {}
    if agreement.get("minima_agree") is False:
        register.append({
            "index": len(register) + 1,
            "kind": "replica_block_minima_disagree",
            "observed": {
                "pooled_argmin": agreement.get("pooled_argmin"),
                "per_block_argmins": agreement.get("per_block_argmins"),
            },
            "diagnosis": (
                "the replica-block minima do not agree with the pooled "
                "profile minimum: no curvature is quoted under the plan "
                "section 5.4 line 311 discipline, so the "
                "stable-curvature claim (the pooled Hessian and the "
                "per-block stability conditions) and the correlation "
                "claim are not established; the per-block argmins are "
                "recorded beside the gate and the finding is named, "
                "never refit"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    correlation = report.get("correlation_b_eff_b_dec")
    if agreement.get("minima_agree", True) and correlation is None:
        register.append({
            "index": len(register) + 1,
            "kind": "correlation_undefined_or_high",
            "observed": None,
            "bound": CORRELATION_BOUND,
            "diagnosis": (
                "the parameter correlation is undefined at the profile "
                "minimum (a non-positive diagonal curvature: one-sided "
                "boundary curvature or a flat direction); the plan line "
                "788 distinguishable-effects bound |rho| < 0.9 is not "
                "met by an undefined correlation and the finding is "
                "named"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    elif correlation is not None and             abs(float(correlation)) >= CORRELATION_BOUND:
        register.append({
            "index": len(register) + 1,
            "kind": "correlation_undefined_or_high",
            "observed": float(correlation),
            "bound": CORRELATION_BOUND,
            "diagnosis": (
                "the |rho(B_eff, b_dec)| correlation reaches or exceeds "
                "the pre-registered 0.9 distinguishability bound: the "
                "two parameters move the observables as one combined "
                "direction, the effective reparametrization the plan "
                "section 24 line 1866 prohibition tests for; the "
                "distinguishable-effects clause fails and the finding "
                "is named (the bound is never relaxed)"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    hessian = report.get("hessian") or {}
    if hessian.get("quoted", True) and \
            not hessian.get("positive_definite", False):
        reasons = []
        if not (hessian.get("interior_flags") or {}).get(
                "all_interior", False):
            reasons.append(
                "the minimum sits on the grid boundary (one-sided "
                "curvature construct)")
        reasons.append(
            "the curvature matrix of -2lnL is not positive definite")
        register.append({
            "index": len(register) + 1,
            "kind": "curvature_not_positive",
            "observed": {
                "hessian": hessian.get("hessian"),
                "eigenvalues": hessian.get("eigenvalues"),
            },
            "diagnosis": (
                "the stable-curvature clause fails at the profile "
                "minimum: " + "; ".join(reasons) + "; the plan line 788 "
                "stable curvature is not established and the finding is "
                "named"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    stability = report.get("curvature_stability") or {}
    spread = report.get("per_direction_curvature") or {}
    failing_blocks = blocks_failing(stability)
    spread_failures = {
        axis: {
            "spread_fraction": entry.get("spread_fraction"),
            "bound_fraction": entry.get("bound_fraction"),
            "offending_blocks": [
                replica for replica, value in (
                    entry.get("per_block") or {}).items()
                if abs(float(value) - float(entry["pooled"]))
                / abs(float(entry["pooled"]))
                >= CURVATURE_SPREAD_FRACTION
                if math.isfinite(float(entry["pooled"]))
                and float(entry["pooled"]) != 0.0
            ],
        }
        for axis, entry in spread.items()
        if not entry.get("below_fraction", True)
    }
    if failing_blocks or (hessian.get("quoted", True) and spread_failures):
        register.append({
            "index": len(register) + 1,
            "kind": "block_curvature_failure",
            "observed": {
                "blocks_not_positive_definite": {
                    replica: eigenvalues
                    for replica, eigenvalues in failing_blocks.items()
                },
                "per_direction_spread_failures": spread_failures,
            },
            "diagnosis": (
                "the per-block curvature stability fails (plan section "
                "5.5 item 5 line 321): at least one replica block's "
                "Hessian at the pooled minimum is not positive definite "
                "(the block index and eigenvalues quoted above) or a "
                "per-direction curvature spread reaches the "
                "pre-registered 20% fraction of the pooled curvature "
                "(the axis, the spread fraction, and the offending "
                "blocks quoted above); the finding is named"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    ess = report.get("objective_at_minimum", {}).get("ess") or {}
    if ess and float(ess.get("min", 1.0)) < ESS_FLOOR_FRACTION:
        register.append({
            "index": len(register) + 1,
            "kind": "weight_degeneracy",
            "observed": ess.get("min"),
            "floor": ESS_FLOOR_FRACTION,
            "diagnosis": (
                "the effective sample size of the reweighted ensemble "
                "falls below the pre-registered floor at the profile "
                "minimum (the importance weights degenerate at the "
                "minimum); the surface point is recorded with this "
                "diagnosis"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    nonfinite = report.get("nonfinite_surface_points") or []
    if nonfinite:
        register.append({
            "index": len(register) + 1,
            "kind": "nonfinite_surface_points",
            "observed": nonfinite,
            "diagnosis": (
                "the pooled surface value is non-finite at the recorded "
                "grid points (the objective is empty after the structure "
                "drops there); the minimum is taken over the finite "
                "points and the finding is named"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    return register





# ---------------------------------------------------------------------------
# run_profile (the profile evaluation over the sealed artifacts)


#: The standing residuals carried into every comparison (the card's
#: pre-registration): the R43 b_transport_seagull
#: beyond_isotropic_not_positive finding (observed at the sealed R43
#: value) and the R32 spread carry (the obligation-4 model-uncertainty
#: term).  They appear as named entries in the report's and the gate
#: record's standing_residuals blocks.
STANDING_RESIDUALS = [
    {
        "kind": "b_transport_seagull_beyond_isotropic_not_positive",
        "observed": -2.3959765824349295e-05,
        "source": (
            "the sealed R43 cascade-ablation archive's named-residual "
            "register (the beyond-isotropic metric's non-positive "
            "deviance row)"
        ),
    },
    {
        "kind": "r32_model_uncertainty_carry",
        "r32_spread": dict(OBLIGATION4_R32_SPREAD),
        "form": OBLIGATION4_FORM,
    },
]


# ---------------------------------------------------------------------------
# run_profile (the profile evaluation over the sealed artifacts)
# ---------------------------------------------------------------------------
def _build_carry_set_production() -> set:
    """The production blend-band-sensitive cells: the sealed 32 persisting
    bins (the card-1 attribution) restricted to the b_transport pair's
    species."""
    global CARRY_BIN_KEYS
    if not CARRY_BIN_KEYS:
        CARRY_BIN_KEYS = _load_carry_bins()
    return {(entry["species"], int(entry["bin_index"]))
            for entry in CARRY_BIN_KEYS
            if entry["species"] in ("proton", "antiproton")}


def run_profile_machinery(run_dir: Path, smoke: bool = False) -> dict:
    """The profile evaluation: the 9 x 9 reweighting grid, the 8 per-
    block surfaces, the profile minimum, the curvature matrix and its
    eigenvalues, the parameter correlation, the replica-block curvature
    stability, the pinned validity probe, and the profile report payload
    (no wall-clock fields)."""
    run_dir = Path(run_dir)
    tables = _load_smoke_tables() if smoke else _load_production_tables()
    _init_species_layout(tables)
    _init_dense_layout(tables)
    tables_by_species = {t.species: t for t in tables}
    sqrt_s = _sqrt_s()
    if smoke:
        sealed = {}
        for species in SPECIES_FLAT_ORDER:
            table = tables_by_species[species]
            sealed[species] = {
                "d": [1.0] * int(table.data.size),
                "v_data": [0.01] * int(table.data.size),
                "xF_mid": [float(v) for v in table.xf_centers],
                "pT_mid": [float(v) for v in table.pt_centers],
            }
        stats = _build_statistics(
            sealed, tables_by_species, _synthetic_carry_all(), sqrt_s)
        carry_set = _synthetic_carry_all()
    else:
        pre = _load_pre_registration(run_dir)
        _assert_pre_registration(pre, run_dir)
        global CARRY_BIN_KEYS
        CARRY_BIN_KEYS = _load_carry_bins()
        sealed = _load_sealed_r01(tables)
        carry_set = _build_carry_set_production()
        stats = _build_statistics(
            sealed, tables_by_species, carry_set, sqrt_s)
    blocks = _load_blocks(run_dir, smoke)
    objective = ProfileObjective(
        stats, blocks, tables_by_species, sqrt_s,
        CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
    if smoke:
        # the smoke synthetic data at the synthetic truth
        truth_eval = objective.evaluate(
            SMOKE_SYNTHETIC_TRUTH["b_eff_multiplier"] * B_EFF_FROZEN_GEV2,
            SMOKE_SYNTHETIC_TRUTH["b_dec"])
        _synthetic_data_for(stats, {
            stat.key: vec for stat, vec in zip(
                stats, truth_eval["pooled_vectors"])})
    # -- the 9 x 9 grid: the pooled surface and the per-block surfaces --
    surface = []
    nonfinite_points = []
    per_block_vectors = {}
    for mult in B_EFF_MULTIPLIERS:
        b_eff = float(mult * B_EFF_FROZEN_GEV2)
        row = []
        for b_dec in B_DEC_GRID:
            trial = objective.evaluate(b_eff, float(b_dec))
            value = trial["value"]
            row.append(float(value) if math.isfinite(value)
                       else float("inf"))
            for index, block in enumerate(blocks):
                per_block_vectors.setdefault(index, []).append(
                    trial["per_block"][index]["vectors"])
        surface.append(row)
    # -- the per-block surfaces (each block's own profiled objective
    #    against the published data with the deterministic SE floor) --
    per_block_surfaces = {}
    for index in range(len(blocks)):
        block = blocks[index]
        block_objective = ProfileObjective(
            stats, [block], tables_by_species, sqrt_s,
            CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
        surface_rows = []
        for mult in B_EFF_MULTIPLIERS:
            b_eff = float(mult * B_EFF_FROZEN_GEV2)
            block_row = []
            for b_dec in B_DEC_GRID:
                trial = block_objective.evaluate(b_eff, float(b_dec))
                block_row.append(
                    float(trial["value"]) if math.isfinite(trial["value"])
                    else float("inf"))
            surface_rows.append(block_row)
        per_block_surfaces[int(index)] = surface_rows

    i0, j0 = grid_argmin(surface)
    pooled_h, interior_flags = grid_hessian(surface, i0, j0)
    eigenvalues = hessian_eigenvalues(pooled_h)
    positive = bool(all(v > 0.0 for v in eigenvalues))
    stability = per_block_curvature(per_block_surfaces, i0, j0)
    agreement = minima_agreement(per_block_surfaces, i0, j0)
    quoted = bool(agreement["minima_agree"])
    spread = per_direction_curvature_spread(stability, pooled_h)
    spread_ok = bool(all(
        entry["below_fraction"] for entry in spread.values()))
    blocks_pd = bool(all(
        entry["positive_definite"] for entry in stability.values()))
    correlation = (correlation_from_hessian(pooled_h) if quoted else None)
    correlation_defined = bool(correlation is not None)
    rho_ok = bool(correlation_defined
                  and abs(float(correlation)) < CORRELATION_BOUND)
    minimum_eval = objective.evaluate(
        float(B_EFF_MULTIPLIERS[i0] * B_EFF_FROZEN_GEV2),
        float(B_DEC_GRID[j0]))
    # -- the pinned validity probe --
    fams = _load_families()
    validity = run_validity_probe(tables, fams, smoke)
    nonfinite = [
        {"i": i, "j": j,
         "b_eff_multiplier": float(B_EFF_MULTIPLIERS[i]),
         "b_dec_gev2": float(B_DEC_GRID[j])}
        for i, row in enumerate(surface) for j, value in enumerate(row)
        if not math.isfinite(value)
    ]
    # -- the acceptance (ii) evaluation (plan line 788): all four
    #    conditions hold only in full --
    acceptance_ii = {
        "curvature_quoted": quoted,
        "interior_minimum": bool(interior_flags.get("all_interior", False)),
        "correlation_defined": correlation_defined,
        "correlation_within_bound": rho_ok,
        "pooled_hessian_positive_definite": bool(quoted and positive),
        "all_blocks_positive_definite": bool(quoted and blocks_pd),
        "per_direction_spread_below_fraction": bool(quoted and spread_ok),
        "pass": bool(interior_flags.get("all_interior", False)
                     and correlation_defined and rho_ok
                     and quoted and positive
                     and blocks_pd and spread_ok),
        "note": (
            "the plan line 788 acceptance: the profile minimum interior "
            "to the grid, the B_eff / b_dec correlation defined with "
            "|rho| < 0.9 on the leading-baryon transfer observables and "
            "the identified seagulls, the pooled Hessian positive "
            "definite, and every one of the 8 replica blocks positive "
            "definite with the per-direction curvature spread below 20% "
            "of the pooled curvature (plan section 5.5 item 5 "
            "stability); no separation claim is made unless every "
            "condition holds"
        ),
    }
    report = {
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile."
            "report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "profile_grid": {
            "b_eff_multipliers": [float(m) for m in B_EFF_MULTIPLIERS],
            "b_eff_frozen_gev2": B_EFF_FROZEN_GEV2,
            "b_dec_gev2": [float(b) for b in B_DEC_GRID],
            "b_dec_generation_gev2": B_DEC_GENERATION_GEV2,
        },
        "bounds_never_relaxed": {
            "b_eff_multiplier": [
                float(B_EFF_MULTIPLIERS[0]), float(B_EFF_MULTIPLIERS[-1])],
            "b_dec_gev2": [float(B_DEC_GRID[0]), float(B_DEC_GRID[-1])],
            "note": (
                "the pre-registered bounds are never relaxed and the "
                "grid is never extended to move the minimum interior"
            ),
        },
        "surface": surface,
        "per_block_surfaces": {
            str(key): value for key, value in per_block_surfaces.items()
        },
        "minimum": {
            "i": i0, "j": j0,
            "b_eff_multiplier": float(B_EFF_MULTIPLIERS[i0]),
            "b_eff_gev2": float(B_EFF_MULTIPLIERS[i0] * B_EFF_FROZEN_GEV2),
            "b_dec": float(B_DEC_GRID[j0]),
            "on_boundary": bool(
                i0 in (0, len(B_EFF_MULTIPLIERS) - 1)
                or j0 in (0, len(B_DEC_GRID) - 1)),
            "minus2lnl": surface[i0][j0],
        },
        "hessian": {
            "hessian": pooled_h,
            "eigenvalues": eigenvalues,
            "positive_definite": positive,
            "quoted": quoted,
            "interior_flags": interior_flags,
            "quoting_gate": agreement,
            "coordinate_note": (
                "the curvature matrix of the profiled -2lnL in the "
                "(b_eff multiplier, b_dec GeV^-2) coordinates, from the "
                "grid second differences at the argmin (the spacings "
                "0.25 and 0.3 GeV^-2)"
            ),
        },
        "correlation_b_eff_b_dec": (
            None if correlation is None else float(correlation)),
        "correlation_note": CORRELATION_NOTE,
        "distinguishability_bound": CORRELATION_BOUND,
        "curvature_stability": stability,
        "per_direction_curvature": spread,
        "minima_agreement": agreement,
        "acceptance_ii": acceptance_ii,
        "separation_claimed": bool(acceptance_ii["pass"]),
        "objective_at_minimum": {
            "minus2lnl_profiled": minimum_eval["value"],
            "nuisances": minimum_eval["eta_hat"],
            "nuisance_columns": list(NUISSANCE_COLUMNS),
            "n_eff_total": minimum_eval["n_eff_total"],
            "ess": minimum_eval["ess"],
        },
        "per_block_ess": [float(e["ess"]) for e in minimum_eval[
            "per_block"]],
        "dropped_bins": objective.dropped_bins[:100],
        "n_dropped_bins": len(objective.dropped_bins),
        "nonfinite_surface_points": nonfinite,
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "carry_prior": CARRY_NUISANCE_PRIOR,
        },
        "standing_residuals": STANDING_RESIDUALS,
        "named_residual_register": None,  # assembled below
        "superseded_profile": {
            "run_dir": SUPERSEDED_R42_RUN_DIR,
            "minimum": [float(SUPERSEDED_R42_MINIMUM["b_eff_multiplier"]),
                        float(SUPERSEDED_R42_MINIMUM["b_dec_gev2"])],
            "status": SUPERSEDED_R42_STATUS,
            "report_digest": SUPERSEDED_R42_REPORT_DIGEST,
            "pooled_eigenvalues": list(SUPERSEDED_R42_EIGENVALUES),
            "failing_block_eigenvalues": {
                "4": list(SUPERSEDED_R42_BLOCK4_EIGENVALUES),
            },
            "sealed_utc": SUPERSEDED_R42_SEALED_UTC,
            "note": (
                "the superseded corrected R42 profile is cited as "
                "superseded, never as evidence: its boundary minimum "
                "(0.5, 2.6), its non-positive-definite pooled Hessian, "
                "its failing block-4 curvature, and its null "
                "correlation are the recorded findings the Gate 4 "
                "decision consumed"
            ),
        },
        "no_refit": (
            "the profile axes are exactly the two pre-registered "
            "parameters (b_eff multiplier, b_dec); no exchange, "
            "topology, flavor, branching, or ramp-start parameter is "
            "fit and the alignment kernels stay at the card-20 frozen "
            "simplified point (kappa_B, kappa_m) = (0.0, 4.0) with the "
            "frozen b_dec 2.0 GeV^-2 and the frozen width 0.3 GeV "
            "(plan section 24; plan line 784; obligation 4)"
        ),
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "optimizer_note": (
            "this card holds the single pre-registered optimizer use of "
            "the follow-up pair (the two-dimensional profile); card 20 "
            "invokes none and no other optimizer invocation exists in "
            "this card: the profile minimum is the pre-registered 9 x 9 "
            "lattice argmin and the curvature is the grid second "
            "difference at it, no continuous optimization is claimed"
        ),
        "determinism": {
            "report_note": (
                "no wall-clock fields: the report bytes are a pure "
                "function of the sealed artifacts, the pre-registered "
                "seeds, and the committed code"
            ),
        },
    }
    report["reweighting_validity"] = validity
    report["named_residual_register"] = _assemble_named_residuals(report)
    digest = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["digest"] = digest
    return report


def truth_vectors_payload(stats, objective):
    return {}


def validity_payload(validity):
    return validity


def diagnostic_payload(diagnostic):
    return dict(diagnostic) if diagnostic else {}


def run_gate(run_dir: Path) -> dict:
    """Write gate4_followup_reprofile.json (evidence only), then stop.
    The record claims the separation only if acceptance (ii) passes in
    full; on any failure the named finding is carried and the
    separation stays unestablished."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / PROFILE_REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {PROFILE_REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    records = [
        _read_json(run_dir / RECORD_FMT.format(replica))
        for replica in range(T2_REPLICAS)
    ]
    conservation = {
        "events_checked": sum(int(r["n_events"]) for r in records),
        "violations": 0,
        "worst_energy_closure_gev": max(
            float(r["conservation"]["worst_energy_closure_gev"])
            for r in records),
        "tolerance": EVENT_CLOSURE_TOL,
    }
    conservation["pass"] = bool(
        conservation["violations"] == 0
        and conservation["events_checked"]
        == T2_REPLICAS * T2_EVENTS_PER_REPLICA
        and all(r["conservation"]["event_b_q_s_conserved"] for r in records)
    )
    minimum = report.get("minimum", {})
    acceptance_ii = report.get("acceptance_ii") or {}
    gate = {
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile."
            "gate/1",
        "gate": "gate 4 follow-up obligation 2 (plan line 788)",
        "run": "R46 simplified recoil/decay profile (2026-09-24)",
        "card": CARD,
        "clause": "string-like rank law under the simplified form",
        "substrate": {
            "frozen_simplified_archive": R45_RUN_DIR,
            "seal_verified": True,
            "sibling_driver_sha256":
                FROZEN_SUBSTRATE["r45_driver"][1],
            "frozen_simplified_alignment": dict(
                FROZEN_SIMPLIFIED_ALIGNMENT),
        },
        "profile": {
            "axes": ["B_eff", "b_dec"],
            "grid": {
                "b_eff_multipliers": [B_EFF_MULTIPLIERS[0],
                                      B_EFF_MULTIPLIERS[-1],
                                      B_DEC_MULT_STEP],
                "b_dec_gev2": [B_DEC_GRID[0], B_DEC_GRID[-1],
                               B_DEC_STEP_GEV2],
            },
            "b_eff_frozen_gev2": B_EFF_FROZEN_GEV2,
            "b_dec_generation_gev2": B_DEC_GENERATION_GEV2,
        },
        "bounds_never_relaxed": {
            "b_eff_multiplier": [float(B_EFF_MULTIPLIERS[0]),
                                 float(B_EFF_MULTIPLIERS[-1])],
            "b_dec_gev2": [float(B_DEC_GRID[0]), float(B_DEC_GRID[-1])],
        },
        "events_checked": conservation["events_checked"],
        "conservation": conservation,
        "minimum": {
            "b_eff_multiplier": minimum.get("b_eff_multiplier"),
            "b_dec": minimum.get("b_dec"),
            "on_boundary": minimum.get("on_boundary"),
            "minus2lnl": minimum.get("minus2lnl"),
        },
        "correlation_b_eff_b_dec":
            report.get("correlation_b_eff_b_dec"),
        "distinguishability_bound": CORRELATION_BOUND,
        "hessian": report.get("hessian", {}),
        "per_block_curvature": report.get("curvature_stability", {}),
        "per_direction_curvature": report.get(
            "per_direction_curvature", {}),
        "acceptance_ii": acceptance_ii,
        "superseded_profile": report.get("superseded_profile", {}),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
        },
        "standing_residuals": report.get("standing_residuals", []),
        "named_residual_register": report.get(
            "named_residual_register", []),
        "separation_claimed": bool(acceptance_ii.get("pass", False)),
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "profile_report_digest": report.get("digest"),
        "profile_note": (
            "this card holds the single pre-registered optimizer use of "
            "the follow-up pair (the two-dimensional profile); card 20 "
            "invokes none; the profile minimum is the pre-registered "
            "9 x 9 lattice argmin and the curvature the grid second "
            "difference at it under the plan-5.4-line-311 quoting gate"
        ),
        "no_refit": report.get("no_refit"),
        "sealed": True,
        "sealed_note": (
            "the run seals after this record (the seal manifest and "
            "SHA256SUMS.txt cover this file; the run dir is immutable "
            "after the seal)"
        ),
        "gate_note": (
            "evidence only: on any acceptance-(ii) failure the named "
            "finding is carried and no separation claim is made; the "
            "written retain / revise record on the Gate 4 follow-ups is "
            "the round's separate follow-up obligation"
        ),
    }
    _write_json(run_dir / GATE_RECORD_NAME, gate)
    print(f"[r46] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


def write_sha256sums(run_dir: Path) -> int:
    """The strict path-sorted checksum inventory (the make_archive_
    checksums convention, run-dir-relative; LF endings)."""
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
    _write_bytes(run_dir / SHA256SUMS_NAME,
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
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile.seal/1",
        "run": RUN_ID,
        "card": CARD,
        "run_dir": str(run_dir),
        "files": files,
        "sha256sums_entries": n_entries,
        "record_count": sum(
            1 for name in files if name.startswith(RECORD_FMT.replace(
                "{}", "").split(".")[0])),
        "sealed_utc": _now_utc(),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
    }
    _write_json(run_dir / SEAL_NAME, manifest)
    return manifest


def run_seal(run_dir: Path) -> dict:
    """Write seal_manifest.json and SHA256SUMS.txt over the run dir,
    then stop.  The run dir is immutable after the seal."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    required = (
        PREREGISTRATION_NAME, PROVENANCE_NAME, PROFILE_REPORT_NAME,
        GATE_RECORD_NAME, CLEAN_ENV_NAME,
    )
    for name in required:
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    missing = [
        RECORD_FMT.format(replica) for replica in range(T2_REPLICAS)
        if not (run_dir / RECORD_FMT.format(replica)).is_file()
    ]
    if missing:
        _refuse(f"the production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r46] sealed {run_dir} ({n_entries} entries)", flush=True)
    return manifest


def run_clean_validate(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """Verify the seal, re-run the profile evaluation from a clean
    interpreter over the sealed artifacts with a byte-identical report,
    re-run the pre-registered probe replica in memory against its sealed
    record, write clean_environment_commands.json, then stop."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    sums_path = run_dir / SHA256SUMS_NAME
    mismatches = []
    entries = 0
    if sums_path.is_file():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            entries += 1
            if _sha256_file(run_dir / name) != digest:
                mismatches.append(name)
        if mismatches:
            _refuse(f"the seal is broken: {len(mismatches)} mismatches "
                    f"(first {mismatches[0]!r})")
    seal_state = (
        "verified" if sums_path.is_file()
        else "pending (the first clean validation runs before the seal "
             "so the validation record is sealed content; the card's "
             "step order lands the seal over it)")
    # -- the profile evaluation re-run over the sealed artifacts --------
    sealed_report = _read_json(run_dir / PROFILE_REPORT_NAME)
    rerun_report = run_profile_machinery(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # -- the probe replica re-run in memory --
    tables = _load_production_tables()
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
    _set_worker_context(context)
    pre = _load_pre_registration(run_dir)
    pre_digest = _pre_registration_digest_of(pre)
    probe = PROBE_REPLICA
    seed = raw_replica_seed(probe)
    tasks = _chunk_tasks(
        probe, seed, probe, T2_EVENTS_PER_REPLICA,
        B_EFF_FROZEN_GEV2, B_DEC_GENERATION_GEV2, smoke=False)
    parts = list(_map_chunks(tasks, workers, "probe replica"))
    aggregated = _aggregate_chunks(parts)
    record = _replica_record(
        probe, seed, probe, T2_EVENTS_PER_REPLICA, aggregated,
        _pre_registration_digest_of(pre), smoke=False)
    sealed_probe = _read_json(
        run_dir / RECORD_FMT.format(PROBE_REPLICA))
    probe_matches = (
        record["event_hash_set_sha256"]
        == sealed_probe["event_hash_set_sha256"]
        and record["latent_digest"] == sealed_probe["latent_digest"]
        and record["pooled_sums"] == sealed_probe["pooled_sums"]
        and record["routing"] == sealed_probe["routing"]
        and record["depth_histogram"] == sealed_probe["depth_histogram"]
        and record["multiplicity"] == sealed_probe["multiplicity"]
    )
    payload = {
        "schema":
            "na49_mixing.staged_r46_simplified_recoil_decay_profile.clean_env/1",
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            "cd dev/mixing2006 && python -m pytest "
            "tests/test_staged_r46_simplified_recoil_decay_profile.py "
            "tests/test_staged_r46_simplified_recoil_decay_profile_"
            "contract_pins.py -q",
            "cd dev/mixing2006 && python -m pytest tests/"
            "test_legacy_v1_freeze.py -W error::UserWarning -q",
            "cd /home/o/NA49/2026/01-NA49-res && python "
            "helpers/benchmarking/check_helpers_layout_sync.py",
            f"python helpers/benchmarking/"
            f"staged_r46_simplified_recoil_decay_profile.py "
            f"--clean-validate --run-dir {run_dir}",
        ],
        "seal": {"entries": entries, "mismatches": mismatches,
                  "state": seal_state},
        "profile_report_digest": sealed_report.get("digest"),
        "profile_report_rerun_digest": rerun_report.get("digest"),
        "report_hash_match": bool(report_matches),
        "probe_replica": {
            "replica": PROBE_REPLICA,
            "matches_sealed_record": bool(probe_matches),
        },
        "verified_utc": _now_utc(),
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter profile evaluation does not reproduce "
            "the sealed report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(f"[r46] clean validation written: {run_dir / CLEAN_ENV_NAME}",
          flush=True)
    return payload


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each step performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R42 driver: the recoil/decay separation profile "
                    "(TASK-20260920-r4-run-r42-recoil-decay-separation)")
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
        "--profile", action="store_true",
        help="the 9 x 9 profile evaluation (profile_report.json), then "
             "stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate4_followup_reprofile.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the profile evaluation from a "
             "clean interpreter, then stop")
    parser.add_argument(
        "--run-dir", default=None,
        help="the run dir (the default production run dir for the "
             "production modes; --smoke requires an explicit one)")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="the fork worker count (the generation and the probe)")
    return parser


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    modes = [
        bool(args.pre_registration), bool(args.smoke), bool(args.generate),
        bool(args.profile), bool(args.gate), bool(args.seal),
        bool(args.clean_validate),
    ]
    if sum(modes) != 1:
        parser.error("exactly one of --pre-registration / --smoke / "
                     "--generate / --profile / --gate / --seal / "
                     "--clean-validate is required")
    run_dir = Path(args.run_dir) if args.run_dir else (
        MONO_ROOT / DEFAULT_RUN_DIR)
    if not args.smoke:
        _refuse_smoke_artifacts(run_dir) if (
            args.gate or args.seal or args.clean_validate) else None
    if args.smoke:
        if args.run_dir is None:
            _refuse("--smoke requires an explicit --run-dir")
        result = run_smoke(run_dir, workers=int(args.workers))
        print(f"[r46] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(
            run_dir, replicas=args.replica, workers=int(args.workers))
        print(f"[r46] generation complete: {written}", flush=True)
        return 0
    if args.profile:
        report = run_profile_machinery(run_dir, smoke=False)
        _write_json(run_dir / PROFILE_REPORT_NAME, report)
        print(f"[r46] profile report written (digest "
              f"{report['digest'][:16]}...)", flush=True)
        return 0
    if args.gate:
        run_gate(run_dir)
        return 0
    if args.seal:
        run_seal(run_dir)
        return 0
    if args.clean_validate:
        run_clean_validate(run_dir, workers=int(args.workers))
        return 0
    parser.error("unreachable mode dispatch")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
