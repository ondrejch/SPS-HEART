#!/usr/bin/env python3
"""R41 driver: the nonstrange pilot fit (Round 4, card 12).

Task ``TASK-20260920-r4-run-r41-nonstrange-pilot-fit`` (plan section 10
run R41, line 768; the Round 4 card set, card 12; plan section 23, lines
1821-1823).  R41 is the round's only fit (plan section 24 applies to the
RUN cards' acceptance checks, not to this fit card): the nonstrange
pilot fit of the aligned resonance cascade, on the four-parameter fit
set of plan lines 770-775 -- kappa_B, kappa_M, b_dec (the section 10.5
hadronic decay form-factor slope, the "one hadronic decay form-factor
slope"), and the baryonic continuum family's ``nominal_width`` (the "at
most one continuum multiplicity/spectral parameter" of plan line 775) --
against the training observables of plan lines 779-782 (the pi_Sigma
spectrum, B_transport, the pooled pion and proton seagulls, and the
pooled multiplicity moments), with the exchange topology and the flavor
parameters fixed (plan line 784).  The fit constrains the alignment
parameters (Gate 4 unconstrained clause, plan section 10 line 806) and
freezes them for the R44 held-out validation (card 15).

Collision (round 4 index item 7)
--------------------------------
The R32 p+p configuration: p+p at sqrt(s) = 17.270 GeV (beam 158 GeV/c),
the two-source (B, Q, S) = (1, 1, 0) topology of the M8-frozen
ten-parameter set (hash ``358989d8...``) applied unchanged, through the
``ResonancePPBuilder`` with the card-6 routing at the nominal ramp start
B = 2.680194110727528 GeV (the v0 variant, the ramp width 0.5 GeV) and
the card-9 feed-down lineage convention.  "Keep exchange topology and
flavor parameters fixed" (plan line 784): the exchange layer is the
byte-unchanged builder source layer, W_flavor = 1.0 (card 4; Round 6
owns the flavor weights), and the blend ramp start stays fixed at B
(obligation 4 bars it from the fit set).

Event assembly (the R41 continuation of the R40 routing)
--------------------------------------------------------
Per event the frozen builder's source layer is reproduced bit for bit
(the M8 mass sampler, the exact two-body scattering, the legacy retry
discipline on the shared per-event generator); the per-source kernel
switch follows the card-6 routing at the frozen source mass:

- the catalog route (M < B, and the in-band draws that land there):
  the R32 builder's own catalog branch -- the mixture record draw on
  the spawned ``decay_channel`` child, the stable-identity semantics,
  and the byte-unchanged Round 2 kernel decay on the labeled
  substreams (the card-9 feed-down lineage convention);
- the continuum route: the single activated continuum family of the
  source (B, Q, S) = (1, 1, 0) realized at the frozen source
  four-momentum (no family-mass redraw), decayed through the card-10
  aligned cascade at the trial parameters (kappa_B, kappa_M, b_dec)
  with the baryonic family's ``nominal_width`` as the trial width: the
  card-9 routing order (the depth guard, the handoff, the continuation,
  the direct termination) with card 10's attached per-vertex step (the
  card-8 W_dec attachment at the trial b_dec, the card-6/7 cos(theta)
  dispatch at the trial kappas).  This is the R40 continuum-route
  convention transplanted to p+p (the plan section 10.2 line 678 "each
  high-mass parent" clause): the aligned cascade is the object R41
  fits, and the continuum route is where its parameters act.

The production axis of each source is card 5's ``initial_source_axis``
of the source four-momentum against the source's own beam direction
(the projectile +z, the target -z; the M8-frozen two-source
longitudinal axis of the round 4 index item 4 pre-registration).  The
per-vertex axis inheritance, the walk, and the caller-side assembly
follow the R40 driver's card-9/card-10 conventions byte for byte (the
private card-10 attachments, the kernel delegation on the same bound
streams, the frozen 1 channel + 2 kinematics per-vertex budget).

Recording convention (the relitigation clause, Round 4 card 16)
---------------------------------------------------------------
The bin finals follow the established R01 accumulate-dense-then-pick
convention (``_bin_track_sums``; the R44 driver's ``_worker_context`` /
``_replica_record`` pattern is the corrected reference): every event
weight accumulates into the species' dense pT-major evaluation lattice
(``n_x * n_p`` cells, the dense offset per species), and the published
rows are picked from the dense totals at record assembly with
``pick = bin_index_pt * n_x + bin_index_xf``.  The published R01 tables
are non-rectangular subsets of their lattices (holes; rows not
pT-major), so a dense cell address and a published row address are
different addresses: the pick map keeps the recorded ``pooled_sums``
aligned 1:1 with the published rows the evaluation consumes, and a hole
cell's weight stays unpicked and is dropped, never bled into a
neighboring published row or species.  The superseded card-12 landing
recorded the finals at the published-row flat index
(``species_offset(species) + ip * n_x + ix`` with the published-row
cumulative offsets), which cell-transposed the spectra on this
geometry (the sealed ``r41_nonstrange_pilot_fit_2026-09-22`` archive
was generated and evaluated under that superseded convention; it stays
intact, its seal verifies, and it is superseded in the docs by the
relitigation run).  A published bin_index pair mapping outside the
dense lattice, or two published rows on one dense lattice cell, is
refused (the pick map must stay one-to-one).

Fit set (plan lines 770-775; round 4 index item 5)
--------------------------------------------------
Exactly four parameters: kappa_B, kappa_M, b_dec (GeV^-2), and the
baryonic family's ``nominal_width`` (GeV).  The blend ramp start is
EXCLUDED by obligation 4 (decision card lines 305-310) and stays fixed
at B.  Initial values (from the sealed R40): the nominal point
(kappa_B, kappa_M, b_dec) = (1, 1, 1.0 GeV^-2) and nominal_width =
0.5 GeV; the R40 per-setting scan tables provide the gradient
information (the house pattern of R20 before R21/R22 and R31 before
R32).  The width is fitted on the pre-registered five-point grid
{0.3, 0.4, 0.5, 0.6, 0.7} GeV (bracketing the initial 0.5 GeV, the
audit's width-probe range): a width change alters the card-4
continuum-daughter mass rule, hence the realized parent masses of the
cascade continuation -- plan section 5.3 line 288 requires
regeneration, and each grid point's prediction is regenerated (the
documented support boundary).  kappa_B, kappa_M, and b_dec alter
probability densities but not kinematic support (the pools, the channel
sets, and the realized masses are unchanged; the alignment kernels
replace only the cos(theta) quantile and the form factor only weights
the channels), so they are evaluated by exact per-vertex path
reweighting of the generated events (plan section 5.3 line 288): the
per-event weight is the product over the event's continuum-route
vertices of the channel-choice ratio (the card-8 W_dec shares) and the
angular density ratio (the card-6/7 kernels), with the normalization
ratios of both laws.

Observables (plan lines 779-782; plan section 7.3, lines 420-431)
----------------------------------------------------------------
The pooled combinations are the training observables (R44's held-out
set is the species-separated projections, index item 11):

- the pi_Sigma = pi+ + pi- spectrum on the aligned (xF, pT) cells of
  the sealed R01 tables (the R11 aligned-cells convention; d and
  V_data from the sealed R01 archive, the published binning
  preserved);
- the B_transport = p - pbar spectrum on the aligned proton/antiproton
  cells;
- the pooled pion seagull (the pi_Sigma combination's per-xF <pT>_x and
  <pT>_y at the pion mass) and the pooled proton seagull (the
  p + pbar combination at the proton mass), the house
  ``fit_observables.seagull_moment`` conventions on both sides;
- the pooled multiplicity moments: the paper-window first moments
  <n> of pi_Sigma and of p + pbar (the plan section 7.3 multiplicity
  observables at the pipeline's published data surface; the full
  multiplicity distribution is not published in the pipeline tables
  and the species-separated moments are R44 held-out territory).  The
  MC-side <n^2> of the pooled combinations is recorded per replica as
  a non-binding diagnostic.

Likelihood (plan section 5.4)
-----------------------------
-2lnL(theta) = (d - m)' V^-1 (d - m) + eta' P^-1 eta, minimized exactly
over the pre-registered nuisance vector at every trial (the profiled
generalized-least-squares form): one multiplicative normalization
nuisance per training block (four blocks: pi_Sigma, B_transport, the
seagulls, the multiplicities; prior width 0.35 -- the pilot stage has
no calibrated normalization and the R01/R32 record shows the standing
model-to-data mismatch is of order one), plus the obligation 4
model-uncertainty carry as one correlated multiplicative nuisance on
the binned predictions of the blend-band-sensitive bins -- the sealed
R32 persisting-residual bins (32 bins, re-listed in the sealed card-1
audit's attribution; the 14 of them that intersect the fit set's
species carry the term in the objective, all 32 are recorded in the
evidence), prior width 0.25 (the sealed p50 relative cross-variant
spread of the 32 persisting bins, rounded up conservatively).  V =
V_data + V_mc with the published binning preserved and the data and MC
contributions reported separately; V_mc is the between-replica
standard error of the trial statistic (std ddof=1 over the 8 replica
blocks, divided by sqrt(8)), floored at the engine's deterministic SE
floor; a statistic whose MC error is zero-by-structure (an empty
model bin) carries an infinite sigma and drops out of the objective
(the R11 convention, recorded, never dropped silently -- the dropped
bins are listed).  The nuisance profile is part of the objective
definition (plan section 5.4), not part of the fit set: the optimizer's
parameter vector is exactly the four fit parameters.

Hessian discipline (plan section 5.4, line 311)
-----------------------------------------------
No Hessian errors are quoted until the replica-block minima agree
(the acceptance (ii) stability table across all 8 blocks, at least
four required by plan section 5.5 item 5 line 321) and the local
curvature is stable (the pooled numerical Hessian positive definite on
the reweightable directions and the width-profile curvature positive);
otherwise the Hessian block records ``quoted: false`` and the failure
is named.

Statistics (T2, plan section 5.1 line 254): 8 replicas x 100,000 p+p
events = 800,000 events (the upper T2 bound, the R23/R32 convention).
Local generation is primary (the measured local envelope: 100,000
events ~ 11 min single-thread, ~ 2 min at 8 fork workers); the
pre-registered gateway fallback covers the envelope (one job per
replica, ``--tasks 1``, single math thread, capacity-paced; record
outputs appended only to the new run dir).  The fit's width-grid
trials regenerate the walk layer at the four off-nominal grid widths
on the production (seed, event_id) pairs (common random numbers,
plan section 5.2 line 275; the regeneration boundary documented per
plan section 5.3 line 288), in memory, never sealed.

Conservation: every p+p event conserves total (B, Q, S) = (2, 2, 0)
exactly and every source its own (B, Q, S); the event energy sum
closes within 1e-9 GeV; a violation raises and stops the run.

Determinism: a per-replica re-run from a clean interpreter reproduces
the record identity digests; the fit evaluation re-run from a clean
interpreter over the sealed artifacts reproduces the report hashes
(fit_report.json and gate4_r41.json carry no wall-clock fields: their
bytes are a pure function of the sealed artifacts, the pre-registered
seeds, and the committed code).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests and the sealed
  R40 / R30-R32 seals first.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, 1 replica x 2,000 events at the
  pre-registered smoke seeds, and the smoke fit, into an explicit
  ``--run-dir`` (the default production run dir is refused).  Two
  identical smoke invocations produce byte-identical artifacts.
  ``--gate`` / ``--seal`` / ``--clean-validate`` refuse smoke
  artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested replica
  records (any subset; the default is every replica), one record file
  per replica, then stop.  Requires the pre-registration; the
  statistics flags must match the pre-registration in production mode.
  A completed record that matches the pre-registered identity is
  reused unchanged, so the declared tier can be completed replica by
  replica (the pre-registered gateway fallback runs one job per
  replica).
- ``--fit``: the fit evaluation -- the per-replica-block likelihoods,
  the pooled fit over the width grid, the per-block fits, the
  replica-block stability table, the Hessian (only after the replica
  minima agree), and ``fit_report.json``, then stop.  Requires all 8
  production records.  The four width-grid trial generations run
  in memory from the pre-registered seeds (the documented regeneration
  boundary); no trial artifacts are written.
- ``--gate``: write ``gate4_r41.json`` (evidence only), then stop.
  Requires the fit report.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the fit evaluation
  from a clean interpreter over the sealed artifacts with a
  byte-identical report, re-run production replica 0 in memory against
  its sealed record, write ``clean_environment_commands.json``, then
  stop.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic in ``--smoke`` (the committed catalog, the committed continuum
config, and the synthetic grid only; no SPS table, no sealed archive
needed for the smoke fit's synthetic data); the production modes read
the committed catalog and continuum config, the sealed archives, and
the SPS tables (the published data the likelihood is defined against).
"""

from __future__ import annotations

import argparse
import base64
import binascii
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
        f"staged_r41_nonstrange_pilot_fit: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_observable_decomposition as r11  # noqa: E402
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
from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL  # noqa: E402
from na49_mixing.event_builder_pp import PPExcitationConfig  # noqa: E402
from na49_mixing.exchange_model import transfer_slope  # noqa: E402
from na49_mixing.excitation_kinematics import (  # noqa: E402
    FourVector,
    scatter_two_body,
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
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402
from na49_mixing import fit_observables as fo  # noqa: E402

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

#: The superseded R41 driver digest (the card-12 landing; the value of
#: the card-12 contract register's ``DRIVER_SHA256`` constant at the
#: relitigation landing, sha256 prefix ``573d833b``).  The sealed
#: ``r41_nonstrange_pilot_fit_2026-09-22`` archive was generated and
#: evaluated under that driver's published-row flat index; it stays
#: intact, its seal verifies, and it is superseded by this run.
SUPERSEDED_DRIVER_SHA256 = (
    "573d833bfbc659885fb83e6c0794207947fbe9e5a6dafe6159650bd45097172b"
)
#: The superseded sealed R41 archive (read-only; its seal verifies and
#: is recorded in the provenance).
SUPERSEDED_RUN_DIR = "dev/00runs/r41_nonstrange_pilot_fit_2026-09-22"


# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260920-r4-run-r41-nonstrange-
# pilot-fit; the fit set, statistics, seeds, likelihood, and observable
# blocks are verbatim from the card and the round 4 index items 5, 7, 9,
# and 11).
# ---------------------------------------------------------------------------

RUN_ID = "R41"
CARD = "TASK-20260920-r4-run-r41-nonstrange-pilot-fit"
RUN_NAME = "r41_nonstrange_pilot_fit_2026-09-23"
DEFAULT_RUN_DIR = "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"
GATE_RECORD_NAME = "gate4_r41.json"
FIT_REPORT_NAME = "fit_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
ROUTING_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
KERNEL_REL = "dev/mixing2006/na49_mixing/resonance_decay.py"
R32_DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"
R40_DRIVER_REL = "helpers/benchmarking/staged_r40_alignment_scan.py"

#: The card chain this run sits on: the Gate 3 closure audits (cards
#: 1-3, sealed; card 1's RECOMMENDATION spawned the rework card 1r,
#: recorded in the obligation block at its execution-time status), the
#: cascade implementation cards (4-10, landed and byte-stable), and the
#: sealed R40 scan (card 11, the initial-value source).
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
)

#: The frozen substrate digests (byte-unchanged inputs; verified before
#: any step).  The cascade surfaces are cards 4-10 (landed this round,
#: consumed read-only); the builder blend path, the card-6 routing, the
#: Round 2 kernel, the engine, the catalog, the continuum config, the
#: R32 driver, and the R40 driver close the list.
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
}

#: The sealed archives consumed read-only (their seals verify).  The R40
#: archive is the initial-value source; the R32 spread is the
#: model-uncertainty carry of obligation 4; the sealed R01 archive is
#: the published-data surface of the likelihood (d, V_data, the binning).
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21",
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
)
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R40_RUN_DIR = "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21"
R4_AUDIT_CARD1_DIR = "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20"

#: The R32 p+p configuration constants (index item 7).
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.2701885890706  # Reaction(158, 2212, 2212).sqrt_s()
BLEND_RAMP_START_GEV = 2.680194110727528
BLEND_RAMP_WIDTH_GEV = 0.5
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)

#: The four-parameter fit set (plan lines 770-775; index item 5).  The
#: blend ramp start is excluded by obligation 4 and stays fixed at B.
FIT_SET: tuple[str, ...] = (
    "kappa_B", "kappa_m", "b_dec", "baryonic_nominal_width",
)
FIT_SET_EXCLUDED = {
    "blend_ramp_start": "obligation 4 (fixed at B = "
    "2.680194110727528 GeV; decision card lines 305-310)",
}
INITIAL_VALUES: dict[str, float] = {
    "kappa_B": 1.0,
    "kappa_m": 1.0,
    "b_dec": 1.0,
    "baryonic_nominal_width": 0.5,
}
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "kappa_B": (0.0, 4.0),
    "kappa_m": (0.0, 4.0),
    "b_dec": (0.0, 2.0),
    "baryonic_nominal_width": (0.3, 0.7),
}
BOUNDS_NOTE = (
    "kappa_B and kappa_M within the R40 scan range [0, 4] (0 the "
    "isotropic kernel, 4 the scan cap); b_dec within the R40 scan "
    "bracket [0, 2.0] GeV^-2 (non-negative: a negative slope is an "
    "unphysical vertex form factor); the width on the pre-registered "
    "grid [0.3, 0.7] GeV.  An optimum on a bound is a named finding "
    "(the fit range does not extend)."
)

#: The pre-registered width grid (the regeneration boundary; plan
#: section 5.3 line 288): five points bracketing the initial 0.5 GeV,
#: the audit's width-probe range.  The nominal width point (0.5) is the
#: production generation; the four off-nominal points regenerate the
#: walk layer in memory on the production (seed, event_id) pairs.
W_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
W_GRID_NOMINAL = 0.5
W_GRID_INDEX_NOMINAL = 2
W_GRID_NOTE = (
    "the baryonic nominal_width is fitted on the pre-registered "
    "five-point grid {0.3, 0.4, 0.5, 0.6, 0.7} GeV: a width change "
    "alters the card-4 continuum-daughter mass rule (the realized "
    "parent masses of the cascade continuation), so plan section 5.3 "
    "line 288 requires regeneration and each grid point's prediction "
    "is regenerated (the documented support boundary); the nominal "
    "point is the production generation, the four off-nominal points "
    "run in memory on the production (seed, event_id) pairs (common "
    "random numbers, plan section 5.2 line 275).  The width resolution "
    "of the pilot fit is the grid spacing 0.1 GeV; no continuous width "
    "optimization is claimed (the optimizer's width coordinate is the "
    "pre-registered lattice argmin)"
)

#: The reweightable directions (plan section 5.3 line 288): kappa_B,
#: kappa_M, and b_dec alter probability densities but not kinematic
#: support (the pools, the channel sets, and the realized masses are
#: unchanged; the alignment kernels replace only the cos(theta)
#: quantile and the form factor only weights the channels).
REWEIGHT_PARAMS = ("kappa_B", "kappa_m", "b_dec")

#: The optimizer contract (pre-registered; scipy Nelder-Mead, bounded,
#: deterministic).  One fit procedure per objective (the pooled fit and
#: the 8 per-block fits); optimizer_uses = 1 for the card.
NM_XATOL = 1e-3
NM_FATOL = 1e-6
NM_MAXITER = 400
NM_MAXFEV = 800
NM_INITIAL_SIMPLEX_NOTE = (
    "the scipy Nelder-Mead default initial simplex at the pre-registered "
    "start point (the deterministic 5% class of the nonzero coordinates)"
)
HESSIAN_STEPS: dict[str, float] = {
    "kappa_B": 0.1,
    "kappa_m": 0.1,
    "b_dec": 0.1,
}
HESSIAN_W_FROM_GRID = (
    "the width curvature comes from the five-point grid profile (the "
    "second difference at the argmin over its neighbours); no off-grid "
    "width evaluation exists (the regeneration boundary)"
)

#: The pre-registered nuisance structure (plan section 5.4): one
#: multiplicative normalization nuisance per training block, plus the
#: obligation 4 carry as one correlated multiplicative nuisance on the
#: blend-band-sensitive bins.
NUISSANCE_BLOCK_KEYS = (
    "pi_sigma_spectrum", "b_transport_spectrum", "seagulls",
    "multiplicity",
)
NORM_NUISANCE_PRIOR = 0.35
NORM_NUISANCE_NOTE = (
    "one multiplicative normalization nuisance per training block "
    "(m -> m (1 + eta_g)) with prior width 0.35: the pilot stage has no "
    "calibrated model-to-data normalization (the sealed R01/R32 record "
    "shows the standing mismatch is of order one) and the profiled "
    "nuisance absorbs the level while the shape residuals constrain the "
    "fit set; the nuisance profile is part of the objective definition "
    "(plan section 5.4), not part of the four-parameter fit set"
)
CARRY_NUISANCE_PRIOR = 0.25
CARRY_PRIOR_NOTE = (
    "the obligation 4 carry is one correlated multiplicative nuisance "
    "(m -> m (1 + eta_carry) on the blend-band-sensitive bins) with "
    "prior width 0.25: the sealed p50 relative cross-variant spread of "
    "the 32 persisting bins is 0.2351 (computed from the sealed card-1 "
    "attribution rows), rounded up conservatively; the term is never a "
    "free parameter of the fit set"
)
ESS_FLOOR_FRACTION = 0.01
ESS_NOTE = (
    "the effective sample size of the reweighted ensemble, "
    "(sum w)^2 / sum w^2 over the pooled events, is recorded at the "
    "pooled optimum; an ESS fraction below 0.01 is a named "
    "weight-degeneracy finding, never a refit"
)

#: The obligation 4 model-uncertainty carry (the sealed R32
#: cross-variant spread), verbatim from the card; the blend-band-
#: sensitive bins are the sealed R32 persisting-residual bins (the card-1
#: audit's attribution; the audit outcome is the card's "final form
#: determined by the closed audits" input: cards 2 and 3 passed with no
#: new spread diagnostic and card 1's rework card 1r had not landed a new
#: spread at run time, so the original R32 spread stands).
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}
OBLIGATION4_FORM = (
    "one correlated multiplicative nuisance on the blend-band-sensitive "
    "bins (the sealed R32 persisting-residual bins; prior 0.25), "
    "profiled exactly with the block normalization nuisances per trial "
    "(plan section 5.4); the audits closed with card 2 and card 3 PASS "
    "and card 1 RECOMMENDATION (its rework card 1r open at run time), "
    "so the original R32 spread stands"
)
CARRY_SPECIES = ("pi_plus", "pi_minus", "proton")

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
STABILITY_SPREAD_BOUND = 0.20
STABILITY_BOUND_NOTE = (
    "the per-block spread (max - min over the 8 independent replica "
    "blocks) of each fitted parameter against 20% of the pooled error "
    "(the pre-registered bound of acceptance (ii)); a failing "
    "parameter is a named finding (the Gate 4 unconstrained clause "
    "consumes it directly)"
)

#: The per-vertex draw budget of the card-4 cascade (1 channel + 2
#: kinematics uniforms), frozen through cards 6-8; the routing consumes
#: one uniform inside the blend band and zero off-band (the card-6
#: surface); the catalog route's record draw consumes one uniform on
#: the same spawned child (the R32 builder branch).
EVENT_CLOSURE_TOL = 1e-9
DEGENERATE_WINDOW_TOL = 1e-12

#: Generation chunking (the R01/R23/R32 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 8

#: The event-seed stage label of the R41 stream.
STAGE_LABEL = "r41_nonstrange_pilot_fit"



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
    entries = 0
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
        entries_ok = _sha256_file(payload) == digest
        if not entries_ok:
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
SEED_STRIDE = 100
T2_REPLICAS = 8
T2_EVENTS_PER_REPLICA = 100_000
RAW_SEED_RULE = (
    "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r per "
    "replica r < 8 (the block 20301020-20301027); the block is disjoint "
    "from every previously used raw block (R23 20300913-20300920; the "
    "R32 production and confirmation blocks; the audit confirmation "
    "block 20301040 + k; the R32 smoke block 20305919 + 100*v; the R40 "
    "raw block 20301020-20301023 shares the block but not the "
    "event-index partition) and from the audit block 20301040 + k"
)
KERNEL_SEED_RULE = (
    "derive_substreams(20260920, r, 400_000_000 + i) per (replica "
    "r < 8, event i < 100,000) on the closed decay_channel/"
    "decay_kinematics labels (the round 2 card 9 and round 3 card 8 "
    "convention: the master seed with the replica folded into the "
    "replica index, the event index encoding the run); the 8 (replica) "
    "kernel-layer spaces are pairwise disjoint and disjoint from the "
    "R40 spaces (the R40 event indices are 300_000_000 + s*10_000_000 "
    "+ p*1_000_000 + i)"
)
EVENT_INDEX_RULE = "400_000_000 + i, i < 100,000"
EVENT_INDEX_BASE = 400_000_000
RAW_REPLICA_SEEDS = tuple(SEED_BASE + SEED_OFFSET + SEED_STRIDE + r
                          for r in range(T2_REPLICAS))

#: The hermetic smoke mode (driver validation; never sealed): 1 replica
#: x 2,000 events at the pre-registered smoke seeds.  The +8000 block
#: (raw smoke seed 20308920, kernel replica index 8000) is disjoint from
#: every production space (replica indices 0..7), from the R40 +7000
#: smoke block, from the R32 +5000 smoke block, from the R32
#: confirmation blocks, and from every previously used block.
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 8000
SMOKE_RAW_SEED = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260920 + 40000 + 8000 = 20308920 (the +8000 "
    "smoke block, one replica; the R40 smoke block is +7000); the smoke "
    "kernel layer draws at derive_substreams(20260920, 8000, "
    "event_index) on the production event-index partition (400_000_000 "
    "+ i, i < 2,000), disjoint from every production space (replica "
    "indices 0..7) and from every previously used block"
)
SMOKE_W_GRID: tuple[float, ...] = (0.5,)
SMOKE_SYNTHETIC_TRUTH: dict[str, float] = {
    "kappa_B": 1.3,
    "kappa_m": 0.8,
    "b_dec": 1.2,
    "baryonic_nominal_width": 0.5,
}
SMOKE_NOTE = (
    "the smoke fit evaluates the objective machinery against a "
    "synthetic data vector: the smoke replica's own prediction at the "
    "pre-registered synthetic-truth point (kappa_B, kappa_m, b_dec, "
    "width) = (1.3, 0.8, 1.2, 0.5), so the smoke fit exercises "
    "pre-registration, generation, reweighting, likelihood, and the "
    "optimizer end to end, hermetically (the smoke data is the "
    "synthetic truth, no SPS table, no sealed archive)"
)

#: The clean-environment validation's pre-registered probe replica.
PROBE_REPLICA = 0


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
    return SMOKE_RAW_SEED


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
# The event assembly (the R41 p+p configuration)
# ---------------------------------------------------------------------------
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


class _LatentCapture:
    """The per-event latent arrays of one generated event.

    ``rows`` holds the event's sparse (dense lattice cell, Jacobian
    weight) contributions of every final particle (the R01 binning
    convention: ``x_F = 2 p_z / sqrt(s)``, the pT-major flat layout, the
    same in-bounds drop; the recorded address is the final's DENSE
    lattice cell -- the species' dense offset plus ``ip * n_x + ix`` --
    and the published rows are picked from the dense totals at record
    assembly, the accumulate-dense-then-pick convention, so a hole
    cell's weight stays unpicked and is dropped); ``vertices`` holds one
    latent tuple per
    continuum-route cascade vertex: ``(chosen, law, c_k, w0, q2)`` with
    the attached pool's normalized card-8 shares ``w0`` and the
    reference transverse momenta squared ``q2`` (the exact data the
    kappa_B / kappa_M / b_dec path reweighting consumes), ``law`` the
    governing kernel (0 baryonic, 1 mesonic), and ``c_k`` the signed
    kernel cosine (the baryonic-trailing mirror applied).  Events with
    no continuum-routed source carry an empty vertex list and their row
    is pooled into the theta-independent fixed sums.
    """

    __slots__ = (
        "event_id", "flat", "weights", "vtx_nchan", "vtx_chosen",
        "vtx_law", "vtx_ck", "ch_vtx", "ch_w0", "ch_q2", "n_vertices",
        "is_var", "route_census", "depth_hist", "bqs", "energy_sum",
        "energy_expected", "worst_closure", "event_hash", "n_finals",
        "mult_sums", "mult_sums2", "mult_with", "payload",
    )

    def __init__(self) -> None:
        self.flat: list[int] = []
        self.weights: list[float] = []
        self.vtx_nchan: list[int] = []
        self.vtx_chosen: list[int] = []
        self.vtx_law: list[int] = []
        self.vtx_ck: list[float] = []
        self.ch_vtx: list[int] = []
        self.ch_w0: list[float] = []
        self.ch_q2: list[float] = []
        self.n_vertices = 0
        self.is_var = False
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
        row (the R01 binning convention: the same grid edges, the same
        ``sps_core.grid_indices`` binning, the pT-major flat layout, the
        same in-bounds drop).  The recorded address is the final's DENSE
        lattice cell (the species' dense offset plus ``ip * n_x + ix``);
        the published rows are picked from the dense totals at record
        assembly (the R01 ``_bin_track_sums`` accumulate-dense-then-pick
        convention), so a hole cell's weight stays unpicked and is
        dropped, never bled into a neighboring published row or
        species."""
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
        self.flat.append(
            context["dense_offsets_by_species"][species] + ip * n_x + ix)
        self.weights.append(e / pt)


def _reference_pool_payload(
    attached: list,
) -> list[float]:
    """The attached pool's normalized shares and reference q_perp^2.

    ``attached[i].weight`` is card 4's base weight scaled by the card-8
    W_dec factor at the reference two-body configuration; the recorded
    ``w0`` is the normalized share the vertex walk draws against at the
    generation-time b_dec, and ``q2`` the reference transverse momentum
    squared per channel (the exact reweighting sufficient statistics).
    """

    total = math.fsum(float(channel.weight) for channel in attached)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            "r41 latent capture: the attached pool weights do not sum "
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

    The R40 walk order byte for byte (the depth guard, the catalog
    handoff, the continuation, the direct termination, the no-channel
    refusal) with the attached pool's (w0, q2, chosen, law, c_k)
    recorded per continuum vertex.  The capture consumes no random
    numbers and adds no draw.  Returns ``(n_vertices, mode)``; every
    final particle of the walk lands in the capture with its binning
    row.
    """

    root = exchange_root_state(root_event)
    current = root
    n_vert = 0
    while True:
        if int(current.depth) > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"r41: the line state of family {current.family_id!r} at "
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
                f"r41: the line state of family {current.family_id!r} at "
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
        qn0 = (int(daughter0.qn[0]), int(daughter0.qn[1]),
               int(daughter0.qn[2]))
        qn1 = (int(daughter1.qn[0]), int(daughter1.qn[1]),
               int(daughter1.qn[2]))
        # The leading daughter's rest-frame cosine against the axis,
        # reconstructed from the realized momenta (deterministic; the
        # round trip of the vertex boost).
        energy = float(current.momentum[0])
        beta = (
            float(current.momentum[1]) / energy,
            float(current.momentum[2]) / energy,
            float(current.momentum[3]) / energy,
        )
        from na49_mixing.resonance_decay import _boost_lab
        rest0 = _boost_lab(
            (float(daughter0.momentum[0]), float(daughter0.momentum[1]),
             float(daughter0.momentum[2]), float(daughter0.momentum[3])),
            (-beta[0], -beta[1], -beta[2]),
        )
        px, py, pz = rest0[1], rest0[2], rest0[3]
        norm = math.sqrt(px * px + py * py + pz * pz)
        ax, ay, az = (float(v) for v in current.axis)
        c = (px * ax + py * ay + pz * az) / norm
        if qn0[0] != 0:
            law, c_k = 0, c
        elif qn1[0] != 0:
            law, c_k = 0, -c
        else:
            law, c_k = 1, c
        q2 = [
            _channel_reference_q2(channel, current)
            for channel in attached
        ]
        capture.vtx_nchan.append(len(attached))
        capture.vtx_chosen.append(int(chosen))
        capture.vtx_law.append(int(law))
        capture.vtx_ck.append(float(c_k))
        capture.ch_w0.extend(w0)
        capture.ch_q2.extend(float(x) for x in q2)
        capture.n_vertices += 1
        capture.is_var = True
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
            "r41 assembly: a continuum daughter state reached the event "
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


def _load_production_tables():
    """The six-species SPS tables (the published binning; the R01
    loader).  The production generation and the fit read the SPS
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


def _worker_context(tables: list, fams: list, smoke: bool) -> dict:
    catalog = _load_catalog()
    for table in tables:
        x_edges = _unique_edges(sps_core.edges_from_centers(
            table.xf_centers))
        p_edges = _unique_edges(sps_core.edges_from_centers(
            table.pt_centers))
        edges_by_species[table.species] = (x_edges, p_edges)
        n_x_by_species[table.species] = int(table.bin_index_xf.max()) + 1
        n_p_by_species[table.species] = int(table.bin_index_pt.max()) + 1
    # -- the R01 accumulate-dense-then-pick convention (the
    #    ``_bin_track_sums`` convention of the sealed R01/R23/R32
    #    pipeline, mirrored from the corrected R44 driver's
    #    ``_worker_context``): the finals accumulate into the species'
    #    dense pT-major lattice (n_p * n_x cells) and the published rows
    #    are then picked from the dense totals at the published rows'
    #    cells, pick = bin_index_pt * n_x + bin_index_xf.  The published
    #    R01 tables are non-rectangular subsets of their lattices
    #    (holes; rows not pT-major), so a dense cell address and a
    #    published row address are different addresses; the pick map is
    #    what keeps the recorded pooled_sums aligned 1:1 with the
    #    published rows the evaluation consumes, and a hole cell (a
    #    lattice cell with no published row) stays unpicked -- its
    #    weight is dropped, never bled into a neighboring row or
    #    species. --
    dense_total = _init_dense_layout(tables)
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


edges_by_species: dict[str, tuple] = {}
n_x_by_species: dict[str, int] = {}
n_p_by_species: dict[str, int] = {}


def _unique_edges(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    keep = [float(values[0])]
    for value in values[1:]:
        if float(value) != keep[-1]:
            keep.append(float(value))
    return np.asarray(keep, dtype=float)



# ---------------------------------------------------------------------------
# The per-event generation (the R41 p+p assembly)
# ---------------------------------------------------------------------------
class ConservationViolation(RuntimeError):
    """The pre-registered conservation check of the card fired (checked,
    never repaired; a violation stops the run)."""


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
    kappa_b: float,
    kappa_m: float,
    b_dec: float,
) -> _LatentCapture:
    """One p+p event of the R41 configuration: the frozen builder source
    layer bit for bit (the M8 mass sampler and the exact two-body
    scattering on the shared legacy generator, the retry discipline),
    the card-6 routing at the frozen source mass, the R32 builder
    catalog branch (the card-9 feed-down lineage convention) and the
    card-10 aligned cascade at the trial (kappa_B, kappa_m, b_dec) on
    the labeled substreams.  The conservation checks are the
    pre-registered per-event and per-source checks (checked, never
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
    fam_b = active_families(list(fams), (1, 1, 0))
    if len(fam_b) != 1:
        _refuse(
            "r41: the blend routing expects exactly one activated "
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
                float(kappa_b), float(kappa_m), float(b_dec),
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
                f"r41: event {event_id} source {sid} ({role}) finals sum "
                f"to (B, Q, S) = {delta} but the source carries "
                "(1, 1, 0); conservation is checked, never repaired "
                "(the run stops)"
            )
    # -- the per-event conservation check --
    if capture.bqs != (2, 2, 0):
        raise ConservationViolation(
            f"r41: event {event_id} finals sum to (B, Q, S) = "
            f"{capture.bqs} but the p+p pair carries (2, 2, 0); "
            "conservation is checked, never repaired (the run stops)"
        )
    closure = abs(capture.energy_sum - capture.energy_expected)
    capture.worst_closure = closure
    if not closure <= EVENT_CLOSURE_TOL:
        raise ConservationViolation(
            f"r41: event {event_id} final energy sum misses the source "
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
        f"r41|{event_id}|{route_census[0]},{route_census[1]},"
        f"{route_census[2]}|{capture.n_finals}|".encode("utf-8"))
    for line in capture.payload:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()



# ---------------------------------------------------------------------------
# Latent packing (the sealed record's reweighting payload)
# ---------------------------------------------------------------------------
_PACK_DTYPES = ("f8", "f4", "i8", "i4", "u2", "i1")


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
     blend_boundary, kappa_b, kappa_m, b_dec, width, smoke) = task
    catalog = context["catalog"]
    pp_config = context["pp_config"]
    blend = BlendSpec(boundary=float(blend_boundary),
                      ramp_width=BLEND_RAMP_WIDTH_GEV)
    fams = list(context["families"])
    if float(width) != W_GRID_NOMINAL:
        # The width-grid trials replace the baryonic family's
        # nominal_width (the documented regeneration boundary).
        import dataclasses

        target = active_families(fams, (1, 1, 0))
        if len(target) != 1:
            _refuse("r41 chunk: no unique active baryonic family")
        replaced = dataclasses.replace(
            target[0], nominal_width=float(width))
        fams = [
            replaced if str(f.family_id) == str(target[0].family_id) else f
            for f in fams
        ]
    captures: list[_LatentCapture] = []
    for event_id in event_ids:
        captures.append(generate_one_event(
            catalog, pp_config, blend, fams, fams,
            seed=int(seed), event_id=int(event_id),
            master_seed=int(master_seed),
            replica_slot=int(replica_slot),
            kappa_b=float(kappa_b), kappa_m=float(kappa_m),
            b_dec=float(b_dec),
        ))
    return _chunk_parts(replica, event_ids[0], captures, smoke)


def _chunk_parts(replica, start_id, captures, smoke) -> dict:
    context = _WORKER_CONTEXT
    # the fixed pool accumulates on the dense evaluation lattice (the
    # accumulate-dense-then-pick convention; the dense layout covers
    # every lattice cell, hole cells included)
    total_flat = _dense_total()
    hashes = [c.event_hash for c in captures]
    fixed = np.zeros(total_flat, dtype=float)
    var_flat: list[int] = []
    var_weight: list[float] = []
    var_ids: list[int] = []
    var_ptr: list[int] = [0]
    row_flat: list[int] = []
    row_weight: list[float] = []
    row_ptr: list[int] = [0]
    vtx_nchan: list[int] = []
    vtx_chosen: list[int] = []
    vtx_law: list[int] = []
    vtx_ck: list[float] = []
    ch_w0: list[float] = []
    ch_q2: list[float] = []
    vtx_ptr: list[int] = [0]
    routing = [0, 0, 0]
    depth_hist: dict[int, int] = {}
    mult = {k: [0.0, 0.0, 0] for k in SPECIES_KEYS}
    n_events = len(captures)
    worst = 0.0
    guard = 0
    vtx_cum = 0
    for capture in captures:
        if capture.is_var:
            var_ids.append(int(capture.event_id))
            row_flat.extend(capture.flat)
            row_weight.extend(capture.weights)
            row_ptr.append(len(row_flat))
            vtx_cum += capture.n_vertices
            vtx_ptr.append(vtx_cum)
            vtx_nchan.extend(capture.vtx_nchan)
            vtx_chosen.extend(capture.vtx_chosen)
            vtx_law.extend(capture.vtx_law)
            vtx_ck.extend(capture.vtx_ck)
            ch_w0.extend(capture.ch_w0)
            ch_q2.extend(capture.ch_q2)
        else:
            np.add.at(fixed, np.asarray(capture.flat, dtype=np.int64),
                      np.asarray(capture.weights, dtype=float))
        routing[0] += int(capture.route_census[0])
        routing[1] += int(capture.route_census[1])
        routing[2] += int(capture.route_census[2])
        for depth, count in capture.depth_hist.items():
            depth_hist[int(depth)] = depth_hist.get(int(depth), 0) + int(count)
        worst = max(worst, float(capture.worst_closure))
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
        "fixed": fixed,
        "var_ids": np.asarray(var_ids, dtype=np.int64),
        "row_flat": np.asarray(row_flat, dtype=np.int32),
        "row_weight": np.asarray(row_weight, dtype=np.float64),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "vtx_ptr": np.asarray(vtx_ptr, dtype=np.int64),
        "vtx_nchan": np.asarray(vtx_nchan, dtype=np.int32),
        "vtx_chosen": np.asarray(vtx_chosen, dtype=np.int32),
        "vtx_law": np.asarray(vtx_law, dtype=np.int8),
        "vtx_ck": np.asarray(vtx_ck, dtype=np.float64),
        "ch_w0": np.asarray(ch_w0, dtype=np.float32),
        "ch_q2": np.asarray(ch_q2, dtype=np.float32),
        "routing": routing,
        "depth_hist": {str(k): int(v) for k, v in sorted(depth_hist.items())},
        "mult": mult,
        "worst_closure": worst,
        "smoke": bool(smoke),
        "n_vertices_total": int(sum(c.n_vertices for c in captures)),
    }



# ---------------------------------------------------------------------------
# The replica record assembly
# ---------------------------------------------------------------------------
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any] | None) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _aggregate_chunks(parts: list) -> dict:
    """Merge the chunk parts into one replica's state (exact integer
    counts; float sums in chunk order, the chunks sorted by start id)."""
    parts = sorted(parts, key=lambda p: int(p["start_id"]))
    # the fixed pool accumulates on the dense evaluation lattice (the
    # accumulate-dense-then-pick convention)
    total_flat = _dense_total()
    hashes: list[str] = []
    fixed = np.zeros(total_flat, dtype=float)
    row_flat: list[int] = []
    row_weight: list[float] = []
    row_ptr: list[int] = [0]
    var_ids: list[int] = []
    vtx_ptr: list[int] = [0]
    vtx_nchan: list[int] = []
    vtx_chosen: list[int] = []
    vtx_law: list[int] = []
    vtx_ck: list[float] = []
    ch_w0: list[float] = []
    ch_q2: list[float] = []
    routing = [0, 0, 0]
    depth_hist: dict[str, int] = {}
    mult = {k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
            for k in SPECIES_KEYS}
    worst = 0.0
    n_vertices = 0
    n_events = 0
    for part in parts:
        hashes.extend(part["hashes"])
        fixed += part["fixed"]
        row_flat.extend(part["row_flat"].tolist())
        row_weight.extend(part["row_weight"].tolist())
        base = int(row_ptr[-1])
        row_ptr.extend((base + int(v)) for v in part["row_ptr"][1:])
        var_ids.extend(part["var_ids"].tolist())
        base_v = vtx_ptr[-1]
        vtx_ptr.extend((base_v + int(v)) for v in part["vtx_ptr"][1:])
        vtx_nchan.extend(part["vtx_nchan"].tolist())
        vtx_chosen.extend(part["vtx_chosen"].tolist())
        vtx_law.extend(part["vtx_law"].tolist())
        vtx_ck.extend(part["vtx_ck"].tolist())
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
        n_vertices += int(part["n_vertices_total"])
        n_events += int(part["n_events"])
    return {
        "hashes": hashes,
        "fixed": fixed,
        "row_flat": np.asarray(row_flat, dtype=np.int32),
        "row_weight": np.asarray(row_weight, dtype=np.float64),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "var_ids": np.asarray(var_ids, dtype=np.int64),
        "vtx_ptr": np.asarray(vtx_ptr, dtype=np.int64),
        "vtx_nchan": np.asarray(vtx_nchan, dtype=np.int32),
        "vtx_chosen": np.asarray(vtx_chosen, dtype=np.int32),
        "vtx_law": np.asarray(vtx_law, dtype=np.int8),
        "vtx_ck": np.asarray(vtx_ck, dtype=np.float64),
        "ch_w0": np.asarray(ch_w0, dtype=np.float32),
        "ch_q2": np.asarray(ch_q2, dtype=np.float32),
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
                    f"[r41] {progress_label}: chunk {completed}/"
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
    cell's weight stays unpicked and is dropped.  The theta-independent
    ``fixed_pool`` and the reweighting latent carry the dense lattice
    addresses (the two-step dense-slice-then-pick conversion happens at
    the evaluation boundary, in ``_species_bin_values``)."""
    context = _WORKER_CONTEXT
    assert context is not None
    dense_offsets = context["dense_offsets_by_species"]
    dense_sizes = context["dense_sizes_by_species"]
    pick_by_species = context["pick_by_species"]
    routing = aggregated["routing"]
    pooled = np.asarray(aggregated["fixed"], dtype=float).copy()
    row_flat = np.asarray(aggregated["row_flat"], dtype=np.int64)
    row_weight = np.asarray(aggregated["row_weight"], dtype=float)
    if row_flat.size:
        np.add.at(pooled, row_flat, row_weight)
    pooled_species = {}
    for key in SPECIES_FLAT_ORDER:
        dense_slice = pooled[
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
        aggregated["var_ids"], aggregated["row_flat"],
        aggregated["row_weight"], aggregated["row_ptr"],
        aggregated["vtx_ptr"], aggregated["vtx_nchan"],
        aggregated["vtx_chosen"], aggregated["vtx_law"],
        aggregated["vtx_ck"], aggregated["ch_w0"], aggregated["ch_q2"],
    ):
        latent_digest.update(np.ascontiguousarray(array).tobytes())
        latent_digest.update(b"|")
    record = {
        "schema": "na49_mixing.staged_r41_nonstrange_pilot_fit.replica/1",
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
                "aligned cascade)"
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
        "fixed_pool": [float(v) for v in np.asarray(
            aggregated["fixed"], dtype=float)],
        "latent": {
            "note": (
                "the per-event reweighting latent of the continuum-route "
                "events: the var rows (dense lattice cell + Jacobian "
                "weight per final, the accumulate-dense-then-pick "
                "addresses; the published-row conversion happens at the "
                "evaluation boundary), the per-event CSR pointers, the "
                "per-vertex "
                "attached pool (the card-8 shares w0 at the generation "
                "b_dec, the reference transverse momenta squared q2, the "
                "chosen channel index, the governing kernel law, and the "
                "signed kernel cosine c_k), and the fixed (theta-"
                "independent) row pool; the kappa_B / kappa_m / b_dec "
                "path reweighting consumes exactly this payload"
            ),
            "packed": pack_arrays([
                aggregated["var_ids"], aggregated["row_flat"],
                aggregated["row_weight"], aggregated["row_ptr"],
                aggregated["vtx_ptr"], aggregated["vtx_nchan"],
                aggregated["vtx_chosen"], aggregated["vtx_law"],
                aggregated["vtx_ck"], aggregated["ch_w0"],
                aggregated["ch_q2"],
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
    "the upper T2 bound, the R23/R32 convention); 800,000 events total"
)


def _chunk_tasks(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    kappa_b: float,
    kappa_m: float,
    b_dec: float,
    width: float,
    smoke: bool,
) -> list[tuple]:
    starts = list(range(0, int(n_events), CHUNK_SIZE))
    return [
        (int(replica), int(seed),
         [event_index(start + j) for j in range(start, min(start + CHUNK_SIZE, int(n_events)))],
         int(MASTER_SEED), int(replica_slot),
         float(BLEND_RAMP_START_GEV),
         float(kappa_b), float(kappa_m), float(b_dec),
         float(width), bool(smoke))
        for start in starts
    ]


def run_generation(
    run_dir: Path,
    pre: dict,
    replicas: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> list[str]:
    """Generate the requested production replica records (any subset;
    the default is every replica).  A completed record that matches the
    pre-registered identity is reused unchanged.  Returns the record
    names written or reused."""
    import os

    pre = _load_pre_registration(run_dir)
    _assert_pre_registration(pre, run_dir)
    pre_digest = hashlib.sha256(json.dumps(
        pre, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
                print(f"[r41] {name}: completed record reused", flush=True)
                written.append(name)
                continue
            _refuse(
                f"{path}: a completed record exists but does not match "
                "the pre-registered identity; a drifted record is never "
                "silently overwritten (delete the run dir and re-run)"
            )
        tasks = _chunk_tasks(replica, seed, replica,
                             T2_EVENTS_PER_REPLICA,
                             INITIAL_VALUES["kappa_B"],
                             INITIAL_VALUES["kappa_m"],
                             INITIAL_VALUES["b_dec"],
                             W_GRID_NOMINAL, smoke=False)
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
        print(f"[r41] {name}: written "
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
    the smoke fit, into an explicit ``--run-dir``."""
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
    # -- the smoke pre-registration (the smoke variant of the contract) --
    pre = build_pre_registration(smoke=True)
    pre["mode"] = "smoke"
    pre["smoke"] = SMOKE_PAYLOAD
    # the recording geometry of the smoke grid (the same pick convention;
    # the synthetic grid is rectangular, so no holes)
    pre["recording_convention"]["geometry"] = _recording_geometry(tables)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    pre_digest = _pre_registration_digest_of(pre)
    context["pre_registration_sha256"] = pre_digest
    _set_worker_context(context)
    # -- the smoke generation --
    replica = 0
    name = f"{SMOKE_PREFIX}replica_{replica}.json"
    seed = smoke_raw_seed()
    tasks = _chunk_tasks(
        replica, seed, SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        INITIAL_VALUES["kappa_B"], INITIAL_VALUES["kappa_m"],
        INITIAL_VALUES["b_dec"], W_GRID_NOMINAL, smoke=True)
    parts = list(_map_chunks(tasks, workers, "smoke"))
    aggregated = _aggregate_chunks(parts)
    record = _replica_record(
        replica, seed, SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        aggregated, _pre_registration_digest_of(pre), smoke=True)
    _write_json(run_dir / name, record)
    # -- the smoke fit (the machinery end to end; synthetic data) --
    report = run_fit_machinery(run_dir, smoke=True)
    _write_json(run_dir / FIT_REPORT_NAME, report)
    return {
        "record": name,
        "fit_report_digest": report.get("digest"),
    }


SMOKE_PAYLOAD = {
    "replicas": SMOKE_REPLICAS,
    "events_per_replica": SMOKE_EVENTS,
    "raw_seed": SMOKE_RAW_SEED,
    "raw_seed_rule": SMOKE_RAW_SEED_RULE,
    "kernel_replica_slot": SMOKE_KERNEL_REPLICA_SLOT,
    "synthetic_truth": SMOKE_SYNTHETIC_TRUTH,
    "w_grid": SMOKE_W_GRID,
    "note": SMOKE_NOTE,
}


# ---------------------------------------------------------------------------
# The pre-registration (written before any generation)
# ---------------------------------------------------------------------------
def build_pre_registration(smoke: bool = False) -> dict:
    """The pre-registered run contract, asserted against the card
    (TASK-20260920-r4-run-r41-nonstrange-pilot-fit)."""
    return {
        "schema":
            "na49_mixing.staged_r41_nonstrange_pilot_fit.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "plan_anchors": {
            "run": "plan section 10 run R41, line 768",
            "fit_set": "plan lines 770-775",
            "observables": "plan lines 779-782 (definitions plan section 7.3, lines 420-431)",
            "exchange_fixed": "plan line 784",
            "likelihood": "plan section 5.4 (lines 290-311)",
            "hessian_discipline": "plan section 5.4 line 311",
            "replica_blocks": "plan section 5.5 item 5, line 321",
            "statistics_tier": "plan section 5.1 line 254",
            "cascade_spec": "plan section 10.2 line 678 (each high-mass parent)",
            "not_do": "plan section 24 line 1866 (exchange slopes never control daughter-level transverse dynamics)",
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
                "convention (the R32 pre-registration, decision card "
                "lines 84-94)"
            ),
            "exchange_layer": (
                "byte-unchanged (the builder's exchange/topology/"
                "spectral-state layers are the frozen surface; the M8 "
                "set is applied unchanged)"
            ),
            "w_flavor": 1.0,
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
                "aligned cascade at the trial (kappa_B, kappa_m, b_dec) "
                "with the baryonic family's nominal_width as the trial "
                "width (the card-4 continuum-daughter mass rule); the "
                "card-9 routing order with card 10's attached per-vertex "
                "step, the frozen 1 channel + 2 kinematics per-vertex "
                "budget; the production axis is card 5's "
                "initial_source_axis against the source's own beam "
                "direction (projectile +z, target -z; the round 4 index "
                "item 4 p+p pre-registration)"
            ),
            "production_axis_note": (
                "the R40 isolated-source axis (the lab z-axis) is "
                "replaced by the p+p convention per index item 4; the "
                "card-10 runner copies the axis verbatim down the line"
            ),
        },
        "fit_set": {
            "parameters": list(FIT_SET),
            "count": len(FIT_SET),
            "excluded": FIT_SET_EXCLUDED,
            "initial_values": dict(INITIAL_VALUES),
            "parameter_bounds": {
                key: [float(lo), float(hi)]
                for key, (lo, hi) in PARAM_BOUNDS.items()
            },
            "width_treatment": W_GRID_NOTE,
            "width_grid": [float(w) for w in W_GRID],
            "reweightable": list(REWEIGHT_PARAMS),
            "reweight_note": (
                "kappa_B, kappa_m, and b_dec alter probability densities "
                "but not kinematic support (plan section 5.3 line 288): "
                "the per-vertex path ratios are exact (the channel-choice "
                "ratio of the card-8 attached shares; the angular density "
                "ratio of the card-6/7 laws at the recorded kernel "
                "cosine, including the card-6 trailing-daughter mirror "
                "and the mesonic even law's symmetry); the nominal_width "
                "alters the card-4 continuum-daughter mass rule, hence "
                "the realized parent masses of the continuation, so the "
                "width is regenerated at each pre-registered grid point "
                "(the documented support boundary)"
            ),
            "reweight_algebra": {
                "channel": (
                    "ratio = exp(-(b_trial - b_gen) q2_chosen) / "
                    "sum_i w0_i exp(-(b_trial - b_gen) q2_i), with w0 the "
                    "normalized attached shares at the generation b_dec"
                ),
                "angular": (
                    "ratio = exp((kappa_trial - kappa_gen) s(c_k)) * "
                    "Z(kappa_gen)/Z(kappa_trial), s = c_k for the "
                    "baryonic kernel and c_k^2 for the mesonic even law, "
                    "Z the closed-form normalizations; the recorded c_k "
                    "carries the card-6 trailing-daughter mirror sign"
                ),
                "mesonic_interpolation_note": (
                    "the mesonic law's draw is the pre-registered 10,001-"
                    "node CDF table with linear interpolation; the path "
                    "ratio uses the closed-form ideal densities, whose "
                    "deviation from the table-interpolated density is "
                    "bounded by the card-7 table fidelity (about 1e-8 "
                    "relative), far below the Monte Carlo errors"
                ),
            },
            "r40_initial_values": (
                "the nominal point (kappa_B, kappa_m, b_dec) = (1, 1, "
                "1.0 GeV^-2) and nominal_width = 0.5 GeV, the "
                "pre-registered R40 starting values (round 4 index item "
                "3); the R40 per-setting scan tables provide the gradient "
                "information (the house pattern of R20 before R21/R22 "
                "and R31 before R32)"
            ),
            "r40_gradient_information": R40_GRADIENT_INFORMATION,
        },
        "observables": {
            "training": (
                "the pooled combinations of plan section 7.3: the "
                "pi_Sigma = pi+ + pi- spectrum on the aligned cells, the "
                "B_transport = p - pbar spectrum on the aligned cells, "
                "the pooled pion and proton seagulls (the per-xF <pT>_x "
                "and <pT>_y conventions of fit_observables on the "
                "combinations at the pion / proton mass), and the pooled "
                "multiplicity moments (the paper-window first moments "
                "<n> of pi_Sigma and of p + pbar; the published data "
                "surface carries no multiplicity distribution, and the "
                "species-separated moments are R44 held-out territory, "
                "index item 11)"
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
                "locally loaded SPS tables at --fit time (the R11 "
                "parse_sealed_binned discipline)"
            ),
        },
        "recording_convention": {
            "convention": (
                "the R01 accumulate-dense-then-pick recording convention "
                "(the corrected convention of the relitigation card; the "
                "R44 driver's _worker_context / _replica_record pattern "
                "is the reference): every event weight accumulates into "
                "the species' dense pT-major evaluation lattice (n_x * "
                "n_p cells at the species' dense offset), and the "
                "published rows are picked from the dense totals at "
                "record assembly"
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
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "superseded_note": (
                "the superseded card-12 landing recorded the finals at "
                "the published-row flat index (species_offset(species) + "
                "ip * n_x + ix with the published-row cumulative "
                "offsets), which cell-transposed the spectra on the "
                "non-rectangular published geometry; the sealed "
                f"{SUPERSEDED_RUN_DIR} archive was generated and "
                "evaluated under that convention and is superseded by "
                "this run (it stays intact; its seal verifies)"
            ),
            "corrected_driver_sha256": DRIVER_SHA256,
        },
        "likelihood": {
            "form": (
                "-2lnL = (d - m)' V^-1 (d - m) + eta' P^-1 eta, with V = "
                "V_data + V_mc (reported separately), the published "
                "binning preserved, and the nuisance vector profiled "
                "exactly per trial: one multiplicative normalization "
                "nuisance per training block (prior 0.35) plus the "
                "obligation 4 carry as one correlated multiplicative "
                "nuisance on the blend-band-sensitive bins (prior 0.25)"
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
                    "audit's attribution; "
                    f"{len(CARRY_BIN_KEYS)} of the 32 intersect the fit "
                    "set's species and carry the term in the objective)"
                ),
            },
            "profile_is_not_fit_set": (
                "the nuisance profile is part of the objective definition "
                "(plan section 5.4), not part of the four-parameter fit "
                "set: the optimizer's parameter vector is exactly "
                f"{list(FIT_SET)}"
            ),
        },
        "hessian_discipline": {
            "rule": (
                "no Hessian errors are quoted until the replica-block "
                "minima agree (the acceptance (ii) stability table "
                "across all 8 blocks; plan section 5.5 item 5 line 321 "
                "requires at least four independent replica blocks) and "
                "the local curvature is stable (the pooled numerical "
                "Hessian positive definite on the reweightable "
                "directions and the width-profile curvature positive)"
            ),
            "steps": dict(HESSIAN_STEPS),
            "width_curvature": HESSIAN_W_FROM_GRID,
        },
        "statistics": {
            "tier": T2_LABEL,
            "replicas": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "total_events": T2_REPLICAS * T2_EVENTS_PER_REPLICA,
            "width_grid_trials": (
                "the four off-nominal width-grid generations run in "
                "memory on the production (seed, event_id) pairs "
                "(common random numbers, plan section 5.2 line 275); "
                "4 x 800,000 trial events never sealed, never written"
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
                "child consumes nothing in R41 (the R32 continuum root "
                "split is replaced by the card-10 attached vertices, "
                "which draw on the labeled ids; recorded so no draw-order "
                "ambiguity remains)"
            ),
            "disjointness": (
                "the R41 master seed 20260920 is disjoint from "
                "20260913 (R20-R23), 20260917 (R31), 20260918 (card 8c), "
                "and 20260919 (R32); the raw block 20301020-20301027 is "
                "disjoint from the R23 raw seeds, from the R32 "
                "production and confirmation blocks, from the audit "
                "block 20301040 + k, and from every previously used "
                "block; the R41 event-index partition 400_000_000 + i "
                "disjoint from the R40 partition 300_000_000 + ... and "
                "from the audit block"
            ),
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
                "fit_report.json and gate4_r41.json carry no wall-clock "
                "fields: their bytes are a pure function of the sealed "
                "artifacts, the pre-registered seeds, and the committed "
                "code"
            ),
        },
        "optimizer": {
            "optimizer_invoked": True,
            "optimizer_uses": 1,
            "optimizer_uses_note": (
                "the round's first and primary optimizer use; R42 "
                "(card 13) is the second and final"
            ),
            "algorithm": "scipy.optimize.minimize, method Nelder-Mead",
            "xatol": NM_XATOL,
            "fatol": NM_FATOL,
            "maxiter": NM_MAXITER,
            "maxfev": NM_MAXFEV,
            "initial_simplex": NM_INITIAL_SIMPLEX_NOTE,
            "start_point": {
                key: float(INITIAL_VALUES[key])
                for key in ("kappa_B", "kappa_m", "b_dec")
            },
            "start_point_note": (
                "the 3-parameter (kappa_B, kappa_m, b_dec) optimizer "
                "starts at the pre-registered initial values at EVERY "
                "width-grid point; the width coordinate of the fit is "
                "the pre-registered grid argmin (no off-grid width "
                "evaluation exists; the regeneration boundary)"
            ),
            "hessian": (
                "the numerical 3x3 Hessian of the pooled objective by "
                "central differences at the pooled optimum with the "
                "pre-registered steps, plus the width curvature from the "
                "grid profile; quoted only after the replica-block "
                "minima agree and the local curvature is stable"
            ),
        },
        "acceptance": {
            "precedent": (
                "the acceptance precedent of "
                "TASK-20260918-r3-acceptance-rescope-card8a-8b: an "
                "MC-driven check set with residuals seals with a named-"
                "residual register, not a raw zero-failure target"
            ),
            "i_fit_convergence": (
                "the optimizer converges on the four-parameter fit set "
                "under the pre-registered contract (xatol 1e-3, fatol "
                "1e-6, within maxiter/maxfev) at every width-grid point; "
                "the fitted values and the per-replica-block table are "
                "recorded"
            ),
            "ii_replica_block_stability": (
                "the fitted parameters stable across all 8 independent "
                "replica blocks, the per-block spread < 20% of the "
                "pooled error per parameter (pre-registered bound); a "
                "failing parameter is a named finding (the Gate 4 "
                "unconstrained clause consumes it directly)"
            ),
            "iii_conservation": (
                "800,000 events, exact B = 2 / Q = 2 and exact "
                "per-source (B, Q, S); zero violations"
            ),
            "iv_determinism": (
                "the clean-environment re-run reproduces the per-replica "
                "record identity digests and the fit evaluation hashes"
            ),
            "v_no_refit_beyond_fit_set": (
                "no exchange, topology, flavor, branching, or ramp-start "
                "parameter is fit (plan section 24; obligation 4)"
            ),
            "vi_seal_and_clean_validation": (
                "the seal is intact and the clean-environment re-run "
                "reproduces the fit evaluation hashes"
            ),
            "vii_pins_and_freeze": (
                "the pin test register lands, the freeze re-pin is "
                "recorded, and the full nested suite is green"
            ),
            "named_residual_register": (
                "every failing verdict named with sigma, a one-line "
                "diagnosis, and the robust-statistics cross-reference; "
                "the 3.0-sigma band is never relaxed; no data selection"
            ),
            "register_triggers": [
                "any acceptance (ii) stability failure (per parameter)",
                "a pooled optimum on any pre-registered parameter bound "
                "(a boundary finding: the fit range does not extend)",
                "an effective sample size fraction below "
                f"{ESS_FLOOR_FRACTION} at the pooled optimum (the "
                "weight-degeneracy diagnosis)",
                "a convergence failure of any pre-registered optimizer "
                "run (recorded; a convergence failure is a valid run "
                "outcome)",
                "the width argmin on the grid edge (the width is "
                "edge-constrained by the pre-registered bracket)",
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
                "the carry nuisance applies to the fit-set bins whose "
                "contributing species cells are in this sealed set"
            ),
            "source": (
                "dev/00runs/r4_audit_transition_routing_formulation_"
                "2026-09-20/persisting_bin_attribution.json rows "
                "(classification == 'persisting', the R32 confirmation "
                "reclassification)"
            ),
            "sigma_range": [2.0001998039779263, 7.532929574244998],
            "entries": 32,
        },
        "sealed_run_dirs": list(SEALED_RUN_DIRS),
        "frozen_substrate": {
            key: value[1] for key, value in FROZEN_SUBSTRATE.items()
        },
        "gateway_fallback": GATEWAY_FALLBACK_RULE,
        "written_before_generation": True,
    }


R40_GRADIENT_INFORMATION = {
    "source": f"dev/00runs/{R40_RUN_DIR}/scan_report.json (sealed)",
    "kappa_B": (
        "strong monotone response on the baryon-line longitudinal "
        "retention at the continuum-routed baryonic points (sigma up to "
        "486); flat at the catalog-routed 2 GeV point (sigma 1.47, the "
        "Round 2 kernel inert)"
    ),
    "kappa_M": (
        "zero central-multiplicity response at every mesonic point (the "
        "even law is angle-only); the supplementary meson alignment "
        "moment <pz^2/|p|^2> strongly increasing across the kappa_M "
        "slice (the even law's actual response)"
    ),
    "b_dec": (
        "the <pT>_x decrease as predicted at the continuum-routed points "
        "for the pions; two named |sigma| ~ 3.6 findings (pi- at the "
        "catalog-routed point 0, pure MC scatter; K+ at the 4 GeV "
        "baryonic point, the cascade-line momentum-redistribution "
        "finding candidate), both inside the R32 finite-MC-scatter "
        "envelope class (|sigma| <= 4.0); diagnosed, not refit"
    ),
    "note": (
        "the R40 scan tables provide the gradient information of the fit "
        "(the house pattern of R20 before R21/R22 and R31 before R32); "
        "they are initial-value context, not fit inputs"
    ),
}

AUDIT_OUTCOME = {
    "card_1": (
        "RECOMMENDATION (sealed dev/00runs/"
        "r4_audit_transition_routing_formulation_2026-09-20/verdict.json): "
        "the rework card 1r "
        "(TASK-20260920-r4-rework-transition-routing-formulation) is "
        "spawned and was open at run time; this run pre-registers "
        "against the closed audits per the card and carries the original "
        "R32 spread until card 1r resolves"
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
    "events ~ 11 min single-thread, ~ 2 min at 8 fork workers); if any "
    "replica exceeds 30 min wall, the remaining replicas run on the "
    "cluster gateway (host cl, workers c0801-c0809 x 32 task slots, one "
    "job per replica, --tasks 1, single math thread (NA49_CLUSTER_GW_"
    "TASKS forces thread counts to 1), no local process pool inside the "
    "chunk command, submission paced by the capacity check; record "
    "outputs are appended only to the new run dir)"
)



def _load_carry_bins() -> list[dict]:
    """The sealed 32 persisting bins (the blend-band-sensitive set),
    read read-only from the sealed card-1 attribution."""
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


CARRY_BIN_KEYS: list = []  # populated at pre-registration / fit time


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
    # the recording geometry of the pre-registered pick convention (the
    # per-species published-row vs lattice-cell counts of the run's
    # tables; the relitigation clause) and the superseded archive's seal
    # state (read-only; the seal verifies).  The smoke mode attaches its
    # own geometry from the synthetic grid (in run_smoke).
    superseded_status = verify_sealed_dir(SUPERSEDED_RUN_DIR)
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
        "schema": "na49_mixing.staged_r41_nonstrange_pilot_fit.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "new_driver_sha256_both_trees": DRIVER_SHA256,
        "driver_note": (
            "byte-identical in the nested tree (dev/mixing2006/helpers/"
            "benchmarking/) and the monorepo mirror (helpers/"
            "benchmarking/), the established _locate_roots pattern"
        ),
        "superseded_r41_archive": {
            "run_dir": SUPERSEDED_RUN_DIR,
            "state": (
                "intact; the seal verifies (verify_sealed_dir); the "
                "archive was generated and evaluated under the "
                "superseded card-12 driver's published-row flat index "
                "and is superseded by this run"
            ),
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "seal": superseded_status,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "corrected_driver_sha256": DRIVER_SHA256,
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "geometry": pre["recording_convention"]["geometry"],
        },
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
        "catalog_sha256": FROZEN_SUBSTRATE["catalog"][1],
        "continuum_config_sha256": FROZEN_SUBSTRATE["continuum_config"][1],
        "r32_driver_sha256": FROZEN_SUBSTRATE["r32_driver"][1],
        "r40_driver_sha256": FROZEN_SUBSTRATE["r40_driver"][1],
        "sealed_run_dirs": sealed,
        "r40_initial_values": (
            "the sealed R40 scan archive at "
            f"{R40_RUN_DIR}: the nominal point (kappa_B, kappa_m, "
            "b_dec) = (1, 1, 1.0 GeV^-2) and nominal_width = 0.5 GeV, "
            "the pre-registered starting values; the per-setting scan "
            "tables provide the gradient information"
        ),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(f"[r41] pre-registration written into {run_dir}", flush=True)
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
    fit_set = pre.get("fit_set", {}).get("parameters")
    if list(fit_set or ()) != list(FIT_SET):
        _refuse("the pre-registered fit set does not match the card")
    if pre.get("fit_set", {}).get("count") != len(FIT_SET):
        _refuse("the pre-registered fit-set count does not match the card")
    if "blend_ramp_start" not in pre.get("fit_set", {}).get("excluded", {}):
        _refuse("the blend ramp start exclusion is missing from the "
                "pre-registration")
    stats = pre.get("statistics", {})
    if int(stats.get("replicas", -1)) != T2_REPLICAS or int(
        stats.get("events_per_replica", -1)
    ) != T2_EVENTS_PER_REPLICA:
        _refuse("the pre-registered statistics do not match the card "
                "(T2: 8 replicas x 100,000)")
    seeds = pre.get("seed_map", {})
    if list(seeds_ := seeds.get("raw_replica_seeds") or ()) != \
            list(RAW_REPLICA_SEEDS):
        _refuse("the pre-registered raw replica seeds do not match the "
                "card's block 20301020-20301027")
    if seeds.get("event_index_rule") != EVENT_INDEX_RULE:
        _refuse("the pre-registered event-index rule does not match the "
                "card (400_000_000 + i, i < 100,000)")
    grid = pre.get("fit_set", {}).get("width_grid")
    if [float(w) for w in (grid or ())] != [float(w) for w in W_GRID]:
        _refuse("the pre-registered width grid does not match the card")
    carry = pre.get("obligation_4_carry", {})
    if carry.get("r32_spread", {}).get("bin_spread_p95") != \
            OBLIGATION4_R32_SPREAD["bin_spread_p95"]:
        _refuse("the obligation 4 carry numbers are not the card's "
                "verbatim R32 spread")
    if not pre.get("written_before_generation", False):
        _refuse("the pre-registration was not marked "
                "written_before_generation")
    for key in ("collision", "observables", "likelihood",
                "hessian_discipline", "conservation", "determinism"):
        if key not in pre:
            _refuse(f"the pre-registration is missing the {key} block")



# ---------------------------------------------------------------------------
# The fit data (the sealed R01 archive + the published binning)
# ---------------------------------------------------------------------------
R01_ARCHIVE = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"
COMBINATION_PAIRS = {
    "pi_sigma": ("pi_plus", "pi_minus", +1.0),
    "b_transport": ("proton", "antiproton", +1.0),
}
SEAGULL_MASS_NOTE = (
    "the pooled-combination seagull is evaluated at the combination's "
    "common species mass (the pion mass for pi_Sigma, the proton mass "
    "for p + pbar; both species of each combination share the mass)"
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
# The replica-block latent (the reweighting data, one width x replica)
# ---------------------------------------------------------------------------
class BlockLatent:
    """One replica block's reweighting payload at one width.

    ``fixed`` is the theta-independent row pool on the DENSE evaluation
    lattice (the events with no continuum-routed source); ``row_*`` is
    the CSR of the continuum-route events' bin rows at their dense
    lattice cells (the accumulate-dense-then-pick addresses; the
    published-row conversion happens at the evaluation boundary in
    ``_species_bin_values``); ``vtx_*``/``ch_*`` is the per-vertex
    attached pool
    of the card-8 shares (``w0`` at the generation b_dec) and the
    reference transverse momenta squared (``q2``), the chosen channel,
    the governing kernel law, and the signed kernel cosine.  The
    kappa_B / kappa_m / b_dec path reweighting consumes exactly this
    payload (plan section 5.3 line 288)."""

    __slots__ = (
        "replica", "seed", "n_events", "fixed", "row_flat", "row_weight",
        "row_ptr", "var_ids", "vtx_ptr", "vtx_nchan", "vtx_chosen",
        "vtx_law", "vtx_ck", "ch_w0", "ch_q2", "n_events_var",
        "_row_event_map", "_vtx_pool_ptr",
    )

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))
        self.n_events = int(self.n_events)
        self.n_events_var = int(self.var_ids.size)
        self.fixed = np.asarray(self.fixed, dtype=float)
        self.row_flat = np.asarray(self.row_flat, dtype=np.int64)
        self.row_weight = np.asarray(self.row_weight, dtype=float)
        self.row_ptr = np.asarray(self.row_ptr, dtype=np.int64)
        self.vtx_ptr = np.asarray(self.vtx_ptr, dtype=np.int64)
        self.vtx_nchan = np.asarray(self.vtx_nchan, dtype=np.int64)
        self.vtx_chosen = np.asarray(self.vtx_chosen, dtype=np.int64)
        self.vtx_law = np.asarray(self.vtx_law, dtype=np.int64)
        self.vtx_ck = np.asarray(self.vtx_ck, dtype=float)
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
                np.arange(self.n_events_var, dtype=np.int64), counts)
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
    (var_ids, row_flat, row_weight, row_ptr, vtx_ptr, vtx_nchan,
     vtx_chosen, vtx_law, vtx_ck, ch_w0, ch_q2) = unpack_arrays(
        record["latent"]["packed"])
    return BlockLatent(
        replica=int(record["replica"]),
        seed=int(record["seed"]),
        n_events=int(record["n_events"]),
        fixed=np.asarray(record["fixed_pool"], dtype=float),
        row_flat=row_flat,
        row_weight=row_weight,
        row_ptr=row_ptr,
        var_ids=var_ids,
        vtx_ptr=vtx_ptr,
        vtx_nchan=vtx_nchan,
        vtx_chosen=vtx_chosen,
        vtx_law=vtx_law,
        vtx_ck=vtx_ck.astype(float),
        ch_w0=ch_w0.astype(float),
        ch_q2=ch_q2.astype(float),
    )


def _block_from_aggregate(aggregate: dict) -> BlockLatent:
    return BlockLatent(
        replica=int(aggregate["replica"]),
        seed=int(aggregate["seed"]),
        n_events=int(aggregate["n_events"]),
        fixed=np.asarray(aggregate["fixed"], dtype=float),
        row_flat=aggregate["row_flat"],
        row_weight=np.asarray(aggregate["row_weight"], dtype=float),
        row_ptr=np.asarray(aggregate["row_ptr"], dtype=np.int64),
        var_ids=np.asarray(aggregate["var_ids"], dtype=np.int64),
        vtx_ptr=np.asarray(aggregate["vtx_ptr"], dtype=np.int64),
        vtx_nchan=np.asarray(aggregate["vtx_nchan"], dtype=np.int64),
        vtx_chosen=np.asarray(aggregate["vtx_chosen"], dtype=np.int64),
        vtx_law=np.asarray(aggregate["vtx_law"], dtype=np.int64),
        vtx_ck=np.asarray(aggregate["vtx_ck"], dtype=float),
        ch_w0=np.asarray(aggregate["ch_w0"], dtype=float),
        ch_q2=np.asarray(aggregate["ch_q2"], dtype=float),
    )


# ---------------------------------------------------------------------------
# The path reweighting (plan section 5.3 line 288; exact per-vertex ratios)
# ---------------------------------------------------------------------------
_BARYONIC_NORM_CACHE: dict[float, float] = {}
_MESONIC_NORM_CACHE: dict[float, float] = {}


def _baryonic_norm(kappa: float) -> float:
    from na49_mixing.cascade_alignment import baryonic_normalization

    key = float(kappa)
    cached = _BARYONIC_NORM_CACHE.get(key)
    if cached is None:
        cached = float(baryonic_normalization(key))
        _BARYONIC_NORM_CACHE[key] = cached
    return cached


def _mesonic_norm(kappa: float) -> float:
    from na49_mixing.cascade_alignment import mesonic_normalization

    key = float(kappa)
    cached = _MESONIC_NORM_CACHE.get(key)
    if cached is None:
        cached = float(mesonic_normalization(key))
        _MESONIC_NORM_CACHE[key] = cached
    return cached


def event_weights(block: BlockLatent, kappa_b: float, kappa_m: float,
                  b_dec: float) -> np.ndarray:
    """The per-event importance weights of one block at the trial
    (kappa_B, kappa_m, b_dec): the exact per-vertex path ratios (the
    card-8 channel-choice ratio; the card-6/7 angular density ratio
    with the trailing-daughter mirror), the product over the event's
    continuum vertices, one per continuum-route event; the theta-
    independent (catalog-only) events sit in the fixed pool with weight
    1 by construction."""
    kappa_b = float(kappa_b)
    kappa_m = float(kappa_m)
    b_dec = float(b_dec)
    n_var = block.n_events_var
    if n_var == 0 or block.vtx_nchan.size == 0:
        return np.ones(n_var, dtype=float)
    pool_ptr = block.vtx_pool_ptr
    delta_b = b_dec - float(INITIAL_VALUES["b_dec"])
    ex = np.exp(-delta_b * block.ch_q2)
    pool_sum = np.add.reduceat(block.ch_w0 * ex, pool_ptr[:-1])
    if not np.all(np.isfinite(pool_sum)) or not np.all(pool_sum > 0.0):
        _refuse(
            "r41 reweight: a vertex pool's trial weights sum to a "
            "non-finite or non-positive total; the trial b_dec is "
            "outside the pre-registered domain (checked, never repaired)"
        )
    log_channel = (
        -delta_b * block.ch_q2[pool_ptr[:-1] + block.vtx_chosen]
        - np.log(pool_sum)
    )
    delta_kb = kappa_b - float(INITIAL_VALUES["kappa_B"])
    delta_km = kappa_m - float(INITIAL_VALUES["kappa_m"])
    log_norm_b = math.log(
        _baryonic_norm(float(INITIAL_VALUES["kappa_B"]))
        / _baryonic_norm(kappa_b))
    log_norm_m = math.log(
        _mesonic_norm(float(INITIAL_VALUES["kappa_m"]))
        / _mesonic_norm(kappa_m))
    law = block.vtx_law
    ck = block.vtx_ck
    ang = np.where(
        law == 0,
        delta_kb * ck + log_norm_b,
        delta_km * ck * ck + log_norm_m,
    )
    log_eta = np.add.reduceat(log_channel + ang, block.vtx_ptr[:-1])
    if log_eta.size != n_var:
        _refuse(
            "r41 reweight: the per-event reduce produced "
            f"{log_eta.size} values for {n_var} continuum-route events"
        )
    return np.exp(log_eta)


def block_weighted_sums(block: BlockLatent, eta: np.ndarray) -> np.ndarray:
    """The block's weighted per-bin sums (the fixed pool plus the
    weighted var rows)."""
    totals = np.array(block.fixed, dtype=float, copy=True)
    if block.row_flat.size:
        counts = np.diff(block.row_ptr)
        event_map = np.repeat(
            np.arange(block.n_events_var, dtype=np.int64), counts)
        np.add.at(totals, block.row_flat, block.row_weight * eta[event_map])
    return totals


def ess_fraction(eta: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 over the weighted continuum-route ensemble
    (the reweighted subensemble whose weights carry the trial
    dependence; the weight-degeneracy diagnostic of the pre-
    registration)."""
    if eta.size == 0:
        return 1.0
    total = float(np.sum(eta))
    squares = float(np.sum(eta * eta))
    if squares <= 0.0:
        return 0.0
    return (total * total) / squares / float(eta.size)



# ---------------------------------------------------------------------------
# The training-observable statistic set
# ---------------------------------------------------------------------------
class Statistic:
    """One training-observable statistic (d, V_data, and the model
    evaluator spec).  Kinds: 'spectrum' (the binned combination cells),
    'seagull' (the two <pT> conventions per xF bin of a combination),
    'integral' (the pooled window first moment of a combination)."""

    __slots__ = ("key", "kind", "block_key", "d", "v_data", "areas",
                 "carry_mask", "combo", "seagull")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def _build_statistics(
    sealed: dict, tables_by_species: dict, carry_set: set,
    sqrt_s: float,
) -> list:
    """The pre-registered training observables: the pi_Sigma spectrum,
    the B_transport spectrum, the pooled pion and proton seagulls (both
    <pT> conventions per xF bin), and the pooled multiplicity moments
    (the paper-window first moments of the two combinations)."""
    stats = []
    for pair_key in ("pi_sigma", "b_transport"):
        spectrum = _combination_statistic(
            sealed, tables_by_species, pair_key, set(carry_set))
        stats.append(Statistic(
            key=spectrum["key"], kind="spectrum", block_key="spectrum",
            d=spectrum["d"], v_data=spectrum["v_data"],
            areas=spectrum["areas"], carry_mask=spectrum["carry_mask"],
            combo=spectrum, seagull=None))
        stats.append(_seagull_statistic(
            sealed, tables_by_species, pair_key, spectrum, sqrt_s))
        stats.append(_integral_statistic(spectrum))
    return stats


def _seagull_statistic(
    sealed: dict, tables_by_species: dict, pair_key: str, spectrum: dict,
    sqrt_s: float,
) -> Statistic:
    """The pooled-combination seagull: the per-xF <pT>_x and <pT>_y of
    the fit_observables conventions on both sides, the data side from
    the sealed d and the first-order delta-method variance over the
    aligned cells."""
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
    # the flat lists aligned with the cells' (ix, ip) on the table grid
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
    moments, v_data = [], []
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
    d_flat = np.asarray([
        float("inf") if m is None else float(m) for m in moments
    ], dtype=float)
    return Statistic(
        key=f"{pair_key}_seagull", kind="seagull", block_key="seagull",
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
        },
    )


def _integral_statistic(spectrum: dict) -> Statistic:
    """The pooled window first moment <n> of a combination (the
    midpoint-rectangle window integral of the combination cells; the
    plan section 7.3 multiplicity observable at the pipeline's
    published data surface)."""
    weights = spectrum["areas"]
    d = float(np.sum(spectrum["d"] * weights))
    v_data = float(np.sum(spectrum["v_data"] * weights * weights))
    return Statistic(
        key=spectrum["key"].replace("_spectrum", "_multiplicity"),
        kind="integral", block_key="multiplicity",
        d=np.asarray([d]), v_data=np.asarray([max(v_data, SE_FLOOR)]),
        areas=weights, carry_mask=np.asarray([0.0]),
        combo=spectrum, seagull=None,
    )


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


def _integral_model(stat: Statistic, species_values: dict) -> float:
    combo_values = _combination_model(stat, species_values)
    return float(np.sum(combo_values * stat.areas))



# ---------------------------------------------------------------------------
# The objective (the profiled plan-5.4 likelihood) and the optimizer
# ---------------------------------------------------------------------------
NUISSANCE_COLUMNS = (
    "norm_pi_sigma_spectrum", "norm_b_transport_spectrum",
    "norm_multiplicity", "carry",
)
SPECTRUM_NORM_COLUMN = {
    "pi_sigma_spectrum": 0,
    "b_transport_spectrum": 1,
}
MULTIPLICITY_NORM_COLUMN = 2
CARRY_COLUMN = 3
#: The normalization-nuisance column of each level statistic (the
#: seagull statistics are normalization-free ratios and carry none).
STAT_BLOCK_COLUMN = {
    "pi_sigma_spectrum": 0,
    "b_transport_spectrum": 1,
    "pi_sigma_multiplicity": 2,
    "b_transport_multiplicity": 2,
}
CARRY_COLUMN_INDEX = 3


class FitObjective:
    """The profiled -2lnL of the pre-registered likelihood (plan
    section 5.4): the four-parameter fit set enters through the exact
    per-vertex path reweighting (the kappa_B / kappa_m / b_dec
    directions) and the pre-registered width grid (the regeneration
    boundary); the nuisance profile (the level-statistic normalization
    nuisances plus the obligation 4 carry) is solved exactly per trial
    by the generalized-least-squares profile.  The seagull statistics
    are normalization-free ratios and carry no normalization nuisance;
    the carry applies to the two binned spectra only.  A statistic bin
    whose pooled MC error is infinite (a structural zero bin) carries
    an infinite sigma and is dropped from the objective (the R11
    convention; the dropped bins are listed, never silently dropped)."""

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
    def block_vectors(self, block, kappa_b: float, kappa_m: float,
                      b_dec: float) -> tuple:
        eta = event_weights(block, kappa_b, kappa_m, b_dec)
        sums = block_weighted_sums(block, eta)
        n_var = int(len(block.var_ids))
        n_eff = float(np.sum(eta)) + float(block.n_events - n_var)
        species_values = _species_bin_values(
            sums, self.tables_by_species, self.sqrt_s, n_eff)
        vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum":
                vectors.append(_combination_model(stat, species_values))
            elif stat.kind == "seagull":
                vectors.append(_seagull_model(stat, species_values))
            else:
                vectors.append(np.asarray(
                    [_integral_model(stat, species_values)]))
        return vectors, n_eff, ess_fraction(eta)

    # -- the trial evaluation -------------------------------------------
    def evaluate(self, kappa_b: float, kappa_m: float,
                 b_dec: float) -> dict:
        per_block = []
        # the pooled sums accumulate on the dense evaluation lattice
        # (the accumulate-dense-then-pick convention; the per-species
        # published-row conversion happens in _species_bin_values)
        sums_total = np.zeros(_dense_total(), dtype=float)
        for block in self.blocks:
            eta = event_weights(block, kappa_b, kappa_m, b_dec)
            sums = block_weighted_sums(block, eta)
            n_var = int(len(block.var_ids))
            n_eff = float(np.sum(eta)) + float(block.n_events - n_var)
            species_values = _species_bin_values(
                sums, self.tables_by_species, self.sqrt_s, n_eff)
            vectors = []
            for stat in self.stats:
                if stat.kind == "spectrum":
                    vectors.append(
                        _combination_model(stat, species_values))
                elif stat.kind == "seagull":
                    vectors.append(_seagull_model(stat, species_values))
                else:
                    vectors.append(np.asarray(
                        [_integral_model(stat, species_values)]))
            per_block.append({
                "vectors": vectors, "n_eff": n_eff,
                "ess": ess_fraction(eta),
            })
            sums_total += sums
        n_eff_total = 0.0
        for entry in per_block:
            n_eff_total += float(entry["n_eff"])
        pooled_binned = _species_bin_values(
            sums_total, self.tables_by_species, self.sqrt_s, n_eff_total)
        pooled_vectors = []
        for stat in self.stats:
            if stat.kind == "spectrum":
                pooled_vectors.append(
                    _combination_model(stat, pooled_binned))
            elif stat.kind == "seagull":
                pooled_vectors.append(_seagull_model(stat, pooled_binned))
            else:
                pooled_vectors.append(np.asarray(
                    [_integral_model(stat, pooled_binned)], dtype=float))
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
                "r41 fit: no finite objective bin (the statistic set is "
                "empty after the structure drops); the run stops"
            )
        r = np.asarray(rows, dtype=float)
        design = np.asarray(columns, dtype=float)
        prior_diag = np.asarray([
            self.norm_prior ** 2, self.norm_prior ** 2,
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
        row = [0.0, 0.0, 0.0, 0.0]
        column = STAT_BLOCK_COLUMN.get(stat.key)
        if column is not None:
            row[column] = 1.0
        if stat.carry_mask is not None and len(stat.carry_mask) > int(index):
            row[CARRY_COLUMN_INDEX] = float(stat.carry_mask[index])
        return row


CARRY_COLUMN_INDEX = 3


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



# ---------------------------------------------------------------------------
# The fit helpers (blocks, optimizer, per-block fits, stability, Hessian)
# ---------------------------------------------------------------------------
def _synthetic_carry_all() -> set:
    """The smoke fit's carry set: every species bin (the synthetic
    machinery validation; never used in production)."""
    out = set()
    for species in SPECIES_FLAT_ORDER:
        size = int(_SPECIES_SIZE.get(species, 0))
        out.update((species, i) for i in range(size))
    return out


def _load_blocks(run_dir: Path, smoke: bool) -> list:
    """The sealed production records' reweighting payloads at the
    nominal width (or the single smoke record)."""
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


def _species_unslice(pooled_sums: dict) -> np.ndarray:
    pooled = np.zeros(_flat_total(), dtype=float)
    for species in SPECIES_FLAT_ORDER:
        start = int(SPECIES_FLAT_OFFSETS[species])
        values = np.asarray(pooled_sums[species], dtype=float)
        pooled[start:start + values.size] = values
    return pooled


def _generate_trial_blocks(run_dir: Path, width: float, smoke: bool) -> list:
    """The in-memory trial generation at an off-nominal width grid
    point: the production (seed, event_id) pairs (common random
    numbers, plan section 5.2 line 275), the walk at the trial width,
    one BlockLatent per replica (never sealed, never written; the
    documented regeneration boundary of plan section 5.3 line 288)."""
    tables = (
        _load_smoke_tables() if smoke else _load_production_tables()
    )
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=True)
    _set_worker_context(context)
    replicas = [0] if smoke else list(range(T2_REPLICAS))
    n_events = SMOKE_EVENTS if smoke else T2_EVENTS_PER_REPLICA
    blocks = []
    for replica in replicas:
        if smoke:
            seed = smoke_raw_seed()
            slot = SMOKE_KERNEL_REPLICA_SLOT
        else:
            seed = raw_replica_seed(replica)
            slot = kernel_replica_slot(replica)
        tasks = _chunk_tasks(
            replica, seed, slot, n_events,
            INITIAL_VALUES["kappa_B"], INITIAL_VALUES["kappa_m"],
            INITIAL_VALUES["b_dec"], float(width), smoke=True)
        parts = list(_map_chunks(
            tasks, DEFAULT_WORKERS,
            f"width {width} trial replica {replica}"))
        aggregated = _aggregate_chunks(parts)
        aggregated["replica"] = int(replica)
        aggregated["seed"] = int(seed)
        blocks.append(_block_from_aggregate(aggregated))
    return blocks


def _minimize_reweight(objective: FitObjective):
    """The pre-registered 3-parameter optimizer over the reweightable
    directions at a fixed grid width (scipy Nelder-Mead, bounded,
    deterministic)."""
    from scipy.optimize import minimize

    bounds = [tuple(PARAM_BOUNDS[key]) for key in
              ("kappa_B", "kappa_m", "b_dec")]
    x0 = [
        float(INITIAL_VALUES["kappa_B"]),
        float(INITIAL_VALUES["kappa_m"]),
        float(INITIAL_VALUES["b_dec"]),
    ]

    def target(x):
        value = objective.evaluate(float(x[0]), float(x[1]),
                                   float(x[2]))["value"]
        if not math.isfinite(value):
            return float("inf")
        return float(value)

    return minimize(
        target, x0, method="Nelder-Mead", bounds=bounds,
        options={
            "xatol": NM_XATOL, "fatol": NM_FATOL,
            "maxiter": NM_MAXITER, "maxfev": NM_MAXFEV,
            "adaptive": True,
        },
    )


def _fit_blocks_individually(
    stats, blocks, tables_by_species, sqrt_s,
) -> list:
    """The 8 per-block fits (the replica-block stability table): the
    same pre-registered procedure on each block's own data."""
    block_fits = []
    for block in blocks:
        objective = FitObjective(
            stats, [block], tables_by_species, sqrt_s,
            CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
        result = _minimize_reweight(objective)
        block_fits.append({
            "replica": int(block.replica),
            "minus2lnl": float(result.fun),
            "kappa_B": float(result.x[0]),
            "kappa_m": float(result.x[1]),
            "b_dec": float(result.x[2]),
            "converged": bool(result.success),
            "nfev": int(result.nfev),
        })
    return block_fits


def _stability_table(best_x, block_fits, pooled_errors: dict) -> dict:
    """The acceptance (ii) stability table: per parameter the 8 block
    values, the spread, and the 20%-of-pooled-error bound."""
    entries = {}
    for name, pooled_error in pooled_errors.items():
        values = [fit[name] for fit in block_fits]
        spread = (
            float(max(values) - min(values)) if len(values) >= 2
            else float("nan")
        )
        bound = STABILITY_SPREAD_BOUND * float(pooled_error)
        entries[name] = {
            "per_block": [float(v) for v in values],
            "spread": spread,
            "pooled_error": float(pooled_error),
            "bound": float(bound),
            "pass": bool(spread < bound),
        }
    return entries


def _pooled_curvature(objective: FitObjective, best_x) -> np.ndarray:
    """The 3x3 numerical Hessian of the pooled objective at the optimum
    (central differences with the pre-registered steps)."""
    steps = [HESSIAN_STEPS["kappa_B"], HESSIAN_STEPS["kappa_m"],
             HESSIAN_STEPS["b_dec"]]
    hessian = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            step_i = steps[i]
            step_j = steps[j]
            args = list(best_x)
            args[i] += step_i
            args[j] += step_j
            f_pp = objective.evaluate(args[0], args[1], args[2])["value"]
            args = list(best_x)
            args[i] += step_i
            args[j] -= step_j
            f_pm = objective.evaluate(args[0], args[1], args[2])["value"]
            args = list(best_x)
            args[i] -= step_i
            args[j] += step_j
            f_mp = objective.evaluate(args[0], args[1], args[2])["value"]
            args = list(best_x)
            args[i] -= step_i
            args[j] -= step_j
            f_mm = objective.evaluate(args[0], args[1], args[2])["value"]
            entry = (f_pp - f_mp - f_pm + f_mm) / (4.0 * step_i * step_j)
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
    for index, name in enumerate(("kappa_B", "kappa_m", "b_dec")):
        diagonal = float(curvature[index][index])
        errors[name] = (
            1.0 / math.sqrt(diagonal) if diagonal > 0.0
            else float("inf"))
    return errors


def _width_curvature(profile: list, best_width: float) -> float:
    """The width curvature from the five-point grid profile (the
    pre-registered second difference at the argmin over its
    neighbours; no off-grid width evaluation exists)."""
    widths = [entry["width_gev"] for entry in profile]
    if float(best_width) not in widths:
        return float("nan")
    index = widths.index(float(best_width))
    if index <= 0 or index >= len(profile) - 1:
        return float("nan")
    left = float(profile[index - 1]["minus2lnl"])
    center = float(profile[index]["minus2lnl"])
    right = float(profile[index + 1]["minus2lnl"])
    dw = (widths[index + 1] - widths[index - 1]) / 2.0
    if not dw > 0.0:
        return float("nan")
    return (left - 2.0 * center + right) / (dw * dw)


def _hessian_block(
    final_objective, best_x, best_width, profile, stability,
    smoke: bool,
) -> dict:
    """The Hessian block: quoted only after the replica minima agree
    (the stability table passes on every parameter; plan section 5.4
    line 311) and the local curvature is stable (positive definite);
    otherwise ``quoted: false`` and the failure named."""
    minima_agree = (
        bool(stability) and all(
            bool(entry["pass"]) for entry in stability.values())
    )
    curvature = _pooled_curvature(final_objective, best_x)
    eigenvalues = [float(v) for v in np.linalg.eigvalsh(curvature)]
    positive = bool(all(v > 0.0 for v in eigenvalues))
    h_ww = _width_curvature(profile, best_width)
    quoted = bool(
        (not smoke) and minima_agree and positive and h_ww > 0.0)
    return {
        "quoted": bool(quoted_flag(minima_agree, positive, h_ww, smoke)),
        "minima_agree": minima_agree,
        "curvature": [[float(v) for v in row] for row in curvature],
        "eigenvalues": eigenvalues,
        "positive_definite": positive,
        "width_curvature": h_ww,
        "width_error_note": (
            "the width error is the profile-likelihood interval of the "
            "grid profile (the min + 1 crossings, linear interpolation "
            "between the grid points; the grid resolution is the "
            "pre-registered width resolution)"
        ),
        "steps": {key: float(value) for key, value in HESSIAN_STEPS.items()},
        "note": HESSIAN_W_FROM_GRID,
    }


def quoted_flag(minima_agree, positive, h_ww, smoke) -> bool:
    return bool((not smoke) and minima_agree and positive and h_ww > 0.0)


def _width_curvature(profile, best_width) -> float:
    widths = [entry["width_gev"] for entry in profile]
    if float(best_width) not in widths:
        return float("nan")
    index = widths.index(float(best_width))
    if index <= 0 or index >= len(profile) - 1:
        return float("nan")
    left = float(profile[index - 1]["minus2lnl"])
    center = float(profile[index]["minus2lnl"])
    right = float(profile[index + 1]["minus2lnl"])
    dw = (widths[index + 1] - widths[index - 1]) / 2.0
    if not dw > 0.0:
        return float("nan")
    return (left - 2.0 * center + right) / (dw * dw)



# ---------------------------------------------------------------------------
# run_fit (the fit evaluation over the sealed artifacts)
# ---------------------------------------------------------------------------
def _build_carry_set() -> set:
    """The blend-band-sensitive cells: the sealed 32 persisting bins
    (the card-1 attribution) restricted to the fit species."""
    global CARRY_BIN_KEYS
    if not CARRY_BIN_KEYS:
        CARRY_BIN_KEYS = _load_carry_bins()
    return {(entry["species"], int(entry["bin_index"]))
            for entry in CARRY_BIN_KEYS}


def _synthetic_data_for(stats: list, truth_vectors: dict) -> None:
    """The smoke fit's synthetic data: the model at the pre-registered
    synthetic-truth point, with a synthetic 10% published error; the
    carry set is every spectrum bin (the machinery validation)."""
    for stat in stats:
        stat.d = np.asarray(truth_vectors[stat.key], dtype=float).copy()
        finite = np.isfinite(stat.d)
        stat.d = np.where(finite, stat.d, 0.0)
        stat.v_data = np.where(finite, (0.1 * np.abs(stat.d)) ** 2,
                               np.inf)
        stat.v_data = np.maximum(stat.v_data, SE_FLOOR)


def run_fit_machinery(run_dir: Path, smoke: bool = False) -> dict:
    """The fit evaluation: the per-replica-block likelihoods, the pooled
    fit over the width grid, the per-block fits, the replica-block
    stability table, and the Hessian (only after the replica minima
    agree).  Returns the fit report payload (no wall-clock fields)."""
    run_dir = Path(run_dir)
    tables = (
        _load_smoke_tables() if smoke else _load_production_tables()
    )
    _init_species_layout(tables)
    # the dense evaluation-lattice layout of the accumulate-dense-then-
    # pick convention (the published-row pick maps; the fit's model side
    # converts the dense weighted sums to published rows per species)
    _init_dense_layout(tables)
    tables_by_species = {t.species: t for t in tables}
    sqrt_s = _sqrt_s()
    if smoke:
        # the smoke statistic structure on a synthetic placeholder data
        # vector (overwritten by the synthetic-truth evaluation below)
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
        carry_set = _build_carry_set()
        stats = _build_statistics(
            sealed, tables_by_species, carry_set, sqrt_s)
    # -- the blocks at the nominal width (the sealed production records
    #    or the smoke record) --
    nominal_blocks = _load_blocks(run_dir, smoke)
    # -- the objective at the nominal width --
    objective = FitObjective(
        stats, nominal_blocks, tables_by_species, sqrt_s,
        CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
    # -- the smoke synthetic data at the synthetic truth --
    if smoke:
        truth_eval = objective.evaluate(
            SMOKE_SYNTHETIC_TRUTH["kappa_B"],
            SMOKE_SYNTHETIC_TRUTH["kappa_m"],
            SMOKE_SYNTHETIC_TRUTH["b_dec"])
        _synthetic_data_for(stats, {
            stat.key: vec for stat, vec in zip(stats, truth_eval[
                "pooled_vectors"])})
    # -- the width grid: the per-width 3-parameter optimizer -------------
    profile = []
    for width in (SMOKE_W_GRID if smoke else W_GRID):
        blocks = (
            nominal_blocks if float(width) == W_GRID_NOMINAL
            else _generate_trial_blocks(run_dir, float(width), smoke)
        )
        trial = FitObjective(
            stats, blocks, tables_by_species, sqrt_s,
            CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
        result = _minimize_reweight(trial)
        profile.append({
            "width_gev": float(width),
            "minus2lnl": float(result.fun),
            "kappa_B": float(result.x[0]),
            "kappa_m": float(result.x[1]),
            "b_dec": float(result.x[2]),
            "converged": bool(result.success),
            "nfev": int(result.nfev),
            "message": str(result.message),
        })
    best = min(
        (entry for entry in profile if math.isfinite(entry["minus2lnl"])),
        key=lambda e: e["minus2lnl"])
    best_width = float(best["width_gev"])
    best_x = (best["kappa_B"], best["kappa_m"], best["b_dec"])
    # -- the evaluation at the pooled optimum ----------------------------
    final_blocks = (
        nominal_blocks if best_width == W_GRID_NOMINAL
        else _generate_trial_blocks(run_dir, best_width, smoke)
    )
    final_objective = FitObjective(
        stats, final_blocks, tables_by_species, sqrt_s,
        CARRY_NUISANCE_PRIOR, NORM_NUISANCE_PRIOR)
    final_eval = final_objective.evaluate(*best_x)
    # -- the per-block fits (the replica-block stability table) ----------
    block_fits = []
    if not smoke:
        block_fits = _fit_blocks_individually(
            stats, final_blocks, tables_by_species, sqrt_s)
    # -- the stability table and the Hessian -----------------------------
    stability = _stability_table(
        best_x, block_fits, _pooled_errors(final_objective, best_x))
    hessian = _hessian_block(
        final_objective, best_x, best_width, profile, stability, smoke)
    report_named_residuals = _assemble_named_residuals({
        "replica_block_stability": stability,
        "optimizer": {
            "converged": all(entry["converged"] for entry in profile),
            "profile": profile,
        },
        "fitted": {
            "kappa_B": best_x[0], "kappa_m": best_x[1],
            "b_dec": best_x[2],
            "baryonic_nominal_width": best_width,
        },
        "objective_at_optimum": {
            "ess": final_eval["ess"],
        },
        "hessian": hessian,
    })
    report = {
        "schema": "na49_mixing.staged_r41_nonstrange_pilot_fit.report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "fit_set": list(FIT_SET),
        "initial_values": dict(INITIAL_VALUES),
        "fitted": {
            "kappa_B": best_x[0],
            "kappa_m": best_x[1],
            "b_dec": best_x[2],
            "baryonic_nominal_width": best_width,
        },
        "optimizer": {
            "invoked": True,
            "uses": 1,
            "algorithm": "scipy Nelder-Mead, bounded, pre-registered "
                         f"xatol {NM_XATOL}, fatol {NM_FATOL}, maxiter "
                         f"{NM_MAXITER}, maxfev {NM_MAXFEV}",
            "width_grid": [float(w) for w in (SMOKE_W_GRID if smoke
                                              else W_GRID)],
            "converged": all(entry["converged"] for entry in profile),
            "profile": profile,
        },
        "width_profile": profile,
        "objective_at_optimum": {
            "minus2lnl_profiled": final_eval["value"],
            "nuisances": [float(v) for v in final_eval["eta_hat"]],
            "nuisance_columns": list(NUISSANCE_COLUMNS),
            "n_eff_total": final_eval["n_eff_total"],
            "ess": final_eval["ess"],
        },
        "per_block_ess": [float(e["ess"]) for e in final_eval[
            "per_block"]],
        "dropped_bins": final_objective.dropped_bins[:50],
        "n_dropped_bins": len(final_objective.dropped_bins),
        "block_fits": block_fits,
        "replica_block_stability": stability,
        "hessian": hessian,
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "carry_prior": CARRY_NUISANCE_PRIOR,
        },
        "named_residual_register": report_named_residuals,
        "no_refit": (
            "the fit set is exactly the four pre-registered parameters; "
            "no exchange, topology, flavor, branching, or ramp-start "
            "parameter is fit (plan section 24; obligation 4)"
        ),
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "determinism": {
            "report_note": (
                "no wall-clock fields: the report bytes are a pure "
                "function of the sealed artifacts, the pre-registered "
                "seeds, and the committed code"
            ),
        },
    }
    digest = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["digest"] = digest
    return report



# ---------------------------------------------------------------------------
# The named-residual register, the gate record, the seal, clean-validate
# ---------------------------------------------------------------------------
def _assemble_named_residuals(report: dict) -> list:
    """The named-residual register of the acceptance precedent: every
    failing verdict named with the observed value, a one-line diagnosis,
    and the robust-statistics cross-reference."""
    register = []
    stability = report.get("replica_block_stability") or {}
    for name, entry in stability.items():
        if not entry["pass"]:
            register.append({
                "index": len(register) + 1,
                "kind": "replica_block_stability",
                "parameter": name,
                "per_block": entry["per_block"],
                "spread": entry["spread"],
                "pooled_error": entry["pooled_error"],
                "bound": entry["bound"],
                "sigma": (
                    None if entry["pooled_error"] == 0.0
                    else float(entry["spread"] / entry["pooled_error"])
                ),
                "diagnosis": (
                    f"the 8-block spread of {name} exceeds 20% of the "
                    "pooled error (the pre-registered acceptance (ii) "
                    "bound); the parameter is weakly constrained by the "
                    "training observables and the Gate 4 unconstrained "
                    "clause consumes it directly"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
    optimizer = report.get("optimizer") or {}
    if not report.get("optimizer", {}).get("converged", False):
        register.append({
            "index": len(register) + 1,
            "kind": "fit_convergence",
            "observed": [
                {"width_gev": e["width_gev"],
                 "converged": e["converged"],
                 "message": e["message"]}
                for e in report.get("optimizer", {}).get("profile", [])
                if not e.get("converged")
            ],
            "diagnosis": (
                "at least one pre-registered optimizer run did not "
                "converge within the pre-registered tolerances; a "
                "convergence failure is a valid run outcome and is "
                "recorded, never silently accepted"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    fitted = report.get("fitted", {})
    for name, (lo, hi) in PARAM_BOUNDS.items():
        value = fitted.get(name)
        if value is None:
            continue
        if float(value) <= float(lo) + 1e-9:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_optimum",
                "parameter": name,
                "observed": float(value),
                "bound": [float(lo), float(hi)],
                "diagnosis": (
                    f"the fitted {name} sits on the pre-registered "
                    "lower bound; the fit range does not extend and the "
                    "finding is named, never refit"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
        elif float(value) >= float(hi) - 1e-9:
            register.append({
                "index": len(register) + 1,
                "kind": "boundary_optimum",
                "parameter": name,
                "observed": float(value),
                "bound": [float(lo), float(hi)],
                "diagnosis": (
                    f"the fitted {name} sits on the pre-registered "
                    "upper bound; the fit range does not extend"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
    hessian = report.get("hessian") or {}
    if not hessian.get("quoted", False):
        reasons = []
        if not hessian.get("minima_agree", False):
            reasons.append("the replica-block minima do not agree")
        if not hessian.get("positive_definite", False):
            reasons.append("the pooled numerical curvature is not "
                           "positive definite (an optimum on the "
                           "pre-registered bounds has one-sided "
                           "curvature)")
        if not (isinstance(hessian.get("width_curvature"), float)
                and hessian.get("width_curvature") == hessian.get(
                    "width_curvature")):
            reasons.append("the width curvature is undefined on the grid")
        if hessian.get("width_curvature") is not None and not (
                isinstance(hessian.get("width_curvature"), (int, float))
                and hessian.get("width_curvature") > 0.0):
            if "the width curvature" not in " ".join(reasons):
                reasons.append("the width-profile curvature is not "
                               "positive")
        register.append({
            "index": len(register) + 1,
            "kind": "hessian_not_quoted",
            "observed": {
                "minima_agree": hessian.get("minima_agree"),
                "positive_definite": hessian.get("positive_definite"),
                "width_curvature": hessian.get("width_curvature"),
            },
            "diagnosis": (
                "the Hessian errors are not quoted (plan section 5.4 "
                "line 311): " + "; ".join(reasons) + " -- a fit pinned "
                "on the pre-registered bounds has one-sided curvature "
                "and the pooled-error bookkeeping is a boundary "
                "construct"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    ess = report.get("objective_at_optimum", {}).get("ess") or {}
    if ess and float(ess.get("min", 1.0)) < ESS_FLOOR_FRACTION:
        register.append({
            "index": len(register) + 1,
            "kind": "weight_degeneracy",
            "observed": ess.get("min"),
            "floor": ESS_FLOOR_FRACTION,
            "diagnosis": (
                "the effective sample size of the reweighted ensemble "
                "falls below the pre-registered floor at the pooled "
                "optimum (the importance weights degenerate at the "
                "fitted parameters); the fit is recorded with this "
                "diagnosis"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    return register


def run_gate(run_dir: Path) -> dict:
    """Write gate4_r41.json (evidence only), then stop."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / FIT_REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {FIT_REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    records = [
        _read_json(run_dir / RECORD_FMT.format(replica))
        for replica in range(T2_REPLICAS)
    ]
    conservation = {
        "events_checked": sum(
            int(r["n_events"]) for r in records),
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
    fitted = report.get("fitted", {})
    stability = report.get("replica_block_stability", {})
    gate = {
        "schema": "na49_mixing.staged_r41_nonstrange_pilot_fit.gate/1",
        "gate": "gate 4 (plan section 10, line 806)",
        "run": RUN_ID,
        "card": CARD,
        "clause": "unconstrained",
        "fit_set": list(FIT_SET),
        "excluded": FIT_SET_EXCLUDED,
        "events_checked": conservation["events_checked"],
        "conservation": conservation,
        "fitted": {
            "kappa_B": fitted.get("kappa_B"),
            "kappa_m": fitted.get("kappa_m"),
            "b_dec": fitted.get("b_dec"),
            "baryonic_nominal_width": fitted.get("baryonic_nominal_width"),
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
        "hessian": report.get("hessian", {}),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
        },
        "named_residual_register": report.get("named_residual_register", []),
        "optimizer_invoked": True,
        "optimizer_uses": 1,
        "fit_report_digest": report.get("digest"),
        "no_fitting_note": (
            "the fit set is the card's pre-registered four-parameter "
            "pilot fit (plan lines 770-775); the fit is the card's own "
            "pre-registered procedure, not a violation of the no-"
            "fitting discipline that governs the RUN cards' acceptance "
            "checks"
        ),
        "sealed": True,
        "sealed_note": (
            "the run seals after this record (the seal manifest and "
            "SHA256SUMS.txt cover this file; the run dir is immutable "
            "after the seal)"
        ),
        "gate_note": (
            "evidence only: the written retain / revise / reject "
            "decision on the Gate 4 clauses is the round's separate "
            "follow-up obligation, recorded in _knowledge/ and in the "
            "round 4 index status field"
        ),
    }
    _write_json(run_dir / GATE_RECORD_NAME, gate)
    print(f"[r41] gate record written: {run_dir / GATE_RECORD_NAME}",
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
        "schema": "na49_mixing.staged_r41_nonstrange_pilot_fit.seal/1",
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
        PREREGISTRATION_NAME, PROVENANCE_NAME, FIT_REPORT_NAME,
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
    print(f"[r41] sealed {run_dir} ({n_entries} entries)", flush=True)
    return manifest


def run_clean_validate(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """Verify the seal, re-run the fit evaluation from a clean
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
    # -- the fit evaluation re-run over the sealed artifacts ----------
    sealed_report = _read_json(run_dir / FIT_REPORT_NAME)
    rerun_report = run_fit_machinery(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest")
    )
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
        INITIAL_VALUES["kappa_B"], INITIAL_VALUES["kappa_m"],
        INITIAL_VALUES["b_dec"], W_GRID_NOMINAL, smoke=False)
    parts = list(_map_chunks(tasks, workers, "probe replica"))
    aggregated = _aggregate_chunks(parts)
    record = _replica_record(
        probe, seed, probe, T2_EVENTS_PER_REPLICA, aggregated,
        pre_digest, smoke=False)
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
            "na49_mixing.staged_r41_nonstrange_pilot_fit.clean_env/1",
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            "cd dev/mixing2006 && python -m pytest "
            "tests/test_staged_r41_nonstrange_pilot_fit.py -q",
            "python -m pytest tests/test_legacy_v1_freeze.py "
            "-W error::UserWarning -q",
            f"python helpers/benchmarking/"
            f"staged_r41_nonstrange_pilot_fit.py --clean-validate "
            f"--run-dir {run_dir}",
        ],
        "seal": {"entries": entries, "mismatches": mismatches,
                  "state": seal_state},
        "fit_report_digest": sealed_report.get("digest"),
        "fit_report_rerun_digest": rerun_report.get("digest"),
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
            "the clean-interpreter fit evaluation does not reproduce the "
            "sealed report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(f"[r41] clean validation written: {run_dir / CLEAN_ENV_NAME}",
          flush=True)
    return payload



# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each step performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R41 driver: the nonstrange pilot fit "
                    "(TASK-20260920-r4-run-r41-nonstrange-pilot-fit)")
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
        "--fit", action="store_true",
        help="the fit evaluation (fit_report.json), then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate4_r41.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the fit evaluation from a "
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
        bool(args.fit), bool(args.gate), bool(args.seal),
        bool(args.clean_validate),
    ]
    if sum(modes) != 1:
        parser.error("exactly one of --pre-registration / --smoke / "
                     "--generate / --fit / --gate / --seal / "
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
        print(f"[r41] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(
            run_dir, _load_pre_registration(run_dir),
            replicas=args.replica, workers=int(args.workers))
        print(f"[r41] generation complete: {written}", flush=True)
        return 0
    if args.fit:
        report = run_fit_machinery(run_dir, smoke=False)
        _write_json(run_dir / FIT_REPORT_NAME, report)
        print(f"[r41] fit report written (digest {report['digest'][:16]}"
              "...)", flush=True)
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
