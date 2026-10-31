#!/usr/bin/env python3
"""R40 driver: the fixed-source alignment scan (Round 4, card 11).

Task ``TASK-20260920-r4-run-r40-fixed-source-alignment-scan`` (plan
section 10 run R40, line 761; the Round 4 card set, card 11; plan
section 23, lines 1821-1823, "Round 4 -- aligned resonance cascade").
R40 is the training-side shape evidence of Gate 4: the fixed-source
scan of the aligned resonance cascade over ten at-rest root points
(the baryonic (B, Q, S) = (1, 1, 0) and the mesonic (0, 0, 0) families
at 2, 3, 4, 6, and 8 GeV), T1 per point, scanning the alignment
parameters kappa_B, kappa_M, and b_dec on pre-registered one-dimensional
slices at the nominal point (1, 1, 1.0 GeV^-2), measuring longitudinal
momentum retention, multiplicity, p_T moments, and cascade depth
(plan line 766).  The run verifies the cascade kernels per source mass
before the p+p fit (R41, card 12) uses them -- the house pattern of R20
before R21/R22 and R31 before R32.  No fitting of any kind (plan
section 24): the scan values are pre-registered inputs, and R41 is the
round's fit.

Source grid (round 4 index item 2, the pre-registered minimal choice)
---------------------------------------------------------------------
Ten at-rest points, the baryonic (B, Q, S) = (1, 1, 0) and the mesonic
(0, 0, 0) families at 2, 3, 4, 6, and 8 GeV (point index p = 0..4
baryonic at 2/3/4/6/8, p = 5..9 mesonic at 2/3/4/6/8).  All ten are
kinematically relevant: every mass exceeds both decay thresholds
(baryonic 1.07324891 GeV, mesonic 0.2699532 GeV, the round 2 index
item 6 values).  The source four-momentum is the at-rest four-vector of
the realized parent mass; the production axis is the lab z-axis (index
item 4, the isolated-source convention).

Routing (the card-6 blend at the frozen source mass)
----------------------------------------------------
Every decay routes through :func:`na49_mixing.spectral_density.
route_source_decay` at the point mass M (the audited card-6 surface):

- M = 2 GeV (below the kernel boundary B = 2.680194110727528 GeV):
  the catalog route, zero routing uniforms; the sampled parent decays
  through the Round 2 ``ResonanceDecayer`` byte-unchanged;
- M = 3 GeV (the blend band [B, B + W) at W = 0.5 GeV): exactly one
  routing uniform on the spawned ``decay_channel`` child, taken before
  any record draw; the continuum route fires exactly when the draw
  ``u < w(3) = (3 - B)/W`` (= 0.6396117785449444, the card's 0.6396);
- M = 4, 6, 8 GeV (at or above B + W): the continuum route, zero
  routing uniforms.

The catalog route follows the R21/R22 flat record mixture (the
source-grid runner convention): the open set is the catalog records
with the source (B, Q, S) whose truncated spectral window
``[mass_range[0], min(mass_range[1], M)]`` is non-empty
(``mass_range[0] <= M``), restricted to the kernel-decayable records
(``stable_for_decay`` excluded: a stable record carries no kernel
decay and no high-mass parent, and the R40 scan measures decay
observables only -- the R21/R22 stable-identity events are not part of
this scan; recorded in the pre-registration).  One uniform on the
spawned ``decay_channel`` child draws the record uniformly; the card-3
``draw_mass`` realizes the parent mass inside the truncated window
(exactly two uniforms on the spawned ``decay_kinematics`` child; no
degenerate window occurs for the eligible records at the R40 masses --
asserted, never repaired).  Every realized catalog parent satisfies the
card-9 handoff predicate (its realized mass lies inside the record
support), so its remainder decays through the byte-unchanged kernel via
card 9's :func:`~na49_mixing.cascade_termination.handoff_decay`.

The continuum route realizes the single activated continuum family of
the source (B, Q, S) at the frozen source mass (no family-mass redraw,
the builder's continuum-route convention) and decays it through the new
sequential cascade: the card-10 ``aligned_sequential`` machinery at the
setting's (kappa_B, kappa_M, b_dec), the plan section 10.2 line 678
"each high-mass parent" clause.

The decay walk (card 9's routing, card 10's attachments)
--------------------------------------------------------
The per-decay walk mirrors card 9's ``run_cascade`` routing order --
the hard depth guard (:class:`~na49_mixing.cascade_termination.
CascadeDepthGuardError` at depth 12), the catalog handoff, the
continuation through the open continuum pool, the direct-channel
termination, and the no-channel refusal -- with card 10's attached
per-vertex step (:func:`~na49_mixing.cascade_ablation.
_draw_attached_vertex` on the card-8 attached pool of
:func:`~na49_mixing.cascade_ablation._attach_form_factor`).  The walk
is implemented in this driver because the event assembly needs the
vertex siblings that the card-9/card-10 line traces drop (card 9's
caller-assembly clause: the decay products of the continuum and direct
vertices are the caller's event assembly, not line states).  The card-9
routing order, the card-10 attachments, and the frozen 1 channel + 2
kinematics per-vertex budget are preserved exactly; the walk adds zero
uniforms of its own.

The event assembly (the caller side of card 9's clause)
-------------------------------------------------------
An established daughter state of the assembly:

- a stable record contributes itself as a final state;
- an unstable catalog record decays through the Round 2 kernel's own
  ``decay_resonance`` on the same closed substream generators (the
  R23/R32 builder ``_emit`` convention: the caller-side resonance
  feed-down; the card-9 handoff predicate governs the cascade line's
  routing, the caller's assembly follows the kernel's own entry).  The
  kernel's non-resonance daughters are the final states.

A continuum-route line never contains a catalog state (the line
successors are continuum daughters), so the handoff fires only at a
catalog-route root.

Seeds (round 4 index item 9)
----------------------------
Master seed ``20260920`` (the round start date).  Raw replica seeds
``20301020 + r`` (r = 0..3; the block 20301020-20301023, disjoint from
R23's 20300913-20300920, the R32 production and confirmation blocks,
and the audit block 20301040 + k).  Per (setting s, point p, replica
r < 4) the event index is ``300_000_000 + s*10_000_000 + p*1_000_000 +
i`` with i < 25,000, and the kernel-layer streams derive as
``derive_substreams(20260920, r, event_index)`` on the closed
``decay_channel`` / ``decay_kinematics`` labels (the round 2 card 9 and
round 3 card 8 convention: the master seed with the replica folded into
the replica index, the event index encoding the setting and the point;
all 360 (setting, point, replica) stream spaces are pairwise disjoint).
The mixture-layer draws (the routing and the record draw, and the
card-3 mass draw) consume one-step spawned children of the two labeled
stream ids (``SeedSequence(stream_id).spawn(1)``, the R21/R22/R23
mixture-layer stream split); the cascade machinery and the Round 2
kernel keep the labeled ids themselves.  The hermetic smoke mode uses
the ``+7000`` block (raw smoke seed 20307920, kernel replica index
7000), disjoint from every production space and every previously used
block.

Observables (plan line 766)
---------------------------
Per setting and per point, pooled over the four replicas:

- **Longitudinal momentum retention**: the light-cone fraction of the
  realized source retained by each paper-window species,
  ``sum(E + p_z)/m_source`` with p_z the component along the lab
  z-axis (the inherited axis; the at-rest source carries
  ``sum(E + p_z) = m_source``, so the per-species fractions plus the
  off-window remainder sum to one, checked per decay within 1e-9), plus
  the card's baryon-line / central partition: the fraction retained by
  the baryon-line daughter (the B != 0 finals; exactly one final at the
  baryonic points, structurally zero at the mesonic points) and by the
  central system (the B = 0 finals).
- **Multiplicity**: the inclusive per-species counts per decay, the six
  paper-window species (pi+, pi-, K+, K-, p, pbar) by exact pdg
  identity (the R01 six-species convention; the neutral and
  neutral-strange finals carry no paper-window species).
- **p_T moments**: the two house conventions with the subscripts
  carried: ``<p_T>_x`` (the x_F-density convention, the density
  p_T/E * f) realized as the per-particle mean ``sum(p_T)/N``, and
  ``<p_T>_y`` (the y-density convention) realized as the
  energy-weighted mean ``sum(E p_T)/sum(E)`` (the engine's
  ``mean_pt_y`` convention, identical to the R21/R22 source-grid
  convention).
- **Cascade depth**: the terminal line state's depth of the card-9
  routing walk per decay (0 for a catalog handoff root: the kernel
  took the remainder at depth 0; the number of cascade vertices for a
  continuum line), histogrammed; the statistics are the pooled mean,
  the 95th percentile (the smallest histogram depth whose cumulative
  count reaches 0.95 x n_decays; no interpolation), the max, and the
  guard activations (a guard raises and stops the run, so the accepted
  count is zero by construction; plan section 10.6 line 757).
- A supplementary (recorded, non-binding) meson alignment moment, the
  mean ``p_z^2/|p|^2`` of the B = 0 finals, documents the even law's
  actual source-level response across the kappa_M slice.

Statistics (T1, plan section 5.1 line 253): 4 replicas x 25,000 decays
per point; 9 settings x 10 points x 100,000 decays = 9,000,000 decays
total.  Local generation is primary (the established local envelope:
9M decays ~ 1.5 h wall, chunked by (setting, point, replica)); the
pre-registered gateway fallback covers the envelope (one job per
chunk, ``--tasks 1``, single math thread, capacity-paced, chunk
outputs appended only to the new run dir).

Conservation: every decay step conserves (B, Q, S) exactly and the
daughter four-momenta sum to the parent's within 1e-9 (the card-4
contract, checked per vertex, never repaired); the driver adds the
per-event assembly checks (the finals' exact (B, Q, S) sum against the
source and the energy sum against the realized parent within
1e-9 GeV); a violation raises and stops the run.

Acceptance (the card's pre-registered criteria, the rescoped
named-residual-register precedent): an MC-driven check set whose
failing verdicts are named with sigma, a one-line diagnosis, and the
robust-statistics cross-reference; the 3.0-sigma band is never relaxed;
no fitting; no refit; no data selection.  The check set (v) is the
pre-registered monotonicity set:

- the kappa_B slice (settings 0, 1, 2, 3) on the baryon-line
  longitudinal retention: predicted increasing at the continuum-routed
  baryonic points (the blend-band point diluted by the catalog branch,
  recorded with its routing census), predicted flat at the
  catalog-routed 2 GeV point (the Round 2 kernel carries no alignment);
- the kappa_M slice (settings 4, 1, 5, 6 ordered by kappa_M = 0, 1, 2,
  4) on the mesonic central multiplicity: predicted flat at every
  mesonic point -- the even law is an angular law (plan section 10.4
  line 724): the alignment dispatch replaces only the cos(theta)
  quantile, the channel draws, daughter masses, and count observables
  carry no kappa_M dependence, so the machinery identity is the
  zero-response prediction; any significant trend is a named finding;
- the b_dec slice (settings 7, 1, 8 ordered by b_dec = 0.5, 1.0,
  2.0 GeV^-2) on <p_T>_x: predicted decreasing at the continuum-routed
  points for every species with a mean multiplicity of at least 0.01
  per decay (the card-8 W_dec factor suppresses large two-body
  momenta), predicted flat at the catalog-routed points.

Verdict statistic: the endpoint contrast of the slice,
``sigma = d * (x_last - x_first) / SE`` with d the predicted direction
(+1 increasing, -1 decreasing, 0 flat), ``SE`` the between-replica
standard error of the difference (std ddof=1 over the 4 replicas of
each endpoint, divided by sqrt(4), combined in quadrature; floored at
the engine's ``DETERMINISTIC_SE_FLOOR``), a direction verdict failing
at ``sigma < -3.0`` and a flat verdict failing at ``|sigma| > 3.0``.
Every failing verdict lands in the named-residual register.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests and the sealed
  R30-R32 seals first.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, 1 setting x 1 point x 1 replica
  x 2,000 decays at the pre-registered smoke seeds, and the smoke
  evaluation, into an explicit ``--run-dir`` (the default production
  run dir is refused).  Two identical smoke invocations produce
  byte-identical artifacts.  ``--gate`` / ``--seal`` /
  ``--clean-validate`` refuse smoke artifacts (the R31/8c convention).
- ``--generate [--setting S] [--point P] [--replica R]``: generate the
  requested (setting, point, replica) chunks (any subset; the default
  is every chunk), one record file per chunk, then stop.  Requires the
  pre-registration; the statistics flags must match the
  pre-registration in production mode.  A completed record that matches
  the pre-registered identity is reused unchanged, so the declared tier
  can be completed chunk by chunk (the pre-registered gateway fallback
  runs one job per chunk).
- ``--evaluate``: run the observable evaluation over the chunk records,
  write ``scan_report.json``, then stop.  Requires all 360 production
  records.  A check-set verdict failure is recorded and named; the step
  still exits 0 (the run seals with the named-residual register; a
  failing verdict is a Gate 4 finding, never a refit).
- ``--gate``: write ``gate4_r40.json`` (evidence only), then stop.
  Requires the scan report.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the evaluation from a
  clean interpreter over the sealed artifacts with a byte-identical
  report, re-run the pre-registered probe chunk (setting 1, point 0,
  replica 0) in memory against its sealed record, write
  ``clean_environment_commands.json``, then stop.

``scan_report.json`` and ``gate4_r40.json`` carry no wall-clock fields:
their bytes are a pure function of the sealed artifacts and the
committed code, which is what the clean-environment validation
re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic in every mode: the committed catalog, the committed continuum
config, and the sealed archives only; no SPS table, no event
generation outside ``--generate`` / ``--smoke``.
"""

from __future__ import annotations

import argparse
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
        f"staged_r40_alignment_scan: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from na49_mixing.cascade_ablation import (  # noqa: E402
    ABLATION_VARIANTS,
    AblationPolicy,
    _attach_form_factor,
    _draw_attached_vertex,
    exchange_root_state,
)
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
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.resonance_widths import draw_mass  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    BlendSpec,
    families_from_json,
    route_source_decay,
)
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260920-r4-run-r40-fixed-source-
# alignment-scan; the source grid, scan grid, statistics, seeds, and
# observables are verbatim from the card and the round 4 index items 2,
# 3, 4, and 9).
# ---------------------------------------------------------------------------

RUN_ID = "R40"
CARD = "TASK-20260920-r4-run-r40-fixed-source-alignment-scan"
RUN_NAME = "r40_fixed_source_alignment_scan_2026-09-21"
DEFAULT_RUN_DIR = "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21"
GATE_RECORD_NAME = "gate4_r40.json"
SCAN_REPORT_NAME = "scan_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "setting_{}_point_{}_replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "dev/mixing2006/config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r40_alignment_scan.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
ROUTING_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
KERNEL_REL = "dev/mixing2006/na49_mixing/resonance_decay.py"
R32_DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"

#: The card chain this run sits on: the Gate 3 closure audits (cards
#: 1-3, sealed before this run starts) and the cascade implementation
#: cards (4-10, landed and byte-stable).  The card-1 rework card (1r)
#: is recorded in the pre-registration's obligation block at its
#: execution-time status; the run pre-registers against the closed
#: audits per the card.
CARD_CHAIN = (
    "TASK-20260920-r4-audit-transition-routing-formulation",
    "TASK-20260920-r4-audit-strange-sector-continuum-families",
    "TASK-20260920-r4-audit-builder-blend-path-routing-weight",
    "TASK-20260920-r4-sequential-color-singlet-resonance-decay",
    "TASK-20260920-r4-inherited-production-axes",
    "TASK-20260920-r4-baryonic-alignment-kernel",
    "TASK-20260920-r4-mesonic-alignment-kernel",
    "TASK-20260920-r4-hadronic-decay-form-factor",
    "TASK-20260920-r4-termination-depth-tests",
    "TASK-20260920-r4-isotropic-aligned-ablation-runner",
)

#: The frozen substrate digests (byte-unchanged inputs; verified before
#: any step).  The cascade surfaces are cards 4-10; the Round 2 kernel,
#: the builder blend path, and the card-6 routing are the inherited
#: frozen substrate; the engine, catalog, continuum config, and R32
#: driver close the list.
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
}

#: The sealed archives consumed read-only (their seals verify; the R32
#: spread is the model-uncertainty carry of obligation 4).
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
)

#: The Round 4 master seed (the round start date; the round 2 master
#: 20260913 and the round 3 master 20260919 precedents).
MASTER_SEED = 20260920
SEED_BASE = 20260920
SEED_OFFSET = 40_000
SEED_STRIDE = 100
RAW_SEED_RULE = (
    "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r per "
    "replica r < 4 (the block 20301020-20301023); the block is disjoint "
    "from every previously used raw block (R23 20300913-20300920; the "
    "R32 production 20301519-20301526 / 20301619-20301626 / "
    "20301719-20301726 and confirmation 20301819-20301834 / "
    "20301919-20301934 / 20302019-20302034 blocks; the audit "
    "confirmation block 20301040 + k; the R32 smoke block "
    "20305919 + 100*v)"
)
KERNEL_SEED_RULE = (
    "derive_substreams(20260920, r, 300_000_000 + s*10_000_000 + "
    "p*1_000_000 + i) per (setting s in 0..8, point p in 0..9, replica "
    "r < 4, event i < 25,000) on the closed decay_channel/"
    "decay_kinematics labels (the round 2 card 9 and round 3 card 8 "
    "convention: the master seed with the replica folded into the "
    "replica index, the event index encoding the setting and the "
    "point); all 360 (setting, point, replica) stream spaces are "
    "pairwise disjoint"
)
EVENT_INDEX_RULE = "300_000_000 + s*10_000_000 + p*1_000_000 + i, i < 25,000"
EVENT_INDEX_BASE = 300_000_000
EVENT_INDEX_SETTING_STRIDE = 10_000_000
EVENT_INDEX_POINT_STRIDE = 1_000_000

#: The hermetic smoke mode (driver validation; never sealed): 1 setting
#: (the nominal s = 1) x 1 point (p = 0) x 1 replica x 2,000 decays at
#: the pre-registered smoke seeds.  The +7000 block (raw smoke seed
#: 20307920, kernel replica index 7000) is disjoint from every
#: production space (replica indices 0..3) and from every previously
#: used block (the R32 smoke block is +5000; the R32 confirmation
#: blocks are +900; the audit block is 20301040 + k).
SMOKE_SETTING = 1
SMOKE_POINT = 0
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 7000
SMOKE_RAW_SEED = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260920 + 40000 + 7000 = 20307920 (the +7000 "
    "smoke block, one replica); the smoke kernel layer draws at "
    "derive_substreams(20260920, 7000, event_index) on the production "
    "event-index partition of (setting 1, point 0), disjoint from every "
    "production space (replica indices 0..3), from the R32 +5000 smoke "
    "block, from the R32 confirmation blocks, and from every previously "
    "used block"
)

#: The clean-environment validation's pre-registered probe chunk: a full
#: (setting, point, replica) chunk re-run from a clean interpreter,
#: compared against its sealed record.
PROBE_SETTING = 1
PROBE_POINT = 0
PROBE_REPLICA = 0

#: T1 statistics (plan section 5.1 line 253): 4 replicas x 25,000
#: decays per point; 9 settings x 10 points x 100,000 decays =
#: 9,000,000 decays total.
N_SETTINGS = 9
N_POINTS = 10
T1_REPLICAS = 4
T1_DECAYS_PER_REPLICA = 25_000
T1_LABEL = (
    "T1: 4 replicas x 25,000 decays per point (plan section 5.1, "
    "line 253); 9 settings x 10 points x 100,000 decays = 9,000,000 "
    "decays total"
)
TOTAL_DECAYS = N_SETTINGS * N_POINTS * T1_REPLICAS * T1_DECAYS_PER_REPLICA

#: The source grid (round 4 index item 2): ten at-rest points, the
#: baryonic (1, 1, 0) and mesonic (0, 0, 0) families at 2, 3, 4, 6,
#: and 8 GeV; the point index p = 0..4 is baryonic at 2/3/4/6/8 and
#: p = 5..9 mesonic at 2/3/4/6/8 (the card's listing order).  The
#: decay thresholds are the round 2 index item 6 values (every point
#: mass exceeds both, so all ten are kinematically relevant).
DECAY_THRESHOLD_BARYONIC_GEV = 1.07324891
DECAY_THRESHOLD_MESONIC_GEV = 0.2699532
POINTS: tuple[dict, ...] = (
    {"point": 0, "family": "baryonic", "qn": (1, 1, 0), "mass_gev": 2.0},
    {"point": 1, "family": "baryonic", "qn": (1, 1, 0), "mass_gev": 3.0},
    {"point": 2, "family": "baryonic", "qn": (1, 1, 0), "mass_gev": 4.0},
    {"point": 3, "family": "baryonic", "qn": (1, 1, 0), "mass_gev": 6.0},
    {"point": 4, "family": "baryonic", "qn": (1, 1, 0), "mass_gev": 8.0},
    {"point": 5, "family": "mesonic", "qn": (0, 0, 0), "mass_gev": 2.0},
    {"point": 6, "family": "mesonic", "qn": (0, 0, 0), "mass_gev": 3.0},
    {"point": 7, "family": "mesonic", "qn": (0, 0, 0), "mass_gev": 4.0},
    {"point": 8, "family": "mesonic", "qn": (0, 0, 0), "mass_gev": 6.0},
    {"point": 9, "family": "mesonic", "qn": (0, 0, 0), "mass_gev": 8.0},
)

#: The scan grid (round 4 index item 3): nine distinct settings,
#: one-dimensional slices at the nominal point (kappa_B, kappa_M,
#: b_dec) = (1, 1, 1.0 GeV^-2); kappa_B in {0, 1, 2, 4} (0 the
#: isotropic baryonic kernel, the ablation baseline), kappa_M in
#: {0, 1, 2, 4} (0 the isotropic mesonic kernel), b_dec in
#: {0.5, 1.0, 2.0 GeV^-2}; the three slices share the nominal point
#: (setting 1).  Pre-registered inputs, not fit parameters (R41 is the
#: round's fit).
NOMINAL_POINT = (1.0, 1.0, 1.0)
SETTINGS: tuple[dict, ...] = (
    {"setting": 0, "kappa_b": 0.0, "kappa_m": 1.0, "b_dec": 1.0,
     "slice": "kappa_B", "label": "kappa_B = 0 (isotropic baryonic kernel)"},
    {"setting": 1, "kappa_b": 1.0, "kappa_m": 1.0, "b_dec": 1.0,
     "slice": "nominal", "label": "the nominal point (1, 1, 1.0 GeV^-2)"},
    {"setting": 2, "kappa_b": 2.0, "kappa_m": 1.0, "b_dec": 1.0,
     "slice": "kappa_B", "label": "kappa_B = 2"},
    {"setting": 3, "kappa_b": 4.0, "kappa_m": 1.0, "b_dec": 1.0,
     "slice": "kappa_B", "label": "kappa_B = 4"},
    {"setting": 4, "kappa_b": 1.0, "kappa_m": 0.0, "b_dec": 1.0,
     "slice": "kappa_M", "label": "kappa_M = 0 (isotropic mesonic kernel)"},
    {"setting": 5, "kappa_b": 1.0, "kappa_m": 2.0, "b_dec": 1.0,
     "slice": "kappa_M", "label": "kappa_M = 2"},
    {"setting": 6, "kappa_b": 1.0, "kappa_m": 4.0, "b_dec": 1.0,
     "slice": "kappa_M", "label": "kappa_M = 4"},
    {"setting": 7, "kappa_b": 1.0, "kappa_m": 1.0, "b_dec": 0.5,
     "slice": "b_dec", "label": "b_dec = 0.5 GeV^-2"},
    {"setting": 8, "kappa_b": 1.0, "kappa_m": 1.0, "b_dec": 2.0,
     "slice": "b_dec", "label": "b_dec = 2.0 GeV^-2"},
)

#: The scan-slice check orders (acceptance (v)): each slice's settings
#: ordered by the scanned parameter, with the predicted direction of
#: the named response.
KAPPA_B_SLICE = (0, 1, 2, 3)
KAPPA_M_SLICE = (4, 1, 5, 6)  # ordered by kappa_M = 0, 1, 2, 4
B_DEC_SLICE = (7, 1, 8)       # ordered by b_dec = 0.5, 1.0, 2.0

#: The six paper-window species (the R01 six-species set) by exact pdg
#: identity; the neutral and neutral-strange finals carry no
#: paper-window species.
SPECIES: tuple[tuple[str, int], ...] = (
    ("pi_plus", 211),
    ("pi_minus", -211),
    ("k_plus", 321),
    ("k_minus", -321),
    ("proton", 2212),
    ("antiproton", -2212),
)
SPECIES_KEYS: tuple[str, ...] = tuple(name for name, _ in SPECIES)

#: The routing regimes of the point masses (the card-6 blend at the
#: frozen source mass).  w(3) is the card's 0.6396 at full precision.
BLEND_WEIGHT_AT_3 = (3.0 - BLEND_BOUNDARY) / BLEND_RAMP_WIDTH
ROUTE_REGIMES: dict[float, str] = {
    2.0: "catalog (below the kernel boundary B)",
    3.0: (
        "blend band [B, B + W) at w = (3 - B)/W = "
        f"{BLEND_WEIGHT_AT_3!r} (the card's 0.6396)"
    ),
    4.0: "continuum (at or above the ramp end B + W)",
    6.0: "continuum (at or above the ramp end B + W)",
    8.0: "continuum (at or above the ramp end B + W)",
}

#: The per-vertex draw budget of the card-4 cascade (1 channel + 2
#: kinematics uniforms), frozen through cards 6-8; the card-3 mass draw
#: consumes exactly two uniforms per non-degenerate catalog-route
#: realization; the routing consumes one uniform inside the blend band
#: and zero off-band (the card-6 surface).
DEGENERATE_WINDOW_TOL = 1e-12
EVENT_CLOSURE_TOL = 1e-9
#: The minimum per-species mean multiplicity for the b_dec <p_T>_x
#: response check (species below it are recorded as skipped, not
#: findings).
PT_CHECK_MIN_MEAN_MULTIPLICITY = 0.01

#: The obligation 4 model-uncertainty carry (the sealed R32
#: cross-variant spread), verbatim from the card; the term applies to
#: the 3 GeV blend-band points (inside the transition window); the
#: 2 GeV catalog points and the 4, 6, 8 GeV continuum points are
#: off-band and the term is recorded as not applicable to them.
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}
OBLIGATION4_APPLIES_TO = "3 GeV blend-band points (w = 0.6396)"

#: The R32 robust-statistics cross-reference of the named-residual
#: register: the between-replica SE convention (std ddof=1 over the 4
#: replicas, divided by sqrt(4)) floored at the engine's deterministic
#: SE floor; the 3.0-sigma band is never relaxed (the R32 acceptance
#: rescope precedent).
SIGMA_BAND = 3.0
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)
SE_CONVENTION = (
    "the between-replica standard error of each endpoint (std ddof=1 "
    "over the 4 T1 replicas, divided by sqrt(4)), combined in "
    "quadrature for the difference and floored at the engine's "
    f"DETERMINISTIC_SE_FLOOR = {SE_FLOOR!r}; the 3.0-sigma band is "
    "never relaxed (the acceptance rescope precedent of "
    "TASK-20260918-r3-acceptance-rescope-card8a-8b)"
)

STAGE_LABEL = "r40_fixed_source_alignment_scan"
CHUNK_WORKERS_DEFAULT = 4
#: The pre-registered gateway fallback (local generation is primary):
#: if any replica chunk exceeds 30 min wall, the remaining chunks run
#: on the cluster gateway (host cl, workers c0801-c0809 x 32 task
#: slots; one job per chunk, --tasks 1, single math thread,
#: capacity-paced; chunk outputs appended only to the new run dir).
GATEWAY_FALLBACK_WALL_LIMIT_S = 30 * 60
GATEWAY_FALLBACK_RULE = (
    "local generation is primary; if any replica chunk exceeds 30 min "
    "wall, the remaining chunks run on the cluster gateway (host cl, "
    "workers c0801-c0809 x 32 task slots, one job per chunk, --tasks 1, "
    "single math thread (NA49_CLUSTER_GW_TASKS forces thread counts to "
    "1), no local process pool inside the chunk command, submission "
    "paced by the capacity check; chunk outputs are appended only to "
    "the new run dir)"
)


# ---------------------------------------------------------------------------
# Small deterministic JSON / hash helpers
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as UTF-8 JSON (indent 2, a trailing newline);
    identical inputs produce identical bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)


def _git_info() -> dict:
    """The git state of both trees (informational; the standing
    pre-existing dirt is never staged by this card)."""
    info: dict[str, Any] = {}
    for name, base in (("outer", MONO_ROOT), ("nested", NESTED_ROOT)):
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(base),
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=str(base),
                capture_output=True, text=True,
            ).stdout
            info[name] = {
                "head": head,
                "n_dirty_entries": len(
                    [line for line in status.splitlines() if line.strip()]
                ),
                "dirty_note": (
                    "the standing pre-existing dirt is never staged by "
                    "this card (the manifest pre_existing_dirt list)"
                ),
            }
        except (OSError, subprocess.SubprocessError) as exc:
            info[name] = {"error": repr(exc)[:200]}
    return info


def environment_fingerprint() -> dict:
    """The environment fingerprint of the run (no secrets; the module
    versions that enter the generation)."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "executable": Path(sys.executable).name,
    }


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(
        NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
    )


def _load_families() -> list:
    return families_from_json(
        str(NESTED_ROOT / "config" / "v2_singlet"
            / "continuum_resonances.json")
    )


def substrate_digests() -> dict:
    """The on-disk digests of the frozen substrate (the pre-registration
    and provenance inputs)."""
    return {
        name: _sha256_file(MONO_ROOT / rel)
        for name, (rel, _want) in FROZEN_SUBSTRATE.items()
    }


def verify_frozen_substrate() -> dict:
    """Verify every frozen substrate digest (a mismatch stops the run
    before any step; the frozen surfaces are byte-unchanged inputs)."""
    problems: list[str] = []
    actual = substrate_digests()
    for name, (rel, want) in FROZEN_SUBSTRATE.items():
        got = actual[name]
        if got != want:
            problems.append(
                f"{name}: {rel} sha256 {got!r} != the pre-registered "
                f"{want!r} (a frozen surface drifted; the run is "
                "invalidated)"
            )
    if problems:
        _refuse("; ".join(problems))
    return {
        "verified": True,
        "digests": {name: {"path": rel, "sha256": want}
                    for name, (rel, want) in FROZEN_SUBSTRATE.items()},
    }


def verify_sealed_dir(rel: str) -> dict:
    """Verify one sealed run dir's ``SHA256SUMS.txt`` (read-only).  The
    entries carry monorepo-root-relative paths (the R21/R31/R32
    convention) or run-dir-relative paths (the R30 convention); both
    resolve."""
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
        entries += 1
        if not payload.is_file() or _sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(
            f"sealed run dir {rel}: {len(mismatches)} of {entries} "
            f"{SHA256SUMS_NAME} entries do not re-hash correctly (first: "
            f"{mismatches[0]!r}); the sealed archive must be unmodified"
        )
    return {"path": rel, "entries": int(entries), "verified": True}


def verify_sealed_inputs() -> list[dict]:
    """The sealed R30-R32 archive states (read-only inputs of the run)."""
    return [verify_sealed_dir(rel) for rel in SEALED_RUN_DIRS]


def raw_replica_seed(replica: int) -> int:
    """The raw replica seed of replica *r* (the M8 rule +100 block)."""
    return SEED_BASE + SEED_OFFSET + SEED_STRIDE + int(replica)


def kernel_replica_slot(replica: int, smoke: bool) -> int:
    """The kernel-layer replica slot: the replica *r* in production
    (the round 3 card 8 convention), the +7000 smoke block in smoke
    mode (the R32 smoke convention)."""
    return SMOKE_KERNEL_REPLICA_SLOT if smoke else int(replica)


def event_index(setting: int, point: int, event: int) -> int:
    """The pre-registered event index of one decay."""
    return (
        EVENT_INDEX_BASE
        + int(setting) * EVENT_INDEX_SETTING_STRIDE
        + int(point) * EVENT_INDEX_POINT_STRIDE
        + int(event)
    )


def point_regime(mass: float) -> str:
    """The routing regime of a point mass under the card-6 blend."""
    if float(mass) < BLEND_BOUNDARY:
        return "catalog"
    if float(mass) >= BLEND_BOUNDARY + BLEND_RAMP_WIDTH:
        return "continuum"
    return "blend_band"


def blend_weight_at(mass: float) -> float:
    """The card-6 ramp weight ``w(M)`` at a point mass (BlendSpec().weight)."""
    return BlendSpec().weight(float(mass))


def active_family_for(fams, qn):
    """The single activated continuum family carrying *qn* (the source
    families are baryonic (1, 1, 0) and mesonic (0, 0, 0); the R32
    builder's one-active-family convention)."""
    wanted = tuple(int(v) for v in qn)
    actives = [
        family for family in fams
        if bool(family.active) and tuple(int(v) for v in family.qn) == wanted
    ]
    if len(actives) != 1:
        _refuse(
            "the committed continuum configuration carries "
            f"{len(actives)} activated families for the source quantum "
            f"numbers {wanted}; the routing needs exactly one"
        )
    return actives[0]


def eligible_catalog_records(catalog: HadronCatalog, qn, mass: float) -> list:
    """The R40 catalog-route record set at the point mass: the catalog
    records with the source (B, Q, S) whose truncated spectral window
    ``[mass_range[0], min(mass_range[1], M)]`` is non-empty
    (``mass_range[0] <= M``), restricted to the kernel-decayable
    records (``stable_for_decay`` excluded: a stable record carries no
    kernel decay and no high-mass parent; the R40 catalog route
    measures kernel decays only -- the R21/R22 stable-identity events
    are not part of this scan).  The card-3 mass draw realizes every
    drawn record inside its own support, so every realized parent is a
    card-9 catalog handoff state (checked at draw time)."""
    out = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != tuple(
                int(v) for v in qn):
            continue
        if record.stable_for_decay:
            continue
        if float(record.mass_range[0]) <= float(mass):
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# The per-decay generation step (pre-registered in the pre-registration)
# ---------------------------------------------------------------------------
class _CountingRng:
    """A transparent uniform-counting wrapper around one generator.

    Delegates every ``random()`` call unchanged (the draw values are
    the wrapped generator's) and counts the calls, so the per-chunk
    draw census is exact while the consumed uniforms are bit-identical
    to the unwrapped generators.  Duck-types the kernel's draw sites.
    """

    __slots__ = ("_rng", "count")

    def __init__(self, rng) -> None:
        self._rng = rng
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return self._rng.random()


_DAUGHTER_MAP_CACHE: dict[int, dict[int, float]] = {}
_BQS_CACHE: dict[int, tuple[int, int, int]] = {}


def _daughter_pole_mass(catalog: HadronCatalog, pdg_id: int) -> float:
    """Pole mass of an established daughter (the Round 2 kernel's
    daughter-mass sources)."""
    try:
        return float(catalog.get(int(pdg_id)).pole_mass)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            raise KeyError(
                f"daughter pdg_id {int(pdg_id)} is neither a catalog "
                "record nor in the vendored PDG table"
            ) from None
        return float(entry.mass)


def _daughter_masses_map(catalog: HadronCatalog, record) -> dict[int, float]:
    """Pole masses of every channel daughter of *record* (the card-3
    width law's daughter-mass map; cached per record)."""
    cached = _DAUGHTER_MAP_CACHE.get(int(record.pdg_id))
    if cached is not None:
        return cached
    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter not in masses:
                masses[daughter] = _daughter_pole_mass(catalog, daughter)
    _DAUGHTER_MAP_CACHE[int(record.pdg_id)] = masses
    return masses


def build_exchange_event(
    catalog: HadronCatalog,
    fams,
    blend: BlendSpec,
    qn,
    mass: float,
    rng_record,
    rng_mass,
) -> tuple[dict, str, dict]:
    """One realized R40 source at the frozen point (qn, mass): the
    card-6 routing decision at the source mass (one uniform inside the
    blend band, zero off-band, on the spawned ``decay_channel`` child),
    the catalog route's record draw (one further uniform on the same
    spawned child, uniform over the kernel-decayable open set) and
    card-3 mass draw (exactly two uniforms on the spawned
    ``decay_kinematics`` child; no degenerate window occurs for the
    eligible records at the R40 masses -- asserted), or the continuum
    route's continuum family state at the frozen source mass (the
    builder's no-family-mass-redraw convention).

    Returns ``(exchange_event, route, diagnostics)``; the diagnostics
    carry the realized catalog record id and realized mass when the
    catalog route fired."""
    route = route_source_decay(
        tuple(int(v) for v in qn), float(mass), blend, catalog, fams,
        rng_record,
    )
    diagnostics: dict[str, Any] = {
        "route": route,
        "blend_weight": float(blend.weight(float(mass))),
        "record_pdg": None,
        "record_name": None,
        "realized_mass_gev": float(mass),
    }
    if route == "catalog":
        records = eligible_catalog_records(catalog, qn, mass)
        if not records:
            _refuse(
                "the catalog route at quantum numbers "
                f"{tuple(int(v) for v in qn)} and mass {mass!r} GeV has "
                "no kernel-decayable open record (the eligible set is "
                "empty); no fallback route exists"
            )
        index = min(
            int(rng_record.random() * len(records)), len(records) - 1
        )
        record = records[int(index)]
        lo = float(record.mass_range[0])
        hi = min(float(record.mass_range[1]), float(mass))
        if not hi - lo > DEGENERATE_WINDOW_TOL:
            _refuse(
                f"the eligible record {record.name} (pdg {record.pdg_id}) "
                f"carries a degenerate truncated window {(lo, hi)!r} at "
                f"the point mass {mass!r} GeV; the pre-registered "
                "non-degenerate assertion failed"
            )
        dmap = _daughter_masses_map(catalog, record)
        drawn = draw_mass((lo, hi), record, dmap, rng_mass)
        if drawn is None:
            _refuse(
                f"the eligible record {record.name} (pdg {record.pdg_id}) "
                f"has no line-shape mass in the window {(lo, hi)!r} at "
                f"the point mass {mass!r} GeV; no fallback mass is "
                "implemented"
            )
        realized = float(drawn)
        exchange = {
            "topology": STAGE_LABEL,
            "momentum": (realized, 0.0, 0.0, 0.0),
            "family_id": f"catalog:{int(record.pdg_id)}",
            "mass": realized,
            "qn": tuple(int(v) for v in qn),
            "axis": (0.0, 0.0, 1.0),
            "ancestry": (STAGE_LABEL,),
        }
        diagnostics["record_pdg"] = int(record.pdg_id)
        diagnostics["record_name"] = str(record.name)
        diagnostics["realized_mass_gev"] = realized
        return exchange, route, diagnostics
    family = active_family_for(fams, qn)
    exchange = {
        "topology": STAGE_LABEL,
        "momentum": (float(mass), 0.0, 0.0, 0.0),
        "family_id": str(family.family_id),
        "mass": float(mass),
        "qn": tuple(int(v) for v in qn),
        "axis": (0.0, 0.0, 1.0),
        "ancestry": (STAGE_LABEL,),
    }
    return exchange, route, diagnostics


def _state_pdg(state) -> int | None:
    """The catalog pdg behind a state's family id, or ``None`` for a
    continuum state."""
    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        return int(family_id[len(CATALOG_FAMILY_PREFIX):])
    return None


def _emit_assembly_state(
    catalog: HadronCatalog,
    finals: list[tuple[int, tuple[float, float, float, float]]],
    state,
    rng_channel,
    rng_kinematics,
) -> None:
    """The event assembly of one terminal established daughter state
    (card 9's caller-assembly clause): a stable record contributes
    itself as a final; an unstable catalog record decays through the
    Round 2 kernel's own ``decay_resonance`` on the same closed streams
    with the card-9 binding pattern (the R23/R32 builder ``_emit``
    convention: the caller-side resonance feed-down; the kernel's decay
    logic, tables, and conservation checks run byte-unchanged)."""
    pdg = _state_pdg(state)
    if pdg is None:
        raise ValueError(
            "r40 assembly: a continuum daughter state reached the event "
            f"assembly (family_id {state.family_id!r}); the cascade line "
            "walks continuum states and emits only established daughters"
        )
    record = catalog.get(pdg)
    p4 = tuple(float(x) for x in state.momentum)
    if record.stable_for_decay or pdg in catalog.stable_states():
        finals.append((pdg, p4))
        return
    decayer = ResonanceDecayer(catalog, 0, 0, 0)  # pure: no random draws
    decayer._rng_channel = rng_channel
    decayer._rng_kinematics = rng_kinematics
    vertices = decayer.decay_resonance(pdg, p4)
    for vertex in vertices:
        for daughter in vertex.daughters:
            if not daughter.is_resonance:
                finals.append((
                    int(daughter.pdg_id),
                    tuple(float(x) for x in daughter.four_momentum),
                ))


def decay_scan_step(
    catalog: HadronCatalog,
    fams,
    exchange_event: dict,
    policy: AblationPolicy,
    rng_channel,
    rng_kinematics,
) -> tuple[list, list[tuple[int, tuple[float, float, float, float]]], str]:
    """One decay of the R40 scan: card 9's routing order with card 10's
    attached per-vertex step, capturing the full event assembly.

    At every line state, in order: the hard depth guard
    (:class:`~na49_mixing.cascade_termination.CascadeDepthGuardError`
    propagates and stops the run); the catalog handoff (card 9's
    :func:`~na49_mixing.cascade_termination.handoff_decay`, the
    byte-unchanged kernel delegation, whose non-resonance daughters are
    the finals); the continuation through one attached vertex over the
    open continuum pool (the continuum daughter continues the line, the
    established sibling is emitted into the assembly); the
    direct-channel termination through one attached vertex over the
    direct pool (both daughters are emitted); the no-channel refusal
    (``ValueError``).  Returns ``(line, finals, mode)`` with the line
    root first and the finals as ``(pdg, (E, px, py, pz))`` tuples.

    The per-vertex budget is card 4's frozen 1 channel + 2 kinematics
    uniforms on the closed labels (the card-10 attachments consume
    none); the handoff and the assembly remainders consume the Round 2
    kernel's own budget on the same streams."""
    root = exchange_root_state(exchange_event)
    line: list = [root]
    finals: list[tuple[int, tuple[float, float, float, float]]] = []
    current = root
    while True:
        if int(current.depth) > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"r40 scan: the line state of family "
                f"{current.family_id!r} at mass {float(current.mass)!r} "
                f"GeV carries depth {int(current.depth)}, exceeding "
                f"MAX_CASCADE_DEPTH = {MAX_CASCADE_DEPTH} (plan section "
                "10.6, line 757); the run stops"
            )
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(
                current, catalog, rng_channel, rng_kinematics)
            for vertex in vertices:
                for daughter in vertex.daughters:
                    if not daughter.is_resonance:
                        finals.append((
                            int(daughter.pdg_id),
                            tuple(
                                float(x) for x in daughter.four_momentum
                            ),
                        ))
            return line, finals, "catalog_handoff"
        channels = enumerate_channels(current, catalog, fams)
        continuum = [
            channel for channel in channels
            if channel.kind == KIND_CONTINUUM
        ]
        pool = continuum if continuum else [
            channel for channel in channels
            if channel.kind in (KIND_ESTABLISHED_HADRON,
                                KIND_ESTABLISHED_RESONANCE)
        ]
        if not pool:
            raise ValueError(
                f"r40 scan: the line state of family "
                f"{current.family_id!r} at mass "
                f"{float(current.mass)!r} GeV carries no open continuum "
                "channel and no valid direct hadronic channel, so "
                "neither termination rule of plan section 10.6 applies "
                "(checked, never repaired)"
            )
        attached = _attach_form_factor(pool, current, policy.b_dec)
        _channel, (daughter0, daughter1) = _draw_attached_vertex(
            current, attached, catalog, fams,
            policy.kappa_b, policy.kappa_m,
            rng_channel, rng_kinematics,
        )
        if continuum:
            successor = next(
                daughter for daughter in (daughter0, daughter1)
                if not str(daughter.family_id).startswith(
                    CATALOG_FAMILY_PREFIX)
            )
            sibling = daughter1 if successor is daughter0 else daughter0
            _emit_assembly_state(
                catalog, finals, sibling, rng_channel, rng_kinematics)
            line.append(successor)
            current = successor
            continue
        _emit_assembly_state(
            catalog, finals, daughter0, rng_channel, rng_kinematics)
        _emit_assembly_state(
            catalog, finals, daughter1, rng_channel, rng_kinematics)
        return line, finals, "direct_channel"


# ---------------------------------------------------------------------------
# The per-chunk accumulator and generation
# ---------------------------------------------------------------------------
_DEPTH_BINS = MAX_CASCADE_DEPTH + 2  # depths 0..MAX+1 inclusive


def _pdg_bqs(catalog: HadronCatalog, pdg: int) -> tuple[int, int, int]:
    """The (B, Q, S) of a final-state pdg (the catalog record's, else
    the vendored quantum-number table's)."""
    try:
        record = catalog.get(int(pdg))
        return (int(record.B), int(record.Q), int(record.S))
    except KeyError:
        from na49_mixing.quantum_numbers import quantum_numbers

        qn = quantum_numbers(int(pdg))
        return (int(qn.B), int(qn.Q), int(qn.S))


class ChunkAccumulator:
    """The per-(setting, point, replica) accumulator: exact integer
    counts, float sums in generation order, the depth histograms, the
    route census, the draw-budget census, the rolling digest, and the
    per-event conservation guards."""

    def __init__(self, catalog: HadronCatalog):
        self.catalog = catalog
        self.n_decays = 0
        self.n_catalog = 0
        self.n_continuum = 0
        self.n_band_catalog = 0
        self.n_band_continuum = 0
        self.mass_sum_catalog = 0.0
        self.mass_sum_continuum = 0.0
        self.depth_hist_all = [0] * _DEPTH_BINS
        self.depth_hist_catalog = [0] * _DEPTH_BINS
        self.depth_hist_continuum = [0] * _DEPTH_BINS
        self.species = {
            name: {"n": 0, "sum_pt": 0.0, "sum_e": 0.0,
                   "sum_ept": 0.0, "sum_lc": 0.0}
            for name in SPECIES_KEYS
        }
        self.baryon_line_lc = 0.0
        self.central_lc = 0.0
        self.baryon_finals = 0
        self.central_finals = 0
        self.moment_n = 0
        self.moment_sum = 0.0
        self.worst_energy_closure = 0.0
        self.worst_lightcone_identity = 0.0
        self.spawned_record_uniforms = 0
        self.spawned_mass_uniforms = 0
        self.labeled_channel_uniforms = 0
        self.labeled_kinematics_uniforms = 0
        self.digest = hashlib.sha256()

    def add_decay(
        self, index: int, route: str, diagnostics: dict, exchange: dict,
        line: list, finals: list, mode: str,
    ) -> None:
        mass_src = float(exchange["mass"])
        energy_sum = 0.0
        qn_sum = [0, 0, 0]
        lc_sum = 0.0
        lc_baryon = 0.0
        lc_central = 0.0
        n_baryon = 0
        n_central = 0
        species_p = dict.fromkeys(SPECIES_KEYS, 0.0)
        species_n = dict.fromkeys(SPECIES_KEYS, 0)
        species_e = dict.fromkeys(SPECIES_KEYS, 0.0)
        species_ept = dict.fromkeys(SPECIES_KEYS, 0.0)
        species_lc = dict.fromkeys(SPECIES_KEYS, 0.0)
        canonical: list[str] = []
        for pdg, p4 in finals:
            e, px, py, pz = (float(x) for x in p4)
            if not all(math.isfinite(x) for x in p4):
                raise ValueError(
                    f"r40 scan: event {index} of the chunk carries a "
                    "non-finite final four-momentum component (the NaN "
                    "guard; checked, never repaired)"
                )
            bqs = _pdg_bqs(self.catalog, pdg)
            for k in range(3):
                qn_sum[k] += bqs[k]
            energy_sum += e
            lc = e + pz
            lc_sum += lc
            if bqs[0] != 0:
                lc_baryon += lc
                n_baryon += 1
            else:
                lc_central += lc
                n_central += 1
                p2 = px * px + py * py + pz * pz
                if p2 > 0.0:
                    self.moment_n += 1
                    self.moment_sum += pz * pz / p2
            pt = math.hypot(px, py)
            for name, want in SPECIES:
                if pdg == want:
                    species_n[name] += 1
                    species_p[name] += pt
                    species_e[name] += e
                    species_ept[name] += e * pt
                    species_lc[name] += lc
                    canonical.append(
                        f"{pdg}:{e.hex()},{px.hex()},{py.hex()},{pz.hex()}"
                    )
                    break
            else:
                canonical.append(f"{pdg}:offwindow")
        if qn_sum != [int(v) for v in exchange["qn"]]:
            raise ValueError(
                f"r40 scan: event {index}'s finals sum to (B, Q, S) = "
                f"{tuple(qn_sum)} but the source carries "
                f"{tuple(int(v) for v in exchange['qn'])}; conservation "
                "is checked, never repaired (the run stops)"
            )
        closure = abs(energy_sum - mass_src)
        identity = abs(lc_sum - mass_src)
        if not (closure <= EVENT_CLOSURE_TOL):
            raise ValueError(
                f"r40 scan: event {index}'s final energy sum "
                f"{energy_sum!r} misses the realized source mass "
                f"{mass_src!r} by {closure!r} GeV (beyond "
                f"{EVENT_CLOSURE_TOL}); conservation is checked, never "
                "repaired (the run stops)"
            )
        if not (identity <= EVENT_CLOSURE_TOL):
            raise ValueError(
                f"r40 scan: event {index}'s light-cone identity deviates "
                f"by {identity!r} GeV (beyond {EVENT_CLOSURE_TOL}); "
                "checked, never repaired"
            )
        self.worst_energy_closure = max(self.worst_energy_closure, closure)
        self.worst_lightcone_identity = max(
            self.worst_lightcone_identity, identity)
        depth = int(line[-1].depth)
        if not 0 <= depth < _DEPTH_BINS:
            raise ValueError(
                f"r40 scan: event {index}'s terminal line depth "
                f"{depth} is outside the histogram range (the depth "
                "guard fires before depth MAX_CASCADE_DEPTH + 2)"
            )
        self.depth_hist_all[depth] += 1
        if route == "catalog":
            self.depth_hist_catalog[depth] += 1
        else:
            self.depth_hist_continuum[depth] += 1
        blend_w = float(diagnostics["blend_weight"])
        if route == "catalog":
            self.n_catalog += 1
            self.mass_sum_catalog += mass_src
            if blend_w != 0.0:
                self.n_band_catalog += 1
        else:
            self.n_continuum += 1
            self.mass_sum_continuum += mass_src
            if blend_w != 1.0:
                self.n_band_continuum += 1
        for name in SPECIES_KEYS:
            slot = self.species[name]
            slot["n"] += species_n[name]
            slot["sum_pt"] += species_p[name]
            slot["sum_e"] += species_e[name]
            slot["sum_ept"] += species_ept[name]
            slot["sum_lc"] += species_lc[name]
        self.baryon_line_lc += lc_baryon
        self.central_lc += lc_central
        self.baryon_finals += n_baryon
        self.central_finals += n_central
        payload = (
            f"{index}|{route}|{exchange['family_id']}|"
            f"{float(exchange['mass']).hex()}|{mode}|{depth}|"
            f"{len(finals)}|" + ",".join(canonical)
        )
        self.digest.update(payload.encode("utf-8"))
        self.digest.update(b"\n")
        self.n_decays += 1


# ---------------------------------------------------------------------------
# The generation context (module-level worker context; set once in the
# parent before the fork pool is created, the R01/R23/R32 convention)
# ---------------------------------------------------------------------------
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _generation_context(smoke: bool) -> dict:
    catalog = _load_catalog()
    return {
        "catalog": catalog,
        "fams": _load_families(),
        "blend": BlendSpec(),
        "smoke": bool(smoke),
    }


def generate_chunk(task: tuple) -> dict:
    """Generate one (setting, point, replica) chunk: 25,000 pre-registered
    decays (2,000 in smoke mode) with the exact per-decay draw sequence,
    returning the accumulator payload (no file writes; the caller owns
    the record)."""
    assert _WORKER_CONTEXT is not None
    setting, point, replica, n_decays = task
    context = _WORKER_CONTEXT
    catalog: HadronCatalog = context["catalog"]
    fams = context["fams"]
    blend: BlendSpec = context["blend"]
    smoke = bool(context["smoke"])
    setting_entry = SETTINGS[int(setting)]
    point_entry = POINTS[int(point)]
    qn = tuple(int(v) for v in point_entry["qn"])
    mass = float(point_entry["mass_gev"])
    policy = AblationPolicy(
        "aligned_sequential",
        float(setting_entry["kappa_b"]),
        float(setting_entry["kappa_m"]),
        float(setting_entry["b_dec"]),
    )
    replica_slot = kernel_replica_slot(replica, smoke)
    acc = ChunkAccumulator(catalog)
    for i in range(int(n_decays)):
        event = event_index(setting, point, i)
        subs = derive_substreams(MASTER_SEED, replica_slot, event)
        rng_record = _CountingRng(np.random.default_rng(
            np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
        ))
        rng_mass = _CountingRng(np.random.default_rng(
            np.random.SeedSequence(
                subs.stream_for("decay_kinematics")).spawn(1)[0]
        ))
        rng_channel = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_channel")))
        rng_kin = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_kinematics")))
        exchange, route, diagnostics = build_exchange_event(
            catalog, fams, blend, qn, mass, rng_record, rng_mass)
        line, finals, mode = decay_scan_step(
            catalog, fams, exchange, policy,
            rng_channel, rng_kin)
        acc.add_decay(i, route, diagnostics, exchange, line, finals, mode)
        acc.spawned_record_uniforms += rng_record.count
        acc.spawned_mass_uniforms += rng_mass.count
        acc.labeled_channel_uniforms += rng_channel.count
        acc.labeled_kinematics_uniforms += rng_kin.count
    return {
        "setting": int(setting),
        "point": int(point),
        "replica": int(replica),
        "n_decays": int(acc.n_decays),
        "route_census": {
            "n_catalog": int(acc.n_catalog),
            "n_continuum": int(acc.n_continuum),
            "n_band_catalog": int(acc.n_band_catalog),
            "n_band_continuum": int(acc.n_band_continuum),
            "mass_sum_catalog": float(acc.mass_sum_catalog),
            "mass_sum_continuum": float(acc.mass_sum_continuum),
        },
        "depth_histograms": {
            "all": [int(v) for v in acc.depth_hist_all],
            "catalog_route": [int(v) for v in acc.depth_hist_catalog],
            "continuum_route": [int(v) for v in acc.depth_hist_continuum],
        },
        "species_sums": {
            name: {
                "n": int(slot["n"]),
                "sum_pt": float(slot["sum_pt"]),
                "sum_e": float(slot["sum_e"]),
                "sum_ept": float(slot["sum_ept"]),
                "sum_lc": float(slot["sum_lc"]),
            }
            for name, slot in acc.species.items()
        },
        "partition_sums": {
            "baryon_line_lc": float(acc.baryon_line_lc),
            "central_lc": float(acc.central_lc),
            "baryon_finals": int(acc.baryon_finals),
            "central_finals": int(acc.central_finals),
        },
        "alignment_moment": {
            "n_mesons": int(acc.moment_n),
            "sum_z2_over_p2": float(acc.moment_sum),
        },
        "draw_budget": {
            "spawned_record_child_uniforms": int(acc.spawned_record_uniforms),
            "spawned_mass_child_uniforms": int(acc.spawned_mass_uniforms),
            "labeled_channel_uniforms": int(acc.labeled_channel_uniforms),
            "labeled_kinematics_uniforms": int(
                acc.labeled_kinematics_uniforms),
        },
        "conservation": {
            "events_checked": int(acc.n_decays),
            "qn_violations": 0,
            "worst_energy_closure_gev": float(acc.worst_energy_closure),
            "worst_lightcone_identity_gev": float(
                acc.worst_lightcone_identity),
        },
        "rolling_digest": acc.digest.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Mode: --pre-registration
# ---------------------------------------------------------------------------
def build_pre_registration(substrate: dict, seals: list) -> dict:
    """The pre-registration payload (the card's assertion contract,
    written before any generation; asserted against this driver by
    every later mode)."""
    points = []
    for entry in POINTS:
        mass = float(entry["mass_gev"])
        qn = tuple(int(v) for v in entry["qn"])
        points.append({
            "point": int(entry["point"]),
            "family": entry["family"],
            "qn": [int(v) for v in qn],
            "mass_gev": mass,
            "route_regime": point_regime(mass),
            "route_regime_note": ROUTE_REGIMES[mass],
            "blend_weight": blend_weight_at(mass),
            "at_rest": True,
            "axis": [0.0, 0.0, 1.0],
            "axis_convention": (
                "the lab z-axis (round 4 index item 4, the "
                "isolated-source convention)"
            ),
        })
    return {
        "schema": "na49_mixing.staged_r40_alignment_scan.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "plan_anchors": {
            "run": "plan section 10 run R40, line 761",
            "grid_clause": "plan line 763 (baryonic and mesonic sources at "
                           "2, 3, 4, 6, and 8 GeV where kinematically "
                           "relevant)",
            "statistics_clause": "plan line 764 (T1 per point)",
            "scan_clause": "plan line 765 (scan kappa_B, kappa_M, b_dec)",
            "observables_clause": "plan line 766",
            "nominal_point": (
                "plan section 10.4 line 732 and plan section 10.5 "
                "(b_dec = 1.0 GeV^-2)"
            ),
            "statistics_tier": "plan section 5.1 line 253",
            "termination_guard": "plan section 10.6 line 757",
        },
        "source_grid": {
            "index_item": 2,
            "points": points,
            "decay_thresholds_gev": {
                "baryonic": DECAY_THRESHOLD_BARYONIC_GEV,
                "mesonic": DECAY_THRESHOLD_MESONIC_GEV,
            },
            "kinematic_relevance": (
                "all ten points are kinematically relevant: every mass "
                "exceeds both decay thresholds (baryonic "
                f"{DECAY_THRESHOLD_BARYONIC_GEV} GeV, mesonic "
                f"{DECAY_THRESHOLD_MESONIC_GEV} GeV, the round 2 index "
                "item 6 values)"
            ),
        },
        "scan_grid": {
            "index_item": 3,
            "nominal_point": {
                "kappa_b": NOMINAL_POINT[0],
                "kappa_m": NOMINAL_POINT[1],
                "b_dec_per_gev_squared": NOMINAL_POINT[2],
            },
            "settings": [
                {
                    "setting": int(entry["setting"]),
                    "kappa_b": float(entry["kappa_b"]),
                    "kappa_m": float(entry["kappa_m"]),
                    "b_dec_per_gev_squared": float(entry["b_dec"]),
                    "slice": entry["slice"],
                    "label": entry["label"],
                }
                for entry in SETTINGS
            ],
            "slices": {
                "kappa_B": list(KAPPA_B_SLICE),
                "kappa_M": list(KAPPA_M_SLICE),
                "b_dec": list(B_DEC_SLICE),
            },
            "note": (
                "one-dimensional slices at the nominal point (kappa_B, "
                "kappa_M, b_dec) = (1, 1, 1.0 GeV^-2); kappa_B in "
                "{0, 1, 2, 4} (0 the isotropic baryonic kernel, the "
                "ablation baseline), kappa_M in {0, 1, 2, 4} (0 the "
                "isotropic mesonic kernel), b_dec in {0.5, 1.0, 2.0 "
                "GeV^-2}; nine distinct settings, the three slices "
                "sharing the nominal point (setting 1); the nominal "
                "point is the pre-registered starting value (plan "
                "section 10.4 line 732, constant kappa of order one; "
                "b_dec = 1.0 GeV^-2 the conventional hadronic vertex "
                "scale of section 10.5); these values are pre-registered "
                "inputs, not fit parameters (R41 is the round's fit)"
            ),
        },
        "statistics": {
            "tier": T1_LABEL,
            "replicas": T1_REPLICAS,
            "decays_per_replica": T1_DECAYS_PER_REPLICA,
            "decays_per_point": T1_REPLICAS * T1_DECAYS_PER_REPLICA,
            "settings": N_SETTINGS,
            "points": N_POINTS,
            "total_decays": TOTAL_DECAYS,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": RAW_SEED_RULE,
            "raw_replica_seeds": [
                raw_replica_seed(r) for r in range(T1_REPLICAS)
            ],
            "kernel_seed_rule": KERNEL_SEED_RULE,
            "event_index_rule": EVENT_INDEX_RULE,
            "stream_split": (
                "the mixture-layer draws (the routing uniform, the "
                "catalog record draw, and the card-3 mass draw) consume "
                "one-step spawned children of the two labeled stream ids "
                "(SeedSequence(stream_id).spawn(1), the R21/R22/R23 "
                "mixture-layer stream split); the cascade machinery and "
                "the Round 2 kernel keep the labeled ids themselves"
            ),
            "smoke": {
                "setting": SMOKE_SETTING,
                "point": SMOKE_POINT,
                "replicas": SMOKE_REPLICAS,
                "decays_per_replica": SMOKE_EVENTS,
                "raw_seed_rule": SMOKE_RAW_SEED_RULE,
                "raw_seed": SMOKE_RAW_SEED,
                "kernel_replica_slot": SMOKE_KERNEL_REPLICA_SLOT,
            },
            "disjointness": (
                "the R40 master seed 20260920 is disjoint from "
                "20260913 (R20-R23), 20260917 (R31), 20260918 (card 8c), "
                "and 20260919 (R32); the raw block 20301020-20301023 is "
                "disjoint from R23's 20300913-20300920, the R32 "
                "production and confirmation blocks, and the audit "
                "block 20301040 + k"
            ),
        },
        "routing": {
            "surface": (
                "na49_mixing/spectral_density.py (the card-6 blend, the "
                "audited surface of card 1)"
            ),
            "boundary_gev": BLEND_BOUNDARY,
            "ramp_width_gev": BLEND_RAMP_WIDTH,
            "weight_at_point_masses": {
                str(entry["mass_gev"]): blend_weight_at(
                    float(entry["mass_gev"]))
                for entry in POINTS
            },
            "uniforms": (
                "zero routing uniforms off-band (M < B and M >= B + W), "
                "exactly one inside the blend band, taken before any "
                "record draw (the card-6 surface)"
            ),
        },
        "catalog_route_mixture": {
            "lineage": (
                "the R21/R22 flat record mixture (the source-grid "
                "runner convention), restricted to the kernel-decayable "
                "records"
            ),
            "open_set_rule": (
                "the catalog records with the source (B, Q, S) whose "
                "truncated spectral window [mass_range[0], "
                "min(mass_range[1], M)] is non-empty (mass_range[0] <= "
                "M), with stable_for_decay records excluded: a stable "
                "record carries no kernel decay and no high-mass "
                "parent, and the R40 scan measures decay observables "
                "only (the R21/R22 stable-identity events are not part "
                "of this scan)"
            ),
            "record_draw": (
                "uniform over the eligible set (one uniform on the "
                "spawned decay_channel child)"
            ),
            "mass_rule": (
                "the card-3 spectral function restricted to the "
                "truncated window (resonance_widths.draw_mass; exactly "
                "two uniforms per draw on the spawned decay_kinematics "
                "child); no degenerate window occurs for the eligible "
                "records at the R40 masses (asserted, never repaired)"
            ),
            "handoff_guarantee": (
                "every realized catalog parent satisfies the card-9 "
                "handoff predicate (its realized mass lies inside the "
                "record support), so its remainder decays through the "
                "byte-unchanged Round 2 kernel via card 9's "
                "handoff_decay"
            ),
        },
        "continuum_route": {
            "family": (
                "the single activated continuum family of the source "
                "(B, Q, S), realized at the frozen source mass (no "
                "family-mass redraw, the builder's continuum-route "
                "convention)"
            ),
            "decay": (
                "the card-10 aligned_sequential machinery at the "
                "setting's (kappa_B, kappa_M, b_dec): plan section 10.2 "
                "line 678, each high-mass parent decays through the new "
                "sequential cascade"
            ),
        },
        "decay_walk": {
            "routing": (
                "card 9's run_cascade routing order (the hard depth "
                "guard, the catalog handoff, the continuum "
                "continuation, the direct-channel termination, the "
                "no-channel refusal), implemented in the driver because "
                "the event assembly needs the vertex siblings the "
                "card-9/card-10 line traces drop (card 9's "
                "caller-assembly clause)"
            ),
            "per_vertex_step": (
                "card 10's attached vertex (the card-8 attached pool "
                "and the card-6/7 alignment dispatch), the frozen 1 "
                "channel + 2 kinematics per-vertex budget; the "
                "attachments consume zero uniforms"
            ),
            "event_assembly": (
                "an established daughter is a final when its record is "
                "stable, else it decays through the Round 2 kernel's "
                "own decay_resonance on the same closed streams (the "
                "R23/R32 builder _emit convention: the caller-side "
                "resonance feed-down)"
            ),
            "depth_convention": (
                "the cascade depth of a decay is the terminal line "
                "state's depth of the card-9 routing walk (0 for a "
                "catalog handoff root: the kernel took the remainder at "
                "depth 0; the number of cascade vertices for a "
                "continuum line)"
            ),
        },
        "observables": {
            "retention": (
                "the light-cone fraction of the realized source "
                "retained by each paper-window species, sum(E + p_z)/"
                "m_source with p_z the component along the lab z-axis "
                "(the at-rest source carries sum(E + p_z) = m_source, "
                "so the per-species fractions plus the off-window "
                "remainder sum to one, checked per decay within 1e-9), "
                "plus the baryon-line / central partition (the B != 0 "
                "finals and the B = 0 finals)"
            ),
            "multiplicity": (
                "the inclusive per-species counts per decay, the six "
                "paper-window species (pi+, pi-, K+, K-, p, pbar) by "
                "exact pdg identity (the R01 six-species convention)"
            ),
            "species_pdgs": {name: pdg for name, pdg in SPECIES},
            "pt_moments": {
                "mean_pt_x": (
                    "the x_F-density convention (the density p_T/E * "
                    "f): the per-particle mean sum(p_T)/N over the "
                    "species' particles"
                ),
                "mean_pt_y": (
                    "the y-density convention: the energy-weighted mean "
                    "sum(E p_T)/sum(E) (the engine's mean_pt_y "
                    "convention, identical to the R21/R22 source-grid "
                    "convention)"
                ),
            },
            "depth": (
                "the pooled depth histogram per (setting, point); the "
                "statistics are the mean, the 95th percentile (the "
                "smallest histogram depth whose cumulative count "
                "reaches 0.95 x n_decays; no interpolation), the max, "
                "and the guard activations (zero by construction: a "
                "guard raises and stops the run)"
            ),
            "supplementary": (
                "the meson alignment moment, the mean p_z^2/|p|^2 of "
                "the B = 0 finals (recorded, non-binding; documents the "
                "even law's source-level response across the kappa_M "
                "slice)"
            ),
        },
        "conservation": {
            "per_vertex": (
                "every decay step conserves (B, Q, S) exactly and the "
                "daughter four-momenta sum to the parent's within 1e-9 "
                "(the card-4 contract, checked per vertex, never "
                "repaired); a violation raises and stops the run"
            ),
            "event_level": (
                "the driver checks the finals' exact (B, Q, S) sum "
                "against the source and the energy sum against the "
                "realized parent within 1e-9 GeV per decay; a violation "
                "raises and stops the run"
            ),
            "nan_guard": (
                "every recorded observable is finite (acceptance (iv)); "
                "a non-finite component raises and stops the run"
            ),
        },
        "determinism": {
            "rolling_digest": (
                "each chunk record carries a sha256 rolling digest over "
                "the per-decay canonical payloads (the event index, the "
                "route, the realized root family and mass, the "
                "termination mode, the terminal depth, and every "
                "final's pdg and exact hex four-momentum); a "
                "per-(setting, point, replica) re-run from a clean "
                "interpreter reproduces it (the clean-environment "
                "contract)"
            ),
            "probe_chunk": {
                "setting": PROBE_SETTING,
                "point": PROBE_POINT,
                "replica": PROBE_REPLICA,
                "rule": (
                    "clean-validate re-runs this full chunk from a "
                    "clean interpreter and compares the rolling digest, "
                    "the route census, and the depth histogram against "
                    "the sealed record"
                ),
            },
            "report_note": (
                "scan_report.json and gate4_r40.json carry no "
                "wall-clock fields: their bytes are a pure function of "
                "the sealed artifacts and the committed code"
            ),
        },
        "acceptance": {
            "precedent": (
                "the acceptance precedent of "
                "TASK-20260918-r3-acceptance-rescope-card8a-8b: an "
                "MC-driven check set with residuals seals with a "
                "named-residual register, not a raw zero-failure target"
            ),
            "i_conservation": (
                "exact (B, Q, S) per vertex and four-momentum sum "
                "within 1e-9 over the 9,000,000 decays; zero violations"
            ),
            "ii_determinism": (
                "the clean-environment re-run reproduces the "
                "per-(setting, point, replica) report hashes (the "
                "rolling digests; probed on the pre-registered probe "
                "chunk and sealed through the seal)"
            ),
            "iii_depth_guard": (
                "zero CascadeDepthGuardError activations across the "
                "full run (plan section 10.6 line 757); the depth "
                "statistics (mean, p95, max) are recorded per setting "
                "and point"
            ),
            "iv_no_nan": "all recorded observables are finite",
            "v_kernel_verification": {
                "kappa_B": {
                    "slice": list(KAPPA_B_SLICE),
                    "observable": (
                        "baryon_line_fraction (the baryon-line "
                        "longitudinal retention)"
                    ),
                    "points_predicted_increasing": [1, 2, 3, 4],
                    "points_predicted_flat": [0],
                    "reason": (
                        "the exp(kappa_B cos(theta)) law biases the "
                        "baryonic daughter forward along the inherited "
                        "axis (plan section 10.4 line 721); the Round 2 "
                        "kernel carries no alignment, so the "
                        "catalog-routed point is flat"
                    ),
                },
                "kappa_M": {
                    "slice": list(KAPPA_M_SLICE),
                    "observable": (
                        "central multiplicity (the mean B = 0 finals "
                        "count per decay) at the mesonic points"
                    ),
                    "points_predicted_flat": [5, 6, 7, 8, 9],
                    "reason": (
                        "the even law is an angular law (plan section "
                        "10.4 line 724): the alignment dispatch "
                        "replaces only the cos(theta) quantile, the "
                        "channel draws, daughter masses, and count "
                        "observables carry no kappa_M dependence, so "
                        "the machinery identity is the zero-response "
                        "prediction; any significant trend is a named "
                        "finding"
                    ),
                },
                "b_dec": {
                    "slice": list(B_DEC_SLICE),
                    "observable": "mean_pt_x per paper-window species",
                    "points_predicted_decreasing": [1, 2, 3, 4, 6, 7, 8,
                                                    9],
                    "points_predicted_flat": [0, 5],
                    "min_mean_multiplicity": PT_CHECK_MIN_MEAN_MULTIPLICITY,
                    "reason": (
                        "the card-8 W_dec factor exp(-b_dec q_perp^2) "
                        "suppresses large two-body momenta, shifting the "
                        "channel mix toward smaller two-body momenta "
                        "(plan section 10.5); the Round 2 kernel's "
                        "vertices are un-attached, so the catalog-routed "
                        "point is flat"
                    ),
                },
                "verdict_statistic": (
                    "the endpoint contrast of the slice, sigma = d * "
                    "(x_last - x_first)/SE with d the predicted "
                    "direction (+1 increasing, -1 decreasing, 0 flat); "
                    "a direction verdict fails at sigma < -3.0 and a "
                    "flat verdict fails at |sigma| > 3.0"
                ),
                "se_convention": SE_CONVENTION,
                "register": (
                    "every failing verdict is named in the "
                    "named_residual_register with sigma, a one-line "
                    "diagnosis, and the robust-statistics "
                    "cross-reference; the 3.0-sigma band is never "
                    "relaxed; no fitting, no refit, no data selection"
                ),
            },
            "vi_seal_and_clean": (
                "the seal is intact and the clean-environment re-run "
                "reproduces the scan evaluation hashes"
            ),
            "vii_pins_and_freeze": (
                "the pin test register lands, the freeze re-pin is "
                "recorded, and the full nested suite is green"
            ),
        },
        "obligation_4_carry": {
            "obligation": 4,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "applies_to": OBLIGATION4_APPLIES_TO,
            "not_applicable_to": (
                "the 2 GeV catalog points and the 4, 6, 8 GeV continuum "
                "points are off-band and the term is recorded as not "
                "applicable to them"
            ),
            "audit_outcome": {
                "card_1": (
                    "RECOMMENDATION (sealed dev/00runs/"
                    "r4_audit_transition_routing_formulation_2026-09-20/"
                    "verdict.json): the rework card 1r "
                    "(TASK-20260920-r4-rework-transition-routing-"
                    "formulation) is spawned; this run pre-registers "
                    "against the closed audits per the card and carries "
                    "the original R32 spread until card 1r resolves"
                ),
                "card_2": (
                    "PASS (dev/00runs/r4_audit_strange_sector_continuum_"
                    "families_2026-09-20/verdict.json)"
                ),
                "card_3": (
                    "PASS (dev/00runs/r4_audit_builder_blend_path_"
                    "routing_weight_2026-09-20/verdict.json)"
                ),
            },
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "the scan values are pre-registered inputs, and R41 (card "
            "12) is the round's fit (plan section 24)"
        ),
        "gateway_fallback": GATEWAY_FALLBACK_RULE,
        "substrate": substrate,
        "sealed_run_dirs": seals,
        "statistics_tier": T1_LABEL,
        "total_decays": TOTAL_DECAYS,
        "written_before_generation": True,
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write ``pre_registration.json`` and ``provenance.json``, stop."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file() and any(run_dir.glob("setting_*.json")):
        _refuse(
            f"{pre_path} exists and chunk records are present; the "
            "pre-registration is immutable after generation has started "
            "(a change invalidates the run; re-seal under a new date)"
        )
    substrate = verify_frozen_substrate()
    seals = verify_sealed_inputs()
    pre = build_pre_registration(substrate, seals)
    _write_json(pre_path, pre)
    digests = substrate_digests()
    provenance = {
        "schema": "na49_mixing.staged_r40_alignment_scan.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": list(CARD_CHAIN),
        "driver": DRIVER_REL,
        "driver_sha256": {
            "nested": _sha256_file(
                NESTED_ROOT / "helpers" / "benchmarking"
                / "staged_r40_alignment_scan.py"),
            "outer": _sha256_file(MONO_ROOT / DRIVER_REL),
            "note": (
                "the dual-tree driver is byte-identical in both trees; "
                "the two entries are the on-disk digests"
            ),
        },
        "cascade_modules_sha256": {
            name: digests[name]
            for name in ("cascade_decay", "cascade_axes",
                         "cascade_alignment", "cascade_form_factor",
                         "cascade_termination", "cascade_ablation")
        },
        "round2_kernel_sha256": digests["round2_kernel"],
        "builder_blend_path_sha256": digests["builder_blend_path"],
        "card6_routing_sha256": digests["card6_routing"],
        "card2_engine_sha256": digests["card2_engine"],
        "catalog": {"path": CATALOG_REL, "sha256": digests["catalog"]},
        "continuum_config": {
            "path": FAMILIES_REL,
            "sha256": digests["continuum_config"],
        },
        "r32_driver_sha256": digests["r32_driver"],
        "sealed_run_dirs": seals,
        "kernel_boundary_gev": float(BLEND_BOUNDARY),
        "ramp_width_gev": float(BLEND_RAMP_WIDTH),
        "blend_weight_at_3gev": BLEND_WEIGHT_AT_3,
        "master_seed": MASTER_SEED,
        "raw_replica_seeds": [
            raw_replica_seed(r) for r in range(T1_REPLICAS)
        ],
        "kernel_seed_rule": KERNEL_SEED_RULE,
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_generation": True,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        "staged_r40_alignment_scan: pre-registration written to "
        f"{pre_path} (9 settings x 10 points x 4 replicas x 25,000 "
        f"decays = {TOTAL_DECAYS}, master seed {MASTER_SEED}, nominal "
        f"point {NOMINAL_POINT}, w(3) = {BLEND_WEIGHT_AT_3!r}); nothing "
        "generated",
        flush=True,
    )


def _load_pre_registration(run_dir: Path, smoke: bool = False) -> dict:
    name = f"{SMOKE_PREFIX}pre_registration.json" if smoke \
        else PREREGISTRATION_NAME
    path = Path(run_dir) / name
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the pre-registration "
            "(run --pre-registration first, or --smoke for the smoke "
            "artifacts)"
        )
    return _read_json(path)


def _assert_pre_registration(run_dir: Path, smoke: bool = False) -> dict:
    """Assert the written pre-registration against this driver (a
    mismatch invalidates the run)."""
    pre = _load_pre_registration(run_dir, smoke)
    problems: list[str] = []
    stats = pre.get("statistics", {})
    if int(stats.get("replicas", -1)) != T1_REPLICAS:
        problems.append("statistics.replicas mismatch")
    if int(stats.get("decays_per_replica", -1)) != T1_DECAYS_PER_REPLICA:
        problems.append("statistics.decays_per_replica mismatch")
    if int(stats.get("settings", -1)) != N_SETTINGS:
        problems.append("statistics.settings mismatch")
    if int(stats.get("points", -1)) != N_POINTS:
        problems.append("statistics.points mismatch")
    if int(stats.get("total_decays", -1)) != TOTAL_DECAYS:
        problems.append("statistics.total_decays mismatch")
    if pre.get("source_grid", {}).get("index_item") != 2:
        problems.append("source grid index item mismatch")
    scan = pre.get("scan_grid", {})
    nominal = scan.get("nominal_point", {})
    if (
        float(nominal.get("kappa_b", -1.0)) != NOMINAL_POINT[0]
        or float(nominal.get("kappa_m", -1.0)) != NOMINAL_POINT[1]
        or float(nominal.get("b_dec_per_gev_squared", -1.0))
        != NOMINAL_POINT[2]
    ):
        problems.append("nominal point mismatch")
    if [
        (int(v.get("setting", -1)), float(v.get("kappa_b", -1.0)),
         float(v.get("kappa_m", -1.0)),
         float(v.get("b_dec_per_gev_squared", -1.0)))
        for v in scan.get("settings", [])
    ] != [
        (int(entry["setting"]), float(entry["kappa_b"]),
         float(entry["kappa_m"]), float(entry["b_dec"]))
        for entry in SETTINGS
    ]:
        problems.append("scan settings mismatch")
    points = pre.get("source_grid", {}).get("points", [])
    if [
        (int(v.get("point", -1)), tuple(v.get("qn", ())),
         float(v.get("mass_gev", -1.0)))
        for v in points
    ] != [
        (int(entry["point"]), tuple(int(x) for x in entry["qn"]),
         float(entry["mass_gev"]))
        for entry in POINTS
    ]:
        problems.append("source grid points mismatch")
    seed_map = pre.get("seed_map", {})
    if int(seed_map.get("master_seed", -1)) != MASTER_SEED:
        problems.append("master seed mismatch")
    if str(seed_map.get("raw_seed_rule", "")) != RAW_SEED_RULE:
        problems.append("raw seed rule mismatch")
    if str(seed_map.get("kernel_seed_rule", "")) != KERNEL_SEED_RULE:
        problems.append("kernel seed rule mismatch")
    if seed_map.get("raw_replica_seeds") != [
            raw_replica_seed(r) for r in range(T1_REPLICAS)]:
        problems.append("raw replica seed blocks mismatch")
    if pre.get("no_fitting") is not True:
        problems.append("no_fitting mismatch")
    if pre.get("optimizer_invoked") is not False:
        problems.append("optimizer_invoked mismatch")
    carry = pre.get("obligation_4_carry")
    if not isinstance(carry, dict) or dict(
            carry.get("r32_spread", {})) != dict(OBLIGATION4_R32_SPREAD):
        problems.append("obligation 4 R32 spread mismatch")
    if problems:
        raise RuntimeError(
            "staged_r40_alignment_scan: the written pre-registration "
            "does not match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return pre


# ---------------------------------------------------------------------------
# Mode: --generate
# ---------------------------------------------------------------------------
def record_name(setting: int, point: int, replica: int, smoke: bool) -> str:
    """The chunk-record file name (smoke records carry the smoke
    prefix; smoke artifacts are never sealed)."""
    base = RECORD_FMT.format(int(setting), int(point), int(replica))
    return f"{SMOKE_PREFIX}{base}" if smoke else base


def _record_identity_ok(
    stored: dict, setting: int, point: int, replica: int,
    n_decays: int, smoke: bool,
) -> bool:
    """The reuse predicate of a completed record (the gateway-fallback
    resume pattern): the chunk identity, the statistics, the mode, the
    policy, the source point, and the code identity must all match."""
    digests = substrate_digests()
    identity = stored.get("identity", {})
    policy = stored.get("policy", {})
    source = stored.get("source", {})
    setting_entry = SETTINGS[int(setting)]
    point_entry = POINTS[int(point)]
    return bool(
        int(stored.get("setting", -1)) == int(setting)
        and int(stored.get("point", -1)) == int(point)
        and int(stored.get("replica", -1)) == int(replica)
        and str(stored.get("mode", ""))
        == ("smoke" if smoke else "production")
        and int(stored.get("statistics", {}).get("n_decays", -1))
        == int(n_decays)
        and float(policy.get("kappa_b", -1.0))
        == float(setting_entry["kappa_b"])
        and float(policy.get("kappa_m", -1.0))
        == float(setting_entry["kappa_m"])
        and float(policy.get("b_dec_per_gev_squared", -1.0))
        == float(setting_entry["b_dec"])
        and str(policy.get("variant", "")) == "aligned_sequential"
        and tuple(int(v) for v in source.get("qn", ()))
        == tuple(int(v) for v in point_entry["qn"])
        and float(source.get("mass_gev", -1.0))
        == float(point_entry["mass_gev"])
        and identity.get("driver_sha256")
        == _sha256_file(Path(__file__).resolve())
        and identity.get("catalog_sha256") == digests["catalog"]
        and identity.get("cascade_modules_sha256") == {
            name: digests[name]
            for name in ("cascade_decay", "cascade_axes",
                         "cascade_alignment", "cascade_form_factor",
                         "cascade_termination", "cascade_ablation")
        }
        and isinstance(stored.get("species_sums"), dict)
        and isinstance(stored.get("depth_histograms"), dict)
        and isinstance(stored.get("route_census"), dict)
        and isinstance(stored.get("rolling_digest"), str)
        and isinstance(stored.get("draw_budget"), dict)
        and isinstance(stored.get("conservation"), dict)
    )


def _chunk_record(
    setting: int, point: int, replica: int, n_decays: int, payload: dict,
    smoke: bool,
) -> dict:
    """Assemble one per-(setting, point, replica) record from the
    chunk payload (the smoke records carry no wall-clock fields: two
    identical smoke invocations must produce byte-identical
    artifacts)."""
    setting_entry = SETTINGS[int(setting)]
    point_entry = POINTS[int(point)]
    mass = float(point_entry["mass_gev"])
    digests = substrate_digests()
    route = payload["route_census"]
    n_catalog = int(route["n_catalog"])
    n_continuum = int(route["n_continuum"])
    n_total = n_catalog + n_continuum
    return {
        "schema": "na49_mixing.staged_r40_alignment_scan.record/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "stage_label": STAGE_LABEL,
        "setting": int(setting),
        "point": int(point),
        "replica": int(replica),
        "policy": {
            "variant": "aligned_sequential",
            "kappa_b": float(setting_entry["kappa_b"]),
            "kappa_m": float(setting_entry["kappa_m"]),
            "b_dec_per_gev_squared": float(setting_entry["b_dec"]),
        },
        "source": {
            "family": point_entry["family"],
            "qn": [int(v) for v in point_entry["qn"]],
            "mass_gev": mass,
            "route_regime": point_regime(mass),
            "blend_weight": blend_weight_at(mass),
        },
        "statistics": {
            "n_decays": int(payload["n_decays"]),
            "decays_per_replica_declared": int(n_decays),
            "replicas": 1,
            "tier_note": (
                "one (setting, point, replica) chunk; the declared tier "
                "pools the four production replicas per (setting, point)"
                if not smoke else "the hermetic smoke statistics"
            ),
        },
        "seeds": {
            "master_seed": MASTER_SEED,
            "raw_replica_seed": (
                SMOKE_RAW_SEED if smoke
                else raw_replica_seed(replica)
            ),
            "kernel_replica_slot": kernel_replica_slot(replica, smoke),
            "event_index_rule": EVENT_INDEX_RULE,
            "event_index_first": event_index(setting, point, 0),
            "event_index_last": event_index(
                setting, point, int(payload["n_decays"]) - 1),
            "kernel_seed_rule": KERNEL_SEED_RULE,
        },
        "route_census": {
            "n_catalog": n_catalog,
            "n_continuum": n_continuum,
            "n_band_catalog": int(route["n_band_catalog"]),
            "n_band_continuum": int(route["n_band_continuum"]),
            "fraction_catalog": (
                float(n_catalog) / float(n_total) if n_total > 0 else 0.0
            ),
            "fraction_continuum": (
                float(n_continuum) / float(n_total) if n_total > 0 else 0.0
            ),
            "mean_realized_mass_catalog_gev": (
                float(route["mass_sum_catalog"]) / float(n_catalog)
                if n_catalog > 0 else 0.0
            ),
            "mean_realized_mass_continuum_gev": (
                float(route["mass_sum_continuum"]) / float(n_continuum)
                if n_continuum > 0 else 0.0
            ),
        },
        "depth_histograms": {
            "all": [int(v) for v in payload["depth_histograms"]["all"]],
            "catalog_route": [
                int(v) for v in payload["depth_histograms"]["catalog_route"]
            ],
            "continuum_route": [
                int(v)
                for v in payload["depth_histograms"]["continuum_route"]
            ],
        },
        "species_sums": {
            name: {
                "n": int(slot["n"]),
                "sum_pt": float(slot["sum_pt"]),
                "sum_e": float(slot["sum_e"]),
                "sum_ept": float(slot["sum_ept"]),
                "sum_lc": float(slot["sum_lc"]),
            }
            for name, slot in payload["species_sums"].items()
        },
        "partition_sums": {
            "baryon_line_lc": float(
                payload["partition_sums"]["baryon_line_lc"]),
            "central_lc": float(payload["partition_sums"]["central_lc"]),
            "baryon_finals": int(payload["partition_sums"]["baryon_finals"]),
            "central_finals": int(payload["partition_sums"]["central_finals"]),
        },
        "alignment_moment": {
            "n_mesons": int(payload["alignment_moment"]["n_mesons"]),
            "sum_z2_over_p2": float(
                payload["alignment_moment"]["sum_z2_over_p2"]),
        },
        "draw_budget": {
            "spawned_record_child_uniforms": int(
                payload["draw_budget"]["spawned_record_child_uniforms"]),
            "spawned_mass_child_uniforms": int(
                payload["draw_budget"]["spawned_mass_child_uniforms"]),
            "labeled_channel_uniforms": int(
                payload["draw_budget"]["labeled_channel_uniforms"]),
            "labeled_kinematics_uniforms": int(
                payload["draw_budget"]["labeled_kinematics_uniforms"]),
            "note": (
                "the spawn-child census covers the routing/record and "
                "card-3 mass uniforms; the labeled census covers the "
                "cascade vertices (1 channel + 2 kinematics per vertex) "
                "and the Round 2 kernel's own draws inside the handoff "
                "delegations and the assembly remainders"
            ),
        },
        "conservation": {
            "events_checked": int(payload["conservation"]["events_checked"]),
            "qn_violations": int(payload["conservation"]["qn_violations"]),
            "worst_energy_closure_gev": float(
                payload["conservation"]["worst_energy_closure_gev"]),
            "worst_lightcone_identity_gev": float(
                payload["conservation"]["worst_lightcone_identity_gev"]),
            "convention": (
                "every decay step conserves (B, Q, S) exactly and the "
                "daughter four-momenta sum to the parent's within 1e-9 "
                "(the card-4 contract, checked per vertex, never "
                "repaired); the driver checks the finals' exact (B, Q, "
                "S) sum and the energy sum against the realized parent "
                "per decay; a violation raises and stops the run"
            ),
        },
        "rolling_digest": str(payload["rolling_digest"]),
        "identity": {
            "driver_sha256": _sha256_file(Path(__file__).resolve()),
            "catalog_sha256": digests["catalog"],
            "families_config_sha256": digests["continuum_config"],
            "cascade_modules_sha256": {
                name: digests[name]
                for name in ("cascade_decay", "cascade_axes",
                             "cascade_alignment", "cascade_form_factor",
                             "cascade_termination", "cascade_ablation")
            },
            "round2_kernel_sha256": digests["round2_kernel"],
            "card6_routing_sha256": digests["card6_routing"],
        },
        "reused": False,
    }


def _generate_one_chunk(
    run_dir: Path, setting: int, point: int, replica: int,
    n_decays: int, workers: int, context: dict, smoke: bool,
) -> dict:
    """One (setting, point, replica) chunk: reuse a matching completed
    record unchanged (the gateway-fallback resume pattern) or generate
    it (the decays are strictly sequential inside the chunk, the
    per-event stream order being the generation order)."""
    name = record_name(setting, point, replica, smoke)
    record_path = Path(run_dir) / name
    if record_path.is_file():
        try:
            stored = _read_json(record_path)
        except ValueError:
            stored = None
        if isinstance(stored, dict):
            try:
                matched = _record_identity_ok(
                    stored, setting, point, replica, n_decays, smoke)
            except (KeyError, TypeError, ValueError):
                matched = False
            if matched:
                stored["reused"] = True
                print(
                    f"[r40] setting {setting} point {point} replica "
                    f"{replica}: reusing completed record "
                    f"({int(stored['statistics']['n_decays'])} decays)",
                    flush=True,
                )
                return stored
    _set_worker_context(context)
    started = time.perf_counter()
    started_iso = _now_utc()
    task = (int(setting), int(point), int(replica), int(n_decays))
    if int(workers) > 1:
        # One chunk is one pool task; the chunk's decays are strictly
        # sequential inside the worker (the per-event stream order is
        # the generation order).
        parts = sorted(
            _map_chunks([task], int(workers), "chunk"),
            key=lambda item: item["setting"] * 1000 + item["point"],
        )
        payload = parts[0]
    else:
        payload = generate_chunk(task)
    wall = time.perf_counter() - started
    record = _chunk_record(setting, point, replica, n_decays, payload, smoke)
    if not smoke:
        record["wall_seconds"] = round(wall, 3)
        record["started_utc"] = started_iso
        record["finished_utc"] = _now_utc()
    _write_json(Path(run_dir) / record_name(setting, point, replica, smoke),
                record)
    return record


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(generate_chunk, tasks, chunksize=1):
            completed += 1
            yield result
            if completed % 4 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r40] {progress_label}: chunk {completed}/"
                    f"{len(tasks)} ({elapsed:.1f}s)",
                    flush=True,
                )


def run_generation(
    run_dir: Path, setting: int | None, point: int | None,
    replica: int | None, workers: int, statistics: dict,
) -> None:
    """Generate the requested (setting, point, replica) chunks; each
    record lands on disk before the next chunk starts."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    verify_frozen_substrate()
    digests = substrate_digests()
    recorded_digests = (pre.get("substrate") or {}).get("digests") or {}
    for name in ("catalog", "continuum_config"):
        if str(recorded_digests.get(name, {}).get("sha256")) != digests[name]:
            _refuse(
                f"the committed {name} sha256 differs from the "
                "pre-registered digest; the run is invalidated (re-"
                "register under a new date)"
            )
    declared = {
        "settings": N_SETTINGS,
        "points": N_POINTS,
        "replicas": T1_REPLICAS,
        "decays-per-replica": T1_DECAYS_PER_REPLICA,
        "master-seed": MASTER_SEED,
    }
    for key, value in declared.items():
        given = statistics.get(key)
        if given is not None and int(given) != int(value):
            _refuse(
                f"--{key} {given} does not match the pre-registered "
                f"{value}; the production statistics are fixed"
            )
    setting_list = (
        [int(setting)] if setting is not None else list(range(N_SETTINGS))
    )
    point_list = (
        [int(point)] if point is not None else list(range(N_POINTS))
    )
    replica_list = (
        [int(replica)] if replica is not None else list(range(T1_REPLICAS))
    )
    for value, bound, label in (
        (setting_list, N_SETTINGS, "setting"),
        (point_list, N_POINTS, "point"),
        (replica_list, T1_REPLICAS, "replica"),
    ):
        for item in value:
            if not 0 <= item < bound:
                _refuse(
                    f"--{label} {item} outside the pre-registered set "
                    f"0..{bound - 1}"
                )
    context = _generation_context(smoke=False)
    n_decays = T1_DECAYS_PER_REPLICA
    # The pending-chunk list: a completed record that matches the
    # pre-registered identity is reused unchanged (the gateway-fallback
    # resume pattern), the rest are batched over one fork pool (the
    # chunked-generation envelope; a single --setting/--point/--replica
    # subset runs through the same path).
    tasks = []
    reused = 0
    for s in setting_list:
        for p in point_list:
            for r in replica_list:
                record_path = run_dir / record_name(s, p, r, False)
                if record_path.is_file():
                    try:
                        stored = _read_json(record_path)
                    except ValueError:
                        stored = None
                    matched = False
                    if isinstance(stored, dict):
                        try:
                            matched = _record_identity_ok(
                                stored, s, p, r, n_decays, False)
                        except (KeyError, TypeError, ValueError):
                            matched = False
                    if matched:
                        reused += 1
                        print(
                            f"[r40] setting {s} point {p} replica {r}: "
                            "reusing completed record",
                            flush=True,
                        )
                        continue
                tasks.append((int(s), int(p), int(r), int(n_decays)))
    if tasks:
        _set_worker_context(context)
        completed = 0
        for payload in _map_chunks(
                tasks, int(workers), "production"):
            completed += 1
            s = int(payload["setting"])
            p = int(payload["point"])
            r = int(payload["replica"])
            record = _chunk_record(s, p, r, n_decays, payload, False)
            record["wall_seconds"] = None
            record["started_utc"] = None
            record["finished_utc"] = None
            _write_json(
                run_dir / record_name(s, p, r, False), record)
            census = record["route_census"]
            print(
                f"[r40] setting {s} point {p} replica {r}: "
                f"{record['statistics']['n_decays']} decays, routing "
                f"catalog {census['n_catalog']} / continuum "
                f"{census['n_continuum']} (fraction continuum "
                f"{census['fraction_continuum']:.6f})",
                flush=True,
            )
    print(
        f"[r40] generation step complete (run dir {run_dir}, mode "
        f"production; {len(tasks)} chunks generated, {reused} reused)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Mode: --smoke (the hermetic end-to-end driver validation)
# ---------------------------------------------------------------------------
def run_smoke(run_dir: Path, workers: int) -> None:
    """The hermetic smoke validation: a smoke pre-registration, 1
    setting x 1 point x 1 replica x 2,000 decays at the pre-registered
    smoke seeds, and the smoke evaluation.  Two identical smoke
    invocations produce byte-identical artifacts (no wall-clock fields
    in the smoke records)."""
    run_dir = Path(run_dir)
    if Path(run_dir).resolve() == (
            (MONO_ROOT / DEFAULT_RUN_DIR).resolve()):
        _refuse(
            "--smoke refuses the default production run dir "
            f"({DEFAULT_RUN_DIR}); pass an explicit --run-dir for the "
            "smoke artifacts (they are never sealed)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    substrate = verify_frozen_substrate()
    seals = verify_sealed_inputs()
    pre = build_pre_registration(substrate, seals)
    pre["mode"] = "smoke"
    pre["smoke"] = {
        "setting": SMOKE_SETTING,
        "point": SMOKE_POINT,
        "replicas": SMOKE_REPLICAS,
        "decays_per_replica": SMOKE_EVENTS,
        "raw_seed_rule": SMOKE_RAW_SEED_RULE,
        "raw_seed": SMOKE_RAW_SEED,
        "kernel_replica_slot": SMOKE_KERNEL_REPLICA_SLOT,
        "determinism_note": (
            "the smoke records carry no wall-clock fields; two "
            "identical smoke invocations produce byte-identical "
            "artifacts"
        ),
    }
    _write_json(run_dir / f"{SMOKE_PREFIX}pre_registration.json", pre)
    context = _generation_context(smoke=True)
    record = _generate_one_chunk(
        run_dir, SMOKE_SETTING, SMOKE_POINT, 0, SMOKE_EVENTS,
        int(workers), context, True,
    )
    census = record["route_census"]
    print(
        f"[r40-smoke] setting {SMOKE_SETTING} point {SMOKE_POINT}: "
        f"{record['statistics']['n_decays']} decays, routing catalog "
        f"{census['n_catalog']} / continuum {census['n_continuum']}",
        flush=True,
    )
    report = run_evaluate(run_dir, smoke=True)
    checks = report.get("check_set", {})
    print(
        "[r40-smoke] smoke validation complete (check set "
        f"{checks.get('status', 'n/a')}; smoke artifacts are never "
        "sealed)",
        flush=True,
    )


def _refuse_smoke_artifacts(run_dir: Path) -> None:
    """The R31/8c convention: the gate / seal / clean-validate modes
    refuse smoke artifacts."""
    run_dir = Path(run_dir)
    smoke_files = sorted(
        path.name for path in run_dir.iterdir()
        if path.is_file() and path.name.startswith(SMOKE_PREFIX)
    )
    if smoke_files:
        _refuse(
            f"{run_dir} carries smoke artifacts (first: "
            f"{smoke_files[0]!r}); the gate / seal / clean-validate "
            "modes refuse smoke artifacts (the R31/8c convention)"
        )


# ---------------------------------------------------------------------------
# Evaluation helpers (pure functions of the sealed record payloads)
# ---------------------------------------------------------------------------
def depth_stats(hist: list, n_decays: int) -> dict:
    """The pooled depth statistics of one histogram: the mean, the 95th
    percentile (the smallest histogram depth whose cumulative count
    reaches 0.95 x n_decays; no interpolation), the max, and the guard
    activations (zero by construction: a guard raises and stops the
    run before any record is written)."""
    hist = [int(v) for v in hist]
    total = sum(hist)
    if total != int(n_decays):
        raise ValueError(
            "depth_stats: the histogram total "
            f"{total} != n_decays {n_decays}"
        )
    if total <= 0:
        raise ValueError("depth_stats: the histogram is empty")
    mean = (
        sum(depth * count for depth, count in enumerate(hist)) / float(total)
    )
    threshold = 0.95 * total
    cumulative = 0
    p95 = None
    for depth, count in enumerate(hist):
        cumulative += count
        if cumulative >= threshold:
            p95 = depth
            break
    max_depth = max(depth for depth, count in enumerate(hist) if count > 0)
    return {
        "mean": mean,
        "p95": int(p95),
        "max": int(max_depth),
        "guard_activations": 0,
        "n_decays": int(total),
        "convention": (
            "the terminal line state's depth of the card-9 routing walk "
            "per decay (0 for a catalog handoff root, the number of "
            "cascade vertices for a continuum line); the 95th "
            "percentile is the smallest histogram depth whose "
            "cumulative count reaches 0.95 x n_decays (no "
            "interpolation); the guard activations are zero by "
            "construction (a guard raises and stops the run)"
        ),
    }


def _mean_se(values: list) -> tuple:
    """The mean and the between-replica SE (std ddof=1 over the
    available values, divided by sqrt(n)) of one per-replica observable
    list; ``(None, None)`` when the observable is undefined
    everywhere."""
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return None, None
    mean = math.fsum(defined) / len(defined)
    if len(defined) < 2:
        return mean, None
    variance = math.fsum(
        (v - mean) ** 2 for v in defined) / (len(defined) - 1)
    if variance < 0.0:
        variance = 0.0
    se = math.sqrt(variance) / math.sqrt(len(defined))
    return mean, se


def _point_species_tables(records: list, mass: float) -> dict:
    """The pooled per-species observable tables of one (setting,
    point): the exact pooled sums over the replicas (in replica order)
    with the between-replica standard errors (std ddof=1 over the four
    per-replica values, divided by sqrt(4))."""
    n_decays = sum(
        int(record["statistics"]["n_decays"]) for record in records)
    if n_decays <= 0:
        raise ValueError("the point tables need a positive decay count")
    denom = float(n_decays) * float(mass)

    def per_replica(extract) -> list:
        return [extract(record) for record in records]

    pooled_species = {}
    for name in SPECIES_KEYS:
        n = sum(
            int(record["species_sums"][name]["n"]) for record in records)
        sum_pt = math.fsum(
            float(record["species_sums"][name]["sum_pt"])
            for record in records)
        sum_e = math.fsum(
            float(record["species_sums"][name]["sum_e"])
            for record in records)
        sum_ept = math.fsum(
            float(record["species_sums"][name]["sum_ept"])
            for record in records)
        sum_lc = math.fsum(
            float(record["species_sums"][name]["sum_lc"])
            for record in records)
        pooled_pt_x = (sum_pt / n) if n > 0 else None
        pooled_pt_y = (sum_ept / sum_e) if sum_e > 0.0 else None
        pooled_lc = (sum_lc / denom) if denom > 0.0 else None
        mult_r = per_replica(
            lambda record, name=name: (
                float(record["species_sums"][name]["n"])
                / float(record["statistics"]["n_decays"])))
        ptx_r = per_replica(
            lambda record, name=name: (
                float(record["species_sums"][name]["sum_pt"])
                / int(record["species_sums"][name]["n"])
                if int(record["species_sums"][name]["n"]) > 0 else None))
        pty_r = per_replica(
            lambda record, name=name: (
                float(record["species_sums"][name]["sum_ept"])
                / float(record["species_sums"][name]["sum_e"])
                if float(record["species_sums"][name]["sum_e"]) > 0.0
                else None))
        lc_r = per_replica(
            lambda record, name=name, mass=mass: (
                float(record["species_sums"][name]["sum_lc"])
                / (float(record["statistics"]["n_decays"]) * float(mass))))
        mult_mean, mult_se = _mean_se(mult_r)
        ptx_mean, ptx_se = _mean_se(ptx_r)
        pty_mean, pty_se = _mean_se(pty_r)
        lc_mean, lc_se = _mean_se(lc_r)
        table = {
            "n": int(n),
            "mean_multiplicity": mult_mean,
            "multiplicity_se": mult_se,
            "mean_pt_x": ptx_mean,
            "mean_pt_x_se": ptx_se,
            "pooled_mean_pt_x": pooled_pt_x,
            "mean_pt_y": pty_mean,
            "mean_pt_y_se": pty_se,
            "pooled_mean_pt_y": pooled_pt_y,
            "lightcone_fraction": lc_mean,
            "lightcone_fraction_se": lc_se,
            "pooled_lightcone_fraction": pooled_lc,
        }
        for key, value in table.items():
            if key != "n" and value is not None and not math.isfinite(
                    float(value)):
                raise ValueError(
                    f"r40 evaluation: the {name} table carries a "
                    f"non-finite {key} (acceptance (iv); the NaN guard)"
                )
        pooled_species[name] = table
    return pooled_species


def _point_partition(records: list, mass: float) -> dict:
    """The baryon-line / central partition, the meson alignment moment,
    the pooled depth statistics, the routing census, and the
    conservation block of one (setting, point)."""
    n_decays = sum(
        int(record["statistics"]["n_decays"]) for record in records)
    denom = float(n_decays) * float(mass)

    def per_replica(extract) -> list:
        return [extract(record) for record in records]

    baryon_lc = math.fsum(
        float(record["partition_sums"]["baryon_line_lc"])
        for record in records)
    central_lc = math.fsum(
        float(record["partition_sums"]["central_lc"])
        for record in records)
    baryon_finals = sum(
        int(record["partition_sums"]["baryon_finals"])
        for record in records)
    central_finals = sum(
        int(record["partition_sums"]["central_finals"])
        for record in records)
    baryon_r = per_replica(
        lambda record: (
            float(record["partition_sums"]["baryon_line_lc"])
            / (float(record["statistics"]["n_decays"]) * float(mass))))
    central_r = per_replica(
        lambda record: (
            float(record["partition_sums"]["central_lc"])
            / (float(record["statistics"]["n_decays"]) * float(mass))))
    central_mult_r = per_replica(
        lambda record: (
            float(record["partition_sums"]["central_finals"])
            / float(record["statistics"]["n_decays"])))
    baryon_mean, baryon_se = _mean_se(baryon_r)
    central_mean, central_se = _mean_se(central_r)
    cmult_mean, cmult_se = _mean_se(central_mult_r)
    moment_n = sum(
        int(record["alignment_moment"]["n_mesons"]) for record in records)
    moment_sum = math.fsum(
        float(record["alignment_moment"]["sum_z2_over_p2"])
        for record in records)
    moment_r = per_replica(
        lambda record: (
            float(record["alignment_moment"]["sum_z2_over_p2"])
            / int(record["alignment_moment"]["n_mesons"])
            if int(record["alignment_moment"]["n_mesons"]) > 0 else None))
    moment_mean, moment_se = _mean_se(moment_r)
    for value in (baryon_mean, central_mean, cmult_mean, moment_mean):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(
                "r40 evaluation: a partition or moment observable is "
                "non-finite (acceptance (iv); the NaN guard)"
            )
    hist_all = [0] * _DEPTH_BINS
    hist_catalog = [0] * _DEPTH_BINS
    hist_continuum = [0] * _DEPTH_BINS
    for record in records:
        for depth, count in enumerate(record["depth_histograms"]["all"]):
            hist_all[depth] += int(count)
        for depth, count in enumerate(
                record["depth_histograms"]["catalog_route"]):
            hist_catalog[depth] += int(count)
        for depth, count in enumerate(
                record["depth_histograms"]["continuum_route"]):
            hist_continuum[depth] += int(count)
    n_catalog = sum(
        int(record["route_census"]["n_catalog"]) for record in records)
    n_continuum = sum(
        int(record["route_census"]["n_continuum"]) for record in records)
    n_band_catalog = sum(
        int(record["route_census"]["n_band_catalog"]) for record in records)
    n_band_continuum = sum(
        int(record["route_census"]["n_band_continuum"])
        for record in records)
    return {
        "partition": {
            "baryon_line_fraction": baryon_mean,
            "baryon_line_fraction_se": baryon_se,
            "central_fraction": central_mean,
            "central_fraction_se": central_se,
            "baryon_line_finals_per_decay": (
                float(baryon_finals) / float(n_decays)),
            "central_finals_per_decay": (
                float(central_finals) / float(n_decays)),
            "central_multiplicity": (
                float(central_finals) / float(n_decays)),
            "central_multiplicity_se": cmult_se,
            "pooled_baryon_line_fraction": (
                baryon_lc / denom if denom > 0.0 else None),
            "pooled_central_fraction": (
                central_lc / denom if denom > 0.0 else None),
            "note": (
                "the light-cone fractions sum to one per decay within "
                "the event closure (checked per decay; the off-window "
                "finals carry the remainder)"
            ),
        },
        "alignment_moment": {
            "n_mesons": int(moment_n),
            "mean_z2_over_p2": moment_mean,
            "se": moment_se,
            "status": (
                "supplementary (recorded, non-binding): the mean "
                "p_z^2/|p|^2 of the B = 0 finals"
            ),
        },
        "depth": {
            "all": depth_stats(hist_all, n_decays),
            "catalog_route": (
                depth_stats(hist_catalog, n_catalog)
                if n_catalog > 0 else None),
            "continuum_route": (
                depth_stats(hist_continuum, n_continuum)
                if n_continuum > 0 else None),
            "guard_activations": 0,
        },
        "routing_census": {
            "n_catalog": int(n_catalog),
            "n_continuum": int(n_continuum),
            "n_band_catalog": int(n_band_catalog),
            "n_band_continuum": int(n_band_continuum),
            "fraction_continuum": (
                float(n_continuum) / float(n_decays)
                if n_decays > 0 else 0.0),
            "blend_weight_at_mass": blend_weight_at(mass),
        },
        "conservation": {
            "events_checked": int(n_decays),
            "qn_violations": sum(
                int(record["conservation"]["qn_violations"])
                for record in records),
            "worst_energy_closure_gev": max(
                float(record["conservation"]["worst_energy_closure_gev"])
                for record in records),
            "worst_lightcone_identity_gev": max(
                float(record["conservation"][
                    "worst_lightcone_identity_gev"])
                for record in records),
        },
    }


def _point_entry_tables(records: list, mass: float) -> dict:
    """The full per-(setting, point) table block: the pooled species
    tables, the partition, the alignment moment, the depth statistics,
    the routing census, and the conservation block."""
    species_block = _point_species_tables(records, mass)
    partition_block = _point_partition(records, mass)
    n_decays = sum(
        int(record["statistics"]["n_decays"]) for record in records)
    return {
        "mass_gev": float(mass),
        "n_decays": int(n_decays),
        "replicas": len(records),
        "species": species_block,
        **partition_block,
    }


# ---------------------------------------------------------------------------
# The acceptance-(v) check set: the pre-registered monotonicity verdicts
# ---------------------------------------------------------------------------
def slice_verdict(values: list, ses: list, direction: int) -> tuple:
    """The endpoint-contrast verdict of one slice: sigma = d *
    (x_last - x_first) / SE with d the predicted direction (+1
    increasing, -1 decreasing, 0 flat) and SE the between-replica SE of
    the difference (the endpoint SEs combined in quadrature, floored at
    the engine's DETERMINISTIC_SE_FLOOR).  A direction verdict fails at
    sigma < -3.0; a flat verdict fails at |sigma| > 3.0.  Returns
    (sigma, verdict) with verdict "pass" / "fail" / "skip"."""
    if any(v is None for v in values) or any(s is None for s in ses):
        return None, "skip"
    delta = float(values[-1]) - float(values[0])
    se_first = float(ses[0])
    se_last = float(ses[-1])
    combined = math.sqrt(se_first * se_first + se_last * se_last)
    if not combined > SE_FLOOR:
        combined = SE_FLOOR
    if int(direction) == 0:
        sigma = delta / combined
        verdict = "fail" if abs(sigma) > SIGMA_BAND else "pass"
    else:
        sigma = float(direction) * delta / combined
        verdict = "fail" if sigma < -SIGMA_BAND else "pass"
    return sigma, verdict


def _fail_diagnosis(response: str, point: int, direction: int,
                    sigma: float) -> str:
    """The one-line first-pass diagnosis of a failing verdict (the
    acceptance rescope precedent: sigma + diagnosis + the
    robust-statistics cross-reference)."""
    if direction == 0:
        return (
            f"the {response} endpoint contrast at point {point} moves "
            f"beyond the 3.0-sigma band (|sigma| = {abs(sigma):.2f}) "
            "against the pre-registered zero-response prediction: a "
            "driver defect or a physics finding, diagnosed in this "
            "register, not refit here (robust-statistics "
            "cross-reference: the between-replica SE convention with "
            "the deterministic SE floor; the R32 finite-MC-scatter "
            "envelope applies at |sigma| <= 4.0)"
        )
    return (
        f"the {response} endpoint contrast at point {point} moved "
        "against the pre-registered direction beyond the 3.0-sigma "
        f"band (sigma = {sigma:.2f}): a driver defect or a physics "
        "finding, diagnosed in this register, not refit here "
        "(robust-statistics cross-reference: the between-replica SE "
        "convention with the deterministic SE floor; the R32 "
        "finite-MC-scatter envelope applies at |sigma| <= 4.0)"
    )


def _predicted_direction(response: str, point: int) -> int:
    """The pre-registered response direction at a point (the
    acceptance-(v) map of the pre-registration)."""
    if response == "kappa_B":
        return 1 if point in (1, 2, 3, 4) else 0
    if response == "kappa_M":
        return 0
    if response == "b_dec":
        return -1 if point in (1, 2, 3, 4, 6, 7, 8, 9) else 0
    raise ValueError(f"unknown response {response!r}")


def _partition_observable(table: dict, observable: str):
    """The partition-block observable and its SE of one point table."""
    block = table["partition"]
    if observable == "baryon_line_fraction":
        return block["baryon_line_fraction"], block[
            "baryon_line_fraction_se"]
    if observable == "central_multiplicity":
        return block["central_multiplicity"], block[
            "central_multiplicity_se"]
    raise ValueError(f"unknown partition observable {observable!r}")


def build_check_set(tables: dict, smoke: bool) -> dict:
    """The pre-registered acceptance-(v) monotonicity check set over
    the pooled tables: the kappa_B slice on the baryon-line retention,
    the kappa_M slice on the mesonic central multiplicity, and the
    b_dec slice on <p_T>_x per paper-window species above the
    pre-registered multiplicity floor.  Every failing verdict is named
    in the named_residual_register; the 3.0-sigma band is never
    relaxed."""
    if smoke:
        return {
            "status": "skipped (smoke statistics)",
            "rows": [],
            "named_residual_register": [],
            "note": (
                "the monotonicity check set requires the nine "
                "pre-registered settings at T1 statistics; the smoke "
                "artifacts record the machinery only"
            ),
        }
    rows = []
    register = []
    plan = (
        ("kappa_B", KAPPA_B_SLICE, "baryon_line_fraction", None),
        ("kappa_M", KAPPA_M_SLICE, "central_multiplicity", None),
        ("b_dec", B_DEC_SLICE, "mean_pt_x", "species"),
    )
    for response, slice_settings, observable, scope in plan:
        for point in range(N_POINTS):
            if response == "kappa_B" and point in (5, 6, 7, 8, 9):
                continue  # no baryon line at the mesonic points
            if scope == "species":
                species_plan = []
                for name in SPECIES_KEYS:
                    table = tables[(int(slice_settings[0]), int(point))]
                    mult = table["species"][name]["mean_multiplicity"]
                    if mult is None or mult < PT_CHECK_MIN_MEAN_MULTIPLICITY:
                        continue
                    species_plan.append(name)
            else:
                species_plan = [None]
            for species in species_plan:
                values = []
                ses = []
                for setting in slice_settings:
                    table = tables[(int(setting), int(point))]
                    if species is None:
                        value, se = _partition_observable(
                            table, observable)
                    else:
                        block = table["species"][species]
                        value = block["mean_pt_x"]
                        se = block["mean_pt_x_se"]
                    values.append(value)
                    ses.append(se)
                direction = _predicted_direction(response, point)
                sigma, verdict = slice_verdict(values, ses, direction)
                row = {
                    "response": response,
                    "point": int(point),
                    "observable": observable,
                    "species": species,
                    "settings_order": [int(s) for s in slice_settings],
                    "predicted_direction": int(direction),
                    "values": list(values),
                    "endpoint_ses": list(ses),
                    "sigma": sigma,
                    "verdict": verdict,
                    "robust_statistics_reference": SE_CONVENTION,
                }
                if verdict == "fail":
                    row["diagnosis"] = _fail_diagnosis(
                        response, point, direction, sigma)
                rows.append(row)
    for index, row in enumerate(rows, start=1):
        if row["verdict"] == "fail":
            register.append({
                "index": index,
                "response": row["response"],
                "point": row["point"],
                "observable": row["observable"],
                "species": row["species"],
                "sigma": row["sigma"],
                "predicted_direction": row["predicted_direction"],
                "diagnosis": row["diagnosis"],
                "robust_statistics_reference":
                    row["robust_statistics_reference"],
            })
    return {
        "status": (
            "evaluated" if rows else "skipped (no rows; not expected)"
        ),
        "sigma_band": SIGMA_BAND,
        "rows": rows,
        "named_residual_register": register,
        "note": (
            "the pre-registered acceptance-(v) check set: the endpoint "
            "contrast of each slice with the between-replica SE; a "
            "failing verdict is named with sigma, a one-line diagnosis, "
            "and the robust-statistics cross-reference; the 3.0-sigma "
            "band is never relaxed; no fitting, no refit, no data "
            "selection"
        ),
    }


# ---------------------------------------------------------------------------
# Mode: --evaluate
# ---------------------------------------------------------------------------
def _load_chunk_records(run_dir: Path, smoke: bool) -> list:
    """The production chunk records (all 360; smoke: the single smoke
    record), ordered by (setting, point, replica)."""
    run_dir = Path(run_dir)
    records = []
    settings = [SMOKE_SETTING] if smoke else list(range(N_SETTINGS))
    points = [SMOKE_POINT] if smoke else list(range(N_POINTS))
    for setting in settings:
        for point in points:
            replicas = range(SMOKE_REPLICAS) if smoke else range(T1_REPLICAS)
            for replica in replicas:
                path = run_dir / record_name(setting, point, replica, smoke)
                if not path.is_file():
                    _refuse(
                        f"{path} is missing; --evaluate requires every "
                        "chunk record (run --generate first)"
                    )
                records.append(_read_json(path))
    expected = SMOKE_REPLICAS if smoke else (
        N_SETTINGS * N_POINTS * T1_REPLICAS)
    if len(records) != expected:
        _refuse(
            f"expected {expected} chunk records, found {len(records)}"
        )
    return records


def run_evaluate(run_dir: Path, smoke: bool = False) -> dict:
    """The observable evaluation: the per-(setting, point) tables pooled
    over the replicas, the pre-registered acceptance-(v) check set with
    its named-residual register, and the acceptance-(i)-(iv) blocks.
    Writes ``scan_report.json`` (no wall-clock fields: the bytes are a
    pure function of the sealed records and the committed code) and
    stops.  A check-set verdict failure is recorded and named; the step
    still exits 0 (the run seals with the named-residual register)."""
    run_dir = Path(run_dir)
    pre = _load_pre_registration(run_dir, smoke)
    _assert_pre_registration(run_dir, smoke)
    records = _load_chunk_records(run_dir, smoke)
    by_setting_point: dict[tuple[int, int], list] = {}
    for record in records:
        key = (int(record["setting"]), int(record["point"]))
        by_setting_point.setdefault(key, []).append(record)
    for key in by_setting_point:
        by_setting_point[key].sort(
            key=lambda record: int(record["replica"]))
    tables = {}
    for (setting, point), chunk_records in sorted(by_setting_point.items()):
        mass = float(POINTS[int(point)]["mass_gev"])
        tables[(setting, point)] = _point_entry_tables(chunk_records, mass)
    check_set = build_check_set(tables, smoke)
    total_decays = sum(
        int(record["statistics"]["n_decays"]) for record in records)
    conservation = {
        "events_checked": int(total_decays),
        "qn_violations": sum(
            int(record["conservation"]["qn_violations"])
            for record in records),
        "worst_energy_closure_gev": max(
            float(record["conservation"]["worst_energy_closure_gev"])
            for record in records),
        "worst_lightcone_identity_gev": max(
            float(record["conservation"]["worst_lightcone_identity_gev"])
            for record in records),
        "tolerance": EVENT_CLOSURE_TOL,
        "pass": all(
            int(record["conservation"]["qn_violations"]) == 0
            and float(
                record["conservation"]["worst_energy_closure_gev"])
            <= EVENT_CLOSURE_TOL
            for record in records
        ),
    }
    nan_violations = sum(
        1 for record in records
        if any(
            not math.isfinite(float(value))
            for value in (
                record["conservation"]["worst_energy_closure_gev"],
                record["conservation"]["worst_lightcone_identity_gev"],
            )
        )
    )
    depth_max = max(
        table["depth"]["all"]["max"] for table in tables.values())
    depth_mean_max = max(
        table["depth"]["all"]["mean"] for table in tables.values())
    depth_p95_max = max(
        table["depth"]["all"]["p95"] for table in tables.values())
    guard_activations = 0
    depth_summary = {
        "guard_activations": int(guard_activations),
        "max_over_points": int(depth_max),
        "p95_max_over_points": int(depth_p95_max),
        "mean_max_over_points": float(depth_mean_max),
        "guard_bound": MAX_CASCADE_DEPTH,
        "per_setting_point": {
            f"setting_{setting}_point_{point}": {
                "mean": table["depth"]["all"]["mean"],
                "p95": table["depth"]["all"]["p95"],
                "max": table["depth"]["all"]["max"],
                "catalog_route": (
                    table["depth"]["catalog_route"]
                    if table["depth"]["catalog_route"] else None),
                "continuum_route": (
                    table["depth"]["continuum_route"]
                    if table["depth"]["continuum_route"] else None),
            }
            for (setting, point), table in sorted(tables.items())
        },
        "pass": True,
        "note": (
            "zero CascadeDepthGuardError activations across the full "
            "run (plan section 10.6 line 757: a guard raises and stops "
            "the run, so the accepted count is zero by construction)"
        ),
    }
    per_setting_point = {}
    for (setting, point), table in sorted(tables.items()):
        per_setting_point[f"setting_{setting}_point_{point}"] = table
    report = {
        "schema": "na49_mixing.staged_r40_alignment_scan.report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "settings": N_SETTINGS,
        "points": N_POINTS,
        "decays_checked": int(total_decays),
        "conservation": conservation,
        "depth": depth_summary,
        "observables": {"per_setting_point": per_setting_point},
        "check_set": check_set,
        "named_residual_register": list(
            check_set["named_residual_register"]),
        "obligation_4_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "applies_to": OBLIGATION4_APPLIES_TO,
            "not_applicable_to": (
                "the 2 GeV catalog points and the 4, 6, 8 GeV continuum "
                "points are off-band"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "determinism": {
            "rolling_digest_convention": (
                "each chunk record carries a sha256 rolling digest over "
                "the per-decay canonical payloads; the clean-"
                "environment validation re-runs the pre-registered "
                "probe chunk and the evaluation over the sealed "
                "artifacts"
            ),
            "records_with_digest": sum(
                1 for record in records
                if isinstance(record.get("rolling_digest"), str)
                and len(record["rolling_digest"]) == 64
            ),
            "records_total": len(records),
        },
        "report_note": (
            "no wall-clock fields: the report bytes are a pure "
            "function of the sealed chunk records and the committed "
            "code, which is what the clean-environment validation "
            "re-hashes"
        ),
    }
    _write_json(run_dir / (f"{SMOKE_PREFIX}scan_report.json" if smoke
                           else SCAN_REPORT_NAME), report)
    return report


# ---------------------------------------------------------------------------
# Mode: --gate
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """Write ``gate4_r40.json`` (evidence only; the written Gate 4
    retain / revise / reject decision on the Gate 4 clauses is the
    round's separate follow-up obligation).  Requires the scan
    report."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    _assert_pre_registration(run_dir)
    report_path = run_dir / SCAN_REPORT_NAME
    if not report_path.is_file():
        _refuse(
            f"{report_path} is missing; --gate requires the scan report "
            "(run --evaluate first)"
        )
    report = _read_json(report_path)
    if report.get("mode") != "production":
        _refuse("the scan report is not a production record")
    conservation = report["conservation"]
    depth = report["depth"]
    carry = report["obligation_4_carry"]
    payload = {
        "schema": "na49_mixing.staged_r40_alignment_scan.gate/1",
        "gate": "gate 4 (plan section 10, line 806)",
        "run": RUN_ID,
        "card": CARD,
        "clause": "shape (training-side scan evidence)",
        "settings": int(report["settings"]),
        "points": int(report["points"]),
        "decays_checked": int(report["decays_checked"]),
        "conservation": {
            "violations": int(conservation["qn_violations"]),
            "pass": bool(conservation["pass"]),
            "worst_energy_closure_gev": float(
                conservation["worst_energy_closure_gev"]),
            "tolerance": float(EVENT_CLOSURE_TOL),
        },
        "depth": {
            "guard_activations": int(depth["guard_activations"]),
            "guard_bound": int(depth["guard_bound"]),
            "max_over_points": int(depth["max_over_points"]),
            "p95_max_over_points": int(depth["p95_max_over_points"]),
            "mean_max_over_points": float(depth["mean_max_over_points"]),
            "per_setting_point": {
                key: {
                    "mean": block["mean"],
                    "p95": block["p95"],
                    "max": block["max"],
                }
                for key, block in depth["per_setting_point"].items()
            },
        },
    }
    payload["observables"] = {
        "per_setting_point": {
            key: {
                "retention": {
                    "baryon_line_fraction": table["partition"][
                        "baryon_line_fraction"],
                    "central_fraction": table["partition"][
                        "central_fraction"],
                },
                "multiplicity": {
                    name: block["mean_multiplicity"]
                    for name, block in table["species"].items()
                },
                "mean_pt_x": {
                    name: block["mean_pt_x"]
                    for name, block in table["species"].items()
                },
                "mean_pt_y": {
                    name: block["mean_pt_y"]
                    for name, block in table["species"].items()
                },
            }
            for key, table in
            report["observables"]["per_setting_point"].items()
        },
        "note": (
            "the full tables (with the between-replica standard errors "
            "and the routing censuses) live in scan_report.json; this "
            "block carries the per-setting-point headline values"
        ),
    }
    payload["model_uncertainty_carry"] = {
        "obligation_4": True,
        "r32_spread": dict(carry["r32_spread"]),
        "applies_to": carry["applies_to"],
    }
    payload["named_residual_register"] = list(
        report["named_residual_register"])
    payload["no_fitting"] = True
    payload["optimizer_invoked"] = False
    payload["sealed"] = True
    payload["sealed_note"] = (
        "the run seals after this record (the seal manifest and "
        "SHA256SUMS.txt cover this file; the run dir is immutable "
        "after the seal)"
    )
    payload["gate_note"] = (
        "evidence only: the written retain / revise / reject decision "
        "on the Gate 4 clauses is the round's separate follow-up "
        "obligation, recorded in _knowledge/ and in the round 4 index "
        "status field"
    )
    _write_json(run_dir / GATE_RECORD_NAME, payload)
    print(
        f"staged_r40_alignment_scan: gate evidence written to "
        f"{run_dir / GATE_RECORD_NAME} (decays_checked "
        f"{payload['decays_checked']}, conservation pass "
        f"{payload['conservation']['pass']}, named residuals "
        f"{len(payload['named_residual_register'])})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --seal
# ---------------------------------------------------------------------------
def _run_dir_files(run_dir: Path) -> list:
    """Every regular file under the run directory, sorted by relative
    path."""
    run_dir = Path(run_dir).resolve()
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def write_sha256sums(run_dir: Path) -> Path:
    """Write ``SHA256SUMS.txt`` over every payload file (sha256sum -c).

    The inventory covers every regular file under the run directory
    except the files that cannot carry their own digests (the inventory
    itself and ``seal_manifest.json``); the entries carry
    monorepo-root-relative paths, so the card's validation command
    ``sha256sum -c dev/00runs/<run>/SHA256SUMS.txt`` verifies from the
    repository root (the R21 convention)."""
    run_dir = Path(run_dir)
    skip = {SHA256SUMS_NAME, SEAL_NAME}
    lines = []
    for path in _run_dir_files(run_dir):
        if path.name in skip:
            continue
        rel = path.relative_to(Path(MONO_ROOT).resolve()).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}\n")
    sums_path = run_dir / SHA256SUMS_NAME
    _write_bytes(sums_path, "".join(lines).encode("utf-8"))
    return sums_path


def write_seal_manifest(run_dir: Path) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the
    sums.  The seal manifest records every regular file under the run
    directory except itself, each with its sha256 and byte size; it
    cannot contain its own digest."""
    run_dir = Path(run_dir).resolve()
    entries = []
    total_bytes = 0
    for path in _run_dir_files(run_dir):
        if path.name == SEAL_NAME:
            continue
        size = int(path.stat().st_size)
        total_bytes += size
        entries.append({
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": size,
        })
    sums_path = run_dir / SHA256SUMS_NAME
    seal = {
        "schema": "na49_mixing.staged_r40_alignment_scan.seal_manifest/1",
        "run": RUN_ID,
        "card": CARD,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": int(total_bytes),
        "sha256_sums_file": SHA256SUMS_NAME,
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": [SEAL_NAME, SHA256SUMS_NAME],
        "path_convention": (
            "the SHA256SUMS.txt entries carry monorepo-root-relative "
            f"paths (dev/00runs/{run_dir.name}/ prefixed), so the run "
            f"card's validation command `sha256sum -c dev/00runs/"
            f"{run_dir.name}/SHA256SUMS.txt` verifies from the repo "
            "root (the R21 convention)"
        ),
        "note": (
            "the run dir is immutable after the seal (the Round 1 "
            "sealing discipline); clean_environment_commands.json is "
            "the post-seal validation record, written after this "
            "manifest and never edited"
        ),
        "files": entries,
    }
    _write_json(run_dir / SEAL_NAME, seal)
    return seal


def run_seal(run_dir: Path) -> dict:
    """Seal the finished run directory (sums inventory plus seal
    manifest).  The pre-registration, the provenance, the 360
    production chunk records, the scan report, and the gate record must
    exist; smoke artifacts are refused."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        SCAN_REPORT_NAME,
        GATE_RECORD_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the gate "
                "JSON is part of the seal; run --evaluate and --gate "
                "first)"
            )
    n_records = len(list(run_dir.glob("setting_*_point_*_replica_*.json")))
    if n_records != N_SETTINGS * N_POINTS * T1_REPLICAS:
        _refuse(
            f"expected {N_SETTINGS * N_POINTS * T1_REPLICAS} chunk "
            f"records before sealing, found {n_records}"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"staged_r40_alignment_scan: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------
def recompute_probe_chunk() -> dict:
    """Recompute the pre-registered probe chunk (setting 1, point 0,
    replica 0) in memory with the exact production draw sequence (no
    file writes), for the clean-environment validation comparison."""
    context = _generation_context(smoke=False)
    _set_worker_context(context)
    return generate_chunk(
        (PROBE_SETTING, PROBE_POINT, PROBE_REPLICA,
         T1_DECAYS_PER_REPLICA))


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the evaluation from a clean interpreter
    over the sealed artifacts (byte-identical report), re-run the
    pre-registered probe chunk in memory against its sealed record, and
    write ``clean_environment_commands.json``."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    _load_pre_registration(run_dir)
    if not (run_dir / SHA256SUMS_NAME).is_file():
        _refuse(
            f"{run_dir / SHA256SUMS_NAME} is missing; --clean-validate "
            "requires the sealed run dir (run --seal first)"
        )
    mismatches = []
    files_checked = 0
    for line in (run_dir / SHA256SUMS_NAME).read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
        files_checked += 1
        if not payload.is_file() or _sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(
            "seal verification failed; "
            f"{len(mismatches)} of {files_checked} SHA256SUMS.txt "
            f"entries do not re-hash correctly (first: "
            f"{mismatches[0]!r}); the sealed run dir must be unmodified"
        )
    report_path = run_dir / SCAN_REPORT_NAME
    before = report_path.read_bytes()
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-I", str(driver_path),
        "--evaluate", "--run-dir", str(run_dir),
    ]
    probe = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(MONO_ROOT))
    if probe.returncode != 0:
        _refuse(
            "the clean-interpreter evaluation probe failed (return code "
            f"{probe.returncode}):\n{probe.stderr}"
        )
    recomputed = report_path.read_bytes()
    report_match = before == recomputed
    if recomputed != before:
        _write_bytes(report_path, before)
    sealed_record = _read_json(run_dir / record_name(
        PROBE_SETTING, PROBE_POINT, PROBE_REPLICA, False))
    recomputed_payload = recompute_probe_chunk()
    probe_checks = {
        "n_decays": (
            int(sealed_record["statistics"]["n_decays"])
            == int(recomputed_payload["n_decays"])),
        "rolling_digest": (
            str(sealed_record["rolling_digest"])
            == str(recomputed_payload["rolling_digest"])),
        "route_census": (
            sealed_record["route_census"]["n_catalog"]
            == int(recomputed_payload["route_census"]["n_catalog"])
            and sealed_record["route_census"]["n_continuum"]
            == int(recomputed_payload["route_census"]["n_continuum"])
            and sealed_record["route_census"]["n_band_catalog"]
            == int(recomputed_payload["route_census"]["n_band_catalog"])
            and sealed_record["route_census"]["n_band_continuum"]
            == int(recomputed_payload["route_census"]["n_band_continuum"])
        ),
        "depth_histograms": (
            [int(v) for v in sealed_record["depth_histograms"]["all"]]
            == [int(v) for v in recomputed_payload["depth_histograms"][
                "all"]]
        ),
        "draw_budget": (
            {
                key: int(value)
                for key, value in sealed_record["draw_budget"].items()
                if key != "note"
            }
            == {
                key: int(value)
                for key, value in
                recomputed_payload["draw_budget"].items()
                if key != "note"
            }
        ),
    }
    probe_ok = all(probe_checks.values())
    payload = {
        "schema": "na49_mixing.staged_r40_alignment_scan"
                  ".clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": [
            {
                "purpose": (
                    "clean-interpreter re-run of the scan evaluation "
                    "over the sealed artifacts (rewrites "
                    "scan_report.json; the sealed bytes must be "
                    "reproduced byte for byte)"
                ),
                "argv": argv,
                "cwd": str(MONO_ROOT),
            },
            {
                "purpose": (
                    "the in-memory re-run of the pre-registered probe "
                    "chunk (setting 1, point 0, replica 0) against its "
                    "sealed record (the per-(setting, point, replica) "
                    "determinism acceptance)"
                ),
                "argv": None,
                "cwd": str(MONO_ROOT),
                "probe_chunk": {
                    "setting": PROBE_SETTING,
                    "point": PROBE_POINT,
                    "replica": PROBE_REPLICA,
                    "n_decays": T1_DECAYS_PER_REPLICA,
                },
            },
        ],
        "environment": environment_fingerprint(),
        "probe_returncode": int(probe.returncode),
        "sealed_report_sha256": {
            SCAN_REPORT_NAME: hashlib.sha256(before).hexdigest(),
        },
        "recomputed_report_sha256": {
            SCAN_REPORT_NAME: hashlib.sha256(recomputed).hexdigest(),
        },
        "report_hashes_match": bool(report_match),
        "probe_chunk_checks": probe_checks,
        "probe_chunk_rolling_digest": {
            "sealed": str(sealed_record["rolling_digest"]),
            "recomputed": str(recomputed_payload["rolling_digest"]),
        },
        "probe_tool": {
            "path": str(driver_path),
            "sha256": _sha256_file(driver_path),
            "note": (
                "the committed dual-tree driver is the probe "
                "instrument; the sealed artifacts and the committed "
                "frozen substrate are the inputs"
            ),
        },
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r40_alignment_scan.py",
        description=(
            "Execute run R40, the fixed-source alignment scan (Round 4, "
            "card 11): ten at-rest root points (the baryonic (1, 1, 0) "
            "and the mesonic (0, 0, 0) families at 2, 3, 4, 6, and 8 "
            "GeV), T1 per point, the pre-registered one-dimensional "
            "kappa_B / kappa_M / b_dec slices at the nominal point "
            "(1, 1, 1.0 GeV^-2), the card-10 aligned cascade machinery "
            "with the card-9 routing, no refit (plan section 10, line "
            "761)."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (never sealed) "
             "into an explicit --run-dir",
    )
    modes.add_argument(
        "--generate", action="store_true",
        help="generate the requested (setting, point, replica) chunks "
             "(any subset; the default is every chunk)",
    )
    modes.add_argument(
        "--evaluate", action="store_true",
        help="run the observable evaluation over the chunk records, "
             "then stop",
    )
    modes.add_argument(
        "--gate", action="store_true",
        help="write gate4_r40.json (acceptance evidence), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the evaluation and the probe "
             "chunk cleanly, then stop",
    )
    parser.add_argument(
        "--setting", type=int, default=None,
        help="with --generate: the setting index s = 0..8",
    )
    parser.add_argument(
        "--point", type=int, default=None,
        help="with --generate: the point index p = 0..9",
    )
    parser.add_argument(
        "--replica", type=int, default=None,
        help="with --generate: the replica index r = 0..3",
    )
    parser.add_argument(
        "--settings", type=int, default=None,
        help="the declared setting count (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--points", type=int, default=None,
        help="the declared point count (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--replicas", type=int, default=None,
        help="the declared replica count per point (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--decays-per-replica", type=int, default=None,
        help="the declared decays per replica (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--master-seed", type=int, default=None,
        help="the declared master seed (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--workers", type=int, default=CHUNK_WORKERS_DEFAULT,
        help=f"fork worker count (default {CHUNK_WORKERS_DEFAULT})",
    )
    parser.add_argument(
        "--run-dir", default=DEFAULT_RUN_DIR,
        help=f"the run directory (default {DEFAULT_RUN_DIR})",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.pre_registration:
        run_pre_registration(run_dir)
        return 0
    if args.smoke:
        run_smoke(run_dir, args.workers)
        return 0
    if args.generate:
        statistics = {
            "settings": args.settings,
            "points": args.points,
            "replicas": args.replicas,
            "decays-per-replica": args.decays_per_replica,
            "master-seed": args.master_seed,
        }
        run_generation(
            run_dir, args.setting, args.point, args.replica,
            args.workers, statistics,
        )
        return 0
    if args.evaluate:
        run_evaluate(run_dir, smoke=False)
        print(
            "staged_r40_alignment_scan: evaluation written to "
            f"{run_dir / SCAN_REPORT_NAME}",
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
        payload = run_clean_validate(run_dir)
        _write_json(run_dir / CLEAN_ENV_NAME, payload)
        probe_ok = bool(payload["report_hashes_match"]) and all(
            payload["probe_chunk_checks"].values())
        if not probe_ok:
            print(
                "staged_r40_alignment_scan: the clean-environment "
                "validation FAILED (the recomputed report bytes or the "
                "probe chunk differ from the sealed run)",
                flush=True,
            )
            raise SystemExit(1)
        print(
            "staged_r40_alignment_scan: clean-environment validation "
            f"passed ({payload['seal_verification']['files_checked']} "
            "seal files verified, the clean interpreter reproduced "
            f"{SCAN_REPORT_NAME} byte for byte, the probe chunk "
            "reproduced its sealed rolling digest)",
            flush=True,
        )
        return 0
    _refuse("exactly one mode is required")


if __name__ == "__main__":
    sys.exit(main())
