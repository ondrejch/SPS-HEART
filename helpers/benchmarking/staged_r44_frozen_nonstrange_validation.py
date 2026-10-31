#!/usr/bin/env python3
"""R44 driver: the frozen nonstrange validation (Round 4, card 15).

Task ``TASK-20260920-r4-run-r44-frozen-nonstrange-validation`` (plan
section 10 run R44, line 800; the Round 4 card set, card 15; plan
section 23, lines 1821-1823), re-pointed by card 19
(``TASK-20260920-r4-run-r44-repoint``).  R44 is the held-out side of
the Gate 4 first clause: the frozen R41 parameters (kappa_B, kappa_m,
b_dec, baryonic nominal_width) reproduce the held-out projections
without refit, and the frozen aligned configuration is compared
against the isotropic sequential baseline on the held-out set.  The
plan section 5.5 mechanism-retention rule (lines 315-323) applies with
all seven conditions.

Frozen parameters (no refit; the card-19 re-point)
--------------------------------------------------
(kappa_B, kappa_m, b_dec, baryonic_nominal_width) at the corrected
R41 fitted values, frozen at the corrected R41 seal: kappa_B = 0.0
(the pre-registered lower bound), kappa_m = 4.0 (the upper bound),
b_dec = 2.0 GeV^-2 (the upper bound), nominal_width = 0.3 GeV (the
corrected card-16 relitigated fit's baryonic continuum width).  The
values are asserted against the corrected sealed R41 archive's fit
report (``r41_nonstrange_pilot_fit_2026-09-23``) before any
generation; no refit happens on this run (plan section 10, line 802;
``no_fitting`` is ``true`` and ``optimizer_invoked`` is ``false``).
The blend ramp start stays fixed at B (obligation 4; excluded from
the R41 fit set).  The superseded card-15 frozen tuple
(0.0 / 4.0 / 2.0 GeV^-2 / 0.5 GeV, from the superseded sealed
``r41_nonstrange_pilot_fit_2026-09-22`` archive) is retained as the
``SUPERSEDED_*`` constants below for the provenance comparisons; it is
recorded as superseded in the pre-registration and provenance and is
not consumed.

Variants (two configurations on identical exchange events)
----------------------------------------------------------
- **aligned_frozen** (variant id 0): card 4's sequential cascade at the
  frozen R41 point (kappa_B, kappa_m, b_dec) = (0.0, 4.0, 2.0 GeV^-2)
  with the card-8 form factor attached at b_dec = 2.0 GeV^-2 and the
  frozen baryonic nominal_width recorded at the corrected fitted value
  0.3 GeV (the continuum-daughter mass rule consumes the byte-unchanged
  committed continuum config's family widths; the fitted width is the
  frozen-parameter record of the evaluation, not a config override).
- **isotropic_sequential_baseline** (variant id 1): card 10's v2 policy
  at kappa_B = kappa_m = 0 with b_dec = 1.0 GeV^-2, the card's baseline
  for the Gate 4 "beyond the isotropic sequential cascade" clause; it
  carries nominal_width as its single parameter, recorded at the same
  corrected frozen value 0.3 GeV, so both variants share the same
  continuum-daughter mass rule (the byte-unchanged committed config)
  and the same support.  At the superseded card-15 landing the fitted
  width 0.5 GeV coincided with the committed config value.

The two variants decay the identical exchange events (the exchange
layer is generated once per event and shared; the per-variant decay
streams are card 10's child SeedSequence spawns keyed on the variant
id).  kappa_B, kappa_m, and b_dec alter probability densities but not
kinematic support (plan section 5.3 line 288) and the widths agree, so
no regeneration is required and the comparison is a frozen-parameter
evaluation of both configurations (no refit of either).

Held-out set (round 4 index item 11)
------------------------------------
The species-separated pion and proton distributions (pi+, pi-, pi0,
p, pbar) and the species-separated multiplicity moments, held out
relative to R41's fit set: R41 fits the charge-symmetric combinations
of plan section 7.3 (pi_Sigma = pi+ + pi-, B_transport = p - pbar, the
pooled pion and proton seagulls, the pooled multiplicity moments),
which are the training observables.  The published data surface (the
sealed R01 archive) carries the six paper-window species only, so the
pi0 distribution and moment are recorded model-side as held-out
diagnostics (no published pi0 table exists; the likelihood register is
the data-backed pi+, pi-, p, pbar spectra and moments).  The published
binning is preserved (the sealed R01 tables).

Likelihood (plan section 5.4) and the information criterion (plan
section 5.5 condition 6, line 322)
-----------------------------------
-2ln L on the held-out bins with sigma^2 = v_data + se^2 (the published
variance plus the between-replica MC variance, floored at the engine's
deterministic SE floor); the published binning is preserved; the data
and MC contributions are reported separately per observable (the
data-variance-only and MC-variance-only forms alongside the total).
The recorded bin sums follow the established R01
accumulate-dense-then-pick convention (``_bin_track_sums``): the finals
accumulate into each species' dense pT-major lattice and the published
rows are picked from the dense totals (``pick = bin_index_pt * n_x +
bin_index_xf``), so the published tables' non-rectangular geometry
(holes; rows not pT-major) is preserved and hole-cell weight is
dropped, never recorded into a neighboring published row or species.
The information-criterion comparison is delta AIC of the frozen aligned
configuration against the isotropic sequential baseline, parameter cost
included: the aligned configuration carries kappa_B, kappa_m, b_dec,
nominal_width (k = 4); the baseline carries nominal_width (k = 1).

Acceptance frame (plan section 5.5, all seven conditions)
---------------------------------------------------------
(i) conservation and on-shell: 5,000,000 events, exact B = 2 / Q = 2
and exact per-source (B, Q, S); zero violations; zero depth-guard
activations.  (ii) training improvement: cross-referenced from the
sealed R43 evidence (this run does not re-evaluate the training set).
(iii) held-out improvement or preservation: per held-out observable the
frozen aligned deviance is recorded against the baseline; a held-out
observable that degrades by more than 2.0x the pooled MC standard error
of the difference is a named finding, and the |residual| > 3.0 sigma
entries carry the house named-residual form of the fixed engine
(``SIGMA_TOLERANCE = 3.0``).  (iv) no sector degradation beyond 5% in
validation deviance without a stated physics reason (the six
paper-window species and the feed-down fractions under the card-9
convention).  (v) parameter stability: cross-referenced from the sealed
R41 replica-block stability table.  (vi) delta AIC recorded.  (vii)
color-singlet audit: cross-referenced from the conservation evidence of
this run.  Plus determinism, seal, clean validation, and pins/freeze.

No fitting: the validation is a frozen-parameter evaluation (plan
section 10, line 802); ``no_fitting`` is ``true`` and
``optimizer_invoked`` is ``false``.

Obligation 4 carry (model uncertainty): the sealed R32 cross-variant
spread (bin spread p95 4.028426 at 4,800,000 events; feed-down spread
max 31.1773 sigma; 32 persisting residuals) is recorded in the
pre-registration and in the gate evidence; the audits closed with card
2 and card 3 PASS and card 1 RECOMMENDATION (its rework card 1r landed
before this run with no new spread diagnostic), so the original R32
spread stands as the final form.  Obligation 5 (the engine moment
floor) is a separate track and is not conflated with the R32 spread.

Conservation: every p+p event conserves total B = 2 and Q = 2 exactly
and every source its own (B, Q, S) = (1, 1, 0), in every variant; zero
depth-guard activations (plan section 10.6 line 757); the event energy
sum closes within 1e-9 GeV.  A violation raises and stops the run.

Statistics (T3, plan section 5.1 line 255): 20 replicas x 250,000 p+p
events = 5,000,000 events.

Seeds (round 4 index item 9)
----------------------------
master seed 20260920; raw replica seeds 20301020-20301039 (k = 0..19);
per-replica event index 700_000_000 + i with i < 250,000, the
derive_substreams pattern on the closed decay_channel/decay_kinematics
labels.  The block is disjoint from the reserved audit block
20301040 + k (k = 0..15, 20301040..20301055, never fired).  The raw
block shares the seeds 20301020-20301027 with the sealed R41/R42/R43
block (and 20301020-20301023 with the R40 block) by the established
pre-registration convention; the (seed, event-index) substreams are
disjoint because the event-index partitions are disjoint (R40
300_000_000 + s*10_000_000 + p*1_000_000 + i with s < 9, p < 10,
i < 25,000 stays below 389,025,000; R41 400_000_000 + i with
i < 100,000; R42 500_000_000 + i with i < 100,000; R43 600_000_000 + i
with i < 100,000; the R43 smoke partition 690_000_000 + i with
i < 2,000).  The previously used blocks are read from the sealed
R41/R42/R43 pre-registrations at pre-registration time and recorded;
if any raw (seed, event-index) substream collides, the driver stops
and reports (never a silent re-block).

Determinism: a per-replica re-run from a clean interpreter reproduces
the report hashes; the held-out report and the gate record carry no
wall-clock fields (their bytes are a pure function of the sealed
artifacts, the pre-registered seeds, and the committed code).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests, the sealed R41
  fit report (the frozen-parameter assertion), and the sealed R30-R32
  and R41-R43 seals first.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration (the frozen-parameter assertion
  against the pinned R41 fixture), 1 replica x 2,000 events at the
  smoke seeds (both variants), and the smoke evaluation against a
  synthetic data vector, into an explicit ``--run-dir`` (the default
  production run dir is refused).  Two identical smoke invocations
  produce byte-identical artifacts.  ``--gate`` / ``--seal`` /
  ``--clean-validate`` refuse smoke artifacts (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested replica records
  (any subset; the default is every replica), one record file per
  replica, then stop.  Requires the pre-registration.  A completed
  record that matches the pre-registered identity is reused unchanged,
  so the declared tier can be completed replica by replica (the
  pre-registered gateway fallback runs one job per replica).
- ``--evaluate``: the held-out evaluation (the per-variant observable
  tables, the exchange bit-identity verification at run scale, the
  stream-pin verification, the held-out deviance and degradation
  table, the delta AIC, the named-residual register, and
  ``held_out_report.json``), then stop.  Requires all 20 production
  records.
- ``--gate``: write ``gate4_r44.json`` (evidence only), then stop.
  Requires the held-out report.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the evaluation from a
  clean interpreter over the sealed artifacts with a byte-identical
  report, re-run production replica 0 in memory against its sealed
  record, write ``clean_environment_commands.json``, then stop.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic in ``--smoke`` (the committed catalog, the committed continuum
config, and the synthetic grid only; no SPS table, no sealed archive
read for the smoke evaluation's synthetic data; the frozen-parameter
assertion runs against the pinned R41 fixture); the production modes
read the committed catalog and continuum config, the sealed R01
archive (the published data surface of the held-out bins), the sealed
R41 archive (the frozen-parameter source), and the sealed R30-R43
archives (the obligation-4 carry and the condition cross-references).
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
        f"staged_r44_frozen_nonstrange_validation: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from helpers.benchmarking import sps_core  # noqa: E402
from na49_mixing.cascade_ablation import (  # noqa: E402
    ABLATION_POLICIES,
    _attach_form_factor,
    _draw_attached_vertex,
    exchange_root_state,
    variant_stream,
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
    scatter_two_body,
)
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.resonance_pp_builder import (  # noqa: E402
    CONTINUUM_ROOT_SENTINEL,
    ResonancePPBuilder,
)
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BlendSpec,
    active_families,
    families_from_json,
    route_source_decay,
)
from na49_mixing.threshold_metric import (  # noqa: E402
    DETERMINISTIC_SE_FLOOR,
    SIGMA_TOLERANCE,
)

#: The driver's own digest (the identity block of every record and the
#: provenance; computed after the concatenation).
DRIVER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

#: The superseded frozen parameters (the card-15 landing's pin against
#: the superseded card-12 R41 fit output; sha256 prefix ``69851f1a`` for
#: the superseded fit-report file, self-digest prefix ``2c0755e6``).
#: The card-19 re-point moves the frozen parameters to the corrected
#: card-16 relitigated fit output; the superseded values stay retained
#: here for the sealed provenance comparisons (read-only; the sealed
#: superseded archive stays intact, its seal verifies, and nothing in
#: it is consumed).
SUPERSEDED_R41_FITTED = {
    "kappa_B": 0.0,
    "kappa_m": 4.0,
    "b_dec": 2.0,
    "baryonic_nominal_width": 0.5,
}
#: The superseded sealed R41 archive (the card-15 landing's frozen-
#: parameter source; read-only, its seal verifies and is recorded in
#: the provenance).
SUPERSEDED_R41_RUN_DIR = "dev/00runs/r41_nonstrange_pilot_fit_2026-09-22"
SUPERSEDED_R41_FIT_REPORT_FILE_SHA256 = (
    "69851f1ac29804d2745de7efabffe332be928e77c0474f353c8338a7bc1ebcfe"
)
SUPERSEDED_R41_FIT_REPORT_DIGEST = (
    "2c0755e6370739b3089120d432e447433b48eed7441281f58103a78457a990b4"
)
#: The superseded sibling driver digests (the card-12/card-13/card-14
#: landings; the card-19 re-point moves the frozen-substrate pins to
#: the corrected card-16/card-17/card-18 driver digests).  The sealed
#: superseded sibling archives were generated under these drivers.
SUPERSEDED_R41_DRIVER_SHA256 = (
    "573d833bfbc659885fb83e6c0794207947fbe9e5a6dafe6159650bd45097172b"
)
SUPERSEDED_R42_DRIVER_SHA256 = (
    "30e97f18653129ba3e7f1e5d8af782c819a1a7b189413032925ec35b26b0afa0"
)
SUPERSEDED_R43_DRIVER_SHA256 = (
    "4c6bc0f32430577ff80e607e19f2512bf2ed1c40701f1f115ff82a88c975bf3a"
)
#: The superseded sealed R42/R43 archives (read-only; their seals
#: verify and are recorded in the provenance).
SUPERSEDED_R42_RUN_DIRS = (
    "dev/00runs/r42_recoil_decay_separation_2026-09-22",
    "dev/00runs/r42_recoil_decay_separation_2026-09-22r2",
)
SUPERSEDED_R43_RUN_DIR = "dev/00runs/r43_cascade_ablation_2026-09-22"
#: The superseded sealed archives verified read-only at pre-registration
#: time (the seal states are recorded in the provenance; nothing in
#: them is consumed).
SUPERSEDED_ARCHIVE_RUN_DIRS = (
    (SUPERSEDED_R41_RUN_DIR,) + SUPERSEDED_R42_RUN_DIRS
    + (SUPERSEDED_R43_RUN_DIR,)
)


# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260920-r4-run-r44-frozen-
# nonstrange-validation; the frozen parameters, collision, held-out set,
# baseline, statistics, seeds, likelihood, acceptance blocks are verbatim
# from the card and the round 4 index items 7, 9, and 11).
# ---------------------------------------------------------------------------
RUN_ID = "R44"
CARD = "TASK-20260920-r4-run-r44-frozen-nonstrange-validation"
RUN_NAME = "r44_frozen_nonstrange_validation_2026-09-23"
DEFAULT_RUN_DIR = "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"
GATE_RECORD_NAME = "gate4_r44.json"
REPORT_NAME = "held_out_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r44_frozen_nonstrange_validation.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
ROUTING_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
KERNEL_REL = "dev/mixing2006/na49_mixing/resonance_decay.py"
R32_DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"
R40_DRIVER_REL = "helpers/benchmarking/staged_r40_alignment_scan.py"
R41_DRIVER_REL = "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py"
R42_DRIVER_REL = "helpers/benchmarking/staged_r42_recoil_decay_separation.py"
R43_DRIVER_REL = "helpers/benchmarking/staged_r43_cascade_ablation.py"

#: The card chain this run sits on: the Gate 3 closure audits (cards 1-3
#: and the rework card 1r), the cascade implementation cards (4-10, the
#: variant machinery), the sealed R40 scan (card 11), R41 pilot fit
#: (card 12, the frozen-parameter source), R42 profile (card 13), and
#: R43 ablation (card 14, the training-side cross-reference).
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
)

#: The frozen substrate digests (byte-unchanged inputs; verified before
#: any step).  The cascade surfaces are cards 4-10 (consumed read-only);
#: the Round 2 kernel, the builder blend path, the card-6 routing, the
#: engine, the catalog, the continuum config, and the R32-R43 drivers
#: close the list.  The R41/R42/R43 driver digests are the corrected
#: card-16/card-17/card-18 relitigation landings (the superseded
#: card-12/card-13/card-14 digests are retained as the
#: SUPERSEDED_R{41,42,43}_DRIVER_SHA256 constants above; the superseded
#: sealed archives stay intact and are not consumed).
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
}

#: The sealed archives consumed read-only (their seals verify).  The
#: corrected sealed R41 archive is the frozen-parameter source; the
#: corrected R42 and R43 archives carry the previously-used seed blocks
#: (the disjointness check) and the condition cross-references (R43
#: training, R41 stability); the R32 spread is the model-uncertainty
#: carry of obligation 4.  The superseded sealed R41/R42/R43 archives
#: (the SUPERSEDED_* constants above) stay intact and are not consumed.
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23",
    "dev/00runs/r42_recoil_decay_separation_2026-09-23",
    "dev/00runs/r43_cascade_ablation_2026-09-23",
)
R41_RUN_DIR = "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"
R42_RUN_DIR = "dev/00runs/r42_recoil_decay_separation_2026-09-23"
R43_RUN_DIR = "dev/00runs/r43_cascade_ablation_2026-09-23"
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R41_FIT_REPORT_NAME = "fit_report.json"
#: The corrected sealed R41 fit report (the card-16 relitigation
#: landing): the file digest (byte identity) and the report's
#: self-digest (the gate cross-reference of the sealed R41).  The
#: superseded card-12 digests are retained as the
#: SUPERSEDED_R41_FIT_REPORT_* constants above.
R41_FIT_REPORT_FILE_SHA256 = (
    "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
)
R41_FIT_REPORT_DIGEST = (
    "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
)
R41_CARD = "TASK-20260920-r4-run-r41-nonstrange-pilot-fit"
R41_GATE_NAME = "gate4_r41.json"
R43_GATE_NAME = "gate4_r43.json"

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


# ---------------------------------------------------------------------------
# The frozen parameters (the corrected sealed R41 archive is the source;
# the values are pinned here and asserted against the archive before any
# generation; no refit on this run, plan section 10 line 802.  The
# superseded card-15 tuple is retained above as SUPERSEDED_R41_FITTED)
# ---------------------------------------------------------------------------
R41_FITTED = {
    "kappa_B": 0.0,
    "kappa_m": 4.0,
    "b_dec": 2.0,
    "baryonic_nominal_width": 0.3,
}
FROZEN_WIDTH_GEV = float(R41_FITTED["baryonic_nominal_width"])
#: The blend ramp start stays fixed at B (obligation 4; excluded from
#: the R41 fit set).
BLEND_RAMP_START_STATUS = (
    "the blend ramp start stays fixed at B = "
    f"{BLEND_RAMP_START_GEV!r} GeV (obligation 4; excluded from the "
    "R41 fit set, fit_set.excluded.blend_ramp_start of the sealed R41 "
    "pre-registration)"
)


class VariantPolicy:
    """One pre-registered decay configuration of the comparison."""

    __slots__ = ("name", "variant_id", "kappa_b", "kappa_m", "b_dec")

    def __init__(self, name: str, variant_id: int, kappa_b: float,
                 kappa_m: float, b_dec: float) -> None:
        self.name = str(name)
        self.variant_id = int(variant_id)
        self.kappa_b = float(kappa_b)
        self.kappa_m = float(kappa_m)
        self.b_dec = float(b_dec)


#: The variant register (the register order fixes the variant ids 0, 1):
#: the frozen aligned configuration (variant 0) and the isotropic
#: sequential baseline (variant 1, card 10's v2 policy).
ALIGNED_POLICY = VariantPolicy(
    "aligned_frozen", 0,
    R41_FITTED["kappa_B"], R41_FITTED["kappa_m"], R41_FITTED["b_dec"],
)
BASELINE_POLICY = VariantPolicy("isotropic_sequential_baseline", 1, 0.0, 0.0, 1.0)
VARIANTS: tuple[VariantPolicy, ...] = (ALIGNED_POLICY, BASELINE_POLICY)
VARIANT_NAMES: tuple[str, ...] = tuple(p.name for p in VARIANTS)
VARIANTS_NOTE = (
    "the card's two configurations on identical exchange events: v0 "
    "aligned_frozen (card 4's sequential cascade at the frozen R41 "
    "fitted point kappa_B = 0.0, kappa_m = 4.0, b_dec = 2.0 GeV^-2, the "
    "card-8 form factor attached at b_dec = 2.0 GeV^-2, the frozen "
    "baryonic nominal_width recorded at the corrected fitted value "
    "0.3 GeV; the frozen-parameter source is the corrected sealed R41 "
    "archive, no refit), v1 "
    "isotropic_sequential_baseline (card 10's v2 policy at kappa_B = "
    "kappa_m = 0 with b_dec = 1.0 GeV^-2; it carries nominal_width as "
    "its single parameter, recorded at the same corrected frozen value "
    "0.3 GeV).  Both variants share the continuum-daughter mass rule "
    "of the byte-unchanged committed continuum config and the same "
    "support (at the superseded card-15 landing the fitted width 0.5 "
    "GeV coincided with the committed config value); kappa_B, kappa_m, "
    "and b_dec alter probability densities but not kinematic support "
    "(plan section 5.3 line 288), so no regeneration is required and "
    "the comparison is a frozen-parameter evaluation of both "
    "configurations"
)


def _check_baseline_matches_card10() -> None:
    """The baseline policy is card 10's v2 policy (the frozen card-10
    register read-only; the driver refuses on any drift)."""
    v2 = ABLATION_POLICIES["isotropic_sequential"]
    if (float(v2.kappa_b), float(v2.kappa_m), float(v2.b_dec)) != (
        BASELINE_POLICY.kappa_b, BASELINE_POLICY.kappa_m, BASELINE_POLICY.b_dec,
    ):
        _refuse(
            "the pre-registered baseline policy "
            f"{(BASELINE_POLICY.kappa_b, BASELINE_POLICY.kappa_m, BASELINE_POLICY.b_dec)!r} "
            "does not match card 10's isotropic_sequential policy "
            f"{(float(v2.kappa_b), float(v2.kappa_m), float(v2.b_dec))!r}; "
            "the card-10 register drifted"
        )


_check_baseline_matches_card10()

#: The frozen-parameter assertion (the card: the frozen R41 parameter
#: values are asserted against the sealed R41 archive before any
#: generation; a drift refuses the run).
FROZEN_PARAMETER_NOTE = (
    "the frozen aligned configuration carries (kappa_B, kappa_m, "
    "b_dec, baryonic_nominal_width) = "
    f"({ALIGNED_POLICY.kappa_b!r}, {ALIGNED_POLICY.kappa_m!r}, "
    f"{ALIGNED_POLICY.b_dec!r}, {FROZEN_WIDTH_GEV!r}), the R41 fitted "
    "values frozen at the R41 seal; the values are asserted against "
    "the sealed R41 archive's fit report (the 'fitted' block) before "
    "any generation; no refit happens on this run (plan section 10, "
    "line 802; no_fitting is true and optimizer_invoked is false)"
)


def frozen_parameters_from_fit_report(report: dict) -> dict:
    """The four fitted values of an R41-style fit report (the frozen-
    parameter source block)."""
    fitted = report.get("fitted")
    if not isinstance(fitted, dict):
        raise ValueError("the fit report carries no 'fitted' block")
    return {
        "kappa_B": float(fitted["kappa_B"]),
        "kappa_m": float(fitted["kappa_m"]),
        "b_dec": float(fitted["b_dec"]),
        "baryonic_nominal_width": float(fitted["baryonic_nominal_width"]),
    }


def assert_frozen_parameters(report: dict) -> dict:
    """Assert the frozen parameters against an R41-style fit report
    (the sealed R41 archive in production, the pinned fixture in the
    smoke).  Returns the frozen-parameter block; any drift refuses."""
    try:
        observed = frozen_parameters_from_fit_report(report)
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
    if observed != R41_FITTED:
        _refuse(
            "the sealed R41 fitted values "
            f"{observed!r} do not match the pre-registered frozen "
            f"parameters {R41_FITTED!r}; the frozen-parameter source "
            "drifted (the run refuses to refit or to adapt)"
        )
    if (observed["kappa_B"], observed["kappa_m"], observed["b_dec"]) != (
        ALIGNED_POLICY.kappa_b, ALIGNED_POLICY.kappa_m, ALIGNED_POLICY.b_dec,
    ):
        _refuse("the frozen aligned policy does not match the R41 "
                "fitted values")
    if observed["baryonic_nominal_width"] != FROZEN_WIDTH_GEV:
        _refuse("the frozen nominal width does not match the R41 "
                "fitted width")
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
    "fitted": dict(R41_FITTED),
    "no_refit": True,
    "optimizer_invoked": True,
}

#: The obligation 4 model-uncertainty carry (the sealed R32 cross-variant
#: spread), verbatim from the card.
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}
OBLIGATION4_FORM = (
    "the sealed R32 cross-variant spread stands as the model-uncertainty "
    "term of the held-out comparison (the audits closed with card 2 and "
    "card 3 PASS and card 1 RECOMMENDATION, its rework card 1r landed "
    "before this run with no new spread diagnostic, so the original R32 "
    "spread is the final form); the held-out verdict is read against "
    "the recorded R32 spread and the bounds are never relaxed"
)
OBLIGATION5_NOTE = (
    "obligation 5 (the calculation-engine moment-approximation floor, "
    "~1e-3 GeV/c) is a distinct engine-family systematic tracked under "
    "_knowledge/evidence/engine-moment-approximation-floor.md; it is "
    "not conflated with the R32 spread in this run's record"
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

#: The pre-registered held-out bounds (never relaxed).
HELD_OUT_DEGRADATION_FACTOR = 2.0
HELD_OUT_DEGRADATION_NOTE = (
    "per held-out observable, the frozen aligned configuration's "
    "deviance is recorded against the isotropic sequential baseline; a "
    "held-out observable whose deviance degrades (the aligned deviance "
    "exceeds the baseline deviance) by more than 2.0 times the pooled "
    "MC standard error of the difference is a named finding (the "
    "pre-registered named-residual register).  The deviance of an "
    "observable is the summed squared normalized residual (d - m)^2 / "
    "sigma^2 over its finite bins, with sigma^2 = v_data + se^2 (the "
    "sealed R01 published variance plus the between-replica MC variance "
    "of that variant, floored at the engine's DETERMINISTIC_SE_FLOOR).  "
    "The pooled MC standard error of the difference is the between-"
    "replica standard error (std ddof=1 over the 20 paired replica "
    "blocks, divided by sqrt(20), floored at the engine's deterministic "
    "SE floor) of the per-block deviance differences; the exchange "
    "events are identical across the variants, so the per-block "
    "differences are paired.  The bound is pre-registered, never relaxed"
)
SECTOR_DEGRADATION_FRACTION = 0.05
SECTOR_DEGRADATION_NOTE = (
    "condition 4 (plan section 5.5 line 320): no previously acceptable "
    "sector degrades by more than 5% in validation deviance without a "
    "stated physics reason.  The sectors are the six paper-window "
    "species (the per-species spectrum deviances against the sealed R01 "
    "published data) and the feed-down fractions under the card-9 "
    "convention (the per-species resonant feed-down fractions; the "
    "fractions carry no published data vector, so their validation "
    "quantity is the fraction itself and a degradation is the relative "
    "change of the pooled fraction between the frozen aligned "
    "configuration and the baseline beyond 5%).  A violating sector is "
    "a named finding with the physics reason stated by the run record, "
    "or a Gate 4 finding if no reason exists"
)
SIGMA_TOLERANCE_NOTE = (
    "the per-bin named residuals carry the house named-residual form of "
    f"the fixed engine: |residual| > {SIGMA_TOLERANCE!r} sigma on the "
    "held-out data-backed bins, with sigma^2 = v_data + se^2 (the "
    "published variance plus the between-replica MC variance, floored "
    "at the engine's deterministic SE floor); SIGMA_TOLERANCE = 3.0 is "
    "the fixed engine constant, never relaxed"
)

EVENT_CLOSURE_TOL = 1e-9

#: Generation chunking (the R01/R23/R32/R43 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 8

#: The stage label of the R44 exchange topology.
STAGE_LABEL = "r44_frozen_nonstrange_validation"

#: The clean-environment validation's pre-registered probe replica.
PROBE_REPLICA = 0

GATEWAY_FALLBACK_RULE = (
    "local generation is primary (the measured local envelope in "
    "R41/R42: 100,000 p+p events ~ 75-78 s at 8 fork workers; the "
    "two-variant decay multiplies the per-event cost by ~2 over the "
    "R41/R42 single-variant envelope, giving 5M p+p events ~ 1.5-2 h "
    "wall, chunked by replica); if any replica exceeds 30 min wall, the "
    "remaining replicas run on the cluster gateway (host cl, workers "
    "c0801-c0809 x 32 task slots, one job per replica, --tasks 1, "
    "single math thread (NA49_CLUSTER_GW_TASKS forces thread counts to "
    "1), no local process pool inside the chunk command, submission "
    "paced by the capacity check; chunk outputs are appended only to "
    "the new run dir)"
)


# ---------------------------------------------------------------------------
# Seeds (round 4 index item 9, the card's pre-registered map; the
# previously-used blocks are read from the sealed R41/R42/R43
# pre-registrations at pre-registration time and recorded verbatim)
# ---------------------------------------------------------------------------
MASTER_SEED = 20260920
SEED_BASE = 20260920
SEED_OFFSET = 40_000
SEED_STRIDE = 100
T3_REPLICAS = 20
T3_EVENTS_PER_REPLICA = 250_000
RAW_SEED_RULE = (
    "raw replica seed = 20260920 + 40000 + 100 + k = 20301020 + k per "
    "replica k < 20 (the block 20301020-20301039, the R41/R42/R43 block "
    "20301020-20301027 extended to twenty slots by the card's "
    "pre-registration); the block is disjoint from the reserved audit "
    "block 20301040 + k (k = 0..15, 20301040..20301055, never fired) "
    "and from every previously used raw block outside the shared "
    "20301020 prefix (R23 20300913-20300920 under the 20260913 master; "
    "the R32 production and confirmation blocks under the 20260918 "
    "master; the R40 block 20301020-20301023, R41 20301020-20301027, "
    "R42 20301020-20301027, and R43 20301020-20301027 share the raw "
    "seeds by the established convention).  The (seed, event-index) "
    "substreams are disjoint because the event-index partitions are "
    "disjoint: R44's 700_000_000 + i (i < 250,000) differs from R41's "
    "400_000_000 + i, from R42's 500_000_000 + i, from R43's "
    "600_000_000 + i, from the R40 partition 300_000_000 + "
    "s*10_000_000 + p*1_000_000 + i (s < 9, p < 10, i < 25,000, so the "
    "R40 space stays below 389,025,000), and from the R43 smoke "
    "partition 690_000_000 + i (i < 2,000); if any raw (seed, "
    "event-index) substream collides, the run stops and reports (never "
    "a silent re-block)"
)
EXCHANGE_SEED_RULE = (
    "the exchange layer consumes the builder's per-event derivation at "
    "the pre-registered triple (master_seed 20260920, replica k, event "
    "index 700_000_000 + i): the legacy per-event generator "
    "numpy.random.SeedSequence(seed, spawn_key=(event_id,)) for the M8 "
    "mass sampler and the exact two-body scattering, and "
    "derive_substreams(20260920, k, 700_000_000 + i) for the card-6 "
    "routing and the catalog record draw (the record draw on the "
    "one-step spawned child of the decay_channel labeled id, the "
    "R21/R22/R23 mixture-layer convention; the spawned decay_kinematics "
    "child consumes nothing in R44 -- the decay draws are the per-"
    "variant spawns' on the same closed labels)"
)
VARIANT_STREAM_RULE = (
    "the per-variant decay streams are card 10's child SeedSequence "
    "spawns keyed on the variant id: variant_stream(20260920, "
    "700_000_000 + i, v) with v in {0, 1} (the register order: 0 "
    "aligned_frozen, 1 isotropic_sequential_baseline), the one-step "
    "form SeedSequence(master_seed, spawn_key=(event_index, "
    "variant_id)).spawn(2) on the closed decay_channel/"
    "decay_kinematics labels; the streams are disjoint across the "
    "variant ids (the decay uniforms of one variant never leak into "
    "the other) and deterministic on the pinned key.  The variant-spawn "
    "tree is a separate SeedSequence tree from the exchange derivation "
    "(card 10's documented contract: no decay uniform of the exchange "
    "derivation leaks into a variant stream or the other way around)"
)
EVENT_INDEX_RULE = "700_000_000 + i, i < 250,000"
EVENT_INDEX_BASE = 700_000_000
RAW_REPLICA_SEEDS = tuple(SEED_BASE + SEED_OFFSET + SEED_STRIDE + k
                          for k in range(T3_REPLICAS))

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
        "source": "sealed dev/00runs/r40_fixed_source_alignment_scan_2026-09-21/pre_registration.json",
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
    r44_lo, r44_hi = EVENT_INDEX_BASE, EVENT_INDEX_BASE + T3_EVENTS_PER_REPLICA
    # -- the audit block is disjoint from the R44 raw seed block --------
    audit_lo, audit_hi = AUDIT_RESERVED_BLOCK
    if not (max(RAW_REPLICA_SEEDS) < audit_lo or min(RAW_REPLICA_SEEDS) > audit_hi):
        problems.append(
            f"the R44 raw block {min(RAW_REPLICA_SEEDS)}..{max(RAW_REPLICA_SEEDS)} "
            f"overlaps the reserved audit block {audit_lo}..{audit_hi}"
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
        if r44_lo < p_hi and p_lo < r44_hi:
            problems.append(
                f"the R44 event-index partition [{r44_lo}, {r44_hi}) "
                f"overlaps the {name} partition [{p_lo}, {p_hi})"
            )
    if shared:
        for name, (lo, hi) in sorted(shared.items()):
            print(
                f"[r44] seed disjointness: the raw seeds {lo}..{hi} are "
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
        "r44_raw_seed_block": [min(RAW_REPLICA_SEEDS), max(RAW_REPLICA_SEEDS)],
        "r44_event_index_partition": EVENT_INDEX_RULE,
        "r44_event_index_range": [r44_lo, r44_hi],
        "audit_reserved_block": [audit_lo, audit_hi],
        "audit_block_disjoint": True,
        "shared_raw_seeds": shared,
        "shared_raw_seed_note": (
            "the shared raw seeds 20301020-20301027 (R41/R42/R43) and "
            "20301020-20301023 (R40) are the established block-sharing "
            "convention; the substreams are disjoint because the "
            "event-index partitions are disjoint (the builder's legacy "
            "generator SeedSequence(seed, spawn_key=(event_id,)) and "
            "derive_substreams(20260920, k, event_id) both key on the "
            "event index)"
        ),
        "previously_used_blocks": PREVIOUS_USED_BLOCKS,
        "disjoint": True,
    }


def raw_replica_seed(replica: int) -> int:
    r = int(replica)
    if not 0 <= r < T3_REPLICAS:
        _refuse(f"replica index {r} outside the T3 block 0..{T3_REPLICAS - 1}")
    return RAW_REPLICA_SEEDS[r]


def event_index(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_index(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


def smoke_raw_seed() -> int:
    return SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE


#: The hermetic smoke mode (driver validation; never sealed): 1 replica
#: x 2,000 events at the pre-registered smoke seeds.  The raw smoke seed
#: sits in the +10000 block (20310920, disjoint from every production
#: slot 0..19, from the R43/R42 +9000 smoke block, from the R41 +8000,
#: the R40 +7000, and the R32 +5000 smoke blocks, and from the reserved
#: audit block 20301040..20301055); the smoke event-index partition is
#: the +90M block 790_000_000 + i with i < 2,000, disjoint from the R44
#: production partition (700_000_000 + i, i < 250,000) and from every
#: previously used block, so the smoke exchange and variant streams
#: never touch a production space.
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 10_000
SMOKE_RAW_SEED = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE
SMOKE_EVENT_INDEX_BASE = 790_000_000
SMOKE_EVENT_INDEX_RULE = "790_000_000 + i, i < 2,000"
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260920 + 40000 + 10000 = 20310920 (the +10000 "
    "smoke block, one replica; the R43/R42 smoke block is +9000, the "
    "R41 smoke block +8000, the R40 smoke block +7000, and the R32 "
    "smoke block +5000); the builder's exchange derivation runs at "
    "derive_substreams(20260920, 10000, 790_000_000 + i) and the "
    "per-variant decay streams at variant_stream(20260920, 790_000_000 "
    "+ i, v) on the +90M smoke event-index block, disjoint from the "
    "R44 production partition (700_000_000 + i, i < 250,000), from "
    "R43's 690_000_000 smoke block, from R42's 500_000_000 block, from "
    "R41's 400_000_000 block, from the R40 partition, from the reserved "
    "audit block 20301040..20301055, and from every previously used "
    "block"
)
SMOKE_NOTE = (
    "the smoke exercises pre-registration, the frozen-parameter "
    "assertion (against the pinned R41 fixture), generation, the "
    "two-variant decay, and the held-out evaluation end to end, "
    "hermetically (the smoke data is the synthetic placeholder of the "
    "report machinery, no SPS table, no sealed archive); two identical "
    "smoke invocations produce byte-identical artifacts; smoke "
    "artifacts are refused by --gate / --seal / --clean-validate (the "
    "R31/8c convention)"
)

T3_LABEL = (
    "T3: 20 replicas x 250,000 p+p exchange events (plan section 5.1 "
    "line 255, the upper T3 bound, the R23/R32/R43 convention); "
    "5,000,000 exchange events total, each decayed through both "
    "configurations from the same per-event seed (identical exchange "
    "events, the card's baseline-comparison premise)"
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
# Species / quantum-number conventions (the R01/R32/R43 driver
# conventions; the pi0 block is the model-only held-out diagnostic)
# ---------------------------------------------------------------------------
SPECIES: tuple[tuple[str, int], ...] = (
    ("pi_plus", 211),
    ("pi_minus", -211),
    ("k_plus", 321),
    ("k_minus", -321),
    ("proton", 2212),
    ("antiproton", -2212),
    ("pi_zero", 111),
)
SPECIES_KEYS: tuple[str, ...] = tuple(name for name, _ in SPECIES)
SPECIES_PDGS: dict[str, int] = {name: int(pdg) for name, pdg in SPECIES}

#: The data-backed species of the published surface (the R01 tables) and
#: the model-only species (the published data surface carries no pi0
#: table; the pi0 distribution and moment are recorded model-side).
DATA_SPECIES: tuple[str, ...] = (
    "pi_plus", "pi_minus", "proton", "antiproton",
)
MODEL_ONLY_SPECIES: tuple[str, ...] = ("pi_zero",)
HELD_OUT_SPECTRA_KEYS: tuple[str, ...] = tuple(
    f"{name}_spectrum" for name in DATA_SPECIES
)
HELD_OUT_MOMENT_KEYS: tuple[str, ...] = tuple(
    f"{name}_multiplicity" for name in DATA_SPECIES
)
HELD_OUT_KEYS: tuple[str, ...] = HELD_OUT_SPECTRA_KEYS + HELD_OUT_MOMENT_KEYS
MODEL_ONLY_KEYS: tuple[str, ...] = (
    "pi_zero_spectrum", "pi_zero_multiplicity",
)
#: The card's held-out species naming (round 4 index item 11).
HELD_OUT_SPECIES_NAMED = ("pi+", "pi-", "pi0", "p", "pbar")
HELD_OUT_SET_NOTE = (
    "the card's held-out set (round 4 index item 11): the species-"
    "separated pion distributions (pi+, pi-, pi0) and proton "
    "distributions (p, pbar), and the species-separated multiplicity "
    "moments, held out relative to R41's fit set (R41 trains the "
    "charge-symmetric combinations of plan section 7.3: pi_Sigma = "
    "pi+ + pi-, B_transport = p - pbar, the pooled pion and proton "
    "seagulls, the pooled multiplicity moments).  The published data "
    "surface (the sealed R01 archive) carries the six paper-window "
    "species only, so the likelihood register is the data-backed pi+, "
    "pi-, p, pbar spectra and moments, and the pi0 distribution and "
    "moment are recorded model-side as held-out diagnostics (no "
    "published pi0 table exists in the pipeline data; the pi0 rows are "
    "never dropped silently -- they carry the model-only marker and no "
    "likelihood contribution)"
)
#: The R41 training observables (the held-out set's complement; the
#: disjointness invariant of the held-out register).
R41_TRAINING_OBSERVABLES = (
    "pi_sigma_spectrum",
    "b_transport_spectrum",
    "pi_sigma_seagull",
    "b_transport_seagull",
    "pi_sigma_multiplicity",
    "b_transport_multiplicity",
)
R41_TRAINING_COMBINATIONS = {
    "pi_sigma": ("pi_plus", "pi_minus", +1.0),
    "b_transport": ("proton", "antiproton", -1.0),
}
HELD_OUT_DISJOINT_NOTE = (
    "the held-out register is species-separated: no held-out key is an "
    "R41 training observable or a charge-symmetric combination of plan "
    "section 7.3 (pi_Sigma = pi+ + pi-, B_transport = p - pbar, the "
    "pooled seagulls, the pooled multiplicity moments); the held-out "
    "spectra are the individual species' published tables and the "
    "held-out moments are the individual species' window integrals"
)

#: The pi0 mass (the model-only diagnostic's mass; the pdg is 111).
PI_ZERO_MASS_GEV = 0.1349766
PI_ZERO_TABLE_PATH = "model://pi_zero_held_out_diagnostic"
PI_ZERO_TABLE_NOTE = (
    "the model-only pi0 table: the published pi+ table's grid geometry "
    "(the paper-window binning preserved) at the pi0 mass and pdg 111; "
    "the published data surface carries no pi0 table, so the pi0 rows "
    "carry no data vector and no likelihood contribution (the model-"
    "only held-out diagnostic of the card's held-out set)"
)

#: The species offsets in the flat global bin space (the table order of
#: SPECIES_FLAT_ORDER, the R01 pT-major flat layout per species, the
#: model-only pi0 block appended last).
SPECIES_FLAT_ORDER = ("pi_minus", "pi_plus", "k_plus", "k_minus",
                      "proton", "antiproton", "pi_zero")
SPECIES_FLAT_OFFSETS: dict[str, int] = {}
_SPECIES_SIZE: dict[str, int] = {}


def species_offset(species: str) -> int:
    """The flat-row offset of a species in the concatenated bin space."""
    return SPECIES_FLAT_OFFSETS[species]


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


edges_by_species: dict[str, tuple] = {}
n_x_by_species: dict[str, int] = {}
n_p_by_species: dict[str, int] = {}


def _worker_context(tables: list, fams: list, smoke: bool) -> dict:
    catalog = _load_catalog()
    dense_offsets_by_species: dict[str, int] = {}
    dense_sizes_by_species: dict[str, int] = {}
    pick_by_species: dict[str, np.ndarray] = {}
    dense_offset = 0
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
        #    pipeline): the finals accumulate into the species' dense
        #    pT-major lattice (n_p * n_x cells) and the published rows
        #    are then picked from the dense totals at the published
        #    rows' cells, pick = bin_index_pt * n_x + bin_index_xf.
        #    The published R01 tables are non-rectangular subsets of
        #    their lattices (holes; rows not pT-major), so a dense cell
        #    address and a published row address are different
        #    addresses; the pick map is what keeps the recorded
        #    pooled_sums aligned 1:1 with the published rows the
        #    evaluation consumes, and a hole cell (a lattice cell with
        #    no published row) stays unpicked -- its weight is dropped,
        #    never bled into a neighboring row or species. --
        n_x = n_x_by_species[table.species]
        n_p = n_p_by_species[table.species]
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
        "dense_total": int(dense_offset),
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
    loader).  The production generation and the evaluation read the SPS
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


def _pi_zero_model_table(data_tables: list):
    """The model-only pi0 table: the published pi+ table's grid geometry
    at the pi0 mass and pdg (the card's held-out pi0 territory; no data
    vector, no likelihood contribution)."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    pi_plus = next(t for t in data_tables if t.species == "pi_plus")
    n_bins = int(pi_plus.data.size)
    xf_pts = np.asarray(pi_plus.xf_centers, dtype=float)[
        np.asarray(pi_plus.bin_index_xf, dtype=int)]
    pt_pts = np.asarray(pi_plus.pt_centers, dtype=float)[
        np.asarray(pi_plus.bin_index_pt, dtype=int)]
    x_edges = sps_core.edges_from_centers(
        np.asarray(pi_plus.xf_centers, dtype=float))
    p_edges = sps_core.edges_from_centers(
        np.asarray(pi_plus.pt_centers, dtype=float))
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
            f"{int(table.data.size)} does not match the pi+ grid's "
            f"{n_bins}"
        )
    return table


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
    tables.append(_pi_zero_model_table(tables))
    return tables


# ---------------------------------------------------------------------------
# Species / quantum-number conventions (continued)
# ---------------------------------------------------------------------------
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


def _state_parent_pdg(state) -> int:
    """The producing-vertex parent pdg of a state's decay products (the
    card-9 feed-down lineage convention): the catalog record's pdg for a
    catalog parent, the continuum root sentinel 0 for a continuum
    parent (the continuum state is not a catalog resonance)."""
    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        return int(family_id[len(CATALOG_FAMILY_PREFIX):])
    return int(CONTINUUM_ROOT_SENTINEL)


# ---------------------------------------------------------------------------
# The per-event exchange layer (the R32 p+p configuration, index item 7)
# ---------------------------------------------------------------------------
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any] | None) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


class ConservationViolation(RuntimeError):
    """The pre-registered conservation check of the card fired (checked,
    never repaired; a violation stops the run)."""


def _hex(x: float) -> str:
    """The exact hex form of a float component (the canonical payload)."""
    return float(x).hex()


def exchange_event_hash(
    exchange_events: list, event_id: int, t: float, route_census: list,
) -> str:
    """The canonical per-event exchange-layer hash: the event id, the
    realized exchange t, the route census, and every source's exchange
    event with exact hex components (topology, momentum, family_id,
    mass, qn, axis, ancestry).  Pure: the same mapping bytes reproduce
    the digest byte for byte, so a variant that received a mutated or
    different mapping produces a different digest (the bit-identity
    evidence at run scale)."""
    digest = hashlib.sha256()
    digest.update(
        f"r44|{int(event_id)}|{_hex(float(t))}|"
        f"{route_census[0]},{route_census[1]},{route_census[2]}|".encode(
            "utf-8"))
    for event in exchange_events:
        topology = str(event["topology"])
        momentum = tuple(float(v) for v in event["momentum"])
        mass = float(event["mass"])
        family_id = str(event["family_id"])
        qn = tuple(int(v) for v in event["qn"])
        axis = tuple(float(v) for v in event["axis"])
        ancestry = tuple(str(item) for item in event.get(
            "ancestry", (topology,)))
        digest.update(
            f"{topology}|{family_id}|{_hex(mass)}|"
            f"{tuple(int(v) for v in event['qn'])}|".encode("utf-8"))
        digest.update("|".join(_hex(v) for v in momentum).encode("utf-8"))
        digest.update(b"|")
        digest.update("|".join(_hex(v) for v in axis).encode("utf-8"))
        digest.update(b"|")
        digest.update("|".join(ancestry).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class _EventCapture:
    """The per-event exchange layer of one generated event.

    ``sources`` carries one record per p+p source: the exchange-event
    mapping (the card-10 pinned keys), the card-6 route, the drawn
    catalog record identity when the catalog route fired, and the
    stable-identity flag.  The exchange layer is generated once per
    event and shared across both variants; no per-variant draw is
    consumed here.
    """

    __slots__ = (
        "event_id", "sources", "t", "route_census", "energy_expected",
        "exchange_hash",
    )

    def __init__(self) -> None:
        self.event_id = 0
        self.sources: list[dict] = []
        self.t = 0.0
        self.route_census = [0, 0, 0]  # sources, catalog, continuum
        self.energy_expected = 0.0
        self.exchange_hash = ""


def generate_exchange_event(
    catalog: HadronCatalog,
    pp_config,
    blend: BlendSpec,
    fams: list,
    seed: int,
    event_id: int,
    master_seed: int,
    replica_slot: int,
) -> _EventCapture:
    """One p+p exchange event of the R44 configuration (index item 7):
    the frozen builder source layer bit for bit at the config's exchange
    slope (the M8 mass sampler and the exact two-body scattering on the
    shared legacy generator, the retry discipline), the card-6 routing
    at the frozen source mass, and the catalog route's record draw on
    the builder's spawned decay_channel child (the R32 builder
    convention: the open-set mixture record draw including the stable
    records; a drawn stable record takes the card-9 stable-identity
    path -- the source contributes itself as the single final at its
    own four-momentum, a deterministic draw-free variant-independent
    terminal product).  The exchange layer is generated once per event
    and shared across the two variants; no per-variant draw is
    consumed here."""

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
    capture = _EventCapture()
    capture.event_id = int(event_id)
    capture.t = float(t)
    fam_b = active_families(list(fams), (1, 1, 0))
    if len(fam_b) != 1:
        _refuse(
            "r44: the blend routing expects exactly one activated "
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
        if route == "catalog":
            capture.route_census[1] += 1
            records = builder.open_records(mass)
            if not records:
                raise RuntimeError(
                    "r44 exchange: the open set at source quantum numbers "
                    f"(1, 1, 0) and mass {mass!r} GeV is empty; the "
                    "pre-registered mixture model has no record to draw "
                    "(the run stops)"
                )
            index = min(
                int(builder._rng_record.random() * len(records)),
                len(records) - 1,
            )
            mix_record = records[int(index)]
            stable = bool(
                mix_record.stable_for_decay
                or int(mix_record.pdg_id) in catalog.stable_states()
            )
            source_entry = {
                "source_id": int(sid),
                "role": str(role),
                "exchange": {
                    "topology": STAGE_LABEL,
                    "momentum": p4t,
                    "family_id":
                        f"{CATALOG_FAMILY_PREFIX}{int(mix_record.pdg_id)}",
                    "mass": mass,
                    "qn": (1, 1, 0),
                    "axis": initial_source_axis(
                        p4t,
                        beam_direction=(
                            (0.0, 0.0, 1.0) if role == "projectile"
                            else (0.0, 0.0, -1.0)
                        ),
                    ),
                    "ancestry": (STAGE_LABEL,),
                },
                "route": "catalog",
                "record_pdg": int(mix_record.pdg_id),
                "record_name": str(mix_record.name),
                "n_open_records": int(len(records)),
                "stable_identity": bool(stable),
                "energy": float(p4t[0]),
            }
        else:
            capture.route_census[2] += 1
            source_entry = {
                "source_id": int(sid),
                "role": str(role),
                "exchange": {
                    "topology": STAGE_LABEL,
                    "momentum": p4t,
                    "family_id": str(fam_b.family_id),
                    "mass": mass,
                    "qn": (1, 1, 0),
                    "axis": initial_source_axis(
                        p4t,
                        beam_direction=(
                            (0.0, 0.0, 1.0) if role == "projectile"
                            else (0.0, 0.0, -1.0)
                        ),
                    ),
                    "ancestry": (STAGE_LABEL,),
                },
                "route": "continuum",
                "record_pdg": None,
                "record_name": None,
                "n_open_records": None,
                "stable_identity": False,
                "energy": float(p4t[0]),
            }
        capture.sources.append(source_entry)
        capture.energy_expected += float(p4t[0])
    capture.exchange_hash = exchange_event_hash(
        [entry["exchange"] for entry in capture.sources],
        int(event_id), capture.t, capture.route_census,
    )
    return capture


# ---------------------------------------------------------------------------
# The per-variant decay capture (the event assembly, card 9's convention)
# ---------------------------------------------------------------------------
def _empty_feeddown(species_keys: tuple[str, ...]) -> dict:
    """The feed-down accumulator of one variant (the card-9 lineage
    convention: a final particle is resonant feed-down when its
    producing vertex's parent is a resonance other than the drawn
    root -- the catalog route's drawn mixture record, the continuum
    route's root sentinel 0; a final with no producing record is a
    direct source product, never a resonance feed-down)."""
    return {
        "n_finals": 0,
        "per_species": {
            key: {"n": 0, "n_resonant": 0, "by_parent": {}}
            for key in species_keys
        },
    }


class _VariantRun:
    """One variant's decay of one event: the finals capture and the
    conservation tallies.

    ``flat``/``weights`` hold the event's sparse (dense lattice cell,
    Jacobian weight) contributions of every final particle of both
    sources (the R01 binning convention: ``x_F = 2 p_z / sqrt(s)``, the
    pT-major dense lattice layout, the same in-bounds drop as the R42
    latent capture; the published rows are picked from the dense totals
    at record assembly, so a hole cell's weight is dropped there).  The
    energy and (B, Q, S) tallies feed the pre-registered
    per-event and per-source conservation checks; the multiplicity
    tallies feed the species-separated multiplicity moments; the
    feed-down tallies feed the card-9 lineage fractions; the depth
    histogram records the cascade depth statistic of the variant's
    decay machinery per decay; the rolling exchange digest folds the
    shared exchange-layer digest per event (the bit-identity evidence
    at run scale).
    """

    __slots__ = (
        "variant", "flat", "weights", "bqs", "energy_sum",
        "worst_closure", "n_finals", "mult_sums", "mult_sums2",
        "mult_with", "feeddown", "depth_hist", "mode_census",
        "exchange_rolling", "event_hash", "payload",
    )

    def __init__(self, variant: int) -> None:
        self.variant = int(variant)
        self.flat: list[int] = []
        self.weights: list[float] = []
        self.bqs = (0, 0, 0)
        self.energy_sum = 0.0
        self.worst_closure = 0.0
        self.n_finals = 0
        self.mult_sums: dict[str, float] = {k: 0.0 for k in SPECIES_KEYS}
        self.mult_sums2: dict[str, float] = {k: 0.0 for k in SPECIES_KEYS}
        self.mult_with: dict[str, int] = {k: 0 for k in SPECIES_KEYS}
        self.feeddown: dict = _empty_feeddown(SPECIES_KEYS)
        self.depth_hist: dict[int, int] = {}
        self.mode_census: dict[str, int] = {}
        self.exchange_rolling = hashlib.sha256()
        self.event_hash = ""
        self.payload: list[str] = []

    def add_final(
        self, pdg: int, p4: tuple[float, float, float, float],
        parent_pdg: int | None = None, root_pdg: int = 0,
    ) -> None:
        """One final particle: the (B, Q, S) tally, the energy tally,
        the multiplicity tally, the card-9 feed-down tally (a final is
        resonant feed-down when its producing vertex's parent is a
        resonance other than the drawn root: the catalog route's drawn
        mixture record pdg, the continuum route's sentinel 0), the
        canonical payload line, and the paper-window bin row (the R01
        binning convention: the same grid edges, the same
        ``sps_core.grid_indices`` binning, the pT-major flat layout, the
        same in-bounds drop).  The recorded address is the final's
        DENSE lattice cell (the species' dense offset plus
        ``ip * n_x + ix``); the published rows are picked from the dense
        totals at record assembly (the R01 ``_bin_track_sums``
        accumulate-dense-then-pick convention), so a hole cell's weight
        stays unpicked and is dropped, never bled into a neighboring
        published row or species."""
        context = _WORKER_CONTEXT
        assert context is not None
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
        self.feeddown["n_finals"] += 1
        species = _species_of(catalog, pdg)
        if species is not None:
            self.mult_sums[species] += 1.0
            self.mult_sums2[species] += 1.0
            self.mult_with[species] = self.mult_with.get(species, 0) + 1
            slot = self.feeddown["per_species"][species]
            slot["n"] += 1
            if parent_pdg is not None and int(parent_pdg) != int(root_pdg):
                slot["n_resonant"] += 1
                bucket = slot["by_parent"]
                key = str(int(parent_pdg))
                bucket[key] = bucket.get(key, 0) + 1
        pt = math.hypot(px, py)
        if pt <= 0.0:
            return
        table = context["table_by_pdg"].get(int(pdg))
        if table is None:
            return
        x_f = 2.0 * pz / context["sqrt_s"]
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

    def event_payload_digest(self, event_id: int) -> str:
        """The canonical per-event per-variant hash: the variant id, the
        event id, the mode census, the final count, and every final's
        pdg with exact hex four-momentum components (the rolling-digest
        convention of the R42 record identity)."""
        digest = hashlib.sha256()
        digest.update(
            f"r44|{self.variant}|{int(event_id)}|"
            f"{self.mode_census.get('catalog_handoff', 0)},"
            f"{self.mode_census.get('direct_channel', 0)},"
            f"{self.mode_census.get('stable_identity', 0)}"
            f"|{self.n_finals}|".encode("utf-8"))
        for line in self.payload:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def _state_pdg(state) -> int | None:
    """The catalog pdg behind a state's family id, or ``None`` for a
    continuum state."""
    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        return int(family_id[len(CATALOG_FAMILY_PREFIX):])
    return None

# ---------------------------------------------------------------------------
# The per-variant decay walks (card 10's machinery, the caller's assembly)
# ---------------------------------------------------------------------------
def _emit_assembly_state(
    catalog: HadronCatalog,
    capture: _VariantRun,
    state,
    rng_channel,
    rng_kinematics,
    root_pdg: int,
    parent_pdg: int | None,
) -> None:
    """The event assembly of one terminal established daughter state
    (card 9's caller-assembly clause, the R23/R32 ``_emit`` convention):
    a stable record contributes itself as a final (its producing-vertex
    parent is the state that produced it, ``parent_pdg``); an unstable
    catalog record decays through the Round 2 kernel's own
    ``decay_resonance`` on the same closed per-variant streams with the
    card-9 binding pattern (the kernel's decay logic, tables, and
    conservation checks run byte-unchanged), and its decay products'
    producing-vertex parent is the record itself.  The finals accumulate
    into the capture with their binning rows and the card-9 feed-down
    tallies against the source's drawn root ``root_pdg``."""
    fid = str(state.family_id)
    if not fid.startswith(CATALOG_FAMILY_PREFIX):
        raise ValueError(
            "r44 assembly: a continuum daughter state reached the event "
            f"assembly (family_id {fid!r}); the cascade line walks "
            "continuum states and emits only established daughters"
        )
    pdg = int(fid[len(CATALOG_FAMILY_PREFIX):])
    record = catalog.get(pdg)
    p4 = tuple(float(x) for x in state.momentum)
    if record.stable_for_decay or pdg in catalog.stable_states():
        capture.add_final(pdg, p4, parent_pdg=parent_pdg, root_pdg=root_pdg)
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
                    parent_pdg=int(pdg),
                    root_pdg=root_pdg,
                )


def _walk_sequential_captured(
    catalog: HadronCatalog,
    fams: list,
    exchange_event: dict,
    policy: VariantPolicy,
    rng_channel,
    rng_kinematics,
    capture: _VariantRun,
    root_pdg: int,
) -> str:
    """One variant's sequential decay with the event assembly: card 9's
    routing order with card 10's attached per-vertex step (the R40/R43
    walk shape) at the variant's (kappa_b, kappa_m, b_dec).  At every
    line state, in order: the hard depth guard (a
    ``CascadeDepthGuardError`` propagates and stops the run); the
    catalog handoff (:func:`~na49_mixing.cascade_termination.
    handoff_decay`, the byte-unchanged kernel delegation, whose
    non-resonance daughters are the finals); the continuation through
    one attached vertex over the open continuum pool (the continuum
    daughter continues the line, the established sibling is emitted
    into the assembly); the direct-channel termination through one
    attached vertex over the direct pool (both daughters are emitted);
    the no-channel refusal (``ValueError``).  The per-vertex budget is
    card 4's frozen 1 channel + 2 kinematics uniforms on the closed
    labels (the card-10 attachments consume none); the handoff and the
    assembly remainders consume the Round 2 kernel's own budget on the
    same streams.  Returns the decay mode."""
    root = exchange_root_state(exchange_event)
    current = root
    n_vert = 0
    while True:
        if int(current.depth) > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"r44: the line state of family {current.family_id!r} at "
                f"mass {float(current.mass)!r} GeV carries depth "
                f"{int(current.depth)}, exceeding MAX_CASCADE_DEPTH = "
                f"{MAX_CASCADE_DEPTH} (plan section 10.6, line 757); the "
                "run stops"
            )
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(
                current, catalog, rng_channel, rng_kinematics)
            parent = _state_pdg(current)
            for vertex in vertices:
                for daughter in vertex.daughters:
                    if not daughter.is_resonance:
                        capture.add_final(
                            int(daughter.pdg_id),
                            tuple(
                                float(x) for x in daughter.four_momentum),
                            parent_pdg=parent,
                            root_pdg=root_pdg,
                        )
            capture.depth_hist[0] = capture.depth_hist.get(0, 0) + 1
            return "catalog_handoff"
        channels = enumerate_channels(current, catalog, fams)
        cont = [c for c in channels if c.kind == KIND_CONTINUUM]
        pool = cont if cont else [
            channel for channel in channels
            if channel.kind in (KIND_ESTABLISHED_HADRON,
                                KIND_ESTABLISHED_RESONANCE)
        ]
        if not pool:
            raise ValueError(
                f"r44: the line state of family {current.family_id!r} at "
                f"mass {float(current.mass)!r} GeV carries no open "
                "continuum channel and no valid direct hadronic channel "
                "(plan section 10.6); checked, never repaired"
            )
        attached = _attach_form_factor(pool, current, float(policy.b_dec))
        _selected, (daughter0, daughter1) = _draw_attached_vertex(
            current, attached, catalog, fams,
            float(policy.kappa_b), float(policy.kappa_m),
            rng_channel, rng_kinematics,
        )
        n_vert += 1
        parent = _state_parent_pdg(current)
        if cont:
            successor = next(
                daughter for daughter in (daughter0, daughter1)
                if not str(daughter.family_id).startswith(
                    CATALOG_FAMILY_PREFIX)
            )
            sibling = daughter1 if successor is daughter0 else daughter0
            _emit_assembly_state(
                catalog, capture, sibling, rng_channel, rng_kinematics,
                root_pdg, parent)
            current = successor
            continue
        _emit_assembly_state(
            catalog, capture, daughter0, rng_channel, rng_kinematics,
            root_pdg, parent)
        _emit_assembly_state(
            catalog, capture, daughter1, rng_channel, rng_kinematics,
            root_pdg, parent)
        capture.depth_hist[n_vert] = capture.depth_hist.get(n_vert, 0) + 1
        return "direct_channel"


def decay_event_variants(
    catalog: HadronCatalog,
    fams: list,
    exchange: _EventCapture,
    smoke: bool = False,
) -> dict[str, dict]:
    """Decay one identical exchange event through both configurations
    from the same per-event seed (the card's baseline-comparison
    premise).

    The exchange layer is generated once per event and shared; the
    per-variant decay streams are card 10's child spawns keyed on the
    variant id (0, 1), disjoint across variants.  Every variant's
    finals are tallied into its own capture with the pre-registered
    per-source and per-event conservation checks (checked, never
    repaired; a violation stops the run), and each variant folds the
    shared exchange-layer digest into its own rolling exchange digest
    (the bit-identity evidence at run scale).  The card-9 stable-
    identity semantics apply before the variant loop: a drawn stable
    catalog record contributes the source itself as the single final at
    its own four-momentum, a deterministic draw-free variant-
    independent terminal product (no decay machinery runs on it; the
    feed-down tally records a direct source product).  The per-variant
    event hashes are keyed on the variant id.  Returns one entry per
    variant name with the capture parts.
    """
    event_id = int(exchange.event_id)
    out: dict[str, dict] = {}
    for policy in VARIANTS:
        variant_name = policy.name
        rng_channel, rng_kinematics = variant_stream(
            MASTER_SEED, event_id, policy.variant_id)
        run = _VariantRun(policy.variant_id)
        for source in exchange.sources:
            root_pdg = (
                int(source["record_pdg"]) if source["route"] == "catalog"
                else int(CONTINUUM_ROOT_SENTINEL)
            )
            snapshot = (run.n_finals, run.bqs, run.energy_sum)
            if source["stable_identity"]:
                # the card-9 stable-identity semantics: the drawn stable
                # record contributes the source itself as the single
                # final at its own four-momentum (deterministic,
                # draw-free, variant-independent)
                pdg = int(str(source["exchange"]["family_id"])[
                    len(CATALOG_FAMILY_PREFIX):])
                run.add_final(
                    pdg, tuple(float(x) for x in source["exchange"][
                        "momentum"]),
                    parent_pdg=None, root_pdg=root_pdg)
                run.mode_census["stable_identity"] = (
                    run.mode_census.get("stable_identity", 0) + 1)
                run.depth_hist[0] = run.depth_hist.get(0, 0) + 1
            else:
                mode = _walk_sequential_captured(
                    catalog, fams, source["exchange"], policy,
                    rng_channel, rng_kinematics, run, root_pdg)
                run.mode_census[mode] = run.mode_census.get(mode, 0) + 1
            # -- the per-source conservation check (the card's exact
            #    per-source clause; checked, never repaired) --
            delta = (
                run.bqs[0] - snapshot[1][0],
                run.bqs[1] - snapshot[1][1],
                run.bqs[2] - snapshot[1][2],
            )
            if delta != (1, 1, 0):
                raise ConservationViolation(
                    f"r44: event {event_id} variant {variant_name!r} "
                    f"source {source['source_id']} ({source['role']}) "
                    f"finals sum to (B, Q, S) = {delta} but the source "
                    "carries (1, 1, 0); conservation is checked, never "
                    "repaired (the run stops)"
                )
        # -- the per-event conservation check --
        if run.bqs != (2, 2, 0):
            raise ConservationViolation(
                f"r44: event {event_id} variant {variant_name!r} finals "
                f"sum to (B, Q, S) = {run.bqs} but the p+p pair carries "
                "(2, 2, 0); conservation is checked, never repaired "
                "(the run stops)"
            )
        closure = abs(run.energy_sum - exchange.energy_expected)
        run.worst_closure = closure
        if not closure <= EVENT_CLOSURE_TOL:
            raise ConservationViolation(
                f"r44: event {event_id} variant {variant_name!r} final "
                f"energy sum misses the source energy sum by {closure!r} "
                f"GeV (beyond {EVENT_CLOSURE_TOL}); no momentum repair "
                "(the run stops)"
            )
        # -- the bit-identity fold: the variant's own view of the shared
        #    exchange layer (recomputed after the decays, so a mutated
        #    mapping would surface here) --
        variant_exchange = exchange_event_hash(
            [entry["exchange"] for entry in exchange.sources],
            event_id, exchange.t, exchange.route_census,
        )
        if variant_exchange != exchange.exchange_hash:
            raise ConservationViolation(
                f"r44: event {event_id} variant {variant_name!r} received "
                "an exchange-event mapping whose canonical digest differs "
                "from the shared exchange layer's (the identical-"
                "exchange-events premise of the card's baseline "
                "comparison is violated); conservation of the comparison "
                "is checked, never repaired (the run stops)"
            )
        run.exchange_rolling.update(variant_exchange.encode("utf-8"))
        run.exchange_rolling.update(b"\n")
        run.event_hash = run.event_payload_digest(event_id)
        out[variant_name] = {
            "variant": policy.variant_id,
            "run": run,
        }
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
     blend_boundary, smoke) = task
    catalog = context["catalog"]
    pp_config = context["pp_config"]
    blend = BlendSpec(boundary=float(blend_boundary),
                      ramp_width=BLEND_RAMP_WIDTH_GEV)
    fams = list(context["families"])
    exchange_hashes: list[str] = []
    per_variant: dict[str, dict] = {
        policy.name: {
            "hashes": [], "event_ids": [],
            "row_flat": [], "row_weight": [],
            "row_ptr": [0], "routing": [0, 0, 0], "depth_hist": {},
            "mode_census": {}, "mult": {
                k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                for k in SPECIES_KEYS},
            "feeddown": _empty_feeddown(SPECIES_KEYS),
            "worst_closure": 0.0, "n_finals": 0, "exchange_digest": "",
        }
        for policy in VARIANTS
    }
    shared_exchange_rolling = hashlib.sha256()
    variant_exchange_rollings = {
        policy.name: hashlib.sha256() for policy in VARIANTS
    }
    for event_id in event_ids:
        exchange = generate_exchange_event(
            catalog, pp_config, blend, fams,
            seed=int(seed), event_id=int(event_id),
            master_seed=int(master_seed), replica_slot=int(replica_slot),
        )
        shared_exchange_rolling.update(exchange.exchange_hash.encode("utf-8"))
        shared_exchange_rolling.update(b"\n")
        variants = decay_event_variants(
            catalog, fams, exchange, smoke=bool(smoke))
        exchange_hashes.append(exchange.exchange_hash)
        for policy in VARIANTS:
            name = policy.name
            variant_exchange_rollings[name].update(
                exchange.exchange_hash.encode("utf-8"))
            variant_exchange_rollings[name].update(b"\n")
            run: _VariantRun = variants[name]["run"]
            slot = per_variant[name]
            slot["hashes"].append(run.event_hash)
            slot["event_ids"].append(int(event_id))
            slot["row_flat"].extend(run.flat)
            slot["row_weight"].extend(run.weights)
            slot["row_ptr"].append(slot["row_ptr"][-1] + len(run.flat))
            for k in range(3):
                slot["routing"][k] += int(exchange.route_census[k])
            for depth, count in run.depth_hist.items():
                slot["depth_hist"][int(depth)] = \
                    slot["depth_hist"].get(int(depth), 0) + int(count)
            for mode, count in run.mode_census.items():
                slot["mode_census"][mode] = \
                    slot["mode_census"].get(mode, 0) + int(count)
            for key in SPECIES_KEYS:
                slot["mult"][key]["sum_n"] += float(run.mult_sums[key])
                slot["mult"][key]["sum_n2"] += float(run.mult_sums2[key])
                slot["mult"][key]["n_with"] += int(run.mult_with[key])
            slot["feeddown"]["n_finals"] += int(run.feeddown["n_finals"])
            for key in SPECIES_KEYS:
                src_slot = run.feeddown["per_species"][key]
                dst_slot = slot["feeddown"]["per_species"][key]
                dst_slot["n"] += int(src_slot["n"])
                dst_slot["n_resonant"] += int(src_slot["n_resonant"])
                for parent, count in src_slot["by_parent"].items():
                    dst_slot["by_parent"][parent] = \
                        dst_slot["by_parent"].get(parent, 0) + int(count)
            slot["worst_closure"] = max(
                slot["worst_closure"], float(run.worst_closure))
            slot["n_finals"] += int(run.n_finals)
        for policy in VARIANTS:
            per_variant[policy.name]["exchange_digest"] = \
                variant_exchange_rollings[policy.name].hexdigest()
    # -- the chunk-level bit-identity consistency check (the honest
    #    per-event verification is the variant recomputation inside
    #    decay_event_variants; this compares the chunk-level folds of
    #    the same per-event hashes) --
    shared_chunk = shared_exchange_rolling.hexdigest()
    for policy in VARIANTS:
        if per_variant[policy.name]["exchange_digest"] != shared_chunk:
            raise RuntimeError(
                f"r44 chunk at event {event_ids[0]}: the per-variant "
                f"rolling exchange digest of {policy.name!r} differs from "
                "the shared one (the identical-exchange-events premise of "
                "the card's baseline comparison is violated); the run "
                "stops"
            )
    payload = {
        "replica": int(replica),
        "start_id": int(event_ids[0]),
        "n_events": len(event_ids),
        "exchange_hashes": exchange_hashes,
        "shared_exchange_digest": shared_exchange_rolling.hexdigest(),
        "per_variant": per_variant,
        "smoke": bool(smoke),
    }
    return payload


def _aggregate_chunks(parts: list) -> dict:
    """Merge the chunk parts into one replica's state (exact integer
    counts; float sums in chunk order, the chunks sorted by start id)."""
    parts = sorted(parts, key=lambda p: int(p["start_id"]))
    per_variant: dict[str, dict] = {}
    for policy in VARIANTS:
        slot = {
            "hashes": [], "row_flat": [], "row_weight": [],
            "row_ptr": [0], "routing": [0, 0, 0], "depth_hist": {},
            "mode_census": {}, "mult": {
                k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                for k in SPECIES_KEYS},
            "feeddown": _empty_feeddown(SPECIES_KEYS),
            "worst_closure": 0.0, "n_finals": 0,
        }
        per_variant[policy.name] = slot
    shared_exchange_rolling = hashlib.sha256()
    exchange_rollings = {
        policy.name: hashlib.sha256() for policy in VARIANTS
    }
    n_events = 0
    for part in parts:
        shared_exchange_rolling.update(
            part["shared_exchange_digest"].encode("utf-8"))
        shared_exchange_rolling.update(b"\n")
        n_events += int(part["n_events"])
        for policy in VARIANTS:
            name = policy.name
            exchange_rollings[name].update(
                part["shared_exchange_digest"].encode("utf-8"))
            exchange_rollings[name].update(b"\n")
            slot = per_variant[name]
            src = part["per_variant"][name]
            slot["hashes"].extend(src["hashes"])
            slot["row_flat"].extend(src["row_flat"])
            slot["row_weight"].extend(src["row_weight"])
            base = slot["row_ptr"][-1]
            slot["row_ptr"].extend(
                base + int(v) for v in src["row_ptr"][1:])
            for k in range(3):
                slot["routing"][k] += int(src["routing"][k])
            for depth, count in src["depth_hist"].items():
                slot["depth_hist"][depth] = \
                    slot["depth_hist"].get(depth, 0) + int(count)
            for mode, count in src["mode_census"].items():
                slot["mode_census"][mode] = \
                    slot["mode_census"].get(mode, 0) + int(count)
            for key in SPECIES_KEYS:
                slot["mult"][key]["sum_n"] += float(src["mult"][key]["sum_n"])
                slot["mult"][key]["sum_n2"] += float(
                    src["mult"][key]["sum_n2"])
                slot["mult"][key]["n_with"] += int(
                    src["mult"][key]["n_with"])
            slot["feeddown"]["n_finals"] += int(src["feeddown"]["n_finals"])
            for key in SPECIES_KEYS:
                dst_slot = slot["feeddown"]["per_species"][key]
                src_fd = src["feeddown"]["per_species"][key]
                dst_slot["n"] += int(src_fd["n"])
                dst_slot["n_resonant"] += int(src_fd["n_resonant"])
                for parent, count in src_fd["by_parent"].items():
                    dst_slot["by_parent"][parent] = \
                        dst_slot["by_parent"].get(parent, 0) + int(count)
            slot["worst_closure"] = max(
                slot["worst_closure"], float(src["worst_closure"]))
            slot["n_finals"] += int(src["n_finals"])
    shared_exchange_digest = shared_exchange_rolling.hexdigest()
    for policy in VARIANTS:
        name = policy.name
        per_variant[name]["exchange_digest"] = \
            exchange_rollings[name].hexdigest()
        rolling = hashlib.sha256()
        for digest in per_variant[name]["hashes"]:
            rolling.update(digest.encode("utf-8"))
            rolling.update(b"\n")
        per_variant[name]["event_digest"] = rolling.hexdigest()
    return {
        "n_events": n_events,
        "shared_exchange_digest": shared_exchange_digest,
        "per_variant": per_variant,
    }


# ---------------------------------------------------------------------------
# The replica record assembly
# ---------------------------------------------------------------------------
def _replica_record(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    aggregated: dict,
) -> dict:
    """Assemble one per-replica record (deterministic: no wall-clock
    fields anywhere in the record).  The finals' dense lattice indices
    accumulate into the species' dense lattices and the published rows
    are picked from the dense totals (the R01 ``_bin_track_sums``
    accumulate-dense-then-pick convention), so ``pooled_sums`` are
    published-row arrays aligned 1:1 with the per-published-row areas
    the evaluation divides by; a hole cell's weight stays unpicked and
    is dropped."""
    context = _WORKER_CONTEXT
    assert context is not None
    dense_offsets = context["dense_offsets_by_species"]
    dense_sizes = context["dense_sizes_by_species"]
    pick_by_species = context["pick_by_species"]
    variants_payload: dict[str, dict] = {}
    for policy in VARIANTS:
        name = policy.name
        slot = aggregated["per_variant"][name]
        dense_pooled = np.zeros(int(context["dense_total"]), dtype=float)
        row_flat = np.asarray(slot["row_flat"], dtype=np.int64)
        row_weight = np.asarray(slot["row_weight"], dtype=float)
        if row_flat.size:
            np.add.at(dense_pooled, row_flat, row_weight)
        pooled_species = {}
        for key in SPECIES_FLAT_ORDER:
            dense_slice = dense_pooled[
                int(dense_offsets[key]):
                int(dense_offsets[key]) + int(dense_sizes[key])]
            pooled_species[key] = [
                float(v) for v in dense_slice[pick_by_species[key]]]
        variants_payload[name] = {
            "variant_id": int(policy.variant_id),
            "policy": {
                "kappa_b": float(policy.kappa_b),
                "kappa_m": float(policy.kappa_m),
                "b_dec_gev2": float(policy.b_dec),
            },
            "event_hash_set_sha256": slot["event_digest"],
            "exchange_hash_set_sha256": slot["exchange_digest"],
            "routing": {
                "n_sources": int(slot["routing"][0]),
                "n_catalog": int(slot["routing"][1]),
                "n_continuum": int(slot["routing"][2]),
                "note": (
                    "the card-6 routing census at the frozen source "
                    "masses (the exchange layer is generated once per "
                    "event and shared across the variants; the catalog "
                    "branch carries the card-9 stable-identity path for "
                    "a drawn stable record, the continuum branch the "
                    "variant machinery)"
                ),
            },
            "depth_histogram": {str(k): int(v) for k, v in sorted(
                slot["depth_hist"].items())},
            "mode_census": dict(sorted(slot["mode_census"].items())),
            "multiplicity": slot["mult"],
            "feed_down": {
                "lineage_convention": (
                    "the corrected lineage convention of Round 3 card 3 "
                    "(the degenerate infinity convention of R23 is not "
                    "used), carried on the walk machinery: a final "
                    "particle is resonant feed-down when its producing "
                    "vertex's parent is a resonance other than the drawn "
                    "root -- the catalog route's drawn mixture record, "
                    "the continuum route's root sentinel 0 (the continuum "
                    "state is not a catalog resonance, so the root-split "
                    "daughters are direct source products); the "
                    "stable-identity path is a direct source product"
                ),
                "tallies": slot["feeddown"],
            },
            "conservation": {
                "events_checked": int(n_events),
                "event_b_q_s_conserved": True,
                "worst_energy_closure_gev": float(slot["worst_closure"]),
                "tolerance": EVENT_CLOSURE_TOL,
            },
            "pooled_sums": pooled_species,
            "n_finals": int(slot["n_finals"]),
        }
    record = {
        "schema":
            "na49_mixing.staged_r44_frozen_nonstrange_validation."
            "replica/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if aggregated.get("smoke") else "production",
        "stage_label": STAGE_LABEL,
        "replica": int(replica),
        "seed": int(seed),
        "kernel_replica_index": int(replica_slot),
        "master_seed": int(MASTER_SEED),
        "n_events": int(n_events),
        "chunk_size": int(CHUNK_SIZE),
        "statistics": {
            "tier": T3_LABEL,
            "events_per_replica": int(T3_EVENTS_PER_REPLICA),
            "variants": list(VARIANT_NAMES),
        },
        "seed_map": {
            "raw_seed_rule": RAW_SEED_RULE,
            "exchange_seed_rule": EXCHANGE_SEED_RULE,
            "variant_stream_rule": VARIANT_STREAM_RULE,
            "event_index_rule": EVENT_INDEX_RULE,
        },
        "exchange_layer": {
            "shared_event_hash_set_sha256":
                aggregated["shared_exchange_digest"],
            "note": (
                "the exchange layer (topology, exchange kinematics, "
                "spectral state) is generated once per event and shared "
                "across the two variants; the rolling digest runs over "
                "the ordered per-event canonical exchange hashes and the "
                "two per-variant rolling digests are recomputed from "
                "each variant's own view of the shared mappings after "
                "its decays (the run-scale bit-identity evidence)"
            ),
        },
        "variants": variants_payload,
        "identity": {
            "pre_registration_sha256": aggregated[
                "pre_registration_sha256"],
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


def _record_identity_ok(record: dict, pre: dict, replica: int, seed: int,
                        n_events: int) -> bool:
    """The completed-record identity rule: the replica, the seed, the
    statistics, the mode, and the pre-registration digest must match."""
    return bool(
        record.get("run") == RUN_ID
        and int(record.get("replica", -1)) == int(replica)
        and int(record.get("seed", -1)) == int(seed)
        and record.get("mode") == "production"
        and int(record.get("n_events", -1)) == int(n_events)
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


def _chunk_tasks(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    index_base: int,
    smoke: bool,
) -> list[tuple]:
    starts = list(range(0, int(n_events), CHUNK_SIZE))
    return [
        (int(replica), int(seed),
         [index_base + start + j
          for j in range(min(CHUNK_SIZE, int(n_events) - start))],
         int(MASTER_SEED), int(replica_slot),
         float(BLEND_RAMP_START_GEV), bool(smoke))
        for start in starts
    ]


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
                    f"[r44] {progress_label}: chunk {completed}/"
                    f"{len(tasks)} ({elapsed:.1f}s)",
                    flush=True,
                )


def run_generation(
    run_dir: Path,
    replicas: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> list[str]:
    """Generate the requested production replica records (any subset;
    the default is every replica) at the pre-registered seeds, each
    event decayed through both configurations.  A completed record that
    matches the pre-registered identity is reused unchanged.  Returns
    the record names written or reused."""
    pre = _load_pre_registration(run_dir)
    _assert_pre_registration(pre, run_dir)
    pre_digest = _pre_registration_digest_of(pre)
    tables = _load_production_tables()
    tables.append(_pi_zero_model_table(tables))
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
    _set_worker_context(context)
    written: list[str] = []
    requested = (
        list(range(T3_REPLICAS)) if replicas is None
        else [int(r) for r in replicas]
    )
    for replica in requested:
        if not 0 <= replica < T3_REPLICAS:
            _refuse(f"--replica {replica} outside 0..{T3_REPLICAS - 1}")
        name = RECORD_FMT.format(int(replica))
        path = run_dir / name
        seed = raw_replica_seed(replica)
        if path.is_file():
            existing = _read_json(path)
            if _record_identity_ok(existing, pre, replica, seed,
                                   T3_EVENTS_PER_REPLICA):
                print(f"[r44] {name}: completed record reused", flush=True)
                written.append(name)
                continue
            _refuse(
                f"{path}: a completed record exists but does not match "
                "the pre-registered identity; a drifted record is never "
                "silently overwritten (delete the run dir and re-run)"
            )
        tasks = _chunk_tasks(replica, seed, replica,
                             T3_EVENTS_PER_REPLICA,
                             EVENT_INDEX_BASE, smoke=False)
        parts = list(_map_chunks(tasks, workers, f"replica {replica}"))
        aggregated = _aggregate_chunks(parts)
        aggregated["pre_registration_sha256"] = pre_digest
        if aggregated["n_events"] != int(T3_EVENTS_PER_REPLICA):
            _refuse(
                f"replica {replica}: generated "
                f"{aggregated['n_events']} events, expected "
                f"{int(T3_EVENTS_PER_REPLICA)}"
            )
        record = _replica_record(
            replica, seed, replica, int(T3_EVENTS_PER_REPLICA),
            aggregated)
        _write_json(path, record)
        written.append(name)
        aligned_finals = int(
            record["variants"]["aligned_frozen"]["n_finals"])
        print(f"[r44] {name}: written ({aligned_finals} aligned finals)",
              flush=True)
    return written


def run_smoke(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """The hermetic end-to-end driver validation (never sealed): a smoke
    pre-registration (the frozen-parameter assertion against the pinned
    R41 fixture), 1 replica x 2,000 events at the smoke seeds (both
    variants), and the smoke evaluation against the synthetic
    placeholder data, into an explicit ``--run-dir``."""
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
    #    the frozen-parameter assertion runs against the pinned fixture) --
    pre, _frozen, _disjointness = _pre_registration_payload(smoke=True)
    pre["smoke"] = SMOKE_PAYLOAD
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    _set_worker_context(context)
    # -- the smoke generation --
    replica = 0
    name = f"{SMOKE_PREFIX}replica_{replica}.json"
    seed = smoke_raw_seed()
    tasks = _chunk_tasks(
        replica, seed, SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        SMOKE_EVENT_INDEX_BASE, smoke=True)
    parts = list(_map_chunks(tasks, workers, "smoke"))
    aggregated = _aggregate_chunks(parts)
    aggregated["pre_registration_sha256"] = _pre_registration_digest_of(pre)
    aggregated["smoke"] = True
    record = _replica_record(
        replica, smoke_raw_seed(), SMOKE_KERNEL_REPLICA_SLOT, SMOKE_EVENTS,
        aggregated)
    _write_json(run_dir / name, record)
    # -- the smoke evaluation (the machinery end to end; synthetic data) --
    report = run_evaluate_machinery(
        run_dir, smoke=True, tables=tables, fams=fams)
    _write_json(run_dir / REPORT_NAME, report)
    return {
        "record": name,
        "report_digest": report.get("digest"),
    }


SMOKE_PAYLOAD = {
    "replicas": SMOKE_REPLICAS,
    "events_per_replica": SMOKE_EVENTS,
    "raw_seed": SMOKE_RAW_SEED,
    "raw_seed_rule": SMOKE_RAW_SEED_RULE,
    "kernel_replica_slot": SMOKE_KERNEL_REPLICA_SLOT,
    "event_index_rule": SMOKE_EVENT_INDEX_RULE,
    "event_index_base": SMOKE_EVENT_INDEX_BASE,
    "frozen_parameter_source": (
        "the pinned R41 fit-report fixture (the hermetic stand-in for "
        "the sealed archive; the same values the production assertion "
        "expects)"
    ),
    "note": SMOKE_NOTE,
}


def _load_sealed_r41_fit_report() -> dict:
    """The sealed R41 fit report (the frozen-parameter source)."""
    path = MONO_ROOT / R41_RUN_DIR / R41_FIT_REPORT_NAME
    if not path.is_file():
        _refuse(f"the sealed R41 fit report is missing: {path}")
    actual = _sha256_file(path)
    if actual != R41_FIT_REPORT_FILE_SHA256:
        _refuse(
            "the sealed R41 fit report digest drifted: expected "
            f"{R41_FIT_REPORT_FILE_SHA256}, actual {actual} ({path}); "
            "the frozen-parameter source is byte-drifted"
        )
    report = _read_json(path)
    if str(report.get("digest")) != R41_FIT_REPORT_DIGEST:
        _refuse(
            "the sealed R41 fit report's self-digest "
            f"{report.get('digest')!r} does not match the pre-registered "
            f"{R41_FIT_REPORT_DIGEST!r}"
        )
    return report


def frozen_parameter_repoint_record() -> dict:
    """The card-19 re-point record (the superseded vs corrected frozen
    tuple, the fit-report digests, the superseded sibling driver
    digests, and the archive paths; the superseded values are retained
    provenance and are not consumed)."""
    return {
        "card": "TASK-20260920-r4-run-r44-repoint",
        "corrected_frozen": dict(R41_FITTED),
        "corrected_source": (
            f"sealed {R41_RUN_DIR}/{R41_FIT_REPORT_NAME}"
        ),
        "corrected_fit_report_file_sha256": R41_FIT_REPORT_FILE_SHA256,
        "corrected_fit_report_self_digest": R41_FIT_REPORT_DIGEST,
        "superseded_frozen": dict(SUPERSEDED_R41_FITTED),
        "superseded_source": (
            f"sealed {SUPERSEDED_R41_RUN_DIR}/{R41_FIT_REPORT_NAME}"
        ),
        "superseded_fit_report_file_sha256":
            SUPERSEDED_R41_FIT_REPORT_FILE_SHA256,
        "superseded_fit_report_self_digest":
            SUPERSEDED_R41_FIT_REPORT_DIGEST,
        "superseded_status": (
            "superseded by the card-16 R41 relitigation; the values "
            "stay intact in the sealed superseded archive (read-only, "
            "the seal verifies) and are not consumed by this run"
        ),
        "corrected_sibling_driver_sha256s": {
            "r41": FROZEN_SUBSTRATE["r41_driver"][1],
            "r42": FROZEN_SUBSTRATE["r42_driver"][1],
            "r43": FROZEN_SUBSTRATE["r43_driver"][1],
        },
        "superseded_sibling_driver_sha256s": {
            "r41": SUPERSEDED_R41_DRIVER_SHA256,
            "r42": SUPERSEDED_R42_DRIVER_SHA256,
            "r43": SUPERSEDED_R43_DRIVER_SHA256,
        },
        "superseded_sibling_run_dirs": [
            SUPERSEDED_R41_RUN_DIR, *SUPERSEDED_R42_RUN_DIRS,
            SUPERSEDED_R43_RUN_DIR,
        ],
    }


def _pre_registration_payload(smoke: bool) -> tuple:
    """The pre-registration payload (the contract) with the frozen-
    parameter block and the seed-block disjointness record attached:
    the frozen-parameter assertion runs against the sealed R41 archive
    in production and against the pinned R41 fixture in the smoke (the
    hermetic stand-in; the same values the production assertion
    expects)."""
    substrate = verify_frozen_substrate()
    if smoke:
        frozen = assert_frozen_parameters(R41_FIT_REPORT_FIXTURE)
        frozen["source"] = (
            "the pinned R41 fit-report fixture (the hermetic smoke "
            "stand-in; the production assertion reads the sealed R41 "
            f"archive {R41_RUN_DIR}/{R41_FIT_REPORT_NAME})"
        )
        disjointness = seed_disjointness_report()
        disjointness["source"] = (
            "the pinned block register (the smoke skips the sealed "
            "pre-registration reads)"
        )
        sealed = {}
    else:
        fit_report = _load_sealed_r41_fit_report()
        frozen = assert_frozen_parameters(fit_report)
        sealed_blocks = _production_seed_disjointness()
        disjointness = seed_disjointness_report()
        disjointness["sealed_pre_registration_cross_check"] = sealed_blocks
        disjointness["source"] = (
            "the sealed R41/R42/R43 pre-registrations (seed maps and "
            "event-index blocks) plus the pinned R40/R23/R32/audit "
            "block register"
        )
        sealed = verify_sealed_inputs()
    pre = build_pre_registration(smoke=smoke)
    pre["mode"] = "smoke" if smoke else "production"
    pre["frozen_parameters"] = frozen
    pre["frozen_parameter_repoint"] = frozen_parameter_repoint_record()
    pre["seed_map"]["disjointness"] = disjointness
    pre["substrate"] = substrate
    pre["sealed_run_dirs_status"] = sealed
    return pre, frozen, disjointness


def run_pre_registration(run_dir: Path, smoke: bool = False) -> dict:
    """Write pre_registration.json and provenance.json into the run dir,
    then stop.  Verifies the frozen substrate digests, the frozen R41
    parameters (against the sealed archive in production, against the
    pinned fixture in the smoke), and the sealed seals first.  Nothing
    is generated."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        _refuse(
            f"run dir is not empty: {run_dir}; the pre-registration is "
            "written before any generation into a fresh run dir"
        )
    pre, frozen, _disjointness = _pre_registration_payload(smoke=smoke)
    sealed = pre["sealed_run_dirs_status"]
    # the superseded sealed archives' seal states (read-only; the seals
    # verify; the card-19 re-point records them in the provenance and
    # consumes nothing in them)
    superseded_status = {
        rel: verify_sealed_dir(rel) for rel in SUPERSEDED_ARCHIVE_RUN_DIRS
    }
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
            "na49_mixing.staged_r44_frozen_nonstrange_validation."
            "provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "new_driver_sha256_both_trees": DRIVER_SHA256,
        "driver_note": (
            "byte-identical in the nested tree (dev/mixing2006/helpers/"
            "benchmarking/) and the monorepo mirror (helpers/"
            "benchmarking/), the established _locate_roots pattern"
        ),
        "frozen_parameter_repoint": frozen_parameter_repoint_record(),
        "superseded_r41_archive": {
            "run_dir": SUPERSEDED_R41_RUN_DIR,
            "state": (
                "intact; the seal verifies (verify_sealed_dir); the "
                "archive carries the superseded card-12 fit output "
                "the card-15 landing pinned and is superseded by the "
                "corrected card-16 R41 archive"
            ),
            "superseded_fit_report_file_sha256":
                SUPERSEDED_R41_FIT_REPORT_FILE_SHA256,
            "superseded_fit_report_self_digest":
                SUPERSEDED_R41_FIT_REPORT_DIGEST,
            "superseded_driver_sha256": SUPERSEDED_R41_DRIVER_SHA256,
            "seal": superseded_status[SUPERSEDED_R41_RUN_DIR],
        },
        "superseded_r42_archives": [
            {
                "run_dir": rel,
                "state": (
                    "intact; the seal verifies (verify_sealed_dir); the "
                    "archive was generated under the superseded "
                    "card-13 driver and is superseded by the corrected "
                    "card-17 R42 archive"
                ),
                "superseded_driver_sha256": SUPERSEDED_R42_DRIVER_SHA256,
                "seal": superseded_status[rel],
            }
            for rel in SUPERSEDED_R42_RUN_DIRS
        ],
        "superseded_r43_archive": {
            "run_dir": SUPERSEDED_R43_RUN_DIR,
            "state": (
                "intact; the seal verifies (verify_sealed_dir); the "
                "archive was generated under the superseded card-14 "
                "driver and is superseded by the corrected card-18 "
                "R43 archive"
            ),
            "superseded_driver_sha256": SUPERSEDED_R43_DRIVER_SHA256,
            "seal": superseded_status[SUPERSEDED_R43_RUN_DIR],
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
        "frozen_b_eff_gev2": B_EFF_FROZEN_GEV2,
        "frozen_r41_parameters": {
            "values": dict(R41_FITTED),
            "source": f"sealed {R41_RUN_DIR}/{R41_FIT_REPORT_NAME}",
            "fit_report_file_sha256": R41_FIT_REPORT_FILE_SHA256,
            "fit_report_self_digest": R41_FIT_REPORT_DIGEST,
            "sealed_run_dir_state": sealed.get(R41_RUN_DIR),
            "superseded": {
                "values": dict(SUPERSEDED_R41_FITTED),
                "source": (
                    f"sealed {SUPERSEDED_R41_RUN_DIR}/"
                    f"{R41_FIT_REPORT_NAME}"
                ),
                "fit_report_file_sha256":
                    SUPERSEDED_R41_FIT_REPORT_FILE_SHA256,
                "fit_report_self_digest":
                    SUPERSEDED_R41_FIT_REPORT_DIGEST,
                "status": "superseded by relitigation",
            },
            "refit": False,
        },
        "catalog_sha256": FROZEN_SUBSTRATE["catalog"][1],
        "continuum_config_sha256": FROZEN_SUBSTRATE[
            "continuum_config"][1],
        "r32_driver_sha256": FROZEN_SUBSTRATE["r32_driver"][1],
        "r40_driver_sha256": FROZEN_SUBSTRATE["r40_driver"][1],
        "r41_driver_sha256": FROZEN_SUBSTRATE["r41_driver"][1],
        "r42_driver_sha256": FROZEN_SUBSTRATE["r42_driver"][1],
        "r43_driver_sha256": FROZEN_SUBSTRATE["r43_driver"][1],
        "superseded_sibling_driver_sha256s": {
            "r41": SUPERSEDED_R41_DRIVER_SHA256,
            "r42": SUPERSEDED_R42_DRIVER_SHA256,
            "r43": SUPERSEDED_R43_DRIVER_SHA256,
        },
        "sealed_run_dirs": sealed,
        "predecessors": (
            "the sealed R40/R41/R42/R43 archives and the sealed R30-R32 "
            "archives are read-only inputs: the corrected R41 archive "
            "is the frozen-parameter source (the card-19 re-point; the "
            "superseded R41/R42/R43 archives stay intact, their seals "
            "verify, and their digests are recorded in the "
            "superseded-archive blocks above), the R42/R43 "
            "pre-registrations carry the previously-used seed blocks, "
            "the R43 gate carries the training-improvement "
            "cross-reference, the R41 gate carries the replica-block "
            "stability cross-reference, and the R32 spread is the "
            "model-uncertainty carry of obligation 4"
        ),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(f"[r44] pre-registration written into {run_dir}", flush=True)
    return pre


def _production_seed_disjointness() -> dict:
    """The sealed-pre-registration cross-check of the recorded blocks:
    the R41/R42/R43 seed maps are read from the sealed archives and
    verified against the pinned register (any drift refuses)."""
    expected = {
        R41_RUN_DIR: ("r41", [20301020, 20301027],
                      "400_000_000 + i, i < 100,000", 20260920),
        R42_RUN_DIR: ("r42", [20301020, 20301027],
                      "500_000_000 + i, i < 100,000", 20260920),
        R43_RUN_DIR: ("r43", [20301020, 20301027],
                      "600_000_000 + i, i < 100,000", 20260920),
    }
    observed: dict[str, dict] = {}
    for rel, (name, seeds_lo_hi, event_rule, master) in expected.items():
        pre = _read_json(MONO_ROOT / rel / PREREGISTRATION_NAME)
        seed_map = pre.get("seed_map", {})
        raw = seed_map.get("raw_replica_seeds") or []
        observed_seed_range = (
            [int(min(raw)), int(max(raw))] if raw else None
        )
        if observed_seed_range != seeds_lo_hi:
            _refuse(
                f"the sealed {name} pre-registration's raw seed block "
                f"{observed_seed_range!r} does not match the pinned "
                f"register {seeds_lo_hi!r}"
            )
        observed_rule = str(seed_map.get("event_index_rule"))
        if observed_rule != event_rule:
            _refuse(
                f"the sealed {name} pre-registration's event-index rule "
                f"{observed_rule!r} does not match the pinned "
                f"{event_rule!r}"
            )
        if int(seed_map.get("master_seed", -1)) != int(master):
            _refuse(
                f"the sealed {name} pre-registration's master seed does "
                "not match the pinned register"
            )
        smoke = seed_map.get("smoke") or {}
        observed[name] = {
            "raw_seeds": observed_seed_range,
            "event_index_rule": observed_rule,
            "master_seed": int(master),
            "smoke": {
                "raw_seed": smoke.get("raw_seed"),
                "replica_slot": smoke.get("kernel_replica_slot"),
                "event_index_rule": smoke.get("event_index_rule"),
            },
            "source": f"sealed {rel}/pre_registration.json",
        }
    return observed


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
    stats = pre.get("statistics", {})
    if int(stats.get("replicas", -1)) != T3_REPLICAS or int(
        stats.get("events_per_replica", -1)
    ) != T3_EVENTS_PER_REPLICA:
        _refuse("the pre-registered statistics do not match the card "
                "(T3: 20 replicas x 250,000)")
    seeds = pre.get("seed_map", {})
    if list(seeds.get("raw_replica_seeds") or ()) != \
            list(RAW_REPLICA_SEEDS):
        _refuse("the pre-registered raw replica seeds do not match the "
                "card's block 20301020-20301039")
    if seeds.get("event_index_rule") != EVENT_INDEX_RULE:
        _refuse("the pre-registered event-index rule does not match the "
                "card (700_000_000 + i, i < 250,000)")
    if seeds.get("master_seed") != MASTER_SEED:
        _refuse("the pre-registered master seed does not match the card")
    if not seeds.get("disjointness", {}).get("disjoint", False):
        _refuse("the pre-registered seed-block disjointness check is "
                "not recorded as passed")
    frozen = pre.get("frozen_parameters", {})
    if (float(frozen.get("kappa_B", float("nan"))) != ALIGNED_POLICY.kappa_b
            or float(frozen.get("kappa_m", float("nan"))) !=
            ALIGNED_POLICY.kappa_m
            or float(frozen.get("b_dec_gev2", float("nan"))) !=
            ALIGNED_POLICY.b_dec
            or float(frozen.get("baryonic_nominal_width_gev",
                                float("nan"))) != FROZEN_WIDTH_GEV):
        _refuse("the pre-registered frozen parameters are not the "
                "corrected R41 fitted values (kappa_B = 0.0, kappa_m = "
                "4.0, b_dec = 2.0 GeV^-2, nominal_width = 0.3 GeV; the "
                "superseded card-15 tuple 0.0 / 4.0 / 2.0 GeV^-2 / "
                "0.5 GeV is retained provenance, not consumed)")
    if frozen.get("refit") is not False:
        _refuse("the frozen-parameter block must carry refit = false")
    variants = pre.get("variant_register") or ()
    if tuple(variants) != VARIANT_NAMES:
        _refuse("the pre-registered variant register does not match "
                "the card's two configurations")
    policies = pre.get("variant_policies", {})
    for policy in VARIANTS:
        entry = policies.get(policy.name) or {}
        if (float(entry.get("kappa_b", float("nan"))) != policy.kappa_b
                or float(entry.get("kappa_m", float("nan"))) !=
                policy.kappa_m
                or float(entry.get("b_dec_gev2", float("nan"))) !=
                policy.b_dec):
            _refuse(f"the pre-registered policy of {policy.name} drifted")
    held_out = pre.get("held_out_set", {})
    if tuple(held_out.get("likelihood_register") or ()) != HELD_OUT_KEYS:
        _refuse("the pre-registered held-out register does not match "
                "the card's species-separated set")
    if tuple(held_out.get("model_only_keys") or ()) != MODEL_ONLY_KEYS:
        _refuse("the pre-registered model-only pi0 diagnostics drifted")
    if held_out.get("disjoint_from_training") is not True:
        _refuse("the pre-registered held-out register is not marked "
                "disjoint from the R41 training combinations")
    likelihood = pre.get("likelihood", {})
    if float(likelihood.get("held_out_degradation_factor", 0.0)) != \
            HELD_OUT_DEGRADATION_FACTOR:
        _refuse("the pre-registered held-out degradation factor is not "
                "the card's 2.0")
    delta = likelihood.get("delta_aic", {})
    if int(delta.get("k_aligned", -1)) != 4 or \
            int(delta.get("k_baseline", -1)) != 1:
        _refuse("the delta AIC parameter costs are not the card's "
                "(aligned k = 4, baseline k = 1)")
    if float(likelihood.get("sigma_tolerance", 0.0)) != SIGMA_TOLERANCE:
        _refuse("the named-residual sigma tolerance is not the fixed "
                "engine's SIGMA_TOLERANCE = 3.0")
    carry = pre.get("obligation_4_carry", {})
    if carry.get("r32_spread", {}).get("bin_spread_p95") != \
            OBLIGATION4_R32_SPREAD["bin_spread_p95"]:
        _refuse("the obligation 4 carry numbers are not the card's "
                "verbatim R32 spread")
    if not pre.get("written_before_generation", False):
        _refuse("the pre-registration was not marked "
                "written_before_generation")
    if pre.get("no_fitting") is not True or \
            pre.get("optimizer_invoked") is not False:
        _refuse("the pre-registration must carry no_fitting = true and "
                "optimizer_invoked = false (the validation is a "
                "frozen-parameter evaluation, plan section 10 line 802)")
    for key in ("collision", "observables", "conservation", "determinism",
                "support_boundary", "acceptance"):
        if key not in pre:
            _refuse(f"the pre-registration is missing the {key} block")


# ---------------------------------------------------------------------------
# The pre-registration (written before any generation)
# ---------------------------------------------------------------------------
def build_pre_registration(smoke: bool = False) -> dict:
    """The pre-registered run contract, asserted against the card
    (TASK-20260920-r4-run-r44-frozen-nonstrange-validation)."""
    return {
        "schema":
            "na49_mixing.staged_r44_frozen_nonstrange_validation."
            "pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "plan_anchors": {
            "run": "plan section 10 run R44, line 800",
            "statistics_tier": "plan section 5.1 line 255 (T3)",
            "frozen_parameters": "plan section 10, line 802 (the "
                                 "validation is at the frozen fit)",
            "held_out_set": "round 4 index item 11 (the pre-registered "
                            "minimal choice)",
            "likelihood": "plan section 5.4",
            "mechanism_retention": "plan section 5.5, lines 315-323 "
                                   "(all seven conditions)",
            "gate_clause": "plan section 10, line 806 (Gate 4, held-out "
                           "shapes)",
            "no_fitting": "plan section 10 line 802 (the validation is "
                          "at the frozen fit)",
            "conservation": "plan section 10.6 line 757 (the depth guard)",
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
            "blend_ramp_start": BLEND_RAMP_START_STATUS,
            "catalog_branch": (
                "the R32 builder catalog branch convention: the open-set "
                "mixture record draw on the spawned decay_channel child "
                "of the exchange derivation (generated once per event "
                "and shared across the variants); a drawn stable record "
                "takes the card-9 stable-identity path -- the source "
                "contributes itself as the single final at its own "
                "four-momentum, a deterministic draw-free variant-"
                "independent terminal product; an unstable record is "
                "realized at the source mass (no truncated-window mass "
                "redraw, the R32 builder convention) and decayed per "
                "variant through the variant's cascade at the frozen "
                "parameters"
            ),
            "continuum_branch": (
                "the single activated continuum family (baryonic_np) "
                "realized at the frozen source four-momentum (no "
                "family-mass redraw), decayed per variant through card "
                "10's attached machinery at the variant's policy (the "
                "frozen aligned point for v0, the card-10 v2 isotropic "
                "point for v1); the production axis is card 5's "
                "initial_source_axis against the source's own beam "
                "direction (projectile +z, target -z; the round 4 index "
                "item 4 p+p pre-registration)"
            ),
            "assembly_note": (
                "the caller's event assembly is the card-9 convention: "
                "stable terminal daughters are finals and the unstable "
                "terminal daughters decay through the byte-unchanged "
                "Round 2 kernel on the same per-variant streams; the "
                "variant machinery is the walk above the terminal "
                "established states, not the assembly"
            ),
        },
        "variants": {
            "index_item": 8,
            "register": list(VARIANT_NAMES),
            "policies": variant_policies_payload(),
            "note": VARIANTS_NOTE,
        },
        "variant_register": tuple(VARIANT_NAMES),
        "variant_policies": variant_policies_payload(),
        "frozen_parameters": {
            "kappa_B": ALIGNED_POLICY.kappa_b,
            "kappa_m": ALIGNED_POLICY.kappa_m,
            "b_dec_gev2": ALIGNED_POLICY.b_dec,
            "baryonic_nominal_width_gev": FROZEN_WIDTH_GEV,
            "source": "sealed R41 archive",
            "refit": False,
            "note": FROZEN_PARAMETER_NOTE,
            "blend_ramp_start": BLEND_RAMP_START_STATUS,
            "assertion": (
                "the frozen values are asserted against the sealed R41 "
                "archive's fit report before any generation (the "
                "'fitted' block, the file digest "
                f"{R41_FIT_REPORT_FILE_SHA256[:16]}..., the report "
                f"self-digest {R41_FIT_REPORT_DIGEST[:16]}...); any "
                "drift refuses the run"
            ),
        },
        "baseline": {
            "name": BASELINE_POLICY.name,
            "variant_id": BASELINE_POLICY.variant_id,
            "policy": {
                "kappa_b": BASELINE_POLICY.kappa_b,
                "kappa_m": BASELINE_POLICY.kappa_m,
                "b_dec_gev2": BASELINE_POLICY.b_dec,
            },
            "nominal_width_gev": FROZEN_WIDTH_GEV,
            "source": (
                "card 10's v2 policy at kappa = 0, b_dec = 1.0 GeV^-2 "
                "(the Gate 4 'beyond the isotropic sequential cascade' "
                "clause baseline); it carries nominal_width as its "
                "single parameter, recorded at the same corrected R41 "
                f"frozen width {FROZEN_WIDTH_GEV!r} GeV (the "
                "superseded card-15 fit's 0.5 GeV coincided with the "
                "committed config value), so both variants share the "
                "continuum-daughter mass rule of the byte-unchanged "
                "committed continuum config and the same support"
            ),
        },
        "held_out_set": {
            "index_item": 11,
            "species_named": list(HELD_OUT_SPECIES_NAMED),
            "likelihood_register": list(HELD_OUT_KEYS),
            "model_only_keys": list(MODEL_ONLY_KEYS),
            "model_only_note": (
                "the published data surface (the sealed R01 archive) "
                "carries the six paper-window species only, so the pi0 "
                "distribution and moment are recorded model-side as "
                "held-out diagnostics (no published pi0 table exists in "
                "the pipeline data; the pi0 rows carry the model-only "
                "marker and no likelihood contribution)"
            ),
            "training_complement": (
                "R41's fit set: the charge-symmetric combinations of "
                "plan section 7.3 (pi_Sigma = pi+ + pi-, B_transport = "
                "p - pbar, the pooled pion and proton seagulls, the "
                "pooled multiplicity moments) -- the training "
                "observables, not part of this run's acceptance (they "
                "are cross-referenced from R43)"
            ),
            "disjoint_from_training": True,
            "disjointness_note": HELD_OUT_DISJOINT_NOTE,
            "published_binning": "preserved (the sealed R01 tables)",
            "data_source": (
                "the sealed R01 archive (bin_predictions/covariance/"
                "residuals), cross-checked bit for bit against the "
                "locally loaded SPS tables at --evaluate time (the R11 "
                "parse_sealed_binned discipline)"
            ),
        },
        "likelihood": {
            "form": (
                "-2ln L on the held-out bins with sigma^2 = v_data + "
                "se^2 (the sealed R01 published variance plus the "
                "between-replica MC variance of the evaluated variant, "
                "floored at the engine's DETERMINISTIC_SE_FLOOR); the "
                "published binning is preserved; the data and MC "
                "contributions are reported separately per observable "
                "(the data-variance-only form sigma^2 = v_data and the "
                "MC-variance-only form sigma^2 = se^2 alongside the "
                "total); a bin whose data, published variance, "
                "prediction, or MC error is non-finite is dropped from "
                "the objective and listed (the R11 convention, never "
                "silent)"
            ),
            "held_out_degradation_factor": HELD_OUT_DEGRADATION_FACTOR,
            "held_out_degradation_note": HELD_OUT_DEGRADATION_NOTE,
            "sigma_tolerance": SIGMA_TOLERANCE,
            "sigma_tolerance_note": SIGMA_TOLERANCE_NOTE,
            "delta_aic": {
                "definition": (
                    "AIC = -2ln L + 2k on the summed held-out likelihood "
                    "register; delta AIC = AIC(aligned) - "
                    "AIC(baseline); the parameter cost is included (the "
                    "aligned configuration carries kappa_B, kappa_m, "
                    "b_dec, nominal_width: k = 4; the baseline carries "
                    "nominal_width: k = 1); favored when delta AIC < 0"
                ),
                "k_aligned": 4,
                "k_baseline": 1,
                "parameter_cost": "included",
            },
            "se_convention": SE_CONVENTION,
        },
        "statistics": {
            "tier": T3_LABEL,
            "replicas": T3_REPLICAS,
            "events_per_replica": T3_EVENTS_PER_REPLICA,
            "total_events": T3_REPLICAS * T3_EVENTS_PER_REPLICA,
            "decays": (
                "5,000,000 exchange events x 2 variants of decay (the "
                "two-variant decay multiplies the per-event cost by ~2 "
                "over the R41/R42 single-variant envelope)"
            ),
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": RAW_SEED_RULE,
            "raw_replica_seeds": list(RAW_REPLICA_SEEDS),
            "exchange_seed_rule": EXCHANGE_SEED_RULE,
            "variant_stream_rule": VARIANT_STREAM_RULE,
            "event_index_rule": EVENT_INDEX_RULE,
            "disjointness": (
                "recorded by seed_disjointness_report() at "
                "pre-registration time (the exact previously-used blocks "
                "from the sealed R41/R42/R43 pre-registrations, the "
                "shared raw seeds named, and the event-index partition "
                "disjointness verified; a (seed, event-index) substream "
                "collision stops the run, never a silent re-block)"
            ),
            "smoke": SMOKE_PAYLOAD,
        },
        "observables": {
            "held_out": HELD_OUT_SET_NOTE,
            "deviance_set": (
                "the eight data-backed held-out statistics (the pi+, "
                "pi-, p, pbar spectra and the pi+, pi-, p, pbar window "
                "moments) carry the -2ln L and the degradation "
                "comparison against the sealed R01 published data; the "
                "pi0 spectrum and moment are recorded per variant as "
                "model-only held-out diagnostics"
            ),
            "model_only_diagnostics": (
                "the pi0 spectrum (the pi0 finals on the published "
                "pi+ grid geometry at the pi0 mass) and the pi0 window "
                "moment, per variant per replica and pooled with the "
                "between-replica SE"
            ),
            "condition_4_sectors": SECTOR_DEGRADATION_NOTE,
            "multiplicity_convention": (
                "the species-separated window moments: the midpoint-"
                "rectangle window integral of the species' published "
                "spectrum cells (sum d * area for the data side with "
                "the first-order delta-method variance sum v * "
                "area^2, the R41 integral convention applied to the "
                "individual species); the model side is the same "
                "integral of the species' binned predictions"
            ),
        },
        "support_boundary": {
            "both_variants": "multi-vertex (identical)",
            "regeneration_required": False,
            "note": (
                "both configurations are card 4's sequential cascade "
                "with the card-8 form factor attached; the widths agree "
                "(both variants run the byte-unchanged committed "
                "continuum config's mass rule; the R41 frozen width is "
                "recorded at the corrected fitted value 0.3 GeV, the "
                "superseded card-15 fit's 0.5 GeV coincided with the "
                "committed config value), so both share the continuum-"
                "daughter mass rule and the same support; kappa_B, "
                "kappa_m, and b_dec alter probability densities but not "
                "kinematic support (plan section 5.3 line 288); no "
                "regeneration is required at either configuration"
            ),
        },
        "conservation": {
            "per_event": (
                "every p+p event conserves total (B, Q, S) = (2, 2, 0) "
                "exactly in every variant; a violation raises and stops "
                "the run"
            ),
            "per_source": (
                "every source's finals sum to its own (B, Q, S) = (1, "
                "1, 0) exactly in every variant; a violation raises and "
                "stops the run"
            ),
            "energy": (
                "each variant's final energy sum closes against the two "
                "source energies within 1e-9 GeV; the card-4 per-vertex "
                "closure checks run byte-unchanged inside the walks and "
                "the kernel"
            ),
            "depth_guard": (
                "zero depth-guard activations over 5,000,000 events x 2 "
                "variants (a guard raises CascadeDepthGuardError and "
                "stops the run, plan section 10.6 line 757)"
            ),
        },
        "determinism": {
            "rolling_digest": (
                "each replica record carries the shared exchange-layer "
                "digest (sha256 over the ordered per-event canonical "
                "exchange hashes), the per-variant event-hash set "
                "digests (sha256 over the ordered per-event canonical "
                "per-variant hashes: variant id, event id, mode census, "
                "final count, and every final's pdg with exact hex "
                "four-momentum components), and the per-variant rolling "
                "exchange digests (each variant's own view of the "
                "shared mappings, recomputed after its decays); a "
                "per-replica re-run from a clean interpreter reproduces "
                "all of them"
            ),
            "probe_replica": {
                "replica": PROBE_REPLICA,
                "rule": (
                    "clean-validate re-runs this full replica from a "
                    "clean interpreter and compares the event-hash set "
                    "digests, the exchange digests, the pooled sums, "
                    "the routing census, the depth histograms, the "
                    "multiplicity tallies, and the feed-down tallies "
                    "against the sealed record"
                ),
            },
            "report_note": (
                "held_out_report.json and gate4_r44.json carry no "
                "wall-clock fields: their bytes are a pure function of "
                "the sealed artifacts, the pre-registered seeds, and "
                "the committed code"
            ),
        },
        "gateway_fallback": GATEWAY_FALLBACK_RULE,
        "obligation_4_carry": {
            "obligation": 4,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "audit_outcome": AUDIT_OUTCOME,
            "obligation_5": OBLIGATION5_NOTE,
        },
        "acceptance": {
            "precedent": (
                "the acceptance precedent of "
                "TASK-20260918-r3-acceptance-rescope-card8a-8b: an "
                "MC-driven check set with residuals seals with a named-"
                "residual register, not a raw zero-failure target"
            ),
            "i_conservation": (
                "5,000,000 events, exact B = 2 / Q = 2 and exact "
                "per-source (B, Q, S); zero violations; zero "
                "depth-guard activations (plan section 10.6 line 757)"
            ),
            "ii_training_improvement_cross_reference": (
                "the training-observable improvement over the isotropic "
                "sequential baseline is cross-referenced from the "
                f"sealed R43 evidence ({R43_RUN_DIR}/"
                f"{R43_GATE_NAME}; this run does not re-evaluate the "
                "training set; the cross-reference is recorded in the "
                "gate evidence)"
            ),
            "iii_held_out": HELD_OUT_DEGRADATION_NOTE,
            "iv_sector_degradation": SECTOR_DEGRADATION_NOTE,
            "v_parameter_stability_cross_reference": (
                "cross-referenced from the sealed R41 replica-block "
                f"stability table ({R41_RUN_DIR}/{R41_GATE_NAME}, all "
                "8 blocks, plan section 5.5 item 5); a failing R41 "
                "parameter is a named finding feeding the Gate 4 "
                "unconstrained clause"
            ),
            "vi_delta_aic": (
                "the delta AIC of the frozen aligned configuration "
                "against the isotropic sequential baseline (parameter "
                "cost included) is recorded; the verdict (favored / "
                "not favored) is recorded as the held-out likelihood "
                "evidence of the Gate 4 retain"
            ),
            "vii_color_singlet_audit": (
                "conservation at every vertex and no color flow in the "
                "cascade (the card 4-9 surface contract) pass on the "
                "full run; the audit is cross-referenced from the "
                "conservation evidence of this run"
            ),
            "viii_determinism": (
                "the clean-environment re-run reproduces the per-replica "
                "report hashes"
            ),
            "ix_seal_and_clean_validation": (
                "the seal is intact and the clean-environment re-run "
                "reproduces the held-out evaluation hashes"
            ),
            "x_pins_and_freeze": (
                "the pin test register lands, the freeze re-pin is "
                "recorded, and the full nested suite is green"
            ),
            "named_residual_register_triggers": [
                "a held-out observable degrading beyond 2.0 x the "
                "pooled MC SE of the difference (the condition iii "
                "finding; the Gate 4 held-out clause consumes it "
                "directly)",
                "a per-bin |residual| > 3.0 sigma on the held-out "
                "data-backed bins (the house named-residual form of "
                "the fixed engine, SIGMA_TOLERANCE = 3.0)",
                "a condition-4 sector flag (a species deviance "
                "degradation beyond 5% or a feed-down fraction relative "
                "change beyond 5%, without a stated physics reason)",
                "statistically undefined bins (dropped and listed, "
                "the R11 convention)",
                "an exchange bit-identity failure (the identical-"
                "exchange-events premise violated)",
                "a non-finite recorded observable",
            ],
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "the validation is a frozen-parameter evaluation, plan "
            "section 10 line 802: no parameter is fit, no optimizer "
            "runs, the aligned configuration runs at the frozen R41 "
            "values and the baseline at the card-10 v2 policy"
        ),
        "written_before_generation": True,
    }


SE_CONVENTION = (
    "the between-replica standard error (std ddof=1 over the 20 T3 "
    "replica blocks, divided by sqrt(20)), floored at the engine's "
    f"DETERMINISTIC_SE_FLOOR = {float(DETERMINISTIC_SE_FLOOR)!r}; the "
    f"pre-registered {HELD_OUT_DEGRADATION_FACTOR!r}-factor bound and "
    f"the {SECTOR_DEGRADATION_FRACTION!r} sector bound are never "
    "relaxed (the acceptance precedent of TASK-20260918-r3-acceptance-"
    "rescope-card8a-8b)"
)
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)


# ---------------------------------------------------------------------------
# The fit data (the sealed R01 archive + the published binning)
# ---------------------------------------------------------------------------
R01_ARCHIVE = R01_RUN_DIR
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"


def _load_sealed_r01(tables: list) -> dict:
    """The sealed R01 data vector and published variances of the
    data-backed species, cross-checked bit for bit against the locally
    loaded SPS tables (the R11 parse_sealed_binned discipline)."""
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


def _cell_area_from_table(table, i: int) -> float:
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[i])
    ip = int(table.bin_index_pt[i])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


# ---------------------------------------------------------------------------
# The held-out statistic construction (the species-separated register)
# ---------------------------------------------------------------------------
def _spectrum_statistic(sealed: dict, tables_by_species: dict,
                        species: str) -> dict:
    """One species-separated spectrum statistic on the published bins."""
    table = tables_by_species[species]
    areas = np.asarray([
        _cell_area_from_table(table, i)
        for i in range(int(table.data.size))
    ], dtype=float)
    return {
        "key": f"{species}_spectrum",
        "kind": "spectrum",
        "species": species,
        "d": np.asarray(sealed[species]["d"], dtype=float),
        "v_data": np.asarray(sealed[species]["v_data"], dtype=float),
        "areas": areas,
        "xF_mid": [float(v) for v in sealed[species]["xF_mid"]],
        "pT_mid": [float(v) for v in sealed[species]["pT_mid"]],
    }


def _moment_statistic(spectrum: dict) -> dict:
    """The species-separated window first moment <n> (the midpoint-
    rectangle window integral of the species' published cells; the R41
    integral convention applied to the individual species)."""
    d = float(np.sum(spectrum["d"] * spectrum["areas"]))
    v_data = float(np.sum(spectrum["v_data"] * spectrum["areas"] ** 2))
    return {
        "key": f"{spectrum['species']}_multiplicity",
        "kind": "moment",
        "species": spectrum["species"],
        "d": np.asarray([max(d, 0.0)], dtype=float),
        "v_data": np.asarray([max(v_data, SE_FLOOR)], dtype=float),
        "areas": spectrum["areas"],
        "spectrum": spectrum,
    }


def _build_held_out_statistics(sealed: dict, tables_by_species: dict) -> list:
    """The pre-registered held-out register (the card's species-
    separated set), in the register order: the four species-separated
    spectra then the four species-separated moments."""
    stats = []
    spectra = {}
    for species in DATA_SPECIES:
        spectra[species] = _spectrum_statistic(
            sealed, tables_by_species, species)
        stats.append({
            "key": spectra[species]["key"],
            "kind": "spectrum",
            "species": species,
            "d": spectra[species]["d"],
            "v_data": spectra[species]["v_data"],
            "areas": spectra[species]["areas"],
        })
    for species in DATA_SPECIES:
        stats.append(_moment_statistic(spectra[species]))
    keys = [stat["key"] for stat in stats]
    if keys != list(HELD_OUT_KEYS):
        _refuse("the held-out register drifted from the pre-registered "
                "order")
    return stats


def _species_bin_values(weighted_sums: np.ndarray, tables_by_species: dict,
                        sqrt_s: float, n_eff: float) -> dict:
    """The per-species binned predictions in the data units (the R01
    convention m = sums / (pi sqrt(s) N_eff dxF dpT); the self-normalized
    estimator)."""
    out = {}
    for species in SPECIES_FLAT_ORDER:
        table = tables_by_species[species]
        sums = _species_slice(weighted_sums, species)
        areas = np.asarray([
            _cell_area_from_table(table, i)
            for i in range(int(table.data.size))
        ], dtype=float)
        out[species] = sums / (math.pi * float(sqrt_s) * float(n_eff)
                               * areas)
    return out


def _stat_vector(stat: dict, species_values: dict) -> np.ndarray:
    """One held-out statistic's model vector."""
    if stat["kind"] == "spectrum":
        return np.asarray(
            species_values[stat["species"]], dtype=float)
    moment = float(np.sum(
        np.asarray(species_values[stat["species"]], dtype=float)
        * stat["areas"]))
    return np.asarray([moment], dtype=float)


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
# The deviance algebra (the held-out degradation metric)
# ---------------------------------------------------------------------------
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


def observable_deviance(stat_key: str, d: np.ndarray, v_data: np.ndarray,
                        m: np.ndarray, se: np.ndarray) -> float:
    """The summed squared normalized residual of one observable over its
    finite bins: (d - m)^2 / sigma^2 with sigma^2 = v_data + se^2 (the
    published variance plus the between-replica MC variance, floored at
    the engine's deterministic SE floor).  A bin whose data, published
    variance, prediction, or MC error is non-finite is dropped from the
    objective and listed (the R11 convention; never a NaN)."""
    value, _mask = _deviance_core(d, v_data, m, se)
    _ = stat_key
    return float(value)


def record_deviance_drops(stat_key: str, d: np.ndarray,
                          v_data: np.ndarray, m: np.ndarray,
                          se: np.ndarray, dropped: list) -> None:
    """The canonical drop list of one observable (recorded once from the
    pooled leg; the same structural rule governs every variant)."""
    d = np.asarray(d, dtype=float)
    v_data = np.asarray(v_data, dtype=float)
    m = np.asarray(m, dtype=float)
    se = np.asarray(se, dtype=float)
    _value, mask = _deviance_core(d, v_data, m, se)
    for i in range(m.size):
        if bool(mask[i]):
            continue
        reason = "structurally undefined statistic bin (the R11 convention)"
        if not (math.isfinite(float(d[i]))
                and math.isfinite(float(v_data[i]))
                and float(v_data[i]) >= 0.0):
            reason = "non-finite data or published variance"
        elif not math.isfinite(float(m[i])):
            reason = ("the pooled MC error or prediction is non-finite "
                      "(a structurally undefined statistic bin; the R11 "
                      "convention)")
        dropped.append({
            "statistic": stat_key,
            "index": int(i),
            "reason": reason,
        })


def sigma_components(d: np.ndarray, v_data: np.ndarray, m: np.ndarray,
                     se: np.ndarray) -> dict:
    """The data and MC contributions of one observable's -2ln L (the
    card's 'data and MC contributions reported separately' clause): the
    total form sigma^2 = v_data + se^2, the data-only form
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
        "minus2lnl": float(np.sum(residual[mask] ** 2)),
        "minus2lnl_data_only": float(np.sum(residual_data[mask] ** 2)),
        "minus2lnl_mc_only": float(np.sum(residual_mc[mask] ** 2)),
        "n_bins": int(mask.sum()),
    }


def pooled_difference_se(deltas: list) -> float:
    """The pooled MC standard error of the per-block deviance
    differences: the std ddof=1 over the 20 paired replica blocks,
    divided by sqrt(20), floored at the engine's deterministic SE floor
    (fewer than two blocks carry no between-replica estimate and the
    floor stands)."""
    values = np.asarray(deltas, dtype=float)
    if values.size < 2:
        return float(SE_FLOOR)
    with np.errstate(invalid="ignore"):
        out = float(np.std(values, ddof=1)) / math.sqrt(values.size)
    if not math.isfinite(out):
        return float(SE_FLOOR)
    return max(out, float(SE_FLOOR))


def held_out_row(aligned_deviance: float, baseline_deviance: float,
                 se: float) -> dict:
    """One held-out observable's row: the deviance difference (aligned
    minus baseline), the pooled SE of the difference, the pre-registered
    threshold, and the degradation verdict (the aligned deviance exceeds
    the baseline deviance by more than 2.0 x the pooled SE)."""
    delta = float(aligned_deviance) - float(baseline_deviance)
    se = float(se)
    degrades = bool(delta > 0.0)
    beyond = bool(delta > HELD_OUT_DEGRADATION_FACTOR * se)
    return {
        "aligned_deviance": float(aligned_deviance),
        "baseline_deviance": float(baseline_deviance),
        "delta_aligned_minus_baseline": delta,
        "pooled_se": se,
        "threshold": (
            f"{HELD_OUT_DEGRADATION_FACTOR!r} x pooled SE = "
            f"{HELD_OUT_DEGRADATION_FACTOR * se!r}"
        ),
        "degrades": degrades,
        "beyond_2x_se": beyond,
        "named_finding": bool(degrades and beyond),
    }


def depth_stats(hist: dict, n_decays: int) -> dict:
    """The pooled depth statistics of one histogram (the R40 convention):
    the mean, the 95th percentile (the smallest histogram depth whose
    cumulative count reaches 0.95 x n_decays; no interpolation), the
    max, and the guard activations (zero by construction: a guard
    raises and stops the run before any record is written)."""
    items = sorted((int(k), int(v)) for k, v in hist.items())
    total = sum(count for _depth, count in items)
    if total != int(n_decays):
        raise ValueError(
            "depth_stats: the histogram total "
            f"{total} != n_decays {n_decays}"
        )
    if total <= 0:
        raise ValueError("depth_stats: the histogram is empty")
    mean = math.fsum(depth * count for depth, count in items) / float(total)
    threshold = 0.95 * total
    cumulative = 0
    p95 = None
    for depth, count in items:
        cumulative += count
        if cumulative >= threshold:
            p95 = depth
            break
    max_depth = max(depth for depth, count in items if count > 0)
    return {
        "mean": mean,
        "p95": int(p95),
        "max": int(max_depth),
        "guard_activations": 0,
        "n_decays": int(total),
        "convention": (
            "the cascade depth of the variant's decay machinery per "
            "source decay (0 for a stable-identity source and for a "
            "catalog-handoff root, the number of cascade vertices for a "
            "direct-channel line); the 95th percentile is the smallest "
            "histogram depth whose cumulative count reaches 0.95 x "
            "n_decays (no interpolation); the guard activations are "
            "zero by construction (a guard raises and stops the run)"
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


# ---------------------------------------------------------------------------
# The blocks (the sealed production records or the smoke record)
# ---------------------------------------------------------------------------
def _load_blocks(run_dir: Path, smoke: bool) -> list:
    """The sealed production records (or the single smoke record)."""
    if smoke:
        record = _read_json(
            Path(run_dir) / f"{SMOKE_PREFIX}replica_0.json")
        return [record]
    blocks = []
    for replica in range(T3_REPLICAS):
        path = Path(run_dir) / RECORD_FMT.format(replica)
        if not path.is_file():
            _refuse(f"the production record is missing: {path} "
                    "(run --generate first)")
        record = _read_json(path)
        if int(record.get("replica", -1)) != replica:
            _refuse(f"{path}: the record's replica field disagrees")
        blocks.append(record)
    return blocks


def _variant_sums(record: dict, variant_name: str) -> np.ndarray:
    """One variant's pooled bin sums of one record (the flat species
    layout)."""
    pooled = np.zeros(_flat_total(), dtype=float)
    for species in SPECIES_FLAT_ORDER:
        values = np.asarray(
            record["variants"][variant_name]["pooled_sums"][species],
            dtype=float)
        start = species_offset(species)
        pooled[start:start + values.size] = values
    return pooled


def variant_policies_payload() -> dict:
    """The pre-registered per-variant policies (the driver register)."""
    out: dict[str, Any] = {}
    for policy in VARIANTS:
        out[policy.name] = {
            "variant_id": int(policy.variant_id),
            "kappa_b": float(policy.kappa_b),
            "kappa_m": float(policy.kappa_m),
            "b_dec_gev2": float(policy.b_dec),
        }
    out["note"] = (
        "v0 aligned_frozen carries the frozen R41 fitted values (the "
        "corrected sealed R41 archive, no refit); v1 isotropic_"
        "sequential_baseline is card 10's v2 policy; both record the "
        "corrected frozen nominal_width 0.3 GeV and share the "
        "continuum-daughter mass rule of the byte-unchanged committed "
        "continuum config (the superseded card-15 fit's 0.5 GeV "
        "coincided with the committed config value)"
    )
    return out


# ---------------------------------------------------------------------------
# The evaluation (the held-out tables, the delta AIC, the named residuals)
# ---------------------------------------------------------------------------
K_ALIGNED = 4   # kappa_B, kappa_m, b_dec, nominal_width
K_BASELINE = 1  # nominal_width


def aic_of(minus2lnl: float, n_parameters: int) -> float:
    """AIC = -2ln L + 2k (the parameter cost included, plan section 5.5
    condition 6 line 322)."""
    return float(minus2lnl) + 2.0 * int(n_parameters)


def run_evaluate_machinery(run_dir: Path, smoke: bool = False,
                           tables=None, fams=None) -> dict:
    """The held-out evaluation: the per-variant observable tables, the
    exchange bit-identity verification at run scale, the stream-pin
    verification, the held-out deviance and degradation table, the
    -2ln L register with the data and MC contributions, the delta AIC,
    the condition-4 sector checks, and the report payload (no
    wall-clock fields; the report bytes are a pure function of the
    sealed artifacts, the pre-registered seeds, and the committed
    code)."""
    _ = fams
    run_dir = Path(run_dir)
    pre = _load_pre_registration(run_dir)
    _assert_pre_registration(pre, run_dir)
    if smoke:
        if tables is None:
            tables = _load_smoke_tables()
        _init_species_layout(tables)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = _sqrt_s()
        sealed = {}
        for species in ("pi_plus", "pi_minus", "k_plus", "k_minus",
                        "proton", "antiproton"):
            table = tables_by_species[species]
            sealed[species] = {
                "d": [1.0] * int(table.data.size),
                "v_data": [0.01] * int(table.data.size),
                "xF_mid": [float(v) for v in table.xf_centers],
                "pT_mid": [float(v) for v in table.pt_centers],
            }
    else:
        _refuse_smoke_artifacts(run_dir)
        if tables is None:
            tables = _load_production_tables()
            tables.append(_pi_zero_model_table(tables))
        _init_species_layout(tables)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = _sqrt_s()
        sealed = _load_sealed_r01([
            t for t in tables if t.species != "pi_zero"
        ])
    stats = _build_held_out_statistics(sealed, tables_by_species)
    stat_keys = [stat["key"] for stat in stats]
    if stat_keys != list(HELD_OUT_KEYS):
        _refuse("the held-out register drifted from the pre-registered "
                "order")
    stat_index = {key: i for i, key in enumerate(HELD_OUT_KEYS)}
    blocks = _load_blocks(run_dir, smoke)
    n_replicas = len(blocks)
    n_events_total = sum(int(b["n_events"]) for b in blocks)

    # -- the per-variant model vectors (per block and pooled) ---------
    per_variant_vectors: dict[str, list] = {
        name: [] for name in VARIANT_NAMES}
    per_variant_species: dict[str, list] = {name: [] for name
                                            in VARIANT_NAMES}
    for block in blocks:
        n_eff = float(block["n_events"])
        for name in VARIANT_NAMES:
            variant_sums = _variant_sums(block, name)
            values = _species_bin_values(
                variant_sums, tables_by_species, sqrt_s, n_eff)
            per_variant_vectors[name].append([
                _stat_vector(stat, values) for stat in stats])
            per_variant_species[name].append({
                species: np.asarray(values[species], dtype=float)
                for species in SPECIES_FLAT_ORDER
            })
    pooled_flat: dict[str, np.ndarray] = {
        name: np.zeros(_flat_total(), dtype=float) for name in VARIANT_NAMES
    }
    for block in blocks:
        for name in VARIANT_NAMES:
            pooled_flat[name] = pooled_flat[name] + _variant_sums(
                block, name)
    pooled_n_eff = float(n_events_total)
    pooled_values = {
        name: _species_bin_values(
            pooled_flat[name], tables_by_species, sqrt_s, pooled_n_eff)
        for name in VARIANT_NAMES
    }
    pooled_vectors = {
        name: [_stat_vector(stat, pooled_values[name]) for stat in stats]
        for name in VARIANT_NAMES
    }
    se_vectors: dict[str, list] = {}
    for name in VARIANT_NAMES:
        se_vectors[name] = [
            se_from(np.asarray([
                per_variant_vectors[name][k][i]
                for k in range(n_replicas)
            ], dtype=float))
            for i in range(len(stats))
        ]

    # -- the deviance and -2ln L register, the degradation table ------
    dropped: list = []
    deviance_table: dict[str, dict] = {}
    likelihood_table: dict[str, dict] = {}
    per_block_deviances: dict[str, dict[str, list]] = {
        name: {} for name in VARIANT_NAMES
    }
    for v_index, stat in enumerate(stats):
        row: dict[str, Any] = {}
        likelihood_row: dict[str, Any] = {}
        for name in VARIANT_NAMES:
            m_vec = np.asarray(pooled_vectors[name][v_index], dtype=float)
            se_vec = np.asarray(se_vectors[name][v_index], dtype=float)
            record_deviance_drops(
                stat["key"], stat["d"], stat["v_data"], m_vec, se_vec,
                dropped)
            value, _mask = _deviance_core(
                stat["d"], stat["v_data"], m_vec, se_vec)
            row[name] = float(value)
            components = sigma_components(
                stat["d"], stat["v_data"], m_vec, se_vec)
            likelihood_row[name] = {
                "minus2lnl": components["minus2lnl"],
                "minus2lnl_data_only": components["minus2lnl_data_only"],
                "minus2lnl_mc_only": components["minus2lnl_mc_only"],
                "n_bins": components["n_bins"],
            }
            per_block_deviances[name][stat["key"]] = [
                _deviance_core(
                    stat["d"], stat["v_data"],
                    np.asarray(
                        per_variant_vectors[name][k][v_index], dtype=float),
                    se_vec)[0]
                for k in range(n_replicas)
            ]
        deviance_table[stat["key"]] = row
        likelihood_table[stat["key"]] = likelihood_row
    held_out_rows: dict[str, dict] = {}
    for stat in stats:
        deltas_b = [
            per_block_deviances[BASELINE_POLICY.name][stat["key"]][k]
            - per_block_deviances[ALIGNED_POLICY.name][stat["key"]][k]
            for k in range(n_replicas)
        ]
        held_out_rows[stat["key"]] = held_out_row(
            deviance_table[stat["key"]][ALIGNED_POLICY.name],
            deviance_table[stat["key"]][BASELINE_POLICY.name],
            pooled_difference_se(deltas_b))
    aligned_total = math.fsum(
        likelihood_table[key][ALIGNED_POLICY.name]["minus2lnl"]
        for key in HELD_OUT_KEYS)
    baseline_total = math.fsum(
        likelihood_table[key][BASELINE_POLICY.name]["minus2lnl"]
        for key in HELD_OUT_KEYS)
    delta_aic = delta_aic_block(aligned_total, baseline_total)
    held_out_all_preserved = all(
        not row["named_finding"] for row in held_out_rows.values())

    # -- the per-bin named residuals (the house |sigma| > 3.0 form) ----
    sigma_residuals: list = []
    for v_index, stat in enumerate(stats):
        for name in VARIANT_NAMES:
            m_vec = np.asarray(pooled_vectors[name][v_index], dtype=float)
            se_vec = np.asarray(se_vectors[name][v_index], dtype=float)
            d = np.asarray(stat["d"], dtype=float)
            v_data = np.asarray(stat["v_data"], dtype=float)
            sigma2 = v_data + np.maximum(se_vec, SE_FLOOR) ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                residual = (d - m_vec) / np.sqrt(
                    np.where(sigma2 > 0.0, sigma2, 1.0))
            for i in range(m_vec.size):
                value = float(residual[i])
                if math.isfinite(value) and abs(value) > SIGMA_TOLERANCE:
                    sigma_residuals.append({
                        "kind": "residual_above_tolerance",
                        "variant": name,
                        "observable": stat["key"],
                        "bin_index": int(i),
                        "observed_sigma": value,
                        "bound": f"|residual| > {SIGMA_TOLERANCE!r} sigma",
                        "robust_statistics_cross_reference":
                            SIGMA_TOLERANCE_NOTE,
                    })
    sigma_n_above = len(sigma_residuals)

    # -- condition 4: the six paper-window species and the feed-down
    #    fractions (the per-sector validation) --------------------------
    six_species = ("pi_plus", "pi_minus", "k_plus", "k_minus",
                   "proton", "antiproton")
    species_sector_rows: dict[str, dict] = {}
    for species in six_species:
        sealed_d = np.asarray(sealed[species]["d"], dtype=float)
        sealed_v = np.asarray(sealed[species]["v_data"], dtype=float)
        row = {}
        for name in VARIANT_NAMES:
            m_vec = np.asarray(pooled_values[name][species], dtype=float)
            if species in DATA_SPECIES:
                se_vec = np.asarray(
                    se_vectors[name][stat_index[f"{species}_spectrum"]],
                    dtype=float)
            else:
                # a condition-4-only sector (k_plus, k_minus): the
                # species MC error from the per-block species values
                stack = np.asarray([
                    per_variant_species[name][k][species]
                    for k in range(n_replicas)
                ], dtype=float)
                se_vec = se_from(stack)
            value, _mask = _deviance_core(sealed_d, sealed_v, m_vec, se_vec)
            row[name] = float(value)
        baseline_v = row[BASELINE_POLICY.name]
        aligned_v = row[ALIGNED_POLICY.name]
        relative = (
            (aligned_v - baseline_v) / baseline_v
            if baseline_v > 0.0 else float("inf")
        )
        species_sector_rows[species] = {
            "aligned_deviance": aligned_v,
            "baseline_deviance": baseline_v,
            "relative_degradation": relative,
            "threshold": (
                f"{SECTOR_DEGRADATION_FRACTION!r} of the baseline "
                "deviance"
            ),
            "flagged": bool(relative > SECTOR_DEGRADATION_FRACTION),
        }
    # -- the feed-down fractions (the card-9 convention) ---------------
    feeddown_rows: dict[str, dict] = {}
    for name in VARIANT_NAMES:
        entries = {}
        for key in SPECIES_KEYS:
            per_replica = []
            for block in blocks:
                tallies = block["variants"][name]["feed_down"]["tallies"]
                slot = tallies["per_species"][key]
                if int(slot["n"]) > 0:
                    per_replica.append(
                        float(slot["n_resonant"]) / float(slot["n"]))
                else:
                    per_replica.append(None)
            mean, se = _mean_se(per_replica)
            pooled_tallies = {
                "n": sum(int(b["variants"][name]["feed_down"]["tallies"]
                             ["per_species"][key]["n"]) for b in blocks),
                "n_resonant": sum(
                    int(block["variants"][name]["feed_down"]["tallies"]
                        ["per_species"][key]["n_resonant"]) for b in blocks),
                "by_parent": {},
            }
            for block in blocks:
                slot = block["variants"][name]["feed_down"]["tallies"][
                    "per_species"][key]
                for parent, count in slot["by_parent"].items():
                    pooled_tallies["by_parent"][parent] = (
                        pooled_tallies["by_parent"].get(parent, 0)
                        + int(count))
            fraction = (
                float(pooled_tallies["n_resonant"])
                / float(pooled_tallies["n"])
                if pooled_tallies["n"] > 0 else None
            )
            entries[key] = {
                "pooled_fraction": fraction,
                "between_replica_se": se,
                "pooled_tallies": pooled_tallies,
            }
        feeddown_rows[name] = entries
    feeddown_sector_flags: dict[str, dict] = {}
    for key in SPECIES_KEYS:
        base_frac = feeddown_rows[BASELINE_POLICY.name][key][
            "pooled_fraction"]
        aligned_frac = feeddown_rows[ALIGNED_POLICY.name][key][
            "pooled_fraction"]
        relative = None
        if base_frac is not None and aligned_frac is not None \
                and base_frac > 0.0:
            relative = abs(aligned_frac - base_frac) / base_frac
        flagged = bool(relative is not None
                       and relative > SECTOR_DEGRADATION_FRACTION)
        feeddown_sector_flags[key] = {
            "aligned_fraction": aligned_frac,
            "baseline_fraction": base_frac,
            "relative_change": relative,
            "threshold": (
                f"{SECTOR_DEGRADATION_FRACTION!r} of the baseline "
                "fraction"
            ),
            "flagged": flagged,
        }

    # -- the exchange bit-identity verification at run scale ----------
    exchange_ok = True
    exchange_detail = []
    for block in blocks:
        shared = str(block["exchange_layer"][
            "shared_event_hash_set_sha256"])
        digests = {
            name: str(block["variants"][name]["exchange_hash_set_sha256"])
            for name in VARIANT_NAMES
        }
        equal = all(d == shared for d in digests.values())
        exchange_ok = exchange_ok and equal
        exchange_detail.append({
            "replica": int(block["replica"]),
            "n_events": int(block["n_events"]),
            "variants_digests_equal": bool(equal),
        })
    if not exchange_ok:
        named_exchange = {
            "index": 0,
            "kind": "exchange_bit_identity_failure",
            "observed": exchange_detail,
            "diagnosis": (
                "the exchange layer is not bit-identical across the two "
                "variants (a per-variant rolling exchange digest differs "
                "from the shared per-event digest); the baseline-"
                "comparison premise of the card is violated and the run "
                "refuses"
            ),
        }
    else:
        named_exchange = None

    # -- the stream pin (the pinned-key probe replay) ------------------
    probe_event = (SMOKE_EVENT_INDEX_BASE if smoke else EVENT_INDEX_BASE)
    stream_probe = {
        "key": {
            "master_seed": MASTER_SEED,
            "event_index": probe_event,
            "variant_ids": [policy.variant_id for policy in VARIANTS],
        },
        "first_channel_uniform_per_variant": [
            float(variant_stream(
                MASTER_SEED, probe_event, policy.variant_id)[0].random())
            for policy in VARIANTS
        ],
        "note": (
            "the pinned-key replay of card 10's variant_stream contract: "
            "the same (master_seed, event_index, variant_id) reproduces "
            "the stream byte for byte and the two variants' streams are "
            "disjoint (the run-scale determinism leg is the "
            "clean-validate probe replica re-run)"
        ),
    }

    # -- the per-variant diagnostic tables -----------------------------
    multiplicity_table: dict[str, dict] = {}
    for name in VARIANT_NAMES:
        entries = {}
        for key in SPECIES_KEYS:
            per_replica_means = [
                float(b["variants"][name]["multiplicity"][key]["sum_n"])
                / float(b["n_events"]) for b in blocks
            ]
            mean, se = _mean_se(per_replica_means)
            entries[key] = {
                "mean_multiplicity": mean,
                "between_replica_se": se,
            }
        multiplicity_table[name] = entries
    model_only_table: dict[str, dict] = {}
    for key in MODEL_ONLY_KEYS:
        species = key.replace("_spectrum", "").replace("_multiplicity", "")
        per_replica_values = []
        for block in blocks:
            n_eff = float(block["n_events"])
            variant_sums = {}
            for name in VARIANT_NAMES:
                values = _species_bin_values(
                    _variant_sums(block, name), tables_by_species, sqrt_s,
                    n_eff)[species]
                variant_sums[name] = float(np.sum(values))
            aligned_pooled = _species_bin_values(
                pooled_flat[ALIGNED_POLICY.name], tables_by_species,
                sqrt_s, pooled_n_eff)[species]
            per_replica_values.append({
                "variant_sums": variant_sums,
                "aligned_pooled_sum": float(np.sum(aligned_pooled)),
            })
        model_only_table[key] = {
            "species": species,
            "model_only": True,
            "note": PI_ZERO_TABLE_NOTE,
            "per_replica_sums": per_replica_values,
        }
    depth_table: dict[str, dict] = {}
    for name in VARIANT_NAMES:
        pooled_hist: dict[int, int] = {}
        for block in blocks:
            for depth, count in block["variants"][name][
                    "depth_histogram"].items():
                pooled_hist[int(depth)] = (
                    pooled_hist.get(int(depth), 0) + int(count))
        depth_table[name] = depth_stats(pooled_hist, n_events_total * 2)

    # -- no NaN (all recorded observables finite; the drops listed) ----
    finite_ok = True
    for name in VARIANT_NAMES:
        for vector in pooled_vectors[name]:
            for value in vector:
                if value is not None and not math.isfinite(float(value)):
                    finite_ok = False
    named_residuals: list = []
    for stat in stats:
        row = held_out_rows[stat["key"]]
        if row["named_finding"]:
            named_residuals.append({
                "index": len(named_residuals) + 1,
                "kind": "held_out_degradation_beyond_2x_se",
                "observable": stat["key"],
                "observed": row["delta_aligned_minus_baseline"],
                "bound": row["threshold"],
                "diagnosis": (
                    "the frozen aligned configuration's deviance "
                    "degrades beyond 2.0 x the pooled MC standard error "
                    "of the difference on this held-out observable "
                    "(condition iii finding; the bound is never relaxed)"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
    for entry in sigma_residuals[:200]:
        named_residuals.append({
            "index": len(named_residuals) + 1,
            **entry,
        })
    sector_flags = []
    for species, row in species_sector_rows.items():
        if row["flagged"]:
            sector_flags.append({"sector": f"{species}_spectrum", **row})
    for key, row in feeddown_sector_flags.items():
        if row["flagged"]:
            sector_flags.append({"sector": f"{key}_feed_down", **row})
    for flag in sector_flags:
        named_residuals.append({
            "index": len(named_residuals) + 1,
            "kind": "condition_4_sector_degradation",
            "sector": flag["sector"],
            "observed": flag.get("relative_degradation",
                                 flag.get("relative_change")),
            "bound": flag["threshold"],
            "stated_physics_reason": None,
            "diagnosis": (
                "the sector degrades beyond the pre-registered 5% in "
                "validation (condition iv); the physics reason is "
                "stated by the run record's follow-up or the finding is "
                "a Gate 4 finding (the bound is never relaxed)"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    if dropped:
        named_residuals.append({
            "index": len(named_residuals) + 1,
            "kind": "structurally_dropped_bins",
            "observed": dropped[:50],
            "n_dropped": len(dropped),
            "diagnosis": (
                "statistic bins dropped by structure (a non-finite data "
                "entry, published variance, prediction, or MC error); "
                "the deviances are sums over the finite bins and the "
                "drops are listed, never silent"
            ),
            "robust_statistics_cross_reference": SE_CONVENTION,
        })
    if named_exchange is not None:
        named_exchange["index"] = len(named_residuals) + 1
        named_residuals.append(named_exchange)

    report = {
        "schema":
            "na49_mixing.staged_r44_frozen_nonstrange_validation."
            "report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "events_checked": int(n_events_total),
        "replicas": int(n_replicas),
        "variants": list(VARIANT_NAMES),
        "frozen_parameters": {
            "kappa_B": ALIGNED_POLICY.kappa_b,
            "kappa_m": ALIGNED_POLICY.kappa_m,
            "b_dec_gev2": ALIGNED_POLICY.b_dec,
            "baryonic_nominal_width_gev": FROZEN_WIDTH_GEV,
            "source": "sealed R41 archive",
            "refit": False,
        },
        "baseline_parameters": {
            "kappa_B": BASELINE_POLICY.kappa_b,
            "kappa_m": BASELINE_POLICY.kappa_m,
            "b_dec_gev2": BASELINE_POLICY.b_dec,
            "baryonic_nominal_width_gev": FROZEN_WIDTH_GEV,
            "source": "card 10's v2 policy (ABLATION_POLICIES)",
        },
        "held_out_set": {
            "species_separated": list(HELD_OUT_SPECIES_NAMED),
            "likelihood_register": list(HELD_OUT_KEYS),
            "model_only_keys": list(MODEL_ONLY_KEYS),
            "multiplicity_moments": "species-separated",
            "disjoint_from_training": True,
        },
        "held_out_set_check": held_out_disjointness_check(),
        "deviance_table": deviance_table,
        "likelihood_table": likelihood_table,
        "held_out": {
            "rows": held_out_rows,
            "threshold_factor": HELD_OUT_DEGRADATION_FACTOR,
            "all_preserved": bool(held_out_all_preserved),
            "note": HELD_OUT_DEGRADATION_NOTE,
        },
        "delta_aic": delta_aic,
        "residuals_above_tolerance": {
            "sigma_tolerance": SIGMA_TOLERANCE,
            "n_above": sigma_n_above,
            "entries": sigma_residuals[:200],
            "note": SIGMA_TOLERANCE_NOTE,
        },
        "sector_checks": {
            "species_deviance_5pct": species_sector_rows,
            "feeddown_fraction_5pct": feeddown_sector_flags,
            "note": SECTOR_DEGRADATION_NOTE,
        },
        "exchange_bit_identity": {
            "variants": len(VARIANT_NAMES),
            "events": int(n_events_total),
            "layers": ["topology", "exchange", "spectral_state"],
            "per_replica": exchange_detail,
            "pass": bool(exchange_ok),
        },
        "stream_pin": stream_probe,
        "depth_statistics": depth_table,
        "multiplicity": multiplicity_table,
        "feed_down": {
            "per_variant": feeddown_rows,
            "convention": (
                "the card-9 lineage convention carried on the walk "
                "machinery (a final particle is resonant feed-down when "
                "its producing vertex's parent is a resonance other than "
                "the drawn root); the fractions cover every blend-routed "
                "source's finals per variant"
            ),
        },
        "model_only_diagnostics": model_only_table,
        "dropped_bins": dropped[:100],
        "n_dropped_bins": len(dropped),
        "all_recorded_observables_finite": bool(finite_ok),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "obligation_5": OBLIGATION5_NOTE,
        },
        "named_residual_register": named_residuals,
        "no_fitting": True,
        "optimizer_invoked": False,
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


def delta_aic_block(aligned_minus2lnl: float,
                    baseline_minus2lnl: float) -> dict:
    """The delta AIC block (the parameter cost included, plan section
    5.5 condition 6 line 322): AIC = -2ln L + 2k with the aligned
    configuration carrying kappa_B, kappa_m, b_dec, nominal_width
    (k = 4) and the baseline carrying nominal_width (k = 1); favored
    when negative."""
    aic_aligned = float(aligned_minus2lnl) + 2.0 * K_ALIGNED
    aic_baseline = float(baseline_minus2lnl) + 2.0 * K_BASELINE
    delta = aic_aligned - aic_baseline
    return {
        "aligned_minus2lnl": float(aligned_minus2lnl),
        "baseline_minus2lnl": float(baseline_minus2lnl),
        "aic_aligned": aic_aligned,
        "aic_baseline": aic_baseline,
        "k_aligned": K_ALIGNED,
        "k_baseline": K_BASELINE,
        "parameter_cost": "included",
        "delta_aic": delta,
        "favored": bool(delta < 0.0),
        "note": (
            "the parameter cost is included: the aligned configuration "
            "carries kappa_B, kappa_m, b_dec, nominal_width (k = 4); "
            "the baseline carries nominal_width (k = 1); the verdict "
            "(favored / not favored) is the held-out likelihood "
            "evidence of the Gate 4 retain"
        ),
    }


def held_out_disjointness_check() -> dict:
    """The held-out set invariants (the card's held-out invariants):
    the register is species-separated and disjoint from the R41 training
    combinations (no held-out key is an R41 training observable, a
    charge-symmetric combination, or a pooled moment)."""
    training = set(R41_TRAINING_OBSERVABLES)
    overlap = sorted(set(HELD_OUT_KEYS) & training)
    model_overlap = sorted(set(MODEL_ONLY_KEYS) & training)
    combination_keys = {
        "pi_sigma_spectrum", "b_transport_spectrum",
        "pi_sigma_seagull", "b_transport_seagull",
        "pi_sigma_multiplicity", "b_transport_multiplicity",
    }
    combination_overlap = sorted(set(HELD_OUT_KEYS) & combination_keys)
    species_separated = all(
        key.split("_")[0] in {"pi", "p"} or key.startswith(
            ("pi_plus", "pi_minus", "pi_zero", "proton", "antiproton"))
        for key in HELD_OUT_KEYS
    )
    disjoint = not overlap and not model_overlap and not combination_overlap
    return {
        "held_out_keys": list(HELD_OUT_KEYS),
        "r41_training_observables": list(R41_TRAINING_OBSERVABLES),
        "overlap": overlap,
        "combination_overlap": combination_overlap,
        "species_separated": bool(species_separated and disjoint),
        "disjoint": bool(disjoint),
        "note": HELD_OUT_DISJOINT_NOTE,
    }


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """Write gate4_r44.json (evidence only), then stop."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    records = [
        _read_json(run_dir / RECORD_FMT.format(replica))
        for replica in range(T3_REPLICAS)
    ]
    conservation = {
        "events_checked": sum(int(r["n_events"]) for r in records),
        "violations": 0,
        "guard_activations": 0,
        "worst_energy_closure_gev": max(
            float(r["variants"][name]["conservation"][
                "worst_energy_closure_gev"])
            for r in records for name in VARIANT_NAMES),
        "tolerance": EVENT_CLOSURE_TOL,
    }
    conservation["pass"] = bool(
        conservation["violations"] == 0
        and conservation["guard_activations"] == 0
        and conservation["events_checked"]
        == T3_REPLICAS * T3_EVENTS_PER_REPLICA
        and all(
            r["variants"][name]["conservation"]["event_b_q_s_conserved"]
            for r in records for name in VARIANT_NAMES
        )
    )
    # -- the condition cross-references (the sealed R43 and R41 gates;
    #    the seals already verify the archives) --------------------------
    r43_gate = _read_json(MONO_ROOT / R43_RUN_DIR / R43_GATE_NAME)
    r41_gate = _read_json(MONO_ROOT / R41_RUN_DIR / R41_GATE_NAME)
    r43_beyond = r43_gate.get("beyond_isotropic") or {}
    r41_stability = r41_gate.get("replica_block_stability") or {}
    held_out_rows = (report.get("held_out") or {}).get("rows") or {}
    delta_aic = report.get("delta_aic") or {}
    sector_checks = report.get("sector_checks") or {}
    sector_findings = []
    for species, row in (sector_checks.get("species_deviance_5pct")
                         or {}).items():
        if row.get("flagged"):
            sector_findings.append(
                {"sector": species, **row})
    for key, row in (sector_checks.get("feeddown_fraction_5pct")
                     or {}).items():
        if row.get("flagged"):
            sector_findings.append(
                {"sector": f"{key}_feed_down", **row})
    held_out_findings = {
        key: {
            "deviance": row["aligned_deviance"],
            "baseline_deviance": row["baseline_deviance"],
            "difference": row["delta_aligned_minus_baseline"],
            "pooled_se": row["pooled_se"],
            "named_finding": row["named_finding"],
        }
        for key, row in held_out_rows.items()
    }
    stability_pass = all(
        bool(entry.get("pass")) for entry in r41_stability.values()
    ) if r41_stability else False
    gate = {
        "schema": "na49_mixing.staged_r44_frozen_nonstrange_validation.gate/1",
        "gate": "gate 4 (plan section 10, line 806)",
        "run": RUN_ID,
        "card": CARD,
        "clause": "held-out shapes",
        "variants": list(VARIANT_NAMES),
        "frozen_parameters": {
            "kappa_B": ALIGNED_POLICY.kappa_b,
            "kappa_m": ALIGNED_POLICY.kappa_m,
            "b_dec": ALIGNED_POLICY.b_dec,
            "baryonic_nominal_width": FROZEN_WIDTH_GEV,
            "source": f"sealed {R41_RUN_DIR}/{R41_FIT_REPORT_NAME}",
            "refit": False,
        },
        "superseded_frozen": {
            "kappa_B": SUPERSEDED_R41_FITTED["kappa_B"],
            "kappa_m": SUPERSEDED_R41_FITTED["kappa_m"],
            "b_dec": SUPERSEDED_R41_FITTED["b_dec"],
            "baryonic_nominal_width":
                SUPERSEDED_R41_FITTED["baryonic_nominal_width"],
            "source": SUPERSEDED_R41_RUN_DIR,
            "status": "superseded by relitigation",
        },
        "events_checked": conservation["events_checked"],
        "held_out_set": {
            "species_separated": list(HELD_OUT_SPECIES_NAMED),
            "multiplicity_moments": "species-separated",
            "likelihood_register": list(HELD_OUT_KEYS),
            "model_only_keys": list(MODEL_ONLY_KEYS),
        },
        "conservation": conservation,
        "conditions": {
            "1_conservation": (
                "pass" if conservation["pass"] else
                "FAIL (the conservation evidence of this run)"),
            "2_training_cross_reference": {
                "source": f"{R43_RUN_DIR}/{R43_GATE_NAME}",
                "digest": _sha256_file(
                    MONO_ROOT / R43_RUN_DIR / R43_GATE_NAME),
                "summary": (
                    "the sealed R43 gate's beyond-isotropic register "
                    "(the training-side comparison at the nominal "
                    "point; this run does not re-evaluate the training "
                    "set)"
                ),
                "all_pass": bool(r43_beyond_all_pass(r43_gate)),
                "rows": (r43_beyond_rows(r43_gate)),
            },
            "3_held_out": {
                "per_observable": held_out_findings,
                "degradation_bound": (
                    f"{HELD_OUT_DEGRADATION_FACTOR!r} x pooled MC SE of "
                    "the difference"
                ),
                "sigma_tolerance": SIGMA_TOLERANCE,
                "all_preserved": bool(
                    report.get("held_out", {}).get("all_preserved")),
            },
            "4_sector_degradation_5pct": sector_findings,
            "5_replica_stability_cross_reference": {
                "source": f"{R41_RUN_DIR}/{R41_GATE_NAME}",
                "digest": _sha256_file(
                    MONO_ROOT / R41_RUN_DIR / R41_GATE_NAME),
                "parameters": {
                    name: {
                        "pass": bool(entry.get("pass")),
                        "spread": entry.get("spread"),
                        "bound": entry.get("bound"),
                    }
                    for name, entry in r41_stability.items()
                },
                "all_pass": bool(stability_pass),
                "note": (
                    "the sealed R41 replica-block stability table (all 8 "
                    "blocks, plan section 5.5 item 5); a failing R41 "
                    "parameter is a named finding feeding the Gate 4 "
                    "unconstrained clause"
                ),
            },
            "6_delta_aic": {
                "aligned_vs_baseline":
                    delta_aic.get("delta_aic"),
                "parameter_cost": delta_aic.get("parameter_cost"),
                "k_aligned": delta_aic.get("k_aligned"),
                "k_baseline": delta_aic.get("k_baseline"),
                "favored": bool(delta_aic.get("favored")),
                "note": delta_aic.get("note"),
            },
            "7_color_singlet_audit": (
                "pass (conservation evidence, this run)"
                if conservation["pass"] else
                "FAIL (the conservation evidence of this run)"
            ),
        },
        "exchange_bit_identity": {
            "variants": (report.get("exchange_bit_identity") or {}).get(
                "variants"),
            "events": (report.get("exchange_bit_identity") or {}).get(
                "events"),
            "layers": (report.get("exchange_bit_identity") or {}).get(
                "layers"),
            "pass": (report.get("exchange_bit_identity") or {}).get("pass"),
        },
        "support_boundary": {
            "both_variants": "multi-vertex (identical)",
            "regeneration_required": False,
        },
        "deviance_table": report.get("deviance_table"),
        "likelihood_table": report.get("likelihood_table"),
        "held_out": report.get("held_out"),
        "delta_aic_record": delta_aic,
        "depth_statistics": report.get("depth_statistics"),
        "multiplicity": report.get("multiplicity"),
        "feed_down": report.get("feed_down"),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "obligation_5": OBLIGATION5_NOTE,
        },
        "named_residual_register": report.get(
            "named_residual_register", []),
        "no_fitting": True,
        "optimizer_invoked": False,
        "report_digest": report.get("digest"),
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
    print(f"[r44] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


def r43_beyond_all_pass(gate: dict) -> bool:
    """The sealed R43 gate's beyond-isotropic aggregate verdict."""
    return bool((gate.get("beyond_isotropic") or {}).get("pass"))


def r43_beyond_rows(gate: dict) -> dict:
    """The sealed R43 gate's per-observable beyond-isotropic rows."""
    rows = (gate.get("beyond_isotropic") or {}).get("rows") or {}
    return {
        key: {
            "delta_v2_minus_v3": row.get("delta_v2_minus_v3"),
            "pass": row.get("pass"),
        }
        for key, row in rows.items()
    }


# ---------------------------------------------------------------------------
# The seal and the clean-environment validation
# ---------------------------------------------------------------------------
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
            "na49_mixing.staged_r44_frozen_nonstrange_validation.seal/1",
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
    }
    _write_json(run_dir / SEAL_NAME, manifest)
    return manifest


def run_seal(run_dir: Path) -> dict:
    """Write seal_manifest.json and SHA256SUMS.txt over the run dir,
    then stop.  The run dir is immutable after the seal."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    required = (
        PREREGISTRATION_NAME, PROVENANCE_NAME, REPORT_NAME,
        GATE_RECORD_NAME, CLEAN_ENV_NAME,
    )
    for name in required:
        if not (run_dir / name).is_file():
            _refuse(f"the run dir is missing {name}: {run_dir}")
    missing = [
        RECORD_FMT.format(replica) for replica in range(T3_REPLICAS)
        if not (run_dir / RECORD_FMT.format(replica)).is_file()
    ]
    if missing:
        _refuse(f"the production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r44] sealed {run_dir} ({n_entries} entries)", flush=True)
    return manifest


def run_clean_validate(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """Verify the seal, re-run the evaluation from a clean interpreter
    over the sealed artifacts with a byte-identical report, re-run the
    pre-registered probe replica in memory against its sealed record,
    write clean_environment_commands.json, then stop."""
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
             "step order lands the seal over it)"
    )
    # -- the evaluation re-run over the sealed artifacts --------------
    sealed_report = _read_json(run_dir / REPORT_NAME)
    rerun_report = run_evaluate_machinery(run_dir, smoke=False)
    report_matches = (
        rerun_report.get("digest") == sealed_report.get("digest"))
    # -- the probe replica re-run in memory ---------------------------
    tables = _load_production_tables()
    tables.append(_pi_zero_model_table(tables))
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
    _set_worker_context(context)
    probe = PROBE_REPLICA
    seed = raw_replica_seed(probe)
    tasks = _chunk_tasks(
        probe, seed, probe, T3_EVENTS_PER_REPLICA,
        EVENT_INDEX_BASE, smoke=False)
    parts = list(_map_chunks(tasks, workers, "probe replica"))
    aggregated = _aggregate_chunks(parts)
    pre = _load_pre_registration(run_dir)
    aggregated["pre_registration_sha256"] = _pre_registration_digest_of(pre)
    record = _replica_record(
        probe, seed, probe, T3_EVENTS_PER_REPLICA, aggregated)
    sealed_probe = _read_json(run_dir / RECORD_FMT.format(probe))
    probe_checks = {}
    for name in VARIANT_NAMES:
        sealed_variant = sealed_probe["variants"][name]
        fresh_variant = record["variants"][name]
        probe_checks[name] = bool(
            fresh_variant["event_hash_set_sha256"]
            == sealed_variant["event_hash_set_sha256"]
            and fresh_variant["exchange_hash_set_sha256"]
            == sealed_variant["exchange_hash_set_sha256"]
            and fresh_variant["pooled_sums"]
            == sealed_variant["pooled_sums"]
            and fresh_variant["routing"] == sealed_variant["routing"]
            and fresh_variant["depth_histogram"]
            == sealed_variant["depth_histogram"]
            and fresh_variant["multiplicity"]
            == sealed_variant["multiplicity"]
            and fresh_variant["feed_down"]["tallies"]
            == sealed_variant["feed_down"]["tallies"]
        )
    probe_matches = bool(
        record["exchange_layer"]["shared_event_hash_set_sha256"]
        == sealed_probe["exchange_layer"]["shared_event_hash_set_sha256"]
        and all(probe_checks.values())
    )
    payload = {
        "schema":
            "na49_mixing.staged_r44_frozen_nonstrange_validation."
            "clean_env/1",
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            "cd dev/mixing2006 && python -m pytest "
            "tests/test_staged_r44_frozen_nonstrange_validation.py -q",
            "python -m pytest tests/test_legacy_v1_freeze.py "
            "-W error::UserWarning -q",
            f"python helpers/benchmarking/"
            f"staged_r44_frozen_nonstrange_validation.py "
            f"--clean-validate --run-dir {run_dir}",
        ],
        "seal": {"entries": entries, "mismatches": mismatches,
                 "state": seal_state},
        "report_digest": sealed_report.get("digest"),
        "report_rerun_digest": rerun_report.get("digest"),
        "report_hash_match": bool(report_matches),
        "probe_replica": {
            "replica": PROBE_REPLICA,
            "per_variant_matches": probe_checks,
            "matches_sealed_record": bool(probe_matches),
        },
        "verified_utc": _now_utc(),
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not report_matches:
        _refuse(
            "the clean-interpreter evaluation does not reproduce the "
            "sealed report digest "
            f"(sealed {sealed_report.get('digest')} vs re-run "
            f"{rerun_report.get('digest')})")
    if not probe_matches:
        _refuse("the probe replica does not match its sealed record")
    print(f"[r44] clean validation written: {run_dir / CLEAN_ENV_NAME}",
          flush=True)
    return payload


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each mode performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R44 driver: the frozen nonstrange validation "
                    "(TASK-20260920-r4-run-r44-frozen-nonstrange-"
                    "validation)")
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
        help="the held-out evaluation (held_out_report.json), then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate4_r44.json, then stop")
    parser.add_argument(
        "--seal", action="store_true",
        help="write the seal manifest and SHA256SUMS.txt, then stop")
    parser.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the evaluation from a clean "
             "interpreter, then stop")
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
        result = run_smoke(run_dir, workers=int(args.workers))
        print(f"[r44] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(
            run_dir, replicas=args.replica, workers=int(args.workers))
        print(f"[r44] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        report = run_evaluate_machinery(run_dir, smoke=False)
        _write_json(run_dir / REPORT_NAME, report)
        print(f"[r44] held-out report written (digest "
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
