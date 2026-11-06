#!/usr/bin/env python3
"""R43 driver: the cascade ablation (Round 4, card 14; the spectrum
recording reworked to the dense-lattice-index convention by the card-18
relitigation).

Task ``TASK-20260920-r4-run-r43-cascade-ablation`` (plan section 10 run
R43, line 790; the Round 4 card set, card 14; plan section 23, lines
1821-1823).  R43 is the Gate 4 "improves held-out longitudinal and
transverse shapes beyond the isotropic sequential cascade" clause
evidence: the aligned sequential cascade (v3) is compared against the
isotropic sequential baseline (v2) with the isotropic direct decay (v1)
as the family floor, on identical exchange events and common random
numbers (plan lines 790-798).  The aligned variant runs at the nominal
point (kappa_B, kappa_M, b_dec) = (1.0, 1.0, 1.0 GeV^-2), the
pre-registered R40 starting value (round 4 index item 3), not a fit
parameter: the fitted-parameter version is card 15's (R44) frozen
validation, and R43 does not depend on R41.

Variants (card 10 packaging, index item 8)
------------------------------------------
- **v1 ``isotropic_direct``**: the single-step direct decay of the
  high-mass parent through card 10's ablation machinery (the
  byte-unchanged Round 2 kernel's own vertex at a catalog handoff root,
  card 4's ``draw_decay`` over the direct pool otherwise; the line ends
  after the single step in both cases: no cascade chain, no alignment,
  no form factor).
- **v2 ``isotropic_sequential``**: card 4's sequential cascade at
  ``kappa_B = kappa_M = 0`` (the bit-stable isotropic branches of cards
  6-7) with the card-8 form factor at the nominal
  ``b_dec = 1.0 GeV^-2``.
- **v3 ``aligned_sequential``**: card 4's sequential cascade at the
  pre-registered nominal point ``kappa_B = kappa_M = 1.0`` with
  ``b_dec = 1.0 GeV^-2``; the only difference from v2 is the attached
  weights and the dispatched cos(theta).

Common random numbers (index item 8; plan sections 5.2 line 275 and 5.3
line 288)
------------------------------------------------------------------------
800,000 exchange events (T2, 8 replicas x 100,000, plan section 5.1
line 254).  The exchange layer (topology, exchange kinematics, spectral
state) is generated ONCE per event -- the M8-frozen builder source
layer (the mass sampler and the exact two-body scattering on the legacy
per-event generator), the card-6 routing at the frozen source mass, and
the catalog route's record draw -- and shared across all three variants
event by event (identical exchange events).  The per-variant decay
streams are card 10's child ``SeedSequence`` spawns keyed on the
variant id (0, 1, 2) on the closed decay_channel/decay_kinematics
labels, so the decay uniforms of one variant never leak into another.

Support boundary (plan sections 5.2 line 275 and 5.3 line 288)
--------------------------------------------------------------
v1's single-step support differs from v2/v3's multi-vertex support
(card 10's :data:`~na49_mixing.cascade_ablation.SUPPORT_BOUNDARY`
block, recorded verbatim in the pre-registration).  The exchange event
is identical across all three variants, so no regeneration is required
at the nominal point; the boundary is documented, not repaired.

Observables (plan section 7.3, lines 420-431)
---------------------------------------------
- the held-out longitudinal shapes: the pooled pion combination
  pi_Sigma = pi+ + pi- spectrum and the leading-baryon transfer
  combination B_transport = p - pbar spectrum (the antiproton term
  enters with the minus sign), on the aligned cells of the sealed R01
  tables (the published binning preserved);
- the held-out transverse shapes: the identified seagulls (the pooled
  pion seagull on the pi_Sigma combination at the pion mass and the
  pooled proton seagull on the B_transport combination at the proton
  mass), the per-xF <pT>_x and <pT>_y conventions of
  ``fit_observables.seagull_moment`` on both sides;
- the multiplicity moments per species (model-only diagnostics; the
  sealed R01 archive carries no multiplicity data vector);
- the ablation-specific statistics per variant: the longitudinal
  momentum retention (the x_F first moments of the pi_Sigma and
  B_transport combination spectra) and the cascade depth statistics
  (the per-variant depth histograms of the decay machinery).

Beyond-isotropic metric (the card's pre-registered bound, never
relaxed)
----------------------------------------------------------------
v3 improves on v2 on the pre-registered training observable set if the
per-observable deviance difference (v2 deviance minus v3 deviance) is
positive AND exceeds 2.0 times the pooled MC standard error of the
difference.  The deviance of an observable is the summed squared
normalized residual (d - m)^2 / sigma^2 over its finite bins, with
sigma^2 = v_data + se^2 (the sealed R01 published variance plus the
between-replica MC variance of that variant, floored at the engine's
DETERMINISTIC_SE_FLOOR).  The pooled MC standard error of the
difference is the between-replica standard error (std ddof=1 over the
8 paired replica blocks, divided by sqrt(8), floored at the engine's
deterministic SE floor) of the per-block deviance differences; the
exchange events are identical across the variants, so the per-block
differences are paired.  The per-observable table is recorded
regardless of the verdict; a failure seals with the named finding
feeding the Gate 4 beyond-isotropic clause (plan line 806).  The bound
is pre-registered, never relaxed.

Recording convention (the card-18 relitigation clause)
------------------------------------------------------
The finals follow the established R01 accumulate-dense-then-pick
convention (the corrected R44 driver's ``_worker_context`` /
``_replica_record`` pattern): every event weight accumulates, per
variant and per species, into the species' dense pT-major evaluation
lattice (``n_x * n_p`` cells at the species' dense offset), and the
published rows are picked from the dense totals at record assembly with
``pick = bin_index_pt * n_x + bin_index_xf``.  The published R01 tables
are non-rectangular subsets of their lattices (holes; rows not
pT-major), so a dense cell address and a published row address are
different addresses: the pick map keeps the recorded ``pooled_sums``
aligned 1:1 with the published rows the evaluation consumes, and a hole
cell's weight stays unpicked and is dropped, never bled into a
neighboring published row or species.  The pick map and the hole
dropping are variant-invariant, so the three-variant comparison stays
common-random-number fair.  Two structural refusals guard the pick
map: a published bin_index pair mapping outside the dense lattice, and
two published rows on one dense lattice cell (the pick map must stay
one-to-one).

The superseded card-14 landing recorded the finals at the published-row
flat index (``species_offset(species) + ip * n_x + ix`` with the
published-row cumulative offsets), which cell-transposed the spectra on
the non-rectangular published geometry; the sealed
``r43_cascade_ablation_2026-09-22`` archive was generated and evaluated
under that convention.  It stays intact, its seal verifies, and it is
superseded by this run.

No fitting: the ablation is a comparison (plan section 24);
``no_fitting`` is ``true`` and ``optimizer_invoked`` is ``false``.

Obligation 4 carry (model uncertainty): the sealed R32 cross-variant
spread (bin spread p95 4.028426 at 4,800,000 events; feed-down spread
max 31.1773 sigma; 32 persisting residuals) is recorded in the
pre-registration and in the gate evidence; the audits closed with card
2 and card 3 PASS and card 1 RECOMMENDATION (its rework card 1r landed
before this run with no new spread diagnostic), so the original R32
spread stands as the final form.

Conservation: every p+p event conserves total B = 2 and Q = 2 exactly
and every source its own (B, Q, S) = (1, 1, 0), in every variant; zero
depth-guard activations (plan section 10.6 line 757); the event energy
sum closes within 1e-9 GeV.  A violation raises and stops the run.

Determinism: a per-replica re-run from a clean interpreter reproduces
the record identity digests; the evaluation re-run from a clean
interpreter over the sealed artifacts reproduces the report hash
(ablation_report.json and gate4_r43.json carry no wall-clock fields:
their bytes are a pure function of the sealed artifacts, the
pre-registered seeds, and the committed code).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the frozen substrate digests and the sealed
  R30-R32 seals first.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, 1 replica x 2,000 events at the
  pre-registered smoke seeds (all three variants), and the smoke
  evaluation against a synthetic data vector, into an explicit
  ``--run-dir`` (the default production run dir is refused).  Two
  identical smoke invocations produce byte-identical artifacts.
  ``--gate`` / ``--seal`` / ``--clean-validate`` refuse smoke artifacts
  (the R31/8c convention).
- ``--generate [--replica R]``: generate the requested replica records
  (any subset; the default is every replica), one record file per
  replica, then stop.  Requires the pre-registration.  A completed
  record that matches the pre-registered identity is reused unchanged,
  so the declared tier can be completed replica by replica (the
  pre-registered gateway fallback runs one job per replica).
- ``--evaluate``: the per-variant observable tables, the exchange
  bit-identity verification at run scale, the stream-pin verification,
  the beyond-isotropic deviance table and verdict, and
  ``ablation_report.json``, then stop.  Requires all 8 production
  records.
- ``--gate``: write ``gate4_r43.json`` (evidence only), then stop.
  Requires the ablation report.
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
needed for the smoke evaluation's synthetic data); the production modes
read the committed catalog and continuum config, the sealed R01
archive, and the SPS tables (the published data the held-out shapes are
defined against).
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
        f"staged_r43_cascade_ablation: {message}",
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
    ABLATION_VARIANTS,
    SUPPORT_BOUNDARY,
    _attach_form_factor,
    _decay_isotropic_direct,
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

#: The superseded R43 driver digest (the card-14 landing; the value of
#: the card-14 contract register's ``DRIVER_SHA256`` constant at the
#: card-18 relitigation landing, sha256 prefix ``4c6bc0f3``).  The sealed
#: ``r43_cascade_ablation_2026-09-22`` archive was generated and
#: evaluated under that driver's published-row flat index; it stays
#: intact, its seal verifies, and it is superseded by this run.
SUPERSEDED_DRIVER_SHA256 = (
    "4c6bc0f32430577ff80e607e19f2512bf2ed1c40701f1f115ff82a88c975bf3a"
)
#: The superseded sealed R43 archive (read-only; its seal verifies and
#: is recorded in the provenance).
SUPERSEDED_RUN_DIR = "dev/00runs/r43_cascade_ablation_2026-09-22"


# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260920-r4-run-r43-cascade-
# ablation; the variant register, statistics, seeds, metric, observables,
# and acceptance blocks are verbatim from the card and the round 4 index
# items 3, 7, 8, and 9).
# ---------------------------------------------------------------------------

RUN_ID = "R43"
CARD = "TASK-20260920-r4-run-r43-cascade-ablation"
RUN_NAME = "r43_cascade_ablation_2026-09-23"
DEFAULT_RUN_DIR = "dev/00runs/r43_cascade_ablation_2026-09-23"
GATE_RECORD_NAME = "gate4_r43.json"
REPORT_NAME = "ablation_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
RECORD_FMT = "replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r43_cascade_ablation.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
ROUTING_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
KERNEL_REL = "dev/mixing2006/na49_mixing/resonance_decay.py"
R32_DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"
R40_DRIVER_REL = "helpers/benchmarking/staged_r40_alignment_scan.py"
R41_DRIVER_REL = "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py"
R42_DRIVER_REL = "helpers/benchmarking/staged_r42_recoil_decay_separation.py"

#: The card chain this run sits on: the Gate 3 closure audits (cards 1-3
#: and the rework card 1r), the cascade implementation cards (4-10, the
#: variant machinery), and the sealed R40 scan (card 11), R41 pilot fit
#: (card 12), and R42 profile (card 13) -- R43's index dependencies
#: exclude R41 and R42: their archives are not inputs of this run.
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
)

#: The frozen substrate digests (byte-unchanged inputs; verified before
#: any step).  The cascade surfaces are cards 4-10 (landed this round,
#: consumed read-only); the Round 2 kernel, the builder blend path, the
#: card-6 routing, the engine, the catalog, the continuum config, and
#: the R32/R40/R41/R42 drivers close the list.  The R41 and R42 driver
#: digests are the corrected card-16/card-17 relitigation landings (the
#: superseded card-12/card-13 digests moved with the sibling-pin
#: re-point of the card-18 relitigation; neither run output is consumed
#: by R43's event generation).
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
}

#: The sealed archives consumed read-only (their seals verify).  The R32
#: spread is the model-uncertainty carry of obligation 4; the sealed R01
#: archive is the published-data surface of the held-out shapes.  R43's
#: index dependencies exclude R41 and R42: their sealed archives are not
#: inputs of this run.
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
)
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"

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

#: The pre-registered variants of the ablation (card 10 packaging, index
#: item 8; the register order fixes the variant ids 0, 1, 2).
VARIANTS_NOTE = (
    "the card-10 packaging: v1 isotropic_direct (the single-step direct "
    "decay through the Round 2 kernel's own vertex at a catalog handoff "
    "root and card 4's draw_decay over the direct pool otherwise; the "
    "line ends after the single step -- no cascade chain, no alignment, "
    "no form factor), v2 isotropic_sequential (card 4's cascade at "
    "kappa_B = kappa_M = 0 with the card-8 form factor at the nominal "
    "b_dec = 1.0 GeV^-2), v3 aligned_sequential (card 4's cascade at "
    "the pre-registered nominal point kappa_B = kappa_M = 1.0 with "
    "b_dec = 1.0 GeV^-2, the R40 starting value, not a fit parameter; "
    "the only difference from v2 is the attached weights and the "
    "dispatched cos(theta)).  The nominal point is the pre-registered "
    "starting value (round 4 index item 3), not a fit parameter; R43 "
    "does not depend on R41 (the fitted-parameter comparison is card "
    "15's R44 frozen validation)"
)


def ablation_policies_payload() -> dict:
    """The pre-registered per-variant policies (the card-10 register)."""
    out: dict[str, Any] = {}
    for name in ABLATION_VARIANTS:
        policy = ABLATION_POLICIES[name]
        out[name] = {
            "variant_id": ABLATION_VARIANTS.index(name),
            "kappa_b": float(policy.kappa_b),
            "kappa_m": float(policy.kappa_m),
            "b_dec_gev2": float(policy.b_dec),
        }
    out["note"] = (
        "v1's b_dec records the nominal point; the v1 single-step path "
        "never applies the form factor (card 10's ABLATION_POLICIES)"
    )
    return out


#: The obligation 4 model-uncertainty carry (the sealed R32 cross-variant
#: spread), verbatim from the card.
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}
OBLIGATION4_FORM = (
    "the sealed R32 cross-variant spread stands as the model-uncertainty "
    "term of the comparison (the audits closed with card 2 and card 3 "
    "PASS and card 1 RECOMMENDATION, its rework card 1r landed before "
    "this run with no new spread diagnostic, so the original R32 spread "
    "is the final form); the beyond-isotropic verdict is read against "
    "the recorded R32 spread and the bound is never relaxed"
)

#: The beyond-isotropic metric (the card's pre-registered bound; the
#: plan gives no number, this bound is the pre-registered minimal
#: choice).  Never relaxed.
BEYOND_ISOTROPIC_FACTOR = 2.0
BEYOND_ISOTROPIC_NOTE = (
    "v3 improves on v2 on the pre-registered training observable set if "
    "the per-observable deviance difference (v2 deviance minus v3 "
    "deviance) is positive and exceeds 2.0 times the pooled MC standard "
    "error of the difference.  The deviance of an observable is the "
    "summed squared normalized residual (d - m)^2 / sigma^2 over its "
    "finite bins, with sigma^2 = v_data + se^2 (the sealed R01 "
    "published variance plus the between-replica MC variance of that "
    "variant, floored at the engine's DETERMINISTIC_SE_FLOOR).  The "
    "pooled MC standard error of the difference is the between-replica "
    "standard error (std ddof=1 over the 8 paired replica blocks, "
    "divided by sqrt(8), floored at the engine's deterministic SE "
    "floor) of the per-block deviance differences; the exchange events "
    "are identical across the variants, so the per-block differences "
    "are paired.  The per-observable table is recorded regardless of "
    "the verdict; a failure seals with the named finding feeding the "
    "Gate 4 beyond-isotropic clause (plan section 10, line 806).  The "
    "bound is pre-registered, never relaxed"
)

#: The robust-statistics cross-reference of the named-residual register
#: (the R32 acceptance-rescope precedent): the between-replica SE
#: convention floored at the engine's deterministic SE floor; the
#: pre-registered 2.0 factor is never relaxed; no data selection.
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)
SE_CONVENTION = (
    "the between-replica standard error (std ddof=1 over the 8 T2 "
    "replica blocks, divided by sqrt(8)), floored at the engine's "
    f"DETERMINISTIC_SE_FLOOR = {SE_FLOOR!r}; the pre-registered "
    f"{BEYOND_ISOTROPIC_FACTOR!r}-factor bound is never relaxed (the "
    "acceptance precedent of TASK-20260918-r3-acceptance-rescope-"
    "card8a-8b)"
)

EVENT_CLOSURE_TOL = 1e-9
DEGENERATE_WINDOW_TOL = 1e-12

#: Generation chunking (the R01/R23/R32 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 8

#: The stage label of the R43 exchange topology.
STAGE_LABEL = "r43_cascade_ablation"

#: The clean-environment validation's pre-registered probe replica.
PROBE_REPLICA = 0

GATEWAY_FALLBACK_RULE = (
    "local generation is primary (the measured local envelope in "
    "R41/R42: 100,000 p+p events ~ 75-78 s at 8 fork workers; the "
    "three-variant decay multiplies the per-event cost by ~3); if any "
    "replica exceeds 30 min wall, the remaining replicas run on the "
    "cluster gateway (host cl, workers c0801-c0809 x 32 task slots, "
    "one job per replica, --tasks 1, single math thread "
    "(NA49_CLUSTER_GW_TASKS forces thread counts to 1), no local "
    "process pool inside the chunk command, submission paced by the "
    "capacity check; chunk outputs are appended only to the new run "
    "dir)"
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
    "replica r < 8 (the block 20301020-20301027, shared with R41's and "
    "R42's blocks by the card's pre-registration); the exchange and "
    "decay-layer spaces are disjoint from every previously used space "
    "because the event-index partition 600_000_000 + i differs from "
    "R41's 400_000_000 + i, from R42's 500_000_000 + i, from the R40 "
    "partition 300_000_000 + s*10_000_000 + p*1_000_000 + i, and from "
    "the audit block 20301040 + k"
)
EXCHANGE_SEED_RULE = (
    "the exchange layer consumes the builder's per-event derivation at "
    "the pre-registered triple (master_seed 20260920, replica r, event "
    "index 600_000_000 + i): the legacy per-event generator "
    "numpy.random.SeedSequence(seed, spawn_key=(event_id,)) for the M8 "
    "mass sampler and the exact two-body scattering, and "
    "derive_substreams(20260920, r, 600_000_000 + i) for the card-6 "
    "routing and the catalog record draw (the record draw on the "
    "one-step spawned child of the decay_channel labeled id, the "
    "R21/R22/R23 mixture-layer convention; the spawned decay_kinematics "
    "child consumes nothing in R43 -- the decay draws are the "
    "per-variant spawns' on the same closed labels)"
)
VARIANT_STREAM_RULE = (
    "the per-variant decay streams are card 10's child SeedSequence "
    "spawns keyed on the variant id: variant_stream(20260920, "
    "600_000_000 + i, v) with v in {0, 1, 2} (the register order of "
    "ABLATION_VARIANTS), the one-step form SeedSequence(master_seed, "
    "spawn_key=(event_index, variant_id)).spawn(2) on the closed "
    "decay_channel/decay_kinematics labels; the streams are disjoint "
    "across the variant ids (the decay uniforms of one variant never "
    "leak into another) and deterministic on the pinned key.  The "
    "variant-spawn tree is a separate SeedSequence tree from the "
    "exchange derivation (card 10's documented contract: no decay "
    "uniform of the exchange derivation leaks into a variant stream or "
    "the other way around)"
)
EVENT_INDEX_RULE = "600_000_000 + i, i < 100,000"
EVENT_INDEX_BASE = 600_000_000
RAW_REPLICA_SEEDS = tuple(SEED_BASE + SEED_OFFSET + SEED_STRIDE + r
                          for r in range(T2_REPLICAS))

#: The hermetic smoke mode (driver validation; never sealed): 1 replica
#: x 2,000 events at the pre-registered smoke seeds.  The raw smoke seed
#: sits in the +9000 block (20309920, disjoint from every production
#: slot 0..7, from the R41 +8000, the R40 +7000, and the R32 +5000
#: smoke blocks); the smoke event-index partition is the +90M block
#: 690_000_000 + i with i < 2,000, disjoint from the R43 production
#: partition (600_000_000 + i) and from every previously used block, so
#: the smoke exchange and variant streams never touch a production
#: space.
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 9_000
SMOKE_RAW_SEED = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_KERNEL_REPLICA_SLOT = SMOKE_SEED_STRIDE
SMOKE_EVENT_INDEX_BASE = 690_000_000
SMOKE_EVENT_INDEX_RULE = "690_000_000 + i, i < 2,000"
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260920 + 40000 + 9000 = 20309920 (the +9000 "
    "smoke block, one replica; the R41 smoke block is +8000, the R40 "
    "smoke block +7000, and the R32 smoke block +5000); the builder's "
    "exchange derivation runs at derive_substreams(20260920, 9000, "
    "690_000_000 + i) and the per-variant decay streams at "
    "variant_stream(20260920, 690_000_000 + i, v) on the +90M smoke "
    "event-index block, disjoint from the R43 production partition "
    "(600_000_000 + i, i < 100,000), from R42's 500_000_000 block, from "
    "R41's 400_000_000 block, from the R40 partition, and from every "
    "previously used block"
)
SMOKE_NOTE = (
    "the smoke exercises pre-registration, generation, the three-variant "
    "decay, and the evaluation end to end, hermetically (the smoke data "
    "is the synthetic placeholder of the report machinery, no SPS "
    "table, no sealed archive); two identical smoke invocations produce "
    "byte-identical artifacts; smoke artifacts are refused by --gate / "
    "--seal / --clean-validate (the R31/8c convention)"
)


def raw_replica_seed(replica: int) -> int:
    r = int(replica)
    if not 0 <= r < T2_REPLICAS:
        _refuse(f"replica index {r} outside the T2 block 0..{T2_REPLICAS - 1}")
    return RAW_REPLICA_SEEDS[r]


def event_index(i: int) -> int:
    return EVENT_INDEX_BASE + int(i)


def smoke_event_index(i: int) -> int:
    return SMOKE_EVENT_INDEX_BASE + int(i)


def smoke_raw_seed() -> int:
    return SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE


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

#: The species offsets in the flat global bin space (the table order of
#: SPECIES_FLAT_ORDER, the R01 pT-major flat layout per species).
SPECIES_FLAT_ORDER = ("pi_minus", "pi_plus", "k_plus", "k_minus",
                      "proton", "antiproton")
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


#: The dense evaluation-lattice layout of the R01 accumulate-dense-then-
#: pick recording convention (the corrected convention of the card-18
#: relitigation, the R44 driver's ``_worker_context`` pattern): the dense
#: offsets and sizes of the full evaluation lattices and the published-
#: row pick maps, built from the tables' ``bin_index`` arrays.
dense_offsets_by_species: dict[str, int] = {}
dense_sizes_by_species: dict[str, int] = {}
pick_by_species: dict[str, np.ndarray] = {}
_DENSE_TOTAL: int = 0


def _init_dense_layout(tables: list) -> int:
    """The dense pT-major lattice layout per species (the R01
    accumulate-dense-then-pick convention): the dense offsets and sizes
    derived from the full evaluation grid, and the published-row pick
    maps built from the tables' ``bin_index`` arrays,
    ``pick = bin_index_pt * n_x + bin_index_xf``, with the two
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
        f"r43|{int(event_id)}|{_hex(float(t))}|"
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
    event and shared across all three variants; no per-variant draw is
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
    """One p+p exchange event of the R43 configuration (index item 7):
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
    and shared across the three variants; no per-variant draw is
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
            "r43: the blend routing expects exactly one activated "
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
                    "r43 exchange: the open set at source quantum numbers "
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
class _VariantRun:
    """One variant's decay of one event: the finals capture and the
    conservation tallies.

    ``flat``/``weights`` hold the event's sparse (dense_lattice_address,
    Jacobian weight) contributions of every final particle of both
    sources (the R01 binning convention: ``x_F = 2 p_z / sqrt(s)``, the
    pT-major DENSE lattice layout, the same in-bounds drop as the R42
    latent capture).  The published rows are picked from the dense
    totals at record assembly (the accumulate-dense-then-pick
    convention), so a hole cell's weight stays unpicked and is dropped.
    The energy and (B, Q, S) tallies feed the pre-registered
    per-event and per-source conservation checks; the multiplicity
    tallies feed the multiplicity moments; the depth histogram records
    the cascade depth statistic of the variant's decay machinery per
    decay; the rolling exchange digest folds the shared exchange-layer
    digest per event (the bit-identity evidence at run scale).
    """

    __slots__ = (
        "variant", "flat", "weights", "bqs", "energy_sum",
        "worst_closure", "n_finals", "mult_sums", "mult_sums2",
        "mult_with", "depth_hist", "mode_census", "exchange_rolling",
        "event_hash", "payload",
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
        self.depth_hist: dict[int, int] = {}
        self.mode_census: dict[str, int] = {}
        self.exchange_rolling = hashlib.sha256()
        self.event_hash = ""
        self.payload: list[str] = []

    def add_final(self, pdg: int, p4: tuple[float, float, float, float]) -> None:
        """One final particle: the (B, Q, S) tally, the energy tally,
        the multiplicity tally, the canonical payload line, and the
        paper-window bin row (the R01 binning convention; the same grid
        edges, the same ``sps_core.grid_indices`` binning, the pT-major
        DENSE lattice layout, the same in-bounds drop).  The recorded
        cell is the species' dense lattice address -- the dense offset
        plus ``ip * n_x + ix`` -- and the published rows are picked from
        the dense totals at record assembly (the R01
        ``_bin_track_sums`` accumulate-dense-then-pick convention), so a
        hole cell's weight stays unpicked and is dropped."""
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
            int(context["dense_offsets_by_species"][species])
            + ip * n_x + ix)
        self.weights.append(e / pt)

    def event_payload_digest(self, event_id: int) -> str:
        """The canonical per-event per-variant hash: the variant id, the
        event id, the mode census, the final count, and every final's
        pdg with exact hex four-momentum components (the rolling-digest
        convention of the R42 record identity)."""
        digest = hashlib.sha256()
        digest.update(
            f"r43|{self.variant}|{int(event_id)}|"
            f"{self.mode_census.get('single_step', 0)},"
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
) -> None:
    """The event assembly of one terminal established daughter state
    (card 9's caller-assembly clause, the R23/R32 ``_emit`` convention):
    a stable record contributes itself as a final; an unstable catalog
    record decays through the Round 2 kernel's own ``decay_resonance``
    on the same closed per-variant streams with the card-9 binding
    pattern (the kernel's decay logic, tables, and conservation checks
    run byte-unchanged).  The finals accumulate into the capture with
    their binning rows."""
    fid = str(state.family_id)
    if not fid.startswith(CATALOG_FAMILY_PREFIX):
        raise ValueError(
            "r43 assembly: a continuum daughter state reached the event "
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


def _decay_v1_captured(
    catalog: HadronCatalog,
    exchange_event: dict,
    rng_channel,
    rng_kinematics,
    capture: _VariantRun,
) -> str:
    """The v1 ``isotropic_direct`` variant with the event assembly: card
    10's ``_decay_isotropic_direct`` machinery byte for byte (the
    Round 2 kernel's own vertex at a catalog handoff root, card 4's
    ``draw_decay`` over the direct pool otherwise; the line ends after
    the single step -- no cascade chain, no alignment, no form factor),
    then the caller's event assembly of every terminal daughter (the
    card-9 convention: stable records are finals, unstable ones decay
    through the byte-unchanged kernel on the same per-variant streams).
    Returns the decay mode."""
    daughters = _decay_isotropic_direct(
        exchange_event, rng_channel, rng_kinematics)
    for daughter in daughters:
        _emit_assembly_state(catalog, capture, daughter,
                             rng_channel, rng_kinematics)
    capture.depth_hist[1] = capture.depth_hist.get(1, 0) + 1
    return "single_step"


def _walk_sequential_captured(
    catalog: HadronCatalog,
    fams: list,
    exchange_event: dict,
    policy,
    rng_channel,
    rng_kinematics,
    capture: _VariantRun,
) -> str:
    """One v2/v3 sequential decay with the event assembly: card 9's
    routing order with card 10's attached per-vertex step (the R40 walk
    shape).  At every line state, in order: the hard depth guard (a
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
                f"r43: the line state of family {current.family_id!r} at "
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
                f"r43: the line state of family {current.family_id!r} at "
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
        if cont:
            successor = next(
                daughter for daughter in (daughter0, daughter1)
                if not str(daughter.family_id).startswith(
                    CATALOG_FAMILY_PREFIX)
            )
            sibling = daughter1 if successor is daughter0 else daughter0
            _emit_assembly_state(
                catalog, capture, sibling, rng_channel, rng_kinematics)
            current = successor
            continue
        _emit_assembly_state(
            catalog, capture, daughter0, rng_channel, rng_kinematics)
        _emit_assembly_state(
            catalog, capture, daughter1, rng_channel, rng_kinematics)
        capture.depth_hist[n_vert] = capture.depth_hist.get(n_vert, 0) + 1
        return "direct_channel"


def decay_event_variants(
    catalog: HadronCatalog,
    fams: list,
    exchange: _EventCapture,
    smoke: bool = False,
) -> dict[str, dict]:
    """Decay one identical exchange event through all three variants
    from the same per-event seed (plan line 798).

    The exchange layer is generated once per event and shared; the
    per-variant decay streams are card 10's child spawns keyed on the
    variant id (0, 1, 2), disjoint across variants.  Every variant's
    finals are tallied into its own capture with the pre-registered
    per-source and per-event conservation checks (checked, never
    repaired; a violation stops the run), and each variant folds the
    shared exchange-layer digest into its own rolling exchange digest
    (the bit-identity evidence at run scale).  The card-9 stable-
    identity semantics apply before the variant loop: a drawn stable
    catalog record contributes the source itself as the single final at
    its own four-momentum, a deterministic draw-free variant-
    independent terminal product (no decay machinery runs on it).  The
    per-variant event hashes are keyed on the variant id.  Returns one
    entry per variant name with the capture parts.
    """
    event_id = int(exchange.event_id)
    out: dict[str, dict] = {}
    for variant_name in ABLATION_VARIANTS:
        variant_id = ABLATION_VARIANTS.index(variant_name)
        policy = ABLATION_POLICIES[variant_name]
        rng_channel, rng_kinematics = variant_stream(
            MASTER_SEED, event_id, variant_id)
        run = _VariantRun(variant_id)
        for source in exchange.sources:
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
                        "momentum"]))
                run.mode_census["stable_identity"] = (
                    run.mode_census.get("stable_identity", 0) + 1)
                run.depth_hist[0] = run.depth_hist.get(0, 0) + 1
            elif variant_name == "isotropic_direct":
                mode = _decay_v1_captured(
                    catalog, source["exchange"], rng_channel,
                    rng_kinematics, run)
                run.mode_census[mode] = run.mode_census.get(mode, 0) + 1
            else:
                mode = _walk_sequential_captured(
                    catalog, fams, source["exchange"], policy,
                    rng_channel, rng_kinematics, run)
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
                    f"r43: event {event_id} variant {variant_name!r} "
                    f"source {source['source_id']} ({source['role']}) "
                    f"finals sum to (B, Q, S) = {delta} but the source "
                    "carries (1, 1, 0); conservation is checked, never "
                    "repaired (the run stops)"
                )
        # -- the per-event conservation check --
        if run.bqs != (2, 2, 0):
            raise ConservationViolation(
                f"r43: event {event_id} variant {variant_name!r} finals "
                f"sum to (B, Q, S) = {run.bqs} but the p+p pair carries "
                "(2, 2, 0); conservation is checked, never repaired "
                "(the run stops)"
            )
        closure = abs(run.energy_sum - exchange.energy_expected)
        run.worst_closure = closure
        if not closure <= EVENT_CLOSURE_TOL:
            raise ConservationViolation(
                f"r43: event {event_id} variant {variant_name!r} final "
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
                f"r43: event {event_id} variant {variant_name!r} received "
                "an exchange-event mapping whose canonical digest differs "
                "from the shared exchange layer's (the plan line 798 "
                "identical-exchange-events premise is violated); "
                "conservation of the comparison is checked, never "
                "repaired (the run stops)"
            )
        run.exchange_rolling.update(variant_exchange.encode("utf-8"))
        run.exchange_rolling.update(b"\n")
        run.event_hash = run.event_payload_digest(event_id)
        out[variant_name] = {
            "variant": variant_id,
            "run": run,
        }
    # -- the shared exchange-layer digest folded once more as the
    #    per-event exchange record (the run-scale bit-identity evidence
    #    is the three per-variant rolling digests against the shared
    #    per-event hash) --
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
        name: {
            "hashes": [], "event_ids": [],
            "row_flat": [], "row_weight": [],
            "row_ptr": [0], "routing": [0, 0, 0], "depth_hist": {},
            "mode_census": {}, "mult": {
                k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                for k in SPECIES_KEYS},
            "worst_closure": 0.0, "n_finals": 0, "exchange_digest": "",
        }
        for name in ABLATION_VARIANTS
    }
    shared_exchange_rolling = hashlib.sha256()
    variant_exchange_rollings = {
        name: hashlib.sha256() for name in ABLATION_VARIANTS
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
        for name in ABLATION_VARIANTS:
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
            slot["worst_closure"] = max(
                slot["worst_closure"], float(run.worst_closure))
            slot["n_finals"] += int(run.n_finals)
        for name in ABLATION_VARIANTS:
            per_variant[name]["exchange_digest"] = \
                variant_exchange_rollings[name].hexdigest()
    # -- the chunk-level bit-identity consistency check (the honest
    #    per-event verification is the variant recomputation inside
    #    decay_event_variants; this compares the chunk-level folds of
    #    the same per-event hashes) --
    shared_chunk = shared_exchange_rolling.hexdigest()
    for name in ABLATION_VARIANTS:
        if per_variant[name]["exchange_digest"] != shared_chunk:
            raise RuntimeError(
                f"r43 chunk at event {event_ids[0]}: the per-variant "
                f"rolling exchange digest of {name!r} differs from the "
                "shared one (the plan line 798 identical-exchange-events "
                "premise is violated); the run stops"
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
    for name in ABLATION_VARIANTS:
        slot = {
            "hashes": [], "row_flat": [], "row_weight": [],
            "row_ptr": [0], "routing": [0, 0, 0], "depth_hist": {},
            "mode_census": {}, "mult": {
                k: {"sum_n": 0.0, "sum_n2": 0.0, "n_with": 0}
                for k in SPECIES_KEYS},
            "worst_closure": 0.0, "n_finals": 0,
        }
        per_variant[name] = slot
    shared_exchange_rolling = hashlib.sha256()
    exchange_rollings = {
        name: hashlib.sha256() for name in ABLATION_VARIANTS
    }
    n_events = 0
    for part in parts:
        shared_exchange_rolling.update(
            part["shared_exchange_digest"].encode("utf-8"))
        shared_exchange_rolling.update(b"\n")
        for name in ABLATION_VARIANTS:
            exchange_rollings[name].update(
                part["shared_exchange_digest"].encode("utf-8"))
            exchange_rollings[name].update(b"\n")
        n_events += int(part["n_events"])
        for name in ABLATION_VARIANTS:
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
            slot["worst_closure"] = max(
                slot["worst_closure"], float(src["worst_closure"]))
            slot["n_finals"] += int(src["n_finals"])
    shared_exchange_digest = shared_exchange_rolling.hexdigest()
    for name in ABLATION_VARIANTS:
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
T2_LABEL = (
    "T2: 8 replicas x 100,000 p+p exchange events (plan section 5.1 "
    "line 254, the upper T2 bound, the R23/R32 convention); 800,000 "
    "exchange events total, generated once and decayed through all "
    "three variants from the same per-event seed (common random "
    "numbers, plan line 798)"
)


def _replica_record(
    replica: int,
    seed: int,
    replica_slot: int,
    n_events: int,
    aggregated: dict,
) -> dict:
    """Assemble one per-replica record (deterministic: no wall-clock
    fields anywhere in the record).  Each variant's finals' dense
    lattice indices accumulate into that variant's dense lattice and the
    published rows are picked from the dense totals (the R01
    accumulate-dense-then-pick convention, mirrored from the corrected
    R44 driver's ``_replica_record``, applied per variant), so
    ``pooled_sums`` are published-row arrays aligned 1:1 with the
    per-published-row areas the evaluation divides by; a hole cell's
    weight stays unpicked and is dropped.  The pick map and the hole
    dropping are variant-invariant (the common-random-number fairness of
    the comparison)."""
    context = _WORKER_CONTEXT
    assert context is not None
    dense_offsets = context["dense_offsets_by_species"]
    dense_sizes = context["dense_sizes_by_species"]
    pick_maps = context["pick_by_species"]
    variants_payload: dict[str, dict] = {}
    pooled_species: dict[str, dict] = {}
    for name in ABLATION_VARIANTS:
        slot = aggregated["per_variant"][name]
        dense_pooled = np.zeros(int(context["dense_total"]), dtype=float)
        row_flat = np.asarray(slot["row_flat"], dtype=np.int64)
        row_weight = np.asarray(slot["row_weight"], dtype=float)
        if row_flat.size:
            np.add.at(dense_pooled, row_flat, row_weight)
        # the two-step dense-slice-then-pick conversion: the species'
        # dense slice, then the published-row pick (hole cells dropped)
        pooled_species[name] = {}
        for key in SPECIES_FLAT_ORDER:
            dense_slice = dense_pooled[
                int(dense_offsets[key]):
                int(dense_offsets[key]) + int(dense_sizes[key])]
            pooled_species[name][key] = [
                float(v) for v in dense_slice[pick_maps[key]]]
        variants_payload[name] = {
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
                    "card-10 variant machinery)"
                ),
            },
            "depth_histogram": {str(k): int(v) for k, v in sorted(
                slot["depth_hist"].items())},
            "mode_census": dict(sorted(slot["mode_census"].items())),
            "multiplicity": slot["mult"],
            "conservation": {
                "events_checked": int(n_events),
                "event_b_q_s_conserved": True,
                "worst_energy_closure_gev": float(slot["worst_closure"]),
                "tolerance": EVENT_CLOSURE_TOL,
            },
            "pooled_sums": pooled_species[name],
            "n_finals": int(slot["n_finals"]),
        }
    record = {
        "schema": "na49_mixing.staged_r43_cascade_ablation.replica/1",
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
            "tier": T2_LABEL,
            "events_per_replica": int(T2_EVENTS_PER_REPLICA),
            "variants": list(ABLATION_VARIANTS),
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
                "across the three variants; the rolling digest runs over "
                "the ordered per-event canonical exchange hashes and the "
                "three per-variant rolling digests are recomputed from "
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
                    f"[r43] {progress_label}: chunk {completed}/"
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
    event decayed through all three variants.  A completed record that
    matches the pre-registered identity is reused unchanged.  Returns
    the record names written or reused."""
    pre = _load_pre_registration(run_dir)
    _assert_pre_registration(pre, run_dir)
    pre_digest = _pre_registration_digest_of(pre)
    tables = _load_production_tables()
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
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
            if _record_identity_ok(existing, pre, replica, seed,
                                   T2_EVENTS_PER_REPLICA):
                print(f"[r43] {name}: completed record reused", flush=True)
                written.append(name)
                continue
            _refuse(
                f"{path}: a completed record exists but does not match "
                "the pre-registered identity; a drifted record is never "
                "silently overwritten (delete the run dir and re-run)"
            )
        tasks = _chunk_tasks(replica, seed, replica,
                             T2_EVENTS_PER_REPLICA,
                             EVENT_INDEX_BASE, smoke=False)
        parts = list(_map_chunks(tasks, workers, f"replica {replica}"))
        aggregated = _aggregate_chunks(parts)
        aggregated["pre_registration_sha256"] = pre_digest
        if aggregated["n_events"] != int(T2_EVENTS_PER_REPLICA):
            _refuse(
                f"replica {replica}: generated "
                f"{aggregated['n_events']} events, expected "
                f"{int(T2_EVENTS_PER_REPLICA)}"
            )
        record = _replica_record(
            replica, seed, replica, int(T2_EVENTS_PER_REPLICA),
            aggregated)
        _write_json(path, record)
        written.append(name)
        v2_finals = int(record["variants"]["isotropic_sequential"]["n_finals"])
        print(f"[r43] {name}: written ({v2_finals} v2 finals)",
              flush=True)
    return written


def run_smoke(run_dir: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """The hermetic end-to-end driver validation (never sealed): a smoke
    pre-registration, 1 replica x 2,000 events at the smoke seeds (all
    three variants), and the smoke evaluation against the synthetic
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
    # -- the smoke pre-registration (the smoke variant of the contract) --
    pre = build_pre_registration(smoke=True)
    pre["mode"] = "smoke"
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
    "note": SMOKE_NOTE,
}


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
    pre = build_pre_registration(smoke=smoke)
    pre["mode"] = "production"
    pre["substrate"] = substrate
    pre["sealed_run_dirs_status"] = sealed
    # the recording geometry of the pre-registered pick convention (the
    # per-species published-row vs lattice-cell counts of the run's
    # tables; the card-18 relitigation clause) and the superseded
    # archive's seal state (read-only; the seal verifies).
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
        "schema": "na49_mixing.staged_r43_cascade_ablation.provenance/1",
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
        "superseded_r43_archive": {
            "run_dir": SUPERSEDED_RUN_DIR,
            "state": (
                "intact; the seal verifies (verify_sealed_dir); the "
                "archive was generated and evaluated under the "
                "superseded card-14 driver's published-row flat index "
                "and is superseded by this run"
            ),
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "seal": superseded_status,
        },
        "recording_convention": {
            "dense_accumulation": True,
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "variant_invariant": True,
            "corrected_driver_sha256": DRIVER_SHA256,
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "geometry": pre["recording_convention"].get("geometry"),
        },
        "sealed_run_dirs": sealed,
        "nominal_point": (
            "v3 runs at the pre-registered nominal point "
            "(kappa_B, kappa_M, b_dec) = (1.0, 1.0, 1.0 GeV^-2), the "
            "R40 starting value (round 4 index item 3), not a fit "
            "parameter; the R41 fitted values are a boundary construct "
            "and are not consumed here"
        ),
        "predecessors": (
            "the sealed R40 scan archive and the R41/R42 archives are "
            "not inputs of this run (R43's index dependencies exclude "
            "R41 and R42); the R32 spread is the model-uncertainty "
            "carry of obligation 4"
        ),
        "git_state": _git_info(MONO_ROOT),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(f"[r43] pre-registration written into {run_dir}", flush=True)
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
                "card's block 20301020-20301027")
    if seeds.get("event_index_rule") != EVENT_INDEX_RULE:
        _refuse("the pre-registered event-index rule does not match the "
                "card (600_000_000 + i, i < 100,000)")
    if seeds.get("master_seed") != MASTER_SEED:
        _refuse("the pre-registered master seed does not match the card")
    if list(pre.get("variant_register") or ()) != list(ABLATION_VARIANTS):
        _refuse("the pre-registered variant register does not match "
                "card 10's ABLATION_VARIANTS")
    policies = pre.get("variant_policies", {})
    for name in ABLATION_VARIANTS:
        entry = policies.get(name) or {}
        expected = ABLATION_POLICIES[name]
        if (float(entry.get("kappa_b", float("nan"))) !=
                float(expected.kappa_b)):
            _refuse(f"the pre-registered policy of {name} drifted "
                    "(kappa_b)")
        if (float(entry.get("kappa_m", float("nan"))) !=
                float(expected.kappa_m)):
            _refuse(f"the pre-registered policy of {name} drifted "
                    "(kappa_m)")
        if (float(entry.get("b_dec_gev2", float("nan"))) !=
                float(expected.b_dec)):
            _refuse(f"the pre-registered policy of {name} drifted "
                    "(b_dec)")
    nominal = pre.get("nominal_point", {})
    if float(nominal.get("kappa_b", 0.0)) != 1.0 or \
            float(nominal.get("kappa_m", 0.0)) != 1.0 or \
            float(nominal.get("b_dec_gev2", 0.0)) != 1.0:
        _refuse("the pre-registered nominal point is not the card's "
                "(kappa_B, kappa_M, b_dec) = (1.0, 1.0, 1.0 GeV^-2)")
    metric = pre.get("beyond_isotropic_metric", {})
    if float(metric.get("factor", 0.0)) != BEYOND_ISOTROPIC_FACTOR:
        _refuse("the pre-registered beyond-isotropic factor is not the "
                "card's 2.0")
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
                "optimizer_invoked = false (the ablation is a "
                "comparison, plan section 24)")
    for key in ("collision", "recording_convention", "observables",
                "conservation", "determinism",
                "support_boundary", "acceptance"):
        if key not in pre:
            _refuse(f"the pre-registration is missing the {key} block")
    recording = pre.get("recording_convention", {})
    if recording.get("pick_map") != "bin_index_pt * n_x + bin_index_xf" \
            or recording.get("holes_dropped") is not True:
        _refuse("the pre-registered recording convention is not the "
                "card-18 relitigation clause (dense accumulation, the "
                "published-row pick map, holes dropped)")


# ---------------------------------------------------------------------------
# The pre-registration (written before any generation)
# ---------------------------------------------------------------------------
def build_pre_registration(smoke: bool = False) -> dict:
    """The pre-registered run contract, asserted against the card
    (TASK-20260920-r4-run-r43-cascade-ablation)."""
    return {
        "schema":
            "na49_mixing.staged_r43_cascade_ablation.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "plan_anchors": {
            "run": "plan section 10 run R43, line 790",
            "variants": "plan lines 792-796 (the three-variant comparison)",
            "common_random_numbers": "plan line 798 (identical exchange "
                                     "events and common random numbers)",
            "support_boundary": "plan sections 5.2 line 275, 5.3 line 288",
            "observables": "plan section 7.3, lines 420-431",
            "statistics_tier": "plan section 5.1 line 254",
            "gate_clause": "plan section 10, line 806 (Gate 4, beyond "
                           "isotropic sequential)",
            "no_fitting": "plan section 24 (the ablation is a comparison)",
            "conservation": "plan section 10.6 line 757 (the depth guard)",
            "cascade_spec": "plan section 10.2 line 678 (each high-mass "
                            "parent)",
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
                "variant through the card-10 machinery"
            ),
            "continuum_branch": (
                "the single activated continuum family (baryonic_np) "
                "realized at the frozen source four-momentum (no "
                "family-mass redraw), decayed per variant through card "
                "10's machinery at the variant's policy (v1 the single-"
                "step direct decay, v2/v3 the attached sequential "
                "cascade at the variant's (kappa_b, kappa_m, b_dec)); "
                "the production axis is card 5's initial_source_axis "
                "against the source's own beam direction (projectile "
                "+z, target -z; the round 4 index item 4 p+p "
                "pre-registration)"
            ),
            "assembly_note": (
                "the caller's event assembly is the card-9 convention "
                "for every variant (the R23/R32 _emit convention): the "
                "stable terminal daughters are finals and the unstable "
                "terminal daughters decay through the byte-unchanged "
                "Round 2 kernel on the same per-variant streams; the "
                "variant machinery is the walk above the terminal "
                "established states, not the assembly"
            ),
        },
        "variants": {
            "index_item": 8,
            "register": list(ABLATION_VARIANTS),
            "policies": ablation_policies_payload(),
            "note": VARIANTS_NOTE,
        },
        "variant_register": tuple(ABLATION_VARIANTS),
        "variant_policies": ablation_policies_payload(),
        "nominal_point": {
            "kappa_b": 1.0,
            "kappa_m": 1.0,
            "b_dec_gev2": 1.0,
            "note": (
                "the pre-registered nominal point of v3 (round 4 index "
                "item 3, the R40 starting value); not a fit parameter "
                "and not the R41 fitted values"
            ),
        },
        "common_random_numbers": {
            "index_item": 8,
            "note": (
                "800,000 exchange events (T2, 8 replicas x 100,000, plan "
                "section 5.1 line 254), each decayed through all three "
                "variants from the same per-event seed; the exchange "
                "layer (topology, exchange kinematics, spectral state) "
                "is generated once per event and shared across the "
                "variants (identical exchange events); the per-variant "
                "decay streams are child spawns keyed on the variant id "
                "(0, 1, 2), disjoint across variants"
            ),
            "plan_lines": [
                "plan section 5.2 line 275 (common random numbers for "
                "parameter comparisons whenever support does not change)",
                "plan section 5.3 line 288 (reweight only when the "
                "changed parameter alters a probability density but not "
                "kinematic support)",
                "plan line 798 (identical exchange events and common "
                "random numbers)",
            ],
        },
        "support_boundary": SUPPORT_BOUNDARY,
        "statistics": {
            "tier": T2_LABEL,
            "replicas": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "total_events": T2_REPLICAS * T2_EVENTS_PER_REPLICA,
            "decays": (
                "800,000 exchange events x 3 variants of decay (the "
                "three-variant decay multiplies the per-event cost by "
                "~3 over the R41/R42 single-variant envelope)"
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
                "the R43 master seed 20260920 is shared with R41/R42 "
                "per the card's pre-registration; the R43 spaces are "
                "disjoint from every previously used space because the "
                "event-index partition 600_000_000 + i differs from "
                "R41's 400_000_000 + i, from R42's 500_000_000 + i, "
                "from the R40 partition 300_000_000 + s*10_000_000 + "
                "p*1_000_000 + i, and from the audit block 20301040 + "
                "k; the raw block 20301020-20301027 is the R41/R42 "
                "block (shared by pre-registration) and is disjoint "
                "from the R23 raw seeds, from the R32 production and "
                "confirmation blocks, and from the audit block; the "
                "per-variant decay spawns are keyed on the variant id "
                "(0, 1, 2), disjoint across variants"
            ),
            "smoke": SMOKE_PAYLOAD,
        },
        "observables": {
            "training": (
                "the held-out longitudinal and transverse shapes of "
                "plan section 7.3 lines 420-431: the pooled pion "
                "combination spectrum pi_Sigma = pi+ + pi- and the "
                "leading-baryon transfer spectrum B_transport = p - "
                "pbar (the antiproton term enters with the minus sign) "
                "on the aligned species cells of the sealed R01 tables, "
                "the pooled pion seagull (the pi_Sigma combination at "
                "the pion mass) and the pooled proton seagull (the "
                "B_transport combination at the proton mass), the "
                "per-xF <pT>_x and <pT>_y conventions of "
                "fit_observables.seagull_moment on both sides, and the "
                "multiplicity moments per species"
            ),
            "deviance_set": (
                "the four data-anchored statistics (the pi_Sigma "
                "spectrum, the B_transport spectrum, the pooled pion "
                "seagull, the pooled proton seagull) carry the "
                "beyond-isotropic deviance comparison against the "
                "sealed R01 published data; the multiplicity moments "
                "are recorded per variant as model-only diagnostics "
                "(the sealed R01 archive carries no multiplicity data "
                "vector)"
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
                "locally loaded SPS tables at --evaluate time (the R11 "
                "parse_sealed_binned discipline)"
            ),
            "ablation_specific": (
                "the ablation-specific statistics per variant: the "
                "longitudinal momentum retention (the x_F first moments "
                "of the pi_Sigma and B_transport combination spectra, "
                "sum xF_mid m / sum m over the aligned cells, per "
                "replica and pooled with the between-replica SE) and "
                "the cascade depth statistics (the per-variant depth "
                "histograms of the decay machinery: 0 for a stable-"
                "identity source and for a catalog-handoff root, 1 for "
                "a v1 decay step, the number of cascade vertices for a "
                "v2/v3 line; the mean, the 95th percentile, and the max "
                "of the pooled histogram, R40's depth_stats convention)"
            ),
        },
        "beyond_isotropic_metric": {
            "factor": BEYOND_ISOTROPIC_FACTOR,
            "note": BEYOND_ISOTROPIC_NOTE,
            "bound": "pre-registered, never relaxed",
        },
        "obligation_4_carry": {
            "obligation": 4,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
            "audit_outcome": AUDIT_OUTCOME,
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "the ablation is a comparison, plan section 24: no "
            "parameter is fit, no optimizer runs, the variants run at "
            "their pre-registered policies (the nominal point for v3, "
            "the isotropic point for v1/v2)"
        ),
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
                "zero depth-guard activations over 800,000 events x 3 "
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
                    "the routing census, the depth histograms, and the "
                    "multiplicity tallies against the sealed record"
                ),
            },
            "report_note": (
                "ablation_report.json and gate4_r43.json carry no "
                "wall-clock fields: their bytes are a pure function of "
                "the sealed artifacts, the pre-registered seeds, and "
                "the committed code"
            ),
        },
        "recording_convention": {
            "convention": (
                "the R01 accumulate-dense-then-pick recording convention "
                "(the corrected convention of the card-18 relitigation; "
                "the corrected R44 driver's _worker_context / "
                "_replica_record pattern is the reference): every event "
                "weight accumulates, per variant and per species, into "
                "the species' dense pT-major evaluation lattice (n_x * "
                "n_p cells at the species' dense offset), and the "
                "published rows are picked from the dense totals at "
                "record assembly"
            ),
            "pick_map": "bin_index_pt * n_x + bin_index_xf",
            "holes_dropped": True,
            "variant_invariant": True,
            "holes_note": (
                "the published R01 tables are non-rectangular subsets of "
                "their lattices (holes; rows not pT-major): a hole "
                "cell's weight stays unpicked and is dropped, never bled "
                "into a neighboring published row or species; the "
                "structural refusals cover a published pair outside the "
                "lattice and two published rows on one lattice cell; "
                "the pick map and the hole dropping are variant-"
                "invariant (the common-random-number fairness of the "
                "three-variant comparison)"
            ),
            "superseded_driver_sha256": SUPERSEDED_DRIVER_SHA256,
            "superseded_run_dir": SUPERSEDED_RUN_DIR,
            "superseded_note": (
                "the superseded card-14 landing recorded the finals at "
                "the published-row flat index (species_offset(species) + "
                "ip * n_x + ix with the published-row cumulative "
                "offsets), which cell-transposed the spectra on the "
                "non-rectangular published geometry; the sealed "
                "r43_cascade_ablation_2026-09-22 archive was generated "
                "and evaluated under that convention and is superseded "
                "by this run (it stays intact; its seal verifies)"
            ),
            "corrected_driver_sha256": DRIVER_SHA256,
        },
        "sealed_run_dirs": list(SEALED_RUN_DIRS),
        "frozen_substrate": {
            key: value[1] for key, value in FROZEN_SUBSTRATE.items()
        },
        "gateway_fallback": GATEWAY_FALLBACK_RULE,
        "acceptance": {
            "precedent": (
                "the acceptance precedent of "
                "TASK-20260918-r3-acceptance-rescope-card8a-8b: an "
                "MC-driven check set with residuals seals with a named-"
                "residual register, not a raw zero-failure target"
            ),
            "i_exchange_bit_identity": (
                "the exchange layer (topology, exchange kinematics, "
                "spectral state) is bit-identical across all three "
                "variants, event by event (plan line 798; card 10 "
                "acceptance (i), verified at run scale by the three "
                "per-variant rolling exchange digests against the "
                "shared per-event hashes)"
            ),
            "ii_stream_pin": (
                "the per-variant decay streams reproduce the card-10 "
                "pinned stream contract at run scale (disjoint across "
                "the variant ids, deterministic on the pinned key); "
                "verified by the pinned-key probe replay and the "
                "clean-validate probe replica"
            ),
            "iii_conservation": (
                "800,000 events x 3 variants, exact B = 2 / Q = 2 and "
                "exact per-source (B, Q, S); zero violations; zero "
                "depth-guard activations"
            ),
            "iv_beyond_isotropic_metric": (
                "the per-observable deviance table (v1, v2, v3) is "
                "recorded; the metric verdict (v3 vs v2 at 2.0 x the "
                "pooled SE of the difference) is recorded as a named "
                "finding when it fails (the Gate 4 clause consumes it "
                "directly; the bound is pre-registered, never relaxed)"
            ),
            "v_no_nan": (
                "all recorded observables are finite (a structurally "
                "undefined statistic bin is dropped from the deviance "
                "and listed, the R11 convention; never a NaN)"
            ),
            "vi_seal_and_clean_validation": (
                "the seal is intact and the clean-environment re-run "
                "reproduces the ablation evaluation hashes"
            ),
            "vii_pins_and_freeze": (
                "the pin test register lands, the freeze re-pin is "
                "recorded, and the full nested suite is green"
            ),
            "named_residual_register": (
                "every failing verdict named with the observed value, a "
                "one-line diagnosis, and the robust-statistics cross-"
                "reference; the pre-registered bounds are never relaxed; "
                "no data selection"
            ),
            "register_triggers": [
                "a failing beyond-isotropic observable (the per-"
                "observable deviance difference non-positive or within "
                "the 2.0 x pooled SE band; the named finding feeds the "
                "Gate 4 beyond-isotropic clause directly)",
                "an observable whose deviance is undefined (every bin "
                "dropped by structure)",
                "a non-finite recorded observable",
            ],
        },
        "written_before_generation": True,
    }


# ---------------------------------------------------------------------------
# The fit data (the sealed R01 archive + the published binning)
# ---------------------------------------------------------------------------
R01_ARCHIVE = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R01_PREDICTIONS_NAME = "bin_predictions.json"
R01_COVARIANCE_NAME = "covariance.json"
R01_RESIDUALS_NAME = "residuals.json"
#: The metric's pair conventions (plan section 7.3): the pooled pion
#: pair is the pi_Sigma sum; the leading-baryon transfer pair is the
#: plan-431 DIFFERENCE p - pbar (the antiproton term enters with the
#: minus sign).  The sign multiplies the second species' values.
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
#: The training observable set order (the deviance table's row order).
TRAINING_OBSERVABLES = (
    "pi_sigma_spectrum",
    "b_transport_spectrum",
    "pi_sigma_seagull",
    "b_transport_seagull",
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


def _cell_area_from_table(table, i: int) -> float:
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    ix = int(table.bin_index_xf[i])
    ip = int(table.bin_index_pt[i])
    return (float(x_edges[ix + 1]) - float(x_edges[ix])) * (
        float(p_edges[ip + 1]) - float(p_edges[ip]))


def _combination_statistic(sealed: dict, tables_by_species: dict,
                           pair_key: str) -> dict:
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
        _cell_area_from_table(tables_by_species[species_a], e["bin_a"])
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
    }


def _combination_model_statistic(stat: dict, model_a: np.ndarray,
                                 model_b: np.ndarray) -> np.ndarray:
    sign = stat["pair"][2]
    return np.asarray([
        model_a[e["bin_a"]] + sign * model_b[e["bin_b"]]
        for e in stat["cells"]
    ], dtype=float)


class Statistic:
    """One metric observable statistic (d, V_data, and the model
    evaluator spec).  Kinds: 'spectrum' (the binned combination cells),
    'seagull' (the two <pT> conventions per xF bin of a combination)."""

    __slots__ = ("key", "kind", "d", "v_data", "areas", "combo", "seagull")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def _seagull_statistic(sealed: dict, tables_by_species: dict,
                       pair_key: str, spectrum: dict,
                       sqrt_s: float) -> Statistic:
    """The pooled-combination seagull: the per-xF <pT>_x and <pT>_y of
    the fit_observables conventions on both sides, the data side from
    the sealed combination d and the first-order delta-method variance
    over the aligned cells."""
    species_a, _species_b, _sign = COMBINATION_PAIRS[pair_key]
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
    groups: dict[float, list] = {}
    for value, (ix, ip), v in zip(
        d_cells, bin_indices, spectrum["v_data"],
    ):
        x_mid = 0.5 * (float(x_edges[ix]) + float(x_edges[ix + 1]))
        p_mid = 0.5 * (float(p_edges[ip]) + float(p_edges[ip + 1]))
        dp = float(p_edges[ip + 1]) - float(p_edges[ip])
        groups.setdefault(x_mid, []).append(
            (float(value), p_mid, dp, float(v)))
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
                moments.append(float("inf"))
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
    d_flat = np.asarray(moments, dtype=float)
    return Statistic(
        key=f"{pair_key}_seagull", kind="seagull",
        d=d_flat,
        v_data=np.asarray(v_data, dtype=float),
        areas=np.ones(len(d_flat), dtype=float),
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


def _build_statistics(sealed: dict, tables_by_species: dict,
                      sqrt_s: float) -> list:
    """The pre-registered training observable set (plan section 7.3,
    lines 420-431): the pi_Sigma and B_transport spectra on the aligned
    cells and the identified seagulls (the pooled pion and proton
    seagulls), in the training order."""
    stats = []
    combos = {}
    for pair_key in ("pi_sigma", "b_transport"):
        combos[pair_key] = _combination_statistic(
            sealed, tables_by_species, pair_key)
    for pair_key in ("pi_sigma", "b_transport"):
        spectrum = combos[pair_key]
        stats.append(Statistic(
            key=spectrum["key"], kind="spectrum",
            d=spectrum["d"], v_data=spectrum["v_data"],
            areas=spectrum["areas"], combo=spectrum, seagull=None))
    for pair_key in ("pi_sigma", "b_transport"):
        stats.append(_seagull_statistic(
            sealed, tables_by_species, pair_key, combos[pair_key],
            sqrt_s))
    return stats


# ---------------------------------------------------------------------------
# The model evaluation (the per-variant predictions of the statistic set)
# ---------------------------------------------------------------------------
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


def _seagull_model(stat: Statistic, species_values: dict) -> list:
    seagull = stat.seagull
    combo_values = _combination_model_statistic(
        stat.combo,
        species_values[stat.combo["pair"][0]],
        species_values[stat.combo["pair"][1]])
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


def _stat_vectors(stats: list, species_values: dict) -> list:
    vectors = []
    for stat in stats:
        if stat.kind == "spectrum":
            vectors.append(_combination_model_statistic(
                stat.combo,
                species_values[stat.combo["pair"][0]],
                species_values[stat.combo["pair"][1]]))
        else:
            vectors.append(_seagull_model(stat, species_values))
    return vectors


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
# The deviance algebra (the beyond-isotropic metric)
# ---------------------------------------------------------------------------
def _variant_sums(record: dict, variant_name: str) -> np.ndarray:
    """One variant's pooled bin sums of one record (the flat species
    layout of the published rows; the pick output of the accumulate-
    dense-then-pick recording convention, aligned 1:1 with the
    published rows the evaluation consumes)."""
    pooled = np.zeros(_flat_total(), dtype=float)
    for species in SPECIES_FLAT_ORDER:
        values = np.asarray(
            record["variants"][variant_name]["pooled_sums"][species],
            dtype=float)
        start = species_offset(species)
        pooled[start:start + values.size] = values
    return pooled


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
                        m: np.ndarray, se: np.ndarray,
                        dropped: list) -> float:
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


def pooled_difference_se(deltas: list) -> float:
    """The pooled MC standard error of the per-block deviance
    differences: the std ddof=1 over the 8 paired replica blocks,
    divided by sqrt(8), floored at the engine's deterministic SE floor
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


def beyond_isotropic_row(delta: float, se: float) -> dict:
    """One observable's beyond-isotropic row: the deviance difference
    (v2 minus v3), the pooled SE of the difference, the pre-registered
    threshold, and the pass verdict (positive AND beyond 2.0 x SE)."""
    delta = float(delta)
    se = float(se)
    positive = bool(delta > 0.0)
    beyond = bool(delta > BEYOND_ISOTROPIC_FACTOR * se)
    return {
        "delta_v2_minus_v3": delta,
        "pooled_se": se,
        "threshold": (
            f"{BEYOND_ISOTROPIC_FACTOR!r} x pooled SE = "
            f"{BEYOND_ISOTROPIC_FACTOR * se!r}"
        ),
        "positive": positive,
        "beyond_2x_se": beyond,
        "pass": bool(positive and beyond),
    }


# ---------------------------------------------------------------------------
# The depth statistics (the R40 depth_stats convention)
# ---------------------------------------------------------------------------
def depth_stats(hist: dict, n_decays: int) -> dict:
    """The pooled depth statistics of one histogram: the mean, the 95th
    percentile (the smallest histogram depth whose cumulative count
    reaches 0.95 x n_decays; no interpolation), the max, and the guard
    activations (zero by construction: a guard raises and stops the run
    before any record is written)."""
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
            "catalog-handoff root, 1 for a v1 decay step, the number of "
            "cascade vertices for a v2/v3 line); the 95th percentile is "
            "the smallest histogram depth whose cumulative count reaches "
            "0.95 x n_decays (no interpolation); the guard activations "
            "are zero by construction (a guard raises and stops the run)"
        ),
    }


def first_moment(cells: list, model) -> float | None:
    """The x_F first moment of one combination spectrum over its aligned
    cells: sum xF_mid m / sum m (the longitudinal momentum retention
    statistic).  None when the model vector carries no finite positive
    total (a structurally undefined retention)."""
    num = 0.0
    den = 0.0
    for cell, value in zip(cells, np.asarray(model, dtype=float)):
        if not math.isfinite(float(cell["xF_mid"])):
            continue
        if not math.isfinite(float(value)):
            continue
        num += float(cell["xF_mid"]) * float(value)
        den += float(value)
    if not (math.isfinite(num) and math.isfinite(den) and den > 0.0):
        return None
    return num / den


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
    for replica in range(T2_REPLICAS):
        path = Path(run_dir) / RECORD_FMT.format(replica)
        if not path.is_file():
            _refuse(f"the production record is missing: {path} "
                    "(run --generate first)")
        record = _read_json(path)
        if int(record.get("replica", -1)) != replica:
            _refuse(f"{path}: the record's replica field disagrees")
        blocks.append(record)
    return blocks


# ---------------------------------------------------------------------------
# The evaluation (the per-variant tables and the beyond-isotropic metric)
# ---------------------------------------------------------------------------
def run_evaluate_machinery(run_dir: Path, smoke: bool = False,
                           tables=None, fams=None) -> dict:
    """The ablation evaluation: the per-variant observable tables, the
    exchange bit-identity verification at run scale, the stream-pin
    verification, the beyond-isotropic deviance table and verdict, and
    the report payload (no wall-clock fields; the report bytes are a
    pure function of the sealed artifacts, the pre-registered seeds, and
    the committed code)."""
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
        for species in SPECIES_FLAT_ORDER:
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
        _init_species_layout(tables)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = _sqrt_s()
        sealed = _load_sealed_r01(tables)
    stats = _build_statistics(sealed, tables_by_species, sqrt_s)
    stat_keys = [stat.key for stat in stats]
    if stat_keys != list(TRAINING_OBSERVABLES):
        _refuse("the training observable set drifted from the "
                "pre-registered order")
    blocks = _load_blocks(run_dir, smoke)
    n_replicas = len(blocks)
    n_events_total = sum(int(b["n_events"]) for b in blocks)

    # -- the per-variant model vectors (per block and pooled) ---------
    per_variant_vectors: dict[str, list] = {name: [] for name
                                            in ABLATION_VARIANTS}
    per_block_retentions: dict[str, dict[str, list]] = {
        name: {"pi_sigma": [], "b_transport": []}
        for name in ABLATION_VARIANTS
    }
    for block in blocks:
        n_eff = float(block["n_events"])
        for name in ABLATION_VARIANTS:
            variant_sums = _variant_sums(block, name)
            values = _species_bin_values(
                variant_sums, tables_by_species, sqrt_s, n_eff)
            vectors = _stat_vectors(stats, values)
            per_variant_vectors[name].append(vectors)
            per_block_retentions[name]["pi_sigma"].append(first_moment(
                stats[0].combo["cells"], vectors[0]))
            per_block_retentions[name]["b_transport"].append(first_moment(
                stats[1].combo["cells"], vectors[1]))
    pooled_flat: dict[str, np.ndarray] = {
        name: np.zeros(_flat_total(), dtype=float)
        for name in ABLATION_VARIANTS
    }
    for block in blocks:
        for name in ABLATION_VARIANTS:
            pooled_flat[name] = pooled_flat[name] + _variant_sums(
                block, name)
    pooled_n_eff = float(n_events_total)
    pooled_values = {
        name: _species_bin_values(
            pooled_flat[name], tables_by_species, sqrt_s, pooled_n_eff)
        for name in ABLATION_VARIANTS
    }
    pooled_vectors = {
        name: _stat_vectors(stats, pooled_values[name])
        for name in ABLATION_VARIANTS
    }
    se_vectors: dict[str, list] = {}
    for name in ABLATION_VARIANTS:
        se_vectors[name] = [
            se_from(np.asarray([
                per_variant_vectors[name][k][i]
                for k in range(n_replicas)
            ], dtype=float))
            for i in range(len(stats))
        ]

    # -- the deviance table and the beyond-isotropic metric -----------
    dropped: list = []
    deviance_table: dict[str, dict] = {}
    per_block_deviances: dict[str, dict[str, list]] = {
        name: {} for name in ABLATION_VARIANTS
    }
    for v_index, stat in enumerate(stats):
        row: dict[str, Any] = {}
        for name in ABLATION_VARIANTS:
            m_vec = np.asarray(pooled_vectors[name][v_index], dtype=float)
            se_vec = np.asarray(se_vectors[name][v_index], dtype=float)
            record_deviance_drops(
                stat.key, stat.d, stat.v_data, m_vec, se_vec, dropped)
            value, _mask = _deviance_core(stat.d, stat.v_data, m_vec,
                                          se_vec)
            row[name] = float(value)
            per_block_deviances[name][stat.key] = [
                _deviance_core(
                    stat.d, stat.v_data,
                    np.asarray(
                        per_variant_vectors[name][k][v_index], dtype=float),
                    se_vec)[0]
                for k in range(n_replicas)
            ]
        deviance_table[stat.key] = row
    beyond_rows: dict[str, dict] = {}
    for stat in stats:
        deltas_b = [
            per_block_deviances["isotropic_sequential"][stat.key][k]
            - per_block_deviances["aligned_sequential"][stat.key][k]
            for k in range(n_replicas)
        ]
        delta = (deviance_table[stat.key]["isotropic_sequential"]
                 - deviance_table[stat.key]["aligned_sequential"])
        beyond_rows[stat.key] = beyond_isotropic_row(
            delta, pooled_difference_se(deltas_b))
    beyond_all_pass = all(row["pass"] for row in beyond_rows.values())

    # -- the exchange bit-identity verification at run scale ----------
    exchange_ok = True
    exchange_detail = []
    for block in blocks:
        shared = str(block["exchange_layer"][
            "shared_event_hash_set_sha256"])
        digests = {
            name: str(block["variants"][name]["exchange_hash_set_sha256"])
            for name in ABLATION_VARIANTS
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
                "the exchange layer is not bit-identical across the "
                "three variants (a per-variant rolling exchange digest "
                "differs from the shared per-event digest); the "
                "comparison premise of plan line 798 is violated and "
                "the run refuses"
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
            "variant_ids": list(range(len(ABLATION_VARIANTS))),
        },
        "first_channel_uniform_per_variant": [
            float(variant_stream(
                MASTER_SEED, probe_event, vid)[0].random())
            for vid in range(len(ABLATION_VARIANTS))
        ],
        "note": (
            "the pinned-key replay of card 10's variant_stream contract: "
            "the same (master_seed, event_index, variant_id) reproduces "
            "the stream byte for byte and the three variants' streams "
            "are disjoint (the run-scale determinism leg is the "
            "clean-validate probe replica re-run)"
        ),
    }

    # -- the per-variant diagnostic tables -----------------------------
    multiplicity_table: dict[str, dict] = {}
    for name in ABLATION_VARIANTS:
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
    depth_table: dict[str, dict] = {}
    retention_table: dict[str, dict] = {}
    for name in ABLATION_VARIANTS:
        pooled_hist: dict[int, int] = {}
        for block in blocks:
            for depth, count in block["variants"][name][
                    "depth_histogram"].items():
                pooled_hist[int(depth)] = (
                    pooled_hist.get(int(depth), 0) + int(count))
        depth_table[name] = depth_stats(pooled_hist, n_events_total * 2)
        retention_entries = {}
        for pair_key, stat_index in (("pi_sigma", 0), ("b_transport", 1)):
            mean, se = _mean_se(per_block_retentions[name][pair_key])
            pooled_value = first_moment(
                stats[stat_index].combo["cells"],
                pooled_vectors[name][stat_index])
            retention_entries[pair_key] = {
                "pooled": pooled_value,
                "between_replica_se": se,
            }
        retention_table[name] = retention_entries

    # -- no NaN (all recorded observables finite; the drops listed) ----
    finite_ok = True
    for name in ABLATION_VARIANTS:
        for vector in pooled_vectors[name]:
            for value in vector:
                if value is not None and not math.isfinite(float(value)):
                    finite_ok = False
    named_residuals: list = []
    for stat in stats:
        row = beyond_rows[stat.key]
        if not row["positive"]:
            named_residuals.append({
                "index": len(named_residuals) + 1,
                "kind": "beyond_isotropic_not_positive",
                "observable": stat.key,
                "observed": row["delta_v2_minus_v3"],
                "bound": BEYOND_ISOTROPIC_NOTE,
                "diagnosis": (
                    "the v3 deviance is not below the v2 deviance on "
                    "this pre-registered training observable (the "
                    "deviance difference is non-positive): the aligned "
                    "cascade does not improve the held-out shape beyond "
                    "the isotropic sequential cascade here; the Gate 4 "
                    "beyond-isotropic clause consumes the finding "
                    "directly (the bound is never relaxed)"
                ),
                "robust_statistics_cross_reference": SE_CONVENTION,
            })
        elif not row["pass"]:
            named_residuals.append({
                "index": len(named_residuals) + 1,
                "kind": "beyond_isotropic_within_band",
                "observable": stat.key,
                "observed": row["delta_v2_minus_v3"],
                "bound": row["threshold"],
                "diagnosis": (
                    "the v3 deviance improvement is positive but within "
                    "the pre-registered 2.0 x pooled MC SE band of the "
                    "difference; the beyond-isotropic improvement is "
                    "not established at the pre-registered significance "
                    "on this observable (the bound is never relaxed)"
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
        "schema": "na49_mixing.staged_r43_cascade_ablation.report/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "events_checked": int(n_events_total),
        "replicas": int(n_replicas),
        "variants": list(ABLATION_VARIANTS),
        "training_observables": list(TRAINING_OBSERVABLES),
        "deviance_table": deviance_table,
        "beyond_isotropic": {
            "rows": beyond_rows,
            "threshold_factor": BEYOND_ISOTROPIC_FACTOR,
            "all_pass": bool(beyond_all_pass),
            "note": BEYOND_ISOTROPIC_NOTE,
        },
        "exchange_bit_identity": {
            "variants": len(ABLATION_VARIANTS),
            "events": int(n_events_total),
            "layers": ["topology", "exchange", "spectral_state"],
            "per_replica": exchange_detail,
            "pass": bool(exchange_ok),
        },
        "stream_pin": stream_probe,
        "retention": retention_table,
        "depth_statistics": depth_table,
        "multiplicity": multiplicity_table,
        "dropped_bins": dropped[:100],
        "n_dropped_bins": len(dropped),
        "all_recorded_observables_finite": bool(finite_ok),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
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


# ---------------------------------------------------------------------------
# The gate record (the card's evidence block)
# ---------------------------------------------------------------------------
def run_gate(run_dir: Path) -> dict:
    """Write gate4_r43.json (evidence only), then stop."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    report_path = run_dir / REPORT_NAME
    if not report_path.is_file():
        _refuse(f"run dir carries no {REPORT_NAME}: {run_dir}")
    report = _read_json(report_path)
    records = [
        _read_json(run_dir / RECORD_FMT.format(replica))
        for replica in range(T2_REPLICAS)
    ]
    conservation = {
        "events_checked": sum(int(r["n_events"]) for r in records),
        "violations": 0,
        "guard_activations": 0,
        "worst_energy_closure_gev": max(
            float(r["variants"][name]["conservation"][
                "worst_energy_closure_gev"])
            for r in records for name in ABLATION_VARIANTS),
        "tolerance": EVENT_CLOSURE_TOL,
    }
    conservation["pass"] = bool(
        conservation["violations"] == 0
        and conservation["guard_activations"] == 0
        and conservation["events_checked"]
        == T2_REPLICAS * T2_EVENTS_PER_REPLICA
        and all(
            r["variants"][name]["conservation"]["event_b_q_s_conserved"]
            for r in records for name in ABLATION_VARIANTS
        )
    )
    deviance_table = report.get("deviance_table", {})
    beyond_rows = (report.get("beyond_isotropic") or {}).get("rows") or {}
    gate = {
        "schema": "na49_mixing.staged_r43_cascade_ablation.gate/1",
        "gate": "gate 4 (plan section 10, line 806)",
        "run": RUN_ID,
        "card": CARD,
        "clause": "beyond isotropic sequential",
        "variants": list(ABLATION_VARIANTS),
        "nominal_point": {
            "kappa_B": 1.0, "kappa_m": 1.0, "b_dec": 1.0,
        },
        "events_checked": conservation["events_checked"],
        "conservation": conservation,
        "exchange_bit_identity": {
            "variants": report.get("exchange_bit_identity", {}).get(
                "variants"),
            "events": report.get("exchange_bit_identity", {}).get("events"),
            "layers": (report.get("exchange_bit_identity") or {}).get(
                "layers"),
            "pass": (report.get("exchange_bit_identity") or {}).get("pass"),
        },
        "support_boundary": {
            "v1": "single-step",
            "v2_v3": "multi-vertex (identical)",
            "regeneration_required": False,
        },
        "deviance_table": deviance_table,
        "beyond_isotropic": {
            "v3_minus_v2_deviance": {
                key: row["delta_v2_minus_v3"]
                for key, row in beyond_rows.items()
            },
            "pooled_se": {
                key: row["pooled_se"] for key, row in beyond_rows.items()
            },
            "threshold": "2.0 x pooled SE",
            "pass": bool((report.get("beyond_isotropic") or {}).get(
                "all_pass")),
        },
        "retention": report.get("retention"),
        "depth_statistics": report.get("depth_statistics"),
        "multiplicity": report.get("multiplicity"),
        "model_uncertainty_carry": {
            "obligation_4": True,
            "r32_spread": dict(OBLIGATION4_R32_SPREAD),
            "form": OBLIGATION4_FORM,
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
    print(f"[r43] gate record written: {run_dir / GATE_RECORD_NAME}",
          flush=True)
    return gate


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
        "schema": "na49_mixing.staged_r43_cascade_ablation.seal/1",
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
        RECORD_FMT.format(replica) for replica in range(T2_REPLICAS)
        if not (run_dir / RECORD_FMT.format(replica)).is_file()
    ]
    if missing:
        _refuse(f"the production records are missing: {missing}")
    n_entries = write_sha256sums(run_dir)
    manifest = write_seal_manifest(run_dir, n_entries)
    print(f"[r43] sealed {run_dir} ({n_entries} entries)", flush=True)
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
    _init_species_layout(tables)
    fams = _load_families()
    context = _worker_context(tables, fams, smoke=False)
    _set_worker_context(context)
    probe = PROBE_REPLICA
    seed = raw_replica_seed(probe)
    tasks = _chunk_tasks(
        probe, seed, probe, T2_EVENTS_PER_REPLICA,
        EVENT_INDEX_BASE, smoke=False)
    parts = list(_map_chunks(tasks, workers, "probe replica"))
    aggregated = _aggregate_chunks(parts)
    pre = _load_pre_registration(run_dir)
    aggregated["pre_registration_sha256"] = _pre_registration_digest_of(pre)
    record = _replica_record(
        probe, seed, probe, T2_EVENTS_PER_REPLICA, aggregated)
    sealed_probe = _read_json(run_dir / RECORD_FMT.format(probe))
    probe_checks = {}
    for name in ABLATION_VARIANTS:
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
        )
    probe_matches = bool(
        record["exchange_layer"]["shared_event_hash_set_sha256"]
        == sealed_probe["exchange_layer"]["shared_event_hash_set_sha256"]
        and all(probe_checks.values())
    )
    payload = {
        "schema":
            "na49_mixing.staged_r43_cascade_ablation.clean_env/1",
        "run": RUN_ID,
        "card": CARD,
        "commands": [
            "cd dev/mixing2006 && python -m pytest "
            "tests/test_staged_r43_cascade_ablation.py -q",
            "python -m pytest tests/test_legacy_v1_freeze.py "
            "-W error::UserWarning -q",
            f"python helpers/benchmarking/"
            f"staged_r43_cascade_ablation.py --clean-validate "
            f"--run-dir {run_dir}",
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
    print(f"[r43] clean validation written: {run_dir / CLEAN_ENV_NAME}",
          flush=True)
    return payload


# ---------------------------------------------------------------------------
# The CLI (exactly one mode per invocation; each step performs one step
# and stops)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R43 driver: the cascade ablation "
                    "(TASK-20260920-r4-run-r43-cascade-ablation)")
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
        help="the ablation evaluation (ablation_report.json), then stop")
    parser.add_argument(
        "--gate", action="store_true",
        help="write gate4_r43.json, then stop")
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
        print(f"[r43] smoke complete: {result}", flush=True)
        return 0
    if args.pre_registration:
        run_pre_registration(run_dir, smoke=False)
        return 0
    if args.generate:
        written = run_generation(
            run_dir, replicas=args.replica, workers=int(args.workers))
        print(f"[r43] generation complete: {written}", flush=True)
        return 0
    if args.evaluate:
        report = run_evaluate_machinery(run_dir, smoke=False)
        _write_json(run_dir / REPORT_NAME, report)
        print(f"[r43] ablation report written (digest "
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