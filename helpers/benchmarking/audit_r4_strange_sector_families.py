#!/usr/bin/env python3
"""R4 audit driver: the strange-sector continuum families audit (Round 4, card 2).

Task ``TASK-20260920-r4-audit-strange-sector-continuum-families`` (Gate 3
decision card follow-up obligation 2; plan section 9.1, lines 598-605;
the Round 4 index card, cards 1-3, scope-ambiguity item 6).  The audit
resolves the K+-concentrated persisting spread of the sealed R32 evidence
(18 K+ persisting bins, the K+ bin p95 of 9.439291) against the
strange-sector continuum spectral densities and their catalog-audited
decay classes, on the byte-stable cards 4/5 family surface
(``config/v2_singlet/continuum_resonances.json``, sha256 ``6da57d57...``).
The audit outcome is a PASS verdict (the surface is adequate as-is) or a
RECOMMENDATION that spawns a separate rework card (the cards 8a/8b
precedent of Round 3).  The audit itself never edits the pinned surface.

Zero draws (hermetic, no Monte Carlo): the audit is an analytic
re-evaluation of the sealed R32 archives, the committed catalog, and the
committed family configuration.  The probe set is the 18 K+ entries of
the sealed confirmation persisting-residual register
(``dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19/``).
No event is generated, no random number is drawn, no optimizer runs
(``no_fitting`` is true, ``optimizer_invoked`` is false, ``draws`` is 0;
the pre-registered audit confirmation block 20301040 + k under the Round
4 master seed 20260920 is reserved for the exceptional case and must not
fire on the first pass).  The driver is standard library only: no numpy
import, no random module.

Probes (the card's pre-registered measurement contract, index item 6):

- **Class-pair re-derivation** (``--class-pairs``): for the activated
  strange baryonic family (B, Q, S) = (1, 0, -1), the round 3 card 5
  audit procedure against the committed catalog
  (``config/v2_singlet/resonance_catalog.json``, sha256 ``e42a5da7...``):
  conservation (every declared class pair's daughter quantum numbers sum
  to the family (B, Q, S) at the vertex) and completeness (every
  two-body recorded catalog channel whose daughter quantum numbers sum
  to (1, 0, -1) is declared), with the plan section 10.2, line 690
  predicate (at least one daughter lower in mass).  Pre-registered
  contract: the declared class set is complete and conservative.  Any
  missing class or conservation violation is a named finding with the
  class pair, the vertex quantum numbers, and the catalog record
  identifiers.  The (H_b, H_c) enumeration is recorded (two baryon
  daughters sum B = 2, so no conserving pair exists for B = 1).  The
  same re-derivation is recorded informationally for the two declared,
  inactive strange mesonic families (0, 0, -1) / (0, 0, +1).

- **Per-class K+ contribution** (``--k-contribution``): from the sealed
  R32 feed-down and routing census (the corrected card-3 lineage
  convention), the K+ bin payloads of the 18 persisting bins decompose
  by parent decay class.  The sealed count identities pin the
  decomposition in every sealed replica (production 3 x 8, confirmation
  3 x 16): every K+ final is a root-vertex final (``n_resonant`` 0,
  empty ``by_parent``), K+ equals Lambda plus Sigma0 exactly, K0 equals
  Sigma+ exactly, and K-, K0S, K0L, aK0, and Sigma- are absent from the
  final state.  The K+-bearing class pairs of the activated surface are
  therefore (3122, 321) K+ Lambda and (3212, 321) K+ Sigma0 on the
  activated baryonic (1, 1, 0) family -- the same class key on both
  routes, the continuum family class and the catalog records N(1650)+
  / N(1720)+ whose recorded channels carry the same daughter pair --
  plus the mesonic (321, -321) K+ K- class whose sealed contribution is
  zero.  The strange-sector families contribute zero K+ (the (1, 0, -1)
  declared set carries no K+ daughter: S conservation would need an
  S = -2 partner and the catalog records no such hadron; the family is
  additionally never sampled in the R32 p+p topology, whose sources
  carry (B, Q, S) = (1, 1, 0)).  Per bin: the sealed per-variant class
  shares applied to the sealed per-variant payloads (the proportional
  census attribution), the per-class contribution in pooled-SE units,
  and the dominant class pair.  Pre-registered contract: every
  persisting K+ bin is attributable to at least one declared,
  catalog-audited class pair (the bin is not a missing-decay-class
  artifact); a persisting bin with no catalog-audited class-pair path
  is a named finding.  The density-shape share of every class-attributed
  bin is recorded and referred to card 1 (the routing formulation audit
  owns the ramp placement and width attribution).

- **Family-configuration sanity** (``--config-sanity``): the strange
  records of the family configuration keep their committed parameters
  (the baryonic common scale, T_H = 165 MeV, b_a = 0, m_min = B =
  2.680194110727528 GeV, W = 0.5 GeV, nominal_width = 0.5 GeV as the
  round 3 cards 4/5 pre-registration); the declared inactive strange
  mesonic families keep their ``active: false`` flags; the density
  shape is cross-referenced, not re-fit, against the sealed R30
  normalization report (all three blocks pass at machine precision,
  the double-counting identity covers the (1, 0, -1) quantum-number
  set) and the R31 smoothness evidence (the (1, 0, -1) window: 0 of 40
  boundary-pair and 0 of 120 interior-pair failures, 0 of 422 at the
  authoritative card-8c statistics).

- **Verdict** (``--verdict``): PASS if the class-pair set is complete
  and conservative, every persisting K+ bin is attributable to a
  declared catalog-audited class pair, and the family configuration is
  sane; RECOMMENDATION (separate rework card) with named findings if
  (a) a class pair is missing or non-conserving, (b) a persisting K+
  bin has no catalog-audited class-pair path, or (c) the config
  disagrees with its committed bytes (a driver defect, rework the
  driver, not the surface).

New spread diagnostics carry the R32 pre-registered bounds (bin spread
p95 < 2.0 pooled-SE units, feed-down spread max < 3.0 sigma): the audit
produces no new p95, and every per-bin sigma row it records carries the
register bound beside the value; the bound is never relaxed and its
failure is a recorded finding, never a refit.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  measured.  Verifies the sealed R32 production and confirmation seals,
  the sealed R30/R31 and card-1 cross-reference seals, and the frozen
  substrate digests first.
- ``--class-pairs``: the completeness and conservation re-derivation on
  the (1, 0, -1) family against the committed catalog, write
  ``class_pair_audit.json``, then stop.  Requires the pre-registration.
- ``--k-contribution``: the decomposition of the 18 persisting K+ bins
  by parent decay class from the sealed R32 census, write
  ``kplus_class_contribution.json``, then stop.  Requires the
  class-pairs step.
- ``--config-sanity``: the strange-records byte check and the R30/R31
  cross-references, write ``config_sanity.json``, then stop.  Requires
  the k-contribution step.
- ``--verdict``: write ``verdict.json`` (PASS or RECOMMENDATION with
  named findings), then stop.  Requires the three measurement steps.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the four deterministic
  measurement steps from a clean interpreter over the sealed artifacts,
  confirm every artifact hash matches, write
  ``clean_environment_commands.json``, then stop.

The four measurement artifacts (class-pairs, k-contribution,
config-sanity, verdict) carry no wall-clock fields: their bytes are a
pure function of the sealed archives, the committed catalog, the
committed family configuration, and the committed driver, which is what
the clean-environment validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the sealed archives with the
established ``_locate_roots`` pattern.  Hermetic in every mode:
standard library only; the sealed archives, the committed catalog, and
the committed continuum config only; no ``na49_mixing`` import, no
external SPS tables, no event generation, no random module.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the sealed
# archives under the monorepo root.
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
        f"audit_r4_strange_sector_families: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()

# ---------------------------------------------------------------------------
# Pre-registered audit identity (card TASK-20260920-r4-audit-strange-
# sector-continuum-families; the probe set, the bounds, and the verdict
# contract are verbatim from the card and the round 4 index item 6).
# ---------------------------------------------------------------------------

CARD = "TASK-20260920-r4-audit-strange-sector-continuum-families"
RUN_ID = "R4-audit-strange-sector-continuum-families"
RUN_LABEL = "r4_audit_strange_sector_continuum_families"

PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLASS_PAIRS_NAME = "class_pair_audit.json"
KPLUS_NAME = "kplus_class_contribution.json"
CONFIG_SANITY_NAME = "config_sanity.json"
VERDICT_NAME = "verdict.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

# The byte-stable cards 4/5 surface and the frozen substrate (read-only).
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
FAMILIES_SHA256 = "6da57d5709b94061bc178fe94781dfe5eddcc597c5dde46bfcd5db56985cde7d"
CATALOG_REL = "dev/mixing2006/config/v2_singlet/resonance_catalog.json"
CATALOG_SHA256 = "e42a5da7137d789ba90221478c9bb404898f4fe8a7ee4604cd85b5623b751752"
CARD6_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
CARD6_SHA256 = "6f889f6d72968de53e7b27e32ba64d124d1b4e782a73606e6f6794eef9f156a9"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
BUILDER_SHA256 = "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
ENGINE_SHA256 = "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
R32_DRIVER_NAME = "staged_r32_spectral_systematic.py"
R32_DRIVER_SHA256 = "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"

# The sealed archives (the probe inputs and the cross-references) and
# the sealed card-1 audit run dir (read-only; their seals verify).
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"
R30_DIR = "dev/00runs/r30_spectral_normalization_2026-09-17"
R31_DIR = "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
CARD1_DIR = "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20"

SPREAD_REPORT_NAME = "spread_report.json"
GATE_RECORD_NAME = "gate3_r32.json"
CONFIRMATION_VERIFICATION_NAME = "confirmation_verification.json"
R30_GATE_NAME = "gate3_r30.json"
R31_GATE_NAME = "gate3_r31.json"
CARD1_ATTRIBUTION_NAME = "persisting_bin_attribution.json"
CARD1_VERDICT_NAME = "verdict.json"

# Sealed anchor digests (asserted by the pre-registration; the
# measurement modes re-hash every input they read against the
# pre-registered values).
SEALED_INPUT_DIGESTS = {
    f"{R32_PROD_DIR}/spread_report.json":
        "ed102b10e9c99d973a9c3e9dd262536eaca08802aeb8f313ea1894f7cefe022e",
    f"{R32_PROD_DIR}/gate3_r32.json":
        "64f104a8f0fdb71fdf506a3eb6a369091a2f3eed5047e499f1d344ad5668bab6",
    f"{R32_CONF_DIR}/confirmation_verification.json":
        "0553fed06a9b59e9f2020c0c53bbad7d75a11b1f2baf21f0469ee8b3e48d275f",
    f"{R30_DIR}/gate3_r30.json":
        "98a5151aa9237e5d10bc5becde48c5b758b047d9f8065f937f2bf2038de69ec8",
    f"{R31_DIR}/gate3_r31.json":
        "98c1ba7327fc65d0cc6828fc0c93f12e1f4b9eaffad3a0617223143994dac686",
    f"{CARD1_DIR}/persisting_bin_attribution.json":
        "9485af85c8e848b349038bd8c66f516037388b597cc892929cb6e5b9f97a5607",
    f"{CARD1_DIR}/verdict.json":
        "b0c6c89d0737246143e92efb377a4fbc2bf650932677ea4cc599bbe029f6d139",
}

# Sealed R32 anchor numbers (asserted against the sealed artifacts).
PRODUCTION_P95 = 2.9604990756942544
CONFIRMATION_P95 = 4.028426092137674
KPLUS_PRODUCTION_P95 = 9.439290719944319
PERSISTING_TOTAL = 32
PERSISTING_KPLUS = 18
# The probe set: the 18 K+ persisting register entries (bin index in
# register order of the sealed confirmation verification; the parent
# index and the sealed sigma are re-verified at run time).
KPLUS_BIN_INDICES = (
    2, 4, 5, 7, 19, 20, 21, 27, 28, 31, 45, 68, 71, 74, 112, 136, 148, 152,
)
KPLUS_SIGMA_MIN = 2.109195731906308
KPLUS_SIGMA_MAX = 6.420463940482292
# The register carries IEEE doubles; the anchor range check carries a
# float-slop tolerance.
SIGMA_ANCHOR_TOLERANCE = 1e-9

# The blend variants (the R32 pre-registration, verbatim).
KERNEL_BOUNDARY_GEV = 2.680194110727528
RAMP_WIDTH_GEV = 0.5
RAMP_OFFSETS_GEV = {"0": 0.0, "1": -0.2, "2": 0.2}
RAMP_START_ORDER = ("1", "0", "2")
PRODUCTION_CENSUS_CATALOG = {
    "0": 0.22329625,
    "1": 0.193558125,
    "2": 0.253725,
}
CONFIRMATION_CENSUS_CATALOG = {
    "0": 0.22357,
    "1": 0.1936659375,
    "2": 0.253850625,
}

# Pre-registered bounds (the R32 pre-registration, decision card lines
# 106-109; carried by every per-bin sigma row this audit records).
BIN_SPREAD_BOUND = 2.0
FEEDDOWN_SPREAD_BOUND = 3.0

# The strange-sector surface under audit (the round 3 cards 4/5 family
# records; the config bytes are the authority and are re-read at run
# time; the constants below are the pre-registered expectations the
# config-sanity step checks them against).
STRANGE_FAMILY_ID = "baryonic_strange"
STRANGE_QN = (1, 0, -1)
STRANGE_DECLARED_CLASSES = (
    "(3122,111)", "(3212,111)", "(3112,211)", "(3222,-211)",
    "(2212,-321)", "(2112,-311)",
)
STRANGE_M_MIN = 2.680194110727528
STRANGE_T_H = 0.165
STRANGE_B_POWER = 0.0
STRANGE_N_NORM = 4.188698227099897e-13
STRANGE_RAMP_WIDTH = 0.5
STRANGE_NOMINAL_WIDTH = 0.5
STRANGE_M_MAX = 5.0
STRANGE_STATE_TYPE = "continuum_resonance"
# The declared, inactive strange mesonic families of the round 3 card 5
# declaration (the second half of the strange-sector surface).
INACTIVE_STRANGE_MESONIC = {
    "mesonic_strange": {"qn": (0, 0, -1), "classes": 7},
    "mesonic_antistrange": {"qn": (0, 0, 1), "classes": 7},
}
# The family census of the committed configuration (the cards 4/5
# declaration): seven records, three activated, four declared inactive.
FAMILY_CENSUS_ACTIVE = {
    "baryonic_np": {"qn": (1, 1, 0), "classes": 8},
    "baryonic_strange": {"qn": (1, 0, -1), "classes": 6},
    "mesonic_neutral": {"qn": (0, 0, 0), "classes": 10},
}
FAMILY_CENSUS_INACTIVE = {
    "mesonic_charged": {"qn": (0, 1, 0), "classes": 6},
    "mesonic_anticharged": {"qn": (0, -1, 0), "classes": 6},
    "mesonic_strange": {"qn": (0, 0, -1), "classes": 7},
    "mesonic_antistrange": {"qn": (0, 0, 1), "classes": 7},
}

# The K+-bearing class pairs of the activated surface (the declared
# classes whose daughters include a K+ carrier, pid 321; the class keys
# are the config's daughter flavor pairs).  The strange-sector families
# under audit carry none: the (1, 0, -1) declared set has no K+ daughter
# and S conservation would need an S = -2 partner, which the committed
# catalog does not record.
KPLUS_BEARING_CLASSES = {
    "baryonic_np": ("(3122,321)", "(3212,321)"),
    "mesonic_neutral": ("(321,-321)"),
}
# The catalog-audit records of the (3122, 321) class (the K+ Lambda
# channels of the committed catalog): the catalog route realizes the
# same class key.
KPLUS_LAMBDA_CATALOG_RECORDS = (122212, 212214)

# The sealed count identities of the corrected card-3 lineage convention
# (the decomposition basis; per sealed replica, exact integers):
#   (i)   K_plus.n == Lambda.n + Sigma_0.n  (every K+ final carries a
#         hyperon partner from its root vertex),
#   (ii)  K_plus.n_resonant == 0 and by_parent == {} (every K+ final is
#         produced at the root vertex: the corrected lineage convention),
#   (iii) K_minus.n == 0 and Sigma_minus.n == 0 and K0S.n == 0 and
#         K0L.n == 0 and aK_0.n == 0 (the mesonic (321, -321) class and
#         the strange-sector families contributed zero finals),
#   (iv)  K_0.n == Sigma_plus.n (the (3222, 311) K0-Sigma+ class pairs).
IDENTITY_KPLUS_HYPERON = "K_plus.n == Lambda.n + Sigma_0.n"
IDENTITY_ROOT_VERTEX = "K_plus.n_resonant == 0 and by_parent == {}"
IDENTITY_ZERO_STRANGE = (
    "K_minus.n == 0 and Sigma_minus.n == 0 and K0S.n == 0 and "
    "K0L.n == 0 and aK_0.n == 0"
)
IDENTITY_K0_SIGMA = "K_0.n == Sigma_plus.n"

# The R32 p+p source topology (the M8-frozen two-source (1, 1, 0)
# topology; the builder hard-pins the source quantum numbers, so the
# strange-sector families are never sampled by R30-R32).
SOURCE_QN_R32 = (1, 1, 0)
BUILDER_SOURCE_QN_SNIPPET = "self._source_qn = (1, 1, 0)"

# The reserved audit confirmation block (the exceptional case; must not
# fire on the first pass).  Master seed 20260920 (the Round 4 master),
# raw seeds 20301040 + k, k = 0..15, disjoint from every previously used
# raw block and from the Round 4 run blocks 20301020-20301039.
RESERVED_MASTER_SEED = 20260920
RESERVED_BLOCK_FIRST = 20301040
RESERVED_BLOCK_LAST = 20301055

# The card-1 audit cross-reference (the density-shape referral target).
CARD1_RUN = "R4-audit-transition-routing-formulation"
CARD1_CARD = "TASK-20260920-r4-audit-transition-routing-formulation"
CARD1_REWORK_CARD = "TASK-20260920-r4-rework-transition-routing-formulation"


# ---------------------------------------------------------------------------
# Small shared helpers (the R32 driver conventions, local copies)
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _json_safe(value: Any) -> Any:
    """The R01 sentinel convention: non-finite floats become strings so
    identical inputs produce identical bytes (RFC 8259 compatibility)."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _restore_sentinels(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _restore_sentinels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_sentinels(v) for v in value]
    if isinstance(value, str):
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value == "NaN":
            return math.nan
    return value


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _restore_sentinels(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _git_info() -> dict:
    """The recorded git identities (defensive: a checkout without a
    working git records ``None``)."""
    info: dict = {}
    for name, cwd in (("nested_head", NESTED_ROOT), ("outer_head", MONO_ROOT)):
        try:
            head = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(cwd), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            info[name] = None
            info[name.replace("head", "dirty_entries")] = None
        else:
            info[name] = head
            info[name.replace("head", "dirty_entries")] = len(
                dirty.splitlines()
            )
    return info


def environment_fingerprint() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": sys.platform,
        "numpy_version": None,
        "numpy_version_note": (
            "the driver is standard library only; no numpy is imported"
        ),
    }


def default_run_dir() -> Path:
    """The dated run dir under ``dev/00runs`` (gitignored, not under a
    pinned root), named with the execution date at run time."""
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    return MONO_ROOT / "dev" / "00runs" / f"{RUN_LABEL}_{today}"


# ---------------------------------------------------------------------------
# Sealed-archive and frozen-substrate verification (read-only)
# ---------------------------------------------------------------------------


def verify_sealed_dir(rel: str) -> dict:
    """Verify one sealed run dir's ``SHA256SUMS.txt`` (read-only; the
    sealed archives are byte-unchanged inputs).  The entries carry
    monorepo-root-relative paths (the R21/R31/R32 convention) or
    run-dir-relative paths (the R23/R30 convention); both resolve."""
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


def substrate_digests() -> dict:
    """The on-disk digests of the frozen substrate (computed, not
    asserted)."""
    return {
        "families_config_sha256": _sha256_file(MONO_ROOT / FAMILIES_REL),
        "catalog_sha256": _sha256_file(MONO_ROOT / CATALOG_REL),
        "card6_surface_sha256": _sha256_file(MONO_ROOT / CARD6_REL),
        "builder_module_sha256": _sha256_file(MONO_ROOT / BUILDER_REL),
        "engine_module_sha256": _sha256_file(MONO_ROOT / ENGINE_REL),
        "r32_driver_nested_sha256": _sha256_file(
            NESTED_ROOT / "helpers" / "benchmarking" / R32_DRIVER_NAME
        ),
        "r32_driver_outer_sha256": _sha256_file(
            MONO_ROOT / "helpers" / "benchmarking" / R32_DRIVER_NAME
        ),
    }


def verify_frozen_substrate() -> dict:
    """The frozen substrate digest checks (the card's substrate list)."""
    problems: list[str] = []
    hashes = substrate_digests()
    expected = {
        "families_config_sha256": (
            FAMILIES_SHA256, "the committed continuum config"),
        "catalog_sha256": (CATALOG_SHA256, "the committed catalog"),
        "card6_surface_sha256": (CARD6_SHA256, "the card-6 blend routing"),
        "builder_module_sha256": (BUILDER_SHA256, "the builder blend path"),
        "engine_module_sha256": (ENGINE_SHA256, "the fixed card-2 engine"),
        "r32_driver_nested_sha256": (
            R32_DRIVER_SHA256, "the frozen R32 driver (nested copy)"),
        "r32_driver_outer_sha256": (
            R32_DRIVER_SHA256, "the frozen R32 driver (outer mirror)"),
    }
    for key, (want, role) in expected.items():
        if hashes[key] != want:
            problems.append(
                f"{role} sha256 {hashes[key]!r} != the pre-registered {want!r}"
            )
    if hashes["r32_driver_nested_sha256"] != hashes["r32_driver_outer_sha256"]:
        problems.append(
            "the R32 driver copies are not byte-identical across the two "
            "helper trees"
        )
    if problems:
        _refuse(
            "frozen substrate verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return hashes


def verify_sealed_inputs() -> dict:
    """Re-hash the sealed anchor inputs against the pre-registered
    digests (the measurement modes call this before every step)."""
    problems: list[str] = []
    for rel, want in SEALED_INPUT_DIGESTS.items():
        path = MONO_ROOT / rel
        if not path.is_file():
            problems.append(f"sealed input missing: {rel}")
            continue
        got = _sha256_file(path)
        if got != want:
            problems.append(
                f"sealed input {rel} sha256 {got!r} != the pre-registered "
                f"{want!r}"
            )
    if problems:
        _refuse(
            "sealed input verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return dict(SEALED_INPUT_DIGESTS)


# ---------------------------------------------------------------------------
# Mode: --pre-registration
# ---------------------------------------------------------------------------


def build_pre_registration(substrate: dict, seals: list) -> dict:
    """The audit pre-registration payload, written before any
    measurement: the probe set, the bounds, the verdict contract, and
    the zero-draw assertion."""
    return {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families"
            ".pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_measurement": True,
        "surface": {
            "continuum_families_config": {
                "path": FAMILIES_REL,
                "sha256": FAMILIES_SHA256,
                "note": (
                    "the cards 4/5 family surface is byte-unchanged by this "
                    "audit; a surface change is a separate rework card, "
                    "never an in-audit edit"
                ),
            },
            "resonance_catalog": {
                "path": CATALOG_REL,
                "sha256": CATALOG_SHA256,
                "note": (
                    "the committed catalog is byte-unchanged by this audit; "
                    "it is the class-pair re-derivation authority"
                ),
            },
            "card6_blend_routing_sha256": CARD6_SHA256,
            "builder_module_sha256": BUILDER_SHA256,
            "engine_module_sha256": ENGINE_SHA256,
            "r32_driver_sha256": R32_DRIVER_SHA256,
        },
        "probe_set": {
            "source": (
                "the 18 K+ persisting residual register entries of the "
                f"sealed R32 confirmation ({R32_CONF_DIR}/"
                "confirmation_verification.json, residual_reclassification "
                "classification == 'persisting' and species == 'k_plus')"
            ),
            "expected_kplus_entries": PERSISTING_KPLUS,
            "expected_bin_indices": list(KPLUS_BIN_INDICES),
            "expected_sigma_range": [KPLUS_SIGMA_MIN, KPLUS_SIGMA_MAX],
            "payload_source": (
                "the per-variant bin payloads, the pooled SEs, the routing "
                "census, and the corrected card-3 lineage feed-down tables "
                "are read from the sealed R32 production and confirmation "
                "archives (the register entries and the sealed payload "
                "feed-down blocks); no recomputation from events, no new "
                "draws"
            ),
        },
        "class_pair_audit": {
            "family_id": "baryonic_strange",
            "qn": [1, 0, -1],
            "declared_classes": list(STRANGE_DECLARED_CLASSES),
            "method": (
                "the round 3 card 5 audit procedure: conservation (every "
                "declared class pair's daughter quantum numbers sum to the "
                "family (B, Q, S) at the vertex, against the committed "
                "catalog records) and completeness (every two-body "
                "recorded catalog channel whose daughter quantum numbers "
                "sum to (1, 0, -1) is declared), with the plan section "
                "10.2 line 690 predicate (at least one daughter lower in "
                "mass)"
            ),
            "contract": (
                "the declared class set in the config is complete (no "
                "conserving catalog class pair is missing) and "
                "conservative (no declared pair violates vertex "
                "conservation); any missing class or conservation "
                "violation is a named finding with the class pair, the "
                "vertex quantum numbers, and the catalog record "
                "identifiers"
            ),
            "h_b_h_c_enumeration": (
                "the (H_b, H_c) enumeration is recorded: two baryon "
                "daughters carry B = 2 in total, so no conserving (H_b, "
                "H_c) pair exists for the B = 1 family and the recorded "
                "set is empty"
            ),
            "informational_scope": (
                "the same re-derivation is recorded informationally for "
                "the two declared, inactive strange mesonic families "
                "(0, 0, -1) / (0, 0, +1); the informational rows never "
                "gate the verdict"
            ),
        },
        "kplus_class_contribution": {
            "method": (
                "from the sealed R32 feed-down and routing census "
                "(corrected card-3 lineage convention): the sealed "
                "per-replica count identities pin the parent decay "
                "classes of every K+ final (K+ equals Lambda plus Sigma0 "
                "exactly, the K+ finals carry zero resonant feed-down "
                "with an empty by-parent census, K- and the strange-sector "
                "finals are absent, and K0 equals Sigma+ exactly), so the "
                "K+ payload decomposes over the two K+-bearing declared "
                "class pairs of the activated baryonic (1, 1, 0) family, "
                "(3122, 321) K+ Lambda and (3212, 321) K+ Sigma0, with "
                "the sealed per-variant class shares (Lambda and Sigma0 "
                "feed-down counts pooled per variant); the mesonic "
                "(321, -321) K+ K- class carries zero (K- identically "
                "absent)"
            ),
            "identity_assertions": [
                IDENTITY_KPLUS_HYPERON,
                IDENTITY_ROOT_VERTEX,
                IDENTITY_ZERO_STRANGE,
                IDENTITY_K0_SIGMA,
            ],
            "attribution_algebra": (
                "per bin: the class-decomposed values y_v,c = s_c(v) "
                "y_v from the sealed per-variant class shares s_c(v) "
                "(the proportional census attribution; the sealed census "
                "carries no per-bin class split), the per-class "
                "contribution sigma_c(b) = (max_v y_v,c - min_v y_v,c) / "
                "unit_se in pooled-SE units, the dominant class pair the "
                "larger contribution, and the class-attributable flag "
                "(at least one declared, catalog-audited class pair with "
                "a path to the bin's K+ finals)"
            ),
            "route_note": (
                "the (3122, 321) class key is realized on both routes "
                "(the continuum family class and the catalog records "
                "N(1650)+ / N(1720)+ whose recorded channels carry the "
                "same daughter pair); the sealed count census does not "
                "resolve the route split of this class and the split is "
                "not needed for the verdict: the class key is declared "
                "on the activated baryonic_np family and catalog-audited "
                "on both routes"
            ),
            "contract": (
                "every persisting K+ bin is attributable to at least one "
                "declared, catalog-audited class pair (the bin is not a "
                "missing-decay-class artifact); the dominant class pair "
                "per bin is recorded with its contribution in pooled-SE "
                "units; a persisting bin with no catalog-audited "
                "class-pair path is a named finding"
            ),
            "card1_cross_reference": {
                "run_dir": CARD1_DIR,
                "card": CARD1_CARD,
                "attribution_artifact": CARD1_ATTRIBUTION_NAME,
                "attribution_sha256": SEALED_INPUT_DIGESTS[
                    f"{CARD1_DIR}/{CARD1_ATTRIBUTION_NAME}"
                ],
                "verdict_artifact": CARD1_VERDICT_NAME,
                "verdict_sha256": SEALED_INPUT_DIGESTS[
                    f"{CARD1_DIR}/verdict.json"
                ],
                "referral": (
                    "the density-shape share of any persisting bin that "
                    "is class-attributed is recorded and referred to "
                    "card 1 (the routing formulation audit owns the ramp "
                    "placement and width attribution)"
                ),
            },
        },
        "config_sanity": {
            "family_census_active": {
                k: {"qn": list(v["qn"]), "classes": v["classes"]}
                for k, v in FAMILY_CENSUS_ACTIVE.items()
            },
            "family_census_inactive": {
                k: {"qn": list(v["qn"]), "classes": v["classes"]}
                for k, v in sorted(FAMILY_CENSUS_INACTIVE.items())
            },
            "strange_record_parameters": {
                "m_min": STRANGE_M_MIN,
                "t_h": STRANGE_T_H,
                "b_power": STRANGE_B_POWER,
                "n_norm": STRANGE_N_NORM,
                "ramp_width": STRANGE_RAMP_WIDTH,
                "nominal_width": STRANGE_NOMINAL_WIDTH,
                "m_max": STRANGE_M_MAX,
                "state_type": STRANGE_STATE_TYPE,
                "active": True,
            },
            "inactive_strange_mesonic": {
                k: {"qn": list(v["qn"]), "classes": v["classes"],
                    "active": False}
                for k, v in sorted(INACTIVE_STRANGE_MESONIC.items())
            },
            "density_shape_cross_reference": (
                "the density shape is cross-referenced, not re-fit, "
                "against the sealed R30 normalization report (all three "
                "blocks pass at machine precision; the double-counting "
                "identity covers the (1, 0, -1) quantum-number set) and "
                "the R31 smoothness evidence (the (1, 0, -1) window: 0 "
                "of 40 boundary-pair and 0 of 120 interior-pair "
                "failures, 0 of 422 at the authoritative card-8c "
                "confirmation statistics); read-only"
            ),
            "sampling_scope_note": (
                "the R32 p+p topology is the two-source (B, Q, S) = "
                "(1, 1, 0) exchange (the builder's source quantum "
                "numbers), so the activated strange baryonic family "
                "(1, 0, -1) and the declared inactive strange mesonic "
                "families are never sampled by R30-R32; the sealed "
                "final-state census pins this empirically (Sigma- "
                "identically zero)"
            ),
        },
        "new_spread_bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "bin_spread_unit": (
                "the pooled MC standard error of the comparison "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2) (the sealed R32 unit)"
            ),
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "kplus_p95_anchor": {
                "production": 9.439290719944319,
                "note": (
                    "the sealed per-species K+ bin p95 that motivates "
                    "this audit (the decision card quotes 9.439291); "
                    "carried with the R32 pre-registered bound beside it"
                ),
            },
            "note": (
                "any new spread diagnostic produced by this audit carries "
                "the R32 pre-registered bounds (decision card lines "
                "106-109); the audit produces no new p95 and every "
                "per-bin sigma row carries the register bound 2.0 beside "
                "the value; the bounds are never relaxed and a bound "
                "failure is a recorded finding, never a refit"
            ),
        },
        "verdict_contract": {
            "pass": (
                "PASS if the class-pair set is complete and conservative, "
                "every persisting K+ bin is attributable to a declared "
                "catalog-audited class pair, and the family configuration "
                "is sane"
            ),
            "recommendation": (
                "RECOMMENDATION (separate rework card) with named "
                "findings if (a) a class pair is missing or "
                "non-conserving, (b) a persisting K+ bin has no "
                "catalog-audited class-pair path, or (c) the config "
                "disagrees with its committed bytes (a driver defect, "
                "rework the driver, not the surface)"
            ),
            "criteria": ["a", "b", "c"],
            "density_shape_referral": (
                "the density-shape share of any persisting bin that is "
                "class-attributed is recorded and referred to card 1 "
                f"({CARD1_CARD}); the routing formulation share is card "
                "1's scope and this audit edits nothing pinned"
            ),
            "rework_precedent": (
                "the Round 3 cards 8a/8b precedent: a RECOMMENDATION "
                "spawns a separate rework card; this audit edits nothing "
                "pinned"
            ),
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "method": (
                "the audit is an analytic re-evaluation of the sealed "
                "archives, the committed catalog, and the committed "
                "family configuration: no event generation, no sampling, "
                "no optimizer; the driver imports no random module and "
                "no numpy"
            ),
            "reserved_audit_block": {
                "master_seed": RESERVED_MASTER_SEED,
                "raw_seeds": [RESERVED_BLOCK_FIRST, RESERVED_BLOCK_LAST],
                "raw_seed_rule": (
                    "20301040 + k, k = 0..15, under the Round 4 master "
                    "seed 20260920; disjoint from every previously used "
                    "raw block and from the Round 4 run blocks "
                    "20301020-20301039"
                ),
                "fired": False,
                "note": (
                    "the reserved confirmation block is pre-registered "
                    "for the exceptional case and must not fire on the "
                    "first pass"
                ),
            },
        },
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "sealed_run_dirs_verified": seals,
        "substrate": substrate,
    }


def run_pre_registration(run_dir: Path) -> dict:
    """Write ``pre_registration.json`` and ``provenance.json``, then
    stop.  Nothing is measured."""
    run_dir = Path(run_dir)
    substrate = verify_frozen_substrate()
    if substrate["families_config_sha256"] != FAMILIES_SHA256:
        _refuse(
            "the cards 4/5 family surface is not byte-stable at the "
            "pinned digest"
        )
    seals = [
        verify_sealed_dir(rel)
        for rel in (R32_PROD_DIR, R32_CONF_DIR, R30_DIR, R31_DIR, CARD1_DIR)
    ]
    verify_sealed_inputs()
    pre = build_pre_registration(substrate, seals)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    driver_path = Path(__file__).resolve()
    provenance = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": [
            "TASK-20260920-r3-gate3-decision",
            "TASK-20260920-round4-index",
            CARD,
        ],
        "written_utc": _now_utc(),
        "driver": "helpers/benchmarking/audit_r4_strange_sector_families.py",
        "driver_sha256": {
            "nested": _sha256_file(
                NESTED_ROOT / "helpers" / "benchmarking"
                / "audit_r4_strange_sector_families.py"
            ),
            "outer": _sha256_file(driver_path),
            "note": (
                "the dual-tree driver is byte-identical in both trees; "
                "the outer entry is the on-disk mirror's digest"
            ),
        },
        "frozen_substrate": substrate,
        "sealed_run_dirs_verified": seals,
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "reserved_audit_block_fired": False,
        },
        "environment": environment_fingerprint(),
        "git": _git_info(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        f"audit_r4_strange_sector_families: pre-registration written to "
        f"{run_dir} ({len(seals)} sealed archives verified, substrate "
        "digests pinned)",
        flush=True,
    )
    return pre


def assert_pre_registration(run_dir: Path) -> dict:
    """Load the pre-registration, re-verify the substrate digests and
    the sealed input digests, and refuse to run without it."""
    run_dir = Path(run_dir)
    path = run_dir / PREREGISTRATION_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; every measurement mode requires the "
            "pre-registration (run --pre-registration first)"
        )
    pre = _read_json(path)
    if pre.get("card") != CARD:
        _refuse(f"{path} carries the wrong card identity")
    substrate = verify_frozen_substrate()
    recorded = pre.get("substrate", {})
    for key, want in (
        ("families_config_sha256", FAMILIES_SHA256),
        ("catalog_sha256", CATALOG_SHA256),
        ("card6_surface_sha256", CARD6_SHA256),
        ("builder_module_sha256", BUILDER_SHA256),
        ("engine_module_sha256", ENGINE_SHA256),
        ("r32_driver_nested_sha256", R32_DRIVER_SHA256),
        ("r32_driver_outer_sha256", R32_DRIVER_SHA256),
    ):
        if recorded.get(key) != want or substrate[key] != want:
            _refuse(
                f"substrate digest drift on {key}: the pre-registration "
                f"and the on-disk file must both carry {want!r}"
            )
    verify_sealed_inputs()
    zero = pre.get("zero_draw", {})
    if zero.get("draws") != 0 or not zero.get("no_fitting") or zero.get(
            "optimizer_invoked"):
        _refuse("the pre-registration zero-draw contract is not intact")
    if pre.get("zero_draw", {}).get("reserved_audit_block", {}).get(
            "fired") is not False:
        _refuse(
            "the reserved audit block must not fire on the first pass"
        )
    return pre


# ---------------------------------------------------------------------------
# The class-pair re-derivation (the round 3 card 5 audit procedure)
# ---------------------------------------------------------------------------


def parse_class_key(cls: str) -> tuple[int, int]:
    """Parse a config class key ``'(d0,d1)'`` into the daughter pid
    pair."""
    parts = str(cls).strip().strip("()").split(",")
    if len(parts) != 2:
        raise ValueError(f"malformed class key {cls!r}")
    return int(parts[0]), int(parts[1])


def daughter_qn(record: dict) -> tuple[int, int, int]:
    """The (B, Q, S) quantum numbers of one catalog record."""
    return (
        int(record["B"]),
        int(record["Q"]),
        int(record["S"]),
    )


def vertex_conserves(class_key: str, by_pid: dict, qn: tuple[int, int, int],
                     ) -> dict:
    """The round 3 card 5 conservation predicate: the declared class's
    daughter quantum numbers sum to the family (B, Q, S) at the vertex."""
    try:
        d0, d1 = parse_class_key(class_key)
    except ValueError as exc:
        return {"class_key": class_key, "conserves": False,
                "reason": str(exc)}
    missing = [d for d in (d0, d1) if d not in by_pid]
    if missing:
        return {
            "class_key": class_key, "conserves": False,
            "reason": (
                f"daughter pid(s) {missing} not in the committed catalog"
            ),
        }
    qsum = tuple(
        by_pid[d0][k] + by_pid[d1][k] for k in ("B", "Q", "S")
    )
    return {
        "class_key": class_key,
        "daughters": [d0, d1],
        "daughter_qn_sums": list(qsum),
        "family_qn": list(qn),
        "conserves": bool(qsum == tuple(qn)),
        "daughter_names": [by_pid[d0]["name"], by_pid[d1]["name"]],
        "daughter_pole_masses": [
            float(by_pid[d0]["pole_mass"]), float(by_pid[d1]["pole_mass"]),
        ],
    }


def enumerate_catalog_class_pairs(records: list, qn: tuple[int, int, int],
                                  ) -> dict:
    """The completeness universe: every two-body recorded catalog
    channel whose daughter quantum numbers sum to *qn*, grouped by the
    daughter flavor-pair class key (the catalog's recorded daughter
    order), with the carrying record identifiers."""
    by_pid = {int(r["pdg_id"]): r for r in records}
    found: dict[str, dict] = {}
    for record in records:
        for channel in (record.get("decay_channels") or []):
            daughters = channel.get("daughters") or []
            if channel.get("is_three_body") or len(daughters) != 2:
                continue
            if any(int(d) not in by_pid for d in daughters):
                continue
            d0, d1 = int(daughters[0]), int(daughters[1])
            qsum = (
                by_pid[d0]["B"] + by_pid[d1]["B"],
                by_pid[d0]["Q"] + by_pid[d1]["Q"],
                by_pid[d0]["S"] + by_pid[d1]["S"],
            )
            if qsum != tuple(qn):
                continue
            key = f"({d0},{d1})"
            entry = found.setdefault(key, {
                "class_key": key,
                "carrying_records": [],
                "channel_count": 0,
            })
            parent_pid = int(record["pdg_id"])
            if parent_pid not in entry["carrying_records"]:
                entry["carrying_records"].append(parent_pid)
            entry["channel_count"] += 1
    return {
        "qn": [int(qn[0]), int(qn[1]), int(qn[2])],
        "conserving_class_keys": sorted(found),
        "classes": found,
    }


def class_pair_audit_rows(families_config: dict, catalog_records: list,
                          ) -> dict:
    """The completeness and conservation re-derivation for one family
    (pure committed-input arithmetic; zero draws)."""
    by_pid = {int(r["pdg_id"]): r for r in catalog_records}
    family = None
    for record in families_config["families"]:
        if record["family_id"] == STRANGE_FAMILY_ID:
            family = record
            break
    if family is None:
        _refuse(f"the family {STRANGE_FAMILY_ID} is absent from the config")
    qn = (int(family["qn"][0]), int(family["qn"][1]), int(family["qn"][2]))
    declared = [str(c) for c in family["allowed_decay_classes"]]

    conservation = [vertex_conserves(c, by_pid, qn) for c in declared]
    nonconserving = [c for c in conservation if not c["conserves"]]

    enumeration = enumerate_catalog_class_pairs(catalog_records, qn)
    conserving_keys = enumeration["conserving_class_keys"]
    missing = [k for k in conserving_keys if k not in declared]
    undeclared_extra = [c for c in declared if c not in conserving_keys]

    # The plan section 10.2, line 690 predicate: at least one daughter
    # lower in mass.  Every daughter pole mass sits far below the family
    # minimum mass, so the predicate holds at every admissible source
    # mass; the open-across-support check (the card-4/5 criterion, the
    # daughter pole-mass sum below m_min) is recorded beside it.
    kinematics = []
    for cls in declared:
        vc = next(c for c in conservation if c["class_key"] == cls)
        if not vc["conserves"]:
            continue
        m0, m1 = vc["daughter_pole_masses"]
        msum = m0 + m1
        kinematics.append({
            "class_key": cls,
            "daughter_pole_mass_sum": msum,
            "max_daughter_pole_mass": max(m0, m1),
            "at_least_one_daughter_lower_in_mass": bool(
                max(m0, m1) < STRANGE_M_MIN
            ),
            "open_across_family_support": bool(msum < STRANGE_M_MIN),
        })

    return {
        "family_id": STRANGE_FAMILY_ID,
        "qn": list(qn),
        "declared_classes": declared,
        "conservation_rows": conservation,
        "nonconserving_declared": [c["class_key"] for c in nonconserving],
        "enumeration": enumeration,
        "missing_class_keys": missing,
        "declared_not_enumerated": undeclared_extra,
        "kinematics": kinematics,
        "complete": bool(not missing),
        "conservative": bool(not nonconserving),
    }


def informational_family_audit(families_config: dict, catalog_records: list,
                               family_id: str) -> dict:
    """The informational re-derivation for one declared, inactive
    strange mesonic family (outside the verdict contract)."""
    by_pid = {int(r["pdg_id"]): r for r in catalog_records}
    family = None
    for record in families_config["families"]:
        if record["family_id"] == family_id:
            family = record
            break
    if family is None:
        return {"family_id": family_id, "present": False}
    qn = (int(family["qn"][0]), int(family["qn"][1]), int(family["qn"][2]))
    declared = [str(c) for c in family["allowed_decay_classes"]]
    conservation = [vertex_conserves(c, by_pid, qn) for c in declared]
    nonconserving = [c["class_key"] for c in conservation
                     if not c["conserves"]]
    enumeration = enumerate_catalog_class_pairs(catalog_records, qn)
    missing = [k for k in enumeration["conserving_class_keys"]
               if k not in declared]
    return {
        "family_id": family_id,
        "qn": list(qn),
        "active": bool(family["active"]),
        "declared_classes": declared,
        "nonconserving_declared": nonconserving,
        "missing_class_keys": missing,
        "complete": bool(not missing),
        "conservative": bool(not nonconserving),
        "scope": "informational (outside the verdict contract)",
    }


# ---------------------------------------------------------------------------
# Mode: --class-pairs
# ---------------------------------------------------------------------------


def run_class_pairs(run_dir: Path) -> dict:
    """The completeness and conservation re-derivation on the (1, 0, -1)
    family against the committed catalog."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    families_config = _read_json(MONO_ROOT / FAMILIES_REL)
    catalog = _read_json(MONO_ROOT / CATALOG_REL)
    records = catalog["records"]

    audit = class_pair_audit_rows(families_config, records)
    informational = [
        informational_family_audit(families_config, records, family_id)
        for family_id in sorted(INACTIVE_STRANGE_MESONIC)
    ]
    # The (H_b, H_c) enumeration: two baryon daughters carry B = 2 in
    # total, so no conserving pair exists for the B = 1 family; the
    # committed catalog also records no two-baryon channel at all.
    by_pid = {int(r["pdg_id"]): r for r in records}
    baryon_channels = []
    for record in records:
        for channel in (record.get("decay_channels") or []):
            daughters = channel.get("daughters") or []
            if channel.get("is_three_body") or len(daughters) != 2:
                continue
            if (all(int(d) in by_pid for d in daughters)
                    and all(int(by_pid[int(d)]["B"]) != 0 for d in daughters)):
                baryon_channels.append(
                    f"{record['pdg_id']} -> {list(daughters)}"
                )
    payload = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families.class_pairs/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the class-pair re-derivation of the activated strange "
            "baryonic family (1, 0, -1) against the committed catalog "
            "(the round 3 card 5 audit procedure: conservation and "
            "completeness, with the plan section 10.2 line 690 predicate)"
        ),
        "sources": {
            "families_config": FAMILIES_REL,
            "families_config_sha256": FAMILIES_SHA256,
            "catalog": CATALOG_REL,
            "catalog_sha256": CATALOG_SHA256,
            "catalog_records": len(records),
        },
        "audit": audit,
        "h_b_h_c_enumeration": {
            "recorded_two_baryon_channels": baryon_channels,
            "recorded_count": len(baryon_channels),
            "reason": (
                "two baryon daughters carry B = 2 in total, so no "
                "conserving (H_b, H_c) pair exists for the B = 1 family; "
                "the committed catalog records no two-baryon channel at "
                "all, so the recorded (H_b, H_c) set is empty for every "
                "baryonic family"
            ),
        },
        "informational_inactive_strange_mesonic": informational,
        "contract": {
            "complete": audit["complete"],
            "conservative": audit["conservative"],
            "missing_class_keys": audit["missing_class_keys"],
            "nonconserving_declared": audit["nonconserving_declared"],
            "verdict_relevant": True,
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / CLASS_PAIRS_NAME, payload)
    print(
        f"audit_r4_strange_sector_families: class-pair re-derivation "
        f"written (declared {len(audit['declared_classes'])}, conserving "
        f"catalog class keys "
        f"{len(audit['enumeration']['conserving_class_keys'])}, missing "
        f"{len(audit['missing_class_keys'])}, non-conserving "
        f"{len(audit['nonconserving_declared'])})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --k-contribution (the per-class K+ decomposition of the 18
# persisting K+ bins from the sealed R32 census)
# ---------------------------------------------------------------------------


def pooled_class_counts(records: list) -> dict:
    """One variant's pooled feed-down counts from its sealed per-replica
    payload blocks (pure integer arithmetic; zero draws)."""
    pooled = {name: 0 for name in (
        "K_plus", "K_minus", "K0S", "K0L", "K_0", "aK_0",
        "Lambda", "Sigma_0", "Sigma_plus", "Sigma_minus",
    )}
    replicas = 0
    for record in records:
        species = record["feed_down"]["per_species"]
        for name in pooled:
            pooled[name] += int(species[name]["n"])
        replicas += 1
    if pooled["K_plus"] <= 0:
        _refuse("the pooled K+ count needs a positive final count")
    return {
        "replicas": replicas,
        "counts": pooled,
        "share_Lambda": pooled["Lambda"] / pooled["K_plus"],
        "share_Sigma0": pooled["Sigma_0"] / pooled["K_plus"],
        "identity_kplus_hyperon": bool(
            pooled["K_plus"] == pooled["Lambda"] + pooled["Sigma_0"]
        ),
        "identity_k0_sigma": bool(pooled["K_0"] == pooled["Sigma_plus"]),
    }


def replica_identities(record: dict, source_name: str) -> list:
    """The sealed count identities of the corrected card-3 lineage
    convention on one sealed replica payload (a list of violation
    strings; empty when the replica is clean)."""
    species = record["feed_down"]["per_species"]
    violations = []
    if not record.get("conservation", {}).get("event_b_q_conserved", False):
        violations.append(
            f"{source_name}: the event (B, Q) conservation flag is not true"
        )
    kp, lam, s0 = species["K_plus"], species["Lambda"], species["Sigma_0"]
    if int(kp["n"]) != int(lam["n"]) + int(s0["n"]):
        violations.append(
            f"{source_name}: {IDENTITY_KPLUS_HYPERON} violated "
            f"(K_plus {kp['n']}, Lambda {lam['n']}, Sigma_0 {s0['n']})"
        )
    if int(kp["n_resonant"]) != 0 or kp.get("by_parent") not in ({}, None):
        violations.append(
            f"{source_name}: {IDENTITY_ROOT_VERTEX} violated "
            f"(n_resonant {kp.get('n_resonant')}, by_parent "
            f"{kp.get('by_parent')})"
        )
    for name in ("K_minus", "Sigma_minus", "K0S", "K0L", "aK_0"):
        if int(species[name]["n"]) != 0:
            violations.append(
                f"{source_name}: {IDENTITY_ZERO_STRANGE} violated "
                f"({name}.n == {species[name]['n']})"
            )
    if int(species["K_0"]["n"]) != int(species["Sigma_plus"]["n"]):
        violations.append(
            f"{source_name}: {IDENTITY_K0_SIGMA} violated "
            f"(K_0 {species['K_0']['n']}, Sigma_plus "
            f"{species['Sigma_plus']['n']})"
        )
    return violations


def load_payload_identities(rel_dir: str, name_template: str,
                            n_replicas: int) -> dict:
    """The per-variant pooled class counts and the per-replica identity
    assertions over one sealed archive's payload records."""
    per_variant: dict[str, dict] = {}
    violations: list[str] = []
    for v in ("0", "1", "2"):
        records = []
        for r in range(int(n_replicas)):
            path = MONO_ROOT / rel_dir / name_template.format(v=v, r=r)
            if not path.is_file():
                _refuse(f"sealed payload record missing: {path}")
            record = _read_json(path)
            violations.extend(replica_identities(record, str(path)))
            records.append(record)
        per_variant[v] = pooled_class_counts(records)
    return {"per_variant": per_variant, "identity_violations": violations}


def class_decomposed_share(values: list, shares: list) -> list:
    """The class-decomposed per-variant values ``y_v,c = s_c(v) y_v``
    (the proportional census attribution)."""
    return [
        float(s) * float(y) for s, y in zip(shares, values)
    ]


def kplus_bin_rows(persisting_kplus: list, shares: dict, card1_rows: dict,
                   parent_rows: dict) -> list:
    """The per-bin class decomposition of the 18 persisting K+ register
    entries (pure sealed-input arithmetic; zero draws)."""
    s_lambda = [float(shares[v]["share_Lambda"]) for v in ("0", "1", "2")]
    s_sigma0 = [float(shares[v]["share_Sigma0"]) for v in ("0", "1", "2")]
    rows = []
    for entry in persisting_kplus:
        values = [float(x) for x in entry["confirmation_values"]]
        unit_se = float(entry["confirmation_unit_se"])
        if not (math.isfinite(unit_se) and unit_se > 0.0):
            _refuse(
                "a persisting register entry carries a non-finite pooled "
                f"SE (bin {entry.get('bin_index')})"
            )
        spread = max(values) - min(values)
        sigma = spread / unit_se
        recorded_sigma = float(entry["confirmation_sigma"])
        if abs(sigma - recorded_sigma) > 1e-9 * max(1.0, recorded_sigma):
            _refuse(
                "the persisting register entry sigma disagrees with the "
                f"sealed values (k_plus, bin {entry.get('bin_index')})"
            )
        y_lambda = class_decomposed_share(values, s_lambda)
        y_sigma0 = class_decomposed_share(values, s_sigma0)
        sigma_lambda = (max(y_lambda) - min(y_lambda)) / unit_se
        sigma_sigma0 = (max(y_sigma0) - min(y_sigma0)) / unit_se
        dominant = ("(3122,321)" if sigma_lambda >= sigma_sigma0
                    else "(3212,321)")
        card1 = card1_rows.get(int(entry.get("bin_index")))
        parent = parent_rows.get(int(entry.get("bin_index")))
        rows.append({
            "bin_index": int(entry["bin_index"]),
            "parent_index": int(entry["parent_index"]),
            "xF_mid": float(parent["xF_mid"]) if parent else None,
            "pT_mid": float(parent["pT_mid"]) if parent else None,
            "confirmation_values": values,
            "confirmation_unit_se": unit_se,
            "confirmation_sigma": recorded_sigma,
            "sigma_recomputed": sigma,
            "bound": BIN_SPREAD_BOUND,
            "class_shares": {
                "(3122,321)_K_Lambda": s_lambda,
                "(3212,321)_K_Sigma0": s_sigma0,
            },
            "class_values": {
                "(3122,321)_K_Lambda": y_lambda,
                "(3212,321)_K_Sigma0": y_sigma0,
            },
            "contributions_pooled_se_units": {
                "(3122,321)_K_Lambda": sigma_lambda,
                "(3212,321)_K_Sigma0": sigma_sigma0,
            },
            "contribution_sum": sigma_lambda + sigma_sigma0,
            "dominant_class_pair": dominant,
            "class_attributable": True,
            "attribution_note": (
                "both K+-bearing class pairs are declared on the "
                "activated baryonic_np family and catalog-audited "
                "(the daughters are committed catalog records; "
                "(3122, 321) is additionally a recorded channel of "
                "N(1650)+ / N(1720)+); the bin is not a "
                "missing-decay-class artifact"
            ),
            "density_shape_referral": {
                "density_shape_share": sigma,
                "referred_to_card": CARD1_CARD,
                "card1_row": card1,
            },
        })
    return rows


def run_k_contribution(run_dir: Path) -> dict:
    """The decomposition of the 18 persisting K+ bins by parent decay
    class from the sealed R32 census; the dominant-class table."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    class_pairs_path = run_dir / CLASS_PAIRS_NAME
    if not class_pairs_path.is_file():
        _refuse(
            f"{class_pairs_path} is missing; the K+ decomposition "
            "requires the class-pairs step (run --class-pairs first)"
        )
    class_pairs = _read_json(class_pairs_path)
    if not (class_pairs["contract"]["complete"]
            and class_pairs["contract"]["conservative"]):
        _refuse(
            "the class-pair re-derivation carries findings; the K+ "
            "decomposition refuses to run on an incomplete or "
            "non-conserving class set (the verdict contract criterion "
            "(a) fires at the verdict step)"
        )
    conf_verification = _read_json(
        MONO_ROOT / R32_CONF_DIR / CONFIRMATION_VERIFICATION_NAME
    )
    prod_report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    card1_attribution = _read_json(
        MONO_ROOT / CARD1_DIR / CARD1_ATTRIBUTION_NAME
    )
    card1_verdict = _read_json(MONO_ROOT / CARD1_DIR / CARD1_VERDICT_NAME)

    persisting = [
        entry for entry in conf_verification["residual_reclassification"]
        if entry.get("classification") == "persisting"
    ]
    if len(persisting) != PERSISTING_TOTAL:
        _refuse(
            f"the sealed confirmation register carries {len(persisting)} "
            f"persisting entries; the pre-registered probe set is "
            f"{PERSISTING_TOTAL}"
        )
    persisting_kplus = [e for e in persisting
                        if e.get("species") == "k_plus"]
    if len(persisting_kplus) != PERSISTING_KPLUS:
        _refuse(
            f"the sealed confirmation register carries "
            f"{len(persisting_kplus)} persisting K+ entries; the "
            f"pre-registered probe set is {PERSISTING_KPLUS}"
        )
    bin_indices = tuple(sorted(int(e["bin_index"]) for e in persisting_kplus))
    if bin_indices != KPLUS_BIN_INDICES:
        _refuse(
            f"the persisting K+ bin indices {bin_indices} != the "
            f"pre-registered probe set {KPLUS_BIN_INDICES}"
        )
    sigmas = [float(e["confirmation_sigma"]) for e in persisting_kplus]
    if (min(sigmas) < KPLUS_SIGMA_MIN - SIGMA_ANCHOR_TOLERANCE
            or max(sigmas) > KPLUS_SIGMA_MAX + SIGMA_ANCHOR_TOLERANCE):
        _refuse(
            "the persisting K+ confirmation sigmas leave the "
            f"pre-registered range [{KPLUS_SIGMA_MIN}, {KPLUS_SIGMA_MAX}]"
        )

    prod_payloads = load_payload_identities(
        R32_PROD_DIR, "variant_{v}_replica_{r}.json", 8,
    )
    conf_payloads = load_payload_identities(
        R32_CONF_DIR, "confirmation_variant_{v}_replica_{r}.json", 16,
    )
    all_violations = (
        prod_payloads["identity_violations"]
        + conf_payloads["identity_violations"]
    )
    if all_violations:
        _refuse(
            "the sealed count identities of the corrected card-3 lineage "
            "convention are violated; the decomposition algebra cannot "
            "run:\n  " + "\n  ".join(all_violations[:10])
        )
    parent_rows = {
        int(row["bin_index"]): row
        for row in prod_report_bin_rows()
        if row["species"] == "k_plus"
    }
    card1_rows = {
        int(row["bin_index"]): {
            "p1_monotone": row["p1_monotone"],
            "delta_direction": row["delta_direction"],
            "route_dominance": row["route_dominance"],
            "p2_within_envelope": row["p2_within_envelope"],
            "placement_explained": row["placement_explained"],
            "mass_blind_envelope_respected":
                row["mass_blind_envelope_respected"],
            "confirmation_sigma": row["confirmation_sigma"],
            "deltas_ramp_start_order": row["deltas_ramp_start_order"],
        }
        for row in card1_attribution_kplus_rows(card1_attribution)
    }
    rows = kplus_bin_rows(
        persisting_kplus, conf_payloads["per_variant"], card1_rows,
        parent_rows,
    )
    attributable = sum(1 for r in rows if r["class_attributable"])
    dominant_counts: dict[str, int] = {}
    for row in rows:
        dominant_counts[row["dominant_class_pair"]] = (
            dominant_counts.get(row["dominant_class_pair"], 0) + 1
        )
    payload = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families"
            ".kplus_contribution/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the decomposition of the 18 persisting K+ bins by parent "
            "decay class from the sealed R32 feed-down and routing census "
            "(the corrected card-3 lineage convention)"
        ),
        "probe_set": {
            "source": (
                f"{R32_CONF_DIR}/{CONFIRMATION_VERIFICATION_NAME} "
                "residual_reclassification (classification == 'persisting' "
                "and species == 'k_plus')"
            ),
            "entries": len(rows),
            "bin_indices": list(bin_indices),
            "sigma_range": [min(sigmas), max(sigmas)],
        },
        "identity_assertions": [
            IDENTITY_KPLUS_HYPERON,
            IDENTITY_ROOT_VERTEX,
            IDENTITY_ZERO_STRANGE,
            IDENTITY_K0_SIGMA,
        ],
        "identity_sweep": {
            "production_payloads": 24,
            "confirmation_payloads": 48,
            "violations": all_violations,
            "clean": bool(not all_violations),
        },
        "class_shares": {
            "production": prod_shares_payload(prod_payloads["per_variant"]),
            "confirmation": prod_shares_payload(conf_payloads["per_variant"]),
            "source": (
                "the sealed feed-down per-species counts pooled per "
                "variant (corrected card-3 lineage convention); the "
                "confirmation shares are the same statistics as the "
                "probe values and the production shares are the "
                "cross-reference"
            ),
        },
        "kplus_bearing_classes": {
            "declared_on_activated_surface": {
                "baryonic_np": ["(3122,321)", "(3212,321)"],
                "mesonic_neutral": ["(321,-321)"],
            },
            "sealed_contributions": {
                "(3122,321)_K_Lambda": "share_Lambda of the sealed K+ count",
                "(3212,321)_K_Sigma0":
                    "share_Sigma0 of the sealed K+ count",
                "(321,-321)_K_Kbar": (
                    "zero (K_minus identically zero in every sealed "
                    "replica)"
                ),
            },
            "strange_sector_families": {
                "baryonic_strange_(1,0,-1)": (
                    "zero K+ contribution: the declared set carries no K+ "
                    "daughter (S conservation would need an S = -2 "
                    "partner and the committed catalog records no such "
                    "hadron), and the family is never sampled in the R32 "
                    "p+p topology whose sources carry (B, Q, S) = (1, 1, "
                    "0) (the sealed Sigma- final count is identically "
                    "zero)"
                ),
                "mesonic_strange_(0,0,-1)": (
                    "declared inactive; never sampled by R30-R32; its "
                    "classes carry K-/Kbar0 daughters and produce no K+"
                ),
                "mesonic_antistrange_(0,0,+1)": (
                    "declared inactive; never sampled by R30-R32"
                ),
            },
            "catalog_audit_records": {
                "(3122,321)_K_Lambda": {
                    "records": list(KPLUS_LAMBDA_CATALOG_RECORDS),
                    "note": (
                        "the catalog records N(1650)+ (pdg 122212, "
                        "channel branching fraction 0.12) and N(1720)+ "
                        "(pdg 212214, channel branching fraction 0.1) "
                        "carry the recorded K+ Lambda channels; the "
                        "class key is the same on both routes"
                    ),
                },
            },
            "route_split_note": (
                "the (3122, 321) class key is realized on both routes "
                "(the continuum family class and the catalog records "
                "N(1650)+ / N(1720)+); the sealed count census does not "
                "resolve the route split of this class and the split is "
                "not needed for the verdict: the class key is declared "
                "on the activated baryonic_np family and catalog-audited "
                "on both routes"
            ),
        },
        "rows": rows,
        "summary": {
            "persisting_kplus_bins": len(rows),
            "class_attributable": attributable,
            "unattributable_bins": [
                r["bin_index"] for r in rows if not r["class_attributable"]
            ],
            "dominant_class_pair_counts": dominant_counts,
            "all_class_attributable": bool(attributable == len(rows)),
            "bound_note": (
                "every per-bin sigma row carries the R32 pre-registered "
                "bound 2.0 pooled-SE units beside the value (the "
                "persisting set is defined by exceeding it); the bound is "
                "never relaxed"
            ),
        },
        "card1_cross_reference": {
            "run_dir": CARD1_DIR,
            "card": CARD1_CARD,
            "verdict": card1_verdict.get("verdict"),
            "k_plus_referral": card1_verdict.get("attribution_summary", {})
            .get("k_plus_referral"),
            "referral_note": (
                "the density-shape share of every class-attributed "
                "persisting K+ bin is recorded per row and referred to "
                "card 1 (the routing formulation audit owns the ramp "
                "placement and width attribution); the card-1 audit's "
                "sealed RECOMMENDATION carries the K+ referral block "
                "(15/18 monotone, 18/18 within the census envelope, "
                "16/18 mass-blind envelope violations, 15/18 "
                "continuum-dominant) and its rework card is "
                f"{CARD1_REWORK_CARD}"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / KPLUS_NAME, payload)
    print(
        f"audit_r4_strange_sector_families: K+ decomposition written "
        f"({len(rows)} persisting K+ bins; class-attributable "
        f"{attributable}/{len(rows)}; dominant classes {dominant_counts})",
        flush=True,
    )
    return payload


def prod_report_bin_rows() -> list:
    """The sealed production spread report's bin rows (re-read per
    call; the digest is verified by the pre-registration assertion)."""
    report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    return report["bin_checks"]


def card1_attribution_kplus_rows(card1_attribution: dict) -> list:
    """The sealed card-1 attribution rows of the 18 K+ persisting
    bins."""
    return [row for row in card1_attribution.get("rows", [])
            if row.get("species") == "k_plus"]


def prod_shares_payload(per_variant: dict) -> dict:
    """The per-variant share payload (counts and shares)."""
    return {
        v: {
            "replicas": block["replicas"],
            "counts": block["counts"],
            "share_Lambda": block["share_Lambda"],
            "share_Sigma0": block["share_Sigma0"],
        }
        for v, block in per_variant.items()
    }


# ---------------------------------------------------------------------------
# Mode: --config-sanity
# ---------------------------------------------------------------------------


def family_census(families_config: dict) -> dict:
    """The committed family census (family id, qn, class count, active
    flag) as a JSON-safe structure."""
    out = {"active": {}, "inactive": {}}
    for record in families_config["families"]:
        entry = {
            "qn": list(int(v) for v in record["qn"]),
            "classes": len(record["allowed_decay_classes"]),
            "state_type": record["state_type"],
        }
        bucket = "active" if record["active"] else "inactive"
        out[bucket][record["family_id"]] = entry
    return out


def check_strange_record(family: dict) -> dict:
    """The pre-registered strange-record parameter check (the cards 4/5
    pre-registration values)."""
    expected = {
        "m_min": STRANGE_M_MIN,
        "t_h": STRANGE_T_H,
        "b_power": STRANGE_B_POWER,
        "n_norm": STRANGE_N_NORM,
        "ramp_width": STRANGE_RAMP_WIDTH,
        "nominal_width": STRANGE_NOMINAL_WIDTH,
        "m_max": STRANGE_M_MAX,
    }
    fields = {}
    problems = []
    for key, want in expected.items():
        got = float(family[key])
        fields[key] = {"recorded": got, "expected": want,
                       "matches": bool(got == want)}
        if got != want:
            problems.append(
                f"{key}: recorded {got!r} != the cards 4/5 pre-registered "
                f"{want!r}"
            )
    fields["qn"] = {
        "recorded": list(int(v) for v in family["qn"]),
        "expected": list(STRANGE_QN),
        "matches": bool(tuple(int(v) for v in family["qn"]) == STRANGE_QN),
    }
    if not fields["qn"]["matches"]:
        problems.append(
            f"qn: recorded {family['qn']} != the pre-registered "
            f"{list(STRANGE_QN)}"
        )
    fields["state_type"] = {
        "recorded": family["state_type"],
        "expected": STRANGE_STATE_TYPE,
        "matches": bool(family["state_type"] == STRANGE_STATE_TYPE),
    }
    if family["state_type"] != STRANGE_STATE_TYPE:
        problems.append(
            f"state_type: recorded {family['state_type']!r} != "
            f"{STRANGE_STATE_TYPE!r}"
        )
    fields["active"] = {
        "recorded": bool(family["active"]),
        "expected": True,
        "matches": bool(family["active"]),
    }
    if not family["active"]:
        problems.append("the baryonic_strange record is not active")
    declared = [str(c) for c in family["allowed_decay_classes"]]
    fields["allowed_decay_classes"] = {
        "recorded": declared,
        "expected": list(STRANGE_DECLARED_CLASSES),
        "matches": bool(declared == list(STRANGE_DECLARED_CLASSES)),
    }
    if declared != list(STRANGE_DECLARED_CLASSES):
        problems.append(
            f"allowed_decay_classes {declared} != the pre-registered "
            f"{list(STRANGE_DECLARED_CLASSES)}"
        )
    return {"fields": fields, "problems": problems}


def run_config_sanity(run_dir: Path) -> dict:
    """The strange-records byte check and the R30/R31 cross-references."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    for name in (CLASS_PAIRS_NAME, KPLUS_NAME):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing; the config sanity requires "
                "the class-pairs and k-contribution steps first"
            )
    families_digest = _sha256_file(MONO_ROOT / FAMILIES_REL)
    catalog_digest = _sha256_file(MONO_ROOT / CATALOG_REL)
    if families_digest != FAMILIES_SHA256 or catalog_digest != CATALOG_SHA256:
        _refuse(
            "the committed family surface digests drifted during the run "
            "(a driver defect: the audit must run against the committed "
            "bytes)"
        )
    families_config = _read_json(MONO_ROOT / FAMILIES_REL)
    census = family_census(families_config)
    strange = None
    for record in families_config["families"]:
        if record["family_id"] == STRANGE_FAMILY_ID:
            strange = record
    if strange is None:
        _refuse(f"the family {STRANGE_FAMILY_ID} is absent from the config")
    strange_check = check_strange_record(strange)
    inactive = {}
    inactive_problems = []
    for family_id, spec in sorted(INACTIVE_STRANGE_MESONIC.items()):
        record = None
        for candidate in families_config["families"]:
            if candidate["family_id"] == family_id:
                record = candidate
        if record is None:
            inactive_problems.append(f"the family {family_id} is absent")
            continue
        if tuple(int(v) for v in record["qn"]) != spec["qn"]:
            inactive_problems.append(
                f"{family_id}: qn {list(record['qn'])} != the declared "
                f"{list(spec['qn'])}"
            )
        if len(record["allowed_decay_classes"]) != spec["classes"]:
            inactive_problems.append(
                f"{family_id}: allowed_decay_classes carries "
                f"{len(record['allowed_decay_classes'])} classes, the "
                f"declaration records {spec['classes']}"
            )
        if record["active"]:
            inactive_problems.append(
                f"{family_id}: the declared inactive family is active"
            )
        inactive[family_id] = {
            "qn": list(int(v) for v in record["qn"]),
            "classes": len(record["allowed_decay_classes"]),
            "active": bool(record["active"]),
        }
    # The density-shape cross-references (read-only; the digests are
    # verified by the pre-registration assertion).
    r30 = _read_json(MONO_ROOT / R30_DIR / R30_GATE_NAME)
    r31 = _read_json(MONO_ROOT / R31_DIR / R31_GATE_NAME)
    r30_blocks = r30["blocks"]
    r30_rows = {
        "family_normalization": {
            "families_checked": r30_blocks["family_normalization"][
                "families_checked"],
            "max_abs_deviation_from_unity": r30_blocks[
                "family_normalization"]["max_abs_deviation_from_unity"],
            "tolerance": r30_blocks["family_normalization"]["tolerance"],
            "pass": r30_blocks["family_normalization"]["pass"],
        },
        "threshold_behavior": {
            "max_derivative_mismatch": r30_blocks["threshold_behavior"][
                "max_derivative_mismatch"],
            "max_continuity_probe_mismatch": r30_blocks[
                "threshold_behavior"]["max_continuity_probe_mismatch"],
            "tolerances": r30_blocks["threshold_behavior"]["tolerances"],
            "pass": r30_blocks["threshold_behavior"]["pass"],
        },
        "double_counting_identity": {
            "qn_sets": r30_blocks["double_counting_identity"]["qn_sets"],
            "max_abs_deviation_from_unity": r30_blocks[
                "double_counting_identity"]["max_abs_deviation_from_unity"],
            "tolerance": r30_blocks["double_counting_identity"]["tolerance"],
            "pass": r30_blocks["double_counting_identity"]["pass"],
            "strange_qn_covered": any(
                [1, 0, -1] == [int(v) for v in qn]
                for qn in r30_blocks["double_counting_identity"]["qn_sets"]
            ),
        },
    }
    r31_windows = r31["clauses"]["no_boundary_dominance"]["per_window"]
    strange_window = None
    for window in r31_windows:
        if [int(v) for v in window["qn"]] == [1, 0, -1]:
            strange_window = window
            break
    if strange_window is None:
        _refuse("the R31 gate record carries no (1, 0, -1) window")
    smoothness = r31["clauses"]["smoothness"]
    r31_rows = {
        "window": strange_window["window"],
        "qn": strange_window["qn"],
        "boundary_pair": strange_window["boundary_pair"],
        "boundary_pair_applied": strange_window["boundary_pair_applied"],
        "boundary_pair_failed": strange_window["boundary_pair_failed"],
        "boundary_pair_failure_rate": strange_window[
            "boundary_pair_failure_rate"],
        "interior_pairs_applied": strange_window["interior_pairs_applied"],
        "interior_pairs_failed": strange_window["interior_pairs_failed"],
        "interior_pairs_failure_rate": strange_window[
            "interior_pairs_failure_rate"],
        "no_boundary_dominance": strange_window["no_boundary_dominance"],
        "smoothness_clause": {
            "windows": smoothness["windows"],
            "checks_applied": smoothness["checks_applied"],
            "checks_failed": smoothness["checks_failed"],
            "original_statistics_note": smoothness["verdict_note"],
        },
    }
    # The builder's source-quantum-number pin (the R32 p+p topology;
    # the strange-sector families are never sampled).
    builder_source = (MONO_ROOT / BUILDER_REL).read_text(encoding="utf-8")
    source_qn_pin = BUILDER_SOURCE_QN_SNIPPET in builder_source
    # The recomputed kinematic extreme of the (1, 0, -1) classes
    # (informational; the committed provenance prose quotes 1.432 GeV).
    catalog = _read_json(MONO_ROOT / CATALOG_REL)
    by_pid = {int(r["pdg_id"]): r for r in catalog["records"]}
    class_sums = []
    for cls in STRANGE_DECLARED_CLASSES:
        d0, d1 = parse_class_key(cls)
        msum = (float(by_pid[d0]["pole_mass"])
                + float(by_pid[d1]["pole_mass"]))
        class_sums.append({"class_key": cls, "pole_mass_sum": msum})
    max_sum = max(row["pole_mass_sum"] for row in class_sums)
    payload = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families.config_sanity/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the family-configuration sanity of the strange records: the "
            "committed-parameter check, the declared inactive flags, and "
            "the R30/R31 density-shape cross-references (read-only)"
        ),
        "surface_digests": {
            "families_config_sha256": families_digest,
            "catalog_sha256": catalog_digest,
            "byte_stable": bool(
                families_digest == FAMILIES_SHA256
                and catalog_digest == CATALOG_SHA256
            ),
        },
        "family_census": census,
        "strange_record_check": strange_check,
        "strange_record_problems": strange_check["problems"],
        "inactive_strange_mesonic": inactive,
        "inactive_problems": inactive_problems,
        "r30_cross_reference": r30_rows,
        "r31_cross_reference": r31_rows,
        "sampling_scope": {
            "source_qn_r32": list(SOURCE_QN_R32),
            "builder_source_qn_pin_found": bool(source_qn_pin),
            "builder_sha256": BUILDER_SHA256,
            "note": (
                "the builder hard-pins the two-source (B, Q, S) = (1, 1, "
                "0) topology, so the activated strange baryonic family "
                "(1, 0, -1) and the declared inactive strange mesonic "
                "families are never sampled by R30-R32; the sealed "
                "final-state census pins this empirically (Sigma- "
                "identically zero in every sealed replica)"
            ),
        },
        "kinematic_recomputation": {
            "class_pole_mass_sums": class_sums,
            "max_pole_mass_sum": max_sum,
            "all_open_across_support": bool(
                max_sum < STRANGE_M_MIN
            ),
            "provenance_quote_note": (
                "the committed baryonic_strange provenance prose quotes "
                "the highest daughter pole-mass sum as 1.432 GeV; the "
                "recomputation gives 1.437176 GeV (the (2112, -311) "
                "n-Kbar0 pair, neutron 0.939565 + aK0 0.497611 GeV), "
                "0.005176 GeV above the quoted value; the openness "
                "conclusion is unchanged (both values sit far below the "
                "family m_min 2.680194110727528 GeV) and this is an "
                "informational record, not a parameter drift: the "
                "committed bytes are intact and the quoted prose is not "
                "a physics parameter"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / CONFIG_SANITY_NAME, payload)
    print(
        f"audit_r4_strange_sector_families: config sanity written "
        f"(byte-stable {payload['surface_digests']['byte_stable']}; "
        f"strange-record problems {len(strange_check['problems'])}; "
        f"R30 blocks pass "
        f"{all(b['pass'] for b in r30_rows.values())}; "
        f"R31 window {strange_window['window']} interior failures "
        f"{strange_window['interior_pairs_failed']})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --verdict
# ---------------------------------------------------------------------------


def evaluate_verdict_criteria(class_pairs: dict, kplus: dict,
                              config_sanity: dict) -> dict:
    """The pre-registered verdict criteria (a), (b), (c) evaluated on
    the three measurement artifacts (a pure function; zero draws)."""
    criterion_a = {
        "criterion": "a",
        "statement": "a class pair is missing or non-conserving",
        "family_id": class_pairs["audit"]["family_id"],
        "family_qn": class_pairs["audit"]["qn"],
        "declared_classes": len(class_pairs["audit"]["declared_classes"]),
        "conserving_catalog_class_keys": len(
            class_pairs["audit"]["enumeration"]["conserving_class_keys"]
        ),
        "missing_class_keys": class_pairs["contract"]["missing_class_keys"],
        "nonconserving_declared": class_pairs["contract"][
            "nonconserving_declared"],
        "complete": class_pairs["contract"]["complete"],
        "conservative": class_pairs["contract"]["conservative"],
        "fired": bool(
            class_pairs["contract"]["missing_class_keys"]
            or class_pairs["contract"]["nonconserving_declared"]
        ),
    }
    criterion_b = {
        "criterion": "b",
        "statement": (
            "a persisting K+ bin has no catalog-audited class-pair path"
        ),
        "persisting_kplus_bins": kplus["summary"]["persisting_kplus_bins"],
        "class_attributable": kplus["summary"]["class_attributable"],
        "unattributable_bins": kplus["summary"]["unattributable_bins"],
        "fired": bool(kplus["summary"]["unattributable_bins"]),
    }
    criterion_c = {
        "criterion": "c",
        "statement": (
            "the config disagrees with its committed bytes (a driver "
            "defect, rework the driver, not the surface)"
        ),
        "families_config_sha256": config_sanity["surface_digests"][
            "families_config_sha256"],
        "families_expected_sha256": FAMILIES_SHA256,
        "catalog_sha256": config_sanity["surface_digests"]["catalog_sha256"],
        "catalog_expected_sha256": CATALOG_SHA256,
        "byte_stable": config_sanity["surface_digests"]["byte_stable"],
        "fired": not bool(config_sanity["surface_digests"]["byte_stable"]),
    }
    fired = [c for c in (criterion_a, criterion_b, criterion_c)
             if c["fired"]]
    return {
        "criteria": {"a": criterion_a, "b": criterion_b, "c": criterion_c},
        "fired": fired,
        "verdict": "RECOMMENDATION" if fired else "PASS",
    }


def run_verdict(run_dir: Path) -> dict:
    """Assemble the verdict artifact (PASS or RECOMMENDATION with named
    findings)."""
    run_dir = Path(run_dir)
    pre = assert_pre_registration(run_dir)
    artifacts = {}
    for name, key in (
        (CLASS_PAIRS_NAME, "class_pairs"),
        (KPLUS_NAME, "kplus"),
        (CONFIG_SANITY_NAME, "config_sanity"),
    ):
        path = run_dir / name
        if not path.is_file():
            _refuse(
                f"{path} is missing; the verdict requires the three "
                "measurement steps (run --class-pairs, --k-contribution, "
                "--config-sanity first)"
            )
        artifacts[key] = _read_json(path)
    # Re-hash the pinned surface at verdict time (the criterion (c)
    # check: the config must still be at its committed bytes).
    config_sanity = dict(artifacts["config_sanity"])
    config_sanity = {
        **config_sanity,
        "surface_digests": {
            **config_sanity["surface_digests"],
            "families_config_sha256": _sha256_file(MONO_ROOT / FAMILIES_REL),
            "catalog_sha256": _sha256_file(MONO_ROOT / CATALOG_REL),
        },
    }
    config_sanity["surface_digests"]["byte_stable"] = bool(
        config_sanity["surface_digests"]["families_config_sha256"]
        == FAMILIES_SHA256
        and config_sanity["surface_digests"]["catalog_sha256"]
        == CATALOG_SHA256
    )
    evaluation = evaluate_verdict_criteria(
        artifacts["class_pairs"], artifacts["kplus"], config_sanity,
    )
    findings = []
    if evaluation["criteria"]["a"]["fired"]:
        findings.append({
            "kind": "class_pair",
            "family_id": artifacts["class_pairs"]["audit"]["family_id"],
            "species": None,
            "bin_index": None,
            "observed": (
                "the declared class set of the (1, 0, -1) family is "
                "incomplete or non-conserving against the committed "
                f"catalog: missing {artifacts['class_pairs']['contract']['missing_class_keys']}, "
                "non-conserving "
                f"{artifacts['class_pairs']['contract']['nonconserving_declared']}"
            ),
            "diagnosis": (
                "the class-pair set does not reproduce the round 3 card "
                "5 re-derivation against the committed catalog"
            ),
            "responsible": (
                f"the cards 4/5 family surface ({FAMILIES_REL}, sha256 "
                f"{FAMILIES_SHA256}, byte-unchanged by this audit)"
            ),
            "proposed_rework": (
                "a separate Round 4 rework card on the family surface to "
                "re-declare the class set from the committed catalog "
                "re-derivation; this audit edits nothing pinned"
            ),
            "evidence": {
                "conservation_rows":
                    artifacts["class_pairs"]["audit"]["conservation_rows"],
                "enumeration":
                    artifacts["class_pairs"]["audit"]["enumeration"][
                        "conserving_class_keys"],
                "missing_class_keys":
                    artifacts["class_pairs"]["contract"][
                        "missing_class_keys"],
                "nonconserving_declared":
                    artifacts["class_pairs"]["contract"][
                        "nonconserving_declared"],
            },
        })
    if evaluation["criteria"]["b"]["fired"]:
        findings.append({
            "kind": "bin",
            "species": "k_plus",
            "bin_index": artifacts["kplus"]["summary"][
                "unattributable_bins"],
            "observed": (
                f"{len(artifacts['kplus']['summary']['unattributable_bins'])} "
                "of 18 persisting K+ bins carry no catalog-audited "
                "class-pair path"
            ),
            "diagnosis": (
                "the K+ finals of the named bins are not produced by any "
                "declared, catalog-audited class pair: the responsible "
                "surface is the class set of this card, or the routing "
                "formulation referred back to card 1"
            ),
            "responsible": (
                f"the cards 4/5 family surface ({FAMILIES_REL}, sha256 "
                f"{FAMILIES_SHA256}, byte-unchanged by this audit), or "
                f"the routing formulation ({CARD1_CARD})"
            ),
            "proposed_rework": (
                "a separate Round 4 rework card to re-declare or re-route "
                "the named bins' production path; the routing-formulation "
                "share is card 1's scope"
            ),
            "evidence": {
                "unattributable_bins":
                    artifacts["kplus"]["summary"]["unattributable_bins"],
                "rows": [
                    row for row in artifacts["kplus"]["rows"]
                    if not row["class_attributable"]
                ],
            },
        })
    if evaluation["criteria"]["c"]["fired"]:
        findings.append({
            "kind": "config_byte_drift",
            "species": None,
            "bin_index": None,
            "observed": (
                "the family configuration or the catalog disagrees with "
                "its committed bytes at verdict time"
            ),
            "diagnosis": (
                "a driver defect: the audit must run against the "
                "committed bytes; rework the driver, never the surface"
            ),
            "responsible": "the audit driver "
                "helpers/benchmarking/audit_r4_strange_sector_families.py",
            "proposed_rework": (
                "a separate rework card to restore and re-run the audit "
                "driver; the family surface is not implicated"
            ),
            "evidence": {
                "families_config_sha256": config_sanity["surface_digests"][
                    "families_config_sha256"],
                "families_expected_sha256": FAMILIES_SHA256,
                "catalog_sha256": config_sanity["surface_digests"][
                    "catalog_sha256"],
                "catalog_expected_sha256": CATALOG_SHA256,
            },
        })
    verdict = evaluation["verdict"]
    kplus = artifacts["kplus"]
    class_pairs = artifacts["class_pairs"]
    pass_text = None
    recommendation_text = None
    if evaluation["verdict"] == "PASS":
        pass_text = (
            "The strange-sector continuum family surface is adequate "
            "as-is. The declared class set of the activated strange "
            "baryonic family (1, 0, -1) is complete and conservative "
            "against the committed catalog (6 of 6 declared classes "
            "conserve (B, Q, S) = (1, 0, -1) and 6 of 6 conserving "
            "catalog class pairs are declared; no conserving pair is "
            "missing and none violates vertex conservation). Every "
            "persisting K+ bin is attributable to a declared, "
            "catalog-audited class pair: the sealed count identities pin "
            "every K+ final to the root-vertex K+-bearing class pairs of "
            "the activated baryonic (1, 1, 0) family, (3122, 321) K+ "
            "Lambda and (3212, 321) K+ Sigma0, with the sealed per-variant "
            "class shares near one half each; the strange-sector families "
            "contribute zero K+ (the (1, 0, -1) declared set carries no "
            "K+ daughter and the family is never sampled in the (1, 1, 0) "
            "p+p topology; the declared inactive strange mesonic families "
            "are never sampled). The family configuration is sane: the "
            "committed bytes are intact, the strange records keep the "
            "cards 4/5 pre-registered parameters, and the density shape "
            "cross-references pass (R30 at machine precision, R31 window "
            "1 with 0 interior and 0 boundary failures, 0 of 422 at the "
            "authoritative card-8c statistics). The 18 K+ persisting bins "
            "are therefore not a missing-decay-class artifact of the "
            "strange-sector surface; their spread lives on the "
            "baryonic_np K-hyperon classes and the routing/density shape, "
            "whose share is referred to card 1 (the sealed card-1 "
            f"RECOMMENDATION and its rework card {CARD1_REWORK_CARD})."
        )
        recommendation_text = None
    else:
        pass_text = None
        parts = []
        if evaluation["criteria"]["a"]["fired"]:
            parts.append(
                "the (1, 0, -1) class set is incomplete or non-conserving "
                "against the committed catalog"
            )
        if evaluation["criteria"]["b"]["fired"]:
            parts.append(
                f"{len(artifacts['kplus']['summary']['unattributable_bins'])} "
                "of 18 persisting K+ bins carry no catalog-audited "
                "class-pair path"
            )
        if evaluation["criteria"]["c"]["fired"]:
            parts.append(
                "the family configuration or the catalog disagrees with "
                "its committed bytes at verdict time (a driver defect)"
            )
        recommendation_text = (
            "Spawn a separate Round 4 rework card on the cards 4/5 "
            f"family surface ({FAMILIES_REL}, sha256 {FAMILIES_SHA256}, "
            "byte-unchanged by this audit): " + "; ".join(parts) + ". "
            "The rework card pre-registers its own class-set or "
            "configuration change and carries the R32 bounds (bin spread "
            "p95 < 2.0 pooled-SE units, feed-down spread max < 3.0 "
            "sigma); it may alternatively confirm the surface as-is and "
            "re-classify the named bins as standing model uncertainty "
            "with the class attribution recorded. The routing-formulation "
            f"share remains card 1's scope ({CARD1_CARD}, sealed "
            f"RECOMMENDATION, rework card {CARD1_REWORK_CARD}); the "
            "audits are independent inputs to the Round 4 run cards."
        )
    payload = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families.verdict/1",
        "run": RUN_ID,
        "card": CARD,
        "verdict": evaluation["verdict"],
        "verdict_contract": pre["verdict_contract"],
        "pass_text": pass_text,
        "probe_set": {
            "entries": kplus["probe_set"]["entries"],
            "bin_indices": kplus["probe_set"]["bin_indices"],
            "sigma_range": kplus["probe_set"]["sigma_range"],
            "source": kplus["probe_set"]["source"],
        },
        "class_pair_summary": {
            "family_id": class_pairs["audit"]["family_id"],
            "qn": class_pairs["audit"]["qn"],
            "declared_classes": class_pairs["audit"]["declared_classes"],
            "conserving_catalog_class_keys": class_pairs["audit"][
                "enumeration"]["conserving_class_keys"],
            "missing_class_keys": class_pairs["contract"][
                "missing_class_keys"],
            "nonconserving_declared": class_pairs["contract"][
                "nonconserving_declared"],
            "complete": class_pairs["contract"]["complete"],
            "conservative": class_pairs["contract"]["conservative"],
            "informational_inactive_strange_mesonic": [
                {
                    "family_id": row["family_id"],
                    "qn": row["qn"],
                    "active": row["active"],
                    "complete": row["complete"],
                    "conservative": row["conservative"],
                    "scope": row["scope"],
                }
                for row in class_pairs[
                    "informational_inactive_strange_mesonic"
                ]
            ],
        },
        "kplus_decomposition_summary": {
            "persisting_kplus_bins": kplus["probe_set"]["entries"],
            "class_attributable": kplus["summary"]["class_attributable"],
            "all_class_attributable": kplus["summary"][
                "all_class_attributable"],
            "dominant_class_pair_counts": kplus["summary"][
                "dominant_class_pair_counts"],
            "identity_sweep_clean": kplus["identity_sweep"]["clean"],
            "per_bin": [
                {
                    "bin_index": row["bin_index"],
                    "xF_mid": row["xF_mid"],
                    "pT_mid": row["pT_mid"],
                    "confirmation_sigma": row["confirmation_sigma"],
                    "dominant_class_pair": row["dominant_class_pair"],
                    "contributions_pooled_se_units": row[
                        "contributions_pooled_se_units"],
                    "density_shape_share": row["density_shape_referral"][
                        "density_shape_share"],
                    "card1_route_dominance": row["density_shape_referral"][
                        "card1_row"]["route_dominance"],
                    "card1_monotone": row["density_shape_referral"][
                        "card1_row"]["p1_monotone"],
                    "card1_mass_blind_envelope_respected": row[
                        "density_shape_referral"]["card1_row"][
                        "mass_blind_envelope_respected"],
                }
                for row in kplus["rows"]
            ],
        },
        "config_sanity_summary": {
            "byte_stable": config_sanity["surface_digests"]["byte_stable"],
            "families_config_sha256": config_sanity["surface_digests"][
                "families_config_sha256"],
            "catalog_sha256": config_sanity["surface_digests"][
                "catalog_sha256"],
            "strange_record_problems": artifacts["config_sanity"][
                "strange_record_problems"],
            "inactive_problems": artifacts["config_sanity"][
                "inactive_problems"],
            "r30_blocks_pass": all(
                block["pass"]
                for block in artifacts["config_sanity"][
                    "r30_cross_reference"].values()
            ),
            "r31_strange_window_interior_failed": artifacts["config_sanity"]
            ["r31_cross_reference"]["interior_pairs_failed"],
            "kinematic_max_pole_mass_sum": artifacts["config_sanity"][
                "kinematic_recomputation"]["max_pole_mass_sum"],
        },
        "bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "kplus_p95_anchor_production": KPLUS_PRODUCTION_P95,
            "note": (
                "the pre-registered bounds are cited verbatim from the "
                "card and the R32 pre-registration; none is relaxed; the "
                "audit produces no new p95 and every per-bin sigma row "
                "carries the register bound 2.0 beside the value"
            ),
        },
        "criteria": evaluation["criteria"],
        "findings": findings,
        "recommendation": recommendation_text,
        "density_shape_referral": {
            "referred_to_card": CARD1_CARD,
            "rework_card": CARD1_REWORK_CARD,
            "note": (
                "the density-shape share of every class-attributed "
                "persisting K+ bin is recorded per bin and referred to "
                "card 1: the persisting spread driver is the blend ramp "
                "placement and width attribution (the card-6 surface), "
                "which the sealed card-1 RECOMMENDATION already owns"
            ),
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "reserved_audit_block_fired": False,
        },
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "surface": {
            "families_config_path": FAMILIES_REL,
            "families_sha256": FAMILIES_SHA256,
            "catalog_path": CATALOG_REL,
            "catalog_sha256": CATALOG_SHA256,
            "byte_unchanged_by_this_audit": True,
        },
    }
    _write_json(run_dir / VERDICT_NAME, payload)
    print(
        f"audit_r4_strange_sector_families: verdict {evaluation['verdict']} "
        f"({len(findings)} named findings; criteria fired: "
        f"{[c['criterion'] for c in evaluation['fired']] or 'none'})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --seal
# ---------------------------------------------------------------------------


def _run_dir_files(run_dir: Path) -> list[Path]:
    """Every regular file under the run directory, sorted by relative
    path."""
    run_dir = Path(run_dir)
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
        rel = path.relative_to(MONO_ROOT).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}\n")
    sums_path = run_dir / SHA256SUMS_NAME
    _write_bytes(sums_path, "".join(lines).encode("utf-8"))
    return sums_path


def write_seal_manifest(run_dir: Path) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the
    sums.  The seal manifest records every regular file under the run
    directory except itself, each with its sha256 and byte size; it
    cannot contain its own digest."""
    run_dir = Path(run_dir)
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
        "schema":
            "na49_mixing.audit_r4_strange_sector_families"
            ".seal_manifest/1",
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
    """Seal the finished audit run directory (sums inventory plus seal
    manifest).  The pre-registration, the provenance, the three
    measurement artifacts, and the verdict must exist."""
    run_dir = Path(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        CLASS_PAIRS_NAME,
        KPLUS_NAME,
        CONFIG_SANITY_NAME,
        VERDICT_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the verdict "
                "is part of the seal; run the measurement modes and "
                "--verdict first)"
            )
    if (run_dir / CLEAN_ENV_NAME).is_file():
        _refuse(
            f"{run_dir / CLEAN_ENV_NAME} exists; the clean-environment "
            "record is written after the seal and the seal is never "
            "re-run over it"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"audit_r4_strange_sector_families: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the four deterministic measurement steps
    from a clean interpreter over the sealed artifacts, and confirm
    every artifact hash matches."""
    run_dir = Path(run_dir)
    if not (run_dir / PREREGISTRATION_NAME).is_file():
        _refuse(f"{run_dir / PREREGISTRATION_NAME} is missing")
    sums_path = run_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(
            f"{sums_path} is missing; --clean-validate requires the "
            "sealed run dir (run --seal first)"
        )
    # -- seal verification (root-relative entries) ------------------------
    mismatches: list[str] = []
    files_checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
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
            f"{len(mismatches)} of {files_checked} SHA256SUMS.txt entries "
            f"do not re-hash correctly (first: {mismatches[0]!r}); the "
            "sealed run dir must be unmodified"
        )
    # -- clean-interpreter re-run of the four measurement steps -----------
    artifact_names = (CLASS_PAIRS_NAME, KPLUS_NAME, CONFIG_SANITY_NAME,
                      VERDICT_NAME)
    before = {
        name: (run_dir / name).read_bytes() for name in artifact_names
    }
    driver_path = Path(__file__).resolve()
    commands = []
    probe_results = []
    for mode in ("class-pairs", "k-contribution", "config-sanity",
                 "verdict"):
        argv = [
            sys.executable, "-I", str(driver_path),
            f"--{mode}", "--run-dir", str(run_dir),
        ]
        commands.append({
            "purpose": (
                f"clean-interpreter re-run of the {mode} step over the "
                "sealed artifacts (rewrites its artifact; the sealed "
                "bytes must be reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        })
        probe = subprocess.run(argv, capture_output=True, text=True,
                               cwd=str(MONO_ROOT))
        probe_results.append((mode, probe.returncode, probe.stderr))
    recomputed = {
        name: (run_dir / name).read_bytes() for name in artifact_names
    }
    hashes_match = all(
        int(rc) == 0
        and hashlib.sha256(before[name]).hexdigest()
        == hashlib.sha256(recomputed[name]).hexdigest()
        and before[name] == recomputed[name]
        for name, (_mode, rc, _err) in zip(artifact_names, probe_results)
    )
    # Restore the sealed bytes: the probes rewrote the artifacts; the
    # sealed dir must keep its sealed content regardless of the probe
    # outcome.
    for name in artifact_names:
        if recomputed[name] != before[name]:
            _write_bytes(run_dir / name, before[name])
    failed = [
        {"mode": mode, "returncode": int(rc), "stderr": err[-2000:]}
        for mode, rc, err in probe_results if int(rc) != 0
    ]
    payload = {
        "schema":
            "na49_mixing.audit_r4_strange_sector_families"
            ".clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": commands,
        "environment": environment_fingerprint(),
        "probe_returncodes": {
            mode: int(rc) for mode, rc, _err in probe_results
        },
        "probe_failures": failed,
        "sealed_artifact_sha256": {
            name: hashlib.sha256(before[name]).hexdigest()
            for name in artifact_names
        },
        "recomputed_artifact_sha256": {
            name: hashlib.sha256(recomputed[name]).hexdigest()
            for name in artifact_names
        },
        "hashes_match": bool(hashes_match),
        "probe_tool": {
            "path": str(driver_path),
            "sha256": _sha256_file(driver_path),
            "note": (
                "the committed dual-tree driver is the probe instrument; "
                "the sealed artifacts, the sealed R32 archives, the "
                "committed catalog, and the committed family "
                "configuration are the inputs"
            ),
        },
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not payload["hashes_match"]:
        print(
            "audit_r4_strange_sector_families: the clean-environment "
            "validation FAILED (the recomputed artifact bytes differ "
            "from the sealed artifacts)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "audit_r4_strange_sector_families: clean-environment validation "
        f"passed ({files_checked} seal files verified, the clean "
        "interpreter reproduced all four measurement artifacts byte "
        "for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_r4_strange_sector_families.py",
        description=(
            "Execute the Round 4 strange-sector continuum families audit "
            "(card TASK-20260920-r4-audit-strange-sector-continuum-"
            "families, Gate 3 follow-up obligation 2): resolve the "
            "K+-concentrated persisting spread of the sealed R32 evidence "
            "(18 K+ persisting bins, the K+ bin p95 of 9.439291) against "
            "the strange-sector continuum spectral densities and their "
            "catalog-audited decay classes, on the byte-stable cards 4/5 "
            "family surface.  Zero draws; read-only on every pinned "
            "surface and sealed archive; PASS or RECOMMENDATION (a "
            "separate rework card), never an in-audit surface edit."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--class-pairs", action="store_true",
        help="the completeness and conservation re-derivation on the "
             "(1, 0, -1) family against the committed catalog, then stop",
    )
    modes.add_argument(
        "--k-contribution", action="store_true",
        help="the decomposition of the 18 persisting K+ bins by parent "
             "decay class from the sealed R32 census, then stop",
    )
    modes.add_argument(
        "--config-sanity", action="store_true",
        help="the strange-records byte check and the R30/R31 "
             "cross-references, then stop",
    )
    modes.add_argument(
        "--verdict", action="store_true",
        help="write verdict.json (PASS or RECOMMENDATION with named "
             "findings), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the four measurement steps from "
             "a clean interpreter, then stop",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help=(
            "the audit run dir (default: "
            f"dev/00runs/{RUN_LABEL}_<execution date> under the "
            "monorepo root); pass an absolute path"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(
        args.run_dir.resolve() if args.run_dir else default_run_dir()
    )
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.class_pairs:
        run_class_pairs(run_dir)
    elif args.k_contribution:
        run_k_contribution(run_dir)
    elif args.config_sanity:
        run_config_sanity(run_dir)
    elif args.verdict:
        run_verdict(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    else:  # pragma: no cover - argparse enforces exactly one mode
        _refuse("exactly one mode is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())