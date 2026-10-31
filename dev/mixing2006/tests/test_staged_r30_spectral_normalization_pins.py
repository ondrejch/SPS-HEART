"""Round 3 card 7: run R30 spectral-normalization committed-run artifact pins.

Task ``TASK-20260917-r3-run-r30-spectral-normalization`` (round 3, card 7;
plan section 9 run R30, line 638; nested feat commit 1900e42, outer
mirror fe78b78).  This module pins the sealed run evidence under
``dev/00runs/r30_spectral_normalization_2026-09-17/``: the
pre-registration (the seven committed families, the fixed 10,001-node
trapezoid reference quadrature, the 1e-6 normalization tolerance, the
threshold-behavior spec, the double-counting identity, and the hermetic
seeds block with master seed 20260917 and ``draws: 0``), the
provenance, the family-integral report (``family_integrals.json``),
the threshold-check report (``threshold_checks.json``), the Gate 3
evidence (``gate3_r30.json``), the seal manifest, ``SHA256SUMS.txt``
(3 payload lines, run-dir-relative, ``sha256sum -c`` format), and the
post-seal clean-environment record.

R30 is hermetic (round 3 index item 6): it draws no random numbers,
generates no events, and fits nothing (``no_fitting``).  It verifies the
cards 4-6 surfaces (the baryonic, strange baryonic, and neutral mesonic
families; the four declared inactive charged and strange mesonic
families; the card-6 blend) before any Monte Carlo run uses them.  The
committed verdict: every one of the seven families integrates to
1.0000000000000002 over its support (max absolute deviation from unity
2.220446049250313e-16, tolerance 1e-6); for each of the three activated
source QN sets the ``rho_total`` integral is exactly 1.0 (deviation
0.0), the split-sum double-counting identity deviation is 0.0, the
``rho_cont`` integrals are 1.0000003402564623 / 1.0000002971062014 /
1.0000005597498782 with matching factors 2.355631289512376e-07 /
9.969592659062824e-07 / 1.3240542705434904e-07; all five threshold
checks pass (the spectrum is exactly zero at and below ``m_min`` on
5,000 of 5,000 fine-grid nodes per family, the C^1 one-sided
derivatives agree to 1.3552527156068805e-14 at worst, tolerance
1e-12, and ``rho_total`` is continuous at the blend boundary B and at
B + W with a probe mismatch of 1.3877787807814457e-17 at worst); the
Gate 3 blocks all pass with ``sealed: true``; and the
clean-interpreter re-run reproduced ``family_integrals.json`` byte for
byte.

Sibling of ``tests/test_staged_r3_threshold_metric_reregistration_pins.
py`` and ``tests/test_staged_r3_feeddown_lineage_convention_pins.py``
(the same pin style: sealed-dir discovery, skip-or-fail when the
gitignored archive is absent, digest pins plus structural pins).  These
pins are read-only over the sealed artifacts: nothing is recomputed,
the sealed run is never re-sealed, and the post-seal
clean-environment record is never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the eight
  top-level files (the five sealed payloads, the seal manifest,
  ``SHA256SUMS.txt``, the post-seal clean-environment record); nothing
  else is present.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (the seven family records in config
  order with their (B, Q, S) quantum numbers and active flags: three
  activated (1,1,0), (1,0,-1), (0,0,0) and four declared inactive
  (0,+1,0), (0,-1,0), (0,0,-1), (0,0,+1); the committed continuum
  config digest; the fixed 10,001-node trapezoid reference quadrature
  over each family's support [m_min, m_max]; the 1e-6 normalization
  tolerance; the threshold-behavior spec with the 1e-12 C^1 derivative
  and the card-6 ``rho_total`` continuity conventions; the
  double-counting identity over the three activated QN sets at 1e-6;
  the blend boundary B = the M8-frozen continuum.match_mass and ramp
  width W = 0.5 GeV; the hermetic seeds block (policy none, master
  seed 20260917, ``draws: 0``); ``no_fitting``; and the eight output
  files).
- ``test_provenance_structure``: the provenance records the committed
  continuum config SHA-256, the catalog SHA-256, the M8-frozen
  continuum.match_mass (cross-checked against the legacy_v1 manifest's
  frozen M8 p+p parameter map), the spectral-density and
  resonance-width module SHA-256s, the dual-tree driver SHA-256 (the
  nested and monorepo copies are byte-identical), the git identity, the
  environment fingerprint, the master seed with zero draws, and
  ``no_fitting``.
- ``test_family_integrals_structure``: the integral report carries
  schema ``na49_mixing.staged_r30_spectral_normalization.integration/1``
  and the committed field set: the 10,001-node quadrature echo, all
  seven family rows with the pinned unit integral
  1.0000000000000002 and the active flags, the family-normalization
  block (7 checked, max absolute deviation from unity
  2.2204460000000e-16, tolerance 1e-6, pass), the per-QN-set table for
  the three activated sets (``rho_total`` integral exactly 1.0,
  ``rho_cont`` integrals 1.0000003402564623 / 1.0000002971062014 /
  1.0000005597498782, the catalog matching factors
  2.355631289512376e-07 / 9.969592659062824e-07 /
  1.3240542705434904e-07, the split integrals summing exactly to 1.0
  with deviation 0.0, and the double-counting identity with deviation
  0.0 at the 1e-6 tolerance), the double-counting summary (max absolute
  deviation 0.0, pass), and the no-Monte-Carlo block.
- ``test_threshold_checks_structure``: the threshold report carries the
  fine-grid spec (node spacing 1e-6 GeV, window half-width 0.005 GeV,
  5,000 nodes per side, the three named junction windows at m_min, B,
  and B + W with the m_min/B node-for-node coincidence note) and all
  five pre-registered checks passing: ``rho_a`` exactly zero on
  5,000/5,000 strictly-below nodes per family with the junction value
  zero; the C^1 junctions at m_min (max derivative mismatch
  5.387274994699222e-16) and at m_min + W (max derivative mismatch
  1.3552527156068805e-14, tolerance 1e-12); ``rho_total`` continuous at
  B (``rho_cont(B)`` exactly 0.0, max probe mismatch
  4.658681209898652e-21) and at B + W (max probe mismatch
  1.3877787807814457e-17); and the top-level maxima mirroring the
  per-check maxima.
- ``test_gate3_structure``: the Gate 3 evidence carries the three
  blocks (family_normalization over 7 families, threshold_behavior
  over the five named checks, double_counting_identity over the three
  activated QN sets) all passing with values consistent with the
  integral and threshold reports, ``no_fitting: true``, ``sealed:
  true``, and ``input_hashes`` matching the actual SHA-256 of
  ``family_integrals.json`` and ``threshold_checks.json`` on disk.
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records 6 sealed files totaling 30,890 bytes, and
  ``SHA256SUMS.txt`` is in ``sha256sum -c`` format with run-dir-
  relative paths (LF, two spaces, relative path), carries the 3 pinned
  payload digests, verifies by recomputed SHA-256 against every payload
  file on disk, and self-consistently accounts for the full byte total.
- ``test_clean_environment_record``: the post-seal clean-environment
  record re-verified the 3 sums-covered files without mismatches, ran
  the clean interpreter (``python -I``) with return code 0, and the
  re-run reproduced ``family_integrals.json`` byte for byte (the
  recorded sealed and recomputed digests match, equal the pinned
  digest, and equal the digest of the file on disk).

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and the committed config, catalog, package, and
drivers; no computation runs.  If the run dir is absent (a fresh
checkout), the artifact pins skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r30_spectral_normalization_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
LEGACY_MANIFEST_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r30_spectral_normalization.py"
)
DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r30_spectral_normalization.py"
)
SPECTRAL_DENSITY_MODULE_PATH = (
    NESTED_ROOT / "na49_mixing" / "spectral_density.py"
)
RESONANCE_WIDTHS_MODULE_PATH = (
    NESTED_ROOT / "na49_mixing" / "resonance_widths.py"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R30"
CARD_ID = "TASK-20260917-r3-run-r30-spectral-normalization"
MASTER_SEED = 20260917
DRAWS = 0
N_FAMILIES = 7
N_QUADRATURE_NODES = 10_001
NORMALIZATION_TOLERANCE = 1e-6
DERIVATIVE_TOLERANCE = 1e-12
BLEND_BOUNDARY = 2.680194110727528  # the M8-frozen continuum.match_mass
RAMP_WIDTH = 0.5
M_MAX = 5.0
M_MAX_SUPPORT = [2.680194110727528, 5.0]

#: The seven family ids in config order (the three activated families
#: first, then the four declared inactive mesonic families).
FAMILY_IDS = [
    "baryonic_np",
    "baryonic_strange",
    "mesonic_neutral",
    "mesonic_charged",
    "mesonic_anticharged",
    "mesonic_strange",
    "mesonic_antistrange",
]

#: The (B, Q, S) quantum numbers and active flags of the seven families.
FAMILY_QN = {
    "baryonic_np": (1, 1, 0),
    "baryonic_strange": (1, 0, -1),
    "mesonic_neutral": (0, 0, 0),
    "mesonic_charged": (0, 1, 0),
    "mesonic_anticharged": (0, -1, 0),
    "mesonic_strange": (0, 0, -1),
    "mesonic_antistrange": (0, 0, 1),
}
FAMILY_ACTIVE = {
    "baryonic_np": True,
    "baryonic_strange": True,
    "mesonic_neutral": True,
    "mesonic_charged": False,
    "mesonic_anticharged": False,
    "mesonic_strange": False,
    "mesonic_antistrange": False,
}
FAMILY_DECAY_CLASSES = {
    "baryonic_np": 8,
    "baryonic_strange": 6,
    "mesonic_neutral": 10,
    "mesonic_charged": 6,
    "mesonic_anticharged": 6,
    "mesonic_strange": 7,
    "mesonic_antistrange": 7,
}

ACTIVATED_QN_SETS = [[1, 1, 0], [1, 0, -1], [0, 0, 0]]
DECLARED_INACTIVE_QN_SETS = [[0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1]]

#: The shared pre-registered family parameters (identical on all seven
#: records of the committed config).
FAMILY_PARAMS = {
    "t_h": 0.165,
    "b_power": 0.0,
    "n_norm": 4.188698227099897e-13,
    "ramp_width": 0.5,
    "nominal_width": 0.5,
}

PRE_SCHEMA = "na49_mixing.staged_r30_spectral_normalization.pre_registration/1"
PROV_SCHEMA = "na49_mixing.staged_r30_spectral_normalization.provenance/1"
INTEGRATION_SCHEMA = (
    "na49_mixing.staged_r30_spectral_normalization.integration/1"
)
THRESHOLD_SCHEMA = (
    "na49_mixing.staged_r30_spectral_normalization.threshold/1"
)
GATE_SCHEMA = "na49_mixing.staged_r30_spectral_normalization.gate/1"
CLEAN_SCHEMA = (
    "na49_mixing.staged_r30_spectral_normalization.clean_environment/1"
)

# ---------------------------------------------------------------------------
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CONFIG_SHA256 = (
    "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
)
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
DRIVER_SHA256 = (
    "d03c5d0e390698baa9c968c607a08538b937dcfd54874ea051229a76d851d349"
)
SPECTRAL_DENSITY_MODULE_SHA256 = (
    "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
)
RESONANCE_WIDTHS_MODULE_SHA256 = (
    "1f87b8108cf4e7b0fa8a57b5b4e2747f8665eb549c77d34d624e54b082e53e1e"
)

# ---------------------------------------------------------------------------
# The pinned sealed-file digests (the sha256 pin set of this module).
# ---------------------------------------------------------------------------
PRE_REGISTRATION_SHA256 = (
    "b765df84edafb5d4515e9ef24a3f2b049bedd5bac2dd0eb64073615997e51f4d"
)
PROVENANCE_SHA256 = (
    "e1b86a044f7a0b5f2fb22c5c496e8d0fd5fb510aad80e39bed638e918ae860c7"
)
FAMILY_INTEGRALS_SHA256 = (
    "251b404168641972d38da04bdbaf9fa64192f8dddb007b0956ed4df742af9d97"
)
THRESHOLD_CHECKS_SHA256 = (
    "8a77759896f1d25248e410485abf9bd19b3dcd28e4578d9c8c134530f2e242e3"
)
GATE_SHA256 = (
    "98a5151aa9237e5d10bc5becde48c5b758b047d9f8065f937f2bf2038de69ec8"
)
SEAL_MANIFEST_SHA256 = (
    "f2b5414709a8d42b1b984e78f86863a027bc8ca6c1ebdecb005a6967d531ffc9"
)
SUMS_FILE_SHA256 = (
    "7b36bcfaaf749f0ab69a24bc6336b55fb7a5603440df0e88e9a5fd55a9a6a358"
)
CLEAN_ENVIRONMENT_SHA256 = (
    "b0f64a57af45855b7cde3ec42f3521c93ece2f5daaa813c6d364ba8d4dc1ba13"
)

# ---------------------------------------------------------------------------
# The pinned sealed verdicts (per the task card, verified bit-exact
# against the on-disk artifacts).
# ---------------------------------------------------------------------------

#: Every one of the seven family trapezoid integrals over its support.
FAMILY_INTEGRAL = 1.0000000000000002
MAX_ABS_DEV_FAMILY_FROM_UNITY = 2.220446049250313e-16

#: The per-activated-QN-set blend verdicts (in activated order).
QN_SET_PIN = [
    {
        "qn": [1, 1, 0],
        "support": [0.93827231, 5.0],
        "rho_total": 1.0,
        "rho_catalog": 2.1199837219199927e-06,
        "rho_cont": 1.0000003402564623,
        "catalog_below_boundary": 1.797649130220582e-06,
        "rho_total_at_or_above_boundary": 0.9999982023508698,
        "matching_factor": 2.355631289512376e-07,
    },
    {
        "qn": [1, 0, -1],
        "support": [1.115683, 5.0],
        "rho_total": 1.0,
        "rho_catalog": 1.9942532743897345e-06,
        "rho_cont": 1.0000002971062014,
        "catalog_below_boundary": 1.9942532743897345e-06,
        "rho_total_at_or_above_boundary": 0.9999980057467256,
        "matching_factor": 9.969592659062824e-07,
    },
    {
        "qn": [0, 0, 0],
        "support": [0.1349766, 5.0],
        "rho_total": 1.0,
        "rho_catalog": 1.8040045816306563e-06,
        "rho_cont": 1.0000005597498782,
        "catalog_below_boundary": 1.6686495699815337e-06,
        "rho_total_at_or_above_boundary": 0.99999833135043,
        "matching_factor": 1.3240542705434904e-07,
    },
]

#: The pinned threshold-check maxima (tolerance 1e-12).
MAX_DERIVATIVE_MISMATCH = 1.3552527156068805e-14
MAX_CONTINUITY_PROBE_MISMATCH = 1.3877787807814457e-17
C1_AT_M_MIN_MISMATCH = 5.387274994699222e-16
C1_AT_M_MIN_W_MISMATCH = MAX_DERIVATIVE_MISMATCH
MAX_PROBE_MISMATCH_AT_B = 4.658681209898652e-21
N_NODES_PER_SIDE = 5000
NODE_SPACING_GEV = 1e-6
WINDOW_HALF_WIDTH_GEV = 0.005

#: The five pre-registered threshold checks (in report order).
CHECK_NAMES = [
    "zero below m_min",
    "C1 at m_min",
    "C1 at m_min+W",
    "rho_total continuous at B",
    "rho_total continuous at B+W",
]

N_SEAL_MANIFEST_ENTRIES = 6
SEAL_TOTAL_BYTES = 30_890

SUMS_TOP_LEVEL_PAYLOADS = (
    "family_integrals.json",
    "pre_registration.json",
    "threshold_checks.json",
)
SUMS_PAYLOAD_SHA256 = {
    "family_integrals.json": FAMILY_INTEGRALS_SHA256,
    "pre_registration.json": PRE_REGISTRATION_SHA256,
    "threshold_checks.json": THRESHOLD_CHECKS_SHA256,
}
#: The sealed report payload the clean-environment record re-verified
#: byte-for-byte (sealed digest == recomputed digest).
CLEAN_SEALED_REPORT_SHA256 = {
    "family_integrals.json": FAMILY_INTEGRALS_SHA256,
}

TOP_LEVEL_FILES = (
    "pre_registration.json",
    "provenance.json",
    "family_integrals.json",
    "threshold_checks.json",
    "gate3_r30.json",
    "clean_environment_commands.json",
    "seal_manifest.json",
    "SHA256SUMS.txt",
)

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")
UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict) -> str:
    """The driver's canonical report hash (indent-2 JSON, trailing
    newline, UTF-8)."""
    return hashlib.sha256(
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _m8_frozen_match_mass() -> float:
    """The M8-frozen continuum.match_mass from the legacy_v1 manifest
    (the single source of truth for the frozen M8 p+p parameters)."""
    manifest = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    return float(
        manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
            "continuum.match_mass"
        ]
    )


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R30 run dir, skipped with a clear message when the
    gitignored artifact is absent or incomplete (a fresh checkout)."""
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the sealed R30 run dir {RUN_DIR} is absent (gitignored "
            "artifact); restore the sealed run to execute the "
            "artifact pins"
        )
    missing = [name for name in TOP_LEVEL_FILES if not (RUN_DIR / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed R30 run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing) + "); restore the sealed run to "
            "execute the artifact pins"
        )
    return RUN_DIR


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the eight top-level files (the five
    sealed payloads, the seal manifest, the sums file, the post-seal
    clean-environment record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES), (
        "the sealed run dir drifted from the committed 8-file top "
        "level set"
    )


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract: the
    seven family records in config order with their (B, Q, S) quantum
    numbers and active flags (three activated: (1,1,0), (1,0,-1),
    (0,0,0); four declared inactive: (0,+1,0), (0,-1,0), (0,0,-1),
    (0,0,+1)), the committed continuum config digest, the fixed
    10,001-node trapezoid reference quadrature over each family's
    support [m_min, m_max], the 1e-6 normalization tolerance, the
    threshold-behavior spec, the double-counting identity over the
    three activated QN sets, the blend boundary B = the M8-frozen
    continuum.match_mass with ramp width W = 0.5 GeV, the hermetic
    seeds block (policy none, master seed 20260917, ``draws: 0``),
    ``no_fitting``, and the eight output files."""
    pre = _load("pre_registration.json")
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_evaluation"] is True

    tier = pre["tier"]
    assert tier["monte_carlo"] is False
    assert tier["replicas"] == 0
    assert tier["events"] == 0

    families = pre["families"]
    assert families["source"] == (
        "config/v2_singlet/continuum_resonances.json"
    )
    assert families["sha256"] == CONFIG_SHA256
    # the recorded config digest matches the committed config on disk
    assert _file_sha256(CONFIG_PATH) == CONFIG_SHA256
    assert families["count"] == N_FAMILIES
    assert families["activated_qn_sets"] == ACTIVATED_QN_SETS
    assert families["declared_inactive_qn_sets"] == DECLARED_INACTIVE_QN_SETS
    assert "round 3 index item 9" in families["note"]

    records = families["records"]
    assert len(records) == N_FAMILIES
    for record in records:
        family_id = record["family_id"]
        assert family_id in FAMILY_IDS
        assert record["qn"] == list(FAMILY_QN[family_id])
        assert record["state_type"] == "continuum_resonance"
        assert record["active"] is FAMILY_ACTIVE[family_id]
        assert record["m_min"] == BLEND_BOUNDARY
        assert record["m_max"] == M_MAX
        for key, value in FAMILY_PARAMS.items():
            assert record[key] == value, (
                f"{family_id}: the pre-registered {key} drifted from "
                "the committed config value"
            )
        assert record["n_allowed_decay_classes"] == (
            FAMILY_DECAY_CLASSES[family_id]
        )
    assert [record["family_id"] for record in records] == FAMILY_IDS

    quadrature = pre["quadrature"]
    assert quadrature["rule"] == "trapezoid"
    assert quadrature["nodes"] == N_QUADRATURE_NODES
    assert quadrature["reference_grid_points"] == N_QUADRATURE_NODES
    assert quadrature["spectral_quadrature_nodes"] == N_QUADRATURE_NODES
    assert quadrature["support_per_family"] == [
        BLEND_BOUNDARY, M_MAX,
    ]

    assert pre["normalization_tolerance"] == NORMALIZATION_TOLERANCE

    threshold = pre["threshold_behavior"]
    assert set(threshold) == {
        "zero_below_m_min", "c1_junctions", "rho_total_continuity",
    }
    zero = threshold["zero_below_m_min"]
    assert "strictly below m_min" in zero["claim"]
    assert zero["fine_grid"] == {
        "node_spacing_gev": NODE_SPACING_GEV,
        "window_half_width_gev": WINDOW_HALF_WIDTH_GEV,
        "nodes_per_side": N_NODES_PER_SIDE,
    }
    c1 = threshold["c1_junctions"]
    assert c1["tolerance"] == DERIVATIVE_TOLERANCE
    assert "C^1" in c1["claim"]
    continuity = threshold["rho_total_continuity"]
    assert continuity["tolerance"] == DERIVATIVE_TOLERANCE
    assert "B and at B + W" in continuity["claim"]
    assert "card-6" in continuity["definition"]

    identity = pre["double_counting_identity"]
    assert identity["tolerance"] == NORMALIZATION_TOLERANCE
    assert identity["qn_sets"] == ACTIVATED_QN_SETS
    assert "1e-6" in identity["statement"]

    blend = pre["blend"]
    assert blend["boundary_B"] == BLEND_BOUNDARY
    assert blend["ramp_width_W"] == RAMP_WIDTH
    assert blend["boundary_B"] == _m8_frozen_match_mass()

    seeds = pre["seeds"]
    assert seeds["master_seed"] == MASTER_SEED
    assert seeds["draws"] == DRAWS
    assert "none" in seeds["policy"]

    assert pre["no_fitting"] is True
    assert pre["output_files"] == list(TOP_LEVEL_FILES)

    inputs = pre["input_hashes"]
    assert inputs["continuum_config_path"] == (
        "config/v2_singlet/continuum_resonances.json"
    )
    assert inputs["continuum_config_sha256"] == CONFIG_SHA256
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert inputs["m8_continuum_match_mass"] == BLEND_BOUNDARY
    assert inputs["spectral_density_module_sha256"] == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert inputs["resonance_widths_module_sha256"] == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )
    assert inputs["driver_sha256"] == DRIVER_SHA256
    assert inputs["driver_mirror_sha256"] == DRIVER_SHA256
    assert _file_sha256(SPECTRAL_DENSITY_MODULE_PATH) == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert _file_sha256(RESONANCE_WIDTHS_MODULE_PATH) == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )

    # the sealed pre-registration is byte-pinned
    assert _file_sha256(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )


def test_provenance_structure(run_dir):
    """The provenance records the committed continuum config SHA-256,
    the catalog SHA-256, the M8-frozen continuum.match_mass (cross-
    checked against the legacy_v1 manifest's frozen M8 p+p parameter
    map), the spectral-density and resonance-width module SHA-256s,
    the dual-tree driver SHA-256 (the nested and monorepo copies are
    byte-identical), the git identity, the environment fingerprint,
    the master seed with zero draws, and ``no_fitting``."""
    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert UTC_STAMP.match(prov["written_utc"])
    assert prov["written_before_evaluation"] is True

    assert prov["driver"] == (
        "dev/mixing2006/helpers/benchmarking/"
        "staged_r30_spectral_normalization.py"
    )
    assert prov["driver_sha256"] == DRIVER_SHA256
    assert prov["driver_mirror_sha256"] == DRIVER_SHA256
    # dual-tree driver: the nested and monorepo copies are
    # byte-identical and carry the pinned digest
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256
    assert _file_sha256(DRIVER_OUTER_PATH) == DRIVER_SHA256

    inputs = prov["inputs"]
    assert inputs["continuum_config_path"] == (
        "config/v2_singlet/continuum_resonances.json"
    )
    assert inputs["continuum_config_sha256"] == CONFIG_SHA256
    assert _file_sha256(CONFIG_PATH) == CONFIG_SHA256
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert inputs["m8_continuum_match_mass"] == BLEND_BOUNDARY
    assert inputs["m8_continuum_match_mass"] == _m8_frozen_match_mass()
    assert inputs["spectral_density_module_sha256"] == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert inputs["resonance_widths_module_sha256"] == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )
    assert inputs["driver_sha256"] == DRIVER_SHA256
    assert inputs["driver_mirror_sha256"] == DRIVER_SHA256

    seeds = prov["seeds"]
    assert seeds == {
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "no_monte_carlo": True,
    }
    for key in ("nested_head", "nested_dirty_entries",
                "outer_head", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_executable", "platform",
                "numpy_version"):
        assert key in prov["environment"]

    assert prov["no_fitting"] is True

    assert _file_sha256(run_dir / "provenance.json") == PROVENANCE_SHA256


def test_family_integrals_structure(run_dir):
    """The integral report carries schema
    ``na49_mixing.staged_r30_spectral_normalization.integration/1`` and
    the committed field set: the 10,001-node quadrature echo, all seven
    family rows with the pinned unit integral 1.0000000000000002 and
    the active flags, the family-normalization block (7 checked, max
    absolute deviation from unity 2.220446049250313e-16, tolerance
    1e-6, pass), the per-QN-set table for the three activated sets
    (``rho_total`` integral exactly 1.0 with deviation 0.0, the
    split-sum identity with deviation 0.0, ``rho_cont`` integrals
    1.0000003402564623 / 1.0000002971062014 / 1.0000005597498782, the
    catalog matching factors 2.355631289512376e-07 /
    9.969592659062824e-07 / 1.3240542705434904e-07), the
    double-counting summary (max absolute deviation 0.0, pass), and
    the no-Monte-Carlo block."""
    payload = _load("family_integrals.json")
    assert payload["schema"] == INTEGRATION_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID

    inputs = payload["input_hashes"]
    assert inputs["continuum_config_sha256"] == CONFIG_SHA256
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    assert inputs["m8_continuum_match_mass"] == BLEND_BOUNDARY
    assert inputs["driver_sha256"] == DRIVER_SHA256

    quadrature = payload["quadrature"]
    assert quadrature["rule"] == "trapezoid"
    assert quadrature["nodes"] == N_QUADRATURE_NODES
    assert quadrature["spectral_quadrature_nodes"] == N_QUADRATURE_NODES

    families = payload["families"]
    assert len(families) == N_FAMILIES
    for row in families:
        family_id = row["family_id"]
        assert family_id in FAMILY_IDS
        assert row["qn"] == list(FAMILY_QN[family_id])
        assert row["grid_points"] == N_QUADRATURE_NODES
        assert row["support"] == [BLEND_BOUNDARY, M_MAX]
        # the pinned unit integral (bit-exact)
        assert row["integral"] == FAMILY_INTEGRAL, (
            f"{family_id}: the sealed integral drifted from the "
            "pinned value"
        )
        assert row["tolerance"] == NORMALIZATION_TOLERANCE
        assert row["within_tolerance"] is True
        assert row["active"] is FAMILY_ACTIVE[family_id]
        assert row["n_allowed_decay_classes"] == (
            FAMILY_DECAY_CLASSES[family_id]
        )
    assert [row["family_id"] for row in families] == FAMILY_IDS

    normalization = payload["family_normalization"]
    assert set(normalization) == {
        "families_checked", "max_abs_deviation_from_unity", "tolerance",
        "pass",
    }
    assert normalization["families_checked"] == N_FAMILIES
    assert normalization["max_abs_deviation_from_unity"] == (
        MAX_ABS_DEV_FAMILY_FROM_UNITY
    )
    # self-consistent: the recorded maximum is the maximum of the rows
    row_max = max(
        abs(row["integral"] - 1.0) for row in families
    )
    assert normalization["max_abs_deviation_from_unity"] == row_max
    assert normalization["tolerance"] == NORMALIZATION_TOLERANCE
    assert normalization["pass"] is True

    qn_sets = payload["activated_qn_sets"]
    assert len(qn_sets) == len(ACTIVATED_QN_SETS)
    for row, pinned in zip(qn_sets, QN_SET_PIN):
        # the three activated source QN sets, in pre-registered order
        assert row["qn"] == pinned["qn"]
        assert row["qn"] in ACTIVATED_QN_SETS
        assert row["grid_points"] == N_QUADRATURE_NODES
        assert row["support"] == pinned["support"]
        assert row["boundary"] == BLEND_BOUNDARY
        assert row["ramp_width"] == RAMP_WIDTH
        assert row["tolerance"] == NORMALIZATION_TOLERANCE

        integrals = row["integrals"]
        assert integrals["rho_total"] == pinned["rho_total"] == 1.0
        assert integrals["rho_catalog"] == pinned["rho_catalog"]
        assert integrals["rho_cont"] == pinned["rho_cont"]
        assert integrals["rho_cont"] > 1.0, (
            f"qn {pinned['qn']}: the continuum integral must exceed "
            "unity (the ramp complement is matched away below the "
            "boundary)"
        )

        split = row["split_integrals"]
        assert split["catalog_below_boundary"] == (
            pinned["catalog_below_boundary"]
        )
        assert split["rho_total_at_or_above_boundary"] == (
            pinned["rho_total_at_or_above_boundary"]
        )
        # the split-sum identity: the two split integrals sum exactly
        # to 1.0 (the deviation from unity is 0.0)
        assert split["sum"] == 1.0
        assert split["catalog_below_boundary"] + (
            split["rho_total_at_or_above_boundary"]
        ) == pytest.approx(1.0, abs=0.0)

        matching = row["catalog_matching"]
        assert matching["cont_integral"] == pinned["rho_cont"]
        assert matching["matching_factor"] == pinned["matching_factor"]

        identity = row["double_counting_identity"]
        assert identity["total"]["integral"] == 1.0
        assert identity["total"]["deviation"] == 0.0
        assert identity["total"]["tolerance"] == NORMALIZATION_TOLERANCE
        assert identity["total"]["within_tolerance"] is True
        assert identity["split"]["integral"] == 1.0
        assert identity["split"]["deviation"] == 0.0
        assert identity["split"]["tolerance"] == NORMALIZATION_TOLERANCE
        assert identity["split"]["within_tolerance"] is True

    summary = payload["double_counting_identity"]
    assert summary["qn_sets"] == ACTIVATED_QN_SETS
    assert summary["max_abs_deviation_from_unity"] == 0.0
    assert summary["tolerance"] == NORMALIZATION_TOLERANCE
    assert summary["pass"] is True

    assert payload["no_fitting"] is True
    assert payload["monte_carlo"] == {
        "draws": DRAWS,
        "no_monte_carlo": True,
        "master_seed_recorded": MASTER_SEED,
    }

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(run_dir / "family_integrals.json") == (
        FAMILY_INTEGRALS_SHA256
    )
    assert _canonical_sha256(payload) == FAMILY_INTEGRALS_SHA256


def test_threshold_checks_structure(run_dir):
    """The threshold report carries the fine-grid spec (node spacing
    1e-6 GeV, window half-width 0.005 GeV, 5,000 nodes per side, the
    three named junction windows at m_min, B, and B + W with the
    m_min/B node-for-node coincidence note) and all five
    pre-registered checks passing: ``rho_a`` exactly zero on
    5,000/5,000 strictly-below nodes per family with the junction
    value zero; the C^1 junctions at m_min (max derivative mismatch
    5.387274994699222e-16) and at m_min + W (max derivative mismatch
    1.3552527156068805e-14, tolerance 1e-12); ``rho_total``
    continuous at B (``rho_cont(B)`` exactly 0.0, max probe mismatch
    4.658681209898652e-21) and at B + W (max probe mismatch
    1.3877787807814457e-17); and the top-level maxima mirroring the
    per-check maxima."""
    payload = _load("threshold_checks.json")
    assert payload["schema"] == THRESHOLD_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID

    grid = payload["fine_grid"]
    assert grid["node_spacing_gev"] == NODE_SPACING_GEV
    assert grid["window_half_width_gev"] == WINDOW_HALF_WIDTH_GEV
    assert grid["nodes_per_side"] == N_NODES_PER_SIDE
    assert grid["nodes_per_window"] == 2 * N_NODES_PER_SIDE + 1
    junctions = grid["junctions"]
    assert set(junctions) == {"m_min", "B", "B+W"}
    assert junctions["m_min"]["center"] == BLEND_BOUNDARY
    assert junctions["B"]["center"] == BLEND_BOUNDARY
    assert junctions["B+W"]["center"] == BLEND_BOUNDARY + RAMP_WIDTH
    # the m_min and B windows coincide node for node (the cards 4-6
    # shared boundary values)
    assert junctions["m_min"]["first_node"] == junctions["B"]["first_node"]
    assert junctions["m_min"]["last_node"] == junctions["B"]["last_node"]
    assert "node for node" in grid["coincidence_note"]

    checks = payload["checks"]
    assert set(checks) == set(CHECK_NAMES)

    zero = checks["zero below m_min"]
    assert zero["pass"] is True
    assert zero["n_families_checked"] == N_FAMILIES
    assert zero["n_nodes_strictly_below_per_family"] == N_NODES_PER_SIDE
    assert len(zero["families"]) == N_FAMILIES
    for row in zero["families"]:
        family_id = row["family_id"]
        assert family_id in FAMILY_IDS
        assert row["qn"] == list(FAMILY_QN[family_id])
        # exactly 0.0 on all 5,000 strictly-below nodes, and at the
        # junction itself
        assert row["n_nodes_strictly_below"] == N_NODES_PER_SIDE
        assert row["n_zero_strictly_below"] == N_NODES_PER_SIDE
        assert row["junction_value_zero"] is True
    assert [row["family_id"] for row in zero["families"]] == FAMILY_IDS

    c1_mmin = checks["C1 at m_min"]
    assert c1_mmin["pass"] is True
    assert c1_mmin["tolerance"] == DERIVATIVE_TOLERANCE
    assert c1_mmin["max_derivative_mismatch"] == C1_AT_M_MIN_MISMATCH
    assert len(c1_mmin["families"]) == N_FAMILIES
    for row in c1_mmin["families"]:
        assert row["left_derivative"] == 0.0, (
            "below m_min the left side is the zero function (left "
            "derivative exactly 0)"
        )
        assert row["mismatch"] == C1_AT_M_MIN_MISMATCH
        assert row["mismatch"] == pytest.approx(
            abs(row["right_derivative"] - row["left_derivative"]),
            rel=1e-12,
        )
        assert row["mismatch"] <= DERIVATIVE_TOLERANCE

    c1_mmin_w = checks["C1 at m_min+W"]
    assert c1_mmin_w["pass"] is True
    assert c1_mmin_w["tolerance"] == DERIVATIVE_TOLERANCE
    assert c1_mmin_w["max_derivative_mismatch"] == (
        C1_AT_M_MIN_W_MISMATCH
    )
    assert len(c1_mmin_w["families"]) == N_FAMILIES
    for row in c1_mmin_w["families"]:
        assert row["mismatch"] <= DERIVATIVE_TOLERANCE
    row_max = max(
        row["mismatch"] for row in c1_mmin_w["families"]
    )
    assert c1_mmin_w["max_derivative_mismatch"] == row_max

    cont_b = checks["rho_total continuous at B"]
    assert cont_b["pass"] is True
    assert cont_b["tolerance"] == DERIVATIVE_TOLERANCE
    assert cont_b["max_probe_mismatch"] == MAX_PROBE_MISMATCH_AT_B
    assert len(cont_b["qn_sets"]) == len(ACTIVATED_QN_SETS)
    for row in cont_b["qn_sets"]:
        assert row["qn"] in ACTIVATED_QN_SETS
        # the card-6 identity at the boundary: rho_total(B-) ==
        # rho_catalog(B) and rho_cont(B) == 0.0 exactly
        assert row["rho_total_at_boundary"] == row[
            "rho_catalog_at_boundary"
        ]
        assert row["rho_cont_at_boundary"] == 0.0
        assert row["rho_cont_at_boundary_is_zero"] is True
        assert row["probe_mismatch"] <= DERIVATIVE_TOLERANCE

    cont_b_w = checks["rho_total continuous at B+W"]
    assert cont_b_w["pass"] is True
    assert cont_b_w["tolerance"] == DERIVATIVE_TOLERANCE
    assert cont_b_w["max_probe_mismatch"] == (
        MAX_CONTINUITY_PROBE_MISMATCH
    )
    assert len(cont_b_w["qn_sets"]) == len(ACTIVATED_QN_SETS)
    for row in cont_b_w["qn_sets"]:
        # above the ramp end rho_total == rho_cont (the card-6
        # identity at B + W)
        assert row["rho_total_at_ramp_end"] == row["rho_cont_at_ramp_end"]
        assert row["probe_mismatch"] <= DERIVATIVE_TOLERANCE
    row_max = max(
        row["probe_mismatch"] for row in cont_b_w["qn_sets"]
    )
    assert cont_b_w["max_probe_mismatch"] == row_max

    # every pre-registered check passes
    assert all(check["pass"] is True for check in checks.values())
    assert payload["check_names"] == CHECK_NAMES
    assert payload["max_derivative_mismatch"] == MAX_DERIVATIVE_MISMATCH
    assert payload["max_continuity_probe_mismatch"] == (
        MAX_CONTINUITY_PROBE_MISMATCH
    )
    assert payload["pass"] is True
    assert payload["no_fitting"] is True
    assert payload["monte_carlo"] == {
        "draws": DRAWS,
        "no_monte_carlo": True,
        "master_seed_recorded": MASTER_SEED,
    }

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(run_dir / "threshold_checks.json") == (
        THRESHOLD_CHECKS_SHA256
    )
    assert _canonical_sha256(payload) == THRESHOLD_CHECKS_SHA256


def test_gate3_structure(run_dir):
    """The Gate 3 evidence carries the three blocks
    (family_normalization over 7 families, threshold_behavior over the
    five named checks, double_counting_identity over the three
    activated QN sets) all passing with values consistent with the
    integral and threshold reports, ``no_fitting: true``, ``sealed:
    true``, and ``input_hashes`` matching the actual SHA-256 of
    ``family_integrals.json`` and ``threshold_checks.json`` on disk."""
    gate = _load("gate3_r30.json")
    assert gate["schema"] == GATE_SCHEMA
    assert gate["gate"] == (
        "gate 3 (plan section 9, line 650), family-side evidence"
    )
    assert gate["run"] == RUN_ID
    assert gate["card"] == CARD_ID

    blocks = gate["blocks"]
    assert set(blocks) == {
        "family_normalization", "threshold_behavior",
        "double_counting_identity",
    }
    for block in blocks.values():
        assert block["pass"] is True

    family_block = blocks["family_normalization"]
    assert family_block["families_checked"] == N_FAMILIES
    assert family_block["max_abs_deviation_from_unity"] == (
        MAX_ABS_DEV_FAMILY_FROM_UNITY
    )
    assert family_block["tolerance"] == NORMALIZATION_TOLERANCE
    # consistent with the integral report's family-normalization block
    assert family_block == _load("family_integrals.json")[
        "family_normalization"
    ]

    threshold_block = blocks["threshold_behavior"]
    assert threshold_block["checks"] == CHECK_NAMES
    assert threshold_block["max_derivative_mismatch"] == (
        MAX_DERIVATIVE_MISMATCH
    )
    assert threshold_block["max_continuity_probe_mismatch"] == (
        MAX_CONTINUITY_PROBE_MISMATCH
    )
    assert threshold_block["tolerances"] == {
        "derivative": DERIVATIVE_TOLERANCE,
        "continuity": DERIVATIVE_TOLERANCE,
    }

    identity_block = blocks["double_counting_identity"]
    assert identity_block["qn_sets"] == ACTIVATED_QN_SETS
    assert identity_block["max_abs_deviation_from_unity"] == 0.0
    assert identity_block["tolerance"] == NORMALIZATION_TOLERANCE
    # consistent with the integral report's double-counting summary
    assert identity_block["max_abs_deviation_from_unity"] == (
        _load("family_integrals.json")["double_counting_identity"][
            "max_abs_deviation_from_unity"
        ]
    )

    # the input hashes match the actual sha256 of the sealed report
    # payloads on disk
    assert gate["input_hashes"]["family_integrals_sha256"] == (
        FAMILY_INTEGRALS_SHA256
    )
    assert gate["input_hashes"]["threshold_checks_sha256"] == (
        THRESHOLD_CHECKS_SHA256
    )
    assert _file_sha256(run_dir / "family_integrals.json") == (
        gate["input_hashes"]["family_integrals_sha256"]
    )
    assert _file_sha256(run_dir / "threshold_checks.json") == (
        gate["input_hashes"]["threshold_checks_sha256"]
    )

    assert gate["no_fitting"] is True
    assert gate["no_monte_carlo"] == {
        "draws": DRAWS,
        "no_monte_carlo": True,
        "master_seed_recorded": MASTER_SEED,
    }
    assert gate["sealed"] is True

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(run_dir / "gate3_r30.json") == GATE_SHA256
    assert _canonical_sha256(gate) == GATE_SHA256


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 6 sealed files (the three
    sums-covered payloads plus ``SHA256SUMS.txt``, ``gate3_r30.json``,
    and ``provenance.json``) totaling 30,890 bytes, and
    ``SHA256SUMS.txt`` is in ``sha256sum -c`` format with run-dir-
    relative paths (LF line endings, two spaces, relative path),
    carries the 3 pinned payload digests, verifies by recomputed
    SHA-256 against every payload file on disk, and self-consistently
    accounts for the full byte total."""
    manifest = _load("seal_manifest.json")
    assert set(manifest) == {
        "sealed_at_utc", "run_directory", "card", "format", "files",
    }
    assert UTC_STAMP.match(manifest["sealed_at_utc"])
    assert manifest["run_directory"] == RUN_NAME
    assert manifest["card"] == CARD_ID

    files = manifest["files"]
    assert len(files) == N_SEAL_MANIFEST_ENTRIES
    paths = [entry["path"] for entry in files]
    assert paths == sorted(paths), (
        "the seal manifest file list is not in sorted relative-path "
        "order"
    )
    assert set(paths) == {
        "SHA256SUMS.txt", "family_integrals.json", "gate3_r30.json",
        "pre_registration.json", "provenance.json", "threshold_checks.json",
    }
    for entry in files:
        assert set(entry) == {"path", "sha256", "bytes"}
        # the recorded digest and byte size match the file on disk
        assert _file_sha256(run_dir / entry["path"]) == entry["sha256"], (
            f"{entry['path']}: the sealed digest does not match the "
            "file on disk; the run dir was modified after the seal"
        )
        assert (run_dir / entry["path"]).stat().st_size == entry["bytes"]
    # self-consistent byte total: the recorded bytes account for the
    # full 30,890-byte seal
    assert sum(entry["bytes"] for entry in files) == SEAL_TOTAL_BYTES
    assert sum(
        (run_dir / entry["path"]).stat().st_size for entry in files
    ) == SEAL_TOTAL_BYTES
    # the sums-file digest recorded in the manifest matches on disk
    sums_entry = next(
        entry for entry in files if entry["path"] == "SHA256SUMS.txt"
    )
    assert sums_entry["sha256"] == SUMS_FILE_SHA256

    assert _file_sha256(run_dir / "seal_manifest.json") == (
        SEAL_MANIFEST_SHA256
    )

    sums_path = run_dir / "SHA256SUMS.txt"
    assert _file_sha256(sums_path) == SUMS_FILE_SHA256, (
        "the sums file drifted from the digest recorded in the seal "
        "manifest; the run dir was modified after the seal"
    )

    raw = sums_path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    lines = text.splitlines()
    assert len(lines) == len(SUMS_TOP_LEVEL_PAYLOADS), (
        "SHA256SUMS.txt covers the drifted payload set"
    )
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format "
            f"(LF, two spaces, relative path): {line!r}"
        )

    entries = [line.split("  ", 1) for line in lines]
    assert [rel for _digest, rel in entries] == list(
        SUMS_TOP_LEVEL_PAYLOADS
    ), "the sums file is not in sorted relative-path order"
    for digest, rel in entries:
        assert digest == SUMS_PAYLOAD_SHA256[rel], (
            f"{rel}: the sums-carrying digest drifted from the pinned "
            "digest"
        )
        # verifies by recomputed sha256 against the file on disk
        assert _file_sha256(run_dir / rel) == digest, (
            f"{rel}: the sealed digest does not match the file on "
            "disk; the run dir was modified after the seal"
        )


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified the 3
    sums-covered files without mismatches, ran the clean interpreter
    (``python -I``) with return code 0, and the re-run reproduced
    ``family_integrals.json`` byte for byte (the recorded sealed and
    recomputed digests match, equal the pinned digest, and equal the
    digest of the file on disk)."""
    record = _load("clean_environment_commands.json")
    assert set(record) == {
        "schema", "run", "card", "verified_utc", "seal_verification",
        "commands", "environment", "probe_returncode",
        "sealed_report_sha256", "recomputed_report_sha256",
        "hashes_match", "probe_tool",
    }
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert UTC_STAMP.match(record["verified_utc"])
    assert record["seal_verification"] == {
        "files_checked": len(SUMS_TOP_LEVEL_PAYLOADS),
        "mismatches": [],
    }
    assert record["probe_returncode"] == 0
    assert record["hashes_match"] is True

    commands = record["commands"]
    assert len(commands) == 1
    command = commands[0]
    assert command["cwd"] == str(MONO_ROOT)
    argv = command["argv"]
    # the clean interpreter: python -I, the nested-tree driver, the
    # integration step, the run dir
    assert argv[1] == "-I"
    assert str(DRIVER_PATH) in argv
    assert "--integrate" in argv
    assert "--run-dir" in argv
    assert argv[argv.index("--run-dir") + 1] == (
        f"dev/00runs/{RUN_NAME}"
    )

    for key in ("python_version", "python_executable", "platform",
                "numpy_version"):
        assert key in record["environment"]

    probe = record["probe_tool"]
    assert probe["path"] == str(DRIVER_PATH)
    assert probe["sha256"] == DRIVER_SHA256
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256

    assert record["sealed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    assert record["recomputed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    for name, digest in CLEAN_SEALED_REPORT_SHA256.items():
        assert _file_sha256(run_dir / name) == digest, (
            f"{name}: the on-disk sealed report drifted from the "
            "clean-environment record"
        )

    assert _file_sha256(run_dir / "clean_environment_commands.json") == (
        CLEAN_ENVIRONMENT_SHA256
    )
