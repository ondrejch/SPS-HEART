"""Round 4, card 20 test-author register: the R45 simplified-alignment
freeze contract pins.

Card ``TASK-20260920-r4-gate4-followup-simplified-alignment`` (the Gate
4 follow-up obligation 1 of the written Gate 4 decision
``TASK-20260920-r4-gate4-decision``; the pre-registered simplification
of plan section 10.4, line 732), phase 1 (no Monte Carlo) plus the
phase-2 re-point to the sealed run
``dev/00runs/r45_simplified_alignment_freeze_2026-09-24/`` (sealed
2026-09-25T00:22:00Z; 25 sealed payload files, 0 mismatches),
mirroring the card-16/17/18 re-point precedent.  This register is the
test-author half of the card: independent pins of the dual-tree run
driver
(``helpers/benchmarking/staged_r45_simplified_alignment_freeze.py``,
byte-identical in the nested tree and the monorepo mirror), of the
hermetic solver test
(``tests/test_staged_r45_simplified_alignment_freeze.py``), of the run
identity, and of the sealed-artifact pins including the R41 source
pins and the sealed follow-up evidence values.  The driver is loaded
once via importlib (its top level is the pre-registration identity
and the import-time constant-kappa enforcement; no events are
generated), and the sealed JSON is read from disk.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist,
  are byte-identical, and sit at the committed digest; the driver's
  own self-computed identity digest agrees; the freeze manifest
  inventory pins both at that digest.
- ``test_driver_identity_constants``: the driver's run identity, the
  file-name constants, the substrate and sealed-dir rel paths, the
  card chain (the Gate 4 decision and this card appended), and the
  frozen-substrate digests.
- ``test_simplified_form_pins``: the constant-kappa form, the zero
  mass-dependence coefficients, the unchanged pre-registered ranges,
  the frozen values on their bounds, and the mass-free kernel
  signatures (the card's acceptance (ii); the enforcement is
  import-time and refuses on any violation).
- ``test_frozen_tuple_r41_source_pins``: the frozen tuple
  (0.0, 4.0, 2.0 GeV^-2, 0.3 GeV) byte-identical to the sealed
  corrected R41 fit report (file sha256
  ``bc419ab1...``, report self-digest ``76a40d34...``; the sealed
  archive is read read-only and skips on a fresh checkout).
- ``test_no_optimizer_and_ramp_pin``: no_fitting true,
  optimizer_invoked false, and the blend ramp start fixed at
  B = 2.680194110727528 GeV (obligation 4).
- ``test_seed_map_partition``: the master seed, the new raw block
  20301056-20301075, the event-index partition 800_000_000 + i, the
  variant-stream rule, the hermetic smoke block (+11000), and the
  seed functions' refusal discipline.
- ``test_solver_test_file_digest``: the hermetic solver test file at
  its committed digest, pinned in the freeze manifest inventory.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 502 entries after the
  Round 5 card 1 test-author re-pin; 501 at the Round 5 card 1
  landing; 499 at the card-21 landing) carries the
  new dual-tree driver in both helper trees and the two new R45 test
  files at their on-disk digests; the card-15/card-19 exceptions stay
  exactly once each; this card's freeze exception names the card
  exactly once with the path-sorted file list; the card-20 naming
  comment is retained in every re-pinned chain file with the card-15
  history).
- ``test_sealed_run_identity``: the sealed run dir constant, the
  driver run identity against it, the T3 statistics, the two variant
  policies, the probe replica, and the eight sealed predecessor dirs.
- ``test_pre_registration_on_disk``: the sealed pre-registration's
  schema, T3 statistics, seed map, frozen parameters, simplified
  form, variant register and policies, sealed-dir verification, and
  the no-fitting clause.
- ``test_frozen_report_on_disk``: the sealed frozen report's schema,
  internal digest, held-out deviance table (all 8 rows improving,
  all named_finding false), delta-AIC with parameter cost, exchange
  bit-identity, depth statistics, the 200-entry named-residual
  register standing class, the standing residuals, the stream pin,
  and the no-fitting clause.
- ``test_gate4_followup_evidence``: the gate record's identity, its
  clause, the frozen parameters and ranges, the conservation summary
  (zero violations over 5,000,000 events), the exchange bit-identity
  block, the deviance table, the held-out verdict (all preserved),
  the delta-AIC with parameter cost, the model-uncertainty carry,
  the standing cross-references, the named-residual register equal
  to the report's, the report-digest cross-reference, and the
  no-fitting clause.
- ``test_seal_manifest_and_checksums``: the seal manifest's 25 sealed
  files (20 replica records plus the five top-level files), the
  checksum file's 25 entries, the 27-file run dir (25 payload plus
  the two meta files), and the full re-hash of all 25 sealed files
  against the checksum digests.
- ``test_clean_environment_record``: the clean-environment
  validation's command list, the byte-identical report re-hash
  (digest equality), the probe replica record, and the pending seal
  state.
- ``test_provenance_on_disk``: the provenance's schema, the frozen
  R41 parameter source, the simplified-form record, the substrate
  digests, the recorded driver digest (the committed dual-tree
  digest: the run pre-registered against the committed driver), and
  the eight sealed predecessor dirs verified.

Hermetic: standard library, the committed config and catalog, and the
sealed archives (read-only) only; no SPS tables, no event generation,
no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import sealed_mismatches, skip_unless_sealed
INVENTORY_COUNT = _inventory_count()

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking"
    / "staged_r45_simplified_alignment_freeze.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r45_simplified_alignment_freeze.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r45_simplified_alignment_freeze_2026-09-24")
R41_ARCHIVE = (MONO_ROOT / "dev" / "00runs"
               / "r41_nonstrange_pilot_fit_2026-09-23")
R44_ARCHIVE = (MONO_ROOT / "dev" / "00runs"
               / "r44_frozen_nonstrange_validation_2026-09-23")

RUN_DIR_REL = "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r45_simplified_alignment_freeze.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-gate4-followup-simplified-alignment"

#: The committed dual-tree driver digest (byte-identical in both
#: trees; the freeze manifest inventory pins both copies here).
DRIVER_SHA256 = (
    "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10"
)

#: The hermetic solver test file's committed digest (the Round 5
#: card 1 count re-pin 499 -> 501 touched the INVENTORY_COUNT
#: constant, so the digest moved from the card-21 landing digest
#: d368a841a45f4534b537874873975c5f47ed6db3ac1dbd6f6fcfcf63e7dc4df7;
#: the Round 5 card 1 test-author count re-pin 501 -> 502 touched the
#: same constant, so the digest moved from the Round 5 card 1 landing
#: digest 0336a9688354a4e0...).
#: the Round 5 card 2 count re-pin 502 -> 504 touched the same constant (TASK-20260925-r5-exchange-quantum-number-ledger).
#: the Round 5 card 2 test-author count re-pin 504 -> 505 touched the same constant (TASK-20260925-r5-exchange-quantum-number-ledger-test-author).
#: the Round 5 card 3 count re-pin 508 -> 510 touched the same constant (TASK-20260925-r5-intact-low-mass-remnants).
#: the Round 5 card 5 test-author count re-pin 516 -> 517 touched the same constant (TASK-20260925-r5-single-diffraction-test-author).
#: re-pinned by the Round 5 card 8 523 -> 525 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-multiple-neutral-excitations).
#: re-pinned by the Round 5 card 8 test-author 525 -> 526 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-multiple-neutral-excitations-test-author).
#: re-pinned by the Round 5 card 9 526 -> 528 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner).
#: re-pinned by the Round 5 card 9 test-author 528 -> 529 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner-test-author).
SOLVER_TEST_SHA256 = (
    "5bfa45528397dff941c01976a505609b07bdbca1d6ab8096ad98ba0adf3c3a91"
)

#: The sealed corrected R41 fit report (the frozen-parameter source):
#: the file digest (byte identity) and the report's self-digest (the
#: gate cross-reference of the sealed R41).
R41_FIT_REPORT_FILE_SHA256 = (
    "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
)
R41_FIT_REPORT_DIGEST = (
    "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
)
R41_FROZEN_TUPLE = (0.0, 4.0, 2.0, 0.3)

#: The pre-registered ranges (unchanged from the corrected R41
#: pre-registration; never extended, never relaxed).
RANGES = {
    "kappa_B": (0.0, 4.0),
    "kappa_m": (0.0, 4.0),
    "b_dec_gev2": (0.0, 2.0),
    "baryonic_nominal_width_gev": (0.3, 0.7),
}

#: The frozen substrate (rel, digest) -- the driver's own table
#: (read-only; verified against disk).
SUBSTRATE = {
    "cascade_decay": (
        "dev/mixing2006/na49_mixing/cascade_decay.py",
        "f63b3649039f12c1b65c87b82b64433e8c746679b90050d9e72eb0110fd923e2",
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
        "65eadd60c3e06007218d83f6baa5f12f9088caacd24512b3de48c594d4a3063e",
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
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563",
    ),
    "card2_engine": (
        "dev/mixing2006/na49_mixing/threshold_metric.py",
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    ),
    "catalog": (
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd",
    ),
    "continuum_config": (
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1",
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

# ----------------------------------------------------------------------
# Pinned sealed-run values (read from disk, verified on disk)
# ----------------------------------------------------------------------

#: The seal manifest's file list (20 replica records plus the five
#: top-level files), in seal order.
SEALED_FILES = (
    "clean_environment_commands.json",
    "frozen_report.json",
    "gate4_followup_simplified.json",
    "pre_registration.json",
    "provenance.json",
) + tuple(f"replica_{r}.json" for r in (
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 3, 4, 5, 6, 7, 8, 9,
))

#: The on-disk digest of every sealed file (the checksum file's 25
#: entries plus the seal manifest's own digest).
SEALED_FILE_DIGESTS = {
    "clean_environment_commands.json": (
        "92439c11056e42eef8ebed4f0370f5de1d0f882326bdae69a161cea438be0c72"
    ),
    "frozen_report.json": (
        "880c8d34d2839bfc2f34e5faf0792ae427964d4d6111dcbe0c2be8faf39ca174"
    ),
    "gate4_followup_simplified.json": (
        "1da69b6059c2cdaaad9d8271465fd9ed718d1e54594ed36e82cb68e7cc6aa869"
    ),
    "pre_registration.json": (
        "87272026b7a5fe8a30cd5ff7c411e1a5222b258cf0e385a8c7e3aeae567bd279"
    ),
    "provenance.json": (
        "62363527346fb0b1e92311812830208bd951afeacbfd6ed94e8b70556f6eded9"
    ),
    "replica_0.json": (
        "8db6c5613e24a0fad112582ca53348b74233a210f47f89fc0bba5b5f4bcb3f95"
    ),
    "replica_1.json": (
        "29c23c62ff29877397bde04b3a79bba4e0b7e67b197d0fad9e4c520040fbe716"
    ),
    "replica_10.json": (
        "6f8fac67456a0c9ecb82478cb28230509ecea46ff2d4038632fab88b36dab888"
    ),
    "replica_11.json": (
        "caf6eaf3f762f8b5ff8c3f78c24847618c73f6d31146594f51f69e0bbdaf259f"
    ),
    "replica_12.json": (
        "e2836a9ec77da47bd48e0c7ba7e9b6bdd3bc1a41db1d95f3ee8b4c463ab4b2a4"
    ),
    "replica_13.json": (
        "604636ee041190ce0b0b6f4171d326e2b11358cfced48455204c01c2e4076b64"
    ),
    "replica_14.json": (
        "b15264c6697073f56e176c23bde74bc454f4cb6dbbf5aab25b33a3db2f1ee615"
    ),
    "replica_15.json": (
        "68e39f016c21ba7daa49f4aea40cad77294e210af5a71188ee0b43b72914e618"
    ),
    "replica_16.json": (
        "09fea054e2e5aa0677dd8a2ee47ec046e168dcf11f577da643844ba963c0730f"
    ),
    "replica_17.json": (
        "8706f2f5c21a0ec1a9b453e949ba171e469a8eee75be06a47e08fdb774f8e17f"
    ),
    "replica_18.json": (
        "301a2c34ad738ffcb79f4f1bb01d0e95d3e9aaf5eda49e005e8799a179eb901e"
    ),
    "replica_19.json": (
        "920c3d6076f31870021170db7ea2a927fe621707a05e96a22c4e2de45a7fbf10"
    ),
    "replica_2.json": (
        "59ce09d128ec793f8df45195e5775492da7765c1f8f806f77a11c37a5a999348"
    ),
    "replica_3.json": (
        "a795689673e97058bd6e2b26098c8d6d3ab2582ab4a1a251f06d7379f72f5f4b"
    ),
    "replica_4.json": (
        "0cffa8d7eaffa7d440728e66e5ddf1ae85c6e549d332e582c217a8b65ed082e6"
    ),
    "replica_5.json": (
        "0c5285cf15e007051ae1e0aab86d77fbd4b4f528070d5e3b838a10218a6d8b93"
    ),
    "replica_6.json": (
        "b0fe856abc4b5add032e5c27b97ab7326a2c977048d99d1a172c6c8eb198430a"
    ),
    "replica_7.json": (
        "3fdde970acd6b2ef1dd0b5e284d48b205928c575a7a1f4e2d530b8e639c6575e"
    ),
    "replica_8.json": (
        "70f0f6b43c1d8172fc77fd677fb499d7ee2e663c8e901d10731bce042cc32c73"
    ),
    "replica_9.json": (
        "6c674cad2db474976aef758ac77c60334a44e29444879695b49e9b92529cb362"
    ),
    "seal_manifest.json": (
        "730943b1505536add004dd13069e706234b631ed184b746f4b11158f0ed007db"
    ),
}

#: The checksum file's own digest (excluded from its own entries).
CHECKSUMS_SHA256 = (
    "abe5927a63d21e3be05cc1c79204cee765b229619c0960262451689b415060f4"
)

#: The frozen report's internal digest (the report bytes carry no
#: wall-clock fields; the clean-environment re-run reproduces it).
FROZEN_REPORT_DIGEST = (
    "dac934160be3d7ca92981ccbadbd8fa9a5d01ecdb88ee2e7a72a10121ea6ab66"
)

#: The held-out deviance table at the sealed values: per observable
#: (aligned_simplified, isotropic_sequential_baseline).  Every row
#: improves or preserves against the baseline; no row is a named
#: finding.
HELD_OUT_DEVIANCE = {
    "antiproton_multiplicity": (
        22435.10349170713, 22435.10349170713),
    "antiproton_spectrum": (
        33612.39765457797, 33612.39765457797),
    "pi_minus_multiplicity": (
        22717.096370810505, 118400.99744006843),
    "pi_minus_spectrum": (
        319776.8257640948, 692695.8691332908),
    "pi_plus_multiplicity": (
        85613.08387277764, 172927.5911364837),
    "pi_plus_spectrum": (
        343759.6040558467, 783203.7328119981),
    "proton_multiplicity": (
        1600.6757332404843, 2090.4500159898103),
    "proton_spectrum": (
        282748.3799584325, 292082.2747851334),
}

#: The held-out comparison rows at the sealed values: per observable
#: (aligned, baseline, aligned-minus-baseline difference, pooled MC
#: SE of the difference).  All eight rows improve (every difference
#: is non-positive); every named_finding is false.
HELD_OUT_ROWS = {
    "antiproton_multiplicity": (
        22435.10349170713, 22435.10349170713, 0.0, 1e-12),
    "antiproton_spectrum": (
        33612.39765457797, 33612.39765457797, 0.0, 1e-12),
    "pi_minus_multiplicity": (
        22717.096370810505, 118400.99744006843,
        -95683.90106925793, 258.4235430524471),
    "pi_minus_spectrum": (
        319776.8257640948, 692695.8691332908,
        -372919.043369196, 1272.123884457511),
    "pi_plus_multiplicity": (
        85613.08387277764, 172927.5911364837,
        -87314.50726370607, 390.9973214792377),
    "pi_plus_spectrum": (
        343759.6040558467, 783203.7328119981,
        -439444.1287561514, 1310.9781864269169),
    "proton_multiplicity": (
        1600.6757332404843, 2090.4500159898103,
        -489.77428274932595, 31.526293047527755),
    "proton_spectrum": (
        282748.3799584325, 292082.2747851334,
        -9333.894826700911, 782.5053951739449),
}

#: The delta-AIC against the isotropic sequential baseline at the
#: sealed values (the parameter cost included: k = 4 aligned against
#: k = 1 baseline; the verdict is the held-out likelihood evidence
#: of the Gate 4 retain).
DELTA_AIC = -1005179.2495677616
K_ALIGNED = 4
K_BASELINE = 1
AIC_ALIGNED = 1112271.1669014879
AIC_BASELINE = 2117450.4164692494
M2L_ALIGNED = 1112263.1669014879
M2L_BASELINE = 2117448.4164692494

#: The conservation summary at the sealed values (zero violations
#: over 5,000,000 events, zero guard activations).
CONSERVATION_EVENTS = 5_000_000
CONSERVATION_WORST_ENERGY_CLOSURE_GEV = 3.907985046680551e-13

#: The exchange bit-identity at the sealed values (verified at run
#: scale over 5,000,000 events x 2 variants).
EXCHANGE_VARIANTS = 2
EXCHANGE_EVENTS = 5_000_000
EXCHANGE_LAYERS = ["topology", "exchange", "spectral_state"]

#: The cascade depth statistics at the sealed values (per variant;
#: 10,000,000 source decays each).
DEPTH_STATISTICS = {
    "aligned_simplified": {
        "guard_activations": 0, "max": 5, "mean": 2.5865198,
        "n_decays": 10_000_000, "p95": 4,
    },
    "isotropic_sequential_baseline": {
        "guard_activations": 0, "max": 5, "mean": 2.5698905,
        "n_decays": 10_000_000, "p95": 4,
    },
}

#: The named-residual register standing class at the sealed values:
#: 200 entries, indices 1..200, all pi_plus_spectrum under the
#: aligned_simplified variant (the report and gate records carry the
#: identical register).
NAMED_REGISTER_COUNT = 200
NAMED_REGISTER_MAX_POSITIVE_SIGMA = 62.11245467656278
NAMED_REGISTER_MAX_POSITIVE_BIN = 89
NAMED_REGISTER_MAX_ABS_SIGMA = -75.30965523271244
NAMED_REGISTER_MAX_ABS_BIN = 188

#: The standing residuals at the sealed values: the R43
#: b_transport_seagull non-positive finding and the obligation-4
#: R32 spread carry.
STANDING_B_TRANSPORT_SEAGULL_OBSERVED = -2.3959765824349295e-05
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}

#: The standing cross-reference digests (the sealed R43 gate's
#: beyond-isotropic register and the sealed R41 replica-block
#: stability table; both all_pass false by construction: the
#: carried findings are the evidence that triggered the
#: pre-registered simplify path).
STANDING_CROSS_REFERENCE_DIGEST = (
    "6541159fefe410574d27a8fc0b0d1fd4f0f625beee007ccb8ad1b3efe6df4532"
)
STABILITY_CROSS_REFERENCE_DIGEST = (
    "c7d36cb27697c13b0767234e4a80471d95eaaa9c608afcd275a0d428c4b4a27c"
)

#: The clean-environment validation timestamp (the validation runs
#: before the seal, so the record is sealed content).
CLEAN_VALIDATED_UTC = "2026-09-25T00:21:47Z"
SEALED_UTC = "2026-09-25T00:22:00Z"


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r45_simplified_alignment_freeze", NESTED_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r45 = _load_driver()


@pytest.fixture(scope="module")
def pre_registration():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "frozen_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_followup_simplified.json")
        .read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def seal_manifest():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "seal_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean_environment():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "clean_environment_commands.json")
        .read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def provenance():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "provenance.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# The driver pins
# ----------------------------------------------------------------------
def test_dual_tree_driver_byte_identity():
    """Both driver copies exist, are byte-identical, and sit at the
    committed digest; the driver's self-computed identity digest
    agrees; the freeze manifest pins both at that digest."""
    assert NESTED_DRIVER.is_file(), NESTED_DRIVER
    assert OUTER_DRIVER.is_file(), OUTER_DRIVER
    nested_digest = hashlib.sha256(
        NESTED_DRIVER.read_bytes()).hexdigest()
    outer_digest = hashlib.sha256(
        OUTER_DRIVER.read_bytes()).hexdigest()
    assert nested_digest == outer_digest
    assert nested_digest == DRIVER_SHA256
    assert r45.DRIVER_SHA256 == DRIVER_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == nested_digest
    assert inventory[DRIVER_REL_OUTER] == outer_digest


def test_driver_identity_constants():
    """The driver's run identity, file-name constants, rel paths, card
    chain, and the frozen substrate digests on disk."""
    assert r45.RUN_ID == "R45"
    assert r45.CARD == SOLVER_CARD
    assert r45.RUN_NAME == "r45_simplified_alignment_freeze_2026-09-24"
    assert r45.DEFAULT_RUN_DIR == (
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24")
    assert r45.GATE_RECORD_NAME == "gate4_followup_simplified.json"
    assert r45.REPORT_NAME == "frozen_report.json"
    assert r45.R41_RUN_DIR == \
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"
    assert r45.R44_RUN_DIR == \
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"
    assert r45.CARD_CHAIN[-2:] == (
        "TASK-20260920-r4-gate4-decision",
        SOLVER_CARD,
    )
    # the frozen substrate byte stability on disk
    for key, (rel, expected) in SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), (key, path)
        assert _sha256_file(path) == expected, key


def test_simplified_form_pins():
    """The constant-kappa form, the zero mass-dependence coefficients,
    the unchanged pre-registered ranges, the frozen values on their
    bounds, and the mass-free kernel signatures (the card's acceptance
    (ii))."""
    assert r45.SIMPLIFIED_FORM == {
        "kappa_B": "constant", "kappa_M": "constant",
        "mass_dependence": "fixed at zero, not fit",
    }
    assert r45.MASS_DEPENDENCE_COEFFICIENTS == {
        "kappa_b_mass_slope_per_gev": 0.0,
        "kappa_m_mass_slope_per_gev": 0.0,
    }
    assert r45.PRE_REGISTERED_RANGES == RANGES
    status = r45._SIMPLIFIED_FORM_STATUS
    assert status["pass"] is True
    on_bounds = status["frozen_values_on_bounds"]
    assert on_bounds["kappa_B"] == {"value": 0.0, "range": [0.0, 4.0]}
    assert on_bounds["kappa_m"] == {"value": 4.0, "range": [0.0, 4.0]}
    assert on_bounds["b_dec"] == {"value": 2.0, "range": [0.0, 2.0]}
    assert on_bounds["baryonic_nominal_width"] == {
        "value": 0.3, "range": [0.3, 0.7]}
    for name, params in status["kernel_signatures_mass_free"].items():
        assert all("mass" not in p.lower() for p in params), (name, params)
    scan = status["mass_independence_scan"]
    assert scan["pass"] is True and scan["worst_mass_spread"] == 0.0
    assert all(row["zero_mass_slope"] is True for row in scan["rows"])
    # the range bounds are never relaxed: a frozen value outside or
    # interior to its range refuses
    saved = r45.ALIGNED_POLICY.kappa_m
    try:
        r45.ALIGNED_POLICY.kappa_m = 3.0   # interior
        with pytest.raises(SystemExit):
            r45.assert_constant_kappa_form()
        r45.ALIGNED_POLICY.kappa_m = 5.0   # outside
        with pytest.raises(SystemExit):
            r45.assert_constant_kappa_form()
    finally:
        r45.ALIGNED_POLICY.kappa_m = saved


def test_frozen_tuple_r41_source_pins():
    """The frozen tuple is byte-identical to the sealed corrected R41
    fit report (the file digest and the report self-digest match the
    sealed archive; read-only, skips on a fresh checkout)."""
    assert (r45.R41_FIT_REPORT_FILE_SHA256, r45.R41_FIT_REPORT_DIGEST) \
        == (R41_FIT_REPORT_FILE_SHA256, R41_FIT_REPORT_DIGEST)
    fit_path = R41_ARCHIVE / "fit_report.json"
    if not fit_path.is_file():
        pytest.skip("the sealed R41 archive is absent on this checkout")
    assert _sha256_file(fit_path) == R41_FIT_REPORT_FILE_SHA256
    report = json.loads(fit_path.read_text(encoding="utf-8"))
    assert report.get("digest") == R41_FIT_REPORT_DIGEST
    observed = tuple(float(report["fitted"][k]) for k in (
        "kappa_B", "kappa_m", "b_dec", "baryonic_nominal_width"))
    assert observed == R41_FROZEN_TUPLE
    # the same tuple is what the driver's production assertion expects
    frozen = r45.assert_frozen_parameters(r45.R41_FIT_REPORT_FIXTURE)
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == R41_FROZEN_TUPLE
    # the prior frozen-point validation (the sealed R44 archive) ran the
    # same tuple; its gate record is present and verifies read-only
    gate44 = R44_ARCHIVE / "gate4_r44.json"
    if gate44.is_file():
        sealed = json.loads(gate44.read_text(encoding="utf-8"))
        prior = sealed.get("frozen_parameters") or {}

        def _f(name):
            value = prior.get(name)
            assert value is not None, name
            return float(value)

        assert (_f("kappa_B"), _f("kappa_m"), _f("b_dec"),
                _f("baryonic_nominal_width")) == R41_FROZEN_TUPLE
        assert prior.get("refit") is False


def test_no_optimizer_and_ramp_pin():
    """No optimizer (the round's two pre-registered optimizer uses were
    consumed by cards 16 and 17) and the blend ramp start fixed at B
    (obligation 4)."""
    pre, _frozen, _disjointness = r45._pre_registration_payload(
        smoke=True)
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False
    assert pre["frozen_parameters"]["refit"] is False
    record = r45.simplified_form_record()
    assert record["no_fitting"] is True
    assert record["optimizer_invoked"] is False
    assert "card 16 the first, card 17 the second and final" in \
        record["optimizer_note"]
    assert r45.BLEND_RAMP_START_GEV == 2.680194110727528
    assert "obligation 4" in r45.BLEND_RAMP_START_STATUS


def test_seed_map_partition():
    """The master seed, the new raw block 20301056-20301075, the
    event-index partition 800_000_000 + i, the variant-stream rule, the
    smoke block (+11000), and the refusal discipline."""
    assert r45.MASTER_SEED == 20260920
    assert r45.RAW_REPLICA_SEEDS == tuple(20301056 + k for k in range(20))
    assert r45.RAW_REPLICA_SEEDS[-1] == 20301075
    assert r45.EVENT_INDEX_RULE == "800_000_000 + i, i < 250,000"
    assert r45.EVENT_INDEX_BASE == 800_000_000
    assert r45.SMOKE_RAW_SEED == 20311920
    assert r45.SMOKE_KERNEL_REPLICA_SLOT == 11_000
    assert r45.SMOKE_EVENT_INDEX_BASE == 890_000_000
    assert "20301056-20301075" in r45.RAW_SEED_RULE
    assert "derive_substreams(20260920, k, 800_000_000 + i)" in \
        r45.EXCHANGE_SEED_RULE
    assert "variant_stream(20260920, 800_000_000 + i, v)" in \
        r45.VARIANT_STREAM_RULE
    report = r45.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r45_raw_seed_block"] == [20301056, 20301075]
    assert report["shared_raw_seeds"] == {}
    with pytest.raises(SystemExit):
        r45.raw_replica_seed(20)


def test_solver_test_file_digest():
    """The hermetic solver test file at its committed digest, pinned in
    the freeze manifest inventory."""
    solver_path = MONO_ROOT / SOLVER_TEST_REL
    assert solver_path.is_file(), solver_path
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_tree_sha256"][SOLVER_TEST_REL] \
        == SOLVER_TEST_SHA256


# ----------------------------------------------------------------------
# The freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
#: The twenty-one register-chain files whose inventory-count pins this
#: card's re-pin bump (491 -> 495, +4 entries): the sixteen chain
#: register files (cards 1, 4, 5, 6, 7, 8, 9, 10, 11, 12) plus the
#: card-13 solver test's and the card-13 test-author register's own
#: count constants, the card-14 solver test's and the card-14
#: test-author register's own count constants, the card-15/19 solver
#: test's own count constant, and the card-12/card-14/card-15 solver
#: registers' chain lists.  Every chain file carries the card-20 naming
#: comment appended to the card-15/card-19 history.
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r44_frozen_nonstrange_validation.py",
)


# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).




# ----------------------------------------------------------------------
# The sealed run identity (the phase-2 re-point)
# ----------------------------------------------------------------------
def test_sealed_run_identity():
    """The sealed run dir constant, the driver run identity against
    it, the T3 statistics, the two variant policies, the probe
    replica, and the eight sealed predecessor dirs."""
    assert RUN_DIR.name == "r45_simplified_alignment_freeze_2026-09-24"
    assert r45.RUN_NAME == RUN_DIR.name
    assert r45.DEFAULT_RUN_DIR == RUN_DIR_REL
    assert r45.RUN_ID == "R45"
    assert r45.CARD == SOLVER_CARD
    assert r45.PREREGISTRATION_NAME == "pre_registration.json"
    assert r45.PROVENANCE_NAME == "provenance.json"
    assert r45.CLEAN_ENV_NAME == "clean_environment_commands.json"
    assert r45.SHA256SUMS_NAME == "SHA256SUMS.txt"
    assert r45.SEAL_NAME == "seal_manifest.json"
    assert r45.RECORD_FMT == "replica_{}.json"
    assert r45.SMOKE_PREFIX == "smoke_"
    # the T3 statistics (plan section 5.1 line 255)
    assert r45.T3_REPLICAS == 20
    assert r45.T3_EVENTS_PER_REPLICA == 250_000
    # the two variant policies: v0 aligned at the frozen R41 point,
    # v1 the isotropic sequential baseline
    assert [(v.name, v.variant_id, v.kappa_b, v.kappa_m, v.b_dec)
            for v in r45.VARIANTS] == [
        ("aligned_simplified", 0, 0.0, 4.0, 2.0),
        ("isotropic_sequential_baseline", 1, 0.0, 0.0, 1.0),
    ]
    assert (r45.ALIGNED_POLICY.name, r45.ALIGNED_POLICY.kappa_b,
            r45.ALIGNED_POLICY.kappa_m) == (
        "aligned_simplified", 0.0, 4.0)
    # the clean-environment validation's pre-registered probe replica
    assert r45.PROBE_REPLICA == 0
    # the eight sealed predecessor archives consumed read-only
    assert r45.SEALED_RUN_DIRS == (
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23",
        "dev/00runs/r42_recoil_decay_separation_2026-09-23",
        "dev/00runs/r43_cascade_ablation_2026-09-23",
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23",
    )


# ----------------------------------------------------------------------
# The sealed pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk(pre_registration):
    """The sealed pre-registration's identity, T3 statistics, seed
    map, frozen parameters, simplified form, variant register and
    policies, sealed-dir verification, and the no-fitting clause."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze."
        "pre_registration/1")
    assert pre_registration["run"] == "R45"
    assert pre_registration["card"] == SOLVER_CARD
    assert pre_registration["mode"] == "production"
    assert pre_registration["written_before_generation"] is True
    # the T3 statistics (plan section 5.1 line 255)
    stats = pre_registration["statistics"]
    assert stats["replicas"] == 20
    assert stats["events_per_replica"] == 250_000
    assert stats["total_events"] == 5_000_000
    # the pre-registered seed map (the new disjoint block)
    seeds = pre_registration["seed_map"]
    assert seeds["master_seed"] == 20260920
    assert seeds["raw_replica_seeds"] == list(
        20301056 + k for k in range(20))
    assert seeds["event_index_rule"] == "800_000_000 + i, i < 250,000"
    assert "20301056-20301075" in seeds["raw_seed_rule"]
    assert "derive_substreams(20260920, k, 800_000_000 + i)" in \
        seeds["exchange_seed_rule"]
    assert "variant_stream(20260920, 800_000_000 + i, v)" in \
        seeds["variant_stream_rule"]
    assert seeds["smoke"]["raw_seed"] == 20311920
    assert seeds["smoke"]["kernel_replica_slot"] == 11_000
    assert seeds["smoke"]["event_index_base"] == 890_000_000
    # the frozen parameters (the sealed R41 tuple, no refit)
    frozen = pre_registration["frozen_parameters"]
    assert (frozen["kappa_B"], frozen["kappa_m"],
            frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == R41_FROZEN_TUPLE
    assert frozen["refit"] is False
    # the simplified form (constant kappa, zero mass slopes)
    form = pre_registration["simplified_form"]
    assert form["simplified_form"] == {
        "kappa_B": "constant", "kappa_M": "constant",
        "mass_dependence": "fixed at zero, not fit",
    }
    assert form["mass_dependence_coefficients"] == {
        "kappa_b_mass_slope_per_gev": 0.0,
        "kappa_m_mass_slope_per_gev": 0.0,
    }
    assert form["frozen"] == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec_gev2": 2.0,
        "baryonic_nominal_width_gev": 0.3,
    }
    # the two-variant register and policies
    assert pre_registration["variant_register"] == [
        "aligned_simplified", "isotropic_sequential_baseline"]
    policies = pre_registration["variant_policies"]
    assert policies["aligned_simplified"]["kappa_b"] == 0.0
    assert policies["aligned_simplified"]["kappa_m"] == 4.0
    assert policies["aligned_simplified"]["variant_id"] == 0
    assert policies["isotropic_sequential_baseline"]["variant_id"] == 1
    # the no-fitting clause (plan section 24; the round's two
    # pre-registered optimizer uses were consumed by cards 16/17)
    assert pre_registration["no_fitting"] is True
    assert pre_registration["optimizer_invoked"] is False
    # the eight sealed predecessor dirs, all verified
    status = pre_registration["sealed_run_dirs_status"]
    assert set(status) == set(r45.SEALED_RUN_DIRS)
    for rel, state in status.items():
        assert state["mismatches"] == 0, rel
        assert state["verified"] is True, rel
    assert status[
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"] == (
            {"entries": 13, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"] == (
            {"entries": 25, "mismatches": 0, "verified": True})
    # the pre-registration's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        SEALED_FILE_DIGESTS["pre_registration.json"])


# ----------------------------------------------------------------------
# The sealed frozen report
# ----------------------------------------------------------------------
def test_frozen_report_on_disk(frozen_report, gate_record):
    """The sealed frozen report's schema, internal digest, held-out
    deviance table (all 8 rows improving, all named_finding false),
    delta-AIC with parameter cost, exchange bit-identity, depth
    statistics, the 200-entry named-residual register standing class,
    the standing residuals, the stream pin, and the no-fitting
    clause."""
    assert frozen_report["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze.report/1")
    assert frozen_report["run"] == "R45"
    assert frozen_report["card"] == SOLVER_CARD
    assert frozen_report["mode"] == "production"
    assert frozen_report["digest"] == FROZEN_REPORT_DIGEST
    assert frozen_report["all_recorded_observables_finite"] is True
    assert frozen_report["events_checked"] == 5_000_000
    assert frozen_report["replicas"] == 20
    assert frozen_report["variants"] == [
        "aligned_simplified", "isotropic_sequential_baseline"]
    # the per-observable deviance table (aligned, baseline)
    table = frozen_report["deviance_table"]
    assert set(table) == set(HELD_OUT_DEVIANCE)
    for obs, (aligned, baseline) in HELD_OUT_DEVIANCE.items():
        assert table[obs]["aligned_simplified"] == aligned, obs
        assert table[obs]["isotropic_sequential_baseline"] == baseline, (
            obs)
    # the held-out verdict (all preserved at the 2.0 x pooled SE
    # threshold; every row improves; no named findings)
    held = frozen_report["held_out"]
    assert held["all_preserved"] is True
    assert held["threshold_factor"] == 2.0
    rows = held["rows"]
    assert set(rows) == set(HELD_OUT_ROWS)
    for obs, (aligned, baseline, diff, pooled_se) in (
            HELD_OUT_ROWS.items()):
        row = rows[obs]
        assert row["aligned_deviance"] == aligned, obs
        assert row["baseline_deviance"] == baseline, obs
        assert row["delta_aligned_minus_baseline"] == diff, obs
        assert row["pooled_se"] == pooled_se, obs
        assert row["degrades"] is False, obs
        assert row["beyond_2x_se"] is False, obs
        assert row["named_finding"] is False, obs
        assert diff <= 0.0, obs
    # the sector check (acceptance (iii) item 6): no sector degrades
    # by more than 5% without a stated physics reason
    sectors = frozen_report["sector_checks"]
    for group in ("species_deviance_5pct", "feeddown_fraction_5pct"):
        for name, check in sectors[group].items():
            assert check["flagged"] is False, (group, name)
    # the delta-AIC with the parameter cost included
    delta = frozen_report["delta_aic"]
    assert delta["delta_aic"] == DELTA_AIC
    assert delta["favored"] is True
    assert delta["k_aligned"] == K_ALIGNED
    assert delta["k_baseline"] == K_BASELINE
    assert delta["aic_aligned"] == AIC_ALIGNED
    assert delta["aic_baseline"] == AIC_BASELINE
    assert delta["aligned_minus2lnl"] == M2L_ALIGNED
    assert delta["baseline_minus2lnl"] == M2L_BASELINE
    assert delta["parameter_cost"] == "included"
    # the exchange bit-identity (verified at run scale, all 20
    # replicas x 250,000 events x 2 variants)
    bit = frozen_report["exchange_bit_identity"]
    assert bit["pass"] is True
    assert bit["variants"] == EXCHANGE_VARIANTS
    assert bit["events"] == EXCHANGE_EVENTS
    assert bit["layers"] == EXCHANGE_LAYERS
    assert len(bit["per_replica"]) == 20
    for entry in bit["per_replica"]:
        assert entry["variants_digests_equal"] is True
        assert entry["n_events"] == 250_000
    # the cascade depth statistics
    depth = frozen_report["depth_statistics"]
    assert set(depth) == set(DEPTH_STATISTICS)
    for variant, stats in DEPTH_STATISTICS.items():
        for key, value in stats.items():
            assert depth[variant][key] == value, (variant, key)
        assert depth[variant]["convention"].startswith(
            "the cascade depth of the variant's decay machinery")
    # the 200-entry named-residual register (the standing class: all
    # pi_plus_spectrum under the aligned variant; the held-out
    # comparison itself carries no named findings)
    register = frozen_report["named_residual_register"]
    assert len(register) == NAMED_REGISTER_COUNT
    assert [entry["index"] for entry in register] == list(
        range(1, NAMED_REGISTER_COUNT + 1))
    assert {entry["observable"] for entry in register} == {
        "pi_plus_spectrum"}
    assert {entry["variant"] for entry in register} == {
        "aligned_simplified"}
    assert {entry["kind"] for entry in register} == {
        "residual_above_tolerance"}
    by_bin = {entry["bin_index"]: entry["observed_sigma"]
              for entry in register}
    assert by_bin[NAMED_REGISTER_MAX_POSITIVE_BIN] == (
        NAMED_REGISTER_MAX_POSITIVE_SIGMA)
    assert by_bin[NAMED_REGISTER_MAX_ABS_BIN] == (
        NAMED_REGISTER_MAX_ABS_SIGMA)
    assert max(by_bin.values()) == NAMED_REGISTER_MAX_POSITIVE_SIGMA
    assert min(by_bin.values()) == NAMED_REGISTER_MAX_ABS_SIGMA
    # the standing residuals carried into every later comparison
    standing = frozen_report["standing_residuals"]
    assert len(standing) == 2
    assert standing[0]["kind"] == (
        "b_transport_seagull_beyond_isotropic_not_positive")
    assert standing[0]["observed"] == (
        STANDING_B_TRANSPORT_SEAGULL_OBSERVED)
    assert standing[1]["kind"] == "r32_model_uncertainty_carry"
    assert standing[1]["r32_spread"] == OBLIGATION4_R32_SPREAD
    # the gate record carries the identical register and standing
    # residuals
    assert gate_record["named_residual_register"] == register
    assert gate_record["standing_residuals"] == standing
    # the stream pin (the pinned-key replay of card 10's
    # variant_stream contract)
    pin = frozen_report["stream_pin"]
    assert pin["key"]["master_seed"] == 20260920
    assert pin["key"]["event_index"] == 800_000_000
    assert pin["key"]["variant_ids"] == [0, 1]
    assert pin["first_channel_uniform_per_variant"] == [
        0.4112077633635626, 0.8036892319615914]
    # the held-out set is species-separated and disjoint from the
    # R41 training combinations
    check = frozen_report["held_out_set_check"]
    assert check["disjoint"] is True
    assert check["species_separated"] is True
    # the no-fitting clause
    assert frozen_report["no_fitting"] is True
    assert frozen_report["optimizer_invoked"] is False
    # the report's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "frozen_report.json") == (
        SEALED_FILE_DIGESTS["frozen_report.json"])


# ----------------------------------------------------------------------
# The Gate 4 follow-up evidence
# ----------------------------------------------------------------------
def test_gate4_followup_evidence(gate_record, frozen_report):
    """The gate record's identity, its clause, the frozen parameters
    and ranges, the conservation summary (zero violations over
    5,000,000 events), the exchange bit-identity block, the deviance
    table, the held-out verdict (all preserved), the delta-AIC with
    parameter cost, the model-uncertainty carry, the standing
    cross-references, the named-residual register equal to the
    report's, the report-digest cross-reference, and the no-fitting
    clause."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze.gate/1")
    assert gate_record["run"] == "R45 simplified-alignment freeze (2026-09-24)"
    assert gate_record["card"] == SOLVER_CARD
    assert gate_record["gate"] == (
        "gate 4 follow-up obligation 1 (plan section 10.4, line 732)")
    assert gate_record["sealed"] is True
    assert gate_record["events_checked"] == 5_000_000
    # the frozen parameters (the sealed R41 tuple, no refit) and the
    # unchanged pre-registered ranges
    frozen = gate_record["frozen_parameters"]
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec"],
            frozen["baryonic_nominal_width"]) == R41_FROZEN_TUPLE
    assert frozen["refit"] is False
    assert gate_record["pre_registered_ranges"] == {
        key: list(value) for key, value in RANGES.items()}
    assert gate_record["simplified_form"] == {
        "kappa_B": "constant", "kappa_M": "constant",
        "mass_dependence": "fixed at zero, not fit",
    }
    assert gate_record["mass_dependence_coefficients"] == {
        "kappa_b_mass_slope_per_gev": 0.0,
        "kappa_m_mass_slope_per_gev": 0.0,
    }
    # the conservation summary (zero violations over 5,000,000
    # events, zero guard activations)
    conservation = gate_record["conservation"]
    assert conservation["violations"] == 0
    assert conservation["guard_activations"] == 0
    assert conservation["pass"] is True
    assert conservation["events_checked"] == CONSERVATION_EVENTS
    assert conservation["tolerance"] == 1e-09
    assert conservation["worst_energy_closure_gev"] == (
        CONSERVATION_WORST_ENERGY_CLOSURE_GEV)
    # the exchange bit-identity block (verified at run scale)
    bit = gate_record["exchange_bit_identity"]
    assert bit["pass"] is True
    assert bit["variants"] == EXCHANGE_VARIANTS
    assert bit["events"] == EXCHANGE_EVENTS
    assert bit["layers"] == EXCHANGE_LAYERS
    # the per-observable deviance table (equal to the report's)
    table = gate_record["deviance_table"]
    assert set(table) == set(HELD_OUT_DEVIANCE)
    for obs, (aligned, baseline) in HELD_OUT_DEVIANCE.items():
        assert table[obs]["aligned_simplified"] == aligned, obs
        assert table[obs]["isotropic_sequential_baseline"] == baseline, (
            obs)
    assert gate_record["depth_statistics"] == (
        frozen_report["depth_statistics"])
    assert gate_record["likelihood_table"] == (
        frozen_report["likelihood_table"])
    # the held-out verdict (all preserved; no sector degrades)
    held = gate_record["held_out"]
    assert held["all_preserved"] is True
    assert held == frozen_report["held_out"]
    conditions = gate_record["conditions"]
    assert conditions["3_held_out"]["all_preserved"] is True
    assert conditions["3_held_out"]["sigma_tolerance"] == 3.0
    assert conditions["4_sector_degradation_5pct"] == []
    # the standing cross-reference (the sealed R43 gate's
    # beyond-isotropic register; all_pass false carries the standing
    # residual, it does not fail this run)
    cross = conditions["2_standing_residual_cross_reference"]
    assert cross["all_pass"] is False
    assert cross["digest"] == STANDING_CROSS_REFERENCE_DIGEST
    assert cross["rows"]["b_transport_seagull"][
        "delta_v2_minus_v3"] == STANDING_B_TRANSPORT_SEAGULL_OBSERVED
    # the stability cross-reference (the sealed R41 replica-block
    # table; the kappa_B/kappa_m FAILs are the evidence that
    # triggered the pre-registered simplify path)
    stability = conditions["5_replica_stability_cross_reference"]
    assert stability["all_pass"] is False
    assert stability["digest"] == STABILITY_CROSS_REFERENCE_DIGEST
    assert stability["parameters"]["kappa_B"]["pass"] is False
    assert stability["parameters"]["kappa_m"]["pass"] is False
    # the delta-AIC with the parameter cost included
    delta = gate_record["delta_aic_record"]
    assert delta["delta_aic"] == DELTA_AIC
    assert delta["favored"] is True
    assert delta["k_aligned"] == K_ALIGNED
    assert delta["k_baseline"] == K_BASELINE
    assert delta["parameter_cost"] == "included"
    assert gate_record["conditions"]["6_delta_aic"][
        "aligned_vs_baseline"] == DELTA_AIC
    # the model-uncertainty carry (obligation 4)
    carry = gate_record["model_uncertainty_carry"]
    assert carry["obligation_4"] is True
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    # the named-residual register (the standing class, equal to the
    # report's)
    register = gate_record["named_residual_register"]
    assert register == frozen_report["named_residual_register"]
    assert len(register) == NAMED_REGISTER_COUNT
    # the report digest cross-reference (the gate's digest equals the
    # frozen report's internal digest)
    assert gate_record["report_digest"] == FROZEN_REPORT_DIGEST
    assert gate_record["report_digest"] == frozen_report["digest"]
    # the support boundary (both variants multi-vertex, no
    # regeneration)
    assert gate_record["support_boundary"][
        "regeneration_required"] is False
    assert gate_record["variants"] == [
        "aligned_simplified", "isotropic_sequential_baseline"]
    # the no-fitting clause
    assert gate_record["no_fitting"] is True
    assert gate_record["optimizer_invoked"] is False
    # the gate record's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "gate4_followup_simplified.json") == (
        SEALED_FILE_DIGESTS["gate4_followup_simplified.json"])


# ----------------------------------------------------------------------
# The seal manifest and the checksum file
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums(seal_manifest):
    """The seal manifest's 25 sealed files (20 replica records plus
    the five top-level files), the checksum file's 25 entries, the
    27-file run dir (25 payload plus the two meta files), and the
    full re-hash of all 25 sealed files against the checksum
    digests."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze.seal/1")
    assert seal_manifest["run"] == "R45"
    assert seal_manifest["card"] == SOLVER_CARD
    assert seal_manifest["record_count"] == 20
    assert seal_manifest["sha256sums_entries"] == 25
    assert seal_manifest["files"] == list(SEALED_FILES)
    assert seal_manifest["files"] == sorted(seal_manifest["files"])
    assert len(seal_manifest["files"]) == 25
    assert seal_manifest["sealed_utc"] == SEALED_UTC
    assert seal_manifest["run_dir"].endswith(RUN_DIR_REL)
    # the run dir is 27 files: the 25 payload plus the two meta files
    # (seal_manifest.json and SHA256SUMS.txt)
    dir_entries = sorted(p.name for p in RUN_DIR.iterdir()
                         if p.is_file())
    assert len(dir_entries) == 27
    assert "seal_manifest.json" in dir_entries
    assert "SHA256SUMS.txt" in dir_entries
    # the checksum file: 25 entries, run-dir-relative names, the
    # self-exclusion, and the full re-hash
    sums_path = RUN_DIR / "SHA256SUMS.txt"
    lines = [
        line for line in sums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 25
    assert sealed_mismatches(RUN_DIR) == []
    names = []
    for line in lines:
        digest, name = line.split("  ", 1)
        names.append(name)
        assert digest == SEALED_FILE_DIGESTS[name]
    assert names == sorted(names), "the checksum entries are name-sorted"
    assert "SHA256SUMS.txt" not in names
    assert "seal_manifest.json" not in names
    # the seal manifest's own on-disk digest
    assert _sha256_file(RUN_DIR / "seal_manifest.json") == (
        SEALED_FILE_DIGESTS["seal_manifest.json"])
    assert _sha256_file(sums_path) == CHECKSUMS_SHA256


# ----------------------------------------------------------------------
# The clean-environment validation record
# ----------------------------------------------------------------------
def test_clean_environment_record(clean_environment, frozen_report):
    """The clean-environment validation's command list, the
    byte-identical report re-hash (digest equality), the probe replica
    record, and the pending seal state."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze.clean_env/1")
    assert clean_environment["run"] == "R45"
    assert clean_environment["card"] == SOLVER_CARD
    commands = clean_environment["commands"]
    assert len(commands) == 3
    assert commands[0] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_staged_r45_simplified_alignment_freeze.py "
        "tests/test_staged_r45_simplified_alignment_freeze_contract_pins.py "
        "-q")
    assert commands[1] == (
        "python -m pytest tests/test_legacy_v1_freeze.py "
        "-W error::UserWarning -q")
    assert commands[2].startswith(
        "python helpers/benchmarking/"
        "staged_r45_simplified_alignment_freeze.py "
        "--clean-validate --run-dir ")
    assert RUN_DIR_REL in commands[2]
    # the byte-identical report re-hash
    assert clean_environment["report_hash_match"] is True
    assert clean_environment["report_digest"] == FROZEN_REPORT_DIGEST
    assert clean_environment["report_rerun_digest"] == (
        FROZEN_REPORT_DIGEST)
    assert clean_environment["report_digest"] == (
        frozen_report["digest"])
    # the probe replica (replica 0, both variants match the sealed
    # record)
    probe = clean_environment["probe_replica"]
    assert probe["replica"] == 0
    assert probe["matches_sealed_record"] is True
    assert probe["per_variant_matches"] == {
        "aligned_simplified": True,
        "isotropic_sequential_baseline": True}
    # the pending seal state (the validation runs before the seal)
    assert clean_environment["seal"]["state"].startswith("pending")
    assert clean_environment["seal"]["entries"] == 0
    assert clean_environment["seal"]["mismatches"] == []
    assert clean_environment["verified_utc"] == CLEAN_VALIDATED_UTC


# ----------------------------------------------------------------------
# The provenance
# ----------------------------------------------------------------------
def test_provenance_on_disk(provenance):
    """The provenance's schema, the frozen R41 parameter source, the
    simplified-form record, the substrate digests, the recorded driver
    digest (equal to the committed dual-tree digest: the run
    pre-registered against the committed driver), and the eight sealed
    predecessor dirs verified."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r45_simplified_alignment_freeze."
        "provenance/1")
    assert provenance["run"] == "R45"
    assert provenance["card"] == SOLVER_CARD
    # the recorded driver digest: the run pre-registered against the
    # committed driver, so the recorded digest equals the committed
    # dual-tree digest
    assert provenance["new_driver_sha256_both_trees"] == DRIVER_SHA256
    assert provenance["driver_note"].startswith(
        "byte-identical in the nested tree")
    # the frozen R41 parameter source (the sealed corrected R41 fit
    # report, no refit)
    source = provenance["frozen_r41_parameters"]
    assert (source["values"]["kappa_B"], source["values"]["kappa_m"],
            source["values"]["b_dec"],
            source["values"]["baryonic_nominal_width"]) == (
                R41_FROZEN_TUPLE)
    assert source["refit"] is False
    assert source["fit_report_file_sha256"] == (
        R41_FIT_REPORT_FILE_SHA256)
    assert source["fit_report_self_digest"] == R41_FIT_REPORT_DIGEST
    assert source["sealed_run_dir_state"] == (
        {"entries": 13, "mismatches": 0, "verified": True})
    # the simplified-form record (constant kappa, zero mass slopes,
    # unchanged ranges)
    form = provenance["simplified_form"]
    assert form["simplified_form"] == {
        "kappa_B": "constant", "kappa_M": "constant",
        "mass_dependence": "fixed at zero, not fit",
    }
    assert form["mass_dependence_coefficients"] == {
        "kappa_b_mass_slope_per_gev": 0.0,
        "kappa_m_mass_slope_per_gev": 0.0,
    }
    assert form["pre_registered_ranges"] == {
        key: list(value) for key, value in RANGES.items()}
    assert form["frozen"] == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec_gev2": 2.0,
        "baryonic_nominal_width_gev": 0.3,
    }
    # the substrate digests, against the frozen table
    assert provenance["builder_sha256"] == (
        SUBSTRATE["builder_blend_path"][1])
    assert provenance["card6_surface_sha256"] == SUBSTRATE["card6_routing"][1]
    assert provenance["engine_sha256"] == SUBSTRATE["card2_engine"][1]
    assert provenance["catalog_sha256"] == SUBSTRATE["catalog"][1]
    assert provenance["continuum_config_sha256"] == (
        SUBSTRATE["continuum_config"][1])
    assert provenance["round2_kernel_sha256"] == (
        SUBSTRATE["round2_kernel"][1])
    assert provenance["r41_driver_sha256"] == SUBSTRATE["r41_driver"][1]
    assert provenance["r42_driver_sha256"] == SUBSTRATE["r42_driver"][1]
    assert provenance["r43_driver_sha256"] == SUBSTRATE["r43_driver"][1]
    assert provenance["cascade_module_sha256"] == {
        name: SUBSTRATE[name][1]
        for name in ("cascade_decay", "cascade_axes", "cascade_alignment",
                     "cascade_form_factor", "cascade_termination",
                     "cascade_ablation")
    }
    assert provenance["m8_frozen_set_hash"] == (
        "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
    # the eight sealed predecessor dirs, all verified
    sealed = provenance["sealed_run_dirs"]
    assert set(sealed) == set(r45.SEALED_RUN_DIRS)
    for rel, state in sealed.items():
        assert state["mismatches"] == 0, rel
        assert state["verified"] is True, rel
    assert sealed["dev/00runs/r44_frozen_nonstrange_validation_2026-09-23"] \
        == {"entries": 25, "mismatches": 0, "verified": True}
    # the provenance's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "provenance.json") == (
        SEALED_FILE_DIGESTS["provenance.json"])
