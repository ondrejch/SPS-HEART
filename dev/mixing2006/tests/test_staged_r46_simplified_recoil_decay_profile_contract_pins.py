"""Round 4, card 21 test-author register: the R46 simplified
recoil/decay re-profile contract pins.

Card ``TASK-20260920-r4-gate4-followup-recoil-decay-reprofile`` (the
Gate 4 follow-up obligation 2 of the written Gate 4 decision
``TASK-20260920-r4-gate4-decision``; plan section 23, lines 1821-1823;
plan section 10 run R42, line 786 re-run under the simplified form),
phase 1 (no Monte Carlo) plus the phase-2 re-point to the sealed run
``dev/00runs/r46_simplified_recoil_decay_profile_2026-09-24/`` (sealed
2026-09-25T06:21:40Z; 13 sealed payload entries, 0 mismatches),
mirroring the card-17/18/20 re-point precedent.  This register is the
test-author half of the card: independent pins of the dual-tree run
driver
(``helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py``,
byte-identical in the nested tree and the monorepo mirror), of the
hermetic solver test
(``tests/test_staged_r46_simplified_recoil_decay_profile.py``), of the
run identity (the pre-registered run dir, sealed by the phase-2 run),
of the card-20 sibling pins (the sealed card-20 R45 archive's
full seal re-hash and its byte-frozen driver digest), of the R41
frozen-parameter source pins, of the superseded corrected R42 profile
pins, and of the freeze re-pin.  The driver is loaded once via
importlib (its top level is the pre-registration identity and the
import-time substrate checks; no events are generated), and the sealed
JSON is read read-only from disk.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist,
  are byte-identical, and sit at the committed digest; the driver's
  own self-computed identity digest agrees; the freeze manifest
  inventory pins both at that digest.
- ``test_driver_identity_constants``: the run identity, the file-name
  constants, the sealed-dir rel paths, the card chain, the frozen
  substrate digests on disk, and the card-20 substrate import status.
- ``test_frozen_simplified_alignment_pins``: the card-20 frozen tuple,
  the import-time substrate cross-check, the frozen-parameter
  assertion on the pinned fixture and its refusal modes, and the R41
  source digests.
- ``test_superseded_profile_pins``: the superseded corrected R42
  profile report (file sha256, self-digest, the boundary minimum (0.5,
  2.6), the pooled eigenvalues, the failing block-4 eigenvalues, and
  the null correlation; the sealed archive is read read-only and skips
  on a fresh checkout).
- ``test_seed_map_partition``: the master seed, the new raw block
  20301076-20301083, the event-index partition 900_000_000 + i, the
  hermetic smoke block (+12000), and the refusal discipline.
- ``test_solver_test_file_digest``: the hermetic solver test file at
  its committed digest, pinned in the freeze manifest inventory.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 502 entries after the
  Round 5 card 1 test-author re-pin; 501 at the Round 5 card 1
  landing; 499 at the card-21 landing) carries the
  new dual-tree driver in both helper trees and the two new R46 test
  files at their on-disk digests; the card-15/19/20 exceptions stay
  exactly once each; this card's freeze exception names the card
  exactly once with the path-sorted file list; the card-21 naming
  comment is retained in every re-pinned chain file with the card-20
  history).
- ``test_sealed_run_identity``: the sealed run dir, the driver run
  identity against it, the T2 statistics, the seed block, the grid,
  and the probe replica.
- ``test_pre_registration_on_disk``: the sealed pre-registration's
  identity, statistics, seed map, frozen parameters, card-20
  substrate pins, superseded-profile pins, and sealed-dir
  verification.
- ``test_profile_report_on_disk``: the sealed profile report's
  schema and digest, the unchanged grid and bounds, the boundary
  minimum, the plan-5.4-line-311 quoting gate (the per-block argmins
  recorded beside the gate, curvature not quoted), the recorded
  curvature matrix and its eigenvalues, the per-direction spread
  table, the reweighting validity probe, the ESS, the acceptance (ii)
  outcome (pass false), the named-residual register, the standing
  residuals, and the superseded-profile citation.
- ``test_gate4_followup_evidence``: the gate record's identity,
  clause, substrate, profile bounds, conservation summary (zero
  violations over 800,000 events), the minimum, the null correlation
  against the 0.9 bound, the unquoted-Hessian block, the per-block
  table, the model-uncertainty carry, the named register equal to the
  report's, the superseded citation, separation_claimed false, and
  the single optimizer use.
- ``test_seal_manifest_and_checksums``: the seal manifest's 13 sealed
  files (8 replica records plus the five top-level files), the
  checksum file's 13 entries, the 15-file run dir (13 payload plus
  the two meta files), and the full re-hash of all 13 sealed files
  against the checksum digests.
- ``test_clean_environment_record``: the clean-environment
  validation's command list, the byte-identical report re-run (digest
  equality), the probe replica record, and the pending seal state.
- ``test_provenance_on_disk``: the provenance's schema, the recorded
  driver digest (the committed dual-tree digest), the card-20
  substrate block, the frozen R41 parameter source, the superseded
  corrected R42 archive block, and the sealed predecessor dirs
  verified.
- ``test_card20_substrate_seal_pins``: the sealed card-20 R45 archive
  verifies read-only with the full re-hash of all 25 sealed payload
  files plus the seal manifest and the checksum inventory (skips on a
  fresh checkout).

Hermetic: standard library, the committed config and catalog, and the
sealed archives (read-only) only; no SPS tables, no event generation,
no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    / "staged_r46_simplified_recoil_decay_profile.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r46_simplified_recoil_decay_profile.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r46_simplified_recoil_decay_profile_2026-09-24")
R45_ARCHIVE = (MONO_ROOT / "dev" / "00runs"
               / "r45_simplified_alignment_freeze_2026-09-24")
R41_ARCHIVE = (MONO_ROOT / "dev" / "00runs"
               / "r41_nonstrange_pilot_fit_2026-09-23")
R42_ARCHIVE = (MONO_ROOT / "dev" / "00runs"
               / "r42_recoil_decay_separation_2026-09-23")

RUN_DIR_REL = "dev/00runs/r46_simplified_recoil_decay_profile_2026-09-24"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r46_simplified_recoil_decay_profile.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-gate4-followup-recoil-decay-reprofile"

#: The committed dual-tree driver digest (byte-identical in both
#: trees; the freeze manifest inventory pins both copies here).
DRIVER_SHA256 = (
    "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f"
)

#: The hermetic solver test file's committed digest (the Round 5
#: card 1 count re-pin 499 -> 501 touched the INVENTORY_COUNT
#: constant, so the digest moved from the card-21 landing digest
#: b431503d3b1a45cdcda3c18603147565af8a4919264e4dcac5be0f52f9ef68fb;
#: the Round 5 card 1 test-author count re-pin 501 -> 502 touched the
#: same constant, so the digest moved from the Round 5 card 1 landing
#: digest 9aaf42ea79c39b8e...).
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
    "2eca3847951c5a85f7f699a8d4b3f5aa6e32137c86b994880243235899b157c3"
)

#: The sealed card-20 R45 substrate archive (the frozen simplified
#: alignment): the run dir, the seal state, and the frozen driver.
R45_RUN_DIR_REL = "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"
CARD20_SIBLING_DRIVER_SHA256 = (
    "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10"
)
CARD20_SEALED_UTC = "2026-09-25T00:22:00Z"

#: The sealed corrected R41 fit report (the frozen-parameter source):
#: the file digest (byte identity) and the report's self-digest.
R41_FIT_REPORT_FILE_SHA256 = (
    "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
)
R41_FIT_REPORT_DIGEST = (
    "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
)
R41_FROZEN_TUPLE = (0.0, 4.0, 2.0, 0.3)

#: The superseded corrected R42 profile (the superseded-profile
#: reference; read-only, cited as superseded and never as evidence).
SUPERSEDED_R42_RUN_DIR_REL = (
    "dev/00runs/r42_recoil_decay_separation_2026-09-23"
)
SUPERSEDED_R42_REPORT_FILE_SHA256 = (
    "430be97de52ba84b1519559b35c5758b2c4d0b08e8a890056ab5c05b78037d62"
)
SUPERSEDED_R42_REPORT_DIGEST = (
    "e18a1e275dda9b72c86ce4f63ea216f250b3a378626f63497a8d252d52755af5"
)
SUPERSEDED_R42_MINIMUM = (0.5, 2.6)
SUPERSEDED_R42_EIGENVALUES = [
    -182549.71181584388, 7.698059717158571,
]
SUPERSEDED_R42_BLOCK4_EIGENVALUES = [
    -1999.037806083929, 15699.862514911674,
]
SUPERSEDED_R42_SEALED_UTC = "2026-09-24T01:53:32Z"

# ----------------------------------------------------------------------
# The sealed R46 run values (the phase-2 re-point; read from the sealed
# run dir and pinned verbatim)
# ----------------------------------------------------------------------
#: The seal manifest's file list (8 replica records plus the five
#: top-level files), in seal order.
SEALED_FILES = (
    "clean_environment_commands.json",
    "gate4_followup_reprofile.json",
    "pre_registration.json",
    "profile_report.json",
    "provenance.json",
) + tuple(f"replica_{r}.json" for r in (0, 1, 2, 3, 4, 5, 6, 7))

#: The on-disk digest of every sealed R46 payload file (the checksum
#: file's 13 entries plus the seal manifest's own digest).
SEALED_FILE_DIGESTS = {
    "clean_environment_commands.json": (
        "af0ebf27555559d6e84a2ddafad4888859db351e11497247778274c88e8066a6"
    ),
    "gate4_followup_reprofile.json": (
        "dd0ea193047b6bbac7fbd9f8fc3095565f4e8cf92362589eec935b66a09837db"
    ),
    "pre_registration.json": (
        "7a7f87b230b2afa48b4bc1dc74be10026e3495de6fa34c6c4ea6e7a27fa18fcd"
    ),
    "profile_report.json": (
        "90f7ed0c9fdc73dee908ad64170f37b6d44ead7541fcbd70581d85d88880fb3d"
    ),
    "provenance.json": (
        "fd4af965b60e8f84156b1346e281ab3289df497a7be36941aba9e516957cf3cc"
    ),
    "replica_0.json": (
        "c0bfdb6eb9d0a135e21deee8dbc325b34a9881b51cdbf68fce00f63445bd8c9b"
    ),
    "replica_1.json": (
        "8c2c620544d25545887f7e3d5ad356b93f99bc68f8834cc3e418abd6ca6da902"
    ),
    "replica_2.json": (
        "c63524f9d8a2e430d2f36609264296cd38498e9b26a31b31fdb0f302ac7e02f2"
    ),
    "replica_3.json": (
        "e23161f6910ccb8773ba8b76753f1f2481c564b12cf4020863d3ddff517013e3"
    ),
    "replica_4.json": (
        "074c7f7bf2db0361ad108b4ef10525390a9857208025fadb00b8bb0b368bc847"
    ),
    "replica_5.json": (
        "cc43662f7f78100e963aa09ad22765369eda9cb24c9b1df6f1f3ac75b3797e3c"
    ),
    "replica_6.json": (
        "7bce49996bcad8bdba6913f65a91dfccbf14257fb535cfcfc812f80682e900c3"
    ),
    "replica_7.json": (
        "f4f9ade4f8e1fafd7778222e56534c5150978b42be7f9f61b10f94e2c268f8c4"
    ),
    "seal_manifest.json": (
        "77f2faf4fddd07a4262afb215e378e554198431dda89a7789070515a995d9a0e"
    ),
}

#: The checksum file's own digest (excluded from its own entries).
CHECKSUMS_SHA256 = (
    "4cfff3c42e9877efe89e35f73d667d56442a60bc365cbfeba5bf2155b2e37e9c"
)

#: The report's internal digest (cross-referenced by the gate record
#: and the clean-environment re-run).
PROFILE_REPORT_DIGEST = (
    "794ca315841ee223f2d23d0f9e9fa4366c0f0d3f1f2023f633b2521b38ed4918"
)

#: The clean-environment validation timestamp (the validation runs
#: before the seal, so the record is sealed content).
CLEAN_VALIDATED_UTC = "2026-09-25T06:21:25Z"
SEALED_UTC = "2026-09-25T06:21:40Z"

#: The profile outcome at the sealed values: the boundary minimum (the
#: upper B_eff-multiplier boundary 2.5 with the lower b_dec boundary
#: 0.2 GeV^-2), the recorded curvature matrix beside the plan-5.4
#: line 311 gate (curvature not quoted: the 8 per-block argmins do not
#: agree with the pooled argmin), the null correlation, and the
#: conservation tally.
PROFILE_MINIMUM = {
    "b_dec": 0.2,
    "b_eff_gev2": 21.260351905209028,
    "b_eff_multiplier": 2.5,
    "i": 8,
    "j": 0,
    "minus2lnl": 20430.362781354015,
    "on_boundary": True,
}
POOLED_HESSIAN = [
    [6194.26495368086, -2802.2388516843785],
    [-2802.2388516843785, -7204.306040043109],
]
POOLED_EIGENVALUES = [-7766.767492115292, 6756.726405753046]
PER_BLOCK_ARGMINS = {
    "0": [0, 8],
    "1": [2, 8],
    "2": [2, 8],
    "3": [0, 8],
    "4": [1, 8],
    "5": [1, 8],
    "6": [1, 8],
    "7": [4, 8],
}
POOLED_ARGMIN = [8, 0]
CONSERVATION = {
    "events_checked": 800000,
    "pass": True,
    "tolerance": 1e-09,
    "violations": 0,
    "worst_energy_closure_gev": 3.836930773104541e-13,
}
VALIDITY_MAX_RELATIVE_DEVIATION = 1.7215327833942586e-14
ESS_MIN_AT_MINIMUM = 0.33194138635521775

# ----------------------------------------------------------------------
# Pinned sealed card-20 substrate values (read from disk, verified)
# ----------------------------------------------------------------------

#: The card-20 seal manifest's file list (20 replica records plus the
#: five top-level files), in seal order.
CARD20_SEALED_FILES = (
    "clean_environment_commands.json",
    "frozen_report.json",
    "gate4_followup_simplified.json",
    "pre_registration.json",
    "provenance.json",
) + tuple(f"replica_{r}.json" for r in (
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 3, 4, 5, 6, 7, 8, 9,
))

#: The on-disk digest of every sealed card-20 payload file (the
#: checksum file's 25 entries plus the seal manifest's own digest).
CARD20_SEALED_FILE_DIGESTS = {
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

#: The card-20 checksum file's own digest (excluded from its own
#: entries).
CARD20_CHECKSUMS_SHA256 = (
    "abe5927a63d21e3be05cc1c79204cee765b229619c0960262451689b415060f4"
)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r46_simplified_recoil_decay_profile", NESTED_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r46 = _load_driver()


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
    assert r46.DRIVER_SHA256 == DRIVER_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == nested_digest
    assert inventory[DRIVER_REL_OUTER] == outer_digest


def test_driver_identity_constants():
    """The driver's run identity, file-name constants, the sealed-dir
    rel paths, and the card chain on disk."""
    assert r46.RUN_ID == "R46"
    assert r46.CARD == SOLVER_CARD
    assert r46.RUN_NAME == "r46_simplified_recoil_decay_profile_2026-09-24"
    assert r46.DEFAULT_RUN_DIR == RUN_DIR_REL
    assert r46.GATE_RECORD_NAME == "gate4_followup_reprofile.json"
    assert r46.PROFILE_REPORT_NAME == "profile_report.json"
    assert r46.PREREGISTRATION_NAME == "pre_registration.json"
    assert r46.PROVENANCE_NAME == "provenance.json"
    assert r46.CLEAN_ENV_NAME == "clean_environment_commands.json"
    assert r46.SHA256SUMS_NAME == "SHA256SUMS.txt"
    assert r46.SEAL_NAME == "seal_manifest.json"
    assert r46.RECORD_FMT == "replica_{}.json"
    assert r46.SMOKE_PREFIX == "smoke_"
    assert r46.R45_RUN_DIR == \
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"
    assert r46.SUPERSEDED_R42_RUN_DIR == \
        "dev/00runs/r42_recoil_decay_separation_2026-09-23"
    assert r46.CARD_CHAIN[-3:] == (
        "TASK-20260920-r4-gate4-decision",
        "TASK-20260920-r4-gate4-followup-simplified-alignment",
        SOLVER_CARD,
    )
    # the frozen substrate byte stability on disk (including the
    # card-20 sibling)
    for key, (rel, expected) in r46.FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), (key, path)
        assert _sha256_file(path) == expected, key
    # the card-20 import cross-check
    assert r46._CARD20_SUBSTRATE_STATUS["verified"] is True
    assert r46._CARD20_SUBSTRATE_STATUS["sibling_driver_sha256"] == (
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10")
    # the checker-required substrate pin: the import status names the
    # sealed card-20 archive under the key the production
    # _assert_pre_registration requires (the writer and the checker
    # agree on the pin)
    assert r46._CARD20_SUBSTRATE_STATUS["frozen_simplified_archive"] == \
        "dev/00runs/r45_simplified_alignment_freeze_2026-09-24"


def test_frozen_simplified_alignment_pins():
    """The card-20 frozen tuple, the import-time substrate status, the
    frozen-parameter assertion on the pinned fixture and its refusal
    modes, and the blend ramp start fixed at B (obligation 4)."""
    assert r46.FROZEN_SIMPLIFIED_ALIGNMENT == {
        "kappa_B": 0.0, "kappa_m": 4.0, "b_dec": 2.0,
        "baryonic_nominal_width": 0.3,
    }
    assert (r46.GENERATION_KAPPA_B, r46.GENERATION_KAPPA_M) == (0.0, 4.0)
    assert r46.FROZEN_WIDTH_GEV == 0.3
    assert r46.B_DEC_GENERATION_GEV2 == 2.0
    frozen = r46.assert_frozen_parameters(r46.R41_FIT_REPORT_FIXTURE)
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == R41_FROZEN_TUPLE
    assert frozen["refit"] is False
    fixture = r46.R41_FIT_REPORT_FIXTURE
    drifted = dict(fixture)
    drifted["fitted"] = dict(fixture["fitted"], kappa_m=3.0)
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(drifted)
    wrong_run = dict(fixture, run="R42")
    with pytest.raises(SystemExit):
        r46.assert_frozen_parameters(wrong_run)
    assert r46.BLEND_RAMP_START_GEV == 2.680194110727528
    # the profile axes are exactly two; nothing else is fit
    assert r46.CURVATURE_SPREAD_FRACTION == 0.20
    assert r46.CORRELATION_BOUND == 0.9


def test_superseded_profile_pins():
    """The superseded corrected R42 profile report pins (read-only;
    skips on a fresh checkout)."""
    superseded_path = (MONO_ROOT / "dev" / "00runs"
                       / "r42_recoil_decay_separation_2026-09-23"
                       / "profile_report.json")
    if not superseded_path.is_file():
        pytest.skip("the sealed R42 archive is absent on this checkout")
    assert r46.SUPERSEDED_R42_REPORT_FILE_SHA256 == _sha256_file(
        superseded_path)
    superseded = json.loads(
        superseded_path.read_text(encoding="utf-8"))
    assert superseded.get("digest") == "e18a1e275dda9b72c86ce4f63ea216f250b3a378626f63497a8d252d52755af5"
    assert (superseded["minimum"]["b_eff_multiplier"],
            superseded["minimum"]["b_dec"]) == SUPERSEDED_R42_MINIMUM
    assert superseded["hessian"]["eigenvalues"] == \
        r46.SUPERSEDED_R42_EIGENVALUES
    assert superseded["curvature_stability"]["4"]["eigenvalues"] == \
        r46.SUPERSEDED_R42_BLOCK4_EIGENVALUES
    assert superseded["correlation_b_eff_b_dec"] is None
    assert r46.SUPERSEDED_R42_STATUS == "superseded by re-profile"
    assert r46.SUPERSEDED_R42_SEALED_UTC == "2026-09-24T01:53:32Z"


def test_seed_map_partition():
    """The master seed, the new raw block 20301076-20301083, the
    event-index partition 900_000_000 + i, the smoke block (+12000),
    and the refusal discipline."""
    assert r46.MASTER_SEED == 20260920
    assert r46.RAW_REPLICA_SEEDS == tuple(20301076 + k for k in range(8))
    assert r46.RAW_REPLICA_SEEDS[-1] == 20301083
    assert r46.EVENT_INDEX_RULE == "900_000_000 + i, i < 100,000"
    assert r46.EVENT_INDEX_BASE == 900_000_000
    assert r46.T2_REPLICAS == 8
    assert r46.T2_EVENTS_PER_REPLICA == 100_000
    assert r46.SMOKE_RAW_SEED == 20312920
    assert r46.SMOKE_KERNEL_REPLICA_SLOT == 12_000
    assert r46.SMOKE_EVENT_INDEX_BASE == 990_000_000
    assert "20301076-20301083" in r46.RAW_SEED_RULE
    assert "20301056-20301075" in r46.RAW_SEED_RULE
    assert "derive_substreams(20260920, k, 900_000_000 + i)" in \
        r46.KERNEL_SEED_RULE
    report = r46.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r46_raw_seed_block"] == [20301076, 20301083]
    assert report["shared_raw_seeds"] == {}
    with pytest.raises(SystemExit):
        r46.raw_replica_seed(8)


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
# The sealed run identity (the phase-2 re-point)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def pre_registration():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profile_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "profile_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_followup_reprofile.json")
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


def test_sealed_run_identity():
    """The sealed run dir, the driver run identity against it, the T2
    statistics, the seed block, the grid, and the probe replica."""
    assert RUN_DIR.name == "r46_simplified_recoil_decay_profile_2026-09-24"
    assert r46.RUN_NAME == RUN_DIR.name
    assert r46.DEFAULT_RUN_DIR == RUN_DIR_REL
    assert r46.RUN_ID == "R46"
    assert r46.CARD == SOLVER_CARD
    # the T2 statistics (plan section 5.1 line 254)
    assert r46.T2_REPLICAS == 8
    assert r46.T2_EVENTS_PER_REPLICA == 100_000
    # the pre-registered probe replica
    assert r46.PROBE_REPLICA == 0


def test_pre_registration_on_disk(pre_registration):
    """The sealed pre-registration's identity, statistics, seed map,
    frozen parameters, card-20 substrate pins, superseded-profile
    pins, and sealed-dir verification."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile."
        "pre_registration/1")
    assert pre_registration["run"] == "R46"
    assert pre_registration["card"] == SOLVER_CARD
    assert pre_registration["mode"] == "production"
    assert pre_registration["written_before_generation"] is True
    stats = pre_registration["statistics"]
    assert stats["replicas"] == 8
    assert stats["events_per_replica"] == 100_000
    assert stats["total_events"] == 800_000
    seeds = pre_registration["seed_map"]
    assert seeds["master_seed"] == 20260920
    assert seeds["raw_replica_seeds"] == list(
        20301076 + k for k in range(8))
    assert seeds["event_index_rule"] == "900_000_000 + i, i < 100,000"
    # the card-20 substrate pins (the checker-required key beside the
    # sibling pins)
    substrate = pre_registration["card20_substrate"]
    assert substrate["frozen_simplified_archive"] == R45_RUN_DIR_REL
    assert substrate["sibling_driver_sha256"] == (
        CARD20_SIBLING_DRIVER_SHA256)
    assert substrate["verified"] is True
    # the frozen simplified alignment (the sealed R41 tuple, no refit)
    frozen = pre_registration["frozen_parameters"]
    assert (frozen["kappa_B"], frozen["kappa_m"],
            frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == R41_FROZEN_TUPLE
    assert frozen["refit"] is False
    # the superseded corrected R42 profile pins
    superseded = pre_registration["superseded_r42_profile"]
    assert superseded["minimum"] == {
        "b_eff_multiplier": 0.5, "b_dec_gev2": 2.6}
    assert superseded["report_digest"] == SUPERSEDED_R42_REPORT_DIGEST
    assert superseded["status"] == "superseded by re-profile"
    # the nine sealed predecessor dirs, all verified (the card-20
    # substrate and the superseded R42 among them)
    status = pre_registration["sealed_run_dirs_status"]
    assert set(status) == set(r46.SEALED_RUN_DIRS)
    for rel, state in status.items():
        assert state["mismatches"] == 0, rel
        assert state["verified"] is True, rel
    assert status[R45_RUN_DIR_REL] == (
        {"entries": 25, "mismatches": 0, "verified": True})
    assert status[SUPERSEDED_R42_RUN_DIR_REL] == (
        {"entries": 13, "mismatches": 0, "verified": True})
    # the optimizer block (the single pre-registered optimizer use of
    # the follow-up pair)
    assert pre_registration["optimizer"]["single_use"].startswith(
        "this card holds the single pre-registered optimizer use")
    assert "card 20 invokes none" in (
        pre_registration["optimizer"]["single_use"])
    # the pre-registration's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        SEALED_FILE_DIGESTS["pre_registration.json"])


def test_profile_report_on_disk(profile_report):
    """The sealed profile report's schema and digest, the unchanged
    grid and bounds, the boundary minimum, the plan-5.4-line-311
    quoting gate (the per-block argmins recorded beside the gate,
    curvature not quoted), the recorded curvature matrix and its
    eigenvalues, the per-direction spread table, the reweighting
    validity probe, the ESS, the acceptance (ii) outcome (pass
    false), the named-residual register, the standing residuals, and
    the superseded-profile citation."""
    assert profile_report["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile."
        "report/1")
    assert profile_report["run"] == "R46"
    assert profile_report["card"] == SOLVER_CARD
    assert profile_report["mode"] == "production"
    assert profile_report["digest"] == PROFILE_REPORT_DIGEST
    assert profile_report["optimizer_invoked"] is True
    assert profile_report["optimizer_uses"] == 1
    # the unchanged 9 x 9 grid and the never-relaxed bounds
    grid = profile_report["profile_grid"]
    assert grid["b_eff_multipliers"] == [0.5, 0.75, 1.0, 1.25, 1.5,
                                         1.75, 2.0, 2.25, 2.5]
    assert grid["b_dec_gev2"] == [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0,
                                  2.3, 2.6]
    assert grid["b_eff_frozen_gev2"] == 8.504140762083612
    assert grid["b_dec_generation_gev2"] == 2.0
    assert profile_report["bounds_never_relaxed"]["b_eff_multiplier"] \
        == [0.5, 2.5]
    assert profile_report["bounds_never_relaxed"]["b_dec_gev2"] \
        == [0.2, 2.6]
    # the profile minimum: the pooled grid corner (multiplier 2.5,
    # b_dec 0.2 GeV^-2), on the boundary
    assert profile_report["minimum"] == PROFILE_MINIMUM
    assert profile_report["minimum"]["on_boundary"] is True
    # the plan-5.4-line-311 quoting gate: the per-block argmins do not
    # agree with the pooled argmin, so no curvature is quoted (the
    # curvature matrix is recorded beside the gate)
    agreement = profile_report["minima_agreement"]
    assert agreement["minima_agree"] is False
    assert agreement["pooled_argmin"] == POOLED_ARGMIN
    assert agreement["per_block_argmins"] == PER_BLOCK_ARGMINS
    hessian = profile_report["hessian"]
    assert hessian["quoted"] is False
    assert hessian["hessian"] == POOLED_HESSIAN
    assert hessian["eigenvalues"] == POOLED_EIGENVALUES
    assert hessian["positive_definite"] is False
    # the null correlation (undefined at the recorded minimum; the
    # pre-registered 0.9 distinguishability bound)
    assert profile_report["correlation_b_eff_b_dec"] is None
    assert profile_report["distinguishability_bound"] == 0.9
    # the per-direction curvature spreads (both axes far above the
    # 20% bound; the offending census is the whole block set)
    spread = profile_report["per_direction_curvature"]
    assert spread["b_eff_multiplier"]["pooled"] == 6194.26495368086
    assert spread["b_eff_multiplier"]["spread_fraction"] == (
        1.9399634777490447)
    assert spread["b_eff_multiplier"]["below_fraction"] is False
    assert spread["b_dec"]["pooled"] == -7204.306040043109
    assert spread["b_dec"]["spread_fraction"] == 11.357903256332406
    assert spread["b_dec"]["below_fraction"] is False
    # the reweighting validity probe (the exact identities at the two
    # pinned grid points; the prediction-level diagnostic is recorded,
    # never asserted at 1e-6)
    validity = profile_report["reweighting_validity"]
    assert validity["pass"] is True
    assert validity["bound_relative"] == 1e-06
    assert validity["max_relative_deviation"] == (
        VALIDITY_MAX_RELATIVE_DEVIATION)
    assert validity["pinned_grid_points"] == 2
    assert validity["pinned_subsample_events"] == 2000
    # the ESS at the minimum (above the 0.01 floor; no
    # weight-degeneracy finding)
    ess = profile_report["objective_at_minimum"]["ess"]
    assert ess["min"] == ESS_MIN_AT_MINIMUM
    assert ess["min"] >= 0.01
    # the objective drops: none (no dropped bins, no non-finite
    # surface points)
    assert profile_report["n_dropped_bins"] == 0
    assert profile_report["dropped_bins"] == []
    assert profile_report["nonfinite_surface_points"] == []
    # the acceptance (ii) outcome: fail, with every condition named
    acceptance = profile_report["acceptance_ii"]
    assert acceptance["pass"] is False
    assert acceptance["interior_minimum"] is False
    assert acceptance["correlation_defined"] is False
    assert acceptance["pooled_hessian_positive_definite"] is False
    assert acceptance["all_blocks_positive_definite"] is False
    assert acceptance["per_direction_spread_below_fraction"] is False
    # the named-residual register: the two boundary findings, the
    # minima-gate finding, and the block-curvature failure (the
    # correlation and pooled-curvature claims are not established
    # under the quoting gate, so they carry no separate finding)
    register = profile_report["named_residual_register"]
    assert [entry["kind"] for entry in register] == [
        "boundary_minimum", "boundary_minimum",
        "replica_block_minima_disagree", "block_curvature_failure"]
    assert register[0]["axis"] == "b_eff_multiplier"
    assert register[0]["observed"] == 2.5
    assert register[1]["axis"] == "b_dec"
    assert register[1]["observed"] == 0.2
    failure = register[3]["observed"]
    assert set(failure["blocks_not_positive_definite"]) == {
        "0", "1", "2", "4", "5", "7"}
    assert failure["blocks_not_positive_definite"]["0"] == [
        -10229.791328650783, 37786.20609152207]
    assert failure["per_direction_spread_failures"]["b_dec"][
        "spread_fraction"] == 11.357903256332406
    # the standing residuals carried (the R43 seagull finding and the
    # R32 spread carry)
    standing = profile_report["standing_residuals"]
    assert standing[0]["kind"] == (
        "b_transport_seagull_beyond_isotropic_not_positive")
    assert standing[0]["observed"] == -2.3959765824349295e-05
    assert standing[1]["kind"] == "r32_model_uncertainty_carry"
    assert standing[1]["r32_spread"] == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    # the superseded corrected R42 profile citation (cited as
    # superseded, never as evidence)
    superseded = profile_report["superseded_profile"]
    assert superseded["minimum"] == [0.5, 2.6]
    assert superseded["status"] == "superseded by re-profile"
    assert superseded["report_digest"] == SUPERSEDED_R42_REPORT_DIGEST
    # the report's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "profile_report.json") == (
        SEALED_FILE_DIGESTS["profile_report.json"])


def test_gate4_followup_evidence(gate_record, profile_report):
    """The gate record's identity, clause, substrate, profile bounds,
    conservation summary (zero violations over 800,000 events), the
    minimum, the null correlation against the 0.9 bound, the
    unquoted-Hessian block, the per-block table, the model-uncertainty
    carry, the named register equal to the report's, the superseded
    citation, separation_claimed false, and the single optimizer
    use."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile.gate/1")
    assert gate_record["gate"] == (
        "gate 4 follow-up obligation 2 (plan line 788)")
    assert gate_record["run"] == "R46 simplified recoil/decay profile (2026-09-24)"
    assert gate_record["card"] == SOLVER_CARD
    assert gate_record["clause"] == (
        "string-like rank law under the simplified form")
    assert gate_record["sealed"] is True
    assert gate_record["events_checked"] == 800_000
    # the substrate (the sealed card-20 archive, the checker-required
    # pin)
    substrate = gate_record["substrate"]
    assert substrate["frozen_simplified_archive"] == R45_RUN_DIR_REL
    assert substrate["seal_verified"] is True
    assert substrate["sibling_driver_sha256"] == (
        CARD20_SIBLING_DRIVER_SHA256)
    # the profile axes and the never-relaxed bounds
    assert gate_record["profile"]["axes"] == ["B_eff", "b_dec"]
    assert gate_record["profile"]["grid"]["b_eff_multipliers"] == [
        0.5, 2.5, 0.25]
    assert gate_record["profile"]["grid"]["b_dec_gev2"] == [0.2, 2.6, 0.3]
    assert gate_record["bounds_never_relaxed"] == {
        "b_eff_multiplier": [0.5, 2.5],
        "b_dec_gev2": [0.2, 2.6],
    }
    # the conservation tally (zero violations; the worst energy
    # closure recorded)
    conservation = gate_record["conservation"]
    assert conservation == CONSERVATION
    assert conservation["violations"] == 0
    assert conservation["pass"] is True
    # the minimum, the null correlation against the 0.9 bound
    minimum = gate_record["minimum"]
    assert minimum["b_eff_multiplier"] == 2.5
    assert minimum["b_dec"] == 0.2
    assert minimum["on_boundary"] is True
    assert minimum["minus2lnl"] == 20430.362781354015
    assert gate_record["correlation_b_eff_b_dec"] is None
    assert gate_record["distinguishability_bound"] == 0.9
    # the unquoted-Hessian block beside the gate
    hessian = gate_record["hessian"]
    assert hessian["quoted"] is False
    assert hessian["hessian"] == POOLED_HESSIAN
    assert hessian["eigenvalues"] == POOLED_EIGENVALUES
    assert hessian["positive_definite"] is False
    # the per-block curvature table (equal to the report's)
    assert gate_record["per_block_curvature"] == (
        profile_report["curvature_stability"])
    assert gate_record["per_direction_curvature"] == (
        profile_report["per_direction_curvature"])
    # the model-uncertainty carry (obligation 4)
    carry = gate_record["model_uncertainty_carry"]
    assert carry["obligation_4"] is True
    assert carry["r32_spread"] == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    # the named-residual register (equal to the report's), the
    # superseded citation, and the no-separation outcome
    assert gate_record["named_residual_register"] == (
        profile_report["named_residual_register"])
    superseded = gate_record["superseded_profile"]
    assert superseded["minimum"] == [0.5, 2.6]
    assert superseded["status"] == "superseded by re-profile"
    assert gate_record["separation_claimed"] is False
    assert gate_record["acceptance_ii"]["pass"] is False
    # the single pre-registered optimizer use of the follow-up pair
    assert gate_record["optimizer_invoked"] is True
    assert gate_record["optimizer_uses"] == 1
    # the report digest cross-reference
    assert gate_record["profile_report_digest"] == PROFILE_REPORT_DIGEST
    assert gate_record["profile_report_digest"] == (
        profile_report["digest"])
    # the standing residuals (equal to the report's)
    assert gate_record["standing_residuals"] == (
        profile_report["standing_residuals"])
    # the gate record's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "gate4_followup_reprofile.json") == (
        SEALED_FILE_DIGESTS["gate4_followup_reprofile.json"])


def test_seal_manifest_and_checksums(seal_manifest):
    """The seal manifest's 13 sealed files (8 replica records plus the
    five top-level files), the checksum file's 13 entries, the 15-file
    run dir (13 payload plus the two meta files), and the full re-hash
    of all 13 sealed files against the checksum digests."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile.seal/1")
    assert seal_manifest["run"] == "R46"
    assert seal_manifest["card"] == SOLVER_CARD
    assert seal_manifest["record_count"] == 8
    assert seal_manifest["sha256sums_entries"] == 13
    assert seal_manifest["files"] == list(SEALED_FILES)
    assert seal_manifest["files"] == sorted(seal_manifest["files"])
    assert len(seal_manifest["files"]) == 13
    assert seal_manifest["sealed_utc"] == SEALED_UTC
    assert seal_manifest["run_dir"].endswith(RUN_DIR_REL)
    # the run dir is 15 files: the 13 payload plus the two meta files
    # (seal_manifest.json and SHA256SUMS.txt)
    dir_entries = sorted(p.name for p in RUN_DIR.iterdir()
                         if p.is_file())
    assert len(dir_entries) == 15
    assert "seal_manifest.json" in dir_entries
    assert "SHA256SUMS.txt" in dir_entries
    # the checksum file: 13 entries, run-dir-relative names, the
    # self-exclusion, and the full re-hash
    sums_path = RUN_DIR / "SHA256SUMS.txt"
    lines = [
        line for line in sums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 13
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


def test_clean_environment_record(clean_environment, profile_report):
    """The clean-environment validation's command list, the
    byte-identical report re-run (digest equality), the probe replica
    record, and the pending seal state."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile."
        "clean_env/1")
    assert clean_environment["run"] == "R46"
    assert clean_environment["card"] == SOLVER_CARD
    commands = clean_environment["commands"]
    assert len(commands) == 4
    assert commands[0] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_staged_r46_simplified_recoil_decay_profile.py "
        "tests/test_staged_r46_simplified_recoil_decay_profile_"
        "contract_pins.py -q")
    assert commands[1] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_legacy_v1_freeze.py -W error::UserWarning -q")
    assert commands[2] == (
        "cd /home/o/NA49/2026/01-NA49-res && python "
        "helpers/benchmarking/check_helpers_layout_sync.py")
    assert commands[3].startswith(
        "python helpers/benchmarking/"
        "staged_r46_simplified_recoil_decay_profile.py "
        "--clean-validate --run-dir ")
    assert RUN_DIR_REL in commands[3]
    # the byte-identical report re-run
    assert clean_environment["report_hash_match"] is True
    assert clean_environment["profile_report_digest"] == (
        PROFILE_REPORT_DIGEST)
    assert clean_environment["profile_report_rerun_digest"] == (
        PROFILE_REPORT_DIGEST)
    assert clean_environment["profile_report_digest"] == (
        profile_report["digest"])
    # the probe replica (replica 0 matches its sealed record; the
    # gateway-generated record reproduced by the local clean
    # interpreter)
    probe = clean_environment["probe_replica"]
    assert probe["replica"] == 0
    assert probe["matches_sealed_record"] is True
    # the pending seal state (the validation runs before the seal)
    assert clean_environment["seal"]["state"].startswith("pending")
    assert clean_environment["seal"]["entries"] == 0
    assert clean_environment["seal"]["mismatches"] == []
    assert clean_environment["verified_utc"] == CLEAN_VALIDATED_UTC


def test_provenance_on_disk(provenance):
    """The provenance's schema, the recorded driver digest (equal to
    the committed dual-tree digest: the run pre-registered against the
    committed driver), the card-20 substrate block, the frozen R41
    parameter source, the superseded corrected R42 archive block, and
    the sealed predecessor dirs verified."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r46_simplified_recoil_decay_profile."
        "provenance/1")
    assert provenance["run"] == "R46"
    assert provenance["card"] == SOLVER_CARD
    # the recorded driver digest: the run pre-registered against the
    # committed driver, so the recorded digest equals the committed
    # dual-tree digest
    assert provenance["new_driver_sha256_both_trees"] == DRIVER_SHA256
    assert provenance["driver_note"].startswith(
        "byte-identical in the nested tree")
    # the card-20 substrate block (the frozen simplified alignment as
    # the generation substrate)
    substrate = provenance["card20_substrate"]
    assert substrate["frozen_simplified_archive"] == R45_RUN_DIR_REL
    assert substrate["sibling_driver_sha256"] == (
        CARD20_SIBLING_DRIVER_SHA256)
    assert substrate["sibling_run_dir_seal"] == {
        "entries": 25, "mismatches": 0, "verified": True}
    # the frozen R41 parameter source (the sealed corrected R41 fit
    # report, no refit)
    frozen = provenance["frozen_parameters"]
    assert (frozen["kappa_B"], frozen["kappa_m"], frozen["b_dec_gev2"],
            frozen["baryonic_nominal_width_gev"]) == R41_FROZEN_TUPLE
    assert frozen["refit"] is False
    assert frozen["r41_fit_report_file_sha256"] == (
        R41_FIT_REPORT_FILE_SHA256)
    assert frozen["r41_fit_report_digest"] == R41_FIT_REPORT_DIGEST
    # the superseded corrected R42 archive block (intact; the seal
    # verifies; cited as superseded, never as evidence)
    superseded = provenance["superseded_r42_profile"]
    assert superseded["run_dir"] == SUPERSEDED_R42_RUN_DIR_REL
    assert superseded["report_file_sha256"] == (
        SUPERSEDED_R42_REPORT_FILE_SHA256)
    assert superseded["report_digest"] == SUPERSEDED_R42_REPORT_DIGEST
    assert superseded["minimum"] == {
        "b_eff_multiplier": 0.5, "b_dec_gev2": 2.6}
    assert superseded["pooled_eigenvalues"] == SUPERSEDED_R42_EIGENVALUES
    assert superseded["failing_block_eigenvalues"] == {
        "4": SUPERSEDED_R42_BLOCK4_EIGENVALUES}
    assert superseded["sealed_utc"] == SUPERSEDED_R42_SEALED_UTC
    assert superseded["seal"] == {
        "entries": 13, "mismatches": 0, "verified": True}
    # the nine sealed predecessor dirs, all verified
    sealed = provenance["sealed_run_dirs"]
    assert set(sealed) == set(r46.SEALED_RUN_DIRS)
    for rel, state in sealed.items():
        assert state["mismatches"] == 0, rel
        assert state["verified"] is True, rel
    assert sealed[R45_RUN_DIR_REL] == {
        "entries": 25, "mismatches": 0, "verified": True}
    # the provenance's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "provenance.json") == (
        SEALED_FILE_DIGESTS["provenance.json"])


# ----------------------------------------------------------------------
# The freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
#: The twenty-three chain count-pin files whose inventory-count pins
#: this card's re-pin bump (495 -> 499, +4 entries): the card-20 chain
#: (twenty-one files) plus the card-20 solver test's and the card-20
#: test-author register's own count constants.
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
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
)




# ----------------------------------------------------------------------
# The card-20 substrate seal pins (the phase-2 sealed archive)
# ----------------------------------------------------------------------
CARD20_SIBLING_DRIVER_SHA256 = (
    "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10"
)


def _card20_nested_sibling() -> Path:
    return (MONO_ROOT / "dev" / "mixing2006" / "helpers" / "benchmarking"
            / "staged_r45_simplified_alignment_freeze.py")


def test_card20_substrate_seal_pins():
    """The sealed card-20 R45 archive verifies read-only with the full
    re-hash of all 25 sealed payload files plus the seal manifest and
    the checksum inventory (skips on a fresh checkout)."""
    archive = R45_ARCHIVE
    if not (archive / "SHA256SUMS.txt").is_file():
        pytest.skip("the sealed card-20 archive is absent on this "
                    "checkout")
    # the checksum file re-hash (25 entries, 0 mismatches)
    for line in (archive / "SHA256SUMS.txt").read_text(
            encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert _sha256_file(archive / name) == digest, name
        assert CARD20_SEALED_FILE_DIGESTS[name] == digest, name
    # the seal manifest identity
    seal = json.loads((archive / "seal_manifest.json")
                      .read_text(encoding="utf-8"))
    assert seal["sealed_utc"] == CARD20_SEALED_UTC
    assert seal["run_dir"].endswith("r45_simplified_alignment_freeze_"
                                    "2026-09-24")
    assert set(CARD20_SEALED_FILE_DIGESTS) == set(CARD20_SEALED_FILES) | {
        "seal_manifest.json"}
    assert len(CARD20_SEALED_FILE_DIGESTS) == 26
    # the checksum file's own digest (excluded from its own entries)
    assert _sha256_file(archive / "SHA256SUMS.txt") == (
        CARD20_CHECKSUMS_SHA256)
    # the sealed card-20 driver digest is the substrate sibling pin
    assert _sha256_file(_card20_nested_sibling()) == (
        CARD20_SIBLING_DRIVER_SHA256)
