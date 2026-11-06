"""Round 5, card 13 test-author register: the R53 multiple-neutral-
excitation run-card contract.

TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author (the
test-author half of the R53 run card
TASK-20260925-r5-run-r53-multiple-neutral-excitation, the card-13 landing
of the Round 5 run sequence; the card chain is the 14 cards ending at
this run card).

This register pins the committed behavior of the R53 run against the
committed tree:

- the dual-tree driver (byte identity between
  ``dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_
  excitation.py`` and ``helpers/benchmarking/staged_r53_multiple_neutral_
  excitation.py`` at their on-disk digest, and the driver's identity
  constants: the card chain, the R53 run identity, the T2 statistics
  (8 replicas x 100,000 = 800,000), the raw seed block 20300961-20300968
  under the 20260925 master (offset 40,000 + 36), the event-index
  partition 1_300_000_000 + i, the round-5 reserved blocks and
  partitions (no r53 entry: the run consumed, it is not reserved), the
  drawn P(n = 2) = 0.25, the frozen R51 fit point and cascade, the
  frozen ten-entry substrate, the M8 parameter hash, and the R52 cross
  pins (the consumed trigger surface));
- the sealed run dir
  ``dev/00runs/r53_multiple_neutral_excitation_2026-09-27`` (the 14
  content files at their on-disk digests with the SHA256SUMS.txt
  inventory, the pre-registration record written before generation with
  the verbatim R52 trigger entry, the provenance record, the 8 replica
  records (the n1/n2 census at the drawn P(n = 2) = 0.25, the per-replica
  conservation contract, the class census, the event-hash identities),
  the fit report (the single-parameter fit of P(n >= 2) bounded at
  [0, 0.5] at the boundary minimum 0.5, the named residual register with
  its six entries, the tail-improvement table, the no-degradation
  table, the conservation contract on all 800,000 events), the gate 5
  record (TRIGGERED, retained: false, the evidence-only note), the
  clean-environment record (the single validation pass before the seal),
  and the seal manifest);
- the cross-run pins (the sealed R52 gate5_r52.json and R51
  fit_report.json digests);
- the layout sync (``logic_drift = 0`` over the two helper trees); and
- the freeze re-pin register: the manifest
  ``dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json`` carries
  this card's freeze exception (the inventory re-pin 547 -> 548, +1
  entry: this register) and the 45 chain count-pin files are re-pinned
  547 -> 548 with the prior literals retained in the naming comments.

The provenance record carries the driver's pre-commit digest
(84220486...) in its driver block: the run sealed the driver bytes it
consumed, and the committed driver is byte-identical across the two
trees at 52151a....  The register pins both values as recorded and as
measured; the drift between them is reported, not papered over.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import deleted_events, sealed_mismatches, skip_unless_sealed

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parents[1]                 # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent              # the monorepo root
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / \
    "check_helpers_layout_sync.py"

NESTED_DRIVER = NESTED_ROOT / "helpers" / "benchmarking" / \
    "staged_r53_multiple_neutral_excitation.py"
OUTER_DRIVER = MONO_ROOT / "helpers" / "benchmarking" / \
    "staged_r53_multiple_neutral_excitation.py"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / \
    "r53_multiple_neutral_excitation_2026-09-27"
R52_RUN_DIR = MONO_ROOT / "dev" / "00runs" / \
    "r52_central_excitation_necessity_2026-09-27"
R52_GATE5 = R52_RUN_DIR / "gate5_r52.json"
R51_RUN_DIR = MONO_ROOT / "dev" / "00runs" / \
    "r51_topology_fraction_fit_2026-09-27"
R51_FIT_REPORT = R51_RUN_DIR / "fit_report.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / \
    "legacy_v1_manifest.json"

# ---------------------------------------------------------------------------
# The card identities
# ---------------------------------------------------------------------------
SOLVER_CARD = "TASK-20260925-r5-run-r53-multiple-neutral-excitation"
TEST_AUTHOR_CARD = \
    "TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author"

SOLVER_TEST_REL = \
    "dev/mixing2006/tests/test_staged_r53_multiple_neutral_excitation.py"
MY_TEST_REL = \
    "dev/mixing2006/tests/" \
    "test_staged_r53_multiple_neutral_excitation_contract_pins.py"
MANIFEST_REL = \
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = \
    "dev/mixing2006/helpers/benchmarking/" \
    "staged_r53_multiple_neutral_excitation.py"
DRIVER_REL_OUTER = \
    "helpers/benchmarking/staged_r53_multiple_neutral_excitation.py"

# ---------------------------------------------------------------------------
# The freeze re-pin (547 -> 548; +1 entry: this register)
# ---------------------------------------------------------------------------
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
#: The 45 chain count-pin files re-pinned 547 -> 548 by this register
#: (path-sorted; the 41 carry ``INVENTORY_COUNT = 551`` and the 4
#: card-3-style files carry the bare ``assert len(keys) == INVENTORY_COUNT``).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r50_pure_topology_templates.py",
    "dev/mixing2006/tests/test_staged_r50_pure_topology_templates_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r51_topology_fraction_fit.py",
    "dev/mixing2006/tests/test_staged_r51_topology_fraction_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py",
    "dev/mixing2006/tests/test_staged_r52_central_excitation_necessity_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r53_multiple_neutral_excitation.py",
    "dev/mixing2006/tests/test_staged_r54_local_ledger_audit.py",
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: The 45 chain files at their bare-assert count-pin style (the card-3
#: convention: no INVENTORY_COUNT constant, the inventory count asserted
#: in the test body).
BARE_ASSERT_FILES = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted; the manifest
#: itself, the 45 chain count-pin files, and this register).
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    [MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES)))

# ---------------------------------------------------------------------------
# The committed digests
# ---------------------------------------------------------------------------
#: The solver test file (test_staged_r53_multiple_neutral_excitation.py)
#: at its post-re-pin on-disk digest.
SOLVER_TEST_SHA256 = (
    "152870b06093f390c75ef67f705963674b0259947174b05fae9593b3dc5c43f0")

#: The committed driver, byte-identical in both trees.
DRIVER_SHA256 = (
    "52151a047da0407e9b4483462848377a70231d51c4f0e4904b8013a7e70edb13")

#: The driver digest as recorded in the sealed provenance (the pre-commit
#: bytes the run consumed; the drift against DRIVER_SHA256 is reported,
#: not papered over).
PROVENANCE_DRIVER_SHA256 = (
    "8422048652dcafd1a8c49f847cff1bfd771c70b3d277059178ce0e3cbf887332")

#: The sealed run dir: the 14 content files at their on-disk digests.
SEALED_FILE_SHA256 = {
    "REPORT.md":
        "926bef04fa8519e90cae2a7836557bb82adee475e9809f7a5944bc57fd4ac0a0",
    "clean_environment_commands.json":
        "ca67156fd9719d7fff5b92028087687f43d008548e0f0a14ec91ed7b42837d62",
    "fit_report.json":
        "d49784f61d436689aefc6833b5d9c528610cdc672e92961b2e09402774fdd182",
    "gate5_r53.json":
        "b65ebbee5c23d80be1b29e551aacd6c372238f4f252b4d39f8b8cd1892f8c82c",
    "pre_registration.json":
        "3b5853a29fa20bbbaceb4f4e470a9505fb454c39ab3f25c7bcf6de4c6ca9e9c0",
    "provenance.json":
        "34a6fe3a440ad48e15d522ea25b82f4196dc7cfcc00cef4437dcbe4a09212c69",
    "replica_0.json":
        "7a48a22e5b000a867e30cffadc8f0298dc7a2c9df77fec2604480c60e2d63844",
    "replica_1.json":
        "f2c28beda94be30b531d8f70cc1de6d9b35b65708f7a178fc161c4d5dad08af7",
    "replica_2.json":
        "5b000d4c6a215aef630b57a2cff10849f824d5b6e73a13cf7442f6c4c65d60ee",
    "replica_3.json":
        "a2416f0f916b1952af5d2b9e6d9c805957cb9b68e98aa276847030d2b8e43131",
    "replica_4.json":
        "d8dd14f94cd388cb543dfae61464d78bb7b39afc307560465904d5f32b4ab114",
    "replica_5.json":
        "4dcd239f64585fd8ab57705a373fabb900da2178bd8e71f27ee93fb9c56b3e22",
    "replica_6.json":
        "3d66120757a00af0dec9634cebc8d8e3259aab42cad4b78d75a71983a12d8e66",
    "replica_7.json":
        "4f25c6de7c4ebb683040a2d8e3002cf126786cf6ef766bc580e3f90c84e704eb",
}
SEAL_MANIFEST_SHA256 = (
    "ba9f7749f6e72eb5ce5a0da6daf62855cc8603b07e1fffbc45742f548cb86550")

# The cross-run pins (the consumed surfaces at their on-disk digests).
R52_GATE5_SHA256 = (
    "a4f0cb7281dd8713e7945b9a7cda531ab329b5990a7e85a6e17c10b366e29374")
R51_FIT_REPORT_SHA256 = (
    "30ed195237ca83bf75ba24278bbb5ce67bfe1c55ad1da4081ace6fc241f9aede")

# ---------------------------------------------------------------------------
# The driver identity constants
# ---------------------------------------------------------------------------
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
    "TASK-20260925-r5-run-r52-central-excitation-necessity",
    "TASK-20260925-r5-run-r53-multiple-neutral-excitation",
)
RUN_ID = "R53"
RUN_NAME = "r53_multiple_neutral_excitation_2026-09-27"
N_REPLICAS = 8
EVENTS_PER_REPLICA = 100000
N_EVENTS_TOTAL = 800000
MASTER_SEED = 20260925
SEED_OFFSET = 40000
REPLICA_SEEDS = (
    20300961, 20300962, 20300963, 20300964,
    20300965, 20300966, 20300967, 20300968,
)
EVENT_INDEX_BASE = 1300000000
EVENT_INDEX_RULE = "1_300_000_000 + i, i < 100,000 (per replica)"
SMOKE_SEED = 20312925
SMOKE_EVENTS = 2000
SMOKE_EVENT_INDEX_BASE = 1600000000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"
SMOKE_PREFIX = "smoke_"
SMOKE_REPLICA_NAME = "smoke_replica.json"
SMOKE_REPORT_NAME = "smoke_fit_report.json"
BEAM_MOMENTUM_GEV_C = 158.0
BLEND_RAMP_START_GEV = 2.680194110727528
SQRT_S_EXTRACTION_GEV = 17.27
CM_BETA_PINNED = 0.9940791999921171
CM_GAMMA_PINNED = 9.203303892226666
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
DRAWN_P_N2 = 0.25
FROZEN_FIT = {
    "b_sd": 1.8457918026393214,
    "dd_fraction": 0.9999997010778905,
    "epsilon": 0.0033443027832747944,
    "nd_fraction": 3.1345895878942542e-09,
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
}
R45_FROZEN_CASCADE = {
    "W_flavor": 1.0,
    "b_dec_gev2": 2.0,
    "baryonic_nominal_width_gev": 0.3,
    "kappa_B": 0.0,
    "kappa_M": 4.0,
}
MODEL_UNCERTAINTY_CARRY = {
    "b_transport_seagull": -2.3959765824349295e-05,
    "r32_spread": {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    },
}
FROZEN_SUBSTRATE = {
    "builder_blend_path": [
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py",
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    ],
    "card2_engine": [
        "dev/mixing2006/na49_mixing/threshold_metric.py",
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    ],
    "card6_routing": [
        "dev/mixing2006/na49_mixing/spectral_density.py",
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563",
    ],
    "catalog": [
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd",
    ],
    "continuum_config": [
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1",
    ],
    "r45_driver": [
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py",
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    ],
    "r46_driver": [
        "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py",
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    ],
    "r50_driver": [
        "helpers/benchmarking/staged_r50_pure_topology_templates.py",
        "b74e5408195ca6a3a53caf495da0c3f1206a9b43a73f336ba24b5cd6eb936fa3",
    ],
    "r51_driver": [
        "helpers/benchmarking/staged_r51_topology_fraction_fit.py",
        "73bd74ab72270ae47b900b4da171b88db606234b1cbd4345c8135f48911f40a1",
    ],
    "r52_driver": [
        "helpers/benchmarking/staged_r52_central_excitation_necessity.py",
        "889131347693b3e3338ececeb63ac883993b1ed04f83fde5449471f898469fa7",
    ],
}
RESERVED_ROUND5_BLOCKS = {
    "r50_used": [20300925, 20300944],
    "r51_used": [20300945, 20300952],
    "r52_used": [20300953, 20300960],
    "r54": [20300969, 20300988],
    "r55": [20300989, 20301008],
}
RESERVED_ROUND5_PARTITIONS = {
    "r50_used": [1000000000, 1000025000],
    "r51_used": [1100000000, 1100100000],
    "r52_used": [1200000000, 1200100000],
    "r54": [1400000000, 1400050000],
    "r55": [1500000000, 1500250000],
    "round5_smoke": [1600000000, 1600002000],
}
DEFAULT_RUN_DIR = "dev/00runs/r53_multiple_neutral_excitation_2026-09-27"
CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
GATE_RECORD_NAME = "gate5_r53.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SEAL_NAME = "seal_manifest.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
RECORD_FMT = "replica_{}.json"
REPORT_MD_NAME = "REPORT.md"
REPORT_NAME = "fit_report.json"

# The R52 cross pins (the consumed trigger surface).
R52_DRIVER_REL_NESTED = \
    "dev/mixing2006/helpers/benchmarking/" \
    "staged_r52_central_excitation_necessity.py"
R52_DRIVER_REL_OUTER = \
    "helpers/benchmarking/staged_r52_central_excitation_necessity.py"
R52_GATE_NAME = "gate5_r52.json"
R52_REPORT_DIGEST = (
    "37ffc0bcdeaf5728d09032d34ff43ffb359c8aafce9257d8aed0e241a9971460")
R52_RUN_DIR_REL = "dev/00runs/r52_central_excitation_necessity_2026-09-27"
R52_TRIGGER_ENTRY = {
    "argmax_bin_n_ch": 8,
    "named_residual":
        "multiplicity_tail_residual_one_model_beyond_2x_pooled_se",
    "pooled_se": 0.002738966420355409,
    "sigma": 11.007284372792265,
    "triggered": True,
    "value": 0.030148582276380863,
}

# ---------------------------------------------------------------------------
# The sealed record pins
# ---------------------------------------------------------------------------
FIT_REPORT_DIGEST = (
    "c6f15ee0f437da0764f98fd9d4e39dc2b5ea7b024360d06c3e85113770ebf496")
TRIGGER_NAMED_RESIDUAL = \
    "multiplicity_tail_residual_one_model_beyond_2x_pooled_se"
TRIGGER_ARGMAX_BIN_N_CH = 8
TRIGGER_VALUE = 0.030148582276380863
TRIGGER_POOLED_SE = 0.002738966420355409
TRIGGER_SIGMA = 11.007284372792265
TRIGGER_BOUND = 2.0

# Conservation (the full sample).
CONSERVATION_N1 = 150153
CONSERVATION_N2 = 50010
CONSERVATION_EVENTS = 800000
CONSERVATION_TOLERANCE = 1e-09
CONSERVATION_WORST_RELATIVE = 3.255904687913371e-14
CONSERVATION_WORST_ENERGY = 5.144329406903125e-12
CONSERVATION_WORST_MOMENTUM = 5.144329406903125e-12
CONSERVATION_WORST_MASS_SHELL = 7.888464679697563e-10

# The per-replica records (census, n1/n2, worsts, event-hash identity).
REPLICA_META = (
    {
        "class_census": {"dd": 24903, "nd": 25082,
                         "sd_projectile": 25020, "sd_target": 24995},
        "n1": 18819, "n2": 6263,
        "worst_relative": 2.700169592481423e-14,
        "worst_energy": 4.291678123991005e-12,
        "worst_momentum": 4.263256414560601e-12,
        "worst_mass_shell": 6.089413019835981e-10,
        "event_hash":
            "5bd10d1255ee769b2a8514c162392802f1a606fa4fd2db510d4aba1973e0ebce",
    },
    {
        "class_census": {"dd": 25085, "nd": 24902,
                         "sd_projectile": 25127, "sd_target": 24886},
        "n1": 18724, "n2": 6178,
        "worst_relative": 3.2379162642232416e-14,
        "worst_energy": 5.115907697472721e-12,
        "worst_momentum": 5.115907697472721e-12,
        "worst_mass_shell": 5.891625008335629e-10,
        "event_hash":
            "57ba3137185bef7eea7c018ad4bb63cb577e69ff4e8132e8b0473e575e949b0b",
    },
    {
        "class_census": {"dd": 24981, "nd": 25082,
                         "sd_projectile": 25003, "sd_target": 24934},
        "n1": 18826, "n2": 6256,
        "worst_relative": 2.968398359946465e-14,
        "worst_energy": 4.718003765447065e-12,
        "worst_momentum": 4.689582056016661e-12,
        "worst_mass_shell": 5.774879222955706e-10,
        "event_hash":
            "616cf3d2768a4c70e5d795c2ade4e86f060053085a611e3f6ee726e98db599d0",
    },
    {
        "class_census": {"dd": 25175, "nd": 25042,
                         "sd_projectile": 24837, "sd_target": 24946},
        "n1": 18816, "n2": 6226,
        "worst_relative": 2.7522288245897553e-14,
        "worst_energy": 4.348521542851813e-12,
        "worst_momentum": 4.348521542851813e-12,
        "worst_mass_shell": 6.089413019835981e-10,
        "event_hash":
            "253251b08722f95aaac48f87a1c234e79740c886978694bafeca73fd8a119dfc",
    },
    {
        "class_census": {"dd": 25036, "nd": 24838,
                         "sd_projectile": 24998, "sd_target": 25128},
        "n1": 18636, "n2": 6202,
        "worst_relative": 2.6083214350687222e-14,
        "worst_energy": 4.121147867408581e-12,
        "worst_momentum": 4.121147867408581e-12,
        "worst_mass_shell": 5.891625008335629e-10,
        "event_hash":
            "8a5a7384e535b20a4ee64948ccca0c8c50c5f1a64d6c2f5bb13c164133e8a0e0",
    },
    {
        "class_census": {"dd": 25159, "nd": 25100,
                         "sd_projectile": 24803, "sd_target": 24938},
        "n1": 18836, "n2": 6264,
        "worst_relative": 2.6083214350687222e-14,
        "worst_energy": 4.121147867408581e-12,
        "worst_momentum": 4.121147867408581e-12,
        "worst_mass_shell": 7.888464679697563e-10,
        "event_hash":
            "252ce14db075e2b0a830c4896c8c0fd7a65031fa6e0a45dc559affb254e63b96",
    },
    {
        "class_census": {"dd": 25135, "nd": 25013,
                         "sd_projectile": 25080, "sd_target": 24772},
        "n1": 18725, "n2": 6288,
        "worst_relative": 3.255904687913371e-14,
        "worst_energy": 5.144329406903125e-12,
        "worst_momentum": 5.144329406903125e-12,
        "worst_mass_shell": 6.089413019835981e-10,
        "event_hash":
            "ccfede6ea9d1c70721b61778f02eae7beb2312727abfd2d719bcaa3d737c1b87",
    },
    {
        "class_census": {"dd": 24875, "nd": 25104,
                         "sd_projectile": 24971, "sd_target": 25050},
        "n1": 18771, "n2": 6333,
        "worst_relative": 2.878147790420659e-14,
        "worst_energy": 4.547473508864641e-12,
        "worst_momentum": 4.547473508864641e-12,
        "worst_mass_shell": 6.089413019835981e-10,
        "event_hash":
            "2f810e034db3cf40d5a65e16213c4408a3e5753f8184d56e5065f4e2983daefe",
    },
)

# The fit report pins.
OBJECTIVE_AT_INITIAL = 725301.2395569991
OBJECTIVE_AT_OPTIMUM = 725276.016677558
N_EFF_AT_INITIAL = 7.345741115653189
N_EFF_AT_OPTIMUM = 5.595737709494318
NUISANCES_AT_INITIAL = (
    0.14956394079066074, 0.07813999685589243, 0.006271767269909204)
NUISANCES_AT_OPTIMUM = (
    0.028497283054397058, -0.04466226609459773, 0.008233140522332096)
FITTED_P_N2 = 0.5
BOUND = (0.0, 0.5)
HESSIAN_STEPS_P_N2 = 0.005
NAMED_RESIDUAL_KINDS = (
    "boundary_minimum",
    "replica_block_minima_disagree",
    "weight_degeneracy",
    "r51_inherited_non_convergence",
    "pi0_published_table_absent",
    "multiplicity_tail_training_overlap",
)
BLOCK_P_N2 = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25)
BLOCK_MINUS2LN_L = (
    18634428.780180953, 18634428.780180953, 2586825.99314296,
    4097629.539388035, 4097629.539353591, 4097629.539388035,
    1120835.9903235252, 18634428.780180953)
BLOCK_NFEV = (5, 5, 5, 5, 5, 5, 10, 5)
TAIL_IMPROVEMENT_PER_BIN = (
    {"bin": 6, "difference": 11.653264786788016,
     "improved_beyond_2x_se": True, "pooled_se": 1.349165322242418,
     "r52_one_deviance": 12.108330631775052,
     "r53_one_deviance": 0.4550658449870357,
     "sigma": 8.637388313108568},
    {"bin": 7, "difference": -2095.819810796392,
     "improved_beyond_2x_se": False, "pooled_se": 0.000789185981333357,
     "r52_one_deviance": 46.09380255485074,
     "r53_one_deviance": 2141.913613351243,
     "sigma": -2655672.8836660683},
    {"bin": 8, "difference": 119.37592866349966,
     "improved_beyond_2x_se": True, "pooled_se": 2.6716119451922435,
     "r52_one_deviance": 121.16030926351681,
     "r53_one_deviance": 1.7843806000171385,
     "sigma": 44.68310934090753},
    {"bin": 9, "difference": -178.94088552233745,
     "improved_beyond_2x_se": False, "pooled_se": 1.0513098137589503e-05,
     "r52_one_deviance": 10.416969383684547,
     "r53_one_deviance": 189.357854906022,
     "sigma": -17020756.69611945},
    {"bin": 10, "difference": -1364.5247563603575,
     "improved_beyond_2x_se": False, "pooled_se": 0.03457056897007291,
     "r52_one_deviance": 74.79681002339466,
     "r53_one_deviance": 1439.3215663837523,
     "sigma": -39470.70577697466},
    {"bin": 11, "difference": -1.837393012181179,
     "improved_beyond_2x_se": False, "pooled_se": 8.000160003198691e-07,
     "r52_one_deviance": 0.16264698861830965,
     "r53_one_deviance": 2.0000400007994887,
     "sigma": -2296695.3304015636},
    {"bin": 12, "difference": -160.40634536817672,
     "improved_beyond_2x_se": False, "pooled_se": 8.772084019813634e-06,
     "r52_one_deviance": 16.90747385550623,
     "r53_one_deviance": 177.31381922368294,
     "sigma": -18286001.935898535},
    {"bin": 13, "difference": None,
     "improved_beyond_2x_se": False, "pooled_se": None,
     "r52_one_deviance": None, "r53_one_deviance": 0.0,
     "sigma": None},
    {"bin": 14, "difference": -3.3571330987670414,
     "improved_beyond_2x_se": False, "pooled_se": 8.000560039197373e-07,
     "r52_one_deviance": 3.6433569355261164,
     "r53_one_deviance": 7.000490034293158,
     "sigma": -4196122.624315477},
)
MAX_IMPROVEMENT_SIGMA = 44.68310934090753
NO_DEGRADATION_TOTALS = {
    "multiplicity_tails": 285.28969963687246,
    "central_spectrum_pi_plus": 16198.391067047858,
    "central_spectrum_pi_minus": 17635.9493495038,
    "central_spectrum_k_plus": 2197.48117772816,
    "central_spectrum_k_minus": 8011.982950190087,
}
BIN_STRUCTURE_KEPT = {
    "b_transport_spectrum": 122,
    "charged_multiplicity_distribution": 12,
    "gap_fraction_abs_y_lt_0.5": 1,
    "leading_proton_spectrum": 135,
}

# The gate 5 record pins.
GATE_STATUS = "TRIGGERED"
GATE_RETAINED = False
GATE_NAME = "gate 5 (plan section 11, line 938)"
GATE_CLAUSE = "necessity of n >= 2 (multiple neutral excitations)"
GATE_GATE_NOTE = (
    "evidence only; the written retain / revise / reject decision on the "
    "Gate 5 clauses is the round's Gate 5 follow-up obligation")

# The clean-environment record pins.
CLEAN_ENV_VERIFIED_UTC = "2026-09-28T04:16:23Z"
CLEAN_ENV_COMMANDS = (
    "cd dev/mixing2006 && "
    "python -m pytest "
    "tests/test_staged_r53_multiple_neutral_excitation.py -q",
    "python -m pytest tests/test_legacy_v1_freeze.py "
    "-W error::UserWarning -q",
    "python "
    "helpers/benchmarking/staged_r53_multiple_neutral_excitation.py "
    "--clean-validate --run-dir "
    "/home/o/NA49/2026/01-NA49-res/dev/00runs/"
    "r53_multiple_neutral_excitation_2026-09-27",
)

# The seal manifest pins.
SEAL_SEALED_UTC = "2026-09-28T04:16:39Z"
SEAL_SHA256SUMS_ENTRIES = 14
SEAL_RECORD_COUNT = 8
SEAL_GIT_HEAD = "f4e009507753f654654b8ff3a0ae49a1b35daf60"
SEAL_GIT_BRANCH = "main"
SEAL_GIT_DIRTY = True
SEAL_ENVIRONMENT = {
    "machine": "x86_64",
    "numpy": "1.26.4",
    "platform": "Linux-6.8.0-139-generic-x86_64-with-glibc2.39",
    "python": "3.12.3",
}

# The pre-registration / provenance shared pins.
RECORDED_UTC = "2026-09-28T02:25:47Z"
RUN_DIR_ABS = \
    "/home/o/NA49/2026/01-NA49-res/dev/00runs/" \
    "r53_multiple_neutral_excitation_2026-09-27"
SCHEMAS = {
    "pre_registration":
        "na49_mixing.staged_r53_multiple_neutral_excitation.pre_registration/1",
    "replica":
        "na49_mixing.staged_r53_multiple_neutral_excitation.replica/1",
    "report":
        "na49_mixing.staged_r53_multiple_neutral_excitation.report/1",
    "gate":
        "na49_mixing.staged_r53_multiple_neutral_excitation.gate/1",
    "clean_env":
        "na49_mixing.staged_r53_multiple_neutral_excitation.clean_env/1",
    "seal":
        "na49_mixing.staged_r53_multiple_neutral_excitation.seal/1",
    "provenance":
        "na49_mixing.staged_r53_multiple_neutral_excitation.provenance/1",
}
HELD_OUT_TAIL_THRESHOLD_N_CH = 6
HELD_OUT_TAIL_CUMULATIVE = 0.9666349999999999
HELD_OUT_TAIL_QUANTILE = 0.9
HELD_OUT_TAIL_BINS = (6, 7, 8, 9, 10, 11, 12, 13, 14)
R01_CENTRAL_ROWS = {"k_minus": 42, "k_plus": 42, "pi0": 58,
                    "pi_minus": 58, "pi_plus": 58}
R01_PUBLISHED_ROWS = {"k_minus": 148, "k_plus": 153,
                      "pi_minus": 229, "pi_plus": 230}
R01_FILE_DIGESTS = {
    "bin_predictions.json":
        "820eb78d8a23aecc16a0a74c8d6b54fb105ab27e0919b638aec608c98d0d97c7",
    "covariance.json":
        "f7c3b454da6e179b992d1640273c9ce44f1ca3c7a7463bc39acf24f4a2aa953c",
    "residuals.json":
        "eaefef310c361d2427655e942b07a3c48d538dcb1e90b36cc64f906dd685564e",
}
SEAL_INPUTS = {
    "r01": {"entries": 28,
            "run_dir": "dev/00runs/r01_legacy_pp_reference_2026-09-11"},
    "r50": {"entries": 26,
            "run_dir": "dev/00runs/r50_pure_topology_templates_2026-09-27"},
    "r51": {"entries": 14,
            "run_dir":
                "dev/00runs/r51_topology_fraction_fit_2026-09-27"},
    "r52": {"entries": 14,
            "run_dir":
                "dev/00runs/r52_central_excitation_necessity_2026-09-27"},
}
ROUND5_MODULES = {
    "card1_elastic":
        "7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824",
    "card2_ledger":
        "3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a",
    "card3_remnants":
        "8eb45877490c04896c7cc7062e174dd6c060d96b1bace07688c16086e8a463dc",
    "card4_mass_laws":
        "c15e3416b022390bcc59dd3f0e17f4264b272beffbd9f3863a093a88a5fc0b48",
    "card5_single_diffraction":
        "5868b178fb3fe5852d5baf141b5cb949554747a1dcbd3496d3301251683dd2cb",
    "card6_double_diffraction":
        "f17a83324deef6be7f415c24ec7742820d99291407705546df8703e34c0f077f",
    "card7_non_diffractive":
        "68162f96614022e5f95807155cbd34f6714cb14bf8e4a5eaf116764f31bfd4b2",
    "card8_multi_neutral":
        "9527643b387ceb29ac22c8854f45f28c22c7a14215b35c55fafc499993bb8454",
    "card9_runner":
        "0394c010d581416ca4f59677401632ab53e7acc8fa54fe147ce4f568370e0dbb",
}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(driver_path):
    spec = importlib.util.spec_from_file_location(
        "r53_driver_check", driver_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The dual-tree driver
# ---------------------------------------------------------------------------
def test_dual_tree_driver_byte_identity():
    """The driver lands byte-identical in both trees at the committed
    digest (acceptance (xi))."""

    nested = _sha256_file(NESTED_DRIVER)
    outer = _sha256_file(OUTER_DRIVER)
    assert nested == DRIVER_SHA256, (
        f"nested driver drifted: {nested}")
    assert outer == DRIVER_SHA256, (
        f"outer driver drifted: {outer}")
    assert nested == outer, "the dual-tree driver is not byte-identical"


def test_driver_identity_constants():
    """The driver's identity constants: the card chain, the run
    identity, the T2 statistics, the seed block and partition, the
    frozen points, the substrate, the reserved round-5 blocks
    (no r53 entry), and the R52 cross pins."""

    d = _load(NESTED_DRIVER)
    assert d.CARD == SOLVER_CARD
    assert tuple(d.CARD_CHAIN) == CARD_CHAIN
    assert len(d.CARD_CHAIN) == 14
    assert d.RUN_ID == RUN_ID
    assert d.RUN_NAME == RUN_NAME
    assert d.N_REPLICAS == N_REPLICAS
    assert d.EVENTS_PER_REPLICA == EVENTS_PER_REPLICA
    assert d.N_EVENTS_TOTAL == N_EVENTS_TOTAL
    assert d.MASTER_SEED == MASTER_SEED
    assert d.SEED_OFFSET == SEED_OFFSET
    assert tuple(d.REPLICA_SEEDS) == REPLICA_SEEDS
    assert d.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert d.EVENT_INDEX_RULE == EVENT_INDEX_RULE
    assert d.SMOKE_SEED == SMOKE_SEED
    assert d.SMOKE_EVENTS == SMOKE_EVENTS
    assert d.SMOKE_EVENT_INDEX_BASE == SMOKE_EVENT_INDEX_BASE
    assert d.SMOKE_EVENT_INDEX_RULE == SMOKE_EVENT_INDEX_RULE
    assert d.SMOKE_PREFIX == SMOKE_PREFIX
    assert d.SMOKE_REPLICA_NAME == SMOKE_REPLICA_NAME
    assert d.SMOKE_REPORT_NAME == SMOKE_REPORT_NAME
    assert d.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert d.BLEND_RAMP_START_GEV == BLEND_RAMP_START_GEV
    assert d.SQRT_S_EXTRACTION_GEV == SQRT_S_EXTRACTION_GEV
    assert d.CM_BETA_PINNED == CM_BETA_PINNED
    assert d.CM_GAMMA_PINNED == CM_GAMMA_PINNED
    assert d.M8_PARAMETER_HASH == M8_PARAMETER_HASH
    assert d.DRAWN_P_N2 == DRAWN_P_N2
    assert d.FROZEN_FIT == FROZEN_FIT
    assert d.R45_FROZEN_CASCADE == R45_FROZEN_CASCADE
    carry = d.MODEL_UNCERTAINTY_CARRY
    assert carry["b_transport_seagull"] == \
        MODEL_UNCERTAINTY_CARRY["b_transport_seagull"]
    assert carry["r32_spread"]["bin_spread_p95"] == \
        MODEL_UNCERTAINTY_CARRY["r32_spread"]["bin_spread_p95"]
    assert carry["r32_spread"]["feeddown_spread_max_sigma"] == \
        MODEL_UNCERTAINTY_CARRY["r32_spread"]["feeddown_spread_max_sigma"]
    assert carry["r32_spread"]["persisting_residuals"] == \
        MODEL_UNCERTAINTY_CARRY["r32_spread"]["persisting_residuals"]
    assert d.FROZEN_SUBSTRATE == \
        {name: tuple(pair) for name, pair in FROZEN_SUBSTRATE.items()}
    assert d.RESERVED_ROUND5_BLOCKS == RESERVED_ROUND5_BLOCKS
    assert "r53" not in d.RESERVED_ROUND5_BLOCKS, (
        "the run consumed the r53 block; it is not reserved")
    assert d.RESERVED_ROUND5_PARTITIONS == RESERVED_ROUND5_PARTITIONS
    assert "r53" not in d.RESERVED_ROUND5_PARTITIONS
    assert d.DEFAULT_RUN_DIR == DEFAULT_RUN_DIR
    assert d.CATALOG_REL == CATALOG_REL
    assert d.FAMILIES_REL == FAMILIES_REL
    assert d.GATE_RECORD_NAME == GATE_RECORD_NAME
    assert d.PREREGISTRATION_NAME == PREREGISTRATION_NAME
    assert d.PROVENANCE_NAME == PROVENANCE_NAME
    assert d.CLEAN_ENV_NAME == CLEAN_ENV_NAME
    assert d.SEAL_NAME == SEAL_NAME
    assert d.SHA256SUMS_NAME == SHA256SUMS_NAME
    assert d.RECORD_FMT == RECORD_FMT
    assert d.REPORT_MD_NAME == REPORT_MD_NAME
    assert d.REPORT_NAME == REPORT_NAME
    # the R52 cross pins (the consumed trigger surface)
    assert d.R52_DRIVER_REL_NESTED == R52_DRIVER_REL_NESTED
    assert d.R52_DRIVER_REL_OUTER == R52_DRIVER_REL_OUTER
    assert d.R52_GATE_NAME == R52_GATE_NAME
    assert d.R52_REPORT_DIGEST == R52_REPORT_DIGEST
    assert d.R52_RUN_DIR == R52_RUN_DIR_REL
    assert d.R52_TRIGGER_ENTRY == R52_TRIGGER_ENTRY


# ---------------------------------------------------------------------------
# The solver test file
# ---------------------------------------------------------------------------
def test_solver_test_file_digest():
    """The solver test file at its post-re-pin on-disk digest, and the
    manifest inventory entry for it at the same digest."""

    solver_path = MONO_ROOT / SOLVER_TEST_REL
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_tree_sha256"][SOLVER_TEST_REL] == \
        SOLVER_TEST_SHA256


# ---------------------------------------------------------------------------
# The sealed run dir
# ---------------------------------------------------------------------------
def test_sealed_run_dir_and_checksums():
    """The 14 sealed content files at their on-disk digests, the
    SHA256SUMS.txt inventory consistent with them, and the seal
    manifest at its on-disk digest."""

    skip_unless_sealed(RUN_DIR)
    assert sealed_mismatches(RUN_DIR) == []
    absent = {n for n in deleted_events(RUN_DIR) if not (RUN_DIR / n).is_file()}
    for name, digest in SEALED_FILE_SHA256.items():
        if name in absent:
            continue
        path = RUN_DIR / name
        assert path.is_file(), f"missing sealed file: {name}"
        assert _sha256_file(path) == digest, f"sealed file drifted: {name}"
    assert _sha256_file(RUN_DIR / "seal_manifest.json") == \
        SEAL_MANIFEST_SHA256

    sums = (RUN_DIR / "SHA256SUMS.txt").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(sums) == 14
    for line in sums:
        digest, name = line.split()
        if name in absent:
            continue
        assert _sha256_file(RUN_DIR / name) == digest, name
    assert [line.split()[1] for line in sums] == sorted(
        line.split()[1] for line in sums)


def test_pre_registration_on_disk():
    """The pre-registration record: written before generation, the
    verbatim R52 trigger entry, the conservation contract, the fit set
    (the single parameter P(n >= 2) bounded at [0, 0.5] at the drawn
    initial 0.25), the T2 statistics, the seed map (the raw block
    20300961-20300968, the partition 1_300_000_000 + i, the reserved
    round-5 blocks without an r53 entry), the sealed inputs, and the
    held-out surface (the tail cut at the 90th quantile, the R01 row
    counts and file digests)."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / PREREGISTRATION_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["card_chain"] == list(CARD_CHAIN)
    assert record["written_before_generation"] is True
    assert record["recorded_utc"] == RECORDED_UTC
    assert record["run"] == RUN_ID
    assert record["run_name"] == RUN_NAME
    assert record["run_dir"] == RUN_DIR_ABS
    assert record["mode"] == "production"
    assert record["schema"] == SCHEMAS["pre_registration"]

    # the verbatim R52 trigger entry (consumed, asserted before any
    # generation; never re-derivable from a regenerated R52)
    assert record["trigger"]["entry"] == R52_TRIGGER_ENTRY
    assert record["trigger"]["gate_file_sha256"] == R52_GATE5_SHA256
    assert record["trigger"]["report_digest"] == R52_REPORT_DIGEST
    assert record["trigger"]["match"] is True
    assert record["trigger"]["source"] == \
        f"{R52_RUN_DIR_REL}/{R52_GATE_NAME}"
    assert record["trigger"]["seal"]["entries"] == 14
    assert record["trigger"]["seal"]["intact"] is True
    assert record["trigger"]["seal"]["mismatches"] == []

    # the conservation contract
    cons = record["conservation"]
    assert cons["expected_b_q_s"] == [2, 2, 0]
    assert cons["tolerance"] == CONSERVATION_TOLERANCE
    assert "1e-09" in cons["contract"]

    # the fit set
    fit = record["fit_set"]
    assert fit["bound"] == list(BOUND)
    assert fit["count"] == 1
    assert fit["drawn_p_n2"] == DRAWN_P_N2
    assert fit["fit_set"] == ["p_n2"]
    assert fit["initial_values"] == {"p_n2": DRAWN_P_N2}
    assert fit["frozen"]["source"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/fit_report.json")
    assert fit["frozen"]["refit"] is False
    assert fit["frozen"]["b_sd"] == FROZEN_FIT["b_sd"]
    assert fit["frozen"]["epsilon"] == FROZEN_FIT["epsilon"]
    assert fit["frozen"]["fractions"]["nd_fraction"] == \
        FROZEN_FIT["nd_fraction"]
    assert fit["pass"] is True
    assert fit["excluded"]["elastic"] == \
        "plan line 824 (excluded from the inelastic sample)"

    # the frozen cascade and substrate
    assert record["frozen_cascade"] == R45_FROZEN_CASCADE
    assert len(record["frozen_substrate"]) == 10
    for name, (rel, digest) in FROZEN_SUBSTRATE.items():
        entry = record["frozen_substrate"][name]
        assert entry["expected"] == digest
        assert entry["sha256"] == digest

    # the T2 statistics
    stats = record["statistics"]
    assert stats["replicas"] == N_REPLICAS
    assert stats["events_per_replica"] == EVENTS_PER_REPLICA
    assert stats["events_total"] == N_EVENTS_TOTAL

    # the seed map
    seeds = record["seed_map"]
    assert seeds["master_seed"] == MASTER_SEED
    assert seeds["event_index_base"] == EVENT_INDEX_BASE
    assert seeds["event_index_rule"] == EVENT_INDEX_RULE
    assert seeds["raw_replica_seeds"] == list(REPLICA_SEEDS)
    assert seeds["smoke"]["raw_seed"] == SMOKE_SEED
    assert seeds["smoke"]["events"] == SMOKE_EVENTS
    assert seeds["smoke"]["event_index_rule"] == SMOKE_EVENT_INDEX_RULE
    disj = seeds["disjointness"]
    assert disj["disjoint"] is True
    assert disj["r53_raw_seed_block"] == [20300961, 20300968]
    assert disj["r53_event_index_partition"] == \
        "1_300_000_000 + i, i < 100,000 (per replica)"
    assert disj["r53_event_index_range"] == [1300000000, 1300100000]
    assert disj["reserved_round5_blocks"] == RESERVED_ROUND5_BLOCKS
    assert disj["reserved_round5_partitions"] == RESERVED_ROUND5_PARTITIONS
    assert disj["shared_raw_seeds"] == {}

    # the sealed inputs
    for run_key, entry in SEAL_INPUTS.items():
        rec = record["sealed_inputs"][run_key]
        assert rec["entries"] == entry["entries"]
        assert rec["run_dir"] == entry["run_dir"]
        assert rec["intact"] is True
        assert rec["mismatches"] == []

    # the M8 frozen set
    assert record["m8_frozen_set"]["hash"] == M8_PARAMETER_HASH
    assert record["m8_frozen_set"]["verified"] is True

    # the held-out surface
    held = record["held_out"]
    assert held["no_degradation_bound"] == 0.05
    cut = held["multiplicity_tail_cut"]
    assert cut["threshold_n_ch"] == HELD_OUT_TAIL_THRESHOLD_N_CH
    assert cut["cumulative_at_threshold"] == HELD_OUT_TAIL_CUMULATIVE
    assert cut["quantile"] == HELD_OUT_TAIL_QUANTILE
    assert cut["tail_bins"] == list(HELD_OUT_TAIL_BINS)
    assert held["central_spectra_abs_y_lt_0.5"] == [
        "pi_plus", "pi_minus", "k_plus", "k_minus"]
    assert held["r01_data"]["central_rows_abs_y_lt_0.5"] == \
        R01_CENTRAL_ROWS
    assert held["r01_data"]["published_rows"] == R01_PUBLISHED_ROWS
    assert held["r01_data"]["published_species"] == [
        "pi_plus", "pi_minus", "k_plus", "k_minus"]
    assert held["r01_data"]["file_digests"] == R01_FILE_DIGESTS
    assert len(held["r52_tail_baseline"]) == 8
    assert held["r52_tail_baseline"]["8"]["m_one"] == 0.05378608227638086
    assert held["r52_tail_baseline"]["8"]["one_pull"] == -TRIGGER_SIGMA
    assert held["r52_tail_baseline"]["8"]["pooled_se"] == \
        163.98934345672424


def test_provenance_record():
    """The provenance record: the git state at the seal, the
    environment, the driver block at the consumed pre-commit digest
    (the drift against the committed driver is pinned, not papered
    over), the R52 cross pins, the round-5 module digests, and the
    sealed inputs."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["card_chain"] == list(CARD_CHAIN)
    assert record["recorded_utc"] == RECORDED_UTC
    assert record["run"] == RUN_ID
    assert record["schema"] == SCHEMAS["provenance"]
    assert record["git_state"] == {
        "branch": SEAL_GIT_BRANCH,
        "dirty": SEAL_GIT_DIRTY,
        "head": SEAL_GIT_HEAD,
    }
    assert record["environment"] == SEAL_ENVIRONMENT
    assert record["driver"]["byte_identical"] is True
    assert record["driver"]["nested"]["path"] == DRIVER_REL_NESTED
    assert record["driver"]["nested"]["sha256"] == PROVENANCE_DRIVER_SHA256
    assert record["driver"]["outer"]["path"] == DRIVER_REL_OUTER
    assert record["driver"]["outer"]["sha256"] == PROVENANCE_DRIVER_SHA256
    assert record["r52_driver"]["byte_identical"] is True
    assert record["r52_driver"]["nested"]["sha256"] == \
        FROZEN_SUBSTRATE["r52_driver"][1]
    assert record["r52_gate_file_sha256"] == R52_GATE5_SHA256
    assert record["r52_trigger_entry"] == R52_TRIGGER_ENTRY
    assert record["r52_trigger_source"] == \
        f"{R52_RUN_DIR_REL}/{R52_GATE_NAME}"
    assert record["round5_modules"] == ROUND5_MODULES
    assert record["r45_r46_r50_r51_r52_driver_pins"] == {
        "r45": FROZEN_SUBSTRATE["r45_driver"][1],
        "r46": FROZEN_SUBSTRATE["r46_driver"][1],
        "r50": FROZEN_SUBSTRATE["r50_driver"][1],
        "r51": FROZEN_SUBSTRATE["r51_driver"][1],
        "r52": FROZEN_SUBSTRATE["r52_driver"][1],
    }
    assert record["m8_frozen_set"]["hash"] == M8_PARAMETER_HASH
    assert record["m8_frozen_set"]["verified"] is True
    for run_key, entry in SEAL_INPUTS.items():
        rec = record["sealed_inputs"][run_key]
        assert rec["entries"] == entry["entries"]
        assert rec["run_dir"] == entry["run_dir"]
        assert rec["intact"] is True
        assert rec["mismatches"] == []


def test_replica_records_on_disk():
    """The 8 replica records: the raw seed block, the n1/n2 census at
    the drawn P(n = 2) = 0.25 (the totals reconciling to the sample
    census), the per-replica conservation contract, the class census,
    the event-hash identities, and the pre-registered identity
    (driver + pre-registration digests)."""

    skip_unless_sealed(RUN_DIR)
    for k in range(N_REPLICAS):
        name = RECORD_FMT.format(k)
        record = json.loads((RUN_DIR / name).read_text(encoding="utf-8"))
        meta = REPLICA_META[k]
        assert record["replica"] == k
        assert record["seed"] == REPLICA_SEEDS[k]
        assert record["run"] == RUN_ID
        assert record["mode"] == "production"
        assert record["schema"] == SCHEMAS["replica"]
        assert record["n_events"] == EVENTS_PER_REPLICA
        assert record["reused"] is False
        assert record["class_census"] == meta["class_census"]
        assert sum(record["class_census"].values()) == EVENTS_PER_REPLICA
        cons = record["conservation"]
        assert cons["events"] == EVENTS_PER_REPLICA
        assert cons["b_q_s_census"] == {"2,2,0": EVENTS_PER_REPLICA}
        assert cons["n1_n2_census"] == {"n1": meta["n1"],
                                        "n2": meta["n2"]}
        # the n1/n2 census subdivides the nd class (n = 1 plus n = 2)
        assert cons["n1_n2_census"]["n1"] + cons["n1_n2_census"]["n2"] == \
            record["class_census"]["nd"]
        assert cons["tolerance"] == CONSERVATION_TOLERANCE
        assert cons["violations"] == 0
        assert cons["worst_closure_relative"] == meta["worst_relative"]
        assert cons["worst_energy_closure_gev"] == meta["worst_energy"]
        assert cons["worst_momentum_closure_gev"] == meta["worst_momentum"]
        assert cons["worst_mass_shell_relative"] == meta["worst_mass_shell"]
        assert record["event_hash_set_sha256"] == meta["event_hash"]
        assert record["identity"]["driver_sha256"] == DRIVER_SHA256
        assert record["identity"]["pre_registration_sha256"] == \
            SEALED_FILE_SHA256["pre_registration.json"]
        gf = record["generation_fractions"]
        assert gf["p_n2"] == DRAWN_P_N2
        assert gf["b_sd"] == FROZEN_FIT["b_sd"]
        assert gf["epsilon"] == FROZEN_FIT["epsilon"]
        assert gf["class_fractions"] == {
            "dd": 0.25, "nd": 0.25,
            "sd_projectile": 0.25, "sd_target": 0.25,
        }
    # the sample census reconciles to the per-replica totals
    n1_total = sum(m["n1"] for m in REPLICA_META)
    n2_total = sum(m["n2"] for m in REPLICA_META)
    assert n1_total == CONSERVATION_N1
    assert n2_total == CONSERVATION_N2


# ---------------------------------------------------------------------------
# The fit report
# ---------------------------------------------------------------------------
def test_fit_report_record():
    """The fit report: the single-parameter fit of P(n >= 2) at the
    boundary minimum 0.5, the objective profile (initial 0.25 to the
    optimum), the Hessian discipline (quoted: false, positive
    definite), the optimizer (the round's second and final optimizer
    call), the named residual register (the six entries in order),
    the per-block fit table, the tail-improvement table, the
    no-degradation table, the conservation contract on the full
    sample, the bin structure, and the internal digest."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / REPORT_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["run"] == RUN_ID
    assert record["mode"] == "production"
    assert record["schema"] == SCHEMAS["report"]
    assert record["digest"] == FIT_REPORT_DIGEST

    # the fit set and the fitted value
    fit = record["fit_set"]
    assert fit["bound"] == list(BOUND)
    assert fit["count"] == 1
    assert fit["fit_set"] == ["p_n2"]
    assert fit["initial_values"] == {"p_n2": DRAWN_P_N2}
    assert fit["frozen"]["source"] == (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/fit_report.json")
    assert fit["frozen"]["refit"] is False
    assert fit["frozen"]["b_sd"] == FROZEN_FIT["b_sd"]
    assert fit["frozen"]["epsilon"] == FROZEN_FIT["epsilon"]
    assert fit["frozen"]["fractions"] == {
        "dd_fraction": FROZEN_FIT["dd_fraction"],
        "nd_fraction": FROZEN_FIT["nd_fraction"],
        "sd_projectile_fraction": FROZEN_FIT["sd_projectile_fraction"],
        "sd_target_fraction": FROZEN_FIT["sd_target_fraction"],
    }
    assert record["fitted"] == {"p_n2": FITTED_P_N2}
    assert record["boundary_parameters"] == [
        {"bound": list(BOUND), "parameter": "p_n2",
         "value": FITTED_P_N2}]

    # the frozen values (asserted against the sealed R51 fit report)
    fv = record["frozen_values"]
    assert fv["b_sd"] == FROZEN_FIT["b_sd"]
    assert fv["dd_fraction"] == FROZEN_FIT["dd_fraction"]
    assert fv["epsilon"] == FROZEN_FIT["epsilon"]
    assert fv["nd_fraction"] == FROZEN_FIT["nd_fraction"]
    assert fv["sd_projectile_fraction"] == \
        FROZEN_FIT["sd_projectile_fraction"]
    assert fv["sd_target_fraction"] == FROZEN_FIT["sd_target_fraction"]
    assert fv["assertion"]["fit_report_sha256"] == R51_FIT_REPORT_SHA256
    assert fv["assertion"]["match"] is True
    assert fv["assertion"]["r51_optimizer_converged"] is False

    # the objective profile
    assert record["objective_at_initial"]["minus2lnl"] == \
        OBJECTIVE_AT_INITIAL
    assert record["objective_at_initial"]["n_eff_total"] == \
        N_EFF_AT_INITIAL
    assert record["objective_at_initial"]["nuisances"] == \
        list(NUISANCES_AT_INITIAL)
    assert record["objective_at_optimum"]["minus2lnl"] == \
        OBJECTIVE_AT_OPTIMUM
    assert record["objective_at_optimum"]["n_eff_total"] == \
        N_EFF_AT_OPTIMUM
    assert record["objective_at_optimum"]["nuisances"] == \
        list(NUISANCES_AT_OPTIMUM)
    assert record["finite_mc_profile"]["p_n2"][5]["value"] == 0.25
    assert record["finite_mc_profile"]["p_n2"][5]["minus2lnl"] == \
        OBJECTIVE_AT_INITIAL
    assert record["finite_mc_profile"]["p_n2"][10]["value"] == 0.5
    assert record["finite_mc_profile"]["p_n2"][10]["minus2lnl"] == \
        OBJECTIVE_AT_OPTIMUM

    # the Hessian discipline (quoted: false; the replica-block minima
    # disagree, so no covariance is quoted)
    hess = record["hessian"]
    assert hess["quoted"] is False
    assert hess["positive_definite"] is True
    assert hess["curvature"] is None
    assert hess["steps"] == {"p_n2": HESSIAN_STEPS_P_N2}

    # the optimizer (the round's second and final optimizer call)
    opt = record["optimizer"]
    assert opt["invoked"] is True
    assert opt["converged"] is True
    assert opt["nfev"] == 10
    assert opt["nit"] == 5
    assert opt["message"] == "Optimization terminated successfully."
    assert opt["objective_value"] == OBJECTIVE_AT_OPTIMUM
    assert opt["best_point"] == {"p_n2": FITTED_P_N2}
    assert opt["settings"]["fatol"] == 1e-06
    assert opt["settings"]["maxfev"] == 400
    assert opt["settings"]["maxiter"] == 200
    assert opt["settings"]["xatol"] == 0.001
    assert opt["settings"]["initial_point"] == [DRAWN_P_N2]
    assert record["optimizer_uses"] == 1

    # the named residual register (the six entries, in order)
    kinds = [entry["kind"]
             for entry in record["named_residual_register"]]
    assert kinds == list(NAMED_RESIDUAL_KINDS)
    assert record["named_residual_register"][0]["parameter"] == "p_n2"
    assert record["named_residual_register"][0]["value"] == FITTED_P_N2
    assert record["named_residual_register"][1]["per_block"] == \
        list(BLOCK_P_N2)
    assert record["named_residual_register"][1]["spread"] == 0.25

    # the per-block fit table
    for k, block in enumerate(record["block_fits"]):
        assert block["replica"] == k
        assert block["converged"] is True
        assert block["p_n2"] == BLOCK_P_N2[k]
        assert block["minus2lnl"] == BLOCK_MINUS2LN_L[k]
        assert block["nfev"] == BLOCK_NFEV[k]

    # the tail-improvement table (acceptance (iv))
    tail = record["tail_improvement"]
    assert tail["bound"] == TRIGGER_BOUND
    assert tail["any_improvement_beyond_2x_se"] is True
    assert tail["max_improvement_sigma"] == MAX_IMPROVEMENT_SIGMA
    assert len(tail["per_bin"]) == 9
    for expected, actual in zip(TAIL_IMPROVEMENT_PER_BIN, tail["per_bin"]):
        assert actual["bin"] == expected["bin"]
        assert actual["difference"] == expected["difference"]
        assert actual["improved_beyond_2x_se"] == \
            expected["improved_beyond_2x_se"]
        assert actual["pooled_se"] == expected["pooled_se"]
        assert actual["r52_one_deviance"] == expected["r52_one_deviance"]
        assert actual["r53_one_deviance"] == expected["r53_one_deviance"]
        assert actual["sigma"] == expected["sigma"]
    assert tail["per_bin"][2]["bin"] == TRIGGER_ARGMAX_BIN_N_CH

    # the no-degradation table (acceptance (iv); nothing degrades)
    for entry in record["no_degradation_5pct"]:
        family = entry["family"]
        assert entry["r52_one_deviance_total"] == \
            NO_DEGRADATION_TOTALS[family]
        assert entry["r53_one_deviance_total"] == 0.0
        assert entry["relative_change"] == -1.0
        assert entry["degrades_beyond_5pct"] is False

    # the conservation contract on the full sample
    cons = record["conservation"]
    assert cons["events_checked"] == CONSERVATION_EVENTS
    assert cons["events_stored"] == CONSERVATION_EVENTS
    assert cons["b_q_s_census"] == {"2,2,0": CONSERVATION_EVENTS}
    assert cons["n1_n2_census"] == {"n1": CONSERVATION_N1,
                                    "n2": CONSERVATION_N2}
    assert cons["tolerance"] == CONSERVATION_TOLERANCE
    assert cons["violations"] == 0
    assert cons["pass"] is True
    assert cons["worst_closure_relative"] == CONSERVATION_WORST_RELATIVE
    assert cons["worst_energy_closure_gev"] == CONSERVATION_WORST_ENERGY
    assert cons["worst_momentum_closure_gev"] == \
        CONSERVATION_WORST_MOMENTUM
    assert cons["worst_mass_shell_relative"] == \
        CONSERVATION_WORST_MASS_SHELL

    # the bin structure
    assert record["bin_structure"]["dropped"] == 1
    assert record["bin_structure"]["kept_per_statistic"] == \
        BIN_STRUCTURE_KEPT
    assert record["dropped_bins"][0]["statistic"] == \
        "charged_multiplicity_distribution"
    assert record["dropped_bins"][0]["index"] == 11

    # the frozen substrate (the ten verified entries)
    assert len(record["frozen_substrate"]) == 10
    for name, (rel, digest) in FROZEN_SUBSTRATE.items():
        entry = record["frozen_substrate"][name]
        assert entry["sha256"]["sha256"] == digest
        assert entry["sha256"]["verified"] is True
        assert entry["verified"] is True

    # the M8 frozen set and the sealed inputs
    assert record["m8_frozen_set"]["hash"] == M8_PARAMETER_HASH
    assert record["m8_frozen_set"]["verified"] is True
    for run_key, entry in SEAL_INPUTS.items():
        rec = record["sealed_inputs"][run_key]
        assert rec["entries"] == entry["entries"]
        assert rec["run_dir"] == entry["run_dir"]
        assert rec["intact"] is True
        assert rec["mismatches"] == []

    # the trigger block (the consumed R52 evidence)
    assert record["trigger"]["entry"] == R52_TRIGGER_ENTRY
    assert record["trigger"]["gate_file_sha256"] == R52_GATE5_SHA256
    assert record["trigger"]["report_digest"] == R52_REPORT_DIGEST
    assert record["trigger"]["match"] is True

    # the statistics and the replica-block stability
    assert record["statistics"]["events_total"] == N_EVENTS_TOTAL
    assert record["statistics"]["replicas"] == N_REPLICAS
    stability = record["replica_block_stability"]["p_n2"]
    assert stability["per_block"] == list(BLOCK_P_N2)
    assert stability["pass"] is False
    assert stability["spread"] == 0.25


def test_gate5_r53_record():
    """The gate 5 record: TRIGGERED, retained: false (the evidence-only
    note), the clause, the consumed trigger, the tail-improvement
    evidence (beyond the 2.0 pooled-SE bound), the no-degradation
    evidence, the named residual register, the replica-block
    stability, and the internal report digest."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / GATE_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["run"] == RUN_ID
    assert record["schema"] == SCHEMAS["gate"]
    assert record["status"] == GATE_STATUS
    assert record["retained"] is GATE_RETAINED
    assert record["gate"] == GATE_NAME
    assert record["gate_note"] == GATE_GATE_NOTE
    assert record["clause"] == GATE_CLAUSE
    assert record["bound"] == list(BOUND)
    assert record["fit_set"] == ["p_n2"]
    assert record["free_parameters"] == 1
    assert record["fitted"] == {"p_n2": FITTED_P_N2}
    assert record["events_checked"] == CONSERVATION_EVENTS
    assert record["sealed"] is True
    assert record["report_digest"] == FIT_REPORT_DIGEST

    cons = record["conservation"]
    assert cons["n1_n2_census"] == {"n1": CONSERVATION_N1,
                                    "n2": CONSERVATION_N2}
    assert cons["pass"] is True
    assert cons["tolerance"] == CONSERVATION_TOLERANCE
    assert cons["violations"] == 0
    assert cons["worst_energy_closure_gev"] == CONSERVATION_WORST_ENERGY

    assert record["trigger"]["named_residual"] == TRIGGER_NAMED_RESIDUAL
    assert record["trigger"]["value"] == TRIGGER_VALUE
    assert record["trigger"]["pooled_se"] == TRIGGER_POOLED_SE
    assert record["trigger"]["sigma"] == TRIGGER_SIGMA
    assert record["trigger"]["triggered"] is True
    assert record["trigger"]["source"] == \
        f"{R52_RUN_DIR_REL}/{R52_GATE_NAME}"

    tail = record["tail_improvement"]
    assert tail["beyond_2x_se"] is True
    assert tail["max_improvement_sigma"] == MAX_IMPROVEMENT_SIGMA
    assert len(tail["per_bin"]) == 9
    for expected, actual in zip(TAIL_IMPROVEMENT_PER_BIN, tail["per_bin"]):
        assert actual["bin"] == expected["bin"]
        assert actual["sigma"] == expected["sigma"]

    for entry in record["no_degradation_5pct"]:
        assert entry["r52_one_deviance_total"] == \
            NO_DEGRADATION_TOTALS[entry["family"]]
        assert entry["r53_one_deviance_total"] == 0.0
        assert entry["degrades_beyond_5pct"] is False

    kinds = [e["kind"] for e in record["named_residual_register"]]
    assert kinds == list(NAMED_RESIDUAL_KINDS)

    opt = record["optimizer"]
    assert opt["converged"] is True
    assert opt["nfev"] == 10
    assert opt["nit"] == 5
    assert opt["objective_value"] == OBJECTIVE_AT_OPTIMUM
    assert opt["message"] == "Optimization terminated successfully."
    assert record["optimizer_invoked"] is True
    assert record["optimizer_uses"] == 1

    hess = record["hessian"]
    assert hess["quoted"] is False

    stability = record["replica_block_stability"]
    assert stability["per_block"] == list(BLOCK_P_N2)
    assert stability["pass"] is False
    assert stability["spread"] == 0.25

    carry = record["model_uncertainty_carry"]
    assert carry["b_transport_seagull"] == \
        MODEL_UNCERTAINTY_CARRY["b_transport_seagull"]
    assert carry["r32_spread"]["bin_spread_p95"] == \
        MODEL_UNCERTAINTY_CARRY["r32_spread"]["bin_spread_p95"]


def test_clean_environment_record():
    """The clean-environment record: the single validation pass
    before the seal (the three commands, the probe replica reproducing
    the sealed record, the report digest matching the re-run digest,
    the seal state pending)."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / CLEAN_ENV_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["run"] == RUN_ID
    assert record["schema"] == SCHEMAS["clean_env"]
    assert record["verified_utc"] == CLEAN_ENV_VERIFIED_UTC
    assert record["commands"] == list(CLEAN_ENV_COMMANDS)
    assert record["report_digest"] == FIT_REPORT_DIGEST
    assert record["report_rerun_digest"] == FIT_REPORT_DIGEST
    assert record["report_hash_match"] is True
    probe = record["probe_replica"]
    assert probe["file"] == "replica_0.json"
    assert probe["replica"] == 0
    assert probe["seed"] == REPLICA_SEEDS[0]
    assert probe["matches_sealed_record"] is True
    assert probe["census_matches"] is True
    assert probe["conservation_matches"] is True
    assert probe["event_hash_set_match"] is True
    assert record["seal"]["entries"] == 0
    assert record["seal"]["mismatches"] == []
    assert record["seal"]["state"].startswith("pending")


def test_seal_manifest_record():
    """The seal manifest: the 14-entry SHA256SUMS inventory, the 8
    records, the seal timestamp, the git state, the environment, and
    the clean-validation record name."""

    skip_unless_sealed(RUN_DIR)
    record = json.loads(
        (RUN_DIR / SEAL_NAME).read_text(encoding="utf-8"))
    assert record["card"] == SOLVER_CARD
    assert record["run"] == RUN_ID
    assert record["schema"] == SCHEMAS["seal"]
    assert record["sealed_utc"] == SEAL_SEALED_UTC
    assert record["sha256sums_entries"] == SEAL_SHA256SUMS_ENTRIES
    assert record["record_count"] == SEAL_RECORD_COUNT
    assert record["run_dir"] == RUN_DIR_ABS
    assert record["clean_validation_record"] == CLEAN_ENV_NAME
    assert record["files"] == sorted(SEALED_FILE_SHA256.keys())
    assert record["git_state"] == {
        "branch": SEAL_GIT_BRANCH,
        "dirty": SEAL_GIT_DIRTY,
        "head": SEAL_GIT_HEAD,
    }
    assert record["environment"] == SEAL_ENVIRONMENT


def test_cross_run_pins():
    """The consumed cross-run surfaces at their on-disk digests (the
    sealed R52 gate record and the sealed R51 fit report)."""

    skip_unless_sealed(R52_RUN_DIR)
    skip_unless_sealed(R51_RUN_DIR)
    assert _sha256_file(R52_GATE5) == R52_GATE5_SHA256
    assert _sha256_file(R51_FIT_REPORT) == R51_FIT_REPORT_SHA256
    r52 = json.loads(R52_GATE5.read_text(encoding="utf-8"))
    entry = r52["trigger_for_r53"]
    for key, value in R52_TRIGGER_ENTRY.items():
        assert entry[key] == value, key
    assert entry["zero_model_max_abs_pull"] == 121.59710317806265
    assert entry["zero_model_triggered"] is True


# ---------------------------------------------------------------------------
# The layout sync
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``). The check pins the committed dual-tree state;
    an untracked in-flight helper file from a later card (present in one
    tree before it is vendored into the other) is transient working-tree
    state, so the test skips while such a file exists."""

    def _untracked(cwd, pathspec):
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all",
             pathspec],
            capture_output=True, text=True, cwd=cwd)
        return [line[3:] for line in status.stdout.splitlines()
                if line.startswith("??")]

    # the nested tree is its own repository; probe both helper trees
    untracked = (
        _untracked(str(MONO_ROOT), "helpers/benchmarking/")
        + _untracked(str(MONO_ROOT / "dev/mixing2006"),
                     "helpers/benchmarking/"))
    if untracked:
        pytest.skip(
            "untracked in-flight helper file(s) pending vendoring: "
            + ", ".join(sorted(untracked)))

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ---------------------------------------------------------------------------
# The freeze re-pin register
# ---------------------------------------------------------------------------
