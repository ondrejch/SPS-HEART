"""Round 5, card 12 test-author register: the R52 central-excitation
necessity run contract.

The card-12 test-author register: the R52 central-excitation necessity
run (models with zero and one neutral central mesonic excitation, the
Gate 5 clause "necessity of the neutral central excitation") is pinned
at the card's landed state.  The register verifies, against the sealed
run directory and the committed dual-tree driver:

* the dual-tree driver byte identity (nested/outer byte-identical at the
  pinned sha256),
* the driver's identity constants (run id, seed map, statistics,
  frozen point, frozen substrate, M8 frozen set, card chain, Round 5
  reserved blocks and partitions, the previous-used block record),
* the card-12 solver test's post re-pin digest,
* the seed map and its disjointness from every previously used block,
* the pre-registration on disk (acceptance, conservation contract,
  frozen substrate digest pins, one-side archive digest pin, zero-model
  draw parameters, retention rule, R53 trigger pre-registration),
* the eight zero-side replica records on disk, with an independent
  re-computation of the per-event hash, the class census, and the
  worst-case closure over all 800,000 stored events,
* the one-side reuse (bit-identity against the sealed R51 archive, the
  frozen point byte-identical to the sealed R51 fit report) and the
  weighted-evaluation identity (the R51 importance ensemble carried
  verbatim),
* the five-family held-out comparison (the four published central
  spectra and the multiplicity tails: per-family degradation,
  preservation, kept rows, verdicts),
* the R53 trigger record,
* the conservation pins (zero B, Q, S violations, per-replica census,
  worst closure residuals, the R51 cross-reference),
* the named notes and the Gate 5 record (named residual register,
  retention verdicts, optimizer absence, sealed state),
* the seal manifest and the 14-entry checksum inventory (re-hashed),
* the clean-environment validation record and the provenance record,
* the freeze re-pin register: the strict path-sorted inventory at 541
  entries after this card's test-author growth (+1: this register),
  the 42 chain count-pin files at the 541 literal with the 540 literal
  retained in the naming comments, and the solver test at its post
  re-pin digest.

No Monte Carlo, no optimizer: every physics value is read from the
sealed artifacts and the committed code; the only recomputation is the
pure-hash verification over the stored zero-side events.
"""

import hashlib
import json
from pathlib import Path
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import deleted_events, sealed_mismatches, skip_unless_sealed

TEST_FILE = (
    "test_staged_r52_central_excitation_necessity_contract_pins.py"
)
NESTED_ROOT = Path(__file__).resolve().parents[1]
MONO_ROOT = NESTED_ROOT.parent.parent

NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking" /
    "staged_r52_central_excitation_necessity.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking" /
    "staged_r52_central_excitation_necessity.py"
)
MANIFEST_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
)
RUN_DIR = (
    MONO_ROOT / "dev" / "00runs" /
    "r52_central_excitation_necessity_2026-09-27"
)
R51_RUN_DIR = (
    MONO_ROOT / "dev" / "00runs" /
    "r51_topology_fraction_fit_2026-09-27"
)
R51_FIT_REPORT = R51_RUN_DIR / "fit_report.json"

SOLVER_CARD = "TASK-20260925-r5-run-r52-central-excitation-necessity"
MY_CARD = "TASK-20260925-r5-run-r52-central-excitation-necessity-test-author"

SOLVER_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r52_central_excitation_necessity.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r52_central_excitation_necessity_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r52_central_excitation_necessity.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r52_central_excitation_necessity.py"
)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DRIVER_SHA256 = (
    "889131347693b3e3338ececeb63ac883993b1ed04f83fde5449471f898469fa7"
)
SOLVER_TEST_SHA256 = (
    "a42d0a2f2c7fc864c2b1c823d6b029c92471916a51f1754559afc18a71677a42"
)

# ---------------------------------------------------------------------------
# Sealed payload digests (the 14-entry SHA256SUMS.txt inventory).
# ---------------------------------------------------------------------------

PRE_REGISTRATION_SHA256 = (
    "61efba85014717175df2885faea19c2c3a332388b6af7e8192d27ec291039ed4"
)
REPORT_FILE_SHA256 = (
    "2bd4877340f6f3cb4cef5d4977d0b4a60ee19ec14886b372a080e71df30b7c33"
)
GATE5_FILE_SHA256 = (
    "a4f0cb7281dd8713e7945b9a7cda531ab329b5990a7e85a6e17c10b366e29374"
)
PROVENANCE_FILE_SHA256 = (
    "053a4da31c2ae7dda9ba5d38bc15035d3f00977221cb036ccb2f652c931e8558"
)
REPORT_MD_FILE_SHA256 = (
    "65d4cec97c0baccf0fa3c33318bef23bb185582ee0c8defa3159e261acbbb3f2"
)
CLEAN_ENV_FILE_SHA256 = (
    "486eb270a62cadd628924e585c313b307d8c6699c1a6a8ef2af5be065c22e19d"
)
REPLICA_FILE_SHA256 = (
    "bafd7cadece916533a91913b157fbde914e9093e85489d92e013dedef5a592bb",
    "497b2ed451cdb054fcbaf6f725674e970ee2bfdd5a4e767a593c2b972589b55a",
    "990a4e1b81cf7a294aeda8cfc30ffcee35e88f7db5b118a02187637436e76f6b",
    "631a3178a6f163765684776653c56679e47e6a9ed7921d237505ae379e8c6a85",
    "2835ed9a3ecf9f9c4590bd8eed5e9347993fe0cc6aca2a8a6820534905929c1e",
    "ca49dba90526d4fe263707e39d1d286d0074cc8191e45318ae8ffccbe680844d",
    "35d49db76ac35562355c7e0f57221151adbb11cf8e21c6d29a5e31a66c35fb51",
    "344291a8688f61db131aab67c5f253621e088930c576d5239036fa0b5ca0a07f",
)

SEALED_FILES = (
    "REPORT.md",
    "clean_environment_commands.json",
    "gate5_r52.json",
    "held_out_report.json",
    "pre_registration.json",
    "provenance.json",
) + tuple(f"replica_{i}.json" for i in range(8))

# ---------------------------------------------------------------------------
# Driver identity constants.
# ---------------------------------------------------------------------------

RUN_ID = "R52"
CARD = SOLVER_CARD
RUN_NAME = "r52_central_excitation_necessity_2026-09-27"

N_REPLICAS = 8
EVENTS_PER_REPLICA = 100_000
N_EVENTS_TOTAL = 800_000

MASTER_SEED = 20260925
SEED_OFFSET = 40_000
REPLICA_SEEDS = tuple(range(20300953, 20300961))
EVENT_INDEX_BASE = 1_200_000_000
EVENT_INDEX_RULE = "1_200_000_000 + i, i < 100,000 (per replica)"
SMOKE_SEED = 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000
SMOKE_EVENTS = 2_000
SMOKE_EVENT_INDEX_RULE = "1_600_000_000 + i, i < 2,000"
SMOKE_RAW_SEED_RULE = (
    "raw smoke seed = 20260925 + 40000 + 12000 = 20312925 (the r"
)
RAW_SEED_RULE = (
    "raw zero-model replica seed = 20260925 + 40000 + 28 + k = "
    "20300953 + k per replica k < 8 (index item 9; the block "
    "20300953-20300960 is disjoint from every previously used raw "
    "block and from every other reserved Round 5 block)"
)
SUBSTREAM_LABELS = (
    "decay_channel",
    "decay_kinematics",
    "exchange_kinematics",
    "nuclear_geometry",
    "spectral_state",
    "topology_selection",
    "transport",
)
CONSUMED_LABELS = (
    "topology_selection",
    "spectral_state",
    "exchange_kinematics",
    "decay_channel",
    "decay_kinematics",
)
# The pre-registration records the label order before the driver sorts
# the substreams for the generation seed map.
PRE_REG_SUBSTREAM_LABELS = (
    "topology_selection",
    "exchange_kinematics",
    "spectral_state",
    "decay_channel",
    "decay_kinematics",
    "nuclear_geometry",
    "transport",
)

MULTIPLICITY_BINS = tuple(range(2, 15))
CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd")
CENTRAL_WINDOW = 0.5
EVENT_CLOSURE_TOL = 1e-09
RETENTION_SE_BOUND = 2.0
TAIL_QUANTILE = 0.9

PUBLISHED_SPECIES = ("pi_plus", "pi_minus", "k_plus", "k_minus")
PI0_SPECIES = "pi0"
PI0_ROW_MASS_GEV = 0.1349766
PI0_GEOMETRY_SPECIES = "pi_minus"
SPECIES_PDGS = {
    "pi_plus": 211,
    "pi_minus": -211,
    "pi0": 111,
    "k_plus": 321,
    "k_minus": -321,
}
SPECIES_MASSES = {
    "pi_plus": 0.1395699,
    "pi_minus": 0.1395699,
    "pi0": 0.1349766,
    "k_plus": 0.493677,
    "k_minus": 0.493677,
}
PROTON_MASS = 0.93827231
BEAM_MOMENTUM_GEV_C = 158.0
BLEND_RAMP_START_GEV = 2.680194110727528
SQRT_S_EXTRACTION_GEV = 17.270
CM_BETA_PINNED = 0.9940791999921171
CM_GAMMA_PINNED = 9.203303892226666

M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)
M8_BLEND_RAMP_START_GEV = 2.680194110727528
M8_EFFECTIVE_TRANSFER_SLOPE_GEV2 = 8.504140762083612

R45_FROZEN_CASCADE = {
    "kappa_B": 0.0,
    "kappa_M": 4.0,
    "b_dec_gev2": 2.0,
    "baryonic_nominal_width_gev": 0.3,
    "W_flavor": 1.0,
}

MODEL_UNCERTAINTY_CARRY = {
    "r32_spread": {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    },
    "b_transport_seagull": -2.3959765824349295e-05,
}

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
)

FROZEN_SUBSTRATE = {
    "builder_blend_path": (
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py",
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    ),
    "card2_engine": (
        "dev/mixing2006/na49_mixing/threshold_metric.py",
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    ),
    "card6_routing": (
        "dev/mixing2006/na49_mixing/spectral_density.py",
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563",
    ),
    "catalog": (
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd",
    ),
    "continuum_config": (
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1",
    ),
    "r45_driver": (
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py",
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    ),
    "r46_driver": (
        "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py",
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    ),
    "r50_driver": (
        "helpers/benchmarking/staged_r50_pure_topology_templates.py",
        "b74e5408195ca6a3a53caf495da0c3f1206a9b43a73f336ba24b5cd6eb936fa3",
    ),
    "r51_driver": (
        "helpers/benchmarking/staged_r51_topology_fraction_fit.py",
        "73bd74ab72270ae47b900b4da171b88db606234b1cbd4345c8135f48911f40a1",
    ),
}

ROUND5_MODULES = {
    "card1_elastic": (
        "dev/mixing2006/na49_mixing/topology_elastic.py",
        "7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824",
    ),
    "card2_ledger": (
        "dev/mixing2006/na49_mixing/exchange_ledger.py",
        "3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a",
    ),
    "card3_remnants": (
        "dev/mixing2006/na49_mixing/topology_remnants.py",
        "8eb45877490c04896c7cc7062e174dd6c060d96b1bace07688c16086e8a463dc",
    ),
    "card4_mass_laws": (
        "dev/mixing2006/na49_mixing/topology_mass_laws.py",
        "c15e3416b022390bcc59dd3f0e17f4264b272beffbd9f3863a093a88a5fc0b48",
    ),
    "card5_single_diffraction": (
        "dev/mixing2006/na49_mixing/topology_single_diffraction.py",
        "5868b178fb3fe5852d5baf141b5cb949554747a1dcbd3496d3301251683dd2cb",
    ),
    "card6_double_diffraction": (
        "dev/mixing2006/na49_mixing/topology_double_diffraction.py",
        "f17a83324deef6be7f415c24ec7742820d99291407705546df8703e34c0f077f",
    ),
    "card7_non_diffractive": (
        "dev/mixing2006/na49_mixing/topology_non_diffractive.py",
        "68162f96614022e5f95807155cbd34f6714cb14bf8e4a5eaf116764f31bfd4b2",
    ),
    "card8_multi_neutral": (
        "dev/mixing2006/na49_mixing/topology_multi_neutral.py",
        "9527643b387ceb29ac22c8854f45f28c22c7a14215b35c55fafc499993bb8454",
    ),
    "card9_runner": (
        "dev/mixing2006/na49_mixing/topology_runner.py",
        "0394c010d581416ca4f59677401632ab53e7acc8fa54fe147ce4f568370e0dbb",
    ),
}

RESERVED_ROUND5_BLOCKS = {
    "r50_used": (20300925, 20300944),
    "r51_used": (20300945, 20300952),
    "r53": (20300961, 20300968),
    "r54": (20300969, 20300988),
    "r55": (20300989, 20301008),
}
RESERVED_ROUND5_PARTITIONS = {
    "r50_used": (1_000_000_000, 1_000_025_000),
    "r51_used": (1_100_000_000, 1_100_100_000),
    "r53": (1_300_000_000, 1_300_100_000),
    "r54": (1_400_000_000, 1_400_050_000),
    "r55": (1_500_000_000, 1_500_250_000),
    "round5_smoke": (1_600_000_000, 1_600_002_000),
}
PREVIOUS_USED_BLOCKS = {
    "r23": {"raw_seeds": (20300913, 20300920)},
    "r32": {"raw_seeds": (20301519, 20301826)},
    "round4_production": {"raw_seeds": (20301020, 20301083)},
    "round4_smoke": {"raw_seeds": (20307920, 20311920)},
    "r46_smoke": {"raw_seeds": (20312920, 20312920)},
    "r50_partition": {
        "raw_seeds": (20300925, 20300944),
        "event_index_range": (1_000_000_000, 1_000_025_000),
    },
    "r51_partition": {
        "raw_seeds": (20300945, 20300952),
        "event_index_range": (1_100_000_000, 1_100_100_000),
    },
}
PROBE_REPLICA = 0

FROZEN_FIT = {
    "sd_projectile_fraction": 3.619686246956965e-09,
    "sd_target_fraction": 2.921678336937106e-07,
    "dd_fraction": 0.9999997010778905,
    "nd_fraction": 3.1345895878942542e-09,
    "epsilon": 0.0033443027832747944,
    "b_sd": 1.8457918026393214,
}
RENORMALIZED_FRACTIONS = {
    "sd_projectile": 3.619686258303196e-09,
    "sd_target": 2.9216783460953685e-07,
    "dd": 0.9999997042124792,
    "nd": 0.0,
}

# ---------------------------------------------------------------------------
# Sealed state.
# ---------------------------------------------------------------------------

REPORT_DIGEST = (
    "37ffc0bcdeaf5728d09032d34ff43ffb359c8aafce9257d8aed0e241a9971460"
)
SEAL_SEALED_UTC = "2026-09-27T22:03:49Z"
PRE_REG_RECORDED_UTC = "2026-09-27T20:22:57Z"
PROVENANCE_RECORDED_UTC = "2026-09-27T20:22:57Z"
CLEAN_ENV_VERIFIED_UTC = "2026-09-27T22:03:26Z"
GIT_HEAD = "d7a037be89179fd757f52b5e4bf465b026c94075"
ENVIRONMENT = {
    "machine": "x86_64",
    "numpy": "1.26.4",
    "python": "3.12.3",
    "platform": "Linux-6.8.0-139-generic-x86_64-with-glibc2.39",
}

# ---------------------------------------------------------------------------
# R51 archive pins (the one-side event source).
# ---------------------------------------------------------------------------

R51_FIT_REPORT_SHA256 = (
    "30ed195237ca83bf75ba24278bbb5ce67bfe1c55ad1da4081ace6fc241f9aede"
)
R51_FIT_REPORT_DIGEST = (
    "73b20d01921531429cc6004035f8fb442a4bbbd704a4f201f3a708976e76e03e"
)
R51_REPLICA_SEEDS = tuple(range(20300945, 20300953))
R51_REPLICA_FILE_SHA256 = (
    "7f97facb838c0ca4c27d23d861c1481f5eba7bae3d0ac5bcf8bb9a953bbd7da6",
    "c7b1329ccdf6d2cda4ec2c9c38072f2a3273cc678535f7ba71b39604f57c3a71",
    "d23a25c9f01e94c708290a69b3416dba4374174b302696580d0fba9a40f210dd",
    "f64c45b66bff2b7880fa51921b3bdf487e461968eb451c6b461ea0b0031f5bce",
    "33edb3ecc5e8ffdd8d43b2db6f0217a8ae29c039306e7aa265c89e2d0a428c12",
    "14566eaf0dbee1de594bfc05d97dd6b140d80ec2129db226f0ac66958054f713",
    "118e73cc0c4bf478b0a799a59b49852a22dae47ec4dcde9a123fd44f55dc9e43",
    "9920fa9b667552a9efb23ccdab3f6a8b4cf0ccdd7cbf642c977786e3ca0f433d",
)
R51_REPLICA_EVENT_HASH_SET = (
    "4755718b085ae571ed7a5c5c651041ac2795753108ad1addd1d28cac1d183103",
    "caab6bcff6c469481c4a8b3e0f8802afb88273776496a4aca9dc9b6e9c8b0b96",
    "623d5e105bbe80878f6002eec2601c799d57ac5dfdff291449674a0f348a4374",
    "ee911ebe5476a34bf713ae87396ea3cdc8c9794a6c988d541a0e14b35a044230",
    "a05900b8c6e1babe76e9bde37081a36300fa8b0ecd8042e5ae5546557b3821da",
    "36ec2b4dd5c560bb6c68eabc400bf11162e151160ad962f8c8d059d39291eec5",
    "a2d4face8670392b6b2984eb00f5afcd3f1ef87b55d50eac3868a91be1b524ea",
    "cd99f5d20a90e732120825ccb47f2dc5ff2afa1bdfc55c4edca5e415e3fc259f",
)
R51_REPLICA_CENSUS = (
    {"dd": 25090, "nd": 25051, "sd_projectile": 25037, "sd_target": 24822},
    {"dd": 25221, "nd": 25048, "sd_projectile": 24889, "sd_target": 24842},
    {"dd": 25103, "nd": 24970, "sd_projectile": 24880, "sd_target": 25047},
    {"dd": 25102, "nd": 25033, "sd_projectile": 25026, "sd_target": 24839},
    {"dd": 25240, "nd": 25006, "sd_projectile": 24995, "sd_target": 24759},
    {"dd": 24955, "nd": 25129, "sd_projectile": 24969, "sd_target": 24947},
    {"dd": 24944, "nd": 25121, "sd_projectile": 24815, "sd_target": 25120},
    {"dd": 25152, "nd": 25068, "sd_projectile": 24783, "sd_target": 24997},
)

# ---------------------------------------------------------------------------
# Zero-side replica pins (the sealed records).
# ---------------------------------------------------------------------------

REPLICA_CENSUS = (
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 100000},
    {"dd": 99999, "sd_target": 1},
)
REPLICA_CONSERVATION = (
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 2.896870688622454e-14,
        "worst_energy_closure_gev": 4.604316927725449e-12,
        "worst_mass_shell_relative": 4.893205172654661e-10,
        "worst_momentum_closure_gev": 4.575895218295045e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 2.968398359946465e-14,
        "worst_energy_closure_gev": 4.718003765447065e-12,
        "worst_mass_shell_relative": 5.891625008335629e-10,
        "worst_momentum_closure_gev": 4.689582056016661e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 2.932113061491046e-14,
        "worst_energy_closure_gev": 4.604316927725449e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 4.632738637155853e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 3.183950993152854e-14,
        "worst_energy_closure_gev": 5.059064278611913e-12,
        "worst_mass_shell_relative": 5.430534098281643e-10,
        "worst_momentum_closure_gev": 5.030642569181509e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 3.05803202732195e-14,
        "worst_energy_closure_gev": 4.803268893738277e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 4.831690603168681e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 3.165962569462725e-14,
        "worst_energy_closure_gev": 5.002220859751105e-12,
        "worst_mass_shell_relative": 5.891625008335629e-10,
        "worst_momentum_closure_gev": 5.002220859751105e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 2.7882056719700134e-14,
        "worst_energy_closure_gev": 4.405364961712621e-12,
        "worst_mass_shell_relative": 5.774879222955706e-10,
        "worst_momentum_closure_gev": 4.405364961712621e-12,
    },
    {
        "b_q_s_census": {"2,2,0": 100000},
        "events": 100000,
        "tolerance": 1e-09,
        "violations": 0,
        "worst_closure_relative": 2.716251977209497e-14,
        "worst_energy_closure_gev": 4.291678123991005e-12,
        "worst_mass_shell_relative": 5.891625008335629e-10,
        "worst_momentum_closure_gev": 4.291678123991005e-12,
    },
)
REPLICA_EVENT_HASH_SET = (
    "fddb460e727effeb40e2e7405cd29d98c5871863135f295cd750ae1fed28a766",
    "92b8534f3507a4264889104c3e10592c5aee26af9a83fa6dd5c150288deadc38",
    "360452a07b66f936b8a6ca8471d840cbea5b55bc1b5ad4b45e3494c1c3dc883b",
    "7dcf3d12b5aca149dd3eb331f66a7335afc580f86ad1455bad7d49f029544fcd",
    "3955b6ee5ae35a3bef1714f5e85a0d6b3a58bb5a969b91e00ca946e726325942",
    "d4e49b4549595f770de2b5cc7724d4c9c52b07d9f893c10ac30d72dc3e1f8182",
    "f28a7313a968b1dd7bea78bf412ca44a5e5996fe9b7c590b46a15535a3969552",
    "4a752f36b0d52e76e92d43b955839ca931fc2904f2c9359330cee741a6608b70",
)
GENERATION_FRACTIONS_PIN = {
    "b_sd": 1.8457918026393214,
    "epsilon": 0.0033443027832747944,
    "p_n2": 0.0,
    "renormalized_fractions": {
        "dd": 0.9999997042124792,
        "nd": 0.0,
        "sd_projectile": 3.619686258303196e-09,
        "sd_target": 2.9216783460953685e-07,
    },
}
IDENTITY_BLOCK_PIN = {
    "driver_sha256": DRIVER_SHA256,
    "pre_registration_sha256": PRE_REGISTRATION_SHA256,
}

# ---------------------------------------------------------------------------
# Report pins (the sealed held_out_report.json).
# ---------------------------------------------------------------------------

ONE_WEIGHTS = {
    "ess_min": 0.00024177313336515557,
    "ess_per_block": [
        0.0011025340542270036,
        0.001297728617848435,
        0.0006023309591162278,
        0.0020419698488048244,
        0.0035078203350823014,
        0.00024177313336515557,
        0.0042599824514324555,
        0.003200009554048086,
    ],
    "log_shift": 9.791795160176466,
    "n_eff_per_block": [
        4.842832036665259,
        4.569974281142578,
        5.852706874827246,
        4.667935919968844,
        4.1307546774118,
        5.002153707834921,
        4.104950067620404,
        4.169562997881534,
    ],
    "n_eff_total": 37.34087056335259,
}
ONE_REUSE_WORST_CLOSURE = (
    3.2379162642232416e-14,
    3.0940088747022085e-14,
    3.111997298392338e-14,
    3.1479741457725955e-14,
    2.968398359946465e-14,
    3.6156731617159525e-14,
    2.986078332561434e-14,
    2.86015936673053e-14,
)

FAMILY_RETENTION = {
    "central_spectrum_pi_plus": {
        "central_rows": 58,
        "kept": 58,
        "max_degradation_pooled_se": 181.2514736160099,
        "one_dropped_out_of_window": 321072,
        "one_model_max_abs_pull": 53.37990301101123,
        "one_model_preserves": False,
        "one_n_eff_total": 37.34087056335259,
        "one_records": 744493,
        "rows": 230,
        "verdict": "not retained",
        "zero_degrades_at_2x_pooled_se": True,
        "zero_dropped_out_of_window": 594170,
        "zero_n_eff_total": 800000.0,
        "zero_records": 1104025,
    },
    "central_spectrum_pi_minus": {
        "central_rows": 58,
        "kept": 58,
        "max_degradation_pooled_se": 158.6314525679463,
        "one_dropped_out_of_window": 201433,
        "one_model_max_abs_pull": 72.07061987564673,
        "one_model_preserves": False,
        "one_n_eff_total": 37.34087056335259,
        "one_records": 492455,
        "rows": 229,
        "verdict": "not retained",
        "zero_degrades_at_2x_pooled_se": True,
        "zero_dropped_out_of_window": 406989,
        "zero_n_eff_total": 800000.0,
        "zero_records": 762395,
    },
    "central_spectrum_k_plus": {
        "central_rows": 42,
        "kept": 42,
        "max_degradation_pooled_se": 13.303313432249068,
        "one_dropped_out_of_window": 62142,
        "one_model_max_abs_pull": 19.74882453385634,
        "one_model_preserves": False,
        "one_n_eff_total": 37.34087056335259,
        "one_records": 138619,
        "rows": 153,
        "verdict": "not retained",
        "zero_degrades_at_2x_pooled_se": True,
        "zero_dropped_out_of_window": 153037,
        "zero_n_eff_total": 800000.0,
        "zero_records": 259627,
    },
    "central_spectrum_k_minus": {
        "central_rows": 42,
        "kept": 42,
        "max_degradation_pooled_se": 17.15754509043264,
        "one_dropped_out_of_window": 18130,
        "one_model_max_abs_pull": 40.06684826722442,
        "one_model_preserves": False,
        "one_n_eff_total": 37.34087056335259,
        "one_records": 41466,
        "rows": 148,
        "verdict": "not retained",
        "zero_degrades_at_2x_pooled_se": True,
        "zero_dropped_out_of_window": 43011,
        "zero_n_eff_total": 800000.0,
        "zero_records": 75236,
    },
    "multiplicity_tails": {
        "bins": [6, 7, 8, 9, 10, 11, 12, 13, 14],
        "kept": 8,
        "max_degradation_pooled_se": 89.424683841745,
        "one_model_max_abs_pull": 11.007284372792265,
        "one_model_preserves": False,
        "threshold_n_ch": 6,
        "verdict": "not retained",
        "zero_degrades_at_2x_pooled_se": True,
    },
}
TAIL_CUT = {
    "cumulative_at_threshold": 0.9666349999999999,
    "quantile": 0.9,
    "tail_bins": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "threshold_n_ch": 6,
}
MULTIPLICITY_TAILS_DROPPED_BIN = (
    13,
    "structurally zero in every block (model: one)",
)
CENTRAL_ROW_COUNTS = {
    "k_minus": 42,
    "k_plus": 42,
    "pi0": 58,
    "pi_minus": 58,
    "pi_plus": 58,
}
PUBLISHED_ROW_COUNTS = {
    "k_minus": 148,
    "k_plus": 153,
    "pi_minus": 229,
    "pi_plus": 230,
}
R01_FILE_DIGESTS = {
    "bin_predictions.json": (
        "820eb78d8a23aecc16a0a74c8d6b54fb105ab27e0919b638aec608c98d0d97c7"
    ),
    "covariance.json": (
        "f7c3b454da6e179b992d1640273c9ce44f1ca3c7a7463bc39acf24f4a2aa953c"
    ),
    "residuals.json": (
        "eaefef310c361d2427655e942b07a3c48d538dcb1e90b36cc64f906dd685564e"
    ),
}
SEAL_R51_ARCHIVE = {
    "entries": 14,
    "intact": True,
    "mismatches": [],
    "run_dir": "dev/00runs/r51_topology_fraction_fit_2026-09-27",
}

TRIGGER_FOR_R53 = {
    "argmax_bin_n_ch": 8,
    "booleans_agree": True,
    "named_residual": (
        "multiplicity_tail_residual_one_model_beyond_2x_pooled_se"
    ),
    "pooled_se": 0.002738966420355409,
    "sigma": 11.007284372792265,
    "triggered": True,
    "value": 0.030148582276380863,
    "zero_model_max_abs_pull": 121.59710317806265,
    "zero_model_triggered": True,
}

CONSERVATION_ZERO = {
    "b_q_s_census": {"2,2,0": 800000},
    "events_checked": 800000,
    "events_stored": 800000,
    "pass": True,
    "tolerance": 1e-09,
    "violations": 0,
    "worst_closure_relative": 3.183950993152854e-14,
    "worst_energy_closure_gev": 5.059064278611913e-12,
    "worst_mass_shell_relative": 6.089413019835981e-10,
    "worst_momentum_closure_gev": 5.030642569181509e-12,
}
CONSERVATION_ONE_XREF = {
    "pass": True,
    "source": (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/gate5_r51.json"
    ),
    "violations": 0,
    "worst_energy_closure_gev": 5.7411853049416095e-12,
}
GATE5_CONSERVATION = {
    "events_checked": 800000,
    "pass": True,
    "r51_cross_reference": (
        "dev/00runs/r51_topology_fraction_fit_2026-09-27/gate5_r51.json"
    ),
    "tolerance": 1e-09,
    "violations": 0,
    "worst_energy_closure_gev": 5.059064278611913e-12,
}

NAMED_RESIDUAL_KINDS = {
    "heldout_bin_degradation_beyond_2x_pooled_se": 157,
    "weight_degeneracy": 1,
    "r51_inherited_non_convergence": 1,
    "pi0_published_table_absent": 1,
    "multiplicity_tail_training_overlap": 1,
    "multiplicity_tail_residual_one_model_beyond_2x_pooled_se": 1,
}
GATE5_CLAUSE = "necessity of the neutral central excitation"
GATE5_RETENTION = {
    "central_spectrum_k_minus": "not retained",
    "central_spectrum_k_plus": "not retained",
    "central_spectrum_pi_minus": "not retained",
    "central_spectrum_pi_plus": "not retained",
    "multiplicity_tails": "not retained",
}

SEALED_INPUTS = {
    "r01": {
        "entries": 28,
        "intact": True,
        "mismatches": [],
        "run_dir": "dev/00runs/r01_legacy_pp_reference_2026-09-11",
    },
    "r50": {
        "entries": 26,
        "intact": True,
        "mismatches": [],
        "run_dir": "dev/00runs/r50_pure_topology_templates_2026-09-27",
    },
    "r51": {
        "entries": 14,
        "intact": True,
        "mismatches": [],
        "run_dir": "dev/00runs/r51_topology_fraction_fit_2026-09-27",
    },
}

# ---------------------------------------------------------------------------
# The freeze re-pin register.
# ---------------------------------------------------------------------------

# The card-12 test-author growth: +1 entry (this register).  The 42 chain
# count-pin files carry the 541 literal; the 540 literal is retained in
# their naming comments (the Round 4 card-20 growth precedent).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
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
MY_CARD_EXCEPTION_FILES = sorted(
    [MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES)
)


def _load_json(rel, root=None):
    p = (root if root is not None else RUN_DIR) / rel
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Driver byte identity and identity constants.
# ---------------------------------------------------------------------------


def test_dual_tree_driver_byte_identity():
    assert NESTED_DRIVER.is_file()
    assert OUTER_DRIVER.is_file()
    nested_bytes = NESTED_DRIVER.read_bytes()
    outer_bytes = OUTER_DRIVER.read_bytes()
    assert nested_bytes == outer_bytes
    assert _sha256_file(NESTED_DRIVER) == DRIVER_SHA256
    assert _sha256_file(OUTER_DRIVER) == DRIVER_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == DRIVER_SHA256
    assert inventory[DRIVER_REL_OUTER] == DRIVER_SHA256


def test_driver_identity_constants():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r52_driver_check", NESTED_DRIVER
    )
    assert spec is not None and spec.loader is not None
    r52 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r52)

    assert r52.RUN_ID == RUN_ID
    assert r52.CARD == CARD
    assert r52.RUN_NAME == RUN_NAME
    assert r52.DRIVER_REL_NESTED == DRIVER_REL_NESTED
    assert r52.DRIVER_REL_OUTER == DRIVER_REL_OUTER
    assert r52.DRIVER_SHA256 == DRIVER_SHA256
    assert r52.N_REPLICAS == N_REPLICAS
    assert r52.EVENTS_PER_REPLICA == EVENTS_PER_REPLICA
    assert r52.N_EVENTS_TOTAL == N_EVENTS_TOTAL
    assert r52.MASTER_SEED == MASTER_SEED
    assert r52.SEED_OFFSET == SEED_OFFSET
    assert tuple(r52.REPLICA_SEEDS) == REPLICA_SEEDS
    assert r52.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert r52.EVENT_INDEX_RULE == EVENT_INDEX_RULE
    assert r52.SMOKE_SEED == SMOKE_SEED
    assert r52.SMOKE_EVENT_INDEX_BASE == SMOKE_EVENT_INDEX_BASE
    assert r52.SMOKE_EVENTS == SMOKE_EVENTS
    assert r52.SMOKE_EVENT_INDEX_RULE == SMOKE_EVENT_INDEX_RULE
    assert tuple(r52.MULTIPLICITY_BINS) == MULTIPLICITY_BINS
    assert tuple(r52.CLASS_ORDER) == CLASS_ORDER
    assert r52.CENTRAL_WINDOW == CENTRAL_WINDOW
    assert r52.EVENT_CLOSURE_TOL == EVENT_CLOSURE_TOL
    assert r52.RETENTION_SE_BOUND == RETENTION_SE_BOUND
    assert r52.TAIL_QUANTILE == TAIL_QUANTILE
    assert tuple(r52.PUBLISHED_SPECIES) == PUBLISHED_SPECIES
    assert r52.PI0_SPECIES == PI0_SPECIES
    assert r52.PI0_ROW_MASS_GEV == PI0_ROW_MASS_GEV
    assert r52.PI0_GEOMETRY_SPECIES == PI0_GEOMETRY_SPECIES
    assert dict(r52.SPECIES_PDGS) == SPECIES_PDGS
    assert dict(r52.SPECIES_MASSES) == SPECIES_MASSES
    assert r52.PROTON_MASS == PROTON_MASS
    assert r52.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert r52.BLEND_RAMP_START_GEV == BLEND_RAMP_START_GEV
    assert r52.SQRT_S_EXTRACTION_GEV == SQRT_S_EXTRACTION_GEV
    assert r52.CM_BETA_PINNED == CM_BETA_PINNED
    assert r52.CM_GAMMA_PINNED == CM_GAMMA_PINNED
    assert r52.M8_PARAMETER_HASH == M8_PARAMETER_HASH
    assert dict(r52.R45_FROZEN_CASCADE) == R45_FROZEN_CASCADE
    assert r52.MODEL_UNCERTAINTY_CARRY["r32_spread"][
        "bin_spread_p95"
    ] == MODEL_UNCERTAINTY_CARRY["r32_spread"]["bin_spread_p95"]
    assert r52.MODEL_UNCERTAINTY_CARRY["r32_spread"][
        "persisting_residuals"
    ] == MODEL_UNCERTAINTY_CARRY["r32_spread"]["persisting_residuals"]
    assert r52.MODEL_UNCERTAINTY_CARRY[
        "b_transport_seagull"
    ] == MODEL_UNCERTAINTY_CARRY["b_transport_seagull"]
    assert tuple(r52.CARD_CHAIN) == CARD_CHAIN
    for name, (path, sha) in FROZEN_SUBSTRATE.items():
        assert r52.FROZEN_SUBSTRATE[name] == (path, sha)
        assert _sha256_file(MONO_ROOT / path) == sha, name
    for name, (path, sha) in ROUND5_MODULES.items():
        assert r52.ROUND5_MODULES[name] == path, name
        assert _sha256_file(MONO_ROOT / path) == sha, name
    assert dict(r52.RESERVED_ROUND5_BLOCKS) == {
        k: list(v) for k, v in RESERVED_ROUND5_BLOCKS.items()
    }
    assert dict(r52.RESERVED_ROUND5_PARTITIONS) == {
        k: list(v) for k, v in RESERVED_ROUND5_PARTITIONS.items()
    }
    for key, pin in PREVIOUS_USED_BLOCKS.items():
        record = r52.PREVIOUS_USED_BLOCKS[key]
        assert record["raw_seeds"] == list(pin["raw_seeds"]), key
        if "event_index_range" in pin:
            assert record["event_index_range"] == list(
                pin["event_index_range"]
            ), key
    assert r52.PROBE_REPLICA == PROBE_REPLICA
    assert dict(r52.FROZEN_FIT) == FROZEN_FIT
    assert dict(r52.RENORMALIZED_FRACTIONS) == RENORMALIZED_FRACTIONS


def test_solver_test_file_digest():
    solver_path = NESTED_ROOT / "tests" / "test_staged_r52_central_excitation_necessity.py"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_tree_sha256"][SOLVER_TEST_REL] == (
        SOLVER_TEST_SHA256
    )


# ---------------------------------------------------------------------------
# Seed map and partition.
# ---------------------------------------------------------------------------


def test_seed_map_partition():
    skip_unless_sealed(RUN_DIR)
    pre = _load_json("pre_registration.json")
    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert tuple(seed_map["raw_replica_seeds"]) == REPLICA_SEEDS
    assert seed_map["event_index_base"] == EVENT_INDEX_BASE
    assert seed_map["event_index_rule"] == EVENT_INDEX_RULE
    assert seed_map["raw_seed_rule"] == RAW_SEED_RULE
    assert tuple(seed_map["substream_labels"]) == PRE_REG_SUBSTREAM_LABELS
    smoke = seed_map["smoke"]
    assert smoke["raw_seed"] == SMOKE_SEED
    assert smoke["events"] == SMOKE_EVENTS
    assert smoke["event_index_rule"] == SMOKE_EVENT_INDEX_RULE
    d = seed_map["disjointness"]
    assert d["disjoint"] is True
    assert d["r52_raw_seed_block"] == [REPLICA_SEEDS[0], REPLICA_SEEDS[-1]]
    assert d["r52_replica_seeds"] == list(REPLICA_SEEDS)
    # The per-replica partition: one replica spans i < 100,000.
    assert d["r52_event_index_range"] == [
        EVENT_INDEX_BASE,
        EVENT_INDEX_BASE + EVENTS_PER_REPLICA,
    ]
    assert d["r52_event_index_partition"] == EVENT_INDEX_RULE
    assert d["shared_raw_seeds"] == {}
    for key, (lo, hi) in RESERVED_ROUND5_BLOCKS.items():
        assert d["reserved_round5_blocks"][key] == [lo, hi], key
    for key, (lo, hi) in RESERVED_ROUND5_PARTITIONS.items():
        assert d["reserved_round5_partitions"][key] == [lo, hi], key
    for key, pin in PREVIOUS_USED_BLOCKS.items():
        assert (
            d["previously_used_blocks"][key]["raw_seeds"]
            == list(pin["raw_seeds"])
        ), key
    assert "20300945-20300952" in seed_map["one_side_reuse"]
    assert (
        "1_100_000_000 + i" in seed_map["one_side_reuse"]
    )


# ---------------------------------------------------------------------------
# The pre-registration on disk.
# ---------------------------------------------------------------------------


def test_pre_registration_on_disk():
    skip_unless_sealed(RUN_DIR)
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )
    pre = _load_json("pre_registration.json")
    assert pre["schema"] == (
        "na49_mixing.staged_r52_central_excitation_necessity.pre_registration/1"
    )
    assert pre["card"] == CARD
    assert pre["run"] == RUN_ID
    assert pre["run_name"] == RUN_NAME
    assert pre["mode"] == "production"
    assert pre["recorded_utc"] == PRE_REG_RECORDED_UTC
    assert pre["written_before_generation"] is True
    assert tuple(pre["card_chain"]) == CARD_CHAIN
    stats = pre["statistics"]
    assert stats["replicas"] == N_REPLICAS
    assert stats["events_per_replica"] == EVENTS_PER_REPLICA
    assert stats["events_total"] == N_EVENTS_TOTAL
    cons = pre["conservation"]
    assert cons["expected_b_q_s"] == [2, 2, 0]
    assert cons["tolerance"] == EVENT_CLOSURE_TOL
    assert dict(pre["frozen_cascade"]) == R45_FROZEN_CASCADE
    for name, (path, sha) in FROZEN_SUBSTRATE.items():
        entry = pre["frozen_substrate"][name]
        assert entry["expected"] == sha
        assert entry["sha256"] == sha
        assert _sha256_file(MONO_ROOT / path) == sha, name
    m8 = pre["m8_frozen_set"]
    assert m8["hash"] == M8_PARAMETER_HASH
    assert m8["blend_ramp_start_gev"] == M8_BLEND_RAMP_START_GEV
    assert m8["effective_transfer_slope_gev2"] == (
        M8_EFFECTIVE_TRANSFER_SLOPE_GEV2
    )
    assert m8["verified"] is True
    carry = pre["model_uncertainty_carry"]
    assert carry["b_transport_seagull"] == (
        MODEL_UNCERTAINTY_CARRY["b_transport_seagull"]
    )
    assert carry["r32_spread"]["bin_spread_p95"] == (
        MODEL_UNCERTAINTY_CARRY["r32_spread"]["bin_spread_p95"]
    )
    om = pre["one_model"]
    assert om["events"] == N_EVENTS_TOTAL
    assert om["regenerated"] is False
    assert om["source"] == "sealed R51 events"
    assert dict(om["frozen_values"]) == FROZEN_FIT
    pin = om["archive_digest_pin"]
    assert pin["fit_report_sha256"] == R51_FIT_REPORT_SHA256
    assert pin["fit_report_digest_field"] == R51_FIT_REPORT_DIGEST
    assert pin["fit_report_name"] == "fit_report.json"
    assert pin["run_dir"] == "dev/00runs/r51_topology_fraction_fit_2026-09-27"
    assert pin["seal"] == SEAL_R51_ARCHIVE
    for i, rec in enumerate(pin["replicas"]):
        assert rec["replica"] == i
        assert rec["seed"] == R51_REPLICA_SEEDS[i]
        assert rec["n_events"] == EVENTS_PER_REPLICA
        assert rec["file_sha256"] == R51_REPLICA_FILE_SHA256[i]
        assert rec["event_hash_set_sha256"] == (
            R51_REPLICA_EVENT_HASH_SET[i]
        )
        assert rec["class_census"] == R51_REPLICA_CENSUS[i]
    assertion = om["assertion_against_sealed_r51"]
    assert assertion["match"] is True
    assert assertion["r51_optimizer_converged"] is False
    assert assertion["r51_nd_fraction"] == FROZEN_FIT["nd_fraction"]
    assert len(assertion["r51_named_residual_kinds"]) == 8
    zm = pre["zero_model"]
    assert zm["nd_fraction"] == 0.0
    assert zm["p_n2"] == 0.0
    assert zm["events"] == N_EVENTS_TOTAL
    assert zm["regenerated"] is True
    assert dict(zm["renormalized_fractions"]) == RENORMALIZED_FRACTIONS
    assert zm["b_sd"] == FROZEN_FIT["b_sd"]
    assert zm["epsilon"] == FROZEN_FIT["epsilon"]
    rule = pre["retention_rule"]
    assert rule["bound"] == RETENTION_SE_BOUND
    assert rule["recorded_per_family"] is True
    trig = pre["r53_trigger"]
    assert trig["bound"] == RETENTION_SE_BOUND
    assert trig["recorded_in"] == "gate5_r52.json"
    ho = pre["held_out"]
    assert tuple(ho["central_spectra_abs_y_lt_0.5"]) == PUBLISHED_SPECIES
    assert dict(ho["central_row_counts"]) == CENTRAL_ROW_COUNTS
    cut = ho["multiplicity_tail_cut"]
    assert cut["quantile"] == TAIL_QUANTILE
    assert cut["threshold_n_ch"] == TAIL_CUT["threshold_n_ch"]
    assert cut["tail_bins"] == TAIL_CUT["tail_bins"]
    assert cut["cumulative_at_threshold"] == (
        TAIL_CUT["cumulative_at_threshold"]
    )
    r01 = ho["r01_data"]
    assert dict(r01["published_rows"]) == PUBLISHED_ROW_COUNTS
    assert tuple(r01["published_species"]) == PUBLISHED_SPECIES
    assert dict(r01["central_rows_abs_y_lt_0.5"]) == CENTRAL_ROW_COUNTS
    assert dict(r01["file_digests"]) == R01_FILE_DIGESTS
    for name, pin in SEALED_INPUTS.items():
        entry = pre["sealed_inputs"][name]
        assert entry["entries"] == pin["entries"], name
        assert entry["intact"] is True, name
        assert entry["mismatches"] == [], name
        assert entry["run_dir"] == pin["run_dir"], name


# ---------------------------------------------------------------------------
# The zero-side replica records on disk (with the independent hash
# re-computation over all stored events).
# ---------------------------------------------------------------------------


def test_replica_records_on_disk():
    skip_unless_sealed(RUN_DIR)
    for i in range(N_REPLICAS):
        path = RUN_DIR / f"replica_{i}.json"
        assert _sha256_file(path) == REPLICA_FILE_SHA256[i]
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["schema"] == (
            "na49_mixing.staged_r52_central_excitation_necessity.replica/1"
        )
        assert rec["run"] == RUN_ID
        assert rec["card"] == CARD
        assert rec["mode"] == "production"
        assert rec["replica"] == i
        assert rec["seed"] == REPLICA_SEEDS[i]
        assert rec["n_events"] == EVENTS_PER_REPLICA
        assert rec["reused"] is False
        assert rec["identity"] == IDENTITY_BLOCK_PIN
        sm = rec["seed_map"]
        assert sm["master_seed"] == MASTER_SEED
        assert sm["event_index_rule"] == EVENT_INDEX_RULE
        assert sm["raw_seed_rule"] == RAW_SEED_RULE
        assert tuple(sm["substream_labels"]) == SUBSTREAM_LABELS
        assert tuple(sm["consumed_labels"]) == CONSUMED_LABELS
        assert dict(rec["generation_fractions"]) == (
            {
                "note": rec["generation_fractions"]["note"],
                **GENERATION_FRACTIONS_PIN,
            }
        )
        assert rec["class_census"] == REPLICA_CENSUS[i]
        assert rec["conservation"] == REPLICA_CONSERVATION[i]
        assert rec["event_hash_set_sha256"] == REPLICA_EVENT_HASH_SET[i]
        # Independent re-computation over the stored events.
        events = rec["events"]
        assert len(events) == EVENTS_PER_REPLICA
        census = {}
        hashes = []
        worst_closure = 0.0
        worst_shell = 0.0
        for ev in events:
            body = {k: v for k, v in ev.items() if k != "h"}
            canonical = json.dumps(
                body, sort_keys=True, separators=(",", ":")
            )
            h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            assert h == ev["h"], (i, ev["i"])
            hashes.append(h)
            census[ev["cl"]] = census.get(ev["cl"], 0) + 1
            # The (B, Q, S) conservation invariant on every event.
            assert ev["bqs"] == [2, 2, 0], (i, ev["i"])
            # The relative four-momentum closure components: every
            # component at or under the pre-registered tolerance.
            for component in ev["clo"]:
                assert component <= EVENT_CLOSURE_TOL, (i, ev["i"])
            closure = max(ev["clo"])
            if closure > worst_closure:
                worst_closure = closure
            if ev["sh"] > worst_shell:
                worst_shell = ev["sh"]
        assert census == REPLICA_CENSUS[i], i
        joined = "\n".join(hashes)
        assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == (
            rec["event_hash_set_sha256"]
        ), i
        cons = rec["conservation"]
        assert cons["worst_closure_relative"] == worst_closure, i
        assert cons["worst_mass_shell_relative"] == worst_shell, i
        assert cons["events"] == EVENTS_PER_REPLICA
        assert cons["violations"] == 0


def test_zero_sample_census_algebra():
    total = 0
    nd = 0
    sd_target = 0
    skip_unless_sealed(RUN_DIR)
    for i in range(N_REPLICAS):
        rec = json.loads(
            (RUN_DIR / f"replica_{i}.json").read_text(encoding="utf-8")
        )
        census = rec["class_census"]
        total += sum(census.values())
        nd += census.get("nd", 0)
        sd_target += census.get("sd_target", 0)
    assert total == N_EVENTS_TOTAL
    assert nd == 0, "the zero model has no ND events"
    assert sd_target == 1, (
        "exactly one target-side single-diffraction event over the "
        "full zero sample (the frozen renormalized fractions draw it "
        "in replica 7)"
    )


# ---------------------------------------------------------------------------
# The one-side reuse and the frozen point.
# ---------------------------------------------------------------------------


def test_one_side_reuse_and_frozen_point():
    # The sealed R51 archive verifies at its pinned digests.
    skip_unless_sealed(R51_RUN_DIR)
    assert _sha256_file(R51_FIT_REPORT) == R51_FIT_REPORT_SHA256
    fit = json.loads(R51_FIT_REPORT.read_text(encoding="utf-8"))
    assert fit["digest"] == R51_FIT_REPORT_DIGEST
    for i in range(N_REPLICAS):
        path = R51_RUN_DIR / f"replica_{i}.json"
        assert _sha256_file(path) == R51_REPLICA_FILE_SHA256[i]
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["event_hash_set_sha256"] == (
            R51_REPLICA_EVENT_HASH_SET[i]
        ), i
    # The report's reuse record: bit identity per replica.
    rep = _load_json("held_out_report.json")
    reuse = rep["one_model"]["reuse"]
    assert reuse["all_bit_identical"] is True
    assert len(reuse["per_replica"]) == N_REPLICAS
    for i, rec in enumerate(reuse["per_replica"]):
        assert rec["replica"] == i
        assert rec["seed"] == R51_REPLICA_SEEDS[i]
        assert rec["events"] == EVENTS_PER_REPLICA
        assert rec["bit_identical"] is True
        assert rec["event_hash_set_match"] is True
        assert rec["per_event_hash_mismatches"] == 0
        assert rec["worst_closure_relative"] == (
            ONE_REUSE_WORST_CLOSURE[i]
        )
    pin = reuse["r51_archive_pin"]
    assert pin["fit_report_sha256"] == R51_FIT_REPORT_SHA256
    assert tuple(pin["replica_file_sha256"]) == R51_REPLICA_FILE_SHA256
    # The frozen point is byte-identical across the driver, the report,
    # and the sealed R51 fit report.
    fv = rep["frozen_values"]
    for key, value in FROZEN_FIT.items():
        assert fv[key] == value, key
    assert fv["assertion"]["match"] is True
    assert fv["assertion"]["fit_report_sha256"] == R51_FIT_REPORT_SHA256
    best = fit["optimizer"]["best_point"]
    # The R51 fit carries the five free parameters; the ND fraction is
    # pinned on the R51 side and does not appear in the best point.
    assert set(best.keys()) == set(FROZEN_FIT.keys()) - {"nd_fraction"}
    for key in best:
        assert best[key] == FROZEN_FIT[key], key
    assert fit["optimizer"]["converged"] is False
    assert rep["one_model"]["refit"] is False
    assert rep["one_model"]["source"] == "sealed R51 events"
    assert rep["one_model"]["events"] == N_EVENTS_TOTAL
    # The clean-environment one-side probe.
    ce = _load_json("clean_environment_commands.json")
    probe = ce["one_side_probe_replica"]
    assert probe["replica"] == PROBE_REPLICA
    assert probe["bit_identical"] is True
    assert probe["event_hash_set_match"] is True
    assert probe["per_event_hash_mismatches"] == 0


def test_weighted_evaluation_identity():
    skip_unless_sealed(RUN_DIR)
    rep = _load_json("held_out_report.json")
    weights = rep["one_model"]["weights"]
    for key, value in ONE_WEIGHTS.items():
        assert weights[key] == value, key
    # Cross-verified against the sealed R51 fit report objective.
    fit = json.loads(R51_FIT_REPORT.read_text(encoding="utf-8"))
    opt = fit["objective_at_optimum"]
    assert opt["n_eff_total"] == ONE_WEIGHTS["n_eff_total"]
    assert opt["ess"]["min"] == ONE_WEIGHTS["ess_min"]
    assert opt["ess"]["per_block"] == ONE_WEIGHTS["ess_per_block"]
    assert opt["log_shift"] == ONE_WEIGHTS["log_shift"]
    # The frozen-point multiplicity vector reproduces the sealed R51
    # model vector bit for bit.
    mc = rep["one_model"]["multiplicity_crosscheck"]
    assert mc["model_fit_matches_sealed_report"] is True
    assert mc["mc_se_matches_sealed_report"] is True
    # Every family carries the same one-side effective sample size.
    for family, pin in FAMILY_RETENTION.items():
        if "one_n_eff_total" in pin:
            assert (
                rep["retention"][family]["one_n_eff_total"]
                == ONE_WEIGHTS["n_eff_total"]
            ), family
        if "zero_n_eff_total" in pin:
            assert (
                rep["retention"][family]["zero_n_eff_total"] == 800000.0
            ), family


# ---------------------------------------------------------------------------
# The five-family held-out comparison.
# ---------------------------------------------------------------------------


def test_held_out_five_family_comparison():
    skip_unless_sealed(RUN_DIR)
    rep = _load_json("held_out_report.json")
    ret = rep["retention"]
    assert ret["bound"] == RETENTION_SE_BOUND
    assert ret["all_families_retained"] is False
    assert set(ret.keys()) == {
        "algebra",
        "all_families_retained",
        "all_families_retained_note",
        "bound",
        *FAMILY_RETENTION.keys(),
    }
    for family, pin in FAMILY_RETENTION.items():
        entry = ret[family]
        for key, value in pin.items():
            assert entry[key] == value, (family, key)
        assert entry["verdict"] == "not retained", family
    # The multiplicity tails: the dropped bin is recorded.
    per_bin = rep["held_out"]["per_bin"]["multiplicity_tails"]
    assert len(per_bin) == 9
    dropped = [
        b for b in per_bin if not b.get("kept", True)
    ]
    assert len(dropped) == 1
    assert dropped[0]["bin"] == MULTIPLICITY_TAILS_DROPPED_BIN[0]
    assert dropped[0]["drop_reason"] == (
        MULTIPLICITY_TAILS_DROPPED_BIN[1]
    )
    kept = [b for b in per_bin if b.get("kept", True)]
    assert [b["bin"] for b in kept] == [
        n for n in TAIL_CUT["tail_bins"]
        if n != MULTIPLICITY_TAILS_DROPPED_BIN[0]
    ]
    # The pre-registered tail cut.
    cut = rep["held_out"]["tail_cut"]
    for key, value in TAIL_CUT.items():
        assert cut[key] == value, key


def test_r53_trigger_record():
    skip_unless_sealed(RUN_DIR)
    rep = _load_json("held_out_report.json")
    trig = rep["trigger_for_r53"]
    for key, value in TRIGGER_FOR_R53.items():
        assert trig[key] == value, key
    gate = _load_json("gate5_r52.json")
    gate_trig = gate["trigger_for_r53"]
    for key, value in TRIGGER_FOR_R53.items():
        if key != "booleans_agree":
            assert gate_trig[key] == value, key
    # The named residual is present in the register with the trigger
    # fields.
    named = [
        e for e in rep["named_residual_register"]
        if e["kind"] == TRIGGER_FOR_R53["named_residual"]
    ]
    assert len(named) == 1
    entry = named[0]
    assert entry["value"] == TRIGGER_FOR_R53["value"]
    assert entry["pooled_se"] == TRIGGER_FOR_R53["pooled_se"]
    assert entry["sigma"] == TRIGGER_FOR_R53["sigma"]
    assert entry["argmax_bin_n_ch"] == TRIGGER_FOR_R53["argmax_bin_n_ch"]
    assert (
        TRIGGER_FOR_R53["triggered"]
        == (TRIGGER_FOR_R53["sigma"] >= RETENTION_SE_BOUND)
    )


# ---------------------------------------------------------------------------
# Conservation.
# ---------------------------------------------------------------------------


def test_conservation_pins():
    skip_unless_sealed(RUN_DIR)
    rep = _load_json("held_out_report.json")
    cons = rep["conservation"]
    assert cons["zero_model"] == CONSERVATION_ZERO
    xref = cons["one_model_cross_reference"]
    for key, value in CONSERVATION_ONE_XREF.items():
        assert xref[key] == value, key
    gate = _load_json("gate5_r52.json")
    assert gate["conservation"] == GATE5_CONSERVATION
    # The zero model record: no ND, no elastic.
    zm = rep["zero_model"]
    assert zm["nd_fraction"] == 0.0
    assert zm["p_n2"] == 0.0
    assert zm["regenerated"] is True
    assert zm["events"] == N_EVENTS_TOTAL
    assert dict(zm["renormalized_fractions"]) == RENORMALIZED_FRACTIONS
    assert dict(zm["conservation"]) == CONSERVATION_ZERO
    # The per-replica census: b, q, s = 2, 2, 0 on every event.
    for i in range(N_REPLICAS):
        rec = json.loads(
            (RUN_DIR / f"replica_{i}.json").read_text(encoding="utf-8")
        )
        census = rec["conservation"]["b_q_s_census"]
        assert list(census.keys()) == ["2,2,0"], i
        assert list(census.values())[0] == EVENTS_PER_REPLICA, i


def test_named_notes_and_gate_record():
    skip_unless_sealed(RUN_DIR)
    gate = _load_json("gate5_r52.json")
    assert gate["schema"] == (
        "na49_mixing.staged_r52_central_excitation_necessity.gate/1"
    )
    assert gate["card"] == CARD
    assert gate["run"] == RUN_ID
    assert gate["gate"] == "gate 5 (plan section 11, line 938)"
    assert gate["clause"] == GATE5_CLAUSE
    assert gate["report_digest"] == REPORT_DIGEST
    assert gate["sealed"] is True
    assert gate["optimizer_invoked"] is False
    assert gate["optimizer_uses"] == 0
    assert gate["one_model"] == {
        "events": N_EVENTS_TOTAL,
        "frozen_source": (
            "dev/00runs/r51_topology_fraction_fit_2026-09-27/"
            "fit_report.json"
        ),
        "refit": False,
        "source": "sealed R51 events",
        "weights": {
            "ess_min": ONE_WEIGHTS["ess_min"],
            "n_eff_total": ONE_WEIGHTS["n_eff_total"],
        },
    }
    assert gate["zero_model"] == {
        "events": N_EVENTS_TOTAL,
        "nd_fraction": 0.0,
        "regenerated": True,
        "renormalized_fractions": RENORMALIZED_FRACTIONS,
    }
    assert gate["retention"] == GATE5_RETENTION
    # The named residual register: the 157 bin degradations plus the
    # five named notes.
    nrr = gate["named_residual_register"]
    assert len(nrr) == sum(NAMED_RESIDUAL_KINDS.values())
    kinds = {}
    for entry in nrr:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    assert kinds == NAMED_RESIDUAL_KINDS
    rep = _load_json("held_out_report.json")
    rep_nrr = rep["named_residual_register"]
    assert len(rep_nrr) == len(nrr)
    for kind in ("pi0_published_table_absent",
                 "multiplicity_tail_training_overlap"):
        assert any(e["kind"] == kind for e in nrr), kind
        assert any(e["kind"] == kind for e in rep_nrr), kind
    # The optimizer absence in the report.
    assert rep["optimizer_invoked"] is False
    assert rep["optimizer_uses"] == 0


# ---------------------------------------------------------------------------
# The seal and the checksum inventory.
# ---------------------------------------------------------------------------


def test_seal_manifest_and_checksums():
    skip_unless_sealed(RUN_DIR)
    seal = _load_json("seal_manifest.json")
    assert seal["schema"] == (
        "na49_mixing.staged_r52_central_excitation_necessity.seal/1"
    )
    assert seal["card"] == CARD
    assert seal["run"] == RUN_ID
    assert seal["sealed_utc"] == SEAL_SEALED_UTC
    assert seal["record_count"] == N_REPLICAS
    assert seal["sha256sums_entries"] == len(SEALED_FILES)
    assert tuple(seal["files"]) == SEALED_FILES
    assert seal["git_state"] == {
        "branch": "main",
        "dirty": True,
        "head": GIT_HEAD,
    }
    assert seal["environment"] == ENVIRONMENT
    assert seal["clean_validation_record"] == (
        "clean_environment_commands.json"
    )
    # The 14-entry checksum inventory re-hashes.
    sums = (RUN_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8")
    lines = [ln for ln in sums.splitlines() if ln.strip()]
    assert len(lines) == len(SEALED_FILES)
    inventory = {}
    for ln in lines:
        sha, rel = ln.split("  ", 1)
        inventory[rel] = sha
    assert set(inventory.keys()) == set(SEALED_FILES)
    assert sealed_mismatches(RUN_DIR) == []
    assert inventory["pre_registration.json"] == PRE_REGISTRATION_SHA256
    assert inventory["held_out_report.json"] == REPORT_FILE_SHA256
    assert inventory["gate5_r52.json"] == GATE5_FILE_SHA256
    assert inventory["provenance.json"] == PROVENANCE_FILE_SHA256
    assert inventory["REPORT.md"] == REPORT_MD_FILE_SHA256
    assert inventory["clean_environment_commands.json"] == (
        CLEAN_ENV_FILE_SHA256
    )
    # The run directory is exactly the sealed set plus the seal
    # artifacts.
    absent = {n for n in deleted_events(RUN_DIR) if not (RUN_DIR / n).is_file()}
    expected = [n for n in [*SEALED_FILES, "SHA256SUMS.txt", "seal_manifest.json"]
                if n not in absent]
    if (RUN_DIR / "EVENTS_DELETED.txt").is_file():
        expected.append("EVENTS_DELETED.txt")
    on_disk = sorted(p.name for p in RUN_DIR.iterdir() if p.is_file())
    assert on_disk == sorted(expected)


# ---------------------------------------------------------------------------
# Clean environment and provenance.
# ---------------------------------------------------------------------------


def test_clean_environment_and_provenance_records():
    skip_unless_sealed(RUN_DIR)
    assert _sha256_file(RUN_DIR / "clean_environment_commands.json") == (
        CLEAN_ENV_FILE_SHA256
    )
    ce = _load_json("clean_environment_commands.json")
    assert ce["schema"] == (
        "na49_mixing.staged_r52_central_excitation_necessity.clean_env/1"
    )
    assert ce["card"] == CARD
    assert ce["run"] == RUN_ID
    assert ce["verified_utc"] == CLEAN_ENV_VERIFIED_UTC
    assert ce["report_digest"] == REPORT_DIGEST
    assert ce["report_rerun_digest"] == REPORT_DIGEST
    assert ce["report_hash_match"] is True
    assert ce["seal"]["entries"] == 0
    assert ce["seal"]["state"].startswith("pending")
    zero_probe = ce["zero_probe_replica"]
    assert zero_probe["replica"] == PROBE_REPLICA
    assert zero_probe["seed"] == REPLICA_SEEDS[PROBE_REPLICA]
    assert zero_probe["matches_sealed_record"] is True
    assert zero_probe["census_matches"] is True
    assert zero_probe["conservation_matches"] is True
    assert zero_probe["event_hash_set_match"] is True
    assert len(ce["commands"]) == 3
    assert (
        "tests/test_staged_r52_central_excitation_necessity.py"
        in ce["commands"][0]
    )
    assert "test_legacy_v1_freeze.py" in ce["commands"][1]
    assert "--clean-validate" in ce["commands"][2]

    assert _sha256_file(RUN_DIR / "provenance.json") == (
        PROVENANCE_FILE_SHA256
    )
    prov = _load_json("provenance.json")
    assert prov["schema"] == (
        "na49_mixing.staged_r52_central_excitation_necessity.provenance/1"
    )
    assert prov["card"] == CARD
    assert prov["run"] == RUN_ID
    assert prov["recorded_utc"] == PROVENANCE_RECORDED_UTC
    assert tuple(prov["card_chain"]) == CARD_CHAIN
    drv = prov["driver"]
    assert drv["byte_identical"] is True
    assert drv["nested"]["path"] == DRIVER_REL_NESTED
    assert drv["nested"]["sha256"] == DRIVER_SHA256
    assert drv["outer"]["path"] == DRIVER_REL_OUTER
    assert drv["outer"]["sha256"] == DRIVER_SHA256
    assert prov["environment"] == ENVIRONMENT
    assert prov["git_state"] == {
        "branch": "main",
        "dirty": True,
        "head": GIT_HEAD,
    }
    for name, (path, sha) in FROZEN_SUBSTRATE.items():
        block = prov["frozen_substrate"][name]
        assert block["sha256"]["sha256"] == sha, name
        assert block["sha256"]["verified"] is True, name
        assert block["verified"] is True, name
    for name, (path, sha) in ROUND5_MODULES.items():
        assert prov["round5_modules"][name] == sha, name
        assert _sha256_file(MONO_ROOT / path) == sha, name
    r51drv = prov["r51_driver"]
    assert r51drv["byte_identical"] is True
    assert (
        r51drv["nested"]["sha256"] == r51drv["outer"]["sha256"]
        == FROZEN_SUBSTRATE["r51_driver"][1]
    )
    pins = prov["r45_r46_r50_r51_driver_pins"]
    assert pins["r45"] == FROZEN_SUBSTRATE["r45_driver"][1]
    assert pins["r46"] == FROZEN_SUBSTRATE["r46_driver"][1]
    assert pins["r50"] == FROZEN_SUBSTRATE["r50_driver"][1]
    assert pins["r51"] == FROZEN_SUBSTRATE["r51_driver"][1]
    apin = prov["r51_archive_pin"]
    assert apin["fit_report_sha256"] == R51_FIT_REPORT_SHA256
    assert apin["fit_report_digest_field"] == R51_FIT_REPORT_DIGEST
    assert apin["seal"] == SEAL_R51_ARCHIVE
    m8 = prov["m8_frozen_set"]
    assert m8["hash"] == M8_PARAMETER_HASH
    assert m8["verified"] is True
    r01 = prov["r01_data"]
    assert dict(r01["published_rows"]) == PUBLISHED_ROW_COUNTS
    assert dict(r01["central_rows_abs_y_lt_0.5"]) == CENTRAL_ROW_COUNTS
    assert dict(r01["file_digests"]) == R01_FILE_DIGESTS
    for name, pin in SEALED_INPUTS.items():
        entry = prov["sealed_inputs"][name]
        assert entry["entries"] == pin["entries"], name
        assert entry["intact"] is True, name
        assert entry["mismatches"] == [], name


# ---------------------------------------------------------------------------
# The freeze re-pin register.
# ---------------------------------------------------------------------------


