"""Round 5, card 11 test-author register: the R51 topology fraction
fit contract pins.

Card ``TASK-20260925-r5-run-r51-topology-fraction-fit-test-author``:
the test-author half of the R51 run card
(``TASK-20260925-r5-run-r51-topology-fraction-fit``, plan section 11
run R51, lines 909-918).  This register independently pins the dual-tree
run driver (``helpers/benchmarking/staged_r51_topology_fraction_fit.py``,
byte-identical in the nested tree and the monorepo mirror), the hermetic
solver test (``tests/test_staged_r51_topology_fraction_fit.py``), the
sealed run ``dev/00runs/r51_topology_fraction_fit_2026-09-27/`` (sealed
2026-09-27T16:28:19Z; 14 sealed payload entries, 0 mismatches), and the
freeze re-pin (the strict path-sorted inventory at 537 entries after this
card's re-pin; 536 at the card-11 solver landing).  The driver is loaded
once via importlib (its top level is the pre-registration identity and
the import-time substrate checks; no events are generated), and the
sealed JSON is read read-only from disk.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist, are
  byte-identical, and sit at the committed digest; the freeze manifest
  inventory pins both at that digest.
- ``test_driver_identity_constants``: the run identity, the file-name
  constants, the class order, the fit set and bounds, the initial and
  nominal parameters, the frozen R45 cascade, the seed-map constants,
  the M8-frozen parameter pins, the model-uncertainty carry, the
  optimizer settings, the closure tolerance, and the bin counts.
- ``test_solver_test_file_digest``: the hermetic solver test at its
  post-re-pin committed digest, pinned in the freeze manifest inventory.
- ``test_seed_map_partition``: the master seed, the raw replica block
  20300945-20300952, the event-index partition 1_100_000_000 + i, the
  smoke block 20312925 at 1_600_000_000 + i, the per-replica seeds, the
  reserved round-5 blocks and partitions, and the driver's disjointness
  report against the pre-registration record.
- ``test_pre_registration_on_disk``: the sealed pre-registration's
  schema, identity, statistics, seed map, initial values, conservation
  contract, M8 frozen set, frozen cascade, frozen substrate digests,
  model-uncertainty carry, optimizer record, fit set, and card chain.
- ``test_replica_records_on_disk``: the 8 replica records (identity,
  class census, seed, conservation block) pinned verbatim, the per-
  replica event hash sets, and the independent recompute of the event
  hash set, class census, and conservation worst-closures from the
  stored events over all 800,000 events.
- ``test_fit_report_on_disk``: the sealed fit report's schema and self-
  digest, the global conservation block, the fitted point, the
  optimizer record (the pre-registered non-convergence), the objective
  at the initial point and the optimum, the Hessian discipline, the bin
  structure, the dropped bin, and the analytic checks.
- ``test_gate5_record_on_disk``: the gate record's identity, clause,
  report digest, conservation summary (zero violations over 800,000
  events), excluded observables, optimizer record, fitted point, the
  named-residual register, and the replica-block stability table.
- ``test_seal_manifest_and_checksums``: the seal manifest's 14 sealed
  files (8 replica records plus the six top-level files), the checksum
  file's 14 entries, the 16-file run dir, and the full re-hash of all
  14 sealed files against the checksum digests.
- ``test_clean_environment_record``: the clean-environment validation's
  command list, the byte-identical report re-run (digest equality), the
  probe replica record, and the pending seal state.
- ``test_provenance_on_disk``: the provenance's schema, the recorded
  driver digests (the committed dual-tree digest), the frozen substrate
  block, the M8 frozen set, the round-5 module digests, the R01 data
  block, the environment, and the git state.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 537 entries, 536 at the
  card-11 solver landing) carries the register at its on-disk digest;
  the card-11 solver exception stays exactly once with the operative
  533 -> 536 re-pin named; this card's freeze exception names the card
  exactly once with the path-sorted 42-file list; the naming comment is
  in every re-pinned chain file with the card-11 history retained;
  the on-disk digests agree with the inventory.

Hermetic: standard library, the committed config and catalog, and the
sealed run dir (read-only) only; no SPS tables, no event generation, no
Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import deleted_events, sealed_mismatches, skip_unless_sealed

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking"
    / "staged_r51_topology_fraction_fit.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r51_topology_fraction_fit.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r51_topology_fraction_fit_2026-09-27")

RUN_DIR_REL = "dev/00runs/r51_topology_fraction_fit_2026-09-27"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r51_topology_fraction_fit.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r51_topology_fraction_fit.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r51_topology_fraction_fit.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r51_topology_fraction_fit_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260925-r5-run-r51-topology-fraction-fit"
MY_CARD = "TASK-20260925-r5-run-r51-topology-fraction-fit-test-author"

#: The committed dual-tree driver digest (byte-identical in both
#: trees; the freeze manifest inventory pins both copies here).
DRIVER_SHA256 = (
    "73bd74ab72270ae47b900b4da171b88db606234b1cbd4345c8135f48911f40a1"
)

#: The hermetic solver test file's committed digest (the Round 5
#: card 11 test-author count re-pin 536 -> 537 touched the
#: INVENTORY_COUNT constant, so the digest moved from the card-11
#: solver landing digest to the post-re-pin value).
SOLVER_TEST_SHA256 = (
    "6fdfc8936368779deab71886e8209f413ad67c7da60a4ab1d741510a5d3fd01d"
)

#: The sealed payload file digests (the checksum file's 14 entries; the
#: seal manifest is excluded from its own checksum file).
PRE_REGISTRATION_SHA256 = "c2b6fd81d8b52fb72405d07b10d9bfa8a96ce2eb4ce71633fe42bdda2bff11d9"
REPORT_FILE_SHA256 = "30ed195237ca83bf75ba24278bbb5ce67bfe1c55ad1da4081ace6fc241f9aede"
GATE5_FILE_SHA256 = "557e7f66a2b06ea08bd0f170db294ca4a4cd6012af8f7728f1d7f2477521e199"
PROVENANCE_FILE_SHA256 = "fa8b0303d7a9ab51d067dab875127e345283d628476b40cc418e100590ad172c"
REPORT_MD_FILE_SHA256 = "0e2b85c903178c1d5ff314d19c17fefe8757317fe61a55690e0536eace867b55"
CLEAN_ENV_FILE_SHA256 = "4fa47048a096f37a44b25ed980e2879005698eb266e009ff11cd973061902090"
REPLICA_FILE_SHA256 = {
    "replica_0.json": "7f97facb838c0ca4c27d23d861c1481f5eba7bae3d0ac5bcf8bb9a953bbd7da6",
    "replica_1.json": "c7b1329ccdf6d2cda4ec2c9c38072f2a3273cc678535f7ba71b39604f57c3a71",
    "replica_2.json": "d23a25c9f01e94c708290a69b3416dba4374174b302696580d0fba9a40f210dd",
    "replica_3.json": "f64c45b66bff2b7880fa51921b3bdf487e461968eb451c6b461ea0b0031f5bce",
    "replica_4.json": "33edb3ecc5e8ffdd8d43b2db6f0217a8ae29c039306e7aa265c89e2d0a428c12",
    "replica_5.json": "14566eaf0dbee1de594bfc05d97dd6b140d80ec2129db226f0ac66958054f713",
    "replica_6.json": "118e73cc0c4bf478b0a799a59b49852a22dae47ec4dcde9a123fd44f55dc9e43",
    "replica_7.json": "9920fa9b667552a9efb23ccdab3f6a8b4cf0ccdd7cbf642c977786e3ca0f433d",
}

#: The seal manifest's file list (8 replica records plus the six
#: top-level files), in seal order.
SEALED_FILES = (
    "REPORT.md",
    "clean_environment_commands.json",
    "fit_report.json",
    "gate5_r51.json",
    "pre_registration.json",
    "provenance.json",
) + tuple(f"replica_{i}.json" for i in range(8))

#: The run identity and file-name constants (pinned from the driver).
RUN_ID = "R51"
RUN_NAME = "r51_topology_fraction_fit_2026-09-27"
N_REPLICAS = 8
EVENTS_PER_REPLICA = 100000
N_EVENTS_TOTAL = 800000
MASTER_SEED = 20260925
SEED_OFFSET = 40000
EVENT_INDEX_BASE = 1100000000
SMOKE_SEED = 20312925
REPLICA_SEEDS = (20300945, 20300946, 20300947, 20300948, 20300949, 20300950, 20300951, 20300952)
CLASS_ORDER = ('sd_projectile', 'sd_target', 'dd', 'nd')
FIT_PARAMETERS = ('sd_projectile_fraction', 'sd_target_fraction', 'dd_fraction', 'epsilon', 'b_sd')
PARAM_BOUNDS = {'sd_projectile_fraction': (0.0, 1.0), 'sd_target_fraction': (0.0, 1.0), 'dd_fraction': (0.0, 1.0), 'epsilon': (-0.5, 0.5), 'b_sd': (1.0, 20.0)}
INITIAL_VALUES = {'sd_projectile_fraction': 0.25, 'sd_target_fraction': 0.25, 'dd_fraction': 0.25, 'epsilon': 0.0, 'b_sd': 8.504140762083612}
NOMINAL_PARAMS = {'epsilon': 0.0, 'b_sd': 8.504140762083612, 'class_fractions': {'sd_projectile': 0.25, 'sd_target': 0.25, 'dd': 0.25, 'nd': 0.25}, 'remnant_weights': {'intact_proton': 0.5, 'low_mass_baryonic_resonance': 0.5, 'high_mass_baryonic_continuum': 0.0}, 'p_n2': 0.0}
R45_FROZEN_CASCADE = {'kappa_B': 0.0, 'kappa_M': 4.0, 'b_dec_gev2': 2.0, 'baryonic_nominal_width_gev': 0.3, 'W_flavor': 1.0}
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
EVENT_CLOSURE_TOL = 1e-09
MULTIPLICITY_BINS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
SPECTRUM_SPECIES = ('proton', 'antiproton')
OPTIMIZER_SETTINGS = {'algorithm': 'scipy.optimize.minimize, method Nelder-Mead, bounded (scipy vertex clipping into PARAM_BOUNDS), adaptive True (the R41 convention)', 'xatol': 0.001, 'fatol': 1e-06, 'maxiter': 600, 'maxfev': 1200, 'initial_point': [0.25, 0.25, 0.25, 0.0, 8.504140762083612], 'determinism_note': 'the objective is a deterministic function of the parameter vector on the fixed generated sample (no sampler runs inside the objective), so the bounded Nelder-Mead walk from the fixed initial point is deterministic and a clean-environment re-run reproduces the fit report bytes', 'invalid_trial_note': 'a trial outside the domain (a negative fraction or f_nd < 0) or with a non-finite objective value returns +inf (the R41 target convention); the simplex avoids it'}
MODEL_UNCERTAINTY_CARRY = {'r32_spread': {'bin_spread_p95': 4.028426, 'feeddown_spread_max_sigma': 31.1773, 'persisting_residuals': 32, 'note': 'the R32 transition-window spread at 4,800,000 events; the model-uncertainty term of the fit (the carry on the B_transport observable)'}, 'b_transport_seagull': -2.3959765824349295e-05, 'b_transport_seagull_note': 'the b_transport_seagull beyond_isotropic_not_positive residual is a carry, not a pass criterion (index item 3)', 'engine_moment_floor_note': 'the calculation-engine moment-approximation floor (~1e-3 GeV/c) is tracked on its separate track (_knowledge/evidence/engine-moment-approximation-floor.md) and is not conflated with the R32 spread'}
CARD_CHAIN = ('TASK-20260920-r4-gate4-decision', 'TASK-20260925-r5-elastic-reference', 'TASK-20260925-r5-exchange-quantum-number-ledger', 'TASK-20260925-r5-intact-low-mass-remnants', 'TASK-20260925-r5-topology-mass-laws', 'TASK-20260925-r5-single-diffraction', 'TASK-20260925-r5-double-diffraction', 'TASK-20260925-r5-non-diffrative-central-excitation', 'TASK-20260925-r5-multiple-neutral-excitations', 'TASK-20260925-r5-topology-template-runner', 'TASK-20260925-r5-run-r50-pure-topology-templates', 'TASK-20260925-r5-run-r51-topology-fraction-fit')
FROZEN_SUBSTRATE = {'builder_blend_path': ('dev/mixing2006/na49_mixing/resonance_pp_builder.py', '4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1'), 'card6_routing': ('dev/mixing2006/na49_mixing/spectral_density.py', '561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563'), 'card2_engine': ('dev/mixing2006/na49_mixing/threshold_metric.py', 'ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63'), 'catalog': ('dev/mixing2006/config/v2_singlet/resonance_catalog.json', '6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd'), 'continuum_config': ('dev/mixing2006/config/v2_singlet/continuum_resonances.json', '242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1'), 'r45_driver': ('helpers/benchmarking/staged_r45_simplified_alignment_freeze.py', '48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10'), 'r46_driver': ('helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py', '121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f'), 'r50_driver': ('helpers/benchmarking/staged_r50_pure_topology_templates.py', 'b74e5408195ca6a3a53caf495da0c3f1206a9b43a73f336ba24b5cd6eb936fa3')}
RESERVED_ROUND5_BLOCKS = {'r50': [20300925, 20300944], 'r52': [20300953, 20300960], 'r53': [20300961, 20300968], 'r54': [20300969, 20300988], 'r55': [20300989, 20301008]}
RESERVED_ROUND5_PARTITIONS = {'r50': [1000000000, 1000025000], 'r52': [1200000000, 1200100000], 'r53': [1300000000, 1300100000], 'r54': [1400000000, 1400050000], 'r55': [1500000000, 1500250000], 'round5_smoke': [1600000000, 1600002000]}
PREVIOUS_USED_BLOCKS = {'r23': {'master_seed': 20260913, 'raw_seeds': [20300913, 20300920], 'event_index_range': None, 'source': 'the sealed R23-era raw block (the R41 disjointness record)'}, 'r32': {'master_seed': 20260918, 'raw_seeds': [20301519, 20301826], 'event_index_range': None, 'source': "the sealed R40 pre-registration's disjointness record (the R32 production blocks and confirmation block under the 20260918 master)"}, 'round4_production': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301083], 'event_index_range': None, 'source': 'the round 4 index card: R40 20301020-20301023, R41/R42/R43 20301020-20301027, R44 20301020-20301039, R45 20301056-20301075, R46 20301076-20301083'}, 'round4_smoke': {'master_seed': 20260920, 'raw_seeds': [20307920, 20311920], 'event_index_range': None, 'source': 'the round 4 index card: the smoke blocks 20307920-20311920 (raw offsets +7000 through +11000)'}, 'r46_smoke': {'master_seed': 20260920, 'raw_seeds': [20312920, 20312920], 'event_index_range': None, 'source': 'the sealed R46 pre-registration smoke block'}, 'r40_partition': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301023], 'event_index_range': [300000000, 389025000], 'event_index_partition': '300_000_000 + s*10_000_000 + p*1_000_000 + i, s < 9, p < 10, i < 25,000 (stays below 389,025,000)', 'source': 'sealed dev/00runs/r40_fixed_source_alignment_scan_2026-09-21/pre_registration.json'}, 'r41_partition': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301027], 'event_index_range': [400000000, 400100000], 'event_index_partition': '400_000_000 + i, i < 100,000', 'source': 'sealed dev/00runs/r41_nonstrange_pilot_fit_2026-09-23/pre_registration.json'}, 'r42_partition': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301027], 'event_index_range': [500000000, 500100000], 'event_index_partition': '500_000_000 + i, i < 100,000', 'source': 'sealed dev/00runs/r42_recoil_decay_separation_2026-09-23/pre_registration.json'}, 'r43_partition': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301027], 'event_index_range': [600000000, 600100000], 'event_index_partition': '600_000_000 + i, i < 100,000', 'source': 'sealed dev/00runs/r43_cascade_ablation_2026-09-23/pre_registration.json'}, 'r44_partition': {'master_seed': 20260920, 'raw_seeds': [20301020, 20301039], 'event_index_range': [700000000, 700250000], 'event_index_partition': '700_000_000 + i, i < 250,000', 'source': 'sealed dev/00runs/r44_frozen_nonstrange_validation_2026-09-23/pre_registration.json'}, 'r45_partition': {'master_seed': 20260920, 'raw_seeds': [20301056, 20301075], 'event_index_range': [800000000, 800250000], 'event_index_partition': '800_000_000 + i, i < 250,000', 'source': 'sealed dev/00runs/r45_simplified_alignment_freeze_2026-09-24/pre_registration.json'}, 'r46_partition': {'master_seed': 20260920, 'raw_seeds': [20301076, 20301083], 'event_index_range': [900000000, 900100000], 'event_index_partition': '900_000_000 + i, i < 100,000', 'source': 'sealed dev/00runs/r46_simplified_recoil_decay_profile_2026-09-24/pre_registration.json'}, 'round4_smoke_partitions': {'master_seed': 20260920, 'raw_seeds': None, 'event_index_range': [400000000, 400100000], 'event_index_partition': 'the Round 4 smoke partitions: R41 400_000_000 + i (slot 8000), R42 500_000_000 + i (slot 9000), R43 690_000_000 + i, R44 790_000_000 + i, R45 890_000_000 + i, R46 990_000_000 + i (each i < 2,000; the recorded union with the R41 range is the loosest bound the record carries)', 'source': 'the sealed R41-R46 pre-registrations (the round 5 index item 9)'}}
BEAM_MOMENTUM = 158.0
PROTON_MASS = 0.93827231
SQRT_S = 17.27

#: The sealed run timestamps and git state (pinned verbatim).
SEAL_SEALED_UTC = "2026-09-27T16:28:19Z"
PRE_REG_RECORDED_UTC = "2026-09-27T15:56:07Z"
PROVENANCE_RECORDED_UTC = "2026-09-27T15:56:07Z"
CLEAN_ENV_VERIFIED_UTC = "2026-09-27T16:27:50Z"
GIT_HEAD = "82638b809d803c8126b24ad74981f15f979aea51"

# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-topology-fraction-fit; +3 entries: the dual-tree driver helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested copy, and the test file tests/test_staged_r51_topology_fraction_fit.py).
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
#: The chain count-pin files re-pinned by this card (the 39 card-11
#: chain files plus the card-11 solver test).
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

#: This card's freeze exception file list (the manifest, this register,
#: and the chain count-pin files; 42 path-sorted entries).
MY_CARD_EXCEPTION_FILES = (
    sorted([MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES)))

#: The per-replica metadata (pinned verbatim from the sealed records).
REPLICA_SEED_BY_INDEX = {
    0: 20300945,
    1: 20300946,
    2: 20300947,
    3: 20300948,
    4: 20300949,
    5: 20300950,
    6: 20300951,
    7: 20300952,
}
REPLICA_CLASS_CENSUS = {
    0: {'dd': 25090, 'nd': 25051, 'sd_projectile': 25037, 'sd_target': 24822},
    1: {'dd': 25221, 'nd': 25048, 'sd_projectile': 24889, 'sd_target': 24842},
    2: {'dd': 25103, 'nd': 24970, 'sd_projectile': 24880, 'sd_target': 25047},
    3: {'dd': 25102, 'nd': 25033, 'sd_projectile': 25026, 'sd_target': 24839},
    4: {'dd': 25240, 'nd': 25006, 'sd_projectile': 24995, 'sd_target': 24759},
    5: {'dd': 24955, 'nd': 25129, 'sd_projectile': 24969, 'sd_target': 24947},
    6: {'dd': 24944, 'nd': 25121, 'sd_projectile': 24815, 'sd_target': 25120},
    7: {'dd': 25152, 'nd': 25068, 'sd_projectile': 24783, 'sd_target': 24997},
}
REPLICA_CONSERVATION = {
    0: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.2379162642232416e-14, 'worst_energy_closure_gev': 5.144329406903125e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 5.115907697472721e-12},
    1: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.0940088747022085e-14, 'worst_energy_closure_gev': 4.916955731459893e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 4.888534022029489e-12},
    2: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.111997298392338e-14, 'worst_energy_closure_gev': 4.945377440890297e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 4.916955731459893e-12},
    3: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.1479741457725955e-14, 'worst_energy_closure_gev': 5.002220859751105e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 4.973799150320701e-12},
    4: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 2.968398359946465e-14, 'worst_energy_closure_gev': 4.718003765447065e-12, 'worst_mass_shell_relative': 5.891625008335629e-10, 'worst_momentum_closure_gev': 4.689582056016661e-12},
    5: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.6156731617159525e-14, 'worst_energy_closure_gev': 5.7411853049416095e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 5.7127635955112055e-12},
    6: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 2.986078332561434e-14, 'worst_energy_closure_gev': 4.689582056016661e-12, 'worst_mass_shell_relative': 8.086252691197916e-10, 'worst_momentum_closure_gev': 4.718003765447065e-12},
    7: {'b_q_s_census': {'2,2,0': 100000}, 'events': 100000, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 2.86015936673053e-14, 'worst_energy_closure_gev': 4.519051799434237e-12, 'worst_mass_shell_relative': 6.089413019835981e-10, 'worst_momentum_closure_gev': 4.519051799434237e-12},
}
REPLICA_EVENT_HASH_SET = {
    0: "4755718b085ae571ed7a5c5c651041ac2795753108ad1addd1d28cac1d183103",
    1: "caab6bcff6c469481c4a8b3e0f8802afb88273776496a4aca9dc9b6e9c8b0b96",
    2: "623d5e105bbe80878f6002eec2601c799d57ac5dfdff291449674a0f348a4374",
    3: "ee911ebe5476a34bf713ae87396ea3cdc8c9794a6c988d541a0e14b35a044230",
    4: "a05900b8c6e1babe76e9bde37081a36300fa8b0ecd8042e5ae5546557b3821da",
    5: "36ec2b4dd5c560bb6c68eabc400bf11162e151160ad962f8c8d059d39291eec5",
    6: "a2d4face8670392b6b2984eb00f5afcd3f1ef87b55d50eac3868a91be1b524ea",
    7: "cd99f5d20a90e732120825ccb47f2dc5ff2afa1bdfc55c4edca5e415e3fc259f",
}

#: The fit report pins (pinned verbatim from the sealed report).
REPORT_DIGEST = "73b20d01921531429cc6004035f8fb442a4bbbd704a4f201f3a708976e76e03e"
FITTED = {'b_sd': 1.8457918026393214, 'dd_fraction': 0.9999997010778905, 'epsilon': 0.0033443027832747944, 'fractions': {'dd': 0.9999997010778905, 'nd': 3.1345895878942542e-09, 'sd_projectile': 3.619686246956965e-09, 'sd_target': 2.921678336937106e-07}, 'nd_fraction': 3.1345895878942542e-09, 'sd_projectile_fraction': 3.619686246956965e-09, 'sd_target_fraction': 2.921678336937106e-07}
OPTIMIZER_RESULT = {'best_point': {'b_sd': 1.8457918026393214, 'dd_fraction': 0.9999997010778905, 'epsilon': 0.0033443027832747944, 'sd_projectile_fraction': 3.619686246956965e-09, 'sd_target_fraction': 2.921678336937106e-07}, 'converged': False, 'invoked': True, 'message': 'Maximum number of iterations has been exceeded.', 'nfev': 1074, 'nit': 600, 'objective_value': 15957.584649927721, 'settings': {'algorithm': 'scipy.optimize.minimize, method Nelder-Mead, bounded (scipy vertex clipping into PARAM_BOUNDS), adaptive True (the R41 convention)', 'determinism_note': 'the objective is a deterministic function of the parameter vector on the fixed generated sample (no sampler runs inside the objective), so the bounded Nelder-Mead walk from the fixed initial point is deterministic and a clean-environment re-run reproduces the fit report bytes', 'fatol': 1e-06, 'initial_point': [0.25, 0.25, 0.25, 0.0, 8.504140762083612], 'invalid_trial_note': 'a trial outside the domain (a negative fraction or f_nd < 0) or with a non-finite objective value returns +inf (the R41 target convention); the simplex avoids it', 'maxfev': 1200, 'maxiter': 600, 'xatol': 0.001}}
CONSERVATION = {'b_q_s_census': {'2,2,0': 800000}, 'events_checked': 800000, 'events_stored': 800000, 'pass': True, 'tolerance': 1e-09, 'violations': 0, 'worst_closure_relative': 3.6156731617159525e-14, 'worst_energy_closure_gev': 5.7411853049416095e-12, 'worst_mass_shell_relative': 8.086252691197916e-10, 'worst_momentum_closure_gev': 5.7127635955112055e-12}
OBJECTIVE_AT_INITIAL = {'ess': {'min': 1.0, 'per_block': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, 'log_shift': 0.0, 'minus2lnl_profiled': 151414.26070437056, 'n_eff_total': 800000.0, 'nuisances': [0.701862826967234, 0.6236244853746062, -0.9468503051253975]}
OBJECTIVE_AT_OPTIMUM = {'dropped_bins': 1, 'ess': {'min': 0.00024177313336515557, 'per_block': [0.0011025340542270036, 0.001297728617848435, 0.0006023309591162278, 0.0020419698488048244, 0.0035078203350823014, 0.00024177313336515557, 0.0042599824514324555, 0.003200009554048086]}, 'log_shift': 9.791795160176466, 'minus2lnl_profiled': 15957.584649927721, 'n_eff_total': 37.34087056335259, 'nuisance_columns': ['norm_leading_proton_spectrum', 'norm_b_transport_spectrum', 'carry'], 'nuisances': [1.136379639875093, 0.9769096934801882, -1.255597735509294]}
HESSIAN = {'curvature': [[float("inf"), float("nan"), float("nan"), float("nan"), float("nan")], [float("nan"), float("inf"), float("nan"), float("nan"), float("nan")], [float("nan"), float("nan"), float("inf"), float("nan"), float("nan")], [float("nan"), float("nan"), float("nan"), 80666.88173104922, -4774.1886677667935], [float("nan"), float("nan"), float("nan"), -4774.1886677667935, 4790.069852857049]], 'eigenvalues': None, 'minima_agree': False, 'note': 'the Hessian is quoted only after the optimizer converged, the replica-block minima agree (the stability table passes on every parameter), and the pooled curvature is positive definite (plan line 1871; the R46 pattern); no Hessian-based covariance is emitted otherwise', 'pooled_errors': {'b_sd': 0.014448710073485022, 'dd_fraction': float("inf"), 'epsilon': 0.0035208892576015596, 'sd_projectile_fraction': float("inf"), 'sd_target_fraction': float("inf")}, 'positive_definite': False, 'quoted': False, 'steps': {'b_sd': 0.05, 'dd_fraction': 0.005, 'epsilon': 0.005, 'sd_projectile_fraction': 0.005, 'sd_target_fraction': 0.005}}
BIN_STRUCTURE = {'dropped': 1, 'kept_per_statistic': {'b_transport_spectrum': 122, 'charged_multiplicity_distribution': 12, 'gap_fraction_abs_y_lt_0.5': 1, 'leading_proton_spectrum': 135}}
DROPPED_BINS = [{'index': 11, 'reason': 'structurally empty at the generation point (no stored event contributes at unit weights; the R11 convention)', 'statistic': 'charged_multiplicity_distribution'}]
ANALYTIC_CHECKS = {'elastic_events': 0, 'elastic_note': 'the mixed draw runs at the elastic fraction exactly 0.0 (plan line 824); a drawn elastic label refuses the run', 'fitted_f_nd': 3.1345895878942542e-09, 'fraction_closure': 'f_nd = 1 - f_sd_p - f_sd_t - f_dd exactly at the fitted point (the sum rule over the inelastic sample)', 'nominal_weight_identity': {'ess_at_initial': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'log_shift_at_initial': 0.0, 'note': "at the pre-registered initial point every importance weight is exactly 1: the log shift is 0.0 and every block's ESS is exactly 1", 'pass': True}, 'xf_pair_sum_max_abs_two_body': 8.704148513061227e-14, 'xf_pair_sum_note': 'the two-body classes (sd_projectile, sd_target, dd) carry the exact back-to-back CM pair sum x_F^p + x_F^t = 0 (the closed kinematics; the elastic normalization reference carries the same identity)', 'xf_pair_sum_tolerance': 1e-09}
AGREEMENT_KEYS = [
    "leading_proton_spectrum",
    "b_transport_spectrum",
    "gap_fraction_abs_y_lt_0.5",
    "charged_multiplicity_distribution",
]
AGREEMENT_DEVANCE = {
    "leading_proton_spectrum":
        (18331.206480095512,
         16468.03984562138),
    "b_transport_spectrum":
        (13746.091499259857,
         11993.567337894774),
    "gap_fraction_abs_y_lt_0.5":
        (5.489036834291573,
         0.007900954760198035),
    "charged_multiplicity_distribution":
        (540.8985662721461,
         0.22335851051293704),
}
AGREEMENT_PULL = {
    "leading_proton_spectrum":
        (64.15173159392114,
         53.32947300113225),
    "b_transport_spectrum":
        (63.41632810989486,
         52.594069517105964),
    "gap_fraction_abs_y_lt_0.5":
        (2.34286935920285,
         0.08888731495662379),
    "charged_multiplicity_distribution":
        (14.738260404265047,
         0.32748598529781486),
}
AGREEMENT_N_KEPT = {
    "leading_proton_spectrum":
        (135,
         135),
    "b_transport_spectrum":
        (122,
         122),
    "gap_fraction_abs_y_lt_0.5":
        (1,
         1),
    "charged_multiplicity_distribution":
        (13,
         12),
}

#: The gate record pins (pinned verbatim from the sealed gate record).
GATE5_CLAUSE = "fractions required by topology-sensitive data"
GATE5_REPORT_DIGEST = "73b20d01921531429cc6004035f8fb442a4bbbd704a4f201f3a708976e76e03e"
GATE5_CONSERVATION = {'events_checked': 800000, 'pass': True, 'tolerance': 1e-09, 'violations': 0, 'worst_energy_closure_gev': 5.7411853049416095e-12}
GATE5_EXCLUDED = {'blend_ramp_start': 'round 4 obligation 4 (fixed at B)', 'elastic': 'plan line 824', 'strange_sector': 'plan line 918'}
GATE5_FITTED = {'b_sd': 1.8457918026393214, 'epsilon': 0.0033443027832747944, 'fractions': [3.619686246956965e-09, 2.921678336937106e-07, 0.9999997010778905, 3.1345895878942542e-09]}
GATE5_NAMED_RESIDUALS = [
    "optimizer_non_convergence",
    "replica_block_minima_disagree",
    "replica_block_minima_disagree",
    "replica_block_minima_disagree",
    "replica_block_minima_disagree",
    "replica_block_minima_disagree",
    "block_curvature_failure",
    "weight_degeneracy",
]
GATE5_STABILITY_PARAMS = ['b_sd', 'dd_fraction', 'epsilon', 'sd_projectile_fraction', 'sd_target_fraction']

#: The provenance pins (pinned verbatim from the sealed provenance).
PROV_DRIVER_SHA256 = "73bd74ab72270ae47b900b4da171b88db606234b1cbd4345c8135f48911f40a1"
PROV_ROUND5_MODULES = {'card1_elastic': '7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824', 'card2_ledger': '3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a', 'card3_remnants': '8eb45877490c04896c7cc7062e174dd6c060d96b1bace07688c16086e8a463dc', 'card4_mass_laws': 'c15e3416b022390bcc59dd3f0e17f4264b272beffbd9f3863a093a88a5fc0b48', 'card5_single_diffraction': '5868b178fb3fe5852d5baf141b5cb949554747a1dcbd3496d3301251683dd2cb', 'card6_double_diffraction': 'f17a83324deef6be7f415c24ec7742820d99291407705546df8703e34c0f077f', 'card7_non_diffractive': '68162f96614022e5f95807155cbd34f6714cb14bf8e4a5eaf116764f31bfd4b2', 'card8_multi_neutral': '9527643b387ceb29ac22c8854f45f28c22c7a14215b35c55fafc499993bb8454', 'card9_runner': '0394c010d581416ca4f59677401632ab53e7acc8fa54fe147ce4f568370e0dbb'}
PROV_M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
PROV_SEALED_INPUTS = {'r01': {'entries': 28, 'intact': True, 'mismatches': [], 'run_dir': 'dev/00runs/r01_legacy_pp_reference_2026-09-11'}, 'r50': {'entries': 26, 'intact': True, 'mismatches': [], 'run_dir': 'dev/00runs/r50_pure_topology_templates_2026-09-27'}}


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r51_topology_fraction_fit", NESTED_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r51 = _load_driver()


def _load_json(name):
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _event_hash(record):
    """Independent recompute of one stored event's hash: the sha256 of
    the record's canonical JSON without its ``h`` key."""
    rec = {k: v for k, v in record.items() if k != "h"}
    return hashlib.sha256(
        json.dumps(rec, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()


def _nan_eq(a, b):
    """Deep equality that treats nan == nan, for the Hessian pin."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    if isinstance(a, dict):
        return set(a) == set(b) and all(_nan_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (len(a) == len(b)
                and all(_nan_eq(x, y) for x, y in zip(a, b)))
    return a == b


def test_dual_tree_driver_byte_identity():
    """Both driver copies exist, are byte-identical, and sit at the
    committed digest; the freeze manifest inventory pins both."""
    assert NESTED_DRIVER.is_file()
    assert OUTER_DRIVER.is_file()
    nested_digest = _sha256_file(NESTED_DRIVER)
    outer_digest = _sha256_file(OUTER_DRIVER)
    assert nested_digest == outer_digest, (
        "the dual-tree driver is not byte-identical")
    assert nested_digest == DRIVER_SHA256, (
        "the driver is not at the committed digest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == DRIVER_SHA256
    assert inventory[DRIVER_REL_OUTER] == DRIVER_SHA256


def test_driver_identity_constants():
    """The run identity, file-name constants, class order, fit set and
    bounds, initial and nominal parameters, frozen R45 cascade, seed-
    map constants, M8 hash, model-uncertainty carry, optimizer settings,
    closure tolerance, and bin counts."""
    assert r51.RUN_ID == RUN_ID
    assert r51.RUN_NAME == RUN_NAME
    assert r51.CARD == SOLVER_CARD
    assert r51.DRIVER_REL_NESTED == DRIVER_REL_NESTED
    assert r51.DRIVER_REL_OUTER == DRIVER_REL_OUTER
    assert r51.DRIVER_SHA256 == DRIVER_SHA256
    assert r51.N_REPLICAS == N_REPLICAS
    assert r51.EVENTS_PER_REPLICA == EVENTS_PER_REPLICA
    assert r51.N_EVENTS_TOTAL == N_EVENTS_TOTAL
    assert r51.MASTER_SEED == MASTER_SEED
    assert r51.SEED_OFFSET == SEED_OFFSET
    assert r51.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert r51.SMOKE_SEED == SMOKE_SEED
    assert tuple(r51.REPLICA_SEEDS) == REPLICA_SEEDS
    assert tuple(r51.CLASS_ORDER) == CLASS_ORDER
    assert tuple(r51.FIT_PARAMETERS) == FIT_PARAMETERS
    assert dict(r51.PARAM_BOUNDS) == PARAM_BOUNDS
    assert dict(r51.INITIAL_VALUES) == INITIAL_VALUES
    assert dict(r51.R45_FROZEN_CASCADE) == R45_FROZEN_CASCADE
    assert r51.M8_PARAMETER_HASH == M8_HASH
    assert r51.EVENT_CLOSURE_TOL == EVENT_CLOSURE_TOL
    assert tuple(r51.MULTIPLICITY_BINS) == MULTIPLICITY_BINS
    assert tuple(r51.SPECTRUM_SPECIES) == SPECTRUM_SPECIES
    assert dict(r51.OPTIMIZER_SETTINGS) == OPTIMIZER_SETTINGS
    assert dict(r51.MODEL_UNCERTAINTY_CARRY) == MODEL_UNCERTAINTY_CARRY
    assert tuple(r51.CARD_CHAIN) == CARD_CHAIN
    assert dict(r51.FROZEN_SUBSTRATE) == FROZEN_SUBSTRATE
    assert dict(r51.RESERVED_ROUND5_BLOCKS) == RESERVED_ROUND5_BLOCKS
    assert dict(r51.RESERVED_ROUND5_PARTITIONS) == RESERVED_ROUND5_PARTITIONS
    assert dict(r51.PREVIOUS_USED_BLOCKS) == PREVIOUS_USED_BLOCKS
    assert r51.BEAM_MOMENTUM == BEAM_MOMENTUM
    assert r51.PROTON_MASS == PROTON_MASS
    assert r51.SQRT_S == SQRT_S
    np_ = r51.NOMINAL_PARAMS
    assert np_.epsilon == NOMINAL_PARAMS["epsilon"]
    assert np_.b_sd == NOMINAL_PARAMS["b_sd"]
    assert dict(np_.class_fractions) == NOMINAL_PARAMS["class_fractions"]
    assert dict(np_.remnant_weights) == NOMINAL_PARAMS["remnant_weights"]
    assert np_.p_n2 == NOMINAL_PARAMS["p_n2"]


def test_solver_test_file_digest():
    """The hermetic solver test at its post-re-pin committed digest,
    pinned in the freeze manifest inventory."""
    solver_path = NESTED_ROOT / "tests" / (
        "test_staged_r51_topology_fraction_fit.py")
    assert solver_path.is_file()
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_tree_sha256"][SOLVER_TEST_REL] == (
        SOLVER_TEST_SHA256)


def test_seed_map_partition():
    """The master seed, the raw replica block 20300945-20300952, the
    event-index partition 1_100_000_000 + i, the smoke block 20312925,
    the per-replica seeds, the reserved round-5 blocks and partitions,
    and the driver's disjointness report against the pre-registration."""
    assert r51.MASTER_SEED == MASTER_SEED
    assert r51.REPLICA_SEEDS == tuple(REPLICA_SEEDS)
    assert r51.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert r51.SMOKE_SEED == SMOKE_SEED
    assert r51.RESERVED_ROUND5_BLOCKS == RESERVED_ROUND5_BLOCKS
    assert r51.RESERVED_ROUND5_PARTITIONS == RESERVED_ROUND5_PARTITIONS
    skip_unless_sealed(RUN_DIR)
    pre = _load_json("pre_registration.json")
    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert seed_map["raw_replica_seeds"] == list(REPLICA_SEEDS)
    assert seed_map["event_index_base"] == EVENT_INDEX_BASE
    assert seed_map["smoke"]["raw_seed"] == SMOKE_SEED
    assert seed_map["smoke"]["events"] == 2000
    disjoint = seed_map["disjointness"]
    assert disjoint["disjoint"] is True
    assert disjoint["r51_raw_seed_block"] == [
        REPLICA_SEEDS[0], REPLICA_SEEDS[-1]]
    assert disjoint["r51_event_index_range"] == [
        EVENT_INDEX_BASE, EVENT_INDEX_BASE + EVENTS_PER_REPLICA]
    assert disjoint["r51_replica_seeds"] == list(REPLICA_SEEDS)
    assert disjoint["shared_raw_seeds"] == {}


def test_pre_registration_on_disk():
    """The sealed pre-registration's schema, identity, statistics, seed
    map, initial values, conservation contract, M8 frozen set, frozen
    cascade, frozen substrate digests, model-uncertainty carry,
    optimizer record, fit set, and card chain."""
    skip_unless_sealed(RUN_DIR)
    pre = _load_json("pre_registration.json")
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256)
    assert pre["schema"] == (
        "na49_mixing.staged_r51_topology_fraction_fit.pre_registration/1")
    assert pre["card"] == SOLVER_CARD
    assert pre["run"] == RUN_ID
    assert pre["run_name"] == RUN_NAME
    assert pre["recorded_utc"] == PRE_REG_RECORDED_UTC
    assert pre["written_before_generation"] is True
    assert pre["card_chain"] == list(CARD_CHAIN)
    stats = pre["statistics"]
    assert stats["replicas"] == N_REPLICAS
    assert stats["events_per_replica"] == EVENTS_PER_REPLICA
    assert stats["events_total"] == N_EVENTS_TOTAL
    assert pre["conservation"]["tolerance"] == EVENT_CLOSURE_TOL
    assert pre["conservation"]["expected_b_q_s"] == [2, 2, 0]
    m8 = pre["m8_frozen_set"]
    assert m8["hash"] == M8_HASH
    assert m8["verified"] is True
    assert pre["frozen_cascade"] == R45_FROZEN_CASCADE
    assert set(pre["frozen_substrate"]) == set(FROZEN_SUBSTRATE)
    for name, block in pre["frozen_substrate"].items():
        assert block["expected"] == FROZEN_SUBSTRATE[name][1]
        assert block["sha256"] == FROZEN_SUBSTRATE[name][1]
    assert pre["model_uncertainty_carry"] == MODEL_UNCERTAINTY_CARRY
    assert pre["optimizer"] == OPTIMIZER_SETTINGS
    fit = pre["fit_set"]
    assert fit["count"] == 5
    assert fit["parameters"] == list(FIT_PARAMETERS)
    assert fit["bounds"] == {k: list(v) for k, v in PARAM_BOUNDS.items()}
    assert fit["initial_values"] == INITIAL_VALUES
    assert fit["pass"] is True
    init = pre["initial_values"]
    assert init["values"] == INITIAL_VALUES
    assert init["assertion_against_sealed_r50"]["match"] is True
    assert pre["multiplicity_bins"] == list(MULTIPLICITY_BINS)
    assert pre["sealed_inputs"] == {
        "r01": {"entries": 28, "intact": True, "mismatches": [],
                "run_dir": "dev/00runs/"
                           "r01_legacy_pp_reference_2026-09-11"},
        "r50": {"entries": 26, "intact": True, "mismatches": [],
                "run_dir": "dev/00runs/"
                           "r50_pure_topology_templates_2026-09-27"}}


def test_replica_records_on_disk():
    """The 8 replica records (identity, class census, seed,
    conservation block) pinned verbatim, the per-replica event hash
    sets, and the independent recompute of the event hash set, class
    census, and conservation worst-closures from the stored events over
    all 800,000 events."""
    skip_unless_sealed(RUN_DIR)
    for i in range(N_REPLICAS):
        rec = _load_json(f"replica_{i}.json")
        assert _sha256_file(RUN_DIR / f"replica_{i}.json") == (
            REPLICA_FILE_SHA256[f"replica_{i}.json"])
        assert rec["schema"] == (
            "na49_mixing.staged_r51_topology_fraction_fit.replica/1")
        assert rec["card"] == SOLVER_CARD
        assert rec["run"] == RUN_ID
        assert rec["replica"] == i
        assert rec["seed"] == REPLICA_SEED_BY_INDEX[i]
        assert rec["n_events"] == EVENTS_PER_REPLICA
        assert rec["reused"] is False
        assert rec["identity"]["driver_sha256"] == DRIVER_SHA256
        assert rec["identity"]["pre_registration_sha256"] == (
            PRE_REGISTRATION_SHA256)
        assert rec["class_census"] == REPLICA_CLASS_CENSUS[i]
        assert rec["conservation"] == REPLICA_CONSERVATION[i]
        assert rec["event_hash_set_sha256"] == REPLICA_EVENT_HASH_SET[i]
        # independent recompute from the stored events
        events = rec["events"]
        assert len(events) == EVENTS_PER_REPLICA
        census = {}
        hashes = []
        worst_closure = 0.0
        worst_shell = 0.0
        for ev in events:
            assert _event_hash(ev) == ev["h"], (
                f"replica {i} event {ev['i']}: the stored hash does "
                "not recompute")
            hashes.append(ev["h"])
            census[ev["cl"]] = census.get(ev["cl"], 0) + 1
            worst_closure = max(worst_closure, max(ev["clo"]))
            worst_shell = max(worst_shell, ev["sh"])
        assert census == rec["class_census"]
        assert hashlib.sha256(
            "\n".join(hashes).encode("utf-8")).hexdigest() == (
                rec["event_hash_set_sha256"])
        cons = rec["conservation"]
        assert cons["b_q_s_census"] == {"2,2,0": EVENTS_PER_REPLICA}
        assert cons["events"] == EVENTS_PER_REPLICA
        assert cons["violations"] == 0
        assert cons["tolerance"] == EVENT_CLOSURE_TOL
        assert cons["worst_closure_relative"] == worst_closure
        assert cons["worst_mass_shell_relative"] == worst_shell


def test_fit_report_on_disk():
    """The sealed fit report's schema and self-digest, the global
    conservation block, the fitted point, the optimizer record (the
    pre-registered non-convergence), the objective at the initial point
    and the optimum, the Hessian discipline, the bin structure, the
    dropped bin, and the analytic checks."""
    skip_unless_sealed(RUN_DIR)
    fr = _load_json("fit_report.json")
    assert _sha256_file(RUN_DIR / "fit_report.json") == REPORT_FILE_SHA256
    assert fr["schema"] == (
        "na49_mixing.staged_r51_topology_fraction_fit.report/1")
    assert fr["card"] == SOLVER_CARD
    assert fr["run"] == RUN_ID
    assert fr["digest"] == REPORT_DIGEST
    assert fr["fitted"] == FITTED
    assert fr["optimizer"] == OPTIMIZER_RESULT
    assert fr["conservation"] == CONSERVATION
    assert fr["objective_at_initial"] == OBJECTIVE_AT_INITIAL
    assert fr["objective_at_optimum"] == OBJECTIVE_AT_OPTIMUM
    # the curvature matrix carries nan entries (the unquoted Hessian);
    # compare nan-aware
    assert _nan_eq(fr["hessian"], HESSIAN)
    assert fr["bin_structure"] == BIN_STRUCTURE
    assert fr["dropped_bins"] == DROPPED_BINS
    assert fr["analytic_checks"] == ANALYTIC_CHECKS
    assert fr["initial_values"] == INITIAL_VALUES
    assert fr["optimizer_invoked"] is True
    assert fr["optimizer_uses"] == 1
    # the optimizer did not converge (the pre-registered budget)
    assert fr["optimizer"]["converged"] is False
    assert fr["hessian"]["quoted"] is False
    assert fr["hessian"]["positive_definite"] is False
    # the agreement block: the four pre-registered observables
    assert [a["key"] for a in fr["agreement"]] == AGREEMENT_KEYS
    for a in fr["agreement"]:
        key = a["key"]
        assert (a["deviance_contribution_fit"],
                a["deviance_contribution_initial"]) == AGREEMENT_DEVANCE[key]
        assert (a["max_abs_pull_fit"], a["max_abs_pull_initial"]) == (
            AGREEMENT_PULL[key])
        assert (a["n_bins"], a["n_kept"]) == AGREEMENT_N_KEPT[key]
    # the conservation pool matches the replica worst-closures
    assert CONSERVATION["events_checked"] == N_EVENTS_TOTAL
    assert CONSERVATION["b_q_s_census"] == {"2,2,0": N_EVENTS_TOTAL}
    assert CONSERVATION["violations"] == 0
    assert CONSERVATION["worst_closure_relative"] == max(
        c["worst_closure_relative"] for c in REPLICA_CONSERVATION.values())
    assert CONSERVATION["worst_mass_shell_relative"] == max(
        c["worst_mass_shell_relative"] for c in
        REPLICA_CONSERVATION.values())


def test_gate5_record_on_disk():
    """The gate record's identity, clause, report digest, conservation
    summary (zero violations over 800,000 events), excluded observables,
    optimizer record, fitted point, the named-residual register, and the
    replica-block stability table."""
    skip_unless_sealed(RUN_DIR)
    g = _load_json("gate5_r51.json")
    assert _sha256_file(RUN_DIR / "gate5_r51.json") == GATE5_FILE_SHA256
    assert g["schema"] == "na49_mixing.staged_r51_topology_fraction_fit.gate/1"
    assert g["card"] == SOLVER_CARD
    assert g["run"] == RUN_ID
    assert g["clause"] == GATE5_CLAUSE
    assert g["report_digest"] == GATE5_REPORT_DIGEST
    assert g["conservation"] == GATE5_CONSERVATION
    assert g["excluded"] == GATE5_EXCLUDED
    assert g["fitted"] == GATE5_FITTED
    assert g["fit_set"] == list(FIT_PARAMETERS)
    assert g["free_parameters"] == 5
    assert g["events_checked"] == N_EVENTS_TOTAL
    assert g["optimizer_invoked"] is True
    assert g["optimizer_uses"] == 1
    assert g["hessian"]["quoted"] is False
    assert g["named_residual_register"] is not None
    assert [entry["kind"] for entry in g["named_residual_register"]] == (
        GATE5_NAMED_RESIDUALS)
    assert sorted(g["replica_block_stability"]) == GATE5_STABILITY_PARAMS
    for param, block in g["replica_block_stability"].items():
        assert block["pass"] is False, (
            f"{param}: the pre-registered replica-block stability bound "
            "is not met (the R51 finite-MC outcome)")
    assert g["sealed"] is True


def test_seal_manifest_and_checksums():
    """The seal manifest's 14 sealed files, the checksum file's 14
    entries, the 16-file run dir, and the full re-hash of all 14 sealed
    files against the checksum digests."""
    skip_unless_sealed(RUN_DIR)
    seal = _load_json("seal_manifest.json")
    assert seal["schema"] == (
        "na49_mixing.staged_r51_topology_fraction_fit.seal/1")
    assert seal["card"] == SOLVER_CARD
    assert seal["run"] == RUN_ID
    assert seal["sealed_utc"] == SEAL_SEALED_UTC
    assert seal["record_count"] == N_REPLICAS
    assert seal["sha256sums_entries"] == len(SEALED_FILES)
    assert seal["files"] == list(SEALED_FILES)
    assert seal["git_state"]["head"] == GIT_HEAD
    assert seal["git_state"]["branch"] == "main"
    assert seal["git_state"]["dirty"] is True
    sums_lines = (RUN_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8")
    sums = {}
    for line in sums_lines.splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    assert set(sums) == set(SEALED_FILES)
    assert sealed_mismatches(RUN_DIR) == []
    absent = {n for n in deleted_events(RUN_DIR) if not (RUN_DIR / n).is_file()}
    expected = [n for n in list(SEALED_FILES) + ["SHA256SUMS.txt", "seal_manifest.json"]
                if n not in absent]
    if (RUN_DIR / "EVENTS_DELETED.txt").is_file():
        expected.append("EVENTS_DELETED.txt")
    run_files = sorted(p.name for p in RUN_DIR.iterdir() if p.is_file())
    assert run_files == sorted(expected)


def test_clean_environment_record():
    """The clean-environment validation's command list, the byte-
    identical report re-run (digest equality), the probe replica record,
    and the pending seal state."""
    skip_unless_sealed(RUN_DIR)
    clean = _load_json("clean_environment_commands.json")
    assert _sha256_file(
        RUN_DIR / "clean_environment_commands.json") == CLEAN_ENV_FILE_SHA256
    assert clean["schema"] == (
        "na49_mixing.staged_r51_topology_fraction_fit.clean_env/1")
    assert clean["card"] == SOLVER_CARD
    assert clean["run"] == RUN_ID
    assert clean["verified_utc"] == CLEAN_ENV_VERIFIED_UTC
    assert clean["report_digest"] == REPORT_DIGEST
    assert clean["report_hash_match"] is True
    assert clean["report_rerun_digest"] == clean["report_digest"]
    probe = clean["probe_replica"]
    assert probe["replica"] == 0
    assert probe["seed"] == REPLICA_SEEDS[0]
    assert probe["file"] == "replica_0.json"
    assert probe["matches_sealed_record"] is True
    assert probe["census_matches"] is True
    assert probe["conservation_matches"] is True
    assert probe["event_hash_set_match"] is True
    seal_state = clean["seal"]
    assert seal_state["entries"] == 0
    assert seal_state["mismatches"] == []
    assert "pending" in seal_state["state"]


def test_provenance_on_disk():
    """The provenance's schema, the recorded driver digests (the
    committed dual-tree digest), the frozen substrate block, the M8
    frozen set, the round-5 module digests, the R01 data block, the
    environment, and the git state."""
    skip_unless_sealed(RUN_DIR)
    prov = _load_json("provenance.json")
    assert _sha256_file(RUN_DIR / "provenance.json") == PROVENANCE_FILE_SHA256
    assert prov["schema"] == (
        "na49_mixing.staged_r51_topology_fraction_fit.provenance/1")
    assert prov["card"] == SOLVER_CARD
    assert prov["run"] == RUN_ID
    assert prov["recorded_utc"] == PROVENANCE_RECORDED_UTC
    assert prov["card_chain"] == list(CARD_CHAIN)
    driver = prov["driver"]
    assert driver["byte_identical"] is True
    assert driver["nested"]["sha256"] == DRIVER_SHA256
    assert driver["outer"]["sha256"] == DRIVER_SHA256
    assert driver["nested"]["path"] == DRIVER_REL_NESTED
    assert driver["outer"]["path"] == DRIVER_REL_OUTER
    assert PROV_DRIVER_SHA256 == DRIVER_SHA256
    for name, block in prov["frozen_substrate"].items():
        assert block["sha256"]["sha256"] == FROZEN_SUBSTRATE[name][1]
        assert block["sha256"]["verified"] is True
        assert block["verified"] is True
    assert prov["m8_frozen_set"]["hash"] == PROV_M8_HASH
    assert prov["round5_modules"] == PROV_ROUND5_MODULES
    assert prov["sealed_inputs"] == PROV_SEALED_INPUTS
    assert prov["git_state"]["head"] == GIT_HEAD
    assert prov["git_state"]["branch"] == "main"
    assert prov["git_state"]["dirty"] is True
    env = prov["environment"]
    assert env == {"machine": "x86_64", "numpy": "1.26.4",
                   "python": "3.12.3",
                   "platform": (
                       "Linux-6.8.0-139-generic-x86_64-with-glibc2.39")}
    r01 = prov["r01_data"]
    assert r01["aligned_cells_b_transport"] == 122
    assert r01["carry_bins"] == 32
    assert r01["published_rows"] == {"antiproton": 122, "proton": 135}
    assert r01["spectrum_species"] == list(SPECTRUM_SPECIES)



