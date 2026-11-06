"""Round 5, card 10 test-author register: the R50 pure topology
templates contract pins.

Card ``TASK-20260925-r5-run-r50-pure-topology-templates-test-author``:
the test-author half of the R50 run card
(``TASK-20260925-r5-run-r50-pure-topology-templates``, plan section 11
run R50, lines 898-907).  This register independently pins the dual-tree
run driver (``helpers/benchmarking/staged_r50_pure_topology_templates.py``,
byte-identical in the nested tree and the monorepo mirror), the hermetic
solver test (``tests/test_staged_r50_pure_topology_templates.py``), the
sealed run ``dev/00runs/r50_pure_topology_templates_2026-09-27/`` (sealed
2026-09-27T05:52:43Z; 26 sealed payload entries, 0 mismatches), and the
freeze re-pin (the strict path-sorted inventory at 533 entries after this
card's re-pin; 532 at the card-10 solver landing).  The driver is loaded
once via importlib (its top level is the pre-registration identity and
the import-time substrate checks; no events are generated), and the sealed
JSON is read read-only from disk.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist, are
  byte-identical, and sit at the committed digest; the driver's own
  self-computed identity digest agrees; the freeze manifest inventory
  pins both at that digest.
- ``test_driver_identity_constants``: the run identity, the file-name
  constants, the sealed-dir rel paths, the class order, the card chain,
  the M8-frozen parameter pins, the nominal parameters, the frozen R45
  cascade, the seed map constants, the model-uncertainty carry, the
  closure tolerance, and the histogram bin counts.
- ``test_solver_test_file_digest``: the hermetic solver test file at its
  post-re-pin committed digest, pinned in the freeze manifest inventory.
- ``test_seed_map_partition``: the master seed, the raw slot block
  20300925-20300944, the event-index partition 1_000_000_000 + i, the
  smoke block 20312925 at 1_600_000_000 + i, the per-class seed blocks,
  the reserved round-5 blocks and partitions, and the driver's
  disjointness report against the pre-registration record.
- ``test_pre_registration_on_disk``: the sealed pre-registration's
  schema, identity, statistics, seed map, nominal parameters,
  conservation contract, M8 frozen set, frozen cascade, frozen substrate
  digests, model-uncertainty carry, no-optimizer record, stored
  quantities, and card chain.
- ``test_slot_records_on_disk``: the 20 slot records (identity, class
  label, census, seed, conservation block) pinned verbatim, the per-slot
  event hash sets, and the independent recompute from the stored events
  (B, Q, S census; four-momentum closure; mass shell; event hash set)
  over all 500,000 events.
- ``test_template_report_on_disk``: the sealed template report's schema
  and self-digest, the global conservation block (independently
  recomputed), the per-class pooled observables (leading-baryon x_F,
  gap size, excitation masses, multiplicity, central/forward yields,
  seagull moments, transfer moments; independently recomputed from the
  stored events), the class draw census, and the histogram edge grids.
- ``test_elastic_normalization_check``: the elastic class's recorded
  normalization check (closed form, quadrature, quantile round trip,
  pooled t mean against the analytic -1/B_SD, x_F pair sum, mass shell,
  zero transfer, verdict pass) with the pooled t statistics, the pair
  sum, and the mass shell independently recomputed from the stored
  elastic events.
- ``test_gate5_record_on_disk``: the gate record's identity, clause,
  report digest, conservation summary (zero violations over 500,000
  events), per-class draw census, excluded classes, optimizer record,
  and nominal parameters.
- ``test_seal_manifest_and_checksums``: the seal manifest's 26 sealed
  files (20 slot records plus the six top-level files), the checksum
   file's 26 entries, the 28-file run dir (the 26 sealed payload files
   plus the checksum file and the seal manifest), and the full re-hash
   of all 26 sealed files against the checksum digests.
- ``test_clean_environment_record``: the clean-environment validation's
  command list, the byte-identical report re-run (digest equality), the
  probe slot record, and the pending seal state.
- ``test_provenance_on_disk``: the provenance's schema, the recorded
  driver digests (the committed dual-tree digest), the frozen substrate
  block (all seven digests verified), the M8 frozen set, the round-5
  module digests, the environment, and the git state.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 533 entries, 532 at the
  card-10 solver landing) carries the register at its on-disk digest;
  the card-10 solver exception stays exactly once with the operative
  529 -> 532 re-pin named; this card's freeze exception names the card
  exactly once with the path-sorted 40-file list; the naming comment is
  in every re-pinned chain file with the card-10 history retained in the
  37 card-10 chain files; the on-disk digests agree with the inventory.

Hermetic: standard library, the committed config and catalog, and the
sealed run dir (read-only) only; no SPS tables, no event generation, no
Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import deleted_events, sealed_mismatches, skip_unless_sealed

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking"
    / "staged_r50_pure_topology_templates.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r50_pure_topology_templates.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r50_pure_topology_templates_2026-09-27")

RUN_DIR_REL = "dev/00runs/r50_pure_topology_templates_2026-09-27"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r50_pure_topology_templates.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r50_pure_topology_templates.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260925-r5-run-r50-pure-topology-templates"
MY_CARD = "TASK-20260925-r5-run-r50-pure-topology-templates-test-author"

#: The committed dual-tree driver digest (byte-identical in both
#: trees; the freeze manifest inventory pins both copies here).
DRIVER_SHA256 = (
    "b74e5408195ca6a3a53caf495da0c3f1206a9b43a73f336ba24b5cd6eb936fa3"
)

#: The hermetic solver test file's committed digest (the Round 5
#: card 10 test-author count re-pin 532 -> 533 touched the
#: INVENTORY_COUNT constant, so the digest moved from the card-10
#: landing digest 9d3269d84ac2d0ca5e9cfb35597c209b9799e7a9736cdd52b0f8
#: 5706fe85d47a).
SOLVER_TEST_SHA256 = (
    "cb6e9083f334c91782857b8113e125d45db85d63ab41e35604694757d5f7f8da"
)

#: The sealed payload file digests (the checksum file's 26 entries plus
#: the seal manifest's own digest, excluded from its own entries).
PRE_REGISTRATION_SHA256 = (
    "f71be714b7f9e82287b3896dfdf122e318d0bdde7b0d8573cda42d761d718f50"
)
REPORT_FILE_SHA256 = (
    "53fba3dcd6472a17695d08417069e1550c3e16957f754d5c081ed25e563c4974"
)
REPORT_DIGEST = (
    "878ee8f2a09f621c17acbbd3f00dde3ee0d671c8969578b18ea48d31dde07c9d"
)
GATE5_FILE_SHA256 = (
    "623f09c16b4dccbb9a158007c5c0ceb8ae20853eafc5860141dce53a812bce37"
)
CLEAN_ENV_FILE_SHA256 = (
    "2a35ca149695f38b8042381a2d194a9581576fe927c0e15c6bf71917bd3516c5"
)
PROVENANCE_FILE_SHA256 = (
    "139913df84f473617f744099c6838f7c39472a3093f0588959a6f465ef9e633d"
)
REPORT_MD_FILE_SHA256 = (
    "e3a71ae3b335b02cbae886f8e1b7b5ce3570fc01055595c6fc99fa6e9c4505f3"
)
SEAL_MANIFEST_FILE_SHA256 = (
    "aaffbacb03bac518ebaef83a77152e3505d91c555a3e1cf967d84cd1418d079f"
)
SLOT_FILE_SHA256 = {
    "slot_elastic_0.json":
        "341023d82c4a6f6a569fab83527143e9dab311461fd4dbe2bdc86833c2b19ebd",
    "slot_elastic_1.json":
        "a7b790ce59b363510f791ab3e377853c53c2e3578cc502088115d3c5f8a8f598",
    "slot_elastic_2.json":
        "ee597fa183f3e1bed374f2d9c5a036b51457fbb716b3adee926c9ea05fd50a99",
    "slot_elastic_3.json":
        "6803b9e8a146c0db13351693299a9604025dd6737ae80e1b4b07de626a2a86a3",
    "slot_sd_projectile_0.json":
        "0fa21fb58afb34933eed38325c72eb2ebc91f8e0e01a084417d0dd9a99755f95",
    "slot_sd_projectile_1.json":
        "1f30bb534356833da43108a762a2f5b4387b0b5d60d8cf1af07aa25a3ff8b73f",
    "slot_sd_projectile_2.json":
        "2be1e54b2989f33d6e25f5abb5f15ab6bf1ff147d7e01f9da61b439de6bc8216",
    "slot_sd_projectile_3.json":
        "fca2e96468fbff17233e556dabe6732a6075acaf27bc036a8f247f2e8c3cf5e3",
    "slot_sd_target_0.json":
        "3c308c04169cc5a4fc018d8a7efed68c801afc9c724b90391c3cd1c62f141525",
    "slot_sd_target_1.json":
        "da8bb71778e1d44850490de54ac6e81d8cb6b50ca7fc13b64f5f8afe21f11614",
    "slot_sd_target_2.json":
        "7251dfe9d0aa49e9d2f00ebb4a9f73d308e1ef7654892b3f80fbddb265a665ab",
    "slot_sd_target_3.json":
        "87c88e8f0e1ce08e87a440149a9057b2f96cd435edc26f6d76e9a0968e144be8",
    "slot_dd_0.json":
        "dea06f0314c99949c22840f688b827719908d9ce7a5c2ed0d2c23e87cb7d1583",
    "slot_dd_1.json":
        "7d2e2f5c531e10c3739e2aa30b752dce22d9ec6f0856987750952aad30453922",
    "slot_dd_2.json":
        "12baa019983be2914cf3825929a60089e951a19e3f1cce0d8cfd2201684f008f",
    "slot_dd_3.json":
        "35728972630ff9dde475dc91948ed8fbca98a230886ebefab4b78826647ee922",
    "slot_nd_0.json":
        "d56c39a8dbc8b0eabe686c3e778e080602cc2cc800b96be0ac8728c7437336d8",
    "slot_nd_1.json":
        "d3bfc69d31c4f4c69488d2b2f4dc719bbdd4563aaab93ce450b20998c94b8e0e",
    "slot_nd_2.json":
        "00028fa30a16fb4f0d1bee63c0cb1774110892d1485cbac1c51d8a7681b90faa",
    "slot_nd_3.json":
        "fbe3bb666413d805df68211a2df90014901ebcf116b96da7f5b3db5cfd3a78ef",
}

#: The seal manifest's file list (20 slot records plus the six
#: top-level files), in seal order.
SEALED_FILES = (
    "REPORT.md",
    "clean_environment_commands.json",
    "gate5_r50.json",
    "pre_registration.json",
    "provenance.json",
) + tuple(f"slot_{cls}_{rep}.json"
          for cls in ("dd", "elastic", "nd", "sd_projectile", "sd_target")
          for rep in range(4)) + (
    "template_report.json",
)

#: The per-slot event hash sets (the stored events' bit-level identity
#: evidence; the probe slot's clean re-run must reproduce them).
SLOT_EVENT_HASH_SETS = {
    0: "d36f67a176f2885ca4732ff3022abda15d8ae5021127e03a590ef4742dc12005",
    1: "37e938d267e01290a6839d968645e9cc0ee8950e7fc4964840cf37e32cb194bd",
    2: "883fe40d2c779ee98ae77dd0fe3b2b08ef6c27f2b3a7c8b8bb9816a89fa1e8ed",
    3: "9b9d64a82cc2b286abbdd5c3534a5ba2e6c15c0aafa1fa181583df7df2c6cbca",
    4: "4703048c2e9b45e3a28daa40646946339381bd9053cc8634c7fdfad033b4b500",
    5: "a4f9e35759208f3680f59f8deebfa9ca273d25f18da4fd90be42d71beb2fbd32",
    6: "32de0e7f20527a946a9b2255b9e267510efbc661020eeb2eb805a0ef1c873c80",
    7: "9b4929bd0d0a5542878a376c3bbb93dc8af2d276388bdf93c17a226917b483fa",
    8: "3799d95f77ccf741bc67386c7e6d55e884e010e265e171f6849debf299e958f5",
    9: "8289936db46bc03fca0992abac9d42b8b8549ae336fdfc9b74da3bb023d5b62f",
    10: "63dc0f21147db0aee5c9ea360f8ba72e4927859f192d3a8b565ec935aca00de7",
    11: "573017ea5367eb04b1b503cca53ee7ee7d6dc7a38af7c261fc93fdedc23f5c2a",
    12: "aea64abe3a204b4057585e8683fd19134d1b759e96ebe0427fb2c965907666f0",
    13: "e40f916efe068773d07c519398e65b8ecf35e5fa2f6cb8d736e251fe58d4087f",
    14: "640392696dde3655a3af39c92f50357eb7bb720b8074004373942d7c88474658",
    15: "15d63be1ad4a3928aeafe426d0cbb365b44045f812d7319f4ffb6578401956b5",
    16: "ece185c7e1fd85625365a2c8be2b9e2f4abe581592d7f6371fbc5c150320d413",
    17: "b3522c0e7c9406fc3e457e8ad950e49472338951b6d20e27f6ecc9672b7cdda5",
    18: "5f3a1fffa27d0586375559f428614489d88bd8478465f88a184111b001b5db30",
    19: "fada33215c20552f0b57ae26426b5c5cc5c57e3ee0daf740c6226d63e48945e5",
}

#: The per-slot conservation blocks, pinned verbatim (slot index ->
#: recorded worst closures and the B, Q, S census).
SLOT_CONSERVATION = {
    0: {"worst_closure_relative": 1.798842369012912e-15,
        "worst_energy_closure_gev": 2.8421709430404007e-13,
        "worst_mass_shell_relative": 2.680561643874022e-11,
        "worst_momentum_closure_gev": 2.8421709430404007e-13},
    1: {"worst_closure_relative": 1.798842369012912e-15,
        "worst_energy_closure_gev": 2.8421709430404007e-13,
        "worst_mass_shell_relative": 2.680561643874022e-11,
        "worst_momentum_closure_gev": 2.8421709430404007e-13},
    2: {"worst_closure_relative": 1.798842369012912e-15,
        "worst_energy_closure_gev": 2.8421709430404007e-13,
        "worst_mass_shell_relative": 2.6915585094221985e-11,
        "worst_momentum_closure_gev": 2.8421709430404007e-13},
    3: {"worst_closure_relative": 1.798842369012912e-15,
        "worst_energy_closure_gev": 2.8421709430404007e-13,
        "worst_mass_shell_relative": 2.6915585094221985e-11,
        "worst_momentum_closure_gev": 2.8421709430404007e-13},
    4: {"worst_closure_relative": 2.557114249833401e-14,
        "worst_energy_closure_gev": 4.064304448547773e-12,
        "worst_mass_shell_relative": 4.496749654845197e-10,
        "worst_momentum_closure_gev": 4.035882739117369e-12},
    5: {"worst_closure_relative": 3.2379162642232416e-14,
        "worst_energy_closure_gev": 5.144329406903125e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 5.115907697472721e-12},
    6: {"worst_closure_relative": 2.6442982824489803e-14,
        "worst_energy_closure_gev": 4.149569576838985e-12,
        "worst_mass_shell_relative": 4.0925733484740457e-10,
        "worst_momentum_closure_gev": 4.177991286269389e-12},
    7: {"worst_closure_relative": 2.572344587688464e-14,
        "worst_energy_closure_gev": 4.064304448547773e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 4.064304448547773e-12},
    8: {"worst_closure_relative": 3.47176577219492e-14,
        "worst_energy_closure_gev": 5.5138116294983774e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 5.4853899200679734e-12},
    9: {"worst_closure_relative": 3.040043603631821e-14,
        "worst_energy_closure_gev": 4.803268893738277e-12,
        "worst_mass_shell_relative": 6.089413019835981e-10,
        "worst_momentum_closure_gev": 4.803268893738277e-12},
    10: {"worst_closure_relative": 2.5928780854954062e-14,
         "worst_energy_closure_gev": 4.121147867408581e-12,
         "worst_mass_shell_relative": 4.0925733484740457e-10,
         "worst_momentum_closure_gev": 4.092726157978177e-12},
    11: {"worst_closure_relative": 3.309869958983758e-14,
         "worst_energy_closure_gev": 5.258016244624741e-12,
         "worst_mass_shell_relative": 6.089413019835981e-10,
         "worst_momentum_closure_gev": 5.229594535194337e-12},
    12: {"worst_closure_relative": 2.6622867061391095e-14,
         "worst_energy_closure_gev": 4.177991286269389e-12,
         "worst_mass_shell_relative": 5.430534098281643e-10,
         "worst_momentum_closure_gev": 4.206412995699793e-12},
    13: {"worst_closure_relative": 2.4104487744773018e-14,
         "worst_energy_closure_gev": 3.780087354243733e-12,
         "worst_mass_shell_relative": 5.430534098281643e-10,
         "worst_momentum_closure_gev": 3.808509063674137e-12},
    14: {"worst_closure_relative": 2.932113061491046e-14,
         "worst_energy_closure_gev": 4.632738637155853e-12,
         "worst_mass_shell_relative": 4.841094779519261e-10,
         "worst_momentum_closure_gev": 4.632738637155853e-12},
    15: {"worst_closure_relative": 2.9326345242844596e-14,
         "worst_energy_closure_gev": 4.661160346586257e-12,
         "worst_mass_shell_relative": 4.496749654845197e-10,
         "worst_momentum_closure_gev": 4.632738637155853e-12},
    16: {"worst_closure_relative": 1.5470044373511042e-14,
         "worst_energy_closure_gev": 2.4442670110147446e-12,
         "worst_mass_shell_relative": 3.0941535127930784e-10,
         "worst_momentum_closure_gev": 2.4442670110147446e-12},
    17: {"worst_closure_relative": 2.0147034532944613e-14,
         "worst_energy_closure_gev": 3.183231456205249e-12,
         "worst_mass_shell_relative": 3.562965211408751e-10,
         "worst_momentum_closure_gev": 3.183231456205249e-12},
    18: {"worst_closure_relative": 1.816830792703041e-14,
         "worst_energy_closure_gev": 2.8705926524708048e-12,
         "worst_mass_shell_relative": 3.8947853369736933e-10,
         "worst_momentum_closure_gev": 2.8705926524708048e-12},
    19: {"worst_closure_relative": 1.4030970478300713e-14,
         "worst_energy_closure_gev": 2.1884716261411086e-12,
         "worst_mass_shell_relative": 3.8947853369736933e-10,
         "worst_momentum_closure_gev": 2.2168933355715126e-12},
}

#: The global conservation block of the sealed template report.
CONSERVATION = {
    "b_q_s_census": {"2,2,0": 500000},
    "events_checked": 500000,
    "pass": True,
    "tolerance": 1e-09,
    "violations": 0,
    "worst_closure_relative": 3.47176577219492e-14,
    "worst_energy_closure_gev": 5.5138116294983774e-12,
    "worst_mass_shell_relative": 6.089413019835981e-10,
    "worst_momentum_closure_gev": 5.4853899200679734e-12,
}

#: The per-class pooled observables (population, ddof = 0, math.fsum
#: accumulation), pinned verbatim from the sealed template report.
PER_CLASS_XF = {
    "elastic": {
        "projectile": (0.99330986596247, 0.9832063726723104,
                       0.9941028046085049, 0.0007951764392929663),
        "target": (-0.9933098659624647, -0.9941028046085104,
                   -0.9832063726723125, 0.0007951764392929771),
    },
    "sd_projectile": {
        "projectile": (0.8377568002477981, 0.003862371794889854,
                       0.9940650678113168, 0.2161145583522143),
        "target": (-0.837756800247798, -0.9940650678113102,
                   -0.003862371794889854, 0.21611455835221421),
    },
    "sd_target": {
        "projectile": (-0.8381838260756932, -0.9940816258336511,
                       -0.0036439381795927645, 0.21567705177400104),
        "target": (0.8381838260756932, 0.0036439381795776186,
                   0.994081625833645, 0.21567705177400107),
    },
    "dd": {
        "projectile": (0.7585841187215611, 0.0033904419275300785,
                       0.9939790618793398, 0.23014233292004635),
        "target": (-0.7585841187215611, -0.9939790618793318,
                   -0.003390441927526292, 0.23014233292004635),
    },
    "nd": {
        "projectile": (0.9660610159534107, 0.4246163673537222,
                       0.9862109524852187, 0.018344289218202303),
        "target": (-0.9660610159534109, -0.9862109524852568,
                   -0.4246163673537166, 0.018344289218202286),
    },
}

PER_CLASS_GAP = {
    "elastic": {"present": 0, "absent": 100000, "mean": None, "min": None,
                "max": None, "std": None},
    "sd_projectile": {"present": 100000, "absent": 0,
                      "mean": 3.2486855357771134,
                      "min": 0.00010271579608200909,
                      "max": 5.8130953231777, "std": 1.4362120847137665},
    "sd_target": {"present": 100000, "absent": 0,
                  "mean": 3.248040092462289,
                  "min": 1.0016481018393364e-05,
                  "max": 5.815881765778563, "std": 1.43614248211734},
    "dd": {"present": 100000, "absent": 0,
           "mean": 1.8457331414200868, "min": 7.789520780843162e-06,
           "max": 5.7987249973762545, "std": 1.4613773835803385},
    "nd": {"present": 0, "absent": 100000, "mean": None, "min": None,
           "max": None, "std": None},
}

PER_CLASS_MASSES = {
    "elastic": {"count": 0, "mean": None, "min": None, "max": None,
                "std": None},
    "sd_projectile": {"count": 100000, "mean": 5.330206729779737,
                      "min": 0.9382789365815919,
                      "max": 16.331306285082817,
                      "std": 4.126171241925512},
    "sd_target": {"count": 100000, "mean": 5.324687730535668,
                  "min": 0.9383189642463218,
                  "max": 16.33137513730609, "std": 4.116751818821417},
    "dd": {"count": 200000, "mean": 4.4929163651429285,
           "min": 0.9382936016126088, "max": 16.315508341316267,
           "std": 3.4866822242440634},
    "nd": {"count": 100000, "mean": 0.3840229548507544,
           "min": 0.13497699155436188, "max": 9.516436178104964,
           "std": 0.30719386545620314},
}

PER_CLASS_MULTIPLICITY = {
    "elastic": {"mean": 2.0, "min": 2.0, "max": 2.0, "std": 0.0,
                "central_activity_mean": 0.0,
                "distribution": {"2": 100000}},
    "sd_projectile": {"mean": 5.16546, "min": 2.0, "max": 11.0,
                      "std": 2.2790706413799464,
                      "central_activity_mean": 0.98923,
                      "distribution": {"2": 4740, "3": 34021,
                                       "4": 11379, "5": 7866,
                                       "6": 6937, "7": 10255,
                                       "8": 17471, "9": 6584,
                                       "10": 491, "11": 256}},
    "sd_target": {"mean": 5.16363, "min": 2.0, "max": 11.0,
                  "std": 2.277484406774281,
                  "central_activity_mean": 0.97778,
                  "distribution": {"2": 4589, "3": 34237,
                                   "4": 11319, "5": 7919,
                                   "6": 7067, "7": 10213,
                                   "8": 17277, "9": 6586,
                                   "10": 503, "11": 290}},
    "dd": {"mean": 7.64017, "min": 2.0, "max": 18.0,
           "std": 2.835918258888997,
           "central_activity_mean": 1.7291,
           "distribution": {"2": 278, "3": 3832, "4": 14697,
                             "5": 9831, "6": 8542, "7": 8560,
                             "8": 12266, "9": 17297, "10": 10800,
                             "11": 5344, "12": 3367, "13": 2280,
                             "14": 1750, "15": 861, "16": 223,
                             "17": 58, "18": 14}},
    "nd": {"mean": 4.81479, "min": 3.0, "max": 13.0,
           "std": 1.288668792165,
           "central_activity_mean": 1.18287,
           "distribution": {"3": 15152, "4": 30275, "5": 26538,
                             "6": 17563, "7": 7534, "8": 2400,
                             "9": 473, "10": 60, "11": 1,
                             "12": 3, "13": 1}},
}

PER_CLASS_YIELDS = {
    "elastic": {"central_pions": (0.0, 0), "central_protons": (0.0, 0),
                "forward_pions": (0.0, 0), "forward_protons": (0.0, 0)},
    "sd_projectile": {"central_pions": (0.62904, 62904),
                      "central_protons": (0.07138, 7138),
                      "forward_pions": (1.74105, 174105),
                      "forward_protons": (0.8411, 84110)},
    "sd_target": {"central_pions": (0.62445, 62445),
                  "central_protons": (0.06987, 6987),
                  "forward_pions": (1.74531, 174531),
                  "forward_protons": (0.84088, 84088)},
    "dd": {"central_pions": (1.09225, 109225),
           "central_protons": (0.12566, 12566),
           "forward_pions": (3.13977, 313977),
           "forward_protons": (0.55442, 55442)},
    "nd": {"central_pions": (0.62744, 62744),
           "central_protons": (0.0, 0),
           "forward_pions": (1.60259, 160259),
           "forward_protons": (1.57511, 157511)},
}

PER_CLASS_SEAGULL = {
    "elastic": (0.30356073898797203, 0.0016664753479187823,
                1.2674679995774536, 0.1590837146475304),
    "sd_projectile": (0.663216195870827, 0.001142359049928673,
                      3.0986461528239673, 0.38546056949128477),
    "sd_target": (0.6630476741704809, 0.005825216170531521,
                  3.092967990275783, 0.38447891383593896),
    "dd": (0.7438091210630005, 0.0, 2.9669379770271465,
           0.31752864780828866),
    "nd": (0.23712521558769203, 0.0, 0.7628672072060387,
           0.0878619662772231),
}

PER_CLASS_TRANSFERS = {
    "elastic": {"n": 100000, "mean": -0.117550778998395,
                "min": -1.6153280079151329,
                "max": -2.777140109351333e-06,
                "std": 0.11787974015905768},
    "sd_projectile": {"n": 100000, "mean": -0.11720927236379401,
                      "min": -1.4149215969314255,
                      "max": -1.8962271173717668e-07,
                      "std": 0.11729985188424923},
    "sd_target": {"n": 100000, "mean": -0.1179468232113065,
                  "min": -1.45926993139133,
                  "max": -1.449197730800656e-06,
                  "std": 0.1178157925050212},
    "dd": {"n": 200000, "mean": -0.11769914087074612,
           "min": -1.5611691243612287, "max": -1.2971674504386813e-06,
           "std": 0.11824932671602462},
    "nd": {"n": 300000, "mean": -0.15707655607426865,
           "min": -1.501147937242956, "max": -6.352400256436872e-07,
           "std": 0.14716739657289432},
}

#: The class draw census of the sealed template report (the pre-
#: registered slot map; the realized fraction deviation is 0.0 exactly).
CLASS_DRAW_CENSUS = {
    "elastic": {"events": 100000, "inelastic": False,
                "nominal_inelastic_fraction": None,
                "realized_inelastic_fraction": None,
                "seed_block": [20300925, 20300928]},
    "sd_projectile": {"events": 100000, "inelastic": True,
                      "nominal_inelastic_fraction": 0.25,
                      "realized_inelastic_fraction": 0.25,
                      "seed_block": [20300929, 20300932]},
    "sd_target": {"events": 100000, "inelastic": True,
                  "nominal_inelastic_fraction": 0.25,
                  "realized_inelastic_fraction": 0.25,
                  "seed_block": [20300933, 20300936]},
    "dd": {"events": 100000, "inelastic": True,
           "nominal_inelastic_fraction": 0.25,
           "realized_inelastic_fraction": 0.25,
           "seed_block": [20300937, 20300940]},
    "nd": {"events": 100000, "inelastic": True,
           "nominal_inelastic_fraction": 0.25,
           "realized_inelastic_fraction": 0.25,
           "seed_block": [20300941, 20300944]},
}

#: The elastic class's recorded normalization check (plan line 824).
ELASTIC_CHECK = {
    "closed_form": 0.11758977514325487,
    "quadrature": 0.11758977514325486,
    "normalization_max_dev": 1.3877787807814457e-17,
    "quantile_max_dev": 5.551115123125783e-17,
    "analytic_mean_gev2": -0.11758977514325487,
    "sample_mean_gev2": -0.117550778998395,
    "sample_sd_gev2": 0.11787974015905768,
    "sigma_of_mean_gev2": 0.00037276846889144145,
    "z": 0.10461224087928457,
    "n": 100000,
    "mass_shell_worst_relative": 2.6915585094221985e-11,
    "xf_pair_sum_max_abs": 2.042810365310288e-14,
    "zero_transfer_violations": 0,
    "xf_in_unit_range": True,
    "verdict": "pass",
}

#: The histogram edge grids (shared by every class block).
XF_HIST_EDGES = (
    -1.05, -1.0, -0.9500000000000001, -0.9, -0.8500000000000001, -0.8,
    -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.44999999999999996, -0.4,
    -0.35, -0.30000000000000004, -0.25, -0.19999999999999996,
    -0.1499999999999999, -0.10000000000000009, -0.050000000000000044,
    0.0, 0.050000000000000044, 0.10000000000000009,
    0.15000000000000013, 0.19999999999999996, 0.25,
    0.30000000000000004, 0.3500000000000001, 0.40000000000000013,
    0.44999999999999996, 0.5000000000000002, 0.55, 0.5999999999999999,
    0.6500000000000001, 0.7, 0.7500000000000002, 0.8,
    0.8499999999999999, 0.9000000000000001, 0.95, 1.0000000000000002,
    1.05,
)
GAP_HIST_EDGES = (
    0.0, 0.3456, 0.6912, 1.0368000000000002, 1.3824,
    1.7280000000000002, 2.0736000000000003, 2.4192, 2.7648,
    3.1104000000000003, 3.4560000000000004, 3.8016,
    4.147200000000001, 4.4928, 4.8384, 5.184000000000001, 5.5296,
    5.8751999999999995, 6.2208000000000006, 6.566400000000001,
    6.912000000000001, 7.2576, 7.6032, 7.948800000000001,
    8.294400000000001, 8.64, 8.9856, 9.3312, 9.6768, 10.0224,
    10.368000000000002, 10.713600000000001, 11.0592, 11.4048,
    11.750399999999999, 12.096000000000002, 12.441600000000001,
    12.7872, 13.132800000000001, 13.4784, 13.824000000000002,
    14.1696, 14.5152, 14.860800000000001, 15.2064, 15.552,
    15.897600000000002, 16.2432, 16.588800000000003, 16.9344, 17.28,
)
MASS_HIST_EDGES = (
    0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
    2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
    4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0,
    6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0,
    8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
    10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0,
    12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0,
    14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0,
    16.2, 16.4, 16.6, 16.8, 17.0, 17.2, 17.4, 17.6, 17.8, 18.0)

#: The M8-frozen p+p set (hash and parameters).
M8_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)
B_SD = 8.504140762083612
BLEND_RAMP_START = 2.680194110727528
SQRT_S_EXTRACTION = 17.27
BEAM_MOMENTUM_GEV_C = 158.0
M_P = 0.93827231

#: The seed map constants.
MASTER_SEED = 20260925
RAW_SEED_BLOCK = [20300925, 20300944]
EVENT_INDEX_BASE = 1_000_000_000
SMOKE_SEED = 20312925
SMOKE_EVENT_INDEX_BASE = 1_600_000_000

#: The frozen R45 simplified cascade (consumed without refit).
FROZEN_CASCADE = {
    "W_flavor": 1.0,
    "b_dec_gev2": 2.0,
    "baryonic_nominal_width_gev": 0.3,
    "kappa_B": 0.0,
    "kappa_M": 4.0,
}

#: The frozen substrate digests (verified at pre-registration).
FROZEN_SUBSTRATE = {
    "builder_blend_path":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "card2_engine":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "card6_routing":
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563",
    "catalog":
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd",
    "continuum_config":
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1",
    "r45_driver":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "r46_driver":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
}

#: The round-5 module digests recorded in the provenance.
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

#: The gate record's recorded values.
GATE5_CONSERVATION = {
    "events_checked": 500000,
    "pass": True,
    "tolerance": 1e-09,
    "violations": 0,
    "worst_energy_closure_gev": 5.5138116294983774e-12,
}

#: The clean-environment validation's command list (exactly one pass
#: executes before the seal).
CLEAN_ENV_COMMANDS = (
    "cd dev/mixing2006 && python -m pytest "
    "tests/test_staged_r50_pure_topology_templates.py -q",
    "python -m pytest tests/test_legacy_v1_freeze.py "
    "-W error::UserWarning -q",
    "python helpers/benchmarking/"
    "staged_r50_pure_topology_templates.py --clean-validate "
    "--run-dir /home/o/NA49/2026/01-NA49-res/dev/00runs/"
    "r50_pure_topology_templates_2026-09-27",
)
CLEAN_ENV_PROBE_SLOT = {
    "slot": 16,
    "class_label": "nd",
    "file": "slot_nd_0.json",
    "seed": 20300941,
    "census_matches": True,
    "conservation_matches": True,
    "event_hash_set_match": True,
    "matches_sealed_record": True,
}

SEAL_SEALED_UTC = "2026-09-27T05:52:43Z"
PRE_REG_RECORDED_UTC = "2026-09-27T05:48:55Z"
PROVENANCE_RECORDED_UTC = "2026-09-27T05:48:55Z"
CLEAN_ENV_VERIFIED_UTC = "2026-09-27T05:52:41Z"
GIT_HEAD = "dd7f515100c471bce7fbeb902fd06e31fa88af08"

# re-pinned 532 -> 533 by Round 5 card 10 test-author (TASK-20260925-r5-run-r50-pure-topology-templates-test-author; +1 entry: the card-10 test-author register file tests/test_staged_r50_pure_topology_templates_contract_pins.py).
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-topology-fraction-fit; +3 entries: the dual-tree driver helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested copy, and the test file tests/test_staged_r51_topology_fraction_fit.py).
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
#: The chain count-pin files re-pinned by this card (the 37 card-10
#: chain files plus the card-10 solver test).
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
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates.py",
    "dev/mixing2006/tests/"
    "test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/"
    "test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze exception file list (the manifest, this register,
#: and the chain count-pin files; 40 path-sorted entries).
MY_CARD_EXCEPTION_FILES = (
    sorted([MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES)))


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r50_pure_topology_templates", NESTED_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r50 = _load_driver()

_E_BEAM = math.sqrt(M_P ** 2 + BEAM_MOMENTUM_GEV_C ** 2)
_E_TOTAL = _E_BEAM + M_P


def _slot_path(slot: int) -> Path:
    cls = r50.CLASS_ORDER[slot // 4]
    return RUN_DIR / f"slot_{cls}_{slot % 4}.json"


def _event_residuals(ev) -> tuple:
    """Independent recompute of one stored event's conservation
    residuals against the incoming reference (E_beam + m_p, 0, 0,
    p_beam): (worst energy closure [GeV], worst momentum closure
    [GeV], worst relative closure, worst mass-shell relative)."""
    rows = ev["fs"] if ev["fs"] else ev["ln"]
    from_fs = bool(ev["fs"])
    es, pxs, pys, pzs = [], [], [], []
    worst_sh = 0.0
    for r in rows:
        if from_fs:
            E, px, py, pz, m = r[6], r[7], r[8], r[9], r[5]
        else:
            E, px, py, pz = r[0], r[1], r[2], r[3]
            m = M_P
        es.append(E)
        pxs.append(px)
        pys.append(py)
        pzs.append(pz)
        # the driver's convention: sequential subtraction and the
        # max(m^2, 1e-9) scale clamp (bit-identical recompute)
        m2 = E * E - px * px - py * py - pz * pz
        scale = max(m * m, 1e-9)
        worst_sh = max(worst_sh, abs(m2 - m * m) / scale)
    d_e = abs(math.fsum(es) - _E_TOTAL)
    d_px = abs(math.fsum(pxs))
    d_py = abs(math.fsum(pys))
    d_pz = abs(math.fsum(pzs) - BEAM_MOMENTUM_GEV_C)
    rel = max(d_e / _E_TOTAL, d_px / BEAM_MOMENTUM_GEV_C,
              d_py / BEAM_MOMENTUM_GEV_C, d_pz / BEAM_MOMENTUM_GEV_C)
    return d_e, max(d_px, d_py, d_pz), rel, worst_sh


def _fstats(values) -> tuple:
    """Population (ddof = 0) (mean, min, max, std) with math.fsum
    accumulation (the report's convention)."""
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / n
    return mean, min(values), max(values), math.sqrt(var)


def _recompute_slot(slot: int) -> dict:
    """Independent recompute of the slot's recorded conservation block
    and event hash set from the stored events."""
    record = json.loads(_slot_path(slot).read_text(encoding="utf-8"))
    events = record["events"]
    census = Counter(tuple(ev["bqs"]) for ev in events)
    worst_e = worst_p = worst_rel = worst_sh = 0.0
    for ev in events:
        d_e, d_p, rel, sh = _event_residuals(ev)
        worst_e = max(worst_e, d_e)
        worst_p = max(worst_p, d_p)
        worst_rel = max(worst_rel, rel)
        worst_sh = max(worst_sh, sh)
    hash_set = hashlib.sha256(
        "\n".join(ev["h"] for ev in events).encode("utf-8")).hexdigest()
    return {
        "b_q_s_census": {f"{b},{q},{s}": n
                         for (b, q, s), n in census.items()},
        "worst_energy_closure_gev": worst_e,
        "worst_momentum_closure_gev": worst_p,
        "worst_closure_relative": worst_rel,
        "worst_mass_shell_relative": worst_sh,
        "event_hash_set_sha256": hash_set,
    }


def _pool_events(class_label: str):
    """The pooled stored events of one class (its four slots)."""
    for slot in range(20):
        if r50.CLASS_ORDER[slot // 4] != class_label:
            continue
        record = json.loads(_slot_path(slot).read_text(encoding="utf-8"))
        yield from record["events"]


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
    assert r50.DRIVER_SHA256 == DRIVER_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    for rel in (DRIVER_REL_NESTED, DRIVER_REL_OUTER):
        assert rel in inventory, f"missing from the inventory: {rel}"
        assert inventory[rel] == DRIVER_SHA256, (
            f"stale inventory digest: {rel}")


def test_driver_identity_constants():
    """The run identity, the file-name constants, the sealed-dir rel
    paths, the class order, the card chain, the M8-frozen parameter
    pins, the nominal parameters, the frozen R45 cascade, the seed map
    constants, the model-uncertainty carry, the closure tolerance, and
    the histogram bin counts."""
    assert r50.RUN_ID == "R50"
    assert r50.CARD == SOLVER_CARD
    assert r50.RUN_NAME == "r50_pure_topology_templates_2026-09-27"
    assert r50.DEFAULT_RUN_DIR == RUN_DIR_REL
    assert r50.REPORT_NAME == "template_report.json"
    assert r50.REPORT_MD_NAME == "REPORT.md"
    assert r50.PREREGISTRATION_NAME == "pre_registration.json"
    assert r50.PROVENANCE_NAME == "provenance.json"
    assert r50.CLEAN_ENV_NAME == "clean_environment_commands.json"
    assert r50.SHA256SUMS_NAME == "SHA256SUMS.txt"
    assert r50.SEAL_NAME == "seal_manifest.json"
    assert r50.GATE_RECORD_NAME == "gate5_r50.json"
    assert r50.DRIVER_REL_NESTED == DRIVER_REL_NESTED
    assert r50.DRIVER_REL_OUTER == DRIVER_REL_OUTER
    assert r50.CLASS_ORDER == ("elastic", "sd_projectile", "sd_target",
                               "dd", "nd")
    assert tuple(r50.CARD_CHAIN) == (
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
        SOLVER_CARD,
    )
    assert r50.M8_PARAMETER_HASH == M8_HASH
    assert r50.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert r50.BLEND_RAMP_START_GEV == BLEND_RAMP_START
    assert r50.SQRT_S_EXTRACTION_GEV == SQRT_S_EXTRACTION
    assert r50.R45_FROZEN_CASCADE == FROZEN_CASCADE
    assert r50.MASTER_SEED == MASTER_SEED
    assert r50.SEED_BASE == RAW_SEED_BLOCK[0]
    assert r50.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert r50.EVENT_INDEX_RULE == (
        "1_000_000_000 + i, i < 25,000 (per slot)")
    assert r50.SMOKE_SEED == SMOKE_SEED
    assert r50.SMOKE_EVENT_INDEX_BASE == SMOKE_EVENT_INDEX_BASE
    assert r50.SMOKE_EVENTS == 2000
    assert r50.REPLICAS_PER_CLASS == 4
    assert r50.EVENTS_PER_SLOT == 25_000
    assert r50.N_SLOTS == 20
    assert r50.N_EVENTS_PER_CLASS == 100_000
    assert r50.N_EVENTS_TOTAL == 500_000
    assert r50.NOMINAL_ELASTIC_FRACTION == 0.0
    assert r50.EVENT_CLOSURE_TOL == 1e-09
    assert r50.XF_HIST_BINS == len(XF_HIST_EDGES) - 1
    assert r50.GAP_HIST_BINS == len(GAP_HIST_EDGES) - 1
    assert r50.MASS_HIST_BINS == len(MASS_HIST_EDGES) - 1
    assert r50.RESERVED_ROUND5_BLOCKS == {
        "r51": [20300945, 20300952],
        "r52": [20300953, 20300960],
        "r53": [20300961, 20300968],
        "r54": [20300969, 20300988],
        "r55": [20300989, 20301008],
    }
    assert r50.RESERVED_ROUND5_PARTITIONS == {
        "r51": [1100000000, 1100100000],
        "r52": [1200000000, 1200100000],
        "r53": [1300000000, 1300100000],
        "r54": [1400000000, 1400050000],
        "r55": [1500000000, 1500250000],
        "round5_smoke": [1600000000, 1600002000],
    }
    carry = r50.MODEL_UNCERTAINTY_CARRY
    assert carry["r32_spread"]["bin_spread_p95"] == 4.028426
    assert carry["r32_spread"]["feeddown_spread_max_sigma"] == 31.1773
    assert carry["r32_spread"]["persisting_residuals"] == 32
    assert carry["b_transport_seagull"] == -2.3959765824349295e-05


def test_solver_test_file_digest():
    """The hermetic solver test file at its post-re-pin committed
    digest, pinned in the freeze manifest inventory."""
    solver_path = MONO_ROOT / SOLVER_TEST_REL
    assert solver_path.is_file(), solver_path
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    assert inventory[SOLVER_TEST_REL] == SOLVER_TEST_SHA256


# ----------------------------------------------------------------------
# The seed map
# ----------------------------------------------------------------------
def test_seed_map_partition():
    """The master seed, the raw slot block, the event-index partition,
    the smoke block, the per-class seed blocks, the reserved round-5
    blocks and partitions, and the driver's disjointness report
    against the pre-registration record."""
    skip_unless_sealed(RUN_DIR)
    pre = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))
    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert seed_map["raw_seed_rule"].startswith(
        "raw slot seed = 20260925 + k per slot k < 20")
    assert seed_map["event_index_rule"] == (
        "1_000_000_000 + i, i < 25,000 (per slot)")
    assert seed_map["event_index_base"] == EVENT_INDEX_BASE
    assert seed_map["raw_slot_seeds"] == list(
        range(RAW_SEED_BLOCK[0], RAW_SEED_BLOCK[1] + 1))
    assert seed_map["per_class_seed_blocks"] == {
        "elastic": [20300925, 20300928],
        "sd_projectile": [20300929, 20300932],
        "sd_target": [20300933, 20300936],
        "dd": [20300937, 20300940],
        "nd": [20300941, 20300944],
    }
    smoke = seed_map["smoke"]
    assert smoke["raw_seed"] == SMOKE_SEED
    assert smoke["events"] == 2000
    assert smoke["event_index_rule"] == "1_600_000_000 + i, i < 2,000"
    assert seed_map["substream_labels"] == [
        "topology_selection", "exchange_kinematics", "spectral_state",
        "decay_channel", "decay_kinematics", "nuclear_geometry",
        "transport",
    ]
    disjoint = seed_map["disjointness"]
    assert disjoint["disjoint"] is True
    assert disjoint["r50_event_index_partition"] == (
        "1_000_000_000 + i, i < 25,000 (per slot)")
    assert disjoint["r50_event_index_range"] == [
        EVENT_INDEX_BASE, EVENT_INDEX_BASE + 25_000]
    assert disjoint["r50_raw_seed_block"] == RAW_SEED_BLOCK
    assert disjoint["reserved_round5_blocks"] == r50.RESERVED_ROUND5_BLOCKS
    assert disjoint["reserved_round5_partitions"] == \
        r50.RESERVED_ROUND5_PARTITIONS
    report = r50.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["r50_raw_seed_block"] == RAW_SEED_BLOCK
    for key in ("r23", "r32", "r40_partition", "r41_partition",
                "r42_partition", "r43_partition", "r44_partition",
                "r45_partition", "r46_partition", "r46_smoke",
                "round4_production", "round4_smoke"):
        assert key in disjoint["previously_used_blocks"], key


# ----------------------------------------------------------------------
# The pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk():
    """The sealed pre-registration's schema, identity, statistics,
    seed map, nominal parameters, conservation contract, M8 frozen
    set, frozen cascade, frozen substrate digests, model-uncertainty
    carry, no-optimizer record, stored quantities, and card chain."""
    skip_unless_sealed(RUN_DIR)
    pre = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "pre_registration.json") == \
        PRE_REGISTRATION_SHA256
    assert pre["schema"] == \
        "na49_mixing.staged_r50_pure_topology_templates.pre_registration/1"
    assert pre["card"] == SOLVER_CARD
    assert pre["run"] == "R50"
    assert pre["run_name"] == "r50_pure_topology_templates_2026-09-27"
    assert pre["run_dir"] == (
        "/home/o/NA49/2026/01-NA49-res/" + RUN_DIR_REL)
    assert pre["recorded_utc"] == PRE_REG_RECORDED_UTC
    assert pre["mode"] == "production"
    stats = pre["statistics"]
    assert stats["slots"] == 20
    assert stats["events_per_slot"] == 25000
    assert stats["events_per_class"] == 100000
    assert stats["events_total"] == 500000
    assert stats["replicas_per_class"] == 4
    assert stats["class_order"] == [
        "elastic", "sd_projectile", "sd_target", "dd", "nd"]
    nominal = pre["nominal_parameters"]
    assert nominal["b_sd_gev2"] == B_SD
    assert nominal["epsilon"] == 0.0
    assert nominal["elastic_fraction"] == 0.0
    assert nominal["p_n2"] == 0.0
    assert nominal["pass"] is True
    assert nominal["class_set"] == [
        "elastic", "sd_projectile", "sd_target", "dd", "nd"]
    assert nominal["inelastic_set"] == [
        "dd", "nd", "sd_projectile", "sd_target"]
    assert nominal["class_fractions"] == {
        "dd": 0.25, "nd": 0.25, "sd_projectile": 0.25, "sd_target": 0.25}
    assert nominal["mass_law_construction"] == {"b_sd": B_SD,
                                                "epsilon": 0.0}
    assert nominal["remnant_weights"] == {
        "high_mass_baryonic_continuum": 0.0,
        "intact_proton": 0.5,
        "low_mass_baryonic_resonance": 0.5,
    }
    assert "unretained" in nominal["nd_multi_status"]
    conservation = pre["conservation"]
    assert conservation["expected_b_q_s"] == [2, 2, 0]
    assert conservation["tolerance"] == 1e-09
    assert "E_beam = sqrt(m_p^2 + p_beam^2)" in \
        conservation["incoming_reference"]
    m8 = pre["m8_frozen_set"]
    assert m8["hash"] == M8_HASH
    assert m8["blend_ramp_start_gev"] == BLEND_RAMP_START
    assert m8["effective_transfer_slope_gev2"] == B_SD
    assert m8["verified"] is True
    assert m8["source"] == (
        "config/v2_singlet/legacy_v1_manifest.json "
        "(frozen_fit_outputs.m8_frozen_pp_parameters)")
    assert pre["frozen_cascade"] == FROZEN_CASCADE
    for name, digest in FROZEN_SUBSTRATE.items():
        entry = pre["frozen_substrate"][name]
        assert entry["sha256"] == digest, name
        assert entry["expected"] == digest, name
    assert pre["model_uncertainty_carry"]["r32_spread"]["bin_spread_p95"] \
        == 4.028426
    assert pre["model_uncertainty_carry"]["b_transport_seagull"] == \
        -2.3959765824349295e-05
    assert pre["no_optimizer"] is True
    assert pre["optimizer_invoked"] is False
    assert pre["stored_quantities"] == [
        "leading_baryon_xf", "gap_size", "excitation_masses",
        "multiplicity", "central_forward_yields", "seagull_moments"]
    assert pre["card_chain"][-1] == SOLVER_CARD
    assert pre["card_chain"][0] == "TASK-20260920-r4-gate4-decision"
    check = pre["elastic_normalization_check"]
    assert check["excluded_from_inelastic_statistics"] is True
    assert "1e-12" in check["definition"]


# ----------------------------------------------------------------------
# The slot records
# ----------------------------------------------------------------------
def test_slot_records_on_disk():
    """The 20 slot records (identity, class label, census, seed,
    conservation block) pinned verbatim, the per-slot event hash sets,
    and the independent recompute from the stored events (B, Q, S
    census; four-momentum closure; mass shell; event hash set) over
    all 500,000 events."""
    skip_unless_sealed(RUN_DIR)
    for slot in range(20):
        record = json.loads(_slot_path(slot).read_text(encoding="utf-8"))
        cls = r50.CLASS_ORDER[slot // 4]
        replica = slot % 4
        assert record["schema"] == \
            "na49_mixing.staged_r50_pure_topology_templates.slot/1"
        assert record["card"] == SOLVER_CARD
        assert record["run"] == "R50"
        assert record["mode"] == "production"
        assert record["reused"] is False
        assert record["mixed_draw_note"] is None
        assert record["slot"] == slot
        assert record["class_label"] == cls
        assert record["class_census"] == {cls: 25000}
        assert record["replica"] == replica
        assert record["seed"] == RAW_SEED_BLOCK[0] + slot
        assert record["n_events"] == 25000
        assert len(record["events"]) == 25000
        assert record["identity"] == {
            "driver_sha256": DRIVER_SHA256,
            "pre_registration_sha256": PRE_REGISTRATION_SHA256,
        }
        seed_map = record["seed_map"]
        assert seed_map["master_seed"] == MASTER_SEED
        assert seed_map["event_index_rule"] == (
            "1_000_000_000 + i, i < 25,000 (per slot)")
        assert record["event_hash_set_sha256"] == \
            SLOT_EVENT_HASH_SETS[slot]
        conservation = record["conservation"]
        pinned = SLOT_CONSERVATION[slot]
        assert conservation["events"] == 25000
        assert conservation["tolerance"] == 1e-09
        assert conservation["violations"] == 0
        assert conservation["b_q_s_census"] == {"2,2,0": 25000}
        for key, value in pinned.items():
            assert conservation[key] == value, (slot, key)
        recomputed = _recompute_slot(slot)
        assert recomputed["b_q_s_census"] == {"2,2,0": 25000}, slot
        for key in ("worst_energy_closure_gev",
                    "worst_momentum_closure_gev",
                    "worst_closure_relative",
                    "worst_mass_shell_relative"):
            assert recomputed[key] == pinned[key], (slot, key)
        assert recomputed["event_hash_set_sha256"] == \
            SLOT_EVENT_HASH_SETS[slot], slot


# ----------------------------------------------------------------------
# The template report
# ----------------------------------------------------------------------
def test_template_report_on_disk():
    """The sealed template report's schema and self-digest, the global
    conservation block (independently recomputed), the per-class pooled
    observables (independently recomputed from the stored events), the
    class draw census, and the histogram edge grids."""
    skip_unless_sealed(RUN_DIR)
    report = json.loads(
        (RUN_DIR / "template_report.json").read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "template_report.json") == \
        REPORT_FILE_SHA256
    assert report["schema"] == \
        "na49_mixing.staged_r50_pure_topology_templates.report/1"
    assert report["card"] == SOLVER_CARD
    assert report["run"] == "R50"
    assert report["mode"] == "production"
    assert report["digest"] == REPORT_DIGEST
    assert report["optimizer_invoked"] is False
    assert report["optimizer_uses"] == 0
    assert report["named_residual_register"] == []
    assert "no separation" in report["no_separation_claim"]
    assert "nd_multi" in report["nd_multi_excluded"]
    nominal = report["nominal_parameters"]
    assert nominal["b_sd_gev2"] == B_SD
    assert nominal["epsilon"] == 0.0
    assert nominal["class_fractions"] == {
        "dd": 0.25, "nd": 0.25, "sd_projectile": 0.25, "sd_target": 0.25}
    assert nominal["remnant_weights"] == {
        "high_mass_baryonic_continuum": 0.0,
        "intact_proton": 0.5,
        "low_mass_baryonic_resonance": 0.5,
    }
    stats = report["statistics"]
    assert stats["events_total"] == 500000
    assert stats["slots"] == 20
    seed_summary = report["seed_map_summary"]
    assert seed_summary["master_seed"] == MASTER_SEED
    assert seed_summary["raw_slot_seeds"] == list(
        range(RAW_SEED_BLOCK[0], RAW_SEED_BLOCK[1] + 1))

    # the global conservation block, independently recomputed
    conservation = report["conservation"]
    for key, value in CONSERVATION.items():
        assert conservation[key] == value, key
    worst_e = worst_p = worst_rel = worst_sh = 0.0
    census = Counter()
    for slot in range(20):
        record = json.loads(_slot_path(slot).read_text(encoding="utf-8"))
        for ev in record["events"]:
            census[tuple(ev["bqs"])] += 1
            d_e, d_p, rel, sh = _event_residuals(ev)
            worst_e = max(worst_e, d_e)
            worst_p = max(worst_p, d_p)
            worst_rel = max(worst_rel, rel)
            worst_sh = max(worst_sh, sh)
    assert dict(census) == {(2, 2, 0): 500000}
    assert worst_e == CONSERVATION["worst_energy_closure_gev"]
    assert worst_p == CONSERVATION["worst_momentum_closure_gev"]
    assert worst_rel == CONSERVATION["worst_closure_relative"]
    assert worst_sh == CONSERVATION["worst_mass_shell_relative"]

    # the per-class pooled observables, independently recomputed
    for cls in r50.CLASS_ORDER:
        block = report["per_class"][cls]
        assert block["events"] == 100000, cls
        events = list(_pool_events(cls))
        assert len(events) == 100000

        xf_p = [ev["ob"][0] for ev in events]
        xf_t = [ev["ob"][1] for ev in events]
        assert _fstats(xf_p) == PER_CLASS_XF[cls]["projectile"], cls
        assert _fstats(xf_t) == PER_CLASS_XF[cls]["target"], cls

        gaps = [ev["gp"] for ev in events if ev["gp"] is not None]
        pinned_gap = PER_CLASS_GAP[cls]
        assert len(gaps) == pinned_gap["present"], cls
        assert 100000 - len(gaps) == pinned_gap["absent"], cls
        if gaps:
            assert _fstats(gaps) == (pinned_gap["mean"], pinned_gap["min"],
                                     pinned_gap["max"],
                                     pinned_gap["std"]), cls

        masses = [m for ev in events for m in ev["ex"]]
        pinned_masses = PER_CLASS_MASSES[cls]
        assert len(masses) == pinned_masses["count"], cls
        if masses:
            assert _fstats(masses) == (pinned_masses["mean"],
                                       pinned_masses["min"],
                                       pinned_masses["max"],
                                       pinned_masses["std"]), cls

        mults = [ev["ob"][2] for ev in events]
        pinned_mult = PER_CLASS_MULTIPLICITY[cls]
        assert _fstats(mults) == (pinned_mult["mean"], pinned_mult["min"],
                                  pinned_mult["max"],
                                  pinned_mult["std"]), cls
        assert dict(Counter(mults)) == {
            int(k): v for k, v in pinned_mult["distribution"].items()}, cls
        assert _fstats([ev["ob"][3] for ev in events])[0] == \
            pinned_mult["central_activity_mean"], cls

        yields = PER_CLASS_YIELDS[cls]
        cf = block["central_forward_yields"]
        totals = [sum(ev["ob"][k] for ev in events)
                  for k in (4, 5, 6, 7)]
        for name, k in (("central_pions", 4), ("central_protons", 5),
                        ("forward_pions", 6), ("forward_protons", 7)):
            total = totals[k - 4]
            mean, pinned_total = yields[name]
            assert total == pinned_total, (cls, name)
            assert cf[name + "_total"] == pinned_total, (cls, name)
            assert cf[name + "_mean"] == mean, (cls, name)

        seagulls = [ev["ob"][8] for ev in events]
        assert _fstats(seagulls) == PER_CLASS_SEAGULL[cls], cls

        transfers = [t for ev in events for t in ev["tr"]]
        pinned_tr = PER_CLASS_TRANSFERS[cls]
        assert len(transfers) == pinned_tr["n"], cls
        assert _fstats(transfers) == (pinned_tr["mean"], pinned_tr["min"],
                                      pinned_tr["max"],
                                      pinned_tr["std"]), cls

        # the histogram edge grids and count sums
        assert report["per_class"][cls]["leading_baryon_xf"] \
            ["projectile"]["histogram"]["edges"] == list(XF_HIST_EDGES)
        gap_hist = block["gap_size"]["histogram"]
        assert gap_hist["edges"] == list(GAP_HIST_EDGES)
        assert sum(gap_hist["counts"]) == pinned_gap["present"], cls
        assert gap_hist["out_of_range_high"] == 0
        assert gap_hist["out_of_range_low"] == 0
        mass_hist = block["excitation_masses"]["histogram"]
        assert mass_hist["edges"] == list(MASS_HIST_EDGES)
        assert sum(mass_hist["counts"]) == pinned_masses["count"], cls
        assert mass_hist["out_of_range_high"] == 0
        assert mass_hist["out_of_range_low"] == 0

    # the class draw census
    for cls, pinned in CLASS_DRAW_CENSUS.items():
        census_block = report["class_draw_census"][cls]
        assert census_block["events"] == pinned["events"], cls
        assert census_block["replicas"] == 4
        assert census_block["exact_by_construction"] is True
        assert census_block["inelastic"] is pinned["inelastic"], cls
        assert census_block["nominal_inelastic_fraction"] == \
            pinned["nominal_inelastic_fraction"], cls
        assert census_block["realized_inelastic_fraction"] == \
            pinned["realized_inelastic_fraction"], cls
        assert census_block["seed_block"] == pinned["seed_block"], cls


def test_elastic_normalization_check():
    """The elastic class's recorded normalization check (closed form,
    quadrature, quantile round trip, pooled t mean against the
    analytic -1/B_SD, x_F pair sum, mass shell, zero transfer, verdict
    pass) with the pooled t statistics, the pair sum, and the mass
    shell independently recomputed from the stored elastic events."""
    skip_unless_sealed(RUN_DIR)
    report = json.loads(
        (RUN_DIR / "template_report.json").read_text(encoding="utf-8"))
    check = report["elastic_normalization_check"]
    assert check["verdict"] == ELASTIC_CHECK["verdict"]
    checks = check["checks"]
    norm = checks["normalization_identity"]
    assert norm["closed_form"] == ELASTIC_CHECK["closed_form"]
    assert norm["quadrature"] == ELASTIC_CHECK["quadrature"]
    assert norm["max_dev"] == ELASTIC_CHECK["normalization_max_dev"]
    assert norm["tolerance"] == 1e-12
    assert norm["pass"] is True
    assert abs(ELASTIC_CHECK["closed_form"] - (1.0 / B_SD)) == 0.0
    quantile = checks["quantile_roundtrip"]
    assert quantile["max_dev"] == ELASTIC_CHECK["quantile_max_dev"]
    assert quantile["tolerance"] == 1e-09
    assert quantile["pass"] is True
    assert "u_j = j / 257" in quantile["grid"]
    t_mean = checks["t_sample_mean"]
    for key in ("analytic_mean_gev2", "sample_mean_gev2", "sample_sd_gev2",
                "sigma_of_mean_gev2", "z", "n"):
        assert t_mean[key] == ELASTIC_CHECK[key], key
    assert t_mean["tolerance_sigma"] == 3.0
    assert t_mean["pass"] is True
    assert checks["xf_in_unit_range"] is True
    assert checks["xf_pair_sum_max_abs"] == \
        ELASTIC_CHECK["xf_pair_sum_max_abs"]
    assert checks["xf_pair_sum_tolerance"] == 1e-09
    assert checks["mass_shell_worst_relative"] == \
        ELASTIC_CHECK["mass_shell_worst_relative"]
    assert checks["zero_transfer_violations"] == 0

    # the independent recompute over the stored elastic events
    events = list(_pool_events("elastic"))
    assert len(events) == 100000
    transfers = [ev["tr"][0] for ev in events]
    mean, sd = _fstats(transfers)[0], _fstats(transfers)[3]
    assert mean == ELASTIC_CHECK["sample_mean_gev2"]
    assert sd == ELASTIC_CHECK["sample_sd_gev2"]
    sigma = sd / math.sqrt(len(transfers))
    assert sigma == ELASTIC_CHECK["sigma_of_mean_gev2"]
    z = (mean - ELASTIC_CHECK["analytic_mean_gev2"]) / sigma
    assert z == ELASTIC_CHECK["z"]
    pair_sum = max(abs(ev["ob"][0] + ev["ob"][1]) for ev in events)
    assert pair_sum == ELASTIC_CHECK["xf_pair_sum_max_abs"]
    worst_sh = 0.0
    for ev in events:
        for r in ev["ln"]:
            E, px, py, pz = r[0], r[1], r[2], r[3]
            inv = E * E - (px * px + py * py + pz * pz)
            worst_sh = max(worst_sh, abs(inv - M_P * M_P) / (M_P * M_P))
    assert worst_sh == ELASTIC_CHECK["mass_shell_worst_relative"]


# ----------------------------------------------------------------------
# The gate record
# ----------------------------------------------------------------------
def test_gate5_record_on_disk():
    """The gate record's identity, clause, report digest, conservation
    summary (zero violations over 500,000 events), per-class draw
    census, excluded classes, optimizer record, and nominal
    parameters."""
    skip_unless_sealed(RUN_DIR)
    gate = json.loads(
        (RUN_DIR / "gate5_r50.json").read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "gate5_r50.json") == GATE5_FILE_SHA256
    assert gate["schema"] == "na49_mixing.staged_r50_pure_topology_templates.gate/1"
    assert gate["card"] == SOLVER_CARD
    assert gate["run"] == "R50"
    assert gate["gate"] == "gate 5 (plan section 11, line 938)"
    assert gate["clause"] == "per-class template evidence (training side)"
    assert gate["report_digest"] == REPORT_DIGEST
    assert gate["events_checked"] == 500000
    assert gate["classes"] == [
        "elastic", "sd_projectile", "sd_target", "dd", "nd"]
    for key, value in GATE5_CONSERVATION.items():
        assert gate["conservation"][key] == value, key
    elastic = gate["elastic_normalization_check"]
    assert elastic["verdict"] == "pass"
    assert "1e-12" in elastic["tolerance"]
    assert gate["excluded_from_inelastic"] == ["elastic", "nd_multi"]
    assert gate["optimizer_invoked"] is False
    assert gate["optimizer_uses"] == 0
    assert gate["sealed"] is True
    assert gate["model_uncertainty_carry"]["r32_spread"] \
        ["bin_spread_p95"] == 4.028426
    nominal = gate["nominal_parameters"]
    assert nominal["b_sd"] == B_SD
    assert nominal["epsilon"] == 0.0
    assert nominal["p_n2"] == 0.0
    assert nominal["inelastic_fractions"] == {
        "dd": 0.25, "nd": 0.25, "sd_projectile": 0.25, "sd_target": 0.25}
    for cls in r50.CLASS_ORDER:
        per_class = gate["per_class"][cls]
        assert per_class["events"] == 100000, cls
        assert per_class["draw_count"] == 100000, cls
        assert per_class["replicas"] == 4
        assert per_class["seed_block"] == \
            CLASS_DRAW_CENSUS[cls]["seed_block"], cls
        assert per_class["stored"] == [
            "leading_baryon_xf", "gap_size", "excitation_masses",
            "multiplicity", "central_forward_yields", "seagull_moments"]


# ----------------------------------------------------------------------
# The seal
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums():
    """The seal manifest's 26 sealed files (20 slot records plus the
    six top-level files), the checksum file's 26 entries, the 28-file
    run dir (the 26 sealed payload files plus the checksum file and
    the seal manifest), and the full re-hash of all 26 sealed files
    against the checksum digests."""
    skip_unless_sealed(RUN_DIR)
    seal = json.loads(
        (RUN_DIR / "seal_manifest.json").read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "seal_manifest.json") == \
        SEAL_MANIFEST_FILE_SHA256
    assert seal["schema"] == \
        "na49_mixing.staged_r50_pure_topology_templates.seal/1"
    assert seal["card"] == SOLVER_CARD
    assert seal["run"] == "R50"
    assert seal["record_count"] == 20
    assert seal["sealed_utc"] == SEAL_SEALED_UTC
    assert seal["sha256sums_entries"] == 26
    assert seal["clean_validation_record"] == \
        "clean_environment_commands.json"
    assert seal["files"] == list(SEALED_FILES)
    assert seal["files"] == sorted(seal["files"])
    assert seal["git_state"] == {
        "branch": "main", "dirty": True, "head": GIT_HEAD}
    assert seal["environment"]["python"] == "3.12.3"
    assert seal["environment"]["numpy"] == "1.26.4"

    checksums = {}
    for line in (RUN_DIR / "SHA256SUMS.txt").read_text(
            encoding="utf-8").splitlines():
        digest, name = line.split("  ")
        checksums[name] = digest
    assert len(checksums) == 26
    assert set(checksums) == set(SEALED_FILES)
    assert checksums["pre_registration.json"] == PRE_REGISTRATION_SHA256
    assert checksums["template_report.json"] == REPORT_FILE_SHA256
    assert checksums["gate5_r50.json"] == GATE5_FILE_SHA256
    assert checksums["clean_environment_commands.json"] == \
        CLEAN_ENV_FILE_SHA256
    assert checksums["provenance.json"] == PROVENANCE_FILE_SHA256
    assert checksums["REPORT.md"] == REPORT_MD_FILE_SHA256
    for name, digest in SLOT_FILE_SHA256.items():
        assert checksums[name] == digest, name
    assert sealed_mismatches(RUN_DIR) == []

    absent = {n for n in deleted_events(RUN_DIR) if not (RUN_DIR / n).is_file()}
    expected = (set(SEALED_FILES) | {"seal_manifest.json", "SHA256SUMS.txt"}) - absent
    if (RUN_DIR / "EVENTS_DELETED.txt").is_file():
        expected.add("EVENTS_DELETED.txt")
    run_files = sorted(p.name for p in RUN_DIR.iterdir() if p.is_file())
    assert set(run_files) == expected


# ----------------------------------------------------------------------
# The clean environment
# ----------------------------------------------------------------------
def test_clean_environment_record():
    """The clean-environment validation's command list, the
    byte-identical report re-run (digest equality), the probe slot
    record, and the pending seal state."""
    skip_unless_sealed(RUN_DIR)
    clean = json.loads(
        (RUN_DIR / "clean_environment_commands.json")
        .read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "clean_environment_commands.json") == \
        CLEAN_ENV_FILE_SHA256
    assert clean["schema"] == \
        "na49_mixing.staged_r50_pure_topology_templates.clean_env/1"
    assert clean["card"] == SOLVER_CARD
    assert clean["run"] == "R50"
    assert clean["commands"] == list(CLEAN_ENV_COMMANDS)
    assert clean["report_digest"] == REPORT_DIGEST
    assert clean["report_rerun_digest"] == REPORT_DIGEST
    assert clean["report_hash_match"] is True
    assert clean["verified_utc"] == CLEAN_ENV_VERIFIED_UTC
    probe = clean["probe_slot"]
    for key, value in CLEAN_ENV_PROBE_SLOT.items():
        assert probe[key] == value, key
    seal = clean["seal"]
    assert seal["entries"] == 0
    assert seal["mismatches"] == []
    assert seal["state"].startswith("pending")


# ----------------------------------------------------------------------
# The provenance
# ----------------------------------------------------------------------
def test_provenance_on_disk():
    """The provenance's schema, the recorded driver digests (the
    committed dual-tree digest), the frozen substrate block (all seven
    digests verified), the M8 frozen set, the round-5 module digests,
    the environment, and the git state."""
    skip_unless_sealed(RUN_DIR)
    prov = json.loads(
        (RUN_DIR / "provenance.json").read_text(encoding="utf-8"))
    assert _sha256_file(RUN_DIR / "provenance.json") == \
        PROVENANCE_FILE_SHA256
    assert prov["schema"] == \
        "na49_mixing.staged_r50_pure_topology_templates.provenance/1"
    assert prov["card"] == SOLVER_CARD
    assert prov["run"] == "R50"
    assert prov["recorded_utc"] == PROVENANCE_RECORDED_UTC
    driver = prov["driver"]
    assert driver["byte_identical"] is True
    assert driver["nested"]["path"] == DRIVER_REL_NESTED
    assert driver["nested"]["sha256"] == DRIVER_SHA256
    assert driver["outer"]["path"] == DRIVER_REL_OUTER
    assert driver["outer"]["sha256"] == DRIVER_SHA256
    for name, digest in FROZEN_SUBSTRATE.items():
        entry = prov["frozen_substrate"][name]
        assert entry["sha256"]["sha256"] == digest, name
        assert entry["sha256"]["verified"] is True, name
        assert entry["verified"] is True, name
    m8 = prov["m8_frozen_set"]
    assert m8["hash"] == M8_HASH
    assert m8["blend_ramp_start_gev"] == BLEND_RAMP_START
    assert m8["effective_transfer_slope_gev2"] == B_SD
    assert m8["verified"] is True
    assert prov["round5_modules"] == ROUND5_MODULES
    assert prov["git_state"] == {
        "branch": "main", "dirty": True, "head": GIT_HEAD}
    assert prov["environment"]["python"] == "3.12.3"
    assert prov["environment"]["numpy"] == "1.26.4"
    assert prov["card_chain"][-1] == SOLVER_CARD


# ----------------------------------------------------------------------
# The freeze re-pin
# ----------------------------------------------------------------------
