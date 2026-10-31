"""R51i driver: the fine floor scan of the joint proton and pion topology
fit on the corrected final state (Gate 5 follow-up; card
TASK-20261001-r5-gate5-followup-r51i-fine-floor-scan).

Differences from R51h (everything else is the R51h contract):
- Floor grid FLOOR_GRID = (1.6, 1.7, 1.8, 1.9) GeV around the minimum that
  R51g and R51h bracket between 1.4 and 2.0 GeV (owner: "maybe try 1.8 and
  1.9 GeV"; 1.6 keeps 1.7 interior).
- Common random numbers across the floors: the raw seed of a slot depends
  on the generation and the slot, not on the floor, so the exchange
  kinematics and the first decays coincide between branches and the
  floor-to-floor differences of the objective carry less Monte Carlo
  noise than independent template sets (the two independent 1.7 GeV
  objectives of R51g and R51h differ by 4%).
- Generation 0 of every floor at the sealed R51h optimum (floor 1.7 GeV).
- Four times the statistics, 32 replicas of 100,000 events per class and
  floor, and a block-parallel objective for the gateway nodes: persistent
  worker processes own contiguous replica blocks and return per-block,
  per-class observation vectors at a trial shape point; the optimizer
  process combines them with the fractions and forms the likelihood.
  Every optimizer start runs its own pool (BLOCK_WORKERS_PER_START), the
  details, Hessian and variant refits use a pool of MASTER_BLOCK_WORKERS,
  and the replica-block fits load one block each.  The pooled objective
  equals the in-process one up to floating summation order (the smoke
  records the comparison).

R51h design (inherited):
Differences from R51g (everything else is the R51g contract):
- Floor grid FLOOR_GRID = (1.7, 2.0, 2.3) GeV: the R51g profile minimum
  lay on its highest value, 1.7 GeV.
- EPS_BOUNDS = (-0.5, 1.5): the R51g diffractive exponent sat on its
  upper bound, 0.5, and the Hessian could not be formed there.
- Generation 0 of every floor at the sealed R51g optimum.

R51g design (inherited):
Differences from R51f (everything else is the R51f contract):
- Substrate: the corrected final state.  Every unstable established
  hadron decays through the Round 2 kernel before it reaches a final
  state (the direct-channel daughters and the single-diffraction remnant
  had entered undecayed), and the continuum config's isospin rule
  (minimal isospin I = |I3|) weights the continuum and direct vertices by
  the Clebsch-Gordan share.
- Floor grid FLOOR_GRID = (1.7, 1.4, 1.1) GeV: the R51f profile minimum
  lay on its lowest value, 1.4 GeV; 1.1 GeV stays above the N pi
  threshold.
- Generation 0 of every floor at the sealed R51f optimum.
- Held out in addition: the pi0 yield and its ratio to the charged-pion
  mean, and the neutron dN/dx_F against the NA49 table (per-event pi0
  counts and neutron x_F records in the slot arrays).

R51f design (inherited):
Differences from R51e (everything else is the R51e contract):
- Substrate: the continuation-floor mechanism (nested 71e3834) with the
  committed Block B stage 1 config, which sets no floor.  The driver sets
  the floor in memory for every family and records it in every slot.
- Floor grid FLOOR_GRID = (2.0, 1.7, 1.4) GeV.  The floor changes the
  decay chains themselves, so importance weights cannot move it: each
  grid value is its own template branch with its own generations, the
  reweightable parameters are fitted within a branch, and the floor is
  profiled across the branch optima.
- Free parameters (4, fractions at the anchors): epsilon, B_SD,
  epsilon_nd and the excited share of the non-diffractive remnants.  The
  intact remnant weight is fixed at 0 (on its lower bound in R51c, R51d
  and R51e); the non-diffractive templates are generated at the remnant
  weights (0, 1/2, 1/2).
- Data: the R51e proton observables plus the pi- (282 bins) and pi+
  (260 bins) tables over x_F in [0, 0.95] and p_T in [0.05, 1.5] GeV/c;
  one shared normalization nuisance (2% prior); a fractional
  model-discrepancy variance (DISCREPANCY x data)^2 per bin, with a
  DISCREPANCY_VARIANT refit at the chosen floor.
- Held out: kaons and antiprotons alone (total yields and paper-window
  sum ratios), the pi- p_T profile, the mesonic hadron count per decay
  against the MARK I e+e- curve, and the single-diffraction n_ch against
  M_X.
- Verdict CONSTRAINED also requires the profiled floor interior to the
  grid.

Steps (one per invocation): --pre-registration, --generate [--floor-index
F] [--generation G] [--slot S ...], --evaluate [--floor-index F],
--profile, --gate, --clean-validate, --seal; --smoke runs a reduced
end-to-end exercise into an explicit --run-dir.  Every branch keeps its
own generation-points file, so branches evaluate concurrently (one
gateway job per branch; orchestrate_gateway_joint_fit.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import run_contract as RC  # noqa: E402

NESTED_ROOT, MONO_ROOT = RC.locate_roots(HERE)
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

import run_observables as OB  # noqa: E402

from na49_mixing.cascade_decay import (  # noqa: E402
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import handoff_decay, has_valid_direct_channel, is_catalog_handoff  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.importance import ess_fraction, logits_from_simplex, simplex_from_logits  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.topology_mass_laws import log_density_ratio  # noqa: E402
from na49_mixing.topology_runner import (  # noqa: E402
    NOMINAL_PARAMS,
    TopologyNominalParams,
    extract_observables,
    run_topology,
)
from na49_mixing.topology_single_diffraction import _final_from_catalog, _kernel_finals  # noqa: E402

# ----------------------------------------------------------------------
# Identity and contract constants
# ----------------------------------------------------------------------
RUN_ID = "r51i_joint_fit_2026-10-01"
CARD = "TASK-20261001-r5-gate5-followup-r51i-fine-floor-scan"
SCHEMA = "na49_mixing.staged_r51i_joint_fit"
LEDGER_RUN = "r51i_joint_fit"

CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd")
REPLICAS = 32
EVENTS_PER_SLOT = 100_000
N_SLOTS = len(CLASS_ORDER) * REPLICAS  # 128
MAX_GENERATIONS = 4
FLOOR_GRID = (1.6, 1.7, 1.8, 1.9)
N_FLOORS = len(FLOOR_GRID)

MASTER_SEED = 20261001
RAW_FIRST = 20314200
RAW_LAST = RAW_FIRST + MAX_GENERATIONS * N_SLOTS - 1  # 20314711 (common random numbers across floors)
PARTITION_BASE = 3_600_000_000
SMOKE_SEED = 20304105
SMOKE_PARTITION = 3_150_000_000
SMOKE_EVENTS_PER_CLASS = 500
SMOKE_FLOOR_INDEX = 1
SMOKE_MESONIC_DECAYS = 200

SQRT_S = OB.SQRT_S
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
EPS_BOUNDS = (-0.5, 1.5)
EPS_ND_BOUNDS = (-1.5, 0.5)
B_SD_BOUNDS = (1.0, 20.0)
LOGIT_BOUNDS = (-8.0, 8.0)
FRACTION_INTERIOR = (0.005, 0.995)
NORM_PRIOR = 0.02
DISCREPANCY = 0.05
DISCREPANCY_VARIANT = 0.10
SE_FLOOR = 1.0e-9
STABILITY_RATIO_BOUND = 1.5
N_STARTS = 8
N_STARTS_VARIANT = 4
# the block-parallel objective: processes per optimizer start (the primary
# fit's 8 starts and the variants' 4) and for the details, Hessian and
# variant evaluations; a 64-core gateway node per branch
BLOCK_WORKERS_PER_START = 6
BLOCK_WORKERS_PER_VARIANT_START = 12
MASTER_BLOCK_WORKERS = 32
EVENT_CLOSURE_TOL = 1.0e-9
HESSIAN_STEPS = {"f_sd_total": 0.01, "f_dd": 0.005, "epsilon": 0.01, "b_sd": 0.1, "w_excited": 0.02, "epsilon_nd": 0.02}
SHAPE_NAMES = ("epsilon", "b_sd", "w_excited", "epsilon_nd")
FRACTION_NAMES = ("f_sd_total", "f_dd")
WEIGHT_CACHE_ENTRIES = 64
SE_CACHE_ENTRIES = 512
# generation 0 of every floor: the sealed R51h optimum (floor 1.7 GeV)
GEN0_EPSILON = 0.49815418511182347
GEN0_B_SD = 2.4114630077319634
GEN0_EPSILON_ND = 0.3294084567271025
R51H_W_LOW = 0.08606901837123448
R51H_W_EXCITED = 0.9139309816287655
GEN0_W_EXCITED = R51H_W_EXCITED
R51H_RUN_ID = "r51h_joint_fit_2026-10-01"
REMNANT_KINDS = ("intact_proton", "low_mass_baryonic_resonance", "high_mass_baryonic_continuum")
ND_GENERATION_REMNANT_WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.5}
FROZEN_REMNANT_WEIGHTS = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}

PROTON_WINDOW = {"xf_min": 0.0, "xf_max": 0.95, "pt_min": 0.05, "pt_max": 1.5}
PION_WINDOW = dict(PROTON_WINDOW)
PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
TABLE_WINDOWS = {"proton": PROTON_WINDOW, "pi_minus": PION_WINDOW, "pi_plus": PION_WINDOW}
TRACK_SPECIES = OB.SPECIES_ORDER
PION_FIT_SPECIES = ("pi_minus", "pi_plus")

# held-out readouts
PT_PROFILE = {"species": "pi_minus", "xf_max": 0.5, "pt_low": 0.3, "pt_high": 1.3}
MARK_I = {"a": 2.103, "b": 0.832, "source": "Siegrist et al., Phys. Rev. D 26 (1982) 969; HEPData ins169326 Table 6; "
          "weighted fit <n_ch> = a + b ln s over 2.6 to 7.8 GeV (chi2/ndf 30.5/36)"}
MESONIC_MASSES = (2.6, 3.0, 3.5)
MESONIC_DECAYS = 2000
# charged decay products per final-state neutral weak decayer, counted as in
# the e+e- measurement (PDG: K0S -> pi+ pi- 69.2%, Lambda -> p pi- 63.9%;
# K0 and K0bar are half K0S; Sigma0 -> Lambda gamma)
CHARGED_FROM_WEAK = {310: 2 * 0.692, 311: 0.692, -311: 0.692, 3122: 2 * 0.639, -3122: 2 * 0.639,
                     3212: 2 * 0.639, -3212: 2 * 0.639}
NEUTRON_TABLE_REL = "spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt"
NEUTRON_XF_EDGES = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.675, 0.825, 0.975])
SD_MX_EDGES = np.asarray([1.08, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.3])

LEAD_XF_EDGES = np.linspace(-1.05, 1.05, 43)
GAP_EDGES = np.linspace(0.0, 17.28, 51)
NCH_MAX = 40

FROZEN_RELS = (
    "dev/mixing2006/na49_mixing/topology_runner.py",
    "dev/mixing2006/na49_mixing/topology_single_diffraction.py",
    "dev/mixing2006/na49_mixing/topology_double_diffraction.py",
    "dev/mixing2006/na49_mixing/topology_non_diffractive.py",
    "dev/mixing2006/na49_mixing/topology_multi_neutral.py",
    "dev/mixing2006/na49_mixing/topology_elastic.py",
    "dev/mixing2006/na49_mixing/topology_remnants.py",
    "dev/mixing2006/na49_mixing/topology_mass_laws.py",
    "dev/mixing2006/na49_mixing/exchange_ledger.py",
    "dev/mixing2006/na49_mixing/cascade_decay.py",
    "dev/mixing2006/na49_mixing/cascade_termination.py",
    "dev/mixing2006/na49_mixing/resonance_decay.py",
    "dev/mixing2006/na49_mixing/resonance_widths.py",
    "dev/mixing2006/na49_mixing/hadron_catalog.py",
    "dev/mixing2006/na49_mixing/spectral_density.py",
    "dev/mixing2006/na49_mixing/flavor_weights.py",
    "dev/mixing2006/na49_mixing/importance.py",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
    "dev/mixing2006/config/v2_singlet/external_cross_sections.json",
    "dev/mixing2006/config/staged_excitation.json",
    "dev/mixing2006/config/v2_singlet/na49_pp_total_yields.json",
    "helpers/benchmarking/run_contract.py",
    "helpers/benchmarking/run_observables.py",
    "helpers/benchmarking/staged_r51c_fraction_and_remnant_fit.py",
    "helpers/benchmarking/staged_r51d_topology_fit.py",
    "helpers/benchmarking/staged_r51e_topology_fit.py",
    "helpers/benchmarking/staged_r51f_joint_fit.py",
    "helpers/benchmarking/staged_r51g_joint_fit.py",
    "helpers/benchmarking/staged_r51h_joint_fit.py",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
)

SLOT_JSON_FMT = "slot_{cls}_{k}_f{f}_g{g}.json"
SLOT_NPZ_FMT = "slot_{cls}_{k}_f{f}_g{g}.npz"
FIT_REPORT_FMT = "fit_report_f{f}_g{g}.json"
BRANCH_REPORT_FMT = "branch_report_f{f}.json"
REGENERATION_FMT = "regeneration_request_f{f}.json"
GATE_RECORD_NAME = "gate5_followup_r51i.json"
GENERATION_POINTS_FMT = "generation_points_f{f}.json"
# the gateway sync record (written by orchestrate_gateway_joint_fit.py at the remote
# root; the remote checkout carries no git metadata)
SYNC_MANIFEST_NAME = "SYNC_MANIFEST.json"


def slot_class(slot: int) -> str:
    return CLASS_ORDER[int(slot) // REPLICAS]


def slot_replica(slot: int) -> int:
    return int(slot) % REPLICAS


def slot_seed(floor_index: int, generation: int, slot: int) -> int:
    """Common random numbers across the floors: the floor index does not
    enter the seed."""
    return RAW_FIRST + N_SLOTS * int(generation) + int(slot)


def slot_names(floor_index: int, generation: int, slot: int) -> tuple[str, str]:
    cls, k = slot_class(slot), slot_replica(slot)
    return (SLOT_JSON_FMT.format(cls=cls, k=k, f=floor_index, g=generation),
            SLOT_NPZ_FMT.format(cls=cls, k=k, f=floor_index, g=generation))


def _contract(run_dir: Path | None) -> RC.RunContract:
    return RC.RunContract(RUN_ID, CARD, driver_file=HERE, schema_prefix=SCHEMA, run_dir=run_dir)


def frozen_digests() -> dict:
    return {rel: RC.sha256_file(MONO_ROOT / rel) for rel in FROZEN_RELS}


def anchors() -> dict:
    ext = RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json")
    f = ext["fractions_of_inelastic"]
    return {
        "f_sd_total": {"value": float(f["f_sd_total"]["value"]), "sigma": float(f["f_sd_total"]["uncertainty"])},
        "f_dd": {"value": float(f["f_dd"]["value"]), "sigma": float(f["f_dd"]["uncertainty"])},
        "source": "config/v2_singlet/external_cross_sections.json",
        "source_sha256": RC.sha256_file(NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json"),
    }


def load_fit_tables() -> dict:
    return OB.load_tables(NESTED_ROOT, windows=TABLE_WINDOWS)


# ----------------------------------------------------------------------
# Event generation (one slot per worker process)
# ----------------------------------------------------------------------
def _load_physics(floor: float):
    """The committed catalog and continuum families with the continuation
    floor set in memory for every family (the config itself sets none)."""
    catalog = HadronCatalog.from_json(NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json")
    fams = families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    if any(f.continuation_floor is not None for f in fams):
        raise RuntimeError("the committed continuum config sets a continuation floor; the driver sets it per branch")
    return catalog, [dataclasses.replace(f, continuation_floor=float(floor)) for f in fams]


def _params_at(eps: float, b_sd: float, cls: str | None = None, eps_nd: float | None = None) -> TopologyNominalParams:
    weights = ND_GENERATION_REMNANT_WEIGHTS if cls == "nd" else FROZEN_REMNANT_WEIGHTS
    return TopologyNominalParams(
        epsilon=float(eps), b_sd=float(b_sd), class_fractions=dict(NOMINAL_PARAMS.class_fractions),
        remnant_weights=dict(weights), p_n2=0.0, flavor=None,
        epsilon_nd=None if eps_nd is None else float(eps_nd),
    )


def _line_qn(h) -> tuple[int, int, int]:
    return int(h.qn[0]), int(h.qn[1]), int(h.qn[2])


def event_conservation(event) -> tuple[float, float]:
    """Exact (B, Q, S) = (2, 2, 0) and the relative four-momentum closure
    against the incoming reference; raises on a violation."""
    e_beam = math.sqrt(OB.PROTON_MASS ** 2 + OB.BEAM_MOMENTUM ** 2)
    e_total = e_beam + OB.PROTON_MASS
    finals = list(event.final_state)
    e_sum = math.fsum(float(h.four_momentum[0]) for h in finals)
    px = math.fsum(float(h.four_momentum[1]) for h in finals)
    py = math.fsum(float(h.four_momentum[2]) for h in finals)
    pz = math.fsum(float(h.four_momentum[3]) for h in finals)
    bqs = tuple(sum(_line_qn(h)[j] for h in finals) for j in range(3))
    if bqs != (2, 2, 0):
        raise RuntimeError(f"conservation: (B, Q, S) = {bqs}, expected (2, 2, 0)")
    worst = max(abs(e_sum - e_total) / e_total, abs(px) / OB.BEAM_MOMENTUM, abs(py) / OB.BEAM_MOMENTUM,
                abs(pz - OB.BEAM_MOMENTUM) / OB.BEAM_MOMENTUM)
    if worst > EVENT_CLOSURE_TOL:
        raise RuntimeError(f"conservation: closure residual {worst!r} exceeds {EVENT_CLOSURE_TOL!r}")
    worst_shell = 0.0
    for h in finals:
        e, hx, hy, hz = (float(x) for x in h.four_momentum)
        m2 = e * e - hx * hx - hy * hy - hz * hz
        worst_shell = max(worst_shell, abs(m2 - float(h.mass) ** 2) / max(float(h.mass) ** 2, 1e-9))
    if worst_shell > EVENT_CLOSURE_TOL:
        raise RuntimeError(f"conservation: mass-shell deviation {worst_shell!r}")
    return worst, worst_shell


def generate_slot(cls: str, seed: int, n_events: int, partition_base: int, eps: float, b_sd: float, eps_nd: float,
                  floor: float) -> dict:
    """Generate one slot at the continuation floor *floor*: returns the
    compact arrays and the metadata."""
    catalog, fams = _load_physics(floor)
    params = _params_at(eps, b_sd, cls, eps_nd)
    beta, gamma = OB.cm_boost(SQRT_S)
    rem_kind = np.full((n_events, 2), -1, dtype=np.int8)
    species_pdg = {name: OB.SPECIES_PDG[name] for name in TRACK_SPECIES}
    shape: dict[str, list] = {}
    lead = np.full(n_events, np.nan)
    gap = np.full(n_events, np.nan)
    n_ch = np.zeros(n_events, dtype=np.int16)
    n_pi0 = np.zeros(n_events, dtype=np.int16)
    neu_event: list[int] = []
    neu_xf: list[float] = []
    trk_event: list[int] = []
    trk_species: list[int] = []
    trk_xf: list[float] = []
    trk_pt: list[float] = []
    trk_w: list[float] = []
    rolling = hashlib.sha256()
    worst_closure = 0.0
    worst_shell = 0.0
    t0 = time.perf_counter()
    for i in range(int(n_events)):
        event_id = int(partition_base) + i
        subs = derive_substreams(int(seed), 0, event_id)
        rngs = {f"rng_{label}": np.random.default_rng(subs.stream_for(label))
                for label in ("topology_selection", "spectral_state", "exchange_kinematics",
                              "decay_channel", "decay_kinematics")}
        event = run_topology(cls, params, catalog, fams, **rngs)
        closure, shell = event_conservation(event)
        worst_closure = max(worst_closure, closure)
        worst_shell = max(worst_shell, shell)
        obs = extract_observables(event, SQRT_S)
        arrays = OB.shape_arrays(event)
        for key, value in arrays.items():
            shape.setdefault(key, []).append(value)
        lead_xf = OB.leading_proton_xf(event.final_state, beta, gamma, SQRT_S)
        lead[i] = np.nan if lead_xf is None else lead_xf
        gap[i] = np.nan if obs.gap_size is None else float(obs.gap_size)
        n_ch[i] = OB.charged_multiplicity(event.final_state)
        for h in event.final_state:
            if h.pdg_id is None:
                continue
            if int(h.pdg_id) == 111:
                n_pi0[i] += 1
            elif int(h.pdg_id) == 2112:
                _e, _px, _py, pz_cm = OB.to_cm(h.four_momentum, beta, gamma)
                neu_event.append(i); neu_xf.append(2.0 * pz_cm / SQRT_S)
        if cls == "nd":
            rem_kind[i] = [REMNANT_KINDS.index(r.kind) for r in event.remnants]
        tracks = OB.track_arrays(event.final_state, species_pdg, beta, gamma, SQRT_S)
        line = [cls, str(i), str(int(n_ch[i])), float(lead[i]).hex(), float(gap[i]).hex(), f"{int(rem_kind[i][0])}/{int(rem_kind[i][1])}"]
        for s_index, name in enumerate(TRACK_SPECIES):
            xs, ps, ws = tracks[name]
            for xf, pt, w in zip(xs, ps, ws):
                trk_event.append(i); trk_species.append(s_index)
                trk_xf.append(xf); trk_pt.append(pt); trk_w.append(w)
                line.append(f"{s_index}:{xf.hex()}:{pt.hex()}:{w.hex()}")
        rolling.update(("|".join(line) + "\n").encode("utf-8"))
    wall = time.perf_counter() - t0
    return {
        "cls": cls, "seed": int(seed), "n_events": int(n_events), "partition_base": int(partition_base),
        "epsilon": float(eps), "b_sd": float(b_sd), "epsilon_nd": float(eps_nd), "continuation_floor": float(floor),
        "remnant_weights": dict(params.remnant_weights),
        "arrays": {
            **{f"shape_{k}": np.asarray(v, dtype=float) for k, v in shape.items()},
            "lead_xf": lead, "gap": gap, "n_ch": n_ch, "rem_kind": rem_kind,
            "n_pi0": n_pi0, "neu_event": np.asarray(neu_event, dtype=np.int32), "neu_xf": np.asarray(neu_xf, dtype=float),
            "trk_event": np.asarray(trk_event, dtype=np.int32),
            "trk_species": np.asarray(trk_species, dtype=np.int8),
            "trk_xf": np.asarray(trk_xf, dtype=float),
            "trk_pt": np.asarray(trk_pt, dtype=float),
            "trk_w": np.asarray(trk_w, dtype=float),
        },
        "event_rolling_sha256": rolling.hexdigest(),
        "conservation": {"events": int(n_events), "violations": 0,
                         "worst_closure_relative": worst_closure, "worst_mass_shell_relative": worst_shell,
                         "tolerance": EVENT_CLOSURE_TOL},
        "tracks": int(len(trk_event)),
        "wall_seconds": wall,
    }


def arrays_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8")); digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8")); digest.update(arr.tobytes())
    return digest.hexdigest()


def write_slot(contract: RC.RunContract, floor_index: int, generation: int, slot: int, result: dict, pre_sha: str) -> dict:
    json_name, npz_name = slot_names(floor_index, generation, slot)
    np.savez(contract.path(npz_name), **result["arrays"])
    record = {
        "schema": f"{SCHEMA}.slot/1", **contract.identity(), "pre_registration_sha256": pre_sha,
        "floor_index": int(floor_index), "continuation_floor": result["continuation_floor"],
        "generation": int(generation), "slot": int(slot), "class_label": result["cls"], "replica": slot_replica(slot),
        "seed": result["seed"], "n_events": result["n_events"], "partition_base": result["partition_base"],
        "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
        "arrays_file": npz_name, "arrays_sha256": arrays_digest(result["arrays"]),
        "event_rolling_sha256": result["event_rolling_sha256"], "conservation": result["conservation"],
        "tracks": result["tracks"], "track_species": list(TRACK_SPECIES),
        "written_utc": RC.now_utc(),
    }
    RC.write_json(contract.path(json_name), record)
    return record


# ----------------------------------------------------------------------
# The held-out cascade readout: charged hadrons per decay at rest
# ----------------------------------------------------------------------
def source_decay_finals(family, mass: float, catalog, fams, rng_channel, rng_kinematics) -> list:
    """The final-state hadrons of one continuum source at rest, through the
    topology generator's excitation cascade (continuation, direct channel,
    catalog handoff; unstable siblings decay through the Round 2 kernel)."""
    current = CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(int(v) for v in family.qn),
                           quantum={}, family_id=family.family_id, axis=(0.0, 0.0, 1.0),
                           alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=("r51i_readout",), depth=0)
    finals: list = []

    def assemble(daughter):
        pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
        if pdg in catalog.stable_states():
            finals.append(_final_from_catalog(catalog, pdg, daughter.momentum, "readout"))
            return
        kernel = ResonanceDecayer(catalog, 0, 0, 0)
        kernel._rng_channel = rng_channel
        kernel._rng_kinematics = rng_kinematics
        finals.extend(_kernel_finals(catalog, kernel.decay_resonance(pdg, tuple(float(x) for x in daughter.momentum)), "readout"))

    while True:
        if is_catalog_handoff(current, catalog):
            finals.extend(_kernel_finals(catalog, handoff_decay(current, catalog, rng_channel, rng_kinematics), "readout"))
            return finals
        channels = enumerate_channels(current, catalog, fams, flavor=None)
        continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
        if continuum:
            _, daughters = draw_decay(current, continuum, catalog, fams, rng_channel, rng_kinematics)
            successor = None
            for d in daughters:
                if d.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    assemble(d)
                else:
                    successor = d
            current = successor
            continue
        if has_valid_direct_channel(current, channels):
            direct = [c for c in channels if c.kind != KIND_CONTINUUM]
            _, daughters = draw_decay(current, direct, catalog, fams, rng_channel, rng_kinematics)
            for d in daughters:
                assemble(d)
        return finals


def mesonic_count_readout(floor: float, n_decays: int = MESONIC_DECAYS) -> dict:
    """The mean charged count per decay of a mesonic_neutral source at rest
    at MESONIC_MASSES against the MARK I curve (the charged products of
    the final neutral weak decayers counted as in the measurement)."""
    catalog, fams = _load_physics(floor)
    family = next(f for f in fams if f.family_id == "mesonic_neutral")
    rows = []
    for m_index, mass in enumerate(MESONIC_MASSES):
        key = [MASTER_SEED, 51, int(round(1000 * floor)), m_index]
        rc = np.random.default_rng(np.random.SeedSequence(key + [1]))
        rk = np.random.default_rng(np.random.SeedSequence(key + [2]))
        total = 0.0
        for _ in range(int(n_decays)):
            for h in source_decay_finals(family, mass, catalog, fams, rc, rk):
                total += (1.0 if int(h.qn[1]) != 0 else 0.0) + CHARGED_FROM_WEAK.get(int(h.pdg_id), 0.0)
        model = total / float(n_decays)
        reference = MARK_I["a"] + MARK_I["b"] * math.log(mass * mass)
        rows.append({"mass": float(mass), "model": model, "mark_i": reference, "ratio": model / reference,
                     "below_open_charm": bool(mass < 3.73)})
    return {"floor": float(floor), "decays_per_mass": int(n_decays), "family": "mesonic_neutral",
            "reference": MARK_I, "charged_from_weak": {str(k): v for k, v in CHARGED_FROM_WEAK.items()},
            "seed_rule": "numpy SeedSequence([MASTER_SEED, 51, round(1000 floor), mass index, 1 or 2])", "rows": rows}


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
class Templates:
    """The loaded templates of one floor branch and generation: per class
    and replica the shape arrays and the per-species binned track sums."""

    def __init__(self, contract: RC.RunContract, floor_index: int, generation: int, tables: dict, *, smoke: bool = False,
                 replicas: list | None = None):
        self.floor_index = int(floor_index)
        self.generation = int(generation)
        self.tables = tables
        self.point = None
        self.by_class: dict[str, list] = {cls: [] for cls in CLASS_ORDER}
        n_slots = N_SLOTS if not smoke else len(CLASS_ORDER)
        wanted = None if replicas is None else {int(r) for r in replicas}
        self.replica_ids = sorted(wanted) if wanted is not None else (list(range(REPLICAS)) if not smoke else [0])
        for slot in range(n_slots):
            cls = slot_class(slot) if not smoke else CLASS_ORDER[slot]
            k = slot_replica(slot) if not smoke else 0
            if wanted is not None and not smoke and k not in wanted:
                continue
            json_name = ("smoke_" if smoke else "") + SLOT_JSON_FMT.format(cls=cls, k=k, f=floor_index, g=generation)
            npz_name = ("smoke_" if smoke else "") + SLOT_NPZ_FMT.format(cls=cls, k=k, f=floor_index, g=generation)
            meta = RC.read_json(contract.path(json_name))
            if float(meta["continuation_floor"]) != float(FLOOR_GRID[floor_index]):
                contract.refuse(f"{json_name} was generated at floor {meta['continuation_floor']}, not {FLOOR_GRID[floor_index]}")
            with np.load(contract.path(npz_name)) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays_digest(arrays) != meta["arrays_sha256"]:
                contract.refuse(f"the arrays of {npz_name} do not match their recorded digest")
            point = (float(meta["epsilon"]), float(meta["b_sd"]), float(meta["epsilon_nd"]))
            if self.point is None:
                self.point = point
            elif self.point != point:
                contract.refuse(f"mixed generation points in branch {floor_index} generation {generation}")
            if cls == "nd" and np.any(arrays["rem_kind"] == REMNANT_KINDS.index("intact_proton")):
                contract.refuse(f"{npz_name} carries an intact remnant; the generation weight of that kind is 0")
            self.by_class[cls].append(self._prepare(cls, arrays, int(meta["n_events"])))
        for cls in CLASS_ORDER:
            if not self.by_class[cls]:
                contract.refuse(f"branch {floor_index} generation {generation} carries no {cls} template")
        self.n_blocks = min(len(v) for v in self.by_class.values())

    def _prepare(self, cls: str, arrays: dict, n_events: int) -> dict:
        shape = {k[len("shape_"):]: v for k, v in arrays.items() if k.startswith("shape_")}
        binned = {}
        for s_index, name in enumerate(TRACK_SPECIES):
            if name not in self.tables:
                continue
            table = self.tables[name]
            sel = arrays["trk_species"] == s_index
            ev = arrays["trk_event"][sel]
            xf = arrays["trk_xf"][sel]; pt = arrays["trk_pt"][sel]; w = arrays["trk_w"][sel]
            xedges = OB.sps_core.edges_from_centers(table.xf_centers)
            pedges = OB.sps_core.edges_from_centers(table.pt_centers)
            n_x = xedges.size - 1; n_p = pedges.size - 1
            if xf.size:
                ix, ip = OB.sps_core.grid_indices(xf, pt, xedges, pedges)
                ok = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < n_p)
                flat = ip[ok] * n_x + ix[ok]
                pick = table.bin_index_pt * n_x + table.bin_index_xf
                lookup = np.full(n_x * n_p, -1, dtype=np.int64)
                lookup[pick] = np.arange(pick.size)
                bin_pos = lookup[flat]
                keep = bin_pos >= 0
                binned[name] = (ev[ok][keep], bin_pos[keep], w[ok][keep], int(table.data.size))
            else:
                binned[name] = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64), np.zeros(0), int(table.data.size))
        totals = {}
        for s_index, name in enumerate(TRACK_SPECIES):
            sel = arrays["trk_species"] == s_index
            totals[name] = np.bincount(arrays["trk_event"][sel].astype(np.int64), minlength=int(n_events))[: int(n_events)]
        return {"n": int(n_events), "shape": shape, "binned": binned,
                "lead_xf": arrays["lead_xf"], "gap": arrays["gap"], "n_ch": arrays["n_ch"],
                "rem_kind": arrays.get("rem_kind"), "totals": totals,
                "n_pi0": arrays["n_pi0"], "neu_event": arrays["neu_event"], "neu_xf": arrays["neu_xf"]}

    # -- weights and moments ------------------------------------------
    def log_weights(self, cls: str, block: dict, eps: float, b_sd: float, eps_nd: float, remnant=None) -> np.ndarray:
        eps0, b0, eps0_nd = self.point
        eps_c, eps0_c = (float(eps_nd), eps0_nd) if cls == "nd" else (float(eps), eps0)
        if (eps_c, float(b_sd)) == (eps0_c, b0):
            log_w = np.zeros(block["n"])
        else:
            log_w = log_density_ratio(cls, block["shape"], eps_c, b_sd, eps0=eps0_c, b_sd0=b0, sqrt_s=SQRT_S)
        if cls == "nd" and remnant is not None:
            # the exact kind-pair reweighting from the generation weights
            # (0, 1/2, 1/2); the intact kind is never generated
            gen = np.asarray([ND_GENERATION_REMNANT_WEIGHTS[k] for k in REMNANT_KINDS], dtype=float)
            ratio = np.full(3, -np.inf)
            ratio[1:] = np.log(np.asarray(remnant, dtype=float)[1:] / gen[1:])
            kinds = block["rem_kind"].astype(np.int64)
            log_w = log_w + ratio[kinds[:, 0]] + ratio[kinds[:, 1]]
        return log_w

    @staticmethod
    def normalized(log_w: np.ndarray) -> np.ndarray:
        w = np.exp(log_w - np.max(log_w))
        return w / np.sum(w)

    def species_mean(self, cls: str, block: dict, wn: np.ndarray, name: str) -> np.ndarray:
        """The weighted mean per-event bin content of one species table."""
        ev, pos, w, nbins = block["binned"][name]
        if ev.size == 0:
            return np.zeros(nbins)
        return np.bincount(pos, weights=w * wn[ev], minlength=nbins)[:nbins]

    def histogram_mean(self, block: dict, wn: np.ndarray, key: str, edges: np.ndarray) -> np.ndarray:
        values = block[key]
        ok = np.isfinite(values)
        if not np.any(ok):
            return np.zeros(edges.size - 1)
        h, _ = np.histogram(values[ok], bins=edges, weights=wn[ok])
        return h

    def nch_mean(self, block: dict, wn: np.ndarray) -> np.ndarray:
        n_ch = np.clip(block["n_ch"].astype(np.int64), 0, NCH_MAX)
        return np.bincount(n_ch, weights=wn, minlength=NCH_MAX + 1)[: NCH_MAX + 1]


def param_names(free: bool) -> tuple:
    return (FRACTION_NAMES if free else ()) + SHAPE_NAMES


def x_from_physical(phys: dict, free: bool) -> np.ndarray:
    """The optimizer vector: [logit f_SD, logit f_DD] (free fractions only)
    + [epsilon, ln B_SD, logit w_excited, epsilon_nd]; w_intact = 0."""
    w_exc = min(max(float(phys["w_excited"]), 1e-12), 1.0 - 1e-12)
    core = [float(phys["epsilon"]), math.log(float(phys["b_sd"])), math.log(w_exc / (1.0 - w_exc)), float(phys["epsilon_nd"])]
    if not free:
        return np.asarray(core, dtype=float)
    z = logits_from_simplex([phys["f_sd_total"], phys["f_dd"], 1.0 - phys["f_sd_total"] - phys["f_dd"]])
    z = z - z[2]
    return np.asarray([float(z[0]), float(z[1])] + core, dtype=float)


def physical_from_x(x, free: bool, anchors_: dict) -> dict:
    x = np.asarray(x, dtype=float)
    if free:
        f = simplex_from_logits([x[0], x[1], 0.0])
        core = x[2:]
    else:
        a_sd, a_dd = float(anchors_["f_sd_total"]["value"]), float(anchors_["f_dd"]["value"])
        f = np.asarray([a_sd, a_dd, 1.0 - a_sd - a_dd], dtype=float)
        core = x
    eps, ln_b, r_exc, eps_nd = (float(v) for v in core)
    w_exc = 1.0 / (1.0 + math.exp(-r_exc))
    return {"f_sd_total": float(f[0]), "f_dd": float(f[1]), "f_nd": float(f[2]),
            "epsilon": eps, "b_sd": float(math.exp(ln_b)),
            "w_intact": 0.0, "w_low_mass": float(1.0 - w_exc), "w_excited": float(w_exc), "epsilon_nd": eps_nd}


def bounds_for(free: bool) -> list:
    core = [EPS_BOUNDS, (math.log(B_SD_BOUNDS[0]), math.log(B_SD_BOUNDS[1])), LOGIT_BOUNDS, EPS_ND_BOUNDS]
    return ([LOGIT_BOUNDS, LOGIT_BOUNDS] if free else []) + core


class Objective:
    """The cached joint objective at one branch and generation.  ``free``
    selects the parametrization (fractions at the anchors, or free on the
    simplex with the anchor terms); ``discrepancy`` is the fractional
    model-discrepancy scale per bin."""

    def __init__(self, templates: Templates, anchors_: dict, free: bool = False, discrepancy: float = DISCREPANCY):
        self.t = templates
        self.anchors = anchors_
        self.free = bool(free)
        self.discrepancy = float(discrepancy)
        tab = templates.tables
        p = tab["proton"]; pb = tab["antiproton"]
        key = lambda t_, i: (round(float(t_.xf_centers[t_.bin_index_xf[i]]), 6),  # noqa: E731
                             round(float(t_.pt_centers[t_.bin_index_pt[i]]), 6))
        p_index = {key(p, i): i for i in range(p.data.size)}
        self.common_p = np.asarray([p_index[key(pb, j)] for j in range(pb.data.size)], dtype=np.int64)
        mask = np.ones(p.data.size, dtype=bool); mask[self.common_p] = False
        self.p_only = np.flatnonzero(mask)
        dz = self.discrepancy
        parts_d = [p.data[self.common_p] - pb.data, p.data[self.p_only]]
        parts_v = [p.sigma[self.common_p] ** 2 + pb.sigma ** 2 + (dz * p.data[self.common_p]) ** 2 + (dz * pb.data) ** 2,
                   p.sigma[self.p_only] ** 2 + (dz * p.data[self.p_only]) ** 2]
        self.segments = {"net_proton": (0, self.common_p.size), "proton_only": (self.common_p.size, self.common_p.size + self.p_only.size)}
        offset = self.common_p.size + self.p_only.size
        for name in PION_FIT_SPECIES:
            t_ = tab[name]
            parts_d.append(t_.data)
            parts_v.append(t_.sigma ** 2 + (dz * t_.data) ** 2)
            self.segments[name] = (offset, offset + t_.data.size)
            offset += t_.data.size
        self.d = np.concatenate(parts_d)
        self.v_data = np.concatenate(parts_v)
        self.areas = {name: OB.bin_areas(tab[name]) for name in ("proton", "antiproton") + PION_FIT_SPECIES}
        self.n_obs = int(self.d.size)
        self.cache: dict = {}
        self.se_cache: dict = {}

    def physical(self, x) -> dict:
        return physical_from_x(x, self.free, self.anchors)

    def class_weights(self, eps: float, b_sd: float, eps_nd: float, blocks: list[int] | None = None, remnant=None) -> tuple[dict, dict]:
        """Per class per block the normalized weights and the ESS fraction."""
        rkey = None if remnant is None else tuple(round(float(v), 12) for v in remnant)
        key = (round(eps, 12), round(b_sd, 12), round(eps_nd, 12), tuple(blocks) if blocks is not None else None, rkey)
        if key in self.cache:
            return self.cache[key]
        weights = {}; ess = {}
        for cls in CLASS_ORDER:
            weights[cls] = []; ess_vals = []
            for k, block in enumerate(self.t.by_class[cls]):
                if blocks is not None and k not in blocks:
                    weights[cls].append(None); continue
                log_w = self.t.log_weights(cls, block, eps, b_sd, eps_nd, remnant if cls == "nd" else None)
                weights[cls].append(self.t.normalized(log_w))
                ess_vals.append(ess_fraction(log_w))
            ess[cls] = min(ess_vals) if ess_vals else 0.0
        self.cache[key] = (weights, ess)
        while len(self.cache) > WEIGHT_CACHE_ENTRIES:
            self.cache.pop(next(iter(self.cache)))
        return weights, ess

    def block_prediction(self, k: int, frac: dict, weights: dict) -> dict:
        """The invariant cross sections of the fitted species in replica block k."""
        f = {"sd_projectile": frac["f_sd_total"] / 2.0, "sd_target": frac["f_sd_total"] / 2.0,
             "dd": frac["f_dd"], "nd": frac["f_nd"]}
        out = {}
        for name, area in self.areas.items():
            m = np.zeros(area.size)
            for cls in CLASS_ORDER:
                m += f[cls] * self.t.species_mean(cls, self.t.by_class[cls][k], weights[cls][k], name)
            out[name] = m / (math.pi * SQRT_S * area)
        return out

    def observation_vector(self, pred: dict) -> np.ndarray:
        return np.concatenate([pred["proton"][self.common_p] - pred["antiproton"], pred["proton"][self.p_only]]
                              + [pred[name] for name in PION_FIT_SPECIES])

    def unweighted_se(self, phys: dict, block_ids: list[int]) -> np.ndarray:
        """The between-block standard error of the mixture at the trial
        fractions with unit shape and remnant weights (cached)."""
        key = ("se", round(phys["f_sd_total"], 12), round(phys["f_dd"], 12), tuple(block_ids))
        if key in self.se_cache:
            return self.se_cache[key]
        if not hasattr(self, "_unit_weights"):
            self._unit_weights = {cls: [np.full(block["n"], 1.0 / block["n"]) for block in self.t.by_class[cls]] for cls in CLASS_ORDER}
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, phys, self._unit_weights)) for k in block_ids])
        se = np.std(vectors, axis=0, ddof=1) / math.sqrt(vectors.shape[0]) if vectors.shape[0] > 1 else np.zeros(vectors.shape[1])
        se = np.maximum(se, SE_FLOOR)
        self.se_cache[key] = se
        while len(self.se_cache) > SE_CACHE_ENTRIES:
            self.se_cache.pop(next(iter(self.se_cache)))
        return se

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        phys = self.physical(x)
        if (not (EPS_BOUNDS[0] <= phys["epsilon"] <= EPS_BOUNDS[1]) or not (B_SD_BOUNDS[0] <= phys["b_sd"] <= B_SD_BOUNDS[1])
                or not (EPS_ND_BOUNDS[0] <= phys["epsilon_nd"] <= EPS_ND_BOUNDS[1])):
            return (float("inf"), None) if details else float("inf")
        weights, ess = self.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"], blocks,
                                          remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
        ess_min = min(ess.values())
        block_ids = list(range(self.t.n_blocks)) if blocks is None else list(blocks)
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, phys, weights)) for k in block_ids])
        m = np.mean(vectors, axis=0)
        se = self.unweighted_se(phys, block_ids)
        total_var = self.v_data + se ** 2
        r = (self.d - m) / np.sqrt(total_var)
        design = m / np.sqrt(total_var)
        normal = float(design @ design) + 1.0 / NORM_PRIOR ** 2
        rhs = float(design @ r)
        eta = rhs / normal
        profiled = float(r @ r) - rhs * eta
        a = self.anchors
        anchor_terms = 0.0
        if self.free:
            anchor_terms = ((phys["f_sd_total"] - a["f_sd_total"]["value"]) / a["f_sd_total"]["sigma"]) ** 2 \
                + ((phys["f_dd"] - a["f_dd"]["value"]) / a["f_dd"]["sigma"]) ** 2
        value = profiled + anchor_terms
        if ess_min < ESS_FLOOR:
            value += TRUST_PENALTY * (ESS_FLOOR - ess_min) / ESS_FLOOR
        if not details:
            return value
        pulls = r - eta * design
        by_segment = {name: float(np.sum(pulls[lo:hi] ** 2)) for name, (lo, hi) in self.segments.items()}
        return value, {
            "physical": phys, "ess": ess, "ess_min": ess_min, "eta_hat": eta,
            "minus2lnl_data": profiled, "anchor_terms": anchor_terms, "chi2_by_segment": by_segment,
            "m": m, "se": se, "residual_pull": pulls, "blocks": block_ids,
            "in_trust_region": bool(ess_min >= ESS_FLOOR), "free_fractions": self.free,
        }


# ----------------------------------------------------------------------
# The block-parallel objective (R51i)
# ----------------------------------------------------------------------
def _class_obs(objective: Objective, weights: dict) -> np.ndarray:
    """Per block and class the observation vector of the class's mean
    per-event content at the given weights: shape (blocks, classes, obs)."""
    t = objective.t
    out = np.zeros((t.n_blocks, len(CLASS_ORDER), objective.n_obs))
    for k in range(t.n_blocks):
        for c, cls in enumerate(CLASS_ORDER):
            block = t.by_class[cls][k]
            wn = weights[cls][k]
            pred = {name: t.species_mean(cls, block, wn, name) / (math.pi * SQRT_S * area) for name, area in objective.areas.items()}
            out[k, c] = objective.observation_vector(pred)
    return out


def block_reply(objective: Objective, msg: tuple) -> tuple:
    """A block worker's answer to ("unit",) (the unit-weight observation
    vectors, for the between-block error) or ("eval", eps, b_sd, eps_nd,
    remnant) (the reweighted vectors and the per-class smallest ESS)."""
    if msg[0] == "unit":
        unit = {cls: [np.full(b["n"], 1.0 / b["n"]) for b in objective.t.by_class[cls]] for cls in CLASS_ORDER}
        return ("unit", _class_obs(objective, unit))
    if msg[0] == "eval":
        _, eps, b_sd, eps_nd, remnant = msg
        weights, ess = objective.class_weights(eps, b_sd, eps_nd, remnant=remnant)
        return ("eval", _class_obs(objective, weights), ess)
    raise ValueError(f"unknown block request {msg[0]!r}")


def _block_worker_main(conn, run_dir: str, floor_index: int, generation: int, replicas: list, smoke: bool) -> None:
    """One block worker: loads its replica blocks, then answers requests
    until ("stop",)."""
    contract = _contract(Path(run_dir))
    templates = Templates(contract, floor_index, generation, load_fit_tables(), smoke=smoke, replicas=None if smoke else replicas)
    objective = Objective(templates, anchors())
    conn.send(("ready", templates.n_blocks, templates.point))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        conn.send(block_reply(objective, msg))
    conn.close()


class BlockPool:
    """Persistent worker processes that each own a contiguous set of replica
    blocks of one branch and generation (block order is replica order)."""

    def __init__(self, run_dir, floor_index: int, generation: int, n_workers: int, *, smoke: bool = False):
        # fork: the workers inherit the imported modules and need no
        # re-import of the driver (it runs as a script on the gateway)
        ctx = mp.get_context("fork")
        reps = [0] if smoke else list(range(REPLICAS))
        n = max(1, min(int(n_workers), len(reps)))
        chunks = [[int(r) for r in c] for c in np.array_split(np.asarray(reps), n)]
        self.conns, self.procs = [], []
        for chunk in chunks:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_block_worker_main, args=(child, str(run_dir), int(floor_index), int(generation), chunk, bool(smoke)))
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        ready = [c.recv() for c in self.conns]
        self.n_blocks = int(sum(r[1] for r in ready))
        if len({tuple(r[2]) for r in ready}) != 1:
            raise RuntimeError("the block workers loaded different generation points")
        self.point = tuple(ready[0][2])
        self._unit = None

    def _ask(self, msg):
        for c in self.conns:
            c.send(msg)
        return [c.recv() for c in self.conns]

    def unit(self) -> np.ndarray:
        if self._unit is None:
            self._unit = np.concatenate([r[1] for r in self._ask(("unit",))], axis=0)
        return self._unit

    def evaluate(self, eps: float, b_sd: float, eps_nd: float, remnant) -> tuple[np.ndarray, dict]:
        replies = self._ask(("eval", float(eps), float(b_sd), float(eps_nd), tuple(float(v) for v in remnant)))
        obs = np.concatenate([r[1] for r in replies], axis=0)
        ess = {cls: min(float(r[2][cls]) for r in replies) for cls in CLASS_ORDER}
        return obs, ess

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=120)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class PooledObjective(Objective):
    """The joint objective of :class:`Objective` with the per-block work
    done by a :class:`BlockPool` (always every block)."""

    def __init__(self, tables: dict, anchors_: dict, pool: BlockPool, free: bool = False, discrepancy: float = DISCREPANCY):
        stub = types.SimpleNamespace(tables=tables, n_blocks=pool.n_blocks, point=pool.point)
        super().__init__(stub, anchors_, free=free, discrepancy=discrepancy)
        self.pool = pool
        self.obs_cache: dict = {}

    def class_obs(self, eps: float, b_sd: float, eps_nd: float, remnant) -> tuple[np.ndarray, dict]:
        key = (round(eps, 12), round(b_sd, 12), round(eps_nd, 12), tuple(round(float(v), 12) for v in remnant))
        if key in self.obs_cache:
            return self.obs_cache[key]
        value = self.pool.evaluate(eps, b_sd, eps_nd, remnant)
        self.obs_cache[key] = value
        while len(self.obs_cache) > WEIGHT_CACHE_ENTRIES:
            self.obs_cache.pop(next(iter(self.obs_cache)))
        return value

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        if blocks is not None:
            raise ValueError("the pooled objective always uses every block")
        phys = self.physical(x)
        if (not (EPS_BOUNDS[0] <= phys["epsilon"] <= EPS_BOUNDS[1]) or not (B_SD_BOUNDS[0] <= phys["b_sd"] <= B_SD_BOUNDS[1])
                or not (EPS_ND_BOUNDS[0] <= phys["epsilon_nd"] <= EPS_ND_BOUNDS[1])):
            return (float("inf"), None) if details else float("inf")
        obs, ess = self.class_obs(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"],
                                  (phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
        ess_min = min(ess.values())
        frac = np.asarray([phys["f_sd_total"] / 2.0, phys["f_sd_total"] / 2.0, phys["f_dd"], phys["f_nd"]])
        vectors = np.einsum("c,kco->ko", frac, obs)
        n = vectors.shape[0]
        m = np.mean(vectors, axis=0)
        unit = np.einsum("c,kco->ko", frac, self.pool.unit())
        se = np.std(unit, axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(unit.shape[1])
        se = np.maximum(se, SE_FLOOR)
        total_var = self.v_data + se ** 2
        r = (self.d - m) / np.sqrt(total_var)
        design = m / np.sqrt(total_var)
        normal = float(design @ design) + 1.0 / NORM_PRIOR ** 2
        rhs = float(design @ r)
        eta = rhs / normal
        profiled = float(r @ r) - rhs * eta
        a = self.anchors
        anchor_terms = 0.0
        if self.free:
            anchor_terms = ((phys["f_sd_total"] - a["f_sd_total"]["value"]) / a["f_sd_total"]["sigma"]) ** 2 \
                + ((phys["f_dd"] - a["f_dd"]["value"]) / a["f_dd"]["sigma"]) ** 2
        value = profiled + anchor_terms
        if ess_min < ESS_FLOOR:
            value += TRUST_PENALTY * (ESS_FLOOR - ess_min) / ESS_FLOOR
        if not details:
            return value
        pulls = r - eta * design
        by_segment = {name: float(np.sum(pulls[lo:hi] ** 2)) for name, (lo, hi) in self.segments.items()}
        return value, {
            "physical": phys, "ess": ess, "ess_min": ess_min, "eta_hat": eta,
            "minus2lnl_data": profiled, "anchor_terms": anchor_terms, "chi2_by_segment": by_segment,
            "m": m, "se": se, "residual_pull": pulls, "blocks": list(range(n)),
            "in_trust_region": bool(ess_min >= ESS_FLOOR), "free_fractions": self.free,
        }


def dispersed_starts(anchors_: dict, free: bool) -> list[np.ndarray]:
    a_sd, a_dd = anchors_["f_sd_total"]["value"], anchors_["f_dd"]["value"]
    shapes = [(GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND), (GEN0_EPSILON, GEN0_B_SD, -0.7), (0.0, 5.0, -0.5), (-0.3, 3.0, -0.8),
              (0.1, 8.0, -1.2), (-0.4, 2.0, -0.3), (0.2, 4.0, 0.0), (-0.2, 6.0, -1.4)]
    excited = [GEN0_W_EXCITED, 0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6]
    fractions = [(a_sd, a_dd), (0.30, 0.10), (0.10, 0.02), (0.40, 0.20), (a_sd, a_dd), (0.15, 0.05), (0.25, 0.15), (0.35, 0.03)]
    n = N_STARTS if not free else N_STARTS_VARIANT
    starts = []
    for (fs, fd), (e, b, en), we in list(zip(fractions, shapes, excited))[:n]:
        phys = {"f_sd_total": fs, "f_dd": fd, "epsilon": e, "b_sd": b, "w_excited": we, "epsilon_nd": en}
        x = x_from_physical(phys, free)
        idx = [0, 1, 4] if free else [2]
        x[idx] = np.clip(x[idx], LOGIT_BOUNDS[0] + 0.5, LOGIT_BOUNDS[1] - 0.5)
        starts.append(x)
    return starts


def run_fit(objective: Objective, *, starts: list[np.ndarray], blocks: list[int] | None = None,
            maxfev: int = 4000) -> dict:
    from scipy.optimize import minimize
    bounds = bounds_for(objective.free)
    best = None
    records = []
    for x0 in starts:
        res = minimize(lambda x: objective.evaluate(x, blocks), np.asarray(x0, dtype=float), method="Powell",
                       bounds=bounds, options={"xtol": 1e-4, "ftol": 1e-7, "maxfev": maxfev})
        records.append({"x0": [float(v) for v in x0], "x": [float(v) for v in res.x], "fun": float(res.fun),
                        "nfev": int(res.nfev), "success": bool(res.success), "message": str(res.message)})
        if best is None or res.fun < best.fun:
            best = res
    return {"x": np.asarray(best.x, dtype=float), "fun": float(best.fun), "converged": bool(best.success),
            "nfev": int(best.nfev), "message": str(best.message), "starts": records}


def _fit_worker(run_dir: str, floor_index: int, generation: int, x0: list, blocks, maxfev: int, free: bool,
                discrepancy: float, pool_workers: int = BLOCK_WORKERS_PER_START) -> dict:
    """One optimizer start in its own process with its own block pool, or
    one replica-block fit (``blocks = [k]``) that loads that block alone."""
    contract = _contract(Path(run_dir))
    tables = load_fit_tables()
    if blocks is None:
        with BlockPool(run_dir, floor_index, generation, pool_workers) as pool:
            objective = PooledObjective(tables, anchors(), pool, free=free, discrepancy=discrepancy)
            return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)
    templates = Templates(contract, floor_index, generation, tables, replicas=blocks)
    objective = Objective(templates, anchors(), free=free, discrepancy=discrepancy)
    return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)


def run_fit_parallel(contract: RC.RunContract, floor_index: int, generation: int, starts: list, *, free: bool,
                     discrepancy: float = DISCREPANCY, blocks=None, maxfev: int = 4000, workers: int | None = None,
                     pool_workers: int = BLOCK_WORKERS_PER_START) -> dict:
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x0], blocks, int(maxfev), bool(free),
             float(discrepancy), int(pool_workers)) for x0 in starts]
    results = RC.run_replicas(_fit_worker, jobs, max_workers=workers or len(jobs))
    records = [r["starts"][0] for r in results]
    best = min(results, key=lambda r: r["fun"])
    return {"x": np.asarray(best["x"], dtype=float), "fun": float(best["fun"]), "converged": bool(best["converged"]),
            "nfev": int(best["nfev"]), "message": best["message"], "starts": records}


def hessian_physical(objective: Objective, phys: dict) -> dict:
    """Central-difference Hessian of the objective in the fitted physical
    parameters (the fractions only when they are free)."""
    names = param_names(objective.free)
    p0 = np.asarray([phys[n] for n in names], dtype=float)
    steps = np.asarray([HESSIAN_STEPS[n] for n in names], dtype=float)

    def value(p):
        trial = dict(phys)
        trial.update({n: float(v) for n, v in zip(names, p)})
        if not (0.0 < trial["w_excited"] < 1.0):
            return float("inf")
        if not (B_SD_BOUNDS[0] < trial["b_sd"] < B_SD_BOUNDS[1]):
            return float("inf")
        if objective.free and (trial["f_sd_total"] <= 0 or trial["f_dd"] <= 0 or trial["f_sd_total"] + trial["f_dd"] >= 1):
            return float("inf")
        return objective.evaluate(x_from_physical(trial, objective.free))

    n = p0.size
    h = np.zeros((n, n))
    f0 = value(p0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = steps[i]
        h[i, i] = (value(p0 + ei) - 2.0 * f0 + value(p0 - ei)) / steps[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = steps[j]
            h[i, j] = h[j, i] = (value(p0 + ei + ej) - value(p0 + ei - ej) - value(p0 - ei + ej) + value(p0 - ei - ej)) / (4.0 * steps[i] * steps[j])
    finite = bool(np.all(np.isfinite(h)))
    eig = np.linalg.eigvalsh(h) if finite else np.full(n, np.nan)
    pd = bool(finite and np.all(eig > 0.0))
    errors = None
    if pd:
        cov = 2.0 * np.linalg.inv(h)
        errors = {name: float(math.sqrt(max(cov[i, i], 0.0))) for i, name in enumerate(names)}
    return {"names": list(names), "steps": [float(s) for s in steps], "matrix": h.tolist() if finite else None,
            "eigenvalues": [float(e) for e in eig] if finite else None, "positive_definite": pd, "quoted": pd,
            "errors": errors}


def _mixture_readouts(templates: Templates, objective: Objective, tables: dict, phys: dict) -> dict:
    """The held-out readouts at a fitted point: paper-window sum ratios, total
    yields, the pi- p_T profile, the SD n_ch against M_X, and predictions."""
    weights, _ = objective.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"],
                                         remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
    frac_map = {"sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0, "dd": phys["f_dd"], "nd": phys["f_nd"]}
    model_bins = {}
    readout = {}
    for name in TRACK_SPECIES:
        if name not in tables:
            continue
        table = tables[name]
        mb = np.zeros(table.data.size)
        for cls in CLASS_ORDER:
            for k, block in enumerate(templates.by_class[cls]):
                mb += frac_map[cls] * templates.species_mean(cls, block, weights[cls][k], name) / templates.n_blocks
        mb = mb / (math.pi * SQRT_S * OB.bin_areas(table))
        model_bins[name] = mb
        paper = np.asarray([table.xf_centers[table.bin_index_xf[i]] <= PAPER_WINDOW["xf_max"] + 1e-9 for i in range(table.data.size)])
        readout[name] = {"bins": int(np.sum(paper)), "model_sum": float(np.sum(mb[paper])),
                         "data_sum": float(np.sum(table.data[paper])),
                         "sum_ratio": float(np.sum(mb[paper]) / np.sum(table.data[paper])),
                         "chi2_data_only": float(np.sum(((table.data[paper] - mb[paper]) / table.sigma[paper]) ** 2))}
    # the pi- p_T profile (x_F <= 0.5): model/data at p_T_high over p_T_low
    t_ = tables[PT_PROFILE["species"]]; mb = model_bins[PT_PROFILE["species"]]
    xfc = np.asarray(t_.xf_centers)[t_.bin_index_xf]; ptc = np.asarray(t_.pt_centers)[t_.bin_index_pt]
    ratios = {}
    for label in ("pt_low", "pt_high"):
        sel = np.isclose(ptc, PT_PROFILE[label]) & (xfc <= PT_PROFILE["xf_max"] + 1e-9)
        ratios[label] = float(np.sum(mb[sel]) / np.sum(t_.data[sel])) if np.any(sel) else float("nan")
    pt_profile = {**PT_PROFILE, "ratio_low": ratios["pt_low"], "ratio_high": ratios["pt_high"],
                  "profile_ratio": ratios["pt_high"] / ratios["pt_low"]}
    reference = RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "na49_pp_total_yields.json")
    yields = {name: 0.0 for name in TRACK_SPECIES}; mean_nch = 0.0; p_proton = 0.0
    for cls in CLASS_ORDER:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            for name in TRACK_SPECIES:
                yields[name] += frac_map[cls] * float(np.sum(wn * block["totals"][name]))
            mean_nch += frac_map[cls] * float(np.sum(wn * block["n_ch"]))
            p_proton += frac_map[cls] * float(np.sum(wn * np.isfinite(block["lead_xf"])))
    ref_y = reference["yields_per_inelastic_event"]
    total_yields = {"model": {**{name: yields[name] for name in TRACK_SPECIES}, "charged_multiplicity": mean_nch, "events_with_a_proton": p_proton},
                    "data": ref_y,
                    "ratio_model_over_data": {name: yields[name] / float(ref_y[name]["value"]) for name in TRACK_SPECIES if name in ref_y},
                    "reference_sha256": RC.sha256_file(NESTED_ROOT / "config" / "v2_singlet" / "na49_pp_total_yields.json")}
    # the single-diffraction event n_ch against the excitation mass (a prediction)
    sums = np.zeros(SD_MX_EDGES.size - 1); norms = np.zeros(SD_MX_EDGES.size - 1)
    for cls in ("sd_projectile", "sd_target"):
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k]
            mx = np.sqrt(np.asarray(block["shape"]["m2"], dtype=float))
            idx = np.digitize(mx, SD_MX_EDGES) - 1
            ok = (idx >= 0) & (idx < sums.size)
            sums += np.bincount(idx[ok], weights=wn[ok] * block["n_ch"][ok], minlength=sums.size)[: sums.size]
            norms += np.bincount(idx[ok], weights=wn[ok], minlength=sums.size)[: sums.size]
    sd_nch = {"edges": SD_MX_EDGES.tolist(),
              "mean_event_n_ch": [float(s / n) if n > 0 else None for s, n in zip(sums, norms)],
              "note": "whole-event charged count of single-diffraction events (the excited system plus the other line, an intact proton or a low-mass resonance by the frozen split)"}
    predictions = {"lead_xf": np.zeros(LEAD_XF_EDGES.size - 1), "gap": np.zeros(GAP_EDGES.size - 1), "n_ch": np.zeros(NCH_MAX + 1)}
    for cls in CLASS_ORDER:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            predictions["lead_xf"] += frac_map[cls] * templates.histogram_mean(block, wn, "lead_xf", LEAD_XF_EDGES)
            predictions["gap"] += frac_map[cls] * templates.histogram_mean(block, wn, "gap", GAP_EDGES)
            predictions["n_ch"] += frac_map[cls] * templates.nch_mean(block, wn)
    # the pi0 yield and the neutron dN/dx_F (held out)
    n_pi0 = 0.0
    neutron = np.zeros(NEUTRON_XF_EDGES.size - 1)
    for cls in CLASS_ORDER:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            n_pi0 += frac_map[cls] * float(np.sum(wn * block["n_pi0"]))
            if block["neu_xf"].size:
                idx = np.searchsorted(NEUTRON_XF_EDGES, block["neu_xf"]) - 1
                ok = (idx >= 0) & (idx < neutron.size)
                neutron += frac_map[cls] * np.bincount(idx[ok], weights=wn[block["neu_event"][ok]], minlength=neutron.size)[: neutron.size]
    neutron = neutron / np.diff(NEUTRON_XF_EDGES)
    rows = [l.split() for l in (NESTED_ROOT / NEUTRON_TABLE_REL).read_text().splitlines() if l.strip() and not l.startswith("#")]
    neutron_readout = {"table": NEUTRON_TABLE_REL, "edges": NEUTRON_XF_EDGES.tolist(),
                       "rows": [{"xf": float(r[0]), "model": float(m), "data": float(r[1]), "data_error": float(r[1]) * float(r[2]) / 100.0,
                                 "ratio": float(m) / float(r[1])} for r, m in zip(rows, neutron)],
                       "note": "dN/dx_F per inelastic event in the forward hemisphere (x_F > 0)"}
    charged_mean = 0.5 * (yields["pi_plus"] + yields["pi_minus"])
    pi0_readout = {"pi0_per_event": n_pi0, "charged_pion_mean": charged_mean, "ratio": n_pi0 / charged_mean if charged_mean > 0 else float("nan"),
                   "neutral_share": n_pi0 / (n_pi0 + 2.0 * charged_mean) if charged_mean > 0 else float("nan")}
    return {"readout_paper_window_sum_ratios": readout, "readout_total_yields": total_yields,
            "readout_pt_profile": pt_profile, "readout_sd_nch_vs_mx": sd_nch,
            "readout_pi0": pi0_readout, "readout_neutron_dndxf": neutron_readout,
            "predictions": {"lead_xf": {"edges": LEAD_XF_EDGES.tolist(), "density": predictions["lead_xf"].tolist()},
                            "gap": {"edges": GAP_EDGES.tolist(), "density": predictions["gap"].tolist()},
                            "n_ch": {"values": list(range(NCH_MAX + 1)), "density": predictions["n_ch"].tolist()}}}


def evaluate_generation(contract: RC.RunContract, floor_index: int, generation: int, tables: dict, anchors_: dict, *,
                        smoke: bool = False) -> dict:
    starts = dispersed_starts(anchors_, free=False)[: (2 if smoke else N_STARTS)]
    master_pool = None
    if smoke:
        templates = Templates(contract, floor_index, generation, tables, smoke=smoke)
        objective = Objective(templates, anchors_, free=False)
        pooled = run_fit(objective, starts=starts, maxfev=600)
    else:
        pooled = run_fit_parallel(contract, floor_index, generation, starts, free=False, maxfev=4000)
        master_pool = BlockPool(contract.run_dir, floor_index, generation, MASTER_BLOCK_WORKERS)
        objective = PooledObjective(tables, anchors_, master_pool, free=False)
    value, det = objective.evaluate(pooled["x"], details=True)
    phys = det["physical"]
    hess = hessian_physical(objective, phys) if not smoke else {"quoted": False, "positive_definite": False, "errors": None, "skipped": "smoke"}
    blocks = []
    n_blocks = objective.t.n_blocks
    if not smoke and n_blocks > 1:
        block_jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in pooled["x"]], [k], 2000, False, DISCREPANCY, 0)
                      for k in range(n_blocks)]
        block_results = RC.run_replicas(_fit_worker, block_jobs, max_workers=n_blocks)
        for k, res in enumerate(block_results):
            blocks.append({"block": k, **physical_from_x(np.asarray(res["x"]), False, anchors_), "fun": res["fun"], "converged": res["converged"]})
    stability = {}
    if blocks and hess.get("errors"):
        for name in param_names(False):
            vals = np.asarray([b[name] for b in blocks])
            spread = float(np.std(vals, ddof=1))
            expected = hess["errors"][name] * math.sqrt(len(blocks))
            ratio = spread / expected if expected > 0 else float("inf")
            stability[name] = {"block_values": [float(v) for v in vals], "spread": spread,
                               "expected_spread": expected, "ratio": ratio, "stable": bool(ratio <= STABILITY_RATIO_BOUND)}
    interior = bool(FRACTION_INTERIOR[0] <= phys["w_excited"] <= FRACTION_INTERIOR[1])
    on_exponent_bound = (abs(phys["epsilon"] - EPS_BOUNDS[0]) < 1e-3 or abs(phys["epsilon"] - EPS_BOUNDS[1]) < 1e-3
                         or abs(phys["epsilon_nd"] - EPS_ND_BOUNDS[0]) < 1e-3 or abs(phys["epsilon_nd"] - EPS_ND_BOUNDS[1]) < 1e-3)
    residuals = []
    if not pooled["converged"]:
        residuals.append({"kind": "optimizer_non_convergence", "message": pooled["message"]})
    if det["ess_min"] < ESS_FLOOR:
        residuals.append({"kind": "weight_degeneracy", "ess_min": det["ess_min"], "floor": ESS_FLOOR})
    if not interior:
        residuals.append({"kind": "fraction_on_boundary", "w_excited": phys["w_excited"]})
    if on_exponent_bound:
        residuals.append({"kind": "exponent_on_boundary", "epsilon": phys["epsilon"], "epsilon_nd": phys["epsilon_nd"],
                          "bounds": {"epsilon": EPS_BOUNDS, "epsilon_nd": EPS_ND_BOUNDS}})
    if not hess.get("positive_definite"):
        residuals.append({"kind": "block_curvature_failure", "eigenvalues": hess.get("eigenvalues")})
    for name, s in stability.items():
        if not s["stable"]:
            residuals.append({"kind": "replica_block_minima_disagree", "parameter": name, "ratio": s["ratio"]})
    regenerate = (not smoke) and det["ess_min"] < ESS_INTERIOR and generation < MAX_GENERATIONS - 1
    constrained = bool(pooled["converged"] and det["ess_min"] >= ESS_FLOOR and interior and not on_exponent_bound
                       and hess.get("positive_definite") and stability and all(s["stable"] for s in stability.values()))
    if smoke:
        variant_obj = Objective(templates, anchors_, free=True)
    else:
        variant_obj = PooledObjective(tables, anchors_, master_pool, free=True)
    v_starts = dispersed_starts(anchors_, free=True)[: (1 if smoke else N_STARTS_VARIANT)]
    if smoke:
        v_fit = run_fit(variant_obj, starts=v_starts, maxfev=400)
    else:
        v_fit = run_fit_parallel(contract, floor_index, generation, v_starts, free=True, maxfev=4000,
                                 pool_workers=BLOCK_WORKERS_PER_VARIANT_START)
    v_value, v_det = variant_obj.evaluate(v_fit["x"], details=True)
    if master_pool is not None:
        master_pool.close()
        # the readouts need every block in process once
        templates = Templates(contract, floor_index, generation, tables)
        objective = Objective(templates, anchors_, free=False)
    variant = {"fitted": v_det["physical"], "objective": {"value": v_value, "minus2lnl_data": v_det["minus2lnl_data"],
                                                          "anchor_terms": v_det["anchor_terms"], "ess_min": v_det["ess_min"],
                                                          "chi2_by_segment": v_det["chi2_by_segment"]},
               "converged": v_fit["converged"], "nfev": v_fit["nfev"], "starts": len(v_starts),
               "note": "fractions free on the simplex with the two anchor terms; reported next to the primary fit"}
    readouts = _mixture_readouts(templates, objective, tables, phys)
    mesonic = mesonic_count_readout(FLOOR_GRID[floor_index], SMOKE_MESONIC_DECAYS if smoke else MESONIC_DECAYS)
    return {
        "floor_index": int(floor_index), "continuation_floor": float(FLOOR_GRID[floor_index]),
        "generation": int(generation),
        "generation_point": {"epsilon": templates.point[0], "b_sd": templates.point[1], "epsilon_nd": templates.point[2]},
        "statistics": {"classes": list(CLASS_ORDER), "replicas": templates.n_blocks,
                       "events_per_slot": int(templates.by_class[CLASS_ORDER[0]][0]["n"]), "tier": "T2" if not smoke else "smoke"},
        "observable_set": {"segments": {k: [int(a), int(b)] for k, (a, b) in objective.segments.items()}, "total": objective.n_obs,
                           "proton_window": PROTON_WINDOW, "pion_window": PION_WINDOW, "antiproton_window": PAPER_WINDOW,
                           "normalization_prior": NORM_PRIOR, "discrepancy": DISCREPANCY, "anchors": anchors_},
        "parametrization": {"primary": "fractions fixed to the anchors; w_intact = 0; free: " + ", ".join(SHAPE_NAMES),
                            "variant": "fractions free on the simplex with the anchor terms; free: " + ", ".join(param_names(True))},
        "fitted": {**phys, "sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0,
                   "remnant_weights": {"intact_proton": phys["w_intact"], "low_mass_baryonic_resonance": phys["w_low_mass"],
                                       "high_mass_baryonic_continuum": phys["w_excited"]},
                   "fractions_fixed_to_anchors": True},
        "objective_at_optimum": {"value": value, "minus2lnl_data": det["minus2lnl_data"], "anchor_terms": det["anchor_terms"],
                                 "chi2_by_segment": det["chi2_by_segment"], "eta_hat": det["eta_hat"], "ess": det["ess"],
                                 "ess_min": det["ess_min"], "in_trust_region": det["in_trust_region"]},
        "optimizer": {"method": "Powell (bounded)", "starts": len(starts), "converged": pooled["converged"], "nfev": pooled["nfev"],
                      "message": pooled["message"], "start_records": pooled["starts"]},
        "hessian": hess, "block_fits": blocks, "replica_block_stability": stability,
        "variant_free_fractions": variant,
        "verdict": {"constrained": constrained, "regeneration_requested": regenerate, "interior_fractions": interior,
                    "exponent_on_boundary": on_exponent_bound, "ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR},
        "named_residual_register": residuals,
        **readouts,
        "readout_mesonic_count": mesonic,
        "per_bin": {"d": objective.d.tolist(), "m": det["m"].tolist(), "se": det["se"].tolist(),
                    "v_data": objective.v_data.tolist(), "pull": det["residual_pull"].tolist()},
        "determinism": {"note": "no wall clock in this record; the slot records carry their own timing"},
    }


# ----------------------------------------------------------------------
# The profile over the floor grid, the acceptance, and the final report
# ----------------------------------------------------------------------
def acceptance(branch: dict) -> dict:
    """The pre-registered Block B stage 2 acceptance at the chosen floor."""
    ratios = branch["readout_total_yields"]["ratio_model_over_data"]
    rows = {round(r["mass"], 3): r for r in branch["readout_mesonic_count"]["rows"]}
    pions_ok = bool(ratios["pi_minus"] >= 0.72 and ratios["pi_plus"] >= 0.72)
    count_ok = bool(all(abs(rows[m]["ratio"] - 1.0) <= 0.15 for m in (3.0, 3.5)))
    profile = branch["readout_pt_profile"]["profile_ratio"]
    profile_ok = bool(math.isfinite(profile) and profile < 3.0)
    return {"criteria": {
                "pion_total_yields": {"rule": "pi- and pi+ total-yield ratios >= 0.72", "values": {"pi_minus": ratios["pi_minus"], "pi_plus": ratios["pi_plus"]}, "met": pions_ok},
                "mesonic_count": {"rule": "within 15% of MARK I at 3.0 and 3.5 GeV", "values": {str(m): rows[m]["ratio"] for m in (3.0, 3.5)}, "met": count_ok},
                "pt_profile": {"rule": "pi- model/data at p_T 1.3 over 0.3 GeV/c below 3", "value": profile, "met": profile_ok}},
            "accepted": bool(pions_ok and count_ok and profile_ok)}


def compose_report(branches: list[dict], chosen: dict, variant_disc: dict | None) -> dict:
    """The final report: the chosen branch's fit, the floor profile, the
    discrepancy variant, the acceptance, and the verdict."""
    profile = [{"floor_index": b["floor_index"], "continuation_floor": b["continuation_floor"], "generation": b["generation"],
                "objective": b["objective_at_optimum"]["value"], "chi2_by_segment": b["objective_at_optimum"]["chi2_by_segment"],
                "fitted": {k: b["fitted"][k] for k in ("epsilon", "b_sd", "epsilon_nd", "w_excited")},
                "ess_min": b["objective_at_optimum"]["ess_min"], "branch_digest": b.get("digest")} for b in branches]
    idx = int(chosen["floor_index"])
    floor_interior = 0 < idx < N_FLOORS - 1
    residuals = list(chosen["named_residual_register"])
    if not floor_interior:
        residuals.append({"kind": "floor_on_grid_edge", "continuation_floor": chosen["continuation_floor"], "grid": list(FLOOR_GRID)})
    report = {k: v for k, v in chosen.items() if k not in ("digest", "schema", "run", "card", "driver", "driver_sha256")}
    report["floor_profile"] = {"grid": list(FLOOR_GRID), "rows": profile, "chosen_floor_index": idx,
                               "chosen_floor": chosen["continuation_floor"], "interior": floor_interior}
    report["discrepancy_variant"] = variant_disc
    report["acceptance_block_b_stage2"] = acceptance(chosen)
    report["named_residual_register"] = residuals
    report["verdict"] = {**chosen["verdict"], "floor_interior": floor_interior,
                         "constrained": bool(chosen["verdict"]["constrained"] and floor_interior), "regeneration_requested": False}
    return report


def render_report_md(report: dict) -> str:
    f = report["fitted"]; o = report["objective_at_optimum"]; v = report["verdict"]
    lines = [f"# R51i: fine floor scan of the joint proton and pion topology fit ({RUN_ID})", "",
             f"Card {CARD}. Chosen floor {report['continuation_floor']} GeV (branch {report['floor_index']}, generation {report['generation']}).",
             "Primary fit: fractions fixed to the external anchors, intact remnant weight fixed at 0.", "",
             "## Floor profile", "", "| floor [GeV] | objective | net p | p only | pi- | pi+ | eps | eps_nd | B_SD | w_exc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["floor_profile"]["rows"]:
        c = row["chi2_by_segment"]; ft = row["fitted"]
        lines.append(f"| {row['continuation_floor']} | {row['objective']:.1f} | {c['net_proton']:.1f} | {c['proton_only']:.1f} | {c['pi_minus']:.1f} | {c['pi_plus']:.1f} | {ft['epsilon']:.3f} | {ft['epsilon_nd']:.3f} | {ft['b_sd']:.2f} | {ft['w_excited']:.3f} |")
    lines += ["", "| parameter | value | error |", "|---|---|---|"]
    errors = (report["hessian"] or {}).get("errors") or {}
    for name in ("f_sd_total", "f_dd", "f_nd", "epsilon", "epsilon_nd", "b_sd", "w_intact", "w_low_mass", "w_excited"):
        err = errors.get(name)
        lines.append(f"| {name} | {f[name]:.6g} | {'' if err is None else f'{err:.3g}'} |")
    lines += ["", f"Objective at optimum {o['value']:.3f} (data {o['minus2lnl_data']:.3f}); eta_hat {o['eta_hat']:.4f}; smallest class ESS {o['ess_min']:.3f}.",
              f"Converged: {report['optimizer']['converged']}. Hessian positive definite: {report['hessian'].get('positive_definite')}. Floor interior: {v['floor_interior']}.",
              f"Verdict: {'CONSTRAINED' if v['constrained'] else 'NOT_CONSTRAINED'}.", "",
              "## Total yields per inelastic event (model / data)", "", "| species | model | data | ratio |", "|---|---|---|---|"]
    ty = report["readout_total_yields"]
    for name, ratio in ty["ratio_model_over_data"].items():
        lines.append(f"| {name} | {ty['model'][name]:.3f} | {ty['data'][name]['value']} | {ratio:.2f} |")
    lines.append(f"\nMean charged multiplicity {ty['model']['charged_multiplicity']:.2f}.")
    pt = report["readout_pt_profile"]
    p0 = report["readout_pi0"]
    lines += ["", f"pi- p_T profile (x_F <= {pt['xf_max']}): model/data {pt['ratio_low']:.3f} at {pt['pt_low']} GeV/c, {pt['ratio_high']:.3f} at {pt['pt_high']} GeV/c; profile ratio {pt['profile_ratio']:.2f}.",
              f"pi0 per event {p0['pi0_per_event']:.3f}; ratio to the charged-pion mean {p0['ratio']:.2f}; neutral share {p0['neutral_share']:.3f}.", "",
              "## Neutron dN/dx_F (held out; NA49 table)", "", "| x_F | model | NA49 | ratio |", "|---|---|---|---|"]
    for row in report["readout_neutron_dndxf"]["rows"]:
        lines.append(f"| {row['xf']} | {row['model']:.3f} | {row['data']:.3f} +- {row['data_error']:.3f} | {row['ratio']:.2f} |")
    lines += ["",
              "## Mesonic count per decay against MARK I", "", "| M [GeV] | model | MARK I | ratio |", "|---|---|---|---|"]
    for row in report["readout_mesonic_count"]["rows"]:
        lines.append(f"| {row['mass']} | {row['model']:.2f} | {row['mark_i']:.2f} | {row['ratio']:.3f} |")
    acc = report["acceptance_block_b_stage2"]
    lines += ["", f"## Block B stage 2 acceptance: {'ACCEPTED' if acc['accepted'] else 'NOT ACCEPTED'}", ""]
    for name, c in acc["criteria"].items():
        lines.append(f"- {name}: {c['rule']}; {json.dumps(c.get('values', c.get('value')))}; met {c['met']}")
    dv = report.get("discrepancy_variant")
    if dv:
        lines += ["", f"Discrepancy variant (delta = {dv['discrepancy']}): eps {dv['fitted']['epsilon']:.3f}, eps_nd {dv['fitted']['epsilon_nd']:.3f}, B_SD {dv['fitted']['b_sd']:.2f}, w_exc {dv['fitted']['w_excited']:.3f}; objective {dv['objective']:.1f}."]
    lines += ["", "## Named residuals", ""]
    for res in report["named_residual_register"] or [{"kind": "none"}]:
        lines.append(f"- {json.dumps(res, sort_keys=True)}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def step_pre_registration(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    ledger = RC.seed_ledger(NESTED_ROOT)
    disjoint = RC.check_seed_disjointness(ledger, LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    frozen = frozen_digests()
    r51h_dir = MONO_ROOT / "dev" / "00runs" / R51H_RUN_ID
    if not (r51h_dir / "seal_manifest.json").is_file():
        contract.refuse(f"the sealed R51h archive {r51h_dir} is required for the generation-0 point")
    r51h_report = RC.read_json(r51h_dir / RC.REPORT_NAME)
    fitted = r51h_report["fitted"]
    if (abs(float(fitted["epsilon"]) - GEN0_EPSILON) > 1e-12 or abs(float(fitted["b_sd"]) - GEN0_B_SD) > 1e-12
            or abs(float(fitted["epsilon_nd"]) - GEN0_EPSILON_ND) > 1e-12
            or abs(float(fitted["w_excited"]) - R51H_W_EXCITED) > 1e-12 or abs(float(fitted["w_low_mass"]) - R51H_W_LOW) > 1e-12):
        contract.refuse("the generation-0 constants do not match the sealed R51h optimum")
    _load_physics(FLOOR_GRID[0])  # refuses a committed config that already sets a floor
    payload = {
        "classes": list(CLASS_ORDER), "replicas": REPLICAS, "events_per_slot": EVENTS_PER_SLOT, "max_generations": MAX_GENERATIONS,
        "floor_grid": {"values_gev": list(FLOOR_GRID), "rule": "one template branch per floor; the reweightable parameters are fitted within a branch and the floor is profiled over the branch optima",
                       "set_in_memory": "dataclasses.replace(family, continuation_floor=F) for every family; the committed config sets no floor"},
        "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST,
                  "rule": "raw seed = RAW_FIRST + 32 * generation + slot (common random numbers across the floors); slot = class_index * 8 + replica",
                  "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * slot + i", "disjointness": disjoint},
        "generation_0_point": {"epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND, "w_excited": GEN0_W_EXCITED,
                               "provenance": {"run": R51H_RUN_ID, "report_digest": r51h_report.get("digest"), "rule": "the sealed R51h optimum, for every floor"}},
        "free_parameters": list(SHAPE_NAMES),
        "fixed": {"w_intact": 0.0, "reason": "on its lower bound in R51c, R51d and R51e (0.0002); the Hessian was never positive definite with it free"},
        "fractions": {"rule": "fixed to the external anchors in the primary fit", "values": {k: anchors()[k]["value"] for k in FRACTION_NAMES}},
        "variant": {"free_parameters": list(param_names(True)), "starts": N_STARTS_VARIANT, "role": "reported next to the primary fit; never drives the regeneration"},
        "discrepancy": {"primary": DISCREPANCY, "variant": DISCREPANCY_VARIANT,
                        "rule": "variance (delta x data)^2 per bin added to the data and Monte Carlo variances; (delta p)^2 + (delta pbar)^2 on the net-proton bins",
                        "reason": "the 5% scale of inter-experiment systematic offsets (Fischer et al. 2022); without it the pion tables (1.4 to 2% errors) dominate"},
        "remnant_inelasticity": {"nd_generation_weights": ND_GENERATION_REMNANT_WEIGHTS, "diffractive_split": FROZEN_REMNANT_WEIGHTS,
                                 "reweighting": "exact kind-pair ratio (w_k1 w_k2)/(w0_k1 w0_k2) per non-diffractive event"},
        "bounds": {"epsilon": EPS_BOUNDS, "epsilon_nd": EPS_ND_BOUNDS, "b_sd": B_SD_BOUNDS, "logits": LOGIT_BOUNDS},
        "trust_region": {"ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR, "penalty": TRUST_PENALTY},
        "data": {"proton_window": PROTON_WINDOW, "pion_window": PION_WINDOW, "antiproton_window": PAPER_WINDOW,
                 "observables": "net proton on the common bins, proton alone elsewhere, pi- and pi+ tables", "normalization_prior": NORM_PRIOR,
                 "normalization": "one shared multiplicative nuisance for all species, profiled", "anchors": anchors()},
        "held_out": ["k_plus", "k_minus", "antiproton (alone)", "lead_xf", "gap", "n_ch", "pi- p_T profile", "mesonic count against MARK I", "SD n_ch against M_X",
                     "pi0 yield and its ratio to the charged-pion mean", "neutron dN/dx_F against the NA49 table"],
        "final_state": {"decays": "every unstable established hadron decays through the Round 2 kernel before it reaches a final state",
                        "isospin_rule": RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json").get("isospin_rule")},
        "readouts": {"pt_profile": PT_PROFILE, "mesonic_count": {"masses": list(MESONIC_MASSES), "decays": MESONIC_DECAYS, "reference": MARK_I,
                                                                "charged_from_weak": {str(k): v for k, v in CHARGED_FROM_WEAK.items()}},
                     "sd_nch_edges": SD_MX_EDGES.tolist()},
        "acceptance_block_b_stage2": {"pion_total_yields": ">= 0.72 for pi- and pi+", "mesonic_count": "within 15% of MARK I at 3.0 and 3.5 GeV",
                                      "pt_profile": "pi- model/data at 1.3 over 0.3 GeV/c below 3", "failure_rule": "a failed criterion names the next mechanism; the floor is not retuned"},
        "verdict_rule": {"constrained": "converged and ess_min >= ess_floor and w_excited interior and neither exponent on its bound and Hessian positive definite and block spreads within 1.5 x pooled error x sqrt(8) and the profiled floor interior to the grid",
                         "fraction_interior": FRACTION_INTERIOR, "stability_ratio_bound": STABILITY_RATIO_BOUND},
        "optimizer": {"method": "Powell (bounded)", "starts": N_STARTS, "hessian_steps": HESSIAN_STEPS},
        "frozen_substrate": frozen,
    }
    sync_path = MONO_ROOT / SYNC_MANIFEST_NAME
    payload["source"] = {"git": {"outer": RC.git_info(MONO_ROOT), "nested": RC.git_info(NESTED_ROOT)},
                         "sync_manifest": RC.read_json(sync_path) if sync_path.is_file() else None,
                         "rule": "the local git heads when the checkout is a repository; otherwise the gateway sync record"}
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()})
    for f in range(N_FLOORS):
        RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), {"points": [
            {"floor_index": f, "generation": 0, "epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND}]})


def _verify_frozen(contract: RC.RunContract) -> dict:
    pre = contract.load_pre_registration()
    frozen = {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}
    status = RC.verify_frozen_substrate(MONO_ROOT, frozen, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    return status


def _points(contract: RC.RunContract, floor_index: int) -> list[dict]:
    return RC.read_json(contract.path(GENERATION_POINTS_FMT.format(f=int(floor_index))))["points"]


def _generation_point(contract: RC.RunContract, floor_index: int, generation: int) -> tuple[float, float, float]:
    for p in _points(contract, floor_index):
        if int(p["generation"]) == int(generation):
            return float(p["epsilon"]), float(p["b_sd"]), float(p["epsilon_nd"])
    contract.refuse(f"branch {floor_index} generation {generation} has no registered point")


def _generation_complete(contract: RC.RunContract, floor_index: int, generation: int) -> bool:
    return all(contract.path(slot_names(floor_index, generation, s)[0]).is_file() for s in range(N_SLOTS))


def _latest_registered(contract: RC.RunContract, floor_index: int) -> int:
    return max(int(p["generation"]) for p in _points(contract, floor_index))


def _latest_complete(contract: RC.RunContract, floor_index: int) -> int:
    gens = sorted({int(p["generation"]) for p in _points(contract, floor_index)})
    complete = [g for g in gens if _generation_complete(contract, floor_index, g)]
    if not complete:
        contract.refuse(f"branch {floor_index} has no complete generation on disk")
    return max(complete)


def _branch_done(contract: RC.RunContract, floor_index: int) -> bool:
    return contract.path(BRANCH_REPORT_FMT.format(f=floor_index)).is_file()


def step_generate(run_dir: Path | None, floor_indices: list[int] | None, generation: int | None, slots: list[int] | None,
                  workers: int | None) -> None:
    """Generate the latest registered generation of every open branch (or
    of the named branches), all slots in one worker pool."""
    contract = _contract(run_dir)
    _verify_frozen(contract)
    pre_sha = RC.sha256_file(contract.path(RC.PREREGISTRATION_NAME))
    branches = list(range(N_FLOORS)) if not floor_indices else [int(f) for f in floor_indices]
    todo_slots = list(range(N_SLOTS)) if not slots else [int(s) for s in slots]
    jobs = []
    for f in branches:
        if _branch_done(contract, f):
            continue
        g = _latest_registered(contract, f) if generation is None else int(generation)
        eps, b_sd, eps_nd = _generation_point(contract, f, g)
        for slot in todo_slots:
            if contract.path(slot_names(f, g, slot)[0]).is_file():
                continue
            jobs.append(((f, g, slot), (slot_class(slot), slot_seed(f, g, slot), EVENTS_PER_SLOT, PARTITION_BASE + 100_000 * slot,
                                         eps, b_sd, eps_nd, FLOOR_GRID[f])))
    if not jobs:
        print(f"[{RUN_ID}] nothing to generate", flush=True)
        return
    t0 = time.perf_counter()
    results = RC.run_replicas(_generate_job, [job[1] for job in jobs], max_workers=workers)
    for ((f, g, slot), _), result in zip(jobs, results):
        write_slot(contract, f, g, slot, result, pre_sha)
    print(f"[{RUN_ID}] generated {len(jobs)} slots over branches {sorted({j[0][0] for j in jobs})} in {time.perf_counter() - t0:.0f} s", flush=True)


def _generate_job(cls, seed, n_events, partition_base, eps, b_sd, eps_nd, floor):
    return generate_slot(cls, seed, n_events, partition_base, eps, b_sd, eps_nd, floor)


def step_evaluate(run_dir: Path | None, floor_indices: list[int] | None) -> None:
    """Evaluate the latest complete generation of every open branch (or of
    the named branches) that has no fit report yet, one branch at a time."""
    contract = _contract(run_dir)
    _verify_frozen(contract)
    tables = load_fit_tables()
    branches = list(range(N_FLOORS)) if not floor_indices else [int(f) for f in floor_indices]
    for f in branches:
        if _branch_done(contract, f):
            continue
        g = _latest_complete(contract, f)
        if contract.path(FIT_REPORT_FMT.format(f=f, g=g)).is_file():
            continue
        report = evaluate_generation(contract, f, g, tables, anchors())
        RC.write_json(contract.path(FIT_REPORT_FMT.format(f=f, g=g)), report)
        if report["verdict"]["regeneration_requested"]:
            nxt = g + 1
            points = RC.read_json(contract.path(GENERATION_POINTS_FMT.format(f=f)))
            points["points"] = [p for p in points["points"] if int(p["generation"]) != nxt] + [
                {"floor_index": f, "generation": nxt, "epsilon": report["fitted"]["epsilon"], "b_sd": report["fitted"]["b_sd"],
                 "epsilon_nd": report["fitted"]["epsilon_nd"], "requested_by_generation": g, "ess_min": report["objective_at_optimum"]["ess_min"]}]
            RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), points)
            RC.write_json(contract.path(REGENERATION_FMT.format(f=f)), {"schema": f"{SCHEMA}.regeneration/1", **contract.identity(),
                                                                         "floor_index": f, "from_generation": g, "to_generation": nxt,
                                                                         "ess_min": report["objective_at_optimum"]["ess_min"], "written_utc": RC.now_utc()})
            print(f"[{RUN_ID}] branch {f} (floor {FLOOR_GRID[f]}) generation {g}: ess_min {report['objective_at_optimum']['ess_min']:.3f} < {ESS_INTERIOR}; "
                  f"regeneration {nxt} requested at (eps, B_SD, eps_nd) = ({report['fitted']['epsilon']:.4f}, {report['fitted']['b_sd']:.3f}, {report['fitted']['epsilon_nd']:.4f})", flush=True)
            continue
        branch = {"schema": f"{SCHEMA}.branch_report/1", **contract.identity(), **report,
                  "generation_history": sorted(p.name for p in contract.run_dir.glob(f"fit_report_f{f}_g*.json"))}
        branch["digest"] = RC.canonical_digest(branch)
        RC.write_json(contract.path(BRANCH_REPORT_FMT.format(f=f)), branch)
        if contract.path(REGENERATION_FMT.format(f=f)).is_file():
            contract.path(REGENERATION_FMT.format(f=f)).unlink()
        print(f"[{RUN_ID}] branch {f} (floor {FLOOR_GRID[f]}) final at generation {g}: objective {report['objective_at_optimum']['value']:.1f}", flush=True)


def _load_branches(contract: RC.RunContract) -> list[dict]:
    branches = []
    for f in range(N_FLOORS):
        path = contract.path(BRANCH_REPORT_FMT.format(f=f))
        if not path.is_file():
            contract.refuse(f"branch {f} has no final branch report; evaluate it first")
        b = RC.read_json(path)
        body = {k: v for k, v in b.items() if k != "digest"}
        if RC.canonical_digest(body) != b["digest"]:
            contract.refuse(f"{path.name} does not reproduce its recorded digest")
        branches.append(b)
    return branches


def _discrepancy_variant(contract: RC.RunContract, chosen: dict, tables: dict) -> dict:
    """The primary parametrization refitted at DISCREPANCY_VARIANT on the
    chosen branch's final generation."""
    f, g = int(chosen["floor_index"]), int(chosen["generation"])
    starts = [x_from_physical(chosen["fitted"], False)] + dispersed_starts(anchors(), free=False)[1:N_STARTS_VARIANT]
    fit = run_fit_parallel(contract, f, g, starts, free=False, discrepancy=DISCREPANCY_VARIANT, maxfev=4000,
                           pool_workers=BLOCK_WORKERS_PER_VARIANT_START)
    with BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS) as pool:
        value, det = PooledObjective(tables, anchors(), pool, free=False, discrepancy=DISCREPANCY_VARIANT).evaluate(fit["x"], details=True)
    templates = Templates(contract, f, g, tables)
    objective = Objective(templates, anchors(), free=False, discrepancy=DISCREPANCY_VARIANT)
    readouts = _mixture_readouts(templates, objective, tables, det["physical"])
    return {"discrepancy": DISCREPANCY_VARIANT, "floor_index": f, "generation": g, "fitted": det["physical"], "objective": value,
            "chi2_by_segment": det["chi2_by_segment"], "ess_min": det["ess_min"], "converged": fit["converged"],
            "total_yield_ratios": readouts["readout_total_yields"]["ratio_model_over_data"], "pt_profile": readouts["readout_pt_profile"]}


def step_profile(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    branches = _load_branches(contract)
    chosen = min(branches, key=lambda b: b["objective_at_optimum"]["value"])
    variant = _discrepancy_variant(contract, chosen, load_fit_tables())
    report = compose_report(branches, chosen, variant)
    body = contract.write_report(report, render_report_md(report))
    print(f"[{RUN_ID}] report written: floor {chosen['continuation_floor']}, verdict "
          f"{'CONSTRAINED' if body['verdict']['constrained'] else 'NOT_CONSTRAINED'}, stage 2 "
          f"{'ACCEPTED' if body['acceptance_block_b_stage2']['accepted'] else 'NOT ACCEPTED'}, digest {body['digest'][:16]}", flush=True)


def step_gate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    pions = report["readout_paper_window_sum_ratios"]
    pion_ratio = 0.5 * (pions["pi_plus"]["sum_ratio"] + pions["pi_minus"]["sum_ratio"])
    record = {
        "schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
        "obligation": "Gate 5 follow-up, the fine floor scan of the joint proton and pion fit (R51i)",
        "verdict": "CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED",
        "chosen_floor": report["continuation_floor"], "floor_profile": report["floor_profile"],
        "fitted": report["fitted"], "errors": (report["hessian"] or {}).get("errors"),
        "ess_min": report["objective_at_optimum"]["ess_min"], "converged": report["optimizer"]["converged"],
        "named_residual_register": report["named_residual_register"],
        "acceptance_block_b_stage2": report["acceptance_block_b_stage2"],
        "readout": {"pion_sum_ratio_mean": pion_ratio, "rule": "0.7 or above lets the flavor round proceed on this point",
                    "flavor_round_proceeds": bool(pion_ratio >= 0.7), "per_species": pions},
        "report_digest": report["digest"], "written_utc": RC.now_utc(),
    }
    RC.write_json(contract.path(GATE_RECORD_NAME), record)
    print(f"[{RUN_ID}] gate record: {record['verdict']}, floor {record['chosen_floor']}, pion sum ratio {pion_ratio:.3f}, "
          f"stage 2 {'ACCEPTED' if record['acceptance_block_b_stage2']['accepted'] else 'NOT ACCEPTED'}", flush=True)


def step_clean_validate(run_dir: Path | None) -> None:
    """Re-evaluate the chosen branch's final generation and the discrepancy
    variant, recompose the report with the other branch reports (verified by
    their digests), and regenerate the chosen branch's probe slot."""
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME, GATE_RECORD_NAME)
    frozen = _verify_frozen(contract)
    sealed = RC.read_json(contract.path(RC.REPORT_NAME))
    branches = _load_branches(contract)
    f = int(sealed["floor_index"])
    g = _latest_complete(contract, f)
    tables = load_fit_tables()
    rerun_branch = evaluate_generation(contract, f, g, tables, anchors())
    recorded = next(b for b in branches if int(b["floor_index"]) == f)
    rerun_branch = {"schema": recorded["schema"], **contract.identity(), **rerun_branch, "generation_history": recorded["generation_history"]}
    rerun_branch["digest"] = RC.canonical_digest(rerun_branch)
    branch_match = bool(rerun_branch["digest"] == recorded["digest"])
    others = [rerun_branch if int(b["floor_index"]) == f else b for b in branches]
    rerun = compose_report(others, rerun_branch, _discrepancy_variant(contract, rerun_branch, tables))
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    eps, b_sd, eps_nd = _generation_point(contract, f, g)
    probe = generate_slot(slot_class(0), slot_seed(f, g, 0), EVENTS_PER_SLOT, PARTITION_BASE, eps, b_sd, eps_nd, FLOOR_GRID[f])
    sealed_probe = RC.read_json(contract.path(slot_names(f, g, 0)[0]))
    probe_match = bool(arrays_digest(probe["arrays"]) == sealed_probe["arrays_sha256"]
                       and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"])
    payload = {"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}",
                            "cd dev/mixing2006 && python -m pytest tests/test_staged_r51i_joint_fit.py -q"],
               "report_digest": sealed["digest"], "report_rerun_digest": rerun_digest, "report_hash_match": bool(rerun_digest == sealed["digest"]),
               "chosen_branch": {"floor_index": f, "generation": g, "branch_digest_match": branch_match},
               "probe_slot": {"floor_index": f, "generation": g, "slot": 0, "arrays_match": probe_match},
               "frozen_substrate": frozen, "seal": {"state": "pending (the seal lands over this record)"}}
    contract.write_clean_validation(payload)
    if not branch_match:
        contract.refuse("the clean re-evaluation of the chosen branch differs from its branch report")
    if rerun_digest != sealed["digest"]:
        contract.refuse(f"the clean re-evaluation digest {rerun_digest} differs from the report {sealed['digest']}")
    if not probe_match:
        contract.refuse("the probe slot does not reproduce its sealed arrays")
    print(f"[{RUN_ID}] clean validation: branch match True, report match True, probe match True", flush=True)


def step_seal(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    required = [GATE_RECORD_NAME]
    for f in range(N_FLOORS):
        required.extend([BRANCH_REPORT_FMT.format(f=f), GENERATION_POINTS_FMT.format(f=f)])
        for g in range(_latest_complete(contract, f) + 1):
            for s in range(N_SLOTS):
                required.extend(slot_names(f, g, s))
    contract.seal(required, "slot_")


def step_smoke(run_dir: Path) -> None:
    contract = _contract(run_dir)
    if contract.run_dir == MONO_ROOT / "dev" / "00runs" / RUN_ID:
        contract.refuse("--smoke needs an explicit --run-dir outside the production run dir")
    contract.run_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_digests()
    f = SMOKE_FLOOR_INDEX
    RC.write_json(contract.path("smoke_pre_registration.json"), {"schema": f"{SCHEMA}.smoke_pre_registration/1", **contract.identity(),
                                                                "seed": SMOKE_SEED, "partition_base": SMOKE_PARTITION,
                                                                "events_per_class": SMOKE_EVENTS_PER_CLASS, "floor_index": f,
                                                                "frozen_substrate": frozen})
    pre_sha = RC.sha256_file(contract.path("smoke_pre_registration.json"))
    jobs = [(cls, SMOKE_SEED + i, SMOKE_EVENTS_PER_CLASS, SMOKE_PARTITION + 10_000 * i, GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND, FLOOR_GRID[f])
            for i, cls in enumerate(CLASS_ORDER)]
    results = RC.run_replicas(_generate_job, jobs)
    for cls, result in zip(CLASS_ORDER, results):
        np.savez(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, f=f, g=0)), **result["arrays"])
        RC.write_json(contract.path("smoke_" + SLOT_JSON_FMT.format(cls=cls, k=0, f=f, g=0)), {
            "schema": f"{SCHEMA}.smoke_slot/1", "pre_registration_sha256": pre_sha, "class_label": cls, "generation": 0, "replica": 0,
            "floor_index": f, "continuation_floor": result["continuation_floor"],
            "seed": result["seed"], "n_events": result["n_events"], "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
            "arrays_sha256": arrays_digest(result["arrays"]), "event_rolling_sha256": result["event_rolling_sha256"],
            "conservation": result["conservation"], "tracks": result["tracks"]})
    report = evaluate_generation(contract, f, 0, load_fit_tables(), anchors(), smoke=True)
    tables = load_fit_tables()
    serial = Objective(Templates(contract, f, 0, tables, smoke=True), anchors())
    x_check = x_from_physical({**report["fitted"], "epsilon": report["fitted"]["epsilon"] + 0.01}, False)
    with BlockPool(contract.run_dir, f, 0, 1, smoke=True) as pool:
        pooled_value = PooledObjective(tables, anchors(), pool).evaluate(x_check)
    serial_value = serial.evaluate(x_check)
    report["pool_check"] = {"in_process": serial_value, "pooled": pooled_value,
                            "relative_difference": abs(pooled_value - serial_value) / max(abs(serial_value), 1e-300)}
    report["schema"] = f"{SCHEMA}.smoke_fit_report/1"
    report["acceptance_block_b_stage2"] = acceptance(report)
    report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_fit_report.json"), report)
    print(f"[{RUN_ID}] smoke: fitted {json.dumps({k: round(v, 4) for k, v in report['fitted'].items() if isinstance(v, float)})}, "
          f"ess_min {report['objective_at_optimum']['ess_min']:.3f}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"R51i driver ({CARD})")
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pre-registration", "--generate", "--evaluate", "--profile", "--gate", "--clean-validate", "--seal", "--smoke"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--floor-index", type=int, action="append", default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--slot", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.pre_registration:
        step_pre_registration(args.run_dir)
    elif args.generate:
        step_generate(args.run_dir, args.floor_index, args.generation, args.slot, args.workers)
    elif args.evaluate:
        step_evaluate(args.run_dir, args.floor_index)
    elif args.profile:
        step_profile(args.run_dir)
    elif args.gate:
        step_gate(args.run_dir)
    elif args.clean_validate:
        step_clean_validate(args.run_dir)
    elif args.seal:
        step_seal(args.run_dir)
    elif args.smoke:
        if args.run_dir is None:
            RC.refuse("--smoke requires --run-dir", prefix=HERE.stem)
        step_smoke(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
