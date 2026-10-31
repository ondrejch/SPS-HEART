"""R66 driver: the joint flavor fit on the constrained R51i topology point
(Round 6 flavor round; stage 4 of the path to the global fit; card
TASK-20261002-r6-joint-flavor-fit-r66).

Design (differences from R51i; the R51i machinery is otherwise inherited):
- The topology and cascade are fixed at the sealed R51i optimum: the
  continuation floor 1.7 GeV; epsilon, B_SD and epsilon_nd; the
  non-diffractive remnant weights (intact 0, low-mass and excited from
  the fit); the class fractions at the external anchors.  The
  non-diffractive class is generated at the fitted remnant weights, so no
  remnant reweighting enters.
- Free: the Round 6 flavor weights of plan section 12.2, in log space:
  gamma_s_had (newly produced s-sbar content), assoc_norm (associated
  kaon-hyperon production by a non-strange baryonic parent) and r_bbar
  (baryon-antibaryon creation).  r_V stays at 1 (no K* or phi table).
  The weights act on continuum-parent vertices only; established
  resonances keep their measured branching fractions.
- Exact reweighting through the flavor ledger: every drawn continuum
  vertex records its chosen exponent key and the pool weight per key
  (flavor_weights.vertex_ledger_record), and a flavor point is reached by
  the per-vertex ratio of the chosen factor and the pool normalizations
  (flavor_fit_core.flavor_log_weights; the driver evaluates the same
  expression on a per-block key table).  Generation 0 is at the identity
  flavor point; a fitted point with a class ESS below 0.3 requests a
  regeneration there (at most 4 generations).
- Objective: the double-differential invariant cross sections per
  inelastic event of p (284 bins), pbar (122), pi- (282), pi+ (260), K+
  (153) and K- (154) in the fit window (x_F 0 to 0.95, p_T 0.05 to
  1.5 GeV/c), 1255 observations.  Variance per bin: the data, the
  between-block Monte Carlo error, and the fractional discrepancy term
  (delta x data)^2 with delta = 0.05.  The shared normalization is fixed
  at eta = 0 (owner decision 2026-10-02: the 2% prior of R51f to R51i was
  overwhelmed, and eta absorbed the high-p_T shape mismatch).  Variants:
  delta = 0.10, eta = +0.02 and eta = -0.02 (refits of the flavor
  weights), and the class fractions free on the simplex with the anchor
  terms.
- The block-parallel objective of R51i (forked block workers own replica
  blocks and return per-class observation vectors at a trial flavor
  point); 32 replicas of 100,000 events per class.
- The acceptance of the flavor round (pre-registered): K+ and K- total
  yields within 0.85 to 1.15 of NA49, pbar within 0.75 to 1.25, and the
  Gate 6 differential clause (for K+, K- and pbar the species chi2 falls
  from the identity point to the optimum; a yield ratio that moves
  toward 1 while its chi2 rises raises the reject flag).
- Verdict CONSTRAINED: converged, smallest class ESS >= 0.1, no flavor
  weight on its bound, positive-definite Hessian, and replica-block
  spreads within 1.5 x the pooled error x sqrt(32).
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

import flavor_fit_core as FC  # noqa: E402
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
from na49_mixing.flavor_weights import NOMINAL_FLAVOR, FlavorParams  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.importance import ess_fraction, logits_from_simplex, simplex_from_logits  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import families_from_json  # noqa: E402
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
RUN_ID = "r66_joint_flavor_fit_2026-10-02"
CARD = "TASK-20261002-r6-joint-flavor-fit-r66"
SCHEMA = "na49_mixing.staged_r66_joint_flavor_fit"
LEDGER_RUN = "r66_joint_flavor_fit"

CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd")
REPLICAS = 32
EVENTS_PER_SLOT = 100_000
N_SLOTS = len(CLASS_ORDER) * REPLICAS  # 128
MAX_GENERATIONS = 4
FLOOR_GRID = (1.7,)  # the R51i floor, fixed
N_FLOORS = len(FLOOR_GRID)

MASTER_SEED = 20261002
RAW_FIRST = 20314800
RAW_LAST = RAW_FIRST + MAX_GENERATIONS * N_SLOTS - 1  # 20315311
PARTITION_BASE = 3_700_000_000
SMOKE_SEED = 20304106
SMOKE_PARTITION = 3_140_000_000
SMOKE_EVENTS_PER_CLASS = 500
SMOKE_FLOOR_INDEX = 0
SMOKE_MESONIC_DECAYS = 200

SQRT_S = OB.SQRT_S
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
LOGIT_BOUNDS = (-8.0, 8.0)
# the flavor weights, in FlavorParams order; the free ones in log space
FLAVOR_ORDER = ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
FLAVOR_FREE = ("gamma_s_had", "assoc_norm", "r_bbar")
R_V_FIXED = 1.0
IDENTITY_FLAVOR = (1.0, 1.0, 1.0, 1.0)
FLAVOR_LOG_BOUNDS = (math.log(0.05), math.log(20.0))
FLAVOR_BOUND_TOL = 1.0e-3
HESSIAN_LOG_STEP = 0.02
# the shared normalization: fixed (owner decision 2026-10-02), with variants
NORM_FIXED = 0.0
NORM_VARIANTS = (0.02, -0.02)
DISCREPANCY = 0.05
DISCREPANCY_VARIANT = 0.10
# the pre-registered acceptance of the flavor round
KAON_YIELD_BAND = (0.85, 1.15)
PBAR_YIELD_BAND = (0.75, 1.25)
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
FRACTION_NAMES = ("f_sd_total", "f_dd")
WEIGHT_CACHE_ENTRIES = 64
SE_CACHE_ENTRIES = 512
# the fixed topology point: the sealed R51i optimum (floor 1.7 GeV, CONSTRAINED)
R51I_RUN_ID = "r51i_joint_fit_2026-10-01"
FIXED_EPSILON = 0.5073987832309154
FIXED_B_SD = 2.4312818539548027
FIXED_EPSILON_ND = 0.3851105135852008
FIXED_W_LOW = 0.05417920942051668
FIXED_W_EXCITED = 0.9458207905794833
FIXED_SHAPE = (FIXED_EPSILON, FIXED_B_SD, FIXED_EPSILON_ND)
FIXED_TOPOLOGY = {"epsilon": FIXED_EPSILON, "b_sd": FIXED_B_SD, "epsilon_nd": FIXED_EPSILON_ND,
                  "w_intact": 0.0, "w_low_mass": FIXED_W_LOW, "w_excited": FIXED_W_EXCITED}
REMNANT_KINDS = ("intact_proton", "low_mass_baryonic_resonance", "high_mass_baryonic_continuum")
# the non-diffractive class is generated at the fitted R51i remnant weights
ND_GENERATION_REMNANT_WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": FIXED_W_LOW,
                                 "high_mass_baryonic_continuum": FIXED_W_EXCITED}
FROZEN_REMNANT_WEIGHTS = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}

FIT_WINDOW = {"xf_min": 0.0, "xf_max": 0.95, "pt_min": 0.05, "pt_max": 1.5}
PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
FIT_SPECIES = ("proton", "antiproton", "pi_minus", "pi_plus", "k_plus", "k_minus")
TABLE_WINDOWS = {name: FIT_WINDOW for name in FIT_SPECIES}
TRACK_SPECIES = OB.SPECIES_ORDER
FLAVOR_TARGETS = ("k_plus", "k_minus", "antiproton")

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
    "helpers/benchmarking/staged_r51i_joint_fit.py",
    "helpers/benchmarking/flavor_fit_core.py",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_proton_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_antiproton_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_piminus.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_piplus.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaplus_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaminus_pp.txt",
)

SLOT_JSON_FMT = "slot_{cls}_{k}_f{f}_g{g}.json"
SLOT_NPZ_FMT = "slot_{cls}_{k}_f{f}_g{g}.npz"
FIT_REPORT_FMT = "fit_report_f{f}_g{g}.json"
BRANCH_REPORT_FMT = "branch_report_f{f}.json"
REGENERATION_FMT = "regeneration_request_f{f}.json"
GATE_RECORD_NAME = "gate6_flavor_r66.json"
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


def flavor_params(point) -> FlavorParams:
    """The FlavorParams of a point in FLAVOR_ORDER (default hidden s-sbar content)."""
    return FlavorParams(**{name: float(v) for name, v in zip(FLAVOR_ORDER, point)})


def flavor_of(phys: dict) -> tuple:
    return tuple(float(phys[name]) for name in FLAVOR_ORDER)


def _params_at(cls: str, flavor_point) -> TopologyNominalParams:
    """The generator parameters of a class at the fixed R51i topology point
    and the generation flavor point."""
    weights = ND_GENERATION_REMNANT_WEIGHTS if cls == "nd" else FROZEN_REMNANT_WEIGHTS
    return TopologyNominalParams(
        epsilon=FIXED_EPSILON, b_sd=FIXED_B_SD, class_fractions=dict(NOMINAL_PARAMS.class_fractions),
        remnant_weights=dict(weights), p_n2=0.0, flavor=flavor_params(flavor_point),
        epsilon_nd=FIXED_EPSILON_ND,
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


def generate_slot(cls: str, seed: int, n_events: int, partition_base: int, flavor_point, floor: float) -> dict:
    """Generate one slot at the fixed topology point and the generation
    flavor point, with the flavor ledger of every drawn continuum vertex:
    returns the compact arrays and the metadata."""
    catalog, fams = _load_physics(floor)
    flavor_point = tuple(float(v) for v in flavor_point)
    params = _params_at(cls, flavor_point)
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
    led_event: list[int] = []
    led_chosen: list = []
    led_gvertex: list[int] = []
    led_gkey: list = []
    led_gw: list[float] = []
    vertex_counter = 0
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
        ledger: list = []
        event = run_topology(cls, params, catalog, fams, **rngs, ledger=ledger)
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
        for record in ledger:
            if record is None:
                continue
            chosen_key, groups = record
            led_event.append(i); led_chosen.append(FC._key_ints(chosen_key))
            for key, weight in sorted(groups.items()):
                led_gvertex.append(vertex_counter); led_gkey.append(FC._key_ints(key)); led_gw.append(float(weight))
                line.append(f"L{vertex_counter}:{'/'.join(str(v) for v in FC._key_ints(key))}:{float(weight).hex()}")
            vertex_counter += 1
        rolling.update(("|".join(line) + "\n").encode("utf-8"))
    wall = time.perf_counter() - t0
    return {
        "cls": cls, "seed": int(seed), "n_events": int(n_events), "partition_base": int(partition_base),
        "epsilon": FIXED_EPSILON, "b_sd": FIXED_B_SD, "epsilon_nd": FIXED_EPSILON_ND, "continuation_floor": float(floor),
        "flavor_generation": dict(zip(FLAVOR_ORDER, flavor_point)),
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
            "led_event": np.asarray(led_event, dtype=np.int32),
            "led_chosen": np.asarray(led_chosen, dtype=np.int16).reshape(-1, 4),
            "led_gvertex": np.asarray(led_gvertex, dtype=np.int32),
            "led_gkey": np.asarray(led_gkey, dtype=np.int16).reshape(-1, 4),
            "led_gw": np.asarray(led_gw, dtype=np.float32),
        },
        "informative_vertices": int(vertex_counter),
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
        "flavor_generation": result["flavor_generation"], "informative_vertices": result["informative_vertices"],
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
def source_decay_finals(family, mass: float, catalog, fams, rng_channel, rng_kinematics, flavor=None) -> list:
    """The final-state hadrons of one continuum source at rest, through the
    topology generator's excitation cascade (continuation, direct channel,
    catalog handoff; unstable siblings decay through the Round 2 kernel)."""
    current = CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(int(v) for v in family.qn),
                           quantum={}, family_id=family.family_id, axis=(0.0, 0.0, 1.0),
                           alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=("r66_readout",), depth=0)
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
        channels = enumerate_channels(current, catalog, fams, flavor=flavor)
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


def mesonic_count_readout(floor: float, n_decays: int = MESONIC_DECAYS, flavor_point=IDENTITY_FLAVOR) -> dict:
    """The mean charged count per decay of a mesonic_neutral source at rest
    at MESONIC_MASSES against the MARK I curve (the charged products of
    the final neutral weak decayers counted as in the measurement), at the
    flavor point *flavor_point*."""
    flavor = flavor_params(flavor_point)
    catalog, fams = _load_physics(floor)
    family = next(f for f in fams if f.family_id == "mesonic_neutral")
    rows = []
    for m_index, mass in enumerate(MESONIC_MASSES):
        key = [MASTER_SEED, 66, int(round(1000 * floor)), m_index]
        rc = np.random.default_rng(np.random.SeedSequence(key + [1]))
        rk = np.random.default_rng(np.random.SeedSequence(key + [2]))
        total = 0.0
        for _ in range(int(n_decays)):
            for h in source_decay_finals(family, mass, catalog, fams, rc, rk, flavor=flavor):
                total += (1.0 if int(h.qn[1]) != 0 else 0.0) + CHARGED_FROM_WEAK.get(int(h.pdg_id), 0.0)
        model = total / float(n_decays)
        reference = MARK_I["a"] + MARK_I["b"] * math.log(mass * mass)
        rows.append({"mass": float(mass), "model": model, "mark_i": reference, "ratio": model / reference,
                     "below_open_charm": bool(mass < 3.73)})
    return {"floor": float(floor), "decays_per_mass": int(n_decays), "family": "mesonic_neutral",
            "flavor": dict(zip(FLAVOR_ORDER, (float(v) for v in flavor_point))),
            "reference": MARK_I, "charged_from_weak": {str(k): v for k, v in CHARGED_FROM_WEAK.items()},
            "seed_rule": "numpy SeedSequence([MASTER_SEED, 66, round(1000 floor), mass index, 1 or 2])", "rows": rows}


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
class Templates:
    """The loaded templates of the branch and one generation: per class and
    replica the flavor ledgers and the per-species binned track sums."""

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
            if (float(meta["epsilon"]), float(meta["b_sd"]), float(meta["epsilon_nd"])) != FIXED_SHAPE:
                contract.refuse(f"{json_name} was not generated at the fixed R51i topology point")
            with np.load(contract.path(npz_name)) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays_digest(arrays) != meta["arrays_sha256"]:
                contract.refuse(f"the arrays of {npz_name} do not match their recorded digest")
            point = tuple(float(meta["flavor_generation"][name]) for name in FLAVOR_ORDER)
            if self.point is None:
                self.point = point
            elif self.point != point:
                contract.refuse(f"mixed generation flavor points in branch {floor_index} generation {generation}")
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
        return {"n": int(n_events), "shape": shape, "binned": binned, "ledger": self._ledger(arrays),
                "lead_xf": arrays["lead_xf"], "gap": arrays["gap"], "n_ch": arrays["n_ch"],
                "rem_kind": arrays.get("rem_kind"), "totals": totals,
                "n_pi0": arrays["n_pi0"], "neu_event": arrays["neu_event"], "neu_xf": arrays["neu_xf"]}

    @staticmethod
    def _ledger(arrays: dict) -> dict | None:
        """The block's flavor ledger on a key table: per pool group its key
        index, vertex and weight; per vertex its event, chosen key index and
        log pool normalization at the generation point."""
        n_v = int(arrays["led_event"].size)
        if n_v == 0:
            return None
        gkey = arrays["led_gkey"].astype(np.int64)
        chosen = arrays["led_chosen"].astype(np.int64)
        both = np.concatenate([gkey, chosen])
        if np.any(both < 0) or np.any(both >= 100):
            raise ValueError("a flavor ledger exponent lies outside [0, 100)")
        codes = ((both[:, 0] * 100 + both[:, 1]) * 100 + both[:, 2]) * 100 + both[:, 3]
        table, inverse = np.unique(codes, return_inverse=True)
        k = np.column_stack([table // 1_000_000, (table // 10_000) % 100, (table // 100) % 100, table % 100]).astype(float)
        k[:, 0] *= 0.5  # the ledger stores twice n_ssbar
        gv = arrays["led_gvertex"].astype(np.int64)
        gw = arrays["led_gw"].astype(float)
        z_from = np.bincount(gv, weights=gw, minlength=n_v)[:n_v]
        return {"n_vertices": n_v, "v_event": arrays["led_event"].astype(np.int64), "gv": gv, "gw": gw,
                "g_idx": inverse[: gkey.shape[0]], "c_idx": inverse[gkey.shape[0]:], "keys": k, "log_z_from": np.log(z_from)}

    # -- weights and moments ------------------------------------------
    def log_weights(self, block: dict, flavor_point) -> np.ndarray:
        """The per-event log importance weights from the generation flavor
        point to *flavor_point* (FLAVOR_ORDER); the expression of
        flavor_fit_core.flavor_log_weights on the block's key table."""
        point = tuple(float(v) for v in flavor_point)
        led = block["ledger"]
        if point == self.point or led is None:
            return np.zeros(block["n"])
        delta = np.log(np.asarray(point)) - np.log(np.asarray(self.point))
        dk = led["keys"] @ delta
        z_to = np.bincount(led["gv"], weights=led["gw"] * np.exp(dk)[led["g_idx"]], minlength=led["n_vertices"])[: led["n_vertices"]]
        per_vertex = dk[led["c_idx"]] + led["log_z_from"] - np.log(z_to)
        return np.bincount(led["v_event"], weights=per_vertex, minlength=block["n"])[: block["n"]]

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
    return (FRACTION_NAMES if free else ()) + FLAVOR_FREE


def x_from_physical(phys: dict, free: bool) -> np.ndarray:
    """The optimizer vector: [logit f_SD, logit f_DD] (free fractions only)
    + [log gamma_s_had, log assoc_norm, log r_bbar]."""
    core = [math.log(float(phys[name])) for name in FLAVOR_FREE]
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
    flavor = {name: float(math.exp(v)) for name, v in zip(FLAVOR_FREE, core)}
    return {"f_sd_total": float(f[0]), "f_dd": float(f[1]), "f_nd": float(f[2]), **flavor, "r_v": R_V_FIXED, **FIXED_TOPOLOGY}


def bounds_for(free: bool) -> list:
    return ([LOGIT_BOUNDS, LOGIT_BOUNDS] if free else []) + [FLAVOR_LOG_BOUNDS] * len(FLAVOR_FREE)


def _in_bounds(x, free: bool) -> bool:
    return all(lo - 1e-12 <= float(v) <= hi + 1e-12 for v, (lo, hi) in zip(np.asarray(x, dtype=float), bounds_for(free)))


class Objective:
    """The cached joint objective at one generation.  ``free`` selects the
    parametrization (fractions at the anchors, or free on the simplex with
    the anchor terms); ``discrepancy`` is the fractional model-discrepancy
    scale per bin; ``norm`` the fixed shared normalization offset eta (the
    model is scaled by 1 + eta)."""

    def __init__(self, templates: Templates, anchors_: dict, free: bool = False, discrepancy: float = DISCREPANCY,
                 norm: float = NORM_FIXED):
        self.t = templates
        self.anchors = anchors_
        self.free = bool(free)
        self.discrepancy = float(discrepancy)
        self.norm = float(norm)
        tab = templates.tables
        dz = self.discrepancy
        parts_d, parts_v = [], []
        self.segments = {}
        offset = 0
        for name in FIT_SPECIES:
            t_ = tab[name]
            parts_d.append(t_.data)
            parts_v.append(t_.sigma ** 2 + (dz * t_.data) ** 2)
            self.segments[name] = (offset, offset + t_.data.size)
            offset += t_.data.size
        self.d = np.concatenate(parts_d)
        self.v_data = np.concatenate(parts_v)
        self.areas = {name: OB.bin_areas(tab[name]) for name in FIT_SPECIES}
        self.n_obs = int(self.d.size)
        self.cache: dict = {}
        self.se_cache: dict = {}

    def physical(self, x) -> dict:
        return physical_from_x(x, self.free, self.anchors)

    def class_weights(self, flavor_point, blocks: list[int] | None = None) -> tuple[dict, dict]:
        """Per class per block the normalized weights and the ESS fraction."""
        point = tuple(float(v) for v in flavor_point)
        key = (tuple(round(v, 12) for v in point), tuple(blocks) if blocks is not None else None)
        if key in self.cache:
            return self.cache[key]
        weights = {}; ess = {}
        for cls in CLASS_ORDER:
            weights[cls] = []; ess_vals = []
            for k, block in enumerate(self.t.by_class[cls]):
                if blocks is not None and k not in blocks:
                    weights[cls].append(None); continue
                log_w = self.t.log_weights(block, point)
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
        return np.concatenate([pred[name] for name in FIT_SPECIES])

    def unweighted_se(self, phys: dict, block_ids: list[int]) -> np.ndarray:
        """The between-block standard error of the mixture at the trial
        fractions with unit flavor weights (cached)."""
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
        if not _in_bounds(x, self.free):
            return (float("inf"), None) if details else float("inf")
        phys = self.physical(x)
        weights, ess = self.class_weights(flavor_of(phys), blocks)
        block_ids = list(range(self.t.n_blocks)) if blocks is None else list(blocks)
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, phys, weights)) for k in block_ids])
        scale = 1.0 + self.norm
        m = np.mean(vectors, axis=0) * scale
        se = self.unweighted_se(phys, block_ids) * scale
        return self._likelihood(phys, m, se, ess, block_ids, details)

    def _likelihood(self, phys: dict, m: np.ndarray, se: np.ndarray, ess: dict, block_ids: list, details: bool):
        ess_min = min(ess.values())
        r = (self.d - m) / np.sqrt(self.v_data + se ** 2)
        data_term = float(r @ r)
        a = self.anchors
        anchor_terms = 0.0
        if self.free:
            anchor_terms = ((phys["f_sd_total"] - a["f_sd_total"]["value"]) / a["f_sd_total"]["sigma"]) ** 2 \
                + ((phys["f_dd"] - a["f_dd"]["value"]) / a["f_dd"]["sigma"]) ** 2
        value = data_term + anchor_terms
        if ess_min < ESS_FLOOR:
            value += TRUST_PENALTY * (ESS_FLOOR - ess_min) / ESS_FLOOR
        if not details:
            return value
        by_segment = {name: float(np.sum(r[lo:hi] ** 2)) for name, (lo, hi) in self.segments.items()}
        return value, {
            "physical": phys, "ess": ess, "ess_min": ess_min, "normalization": self.norm,
            "minus2lnl_data": data_term, "anchor_terms": anchor_terms, "chi2_by_segment": by_segment,
            "m": m, "se": se, "residual_pull": r, "blocks": block_ids,
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
    vectors, for the between-block error) or ("eval", flavor point) (the
    reweighted vectors and the per-class smallest ESS)."""
    if msg[0] == "unit":
        unit = {cls: [np.full(b["n"], 1.0 / b["n"]) for b in objective.t.by_class[cls]] for cls in CLASS_ORDER}
        return ("unit", _class_obs(objective, unit))
    if msg[0] == "eval":
        weights, ess = objective.class_weights(tuple(msg[1]))
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

    def evaluate(self, flavor_point) -> tuple[np.ndarray, dict]:
        replies = self._ask(("eval", tuple(float(v) for v in flavor_point)))
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

    def __init__(self, tables: dict, anchors_: dict, pool: BlockPool, free: bool = False, discrepancy: float = DISCREPANCY,
                 norm: float = NORM_FIXED):
        stub = types.SimpleNamespace(tables=tables, n_blocks=pool.n_blocks, point=pool.point)
        super().__init__(stub, anchors_, free=free, discrepancy=discrepancy, norm=norm)
        self.pool = pool
        self.obs_cache: dict = {}

    def class_obs(self, flavor_point) -> tuple[np.ndarray, dict]:
        key = tuple(round(float(v), 12) for v in flavor_point)
        if key in self.obs_cache:
            return self.obs_cache[key]
        value = self.pool.evaluate(flavor_point)
        self.obs_cache[key] = value
        while len(self.obs_cache) > WEIGHT_CACHE_ENTRIES:
            self.obs_cache.pop(next(iter(self.obs_cache)))
        return value

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        if blocks is not None:
            raise ValueError("the pooled objective always uses every block")
        if not _in_bounds(x, self.free):
            return (float("inf"), None) if details else float("inf")
        phys = self.physical(x)
        obs, ess = self.class_obs(flavor_of(phys))
        frac = np.asarray([phys["f_sd_total"] / 2.0, phys["f_sd_total"] / 2.0, phys["f_dd"], phys["f_nd"]])
        vectors = np.einsum("c,kco->ko", frac, obs)
        n = vectors.shape[0]
        scale = 1.0 + self.norm
        m = np.mean(vectors, axis=0) * scale
        unit = np.einsum("c,kco->ko", frac, self.pool.unit())
        se = np.std(unit, axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(unit.shape[1])
        se = np.maximum(se, SE_FLOOR) * scale
        return self._likelihood(phys, m, se, ess, list(range(n)), details)


def dispersed_starts(anchors_: dict, free: bool) -> list[np.ndarray]:
    """Dispersed starts in (gamma_s_had, assoc_norm, r_bbar), with the
    fractions of the variant dispersed around the anchors."""
    a_sd, a_dd = anchors_["f_sd_total"]["value"], anchors_["f_dd"]["value"]
    flavors = [(1.0, 1.0, 1.0), (0.6, 1.0, 3.0), (0.8, 0.6, 5.0), (0.5, 1.5, 2.0),
               (1.4, 0.8, 8.0), (0.7, 2.0, 1.0), (0.4, 0.5, 4.0), (1.0, 1.2, 12.0)]
    fractions = [(a_sd, a_dd), (0.30, 0.10), (0.10, 0.02), (0.40, 0.20), (a_sd, a_dd), (0.15, 0.05), (0.25, 0.15), (0.35, 0.03)]
    n = N_STARTS if not free else N_STARTS_VARIANT
    starts = []
    for (fs, fd), flavor in list(zip(fractions, flavors))[:n]:
        phys = {"f_sd_total": fs, "f_dd": fd, **dict(zip(FLAVOR_FREE, flavor))}
        x = x_from_physical(phys, free)
        if free:
            x[[0, 1]] = np.clip(x[[0, 1]], LOGIT_BOUNDS[0] + 0.5, LOGIT_BOUNDS[1] - 0.5)
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
                discrepancy: float, pool_workers: int = BLOCK_WORKERS_PER_START, norm: float = NORM_FIXED) -> dict:
    """One optimizer start in its own process with its own block pool, or
    one replica-block fit (``blocks = [k]``) that loads that block alone."""
    contract = _contract(Path(run_dir))
    tables = load_fit_tables()
    if blocks is None:
        with BlockPool(run_dir, floor_index, generation, pool_workers) as pool:
            objective = PooledObjective(tables, anchors(), pool, free=free, discrepancy=discrepancy, norm=norm)
            return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)
    templates = Templates(contract, floor_index, generation, tables, replicas=blocks)
    objective = Objective(templates, anchors(), free=free, discrepancy=discrepancy, norm=norm)
    return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)


def run_fit_parallel(contract: RC.RunContract, floor_index: int, generation: int, starts: list, *, free: bool,
                     discrepancy: float = DISCREPANCY, norm: float = NORM_FIXED, blocks=None, maxfev: int = 4000,
                     workers: int | None = None, pool_workers: int = BLOCK_WORKERS_PER_START) -> dict:
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x0], blocks, int(maxfev), bool(free),
             float(discrepancy), int(pool_workers), float(norm)) for x0 in starts]
    results = RC.run_replicas(_fit_worker, jobs, max_workers=workers or len(jobs))
    records = [r["starts"][0] for r in results]
    best = min(results, key=lambda r: r["fun"])
    return {"x": np.asarray(best["x"], dtype=float), "fun": float(best["fun"]), "converged": bool(best["converged"]),
            "nfev": int(best["nfev"]), "message": best["message"], "starts": records}


def hessian_log(objective: Objective, phys: dict) -> dict:
    """Central-difference Hessian of the primary objective in the log flavor
    weights; the physical errors are the log errors times the values."""
    names = FLAVOR_FREE
    p0 = x_from_physical(phys, False)
    h_step = HESSIAN_LOG_STEP

    def value(p):
        return objective.evaluate(np.asarray(p, dtype=float))

    n = p0.size
    h = np.zeros((n, n))
    f0 = value(p0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h_step
        h[i, i] = (value(p0 + ei) - 2.0 * f0 + value(p0 - ei)) / h_step ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h_step
            h[i, j] = h[j, i] = (value(p0 + ei + ej) - value(p0 + ei - ej) - value(p0 - ei + ej) + value(p0 - ei - ej)) / (4.0 * h_step ** 2)
    finite = bool(np.all(np.isfinite(h)))
    eig = np.linalg.eigvalsh(h) if finite else np.full(n, np.nan)
    pd = bool(finite and np.all(eig > 0.0))
    errors_log = errors = correlation = None
    if pd:
        cov = 2.0 * np.linalg.inv(h)
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        errors_log = {name: float(sd[i]) for i, name in enumerate(names)}
        errors = {name: float(phys[name] * sd[i]) for i, name in enumerate(names)}
        correlation = (cov / np.outer(sd, sd)).tolist()
    return {"names": list(names), "space": "log", "step": h_step, "matrix": h.tolist() if finite else None,
            "eigenvalues": [float(e) for e in eig] if finite else None, "positive_definite": pd, "quoted": pd,
            "errors_log": errors_log, "errors": errors, "correlation": correlation}


def _mixture_readouts(templates: Templates, objective: Objective, tables: dict, phys: dict) -> dict:
    """The readouts at a fitted point: paper-window sum ratios, total yields
    and their class breakdown, the pi- p_T profile, the SD n_ch against M_X,
    the pi0 and neutron readouts, and predictions."""
    weights, _ = objective.class_weights(flavor_of(phys))
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
    # the yields per event of each class (the class breakdown of the mixture)
    class_yields = {}
    for cls in CLASS_ORDER:
        y = {name: 0.0 for name in TRACK_SPECIES}
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            for name in TRACK_SPECIES:
                y[name] += float(np.sum(wn * block["totals"][name]))
        class_yields[cls] = {"fraction": frac_map[cls], "per_event": y,
                             "ratio_model_over_data": {name: y[name] / float(ref_y[name]["value"]) for name in TRACK_SPECIES if name in ref_y}}
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
    return {"readout_paper_window_sum_ratios": readout, "readout_total_yields": total_yields, "readout_class_yields": class_yields,
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
    identity_phys = {**phys, **{name: 1.0 for name in FLAVOR_FREE}}
    identity_value, identity_det = objective.evaluate(x_from_physical(identity_phys, False), details=True)
    hess = hessian_log(objective, phys) if not smoke else {"quoted": False, "positive_definite": False, "errors": None,
                                                           "errors_log": None, "skipped": "smoke"}
    blocks = []
    n_blocks = objective.t.n_blocks
    if not smoke and n_blocks > 1:
        block_jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in pooled["x"]], [k], 2000, False,
                       DISCREPANCY, 0, NORM_FIXED) for k in range(n_blocks)]
        block_results = RC.run_replicas(_fit_worker, block_jobs, max_workers=n_blocks)
        for k, res in enumerate(block_results):
            fitted_k = physical_from_x(np.asarray(res["x"]), False, anchors_)
            blocks.append({"block": k, **{name: fitted_k[name] for name in FLAVOR_FREE}, "fun": res["fun"], "converged": res["converged"]})
    stability = {}
    if blocks and hess.get("errors_log"):
        for name in FLAVOR_FREE:
            vals = np.log(np.asarray([b[name] for b in blocks]))
            spread = float(np.std(vals, ddof=1))
            expected = hess["errors_log"][name] * math.sqrt(len(blocks))
            ratio = spread / expected if expected > 0 else float("inf")
            stability[name] = {"block_log_values": [float(v) for v in vals], "spread_log": spread,
                               "expected_spread_log": expected, "ratio": ratio, "stable": bool(ratio <= STABILITY_RATIO_BOUND)}
    on_bound = [name for name in FLAVOR_FREE
                if min(abs(math.log(phys[name]) - b) for b in FLAVOR_LOG_BOUNDS) < FLAVOR_BOUND_TOL]
    residuals = []
    if not pooled["converged"]:
        residuals.append({"kind": "optimizer_non_convergence", "message": pooled["message"]})
    if det["ess_min"] < ESS_FLOOR:
        residuals.append({"kind": "weight_degeneracy", "ess_min": det["ess_min"], "floor": ESS_FLOOR})
    if on_bound:
        residuals.append({"kind": "flavor_weight_on_bound", "parameters": on_bound, "bounds": [math.exp(b) for b in FLAVOR_LOG_BOUNDS]})
    if not hess.get("positive_definite"):
        residuals.append({"kind": "block_curvature_failure", "eigenvalues": hess.get("eigenvalues")})
    for name, s in stability.items():
        if not s["stable"]:
            residuals.append({"kind": "replica_block_minima_disagree", "parameter": name, "ratio": s["ratio"]})
    regenerate = (not smoke) and det["ess_min"] < ESS_INTERIOR and generation < MAX_GENERATIONS - 1
    constrained = bool(pooled["converged"] and det["ess_min"] >= ESS_FLOOR and not on_bound
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
               "note": "fractions free on the simplex with the two anchor terms, flavor weights free; reported next to the primary fit"}
    readouts = _mixture_readouts(templates, objective, tables, phys)
    identity_readouts = _mixture_readouts(templates, objective, tables, identity_phys)
    mesonic = mesonic_count_readout(FLOOR_GRID[floor_index], SMOKE_MESONIC_DECAYS if smoke else MESONIC_DECAYS, flavor_of(phys))
    return {
        "floor_index": int(floor_index), "continuation_floor": float(FLOOR_GRID[floor_index]),
        "generation": int(generation),
        "generation_point": dict(zip(FLAVOR_ORDER, templates.point)),
        "fixed_topology": {**FIXED_TOPOLOGY, "continuation_floor": float(FLOOR_GRID[floor_index]),
                           "fractions": {k: anchors_[k]["value"] for k in FRACTION_NAMES}, "provenance": R51I_RUN_ID},
        "statistics": {"classes": list(CLASS_ORDER), "replicas": templates.n_blocks,
                       "events_per_slot": int(templates.by_class[CLASS_ORDER[0]][0]["n"]), "tier": "T2" if not smoke else "smoke"},
        "observable_set": {"segments": {k: [int(a), int(b)] for k, (a, b) in objective.segments.items()}, "total": objective.n_obs,
                           "fit_window": FIT_WINDOW, "paper_window": PAPER_WINDOW,
                           "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS)},
                           "discrepancy": DISCREPANCY, "anchors": anchors_},
        "parametrization": {"primary": "topology fixed at R51i, fractions at the anchors, r_V = 1; free (log): " + ", ".join(FLAVOR_FREE),
                            "variant": "fractions free on the simplex with the anchor terms; free: " + ", ".join(param_names(True))},
        "fitted": {**phys, "sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0,
                   "remnant_weights": {"intact_proton": phys["w_intact"], "low_mass_baryonic_resonance": phys["w_low_mass"],
                                       "high_mass_baryonic_continuum": phys["w_excited"]},
                   "fractions_fixed_to_anchors": True},
        "objective_at_optimum": {"value": value, "minus2lnl_data": det["minus2lnl_data"], "anchor_terms": det["anchor_terms"],
                                 "chi2_by_segment": det["chi2_by_segment"], "normalization": det["normalization"], "ess": det["ess"],
                                 "ess_min": det["ess_min"], "in_trust_region": det["in_trust_region"]},
        "identity_point": {"flavor": dict(zip(FLAVOR_ORDER, IDENTITY_FLAVOR)), "objective": identity_value,
                           "chi2_by_segment": identity_det["chi2_by_segment"], "ess_min": identity_det["ess_min"],
                           "total_yield_ratios": identity_readouts["readout_total_yields"]["ratio_model_over_data"],
                           "paper_window_sum_ratios": {k: v["sum_ratio"] for k, v in identity_readouts["readout_paper_window_sum_ratios"].items()}},
        "optimizer": {"method": "Powell (bounded, log flavor weights)", "starts": len(starts), "converged": pooled["converged"], "nfev": pooled["nfev"],
                      "message": pooled["message"], "start_records": pooled["starts"]},
        "hessian": hess, "block_fits": blocks, "replica_block_stability": stability,
        "variant_free_fractions": variant,
        "verdict": {"constrained": constrained, "regeneration_requested": regenerate, "flavor_on_bound": on_bound,
                    "ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR},
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
    """The pre-registered acceptance of the flavor round."""
    ratios = branch["readout_total_yields"]["ratio_model_over_data"]
    ident = branch["identity_point"]
    kaon_ok = bool(all(KAON_YIELD_BAND[0] <= ratios[s] <= KAON_YIELD_BAND[1] for s in ("k_plus", "k_minus")))
    pbar_ok = bool(PBAR_YIELD_BAND[0] <= ratios["antiproton"] <= PBAR_YIELD_BAND[1])
    differential = {}
    for s in FLAVOR_TARGETS:
        chi_opt = float(branch["objective_at_optimum"]["chi2_by_segment"][s])
        chi_id = float(ident["chi2_by_segment"][s])
        r_opt, r_id = float(ratios[s]), float(ident["total_yield_ratios"][s])
        toward = bool(abs(math.log(r_opt)) < abs(math.log(r_id)))
        differential[s] = {"chi2_identity": chi_id, "chi2_optimum": chi_opt, "yield_ratio_identity": r_id, "yield_ratio_optimum": r_opt,
                           "integrated_improves": toward, "shape_improves": bool(chi_opt < chi_id),
                           "reject_flag": bool(toward and chi_opt >= chi_id)}
    diff_ok = bool(all(v["shape_improves"] and not v["reject_flag"] for v in differential.values()))
    return {"criteria": {
                "kaon_total_yields": {"rule": f"K+ and K- total-yield ratios within {KAON_YIELD_BAND}",
                                      "values": {s: ratios[s] for s in ("k_plus", "k_minus")}, "met": kaon_ok},
                "antiproton_total_yield": {"rule": f"pbar total-yield ratio within {PBAR_YIELD_BAND}", "value": ratios["antiproton"], "met": pbar_ok},
                "differential_clause": {"rule": "for K+, K- and pbar the species chi2 falls from the identity point to the optimum; "
                                                "a yield ratio moving toward 1 with a rising chi2 raises the reject flag",
                                        "values": differential, "met": diff_ok}},
            "accepted": bool(kaon_ok and pbar_ok and diff_ok)}


def compose_report(branch: dict, variants: dict | None) -> dict:
    """The final report: the branch's fit, the variants, the acceptance,
    and the verdict."""
    report = {k: v for k, v in branch.items() if k not in ("digest", "schema", "run", "card", "driver", "driver_sha256")}
    report["branch_digest"] = branch.get("digest")
    report["variants"] = variants
    report["acceptance_flavor_round"] = acceptance(branch)
    report["verdict"] = {**branch["verdict"], "regeneration_requested": False}
    return report


def render_report_md(report: dict) -> str:
    f = report["fitted"]; o = report["objective_at_optimum"]; v = report["verdict"]; ident = report["identity_point"]
    errors = (report["hessian"] or {}).get("errors") or {}
    lines = [f"# R66: joint flavor fit on the R51i topology point ({RUN_ID})", "",
             f"Card {CARD}. Generation {report['generation']}; floor {report['continuation_floor']} GeV; topology fixed at {R51I_RUN_ID}.",
             "Primary fit: fractions at the external anchors, shared normalization fixed at eta = 0, r_V = 1.", "",
             "| flavor weight | value | error |", "|---|---|---|"]
    for name in FLAVOR_ORDER:
        err = errors.get(name)
        lines.append(f"| {name} | {f[name]:.4g} | {'fixed' if name not in FLAVOR_FREE else ('' if err is None else f'{err:.3g}')} |")
    lines += ["", "| segment | chi2 at identity | chi2 at optimum |", "|---|---|---|"]
    for name in FIT_SPECIES:
        lines.append(f"| {name} | {ident['chi2_by_segment'][name]:.1f} | {o['chi2_by_segment'][name]:.1f} |")
    lines += ["", f"Objective at optimum {o['value']:.3f} (identity point {ident['objective']:.3f}); smallest class ESS {o['ess_min']:.3f}.",
              f"Converged: {report['optimizer']['converged']}. Hessian positive definite: {report['hessian'].get('positive_definite')}.",
              f"Verdict: {'CONSTRAINED' if v['constrained'] else 'NOT_CONSTRAINED'}.", "",
              "## Total yields per inelastic event (model / data)", "", "| species | model | data | ratio | ratio at identity |", "|---|---|---|---|---|"]
    ty = report["readout_total_yields"]
    for name, ratio in ty["ratio_model_over_data"].items():
        lines.append(f"| {name} | {ty['model'][name]:.4f} | {ty['data'][name]['value']} | {ratio:.3f} | {ident['total_yield_ratios'][name]:.3f} |")
    lines.append(f"\nMean charged multiplicity {ty['model']['charged_multiplicity']:.2f}.")
    lines += ["", "## Class breakdown (model / data per inelastic event)", "", "| class | fraction | " + " | ".join(FIT_SPECIES) + " |",
              "|---|---|" + "---|" * len(FIT_SPECIES)]
    for cls, row in report["readout_class_yields"].items():
        lines.append(f"| {cls} | {row['fraction']:.3f} | " + " | ".join(f"{row['ratio_model_over_data'][s]:.2f}" for s in FIT_SPECIES) + " |")
    pt = report["readout_pt_profile"]
    p0 = report["readout_pi0"]
    lines += ["", f"pi- p_T profile (x_F <= {pt['xf_max']}): model/data {pt['ratio_low']:.3f} at {pt['pt_low']} GeV/c, {pt['ratio_high']:.3f} at {pt['pt_high']} GeV/c; profile ratio {pt['profile_ratio']:.2f}.",
              f"pi0 per event {p0['pi0_per_event']:.3f}; ratio to the charged-pion mean {p0['ratio']:.2f}.", "",
              "## Mesonic count per decay against MARK I (fitted flavor point)", "", "| M [GeV] | model | MARK I | ratio |", "|---|---|---|---|"]
    for row in report["readout_mesonic_count"]["rows"]:
        lines.append(f"| {row['mass']} | {row['model']:.2f} | {row['mark_i']:.2f} | {row['ratio']:.3f} |")
    acc = report["acceptance_flavor_round"]
    lines += ["", f"## Flavor round acceptance: {'ACCEPTED' if acc['accepted'] else 'NOT ACCEPTED'}", ""]
    for name, c in acc["criteria"].items():
        lines.append(f"- {name}: {c['rule']}; {json.dumps(c.get('values', c.get('value')))}; met {c['met']}")
    lines += ["", "## Variants", "", "| variant | gamma_s_had | assoc_norm | r_bbar | objective | K+ | K- | pbar |", "|---|---|---|---|---|---|---|---|"]
    for label, var in (report.get("variants") or {}).items():
        fv = var["fitted"]; yr = var["total_yield_ratios"]
        lines.append(f"| {label} | {fv['gamma_s_had']:.3f} | {fv['assoc_norm']:.3f} | {fv['r_bbar']:.3f} | {var['objective']:.1f} | "
                     f"{yr['k_plus']:.2f} | {yr['k_minus']:.2f} | {yr['antiproton']:.2f} |")
    vf = report["variant_free_fractions"]
    lines.append(f"\nFree-fraction variant: f_SD {vf['fitted']['f_sd_total']:.4f}, f_DD {vf['fitted']['f_dd']:.4f}, "
                 f"gamma_s_had {vf['fitted']['gamma_s_had']:.3f}, assoc_norm {vf['fitted']['assoc_norm']:.3f}, r_bbar {vf['fitted']['r_bbar']:.3f}; "
                 f"objective {vf['objective']['value']:.1f}.")
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
    r51i_dir = MONO_ROOT / "dev" / "00runs" / R51I_RUN_ID
    if not (r51i_dir / "seal_manifest.json").is_file():
        contract.refuse(f"the sealed R51i archive {r51i_dir} is required for the fixed topology point")
    r51i_report = RC.read_json(r51i_dir / RC.REPORT_NAME)
    fitted = r51i_report["fitted"]
    if (any(float(fitted[k]) != v for k, v in FIXED_TOPOLOGY.items())
            or float(r51i_report["continuation_floor"]) != FLOOR_GRID[0] or not r51i_report["verdict"]["constrained"]):
        contract.refuse("the fixed topology constants do not match the sealed, constrained R51i optimum")
    a = anchors()
    if (float(fitted["f_sd_total"]), float(fitted["f_dd"])) != (a["f_sd_total"]["value"], a["f_dd"]["value"]):
        contract.refuse("the R51i fractions differ from the external anchors")
    _load_physics(FLOOR_GRID[0])  # refuses a committed config that already sets a floor
    payload = {
        "classes": list(CLASS_ORDER), "replicas": REPLICAS, "events_per_slot": EVENTS_PER_SLOT, "max_generations": MAX_GENERATIONS,
        "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST,
                  "rule": f"raw seed = RAW_FIRST + {N_SLOTS} * generation + slot; slot = class_index * {REPLICAS} + replica",
                  "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * slot + i", "disjointness": disjoint},
        "fixed_topology": {**FIXED_TOPOLOGY, "continuation_floor": FLOOR_GRID[0], "fractions": {k: a[k]["value"] for k in FRACTION_NAMES},
                           "provenance": {"run": R51I_RUN_ID, "report_digest": r51i_report.get("digest"), "verdict": "CONSTRAINED"},
                           "nd_generation_remnant_weights": ND_GENERATION_REMNANT_WEIGHTS, "diffractive_split": FROZEN_REMNANT_WEIGHTS},
        "flavor": {"free": list(FLAVOR_FREE), "fixed": {"r_v": R_V_FIXED, "reason": "no K* or phi table (the R63 prior-dominance branch)"},
                   "generation_0": dict(zip(FLAVOR_ORDER, IDENTITY_FLAVOR)), "log_bounds": [math.exp(b) for b in FLAVOR_LOG_BOUNDS],
                   "scope": "continuum-parent vertices only; established resonances keep their branching fractions (plan section 12.2)",
                   "reweighting": "exact, through the per-vertex flavor ledger (flavor_fit_core.flavor_log_weights)",
                   "hidden_ssbar_content": "the flavor_weights default (phi 1, eta and eta-prime 0)"},
        "variants": {"free_fractions": {"free_parameters": list(param_names(True)), "starts": N_STARTS_VARIANT},
                     "discrepancy": DISCREPANCY_VARIANT, "normalization": list(NORM_VARIANTS),
                     "role": "reported next to the primary fit; never drive the regeneration"},
        "discrepancy": {"primary": DISCREPANCY, "variant": DISCREPANCY_VARIANT,
                        "rule": "variance (delta x data)^2 per bin added to the data and Monte Carlo variances",
                        "reason": "the 5% scale of inter-experiment systematic offsets (Fischer et al. 2022)"},
        "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS),
                          "decision": "owner, 2026-10-02: fixed at 0 with +-2% variants; the profiled 2% prior of R51f to R51i was overwhelmed"},
        "trust_region": {"ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR, "penalty": TRUST_PENALTY},
        "data": {"fit_window": FIT_WINDOW, "species": list(FIT_SPECIES), "observables": "the double-differential invariant cross sections per inelastic event",
                 "anchors": a},
        "held_out": ["lead_xf", "gap", "n_ch", "pi- p_T profile", "mesonic count against MARK I at the fitted flavor point", "SD n_ch against M_X",
                     "pi0 yield", "neutron dN/dx_F against the NA49 table"],
        "final_state": {"decays": "every unstable established hadron decays through the Round 2 kernel before it reaches a final state",
                        "isospin_rule": RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json").get("isospin_rule")},
        "acceptance_flavor_round": {"kaon_total_yields": f"K+ and K- within {KAON_YIELD_BAND}", "antiproton_total_yield": f"pbar within {PBAR_YIELD_BAND}",
                                    "differential_clause": "for K+, K- and pbar the species chi2 falls from the identity point to the optimum; a yield ratio moving toward 1 with a rising chi2 raises the reject flag",
                                    "failure_rule": "a failed criterion names the next mechanism; the flavor weights are not given new freedom in this run"},
        "verdict_rule": {"constrained": "converged and ess_min >= ess_floor and no flavor weight on its bound and Hessian positive definite and block spreads within 1.5 x pooled error x sqrt(32)",
                         "stability_ratio_bound": STABILITY_RATIO_BOUND},
        "optimizer": {"method": "Powell (bounded, log flavor weights)", "starts": N_STARTS, "hessian_log_step": HESSIAN_LOG_STEP},
        "frozen_substrate": frozen,
    }
    sync_path = MONO_ROOT / SYNC_MANIFEST_NAME
    payload["source"] = {"git": {"outer": RC.git_info(MONO_ROOT), "nested": RC.git_info(NESTED_ROOT)},
                         "sync_manifest": RC.read_json(sync_path) if sync_path.is_file() else None,
                         "rule": "the local git heads when the checkout is a repository; otherwise the gateway sync record"}
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()})
    for f in range(N_FLOORS):
        RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), {"points": [
            {"floor_index": f, "generation": 0, **dict(zip(FLAVOR_ORDER, IDENTITY_FLAVOR))}]})


def _verify_frozen(contract: RC.RunContract) -> dict:
    pre = contract.load_pre_registration()
    frozen = {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}
    status = RC.verify_frozen_substrate(MONO_ROOT, frozen, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    return status


def _points(contract: RC.RunContract, floor_index: int) -> list[dict]:
    return RC.read_json(contract.path(GENERATION_POINTS_FMT.format(f=int(floor_index))))["points"]


def _generation_point(contract: RC.RunContract, floor_index: int, generation: int) -> tuple:
    """The registered generation flavor point (FLAVOR_ORDER)."""
    for p in _points(contract, floor_index):
        if int(p["generation"]) == int(generation):
            return tuple(float(p[name]) for name in FLAVOR_ORDER)
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
        point = _generation_point(contract, f, g)
        for slot in todo_slots:
            if contract.path(slot_names(f, g, slot)[0]).is_file():
                continue
            jobs.append(((f, g, slot), (slot_class(slot), slot_seed(f, g, slot), EVENTS_PER_SLOT, PARTITION_BASE + 100_000 * slot,
                                         point, FLOOR_GRID[f])))
    if not jobs:
        print(f"[{RUN_ID}] nothing to generate", flush=True)
        return
    t0 = time.perf_counter()
    results = RC.run_replicas(_generate_job, [job[1] for job in jobs], max_workers=workers)
    for ((f, g, slot), _), result in zip(jobs, results):
        write_slot(contract, f, g, slot, result, pre_sha)
    print(f"[{RUN_ID}] generated {len(jobs)} slots over branches {sorted({j[0][0] for j in jobs})} in {time.perf_counter() - t0:.0f} s", flush=True)


def _generate_job(cls, seed, n_events, partition_base, flavor_point, floor):
    return generate_slot(cls, seed, n_events, partition_base, flavor_point, floor)


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
                {"floor_index": f, "generation": nxt, **dict(zip(FLAVOR_ORDER, flavor_of(report["fitted"]))),
                 "requested_by_generation": g, "ess_min": report["objective_at_optimum"]["ess_min"]}]
            RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), points)
            RC.write_json(contract.path(REGENERATION_FMT.format(f=f)), {"schema": f"{SCHEMA}.regeneration/1", **contract.identity(),
                                                                         "floor_index": f, "from_generation": g, "to_generation": nxt,
                                                                         "ess_min": report["objective_at_optimum"]["ess_min"], "written_utc": RC.now_utc()})
            print(f"[{RUN_ID}] generation {g}: ess_min {report['objective_at_optimum']['ess_min']:.3f} < {ESS_INTERIOR}; "
                  f"regeneration {nxt} requested at (gamma_s, assoc, r_bbar) = ({report['fitted']['gamma_s_had']:.4f}, "
                  f"{report['fitted']['assoc_norm']:.4f}, {report['fitted']['r_bbar']:.4f})", flush=True)
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


VARIANTS = (("discrepancy_0.10", DISCREPANCY_VARIANT, NORM_FIXED), ("normalization_+0.02", DISCREPANCY, NORM_VARIANTS[0]),
            ("normalization_-0.02", DISCREPANCY, NORM_VARIANTS[1]))


def _variants(contract: RC.RunContract, chosen: dict, tables: dict) -> dict:
    """The primary parametrization refitted at the discrepancy and the
    normalization variants on the branch's final generation."""
    f, g = int(chosen["floor_index"]), int(chosen["generation"])
    templates = None
    out = {}
    for label, dz, norm in VARIANTS:
        starts = [x_from_physical(chosen["fitted"], False)] + dispersed_starts(anchors(), free=False)[1:N_STARTS_VARIANT]
        fit = run_fit_parallel(contract, f, g, starts, free=False, discrepancy=dz, norm=norm, maxfev=4000,
                               pool_workers=BLOCK_WORKERS_PER_VARIANT_START)
        with BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS) as pool:
            value, det = PooledObjective(tables, anchors(), pool, discrepancy=dz, norm=norm).evaluate(fit["x"], details=True)
        if templates is None:
            templates = Templates(contract, f, g, tables)
        objective = Objective(templates, anchors(), discrepancy=dz, norm=norm)
        readouts = _mixture_readouts(templates, objective, tables, det["physical"])
        out[label] = {"discrepancy": dz, "normalization": norm, "fitted": {n: det["physical"][n] for n in FLAVOR_ORDER},
                      "objective": value, "chi2_by_segment": det["chi2_by_segment"], "ess_min": det["ess_min"], "converged": fit["converged"],
                      "total_yield_ratios": readouts["readout_total_yields"]["ratio_model_over_data"]}
    return out


def step_profile(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    chosen = _load_branches(contract)[0]
    variants = _variants(contract, chosen, load_fit_tables())
    report = compose_report(chosen, variants)
    body = contract.write_report(report, render_report_md(report))
    print(f"[{RUN_ID}] report written: verdict {'CONSTRAINED' if body['verdict']['constrained'] else 'NOT_CONSTRAINED'}, flavor round "
          f"{'ACCEPTED' if body['acceptance_flavor_round']['accepted'] else 'NOT ACCEPTED'}, digest {body['digest'][:16]}", flush=True)


def step_gate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    pions = report["readout_paper_window_sum_ratios"]
    pion_ratio = 0.5 * (pions["pi_plus"]["sum_ratio"] + pions["pi_minus"]["sum_ratio"])
    record = {
        "schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
        "obligation": "Round 6 flavor round (stage 4 of the path to the global fit): the joint flavor fit on the constrained R51i topology point (R66)",
        "verdict": "CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED",
        "fitted_flavor": {name: report["fitted"][name] for name in FLAVOR_ORDER},
        "errors": (report["hessian"] or {}).get("errors"), "errors_log": (report["hessian"] or {}).get("errors_log"),
        "ess_min": report["objective_at_optimum"]["ess_min"], "converged": report["optimizer"]["converged"],
        "generation": report["generation"], "named_residual_register": report["named_residual_register"],
        "acceptance_flavor_round": report["acceptance_flavor_round"],
        "total_yield_ratios": report["readout_total_yields"]["ratio_model_over_data"],
        "identity_point": report["identity_point"],
        "variants": {label: {"fitted": v["fitted"], "total_yield_ratios": v["total_yield_ratios"]} for label, v in (report["variants"] or {}).items()},
        "readout": {"pion_sum_ratio_mean": pion_ratio, "per_species": pions},
        "report_digest": report["digest"], "written_utc": RC.now_utc(),
    }
    RC.write_json(contract.path(GATE_RECORD_NAME), record)
    print(f"[{RUN_ID}] gate record: {record['verdict']}, flavor {json.dumps({k: round(v, 3) for k, v in record['fitted_flavor'].items()})}, "
          f"flavor round {'ACCEPTED' if record['acceptance_flavor_round']['accepted'] else 'NOT ACCEPTED'}", flush=True)


def step_clean_validate(run_dir: Path | None) -> None:
    """Re-evaluate the final generation and the variants, recompose the
    report, and regenerate the probe slot."""
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
    rerun = compose_report(rerun_branch, _variants(contract, rerun_branch, tables))
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    point = _generation_point(contract, f, g)
    probe = generate_slot(slot_class(0), slot_seed(f, g, 0), EVENTS_PER_SLOT, PARTITION_BASE, point, FLOOR_GRID[f])
    sealed_probe = RC.read_json(contract.path(slot_names(f, g, 0)[0]))
    probe_match = bool(arrays_digest(probe["arrays"]) == sealed_probe["arrays_sha256"]
                       and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"])
    payload = {"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}",
                            "cd dev/mixing2006 && python -m pytest tests/test_staged_r66_joint_flavor_fit.py -q"],
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
    jobs = [(cls, SMOKE_SEED + i, SMOKE_EVENTS_PER_CLASS, SMOKE_PARTITION + 10_000 * i, IDENTITY_FLAVOR, FLOOR_GRID[f])
            for i, cls in enumerate(CLASS_ORDER)]
    results = RC.run_replicas(_generate_job, jobs)
    for cls, result in zip(CLASS_ORDER, results):
        np.savez(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, f=f, g=0)), **result["arrays"])
        RC.write_json(contract.path("smoke_" + SLOT_JSON_FMT.format(cls=cls, k=0, f=f, g=0)), {
            "schema": f"{SCHEMA}.smoke_slot/1", "pre_registration_sha256": pre_sha, "class_label": cls, "generation": 0, "replica": 0,
            "floor_index": f, "continuation_floor": result["continuation_floor"], "flavor_generation": result["flavor_generation"],
            "informative_vertices": result["informative_vertices"],
            "seed": result["seed"], "n_events": result["n_events"], "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
            "arrays_sha256": arrays_digest(result["arrays"]), "event_rolling_sha256": result["event_rolling_sha256"],
            "conservation": result["conservation"], "tracks": result["tracks"]})
    report = evaluate_generation(contract, f, 0, load_fit_tables(), anchors(), smoke=True)
    tables = load_fit_tables()
    smoke_templates = Templates(contract, f, 0, tables, smoke=True)
    serial = Objective(smoke_templates, anchors())
    check_point = {**report["fitted"], "gamma_s_had": 0.7, "assoc_norm": 0.8, "r_bbar": 3.0}
    x_check = x_from_physical(check_point, False)
    with BlockPool(contract.run_dir, f, 0, 1, smoke=True) as pool:
        pooled_value = PooledObjective(tables, anchors(), pool).evaluate(x_check)
    serial_value = serial.evaluate(x_check)
    report["pool_check"] = {"in_process": serial_value, "pooled": pooled_value,
                            "relative_difference": abs(pooled_value - serial_value) / max(abs(serial_value), 1e-300)}
    # the key-table reweighting against flavor_fit_core.flavor_log_weights
    worst = 0.0
    for i, cls in enumerate(CLASS_ORDER):
        with np.load(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, f=f, g=0))) as data:
            arrays = {key: data[key] for key in data.files}
        block = smoke_templates.by_class[cls][0]
        mine = smoke_templates.log_weights(block, flavor_of(check_point))
        theirs = FC.flavor_log_weights(arrays, block["n"], NOMINAL_FLAVOR, flavor_params(flavor_of(check_point)))
        worst = max(worst, float(np.max(np.abs(mine - theirs))) if mine.size else 0.0)
    report["ledger_check"] = {"flavor_point": dict(zip(FLAVOR_ORDER, flavor_of(check_point))), "max_abs_log_weight_difference": worst}
    report["schema"] = f"{SCHEMA}.smoke_fit_report/1"
    report["acceptance_flavor_round"] = acceptance(report)
    report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_fit_report.json"), report)
    print(f"[{RUN_ID}] smoke: fitted {json.dumps({k: round(report['fitted'][k], 4) for k in FLAVOR_ORDER})}, "
          f"ess_min {report['objective_at_optimum']['ess_min']:.3f}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"R66 driver ({CARD})")
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
