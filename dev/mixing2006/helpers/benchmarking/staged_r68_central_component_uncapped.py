"""R68 driver: the test of the central component on the uncapped cascade
(Round 6; card TASK-20261002-r6-central-component-uncapped-r68).

R67 rejected f_central = 0 (Delta chi2 5,171 against 188).  The diagnosis
that followed (_knowledge, "Why the baryonic cascade cannot fill the
central region") found the 5 GeV support cap of the continuum families to
be the main defect: a continuation daughter could not be realized above
SUPPORT_M_MAX = 5.0 GeV, so a heavier excited system shed its excess into
its first emitted hadron.  The owner's picture has the resonance tower
cascading up to the perturbative limit, with no cap at 5 GeV (owner:
"go").

Differences from R67 (everything else is the R67 contract, including the
pre-registered test rule):
- The continuation ceiling of every continuum family is set in memory to
  CONTINUATION_CEILING = 17.3 GeV (above sqrt(s) = 17.27 GeV), so every
  continuation step realizes its daughter one nominal width below the
  budget edge (the kernel's continuation_ceiling, 2026-10-02).  The
  source density, its support and n_norm keep the support edge.
- Generation 0 at the sealed R67 free optimum (shapes and remnant
  weights); the flavor point stays the sealed R66 point.
- New seeds (ledger block r68_central_component_uncapped).
- The diffractive exponent bound opens to (-0.5, 3.0): R67's null fit
  sat on 1.5.  The interior and bound checks of the verdict apply to the
  free and the null fit (R67 checked the free fit only).
The test outcome is reported as measured, as in R67.
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
RUN_ID = "r68_central_component_uncapped_2026-10-02"
CARD = "TASK-20261002-r6-central-component-uncapped-r68"
SCHEMA = "na49_mixing.staged_r68_central_component_uncapped"
LEDGER_RUN = "r68_central_component_uncapped"

CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd", "nd0")
ND_CLASSES = ("nd", "nd0")
#: the runner class of each template class (nd0 is the runner's nd with
#: nd_central=False)
RUNNER_CLASS = {"nd0": "nd"}
REPLICAS = 32
EVENTS_PER_SLOT = 100_000
N_SLOTS = len(CLASS_ORDER) * REPLICAS  # 160
#: the slots that draw their own seeds (nd0 shares the nd seeds)
SEEDED_SLOTS = (len(CLASS_ORDER) - 1) * REPLICAS  # 128
MAX_GENERATIONS = 4
FLOOR_GRID = (1.7,)  # the R51i floor, fixed
N_FLOORS = len(FLOOR_GRID)

MASTER_SEED = 20261002
RAW_FIRST = 20316000
RAW_LAST = RAW_FIRST + MAX_GENERATIONS * SEEDED_SLOTS - 1  # 20316511
PARTITION_BASE = 3_900_000_000
SMOKE_SEED = 20304108
SMOKE_PARTITION = 3_120_000_000
# the continuation ceiling set in memory for every continuum family
CONTINUATION_CEILING = 17.3
SMOKE_EVENTS_PER_CLASS = 500
SMOKE_FLOOR_INDEX = 0
SMOKE_MESONIC_DECAYS = 200

SQRT_S = OB.SQRT_S
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
EPS_BOUNDS = (-0.5, 3.0)  # widened from R67's 1.5, where its null fit sat on the bound
EPS_ND_BOUNDS = (-1.5, 1.5)  # widened from R51i's 0.5: both fits of the test must reach their minima
B_SD_BOUNDS = (1.0, 20.0)
LOGIT_BOUNDS = (-8.0, 8.0)
FC_BOUNDS = (0.0, 1.0)
FRACTION_INTERIOR = (0.005, 0.995)
HESSIAN_STEPS = {"epsilon": 0.01, "b_sd": 0.1, "w_excited": 0.02, "epsilon_nd": 0.02}
SHAPE_NAMES = ("epsilon", "b_sd", "w_excited", "epsilon_nd")
# the central-component test (pre-registered, owner decision 2026-10-02)
FC_SCAN = (0.1, 0.25, 0.5, 0.75, 1.0)
TEST_DELTA_CHI2 = 3.84
N_FREE_PARAMETERS = 5
N_STARTS_NULL = 4
# the flavor weights, in FlavorParams order: fixed at the sealed R66 point
FLAVOR_ORDER = ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
R66_RUN_ID = "r66_joint_flavor_fit_2026-10-02"
GEN_FLAVOR = (0.6533549042510345, 1.0, 1.6352390738697378, 0.26023655383916866)
# the shared normalization: fixed (owner decision 2026-10-02), with variants
NORM_FIXED = 0.0
NORM_VARIANTS = (0.02, -0.02)
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
FRACTION_NAMES = ("f_sd_total", "f_dd")
WEIGHT_CACHE_ENTRIES = 64
SE_CACHE_ENTRIES = 512
# generation 0: the sealed R67 free optimum (the capped cascade)
R67_RUN_ID = "r67_central_component_test_2026-10-02"
GEN0_EPSILON = 1.1924569206590134
GEN0_B_SD = 2.5964485709629486
GEN0_EPSILON_ND = 0.35169862901173005
GEN0_W_LOW = 0.03216360693582332
GEN0_W_EXCITED = 0.9678363930641767
GEN0_SHAPE = (GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND)
REMNANT_KINDS = ("intact_proton", "low_mass_baryonic_resonance", "high_mass_baryonic_continuum")
# both non-diffractive variants are generated at the fitted R51i remnant
# weights; the fit reweights the kind pair from there
ND_GENERATION_REMNANT_WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": GEN0_W_LOW,
                                 "high_mass_baryonic_continuum": GEN0_W_EXCITED}
FROZEN_REMNANT_WEIGHTS = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}

FIT_WINDOW = {"xf_min": 0.0, "xf_max": 0.95, "pt_min": 0.05, "pt_max": 1.5}
PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
FIT_SPECIES = ("proton", "antiproton", "pi_minus", "pi_plus", "k_plus", "k_minus")
TABLE_WINDOWS = {name: FIT_WINDOW for name in FIT_SPECIES}
TRACK_SPECIES = OB.SPECIES_ORDER

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
    "helpers/benchmarking/staged_r66_joint_flavor_fit.py",
    "helpers/benchmarking/staged_r67_central_component_test.py",
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
GATE_RECORD_NAME = "gate6_central_component_r68.json"
GENERATION_POINTS_FMT = "generation_points_f{f}.json"
# the gateway sync record (written by orchestrate_gateway_joint_fit.py at the remote
# root; the remote checkout carries no git metadata)
SYNC_MANIFEST_NAME = "SYNC_MANIFEST.json"


def slot_class(slot: int) -> str:
    return CLASS_ORDER[int(slot) // REPLICAS]


def slot_replica(slot: int) -> int:
    return int(slot) % REPLICAS


def seed_slot(slot: int) -> int:
    """The slot whose seed and event ids a slot draws: an nd0 replica
    shares those of the nd replica with the same index (common random
    numbers between the two non-diffractive variants)."""
    cls, k = slot_class(slot), slot_replica(slot)
    return CLASS_ORDER.index("nd") * REPLICAS + k if cls == "nd0" else int(slot)


def slot_seed(floor_index: int, generation: int, slot: int) -> int:
    return RAW_FIRST + SEEDED_SLOTS * int(generation) + seed_slot(slot)


def slot_partition(slot: int) -> int:
    return PARTITION_BASE + 100_000 * seed_slot(slot)


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
    floor and the lifted continuation ceiling set in memory for every
    family (the config itself sets neither)."""
    catalog = HadronCatalog.from_json(NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json")
    fams = families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    if any(f.continuation_floor is not None or f.continuation_ceiling is not None for f in fams):
        raise RuntimeError("the committed continuum config sets a continuation floor or ceiling; the driver sets them in memory")
    return catalog, [dataclasses.replace(f, continuation_floor=float(floor), continuation_ceiling=CONTINUATION_CEILING) for f in fams]


def flavor_params(point) -> FlavorParams:
    """The FlavorParams of a point in FLAVOR_ORDER (default hidden s-sbar content)."""
    return FlavorParams(**{name: float(v) for name, v in zip(FLAVOR_ORDER, point)})


def flavor_of(phys: dict) -> tuple:
    return tuple(float(phys[name]) for name in FLAVOR_ORDER)


def _params_at(cls: str, shape_point) -> TopologyNominalParams:
    """The generator parameters of a template class at the shape point
    (epsilon, B_SD, epsilon_nd) and the fixed R66 flavor point."""
    eps, b_sd, eps_nd = (float(v) for v in shape_point)
    weights = ND_GENERATION_REMNANT_WEIGHTS if cls in ND_CLASSES else FROZEN_REMNANT_WEIGHTS
    return TopologyNominalParams(
        epsilon=eps, b_sd=b_sd, class_fractions=dict(NOMINAL_PARAMS.class_fractions),
        remnant_weights=dict(weights), p_n2=0.0, flavor=flavor_params(GEN_FLAVOR),
        epsilon_nd=eps_nd, nd_central=(cls != "nd0"),
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


def generate_slot(cls: str, seed: int, n_events: int, partition_base: int, shape_point, floor: float) -> dict:
    """Generate one slot of a template class at the shape point and the
    fixed R66 flavor point, with the flavor ledger of every drawn
    continuum vertex: returns the compact arrays and the metadata."""
    catalog, fams = _load_physics(floor)
    shape_point = tuple(float(v) for v in shape_point)
    flavor_point = tuple(float(v) for v in GEN_FLAVOR)
    params = _params_at(cls, shape_point)
    runner_cls = RUNNER_CLASS.get(cls, cls)
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
        event = run_topology(runner_cls, params, catalog, fams, **rngs, ledger=ledger)
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
        if cls in ND_CLASSES:
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
        "epsilon": shape_point[0], "b_sd": shape_point[1], "epsilon_nd": shape_point[2], "continuation_floor": float(floor),
        "flavor_generation": dict(zip(FLAVOR_ORDER, flavor_point)), "nd_central": bool(params.nd_central),
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
        "nd_central": result["nd_central"],
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
                           alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=("r68_readout",), depth=0)
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


def mesonic_count_readout(floor: float, n_decays: int = MESONIC_DECAYS, flavor_point=GEN_FLAVOR) -> dict:
    """The mean charged count per decay of a mesonic_neutral source at rest
    at MESONIC_MASSES against the MARK I curve (the charged products of
    the final neutral weak decayers counted as in the measurement), at the
    flavor point *flavor_point*."""
    flavor = flavor_params(flavor_point)
    catalog, fams = _load_physics(floor)
    family = next(f for f in fams if f.family_id == "mesonic_neutral")
    rows = []
    for m_index, mass in enumerate(MESONIC_MASSES):
        key = [MASTER_SEED, 68, int(round(1000 * floor)), m_index]
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
            "seed_rule": "numpy SeedSequence([MASTER_SEED, 68, round(1000 floor), mass index, 1 or 2])", "rows": rows}


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
class Templates:
    """The loaded templates of the branch and one generation: per class and
    replica the shape arrays and the per-species binned track sums."""

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
            if tuple(float(meta["flavor_generation"][name]) for name in FLAVOR_ORDER) != GEN_FLAVOR:
                contract.refuse(f"{json_name} was not generated at the fixed R66 flavor point")
            if bool(meta.get("nd_central", True)) != (cls != "nd0"):
                contract.refuse(f"{json_name} carries the wrong central-source setting for class {cls}")
            with np.load(contract.path(npz_name)) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays_digest(arrays) != meta["arrays_sha256"]:
                contract.refuse(f"the arrays of {npz_name} do not match their recorded digest")
            point = (float(meta["epsilon"]), float(meta["b_sd"]), float(meta["epsilon_nd"]))
            if self.point is None:
                self.point = point
            elif self.point != point:
                contract.refuse(f"mixed generation points in branch {floor_index} generation {generation}")
            if cls in ND_CLASSES and np.any(arrays["rem_kind"] == REMNANT_KINDS.index("intact_proton")):
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
        """The per-event log importance weights from the generation shape
        point; both non-diffractive variants carry the nd mass law (nd0
        without its central term) and the exact remnant kind-pair ratio."""
        eps0, b0, eps0_nd = self.point
        nd_like = cls in ND_CLASSES
        eps_c, eps0_c = (float(eps_nd), eps0_nd) if nd_like else (float(eps), eps0)
        law = "nd" if nd_like else cls
        if (eps_c, float(b_sd)) == (eps0_c, b0):
            log_w = np.zeros(block["n"])
        else:
            log_w = log_density_ratio(law, block["shape"], eps_c, b_sd, eps0=eps0_c, b_sd0=b0, sqrt_s=SQRT_S)
        if nd_like and remnant is not None:
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


def param_names(fc_fixed) -> tuple:
    return SHAPE_NAMES + (() if fc_fixed is not None else ("f_central",))


def x_from_physical(phys: dict, fc_fixed=None) -> np.ndarray:
    """The optimizer vector [epsilon, ln B_SD, logit w_excited, epsilon_nd]
    + [f_central] (f_central free only); w_intact = 0."""
    w_exc = min(max(float(phys["w_excited"]), 1e-12), 1.0 - 1e-12)
    core = [float(phys["epsilon"]), math.log(float(phys["b_sd"])), math.log(w_exc / (1.0 - w_exc)), float(phys["epsilon_nd"])]
    if fc_fixed is None:
        core.append(float(phys["f_central"]))
    return np.asarray(core, dtype=float)


def physical_from_x(x, fc_fixed, anchors_: dict) -> dict:
    x = np.asarray(x, dtype=float)
    eps, ln_b, r_exc, eps_nd = (float(v) for v in x[:4])
    f_central = float(fc_fixed) if fc_fixed is not None else float(x[4])
    a_sd, a_dd = float(anchors_["f_sd_total"]["value"]), float(anchors_["f_dd"]["value"])
    w_exc = 1.0 / (1.0 + math.exp(-r_exc))
    return {"f_sd_total": a_sd, "f_dd": a_dd, "f_nd": 1.0 - a_sd - a_dd, "f_central": f_central,
            "epsilon": eps, "b_sd": float(math.exp(ln_b)), "w_intact": 0.0, "w_low_mass": float(1.0 - w_exc),
            "w_excited": float(w_exc), "epsilon_nd": eps_nd, **dict(zip(FLAVOR_ORDER, GEN_FLAVOR))}


def class_fractions(phys: dict) -> dict:
    """The template-class fractions: the anchors, with the non-diffractive
    share split by f_central between nd (with the cluster) and nd0."""
    fc = float(phys["f_central"])
    return {"sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0, "dd": phys["f_dd"],
            "nd": phys["f_nd"] * fc, "nd0": phys["f_nd"] * (1.0 - fc)}


def bounds_for(fc_fixed) -> list:
    core = [EPS_BOUNDS, (math.log(B_SD_BOUNDS[0]), math.log(B_SD_BOUNDS[1])), LOGIT_BOUNDS, EPS_ND_BOUNDS]
    return core + ([FC_BOUNDS] if fc_fixed is None else [])


def _in_bounds(x, fc_fixed) -> bool:
    return all(lo - 1e-12 <= float(v) <= hi + 1e-12 for v, (lo, hi) in zip(np.asarray(x, dtype=float), bounds_for(fc_fixed)))


class Objective:
    """The cached joint objective at one generation.  ``fc_fixed`` holds
    f_central fixed (the null fit at 0, the profile scan) or leaves it
    free (None); ``discrepancy`` is the fractional model-discrepancy scale
    per bin; ``norm`` the fixed shared normalization offset eta."""

    def __init__(self, templates: Templates, anchors_: dict, fc_fixed=None, discrepancy: float = DISCREPANCY,
                 norm: float = NORM_FIXED):
        self.t = templates
        self.anchors = anchors_
        self.fc_fixed = None if fc_fixed is None else float(fc_fixed)
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
        return physical_from_x(x, self.fc_fixed, self.anchors)

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
                log_w = self.t.log_weights(cls, block, eps, b_sd, eps_nd, remnant if cls in ND_CLASSES else None)
                weights[cls].append(self.t.normalized(log_w))
                ess_vals.append(ess_fraction(log_w))
            ess[cls] = min(ess_vals) if ess_vals else 0.0
        self.cache[key] = (weights, ess)
        while len(self.cache) > WEIGHT_CACHE_ENTRIES:
            self.cache.pop(next(iter(self.cache)))
        return weights, ess

    def block_prediction(self, k: int, frac: dict, weights: dict) -> dict:
        """The invariant cross sections of the fitted species in replica block k."""
        out = {}
        for name, area in self.areas.items():
            m = np.zeros(area.size)
            for cls in CLASS_ORDER:
                m += frac[cls] * self.t.species_mean(cls, self.t.by_class[cls][k], weights[cls][k], name)
            out[name] = m / (math.pi * SQRT_S * area)
        return out

    def observation_vector(self, pred: dict) -> np.ndarray:
        return np.concatenate([pred[name] for name in FIT_SPECIES])

    def unweighted_se(self, frac: dict, block_ids: list[int]) -> np.ndarray:
        """The between-block standard error of the mixture at the trial
        class fractions with unit weights (cached)."""
        key = ("se",) + tuple(round(float(frac[c]), 12) for c in CLASS_ORDER) + (tuple(block_ids),)
        if key in self.se_cache:
            return self.se_cache[key]
        if not hasattr(self, "_unit_weights"):
            self._unit_weights = {cls: [np.full(block["n"], 1.0 / block["n"]) for block in self.t.by_class[cls]] for cls in CLASS_ORDER}
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, frac, self._unit_weights)) for k in block_ids])
        se = np.std(vectors, axis=0, ddof=1) / math.sqrt(vectors.shape[0]) if vectors.shape[0] > 1 else np.zeros(vectors.shape[1])
        se = np.maximum(se, SE_FLOOR)
        self.se_cache[key] = se
        while len(self.se_cache) > SE_CACHE_ENTRIES:
            self.se_cache.pop(next(iter(self.se_cache)))
        return se

    @staticmethod
    def _physical_ok(phys: dict) -> bool:
        return (EPS_BOUNDS[0] <= phys["epsilon"] <= EPS_BOUNDS[1] and B_SD_BOUNDS[0] <= phys["b_sd"] <= B_SD_BOUNDS[1]
                and EPS_ND_BOUNDS[0] <= phys["epsilon_nd"] <= EPS_ND_BOUNDS[1] and FC_BOUNDS[0] <= phys["f_central"] <= FC_BOUNDS[1])

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        phys = self.physical(x)
        if not self._physical_ok(phys):
            return (float("inf"), None) if details else float("inf")
        weights, ess = self.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"], blocks,
                                          remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
        frac = class_fractions(phys)
        block_ids = list(range(self.t.n_blocks)) if blocks is None else list(blocks)
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, frac, weights)) for k in block_ids])
        scale = 1.0 + self.norm
        m = np.mean(vectors, axis=0) * scale
        se = self.unweighted_se(frac, block_ids) * scale
        return self._likelihood(phys, m, se, ess, block_ids, details)

    def _likelihood(self, phys: dict, m: np.ndarray, se: np.ndarray, ess: dict, block_ids: list, details: bool):
        # the trust region applies to the classes the mixture uses
        frac = class_fractions(phys)
        used = {cls: v for cls, v in ess.items() if frac[cls] > 0.0}
        ess_min = min(used.values())
        r = (self.d - m) / np.sqrt(self.v_data + se ** 2)
        data_term = float(r @ r)
        value = data_term
        if ess_min < ESS_FLOOR:
            value += TRUST_PENALTY * (ESS_FLOOR - ess_min) / ESS_FLOOR
        if not details:
            return value
        by_segment = {name: float(np.sum(r[lo:hi] ** 2)) for name, (lo, hi) in self.segments.items()}
        return value, {
            "physical": phys, "ess": ess, "ess_min": ess_min, "normalization": self.norm,
            "minus2lnl_data": data_term, "chi2_by_segment": by_segment,
            "m": m, "se": se, "residual_pull": r, "blocks": block_ids,
            "in_trust_region": bool(ess_min >= ESS_FLOOR), "f_central_fixed": self.fc_fixed,
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
    reweighted vectors and the per-class smallest ESS at a shape point)."""
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

    def __init__(self, tables: dict, anchors_: dict, pool: BlockPool, fc_fixed=None, discrepancy: float = DISCREPANCY,
                 norm: float = NORM_FIXED):
        stub = types.SimpleNamespace(tables=tables, n_blocks=pool.n_blocks, point=pool.point)
        super().__init__(stub, anchors_, fc_fixed=fc_fixed, discrepancy=discrepancy, norm=norm)
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
        if not self._physical_ok(phys):
            return (float("inf"), None) if details else float("inf")
        obs, ess = self.class_obs(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"],
                                  (phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
        fr = class_fractions(phys)
        frac = np.asarray([fr[c] for c in CLASS_ORDER])
        vectors = np.einsum("c,kco->ko", frac, obs)
        n = vectors.shape[0]
        scale = 1.0 + self.norm
        m = np.mean(vectors, axis=0) * scale
        unit = np.einsum("c,kco->ko", frac, self.pool.unit())
        se = np.std(unit, axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(unit.shape[1])
        se = np.maximum(se, SE_FLOOR) * scale
        return self._likelihood(phys, m, se, ess, list(range(n)), details)


def dispersed_starts(fc_fixed=None) -> list[np.ndarray]:
    """Dispersed starts in the shapes, and in f_central when it is free."""
    shapes = [(GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND), (0.3, 3.0, 0.2), (0.7, 2.0, 0.45), (0.5, 4.0, 0.0),
              (0.2, 2.5, 0.4), (0.8, 1.5, 0.3), (0.4, 6.0, -0.2), (0.6, 2.2, 0.1)]
    excited = [GEN0_W_EXCITED, 0.8, 0.9, 0.7, 0.95, 0.85, 0.6, 0.75]
    centrals = [1.0, 0.0, 0.5, 0.25, 0.75, 0.1, 0.9, 0.5]
    n = N_STARTS if fc_fixed is None else N_STARTS_NULL
    starts = []
    for (e, b, en), we, fc in list(zip(shapes, excited, centrals))[:n]:
        phys = {"epsilon": e, "b_sd": b, "w_excited": we, "epsilon_nd": en, "f_central": fc}
        x = x_from_physical(phys, fc_fixed)
        x[2] = float(np.clip(x[2], LOGIT_BOUNDS[0] + 0.5, LOGIT_BOUNDS[1] - 0.5))
        starts.append(x)
    return starts


def run_fit(objective: Objective, *, starts: list[np.ndarray], blocks: list[int] | None = None,
            maxfev: int = 4000) -> dict:
    from scipy.optimize import minimize
    bounds = bounds_for(objective.fc_fixed)
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


def _fit_worker(run_dir: str, floor_index: int, generation: int, x0: list, blocks, maxfev: int, fc_fixed,
                discrepancy: float, pool_workers: int = BLOCK_WORKERS_PER_START, norm: float = NORM_FIXED) -> dict:
    """One optimizer start in its own process with its own block pool, or
    one replica-block fit (``blocks = [k]``) that loads that block alone."""
    contract = _contract(Path(run_dir))
    tables = load_fit_tables()
    if blocks is None:
        with BlockPool(run_dir, floor_index, generation, pool_workers) as pool:
            objective = PooledObjective(tables, anchors(), pool, fc_fixed=fc_fixed, discrepancy=discrepancy, norm=norm)
            return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)
    templates = Templates(contract, floor_index, generation, tables, replicas=blocks)
    objective = Objective(templates, anchors(), fc_fixed=fc_fixed, discrepancy=discrepancy, norm=norm)
    return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)


def _best_of(results: list) -> dict:
    records = [r["starts"][0] for r in results]
    best = min(results, key=lambda r: r["fun"])
    return {"x": np.asarray(best["x"], dtype=float), "fun": float(best["fun"]), "converged": bool(best["converged"]),
            "nfev": int(best["nfev"]), "message": best["message"], "starts": records}


def run_fit_parallel(contract: RC.RunContract, floor_index: int, generation: int, starts: list, *, fc_fixed=None,
                     discrepancy: float = DISCREPANCY, norm: float = NORM_FIXED, blocks=None, maxfev: int = 4000,
                     workers: int | None = None, pool_workers: int = BLOCK_WORKERS_PER_START) -> dict:
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x0], blocks, int(maxfev), fc_fixed,
             float(discrepancy), int(pool_workers), float(norm)) for x0 in starts]
    return _best_of(RC.run_replicas(_fit_worker, jobs, max_workers=workers or len(jobs)))


def run_scan_parallel(contract: RC.RunContract, floor_index: int, generation: int, x_shapes: list, values: tuple, *,
                      discrepancy: float = DISCREPANCY, norm: float = NORM_FIXED, maxfev: int = 4000) -> list:
    """The profile scan: one fit at each fixed f_central, all in parallel,
    each from the free optimum's shapes."""
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x_shapes], None, int(maxfev), float(fc),
             float(discrepancy), BLOCK_WORKERS_PER_START, float(norm)) for fc in values]
    return [_best_of([r]) for r in RC.run_replicas(_fit_worker, jobs, max_workers=len(jobs))]


def hessian_shapes(objective: Objective, phys: dict) -> dict:
    """Central-difference Hessian of the objective in the physical shape
    parameters at the objective's fixed f_central."""
    names = SHAPE_NAMES
    p0 = np.asarray([phys[n] for n in names], dtype=float)
    steps = np.asarray([HESSIAN_STEPS[n] for n in names], dtype=float)

    def value(p):
        trial = dict(phys)
        trial.update({n: float(v) for n, v in zip(names, p)})
        if not (0.0 < trial["w_excited"] < 1.0) or not (B_SD_BOUNDS[0] < trial["b_sd"] < B_SD_BOUNDS[1]):
            return float("inf")
        return objective.evaluate(x_from_physical(trial, objective.fc_fixed))

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
    return {"names": list(names), "steps": [float(s) for s in steps], "at_f_central": objective.fc_fixed,
            "matrix": h.tolist() if finite else None, "eigenvalues": [float(e) for e in eig] if finite else None,
            "positive_definite": pd, "quoted": pd, "errors": errors}


def _mixture_readouts(templates: Templates, objective: Objective, tables: dict, phys: dict) -> dict:
    """The readouts at a fitted point: paper-window sum ratios, total yields
    and their class breakdown, the pi- p_T profile, the SD n_ch against M_X,
    the pi0 and neutron readouts, and predictions."""
    weights, _ = objective.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"],
                                         remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
    frac_map = class_fractions(phys)
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


def central_test(chi2_free: float, chi2_null: float, n_obs: int, f_hat: float, scan: list) -> dict:
    """The pre-registered test: Delta chi2 = chi2_null - chi2_min against
    3.84 S^2 with S^2 = chi2_min / ndf (chi2_min the lowest of the free,
    null and scan fits), and the scaled interval of f_central from the
    profile (linear interpolation between the points)."""
    ndf = int(n_obs - N_FREE_PARAMETERS)
    points = sorted({(0.0, float(chi2_null)), (float(f_hat), float(chi2_free))} | {(float(p["f_central"]), float(p["objective"])) for p in scan})
    # chi2_min is the lowest of the fits; a scan point below the free fit
    # means the free fit missed the global minimum (flagged)
    f_min, chi2_min = min(points, key=lambda p: p[1])
    free_is_minimum = bool(chi2_min >= chi2_free - 1e-6)
    if not free_is_minimum:
        f_hat = f_min
    s2 = float(chi2_min / ndf)
    threshold = TEST_DELTA_CHI2 * s2
    delta = float(chi2_null - chi2_min)
    profile = [(fc, v - chi2_min) for fc, v in points]

    def crossing(direction):
        seq = [p for p in profile if (p[0] >= f_hat if direction > 0 else p[0] <= f_hat)]
        seq = sorted(seq, key=lambda p: p[0], reverse=direction < 0)
        for (f0, d0), (f1, d1) in zip(seq, seq[1:]):
            if d0 < threshold <= d1:
                return float(f0 + (threshold - d0) * (f1 - f0) / (d1 - d0))
        return None

    upper, lower = crossing(+1), crossing(-1)
    return {"chi2_free": float(chi2_free), "chi2_min": float(chi2_min), "free_fit_is_minimum": free_is_minimum,
            "chi2_null": float(chi2_null), "delta_chi2": delta, "ndf": ndf,
            "scale_factor_squared": s2, "threshold": threshold, "raw_threshold": TEST_DELTA_CHI2,
            "f_central_hat": float(f_hat), "compatible_with_zero": bool(delta <= threshold),
            "compatible_with_zero_raw": bool(delta <= TEST_DELTA_CHI2),
            "scaled_interval": {"lower": 0.0 if lower is None and delta <= threshold else lower, "upper": upper,
                                "rule": "where the profile Delta chi2 crosses 3.84 S^2 (None: not crossed on the scan)"},
            "profile": [{"f_central": fc, "delta_chi2": d} for fc, d in profile]}


def evaluate_generation(contract: RC.RunContract, floor_index: int, generation: int, tables: dict, anchors_: dict, *,
                        smoke: bool = False) -> dict:
    f, g = int(floor_index), int(generation)
    free_starts = dispersed_starts(None)[: (2 if smoke else N_STARTS)]
    master_pool = None
    if smoke:
        templates = Templates(contract, f, g, tables, smoke=True)
        free_fit = run_fit(Objective(templates, anchors_), starts=free_starts, maxfev=600)
    else:
        free_fit = run_fit_parallel(contract, f, g, free_starts, maxfev=4000)
    phys_free = physical_from_x(free_fit["x"], None, anchors_)
    null_starts = [x_from_physical(phys_free, 0.0)] + dispersed_starts(0.0)[1: (1 if smoke else N_STARTS_NULL)]
    scan_values = (0.5,) if smoke else FC_SCAN
    if smoke:
        null_fit = run_fit(Objective(templates, anchors_, fc_fixed=0.0), starts=null_starts, maxfev=600)
        scan_fits = [run_fit(Objective(templates, anchors_, fc_fixed=fc), starts=[x_from_physical(phys_free, fc)], maxfev=300)
                     for fc in scan_values]
    else:
        null_fit = run_fit_parallel(contract, f, g, null_starts, fc_fixed=0.0, maxfev=4000)
        scan_fits = run_scan_parallel(contract, f, g, list(x_from_physical(phys_free, 0.0)), scan_values)
        master_pool = BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS)

    def objective_at(fc_fixed):
        if smoke:
            return Objective(templates, anchors_, fc_fixed=fc_fixed)
        return PooledObjective(tables, anchors_, master_pool, fc_fixed=fc_fixed)

    value, det = objective_at(None).evaluate(free_fit["x"], details=True)
    phys = det["physical"]
    v_null, d_null = objective_at(0.0).evaluate(null_fit["x"], details=True)
    scan = []
    for fc, fit in zip(scan_values, scan_fits):
        v_s, d_s = objective_at(fc).evaluate(fit["x"], details=True)
        scan.append({"f_central": float(fc), "objective": v_s, "converged": fit["converged"], "ess_min": d_s["ess_min"],
                     "fitted": {n: d_s["physical"][n] for n in SHAPE_NAMES}, "chi2_by_segment": d_s["chi2_by_segment"]})
    test = central_test(value, v_null, det["m"].size, phys["f_central"], scan)
    if not test["free_fit_is_minimum"]:
        residuals_pre = [{"kind": "free_fit_not_global_minimum", "chi2_free": value, "chi2_min": test["chi2_min"]}]
    else:
        residuals_pre = []
    hess = (hessian_shapes(objective_at(phys["f_central"]), phys) if not smoke
            else {"quoted": False, "positive_definite": False, "errors": None, "skipped": "smoke"})
    blocks = []
    n_blocks = (templates.n_blocks if smoke else master_pool.n_blocks)
    if not smoke and n_blocks > 1:
        block_jobs = [(str(contract.run_dir), f, g, [float(v) for v in free_fit["x"]], [k], 2000, None, DISCREPANCY, 0, NORM_FIXED)
                      for k in range(n_blocks)]
        for k, res in enumerate(RC.run_replicas(_fit_worker, block_jobs, max_workers=n_blocks)):
            blocks.append({"block": k, **{n: v for n, v in physical_from_x(np.asarray(res["x"]), None, anchors_).items()
                                          if n in SHAPE_NAMES + ("f_central",)}, "fun": res["fun"], "converged": res["converged"]})
    stability = {}
    if blocks and hess.get("errors"):
        for name in SHAPE_NAMES:
            vals = np.asarray([b[name] for b in blocks])
            spread = float(np.std(vals, ddof=1))
            expected = hess["errors"][name] * math.sqrt(len(blocks))
            ratio = spread / expected if expected > 0 else float("inf")
            stability[name] = {"block_values": [float(v) for v in vals], "spread": spread,
                               "expected_spread": expected, "ratio": ratio, "stable": bool(ratio <= STABILITY_RATIO_BOUND)}
    f_central_blocks = [float(b["f_central"]) for b in blocks]
    def _interior(p):
        return bool(FRACTION_INTERIOR[0] <= p["w_excited"] <= FRACTION_INTERIOR[1])

    def _on_bound(p):
        return bool(abs(p["epsilon"] - EPS_BOUNDS[0]) < 1e-3 or abs(p["epsilon"] - EPS_BOUNDS[1]) < 1e-3
                    or abs(p["epsilon_nd"] - EPS_ND_BOUNDS[0]) < 1e-3 or abs(p["epsilon_nd"] - EPS_ND_BOUNDS[1]) < 1e-3)

    # R68: the interior and bound checks apply to the free and the null fit
    # (R67 checked the free fit only)
    interior = _interior(phys) and _interior(d_null["physical"])
    on_exponent_bound = _on_bound(phys) or _on_bound(d_null["physical"])
    residuals = list(residuals_pre)
    for label, fit, d_ in (("free", free_fit, det), ("null", null_fit, d_null)):
        if not fit["converged"]:
            residuals.append({"kind": "optimizer_non_convergence", "fit": label, "message": fit["message"]})
        if d_["ess_min"] < ESS_FLOOR:
            residuals.append({"kind": "weight_degeneracy", "fit": label, "ess_min": d_["ess_min"], "floor": ESS_FLOOR})
    for label, p_ in (("free", phys), ("null", d_null["physical"])):
        if not _interior(p_):
            residuals.append({"kind": "fraction_on_boundary", "fit": label, "w_excited": p_["w_excited"]})
        if _on_bound(p_):
            residuals.append({"kind": "exponent_on_boundary", "fit": label, "epsilon": p_["epsilon"], "epsilon_nd": p_["epsilon_nd"]})
    if not hess.get("positive_definite"):
        residuals.append({"kind": "block_curvature_failure", "eigenvalues": hess.get("eigenvalues")})
    for name, s in stability.items():
        if not s["stable"]:
            residuals.append({"kind": "replica_block_minima_disagree", "parameter": name, "ratio": s["ratio"]})
    ess_low = min(det["ess_min"], d_null["ess_min"])
    regenerate = (not smoke) and ess_low < ESS_INTERIOR and g < MAX_GENERATIONS - 1
    regeneration_point = {n: 0.5 * (phys[n] + d_null["physical"][n]) for n in ("epsilon", "b_sd", "epsilon_nd")}
    constrained = bool(free_fit["converged"] and null_fit["converged"] and ess_low >= ESS_FLOOR and interior and not on_exponent_bound
                       and hess.get("positive_definite") and stability and all(s["stable"] for s in stability.values()))
    if master_pool is not None:
        master_pool.close()
        # the readouts need every block in process once
        templates = Templates(contract, f, g, tables)
    readouts = _mixture_readouts(templates, Objective(templates, anchors_), tables, phys)
    null_readouts = _mixture_readouts(templates, Objective(templates, anchors_, fc_fixed=0.0), tables, d_null["physical"])
    mesonic = mesonic_count_readout(FLOOR_GRID[f], SMOKE_MESONIC_DECAYS if smoke else MESONIC_DECAYS, GEN_FLAVOR)
    objective = Objective(templates, anchors_)
    return {
        "floor_index": f, "continuation_floor": float(FLOOR_GRID[f]), "generation": g,
        "generation_point": dict(zip(("epsilon", "b_sd", "epsilon_nd"), templates.point)),
        "fixed": {"continuation_floor": float(FLOOR_GRID[f]), "flavor": dict(zip(FLAVOR_ORDER, GEN_FLAVOR)), "flavor_provenance": R66_RUN_ID,
                  "fractions": {k: anchors_[k]["value"] for k in FRACTION_NAMES}, "w_intact": 0.0},
        "statistics": {"classes": list(CLASS_ORDER), "replicas": templates.n_blocks,
                       "events_per_slot": int(templates.by_class[CLASS_ORDER[0]][0]["n"]), "tier": "T2" if not smoke else "smoke",
                       "common_random_numbers": "nd0 replica k draws the seeds and event ids of nd replica k"},
        "observable_set": {"segments": {k: [int(a), int(b)] for k, (a, b) in objective.segments.items()}, "total": objective.n_obs,
                           "fit_window": FIT_WINDOW, "paper_window": PAPER_WINDOW,
                           "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS)},
                           "discrepancy": DISCREPANCY, "anchors": anchors_},
        "parametrization": {"free": list(param_names(None)), "null": "f_central = 0; free: " + ", ".join(SHAPE_NAMES)},
        "fitted": {**phys, "class_fractions": class_fractions(phys),
                   "remnant_weights": {"intact_proton": phys["w_intact"], "low_mass_baryonic_resonance": phys["w_low_mass"],
                                       "high_mass_baryonic_continuum": phys["w_excited"]}},
        "objective_at_optimum": {"value": value, "minus2lnl_data": det["minus2lnl_data"], "chi2_by_segment": det["chi2_by_segment"],
                                 "normalization": det["normalization"], "ess": det["ess"], "ess_min": det["ess_min"],
                                 "in_trust_region": det["in_trust_region"]},
        "null_fit": {"fitted": {n: d_null["physical"][n] for n in SHAPE_NAMES + ("w_low_mass", "f_central")}, "objective": v_null,
                     "chi2_by_segment": d_null["chi2_by_segment"], "ess_min": d_null["ess_min"], "converged": null_fit["converged"],
                     "nfev": null_fit["nfev"], "start_records": null_fit["starts"],
                     "total_yield_ratios": null_readouts["readout_total_yields"]["ratio_model_over_data"],
                     "class_yields": null_readouts["readout_class_yields"], "pt_profile": null_readouts["readout_pt_profile"],
                     "paper_window_sum_ratios": {k: v["sum_ratio"] for k, v in null_readouts["readout_paper_window_sum_ratios"].items()}},
        "profile_scan": scan,
        "central_component_test": test,
        "optimizer": {"method": "Powell (bounded)", "starts": len(free_starts), "converged": free_fit["converged"], "nfev": free_fit["nfev"],
                      "message": free_fit["message"], "start_records": free_fit["starts"]},
        "hessian": hess, "block_fits": blocks, "replica_block_stability": stability,
        "f_central_blocks": {"values": f_central_blocks,
                             "spread": float(np.std(f_central_blocks, ddof=1)) if len(f_central_blocks) > 1 else None},
        "verdict": {"constrained": constrained, "regeneration_requested": regenerate, "regeneration_point": regeneration_point,
                    "interior_fractions": interior, "exponent_on_boundary": on_exponent_bound,
                    "ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR},
        "named_residual_register": residuals,
        **readouts,
        "readout_mesonic_count": mesonic,
        "per_bin": {"d": objective.d.tolist(), "m": det["m"].tolist(), "se": det["se"].tolist(),
                    "v_data": objective.v_data.tolist(), "pull": det["residual_pull"].tolist(),
                    "m_null": d_null["m"].tolist(), "pull_null": d_null["residual_pull"].tolist()},
        "determinism": {"note": "no wall clock in this record; the slot records carry their own timing"},
    }


# ----------------------------------------------------------------------
# The profile over the floor grid, the acceptance, and the final report
# ----------------------------------------------------------------------
def test_summary(branch: dict, variants: dict | None) -> dict:
    """The test outcome in the primary fit and in each variant."""
    rows = {"primary": branch["central_component_test"]}
    for label, var in (variants or {}).items():
        rows[label] = var["central_component_test"]
    return {"rule": "f_central compatible with zero when Delta chi2 = chi2(f_central = 0) - chi2_min <= 3.84 chi2_min / ndf",
            "outcomes": {k: {"f_central_hat": v["f_central_hat"], "delta_chi2": v["delta_chi2"], "threshold": v["threshold"],
                             "compatible_with_zero": v["compatible_with_zero"], "compatible_with_zero_raw": v["compatible_with_zero_raw"]}
                         for k, v in rows.items()},
            "compatible_in_primary": bool(rows["primary"]["compatible_with_zero"]),
            "compatible_in_every_variant": bool(all(v["compatible_with_zero"] for v in rows.values()))}


def compose_report(branch: dict, variants: dict | None) -> dict:
    """The final report: the branch's fits and test, the variants, the
    test summary, and the verdict."""
    report = {k: v for k, v in branch.items() if k not in ("digest", "schema", "run", "card", "driver", "driver_sha256")}
    report["branch_digest"] = branch.get("digest")
    report["variants"] = variants
    report["test_summary"] = test_summary(branch, variants)
    report["verdict"] = {**branch["verdict"], "regeneration_requested": False}
    return report


def render_report_md(report: dict) -> str:
    f = report["fitted"]; o = report["objective_at_optimum"]; v = report["verdict"]; nf = report["null_fit"]
    test = report["central_component_test"]
    errors = (report["hessian"] or {}).get("errors") or {}
    lines = [f"# R68: test of the central component on the uncapped cascade ({RUN_ID})", "",
             f"Card {CARD}. Generation {report['generation']}; floor {report['continuation_floor']} GeV; flavor fixed at {R66_RUN_ID}.",
             "Fractions at the external anchors; the non-diffractive share split by f_central between the class with the central cluster and the class without it.", "",
             "## The test", "",
             f"f_central = {test['f_central_hat']:.4f}. chi2_min {test['chi2_min']:.1f} (free fit {test['chi2_free']:.1f}); with f_central = 0, {test['chi2_null']:.1f}.",
             f"Delta chi2 = {test['delta_chi2']:.1f}; S^2 = chi2/ndf = {test['scale_factor_squared']:.2f} (ndf {test['ndf']}); threshold 3.84 S^2 = {test['threshold']:.1f}.",
             f"Compatible with zero: {test['compatible_with_zero']} (raw threshold 3.84: {test['compatible_with_zero_raw']}).",
             f"Scaled interval of f_central: {json.dumps(test['scaled_interval'])}.", "",
             "| f_central | Delta chi2 |", "|---|---|"]
    for row in test["profile"]:
        lines.append(f"| {row['f_central']:.3f} | {row['delta_chi2']:.1f} |")
    lines += ["", "| segment | chi2 free | chi2 null |", "|---|---|---|"]
    for name in FIT_SPECIES:
        lines.append(f"| {name} | {o['chi2_by_segment'][name]:.1f} | {nf['chi2_by_segment'][name]:.1f} |")
    lines += ["", "| parameter | free fit | error | null fit |", "|---|---|---|---|"]
    for name in SHAPE_NAMES:
        err = errors.get(name)
        lines.append(f"| {name} | {f[name]:.5g} | {'' if err is None else f'{err:.3g}'} | {nf['fitted'][name]:.5g} |")
    lines += ["", f"Smallest class ESS {o['ess_min']:.3f} (null {nf['ess_min']:.3f}). Hessian positive definite: {report['hessian'].get('positive_definite')}.",
              f"Verdict (fit quality): {'CONSTRAINED' if v['constrained'] else 'NOT_CONSTRAINED'}.", "",
              "## Total yields per inelastic event (model / data)", "", "| species | free fit | null fit |", "|---|---|---|"]
    ty = report["readout_total_yields"]["ratio_model_over_data"]
    for name, ratio in ty.items():
        lines.append(f"| {name} | {ratio:.3f} | {nf['total_yield_ratios'][name]:.3f} |")
    pt = report["readout_pt_profile"]
    lines += ["", f"pi- p_T profile ratio {pt['profile_ratio']:.2f} (null {nf['pt_profile']['profile_ratio']:.2f}).", "",
              "## Variants", "", "| variant | f_central | Delta chi2 | threshold | compatible |", "|---|---|---|---|---|"]
    for label, var in (report.get("variants") or {}).items():
        tv = var["central_component_test"]
        lines.append(f"| {label} | {tv['f_central_hat']:.4f} | {tv['delta_chi2']:.1f} | {tv['threshold']:.1f} | {tv['compatible_with_zero']} |")
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
    sealed = {}
    for run_id in (R67_RUN_ID, R66_RUN_ID):
        d = MONO_ROOT / "dev" / "00runs" / run_id
        if not (d / "seal_manifest.json").is_file():
            contract.refuse(f"the sealed archive {d} is required")
        sealed[run_id] = RC.read_json(d / RC.REPORT_NAME)
    r67, r66 = sealed[R67_RUN_ID], sealed[R66_RUN_ID]
    if ((float(r67["fitted"]["epsilon"]), float(r67["fitted"]["b_sd"]), float(r67["fitted"]["epsilon_nd"])) != GEN0_SHAPE
            or float(r67["fitted"]["w_excited"]) != GEN0_W_EXCITED or float(r67["fitted"]["w_low_mass"]) != GEN0_W_LOW):
        contract.refuse("the generation-0 constants do not match the sealed R67 free optimum")
    if tuple(float(r66["fitted"][n]) for n in FLAVOR_ORDER) != GEN_FLAVOR:
        contract.refuse("the fixed flavor point does not match the sealed R66 optimum")
    _load_physics(FLOOR_GRID[0])  # refuses a committed config that already sets a floor
    a = anchors()
    payload = {
        "classes": list(CLASS_ORDER), "replicas": REPLICAS, "events_per_slot": EVENTS_PER_SLOT, "max_generations": MAX_GENERATIONS,
        "hypothesis": "the owner's pure two-component picture: no central component (owner, 2026-10-02: 'I would like ND to be shown as compatible with zero, validating no central component')",
        "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST,
                  "rule": f"raw seed = RAW_FIRST + {SEEDED_SLOTS} * generation + seed slot; slot = class_index * {REPLICAS} + replica; "
                          "an nd0 slot draws the seed slot of the nd replica with the same index",
                  "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * seed slot + i", "disjointness": disjoint},
        "generation_0_point": {"epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND, "w_excited": GEN0_W_EXCITED,
                               "provenance": {"run": R67_RUN_ID, "report_digest": r67.get("digest"),
                                              "r67_outcome": r67["central_component_test"]["compatible_with_zero"]}},
        "continuation_ceiling": {"value_gev": CONTINUATION_CEILING, "set_in_memory": "dataclasses.replace(family, continuation_ceiling=C) for every family",
                                 "reason": "the R67 diagnosis: the 5 GeV support cap made heavy excitations shed their excess into one hadron",
                                 "kernel": "cascade_decay._continuation_ceiling; the absent key is the support edge, bit for bit"},
        "fixed": {"continuation_floor": FLOOR_GRID[0], "flavor": dict(zip(FLAVOR_ORDER, GEN_FLAVOR)),
                  "flavor_provenance": {"run": R66_RUN_ID, "report_digest": r66.get("digest"), "verdict": r66["verdict"],
                                        "note": "fitted on the capped cascade; held fixed here"},
                  "fractions": {k: a[k]["value"] for k in FRACTION_NAMES}, "w_intact": 0.0,
                  "nd_generation_remnant_weights": ND_GENERATION_REMNANT_WEIGHTS, "diffractive_split": FROZEN_REMNANT_WEIGHTS},
        "free_parameters": list(param_names(None)),
        "two_component_variant": "the kernel's generate_nd(central=False): the same remnant selections, transfers and remnant decays, no central draw, the two baryonic systems sharing the full sqrt(s)",
        "test": {"statistic": "Delta chi2 = chi2(f_central = 0) - chi2_min, each minimized over the shapes",
                 "rule": "compatible with zero when Delta chi2 <= 3.84 S^2, S^2 = chi2_min / (1255 - 5); the raw Delta chi2 is reported beside it",
                 "profile_scan": list(FC_SCAN), "decision": "owner, 2026-10-02",
                 "outcome_rule": "the outcome is reported as measured; a nonzero f_central is a result, not a reason to change the model in this run"},
        "variants": {"discrepancy": DISCREPANCY_VARIANT, "normalization": list(NORM_VARIANTS),
                     "role": "the same test under each variant; never drive the regeneration"},
        "discrepancy": {"primary": DISCREPANCY, "variant": DISCREPANCY_VARIANT},
        "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS)},
        "bounds": {"epsilon": EPS_BOUNDS, "epsilon_nd": EPS_ND_BOUNDS, "b_sd": B_SD_BOUNDS, "logits": LOGIT_BOUNDS, "f_central": FC_BOUNDS},
        "trust_region": {"ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR, "penalty": TRUST_PENALTY,
                         "regeneration": "at the midpoint of the free and null shape optima when either has a class ESS below ess_interior"},
        "data": {"fit_window": FIT_WINDOW, "species": list(FIT_SPECIES), "anchors": a},
        "verdict_rule": {"constrained": "both fits converged and in the trust region, w_excited interior, no exponent on its bound, shape Hessian positive definite at the fitted f_central, shape block spreads within 1.5 x pooled error x sqrt(32)",
                         "test": "reported separately; does not enter the verdict"},
        "optimizer": {"method": "Powell (bounded)", "starts_free": N_STARTS, "starts_null": N_STARTS_NULL, "hessian_steps": HESSIAN_STEPS},
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


def _generation_point(contract: RC.RunContract, floor_index: int, generation: int) -> tuple:
    """The registered generation shape point (epsilon, B_SD, epsilon_nd)."""
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
        point = _generation_point(contract, f, g)
        for slot in todo_slots:
            if contract.path(slot_names(f, g, slot)[0]).is_file():
                continue
            jobs.append(((f, g, slot), (slot_class(slot), slot_seed(f, g, slot), EVENTS_PER_SLOT, slot_partition(slot),
                                         point, FLOOR_GRID[f])))
    if not jobs:
        print(f"[{RUN_ID}] nothing to generate", flush=True)
        return
    t0 = time.perf_counter()
    results = RC.run_replicas(_generate_job, [job[1] for job in jobs], max_workers=workers)
    for ((f, g, slot), _), result in zip(jobs, results):
        write_slot(contract, f, g, slot, result, pre_sha)
    print(f"[{RUN_ID}] generated {len(jobs)} slots over branches {sorted({j[0][0] for j in jobs})} in {time.perf_counter() - t0:.0f} s", flush=True)


def _generate_job(cls, seed, n_events, partition_base, shape_point, floor):
    return generate_slot(cls, seed, n_events, partition_base, shape_point, floor)


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
                {"floor_index": f, "generation": nxt, **report["verdict"]["regeneration_point"],
                 "requested_by_generation": g, "ess_min": report["objective_at_optimum"]["ess_min"],
                 "ess_min_null": report["null_fit"]["ess_min"]}]
            RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), points)
            RC.write_json(contract.path(REGENERATION_FMT.format(f=f)), {"schema": f"{SCHEMA}.regeneration/1", **contract.identity(),
                                                                         "floor_index": f, "from_generation": g, "to_generation": nxt,
                                                                         "ess_min": report["objective_at_optimum"]["ess_min"], "written_utc": RC.now_utc()})
            print(f"[{RUN_ID}] generation {g}: ess_min {report['objective_at_optimum']['ess_min']:.3f} (null {report['null_fit']['ess_min']:.3f}) "
                  f"< {ESS_INTERIOR}; regeneration {nxt} requested at {json.dumps(report['verdict']['regeneration_point'])}", flush=True)
            continue
        branch = {"schema": f"{SCHEMA}.branch_report/1", **contract.identity(), **report,
                  "generation_history": sorted(p.name for p in contract.run_dir.glob(f"fit_report_f{f}_g*.json"))}
        branch["digest"] = RC.canonical_digest(branch)
        RC.write_json(contract.path(BRANCH_REPORT_FMT.format(f=f)), branch)
        if contract.path(REGENERATION_FMT.format(f=f)).is_file():
            contract.path(REGENERATION_FMT.format(f=f)).unlink()
        test = report["central_component_test"]
        print(f"[{RUN_ID}] final at generation {g}: f_central {test['f_central_hat']:.4f}, Delta chi2 {test['delta_chi2']:.1f} "
              f"against {test['threshold']:.1f}", flush=True)


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
    """The test under the discrepancy and normalization variants: the free
    and the null fit refitted on the branch's final generation."""
    f, g = int(chosen["floor_index"]), int(chosen["generation"])
    out = {}
    for label, dz, norm in VARIANTS:
        free_starts = [x_from_physical(chosen["fitted"], None)] + dispersed_starts(None)[1:N_STARTS_VARIANT]
        free_fit = run_fit_parallel(contract, f, g, free_starts, discrepancy=dz, norm=norm, maxfev=4000,
                                    pool_workers=BLOCK_WORKERS_PER_VARIANT_START)
        null_starts = [x_from_physical(chosen["null_fit"]["fitted"], 0.0)] + dispersed_starts(0.0)[1:N_STARTS_VARIANT]
        null_fit = run_fit_parallel(contract, f, g, null_starts, fc_fixed=0.0, discrepancy=dz, norm=norm, maxfev=4000,
                                    pool_workers=BLOCK_WORKERS_PER_VARIANT_START)
        with BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS) as pool:
            v_free, d_free = PooledObjective(tables, anchors(), pool, discrepancy=dz, norm=norm).evaluate(free_fit["x"], details=True)
            v_null, d_null = PooledObjective(tables, anchors(), pool, fc_fixed=0.0, discrepancy=dz, norm=norm).evaluate(null_fit["x"], details=True)
        test = central_test(v_free, v_null, d_free["m"].size, d_free["physical"]["f_central"], [])
        out[label] = {"discrepancy": dz, "normalization": norm,
                      "fitted": {n: d_free["physical"][n] for n in SHAPE_NAMES + ("f_central",)},
                      "null_fitted": {n: d_null["physical"][n] for n in SHAPE_NAMES},
                      "objective": v_free, "objective_null": v_null, "converged": bool(free_fit["converged"] and null_fit["converged"]),
                      "ess_min": min(d_free["ess_min"], d_null["ess_min"]), "central_component_test": test}
    return out


def step_profile(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    chosen = _load_branches(contract)[0]
    variants = _variants(contract, chosen, load_fit_tables())
    report = compose_report(chosen, variants)
    body = contract.write_report(report, render_report_md(report))
    s = body["test_summary"]
    print(f"[{RUN_ID}] report written: verdict {'CONSTRAINED' if body['verdict']['constrained'] else 'NOT_CONSTRAINED'}; "
          f"f_central compatible with zero: primary {s['compatible_in_primary']}, every variant {s['compatible_in_every_variant']}; "
          f"digest {body['digest'][:16]}", flush=True)


def step_gate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    test = report["central_component_test"]
    record = {
        "schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
        "obligation": "the test of the central component against the pure two-component picture, on the uncapped cascade (R68)",
        "verdict": "CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED",
        "test": {k: test[k] for k in ("f_central_hat", "chi2_free", "chi2_min", "free_fit_is_minimum", "chi2_null", "delta_chi2", "ndf", "scale_factor_squared",
                                      "threshold", "compatible_with_zero", "compatible_with_zero_raw", "scaled_interval")},
        "test_summary": report["test_summary"],
        "fitted": {n: report["fitted"][n] for n in SHAPE_NAMES + ("f_central",)},
        "null_fitted": report["null_fit"]["fitted"], "errors": (report["hessian"] or {}).get("errors"),
        "ess_min": report["objective_at_optimum"]["ess_min"], "ess_min_null": report["null_fit"]["ess_min"],
        "generation": report["generation"], "named_residual_register": report["named_residual_register"],
        "total_yield_ratios": report["readout_total_yields"]["ratio_model_over_data"],
        "total_yield_ratios_null": report["null_fit"]["total_yield_ratios"],
        "report_digest": report["digest"], "written_utc": RC.now_utc(),
    }
    RC.write_json(contract.path(GATE_RECORD_NAME), record)
    print(f"[{RUN_ID}] gate record: {record['verdict']}; f_central {test['f_central_hat']:.4f}, Delta chi2 {test['delta_chi2']:.1f} "
          f"against {test['threshold']:.1f}: compatible with zero {test['compatible_with_zero']}", flush=True)


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
    probe = generate_slot(slot_class(0), slot_seed(f, g, 0), EVENTS_PER_SLOT, slot_partition(0), point, FLOOR_GRID[f])
    sealed_probe = RC.read_json(contract.path(slot_names(f, g, 0)[0]))
    probe_match = bool(arrays_digest(probe["arrays"]) == sealed_probe["arrays_sha256"]
                       and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"])
    payload = {"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}",
                            "cd dev/mixing2006 && python -m pytest tests/test_staged_r68_central_component_uncapped.py -q"],
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
    seed_index = {cls: (CLASS_ORDER.index("nd") if cls == "nd0" else i) for i, cls in enumerate(CLASS_ORDER)}
    jobs = [(cls, SMOKE_SEED + seed_index[cls], SMOKE_EVENTS_PER_CLASS, SMOKE_PARTITION + 10_000 * seed_index[cls], GEN0_SHAPE, FLOOR_GRID[f])
            for cls in CLASS_ORDER]
    results = RC.run_replicas(_generate_job, jobs)
    for cls, result in zip(CLASS_ORDER, results):
        np.savez(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, f=f, g=0)), **result["arrays"])
        RC.write_json(contract.path("smoke_" + SLOT_JSON_FMT.format(cls=cls, k=0, f=f, g=0)), {
            "schema": f"{SCHEMA}.smoke_slot/1", "pre_registration_sha256": pre_sha, "class_label": cls, "generation": 0, "replica": 0,
            "floor_index": f, "continuation_floor": result["continuation_floor"], "flavor_generation": result["flavor_generation"],
            "informative_vertices": result["informative_vertices"], "nd_central": result["nd_central"],
            "seed": result["seed"], "n_events": result["n_events"], "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
            "arrays_sha256": arrays_digest(result["arrays"]), "event_rolling_sha256": result["event_rolling_sha256"],
            "conservation": result["conservation"], "tracks": result["tracks"]})
    # common random numbers: nd and nd0 share the remnant selections and transfers
    nd, nd0 = (results[CLASS_ORDER.index(c)]["arrays"] for c in ("nd", "nd0"))
    shared = {key: bool(np.array_equal(nd[key], nd0[key])) for key in ("rem_kind", "shape_m1", "shape_m2", "shape_t1", "shape_t2")}
    tables = load_fit_tables()
    report = evaluate_generation(contract, f, 0, tables, anchors(), smoke=True)
    smoke_templates = Templates(contract, f, 0, tables, smoke=True)
    # the check point sits at the generation point (a smoke fit can end on a bound)
    check = {**report["fitted"], "epsilon": GEN0_EPSILON + 0.01, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND,
             "w_excited": GEN0_W_EXCITED, "w_low_mass": GEN0_W_LOW, "f_central": 0.4}
    x_check = x_from_physical(check, None)
    with BlockPool(contract.run_dir, f, 0, 1, smoke=True) as pool:
        pooled_value = PooledObjective(tables, anchors(), pool).evaluate(x_check)
    serial_value = Objective(smoke_templates, anchors()).evaluate(x_check)
    report["pool_check"] = {"in_process": serial_value, "pooled": pooled_value,
                            "relative_difference": abs(pooled_value - serial_value) / max(abs(serial_value), 1e-300)}
    report["common_random_numbers"] = shared
    report["schema"] = f"{SCHEMA}.smoke_fit_report/1"
    report["test_summary"] = test_summary(report, None)
    report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_fit_report.json"), report)
    print(f"[{RUN_ID}] smoke: f_central {report['fitted']['f_central']:.4f}, Delta chi2 {report['central_component_test']['delta_chi2']:.1f}, "
          f"ess_min {report['objective_at_optimum']['ess_min']:.3f}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"R68 driver ({CARD})")
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
