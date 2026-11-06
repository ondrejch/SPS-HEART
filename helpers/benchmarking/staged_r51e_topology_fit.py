"""R51e driver: the topology refit on the Block B stage 1 substrate, the
continuum pools extended with the two-body channels of heavier established
states (Gate 5 follow-up; cards TASK-20260929-r5-gate5-followup-r51e-topology-fit
and TASK-20260929-r5-gate5-followup-cascade-multiplicity).

Differences from R51d (everything else is the R51d contract):
- Substrate: config/v2_singlet/continuum_resonances.json with the Block B
  stage 1 channels (30 baryonic, 23 neutral mesonic classes, unit
  couplings times phase space and spin), recorded as frozen substrate.
- Generation 0 at the sealed R51d optimum (epsilon, B_SD, epsilon_nd),
  checked against the sealed R51d report at pre-registration; the first
  start carries the R51d remnant weights.
- The objective cache holds the mixture weights and the unweighted-template
  standard errors in two bounded caches (the R51c/R51d cache leaked memory
  in free-fraction fits; FAB-13).

R51d design (inherited):
Differences from R51c (everything else is the R51c contract):
- One more shape parameter, epsilon_nd, the exponent of the non-diffractive
  class's mass laws (central production and excited remnants);
  epsilon stays the exponent of the diffractive classes.  Bounds
  epsilon_nd in [-1.5, 0.5] (the flat proton x_F spectrum points at -1).
- The primary fit fixes the class fractions to the external anchors
  (R51c showed that two anchor terms cannot hold the fractions against a
  factor-two yield deficit over 284 bins); the free-fraction fit is run
  as a variant and reported next to it.
- The exponent reweighting carries the excited remnants' own mass law
  (the kind and window columns of the non-diffractive shape arrays;
  topology_mass_laws.log_density_ratio, 2026-09-29).
- Generation 0 sits at the sealed R51c optimum (epsilon, B_SD) with
  epsilon_nd = epsilon, so the first generation reproduces the R51c
  substrate up to the recorded windows.

Original R51c design:

Differences from R51b (everything else is the R51b contract):

- Free parameters (6): f_SD_total, f_DD, epsilon, B_SD, and the two free
  remnant-inelasticity weights of the non-diffractive class (intact,
  low-mass resonance, excited continuum on the simplex).
- The non-diffractive templates are generated at equal remnant weights
  (1/3, 1/3, 1/3) with the two drawn kinds recorded per event; the remnant
  weights are fit by the exact kind-pair reweighting
  ``(w_k1 w_k2) / (w0_k1 w0_k2)`` (a discrete draw whose mass law does not
  depend on the weights, so the support is unchanged).  The diffractive
  templates keep the frozen split (the non-diffracted line of a
  single-diffraction event is intact or a low-mass resonance by class
  definition).
- Verdict CONSTRAINED also requires the three remnant weights interior.

Original R51b design:

Design (pre-registered in the card; every constant below is the contract):

- Substrate: the corrected kernels (FAB-03, FAB-04, FAB-07, FAB-08,
  FAB-09), the committed catalog and continuum families, the identity
  flavor point, the Round 5 remnant weights, the blend ramp start fixed at
  B.  The frozen-substrate digests are recorded at pre-registration and
  every later step refuses on drift.
- Classes: the four inelastic topology classes; elastic excluded from the
  inelastic sample (plan line 824); nd_multi absent (Gate 5).
- Templates: per class, 8 replicas x 100,000 events at the generation
  point (epsilon_g, B_SD_g); generation 0 at the nominal point.  The
  fractions enter the mixture linearly; the two shape parameters enter
  through the exact per-event mass-law density ratio
  (``topology_mass_laws.log_density_ratio``) with self-normalized
  importance weights per class.
- Trust region: the objective is finite only where every class keeps an
  effective sample size fraction of at least ESS_FLOOR; a fitted point
  whose smallest class ESS lies below ESS_INTERIOR requests a
  regeneration at that point (at most MAX_GENERATIONS - 1 regenerations).
- Free parameters (4): f_SD_total (shared equally by the two
  single-diffraction classes), f_DD, epsilon, B_SD; f_ND by the sum rule.
- Data: the NA49 p+p proton table over x_F in [0, 0.95] and p_T in
  [0.05, 1.5] (the leading-baryon region opened; 284 bins) and the
  antiproton table in the paper window (122 bins); the net-proton
  difference on the 122 common (x_F, p_T) bins and the proton alone on
  the 162 remaining bins.  One multiplicative normalization nuisance with
  a 2% Gaussian prior (the published NA49 normalization uncertainty),
  profiled.  Two external Gaussian anchors on f_SD_total and f_DD from
  ``config/v2_singlet/external_cross_sections.json``.  No multiplicity or
  gap data exist locally; those distributions are recorded as
  predictions.  Pions and kaons are held out; the paper-window sum
  ratios at the optimum are the pre-registered readout.
- Likelihood: plan section 5.4 with V_MC from the between-replica scatter.
- Optimizer: bounded Powell from 8 dispersed starts on the cached
  objective; 8 replica-block fits; a central-difference Hessian in the
  physical parameters; parameter errors sqrt(2 diag(H^-1)).
- Verdict CONSTRAINED requires: converged, smallest class ESS at the
  optimum >= ESS_FLOOR, every fraction interior to [0.005, 0.995], a
  positive-definite Hessian, and block spreads within 1.5 times their
  statistical expectation (pooled error times sqrt(8)).  Otherwise
  NOT_CONSTRAINED with named residuals.

Steps (one per invocation): --pre-registration, --generate [--generation
G] [--slot S ...], --evaluate, --gate, --clean-validate, --seal; --smoke
runs a 4 x 500 event end-to-end exercise into an explicit --run-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import run_contract as RC  # noqa: E402

NESTED_ROOT, MONO_ROOT = RC.locate_roots(HERE)
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

import run_observables as OB  # noqa: E402

from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.importance import ess_fraction, logits_from_simplex, simplex_from_logits  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.topology_mass_laws import log_density_ratio  # noqa: E402
from na49_mixing.topology_runner import (  # noqa: E402
    NOMINAL_PARAMS,
    TopologyNominalParams,
    extract_observables,
    run_topology,
)

# ----------------------------------------------------------------------
# Identity and contract constants
# ----------------------------------------------------------------------
RUN_ID = "r51e_topology_fit_2026-09-29"
CARD = "TASK-20260929-r5-gate5-followup-r51e-topology-fit"
SCHEMA = "na49_mixing.staged_r51e_topology_fit"
LEDGER_RUN = "r51e_topology_fit"

CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd")
REPLICAS = 8
EVENTS_PER_SLOT = 100_000
N_SLOTS = len(CLASS_ORDER) * REPLICAS  # 32
MAX_GENERATIONS = 4

MASTER_SEED = 20260929
RAW_FIRST = 20304120
RAW_LAST = RAW_FIRST + N_SLOTS * MAX_GENERATIONS - 1  # 20304247
PARTITION_BASE = 3_200_000_000
SMOKE_SEED = 20304101
SMOKE_PARTITION = 3_190_000_000
SMOKE_EVENTS_PER_CLASS = 500

SQRT_S = OB.SQRT_S
NOMINAL_EPSILON = float(NOMINAL_PARAMS.epsilon)
NOMINAL_B_SD = float(NOMINAL_PARAMS.b_sd)
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
EPS_BOUNDS = (-0.5, 0.5)
EPS_ND_BOUNDS = (-1.5, 0.5)
B_SD_BOUNDS = (1.0, 20.0)
LOGIT_BOUNDS = (-8.0, 8.0)
FRACTION_INTERIOR = (0.005, 0.995)
NORM_PRIOR = 0.02
SE_FLOOR = 1.0e-9
STABILITY_RATIO_BOUND = 1.5
N_STARTS = 8
EVENT_CLOSURE_TOL = 1.0e-9
HESSIAN_STEPS = {"f_sd_total": 0.01, "f_dd": 0.005, "epsilon": 0.01, "b_sd": 0.1, "w_intact": 0.02, "w_low_mass": 0.02, "epsilon_nd": 0.02}
SHAPE_NAMES = ("epsilon", "b_sd", "w_intact", "w_low_mass", "epsilon_nd")
FRACTION_NAMES = ("f_sd_total", "f_dd")
N_STARTS_VARIANT = 4
# objective cache bounds (amendment of 2026-09-29; the R51c-derived cache
# kept the standard-error entries in the weight cache without eviction,
# which leaked about 25 MB per fraction move in the free-fraction variant)
WEIGHT_CACHE_ENTRIES = 64
SE_CACHE_ENTRIES = 512
# generation 0: the sealed R51d optimum (report digest 9ef34821540967f6...)
GEN0_EPSILON = 0.14889020297261293
GEN0_B_SD = 2.660333074710886
GEN0_EPSILON_ND = -0.7570775320813953
GEN0_W_INTACT = 0.00016148079835122143
GEN0_W_LOW = 0.5184710496002284
R51D_RUN_ID = "r51d_topology_fit_2026-09-29"
REMNANT_KINDS = ("intact_proton", "low_mass_baryonic_resonance", "high_mass_baryonic_continuum")
ND_GENERATION_REMNANT_WEIGHTS = {k: 1.0 / 3.0 for k in REMNANT_KINDS}
FROZEN_REMNANT_WEIGHTS = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}

PROTON_WINDOW = {"xf_min": 0.0, "xf_max": 0.95, "pt_min": 0.05, "pt_max": 1.5}
PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
TRACK_SPECIES = OB.SPECIES_ORDER

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
)

SLOT_JSON_FMT = "slot_{cls}_{k}_g{g}.json"
SLOT_NPZ_FMT = "slot_{cls}_{k}_g{g}.npz"
FIT_REPORT_NAME = "fit_report.json"
GATE_RECORD_NAME = "gate5_followup_r51e.json"
REGENERATION_NAME = "regeneration_request.json"
GENERATION_POINTS_NAME = "generation_points.json"


def slot_class(slot: int) -> str:
    return CLASS_ORDER[int(slot) // REPLICAS]


def slot_replica(slot: int) -> int:
    return int(slot) % REPLICAS


def slot_seed(generation: int, slot: int) -> int:
    return RAW_FIRST + N_SLOTS * int(generation) + int(slot)


def slot_names(generation: int, slot: int) -> tuple[str, str]:
    cls, k = slot_class(slot), slot_replica(slot)
    return (SLOT_JSON_FMT.format(cls=cls, k=k, g=generation), SLOT_NPZ_FMT.format(cls=cls, k=k, g=generation))


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


# ----------------------------------------------------------------------
# Event generation (one slot per worker process)
# ----------------------------------------------------------------------
def _load_physics():
    catalog = HadronCatalog.from_json(NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json")
    fams = families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    return catalog, fams


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


def generate_slot(cls: str, seed: int, n_events: int, partition_base: int, eps: float, b_sd: float, eps_nd: float) -> dict:
    """Generate one slot: returns the compact arrays and the metadata."""
    catalog, fams = _load_physics()
    params = _params_at(eps, b_sd, cls, eps_nd)
    beta, gamma = OB.cm_boost(SQRT_S)
    rem_kind = np.full((n_events, 2), -1, dtype=np.int8)
    species_pdg = {name: OB.SPECIES_PDG[name] for name in TRACK_SPECIES}
    shape: dict[str, list] = {}
    lead = np.full(n_events, np.nan)
    gap = np.full(n_events, np.nan)
    n_ch = np.zeros(n_events, dtype=np.int16)
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
    out = {
        "cls": cls, "seed": int(seed), "n_events": int(n_events), "partition_base": int(partition_base),
        "epsilon": float(eps), "b_sd": float(b_sd), "epsilon_nd": float(eps_nd), "remnant_weights": dict(params.remnant_weights),
        "arrays": {
            **{f"shape_{k}": np.asarray(v, dtype=float) for k, v in shape.items()},
            "lead_xf": lead, "gap": gap, "n_ch": n_ch, "rem_kind": rem_kind,
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
    return out


def arrays_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8")); digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8")); digest.update(arr.tobytes())
    return digest.hexdigest()


def write_slot(contract: RC.RunContract, generation: int, slot: int, result: dict, pre_sha: str) -> dict:
    json_name, npz_name = slot_names(generation, slot)
    np.savez(contract.path(npz_name), **result["arrays"])
    record = {
        "schema": f"{SCHEMA}.slot/1", **contract.identity(), "pre_registration_sha256": pre_sha,
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
# The fit
# ----------------------------------------------------------------------
class Templates:
    """The loaded templates of one generation: per class and replica the
    shape arrays and the per-species binned track sums, ready for the
    weighted mixture at any (f, epsilon, B_SD)."""

    def __init__(self, contract: RC.RunContract, generation: int, tables: dict, *, smoke: bool = False):
        self.generation = int(generation)
        self.tables = tables
        self.point = None
        self.by_class: dict[str, list] = {cls: [] for cls in CLASS_ORDER}
        n_slots = N_SLOTS if not smoke else len(CLASS_ORDER)
        for slot in range(n_slots):
            cls = slot_class(slot) if not smoke else CLASS_ORDER[slot]
            k = slot_replica(slot) if not smoke else 0
            json_name = ("smoke_" if smoke else "") + SLOT_JSON_FMT.format(cls=cls, k=k, g=generation)
            npz_name = ("smoke_" if smoke else "") + SLOT_NPZ_FMT.format(cls=cls, k=k, g=generation)
            meta = RC.read_json(contract.path(json_name))
            with np.load(contract.path(npz_name)) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays_digest(arrays) != meta["arrays_sha256"]:
                contract.refuse(f"the arrays of {npz_name} do not match their recorded digest")
            point = (float(meta["epsilon"]), float(meta["b_sd"]), float(meta["epsilon_nd"]))
            if self.point is None:
                self.point = point
            elif self.point != point:
                contract.refuse(f"mixed generation points in generation {generation}")
            self.by_class[cls].append(self._prepare(cls, arrays, int(meta["n_events"])))
        for cls in CLASS_ORDER:
            if not self.by_class[cls]:
                contract.refuse(f"generation {generation} carries no {cls} template")
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
                # map the dense flat cell to the table's bin position (-1 if the cell is a hole)
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
                "rem_kind": arrays.get("rem_kind"), "totals": totals}

    # -- weights and moments ------------------------------------------
    def log_weights(self, cls: str, block: dict, eps: float, b_sd: float, eps_nd: float, remnant=None) -> np.ndarray:
        eps0, b0, eps0_nd = self.point
        eps_c, eps0_c = (float(eps_nd), eps0_nd) if cls == "nd" else (float(eps), eps0)
        if (eps_c, float(b_sd)) == (eps0_c, b0):
            log_w = np.zeros(block["n"])
        else:
            log_w = log_density_ratio(cls, block["shape"], eps_c, b_sd, eps0=eps0_c, b_sd0=b0, sqrt_s=SQRT_S)
        if cls == "nd" and remnant is not None:
            # the exact kind-pair reweighting from the equal generation weights
            ratio = np.log(np.asarray(remnant, dtype=float) / (1.0 / 3.0))
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
    + [epsilon, ln B_SD, logit w_intact, logit w_low, epsilon_nd]."""
    w_high = max(1.0 - float(phys["w_intact"]) - float(phys["w_low_mass"]), 1e-9)
    triple = np.asarray([phys["w_intact"], phys["w_low_mass"], w_high], dtype=float)
    r = logits_from_simplex(triple / triple.sum())
    r = r - r[2]
    core = [float(phys["epsilon"]), math.log(float(phys["b_sd"])), float(r[0]), float(r[1]), float(phys["epsilon_nd"])]
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
    eps, ln_b, r0, r1, eps_nd = (float(v) for v in core)
    w = simplex_from_logits([r0, r1, 0.0])
    return {"f_sd_total": float(f[0]), "f_dd": float(f[1]), "f_nd": float(f[2]),
            "epsilon": eps, "b_sd": float(math.exp(ln_b)),
            "w_intact": float(w[0]), "w_low_mass": float(w[1]), "w_excited": float(w[2]), "epsilon_nd": eps_nd}


def bounds_for(free: bool) -> list:
    core = [EPS_BOUNDS, (math.log(B_SD_BOUNDS[0]), math.log(B_SD_BOUNDS[1])), LOGIT_BOUNDS, LOGIT_BOUNDS, EPS_ND_BOUNDS]
    return ([LOGIT_BOUNDS, LOGIT_BOUNDS] if free else []) + core


class Objective:
    """The cached objective at one generation.  ``free`` selects the
    parametrization: fractions fixed to the anchors (the primary fit) or
    free on the simplex with the two anchor terms (the variant)."""

    def __init__(self, templates: Templates, anchors_: dict, free: bool = False):
        self.t = templates
        self.anchors = anchors_
        self.free = bool(free)
        p = templates.tables["proton"]; pb = templates.tables["antiproton"]
        key = lambda tab, i: (round(float(tab.xf_centers[tab.bin_index_xf[i]]), 6),  # noqa: E731
                              round(float(tab.pt_centers[tab.bin_index_pt[i]]), 6))
        p_index = {key(p, i): i for i in range(p.data.size)}
        self.common_p = np.asarray([p_index[key(pb, j)] for j in range(pb.data.size)], dtype=np.int64)
        self.common_pb = np.arange(pb.data.size)
        mask = np.ones(p.data.size, dtype=bool); mask[self.common_p] = False
        self.p_only = np.flatnonzero(mask)
        self.d = np.concatenate([p.data[self.common_p] - pb.data, p.data[self.p_only]])
        self.v_data = np.concatenate([p.sigma[self.common_p] ** 2 + pb.sigma ** 2, p.sigma[self.p_only] ** 2])
        self.area_p = OB.bin_areas(p); self.area_pb = OB.bin_areas(pb)
        self.n_obs = int(self.d.size)
        # two bounded caches (amendment of 2026-09-29): the mixture weights per
        # shape point (about 25 MB each) and the unweighted-template standard
        # errors per fraction point (small).  Both are pure functions of their
        # keys, so eviction changes run time only, never a value.
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

    def block_prediction(self, k: int, frac: dict, weights: dict) -> tuple[np.ndarray, np.ndarray]:
        """(m_p over 284 bins, m_pbar over 122 bins) of replica block k."""
        f = {"sd_projectile": frac["f_sd_total"] / 2.0, "sd_target": frac["f_sd_total"] / 2.0,
             "dd": frac["f_dd"], "nd": frac["f_nd"]}
        m_p = np.zeros(self.t.tables["proton"].data.size); m_pb = np.zeros(self.t.tables["antiproton"].data.size)
        for cls in CLASS_ORDER:
            block = self.t.by_class[cls][k]; wn = weights[cls][k]
            m_p += f[cls] * self.t.species_mean(cls, block, wn, "proton")
            m_pb += f[cls] * self.t.species_mean(cls, block, wn, "antiproton")
        return m_p / (math.pi * SQRT_S * self.area_p), m_pb / (math.pi * SQRT_S * self.area_pb)

    def unweighted_se(self, phys: dict, block_ids: list[int]) -> np.ndarray:
        """The between-block standard error of the mixture at the trial
        fractions with unit shape and remnant weights (cached per
        fraction triple and block set)."""
        key = ("se", round(phys["f_sd_total"], 12), round(phys["f_dd"], 12), tuple(block_ids))
        if key in self.se_cache:
            return self.se_cache[key]
        if not hasattr(self, "_unit_weights"):
            self._unit_weights = {cls: [np.full(block["n"], 1.0 / block["n"]) for block in self.t.by_class[cls]] for cls in CLASS_ORDER}
        vectors = np.asarray([self.observation_vector(*self.block_prediction(k, phys, self._unit_weights)) for k in block_ids])
        se = np.std(vectors, axis=0, ddof=1) / math.sqrt(vectors.shape[0]) if vectors.shape[0] > 1 else np.zeros(vectors.shape[1])
        se = np.maximum(se, SE_FLOOR)
        self.se_cache[key] = se
        while len(self.se_cache) > SE_CACHE_ENTRIES:
            self.se_cache.pop(next(iter(self.se_cache)))
        return se

    def observation_vector(self, m_p: np.ndarray, m_pb: np.ndarray) -> np.ndarray:
        return np.concatenate([m_p[self.common_p] - m_pb, m_p[self.p_only]])

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        phys = self.physical(x)
        if (not (EPS_BOUNDS[0] <= phys["epsilon"] <= EPS_BOUNDS[1]) or not (B_SD_BOUNDS[0] <= phys["b_sd"] <= B_SD_BOUNDS[1])
                or not (EPS_ND_BOUNDS[0] <= phys["epsilon_nd"] <= EPS_ND_BOUNDS[1])):
            return (float("inf"), None) if details else float("inf")
        weights, ess = self.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"], blocks,
                                          remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
        ess_min = min(ess.values())
        block_ids = list(range(self.t.n_blocks)) if blocks is None else list(blocks)
        vectors = []
        for k in block_ids:
            m_p, m_pb = self.block_prediction(k, phys, weights)
            vectors.append(self.observation_vector(m_p, m_pb))
        vectors = np.asarray(vectors)
        m = np.mean(vectors, axis=0)
        # The Monte Carlo variance is the between-block scatter of the
        # UNWEIGHTED templates at the trial fractions (R51b/R51c rule).
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
        return value, {
            "physical": phys, "ess": ess, "ess_min": ess_min, "eta_hat": eta,
            "minus2lnl_data": profiled, "anchor_terms": anchor_terms,
            "m": m, "se": se, "residual_pull": r, "blocks": block_ids,
            "in_trust_region": bool(ess_min >= ESS_FLOOR), "free_fractions": self.free,
        }


def dispersed_starts(anchors_: dict, free: bool) -> list[np.ndarray]:
    a_sd, a_dd = anchors_["f_sd_total"]["value"], anchors_["f_dd"]["value"]
    shapes = [(GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND), (GEN0_EPSILON, GEN0_B_SD, -1.0), (0.0, 5.0, -0.5), (-0.3, 3.0, -0.8),
              (0.1, 8.0, -1.2), (-0.4, 2.0, -0.3), (0.2, 4.0, 0.0), (-0.2, 6.0, -1.4)]
    remnants = [(max(GEN0_W_INTACT, 1e-3), GEN0_W_LOW), (0.1, 0.4), (0.25, 0.25), (0.05, 0.6), (0.34, 0.33), (0.2, 0.2), (0.5, 0.5), (0.15, 0.35)]
    fractions = [(a_sd, a_dd), (0.30, 0.10), (0.10, 0.02), (0.40, 0.20), (a_sd, a_dd), (0.15, 0.05), (0.25, 0.15), (0.35, 0.03)]
    n = N_STARTS if not free else N_STARTS_VARIANT
    starts = []
    for (fs, fd), (e, b, en), (wi, wl) in list(zip(fractions, shapes, remnants))[:n]:
        phys = {"f_sd_total": fs, "f_dd": fd, "epsilon": e, "b_sd": b, "w_intact": wi, "w_low_mass": wl, "epsilon_nd": en}
        x = x_from_physical(phys, free)
        idx = [0, 1, 4, 5] if free else [2, 3]
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


def _fit_worker(run_dir: str, generation: int, x0: list, blocks, maxfev: int, free: bool) -> dict:
    """One optimizer start (or one replica-block fit) in its own process:
    the templates are reloaded from the run dir, so the result is the
    same as the sequential :func:`run_fit` on the same start."""
    contract = _contract(Path(run_dir))
    tables = OB.load_tables(NESTED_ROOT, windows={"proton": PROTON_WINDOW})
    templates = Templates(contract, generation, tables)
    objective = Objective(templates, anchors(), free=free)
    return run_fit(objective, starts=[np.asarray(x0, dtype=float)], blocks=blocks, maxfev=maxfev)


def run_fit_parallel(contract: RC.RunContract, generation: int, starts: list, *, free: bool, blocks=None, maxfev: int = 4000,
                     workers: int | None = None) -> dict:
    """The dispersed starts as parallel processes (each deterministic in its
    start); the best start is the fit, exactly as in :func:`run_fit`."""
    jobs = [(str(contract.run_dir), int(generation), [float(v) for v in x0], blocks, int(maxfev), bool(free)) for x0 in starts]
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
        if trial["w_intact"] <= 0 or trial["w_low_mass"] <= 0 or trial["w_intact"] + trial["w_low_mass"] >= 1:
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


def evaluate_generation(contract: RC.RunContract, generation: int, tables: dict, anchors_: dict, *,
                        smoke: bool = False) -> dict:
    templates = Templates(contract, generation, tables, smoke=smoke)
    objective = Objective(templates, anchors_, free=False)
    starts = dispersed_starts(anchors_, free=False)[: (2 if smoke else N_STARTS)]
    if smoke:
        pooled = run_fit(objective, starts=starts, maxfev=600)
    else:
        pooled = run_fit_parallel(contract, generation, starts, free=False, maxfev=4000)
    value, det = objective.evaluate(pooled["x"], details=True)
    phys = det["physical"]
    hess = hessian_physical(objective, phys) if not smoke else {"quoted": False, "positive_definite": False, "errors": None, "skipped": "smoke"}
    # replica-block fits from the pooled optimum
    blocks = []
    if not smoke and templates.n_blocks > 1:
        block_jobs = [(str(contract.run_dir), int(generation), [float(v) for v in pooled["x"]], [k], 2000, False) for k in range(templates.n_blocks)]
        block_results = RC.run_replicas(_fit_worker, block_jobs, max_workers=templates.n_blocks)
        for k, res in enumerate(block_results):
            blocks.append({"block": k, **physical_from_x(np.asarray(res["x"]), False, anchors_), "fun": res["fun"], "converged": res["converged"]})
    stability = {}
    names = param_names(False)
    if blocks and hess.get("errors"):
        for name in names:
            vals = np.asarray([b[name] for b in blocks])
            spread = float(np.std(vals, ddof=1))
            expected = hess["errors"][name] * math.sqrt(len(blocks))
            ratio = spread / expected if expected > 0 else float("inf")
            stability[name] = {"block_values": [float(v) for v in vals], "spread": spread,
                               "expected_spread": expected, "ratio": ratio, "stable": bool(ratio <= STABILITY_RATIO_BOUND)}
    interior = all(FRACTION_INTERIOR[0] <= phys[n] <= FRACTION_INTERIOR[1] for n in ("w_intact", "w_low_mass", "w_excited"))
    on_exponent_bound = (abs(phys["epsilon"] - EPS_BOUNDS[0]) < 1e-3 or abs(phys["epsilon"] - EPS_BOUNDS[1]) < 1e-3
                         or abs(phys["epsilon_nd"] - EPS_ND_BOUNDS[0]) < 1e-3 or abs(phys["epsilon_nd"] - EPS_ND_BOUNDS[1]) < 1e-3)
    residuals = []
    if not pooled["converged"]:
        residuals.append({"kind": "optimizer_non_convergence", "message": pooled["message"]})
    if det["ess_min"] < ESS_FLOOR:
        residuals.append({"kind": "weight_degeneracy", "ess_min": det["ess_min"], "floor": ESS_FLOOR})
    if not interior:
        residuals.append({"kind": "fraction_on_boundary", "fractions": {n: phys[n] for n in ("w_intact", "w_low_mass", "w_excited")}})
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
    # the free-fraction variant (reported, never the regeneration driver)
    variant_obj = Objective(templates, anchors_, free=True)
    v_starts = dispersed_starts(anchors_, free=True)[: (1 if smoke else N_STARTS_VARIANT)]
    if smoke:
        v_fit = run_fit(variant_obj, starts=v_starts, maxfev=400)
    else:
        v_fit = run_fit_parallel(contract, generation, v_starts, free=True, maxfev=4000)
    v_value, v_det = variant_obj.evaluate(v_fit["x"], details=True)
    variant = {"fitted": v_det["physical"], "objective": {"value": v_value, "minus2lnl_data": v_det["minus2lnl_data"],
                                                          "anchor_terms": v_det["anchor_terms"], "ess_min": v_det["ess_min"]},
               "converged": v_fit["converged"], "nfev": v_fit["nfev"], "starts": len(v_starts),
               "note": "fractions free on the simplex with the two anchor terms (the R51c parametrization plus epsilon_nd); "
                       "reported next to the primary fit, which fixes the fractions to the anchors"}
    # readouts and predictions at the primary optimum
    weights, _ = objective.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"], remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]))
    frac_map = {"sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0, "dd": phys["f_dd"], "nd": phys["f_nd"]}
    readout = {}
    for name in TRACK_SPECIES:
        table = tables[name]
        model_bins = np.zeros(table.data.size)
        for cls in CLASS_ORDER:
            for k, block in enumerate(templates.by_class[cls]):
                model_bins += frac_map[cls] * templates.species_mean(cls, block, weights[cls][k], name) / templates.n_blocks
        model_bins = model_bins / (math.pi * SQRT_S * OB.bin_areas(table))
        paper = np.asarray([table.xf_centers[table.bin_index_xf[i]] <= PAPER_WINDOW["xf_max"] + 1e-9 for i in range(table.data.size)])
        readout[name] = {"bins": int(np.sum(paper)), "model_sum": float(np.sum(model_bins[paper])),
                         "data_sum": float(np.sum(table.data[paper])),
                         "sum_ratio": float(np.sum(model_bins[paper]) / np.sum(table.data[paper])),
                         "chi2_data_only": float(np.sum(((table.data[paper] - model_bins[paper]) / table.sigma[paper]) ** 2))}
    reference = RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "na49_pp_total_yields.json")
    yields = {name: 0.0 for name in TRACK_SPECIES}; mean_nch = 0.0; p_proton = 0.0
    for cls in CLASS_ORDER:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            for name in TRACK_SPECIES:
                yields[name] += frac_map[cls] * float(np.sum(wn * block["totals"][name]))
            mean_nch += frac_map[cls] * float(np.sum(wn * block["n_ch"]))
            p_proton += frac_map[cls] * float(np.sum(wn * np.isfinite(block["lead_xf"])))
    total_yields = {"model": {**{name: yields[name] for name in TRACK_SPECIES}, "charged_multiplicity": mean_nch, "events_with_a_proton": p_proton},
                    "data": reference["yields_per_inelastic_event"],
                    "ratio_model_over_data": {name: yields[name] / float(reference["yields_per_inelastic_event"][name]["value"]) for name in TRACK_SPECIES if name in reference["yields_per_inelastic_event"]},
                    "reference_sha256": RC.sha256_file(NESTED_ROOT / "config" / "v2_singlet" / "na49_pp_total_yields.json")}
    predictions = {"lead_xf": np.zeros(LEAD_XF_EDGES.size - 1), "gap": np.zeros(GAP_EDGES.size - 1), "n_ch": np.zeros(NCH_MAX + 1)}
    for cls in CLASS_ORDER:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            predictions["lead_xf"] += frac_map[cls] * templates.histogram_mean(block, wn, "lead_xf", LEAD_XF_EDGES)
            predictions["gap"] += frac_map[cls] * templates.histogram_mean(block, wn, "gap", GAP_EDGES)
            predictions["n_ch"] += frac_map[cls] * templates.nch_mean(block, wn)
    report = {
        "generation": int(generation),
        "generation_point": {"epsilon": templates.point[0], "b_sd": templates.point[1], "epsilon_nd": templates.point[2]},
        "statistics": {"classes": list(CLASS_ORDER), "replicas": templates.n_blocks,
                       "events_per_slot": int(templates.by_class[CLASS_ORDER[0]][0]["n"]),
                       "tier": "T2" if not smoke else "smoke"},
        "observable_set": {"net_proton_common_bins": int(objective.common_p.size), "proton_only_bins": int(objective.p_only.size),
                           "total": objective.n_obs, "proton_window": PROTON_WINDOW, "antiproton_window": PAPER_WINDOW,
                           "normalization_prior": NORM_PRIOR, "anchors": anchors_},
        "parametrization": {"primary": "fractions fixed to the anchors; free: " + ", ".join(SHAPE_NAMES),
                            "variant": "fractions free on the simplex with the anchor terms; free: " + ", ".join(param_names(True))},
        "fitted": {**phys, "sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0,
                   "remnant_weights": {"intact_proton": phys["w_intact"], "low_mass_baryonic_resonance": phys["w_low_mass"], "high_mass_baryonic_continuum": phys["w_excited"]},
                   "fractions_fixed_to_anchors": True},
        "objective_at_optimum": {"value": value, "minus2lnl_data": det["minus2lnl_data"], "anchor_terms": det["anchor_terms"],
                                 "eta_hat": det["eta_hat"], "ess": det["ess"], "ess_min": det["ess_min"], "in_trust_region": det["in_trust_region"]},
        "optimizer": {"method": "Powell (bounded)", "starts": len(starts), "converged": pooled["converged"], "nfev": pooled["nfev"],
                      "message": pooled["message"], "start_records": pooled["starts"]},
        "hessian": hess, "block_fits": blocks, "replica_block_stability": stability,
        "variant_free_fractions": variant,
        "verdict": {"constrained": constrained, "regeneration_requested": regenerate,
                    "interior_fractions": interior, "exponent_on_boundary": on_exponent_bound,
                    "ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR},
        "named_residual_register": residuals,
        "readout_paper_window_sum_ratios": readout,
        "readout_total_yields": total_yields,
        "predictions": {"lead_xf": {"edges": LEAD_XF_EDGES.tolist(), "density": predictions["lead_xf"].tolist()},
                        "gap": {"edges": GAP_EDGES.tolist(), "density": predictions["gap"].tolist()},
                        "n_ch": {"values": list(range(NCH_MAX + 1)), "density": predictions["n_ch"].tolist()}},
        "per_bin": {"d": objective.d.tolist(), "m": det["m"].tolist(), "se": det["se"].tolist(),
                    "v_data": objective.v_data.tolist(), "pull": det["residual_pull"].tolist()},
        "determinism": {"note": "no wall clock in this record; the slot records carry their own timing"},
    }
    return report


def render_report_md(report: dict) -> str:
    f = report["fitted"]; o = report["objective_at_optimum"]; v = report["verdict"]
    lines = [f"# R51e: topology refit on the Block B stage 1 substrate ({RUN_ID})", "",
             f"Card {CARD}. Generation {report['generation']} at (epsilon, B_SD, epsilon_nd) = ({report['generation_point']['epsilon']}, {report['generation_point']['b_sd']}, {report['generation_point']['epsilon_nd']}).",
             "Primary fit: fractions fixed to the external anchors.", "",
             "| parameter | value | error |", "|---|---|---|"]
    errors = (report["hessian"] or {}).get("errors") or {}
    for name in ("f_sd_total", "f_dd", "f_nd", "epsilon", "epsilon_nd", "b_sd", "w_intact", "w_low_mass", "w_excited"):
        err = errors.get(name)
        lines.append(f"| {name} | {f[name]:.6g} | {'' if err is None else f'{err:.3g}'} |")
    lines += ["", f"Objective at optimum {o['value']:.3f} (data {o['minus2lnl_data']:.3f}, anchors {o['anchor_terms']:.3f}); eta_hat {o['eta_hat']:.4f}; smallest class ESS {o['ess_min']:.3f}.",
              f"Converged: {report['optimizer']['converged']}. Hessian positive definite: {report['hessian'].get('positive_definite')}.",
              f"Verdict: {'CONSTRAINED' if v['constrained'] else 'NOT_CONSTRAINED'}; regeneration requested: {v['regeneration_requested']}.", "",
              "## Readout (paper-window sum ratios model/data, held out)", "", "| species | bins | ratio | chi2 (data only) |", "|---|---|---|---|"]
    for name, r in report["readout_paper_window_sum_ratios"].items():
        lines.append(f"| {name} | {r['bins']} | {r['sum_ratio']:.3f} | {r['chi2_data_only']:.1f} |")
    ty = report["readout_total_yields"]
    lines += ["", "## Total yields per inelastic event (model / data)", "", "| species | model | data | ratio |", "|---|---|---|---|"]
    for name, ratio in ty["ratio_model_over_data"].items():
        lines.append(f"| {name} | {ty['model'][name]:.3f} | {ty['data'][name]['value']} | {ratio:.2f} |")
    lines.append(f"\nMean charged multiplicity {ty['model']['charged_multiplicity']:.2f} (data {ty['data'].get('charged_multiplicity', {}).get('value', 'n/a')}); events with a final proton {ty['model']['events_with_a_proton']:.2f}.")
    v = report.get("variant_free_fractions")
    if v:
        lines += ["", "## Variant: fractions free with the anchor terms", "", "| parameter | value |", "|---|---|"]
        for name in ("f_sd_total", "f_dd", "f_nd", "epsilon", "epsilon_nd", "b_sd", "w_intact", "w_low_mass", "w_excited"):
            lines.append(f"| {name} | {v['fitted'][name]:.6g} |")
        lines.append(f"\nVariant objective {v['objective']['value']:.3f} (data {v['objective']['minus2lnl_data']:.3f}, anchors {v['objective']['anchor_terms']:.3f}); ESS {v['objective']['ess_min']:.3f}; converged {v['converged']}.")
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
    r51d_dir = MONO_ROOT / "dev" / "00runs" / R51D_RUN_ID
    if not (r51d_dir / "seal_manifest.json").is_file():
        contract.refuse(f"the sealed R51d archive {r51d_dir} is required for the generation-0 point")
    r51d_report = RC.read_json(r51d_dir / RC.REPORT_NAME)
    fitted = r51d_report["fitted"]
    if (abs(float(fitted["epsilon"]) - GEN0_EPSILON) > 1e-12 or abs(float(fitted["b_sd"]) - GEN0_B_SD) > 1e-12
            or abs(float(fitted["epsilon_nd"]) - GEN0_EPSILON_ND) > 1e-12):
        contract.refuse("the generation-0 constants do not match the sealed R51d optimum")
    payload = {
        "classes": list(CLASS_ORDER), "replicas": REPLICAS, "events_per_slot": EVENTS_PER_SLOT, "max_generations": MAX_GENERATIONS,
        "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST, "rule": "raw seed = RAW_FIRST + 32 * generation + slot; slot = class_index * 8 + replica",
                  "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * slot + i", "disjointness": disjoint},
        "generation_0_point": {"epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND,
                               "provenance": {"run": R51D_RUN_ID, "report_digest": r51d_report.get("digest"), "rule": "the sealed R51d optimum"}},
        "free_parameters": list(SHAPE_NAMES),
        "fractions": {"rule": "fixed to the external anchors in the primary fit", "values": {k: anchors()[k]["value"] for k in FRACTION_NAMES},
                      "reason": "R51c: two anchor terms cannot hold the fractions against a factor-two yield deficit over 284 bins; the fixed-anchor fit measures the shape and remnant parameters conditional on the external diffractive content"},
        "variant": {"free_parameters": list(param_names(True)), "starts": N_STARTS_VARIANT, "role": "reported next to the primary fit; never drives the regeneration"},
        "sum_rule": "f_nd = 1 - f_sd_total - f_dd; f_sd_projectile = f_sd_target = f_sd_total / 2; w_excited = 1 - w_intact - w_low_mass (non-diffractive remnants)",
        "remnant_inelasticity": {"nd_generation_weights": ND_GENERATION_REMNANT_WEIGHTS, "diffractive_split": FROZEN_REMNANT_WEIGHTS,
                                 "reweighting": "exact kind-pair ratio (w_k1 w_k2)/(w0_k1 w0_k2) per non-diffractive event"},
        "mass_laws": {"diffractive": "(M^2)^(-1-epsilon) exp[B_SD t M^2] for the single- and double-diffraction excitations",
                      "non_diffractive": "(M^2)^(-1-epsilon_nd) for the central production (with the transfer factor) and for the excited remnants (zero transfer)",
                      "reweighting": "exact per-class density ratio in (epsilon or epsilon_nd, B_SD) including the excited remnants' own law on their recorded windows (topology_mass_laws.log_density_ratio, 2026-09-29)"},
        "bounds": {"epsilon": EPS_BOUNDS, "epsilon_nd": EPS_ND_BOUNDS, "b_sd": B_SD_BOUNDS, "logits": LOGIT_BOUNDS},
        "trust_region": {"ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR, "penalty": TRUST_PENALTY},
        "mc_variance": "between-block standard error of the unweighted templates at the trial fractions (fixed in the shape trial)",
        "data": {"proton_window": PROTON_WINDOW, "antiproton_window": PAPER_WINDOW, "observable": "net proton on common bins, proton alone elsewhere",
                 "normalization_prior": NORM_PRIOR, "se_floor": SE_FLOOR, "anchors": anchors()},
        "held_out": ["pi_minus", "pi_plus", "k_plus", "k_minus", "lead_xf", "gap", "n_ch"],
        "readout_rule": "paper-window sum ratio model/data of pi+- at the optimum; >= 0.7 lets the flavor round proceed on this point",
        "verdict_rule": {"constrained": "converged and ess_min >= ess_floor and remnant weights interior and neither exponent on its bound and Hessian positive definite and block spreads within 1.5 x pooled error x sqrt(8)",
                         "fraction_interior": FRACTION_INTERIOR, "stability_ratio_bound": STABILITY_RATIO_BOUND},
        "optimizer": {"method": "Powell (bounded)", "starts": N_STARTS, "hessian_steps": HESSIAN_STEPS},
        "frozen_substrate": frozen,
    }
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()})
    RC.write_json(contract.path(GENERATION_POINTS_NAME), {"points": [{"generation": 0, "epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND}]})


def _verify_frozen(contract: RC.RunContract) -> dict:
    pre = contract.load_pre_registration()
    frozen = {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}
    status = RC.verify_frozen_substrate(MONO_ROOT, frozen, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    return status


def _generation_point(contract: RC.RunContract, generation: int) -> tuple[float, float, float]:
    points = RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"]
    for p in points:
        if int(p["generation"]) == int(generation):
            return float(p["epsilon"]), float(p["b_sd"]), float(p["epsilon_nd"])
    contract.refuse(f"generation {generation} has no registered point; run --evaluate on the previous generation first")


def step_generate(run_dir: Path | None, generation: int, slots: list[int] | None, workers: int | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    pre_sha = RC.sha256_file(contract.path(RC.PREREGISTRATION_NAME))
    eps, b_sd, eps_nd = _generation_point(contract, generation)
    todo = list(range(N_SLOTS)) if not slots else [int(s) for s in slots]
    jobs = []
    for slot in todo:
        json_name, _ = slot_names(generation, slot)
        if contract.path(json_name).is_file():
            print(f"[{RUN_ID}] slot {slot} of generation {generation} exists; kept", flush=True)
            continue
        jobs.append((slot_class(slot), slot_seed(generation, slot), EVENTS_PER_SLOT, PARTITION_BASE + 100_000 * slot, eps, b_sd, eps_nd))
        jobs[-1] = (slot,) + jobs[-1]
    if not jobs:
        return
    t0 = time.perf_counter()
    results = RC.run_replicas(_generate_job, [job[1:] for job in jobs], max_workers=workers)
    for (slot, *_), result in zip(jobs, results):
        write_slot(contract, generation, slot, result, pre_sha)
    print(f"[{RUN_ID}] generation {generation}: {len(jobs)} slots in {time.perf_counter() - t0:.0f} s", flush=True)


def _generate_job(cls, seed, n_events, partition_base, eps, b_sd, eps_nd):
    return generate_slot(cls, seed, n_events, partition_base, eps, b_sd, eps_nd)


def _latest_generation(contract: RC.RunContract) -> int:
    points = RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"]
    complete = [int(p["generation"]) for p in points if all(contract.path(slot_names(int(p["generation"]), s)[0]).is_file() for s in range(N_SLOTS))]
    if not complete:
        contract.refuse("no complete generation on disk")
    return max(complete)


def step_evaluate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    generation = _latest_generation(contract)
    tables = OB.load_tables(NESTED_ROOT, windows={"proton": PROTON_WINDOW})
    report = evaluate_generation(contract, generation, tables, anchors())
    if report["verdict"]["regeneration_requested"]:
        nxt = generation + 1
        points = RC.read_json(contract.path(GENERATION_POINTS_NAME))
        points["points"] = [p for p in points["points"] if int(p["generation"]) != nxt] + [
            {"generation": nxt, "epsilon": report["fitted"]["epsilon"], "b_sd": report["fitted"]["b_sd"], "epsilon_nd": report["fitted"]["epsilon_nd"],
             "requested_by_generation": generation, "ess_min": report["objective_at_optimum"]["ess_min"]}]
        RC.write_json(contract.path(GENERATION_POINTS_NAME), points)
        RC.write_json(contract.path(f"fit_report_g{generation}.json"), report)
        RC.write_json(contract.path(REGENERATION_NAME), {"schema": f"{SCHEMA}.regeneration/1", **contract.identity(),
                                                         "from_generation": generation, "to_generation": nxt,
                                                         "point": {"epsilon": report["fitted"]["epsilon"], "b_sd": report["fitted"]["b_sd"], "epsilon_nd": report["fitted"]["epsilon_nd"]},
                                                         "ess_min": report["objective_at_optimum"]["ess_min"], "written_utc": RC.now_utc()})
        print(f"[{RUN_ID}] generation {generation}: ess_min {report['objective_at_optimum']['ess_min']:.3f} < {ESS_INTERIOR}; "
              f"regeneration {nxt} requested at (eps, B_SD, eps_nd) = ({report['fitted']['epsilon']:.4f}, {report['fitted']['b_sd']:.3f}, {report['fitted']['epsilon_nd']:.4f})", flush=True)
        return
    history = sorted(str(p.name) for p in contract.run_dir.glob("fit_report_g*.json"))
    report["generation_history"] = history
    body = contract.write_report(report, render_report_md(report))
    if contract.path(REGENERATION_NAME).is_file():
        contract.path(REGENERATION_NAME).unlink()
    print(f"[{RUN_ID}] fit report written (generation {generation}, verdict "
          f"{'CONSTRAINED' if body['verdict']['constrained'] else 'NOT_CONSTRAINED'}, digest {body['digest'][:16]})", flush=True)


def step_gate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    pions = report["readout_paper_window_sum_ratios"]
    pion_ratio = 0.5 * (pions["pi_plus"]["sum_ratio"] + pions["pi_minus"]["sum_ratio"])
    record = {
        "schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
        "obligation": "Gate 5 follow-up, Block B stage 1 (heavier direct channels) topology refit (R51e)",
        "verdict": "CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED",
        "fitted": report["fitted"], "errors": (report["hessian"] or {}).get("errors"),
        "ess_min": report["objective_at_optimum"]["ess_min"], "converged": report["optimizer"]["converged"],
        "named_residual_register": report["named_residual_register"],
        "readout": {"pion_sum_ratio_mean": pion_ratio, "rule": "0.7 or above lets the flavor round proceed on this point",
                    "flavor_round_proceeds": bool(pion_ratio >= 0.7), "per_species": pions},
        "report_digest": report["digest"], "written_utc": RC.now_utc(),
    }
    RC.write_json(contract.path(GATE_RECORD_NAME), record)
    print(f"[{RUN_ID}] gate record: {record['verdict']}, pion sum ratio {pion_ratio:.3f}", flush=True)


def step_clean_validate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME, GATE_RECORD_NAME)
    frozen = _verify_frozen(contract)
    sealed = RC.read_json(contract.path(RC.REPORT_NAME))
    generation = _latest_generation(contract)
    tables = OB.load_tables(NESTED_ROOT, windows={"proton": PROTON_WINDOW})
    rerun = evaluate_generation(contract, generation, tables, anchors())
    rerun["generation_history"] = sealed.get("generation_history", [])
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    # the probe slot (slot 0 of the final generation) regenerated in memory
    eps, b_sd, eps_nd = _generation_point(contract, generation)
    probe = generate_slot(slot_class(0), slot_seed(generation, 0), EVENTS_PER_SLOT, PARTITION_BASE, eps, b_sd, eps_nd)
    sealed_probe = RC.read_json(contract.path(slot_names(generation, 0)[0]))
    probe_match = bool(arrays_digest(probe["arrays"]) == sealed_probe["arrays_sha256"]
                       and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"])
    payload = {"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}",
                            "cd dev/mixing2006 && python -m pytest tests/test_staged_r51e_topology_fit.py -q"],
               "report_digest": sealed["digest"], "report_rerun_digest": rerun_digest, "report_hash_match": bool(rerun_digest == sealed["digest"]),
               "probe_slot": {"generation": generation, "slot": 0, "arrays_match": probe_match},
               "frozen_substrate": frozen, "seal": {"state": "pending (the seal lands over this record)"}}
    contract.write_clean_validation(payload)
    if rerun_digest != sealed["digest"]:
        contract.refuse(f"the clean re-evaluation digest {rerun_digest} differs from the report {sealed['digest']}")
    if not probe_match:
        contract.refuse("the probe slot does not reproduce its sealed arrays")
    print(f"[{RUN_ID}] clean validation: report match True, probe match True", flush=True)


def step_seal(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    generation = _latest_generation(contract)
    required = [GATE_RECORD_NAME, GENERATION_POINTS_NAME]
    for g in range(generation + 1):
        for s in range(N_SLOTS):
            required.extend(slot_names(g, s))
    contract.seal(required, "slot_")


def step_smoke(run_dir: Path) -> None:
    contract = _contract(run_dir)
    if contract.run_dir == MONO_ROOT / "dev" / "00runs" / RUN_ID:
        contract.refuse("--smoke needs an explicit --run-dir outside the production run dir")
    contract.run_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_digests()
    RC.write_json(contract.path("smoke_pre_registration.json"), {"schema": f"{SCHEMA}.smoke_pre_registration/1", **contract.identity(),
                                                                "seed": SMOKE_SEED, "partition_base": SMOKE_PARTITION, "events_per_class": SMOKE_EVENTS_PER_CLASS,
                                                                "frozen_substrate": frozen})
    pre_sha = RC.sha256_file(contract.path("smoke_pre_registration.json"))
    jobs = [(cls, SMOKE_SEED + i, SMOKE_EVENTS_PER_CLASS, SMOKE_PARTITION + 10_000 * i, GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND) for i, cls in enumerate(CLASS_ORDER)]
    results = RC.run_replicas(_generate_job, jobs)
    for i, (cls, result) in enumerate(zip(CLASS_ORDER, results)):
        np.savez(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, g=0)), **result["arrays"])
        RC.write_json(contract.path("smoke_" + SLOT_JSON_FMT.format(cls=cls, k=0, g=0)), {
            "schema": f"{SCHEMA}.smoke_slot/1", "pre_registration_sha256": pre_sha, "class_label": cls, "generation": 0, "replica": 0,
            "seed": result["seed"], "n_events": result["n_events"], "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
            "arrays_sha256": arrays_digest(result["arrays"]), "event_rolling_sha256": result["event_rolling_sha256"],
            "conservation": result["conservation"], "tracks": result["tracks"]})
    tables = OB.load_tables(NESTED_ROOT, windows={"proton": PROTON_WINDOW})
    report = evaluate_generation(contract, 0, tables, anchors(), smoke=True)
    report["schema"] = f"{SCHEMA}.smoke_fit_report/1"
    report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_fit_report.json"), report)
    print(f"[{RUN_ID}] smoke: fitted {json.dumps({k: round(v, 4) for k, v in report['fitted'].items() if isinstance(v, float)})}, ess_min {report['objective_at_optimum']['ess_min']:.3f}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"R51e driver ({CARD})")
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pre-registration", "--generate", "--evaluate", "--gate", "--clean-validate", "--seal", "--smoke"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--slot", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.pre_registration:
        step_pre_registration(args.run_dir)
    elif args.generate:
        contract = _contract(args.run_dir)
        generation = args.generation if args.generation is not None else max(
            int(p["generation"]) for p in RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"])
        step_generate(args.run_dir, generation, args.slot, args.workers)
    elif args.evaluate:
        step_evaluate(args.run_dir)
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
