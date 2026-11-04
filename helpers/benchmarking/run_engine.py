"""The run engine: one driver for the topology fit runs, configured by a
declarative run spec (TASK-20261004-refactor-run-engine-kernel-config, P2).

Before the engine every run copied the previous ~2,000-line driver and edited
it (R67 to R74).  The engine is the sealed R74 driver's machinery with its
run constants moved into ``dev/mixing2006/config/runs/<run>.json``:

- identity: run id, card, schema, ledger run, the raw seed block, the
  partition base, the smoke seeds, the check-file name;
- the kernel: a ``na49_mixing.kernel_config.KernelConfig`` (profile or
  options);
- the statistics: replicas, events per slot, generations, the floor grid;
- the parameters: a registry of the shape parameters (epsilon, b_sd,
  epsilon_nd, w_excited, the free flavor weights, f_central) with
  transform (identity, log, logit), bounds, Hessian step, free or fixed
  value, the dispersed starts and the regeneration midpoint rule; the
  optimizer vector, the bounds, the starts, the box-bounded Hessian and the
  bound checks follow from the registry;
- generation 0: the sealed source run, the fit (free or null) checked
  against the branch digest, and named judgment values;
- the test (the f_central scan) and the fit settings;
- the pre-registration and report texts.

``configure(spec)`` sets every constant under the name the sealed drivers
used, so the generation, the objective, the fits, the fan-out, the readouts,
the gate, the clean validation and the seal are the R74 code unchanged.
The isovector exchange (R75, 2026-10-04): a spec with an ``isovector`` block
(``generation_fraction`` f0) and a ``f_isovector`` parameter generates every
class at the isovector fraction f0, records each event's exchange
configuration (``iv_config``, -1 for the vacuum), and reweights an event by
f/f0 (isovector) or (1 - f)/(1 - f0) (vacuum); the fit adds the vacuum-only
refit at f_isovector = 0 and its test.  ``fit.neutron_segment`` adds the NA49
neutron dN/dx_F to the objective.  ``isovector.own_slope`` with a
``b_isovector`` parameter (R76) gives the isovector events their own transfer
slope: a fourth coordinate of the generation point, reweighted on the
isovector events only.  Without them the engine runs as before.

The engine supports the R69 to R74 layouts (epsilon_nd, w_excited and the
flavor weights free; epsilon and b_sd free or fixed).

usage: python helpers/benchmarking/run_engine.py --spec SPEC (--pre-registration | --generate | ...)
"""

from __future__ import annotations

import argparse
import collections as _collections
import dataclasses as _dc
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
import fit_fanout as FO  # noqa: E402

from na49_mixing.cascade_decay import (  # noqa: E402
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import handoff_decay, has_valid_direct_channel, is_catalog_handoff  # noqa: E402
from na49_mixing.flavor_weights import CHANNEL_FACTOR_DAUGHTERS, NOMINAL_FLAVOR, FlavorParams  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.kernel_config import KernelConfig, load_kernel  # noqa: E402
from na49_mixing.importance import ess_fraction, logits_from_simplex, simplex_from_logits  # noqa: E402
from na49_mixing.isovector_exchange import ISOVECTOR_CONFIGURATIONS  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import families_from_json  # noqa: E402
import na49_mixing.cascade_decay as CASCADE_DECAY  # noqa: E402
from na49_mixing import split_reweight as SPLIT_RW  # noqa: E402
from na49_mixing.topology_mass_laws import (PROTON_MASS as MASS_LAW_PROTON, log_density_ratio, log_power_integral,  # noqa: E402
                                            log_sd_pomeron_mass_norm, sd_limits)
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
# Run constants: set by configure(spec) under the names of the sealed
# drivers (None until a spec is loaded)
SPEC = None
SPEC_PATH = None
SPEC_REL = None
RUN_ID = CARD = SCHEMA = LEDGER_RUN = GATE_RECORD_NAME = None
REPLICAS = EVENTS_PER_SLOT = N_SLOTS = SEEDED_SLOTS = MAX_GENERATIONS = None
FLOOR_GRID = N_FLOORS = None
MASTER_SEED = RAW_FIRST = RAW_LAST = PARTITION_BASE = SMOKE_SEED = SMOKE_PARTITION = None
KERNEL = CONTINUATION_CEILING = None
PARAMS = PARAM_INDEX = None
EPS_BOUNDS = EPS_ND_BOUNDS = B_SD_BOUNDS = LOGIT_BOUNDS = FC_BOUNDS = FLAVOR_LOG_BOUNDS = None
HESSIAN_STEPS = SHAPE_NAMES = FLAVOR_FREE = R_V_FIXED = FIXED_VALUES = None
FC_SCAN = TEST_DELTA_CHI2 = N_FREE_PARAMETERS = N_STARTS = N_STARTS_NULL = N_STARTS_VARIANT = FIT_MAXFEV = None
GEN_FLAVOR = GEN0_EPSILON = GEN0_B_SD = GEN0_EPSILON_ND = GEN0_W_EXCITED = GEN0_W_LOW = GEN0_F_CENTRAL = GEN0_SHAPE = None
ND_GENERATION_REMNANT_WEIGHTS = None
TEXTS = None
READOUT_LABEL = None
#: the isovector generation fraction f0 (None: the vacuum exchange only)
ISOVECTOR_GEN = None
#: the NA49 neutron dN/dx_F as an objective segment
NEUTRON_SEGMENT = False
SEGMENT_NAMES = None
#: the correlated error model of the NA49 spectra (2026-10-28; spec fit.correlated_errors): per species a normalization
#: scale with a Gaussian prior (the papers' 1.5%), for the kaons an identification scale on the bins above xf_min (the
#: papers' 4-12% K+ and 0-6% K- at x_F > 0.25), for the neutrons one scale (the table's 10% in quadrature) with the
#: per-point remainder floored; the scales are profiled in closed form at every evaluation. None: the committed
#: objective bit for bit (the per-bin errors and the discrepancy floor only)
CORRELATED_ERRORS = None
CORRELATED_ERROR_KEYS = frozenset({"normalization", "kaon_identification", "neutron_scale", "neutron_floor"})
ISO_NAME = "f_isovector"
#: the isovector exchange's own transfer slope (R76)
ISOVECTOR_SLOPE = False
BIV_NAME = "b_isovector"
#: the power central mass law with its own exponent (2026-10-07, R81)
CENTRAL_POWER = False
EPSC_NAME = "epsilon_c"
#: the intact proton's vertex factor of the single-diffraction classes as a
#: free registry parameter (2026-10-15, R97): a coordinate of the generation
#: point, reweighted on the single-diffraction events with an intact proton
INTACT_FREE = False
IVF_NAME = "intact_vertex_factor"
#: the leading-proton tail of the single-diffraction classes (2026-10-23, R112):
#: the share of the class beyond the coherence region, a free parameter with the
#: tail's law (topology "sd_tail": xi_lo, xi_hi, a, b) fixed
TAIL_FREE = False
TAIL_NAME = "sd_tail_share"
SD_TAIL_LAW = None
#: the one-pion exchange of the single-diffraction classes (2026-10-26): its share free beside the tail (OPE_NAME),
#: its law (topology "sd_ope": xi_lo, xi_hi, alpha', r2 or "proton"[, "regge"[, "isospin"[, "eikonal"]]]) fixed; the
#: two-sided hadronization of
#: the heavy systems (topology "sd_two_sided": m_cut, eps_line, b, scope) fixed
OPE_FREE = False
OPE_NAME = "sd_ope_share"
#: the leading splits as fitted parameters (2026-11-03, the fourteen-parameter fit): the leading-baryon split share,
#: the leading-meson split share, the baryon split's direction kappa and the meson split's slope b, named as their
#: KernelConfig fields. Free together; the generation kernel takes them from the shape point, the events carry the
#: split records (cascade_decay.SPLIT_RECORDER, shape_split_*), and the weights are split_reweight.split_log_ratio. They
#: ride on the share argument of the single-diffraction classes: (s_tail, s_pi, F_B, F_M, kappa_B, b_M).
SPLIT_FREE: tuple = ()
SD_OPE_LAW = None
SD_TWO_SIDED = None
#: the intact proton's two-slope transfer in the single-diffraction classes
#: (topology "intact_t_break": t_b, rho; 2026-10-24), fixed
T_BREAK = None
# r_BBbar on the pair steps that continue only (spec flavor_bbar_scope = "continuation", 2026-10-25)
BBAR_CONTINUATION_ONLY = False
#: the flavor ledger's scale of n_ssbar (2026-10-15): twice n_ssbar through
#: R97 (half-integral counting); ten times with the eta'(958), whose hidden
#: s-sbar content is 0.6; slots without the led_ssbar_scale array read as 2
LEDGER_SSBAR_SCALE = 2
#: the shape coordinates beyond (epsilon, B_SD, epsilon_nd), in order
EXTRA_SHAPE: tuple = ()

CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd", "nd0")
#: the generated and fitted classes: CLASS_ORDER, or without "nd" when the
#: run spec sets "central_component": false (configure)
ACTIVE_CLASSES = CLASS_ORDER
#: the two-component mode (configure): f_central = 0 by construction
CENTRAL_COMPONENT = True
#: the f_central of a fit called with fc_fixed None: None (free) or 0.0 in
#: the two-component mode
FC_DEFAULT = None
#: the generator options of the run spec's "topology" block
TOPOLOGY_OPTIONS: dict = {}
TOPOLOGY_KEYS = ("exact_transfer_range", "line_thresholds", "central_recoil", "low_mass_spin_isospin",
                 # 2026-10-18: the duality picks with the records' own line shapes (na49_mixing.nd_duality)
                 "duality_line_shapes",
                 # 2026-10-20: the redraw below the shape mass on the same line shapes
                 "duality_redraw_shapes")
#: the numeric generator options of the "topology" block (2026-10-08): the
#: exponent of the factorized vertex slopes (needs exact_transfer_range);
#: the duality and shape masses of the low-mass non-diffractive lines
#: (2026-10-12, na49_mixing.nd_duality); the intact proton's vertex factor
#: of the single-diffraction classes (2026-10-15, needs the vertex slopes); the
#: mass below which the diffractive lines decay as I = 1/2 records (2026-10-15);
#: the coherence edge M^2/s of the diffractive classes' excited masses (2026-10-23)
TOPOLOGY_NUMERIC_KEYS = ("transfer_slope_mass_exponent", "nd_duality_mass", "nd_duality_shape_mass", "intact_vertex_factor",
                         "diffractive_duality_mass", "diffractive_xi_max",
                         # 2026-11-01: alpha'_P of the Pomeron's shrinkage in the diffractive classes' coherence region
                         # (topology_mass_laws, "The Pomeron's shrinkage"; needs the exact window and the coherence edge)
                         "diffractive_pomeron_slope",
                         # 2026-11-01: t_lin, the shrinkage down to t = -t_lin only (the trajectory's linear range)
                         "diffractive_pomeron_linear_t")
ISOVECTOR_KEYS = frozenset({"generation_fraction", "own_slope", "texts"})
EXACT_RANGE = False
ND_CLASSES = ("nd", "nd0")
#: the runner class of each template class (nd0 is the runner's nd with
#: nd_central=False)
RUNNER_CLASS = {"nd0": "nd"}
SMOKE_EVENTS_PER_CLASS = 500
SMOKE_FLOOR_INDEX = 0
SMOKE_MESONIC_DECAYS = 200

#: the generators' recorded draw value: the reweighting mirrors the draw with it
SQRT_S = OB.SQRT_S
#: the exact invariant mass of the beam pair: the CM boost, x_F and the
#: estimator's pi sqrt(s) (2026-10-07; review 2026-10-06, Muse L-1)
SQRT_S_CM = OB.SQRT_S_CM
#: the bin-center correction of the model bins (spec fit.bin_center_correction)
BIN_CENTER = False
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
FRACTION_INTERIOR = (0.005, 0.995)
# the flavor weights, in FlavorParams order
BASE_FLAVOR_ORDER = ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
#: the channel factors of a diagnostic fit (2026-10-17; the spec's "flavor_channel_factors", in
#: na49_mixing.flavor_weights.CHANNEL_FACTOR_DAUGHTERS order): effective factors after the four plan weights, whose
#: daughter counts extend the ledger keys; empty in every model run
CHANNEL_FACTORS: tuple = ()
FLAVOR_ORDER = BASE_FLAVOR_ORDER + CHANNEL_FACTORS
# the shared normalization: fixed (owner decision 2026-10-02), with variants
NORM_FIXED = 0.0
NORM_VARIANTS = (0.02, -0.02)
DISCREPANCY = 0.05
DISCREPANCY_VARIANT = 0.10
SE_FLOOR = 1.0e-9
STABILITY_RATIO_BOUND = 1.5
# the block-parallel objective: processes per optimizer start (the primary
# fit's 8 starts and the variants' 4) and for the details, Hessian and
# variant evaluations; a 64-core gateway node per branch
BLOCK_WORKERS_PER_START = 6
BLOCK_WORKERS_PER_VARIANT_START = 12
MASTER_BLOCK_WORKERS = 32
#: Fan-out: a fit run as its own gateway job owns a full 32-task allocation,
#: so its block pool takes 32 workers (the block values do not depend on
#: the pool size).
FANOUT_POOL_WORKERS = 32
EVENT_CLOSURE_TOL = 1.0e-9
FRACTION_NAMES = ("f_sd_total", "f_dd")
#: the single-diffraction anchor's window (2026-10-23, R108): None applies the
#: anchor f_sd_total to the whole class (R51 to R108); a run spec's
#: "single_diffraction_window" x applies the measured share f_sd_window to the
#: class's masses with M^2/s < x, the class share then f_sd_window / F(epsilon)
SD_WINDOW = None
#: "single_diffraction_window_counts" (2026-11-01; the review of 2026-11-01, item R6): what the measured window share
#: counts. "coherence" (the committed reading) anchors the coherence component alone; "intact_protons" anchors every
#: intact proton below the window edge, as the inclusive p + p -> p + X measurement does: the coherence component and
#: the neutral pion exchange (p + p -> p + X+), whose in-window share of the exchange is NEUTRAL_OPE_WINDOW (the neutral
#: fraction times the neutral law's share below the edge, fixed by the exchange's law)
SD_WINDOW_INTACT = False
NEUTRAL_OPE_WINDOW = 0.0
WINDOW_NAME = "f_sd_window"
WEIGHT_CACHE_ENTRIES = 64
SE_CACHE_ENTRIES = 512
REMNANT_KINDS = ("intact_proton", "low_mass_baryonic_resonance", "high_mass_baryonic_continuum")
#: the nominal R50 split of the single-diffraction classes' non-excited line
#: (fixed, never fitted); a run spec may name its own as "diffractive_split"
#: (2026-10-09, from R87), which configure() sets as FROZEN_REMNANT_WEIGHTS
NOMINAL_DIFFRACTIVE_SPLIT = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}
FROZEN_REMNANT_WEIGHTS = dict(NOMINAL_DIFFRACTIVE_SPLIT)

SPEC_SCHEMA = "na49_mixing.run_spec/1"
TRANSFORMS = ("identity", "log", "logit")


#: One shape parameter of the registry (the optimizer order is the registry
#: order; f_central is last).  ``transform``: the optimizer coordinate
#: (identity, log, logit); ``bounds``: the physical bounds for identity and
#: log, the logit bounds for logit; ``hessian_box``: the physical box of the
#: box-bounded Hessian (default: the bounds, exp(log bounds) for log, the
#: logistic of the logit bounds); ``starts``: the dispersed starts in
#: physical units ("gen0" is the generation-0 value); ``midpoint``: the
#: regeneration rule; ``bound_check``: the parameter counts in the on-bound
#: residual.  (A namedtuple: the engine is also loaded by file path.)
ParamSpec = _collections.namedtuple(
    "ParamSpec", ["name", "transform", "bounds", "free", "value", "gen0", "hessian_step", "hessian_box", "starts", "midpoint",
                  "bound_check"],
    defaults=[True, None, None, None, None, (), "arithmetic", False])


def configure(spec_path) -> dict:
    """Load a run spec and set the run constants (under the names of the
    sealed drivers).  Returns the spec."""
    import json as _json
    g = globals()
    path = Path(spec_path)
    if not path.is_absolute():
        path = (MONO_ROOT / path) if (MONO_ROOT / path).is_file() else path.resolve()
    spec = _json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"run spec {path}: schema {spec.get('schema')!r} is not {SPEC_SCHEMA!r}")
    ident, stats, seeds, fit = spec["identity"], spec["statistics"], spec["seeds"], spec["fit"]
    factors = spec.get("flavor_channel_factors", [])
    if not isinstance(factors, list) or not set(factors) <= set(CHANNEL_FACTOR_DAUGHTERS) or len(set(factors)) != len(factors):
        raise ValueError(f"run spec: flavor_channel_factors must list distinct names among {tuple(CHANNEL_FACTOR_DAUGHTERS)}, "
                         f"got {factors!r}")
    g["CHANNEL_FACTORS"] = tuple(n for n in CHANNEL_FACTOR_DAUGHTERS if n in factors)
    # flavor_bbar_scope (2026-10-25): "every" applies r_BBbar to every channel that creates a baryon pair,
    # "continuation" to the pair steps that continue only (the floor's direct pairs keep their bare weight)
    bbar_scope = spec.get("flavor_bbar_scope", "every")
    if bbar_scope not in ("every", "continuation"):
        raise ValueError(f"run spec: flavor_bbar_scope must be 'every' or 'continuation', got {bbar_scope!r}")
    g["BBAR_CONTINUATION_ONLY"] = bbar_scope == "continuation"
    g["FLAVOR_ORDER"] = BASE_FLAVOR_ORDER + g["CHANNEL_FACTORS"]
    params = []
    for p in spec["parameters"]:
        if p["transform"] not in TRANSFORMS:
            raise ValueError(f"run spec: parameter {p['name']}: unknown transform {p['transform']!r}")
        params.append(ParamSpec(name=p["name"], transform=p["transform"], bounds=tuple(p["bounds"]), free=bool(p.get("free", True)),
                                value=p.get("value"), gen0=p.get("gen0"), hessian_step=p.get("hessian_step"),
                                hessian_box=tuple(p["hessian_box"]) if p.get("hessian_box") is not None else None,
                                starts=tuple(p.get("starts", ())), midpoint=p.get("midpoint", "arithmetic"),
                                bound_check=bool(p.get("bound_check", False))))
    names = [p.name for p in params]
    if names[-1] != "f_central" or len(set(names)) != len(names):
        raise ValueError("run spec: the parameter registry must list each parameter once, f_central last")
    by_name = {p.name: p for p in params}
    for required in ("epsilon", "b_sd", "epsilon_nd", "w_excited", "f_central"):
        if required not in by_name:
            raise ValueError(f"run spec: the registry lacks {required}")
    shape = tuple(p.name for p in params if p.free and p.name != "f_central")
    # every free parameter carries one dispersed start per fit start (2026-10-15: a short list
    # failed inside the first fit of a smoke run)
    n_starts_max = max(int(fit[k]) for k in ("n_starts", "n_starts_null", "n_starts_variant"))
    short = [p.name for p in params if p.free and len(p.starts) < n_starts_max]
    if short:
        raise ValueError(f"run spec: the parameters {short} list fewer than {n_starts_max} dispersed starts")
    flavor_free = tuple(n for n in shape if n in FLAVOR_ORDER)
    # r_V fixed (the "fixed" block or a fixed registry entry) or free in the registry (2026-10-13, R96)
    if "r_v" not in by_name and "r_v" not in spec.get("fixed", {}):
        raise ValueError("run spec: r_v is neither fixed nor in the parameter registry")
    flavor_bounds = {by_name[n].bounds for n in flavor_free}
    if len(flavor_bounds) > 1:
        raise ValueError("run spec: the free flavor weights must share their bounds")
    fixed = {p.name: float(p.value) for p in params if not p.free}
    fixed.update({k: float(v) for k, v in spec.get("fixed", {}).items()})
    gen0 = {p.name: (float(p.gen0) if p.gen0 is not None else fixed.get(p.name)) for p in params}
    gen0.update({k: float(v) for k, v in spec["generation_0"].get("extra", {}).items()})
    kernel = KernelConfig(**spec["kernel"])
    central = spec.get("central_component", True)
    if not isinstance(central, bool):
        raise ValueError(f"run spec: central_component must be a bool, got {central!r}")
    if not central:
        fc_spec = by_name["f_central"]
        if fc_spec.free or fc_spec.value is None or float(fc_spec.value) != 0.0:
            raise ValueError("run spec: without the central component f_central is fixed at 0 (free: false, value: 0.0)")
    topology = dict(spec.get("topology", {}))
    tail_law = topology.pop("sd_tail", None)
    if tail_law is not None:
        # [xi_lo, xi_hi, a, b], or with the smooth upper edge [xi_lo, xi_hi, a, b, xi_c, w] (2026-10-25: the factor
        # 1 / (1 + exp((xi - xi_c) / w)) in place of the wall at xi_hi; generation only, the tail events keep their
        # fixed law in the reweighting)
        if (not isinstance(tail_law, list) or len(tail_law) not in (4, 6)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in tail_law)
                or not 0.0 < float(tail_law[0]) < float(tail_law[1]) < 1.0 or float(tail_law[3]) <= 0.0
                or (len(tail_law) == 6 and not (float(tail_law[0]) < float(tail_law[4]) < float(tail_law[1])
                                                and float(tail_law[5]) > 0.0))):
            raise ValueError(f"run spec: topology sd_tail is [xi_lo, xi_hi, a, b] or [xi_lo, xi_hi, a, b, xi_c, w] with "
                             f"0 < xi_lo < xi_c < xi_hi < 1, b > 0, w > 0, got {tail_law!r}")
        tail_law = tuple(float(v) for v in tail_law)
    ope_law = topology.pop("sd_ope", None)
    if ope_law is not None:
        # the one-pion exchange's law (2026-10-26): [xi_lo, xi_hi, alpha', r2], the form factor r2 or "proton" (the
        # proton's Dirac form factor), an optional fifth entry the pi+ p cross section's law ("constant" or "regge"), a
        # sixth the exchanged pion (2026-10-27: "charged", or "isospin" for the share of the whole pion exchange with its
        # neutral partner p + p -> p + X+), a seventh the absorptive correction (2026-10-27: "none" or "eikonal"; 2026-10-30:
        # "eikonal_pdg", the PDG compilation's p+p elastic profile)
        numbers = ope_law[:3] + ([] if ope_law[3:4] == ["proton"] else ope_law[3:4]) if isinstance(ope_law, list) else None
        if (numbers is None or len(ope_law) not in (4, 5, 6, 7) or ope_law[4:5] not in ([], ["constant"], ["regge"])
                or ope_law[5:6] not in ([], ["charged"], ["isospin"]) or ope_law[6:] not in ([], ["none"], ["eikonal"], ["eikonal_pdg"])
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in numbers)
                or not 0.0 < float(ope_law[0]) < float(ope_law[1]) < 1.0 or min(float(v) for v in numbers[2:]) < 0.0):
            raise ValueError(f"run spec: topology sd_ope is [xi_lo, xi_hi, alpha', r2 | \"proton\"(, \"constant\" | \"regge\""
                             f"(, \"charged\" | \"isospin\"(, \"none\" | \"eikonal\" | \"eikonal_pdg\")))] "
                             f"with 0 < xi_lo < xi_hi < 1 and alpha', r2 >= 0, got {ope_law!r}")
        ope_law = tuple(v if isinstance(v, str) else float(v) for v in ope_law)
    two_sided = topology.pop("sd_two_sided", None)
    if two_sided is not None:
        # the two-sided hadronization of the heavy single-diffraction systems (2026-10-26): [m_cut, eps_line, b, scope]
        if (not isinstance(two_sided, list) or len(two_sided) != 4 or not isinstance(two_sided[3], str) or not two_sided[3]
                or set(two_sided[3]) - set("otc")
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in two_sided[:3])
                or float(two_sided[0]) <= 0.0 or float(two_sided[2]) <= 0.0):
            raise ValueError(f"run spec: topology sd_two_sided is [m_cut, eps_line, b, scope] with m_cut, b > 0 and the "
                             f"scope's letters from 'otc', got {two_sided!r}")
        two_sided = (float(two_sided[0]), float(two_sided[1]), float(two_sided[2]), str(two_sided[3]))
    t_break = topology.pop("intact_t_break", None)
    if t_break is not None:
        if (not isinstance(t_break, list) or len(t_break) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in t_break)
                or not float(t_break[0]) > 0.0 or not 0.0 < float(t_break[1]) <= 1.0):
            raise ValueError(f"run spec: topology intact_t_break is [t_b, rho] with t_b > 0 and 0 < rho <= 1, got {t_break!r}")
        if not (topology.get("exact_transfer_range") and "transfer_slope_mass_exponent" in topology):
            raise ValueError("run spec: intact_t_break needs exact_transfer_range and transfer_slope_mass_exponent")
        t_break = tuple(float(v) for v in t_break)
    for key, value in topology.items():
        if key in TOPOLOGY_NUMERIC_KEYS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"run spec: topology option {key!r} must be a finite non-negative number, got {value!r}")
            topology[key] = float(value)
            continue
        if key not in TOPOLOGY_KEYS or not isinstance(value, bool):
            raise ValueError(f"run spec: topology options are booleans among {TOPOLOGY_KEYS}, got {key!r}: {value!r}")
    if "transfer_slope_mass_exponent" in topology and not topology.get("exact_transfer_range"):
        raise ValueError("run spec: transfer_slope_mass_exponent needs exact_transfer_range true")
    if "intact_vertex_factor" in topology and not (topology["intact_vertex_factor"] > 0.0 and "transfer_slope_mass_exponent" in topology):
        raise ValueError("run spec: intact_vertex_factor is a positive factor of the vertex slopes (needs transfer_slope_mass_exponent)")
    if "diffractive_pomeron_slope" in topology and not (topology.get("exact_transfer_range") and "diffractive_xi_max" in topology):
        raise ValueError("run spec: diffractive_pomeron_slope needs exact_transfer_range and diffractive_xi_max")
    if "diffractive_pomeron_linear_t" in topology:
        t_lin = topology["diffractive_pomeron_linear_t"]
        if "diffractive_pomeron_slope" not in topology or not t_lin > 0.0:
            raise ValueError("run spec: diffractive_pomeron_linear_t is a positive |t| beside diffractive_pomeron_slope")
        if t_break is not None and float(t_break[0]) != float(t_lin):
            raise ValueError(f"run spec: diffractive_pomeron_linear_t {t_lin!r} must equal the intact proton's break {t_break[0]!r}")
    # (2026-10-07) the exact window applies to the 2 -> 2 classes; the class
    # with the central source keeps its own construction (at rest, or the
    # recoiling Stage B solve under central_recoil) and its committed draw
    if topology.get("central_recoil") and not central:
        raise ValueError("run spec: central_recoil needs central_component true")
    # the non-excited line of the single-diffraction classes (2026-10-09, from
    # R87): an intact proton or a low-mass baryonic resonance (plan line 825),
    # with weights that sum to one; the nominal R50 split when the spec names none
    split = spec.get("diffractive_split", NOMINAL_DIFFRACTIVE_SPLIT)
    if not isinstance(split, dict) or set(split) != set(REMNANT_KINDS):
        raise ValueError(f"run spec: diffractive_split takes the weights of {REMNANT_KINDS}, got {split!r}")
    for kind, value in split.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"run spec: diffractive_split {kind!r} must be a weight in [0, 1], got {value!r}")
    if split["high_mass_baryonic_continuum"] != 0.0:
        raise ValueError("run spec: diffractive_split: the non-excited line is an intact proton or a low-mass resonance "
                         "(high_mass_baryonic_continuum 0)")
    if abs(math.fsum(split.values()) - 1.0) > 1e-12:
        raise ValueError(f"run spec: diffractive_split weights must sum to 1, got {split!r}")
    split = {kind: float(split[kind]) for kind in REMNANT_KINDS}
    # the single-diffraction anchor's window (2026-10-23, from R109): the
    # measured share applies to the class's masses with M^2/s below it
    window = spec.get("single_diffraction_window")
    if window is not None and (isinstance(window, bool) or not isinstance(window, (int, float))
                               or not 0.0 < float(window) < 1.0):
        raise ValueError(f"run spec: single_diffraction_window is an M^2/s edge in (0, 1), got {window!r}")
    counts = spec.get("single_diffraction_window_counts", "coherence")
    if counts not in ("coherence", "intact_protons"):
        raise ValueError(f"run spec: single_diffraction_window_counts is 'coherence' or 'intact_protons', got {counts!r}")
    if counts == "intact_protons":
        if window is None or ope_law is None:
            raise ValueError("run spec: single_diffraction_window_counts 'intact_protons' needs the window and the sd_ope law")
        if tail_law is not None and float(tail_law[0]) < float(window):
            raise ValueError("run spec: 'intact_protons' counts the tail only above the window (its xi_lo below the window edge)")
    xi_cap = topology.get("diffractive_xi_max")
    if xi_cap is not None:
        if not 0.0 < float(xi_cap) <= 0.25:
            raise ValueError(f"run spec: diffractive_xi_max is an M^2/s edge in (0, 0.25], got {xi_cap!r}")
        if window is not None and float(window) > float(xi_cap):
            raise ValueError(f"run spec: the single-diffraction window {window!r} reaches beyond the coherence edge {xi_cap!r}")
    floor_grid = tuple(float(v) for v in spec["floor_grid"])
    iso = spec.get("isovector")
    iso_gen = None if iso is None else float(iso["generation_fraction"])
    if (iso_gen is None) != (ISO_NAME not in by_name):
        raise ValueError(f"run spec: the isovector block and the {ISO_NAME} parameter go together")
    if iso_gen is not None:
        if not 0.0 < iso_gen < 1.0:
            raise ValueError(f"run spec: the isovector generation fraction must lie in (0, 1), got {iso_gen!r}")
        if by_name[ISO_NAME].transform != "identity" or tuple(by_name[ISO_NAME].bounds) != (0.0, 1.0):
            raise ValueError(f"run spec: {ISO_NAME} takes the identity transform on the bounds [0, 1]")
        if not kernel.delta_lines:
            raise ValueError("run spec: the isovector exchange needs the kernel's Delta-like lines (delta_lines)")
    neutron = bool(fit.get("neutron_segment", False))
    bin_center = fit.get("bin_center_correction", False)
    correlated = parse_correlated_errors(fit.get("correlated_errors"), neutron)
    if not isinstance(bin_center, bool):
        raise ValueError(f"run spec: fit.bin_center_correction must be a bool, got {bin_center!r}")
    # the isovector block is read by name: an unknown key would be dropped
    # silently (2026-10-07; review 2026-10-06, Grok L3)
    unknown = sorted(set(iso or {}) - ISOVECTOR_KEYS)
    if unknown:
        raise ValueError(f"run spec: the isovector block takes the keys {sorted(ISOVECTOR_KEYS)}, got {unknown}")
    own_slope = bool((iso or {}).get("own_slope", False))
    if own_slope != (BIV_NAME in by_name):
        raise ValueError(f"run spec: isovector.own_slope and the {BIV_NAME} parameter go together")
    if own_slope and by_name[BIV_NAME].transform != "log":
        raise ValueError(f"run spec: {BIV_NAME} takes the log transform")
    central_power = EPSC_NAME in by_name
    if central_power and not central:
        raise ValueError(f"run spec: {EPSC_NAME} (the power central mass law) needs central_component true")
    if central_power and by_name[EPSC_NAME].transform != "identity":
        raise ValueError(f"run spec: {EPSC_NAME} takes the identity transform")
    intact_free = IVF_NAME in by_name
    if intact_free:
        if not by_name[IVF_NAME].free or by_name[IVF_NAME].transform != "log":
            raise ValueError(f"run spec: {IVF_NAME} in the registry is free on the log transform (a fixed factor is a topology option)")
        if IVF_NAME in topology or "transfer_slope_mass_exponent" not in topology:
            raise ValueError(f"run spec: the free {IVF_NAME} needs transfer_slope_mass_exponent and no {IVF_NAME} topology option")
    tail_free = TAIL_NAME in by_name
    if tail_free != (tail_law is not None):
        raise ValueError(f"run spec: the topology sd_tail and the {TAIL_NAME} parameter go together")
    if tail_free:
        if not by_name[TAIL_NAME].free or by_name[TAIL_NAME].transform != "identity":
            raise ValueError(f"run spec: {TAIL_NAME} is free on the identity transform")
        if not topology.get("exact_transfer_range"):
            raise ValueError("run spec: the single-diffraction tail needs exact_transfer_range")
    ope_free = OPE_NAME in by_name
    if ope_free != (ope_law is not None):
        raise ValueError(f"run spec: the topology sd_ope and the {OPE_NAME} parameter go together")
    if ope_free:
        if not by_name[OPE_NAME].free or by_name[OPE_NAME].transform != "identity":
            raise ValueError(f"run spec: {OPE_NAME} is free on the identity transform")
        if not tail_free:
            raise ValueError("run spec: the one-pion exchange shares the tail's component draw and needs the free tail")
    split_free = tuple(n for n in SPLIT_RW.SPLIT_NAMES if n in by_name)
    if split_free:
        if split_free != SPLIT_RW.SPLIT_NAMES:
            raise ValueError(f"run spec: the split parameters are free together, {SPLIT_RW.SPLIT_NAMES}; got {split_free}")
        if not ope_free:
            raise ValueError("run spec: the free split parameters ride on the share argument and need the free one-pion exchange")
        for n in split_free:
            want = "logit" if n.endswith("_fraction") else "log"
            if not by_name[n].free or by_name[n].transform != want:
                raise ValueError(f"run spec: {n} is free on the {want} transform")
            if getattr(kernel, n) is None:
                raise ValueError(f"run spec: the free {n} needs its kernel option set (the kernel then takes the generation point)")
    extra_shape = (((BIV_NAME,) if own_slope else ()) + ((EPSC_NAME,) if central_power else ())
                   + ((IVF_NAME,) if intact_free else ()) + ((TAIL_NAME,) if tail_free else ())
                   + ((OPE_NAME,) if ope_free else ()) + split_free)
    g.update(
        SPEC=spec, SPEC_PATH=path,
        SPEC_REL=(path.resolve().relative_to(MONO_ROOT.resolve()).as_posix()
                  if path.resolve().is_relative_to(MONO_ROOT.resolve()) else path.resolve().as_posix()),
        RUN_ID=ident["run_id"], CARD=ident["card"], SCHEMA=ident["schema"], LEDGER_RUN=ident["ledger_run"],
        GATE_RECORD_NAME=ident["gate_record"],
        REPLICAS=int(stats["replicas"]), EVENTS_PER_SLOT=int(stats["events_per_slot"]), MAX_GENERATIONS=int(stats["max_generations"]),
        FLOOR_GRID=floor_grid, N_FLOORS=len(floor_grid),
        MASTER_SEED=int(seeds["master"]), RAW_FIRST=int(seeds["raw_first"]), PARTITION_BASE=int(seeds["partition_base"]),
        SMOKE_SEED=int(seeds["smoke_seed"]), SMOKE_PARTITION=int(seeds["smoke_partition"]),
        KERNEL=kernel, CONTINUATION_CEILING=kernel.continuation_ceiling,
        PARAMS=tuple(params), PARAM_INDEX=by_name,
        EPS_BOUNDS=by_name["epsilon"].bounds, B_SD_BOUNDS=by_name["b_sd"].bounds, EPS_ND_BOUNDS=by_name["epsilon_nd"].bounds,
        LOGIT_BOUNDS=by_name["w_excited"].bounds, FC_BOUNDS=by_name["f_central"].bounds,
        FLAVOR_LOG_BOUNDS=tuple(math.log(b) for b in next(iter(flavor_bounds))) if flavor_bounds else None,
        HESSIAN_STEPS={n: float(by_name[n].hessian_step) for n in shape},
        SHAPE_NAMES=shape, FLAVOR_FREE=flavor_free, R_V_FIXED=fixed.get("r_v"), FIXED_VALUES=fixed,
        FC_SCAN=tuple(float(v) for v in fit["fc_scan"]), TEST_DELTA_CHI2=float(fit["test_delta_chi2"]),
        N_FREE_PARAMETERS=len(shape) + (1 if central else 0), N_STARTS=int(fit["n_starts"]), N_STARTS_NULL=int(fit["n_starts_null"]),
        N_STARTS_VARIANT=int(fit["n_starts_variant"]), FIT_MAXFEV=int(fit["maxfev"]),
        GEN0_EPSILON=gen0["epsilon"], GEN0_B_SD=gen0["b_sd"], GEN0_EPSILON_ND=gen0["epsilon_nd"],
        GEN0_W_EXCITED=gen0["w_excited"], GEN0_W_LOW=gen0["w_low_mass"], GEN0_F_CENTRAL=gen0["f_central"],
        GEN_FLAVOR=tuple(gen0[n] if gen0.get(n) is not None else fixed[n] for n in FLAVOR_ORDER),
        TEXTS=spec["texts"], READOUT_LABEL=spec.get("readout_label", "engine_readout"),
        ISOVECTOR_GEN=iso_gen, NEUTRON_SEGMENT=neutron, CORRELATED_ERRORS=correlated, ISOVECTOR_SLOPE=own_slope,
        SEGMENT_NAMES=FIT_SPECIES + (("neutron",) if neutron else ()),
        CENTRAL_COMPONENT=central, FC_DEFAULT=None if central else 0.0,
        ACTIVE_CLASSES=CLASS_ORDER if central else tuple(c for c in CLASS_ORDER if c != "nd"),
        TOPOLOGY_OPTIONS=topology, EXACT_RANGE=bool(topology.get("exact_transfer_range", False)),
        CENTRAL_POWER=central_power, EXTRA_SHAPE=extra_shape, INTACT_FREE=intact_free,
        TAIL_FREE=tail_free, SD_TAIL_LAW=tail_law, T_BREAK=t_break,
        OPE_FREE=ope_free, SD_OPE_LAW=ope_law, SD_TWO_SIDED=two_sided, SPLIT_FREE=split_free,
        LEDGER_SSBAR_SCALE=10 if kernel.eta_prime else 2,
        BIN_CENTER=bin_center, FROZEN_REMNANT_WEIGHTS=split,
        SD_WINDOW=None if window is None else float(window),
        SD_WINDOW_INTACT=(counts == "intact_protons"),
        NEUTRAL_OPE_WINDOW=(neutral_ope_window_fraction(ope_law, float(window)) if counts == "intact_protons" else 0.0),
    )
    g.update(N_SLOTS=len(CLASS_ORDER) * REPLICAS, SEEDED_SLOTS=(len(CLASS_ORDER) - 1) * REPLICAS)
    g.update(ACTIVE_SLOTS=tuple(s for s in range(N_SLOTS) if slot_class(s) in ACTIVE_CLASSES))
    g.update(RAW_LAST=RAW_FIRST + MAX_GENERATIONS * SEEDED_SLOTS - 1,
             GEN0_SHAPE=(GEN0_EPSILON, GEN0_B_SD, GEN0_EPSILON_ND) + tuple(float(gen0[n]) for n in EXTRA_SHAPE),
             ND_GENERATION_REMNANT_WEIGHTS={"intact_proton": 0.0, "low_mass_baryonic_resonance": GEN0_W_LOW,
                                            "high_mass_baryonic_continuum": GEN0_W_EXCITED})
    g.update(FROZEN_RELS=BASE_FROZEN_RELS + (HERE.relative_to(MONO_ROOT.resolve()).as_posix(), SPEC_REL)
             + tuple(spec.get("frozen_extra", ())),
             VARIANTS=(("discrepancy_0.10", DISCREPANCY_VARIANT, NORM_FIXED), ("normalization_+0.02", DISCREPANCY, NORM_VARIANTS[0]),
                       ("normalization_-0.02", DISCREPANCY, NORM_VARIANTS[1]))
             + tuple(v[:3] for v in systematic_variants(spec)),
             VARIANT_ANCHORS={v[0]: v[3] for v in systematic_variants(spec) if v[3]})
    return spec


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

#: the frozen files of every engine run; configure() appends the engine,
#: the spec and the spec's frozen extras
BASE_FROZEN_RELS = (
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
    "helpers/benchmarking/fit_fanout.py",
    "helpers/benchmarking/flavor_fit_core.py",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_proton_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_antiproton_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_piminus.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_piplus.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaplus_pp.txt",
    "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaminus_pp.txt",
)
FROZEN_RELS = ()

SLOT_JSON_FMT = "slot_{cls}_{k}_f{f}_g{g}.json"
SLOT_NPZ_FMT = "slot_{cls}_{k}_f{f}_g{g}.npz"
FIT_REPORT_FMT = "fit_report_f{f}_g{g}.json"
BRANCH_REPORT_FMT = "branch_report_f{f}.json"
REGENERATION_FMT = "regeneration_request_f{f}.json"
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


def frozen_rels() -> tuple:
    """FROZEN_RELS together with every loaded project module (external
    review q1 H1), in a fixed order."""
    return tuple(dict.fromkeys(list(FROZEN_RELS) + RC.loaded_module_rels(MONO_ROOT)))


def frozen_digests() -> dict:
    return {rel: RC.sha256_file(MONO_ROOT / rel) for rel in frozen_rels()}


def anchors(override: dict | None = None) -> dict:
    """The class-fraction anchors of the external cross sections; *override*
    (a systematic variant of the spec, 2026-10-22) replaces the values of
    ``f_sd_total`` and ``f_dd`` it names, the uncertainties kept."""
    ext = RC.read_json(NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json")
    f = ext["fractions_of_inelastic"]
    out = {
        "f_sd_total": {"value": float(f["f_sd_total"]["value"]), "sigma": float(f["f_sd_total"]["uncertainty"])},
        "f_dd": {"value": float(f["f_dd"]["value"]), "sigma": float(f["f_dd"]["uncertainty"])},
        "source": "config/v2_singlet/external_cross_sections.json",
        "source_sha256": RC.sha256_file(NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json"),
    }
    names = FRACTION_NAMES
    if SD_WINDOW is not None:
        w = f[WINDOW_NAME]
        if float(w["xi_max"]) != float(SD_WINDOW):
            raise ValueError(f"anchors: the measured window M^2/s < {w['xi_max']} differs from the spec's {SD_WINDOW}")
        out[WINDOW_NAME] = {"value": float(w["value"]), "sigma": float(w["uncertainty"]), "xi_max": float(SD_WINDOW)}
        names = FRACTION_NAMES + (WINDOW_NAME,)
    for name, value in (override or {}).items():
        if name not in names:
            raise ValueError(f"anchors: the override names {name!r}, not one of {names}")
        out[name] = {**out[name], "value": float(value)}
    if override:
        out["override"] = {k: float(v) for k, v in override.items()}
    return out


def neutron_table() -> tuple[np.ndarray, np.ndarray]:
    """The NA49 neutron dN/dx_F per inelastic event (x_F > 0) at the bin
    centers of NEUTRON_XF_EDGES and its error (the table gives percent)."""
    rows = [l.split() for l in (NESTED_ROOT / NEUTRON_TABLE_REL).read_text().splitlines() if l.strip() and not l.startswith("#")]
    data = np.asarray([float(r[1]) for r in rows])
    sigma = np.asarray([float(r[1]) * float(r[2]) / 100.0 for r in rows])
    if data.size != NEUTRON_XF_EDGES.size - 1:
        raise ValueError(f"the neutron table has {data.size} rows for {NEUTRON_XF_EDGES.size - 1} bins")
    return data, sigma


def species_divisor(table) -> np.ndarray:
    """The divisor that turns a species' per-bin weight sums into the
    model's invariant cross section per inelastic event: pi sqrt(s) times
    the bin area, divided under BIN_CENTER by the bin-center factors (the
    model's bin average of f converted to its value at the tabulated point
    from the data's own local shape, ``run_observables.bin_center_factors``;
    2026-10-07, card TASK-20261005-bin-center-convention)."""
    divisor = math.pi * SQRT_S_CM * OB.bin_areas(table)
    return divisor / OB.bin_center_factors(table) if BIN_CENTER else divisor


def neutron_divisor(data: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """The divisor of the neutron dN/dx_F segment: the x_F bin width,
    divided under BIN_CENTER by the one-dimensional bin-center factors."""
    widths = np.diff(NEUTRON_XF_EDGES)
    if not BIN_CENTER:
        return widths
    centers = 0.5 * (NEUTRON_XF_EDGES[:-1] + NEUTRON_XF_EDGES[1:])
    return widths / OB.bin_center_factors_1d(centers, data, NEUTRON_XF_EDGES, sigma)


def load_fit_tables() -> dict:
    return OB.load_tables(NESTED_ROOT, windows=TABLE_WINDOWS)


# ----------------------------------------------------------------------
# Event generation (one slot per worker process)
# ----------------------------------------------------------------------
def _load_physics(floor: float, shape_point=None):
    """The committed catalog and continuum families with the spec's kernel
    configuration applied, the continuation floor of the branch set
    (``na49_mixing.kernel_config.load_kernel``; the committed config sets no
    kernel option); with free split parameters (2026-11-03) the kernel takes
    their values from *shape_point*, the generation point."""
    split = ({n: float(shape_point[shape_pos(n)]) for n in SPLIT_FREE} if (SPLIT_FREE and shape_point is not None) else {})
    return load_kernel(_dc.replace(KERNEL, continuation_floor=float(floor), **split), NESTED_ROOT)


def flavor_params(point) -> FlavorParams:
    """The FlavorParams of a point in FLAVOR_ORDER (default hidden s-sbar content; the channel factors of a
    diagnostic run as its channel_factors)."""
    values = dict(zip(FLAVOR_ORDER, (float(v) for v in point)))
    if len(values) != len(FLAVOR_ORDER):
        raise ValueError(f"flavor_params: {len(values)} values against FLAVOR_ORDER {FLAVOR_ORDER}")
    return FlavorParams(**{name: values[name] for name in BASE_FLAVOR_ORDER},
                        channel_factors={name: values[name] for name in CHANNEL_FACTORS},
                        bbar_continuation_only=BBAR_CONTINUATION_ONLY)


def flavor_of(phys: dict) -> tuple:
    return tuple(float(phys[name]) for name in FLAVOR_ORDER)


def shape_pos(name: str) -> int:
    """The index of an extra shape coordinate in a shape point."""
    return 3 + EXTRA_SHAPE.index(name)


def _params_at(cls: str, shape_point, flavor_point=None) -> TopologyNominalParams:
    """The generator parameters of a template class at the shape point
    (epsilon, B_SD, epsilon_nd) and the flavor point (FLAVOR_ORDER)."""
    flavor_point = GEN_FLAVOR if flavor_point is None else flavor_point
    eps, b_sd, eps_nd = (float(v) for v in shape_point[:3])
    weights = ND_GENERATION_REMNANT_WEIGHTS if cls in ND_CLASSES else FROZEN_REMNANT_WEIGHTS
    extra = {} if ISOVECTOR_GEN is None else {"isovector_fraction": float(ISOVECTOR_GEN)}
    if ISOVECTOR_SLOPE:
        extra["b_isovector"] = float(shape_point[shape_pos(BIV_NAME)])
    if CENTRAL_POWER:
        extra["epsilon_c"] = float(shape_point[shape_pos(EPSC_NAME)])
    if INTACT_FREE and cls != "nd":
        # the class with the central source draws no factorized vertex slopes
        extra["intact_vertex_factor"] = float(shape_point[shape_pos(IVF_NAME)])
    if T_BREAK is not None and cls in ("sd_projectile", "sd_target"):
        # the intact proton's two-slope transfer (2026-10-24)
        extra["intact_t_break"] = tuple(T_BREAK)
    if TAIL_FREE and cls in ("sd_projectile", "sd_target"):
        # the leading-proton tail at the generation share (2026-10-23, R112)
        extra["sd_tail"] = (float(shape_point[shape_pos(TAIL_NAME)]),) + tuple(SD_TAIL_LAW)
    if OPE_FREE and cls in ("sd_projectile", "sd_target"):
        # the one-pion exchange at the generation share (2026-10-26)
        extra["sd_ope"] = (float(shape_point[shape_pos(OPE_NAME)]),) + tuple(SD_OPE_LAW)
    if SD_TWO_SIDED is not None and cls in ("sd_projectile", "sd_target"):
        extra["sd_two_sided"] = tuple(SD_TWO_SIDED)
    return TopologyNominalParams(
        epsilon=eps, b_sd=b_sd, class_fractions=dict(NOMINAL_PARAMS.class_fractions),
        remnant_weights=dict(weights), p_n2=0.0, flavor=flavor_params(flavor_point),
        epsilon_nd=eps_nd, nd_central=(cls != "nd0"), **extra,
        **{k: (False if (k == "exact_transfer_range" and cls == "nd") else v) for k, v in TOPOLOGY_OPTIONS.items()
           if not (k in ("transfer_slope_mass_exponent", "intact_vertex_factor", "diffractive_duality_mass") and cls == "nd")},
    )


#: the index of each isovector configuration (the event's iv_config; -1 is
#: the vacuum exchange)
_IV_INDEX = {c: k for k, (_p, c) in enumerate(ISOVECTOR_CONFIGURATIONS)}


def iv_config_index(event) -> int:
    """The exchange configuration of an event: its index in
    ISOVECTOR_CONFIGURATIONS, or -1 for the vacuum exchange."""
    config = getattr(event, "exchange_configuration", None)
    if config is None or config.is_vacuum or getattr(event, "ope", None):
        # a one-pion-exchange event (2026-10-26) belongs to the vacuum share of the isovector draw
        return -1
    return int(_IV_INDEX[config])


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
        # on shell relative to max(E^2, 1 GeV^2): the float rounding of
        # E^2 - p^2 follows the GeV-scale momenta of the decay chain, so a soft
        # final state carries it in absolute terms (two R71 launches failed on a
        # 129 keV photon from a boosted radiative decay, m^2 = 1.75e-17 GeV^2,
        # 1.05e-9 of its E^2); a fast hadron is checked against its own E^2
        worst_shell = max(worst_shell, abs(m2 - float(h.mass) ** 2) / max(e * e, 1.0))
    if worst_shell > EVENT_CLOSURE_TOL:
        raise RuntimeError(f"conservation: mass-shell deviation {worst_shell!r}")
    return worst, worst_shell


def _generate_range(cls: str, seed: int, n_events: int, partition_base: int, shape_point, floor: float,
                    flavor_point, start: int, stop: int) -> dict:
    """Generate one slot of a template class at the shape point and the
    fixed R66 flavor point, with the flavor ledger of every drawn
    continuum vertex: returns the compact arrays and the metadata."""
    shape_point = tuple(float(v) for v in shape_point)
    catalog, fams = _load_physics(floor, shape_point)
    flavor_point = tuple(float(v) for v in flavor_point)
    params = _params_at(cls, shape_point, flavor_point)
    runner_cls = RUNNER_CLASS.get(cls, cls)
    beta, gamma = OB.cm_boost(SQRT_S_CM)
    m = int(stop) - int(start)
    rem_kind = np.full((m, 2), -1, dtype=np.int8)
    species_pdg = {name: OB.SPECIES_PDG[name] for name in TRACK_SPECIES}
    shape: dict[str, list] = {}
    lead = np.full(m, np.nan)
    gap = np.full(m, np.nan)
    n_ch = np.zeros(m, dtype=np.int16)
    n_pi0 = np.zeros(m, dtype=np.int16)
    neu_event: list[int] = []
    neu_xf: list[float] = []
    iv_config = np.full(m, -1, dtype=np.int8)
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
    event_digests: list[bytes] = []
    worst_closure = 0.0
    worst_shell = 0.0
    t0 = time.perf_counter()
    for i in range(int(start), int(stop)):
        j = i - int(start)
        event_id = int(partition_base) + i
        subs = derive_substreams(int(seed), 0, event_id)
        rngs = {f"rng_{label}": np.random.default_rng(subs.stream_for(label))
                for label in ("topology_selection", "spectral_state", "exchange_kinematics",
                              "decay_channel", "decay_kinematics")}
        ledger: list = []
        if SPLIT_FREE:
            # the split records of the event's roots (2026-11-03): the recorder draws no random number
            CASCADE_DECAY.SPLIT_RECORDER = []
            try:
                event = run_topology(runner_cls, params, catalog, fams, **rngs, ledger=ledger)
                split_rows = SPLIT_RW.records_to_row(CASCADE_DECAY.SPLIT_RECORDER)
            finally:
                CASCADE_DECAY.SPLIT_RECORDER = None
            for key, value in split_rows.items():
                shape.setdefault(key, []).append(value)
        else:
            event = run_topology(runner_cls, params, catalog, fams, **rngs, ledger=ledger)
        closure, shell = event_conservation(event)
        worst_closure = max(worst_closure, closure)
        worst_shell = max(worst_shell, shell)
        obs = extract_observables(event, SQRT_S_CM)
        arrays = OB.shape_arrays(event)
        for key, value in arrays.items():
            shape.setdefault(key, []).append(value)
        lead_xf = OB.leading_proton_xf(event.final_state, beta, gamma, SQRT_S_CM)
        lead[j] = np.nan if lead_xf is None else lead_xf
        gap[j] = np.nan if obs.gap_size is None else float(obs.gap_size)
        n_ch[j] = OB.charged_multiplicity(event.final_state)
        for h in event.final_state:
            if h.pdg_id is None:
                continue
            if int(h.pdg_id) == 111:
                n_pi0[j] += 1
            elif int(h.pdg_id) == 2112:
                _e, _px, _py, pz_cm = OB.to_cm(h.four_momentum, beta, gamma)
                neu_event.append(i); neu_xf.append(2.0 * pz_cm / SQRT_S_CM)
        if cls in ND_CLASSES:
            rem_kind[j] = [REMNANT_KINDS.index(r.kind) for r in event.remnants]
        tracks = OB.track_arrays(event.final_state, species_pdg, beta, gamma, SQRT_S_CM)
        line = [cls, str(i), str(int(n_ch[j])), float(lead[j]).hex(), float(gap[j]).hex(), f"{int(rem_kind[j][0])}/{int(rem_kind[j][1])}"]
        if ISOVECTOR_GEN is not None:
            iv_config[j] = iv_config_index(event)
            line.append(f"X{int(iv_config[j])}")
        for s_index, name in enumerate(TRACK_SPECIES):
            xs, ps, ws = tracks[name]
            for xf, pt, w in zip(xs, ps, ws):
                trk_event.append(i); trk_species.append(s_index)
                trk_xf.append(xf); trk_pt.append(pt); trk_w.append(w)
                line.append(f"{s_index}:{xf.hex()}:{pt.hex()}:{w.hex()}")
        event_vertex = 0
        for record in ledger:
            if record is None:
                continue
            chosen_key, groups = record
            led_event.append(i); led_chosen.append(FC._key_ints(chosen_key, LEDGER_SSBAR_SCALE))
            for key, weight in sorted(groups.items()):
                led_gvertex.append(vertex_counter); led_gkey.append(FC._key_ints(key, LEDGER_SSBAR_SCALE)); led_gw.append(float(weight))
                line.append(f"L{event_vertex}:{'/'.join(str(v) for v in FC._key_ints(key, LEDGER_SSBAR_SCALE))}:{float(weight).hex()}")
            vertex_counter += 1
            event_vertex += 1
        event_digests.append(hashlib.sha256(("|".join(line) + "\n").encode("utf-8")).digest())
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
            "led_chosen": np.asarray(led_chosen, dtype=np.int16).reshape(-1, len(FLAVOR_ORDER)),
            "led_gvertex": np.asarray(led_gvertex, dtype=np.int32),
            "led_gkey": np.asarray(led_gkey, dtype=np.int16).reshape(-1, len(FLAVOR_ORDER)),
            "led_gw": np.asarray(led_gw, dtype=np.float32),
            **({"led_ssbar_scale": np.asarray([LEDGER_SSBAR_SCALE], dtype=np.int16)} if LEDGER_SSBAR_SCALE != 2 else {}),
            **({"iv_config": iv_config} if ISOVECTOR_GEN is not None else {}),
        },
        **({"isovector_generation_fraction": float(ISOVECTOR_GEN)} if ISOVECTOR_GEN is not None else {}),
        **({BIV_NAME: float(shape_point[shape_pos(BIV_NAME)])} if ISOVECTOR_SLOPE else {}),
        **({EPSC_NAME: float(shape_point[shape_pos(EPSC_NAME)])} if CENTRAL_POWER else {}),
        **({IVF_NAME: float(shape_point[shape_pos(IVF_NAME)])} if INTACT_FREE else {}),
        **({TAIL_NAME: float(shape_point[shape_pos(TAIL_NAME)])} if TAIL_FREE else {}),
        **({OPE_NAME: float(shape_point[shape_pos(OPE_NAME)])} if OPE_FREE else {}),
        **{n: float(shape_point[shape_pos(n)]) for n in SPLIT_FREE},
        "informative_vertices": int(vertex_counter),
        "event_digests": event_digests,
        "conservation": {"events": int(m), "violations": 0,
                         "worst_closure_relative": worst_closure, "worst_mass_shell_relative": worst_shell,
                         "tolerance": EVENT_CLOSURE_TOL},
        "tracks": int(len(trk_event)),
        "wall_seconds": wall,
    }


def _generate_range_job(*args) -> dict:
    return _generate_range(*args)


def generate_slot(cls: str, seed: int, n_events: int, partition_base: int, shape_point, floor: float,
                  flavor_point=None, chunks: int = 1) -> dict:
    """Generate one slot, its events split into *chunks* contiguous ranges
    run in parallel and merged in order (the arrays and the event digest
    do not depend on the split)."""
    flavor_point = GEN_FLAVOR if flavor_point is None else flavor_point
    n = int(n_events)
    k = max(1, min(int(chunks), n))
    bounds = [(int(c[0]), int(c[-1]) + 1) for c in np.array_split(np.arange(n), k) if c.size]
    jobs = [(cls, int(seed), n, int(partition_base), tuple(float(v) for v in shape_point), float(floor),
             tuple(float(v) for v in flavor_point), a, b) for a, b in bounds]
    parts = RC.run_replicas(_generate_range_job, jobs, max_workers=len(jobs)) if len(jobs) > 1 else [_generate_range(*jobs[0])]
    first = parts[0]
    arrays = {key: np.concatenate([p["arrays"][key] for p in parts], axis=0) for key in first["arrays"]}
    offsets = np.cumsum([0] + [int(p["informative_vertices"]) for p in parts[:-1]])
    arrays["led_gvertex"] = np.concatenate([p["arrays"]["led_gvertex"] + np.int32(o) for p, o in zip(parts, offsets)]).astype(np.int32)
    return {**{key: first[key] for key in ("cls", "seed", "n_events", "partition_base", "epsilon", "b_sd", "epsilon_nd",
                                           "continuation_floor", "flavor_generation", "nd_central", "remnant_weights")},
            **({"isovector_generation_fraction": first["isovector_generation_fraction"]}
               if "isovector_generation_fraction" in first else {}),
            **({BIV_NAME: first[BIV_NAME]} if BIV_NAME in first else {}),
            **({EPSC_NAME: first[EPSC_NAME]} if EPSC_NAME in first else {}),
            **({IVF_NAME: first[IVF_NAME]} if IVF_NAME in first else {}),
            **({TAIL_NAME: first[TAIL_NAME]} if TAIL_NAME in first else {}),
            **({OPE_NAME: first[OPE_NAME]} if OPE_NAME in first else {}),
            **{n: first[n] for n in SPLIT_FREE if n in first},
            "arrays": arrays,
            "informative_vertices": int(sum(int(p["informative_vertices"]) for p in parts)),
            "event_rolling_sha256": hashlib.sha256(b"".join(d for p in parts for d in p["event_digests"])).hexdigest(),
            "event_digest_rule": "sha256 over the per-event line digests in event order (vertices numbered within each event)",
            "conservation": {"events": n, "violations": 0,
                             "worst_closure_relative": max(p["conservation"]["worst_closure_relative"] for p in parts),
                             "worst_mass_shell_relative": max(p["conservation"]["worst_mass_shell_relative"] for p in parts),
                             "tolerance": EVENT_CLOSURE_TOL},
            "tracks": int(sum(int(p["tracks"]) for p in parts)),
            "chunks": len(parts),
            "wall_seconds": float(sum(p["wall_seconds"] for p in parts))}


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
        **({"isovector_generation_fraction": result["isovector_generation_fraction"]}
           if "isovector_generation_fraction" in result else {}),
        **({BIV_NAME: result[BIV_NAME]} if BIV_NAME in result else {}),
        **({EPSC_NAME: result[EPSC_NAME]} if EPSC_NAME in result else {}),
        **({IVF_NAME: result[IVF_NAME]} if IVF_NAME in result else {}),
        **({TAIL_NAME: result[TAIL_NAME]} if TAIL_NAME in result else {}),
        **({OPE_NAME: result[OPE_NAME]} if OPE_NAME in result else {}),
        **{n: result[n] for n in SPLIT_FREE if n in result},
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
                           alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=(READOUT_LABEL,), depth=0)
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


def mesonic_count_readout(floor: float, n_decays: int = MESONIC_DECAYS, flavor_point=None) -> dict:
    """The mean charged count per decay of a mesonic_neutral source at rest
    at MESONIC_MASSES against the MARK I curve (the charged products of
    the final neutral weak decayers counted as in the measurement), at the
    flavor point *flavor_point*."""
    flavor_point = GEN_FLAVOR if flavor_point is None else flavor_point
    flavor = flavor_params(flavor_point)
    catalog, fams = _load_physics(floor)
    family = next(f for f in fams if f.family_id == "mesonic_neutral")
    rows = []
    for m_index, mass in enumerate(MESONIC_MASSES):
        key = [MASTER_SEED, 69, int(round(1000 * floor)), m_index]
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
            "seed_rule": "numpy SeedSequence([MASTER_SEED, 69, round(1000 floor), mass index, 1 or 2])", "rows": rows}


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
        self.flavor_point = None
        self.by_class: dict[str, list] = {cls: [] for cls in ACTIVE_CLASSES}
        n_slots = N_SLOTS if not smoke else len(CLASS_ORDER)
        wanted = None if replicas is None else {int(r) for r in replicas}
        self.replica_ids = sorted(wanted) if wanted is not None else (list(range(REPLICAS)) if not smoke else [0])
        for slot in range(n_slots):
            cls = slot_class(slot) if not smoke else CLASS_ORDER[slot]
            k = slot_replica(slot) if not smoke else 0
            if cls not in ACTIVE_CLASSES:
                continue
            if wanted is not None and not smoke and k not in wanted:
                continue
            json_name = ("smoke_" if smoke else "") + SLOT_JSON_FMT.format(cls=cls, k=k, f=floor_index, g=generation)
            npz_name = ("smoke_" if smoke else "") + SLOT_NPZ_FMT.format(cls=cls, k=k, f=floor_index, g=generation)
            meta = RC.read_json(contract.path(json_name))
            if float(meta["continuation_floor"]) != float(FLOOR_GRID[floor_index]):
                contract.refuse(f"{json_name} was generated at floor {meta['continuation_floor']}, not {FLOOR_GRID[floor_index]}")
            flavor_point = tuple(float(meta["flavor_generation"][name]) for name in FLAVOR_ORDER)
            if self.flavor_point is None:
                self.flavor_point = flavor_point
            elif self.flavor_point != flavor_point:
                contract.refuse(f"mixed generation flavor points in branch {floor_index} generation {generation}")
            if bool(meta.get("nd_central", True)) != (cls != "nd0"):
                contract.refuse(f"{json_name} carries the wrong central-source setting for class {cls}")
            if meta.get("isovector_generation_fraction") != ISOVECTOR_GEN:
                contract.refuse(f"{json_name} was generated at the isovector fraction {meta.get('isovector_generation_fraction')}, "
                                f"not {ISOVECTOR_GEN}")
            with np.load(contract.path(npz_name)) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays_digest(arrays) != meta["arrays_sha256"]:
                contract.refuse(f"the arrays of {npz_name} do not match their recorded digest")
            point = (float(meta["epsilon"]), float(meta["b_sd"]), float(meta["epsilon_nd"])) + tuple(
                float(meta[n]) for n in EXTRA_SHAPE)
            if self.point is None:
                self.point = point
            elif self.point != point:
                contract.refuse(f"mixed generation points in branch {floor_index} generation {generation}")
            if cls in ND_CLASSES and np.any(arrays["rem_kind"] == REMNANT_KINDS.index("intact_proton")):
                contract.refuse(f"{npz_name} carries an intact remnant; the generation weight of that kind is 0")
            self.by_class[cls].append(self._prepare(cls, arrays, int(meta["n_events"])))
        for cls in ACTIVE_CLASSES:
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
        if NEUTRON_SEGMENT:
            n_bins = NEUTRON_XF_EDGES.size - 1
            idx = np.searchsorted(NEUTRON_XF_EDGES, arrays["neu_xf"]) - 1
            ok = (idx >= 0) & (idx < n_bins)
            binned["neutron"] = (arrays["neu_event"][ok], idx[ok].astype(np.int64), np.ones(int(np.sum(ok))), n_bins)
        totals = {}
        for s_index, name in enumerate(TRACK_SPECIES):
            sel = arrays["trk_species"] == s_index
            totals[name] = np.bincount(arrays["trk_event"][sel].astype(np.int64), minlength=int(n_events))[: int(n_events)]
        return {"n": int(n_events), "shape": shape, "binned": binned, "ledger": self._ledger(arrays),
                "lead_xf": arrays["lead_xf"], "gap": arrays["gap"], "n_ch": arrays["n_ch"],
                "rem_kind": arrays.get("rem_kind"), "totals": totals,
                "n_pi0": arrays["n_pi0"], "neu_event": arrays["neu_event"], "neu_xf": arrays["neu_xf"],
                "iv": (arrays["iv_config"] >= 0) if "iv_config" in arrays else None}

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
        # base-100 codes of the key entries (four plan exponents, then a diagnostic run's channel-factor counts)
        dims = int(both.shape[1])
        if dims != len(FLAVOR_ORDER):
            raise ValueError(f"the ledger keys carry {dims} entries against FLAVOR_ORDER {FLAVOR_ORDER}")
        codes = both[:, 0].copy()
        for j in range(1, dims):
            codes = codes * 100 + both[:, j]
        table, inverse = np.unique(codes, return_inverse=True)
        k = np.empty((table.size, dims), dtype=float)
        rest = table.copy()
        for j in range(dims - 1, -1, -1):
            k[:, j] = rest % 100
            rest = rest // 100
        # the ledger stores n_ssbar times its scale: 2 through R97, 10 with the eta' (2026-10-15)
        scale = int(arrays["led_ssbar_scale"][0]) if "led_ssbar_scale" in arrays else 2
        k[:, 0] /= float(scale)
        gv = arrays["led_gvertex"].astype(np.int64)
        gw = arrays["led_gw"].astype(float)
        z_from = np.bincount(gv, weights=gw, minlength=n_v)[:n_v]
        return {"n_vertices": n_v, "v_event": arrays["led_event"].astype(np.int64), "gv": gv, "gw": gw,
                "g_idx": inverse[: gkey.shape[0]], "c_idx": inverse[gkey.shape[0]:], "keys": k, "log_z_from": np.log(z_from)}

    def flavor_log_weights(self, block: dict, flavor_point) -> np.ndarray:
        """The per-event log importance weights from the generation flavor
        point to *flavor_point* (the R66 key-table expression of
        flavor_fit_core.flavor_log_weights)."""
        point = tuple(float(v) for v in flavor_point)
        led = block["ledger"]
        if point == self.flavor_point or led is None:
            return np.zeros(block["n"])
        # the last value per block (2026-10-20, the fit profile: a Powell line
        # search along a shape direction repeats the flavor point; the array is
        # read-only, the same floats as a recomputation)
        memo = self.__dict__.setdefault("_flavor_memo", {})
        hit = memo.get(id(block))
        if hit is not None and hit[0] == point:
            return hit[1]
        value = Templates._flavor_log_weights(self, block, led, point)
        value.setflags(write=False)
        memo[id(block)] = (point, value)
        return value

    def _flavor_log_weights(self, block: dict, led: dict, point: tuple) -> np.ndarray:
        delta = np.log(np.asarray(point)) - np.log(np.asarray(self.flavor_point))
        dk = led["keys"] @ delta
        z_to = np.bincount(led["gv"], weights=led["gw"] * np.exp(dk)[led["g_idx"]], minlength=led["n_vertices"])[: led["n_vertices"]]
        per_vertex = dk[led["c_idx"]] + led["log_z_from"] - np.log(z_to)
        return np.bincount(led["v_event"], weights=per_vertex, minlength=block["n"])[: block["n"]]

    # -- weights and moments ------------------------------------------
    def log_weights(self, cls: str, block: dict, eps: float, b_sd: float, eps_nd: float, remnant=None, flavor=None,
                    iso=None, b_iv=None, eps_central=None, f_ivf=None, f_tail=None) -> np.ndarray:
        """The per-event log importance weights from the generation shape
        point; both non-diffractive variants carry the nd mass law (nd0
        without its central term) and the exact remnant kind-pair ratio;
        *iso* reweights the exchange class from the generation fraction."""
        eps0, b0, eps0_nd = self.point[:3]
        nd_like = cls in ND_CLASSES
        eps_c, eps0_c = (float(eps_nd), eps0_nd) if nd_like else (float(eps), eps0)
        law = "nd" if nd_like else cls
        # the power central law of the central class (2026-10-07, R81)
        central = ({"central_law": "power", "eps_c": float(eps_central), "eps_c0": float(self.point[shape_pos(EPSC_NAME)])}
                   if (CENTRAL_POWER and cls == "nd") else {})
        # the intact proton's vertex factor of the single-diffraction classes (2026-10-15, R97)
        intact = ({"intact_factor": (float(f_ivf), float(self.point[shape_pos(IVF_NAME)]))}
                  if (INTACT_FREE and f_ivf is not None and cls in ("sd_projectile", "sd_target")) else {})
        if ((eps_c, float(b_sd)) == (eps0_c, b0) and (not central or central["eps_c"] == central["eps_c0"])
                and (not intact or intact["intact_factor"][0] == intact["intact_factor"][1])):
            log_w = np.zeros(block["n"])
        else:
            # through the class: callers pass a stand-in self (tests/test_intact_vertex_factor.py)
            log_w = Templates._mass_law(self, "main", cls, law, block, eps_c, b_sd, eps0_c, b0, central, intact)
        if b_iv is not None and block.get("iv") is not None and np.any(block["iv"]):
            # the isovector events carry their own transfer slope
            b_iv0 = self.point[shape_pos(BIV_NAME)]
            if ((eps_c, float(b_iv)) == (eps0_c, b_iv0) and (not central or central["eps_c"] == central["eps_c0"])
                    and (not intact or intact["intact_factor"][0] == intact["intact_factor"][1])):
                log_iv = np.zeros(block["n"])
            else:
                log_iv = Templates._mass_law(self, "iv", cls, law, block, eps_c, float(b_iv), eps0_c, b_iv0, central, intact)
            log_w = np.where(block["iv"], log_iv, log_w)
        if TAIL_FREE and f_tail is not None and cls in ("sd_projectile", "sd_target"):
            # the leading-proton tail (2026-10-23, R112): its events keep their own fixed law (no mass-law term) and
            # carry the share ratio s / s0, the coherence-region events (1 - s) / (1 - s0)
            tail = np.asarray(block["shape"]["tail"]) > 0.5
            if OPE_FREE:
                # the one-pion exchange beside the tail (2026-10-26): its events keep their fixed law and carry s_pi / s_pi0,
                # the coherence-region events (1 - s - s_pi) / (1 - s0 - s_pi0)
                ope = np.asarray(block["shape"]["ope"]) > 0.5
                s, s_o = float(f_tail[0]), float(f_tail[1])
                s0, s_o0 = float(self.point[shape_pos(TAIL_NAME)]), float(self.point[shape_pos(OPE_NAME)])
                log_tail = math.log(s / s0) if s > 0.0 else -math.inf
                log_ope = math.log(s_o / s_o0) if s_o > 0.0 else -math.inf
                log_w = np.where(tail, log_tail, np.where(ope, log_ope, log_w + math.log((1.0 - s - s_o) / (1.0 - s0 - s_o0))))
            else:
                s, s0 = float(f_tail), float(self.point[shape_pos(TAIL_NAME)])
                log_tail = math.log(s / s0) if s > 0.0 else -math.inf
                log_w = np.where(tail, log_tail, log_w + math.log((1.0 - s) / (1.0 - s0)))
        if nd_like and remnant is not None:
            ratio = remnant_log_ratio(remnant, [ND_GENERATION_REMNANT_WEIGHTS[k] for k in REMNANT_KINDS])
            kinds = block["rem_kind"].astype(np.int64)
            log_w = log_w + ratio[kinds[:, 0]] + ratio[kinds[:, 1]]
        if flavor is not None:
            log_w = log_w + self.flavor_log_weights(block, flavor)
        if SPLIT_FREE and f_tail is not None:
            # the leading splits (2026-11-03): every class holds baryonic roots
            trial = tuple(float(v) for v in f_tail[2:])
            generation = tuple(float(self.point[shape_pos(n)]) for n in SPLIT_FREE)
            if trial != generation:
                log_w = log_w + SPLIT_RW.split_log_ratio(block["shape"], trial, generation)
        if iso is not None and block.get("iv") is not None:
            f, f0 = float(iso), float(ISOVECTOR_GEN)
            with np.errstate(divide="ignore"):
                ratio = np.log(np.asarray([(1.0 - f) / (1.0 - f0), f / f0]))
            log_w = log_w + ratio[block["iv"].astype(np.int64)]
        return log_w

    def _mass_law(self, tag: str, cls: str, law: str, block: dict, eps_c: float, b: float, eps0_c: float, b0: float, central: dict,
                  intact: dict) -> np.ndarray:
        """``log_density_ratio`` of one block, with the last value per block,
        class and tag kept (2026-10-20, the fit profile: a Powell line search
        along a flavor direction repeats the shape point; the array is
        read-only, the same floats as a recomputation)."""
        exact = EXACT_RANGE and cls != "nd"
        # the coherence edge of the diffractive classes (2026-10-23)
        xi = ({"xi_max": float(TOPOLOGY_OPTIONS["diffractive_xi_max"])}
              if ("diffractive_xi_max" in TOPOLOGY_OPTIONS and cls in ("sd_projectile", "sd_target", "dd")) else {})
        if T_BREAK is not None and cls in ("sd_projectile", "sd_target"):
            # the intact proton's two-slope transfer (2026-10-24)
            xi = {**xi, "t_break": tuple(T_BREAK)}
        if "diffractive_pomeron_slope" in TOPOLOGY_OPTIONS and cls in ("sd_projectile", "sd_target", "dd"):
            # the Pomeron's shrinkage (2026-11-01): (alpha', the vertex exponent, the intact vertex factor when it is not
            # reweighted: the generation's free value or the fixed option)
            f_fix = (float(self.point[shape_pos(IVF_NAME)]) if INTACT_FREE
                     else TOPOLOGY_OPTIONS.get("intact_vertex_factor"))
            xi = {**xi, "pomeron": (float(TOPOLOGY_OPTIONS["diffractive_pomeron_slope"]),
                                    TOPOLOGY_OPTIONS.get("transfer_slope_mass_exponent"), f_fix,
                                    TOPOLOGY_OPTIONS.get("diffractive_pomeron_linear_t"))}
        args = (law, float(eps_c), float(b), float(eps0_c), float(b0), bool(exact), tuple(sorted(central.items())),
                tuple(sorted((k, tuple(v)) for k, v in intact.items())), tuple(xi.items()))
        memo = self.__dict__.setdefault("_mass_law_memo", {})
        key = (tag, cls, id(block))
        hit = memo.get(key)
        if hit is not None and hit[0] == args:
            return hit[1]
        value = log_density_ratio(law, block["shape"], eps_c, b, eps0=eps0_c, b_sd0=b0, sqrt_s=SQRT_S, exact_range=exact,
                                  **central, **intact, **xi)
        value.setflags(write=False)
        memo[key] = (args, value)
        return value

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


def free_shape_names(iv_fixed=None) -> tuple:
    """The free shape parameters; a fixed isovector fraction (the
    vacuum-only refit) drops f_isovector."""
    return SHAPE_NAMES if iv_fixed is None else tuple(n for n in SHAPE_NAMES if n not in (ISO_NAME, BIV_NAME))


def blocks_on_bound(name: str, values) -> int:
    """The replica-block minima of a parameter within 1e-3 of one of its
    bounds (in log for a log parameter, as the fit's bound check reads
    it).  2026-10-27, R122: all 32 block fits of the one-pion-exchange
    share ended at its bound 0.35, the single-block objective biased along
    the share by the block's own Monte Carlo variance, and the spread of
    zero read as stable."""
    spec_ = PARAM_INDEX[name]
    if spec_.transform == "log":
        return sum(1 for v in values if min(abs(math.log(v) - math.log(b)) for b in spec_.bounds) < 1e-3)
    return sum(1 for v in values if min(abs(v - b) for b in spec_.bounds) < 1e-3)


def iso_of(phys: dict):
    """The isovector fraction of a physical point (None without the class)."""
    return None if ISOVECTOR_GEN is None else float(phys[ISO_NAME])


def biv_of(phys: dict):
    """The isovector transfer slope of a physical point (None when shared)."""
    return float(phys[BIV_NAME]) if ISOVECTOR_SLOPE else None


def epsc_of(phys: dict):
    """The exponent of the power central mass law (None for the committed law)."""
    return float(phys[EPSC_NAME]) if CENTRAL_POWER else None


def tail_of(phys: dict):
    """The tail share of the single-diffraction classes, or None; with the free one-pion exchange
    (2026-10-26) the pair (tail share, one-pion-exchange share)."""
    if not TAIL_FREE:
        return None
    if SPLIT_FREE:
        # the free split parameters ride on the share argument (2026-11-03): (s_tail, s_pi, F_B, F_M, kappa_B, b_M)
        return (float(phys[TAIL_NAME]), float(phys[OPE_NAME])) + tuple(float(phys[n]) for n in SPLIT_FREE)
    return (float(phys[TAIL_NAME]), float(phys[OPE_NAME])) if OPE_FREE else float(phys[TAIL_NAME])


def _share_key(f_tail):
    """The cache key of a share argument: the tail share or the (tail, one-pion-exchange) pair."""
    return tuple(round(float(v), 12) for v in f_tail) if isinstance(f_tail, tuple) else round(float(f_tail), 12)


def ivf_of(phys: dict):
    """The intact proton's vertex factor of a physical point (None when it is no free parameter)."""
    return float(phys[IVF_NAME]) if INTACT_FREE else None


def _fc(fc_fixed):
    """The f_central of a fit: the caller's value, else the run default
    (None, free; 0.0 in the two-component mode)."""
    return FC_DEFAULT if fc_fixed is None else fc_fixed


def param_names(fc_fixed, iv_fixed=None) -> tuple:
    return free_shape_names(iv_fixed) + (() if _fc(fc_fixed) is not None else ("f_central",))


def _to_x(p: ParamSpec, value) -> float:
    if p.transform == "log":
        return math.log(float(value))
    if p.transform == "logit":
        w = min(max(float(value), 1e-12), 1.0 - 1e-12)
        return math.log(w / (1.0 - w))
    return float(value)


def remnant_log_ratio(remnant, generation) -> np.ndarray:
    """The per-kind log ratio of the non-diffractive remnant weights to the
    generation weights (intact, low-mass, excited).  The intact kind and a
    kind the generation never drew (weight 0, as the low-mass kind when the
    spec fixes w_excited at 1; 2026-10-13) carry -inf: no event of theirs
    exists."""
    trial = np.asarray(remnant, dtype=float)
    gen = np.asarray(generation, dtype=float)
    ratio = np.full(3, -np.inf)
    drawn = gen[1:] > 0.0
    ratio[1:][drawn] = np.log(trial[1:][drawn] / gen[1:][drawn])
    return ratio


def x_from_physical(phys: dict, fc_fixed=None, iv_fixed=None) -> np.ndarray:
    """The optimizer vector: the free shape parameters in registry order,
    each in its transform, and f_central when it is free."""
    core = [_to_x(PARAM_INDEX[n], phys[n]) for n in free_shape_names(iv_fixed)]
    if _fc(fc_fixed) is None:
        core.append(float(phys["f_central"]))
    return np.asarray(core, dtype=float)


def sd_window_fraction(epsilon: float, xi_max: float, b_sd: "float | None" = None, f_ivf: "float | None" = None) -> float:
    """The share of the single-diffraction class with M^2/s < *xi_max*: the
    class's marginal (M^2)^(-1-epsilon) on its window [EXCITATION_M_MIN^2,
    (sqrt s - m_p)^2] (topology_mass_laws.sd_limits, the proton line; the
    transfer factor is normalized per mass and drops out), the upper edge at
    the coherence edge of the topology option diffractive_xi_max when the
    spec sets it (2026-10-23).  Under the topology option
    diffractive_pomeron_slope (2026-11-01) the marginal carries the
    shrinkage factor Z_{B+c} / Z_B of the intact proton's vertex law, which
    depends on the vertex slope *b_sd* and the intact vertex factor *f_ivf*
    (topology_mass_laws.log_sd_pomeron_mass_norm)."""
    lo, hi = sd_limits(MASS_LAW_PROTON, SQRT_S)
    if "diffractive_xi_max" in TOPOLOGY_OPTIONS:
        hi = min(hi, float(TOPOLOGY_OPTIONS["diffractive_xi_max"]) * SQRT_S * SQRT_S)
    cut = float(xi_max) * SQRT_S * SQRT_S
    if not lo < cut <= hi:
        raise ValueError(f"sd_window_fraction: the window edge {cut!r} GeV^2 lies outside the class window ({lo!r}, {hi!r})")
    if cut == hi:
        return 1.0
    if "diffractive_pomeron_slope" in TOPOLOGY_OPTIONS:
        if b_sd is None:
            raise ValueError("sd_window_fraction: the Pomeron's shrinkage needs the vertex slope b_sd")
        kappa = TOPOLOGY_OPTIONS.get("transfer_slope_mass_exponent")
        f = f_ivf if f_ivf is not None else TOPOLOGY_OPTIONS.get("intact_vertex_factor")
        f_term = 0.5 * (float(f) - 1.0) if (kappa is not None and f is not None) else 0.0
        args = (float(epsilon), float(b_sd), kappa, f_term, float(TOPOLOGY_OPTIONS["diffractive_pomeron_slope"]),
                None if T_BREAK is None else tuple(T_BREAK))
        t_lin = TOPOLOGY_OPTIONS.get("diffractive_pomeron_linear_t")
        return math.exp(log_sd_pomeron_mass_norm(*args, lo, cut, MASS_LAW_PROTON, t_lin=t_lin)
                        - log_sd_pomeron_mass_norm(*args, lo, hi, MASS_LAW_PROTON, t_lin=t_lin))
    return math.exp(log_power_integral(epsilon, lo, cut) - log_power_integral(epsilon, lo, hi))


def neutral_ope_window_fraction(ope_law: tuple, window: float) -> float:
    """The share of the one-pion exchange that leaves an intact proton below M^2/s = *window* (2026-11-01, review R6):
    the neutral fraction of the law (topology_single_diffraction._ope_neutral_fraction) times the neutral law's share
    of its excited masses below the edge, on the window the generator draws (the p pi0 threshold or xi_lo up to xi_hi,
    the proton remnant).  Zero for the charged exchange alone."""
    from na49_mixing.one_pion_exchange import ope_component_of, ope_marginal_cells, ope_parts
    from na49_mixing.topology_mass_laws import EXCITATION_M_MIN
    from na49_mixing.topology_single_diffraction import SQRT_S as SD_SQRT_S, _ope_neutral_fraction
    *vals, law = ope_component_of((0.5,) + tuple(ope_law))
    if law.pion != "isospin":
        return 0.0
    f0 = _ope_neutral_fraction(tuple(vals), law)
    s = float(SD_SQRT_S) ** 2
    lo = max(float(vals[1]) * s, EXCITATION_M_MIN ** 2)
    hi = float(vals[2]) * s
    ys, cell = ope_marginal_cells(lo, hi, MASS_LAW_PROTON ** 2, s, ope_parts(law)[1])
    cdf = np.concatenate([[0.0], np.cumsum(cell)]) / float(np.sum(cell))
    return float(f0 * np.interp(math.log(float(window) * s), ys, cdf))


def sd_class_share(anchors_: dict, epsilon: float, b_sd: "float | None" = None, f_ivf: "float | None" = None) -> float:
    """The single-diffraction class share of the inelastic cross section:
    the anchor f_sd_total for the whole class, or under the window
    (2026-10-23) the measured window share over the class's window fraction
    at *epsilon* (and, under the Pomeron's shrinkage, at the vertex slope
    *b_sd* and the intact vertex factor *f_ivf*)."""
    if WINDOW_NAME not in anchors_:
        return float(anchors_["f_sd_total"]["value"])
    w = anchors_[WINDOW_NAME]
    return float(w["value"]) / sd_window_fraction(epsilon, w["xi_max"], b_sd, f_ivf)


def physical_from_x(x, fc_fixed, anchors_: dict, iv_fixed=None) -> dict:
    x = np.asarray(x, dtype=float)
    fc_fixed = _fc(fc_fixed)
    values = dict(FIXED_VALUES)
    names = free_shape_names(iv_fixed)
    for i, n in enumerate(names):
        p, v = PARAM_INDEX[n], float(x[i])
        values[n] = float(math.exp(v)) if p.transform == "log" else (1.0 / (1.0 + math.exp(-v)) if p.transform == "logit" else v)
    if iv_fixed is not None:
        values[ISO_NAME] = float(iv_fixed)
        if ISOVECTOR_SLOPE:
            values[BIV_NAME] = float(GEN0_SHAPE[shape_pos(BIV_NAME)])  # no isovector event: the slope is inert
    f_central = float(fc_fixed) if fc_fixed is not None else float(x[len(names)])
    a_sd = sd_class_share(anchors_, values["epsilon"], values["b_sd"], values.get(IVF_NAME))
    a_dd = float(anchors_["f_dd"]["value"])
    if TAIL_FREE:
        # the anchors constrain the coherence-region part; the tail adds the share s on top (2026-10-23, R112)
        s_rest = 1.0 - float(values[TAIL_NAME]) - (float(values[OPE_NAME]) if OPE_FREE else 0.0)
        if SD_WINDOW_INTACT and OPE_FREE:
            # every intact proton below the edge (2026-11-01, review R6): the coherence component's window share and the
            # neutral pion exchange's, a_sd [s_rest F + s_pi n_0] = the measured window share
            w_share = float(anchors_[WINDOW_NAME]["value"])
            frac = w_share / a_sd
            a_sd = w_share / (s_rest * frac + float(values[OPE_NAME]) * NEUTRAL_OPE_WINDOW)
        else:
            a_sd = a_sd / s_rest
    w_exc = values["w_excited"]
    flavor = {n: values[n] for n in FLAVOR_FREE}
    return {"f_sd_total": a_sd, "f_dd": a_dd, "f_nd": 1.0 - a_sd - a_dd, "f_central": f_central,
            "epsilon": values["epsilon"], "b_sd": values["b_sd"], "w_intact": 0.0, "w_low_mass": float(1.0 - w_exc),
            "w_excited": float(w_exc), "epsilon_nd": values["epsilon_nd"], "r_v": R_V_FIXED, **flavor,
            **({ISO_NAME: float(values[ISO_NAME])} if ISOVECTOR_GEN is not None else {}),
            **({BIV_NAME: float(values[BIV_NAME])} if ISOVECTOR_SLOPE else {}),
            **({EPSC_NAME: float(values[EPSC_NAME])} if CENTRAL_POWER else {}),
            **({IVF_NAME: float(values[IVF_NAME])} if INTACT_FREE else {}),
            **({TAIL_NAME: float(values[TAIL_NAME])} if TAIL_FREE else {}),
            **({OPE_NAME: float(values[OPE_NAME])} if OPE_FREE else {}),
            **{n: float(values[n]) for n in SPLIT_FREE},
            **({WINDOW_NAME: float(anchors_[WINDOW_NAME]["value"])} if WINDOW_NAME in anchors_ else {})}


def class_fractions(phys: dict) -> dict:
    """The template-class fractions: the anchors, with the non-diffractive
    share split by f_central between nd (with the cluster) and nd0."""
    fc = float(phys["f_central"])
    return {"sd_projectile": phys["f_sd_total"] / 2.0, "sd_target": phys["f_sd_total"] / 2.0, "dd": phys["f_dd"],
            "nd": phys["f_nd"] * fc, "nd0": phys["f_nd"] * (1.0 - fc)}


def bounds_for(fc_fixed, iv_fixed=None) -> list:
    core = []
    for n in free_shape_names(iv_fixed):
        p = PARAM_INDEX[n]
        core.append((math.log(p.bounds[0]), math.log(p.bounds[1])) if p.transform == "log" else tuple(p.bounds))
    return core + ([FC_BOUNDS] if _fc(fc_fixed) is None else [])


def _in_bounds(x, fc_fixed, iv_fixed=None) -> bool:
    return all(lo - 1e-12 <= float(v) <= hi + 1e-12 for v, (lo, hi) in zip(np.asarray(x, dtype=float), bounds_for(fc_fixed, iv_fixed)))


def parse_correlated_errors(block, neutron: bool):
    """The spec's ``fit.correlated_errors`` block (2026-10-28), validated:
    ``normalization`` (the per-species scale prior, a fraction),
    ``kaon_identification`` ({k_plus, k_minus, xf_min}: the identification
    scale priors of the kaons above xf_min), ``neutron_scale`` (the neutron
    table's common scale prior) and ``neutron_floor`` (the floor of the
    per-point remainder, a fraction of the value); ``None`` is the committed
    objective.  A malformed block raises ``ValueError``."""
    if block is None:
        return None
    if not isinstance(block, dict) or set(block) - CORRELATED_ERROR_KEYS or not block:
        raise ValueError(f"run spec: fit.correlated_errors takes the keys {sorted(CORRELATED_ERROR_KEYS)}, got {block!r}")
    def frac(name, value, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0 \
                or (positive and not value > 0.0) or value >= 1.0:
            raise ValueError(f"run spec: fit.correlated_errors.{name} must be a fraction in [0, 1), got {value!r}")
        return float(value)
    out = {}
    if "normalization" in block:
        out["normalization"] = frac("normalization", block["normalization"])
    if "kaon_identification" in block:
        kid = block["kaon_identification"]
        if isinstance(kid, dict) and set(kid) == {"k_plus", "k_minus", "xf_min"}:
            # the R125 form: one scale per kaon charge on the bins above xf_min, the prior a fraction
            out["kaon_identification"] = {"k_plus": frac("kaon_identification.k_plus", kid["k_plus"]),
                                          "k_minus": frac("kaon_identification.k_minus", kid["k_minus"]),
                                          "xf_min": frac("kaon_identification.xf_min", kid["xf_min"])}
        elif isinstance(kid, dict) and set(kid) == {"k_plus", "k_minus"}:
            # the knot form (2026-10-28, the kaon paper's Table 1 and Sect. 4.3): per kaon charge the systematic's
            # amplitude as [x_F, fraction] knots, zero below the first knot, linear between knots, held beyond the
            # last; one correlated scale per charge with the prior 1 multiplies the amplitude
            knots = {}
            for name in ("k_plus", "k_minus"):
                pts = kid[name]
                if (not isinstance(pts, (list, tuple)) or len(pts) < 2
                        or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)):
                    raise ValueError(f"run spec: fit.correlated_errors.kaon_identification.{name} takes at least two "
                                     f"[x_F, fraction] knots, got {pts!r}")
                xs = [frac(f"kaon_identification.{name}.x_F", p[0], positive=True) for p in pts]
                fs = [frac(f"kaon_identification.{name}.fraction", p[1]) for p in pts]
                if any(b <= a for a, b in zip(xs, xs[1:])) or not any(f > 0.0 for f in fs):
                    raise ValueError(f"run spec: fit.correlated_errors.kaon_identification.{name} knots need increasing "
                                     f"x_F and a positive fraction, got {pts!r}")
                knots[name] = [[x, f] for x, f in zip(xs, fs)]
            out["kaon_identification"] = knots
        else:
            raise ValueError(f"run spec: fit.correlated_errors.kaon_identification takes k_plus, k_minus and xf_min, "
                             f"or k_plus and k_minus as [x_F, fraction] knots, got {kid!r}")
    if ("neutron_scale" in block) != ("neutron_floor" in block):
        raise ValueError("run spec: fit.correlated_errors.neutron_scale and neutron_floor go together")
    if "neutron_scale" in block:
        if not neutron:
            raise ValueError("run spec: fit.correlated_errors.neutron_scale needs fit.neutron_segment")
        out["neutron_scale"] = frac("neutron_scale", block["neutron_scale"], positive=True)
        out["neutron_floor"] = frac("neutron_floor", block["neutron_floor"], positive=True)
    return out


def identification_amplitude(xf: np.ndarray, knots) -> np.ndarray:
    """The kaon identification systematic's amplitude per bin from its
    [x_F, fraction] knots (the knot form of ``kaon_identification``): zero
    below the first knot, linear between knots, the last fraction beyond
    the last knot."""
    xs = np.asarray([k[0] for k in knots], dtype=float); fs = np.asarray([k[1] for k in knots], dtype=float)
    xf = np.asarray(xf, dtype=float)
    out = np.interp(xf, xs, fs, left=0.0, right=float(fs[-1]))
    out[xf < xs[0] - 1e-9] = 0.0
    return out


def neutron_remainder(sigma: np.ndarray, data: np.ndarray, scale: float, floor: float) -> np.ndarray:
    """The per-point error of the neutron table once its common scale
    (a fraction *scale* of the value) is a nuisance: the quadratic
    remainder, floored at *floor* of the value."""
    sigma = np.asarray(sigma, dtype=float); data = np.asarray(data, dtype=float)
    rem2 = sigma ** 2 - (float(scale) * data) ** 2
    return np.sqrt(np.maximum(rem2, (float(floor) * data) ** 2))


def profile_scales(y: np.ndarray, design: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The scale nuisances profiled in closed form: minimize
    ``|y - design eta|^2 + sum (eta_k / prior_k)^2`` over eta.  Returns
    ``(eta, residual, penalty)`` with ``residual = y - design eta`` and
    ``penalty = sum (eta_k / prior_k)^2``; the objective's data term is
    ``residual . residual + penalty``."""
    design = np.asarray(design, dtype=float); priors = np.asarray(priors, dtype=float)
    normal = design.T @ design + np.diag(1.0 / priors ** 2)
    eta = np.linalg.solve(normal, design.T @ np.asarray(y, dtype=float))
    residual = y - design @ eta
    return eta, residual, float(np.sum((eta / priors) ** 2))


class Objective:
    """The cached joint objective at one generation.  ``fc_fixed`` holds
    f_central fixed (the null fit at 0, the profile scan) or leaves it
    free (None); ``discrepancy`` is the fractional model-discrepancy scale
    per bin; ``norm`` the fixed shared normalization offset eta."""

    def __init__(self, templates: Templates, anchors_: dict, fc_fixed=None, discrepancy: float = DISCREPANCY,
                 norm: float = NORM_FIXED, iv_fixed=None):
        self.t = templates
        self.anchors = anchors_
        fc_fixed = _fc(fc_fixed)
        self.fc_fixed = None if fc_fixed is None else float(fc_fixed)
        self.iv_fixed = None if iv_fixed is None else float(iv_fixed)
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
        # the divisor of each segment: the invariant cross section of a
        # species table, the x_F bin width of the neutron dN/dx_F
        self.divisors = {name: species_divisor(tab[name]) for name in FIT_SPECIES}
        if NEUTRON_SEGMENT:
            n_data, n_sigma = neutron_table()
            parts_d.append(n_data)
            if CORRELATED_ERRORS and "neutron_scale" in CORRELATED_ERRORS:
                # the table's errors are its systematics (EPJC 65 (2010) 9, Table 11): the common scale is a
                # nuisance below, the per-point remainder stays
                n_sigma_bins = neutron_remainder(n_sigma, n_data, CORRELATED_ERRORS["neutron_scale"],
                                                 CORRELATED_ERRORS["neutron_floor"])
            else:
                n_sigma_bins = n_sigma
            parts_v.append(n_sigma_bins ** 2 + (dz * n_data) ** 2)
            self.segments["neutron"] = (offset, offset + n_data.size)
            offset += n_data.size
            self.divisors["neutron"] = neutron_divisor(n_data, n_sigma)
        self.d = np.concatenate(parts_d)
        self.v_data = np.concatenate(parts_v)
        self.n_obs = int(self.d.size)
        # the correlated error model (2026-10-28): the scale nuisances as indicator columns over the observation vector
        self.nuisance_names, self.nuisance_priors, columns = [], [], []
        if CORRELATED_ERRORS:
            def group(name, prior, column):
                # column: the scale's amplitude per observation (an indicator, or the knot form's amplitude)
                if prior > 0.0 and np.any(column != 0.0):
                    self.nuisance_names.append(name); self.nuisance_priors.append(float(prior)); columns.append(np.asarray(column, dtype=float))
            norm_prior = CORRELATED_ERRORS.get("normalization", 0.0)
            kid = CORRELATED_ERRORS.get("kaon_identification")
            for name in FIT_SPECIES:
                lo, hi = self.segments[name]
                mask = np.zeros(self.n_obs, dtype=bool); mask[lo:hi] = True
                group(f"normalization_{name}", norm_prior, mask)
                if kid is not None and name in ("k_plus", "k_minus"):
                    t_ = tab[name]
                    xfc = np.asarray(t_.xf_centers, dtype=float)[np.asarray(t_.bin_index_xf)]
                    if "xf_min" in kid:
                        pid = np.zeros(self.n_obs, dtype=bool); pid[lo:hi] = xfc > float(kid["xf_min"]) + 1e-9
                        group(f"identification_{name}", kid[name], pid)
                    else:
                        # the knot form: the amplitude per bin, one scale of prior 1 (eta in units of the amplitude)
                        amp = np.zeros(self.n_obs, dtype=float); amp[lo:hi] = identification_amplitude(xfc, kid[name])
                        group(f"identification_{name}", 1.0, amp)
            if "neutron_scale" in CORRELATED_ERRORS and "neutron" in self.segments:
                lo, hi = self.segments["neutron"]
                mask = np.zeros(self.n_obs, dtype=bool); mask[lo:hi] = True
                group("scale_neutron", CORRELATED_ERRORS["neutron_scale"], mask)
        self.nuisance_design = np.column_stack(columns) if columns else None
        self.nuisance_priors = np.asarray(self.nuisance_priors, dtype=float)
        self.cache: dict = {}
        self.se_cache: dict = {}

    def physical(self, x) -> dict:
        return physical_from_x(x, self.fc_fixed, self.anchors, self.iv_fixed)

    def class_weights(self, eps: float, b_sd: float, eps_nd: float, blocks: list[int] | None = None, remnant=None,
                      flavor=None, iso=None, b_iv=None, eps_central=None, f_ivf=None, f_tail=None) -> tuple[dict, dict]:
        """Per class per block the normalized weights and the ESS fraction."""
        rkey = None if remnant is None else tuple(round(float(v), 12) for v in remnant)
        fkey = None if flavor is None else tuple(round(float(v), 12) for v in flavor)
        key = (round(eps, 12), round(b_sd, 12), round(eps_nd, 12), tuple(blocks) if blocks is not None else None, rkey, fkey)
        if iso is not None:
            key = key + (round(float(iso), 12),)
        if b_iv is not None:
            key = key + ("biv", round(float(b_iv), 12))
        if eps_central is not None:
            key = key + ("epsc", round(float(eps_central), 12))
        if f_ivf is not None:
            key = key + ("ivf", round(float(f_ivf), 12))
        if f_tail is not None:
            key = key + ("tail", _share_key(f_tail))
        if key in self.cache:
            return self.cache[key]
        weights = {}; ess = {}
        for cls in ACTIVE_CLASSES:
            weights[cls] = []; ess_vals = []
            for k, block in enumerate(self.t.by_class[cls]):
                if blocks is not None and k not in blocks:
                    weights[cls].append(None); continue
                log_w = self.t.log_weights(cls, block, eps, b_sd, eps_nd, remnant if cls in ND_CLASSES else None, flavor=flavor,
                                          iso=iso, b_iv=b_iv, eps_central=eps_central, f_ivf=f_ivf, f_tail=f_tail)
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
        for name, divisor in self.divisors.items():
            m = np.zeros(divisor.size)
            for cls in ACTIVE_CLASSES:
                m += frac[cls] * self.t.species_mean(cls, self.t.by_class[cls][k], weights[cls][k], name)
            out[name] = m / divisor
        return out

    def observation_vector(self, pred: dict) -> np.ndarray:
        return np.concatenate([pred[name] for name in SEGMENT_NAMES])

    def unweighted_se(self, frac: dict, block_ids: list[int]) -> np.ndarray:
        """The between-block standard error of the mixture at the trial
        class fractions with unit weights (cached)."""
        key = ("se",) + tuple(round(float(frac[c]), 12) for c in ACTIVE_CLASSES) + (tuple(block_ids),)
        if key in self.se_cache:
            return self.se_cache[key]
        if not hasattr(self, "_unit_weights"):
            self._unit_weights = {cls: [np.full(block["n"], 1.0 / block["n"]) for block in self.t.by_class[cls]] for cls in ACTIVE_CLASSES}
        vectors = np.asarray([self.observation_vector(self.block_prediction(k, frac, self._unit_weights)) for k in block_ids])
        se = np.std(vectors, axis=0, ddof=1) / math.sqrt(vectors.shape[0]) if vectors.shape[0] > 1 else np.zeros(vectors.shape[1])
        se = np.maximum(se, SE_FLOOR)
        self.se_cache[key] = se
        while len(self.se_cache) > SE_CACHE_ENTRIES:
            self.se_cache.pop(next(iter(self.se_cache)))
        return se

    @staticmethod
    def _physical_ok(phys: dict) -> bool:
        lo, hi = (math.exp(b) for b in FLAVOR_LOG_BOUNDS)
        return (EPS_ND_BOUNDS[0] <= phys["epsilon_nd"] <= EPS_ND_BOUNDS[1] and FC_BOUNDS[0] <= phys["f_central"] <= FC_BOUNDS[1]
                and all(lo * (1 - 1e-12) <= phys[n] <= hi * (1 + 1e-12) for n in FLAVOR_FREE)
                and (ISOVECTOR_GEN is None or 0.0 <= phys[ISO_NAME] <= 1.0))

    def evaluate(self, x, blocks: list[int] | None = None, *, details: bool = False):
        phys = self.physical(x)
        if not self._physical_ok(phys):
            return (float("inf"), None) if details else float("inf")
        weights, ess = self.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"], blocks,
                                          remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]), flavor=flavor_of(phys),
                                          iso=iso_of(phys), b_iv=biv_of(phys), eps_central=epsc_of(phys), f_ivf=ivf_of(phys),
                                          f_tail=tail_of(phys))
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
        w = np.sqrt(self.v_data + se ** 2)
        r = (self.d - m) / w
        nuisances = None
        m_scaled = m
        if self.nuisance_design is not None:
            # the correlated error model (2026-10-28): the scale nuisances profiled in closed form at this point
            design = self.nuisance_design * (m / w)[:, None]
            eta, r, penalty = profile_scales(r, design, self.nuisance_priors)
            m_scaled = m * (1.0 + self.nuisance_design @ eta)
            data_term = float(r @ r) + penalty
            nuisances = {name: {"value": float(e), "prior": float(p_), "pull": float(e / p_)}
                         for name, e, p_ in zip(self.nuisance_names, eta, self.nuisance_priors)}
            nuisances["penalty"] = penalty
        else:
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
            "m": m_scaled, "se": se, "residual_pull": r, "blocks": block_ids,
            **({"m_bare": m, "nuisances": nuisances} if nuisances is not None else {}),
            "in_trust_region": bool(ess_min >= ESS_FLOOR), "f_central_fixed": self.fc_fixed,
            **({"f_isovector_fixed": self.iv_fixed} if ISOVECTOR_GEN is not None else {}),
        }


# ----------------------------------------------------------------------
# The block-parallel objective (R51i)
# ----------------------------------------------------------------------
def _class_obs(objective: Objective, weights: dict) -> np.ndarray:
    """Per block and class the observation vector of the class's mean
    per-event content at the given weights: shape (blocks, classes, obs)."""
    t = objective.t
    out = np.zeros((t.n_blocks, len(ACTIVE_CLASSES), objective.n_obs))
    for k in range(t.n_blocks):
        for c, cls in enumerate(ACTIVE_CLASSES):
            block = t.by_class[cls][k]
            wn = weights[cls][k]
            pred = {name: t.species_mean(cls, block, wn, name) / divisor for name, divisor in objective.divisors.items()}
            out[k, c] = objective.observation_vector(pred)
    return out


def block_reply(objective: Objective, msg: tuple) -> tuple:
    """A block worker's answer to ("unit",) (the unit-weight observation
    vectors, for the between-block error) or ("eval", flavor point) (the
    reweighted vectors and the per-class smallest ESS at a shape point)."""
    if msg[0] == "unit":
        unit = {cls: [np.full(b["n"], 1.0 / b["n"]) for b in objective.t.by_class[cls]] for cls in ACTIVE_CLASSES}
        return ("unit", _class_obs(objective, unit))
    if msg[0] == "eval":
        _, eps, b_sd, eps_nd, remnant, flavor, iso, b_iv, eps_central, f_ivf, f_tail = msg
        weights, ess = objective.class_weights(eps, b_sd, eps_nd, remnant=remnant, flavor=flavor, iso=iso, b_iv=b_iv,
                                               eps_central=eps_central, f_ivf=f_ivf, f_tail=f_tail)
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

    def evaluate(self, eps: float, b_sd: float, eps_nd: float, remnant, flavor, iso=None, b_iv=None,
                 eps_central=None, f_ivf=None, f_tail=None) -> tuple[np.ndarray, dict]:
        replies = self._ask(("eval", float(eps), float(b_sd), float(eps_nd), tuple(float(v) for v in remnant),
                             tuple(float(v) for v in flavor), None if iso is None else float(iso),
                             None if b_iv is None else float(b_iv), None if eps_central is None else float(eps_central),
                             None if f_ivf is None else float(f_ivf),
                             None if f_tail is None else (tuple(float(v) for v in f_tail) if isinstance(f_tail, tuple) else float(f_tail))))
        obs = np.concatenate([r[1] for r in replies], axis=0)
        ess = {cls: min(float(r[2][cls]) for r in replies) for cls in ACTIVE_CLASSES}
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
                 norm: float = NORM_FIXED, iv_fixed=None):
        stub = types.SimpleNamespace(tables=tables, n_blocks=pool.n_blocks, point=pool.point)
        super().__init__(stub, anchors_, fc_fixed=fc_fixed, discrepancy=discrepancy, norm=norm, iv_fixed=iv_fixed)
        self.pool = pool
        self.obs_cache: dict = {}

    def class_obs(self, eps: float, b_sd: float, eps_nd: float, remnant, flavor, iso=None, b_iv=None,
                  eps_central=None, f_ivf=None, f_tail=None) -> tuple[np.ndarray, dict]:
        key = (round(eps, 12), round(b_sd, 12), round(eps_nd, 12), tuple(round(float(v), 12) for v in remnant),
               tuple(round(float(v), 12) for v in flavor))
        if iso is not None:
            key = key + (round(float(iso), 12),)
        if b_iv is not None:
            key = key + ("biv", round(float(b_iv), 12))
        if eps_central is not None:
            key = key + ("epsc", round(float(eps_central), 12))
        if f_ivf is not None:
            key = key + ("ivf", round(float(f_ivf), 12))
        if f_tail is not None:
            key = key + ("tail", _share_key(f_tail))
        if key in self.obs_cache:
            return self.obs_cache[key]
        value = self.pool.evaluate(eps, b_sd, eps_nd, remnant, flavor, iso, b_iv, eps_central, f_ivf, f_tail)
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
                                  (phys["w_intact"], phys["w_low_mass"], phys["w_excited"]), flavor_of(phys), iso_of(phys),
                                  biv_of(phys), epsc_of(phys), ivf_of(phys), tail_of(phys))
        fr = class_fractions(phys)
        frac = np.asarray([fr[c] for c in ACTIVE_CLASSES])
        vectors = np.einsum("c,kco->ko", frac, obs)
        n = vectors.shape[0]
        scale = 1.0 + self.norm
        m = np.mean(vectors, axis=0) * scale
        unit = np.einsum("c,kco->ko", frac, self.pool.unit())
        se = np.std(unit, axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(unit.shape[1])
        se = np.maximum(se, SE_FLOOR) * scale
        return self._likelihood(phys, m, se, ess, list(range(n)), details)


def dispersed_starts(fc_fixed=None, iv_fixed=None) -> list[np.ndarray]:
    """The dispersed starts of the registry (the first start is the
    generation-0 point), with f_central when it is free; a logit
    coordinate is kept 0.5 inside its bounds."""
    n = N_STARTS if (fc_fixed is None and iv_fixed is None) else N_STARTS_NULL
    shape_names = free_shape_names(iv_fixed)
    names = shape_names + (("f_central",) if _fc(fc_fixed) is None else ())
    starts = []
    for i in range(n):
        phys = {}
        for name in names:
            p = PARAM_INDEX[name]
            v = p.starts[i]
            phys[name] = p.gen0 if v == "gen0" else float(v)
        x = x_from_physical(phys, fc_fixed, iv_fixed)
        for k, name in enumerate(shape_names):
            p = PARAM_INDEX[name]
            if p.transform == "logit":
                x[k] = float(np.clip(x[k], p.bounds[0] + 0.5, p.bounds[1] - 0.5))
        starts.append(x)
    return starts


def run_fit(objective: Objective, *, starts: list[np.ndarray], blocks: list[int] | None = None,
            maxfev: int = 4000) -> dict:
    from scipy.optimize import minimize
    bounds = bounds_for(objective.fc_fixed, objective.iv_fixed)
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
                discrepancy: float, pool_workers: int = BLOCK_WORKERS_PER_START, norm: float = NORM_FIXED, iv_fixed=None,
                anchor_override=None) -> dict:
    """One optimizer start in its own process with its own block pool, or
    one replica-block fit (``blocks = [k]``) that loads that block alone."""
    contract = _contract(Path(run_dir))
    tables = load_fit_tables()
    if blocks is None:
        with BlockPool(run_dir, floor_index, generation, pool_workers) as pool:
            objective = PooledObjective(tables, anchors(anchor_override), pool, fc_fixed=fc_fixed, discrepancy=discrepancy,
                                        norm=norm, iv_fixed=iv_fixed)
            return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)
    templates = Templates(contract, floor_index, generation, tables, replicas=blocks)
    objective = Objective(templates, anchors(anchor_override), fc_fixed=fc_fixed, discrepancy=discrepancy, norm=norm,
                          iv_fixed=iv_fixed)
    return run_fit(objective, starts=[np.asarray(x0, dtype=float)], maxfev=maxfev)


def _best_of(results: list) -> dict:
    records = [r["starts"][0] for r in results]
    best = min(results, key=lambda r: r["fun"])
    return {"x": np.asarray(best["x"], dtype=float), "fun": float(best["fun"]), "converged": bool(best["converged"]),
            "nfev": int(best["nfev"]), "message": best["message"], "starts": records}


def run_fit_parallel(contract: RC.RunContract, floor_index: int, generation: int, starts: list, *, fc_fixed=None,
                     discrepancy: float = DISCREPANCY, norm: float = NORM_FIXED, blocks=None, maxfev: int = 4000,
                     workers: int | None = None, pool_workers: int = BLOCK_WORKERS_PER_START, iv_fixed=None,
                     anchor_override=None) -> dict:
    fo = _fanout()
    if fo.mode == "gateway" and blocks is None:
        pool_workers = FANOUT_POOL_WORKERS
    tail = () if iv_fixed is None else (float(iv_fixed),)
    if anchor_override:
        # a class-fraction variant (2026-10-22): the override rides after iv_fixed (None when absent)
        tail = (None if iv_fixed is None else float(iv_fixed), {k: float(v) for k, v in anchor_override.items()})
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x0], blocks, int(maxfev), fc_fixed,
             float(discrepancy), int(pool_workers), float(norm)) + tail for x0 in starts]
    return _best_of(fo.map("fit", jobs, local_workers=workers or len(jobs)))


SCAN_BLOCK_WORKERS = 5


def run_scan_parallel(contract: RC.RunContract, floor_index: int, generation: int, starts: list, values: tuple, *,
                      discrepancy: float = DISCREPANCY, norm: float = NORM_FIXED, maxfev: int | None = None) -> list:
    """The profile scan: at each fixed f_central one fit from every start
    (R69: the free and the null optimum), all in parallel; the better fit
    of each point is kept."""
    maxfev = FIT_MAXFEV if maxfev is None else maxfev
    fo = _fanout()
    pool_workers = FANOUT_POOL_WORKERS if fo.mode == "gateway" else SCAN_BLOCK_WORKERS
    jobs = [(str(contract.run_dir), int(floor_index), int(generation), [float(v) for v in x0], None, int(maxfev), float(fc),
             float(discrepancy), pool_workers, float(norm)) for fc in values for x0 in starts]
    results = fo.map("fit", jobs, local_workers=len(jobs))
    k = len(starts)
    return [_best_of(results[i * k:(i + 1) * k]) for i in range(len(values))]


def _hessian_box(p: ParamSpec) -> tuple:
    if p.hessian_box is not None:
        return tuple(p.hessian_box)
    if p.transform == "log":
        return (math.exp(math.log(p.bounds[0])), math.exp(math.log(p.bounds[1])))
    if p.transform == "logit":
        return (1.0 / (1.0 + math.exp(-p.bounds[0])), 1.0 / (1.0 + math.exp(-p.bounds[1])))
    return tuple(p.bounds)


def hessian_shapes(objective: Objective, phys: dict) -> dict:
    """The finite-difference Hessian of the objective in the physical shape
    parameters at the objective's fixed f_central, inside the parameter box
    (run_contract.bounded_hessian: one-sided differences at a bound)."""
    names = SHAPE_NAMES
    p0 = np.asarray([phys[n] for n in names], dtype=float)
    steps = np.asarray([HESSIAN_STEPS[n] for n in names], dtype=float)
    box = {n: _hessian_box(PARAM_INDEX[n]) for n in names}

    def value(p):
        trial = dict(phys)
        trial.update({n: float(v) for n, v in zip(names, p)})
        for n in names:
            spec_ = PARAM_INDEX[n]
            if spec_.transform == "logit" and not (0.0 < trial[n] < 1.0):
                return float("inf")
            if spec_.transform == "log" and trial[n] <= 0.0:
                return float("inf")
        return objective.evaluate(x_from_physical(trial, objective.fc_fixed, objective.iv_fixed))

    rec = RC.bounded_hessian(value, p0, steps, [box[n][0] for n in names], [box[n][1] for n in names], list(names))
    return {"names": list(names), "steps": [float(s) for s in steps], "at_f_central": objective.fc_fixed,
            "matrix": rec.get("matrix"), "eigenvalues": rec.get("eigenvalues"), "positive_definite": bool(rec.get("positive_definite")),
            "quoted": bool(rec.get("positive_definite")), "errors": rec.get("errors"), "directions": rec.get("directions"),
            "bound_limited": rec.get("bound_limited"), "non_finite": rec.get("non_finite")}


def _mixture_readouts(templates: Templates, objective: Objective, tables: dict, phys: dict) -> dict:
    """The readouts at a fitted point: paper-window sum ratios, total yields
    and their class breakdown, the pi- p_T profile, the SD n_ch against M_X,
    the pi0 and neutron readouts, and predictions."""
    weights, _ = objective.class_weights(phys["epsilon"], phys["b_sd"], phys["epsilon_nd"],
                                         remnant=(phys["w_intact"], phys["w_low_mass"], phys["w_excited"]), flavor=flavor_of(phys),
                                         iso=iso_of(phys), b_iv=biv_of(phys), eps_central=epsc_of(phys), f_ivf=ivf_of(phys),
                                         f_tail=tail_of(phys))
    frac_map = class_fractions(phys)
    model_bins = {}
    readout = {}
    for name in TRACK_SPECIES:
        if name not in tables:
            continue
        table = tables[name]
        mb = np.zeros(table.data.size)
        for cls in ACTIVE_CLASSES:
            for k, block in enumerate(templates.by_class[cls]):
                mb += frac_map[cls] * templates.species_mean(cls, block, weights[cls][k], name) / templates.n_blocks
        mb = mb / species_divisor(table)
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
    for cls in ACTIVE_CLASSES:
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
    for cls in ACTIVE_CLASSES:
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
    for cls in ACTIVE_CLASSES:
        for k, block in enumerate(templates.by_class[cls]):
            wn = weights[cls][k] / templates.n_blocks
            predictions["lead_xf"] += frac_map[cls] * templates.histogram_mean(block, wn, "lead_xf", LEAD_XF_EDGES)
            predictions["gap"] += frac_map[cls] * templates.histogram_mean(block, wn, "gap", GAP_EDGES)
            predictions["n_ch"] += frac_map[cls] * templates.nch_mean(block, wn)
    # the pi0 yield and the neutron dN/dx_F (held out)
    n_pi0 = 0.0
    neutron = np.zeros(NEUTRON_XF_EDGES.size - 1)
    for cls in ACTIVE_CLASSES:
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


def central_absent(chi2: float, n_obs: int) -> dict:
    """The central-component record of a run without the central component:
    f_central = 0 by construction, no null fit and no test."""
    ndf = int(n_obs - N_FREE_PARAMETERS)
    return {"tested": False, "absent_by_construction": True,
            "rule": "central_component false in the run spec: f_central = 0 by construction, no class with the central source",
            "f_central_hat": 0.0, "chi2_free": float(chi2), "chi2_min": float(chi2), "free_fit_is_minimum": True,
            "chi2_null": float(chi2), "delta_chi2": 0.0, "ndf": ndf, "scale_factor_squared": float(chi2) / ndf,
            "threshold": None, "compatible_with_zero": None, "compatible_with_zero_raw": None, "scaled_interval": None,
            "profile": []}


def isovector_test(chi2_min: float, chi2_vacuum: float, n_obs: int, f_hat: float) -> dict:
    """The isovector test: Delta chi2 = chi2(f_isovector = 0) - chi2_min
    against 3.84 S^2 with S^2 = chi2_min / ndf (chi2_min the lowest of every
    fit of the generation); the isovector exchange is required when Delta
    chi2 exceeds the threshold."""
    ndf = int(n_obs - N_FREE_PARAMETERS)
    s2 = float(chi2_min) / ndf
    delta = float(chi2_vacuum) - float(chi2_min)
    threshold = TEST_DELTA_CHI2 * s2
    return {"f_isovector_hat": float(f_hat), "chi2_min": float(chi2_min), "chi2_vacuum": float(chi2_vacuum), "delta_chi2": delta,
            "ndf": ndf, "scale_factor_squared": s2, "threshold": threshold, "required": bool(delta > threshold),
            "required_raw": bool(delta > TEST_DELTA_CHI2)}


def regeneration_point_of(d_free: dict, d_null: dict) -> tuple[dict, str]:
    """The regeneration point from the free and the null fit (their details
    records): the midpoint of the shapes and the geometric mean of the
    flavor weights when both fits lie in the trust region; the point of the
    one that does otherwise ("free only", "null only"); the free fit's point
    when neither does ("free, neither in the trust region").  A fit in the
    trust-region penalty sits at an arbitrary point and must not move the
    generation.  The central mass exponent (``epsilon_c``) always takes the
    free fit's value: the null fit has no central events, so its value there
    is unidentified (2026-10-08)."""
    names = ("epsilon", "b_sd", "epsilon_nd") + EXTRA_SHAPE
    p_free, p_null = d_free["physical"], d_null["physical"]
    ok_free, ok_null = bool(d_free["in_trust_region"]), bool(d_null["in_trust_region"])
    if ok_free and ok_null:
        point = {**{n: (0.5 * (p_free[n] + p_null[n]) if n in SHAPE_NAMES else FIXED_VALUES[n]) for n in names},
                 **{n: math.sqrt(p_free[n] * p_null[n]) for n in FLAVOR_ORDER}}
        basis = "midpoint"
    else:
        if ok_null:
            source, basis = p_null, "null only"
        else:
            source, basis = p_free, ("free only" if ok_free else "free, neither in the trust region")
        point = {**{n: (source[n] if n in SHAPE_NAMES else FIXED_VALUES[n]) for n in names},
                 **{n: source[n] for n in FLAVOR_ORDER}}
    if EPSC_NAME in names and EPSC_NAME in SHAPE_NAMES:
        point[EPSC_NAME] = p_free[EPSC_NAME]
    return point, basis


def evaluate_generation(contract: RC.RunContract, floor_index: int, generation: int, tables: dict, anchors_: dict, *,
                        smoke: bool = False) -> dict:
    f, g = int(floor_index), int(generation)
    free_starts = dispersed_starts(None)[: (2 if smoke else N_STARTS)]
    master_pool = None
    if smoke:
        templates = Templates(contract, f, g, tables, smoke=True)
        free_fit = run_fit(Objective(templates, anchors_), starts=free_starts, maxfev=600)
    else:
        free_fit = run_fit_parallel(contract, f, g, free_starts, maxfev=FIT_MAXFEV)
    phys_free = physical_from_x(free_fit["x"], None, anchors_)
    null_starts = [x_from_physical(phys_free, 0.0)] + dispersed_starts(0.0)[1: (1 if smoke else N_STARTS_NULL)]
    scan_values = (0.5,) if smoke else FC_SCAN
    if not CENTRAL_COMPONENT:
        # f_central = 0 by construction: the free fit is the null fit
        null_fit, scan_values, scan_fits = free_fit, (), []
    elif smoke:
        null_fit = run_fit(Objective(templates, anchors_, fc_fixed=0.0), starts=null_starts, maxfev=600)
        scan_fits = [run_fit(Objective(templates, anchors_, fc_fixed=fc), starts=[x_from_physical(phys_free, fc), null_fit["x"]], maxfev=300)
                     for fc in scan_values]
    else:
        null_fit = run_fit_parallel(contract, f, g, null_starts, fc_fixed=0.0, maxfev=FIT_MAXFEV)
        scan_fits = run_scan_parallel(contract, f, g, [list(x_from_physical(phys_free, 0.0)), list(null_fit["x"])], scan_values)
    # the vacuum-only refit (f_isovector = 0, f_central free)
    iv_fit = None
    if ISOVECTOR_GEN is not None:
        iv_starts = [x_from_physical(phys_free, None, 0.0)] + dispersed_starts(None, 0.0)[1: (1 if smoke else N_STARTS_NULL)]
        if smoke:
            iv_fit = run_fit(Objective(templates, anchors_, iv_fixed=0.0), starts=iv_starts, maxfev=600)
        else:
            iv_fit = run_fit_parallel(contract, f, g, iv_starts, iv_fixed=0.0, maxfev=FIT_MAXFEV)
    if not smoke:
        master_pool = BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS)

    def objective_at(fc_fixed, iv_fixed=None):
        if smoke:
            return Objective(templates, anchors_, fc_fixed=fc_fixed, iv_fixed=iv_fixed)
        return PooledObjective(tables, anchors_, master_pool, fc_fixed=fc_fixed, iv_fixed=iv_fixed)

    value, det = objective_at(None).evaluate(free_fit["x"], details=True)
    phys = det["physical"]
    v_null, d_null = objective_at(0.0).evaluate(null_fit["x"], details=True)
    scan = []
    for fc, fit in zip(scan_values, scan_fits):
        v_s, d_s = objective_at(fc).evaluate(fit["x"], details=True)
        scan.append({"f_central": float(fc), "objective": v_s, "converged": fit["converged"], "ess_min": d_s["ess_min"],
                     "fitted": {n: d_s["physical"][n] for n in SHAPE_NAMES}, "chi2_by_segment": d_s["chi2_by_segment"]})
    test = (central_test(value, v_null, det["m"].size, phys["f_central"], scan) if CENTRAL_COMPONENT
            else central_absent(value, det["m"].size))
    iv_record = None
    if iv_fit is not None:
        v_iv, d_iv = objective_at(None, 0.0).evaluate(iv_fit["x"], details=True)
        iv_test = isovector_test(min(test["chi2_min"], v_iv), v_iv, det["m"].size, phys[ISO_NAME])
        iv_record = {"test": iv_test, "fitted": {n: d_iv["physical"][n] for n in param_names(None, 0.0)}, "objective": v_iv,
                     "chi2_by_segment": d_iv["chi2_by_segment"], "ess_min": d_iv["ess_min"], "converged": iv_fit["converged"],
                     "nfev": iv_fit["nfev"], "start_records": iv_fit["starts"], "physical": d_iv["physical"]}
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
        # a fraction the spec fixes is no fitted boundary (2026-10-13, R95:
        # the low-mass record kind dropped at w_excited = 1)
        if not PARAM_INDEX["w_excited"].free:
            return True
        return bool(FRACTION_INTERIOR[0] <= p["w_excited"] <= FRACTION_INTERIOR[1])

    def _on_bound(p):
        # the registry parameters with a bound check (the free diffractive
        # shapes, epsilon_nd and the free flavor weights in R71 to R74)
        for n in SHAPE_NAMES:
            spec_ = PARAM_INDEX[n]
            if not spec_.bound_check:
                continue
            if n == EPSC_NAME and float(p.get("f_central", 0.0)) == 0.0:
                # the central mass law has no events without the central
                # component (the null fit): unidentified there (2026-10-08)
                continue
            if spec_.transform == "log":
                if min(abs(math.log(p[n]) - math.log(b)) for b in spec_.bounds) < 1e-3:
                    return True
            elif min(abs(p[n] - b) for b in spec_.bounds) < 1e-3:
                return True
        return False

    # the interior and bound checks apply to the free and the null fit
    interior = _interior(phys) and _interior(d_null["physical"])
    on_exponent_bound = _on_bound(phys) or _on_bound(d_null["physical"])
    residuals = list(residuals_pre)
    for label, fit, d_ in (("free", free_fit, det), ("null", null_fit, d_null)):
        if not fit["converged"]:
            residuals.append({"kind": "optimizer_non_convergence", "fit": label, "message": fit["message"]})
    if iv_fit is not None:
        if not iv_fit["converged"]:
            residuals.append({"kind": "optimizer_non_convergence", "fit": "vacuum_only", "message": iv_fit["message"]})
        if iv_record["objective"] < test["chi2_min"] - 1e-6:
            residuals.append({"kind": "vacuum_only_fit_below_free_minimum", "chi2_vacuum": iv_record["objective"],
                              "chi2_min": test["chi2_min"]})
    # every fit with its own ESS (2026-10-07; review 2026-10-06, Muse H-2)
    for label, d_ in (("free", det), ("null", d_null)) + ((("vacuum_only", iv_record),) if iv_record is not None else ()):
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
        on_bound = blocks_on_bound(name, s["block_values"])
        if 2 * on_bound >= len(blocks):
            # the spread of minima piled at a bound tests nothing (R122); named, the verdict unchanged
            residuals.append({"kind": "replica_block_minima_on_bound", "parameter": name, "blocks_on_bound": on_bound,
                              "blocks": len(blocks)})
    ess_low = min(det["ess_min"], d_null["ess_min"])
    regenerate = (not smoke) and ess_low < ESS_INTERIOR and g < MAX_GENERATIONS - 1
    regeneration_point, regeneration_basis = regeneration_point_of(det, d_null)
    if regeneration_basis != "midpoint":
        residuals.append({"kind": "regeneration_fit_excluded", "basis": regeneration_basis,
                          "ess_min_free": det["ess_min"], "ess_min_null": d_null["ess_min"]})
    constrained = bool(free_fit["converged"] and null_fit["converged"] and ess_low >= ESS_FLOOR and interior and not on_exponent_bound
                       and hess.get("positive_definite") and stability and all(s["stable"] for s in stability.values()))
    if master_pool is not None:
        master_pool.close()
        # the readouts need every block in process once
        templates = Templates(contract, f, g, tables)
    readouts = _mixture_readouts(templates, Objective(templates, anchors_), tables, phys)
    null_readouts = (_mixture_readouts(templates, Objective(templates, anchors_, fc_fixed=0.0), tables, d_null["physical"])
                     if CENTRAL_COMPONENT else readouts)
    isovector_block = {}
    if iv_record is not None:
        iv_readouts = _mixture_readouts(templates, Objective(templates, anchors_, iv_fixed=0.0), tables, iv_record.pop("physical"))
        isovector_block = {
            "isovector_test": iv_record["test"],
            "vacuum_only_fit": {**{k: v for k, v in iv_record.items() if k != "test"},
                                "total_yield_ratios": iv_readouts["readout_total_yields"]["ratio_model_over_data"],
                                "neutron_dndxf": iv_readouts["readout_neutron_dndxf"],
                                "paper_window_sum_ratios": {k: v["sum_ratio"] for k, v in
                                                            iv_readouts["readout_paper_window_sum_ratios"].items()}},
            "isovector_generation_fraction": float(ISOVECTOR_GEN),
        }
    mesonic = mesonic_count_readout(FLOOR_GRID[f], SMOKE_MESONIC_DECAYS if smoke else MESONIC_DECAYS, flavor_of(phys))
    objective = Objective(templates, anchors_)
    return {
        "floor_index": f, "continuation_floor": float(FLOOR_GRID[f]), "generation": g,
        "generation_point": {**dict(zip(("epsilon", "b_sd", "epsilon_nd") + EXTRA_SHAPE, templates.point)),
                             **dict(zip(FLAVOR_ORDER, templates.flavor_point))},
        "fixed": {"continuation_floor": float(FLOOR_GRID[f]), "continuation_ceiling": CONTINUATION_CEILING,
                  **SPEC["report_fixed"], **({"r_v": R_V_FIXED} if R_V_FIXED is not None else {}),
                  "fractions": {k: anchors_[k]["value"] for k in FRACTION_NAMES + (WINDOW_NAME,) if k in anchors_},
                  "w_intact": 0.0},
        "statistics": {"classes": list(ACTIVE_CLASSES), "replicas": templates.n_blocks,
                       "events_per_slot": int(templates.by_class[CLASS_ORDER[0]][0]["n"]), "tier": "T2" if not smoke else "smoke",
                       "common_random_numbers": "nd0 replica k draws the seeds and event ids of nd replica k",
                       # recorded only when set, so a committed-mode report is unchanged
                       **({"central_component": False} if not CENTRAL_COMPONENT else {}),
                       **({"topology_options": dict(TOPOLOGY_OPTIONS)} if TOPOLOGY_OPTIONS else {})},
        "observable_set": {"segments": {k: [int(a), int(b)] for k, (a, b) in objective.segments.items()}, "total": objective.n_obs,
                           "fit_window": FIT_WINDOW, "paper_window": PAPER_WINDOW,
                           "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS)},
                           "discrepancy": DISCREPANCY, "anchors": anchors_},
        "parametrization": {"free": list(param_names(None)),
                            "null": ("f_central = 0; free: " + ", ".join(SHAPE_NAMES)) if CENTRAL_COMPONENT
                            else "none: f_central = 0 by construction (the free fit)",
                            "space": TEXTS["parametrization_space"]},
        "fitted": {**phys, "class_fractions": class_fractions(phys),
                   "remnant_weights": {"intact_proton": phys["w_intact"], "low_mass_baryonic_resonance": phys["w_low_mass"],
                                       "high_mass_baryonic_continuum": phys["w_excited"]}},
        "objective_at_optimum": {"value": value, "minus2lnl_data": det["minus2lnl_data"], "chi2_by_segment": det["chi2_by_segment"],
                                 "normalization": det["normalization"], "ess": det["ess"], "ess_min": det["ess_min"],
                                 "in_trust_region": det["in_trust_region"],
                                 **({"nuisances": det.get("nuisances"), "correlated_errors": CORRELATED_ERRORS}
                                    if CORRELATED_ERRORS else {})},
        "null_fit": {"fitted": {n: d_null["physical"][n] for n in SHAPE_NAMES + ("w_low_mass", "f_central")}, "objective": v_null,
                     "chi2_by_segment": d_null["chi2_by_segment"], "ess_min": d_null["ess_min"], "converged": null_fit["converged"],
                     "nfev": null_fit["nfev"], "start_records": null_fit["starts"],
                     "total_yield_ratios": null_readouts["readout_total_yields"]["ratio_model_over_data"],
                     "class_yields": null_readouts["readout_class_yields"], "pt_profile": null_readouts["readout_pt_profile"],
                     "paper_window_sum_ratios": {k: v["sum_ratio"] for k, v in null_readouts["readout_paper_window_sum_ratios"].items()}},
        "profile_scan": scan,
        "central_component_test": test,
        **isovector_block,
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
                    **({"m_bare": det["m_bare"].tolist()} if "m_bare" in det else {}),
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
    if rows["primary"].get("absent_by_construction"):
        summary = {"rule": rows["primary"]["rule"], "central_component": "absent by construction (f_central = 0)",
                   "outcomes": {}, "compatible_in_primary": None, "compatible_in_every_variant": None}
    else:
        summary = {"rule": "f_central compatible with zero when Delta chi2 = chi2(f_central = 0) - chi2_min <= 3.84 chi2_min / ndf",
                   "outcomes": {k: {"f_central_hat": v["f_central_hat"], "delta_chi2": v["delta_chi2"], "threshold": v["threshold"],
                                    "compatible_with_zero": v["compatible_with_zero"],
                                    "compatible_with_zero_raw": v["compatible_with_zero_raw"]}
                                for k, v in rows.items()},
                   "compatible_in_primary": bool(rows["primary"]["compatible_with_zero"]),
                   "compatible_in_every_variant": bool(all(v["compatible_with_zero"] for v in rows.values()))}
    if "isovector_test" in branch:
        iv_rows = {"primary": branch["isovector_test"]}
        for label, var in (variants or {}).items():
            if "isovector_test" in var:
                iv_rows[label] = var["isovector_test"]
        summary["isovector"] = {
            "rule": "the isovector exchange is required when Delta chi2 = chi2(f_isovector = 0) - chi2_min > 3.84 chi2_min / ndf",
            "outcomes": {k: {"f_isovector_hat": v["f_isovector_hat"], "delta_chi2": v["delta_chi2"], "threshold": v["threshold"],
                             "required": v["required"], "required_raw": v["required_raw"]} for k, v in iv_rows.items()},
            "required_in_primary": bool(iv_rows["primary"]["required"]),
            "required_in_every_variant": bool(all(v["required"] for v in iv_rows.values()))}
    return summary


def compose_report(branch: dict, variants: dict | None) -> dict:
    """The final report: the branch's fits and test, the variants, the
    test summary, and the verdict."""
    report = {k: v for k, v in branch.items() if k not in ("digest", "schema", "run", "card", "driver", "driver_sha256")}
    report["branch_digest"] = branch.get("digest")
    report["variants"] = variants
    report["test_summary"] = test_summary(branch, variants)
    report["verdict"] = {**branch["verdict"], "regeneration_requested": False}
    # the Hessian errors are Delta chi2 = 1 errors; at S^2 = chi2/ndf > 1 the
    # PDG-scaled errors multiply them by S (2026-10-07; review 2026-10-06,
    # Muse H-3)
    errors = (report.get("hessian") or {}).get("errors")
    s2 = (report.get("central_component_test") or {}).get("scale_factor_squared")
    if errors and s2:
        report["errors_scaled_by_s"] = {"s": math.sqrt(max(float(s2), 1.0)),
                                        "errors": {n: float(e) * math.sqrt(max(float(s2), 1.0)) for n, e in errors.items()}}
    return report


def render_report_md(report: dict) -> str:
    f = report["fitted"]; o = report["objective_at_optimum"]; v = report["verdict"]; nf = report["null_fit"]
    test = report["central_component_test"]
    errors = (report["hessian"] or {}).get("errors") or {}
    lines = [f"# {TEXTS['report_title']} ({RUN_ID})", "",
             f"Card {CARD}. Generation {report['generation']}; floor {report['continuation_floor']} GeV; {TEXTS['report_kernel_line']}",
             "Fractions at the external anchors; the non-diffractive share split by f_central between the class with the central cluster and the class without it.", "",
             "## The test", ""]
    if test.get("absent_by_construction"):
        lines += [f"The central component is absent by construction (f_central = 0). chi2_min {test['chi2_min']:.1f}; "
                  f"S^2 = chi2/ndf = {test['scale_factor_squared']:.2f} (ndf {test['ndf']}).", ""]
    else:
        lines += [
             f"f_central = {test['f_central_hat']:.4f}. chi2_min {test['chi2_min']:.1f} (free fit {test['chi2_free']:.1f}); with f_central = 0, {test['chi2_null']:.1f}.",
             f"Delta chi2 = {test['delta_chi2']:.1f}; S^2 = chi2/ndf = {test['scale_factor_squared']:.2f} (ndf {test['ndf']}); threshold 3.84 S^2 = {test['threshold']:.1f}.",
             f"Compatible with zero: {test['compatible_with_zero']} (raw threshold 3.84: {test['compatible_with_zero_raw']}).",
             f"Scaled interval of f_central: {json.dumps(test['scaled_interval'])}.", "",
             "| f_central | Delta chi2 |", "|---|---|"]
    for row in test["profile"]:
        lines.append(f"| {row['f_central']:.3f} | {row['delta_chi2']:.1f} |")
    if "isovector_test" in report:
        it = report["isovector_test"]; vo = report["vacuum_only_fit"]
        lines += ["", "## The isovector test", "",
                  f"f_isovector = {it['f_isovector_hat']:.4f} (generated at {report['isovector_generation_fraction']}). "
                  f"chi2_min {it['chi2_min']:.1f}; vacuum only (f_isovector = 0), {it['chi2_vacuum']:.1f}.",
                  f"Delta chi2 = {it['delta_chi2']:.1f}; threshold 3.84 S^2 = {it['threshold']:.1f}. Required: {it['required']} "
                  f"(raw threshold 3.84: {it['required_raw']})."]
    # without the central component the central null fit is the free fit
    # itself, so its columns are left out (2026-10-07; review 2026-10-06,
    # Muse M-7)
    two = bool(test.get("absent_by_construction"))
    iso_cols = "isovector_test" in report
    lines += ["", "| segment | chi2 |" + ("" if two else " chi2 null |") + (" chi2 vacuum only |" if iso_cols else ""),
              "|---|---|" + ("" if two else "---|") + ("---|" if iso_cols else "")]
    for name in SEGMENT_NAMES:
        extra = f" {report['vacuum_only_fit']['chi2_by_segment'][name]:.1f} |" if iso_cols else ""
        null = "" if two else f" {nf['chi2_by_segment'][name]:.1f} |"
        lines.append(f"| {name} | {o['chi2_by_segment'][name]:.1f} |" + null + extra)
    scaled = (report.get("errors_scaled_by_s") or {}).get("errors") or {}
    s_val = (report.get("errors_scaled_by_s") or {}).get("s")
    lines += ["", f"| parameter | fit | error | error x S (S = {s_val:.2f}) |" if s_val else "| parameter | fit | error | |",
              "|---|---|---|---|"] if two else ["", "| parameter | free fit | error | error x S | null fit |", "|---|---|---|---|---|"]
    for name in SHAPE_NAMES:
        err = errors.get(name); es = scaled.get(name)
        cells = f"| {name} | {f[name]:.5g} | {'' if err is None else f'{err:.3g}'} | {'' if es is None else f'{es:.3g}'} |"
        lines.append(cells if two else cells + f" {nf['fitted'][name]:.5g} |")
    ess_line = f"Smallest class ESS {o['ess_min']:.3f}" + ("" if two else f" (null {nf['ess_min']:.3f})")
    if iso_cols and report["vacuum_only_fit"].get("ess_min") is not None:
        ess_line += f"; vacuum-only fit {report['vacuum_only_fit']['ess_min']:.3f}"
    lines += ["", f"{ess_line}. Hessian positive definite: {report['hessian'].get('positive_definite')}.",
              f"Verdict (fit quality): {'CONSTRAINED' if v['constrained'] else 'NOT_CONSTRAINED'}.", "",
              "## Total yields per inelastic event (model / data)", "",
              "| species | model / data |" if two else "| species | free fit | null fit |", "|---|---|" if two else "|---|---|---|"]
    ty = report["readout_total_yields"]["ratio_model_over_data"]
    for name, ratio in ty.items():
        lines.append(f"| {name} | {ratio:.3f} |" if two else f"| {name} | {ratio:.3f} | {nf['total_yield_ratios'][name]:.3f} |")
    pt = report["readout_pt_profile"]
    lines += ["", f"pi- p_T profile ratio {pt['profile_ratio']:.2f}" + ("" if two else f" (null {nf['pt_profile']['profile_ratio']:.2f})") + ".", "",
              "## Variants", "", "| variant | f_central | Delta chi2 | threshold | compatible |", "|---|---|---|---|---|"]
    for label, var in (report.get("variants") or {}).items():
        tv = var["central_component_test"]
        if tv.get("absent_by_construction"):
            lines.append(f"| {label} | 0 (by construction) | -- | -- | -- |")
            continue
        lines.append(f"| {label} | {tv['f_central_hat']:.4f} | {tv['delta_chi2']:.1f} | {tv['threshold']:.1f} | {tv['compatible_with_zero']} |")
    if any("isovector_test" in var for var in (report.get("variants") or {}).values()):
        lines += ["", "| variant | f_isovector | Delta chi2 (vacuum only) | threshold | required |", "|---|---|---|---|---|"]
        for label, var in report["variants"].items():
            tv = var["isovector_test"]
            lines.append(f"| {label} | {tv['f_isovector_hat']:.4f} | {tv['delta_chi2']:.1f} | {tv['threshold']:.1f} | {tv['required']} |")
    lines += ["", "## Named residuals", ""]
    for res in report["named_residual_register"] or [{"kind": "none"}]:
        lines.append(f"- {json.dumps(res, sort_keys=True)}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def _probe_worker(f: int, g: int, shape_point, flavor_point) -> dict:
    """The clean-validation probe: slot 0 of the final generation,
    regenerated; returns its digests."""
    probe = generate_slot(slot_class(0), slot_seed(f, g, 0), EVENTS_PER_SLOT, slot_partition(0), tuple(shape_point),
                          FLOOR_GRID[f], tuple(flavor_point), chunks=FANOUT_POOL_WORKERS)
    return {"arrays_sha256": arrays_digest(probe["arrays"]), "event_rolling_sha256": probe["event_rolling_sha256"]}


FANOUT_REGISTRY = {"fit": _fit_worker, "probe": _probe_worker}
_FANOUT = None


def _fanout() -> "FO.FanOut":
    """The dispatcher of the running step (local mode when no step set one)."""
    return _FANOUT if _FANOUT is not None else FO.FanOut(Path("."), FANOUT_REGISTRY)


def step_pre_registration(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    ledger = RC.seed_ledger(NESTED_ROOT)
    disjoint = RC.check_seed_disjointness(ledger, LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    frozen = frozen_digests()
    # generation 0: the sealed source run's fit (its final branch report,
    # verified by its digest); the spec's judgment values are named
    g0 = SPEC["generation_0"]
    src_dir = MONO_ROOT / "dev" / "00runs" / g0["run"]
    if not (src_dir / "seal_manifest.json").is_file():
        contract.refuse(f"the sealed archive {src_dir} is required")
    src = RC.read_json(src_dir / BRANCH_REPORT_FMT.format(f=0))
    if RC.canonical_digest({k: v for k, v in src.items() if k != "digest"}) != src["digest"] or src["digest"] != g0["branch_digest"]:
        contract.refuse(f"the {g0['run']} branch report does not reproduce its recorded digest")
    fitted_src = src["fitted"] if g0["fit"] == "free" else src["null_fit"]["fitted"]
    gen0_values = {**{p.name: p.gen0 for p in PARAMS}, "w_low_mass": GEN0_W_LOW}
    for name in g0["from_fit"]:
        if float(fitted_src[name]) != float(gen0_values[name]):
            contract.refuse(f"the generation-0 value of {name} does not match the {g0['run']} {g0['fit']} optimum")
    _load_physics(FLOOR_GRID[0])  # refuses a committed config that already sets a floor
    a = anchors()
    fit_tables = load_fit_tables()
    n_obs_total = sum(int(fit_tables[name].data.size) for name in FIT_SPECIES) + (
        NEUTRON_XF_EDGES.size - 1 if NEUTRON_SEGMENT else 0)
    payload = {
        "classes": list(ACTIVE_CLASSES), "replicas": REPLICAS, "events_per_slot": EVENTS_PER_SLOT, "max_generations": MAX_GENERATIONS,
        "hypothesis": TEXTS["hypothesis"],
        "kernel_change": TEXTS["kernel_change"],
        "kernel_config": KERNEL.as_dict(),
        **({"central_component": False} if not CENTRAL_COMPONENT else {}),
        **({"topology_options": dict(TOPOLOGY_OPTIONS)} if TOPOLOGY_OPTIONS else {}),
        "run_spec": {"path": SPEC_REL, "sha256": RC.sha256_file(SPEC_PATH)},
        "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST,
                  "rule": f"raw seed = RAW_FIRST + {SEEDED_SLOTS} * generation + seed slot; slot = class_index * {REPLICAS} + replica; "
                          "an nd0 slot draws the seed slot of the nd replica with the same index",
                  "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * seed slot + i", "disjointness": disjoint},
        "generation_0_point": {"epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND, "w_excited": GEN0_W_EXCITED,
                               "f_central": GEN0_F_CENTRAL, "flavor": dict(zip(FLAVOR_ORDER, GEN_FLAVOR)),
                               "provenance": {"run": g0["run"], "branch_report_digest": src["digest"], "fit": g0["fit"],
                                              "source_test": src["central_component_test"]["compatible_with_zero"],
                                              "from_fit": list(g0["from_fit"]), "judgment": g0.get("judgment", {})}},
        "fixed": {"continuation_floor": FLOOR_GRID[0], **({"r_v": R_V_FIXED} if R_V_FIXED is not None else {}),
                  "shapes_note": TEXTS["shapes_note"],
                  "fixed_parameters": {p.name: p.value for p in PARAMS if not p.free},
                  "fractions": {k: a[k]["value"] for k in FRACTION_NAMES}, "w_intact": 0.0,
                  "nd_generation_remnant_weights": ND_GENERATION_REMNANT_WEIGHTS, "diffractive_split": FROZEN_REMNANT_WEIGHTS},
        "free_parameters": list(param_names(None)),
        "flavor": {"free": list(FLAVOR_FREE), "log_bounds": [math.exp(b) for b in FLAVOR_LOG_BOUNDS],
                   "reweighting": "exact, through the per-vertex flavor ledger (the R66 key-table expression)"},
        "two_component_variant": "the kernel's generate_nd(central=False): the same remnant selections, transfers and remnant decays, no central draw, the two baryonic systems sharing the full sqrt(s)",
        "test": ({"statistic": "Delta chi2 = chi2(f_central = 0) - chi2_min, each minimized over the shapes",
                  "rule": f"compatible with zero when Delta chi2 <= {TEST_DELTA_CHI2} S^2, S^2 = chi2_min / ({n_obs_total} - {N_FREE_PARAMETERS}); the raw Delta chi2 is reported beside it",
                  "profile_scan": list(FC_SCAN), "decision": "owner, 2026-10-02",
                  "outcome_rule": "the outcome is reported as measured; a nonzero f_central is a result, not a reason to change the model in this run"}
                 if CENTRAL_COMPONENT else
                 {"statistic": None, "rule": "no central-component test: f_central = 0 by construction (central_component false)",
                  "basis": SPEC.get("texts", {}).get("central_component_basis")}),
        "variants": {"discrepancy": DISCREPANCY_VARIANT, "normalization": list(NORM_VARIANTS),
                     "role": "the same test under each variant; never drive the regeneration",
                     **({"systematic": list(SPEC.get("systematic_variants", []))} if SPEC.get("systematic_variants") else {})},
        "discrepancy": {"primary": DISCREPANCY, "variant": DISCREPANCY_VARIANT},
        "normalization": {"fixed": NORM_FIXED, "variants": list(NORM_VARIANTS)},
        "parameters": [{**p._asdict(), "bounds": list(p.bounds)} for p in PARAMS],
        "trust_region": {"ess_floor": ESS_FLOOR, "ess_interior": ESS_INTERIOR, "penalty": TRUST_PENALTY,
                         "regeneration": "when either fit has a class ESS below ess_interior: at the midpoint of the free and null shape "
                                         "optima when both lie in the trust region, else at the optimum of the fit that does (the free "
                                         "optimum when neither does)"},
        "data": {"fit_window": FIT_WINDOW, "species": list(FIT_SPECIES), "anchors": a,
                 **({"bin_center_correction": "the model bins at the tabulated points: each bin average divided by the "
                                              "data's own local shape factor <f>_bin / f(x0, p0) (run_observables."
                                              "bin_center_factors; the neutron dN/dx_F by the one-dimensional factor)"}
                    if BIN_CENTER else {})},
        "verdict_rule": {"constrained": "both fits converged and in the trust region, w_excited interior, no exponent on its bound, shape Hessian positive definite at the fitted f_central, shape block spreads within 1.5 x pooled error x sqrt(32)",
                         "test": "reported separately; does not enter the verdict"},
        "optimizer": {"method": "Powell (bounded)", "starts_free": N_STARTS, "starts_null": N_STARTS_NULL, "maxfev": FIT_MAXFEV,
                      "hessian_steps": HESSIAN_STEPS},
        "frozen_substrate": frozen,
    }
    if ISOVECTOR_GEN is not None:
        payload["isovector"] = {
            "generation_fraction": float(ISOVECTOR_GEN),
            "class": "na49_mixing.isovector_exchange: the I = 1 exchange with the Clebsch-Gordan line states, drawn per SD, DD and ND event",
            "reweighting": "exact: an isovector event by f/f0, a vacuum event by (1 - f)/(1 - f0) (the event's recorded iv_config)",
            "own_slope": ("the isovector events draw their transfer from exp(b_isovector t), a coordinate of the generation point "
                          "reweighted on the isovector events only; the vacuum events keep B_SD") if ISOVECTOR_SLOPE else None,
            "test": {"statistic": "Delta chi2 = chi2(f_isovector = 0) - chi2_min, each minimized over the other parameters"
                                  + (" with f_central free" if CENTRAL_COMPONENT else " (f_central = 0 by construction)"),
                     "rule": f"the isovector exchange is required when Delta chi2 > {TEST_DELTA_CHI2} S^2, S^2 = chi2_min / ({n_obs_total} - {N_FREE_PARAMETERS})",
                     "outcome_rule": "reported as measured beside the central-component test; neither enters the verdict"},
            "texts": SPEC.get("isovector", {}).get("texts", {})}
    if CORRELATED_ERRORS:
        payload["correlated_errors"] = {
            **CORRELATED_ERRORS,
            "rule": "per species a normalization scale with a Gaussian prior, for the kaons an identification scale on the bins "
                    "above xf_min, for the neutrons one scale with the per-point remainder floored; every scale profiled in "
                    "closed form at each evaluation, its penalty (eta / prior)^2 added to the objective; the sources: NA49 pp "
                    "pions EPJC 45 (2006) 343 Table 3, kaons EPJC 68 (2010) 1 Table 1, protons and neutrons EPJC 65 (2010) 9 "
                    "Table 2"}
    if NEUTRON_SEGMENT:
        payload["neutron_segment"] = {"table": NEUTRON_TABLE_REL, "edges": NEUTRON_XF_EDGES.tolist(),
                                      "variance": "the table error (percent of the value) and the model discrepancy in quadrature",
                                      "model": "dN/dx_F per inelastic event of the final-state neutrons at x_F > 0"}
    sync_path = MONO_ROOT / SYNC_MANIFEST_NAME
    payload["source"] = {"git": {"outer": RC.git_info(MONO_ROOT), "nested": RC.git_info(NESTED_ROOT)},
                         "sync_manifest": RC.read_json(sync_path) if sync_path.is_file() else None,
                         "rule": "the local git heads when the checkout is a repository; otherwise the gateway sync record"}
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()})
    for f in range(N_FLOORS):
        RC.write_json(contract.path(GENERATION_POINTS_FMT.format(f=f)), {"points": [
            {"floor_index": f, "generation": 0, "epsilon": GEN0_EPSILON, "b_sd": GEN0_B_SD, "epsilon_nd": GEN0_EPSILON_ND,
             **{n: GEN0_SHAPE[shape_pos(n)] for n in EXTRA_SHAPE}, **dict(zip(FLAVOR_ORDER, GEN_FLAVOR))}]})


def _verify_frozen(contract: RC.RunContract) -> dict:
    pre = contract.load_pre_registration()
    frozen = {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}
    status = RC.verify_frozen_substrate(MONO_ROOT, frozen, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    return status


def _points(contract: RC.RunContract, floor_index: int) -> list[dict]:
    return RC.read_json(contract.path(GENERATION_POINTS_FMT.format(f=int(floor_index))))["points"]


def _generation_point(contract: RC.RunContract, floor_index: int, generation: int) -> tuple:
    """The registered generation point: ((epsilon, B_SD, epsilon_nd), flavor point)."""
    for p in _points(contract, floor_index):
        if int(p["generation"]) == int(generation):
            return ((float(p["epsilon"]), float(p["b_sd"]), float(p["epsilon_nd"])) + tuple(float(p[n]) for n in EXTRA_SHAPE),
                    tuple(float(p[name]) for name in FLAVOR_ORDER))
    contract.refuse(f"branch {floor_index} generation {generation} has no registered point")


def _generation_complete(contract: RC.RunContract, floor_index: int, generation: int) -> bool:
    return all(contract.path(slot_names(floor_index, generation, s)[0]).is_file() for s in ACTIVE_SLOTS)


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
    todo_slots = list(ACTIVE_SLOTS) if not slots else [int(s) for s in slots]
    if any(s not in ACTIVE_SLOTS for s in todo_slots):
        contract.refuse(f"slots {sorted(set(todo_slots) - set(ACTIVE_SLOTS))} belong to no generated class of this run")
    jobs = []
    for f in branches:
        if _branch_done(contract, f):
            continue
        g = _latest_registered(contract, f) if generation is None else int(generation)
        shape_point, flavor_point = _generation_point(contract, f, g)
        for slot in todo_slots:
            if contract.path(slot_names(f, g, slot)[0]).is_file():
                continue
            jobs.append(((f, g, slot), (slot_class(slot), slot_seed(f, g, slot), EVENTS_PER_SLOT, slot_partition(slot),
                                         shape_point, FLOOR_GRID[f], flavor_point)))
    if not jobs:
        print(f"[{RUN_ID}] nothing to generate", flush=True)
        return
    t0 = time.perf_counter()
    if len(jobs) == 1 and (workers or 1) > 1:
        results = [generate_slot(*jobs[0][1], chunks=int(workers))]
    else:
        results = RC.run_replicas(_generate_job, [job[1] for job in jobs], max_workers=workers)
    for ((f, g, slot), _), result in zip(jobs, results):
        write_slot(contract, f, g, slot, result, pre_sha)
    print(f"[{RUN_ID}] generated {len(jobs)} slots over branches {sorted({j[0][0] for j in jobs})} in {time.perf_counter() - t0:.0f} s", flush=True)


def _generate_job(cls, seed, n_events, partition_base, shape_point, floor, flavor_point=None):
    return generate_slot(cls, seed, n_events, partition_base, shape_point, floor, flavor_point)


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


VARIANTS = ()  # set by configure()
#: the class-fraction overrides of the spec's systematic variants, by label (2026-10-22; set by configure())
VARIANT_ANCHORS: dict = {}


def systematic_variants(spec: dict) -> tuple:
    """The spec's extra systematic variants (2026-10-22, the systematics
    campaign): each ``{"label", "discrepancy", "normalization", "f_sd_total",
    "f_dd", "f_sd_window"}`` (all but the label optional) refitted on the
    run's sample like the three committed variants, the class fractions of
    the anchors replaced where named (``f_sd_window`` under the spec's
    single-diffraction window only, 2026-10-23)."""
    out = []
    for entry in spec.get("systematic_variants", []):
        unknown = sorted(set(entry) - {"label", "discrepancy", "normalization", "f_sd_total", "f_dd", WINDOW_NAME})
        if unknown or "label" not in entry:
            raise ValueError(f"run spec: a systematic variant takes a label and the keys discrepancy, normalization, "
                             f"f_sd_total, f_dd, f_sd_window; got {sorted(entry)}")
        if WINDOW_NAME in entry and spec.get("single_diffraction_window") is None:
            raise ValueError("run spec: a variant of f_sd_window needs the single_diffraction_window")
        override = {k: float(entry[k]) for k in FRACTION_NAMES + (WINDOW_NAME,) if k in entry} or None
        out.append((str(entry["label"]), float(entry.get("discrepancy", DISCREPANCY)),
                    float(entry.get("normalization", NORM_FIXED)), override))
    labels = [v[0] for v in out]
    if len(set(labels)) != len(labels) or set(labels) & {"discrepancy_0.10", "normalization_+0.02", "normalization_-0.02"}:
        raise ValueError(f"run spec: the systematic variant labels must be distinct and new, got {labels}")
    return tuple(out)


def _variants(contract: RC.RunContract, chosen: dict, tables: dict) -> dict:
    """The test under the discrepancy and normalization variants: the free
    and the null fit refitted on the branch's final generation."""
    f, g = int(chosen["floor_index"]), int(chosen["generation"])
    free_starts = [x_from_physical(chosen["fitted"], None)] + dispersed_starts(None)[1:N_STARTS_VARIANT]
    null_starts = [x_from_physical(chosen["null_fit"]["fitted"], 0.0)] + dispersed_starts(0.0)[1:N_STARTS_VARIANT]
    iv = ISOVECTOR_GEN is not None and "vacuum_only_fit" in chosen
    if iv:
        iv_starts = ([x_from_physical(chosen["vacuum_only_fit"]["fitted"], None, 0.0)]
                     + dispersed_starts(None, 0.0)[1:N_STARTS_VARIANT])
    per = (3 if iv else 2) - (0 if CENTRAL_COMPONENT else 1)
    thunks = []
    for _label, dz, norm in VARIANTS:
        ov = VARIANT_ANCHORS.get(_label)
        thunks.append(lambda dz=dz, norm=norm, ov=ov: run_fit_parallel(contract, f, g, free_starts, discrepancy=dz, norm=norm,
                                                                     maxfev=FIT_MAXFEV, pool_workers=BLOCK_WORKERS_PER_VARIANT_START,
                                                                     anchor_override=ov))
        if CENTRAL_COMPONENT:
            thunks.append(lambda dz=dz, norm=norm, ov=ov: run_fit_parallel(contract, f, g, null_starts, fc_fixed=0.0, discrepancy=dz,
                                                                         norm=norm, maxfev=FIT_MAXFEV,
                                                                         pool_workers=BLOCK_WORKERS_PER_VARIANT_START, anchor_override=ov))
        if iv:
            thunks.append(lambda dz=dz, norm=norm, ov=ov: run_fit_parallel(contract, f, g, iv_starts, iv_fixed=0.0, discrepancy=dz,
                                                                         norm=norm, maxfev=FIT_MAXFEV,
                                                                         pool_workers=BLOCK_WORKERS_PER_VARIANT_START, anchor_override=ov))
    fits = _fanout().gather(thunks)
    out = {}
    for v, (label, dz, norm) in enumerate(VARIANTS):
        ov = VARIANT_ANCHORS.get(label)
        free_fit = fits[per * v]
        null_fit = fits[per * v + 1] if CENTRAL_COMPONENT else free_fit
        with BlockPool(contract.run_dir, f, g, MASTER_BLOCK_WORKERS) as pool:
            v_free, d_free = PooledObjective(tables, anchors(ov), pool, discrepancy=dz, norm=norm).evaluate(free_fit["x"], details=True)
            if CENTRAL_COMPONENT:
                v_null, d_null = PooledObjective(tables, anchors(ov), pool, fc_fixed=0.0, discrepancy=dz, norm=norm).evaluate(
                    null_fit["x"], details=True)
            else:
                v_null, d_null = v_free, d_free
            if iv:
                iv_fit = fits[per * v + per - 1]
                v_iv, d_iv = PooledObjective(tables, anchors(ov), pool, discrepancy=dz, norm=norm, iv_fixed=0.0).evaluate(
                    iv_fit["x"], details=True)
        test = (central_test(v_free, v_null, d_free["m"].size, d_free["physical"]["f_central"], []) if CENTRAL_COMPONENT
                else central_absent(v_free, d_free["m"].size))
        out[label] = {"discrepancy": dz, "normalization": norm, **({"anchors": dict(ov)} if ov else {}),
                      "fitted": {n: d_free["physical"][n] for n in SHAPE_NAMES + ("f_central",)},
                      "null_fitted": {n: d_null["physical"][n] for n in SHAPE_NAMES},
                      "objective": v_free, "objective_null": v_null, "converged": bool(free_fit["converged"] and null_fit["converged"]),
                      "ess_min": min(d_free["ess_min"], d_null["ess_min"]), "central_component_test": test}
        if iv:
            out[label]["isovector_test"] = isovector_test(min(test["chi2_min"], v_iv), v_iv, d_free["m"].size, d_free["physical"][ISO_NAME])
            out[label]["vacuum_only_fitted"] = {n: d_iv["physical"][n] for n in param_names(None, 0.0)}
            out[label]["converged_vacuum_only"] = bool(iv_fit["converged"])
    return out


def step_profile(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    _verify_frozen(contract)
    chosen = _load_branches(contract)[0]
    variants = _variants(contract, chosen, load_fit_tables())
    report = compose_report(chosen, variants)
    body = contract.write_report(report, render_report_md(report))
    s = body["test_summary"]
    central_line = (f"f_central compatible with zero: primary {s['compatible_in_primary']}, every variant "
                    f"{s['compatible_in_every_variant']}" if CENTRAL_COMPONENT else "f_central = 0 by construction")
    print(f"[{RUN_ID}] report written: verdict {'CONSTRAINED' if body['verdict']['constrained'] else 'NOT_CONSTRAINED'}; "
          f"{central_line}; digest {body['digest'][:16]}", flush=True)


def step_gate(run_dir: Path | None) -> None:
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    test = report["central_component_test"]
    record = {
        "schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
        "obligation": TEXTS["obligation"],
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
    if "isovector_test" in report:
        record["isovector_test"] = report["isovector_test"]
        record["vacuum_only_fitted"] = report["vacuum_only_fit"]["fitted"]
    RC.write_json(contract.path(GATE_RECORD_NAME), record)
    central_line = (f"f_central {test['f_central_hat']:.4f}, Delta chi2 {test['delta_chi2']:.1f} against {test['threshold']:.1f}: "
                    f"compatible with zero {test['compatible_with_zero']}" if CENTRAL_COMPONENT else "f_central = 0 by construction")
    print(f"[{RUN_ID}] gate record: {record['verdict']}; {central_line}", flush=True)


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
    shape_point, flavor_point = _generation_point(contract, f, g)
    fo = _fanout()
    rerun_branch, probe = fo.gather([lambda: evaluate_generation(contract, f, g, tables, anchors()),
                                     lambda: fo.map("probe", [[f, g, [float(v) for v in shape_point],
                                                               [float(v) for v in flavor_point]]])[0]])
    recorded = next(b for b in branches if int(b["floor_index"]) == f)
    rerun_branch = {"schema": recorded["schema"], **contract.identity(), **rerun_branch, "generation_history": recorded["generation_history"]}
    rerun_branch["digest"] = RC.canonical_digest(rerun_branch)
    branch_match = bool(rerun_branch["digest"] == recorded["digest"])
    rerun = compose_report(rerun_branch, _variants(contract, rerun_branch, tables))
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    sealed_probe = RC.read_json(contract.path(slot_names(f, g, 0)[0]))
    probe_match = bool(probe["arrays_sha256"] == sealed_probe["arrays_sha256"]
                       and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"])
    payload = {"commands": [f"python helpers/benchmarking/{HERE.name} --spec {SPEC_REL} --clean-validate --run-dir {contract.run_dir}",
                            TEXTS["register_command"]],
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
            for s in ACTIVE_SLOTS:
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
            for cls in ACTIVE_CLASSES]
    results = RC.run_replicas(_generate_job, jobs)
    for cls, result in zip(ACTIVE_CLASSES, results):
        np.savez(contract.path("smoke_" + SLOT_NPZ_FMT.format(cls=cls, k=0, f=f, g=0)), **result["arrays"])
        RC.write_json(contract.path("smoke_" + SLOT_JSON_FMT.format(cls=cls, k=0, f=f, g=0)), {
            "schema": f"{SCHEMA}.smoke_slot/1", "pre_registration_sha256": pre_sha, "class_label": cls, "generation": 0, "replica": 0,
            "floor_index": f, "continuation_floor": result["continuation_floor"], "flavor_generation": result["flavor_generation"],
            "informative_vertices": result["informative_vertices"], "nd_central": result["nd_central"],
            **({"isovector_generation_fraction": result["isovector_generation_fraction"]}
               if "isovector_generation_fraction" in result else {}),
            **({BIV_NAME: result[BIV_NAME]} if BIV_NAME in result else {}),
            **({EPSC_NAME: result[EPSC_NAME]} if EPSC_NAME in result else {}),
            **({IVF_NAME: result[IVF_NAME]} if IVF_NAME in result else {}),
            **({TAIL_NAME: result[TAIL_NAME]} if TAIL_NAME in result else {}),
        **({OPE_NAME: result[OPE_NAME]} if OPE_NAME in result else {}),
            **{n: result[n] for n in SPLIT_FREE if n in result},
            "seed": result["seed"], "n_events": result["n_events"], "epsilon": result["epsilon"], "b_sd": result["b_sd"], "epsilon_nd": result["epsilon_nd"],
            "arrays_sha256": arrays_digest(result["arrays"]), "event_rolling_sha256": result["event_rolling_sha256"],
            "conservation": result["conservation"], "tracks": result["tracks"]})
    # common random numbers: nd and nd0 share the remnant selections and transfers
    shared = None
    if CENTRAL_COMPONENT:
        nd, nd0 = (results[ACTIVE_CLASSES.index(c)]["arrays"] for c in ("nd", "nd0"))
        shared = {key: bool(np.array_equal(nd[key], nd0[key])) for key in ("rem_kind", "shape_m1", "shape_m2", "shape_t1", "shape_t2")}
    tables = load_fit_tables()
    report = evaluate_generation(contract, f, 0, tables, anchors(), smoke=True)
    smoke_templates = Templates(contract, f, 0, tables, smoke=True)
    # the check point sits at the generation point (a smoke fit can end on a bound)
    check = {**report["fitted"], "epsilon_nd": GEN0_EPSILON_ND + 0.02, "w_excited": GEN0_W_EXCITED, "w_low_mass": GEN0_W_LOW,
             "gamma_s_had": GEN_FLAVOR[0] * 1.1, "assoc_norm": GEN_FLAVOR[3] * 0.9, "r_bbar": GEN_FLAVOR[2] * 1.2,
             "f_central": 0.4 if CENTRAL_COMPONENT else 0.0,
             **({ISO_NAME: 0.9 * float(ISOVECTOR_GEN)} if ISOVECTOR_GEN is not None else {}),
             **({BIV_NAME: 1.1 * float(GEN0_SHAPE[shape_pos(BIV_NAME)])} if ISOVECTOR_SLOPE else {}),
             **({EPSC_NAME: float(GEN0_SHAPE[shape_pos(EPSC_NAME)]) + 0.1} if CENTRAL_POWER else {}),
             **({IVF_NAME: 1.1 * float(GEN0_SHAPE[shape_pos(IVF_NAME)])} if INTACT_FREE else {}),
             **({TAIL_NAME: 0.9 * float(GEN0_SHAPE[shape_pos(TAIL_NAME)])} if TAIL_FREE else {}),
             **({OPE_NAME: 0.9 * float(GEN0_SHAPE[shape_pos(OPE_NAME)])} if OPE_FREE else {}),
             **{n: 1.1 * float(GEN0_SHAPE[shape_pos(n)]) for n in SPLIT_FREE}}
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
    parser = argparse.ArgumentParser(description="the run engine (one spec per run)")
    parser.add_argument("--spec", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pre-registration", "--generate", "--evaluate", "--profile", "--gate", "--clean-validate", "--seal", "--smoke"):
        group.add_argument(flag, action="store_true")
    group.add_argument("--fanout-run-task", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--floor-index", type=int, action="append", default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--slot", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    FO.add_arguments(parser)
    return parser


def main(argv: list | None = None) -> int:
    global _FANOUT
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--spec", type=Path, required=True)
    configure(pre.parse_known_args(argv)[0].spec)
    if "--fanout-task" in argv:
        # a fan-out job: one fit (or probe) of a requested batch
        args = _build_parser().parse_args(argv + ["--fanout-run-task"])
        contract = _contract(args.run_dir)
        _verify_frozen(contract)
        ns, batch, index = args.fanout_task
        out = FO.run_task(contract.run_dir, FANOUT_REGISTRY, ns, batch, int(index))
        print(f"[{RUN_ID}] fan-out task {ns}/{batch}/{index}: {out.name}", flush=True)
        return 0
    args = _build_parser().parse_args(argv)
    if args.fanout == "gateway" and not args.fanout_key:
        RC.refuse("--fanout gateway requires --fanout-key", prefix=HERE.stem)
    _FANOUT = FO.FanOut(_contract(args.run_dir).run_dir, FANOUT_REGISTRY, mode=args.fanout,
                        namespace="clean" if args.clean_validate else "main")
    try:
        return _main_steps(args)
    except FO.FanOutPending as exc:
        if args.fanout != "gateway":
            raise
        path = FO.write_pending(_FANOUT.run_dir, args.fanout_key, exc)
        print(f"[{RUN_ID}] {exc}; pending record {path.name}", flush=True)
        return 0


def _main_steps(args) -> int:
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
