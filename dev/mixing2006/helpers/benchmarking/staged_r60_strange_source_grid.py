"""R60 driver: the strange source grid (plan section 12, run R60; card
TASK-20260928-r6-run-r60-strange-source-grid).

Plan text: "Generate neutral mesonic and baryonic continuum sources over
thresholds relevant to K Kbar, K*, phi, K Lambda, and K Sigma.  Measure
branching fractions and spectra as functions of parent mass."

Design (pre-registered; the constants are the contract):

- Sources: the four continuum families that carry the Round 6 channel
  classes (mesonic_neutral (0, 0, 0), mesonic_charged (0, 1, 0),
  baryonic_np (1, 1, 0), baryonic_strange (1, 0, -1)), each at rest, on a
  grid of 30 parent masses from 1.05 to 3.95 GeV in steps of 0.1 GeV
  (the K Kbar threshold 0.99 GeV to well above the baryon-pair and
  excited-hyperon thresholds).
- Decay: the frozen simplified cascade of the topology classes (the
  central-decay loop of the non-diffractive module: depth guard, catalog
  handoff, continuum continuation with the established sibling assembled,
  direct termination, below-threshold single state) at the identity flavor
  point with the flavor ledger recorded, so the flavor response of every
  yield is read analytically by reweighting (no regeneration).
- Per node: 40,000 decays; the root-vertex channel census, the produced
  resonance census over every continuum vertex (phi, K*, Sigma(1385),
  Lambda(1520), ...), the final-state species multiplicities, rest-frame
  momentum spectra of K+-, neutral kaons, Lambda, antiproton, pions, and
  the reweighted strange, vector, and baryon-pair yields at the
  pre-registered flavor points with their effective sample sizes.
- Seeds: master 20260928; raw seed 20303556 + job, job = family_index * 30
  + node; event ids 2,100,000,000 + 100,000 job + i.

Steps: --pre-registration, --generate [--job J ...], --evaluate, --gate,
--clean-validate, --seal; --smoke (2 families x 3 nodes x 400 decays into
an explicit --run-dir).
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

from na49_mixing.cascade_decay import (  # noqa: E402
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    MAX_CASCADE_DEPTH,
    CascadeState,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import (  # noqa: E402
    handoff_decay,
    has_valid_direct_channel,
    is_catalog_handoff,
)
from na49_mixing.flavor_weights import FlavorParams, ledger_log_weight, vertex_ledger_record  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.importance import ess_fraction  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import families_from_json  # noqa: E402
from na49_mixing.topology_single_diffraction import _final_from_catalog, _kernel_finals  # noqa: E402

RUN_ID = "r60_strange_source_grid_2026-09-28"
CARD = "TASK-20260928-r6-run-r60-strange-source-grid"
SCHEMA = "na49_mixing.staged_r60_strange_source_grid"
LEDGER_RUN = "r60_strange_source_grid"

FAMILIES = ("mesonic_neutral", "mesonic_charged", "baryonic_np", "baryonic_strange")
MASS_NODES = tuple(round(1.05 + 0.1 * n, 2) for n in range(30))
N_JOBS = len(FAMILIES) * len(MASS_NODES)  # 120
DECAYS_PER_NODE = 40_000
MASTER_SEED = 20260928
RAW_FIRST = 20303556
RAW_LAST = RAW_FIRST + N_JOBS - 1  # 20303675
PARTITION_BASE = 2_100_000_000
SMOKE_SEED = 20303901
SMOKE_PARTITION = 2_910_000_000
SMOKE_DECAYS = 400
SMOKE_FAMILIES = ("mesonic_neutral", "baryonic_np")
SMOKE_NODES = (1.25, 2.05, 3.05)

FLAVOR_POINTS = {
    "gamma_s_0.5": FlavorParams(gamma_s_had=0.5), "gamma_s_0.7": FlavorParams(gamma_s_had=0.7),
    "gamma_s_0.85": FlavorParams(gamma_s_had=0.85), "identity": FlavorParams(),
    "gamma_s_1.2": FlavorParams(gamma_s_had=1.2), "gamma_s_1.5": FlavorParams(gamma_s_had=1.5),
    "r_v_0.5": FlavorParams(r_v=0.5), "r_v_2.0": FlavorParams(r_v=2.0),
    "r_bbar_0.5": FlavorParams(r_bbar=0.5), "r_bbar_2.0": FlavorParams(r_bbar=2.0),
}
SPECTRUM_SPECIES = {"k_plus": (321,), "k_minus": (-321,), "k_zero": (311, -311, 310, 130), "lambda": (3122,),
                    "antiproton": (-2212,), "pi_plus": (211,), "pi_minus": (-211,)}
SPECTRUM_EDGES = np.linspace(0.0, 2.0, 41)
RESONANCE_IDS = {"phi": (333,), "k_star": (323, -323, 313, -313), "sigma_1385": (3224, 3214, 3114, -3224, -3214, -3114),
                 "lambda_1520": (102134, -102134), "rho": (213, -213, 113), "omega": (223,), "delta": (2224, 2214, 2114, 1114)}
COUNT_GROUPS = {**OB.STRANGE_GROUPS, "proton": (2212,), "neutron": (2112,), "pi_plus": (211,), "pi_minus": (-211,), "pi_zero": (111,)}

FROZEN_RELS = (
    "dev/mixing2006/na49_mixing/cascade_decay.py", "dev/mixing2006/na49_mixing/cascade_termination.py",
    "dev/mixing2006/na49_mixing/resonance_decay.py", "dev/mixing2006/na49_mixing/resonance_widths.py",
    "dev/mixing2006/na49_mixing/hadron_catalog.py", "dev/mixing2006/na49_mixing/spectral_density.py",
    "dev/mixing2006/na49_mixing/flavor_weights.py", "dev/mixing2006/na49_mixing/topology_single_diffraction.py",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json", "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
    "helpers/benchmarking/run_contract.py", "helpers/benchmarking/run_observables.py",
)
JOB_JSON_FMT = "node_{family}_{mass:.2f}.json"
GATE_RECORD_NAME = "gate6_r60.json"


def job_family(job: int) -> str:
    return FAMILIES[int(job) // len(MASS_NODES)]


def job_mass(job: int) -> float:
    return MASS_NODES[int(job) % len(MASS_NODES)]


def job_seed(job: int) -> int:
    return RAW_FIRST + int(job)


def job_name(family: str, mass: float) -> str:
    return JOB_JSON_FMT.format(family=family, mass=mass)


def _contract(run_dir):
    return RC.RunContract(RUN_ID, CARD, driver_file=HERE, schema_prefix=SCHEMA, run_dir=run_dir)


def frozen_digests() -> dict:
    return {rel: RC.sha256_file(MONO_ROOT / rel) for rel in FROZEN_RELS}


def _load_physics():
    catalog = HadronCatalog.from_json(NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json")
    fams = families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    return catalog, fams


# ----------------------------------------------------------------------
# One source decay (the central-decay loop of the topology modules,
# generic over the family, returning the finals and the vertex records)
# ----------------------------------------------------------------------
def decay_source(family, mass: float, catalog, fams, rng_channel, rng_kinematics, ledger: list):
    root = CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(int(v) for v in family.qn),
                        quantum={}, family_id=family.family_id, axis=(0.0, 0.0, 1.0),
                        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=("r60",), depth=0)
    finals = []
    produced: list[int] = []
    root_channel = None
    depth_reached = 0
    current = root
    while True:
        if int(current.depth) > MAX_CASCADE_DEPTH:
            raise RuntimeError(f"depth guard at {current.family_id} {current.mass}")
        depth_reached = int(current.depth)
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(current, catalog, rng_channel, rng_kinematics)
            finals.extend(_kernel_finals(catalog, vertices, "source"))
            return finals, produced, root_channel, depth_reached, "catalog_handoff"
        channels = enumerate_channels(current, catalog, fams, flavor=None)
        continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
        if continuum:
            channel, daughters = draw_decay(current, continuum, catalog, fams, rng_channel, rng_kinematics)
            ledger.append(vertex_ledger_record(current.qn, continuum, channel, catalog, fams, None))
            if root_channel is None:
                root_channel = tuple(channel.daughters)
            successor = None
            for daughter in daughters:
                if daughter.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                    produced.append(pdg)
                    if pdg in catalog.stable_states():
                        finals.append(_final_from_catalog(catalog, pdg, daughter.momentum, "source"))
                    else:
                        kernel = ResonanceDecayer(catalog, 0, 0, 0)
                        kernel._rng_channel = rng_channel
                        kernel._rng_kinematics = rng_kinematics
                        finals.extend(_kernel_finals(catalog, kernel.decay_resonance(pdg, tuple(float(x) for x in daughter.momentum)), "source"))
                else:
                    successor = daughter
            if successor is None:
                raise RuntimeError("continuum vertex without a continuing daughter")
            current = successor
            continue
        if has_valid_direct_channel(current, channels):
            direct = [c for c in channels if c.kind != KIND_CONTINUUM]
            channel, daughters = draw_decay(current, direct, catalog, fams, rng_channel, rng_kinematics)
            ledger.append(vertex_ledger_record(current.qn, direct, channel, catalog, fams, None))
            if root_channel is None:
                root_channel = tuple(channel.daughters)
            for daughter in daughters:
                pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                produced.append(pdg)
                if pdg in catalog.stable_states():
                    finals.append(_final_from_catalog(catalog, pdg, daughter.momentum, "source"))
                else:
                    kernel = ResonanceDecayer(catalog, 0, 0, 0)
                    kernel._rng_channel = rng_channel
                    kernel._rng_kinematics = rng_kinematics
                    finals.extend(_kernel_finals(catalog, kernel.decay_resonance(pdg, tuple(float(x) for x in daughter.momentum)), "source"))
            return finals, produced, root_channel, depth_reached, "direct_channel"
        return finals, produced, root_channel, depth_reached, "below_threshold"


def generate_node(family_id: str, mass: float, seed: int, n_decays: int, partition_base: int) -> dict:
    catalog, fams = _load_physics()
    family = next(f for f in fams if f.family_id == family_id)
    root_census: dict[str, int] = {}
    produced_census: dict[str, int] = {}
    termination: dict[str, int] = {}
    counts = {name: np.zeros(n_decays, dtype=np.int16) for name in COUNT_GROUPS}
    spectra = {name: np.zeros(SPECTRUM_EDGES.size - 1) for name in SPECTRUM_SPECIES}
    log_w = {name: np.zeros(n_decays) for name in FLAVOR_POINTS}
    depth_hist = np.zeros(MAX_CASCADE_DEPTH + 2, dtype=np.int64)
    n_final = np.zeros(n_decays, dtype=np.int16)
    rolling = hashlib.sha256()
    worst_closure = 0.0
    t0 = time.perf_counter()
    for i in range(int(n_decays)):
        subs = derive_substreams(int(seed), 0, int(partition_base) + i)
        rng_c = np.random.default_rng(subs.stream_for("decay_channel"))
        rng_k = np.random.default_rng(subs.stream_for("decay_kinematics"))
        ledger: list = []
        finals, produced, root_channel, depth, mode = decay_source(family, mass, catalog, fams, rng_c, rng_k, ledger)
        # conservation of the source decay (exact quantum numbers, four-momentum closure)
        e_sum = math.fsum(float(h.four_momentum[0]) for h in finals) if finals else float(mass)
        p_sum = [math.fsum(float(h.four_momentum[j]) for h in finals) if finals else 0.0 for j in (1, 2, 3)]
        closure = max(abs(e_sum - mass) / mass, *(abs(p) / mass for p in p_sum))
        worst_closure = max(worst_closure, closure)
        if finals:
            bqs = tuple(sum(int(h.qn[j]) for h in finals) for j in range(3))
            if bqs != tuple(int(v) for v in family.qn):
                raise RuntimeError(f"source decay violates (B, Q, S): {bqs} vs {family.qn}")
        key = "|".join(root_channel) if root_channel else "none"
        root_census[key] = root_census.get(key, 0) + 1
        for pdg in produced:
            produced_census[str(pdg)] = produced_census.get(str(pdg), 0) + 1
        termination[mode] = termination.get(mode, 0) + 1
        depth_hist[min(depth, MAX_CASCADE_DEPTH + 1)] += 1
        n_final[i] = len(finals)
        for h in finals:
            pdg = int(h.pdg_id) if h.pdg_id is not None else None
            if pdg is None:
                continue
            for name, ids in COUNT_GROUPS.items():
                if pdg in ids:
                    counts[name][i] += 1
            p_abs = math.sqrt(sum(float(h.four_momentum[j]) ** 2 for j in (1, 2, 3)))
            for name, ids in SPECTRUM_SPECIES.items():
                if pdg in ids:
                    b = int(np.searchsorted(SPECTRUM_EDGES, p_abs, side="right")) - 1
                    if 0 <= b < spectra[name].size:
                        spectra[name][b] += 1.0
        for name, params in FLAVOR_POINTS.items():
            log_w[name][i] = ledger_log_weight(ledger, None, params)
        rolling.update((f"{i}|{mode}|{key}|{len(finals)}|" + ";".join(f"{int(h.pdg_id) if h.pdg_id is not None else 0}:{float(h.four_momentum[0]).hex()}" for h in finals) + "\n").encode())
    wall = time.perf_counter() - t0
    # the flavor response of the yields (self-normalized reweighting)
    response = {}
    for name in FLAVOR_POINTS:
        w = np.exp(log_w[name] - np.max(log_w[name])); w /= np.sum(w)
        response[name] = {"ess_fraction": float(ess_fraction(log_w[name])),
                          "mean_multiplicity": {group: float(np.sum(w * counts[group])) for group in COUNT_GROUPS},
                          "strange_event_fraction": float(np.sum(w * (sum(counts[g] for g in ("k_plus", "k_minus", "k_zero", "lambda", "antilambda", "sigma", "antisigma")) > 0))),
                          "baryon_pair_fraction": float(np.sum(w * ((counts["pbar"] + counts["antineutron"]) > 0)))}
    return {
        "family": family_id, "mass": float(mass), "seed": int(seed), "n_decays": int(n_decays), "partition_base": int(partition_base),
        "root_channel_census": dict(sorted(root_census.items(), key=lambda kv: -kv[1])),
        "produced_resonance_census": produced_census, "termination": termination,
        "depth_histogram": depth_hist.tolist(),
        "mean_multiplicity": {name: float(np.mean(counts[name])) for name in COUNT_GROUPS},
        "mean_final_count": float(np.mean(n_final)),
        "spectra": {"edges": SPECTRUM_EDGES.tolist(), **{name: spectra[name].tolist() for name in SPECTRUM_SPECIES}},
        "flavor_response": response,
        "conservation": {"decays": int(n_decays), "violations": 0, "worst_closure_relative": worst_closure},
        "event_rolling_sha256": rolling.hexdigest(), "wall_seconds": wall,
    }


def _resonance_rates(node: dict) -> dict:
    n = node["n_decays"]
    census = node["produced_resonance_census"]
    return {name: sum(census.get(str(pdg), 0) for pdg in ids) / n for name, ids in RESONANCE_IDS.items()}


def aggregate(contract: RC.RunContract, nodes: list[dict]) -> dict:
    grid = {}
    for node in nodes:
        fam = node["family"]
        entry = {"mass": node["mass"], "n_decays": node["n_decays"], "termination": node["termination"],
                 "mean_multiplicity": node["mean_multiplicity"], "resonance_per_decay": _resonance_rates(node),
                 "strange_event_fraction": node["flavor_response"]["identity"]["strange_event_fraction"],
                 "baryon_pair_fraction": node["flavor_response"]["identity"]["baryon_pair_fraction"],
                 "flavor_response": {k: {"ess_fraction": v["ess_fraction"], "strange_event_fraction": v["strange_event_fraction"],
                                         "baryon_pair_fraction": v["baryon_pair_fraction"],
                                         "k_plus": v["mean_multiplicity"]["k_plus"], "k_minus": v["mean_multiplicity"]["k_minus"],
                                         "lambda": v["mean_multiplicity"]["lambda"], "pbar": v["mean_multiplicity"]["pbar"]}
                                     for k, v in node["flavor_response"].items()},
                 "top_root_channels": dict(list(node["root_channel_census"].items())[:6]),
                 "event_rolling_sha256": node["event_rolling_sha256"]}
        grid.setdefault(fam, []).append(entry)
    for fam in grid:
        grid[fam].sort(key=lambda e: e["mass"])
    # monotonicity of the flavor response (plan section 12: gamma_s suppresses created strangeness)
    checks = []
    for fam, entries in grid.items():
        for e in entries:
            fr = e["flavor_response"]
            s = [fr[k]["strange_event_fraction"] for k in ("gamma_s_0.5", "gamma_s_0.7", "gamma_s_0.85", "identity", "gamma_s_1.2", "gamma_s_1.5")]
            b = [fr[k]["baryon_pair_fraction"] for k in ("r_bbar_0.5", "identity", "r_bbar_2.0")]
            checks.append({"family": fam, "mass": e["mass"], "strange_monotone": bool(all(x <= y + 1e-12 for x, y in zip(s, s[1:]))),
                           "pair_monotone": bool(all(x <= y + 1e-12 for x, y in zip(b, b[1:]))),
                           "min_ess": float(min(v["ess_fraction"] for v in fr.values()))})
    return {"grid": grid, "monotonicity_checks": checks,
            "summary": {"families": list(FAMILIES), "mass_nodes": list(MASS_NODES), "decays_per_node": nodes[0]["n_decays"],
                        "nodes": len(nodes), "all_strange_monotone": bool(all(c["strange_monotone"] for c in checks)),
                        "all_pair_monotone": bool(all(c["pair_monotone"] for c in checks)),
                        "min_ess_over_grid": float(min(c["min_ess"] for c in checks))}}


def render_report_md(report: dict) -> str:
    lines = [f"# R60: strange source grid ({RUN_ID})", "", f"Card {CARD}. {report['summary']['nodes']} nodes, {report['summary']['decays_per_node']} decays per node.", ""]
    for fam, entries in report["grid"].items():
        lines += [f"## {fam}", "", "| M [GeV] | <K+> | <K-> | <K0> | <Lambda> | <pbar> | strange frac | pair frac | phi/decay | K*/decay | Sigma(1385)/decay |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in entries:
            m = e["mean_multiplicity"]; r = e["resonance_per_decay"]
            lines.append(f"| {e['mass']:.2f} | {m['k_plus']:.4f} | {m['k_minus']:.4f} | {m['k_zero']:.4f} | {m['lambda']:.4f} | {m['pbar']:.4f} | {e['strange_event_fraction']:.4f} | {e['baryon_pair_fraction']:.4f} | {r['phi']:.4f} | {r['k_star']:.4f} | {r['sigma_1385']:.4f} |")
        lines.append("")
    s = report["summary"]
    lines += ["## Flavor response", "", f"Strange-event fraction monotone in gamma_s at every node: {s['all_strange_monotone']}; baryon-pair fraction monotone in r_BBbar: {s['all_pair_monotone']}; smallest reweighting ESS fraction over the grid and the ten flavor points: {s['min_ess_over_grid']:.3f}.", ""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def step_pre_registration(run_dir):
    contract = _contract(run_dir)
    ledger = RC.seed_ledger(NESTED_ROOT)
    disjoint = RC.check_seed_disjointness(ledger, LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    frozen = frozen_digests()
    payload = {"families": list(FAMILIES), "mass_nodes": list(MASS_NODES), "decays_per_node": DECAYS_PER_NODE, "jobs": N_JOBS,
               "seeds": {"master": MASTER_SEED, "raw_first": RAW_FIRST, "raw_last": RAW_LAST, "rule": "raw seed = RAW_FIRST + job; job = family_index * 30 + node",
                         "partition_base": PARTITION_BASE, "event_id_rule": "PARTITION_BASE + 100000 * job + i", "disjointness": disjoint},
               "flavor_points": {k: {"gamma_s_had": v.gamma_s_had, "r_v": v.r_v, "r_bbar": v.r_bbar, "assoc_norm": v.assoc_norm} for k, v in FLAVOR_POINTS.items()},
               "measurements": ["root_channel_census", "produced_resonance_census", "mean_multiplicity", "rest_frame_momentum_spectra", "flavor_response_by_reweighting"],
               "spectrum_edges_gev": SPECTRUM_EDGES.tolist(), "frozen_substrate": frozen}
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()})


def _verify(contract):
    pre = contract.load_pre_registration()
    status = RC.verify_frozen_substrate(MONO_ROOT, {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), LEDGER_RUN, (RAW_FIRST, RAW_LAST), PARTITION_BASE)
    return status


def _node_job(family_id, mass, seed, n, partition):
    return generate_node(family_id, mass, seed, n, partition)


def step_generate(run_dir, jobs_arg, workers):
    contract = _contract(run_dir)
    _verify(contract)
    pre_sha = RC.sha256_file(contract.path(RC.PREREGISTRATION_NAME))
    todo = [int(j) for j in jobs_arg] if jobs_arg else list(range(N_JOBS))
    jobs = [j for j in todo if not contract.path(job_name(job_family(j), job_mass(j))).is_file()]
    if not jobs:
        print(f"[{RUN_ID}] every requested node exists", flush=True); return
    t0 = time.perf_counter()
    results = RC.run_replicas(_node_job, [(job_family(j), job_mass(j), job_seed(j), DECAYS_PER_NODE, PARTITION_BASE + 100_000 * j) for j in jobs], max_workers=workers)
    for j, result in zip(jobs, results):
        record = {"schema": f"{SCHEMA}.node/1", **contract.identity(), "pre_registration_sha256": pre_sha, "job": j, **result, "written_utc": RC.now_utc()}
        RC.write_json(contract.path(job_name(job_family(j), job_mass(j))), record)
    print(f"[{RUN_ID}] {len(jobs)} nodes in {time.perf_counter() - t0:.0f} s", flush=True)


def _load_nodes(contract):
    nodes = []
    for j in range(N_JOBS):
        path = contract.path(job_name(job_family(j), job_mass(j)))
        if not path.is_file():
            contract.refuse(f"node record missing: {path.name}")
        nodes.append(RC.read_json(path))
    return nodes


def step_evaluate(run_dir):
    contract = _contract(run_dir)
    _verify(contract)
    report = aggregate(contract, _load_nodes(contract))
    body = contract.write_report(report, render_report_md(report))
    print(f"[{RUN_ID}] report written (digest {body['digest'][:16]}); strange monotone {report['summary']['all_strange_monotone']}, pair monotone {report['summary']['all_pair_monotone']}", flush=True)


def step_gate(run_dir):
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    record = {"schema": f"{SCHEMA}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
              "gate": "Gate 6 (plan section 12): parent-mass branching and spectra evidence for the flavor layer",
              "summary": report["summary"], "report_digest": report["digest"], "written_utc": RC.now_utc()}
    RC.write_json(contract.path(GATE_RECORD_NAME), record)


def step_clean_validate(run_dir):
    contract = _contract(run_dir)
    contract.require(RC.REPORT_NAME, GATE_RECORD_NAME)
    frozen = _verify(contract)
    sealed = RC.read_json(contract.path(RC.REPORT_NAME))
    rerun = aggregate(contract, _load_nodes(contract))
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    probe = generate_node(job_family(0), job_mass(0), job_seed(0), DECAYS_PER_NODE, PARTITION_BASE)
    sealed_probe = RC.read_json(contract.path(job_name(job_family(0), job_mass(0))))
    probe_match = probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"]
    contract.write_clean_validation({"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}"],
                                     "report_digest": sealed["digest"], "report_rerun_digest": rerun_digest, "report_hash_match": bool(rerun_digest == sealed["digest"]),
                                     "probe_node": {"job": 0, "rolling_match": bool(probe_match)}, "frozen_substrate": frozen,
                                     "seal": {"state": "pending (the seal lands over this record)"}})
    if rerun_digest != sealed["digest"] or not probe_match:
        contract.refuse("the clean validation does not reproduce the sealed report or probe")
    print(f"[{RUN_ID}] clean validation: report match True, probe match True", flush=True)


def step_seal(run_dir):
    contract = _contract(run_dir)
    contract.seal([GATE_RECORD_NAME] + [job_name(job_family(j), job_mass(j)) for j in range(N_JOBS)], "node_")


def step_smoke(run_dir: Path):
    contract = _contract(run_dir)
    if contract.run_dir == MONO_ROOT / "dev" / "00runs" / RUN_ID:
        contract.refuse("--smoke needs an explicit --run-dir")
    contract.run_dir.mkdir(parents=True, exist_ok=True)
    RC.write_json(contract.path("smoke_pre_registration.json"), {"schema": f"{SCHEMA}.smoke_pre_registration/1", **contract.identity(), "seed": SMOKE_SEED,
                                                                "partition_base": SMOKE_PARTITION, "decays": SMOKE_DECAYS, "frozen_substrate": frozen_digests()})
    jobs = [(fam, mass, SMOKE_SEED + i, SMOKE_DECAYS, SMOKE_PARTITION + 10_000 * i) for i, (fam, mass) in enumerate((f, m) for f in SMOKE_FAMILIES for m in SMOKE_NODES)]
    results = RC.run_replicas(_node_job, jobs)
    for (fam, mass, *_), result in zip(jobs, results):
        RC.write_json(contract.path("smoke_" + job_name(fam, mass)), {"schema": f"{SCHEMA}.smoke_node/1", **result})
    report = aggregate(contract, results)
    report["schema"] = f"{SCHEMA}.smoke_report/1"; report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_report.json"), report)
    contract.path("smoke_REPORT.md").write_text(render_report_md(report), encoding="utf-8")
    print(f"[{RUN_ID}] smoke: {len(results)} nodes; strange monotone {report['summary']['all_strange_monotone']}; min ESS {report['summary']['min_ess_over_grid']:.3f}", flush=True)


def _build_parser():
    parser = argparse.ArgumentParser(description=f"R60 driver ({CARD})")
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pre-registration", "--generate", "--evaluate", "--gate", "--clean-validate", "--seal", "--smoke"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--job", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.pre_registration: step_pre_registration(args.run_dir)
    elif args.generate: step_generate(args.run_dir, args.job, args.workers)
    elif args.evaluate: step_evaluate(args.run_dir)
    elif args.gate: step_gate(args.run_dir)
    elif args.clean_validate: step_clean_validate(args.run_dir)
    elif args.seal: step_seal(args.run_dir)
    elif args.smoke:
        if args.run_dir is None: RC.refuse("--smoke requires --run-dir", prefix=HERE.stem)
        step_smoke(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
