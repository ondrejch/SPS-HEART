"""The spec-driven driver of the Round 6 flavor runs R61 to R65 (review
record code fix C2).  A thin per-run file declares a :class:`RunSpec`
and calls :func:`main`; this module performs the steps on the shared
``run_contract``, ``run_observables`` and ``flavor_fit_core`` machinery:

- ``--pre-registration``: the topology substrate (the sealed R51c fit
  report: fractions, shapes, and the non-diffractive remnant weights), the inherited frozen flavor parameters (sealed earlier flavor
  reports), the frozen-substrate digests, the seed ledger check;
- ``--generate [--generation g]``: the mixture slots of one generation
  (generation g > 0 regenerates at the flavor point requested by the
  previous evaluation; the raw seeds stay the run's block, the event-index
  partition moves by 10,000,000 per generation);
- ``--evaluate``: the fit (or, for a validation run, the frozen
  comparison), the replica blocks, the Hessian, the readouts, the Gate 6
  shape-versus-ratio check; writes ``fit_report.json`` and ``REPORT.md``
  or a regeneration request;
- ``--gate``, ``--clean-validate``, ``--seal``, ``--smoke``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import run_contract as RC  # noqa: E402

NESTED_ROOT, MONO_ROOT = RC.locate_roots(HERE)
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

import flavor_fit_core as FC  # noqa: E402
import run_observables as OB  # noqa: E402

from na49_mixing.flavor_weights import FlavorParams  # noqa: E402

PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
TOPOLOGY_RUN_ID = "r51c_fraction_and_remnant_fit_2026-09-28"  # the R51c point (R51b superseded, see its card)
GENERATION_POINTS_NAME = "generation_points.json"
REGENERATION_NAME = "regeneration_request.json"
NOT_TRIGGERED_NAME = "not_triggered.json"
FIT_REPORT_NAME = "fit_report.json"
SLOT_JSON_FMT = "replica_{k}_g{g}.json"
SLOT_NPZ_FMT = "replica_{k}_g{g}.npz"
FROZEN_RELS = (
    "dev/mixing2006/na49_mixing/topology_runner.py", "dev/mixing2006/na49_mixing/topology_single_diffraction.py",
    "dev/mixing2006/na49_mixing/topology_double_diffraction.py", "dev/mixing2006/na49_mixing/topology_non_diffractive.py",
    "dev/mixing2006/na49_mixing/topology_mass_laws.py", "dev/mixing2006/na49_mixing/cascade_decay.py",
    "dev/mixing2006/na49_mixing/cascade_termination.py", "dev/mixing2006/na49_mixing/resonance_decay.py",
    "dev/mixing2006/na49_mixing/resonance_widths.py", "dev/mixing2006/na49_mixing/hadron_catalog.py",
    "dev/mixing2006/na49_mixing/spectral_density.py", "dev/mixing2006/na49_mixing/flavor_weights.py",
    "dev/mixing2006/na49_mixing/importance.py", "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
    "dev/mixing2006/config/staged_excitation.json", "helpers/benchmarking/run_contract.py",
    "helpers/benchmarking/run_observables.py", "helpers/benchmarking/flavor_fit_core.py",
    "helpers/benchmarking/flavor_run_driver.py",
)


@dataclasses.dataclass(frozen=True)
class RunSpec:
    run_id: str
    card: str
    ledger_run: str
    gate_record: str
    raw_first: int
    raw_last: int
    partition_base: int
    replicas: int
    events_per_replica: int
    statistics: tuple            # ({"name", "species", ["minus"]}, ...)
    free: tuple                  # flavor parameter names fit here
    inherit: tuple               # ((parameter, run_id_of_sealed_report), ...)
    priors: dict                 # {parameter: (center, sigma_ln)}
    starts: tuple                # dispersed start values per free parameter (tuples)
    readout_species: tuple = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
    moments: tuple = (("k_minus", "spshadrons.web.cern.ch/data/xsect_ptintegrated_kaminus_pp.txt"),
                      ("k_plus", "spshadrons.web.cern.ch/data/xsect_ptintegrated_kaplus_pp.txt"))
    max_generations: int = 4
    smoke_seed: int = 20303902
    smoke_partition: int = 2_920_000_000
    smoke_events: int = 300
    stability_ratio_bound: float = 1.5
    validation_only: bool = False
    conditional_trigger: dict | None = None   # R62: {"statistic": ..., "rule": ...}
    schema: str = ""
    description: str = ""


def _contract(spec: RunSpec, run_dir):
    return RC.RunContract(spec.run_id, spec.card, driver_file=HERE, schema_prefix=spec.schema or f"na49_mixing.{spec.ledger_run}", run_dir=run_dir)


def frozen_digests() -> dict:
    return {rel: RC.sha256_file(MONO_ROOT / rel) for rel in FROZEN_RELS}


def _sealed_report(run_id: str, name: str = RC.REPORT_NAME) -> tuple[dict, dict]:
    run_dir = MONO_ROOT / "dev" / "00runs" / run_id
    if not (run_dir / "seal_manifest.json").is_file():
        RC.refuse(f"the required sealed run {run_id} is not sealed at {run_dir}", prefix=HERE.stem)
    check = RC.verify_sha256sums(run_dir)
    if check["mismatches"]:
        RC.refuse(f"the sealed run {run_id} fails its checksum inventory: {check['mismatches'][:5]}", prefix=HERE.stem)
    report = RC.read_json(run_dir / name)
    return report, {"run": run_id, "report_digest": report.get("digest"), "seal_entries": check["entries"]}


def topology_substrate() -> tuple[dict, dict]:
    report, provenance = _sealed_report(TOPOLOGY_RUN_ID)
    fitted = report["fitted"]
    topology = {"fractions": {"sd_projectile": float(fitted["sd_projectile"]), "sd_target": float(fitted["sd_target"]),
                              "dd": float(fitted["f_dd"]), "nd": float(fitted["f_nd"])},
                "epsilon": float(fitted["epsilon"]), "b_sd": float(fitted["b_sd"]),
                "remnant_weights": {k: float(v) for k, v in fitted["remnant_weights"].items()},
                "diffractive_split": dict(FC.DIFFRACTIVE_SPLIT)}
    provenance = {**provenance, "verdict": "CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED",
                  "named_residual_register": report.get("named_residual_register", [])}
    return topology, provenance


def inherited_flavor(spec: RunSpec) -> tuple[dict, list]:
    values = {name: 1.0 for name in FC.FLAVOR_NAMES}
    sources = []
    for parameter, run_id in spec.inherit:
        run_dir = MONO_ROOT / "dev" / "00runs" / run_id
        if (run_dir / NOT_TRIGGERED_NAME).is_file():
            record = RC.read_json(run_dir / NOT_TRIGGERED_NAME)
            values[parameter] = 1.0
            sources.append({"parameter": parameter, "run": run_id, "not_triggered": True, "value": 1.0, "trigger_record": record.get("trigger")})
            continue
        report, provenance = _sealed_report(run_id)
        values[parameter] = float(report["fitted"][parameter])
        sources.append({"parameter": parameter, **provenance, "value": values[parameter]})
    return values, sources


def _slot_names(k: int, g: int) -> tuple[str, str]:
    return SLOT_JSON_FMT.format(k=k, g=g), SLOT_NPZ_FMT.format(k=k, g=g)


def _partition(spec: RunSpec, generation: int, k: int) -> int:
    return spec.partition_base + 10_000_000 * int(generation) + 100_000 * int(k)


def _generation_point(contract: RC.RunContract, generation: int) -> FlavorParams:
    points = RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"]
    for p in points:
        if int(p["generation"]) == int(generation):
            return FC.flavor_from_dict(p["flavor"])
    contract.refuse(f"generation {generation} has no registered flavor point")


def _job(seed, n_events, partition, topology, flavor_dict):
    return FC.generate_mixture_slot(seed, n_events, partition, topology, FC.flavor_from_dict(flavor_dict))


# ----------------------------------------------------------------------
def step_pre_registration(spec: RunSpec, run_dir):
    contract = _contract(spec, run_dir)
    ledger = RC.seed_ledger(NESTED_ROOT)
    disjoint = RC.check_seed_disjointness(ledger, spec.ledger_run, (spec.raw_first, spec.raw_last), spec.partition_base)
    topology, topo_prov = topology_substrate()
    fixed, sources = inherited_flavor(spec)
    frozen = frozen_digests()
    payload = {
        "description": spec.description, "validation_only": spec.validation_only,
        "statistics": [dict(s) for s in spec.statistics], "free_parameters": list(spec.free), "fixed_parameters": fixed,
        "inherited_from": sources, "priors": {k: {"center": v[0], "sigma_ln": v[1]} for k, v in spec.priors.items()},
        "starts": [list(s) for s in spec.starts], "topology_substrate": topology, "topology_provenance": topo_prov,
        "statistics_tier": {"replicas": spec.replicas, "events_per_replica": spec.events_per_replica},
        "seeds": {"master": 20260928, "raw_first": spec.raw_first, "raw_last": spec.raw_last, "rule": "raw seed = raw_first + replica",
                  "partition_base": spec.partition_base, "event_id_rule": "partition_base + 10000000 * generation + 100000 * replica + i", "disjointness": disjoint},
        "trust_region": {"ess_floor": FC.ESS_FLOOR, "ess_interior": FC.ESS_INTERIOR, "max_generations": spec.max_generations},
        "normalization_prior": FC.NORM_PRIOR, "window": PAPER_WINDOW, "readout_species": list(spec.readout_species),
        "moments": [list(m) for m in spec.moments], "stability_ratio_bound": spec.stability_ratio_bound,
        "conditional_trigger": spec.conditional_trigger, "frozen_substrate": frozen,
    }
    trigger_record = None
    if spec.conditional_trigger:
        trig = spec.conditional_trigger
        source, src_prov = _sealed_report(trig["source_run"])
        pull_rms = float(source["readout_at_optimum"][trig["readout_species"]]["pull_rms"])
        fired = bool(pull_rms > float(trig["pull_rms_threshold"]))
        trigger_record = {**trig, "observed_pull_rms": pull_rms, "fired": fired, "source": src_prov}
        payload["trigger_evaluation"] = trigger_record
    contract.write_pre_registration(payload, {rel: {"sha256": d, "verified": True} for rel, d in frozen.items()},
                                    {"topology_provenance": topo_prov, "inherited_from": sources})
    if trigger_record is not None and not trigger_record["fired"]:
        RC.write_json(contract.path(NOT_TRIGGERED_NAME), {"schema": f"{contract.schema_prefix}.not_triggered/1", **contract.identity(),
                                                          "trigger": trigger_record, "written_utc": RC.now_utc()})
        print(f"[{spec.run_id}] NOT_TRIGGERED: {trig['readout_species']} pull rms {pull_rms:.3f} <= {trig['pull_rms_threshold']}; the run records the trigger and stops", flush=True)
        return
    RC.write_json(contract.path(GENERATION_POINTS_NAME), {"points": [{"generation": 0, "flavor": fixed}]})


def _verify(contract, spec: RunSpec | None = None):
    pre = contract.load_pre_registration()
    status = RC.verify_frozen_substrate(MONO_ROOT, {rel: (rel, sha) for rel, sha in pre["frozen_substrate"].items()}, prefix=contract.prefix, skip={RC.SEED_LEDGER_MONO_REL})
    if spec is not None:
        RC.check_seed_disjointness(RC.seed_ledger(NESTED_ROOT), spec.ledger_run, (spec.raw_first, spec.raw_last), spec.partition_base)
    return status, pre


def step_generate(spec: RunSpec, run_dir, generation: int | None, replicas: Sequence[int] | None, workers):
    contract = _contract(spec, run_dir)
    _, pre = _verify(contract, spec)
    pre_sha = RC.sha256_file(contract.path(RC.PREREGISTRATION_NAME))
    if generation is None:
        generation = max(int(p["generation"]) for p in RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"])
    flavor = _generation_point(contract, generation)
    todo = list(range(spec.replicas)) if not replicas else [int(k) for k in replicas]
    jobs = [k for k in todo if not contract.path(_slot_names(k, generation)[0]).is_file()]
    if not jobs:
        print(f"[{spec.run_id}] generation {generation} complete", flush=True); return
    t0 = time.perf_counter()
    results = RC.run_replicas(_job, [(spec.raw_first + k, spec.events_per_replica, _partition(spec, generation, k), pre["topology_substrate"], FC.flavor_to_dict(flavor)) for k in jobs], max_workers=workers)
    for k, result in zip(jobs, results):
        json_name, npz_name = _slot_names(k, generation)
        np.savez(contract.path(npz_name), **result["arrays"])
        meta = {"schema": f"{contract.schema_prefix}.replica/1", **contract.identity(), "pre_registration_sha256": pre_sha,
                "generation": int(generation), "replica": int(k), **{key: v for key, v in result.items() if key != "arrays"},
                "arrays_file": npz_name, "arrays_sha256": FC.arrays_digest(result["arrays"]), "written_utc": RC.now_utc()}
        RC.write_json(contract.path(json_name), meta)
    print(f"[{spec.run_id}] generation {generation}: {len(jobs)} replicas in {time.perf_counter() - t0:.0f} s", flush=True)


def _load_generation(contract, spec: RunSpec, generation: int, tables: dict, *, smoke: bool = False) -> FC.FlavorSample:
    slots = []
    names = [(f"smoke_replica_{k}.json", f"smoke_replica_{k}.npz") for k in range(2)] if smoke else [_slot_names(k, generation) for k in range(spec.replicas)]
    for json_name, npz_name in names:
        meta = RC.read_json(contract.path(json_name))
        with np.load(contract.path(npz_name)) as data:
            arrays = {key: data[key] for key in data.files}
        if FC.arrays_digest(arrays) != meta["arrays_sha256"]:
            contract.refuse(f"{npz_name} does not match its recorded digest")
        slots.append((meta, arrays))
    return FC.FlavorSample(slots, tables)


def _latest_generation(contract, spec: RunSpec) -> int:
    points = RC.read_json(contract.path(GENERATION_POINTS_NAME))["points"]
    complete = [int(p["generation"]) for p in points if all(contract.path(_slot_names(k, int(p["generation"]))[0]).is_file() for k in range(spec.replicas))]
    if not complete:
        contract.refuse("no complete generation on disk")
    return max(complete)


def _readouts(spec: RunSpec, sample: FC.FlavorSample, weights: list, tables: dict) -> dict:
    out = {}
    for name in spec.readout_species:
        table = tables[name]
        model = np.zeros(table.data.size)
        for k, block in enumerate(sample.blocks):
            model += sample.species_prediction(block, weights[k], name) / sample.n_blocks
        out[name] = {"bins": int(table.data.size), "model_sum": float(np.sum(model)), "data_sum": float(np.sum(table.data)),
                     "sum_ratio": float(np.sum(model) / np.sum(table.data)),
                     "chi2_data_only": float(np.sum(((table.data - model) / table.sigma) ** 2)),
                     "pull_rms": float(np.sqrt(np.mean(((table.data - model) / table.sigma) ** 2)))}
    if {"k_plus", "k_minus"} <= set(out):
        out["k_plus_over_k_minus"] = {"model": out["k_plus"]["model_sum"] / max(out["k_minus"]["model_sum"], 1e-300),
                                      "data": out["k_plus"]["data_sum"] / out["k_minus"]["data_sum"]}
    if {"proton", "antiproton"} <= set(out):
        out["antiproton_over_proton"] = {"model": out["antiproton"]["model_sum"] / max(out["proton"]["model_sum"], 1e-300),
                                         "data": out["antiproton"]["data_sum"] / out["proton"]["data_sum"]}
    moments = {}
    for species, rel in spec.moments:
        moments[species] = FC.kaon_pt_moments(sample, weights, species, NESTED_ROOT / rel)
    out["pt_moments_held_out"] = moments
    return out


def evaluate_generation(spec: RunSpec, contract, generation: int, pre: dict, *, smoke: bool = False) -> dict:
    tables = OB.load_tables(NESTED_ROOT)
    sample = _load_generation(contract, spec, generation, tables, smoke=smoke)
    fixed = FC.flavor_from_dict(pre["fixed_parameters"])
    objective = FC.FlavorObjective(sample, spec.statistics, spec.free, fixed, spec.priors)
    identity_value, identity_details = objective.evaluate(objective.x_from_params(fixed), details=True)
    residuals = []
    if spec.validation_only or not spec.free:
        x_opt = objective.x_from_params(fixed); pooled = {"converged": True, "nfev": 0, "message": "frozen comparison", "starts": []}
        hess = {"quoted": False, "positive_definite": None, "errors": None, "skipped": "no free parameter"}
        blocks = []; stability = {}
    else:
        starts = [np.log(np.asarray(s, dtype=float)) for s in spec.starts][: (2 if smoke else len(spec.starts))]
        pooled = FC.run_fit(objective, starts, maxfev=(300 if smoke else 3000))
        x_opt = pooled["x"]
        hess = FC.hessian(objective, x_opt) if not smoke else {"quoted": False, "positive_definite": False, "errors": None, "skipped": "smoke"}
        blocks = []
        if not smoke and sample.n_blocks > 1:
            for k in range(sample.n_blocks):
                res = FC.run_fit(objective, [x_opt], blocks=[k], maxfev=1500)
                blocks.append({"block": k, **FC.flavor_to_dict(objective.params_from_x(res["x"])), "fun": res["fun"], "converged": res["converged"]})
        stability = {}
        if blocks and hess.get("errors"):
            for name in spec.free:
                vals = np.asarray([b[name] for b in blocks]); spread = float(np.std(vals, ddof=1))
                expected = hess["errors"][name]["sigma"] * math.sqrt(len(blocks)); ratio = spread / expected if expected > 0 else float("inf")
                stability[name] = {"block_values": vals.tolist(), "spread": spread, "expected_spread": expected, "ratio": ratio,
                                   "stable": bool(ratio <= spec.stability_ratio_bound)}
        if not pooled["converged"]:
            residuals.append({"kind": "optimizer_non_convergence", "message": pooled["message"]})
        if not hess.get("positive_definite"):
            residuals.append({"kind": "block_curvature_failure", "eigenvalues": hess.get("eigenvalues")})
        for name, s in stability.items():
            if not s["stable"]:
                residuals.append({"kind": "replica_block_minima_disagree", "parameter": name, "ratio": s["ratio"]})
    value, det = objective.evaluate(x_opt, details=True)
    params = objective.params_from_x(x_opt)
    for name in spec.free:
        v = getattr(params, name)
        if v <= math.exp(FC.LOG_BOUNDS[0]) * 1.001 or v >= math.exp(FC.LOG_BOUNDS[1]) * 0.999:
            residuals.append({"kind": "parameter_on_boundary", "parameter": name, "value": v})
    if det["ess_min"] < FC.ESS_FLOOR:
        residuals.append({"kind": "weight_degeneracy", "ess_min": det["ess_min"]})
    regenerate = (not smoke) and (not spec.validation_only) and det["ess_min"] < FC.ESS_INTERIOR and generation < spec.max_generations - 1
    weights, _ = objective.weights(params)
    readout = _readouts(spec, sample, weights, tables)
    weights0, _ = objective.weights(fixed)
    readout_identity = _readouts(spec, sample, weights0, tables) if flavor_dict_differs(params, fixed) else None
    # Gate 6 clause: improves integrated ratios but worsens differential shapes?
    shape_check = None
    if readout_identity is not None:
        stat_species = [s["species"] for s in spec.statistics]
        ratio_before = np.mean([abs(math.log(readout_identity[s]["sum_ratio"])) for s in stat_species])
        ratio_after = np.mean([abs(math.log(readout[s]["sum_ratio"])) for s in stat_species])
        shape_check = {"integrated_ratio_log_deviation": {"before": float(ratio_before), "after": float(ratio_after)},
                       "profiled_shape_minus2lnl": {"before": float(identity_details["minus2lnl_data"]), "after": float(det["minus2lnl_data"])},
                       "improves_integrated": bool(ratio_after < ratio_before), "worsens_shape": bool(det["minus2lnl_data"] > identity_details["minus2lnl_data"]),
                       "reject_flag": bool(ratio_after < ratio_before and det["minus2lnl_data"] > identity_details["minus2lnl_data"])}
        if shape_check["reject_flag"]:
            residuals.append({"kind": "gate6_integrated_improves_shape_worsens", **shape_check})
    prior_dominance = None
    if hess.get("errors"):
        prior_dominance = {}
        for name, (center, sigma_ln) in spec.priors.items():
            if name in hess["errors"]:
                post = hess["errors"][name]["sigma_ln"]
                prior_dominance[name] = {"prior_sigma_ln": sigma_ln, "posterior_sigma_ln": post, "ratio": post / sigma_ln, "prior_dominated": bool(post / sigma_ln > 0.8)}
    constrained = bool(spec.free) and bool(pooled["converged"] and det["ess_min"] >= FC.ESS_FLOOR and hess.get("positive_definite")
                                           and stability and all(s["stable"] for s in stability.values())
                                           and not any(r["kind"] == "parameter_on_boundary" for r in residuals))
    return {
        "generation": int(generation), "generation_flavor_point": FC.flavor_to_dict(sample.generation_point),
        "statistics": [{"name": s["name"], "species": s["species"], "minus": s.get("minus"), "bins": int(b)} for s, b in zip(spec.statistics, [st["bins"] for st in objective.stats])],
        "topology_substrate": pre["topology_substrate"], "topology_provenance": pre.get("topology_provenance"),
        "fixed_parameters": pre["fixed_parameters"], "free_parameters": list(spec.free),
        "fitted": FC.flavor_to_dict(params), "objective_at_optimum": {"value": value, **{k: v for k, v in det.items() if k in ("ess_min", "eta_hat", "minus2lnl_data", "prior_terms", "in_trust_region")}},
        "objective_at_fixed_point": {"value": identity_value, **{k: v for k, v in identity_details.items() if k in ("ess_min", "eta_hat", "minus2lnl_data")}},
        "optimizer": {"method": "Powell (bounded, log parameters)", "converged": pooled["converged"], "nfev": pooled["nfev"], "message": pooled["message"], "start_records": pooled["starts"]},
        "hessian": hess, "block_fits": blocks, "replica_block_stability": stability, "prior_dominance": prior_dominance,
        "verdict": {"constrained": constrained, "regeneration_requested": regenerate, "validation_only": spec.validation_only},
        "gate6_shape_check": shape_check, "named_residual_register": residuals,
        "readout_at_optimum": readout, "readout_at_fixed_point": readout_identity,
        "per_bin": {"d": objective.d.tolist(), "m": det["m"].tolist(), "se": det["se"].tolist(), "v_data": objective.v_data.tolist(), "pull": det["pull"].tolist()},
        "statistics_tier": {"replicas": sample.n_blocks, "events_per_replica": int(sample.blocks[0]["n"])},
    }


def flavor_dict_differs(a: FlavorParams, b: FlavorParams) -> bool:
    return FC.flavor_to_dict(a) != FC.flavor_to_dict(b)


def render_report_md(spec: RunSpec, report: dict) -> str:
    lines = [f"# {spec.run_id}", "", f"Card {spec.card}. {spec.description}", "",
             f"Topology substrate: R51c point {json.dumps(report['topology_substrate'])} ({report['topology_provenance'].get('verdict') if report.get('topology_provenance') else 'n/a'}).", "",
             "| parameter | value | error | free |", "|---|---|---|---|"]
    errors = (report["hessian"] or {}).get("errors") or {}
    for name in FC.FLAVOR_NAMES:
        err = errors.get(name); sig = "" if not err else f"{err['sigma']:.3g}"
        lines.append(f"| {name} | {report['fitted'][name]:.5g} | {sig} | {'yes' if name in report['free_parameters'] else 'no'} |")
    o = report["objective_at_optimum"]
    lines += ["", f"Objective {o['value']:.3f} (data {o['minus2lnl_data']:.3f}); eta_hat {o['eta_hat']:.4f}; smallest ESS {o['ess_min']:.3f}; converged {report['optimizer']['converged']}; Hessian PD {report['hessian'].get('positive_definite')}.",
              f"Verdict: {'CONSTRAINED' if report['verdict']['constrained'] else ('VALIDATION' if report['verdict']['validation_only'] else 'NOT_CONSTRAINED')}.", "",
              "## Readout at the optimum (paper window)", "", "| species | bins | sum ratio | pull rms |", "|---|---|---|---|"]
    for name, r in report["readout_at_optimum"].items():
        if isinstance(r, dict) and "bins" in r:
            lines.append(f"| {name} | {r['bins']} | {r['sum_ratio']:.3f} | {r['pull_rms']:.2f} |")
    for key in ("k_plus_over_k_minus", "antiproton_over_proton"):
        if key in report["readout_at_optimum"]:
            r = report["readout_at_optimum"][key]; lines.append(f"\n{key}: model {r['model']:.3f}, data {r['data']:.3f}")
    lines += ["", "## Named residuals", ""] + [f"- {json.dumps(r, sort_keys=True, default=str)[:400]}" for r in report["named_residual_register"] or [{"kind": "none"}]]
    return "\n".join(lines) + "\n"


def step_evaluate(spec: RunSpec, run_dir):
    contract = _contract(spec, run_dir)
    _, pre = _verify(contract, spec)
    generation = _latest_generation(contract, spec)
    report = evaluate_generation(spec, contract, generation, pre)
    if report["verdict"]["regeneration_requested"]:
        nxt = generation + 1
        points = RC.read_json(contract.path(GENERATION_POINTS_NAME))
        points["points"] = [p for p in points["points"] if int(p["generation"]) != nxt] + [{"generation": nxt, "flavor": report["fitted"], "requested_by_generation": generation, "ess_min": report["objective_at_optimum"]["ess_min"]}]
        RC.write_json(contract.path(GENERATION_POINTS_NAME), points)
        RC.write_json(contract.path(f"fit_report_g{generation}.json"), report)
        RC.write_json(contract.path(REGENERATION_NAME), {"schema": f"{contract.schema_prefix}.regeneration/1", **contract.identity(), "from_generation": generation, "to_generation": nxt, "flavor": report["fitted"], "written_utc": RC.now_utc()})
        print(f"[{spec.run_id}] generation {generation}: ess_min {report['objective_at_optimum']['ess_min']:.3f}; regeneration {nxt} at {json.dumps(report['fitted'])}", flush=True)
        return
    report["generation_history"] = sorted(str(p.name) for p in contract.run_dir.glob("fit_report_g*.json"))
    body = contract.write_report(report, render_report_md(spec, report))
    if contract.path(REGENERATION_NAME).is_file():
        contract.path(REGENERATION_NAME).unlink()
    print(f"[{spec.run_id}] report written (generation {generation}; fitted {json.dumps(report['fitted'])}; verdict constrained={report['verdict']['constrained']})", flush=True)


def step_gate(spec: RunSpec, run_dir):
    contract = _contract(spec, run_dir)
    contract.require(RC.REPORT_NAME)
    report = RC.read_json(contract.path(RC.REPORT_NAME))
    record = {"schema": f"{contract.schema_prefix}.gate/1", **contract.identity(), "sealed": True, "evidence_only": True,
              "gate": "Gate 6 (plan section 12)", "verdict": ("VALIDATION" if report["verdict"]["validation_only"] else ("CONSTRAINED" if report["verdict"]["constrained"] else "NOT_CONSTRAINED")),
              "fitted": report["fitted"], "errors": (report["hessian"] or {}).get("errors"), "shape_check": report.get("gate6_shape_check"),
              "prior_dominance": report.get("prior_dominance"), "named_residual_register": report["named_residual_register"],
              "readout": {k: v for k, v in report["readout_at_optimum"].items() if k != "pt_moments_held_out"},
              "pt_moments_pull_rms": {k: v.get("pull_rms") for k, v in report["readout_at_optimum"]["pt_moments_held_out"].items()},
              "report_digest": report["digest"], "written_utc": RC.now_utc()}
    if spec.conditional_trigger:
        record["conditional_trigger"] = spec.conditional_trigger
    RC.write_json(contract.path(spec.gate_record), record)
    print(f"[{spec.run_id}] gate record: {record['verdict']}", flush=True)


def step_clean_validate(spec: RunSpec, run_dir):
    contract = _contract(spec, run_dir)
    contract.require(RC.REPORT_NAME, spec.gate_record)
    frozen, pre = _verify(contract, spec)
    sealed = RC.read_json(contract.path(RC.REPORT_NAME))
    generation = _latest_generation(contract, spec)
    rerun = evaluate_generation(spec, contract, generation, pre); rerun["generation_history"] = sealed.get("generation_history", [])
    rerun_digest = contract.report_digest_of({**rerun, "schema": sealed["schema"], **contract.identity()})
    flavor = _generation_point(contract, generation)
    probe = FC.generate_mixture_slot(spec.raw_first, spec.events_per_replica, _partition(spec, generation, 0), pre["topology_substrate"], flavor)
    sealed_probe = RC.read_json(contract.path(_slot_names(0, generation)[0]))
    probe_match = FC.arrays_digest(probe["arrays"]) == sealed_probe["arrays_sha256"] and probe["event_rolling_sha256"] == sealed_probe["event_rolling_sha256"]
    contract.write_clean_validation({"commands": [f"python helpers/benchmarking/{HERE.name} --clean-validate --run-dir {contract.run_dir}"],
                                     "report_digest": sealed["digest"], "report_rerun_digest": rerun_digest, "report_hash_match": bool(rerun_digest == sealed["digest"]),
                                     "probe_replica": {"generation": generation, "replica": 0, "arrays_match": bool(probe_match)}, "frozen_substrate": frozen,
                                     "seal": {"state": "pending (the seal lands over this record)"}})
    if rerun_digest != sealed["digest"] or not probe_match:
        contract.refuse("the clean validation does not reproduce the sealed report or probe")
    print(f"[{spec.run_id}] clean validation: report match True, probe match True", flush=True)


def step_seal(spec: RunSpec, run_dir):
    contract = _contract(spec, run_dir)
    if contract.path(NOT_TRIGGERED_NAME).is_file():
        contract.forbid(RC.SEAL_NAME)
        contract.require(RC.PREREGISTRATION_NAME, RC.PROVENANCE_NAME, NOT_TRIGGERED_NAME)
        contract.path(RC.REPORT_NAME).write_text(json.dumps({"schema": f"{contract.schema_prefix}.report/1", **contract.identity(), "not_triggered": True}, indent=1, sort_keys=True) + "\n")
        contract.path(RC.REPORT_MD_NAME).write_text(f"# {spec.run_id}\n\nNOT_TRIGGERED: see not_triggered.json.\n")
        contract.write_clean_validation({"not_triggered": True, "commands": []})
        contract.seal([NOT_TRIGGERED_NAME], "replica_")
        return
    generation = _latest_generation(contract, spec)
    required = [spec.gate_record, GENERATION_POINTS_NAME]
    for g in range(generation + 1):
        for k in range(spec.replicas):
            required.extend(_slot_names(k, g))
    contract.seal(required, "replica_")


def step_smoke(spec: RunSpec, run_dir: Path, topology: dict | None = None):
    contract = _contract(spec, run_dir)
    if contract.run_dir == MONO_ROOT / "dev" / "00runs" / spec.run_id:
        contract.refuse("--smoke needs an explicit --run-dir")
    contract.run_dir.mkdir(parents=True, exist_ok=True)
    topology = topology or {"fractions": {"sd_projectile": 0.1, "sd_target": 0.1, "dd": 0.05, "nd": 0.75}, "epsilon": 0.0, "b_sd": 8.504140762083612}
    fixed = {name: 1.0 for name in FC.FLAVOR_NAMES}
    pre = {"topology_substrate": topology, "fixed_parameters": fixed, "topology_provenance": {"verdict": "smoke"}}
    RC.write_json(contract.path("smoke_pre_registration.json"), {"schema": f"{contract.schema_prefix}.smoke_pre_registration/1", **contract.identity(), **pre, "frozen_substrate": frozen_digests()})
    results = RC.run_replicas(_job, [(spec.smoke_seed + k, spec.smoke_events, spec.smoke_partition + 100_000 * k, topology, fixed) for k in range(2)])
    for k, result in enumerate(results):
        np.savez(contract.path(f"smoke_replica_{k}.npz"), **result["arrays"])
        RC.write_json(contract.path(f"smoke_replica_{k}.json"), {"schema": f"{contract.schema_prefix}.smoke_replica/1", "replica": k, **{key: v for key, v in result.items() if key != "arrays"}, "arrays_sha256": FC.arrays_digest(result["arrays"])})
    report = evaluate_generation(spec, contract, 0, pre, smoke=True)
    report["schema"] = f"{contract.schema_prefix}.smoke_fit_report/1"; report["digest"] = RC.canonical_digest(report)
    RC.write_json(contract.path("smoke_fit_report.json"), report)
    contract.path("smoke_REPORT.md").write_text(render_report_md(spec, report), encoding="utf-8")
    print(f"[{spec.run_id}] smoke: fitted {json.dumps(report['fitted'])}, ess_min {report['objective_at_optimum']['ess_min']:.3f}", flush=True)


def main(spec: RunSpec, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"{spec.run_id} ({spec.card})")
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pre-registration", "--generate", "--evaluate", "--gate", "--clean-validate", "--seal", "--smoke"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--replica", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.pre_registration: step_pre_registration(spec, args.run_dir)
    elif args.generate: step_generate(spec, args.run_dir, args.generation, args.replica, args.workers)
    elif args.evaluate: step_evaluate(spec, args.run_dir)
    elif args.gate: step_gate(spec, args.run_dir)
    elif args.clean_validate: step_clean_validate(spec, args.run_dir)
    elif args.seal: step_seal(spec, args.run_dir)
    elif args.smoke:
        if args.run_dir is None: RC.refuse("--smoke requires --run-dir", prefix=HERE.stem)
        step_smoke(spec, args.run_dir)
    return 0
