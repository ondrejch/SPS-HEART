"""Freeze the p+p baseline (plan section 23, Round 7: "freeze the p+p
baseline"; plan section 13.4 acceptance criteria).

Reads the sealed R51b, R60 to R65, and P7 archives, writes
``config/v2_singlet/pp_baseline_<date>.json`` with the frozen topology
and flavor parameters, their errors, the archive digests, and the
acceptance table of plan section 13.4 evaluated against the sealed R01
legacy reference; prints the Markdown summary.  Refuses on a missing or
broken seal.  Run from either tree:

    python helpers/benchmarking/freeze_pp_baseline.py [--date YYYY-MM-DD] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import run_contract as RC  # noqa: E402

NESTED_ROOT, MONO_ROOT = RC.locate_roots(HERE)
RUNS = MONO_ROOT / "dev" / "00runs"
R01 = RUNS / "r01_legacy_pp_reference_2026-09-11"
ARCHIVES = {
    "r51b": "r51b_topology_fraction_reconstraint_2026-09-28", "r60": "r60_strange_source_grid_2026-09-28",
    "r61": "r61_pair_strangeness_fit_2026-09-28", "r62": "r62_associated_strangeness_fit_2026-09-28",
    "r63": "r63_vector_hidden_strangeness_fit_2026-09-28", "r64": "r64_baryon_pair_fit_2026-09-28",
    "r65": "r65_flavor_closure_validation_2026-09-28", "p7": "r7_p7_joint_flavor_refinement_2026-09-28",
}
SPECIES = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
BOUND_MARGIN = 1.001


def sealed(run_id: str) -> tuple[Path, dict]:
    run_dir = RUNS / run_id
    if not (run_dir / "seal_manifest.json").is_file():
        RC.refuse(f"{run_id} is not sealed", prefix=HERE.stem)
    check = RC.verify_sha256sums(run_dir)
    if check["mismatches"]:
        RC.refuse(f"{run_id}: checksum mismatches {check['mismatches'][:3]}", prefix=HERE.stem)
    return run_dir, check


def legacy_chi2_by_species() -> dict:
    residuals = RC.read_json(R01 / "residuals.json")
    out = {}
    for row in residuals:
        out.setdefault(row["species"], {"chi2": 0.0, "bins": 0})
        out[row["species"]]["chi2"] += float(row["delta_over_sigma"]) ** 2
        out[row["species"]]["bins"] += 1
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="freeze the p+p baseline")
    parser.add_argument("--date", default="2026-09-28")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    provenance = {}
    reports = {}
    for key, run_id in ARCHIVES.items():
        run_dir, check = sealed(run_id)
        report_name = "report.json"  # every run contract writes its final report as report.json
        report = RC.read_json(run_dir / report_name)
        reports[key] = report
        provenance[key] = {"run": run_id, "report": report_name, "report_digest": report.get("digest"),
                           "seal_entries": check["entries"], "not_triggered": bool(report.get("not_triggered"))}
    r51b = reports["r51b"]; p7 = reports["p7"]; r65 = reports["r65"]
    topo_err = (r51b["hessian"] or {}).get("errors") or {}
    flavor = dict(p7["fitted"])
    flavor_err = {}
    for key in ("r61", "r62", "r63", "r64", "p7"):
        rep = reports[key]
        if rep.get("not_triggered"):
            continue
        for name, err in ((rep.get("hessian") or {}).get("errors") or {}).items():
            flavor_err[name] = {"sigma": err["sigma"], "from": ARCHIVES[key]}
    # acceptance table (plan section 13.4)
    legacy = legacy_chi2_by_species()
    readout = r65["readout_at_optimum"]
    improvement = {}
    for sp in SPECIES:
        if sp in readout and sp in legacy:
            model_chi2 = float(readout[sp]["chi2_data_only"]); bins = int(readout[sp]["bins"])
            leg = legacy[sp]
            improvement[sp] = {"legacy_chi2_per_bin": leg["chi2"] / leg["bins"], "model_chi2_per_bin": model_chi2 / bins,
                               "ratio_model_over_legacy": (model_chi2 / bins) / (leg["chi2"] / leg["bins"]),
                               "sum_ratio_model_over_data": readout[sp]["sum_ratio"]}
    on_bound = []
    for name, value in flavor.items():
        if value <= 0.05 * BOUND_MARGIN or value >= 20.0 / BOUND_MARGIN:
            on_bound.append(name)
    for name in ("f_sd_total", "f_dd", "f_nd"):
        v = r51b["fitted"][name]
        if v <= 0.005 * BOUND_MARGIN or v >= 0.995 / BOUND_MARGIN:
            on_bound.append(name)
    stable = all(s.get("stable", False) for rep in (r51b, p7) for s in (rep.get("replica_block_stability") or {}).values())
    criteria = {
        "all_six_species_improve_substantially_vs_legacy_v1": {"pass": all(v["ratio_model_over_legacy"] < 0.5 for v in improvement.values()) if improvement else False, "detail": improvement},
        "no_species_described_solely_by_normalization": {"pass": all(0.5 <= v["sum_ratio_model_over_data"] <= 2.0 for v in improvement.values()) if improvement else False,
                                                          "detail": {sp: v["sum_ratio_model_over_data"] for sp, v in improvement.items()}},
        "leading_baryons_and_multiplicity_simultaneously": {"pass": None, "detail": "leading protons fit in R51b (net-proton and proton-only bins); no local multiplicity distribution: recorded as prediction only"},
        "kaon_and_proton_ratios_differential": {"pass": bool(readout.get("k_minus", {}).get("pull_rms", 99) < 3.0 and readout.get("antiproton", {}).get("pull_rms", 99) < 3.0),
                                                 "detail": {"k_plus_over_k_minus": readout.get("k_plus_over_k_minus"), "antiproton_over_proton": readout.get("antiproton_over_proton"),
                                                            "pull_rms": {sp: readout[sp]["pull_rms"] for sp in SPECIES if sp in readout}}},
        "seagull_trends": {"pass": None, "detail": "no seagull comparison in the Round 6/7 archives; carried from R55 (predictions recorded)"},
        "held_out_not_systematically_worse": {"pass": None, "detail": {"pt_moments_pull_rms": {k: v.get("pull_rms") for k, v in readout.get("pt_moments_held_out", {}).items()}}},
        "fitted_parameters_not_on_bounds": {"pass": not on_bound, "detail": on_bound},
        "stable_across_independent_mc_blocks": {"pass": stable, "detail": {"r51b": r51b.get("replica_block_stability"), "p7": p7.get("replica_block_stability")}},
    }
    baseline = {
        "label": f"pp_baseline_{args.date}", "purpose": "the frozen p+p baseline of plan section 23 Round 7 (topology point of R51b, flavor point of P7), with errors, archive provenance, and the plan section 13.4 acceptance table",
        "sqrt_s_gev": 17.27, "beam_momentum_gev_c": 158.0, "sigma_inel_mb": 31.78,
        "topology": {"fractions": {k: r51b["fitted"][k] for k in ("sd_projectile", "sd_target", "f_dd", "f_nd", "f_sd_total")},
                     "epsilon": r51b["fitted"]["epsilon"], "b_sd": r51b["fitted"]["b_sd"], "errors": topo_err,
                     "verdict": "CONSTRAINED" if r51b["verdict"]["constrained"] else "NOT_CONSTRAINED",
                     "named_residual_register": r51b.get("named_residual_register", []), "p_n2": 0.0, "nd_multi": "absent (Gate 5)"},
        "flavor": {"values": flavor, "errors": flavor_err, "hidden_ssbar_content": {"333": 1.0}, "p7_verdict": "CONSTRAINED" if p7["verdict"]["constrained"] else "NOT_CONSTRAINED",
                   "r62_not_triggered": provenance["r62"]["not_triggered"], "r63_prior_dominance": reports["r63"].get("prior_dominance")},
        "frozen_by_gate_decisions": {"blend_ramp_start_gev": 2.680194110727528, "alignment": "isotropic in the topology cascade (Gate 4 SIMPLIFY; R45 point for the Round 4 ablation path)",
                                     "remnant_weights": {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}},
        "validation": {"r65_statistics": r65.get("statistics_tier"), "readout": {k: v for k, v in readout.items() if k != "pt_moments_held_out"}},
        "acceptance_plan_13_4": criteria,
        "acceptance_summary": {"passed": [k for k, v in criteria.items() if v["pass"] is True], "failed": [k for k, v in criteria.items() if v["pass"] is False],
                               "not_assessable_locally": [k for k, v in criteria.items() if v["pass"] is None]},
        "provenance": provenance, "recorded": args.date,
        "status": "frozen baseline of the current model; conditional on the plan owner's Gate 6 decision and on the Round 5 central-excitation revision named in TASK-20260928-r7-fit-p2-disposition",
    }
    out = NESTED_ROOT / "config" / "v2_singlet" / f"pp_baseline_{args.date}.json"
    text = json.dumps(baseline, indent=1, sort_keys=True, default=float) + "\n"
    if args.dry_run:
        print(text[:3000]); return 0
    out.write_text(text)
    print(f"baseline written: {out}")
    print("acceptance:", json.dumps(baseline["acceptance_summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
