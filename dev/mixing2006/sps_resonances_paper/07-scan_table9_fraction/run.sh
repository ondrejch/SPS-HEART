#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

for mode in table9 pdg; do
  for frac_lam in $TABLE9_FRACTION_LAMBDAS; do
    frac_tag="$(table9_fraction_tag "$frac_lam")"
    run_name="constrained_refit_${mode}_absyield_t9frac_${frac_tag}_${TABLE9_FRACTION_STAMP}"
    run_dir="$RUNS_ROOT/$run_name"
    summary_json="$run_dir/multispecies_constrained_summary.json"
    if [[ "$SKIP_EXISTING" == "1" && -s "$summary_json" ]]; then
      log "reusing $run_name"
      continue
    fi
    log "running $run_name"
    run_python "$REPO_ROOT/helpers/benchmarking/run_constrained_refit.py" \
      --config-json "$CONSTRAINED_CONFIG_JSON" \
      --run-description "$run_name" \
      --out-dir "$run_dir" \
      --external-run-dir "$EXTERNAL_RUN_DIR" \
      --res-cache-dir "$SHARED_RES_CACHE_DIR" \
      --set "prior_table9_json=$([[ "$mode" == "pdg" ]] && printf '%s' "$PDG_PRIOR_JSON" || printf '%s' "$TABLE9_PRIOR_JSON")" \
      --set "prior_table9_lambda=1.0" \
      --set "prior_table9_total_lambda=0.0" \
      --set "prior_table9_fraction_lambda=$frac_lam" \
      --set "prior_table9_fraction_eps=1e-6" \
      --set "prior_abs_yield_lambda=0.3" \
      --set "prior_abs_yield_min_sigma_rel=0.05" \
      --set "fit_objective=gaussian_point_mle" \
      --set "fit_weight_mode=sigma" \
      --set "res_events_per_res=$RES_EVENTS_PER_RES" \
      --set "res_max_decay_passes=$RES_MAX_DECAY_PASSES" \
      --set "res_workers=$RES_WORKERS"
  done
done

log "summarizing Table-9 fraction scan"
run_python "$REPO_ROOT/helpers/benchmarking/summarize_table9_fraction_scan.py" \
  --run-stamp "$TABLE9_FRACTION_STAMP" \
  --out-dir "$TABLE9_FRACTION_SUMMARY_DIR"

log "Table-9 fraction summary ready: $TABLE9_FRACTION_SUMMARY_DIR"
