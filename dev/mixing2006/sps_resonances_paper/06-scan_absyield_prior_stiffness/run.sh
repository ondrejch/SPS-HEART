#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

for mode in table9 pdg; do
  for lam in $ABSYIELD_SCAN_LAMBDAS; do
    for sigma in $ABSYIELD_SCAN_SIGMAS; do
      lam_tag="$(scan_tag_from_float "$lam")"
      sigma_tag="$(scan_tag_from_float "$sigma")"
      run_name="constrained_refit_${mode}_absyield_scan_l${lam_tag}_s${sigma_tag}_${ABSYIELD_SCAN_STAMP}"
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
        --set "prior_abs_yield_lambda=$lam" \
        --set "prior_abs_yield_min_sigma_rel=$sigma" \
        --set "fit_objective=gaussian_point_mle" \
        --set "fit_weight_mode=sigma" \
        --set "res_events_per_res=$RES_EVENTS_PER_RES" \
        --set "res_max_decay_passes=$RES_MAX_DECAY_PASSES" \
        --set "res_workers=$RES_WORKERS"
    done
  done
done

log "summarizing absolute-yield prior stiffness scan"
run_python "$REPO_ROOT/helpers/benchmarking/summarize_absyield_prior_stiffness.py" \
  --run-stamp "$ABSYIELD_SCAN_STAMP" \
  --out-dir "$ABSYIELD_SUMMARY_DIR"

log "absolute-yield scan summary ready: $ABSYIELD_SUMMARY_DIR"
