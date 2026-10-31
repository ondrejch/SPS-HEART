#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

ensure_dir "$BALANCED_RUN_DIR"
ensure_dir "$TABLE9_CHECK_DIR"

summary_json="$BALANCED_RUN_DIR/multispecies_constrained_summary.json"
WEAK_FEED_POLICY="${WEAK_FEED_POLICY:-primary_only}"
if [[ "$SKIP_EXISTING" != "1" || ! -s "$summary_json" ]]; then
  log "running balanced constrained fit: $BALANCED_RUN_NAME"
  run_python "$REPO_ROOT/helpers/benchmarking/run_constrained_refit.py" \
    --config-json "$CONSTRAINED_CONFIG_JSON" \
    --run-description "$BALANCED_RUN_NAME" \
    --out-dir "$BALANCED_RUN_DIR" \
    --external-run-dir "$EXTERNAL_RUN_DIR" \
    --res-cache-dir "$SHARED_RES_CACHE_DIR" \
    --set "prior_table9_json=$PDG_PRIOR_JSON" \
    --set "prior_abs_yield_lambda=0.3" \
    --set "prior_abs_yield_min_sigma_rel=0.05" \
    --set "fit_objective=gaussian_point_mle" \
    --set "fit_weight_mode=sigma" \
    --set "res_events_per_res=$RES_EVENTS_PER_RES" \
    --set "res_max_decay_passes=$RES_MAX_DECAY_PASSES" \
    --set "res_workers=$RES_WORKERS" \
    --set "weak_feed_policy=$WEAK_FEED_POLICY"
else
  log "reusing balanced constrained fit: $BALANCED_RUN_NAME"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$TABLE9_CHECK_JSON" ]]; then
  log "building Table-9 yield check"
  run_python "$REPO_ROOT/helpers/benchmarking/check_fischer2022_table9_pi_minus.py" \
    --run-description "table9_check_refresh_${PIPELINE_STAMP}" \
    --out-json "$TABLE9_CHECK_JSON" \
    --out-md "$TABLE9_CHECK_DIR/FISCHER2022_TABLE9_PI_MINUS_YIELD_CHECK.md"
else
  log "reusing Table-9 yield check JSON"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$BALANCED_RUN_DIR/table9_pdg_fit_weight_compare.json" ]]; then
  log "comparing Table-9 selectors with fitted balanced weights"
  run_python "$REPO_ROOT/helpers/benchmarking/compare_table9_pdg_fit_weights.py" \
    --summary-json "$BALANCED_RUN_DIR/multispecies_constrained_summary.json" \
    --table9-check-json "$TABLE9_CHECK_JSON" \
    --run-description "$BALANCED_RUN_NAME" \
    --out-csv "$BALANCED_RUN_DIR/table9_pdg_fit_weight_compare.csv" \
    --out-md "$BALANCED_RUN_DIR/TABLE9_PDG_FIT_WEIGHT_COMPARE.md" \
    --out-json "$BALANCED_RUN_DIR/table9_pdg_fit_weight_compare.json"
else
  log "reusing Table-9 weight comparison"
fi

log "balanced constrained fit ready: $BALANCED_RUN_DIR"
