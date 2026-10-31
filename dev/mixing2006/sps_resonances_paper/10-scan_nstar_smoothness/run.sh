#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

metrics_png="$NSTAR_SMOOTH_RUN_DIR/nstar_smooth_prior_scan_metrics.png"
if [[ "$SKIP_EXISTING" == "1" && -s "$metrics_png" ]]; then
  log "reusing N* smoothness scan outputs"
else
  log "running N* smoothness scan"
  run_python "$REPO_ROOT/helpers/benchmarking/run_nstar_smooth_prior_scan.py" \
    --base-manifest "$BALANCED_RUN_DIR/constrained_refit_manifest.json" \
    --out-root "$NSTAR_SMOOTH_RUN_DIR" \
    --lambdas "$NSTAR_SMOOTH_LAMBDAS" \
    --python "$PYTHON_BIN"
fi

log "N* smoothness summary ready: $NSTAR_SMOOTH_RUN_DIR"
