#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

plot_png="$BOOTSTRAP_RUN_DIR/bootstrap_model_bands_pdg_balanced_scan.png"
if [[ "$SKIP_EXISTING" == "1" && -s "$plot_png" ]]; then
  log "reusing bootstrap robustness outputs"
else
  log "running bootstrap robustness study"
  run_python "$REPO_ROOT/helpers/benchmarking/bootstrap_multispecies_robustness.py" \
    --run-dir "$BALANCED_RUN_DIR" \
    --label "pdg_balanced_scan" \
    --n-bootstrap "$BOOTSTRAP_REPLICATES" \
    --seed 20260331 \
    --out-dir "$BOOTSTRAP_RUN_DIR"
fi

log "bootstrap summary ready: $BOOTSTRAP_RUN_DIR"
