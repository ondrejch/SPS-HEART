#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

out_dir="$BALANCED_RUN_DIR/component_identifiability"
if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/component_identifiability_overview.png" ]]; then
  log "reusing component identifiability outputs"
else
  log "running component identifiability study"
  run_python "$REPO_ROOT/helpers/benchmarking/study_component_identifiability.py" \
    --run-dir "$BALANCED_RUN_DIR" \
    --out-dir "$out_dir" \
    --run-reduced-basis
fi

log "component identifiability ready: $out_dir"
