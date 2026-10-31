#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/common.sh"

steps=(
  "00-download_spshadrons"
  "01-create_python_env"
  "02-build_external_generators"
  "03-run_external_baselines"
  "04-run_balanced_constrained_fit"
  "05-make_bestmix_plots_and_tables"
  "06-scan_absyield_prior_stiffness"
  "07-scan_table9_fraction"
  "08-scan_nstar_extended_levels"
  "09-study_component_identifiability"
  "10-scan_nstar_smoothness"
  "11-bootstrap_robustness"
  "12-assemble_paper"
  "13-build_pdf"
)

if [[ "$#" -gt 0 ]]; then
  steps=("$@")
fi

for step in "${steps[@]}"; do
  log "running $step"
  "$ROOT/$step/run.sh"
done
