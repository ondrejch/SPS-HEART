#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

ensure_dir "$BALANCED_ASSET_DIR"

plot_species() {
  local species="$1"
  local out_png="$2"
  log "plotting $species spectra panel"
  run_python "$REPO_ROOT/helpers/benchmarking/plot_multispecies_sharedfit_spectra.py" \
    --run-description "$BALANCED_RUN_NAME" \
    --species "$species" \
    --out-png "$out_png"
}

plot_species "pi-" "$BALANCED_ASSET_DIR/piminus_spectra_sharedfit_bestmix.png"
plot_species "pi+" "$BALANCED_ASSET_DIR/piplus_spectra_sharedfit_bestmix.png"
plot_species "k+" "$BALANCED_ASSET_DIR/kplus_spectra_sharedfit_bestmix.png"
plot_species "k-" "$BALANCED_ASSET_DIR/kminus_spectra_sharedfit_bestmix.png"
plot_species "p" "$BALANCED_ASSET_DIR/p_spectra_sharedfit_bestmix.png"
plot_species "pbar" "$BALANCED_ASSET_DIR/antiproton_spectra_sharedfit_bestmix.png"

log "building six-species spectra grid"
run_python "$REPO_ROOT/helpers/benchmarking/make_sharedfit_species_grid.py" \
  --fig-dir "$BALANCED_ASSET_DIR" \
  --out-png "$BALANCED_ASSET_DIR/spectra_sharedfit_bestmix_all6_species_grid.png"

log "building particle-ratio figure"
run_python "$REPO_ROOT/helpers/benchmarking/plot_multispecies_particle_ratios.py" \
  --run-dir "$BALANCED_RUN_DIR" \
  --out-png "$BALANCED_ASSET_DIR/particle_ratio_by_xf_window_bestmix.png"

log "building seagull profile figure"
run_python "$REPO_ROOT/helpers/benchmarking/plot_multispecies_seagull.py" \
  --run-dir "$BALANCED_RUN_DIR" \
  --split full \
  --min-xf 0.0 \
  --out-grid-png "$BALANCED_ASSET_DIR/seagull_xf_meanpt_all6_species_grid.png" \
  --with-slope-summary \
  --out-slope-png "$BALANCED_ASSET_DIR/seagull_xf_slope_all6_species.png" \
  --out-slope-csv "$BALANCED_ASSET_DIR/seagull_xf_slope_all6_species.csv"

log "balanced-run plot assets ready: $BALANCED_ASSET_DIR"
