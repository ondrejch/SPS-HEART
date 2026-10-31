#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

ensure_dir "$PAPER_FIG_DIR"
ensure_dir "$PAPER_GENERATED_DIR"

sync_paper_assets=0
if [[ -n "$PAPER_ROOT" ]]; then
  ensure_dir "$PAPER_ROOT/fig"
  ensure_dir "$PAPER_ROOT/generated"
  sync_paper_assets=1
fi

copy_asset() {
  local src="$1"
  local dst="$2"
  if [[ ! -s "$src" ]]; then
    printf 'missing asset: %s\n' "$src" >&2
    exit 1
  fi
  cp "$src" "$dst"
  log "copied $(basename "$dst")"
  if (( sync_paper_assets == 1 )); then
    cp "$src" "$PAPER_ROOT/fig/$(basename "$dst")"
    log "synced $(basename "$dst") into $PAPER_ROOT/fig"
  fi
}

copy_asset "$BALANCED_ASSET_DIR/piminus_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/piminus_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/piplus_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/piplus_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/kplus_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/kplus_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/kminus_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/kminus_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/p_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/p_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/antiproton_spectra_sharedfit_bestmix.png" "$PAPER_FIG_DIR/antiproton_spectra_sharedfit_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/spectra_sharedfit_bestmix_all6_species_grid.png" "$PAPER_FIG_DIR/spectra_sharedfit_bestmix_all6_species_grid.png"
copy_asset "$BALANCED_ASSET_DIR/particle_ratio_by_xf_window_bestmix.png" "$PAPER_FIG_DIR/particle_ratio_by_xf_window_bestmix.png"
copy_asset "$BALANCED_ASSET_DIR/seagull_xf_meanpt_all6_species_grid.png" "$PAPER_FIG_DIR/seagull_xf_meanpt_all6_species_grid.png"
copy_asset "$ABSYIELD_SUMMARY_DIR/absyield_prior_stiffness_heatmaps.png" "$PAPER_FIG_DIR/absyield_prior_stiffness_heatmaps.png"
copy_asset "$TABLE9_FRACTION_SUMMARY_DIR/table9_fraction_scan_metrics.png" "$PAPER_FIG_DIR/table9_fraction_scan_metrics.png"
copy_asset "$BOOTSTRAP_RUN_DIR/bootstrap_model_bands_pdg_balanced_scan.png" "$PAPER_FIG_DIR/bootstrap_model_bands_bestmix.png"
copy_asset "$NSTAR_SUMMARY_DIR/nstar_likelihood_map_10x_threeway_extended_levels.png" "$PAPER_FIG_DIR/nstar_likelihood_map_10x_threeway_extended_levels.png"
copy_asset "$BALANCED_RUN_DIR/component_identifiability/component_identifiability_overview.png" "$PAPER_FIG_DIR/component_identifiability_overview.png"
copy_asset "$NSTAR_SMOOTH_RUN_DIR/nstar_smooth_prior_scan_metrics.png" "$PAPER_FIG_DIR/nstar_smooth_prior_scan_metrics.png"

log "generating LaTeX table rows"
run_python "$ROOT/extract_paper_tables.py" \
  --run-dir "$BALANCED_RUN_DIR" \
  --out-dir "$PAPER_GENERATED_DIR"

if (( sync_paper_assets == 1 )); then
  cp "$PAPER_GENERATED_DIR"/* "$PAPER_ROOT/generated/"
  log "synced generated table snippets into $PAPER_ROOT/generated"
fi

log "paper asset bundle ready under $PAPER_ASSET_ROOT"
