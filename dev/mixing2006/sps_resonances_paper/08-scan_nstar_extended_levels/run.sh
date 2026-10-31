#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

ensure_dir "$NSTAR_SUMMARY_DIR"

run_mode() {
  local mode="$1"
  local run_name run_dir out_prefix prior_lambda prior_json prior_strange prior_hyperon res_cache_dir summary_csv target_csv
  case "$mode" in
    unconstrained)
      run_name="$NSTAR_UNCONSTRAINED_RUN_NAME"
      out_prefix="nstar_ladder_summary_10x_unconstrained_extended_levels"
      prior_lambda="0.0"
      prior_json="$TABLE9_PRIOR_JSON"
      prior_strange="0.0"
      prior_hyperon="0.0"
      ;;
    table9)
      run_name="$NSTAR_TABLE9_RUN_NAME"
      out_prefix="nstar_ladder_summary_10x_table9_extended_levels"
      prior_lambda="1.0"
      prior_json="$TABLE9_PRIOR_JSON"
      prior_strange="1.0"
      prior_hyperon="1.0"
      ;;
    pdg)
      run_name="$NSTAR_PDG_RUN_NAME"
      out_prefix="nstar_ladder_summary_10x_pdg_extended_levels"
      prior_lambda="1.0"
      prior_json="$PDG_PRIOR_JSON"
      prior_strange="1.0"
      prior_hyperon="1.0"
      ;;
    *)
      printf 'unknown nstar mode: %s\n' "$mode" >&2
      exit 1
      ;;
  esac

  run_dir="$RUNS_ROOT/$run_name"
  res_cache_dir="$run_dir/res_cache"
  summary_csv="$run_dir/${out_prefix}.csv"
  target_csv="$NSTAR_SUMMARY_DIR/${out_prefix}.csv"

  if [[ "$SKIP_EXISTING" != "1" || ! -s "$summary_csv" ]]; then
    log "running N* extended-level grid: $mode"
    run_python "$REPO_ROOT/helpers/benchmarking/run_nstar_ladder_grid.py" \
      --run-description "$run_name" \
      --out-root "$run_dir" \
      --tower-min-mass 1.9 \
      --tower-max-values "$NSTAR_TOWER_MAX_VALUES" \
      --level-counts "$NSTAR_LEVEL_COUNTS" \
      --shapes "$NSTAR_SHAPES" \
      --seed 20260331 \
      --res-events-per-res "$RES_EVENTS_PER_RES" \
      --res-workers "$RES_WORKERS" \
      --max-decay-passes "$RES_MAX_DECAY_PASSES" \
      --fit-objective nnls_weighted \
      --fit-weight-mode sigma \
      --prior-table9-lambda "$prior_lambda" \
      --prior-table9-json "$prior_json" \
      --prior-strange-lambda "$prior_strange" \
      --prior-hyperon-lambda "$prior_hyperon" \
      --external-run-dir "$EXTERNAL_RUN_DIR" \
      --res-cache-dir "$res_cache_dir"

    run_python "$REPO_ROOT/helpers/benchmarking/summarize_nstar_ladder_results.py" \
      --run-dir "$run_dir" \
      --out-prefix "$out_prefix"
  else
    log "reusing N* summary CSV for $mode"
  fi

  cp "$summary_csv" "$target_csv"
}

run_mode unconstrained
run_mode table9
run_mode pdg

log "plotting N* likelihood maps"
run_python "$REPO_ROOT/helpers/benchmarking/plot_nstar_likelihood_map.py" \
  --csv-a "$NSTAR_SUMMARY_DIR/nstar_ladder_summary_10x_unconstrained_extended_levels.csv" \
  --csv-b "$NSTAR_SUMMARY_DIR/nstar_ladder_summary_10x_table9_extended_levels.csv" \
  --csv-c "$NSTAR_SUMMARY_DIR/nstar_ladder_summary_10x_pdg_extended_levels.csv" \
  --label-a "Unconstrained" \
  --label-b "Table-9 prior" \
  --label-c "PDG prior" \
  --out-a "$NSTAR_SUMMARY_DIR/nstar_likelihood_map_10x_unconstrained_extended_levels.png" \
  --out-b "$NSTAR_SUMMARY_DIR/nstar_likelihood_map_10x_table9_extended_levels.png" \
  --out-c "$NSTAR_SUMMARY_DIR/nstar_likelihood_map_10x_pdg_extended_levels.png" \
  --out-combined "$NSTAR_SUMMARY_DIR/nstar_likelihood_map_10x_threeway_extended_levels.png"

log "N* likelihood maps ready: $NSTAR_SUMMARY_DIR"
