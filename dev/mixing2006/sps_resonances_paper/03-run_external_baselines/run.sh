#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"
activate_venv

require_cmd bash

ensure_dir "$EXTERNAL_RUN_DIR"
ensure_dir "$EXTERNAL_RUN_DIR/logs"

if [[ ! -x "$EXTERNAL_ROOT/urqmd-4.0/urqmd.x86_64" ]]; then
  printf 'missing UrQMD executable under %s\n' "$EXTERNAL_ROOT/urqmd-4.0" >&2
  printf 'run 02-build_external_generators/run.sh first\n' >&2
  exit 1
fi
if [[ ! -x "$EXTERNAL_ROOT/crmc/install/bin/crmc" ]]; then
  printf 'missing CRMC executable under %s\n' "$EXTERNAL_ROOT/crmc/install" >&2
  printf 'run 02-build_external_generators/run.sh first\n' >&2
  exit 1
fi
if [[ ! -d "$EXTERNAL_ROOT/hepmc2-install/lib" ]]; then
  printf 'missing HepMC2 libraries under %s\n' "$EXTERNAL_ROOT/hepmc2-install/lib" >&2
  printf 'run 02-build_external_generators/run.sh first\n' >&2
  exit 1
fi

chunk_complete() {
  local model="$1"
  local chunk_tag="$2"
  local species
  for species in piminus piplus kaplus kaminus proton antiproton; do
    if [[ ! -s "$EXTERNAL_RUN_DIR/${model}_chunk_${chunk_tag}_${species}.csv" ]]; then
      return 1
    fi
  done
  return 0
}

throttle_jobs() {
  local max_jobs="$1"
  if (( max_jobs <= 1 )); then
    return
  fi
  while (( $(jobs -pr | wc -l) >= max_jobs )); do
    wait -n
  done
}

run_urqmd_chunk() {
  local chunk="$1"
  local chunk_tag seed template tag log_file
  printf -v chunk_tag '%02d' "$chunk"
  seed=$((URQMD_SEED_BASE + chunk))
  template="$EXTERNAL_RUN_DIR/urqmd_chunk_${chunk_tag}_{species_tag}.csv"
  tag="urqmd_chunk_${chunk_tag}"
  log_file="$EXTERNAL_RUN_DIR/logs/${tag}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] && chunk_complete urqmd "$chunk_tag"; then
    log "reusing $tag"
    return
  fi
  log "running $tag"
  run_python "$REPO_ROOT/helpers/benchmarking/run_urqmd_to_csv.py" \
    --species all \
    --urqmd-dir "$EXTERNAL_ROOT/urqmd-4.0" \
    --events "$EVENTS_PER_CHUNK" \
    --seed "$seed" \
    --beam-momentum "$BEAM_MOMENTUM" \
    --target-momentum "$TARGET_MOMENTUM" \
    --out-csv-template "$template" \
    --tag "$tag" >"$log_file" 2>&1
}

run_epos_chunk() {
  local chunk="$1"
  local chunk_tag seed template tag log_file
  printf -v chunk_tag '%02d' "$chunk"
  seed=$((EPOS_SEED_BASE + chunk))
  template="$EXTERNAL_RUN_DIR/epos_chunk_${chunk_tag}_{species_tag}.csv"
  tag="epos_chunk_${chunk_tag}"
  log_file="$EXTERNAL_RUN_DIR/logs/${tag}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] && chunk_complete epos "$chunk_tag"; then
    log "reusing $tag"
    return
  fi
  log "running $tag"
  run_python "$REPO_ROOT/helpers/benchmarking/run_epos_crmc_to_csv.py" \
    --species all \
    --crmc-install "$EXTERNAL_ROOT/crmc/install" \
    --hepmc-lib "$EXTERNAL_ROOT/hepmc2-install/lib" \
    --events "$EVENTS_PER_CHUNK" \
    --seed "$seed" \
    --projectile-momentum "$BEAM_MOMENTUM" \
    --target-momentum "$TARGET_MOMENTUM" \
    --out-csv-template "$template" \
    --tag "$tag" >"$log_file" 2>&1
}

run_pythia_chunk() {
  local chunk="$1"
  local chunk_tag seed template tag log_file
  printf -v chunk_tag '%02d' "$chunk"
  seed=$((PYTHIA_SEED_BASE + chunk))
  template="$EXTERNAL_RUN_DIR/pythia_chunk_${chunk_tag}_{species_tag}.csv"
  tag="pythia_chunk_${chunk_tag}"
  log_file="$EXTERNAL_RUN_DIR/logs/${tag}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] && chunk_complete pythia "$chunk_tag"; then
    log "reusing $tag"
    return
  fi
  log "running $tag"
  run_python "$REPO_ROOT/helpers/benchmarking/run_pythia8_to_csv.py" \
    --species all \
    --events "$EVENTS_PER_CHUNK" \
    --seed "$seed" \
    --beam-momentum "$BEAM_MOMENTUM" \
    --target-momentum "$TARGET_MOMENTUM" \
    --out-csv-template "$template" \
    --tag "$tag" >"$log_file" 2>&1
}

run_chunk_family() {
  local model="$1"
  local chunk
  for ((chunk = 0; chunk < CHUNK_COUNT; chunk++)); do
    throttle_jobs "$PARALLEL_CHUNKS"
    case "$model" in
      urqmd) run_urqmd_chunk "$chunk" & ;;
      epos) run_epos_chunk "$chunk" & ;;
      pythia) run_pythia_chunk "$chunk" & ;;
      *) printf 'unknown model family: %s\n' "$model" >&2; exit 1 ;;
    esac
  done
  wait
}

run_chunk_family urqmd
run_chunk_family epos
run_chunk_family pythia

run_python "$ROOT/merge_model_chunks.py" --run-dir "$EXTERNAL_RUN_DIR"

for species in pi- pi+ k+ k- p pbar; do
  tag="$(species_tag "$species")"
  summary="$EXTERNAL_RUN_DIR/na49_benchmark_${tag}_summary.json"
  if [[ "$SKIP_EXISTING" == "1" && -s "$summary" ]]; then
    log "reusing benchmark summary for $species"
    continue
  fi
  log "benchmarking external baselines for $species"
  run_python "$REPO_ROOT/helpers/benchmarking/benchmark_sps_species.py" \
    --species "$species" \
    --run-description "$EXTERNAL_RUN_NAME" \
    --disable-resonance \
    --beam-momentum "$BEAM_MOMENTUM" \
    --xf-min 0.0 --xf-max 0.4 \
    --pt-min 0.05 --pt-max 1.5 \
    --pythia-csv "$EXTERNAL_RUN_DIR/pythia8_softqcd_pp158_${tag}_gateway.csv" \
    --epos-csv "$EXTERNAL_RUN_DIR/epos_lhc_pp158_${tag}_gateway.csv" \
    --urqmd-csv "$EXTERNAL_RUN_DIR/urqmd_pp158_${tag}_gateway.csv"
done

log "external baseline generation complete: $EXTERNAL_RUN_DIR"
