#!/usr/bin/env bash
# Offline smoke run of the paper's production fit path.
#
# One complete fit-to-assets pass of
# helpers/benchmarking/benchmark_multispecies_constrained.py on the
# committed mini-pipeline fixture: data loading from the committed reduced
# NA49 p+Pb pion tables, component assembly from the committed component
# cache, an absolute-yield prior, the production non-negative solver, all
# summary/details/table exports, and the run-provenance record. No network,
# no spshadrons.web.cern.ch tables, no external generators, no UrQMD
# tarball. Runtime is well under a minute.
#
# Usage:
#   ./run_smoke.sh [OUTPUT_DIR]
#
# The output directory defaults to $RUNS_ROOT/smoke_mini_pipeline_smoke
# (RUNS_ROOT follows common.sh and stays overridable from the environment).
# The run always refits and revalidates; artifacts in the output directory
# are overwritten, so a second invocation proves reproducibility instead of
# reusing the first result. The fixture cache is copied into the output
# directory first, so even a cache rebuild after a decay-library change
# writes only into the run directory and never mutates the committed
# fixture.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$ROOT/common.sh"

MINI_DIR="$PIPELINE_ROOT/mini_pipeline"
FIXTURE_DIR="$MINI_DIR/fixture"
SMOKE_STAMP="${SMOKE_STAMP:-mini_pipeline_smoke}"
OUT_DIR="${1:-$RUNS_ROOT/smoke_${SMOKE_STAMP}}"

log "mini-pipeline smoke run (always refits and revalidates; output directory: $OUT_DIR)"

if [[ -x "$PYTHON_BIN" ]]; then
  RUN_PY="$PYTHON_BIN"
  log "using pinned reference environment: $RUN_PY"
else
  require_cmd python3
  RUN_PY="$(command -v python3)"
  log "reference venv not found at $VENV_DIR; using system python3: $RUN_PY"
fi

if [[ ! -d "$FIXTURE_DIR/resonance_component_cache" ]]; then
  printf 'missing committed fixture: %s\n' "$FIXTURE_DIR/resonance_component_cache" >&2
  printf 'regenerate it with: %s %s/make_fixture.py\n' "$RUN_PY" "$MINI_DIR" >&2
  exit 1
fi
if [[ ! -f "$FIXTURE_DIR/absyield_prior.json" ]]; then
  printf 'missing committed fixture: %s\n' "$FIXTURE_DIR/absyield_prior.json" >&2
  exit 1
fi

# helpers.benchmarking imports resolve against the mixing2006 tree; the
# generator modules resolve na49_mixing through the same path.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ensure_dir "$OUT_DIR"
ensure_dir "$OUT_DIR/resonance_component_cache"
cp -R "$FIXTURE_DIR/resonance_component_cache/." "$OUT_DIR/resonance_component_cache/"

SECONDS=0
log "running the production constrained fit on the mini-pipeline fixture"
"$RUN_PY" "$REPO_ROOT/helpers/benchmarking/benchmark_multispecies_constrained.py" \
  --run-description "mini_pipeline_smoke" \
  --out-dir "$OUT_DIR" \
  --species "pi-,pi+" \
  --data-system pPb \
  --pa-a 208 \
  --pa-z 82 \
  --pa-name Pb208 \
  --beam-momentum 158.0 \
  --convention invariant \
  --seed 20260830 \
  --res-events-per-res 40 \
  --res-max-decay-passes 8 \
  --res-workers 1 \
  --res-fragmentation-mode both \
  --res-disable-regge-tower \
  --res-cache-dir "$OUT_DIR/resonance_component_cache" \
  --prior-abs-yield-lambda 0.5 \
  --prior-abs-yield-json "$FIXTURE_DIR/absyield_prior.json" \
  --prior-abs-yield-window 0.0:0.4 \
  --skip-external

log "auditing the run provenance record"
"$RUN_PY" "$REPO_ROOT/helpers/benchmarking/benchmark_multispecies_constrained.py" \
  --validate-provenance "$OUT_DIR"

log "validating fit artifacts, identifiability, and pinned reference numbers"
"$RUN_PY" "$MINI_DIR/validate_smoke.py" "$OUT_DIR" --fixture-dir "$FIXTURE_DIR"

log "mini-pipeline smoke completed in ${SECONDS}s"
printf 'PASS mini-pipeline smoke run directory: %s\n' "$OUT_DIR"
