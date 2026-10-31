#!/usr/bin/env bash
#
# Create the Python virtualenv for the paper pipeline.
#
# Default (REPRO_LOCK=1): install the exact package versions pinned in
# sps_resonances_paper/requirements-paper.lock.txt (pip included), then the
# mixing2006 package itself in editable mode. This reproduces the reference
# environment recorded in reproducibility.lock.json.
#
# REPRO_LOCK=0 restores the development behavior: upgrade pip to the latest
# release, install the package with latest-compatible dependencies from
# pyproject.toml, and install pythia8mc from PYTHIA8MC_SPEC. Use this mode
# for library development; it does not reproduce the reference environment.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"

require_cmd python3

repro_lock="${REPRO_LOCK:-1}"
case "$repro_lock" in
  0|1) ;;
  *)
    printf 'REPRO_LOCK must be 0 or 1 (got %s)\n' "$repro_lock" >&2
    exit 1
    ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
  log "creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  log "reusing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "$repro_lock" == "1" ]]; then
  if [[ ! -f "$REQUIREMENTS_LOCK_FILE" ]]; then
    printf 'missing pinned requirements file: %s\n' "$REQUIREMENTS_LOCK_FILE" >&2
    printf 'set REPRO_LOCK=0 to install latest-compatible packages instead\n' >&2
    exit 1
  fi
  pip_pin="$(lock_pip_pin)"
  if [[ -z "$pip_pin" ]]; then
    printf 'no pip pin found in %s\n' "$REQUIREMENTS_LOCK_FILE" >&2
    exit 1
  fi
  log "installing pinned pip $pip_pin"
  python -m pip install --upgrade "pip==$pip_pin"
  log "installing pinned packages from $(basename "$REQUIREMENTS_LOCK_FILE")"
  python -m pip install -r "$REQUIREMENTS_LOCK_FILE"
  log "installing mixing2006 in editable mode (dependencies already pinned)"
  python -m pip install --no-deps -e "${MIXING2006_DIR}[dev]"
else
  python -m pip install --upgrade pip
  python -m pip install -e "${MIXING2006_DIR}[dev]"
  python -m pip install "${PYTHIA8MC_SPEC:-pythia8mc}"
fi

log "python environment ready"
