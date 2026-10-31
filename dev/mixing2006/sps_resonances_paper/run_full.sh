#!/usr/bin/env bash
# Full paper reproduction: preflight audit followed by the complete numbered
# stage sequence 00..13.
#
# This script supersedes run_all.sh, which stays unchanged for compatibility.
# Stage order and SKIP_EXISTING semantics are identical to run_all.sh.
#
# Usage:
#   ./run_full.sh [--dry-run]
#   RUN_FULL_STAGES="00 01 02" ./run_full.sh [--dry-run]
#
# Behavior:
#   - Runs preflight.sh first and aborts when the audit verdict is FAIL.
#   - The preflight verdicts follow BOOTSTRAP_REPRO_LOCK (common.sh): in the
#     default reference mode, unresolved HepMC2/CRMC pins and unverified
#     external-generator source trees are FAIL, so the run aborts at
#     preflight; BOOTSTRAP_REPRO_LOCK=1 restores the discovery WARNs.
#   - Executes the stage scripts in the same order as run_all.sh.
#   - Before stage 02, stops with instructions when the user-supplied UrQMD
#     input is missing or its sha256 is unrecorded, instead of failing deep
#     inside the build stage.
#   - RUN_FULL_STAGES selects a subset of stages; entries are the two-digit
#     stage numbers (00..13) or full stage names. Unset means all stages.
#   - --dry-run prints the plan (preflight plus the selected stages) and
#     executes nothing.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$ROOT/common.sh"

ALL_STAGES=(
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

usage() {
  cat <<EOF
usage: ./run_full.sh [--dry-run]
       RUN_FULL_STAGES="00 01 02" ./run_full.sh [--dry-run]

--dry-run prints the stage plan and executes nothing.
RUN_FULL_STAGES selects a subset (stage numbers 00..13 or full stage names).
EOF
}

dry_run=0
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == "--dry-run" && "$#" -eq 1 ]]; then
    dry_run=1
  else
    usage
    exit 2
  fi
fi

# Stage selection: accept two-digit numbers or full stage names.
planned=()
if [[ -n "${RUN_FULL_STAGES:-}" ]]; then
  # shellcheck disable=SC2086  # word splitting on the stage list is intended
  for token in $RUN_FULL_STAGES; do
    matched=""
    for candidate in "${ALL_STAGES[@]}"; do
      if [[ "$token" == "$candidate" || "$token" == "${candidate%%-*}" ]]; then
        matched="$candidate"
        break
      fi
    done
    if [[ -z "$matched" ]]; then
      printf 'invalid RUN_FULL_STAGES entry: %s\n' "$token" >&2
      printf 'valid entries are stage numbers 00..13 or full stage names:\n' >&2
      for candidate in "${ALL_STAGES[@]}"; do
        printf '  %s\n' "$candidate" >&2
      done
      exit 2
    fi
    planned+=("$matched")
  done
  if [[ "${#planned[@]}" -eq 0 ]]; then
    printf 'RUN_FULL_STAGES is set but selects no stage\n' >&2
    exit 2
  fi
else
  planned=("${ALL_STAGES[@]}")
fi

# UrQMD stop check before stage 02. UrQMD is not redistributed by this
# repository, so stage 02 cannot fetch it; stop here, with instructions,
# instead of failing deep inside the build.
# Prints nothing; the return code says:
#   0  the UrQMD input is supplied (tree, tarball with recorded sha256, or
#      URQMD_SOURCE_DIR)
#   1  no UrQMD input at all
#   2  tarball present but the manifest sha256 is still the TODO marker
urqmd_input_status() {
  if [[ -d "$EXTERNAL_ROOT/urqmd-4.0" ]]; then
    return 0
  fi
  if [[ -n "${URQMD_SOURCE_DIR:-}" ]]; then
    return 0
  fi
  local tarball="${URQMD_TARBALL:-$EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz}"
  local recorded
  recorded="$(lock_download_field urqmd_source_tarball sha256)"
  if [[ ! -s "$tarball" ]]; then
    return 1
  fi
  if ! is_sha256 "$recorded"; then
    return 2
  fi
  return 0
}

urqmd_stop_missing() {
  local recorded
  recorded="$(lock_download_field urqmd_source_tarball sha256)"
  cat >&2 <<EOF
run_full stops here: the UrQMD input required by stage 02 is missing.

UrQMD is not redistributed by this repository, so stage 02 cannot fetch it.
Provide one of:
  URQMD_SOURCE_DIR=/absolute/path/to/an-unpacked-urqmd-4.0-tree
  URQMD_TARBALL=/absolute/path/to/urqmd-4.0.tar.gz

The default tarball location is $EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz
(override with URQMD_TARBALL).

After placing the tarball, record its sha256 in
$REPRO_LOCK_FILE under downloads.urqmd_source_tarball.sha256
(currently "$recorded") so stage 02 can verify it:

  sha256sum $EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz

Then rerun ./run_full.sh.
EOF
  exit 2
}

urqmd_stop_unverified() {
  local tarball="${URQMD_TARBALL:-$EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz}"
  local recorded
  recorded="$(lock_download_field urqmd_source_tarball sha256)"
  cat >&2 <<EOF
run_full stops here: the UrQMD tarball has no verified sha256.

Tarball:        $tarball
Manifest field: downloads.urqmd_source_tarball.sha256 in $REPRO_LOCK_FILE
Current value:  $recorded

Record the tarball sha256 so stage 02 can verify it before extracting:

  sha256sum $tarball

Then replace the marker above with the 64-character value and rerun
./run_full.sh. To use an unpacked source tree instead, set
URQMD_SOURCE_DIR=/absolute/path/to/an-unpacked-urqmd-4.0-tree; the tarball
hash check does not apply to that path.
EOF
  exit 2
}

if [[ "$dry_run" == "1" ]]; then
  log "dry run: preflight.sh would run first and abort on a FAIL verdict"
  log "dry run: plan holds ${#planned[@]} stage(s)"
  for stage in "${planned[@]}"; do
    log "dry run: $stage"
  done
  urqmd_status=0
  urqmd_input_status || urqmd_status=$?
  case "$urqmd_status" in
    0) log "dry run: the UrQMD input is supplied, so stage 02 would proceed" ;;
    1) log "dry run: stage 02 would stop: no UrQMD source tree, no tarball, and URQMD_SOURCE_DIR is unset" ;;
    2) log "dry run: stage 02 would stop: the UrQMD tarball sha256 is not recorded in reproducibility.lock.json" ;;
  esac
  log "dry run: nothing was executed"
  exit 0
fi

log "running preflight"
bash "$ROOT/preflight.sh"

for stage in "${planned[@]}"; do
  if [[ "$stage" == "02-build_external_generators" ]]; then
    urqmd_status=0
    urqmd_input_status || urqmd_status=$?
    case "$urqmd_status" in
      0) log "UrQMD input check passed" ;;
      1) urqmd_stop_missing ;;
      2) urqmd_stop_unverified ;;
    esac
  fi
  log "running $stage"
  "$ROOT/$stage/run.sh"
done

log "run_full complete"
