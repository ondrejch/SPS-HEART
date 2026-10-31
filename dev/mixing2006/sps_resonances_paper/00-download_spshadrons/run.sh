#!/usr/bin/env bash
#
# Refresh the SPS HADRONS tables, landing pages, and source PDFs with sha256
# verification against sps_resonances_paper/reproducibility.lock.json.
#
# CHECK_ONLY=1 verifies the sha256 of every file recorded in the manifest,
# prints one PASS/FAIL line per file plus one machine-readable summary line
# (CHECK_ONLY_SUMMARY verified=N absent=N mismatch=N), and downloads nothing.
# CHECK_ONLY exit codes: 0 all verified, 3 files absent only, 2 at least one
# present file mismatches the manifest (1 is reserved for operational
# failures such as missing commands).
# FORCE_DOWNLOAD=1 re-downloads existing files; a replacement is installed
# only after the fresh copy passes the recorded sha256 check.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"

require_cmd curl
require_cmd python3
require_cmd sha256sum
require_repro_lock

check_only="${CHECK_ONLY:-0}"

declare -A SPS_URL SPS_SHA
SPS_RELS=()
while IFS=$'\t' read -r rel url sha; do
  SPS_RELS+=("$rel")
  SPS_URL["$rel"]="$url"
  SPS_SHA["$rel"]="$sha"
done < <(lock_sps_artifacts)

if [[ "${#SPS_RELS[@]}" -eq 0 ]]; then
  printf 'no SPS artifacts found in %s\n' "$REPRO_LOCK_FILE" >&2
  exit 1
fi

if [[ "$check_only" == "1" ]]; then
  n_absent=0
  n_mismatch=0
  n_verified=0
  for rel in "${SPS_RELS[@]}"; do
    if [[ ! -e "$SPS_ROOT/$rel" ]]; then
      check_file_sha256 "$SPS_ROOT/$rel" "${SPS_SHA[$rel]}" "$rel" || true
      n_absent=$((n_absent + 1))
    elif check_file_sha256 "$SPS_ROOT/$rel" "${SPS_SHA[$rel]}" "$rel"; then
      n_verified=$((n_verified + 1))
    else
      n_mismatch=$((n_mismatch + 1))
    fi
  done
  # Machine-readable verdict; preflight.sh parses only this line so that the
  # human-facing per-file lines and the summary below can never be counted
  # as failures.
  printf 'CHECK_ONLY_SUMMARY verified=%d absent=%d mismatch=%d\n' \
    "$n_verified" "$n_absent" "$n_mismatch"
  if [[ "$n_mismatch" -gt 0 ]]; then
    log "SPS hash verification failed for ${n_mismatch} file(s) above"
    exit 2
  fi
  if [[ "$n_absent" -gt 0 ]]; then
    log "SPS hash check: ${n_absent} of ${#SPS_RELS[@]} files absent (stage 00 downloads them on the next full run)"
    exit 3
  fi
  log "SPS hash check passed for all files recorded in $(basename "$REPRO_LOCK_FILE")"
  exit 0
fi

ensure_dir "$SPS_ROOT/data"
ensure_dir "$SPS_ROOT/doc"

fetch() {
  local rel="$1"
  fetch_verified "$SPS_ROOT/$rel" "${SPS_URL[$rel]}" "${SPS_SHA[$rel]}" "$rel"
}

fetch "index.html"
fetch "docs.html"
fetch "ppdata.html"
fetch "pp_piminus.html"
fetch "pp_kaon_sdep.html"

fetch "data/xsect_piminus.txt"
fetch "data/xsect_piplus.txt"
fetch "data/xsect_kaplus_pp.txt"
fetch "data/xsect_kaminus_pp.txt"
fetch "data/xsect_proton_pp.txt"
fetch "data/xsect_antiproton_pp.txt"

fetch "data/xsect_ptintegrated_piminus.txt"
fetch "data/xsect_ptintegrated_piplus.txt"
fetch "data/xsect_ptintegrated_kaplus_pp.txt"
fetch "data/xsect_ptintegrated_kaminus_pp.txt"
fetch "data/xsect_ptintegrated_proton_pp.txt"
fetch "data/xsect_ptintegrated_antiproton_pp.txt"

fetch "doc/pp-pi.pdf"
fetch "doc/pp-proton.pdf"
fetch "doc/pp-kaon.pdf"
fetch "doc/pppiminus.pdf"

log "download refresh complete"
