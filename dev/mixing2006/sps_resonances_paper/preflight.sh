#!/usr/bin/env bash
# Pre-flight environment audit for the paper reproduction pipeline.
#
# Run this before committing to the expensive stages (03, 06, 07, and
# especially 08). The audit is read-only with respect to expensive work: it
# checks the required commands, parses the reproducibility manifest, verifies
# the sha256 of every downloaded SPS file through stage 00 in CHECK_ONLY
# mode (nothing is downloaded), reports free disk space on the run and
# external filesystems, inspects the Python environment against the manifest,
# and reports the state of the user-supplied UrQMD input.
#
# Every check prints one line:
#   check PASS|WARN|FAIL <name>: <one-line reason>
#
# Verdict meanings:
#   PASS  the item is ready.
#   WARN  the item blocks part of the full reproduction but not the offline
#         smoke path (missing UrQMD, unrecorded manifest hashes in bootstrap
#         mode, low disk space, missing build tools).
#   FAIL  the item breaks the full reproduction (a broken reproducibility
#         manifest, SPS tables whose sha256 no longer matches the manifest,
#         a venv missing a pinned package, and -- in reference mode --
#         unresolved generator pins or an external-generator tree without a
#         verified source identity).
#
# Exit status: 0 for PASS and PASS WITH WARNINGS, 1 for FAIL.
#
# Reference mode is the default (BOOTSTRAP_REPRO_LOCK unset or not 1):
# unresolved HepMC2/CRMC pins are FAIL instead of WARN, and an existing
# HepMC2/CRMC/UrQMD source tree is accepted only when its build-source
# receipt (or, for CRMC, its .git checkout) verifies the source identity.
# BOOTSTRAP_REPRO_LOCK=1 restores the first-run discovery verdicts. The
# offline smoke path never runs this script, so strictness here only
# affects the full reproduction.
#
# Overrides:
#   PREFLIGHT_MIN_FREE_GB  minimum free space in GiB on the filesystems of
#                          RUNS_ROOT and EXTERNAL_ROOT. Default: 20.
#   URQMD_TARBALL, URQMD_SOURCE_DIR, RUNS_ROOT, EXTERNAL_ROOT, VENV_DIR
#     follow common.sh and are honored by the matching checks.
#   BOOTSTRAP_REPRO_LOCK   1 switches the generator-identity verdicts to the
#                          bootstrap (discovery) behavior. Default: 0.
#   REPRO_LOCK_FILE, REQUIREMENTS_LOCK_FILE
#     follow common.sh and are honored by the manifest checks (tests point
#     them at fixture manifests).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$ROOT/common.sh"

MIN_FREE_GB="${PREFLIGHT_MIN_FREE_GB:-20}"

n_pass=0
n_warn=0
n_fail=0

record() {
  local verdict="$1" name="$2" reason="$3"
  log "check $verdict $name: $reason"
  case "$verdict" in
    PASS) n_pass=$((n_pass + 1)) ;;
    WARN) n_warn=$((n_warn + 1)) ;;
    FAIL) n_fail=$((n_fail + 1)) ;;
    *)
      printf 'internal error: unknown check verdict %s\n' "$verdict" >&2
      exit 1
      ;;
  esac
}

# ---------------------------------------------------------------- commands

missing_core=()
for cmd in bash python3 sha256sum; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing_core+=("$cmd")
  fi
done
if [[ "${#missing_core[@]}" -gt 0 ]]; then
  record FAIL "commands_core" "missing required commands: ${missing_core[*]}"
else
  record PASS "commands_core" "bash, python3, and sha256sum are present"
fi

build_required=(curl git tar make cmake)
if [[ -n "${PAPER_ROOT:-}" && -f "$PAPER_ROOT/main.tex" ]]; then
  build_required+=(latexmk pdflatex bibtex)
fi
missing_build=()
for cmd in "${build_required[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing_build+=("$cmd")
  fi
done
if [[ "${#missing_build[@]}" -gt 0 ]]; then
  record WARN "commands_build" "missing commands needed by the full stages but not by the smoke run: ${missing_build[*]}"
else
  record PASS "commands_build" "curl, git, tar, make, and cmake are present (latex tools included when PAPER_ROOT points at a paper tree)"
fi

# ---------------------------------------------------------------- manifest

manifest_ok=1
manifest_parse_error=""
manifest_python_version="unknown"
manifest_hepmc_sha="unknown"
manifest_urqmd_sha="unknown"
manifest_crmc_commit="unknown"
manifest_crmc_tag="unknown"
manifest_sps_count="unknown"

manifest_report="$(python3 - "$REPRO_LOCK_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        lock = json.load(fh)
except Exception as exc:  # missing file, invalid JSON, wrong structure
    print("PARSE_ERROR %s" % exc)
    raise SystemExit(0)

def dig(*keys):
    node = lock
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return "<missing>"
        node = node[key]
    return node

print("python_version %s" % dig("python", "version"))
print("hepmc_sha %s" % dig("downloads", "hepmc2_source_tarball", "sha256"))
print("urqmd_sha %s" % dig("downloads", "urqmd_source_tarball", "sha256"))
print("crmc_commit %s" % dig("generators", "crmc", "pinned_commit"))
print("crmc_tag %s" % dig("generators", "crmc", "tag_reference"))
print("sps_artifacts %s" % len(dig("downloads", "sps_web", "artifacts") or []))
PY
)"

while read -r key value; do
  case "$key" in
    python_version) manifest_python_version="$value" ;;
    hepmc_sha) manifest_hepmc_sha="$value" ;;
    urqmd_sha) manifest_urqmd_sha="$value" ;;
    crmc_commit) manifest_crmc_commit="$value" ;;
    crmc_tag) manifest_crmc_tag="$value" ;;
    sps_artifacts) manifest_sps_count="$value" ;;
    PARSE_ERROR) manifest_parse_error="$value" ;;
  esac
done <<< "$manifest_report"

if [[ -n "$manifest_parse_error" ]]; then
  record FAIL "repro_manifest" "$(basename "$REPRO_LOCK_FILE") does not parse: ${manifest_parse_error}"
  manifest_ok=0
else
  record PASS "repro_manifest" "$(basename "$REPRO_LOCK_FILE") parses (python ${manifest_python_version}, ${manifest_sps_count} SPS artifacts recorded)"
fi

# ------------------------------------------------ SPS hash check (stage 00)

if [[ "$manifest_ok" == "1" ]]; then
  hash_log="$(CHECK_ONLY=1 bash "$ROOT/00-download_spshadrons/run.sh" 2>&1)" && hash_rc=0 || hash_rc=$?
  printf '%s\n' "$hash_log"
  # Parse only the machine-readable CHECK_ONLY_SUMMARY line emitted by stage
  # 00; the human-facing per-file lines and the trailing summary are never
  # parsed, so they cannot be miscounted as failures.
  summary_line="$(grep '^CHECK_ONLY_SUMMARY' <<< "$hash_log" | tail -n 1 || true)"
  if [[ -n "$summary_line" ]]; then
    n_verified="$(sed -n 's/^CHECK_ONLY_SUMMARY verified=\([0-9][0-9]*\).*/\1/p' <<< "$summary_line")"
    n_absent="$(sed -n 's/^CHECK_ONLY_SUMMARY .* absent=\([0-9][0-9]*\).*/\1/p' <<< "$summary_line")"
    n_mismatch="$(sed -n 's/^CHECK_ONLY_SUMMARY .* mismatch=\([0-9][0-9]*\)$/\1/p' <<< "$summary_line")"
    if [[ -z "$n_verified" || -z "$n_absent" || -z "$n_mismatch" ]]; then
      record FAIL "sps_hash_check" "CHECK_ONLY_SUMMARY line present but unparseable: $summary_line"
    elif [[ "$n_mismatch" -gt 0 ]]; then
      record FAIL "sps_hash_check" "${n_mismatch} of ${manifest_sps_count} SPS files are present but their sha256 does not match the manifest (delete and re-download them, or update the manifest only if the reference input legitimately changed)"
    elif [[ "$n_absent" -gt 0 ]]; then
      record WARN "sps_hash_check" "${n_absent} of ${manifest_sps_count} manifest files are absent; stage 00 downloads them on the next full run, so this blocks only the full reproduction"
    else
      record PASS "sps_hash_check" "CHECK_ONLY verification passed for all ${manifest_sps_count} files recorded in $(basename "$REPRO_LOCK_FILE") (verified=${n_verified})"
    fi
  elif [[ "$hash_rc" == "0" ]]; then
    record PASS "sps_hash_check" "CHECK_ONLY verification passed for all ${manifest_sps_count} files recorded in $(basename "$REPRO_LOCK_FILE")"
  elif grep -q 'missing required command' <<< "$hash_log"; then
    record WARN "sps_hash_check" "CHECK_ONLY verification could not run: $(grep 'missing required command' <<< "$hash_log" | head -n 1)"
  else
    record FAIL "sps_hash_check" "CHECK_ONLY verification exited ${hash_rc} without a CHECK_ONLY_SUMMARY line; last line: $(printf '%s\n' "$hash_log" | tail -n 1)"
  fi
else
  record WARN "sps_hash_check" "skipped: the reproducibility manifest does not parse"
fi

# ---------------------------------------------------------------- disk space

free_gib_of() {
  # Available space in GiB on the filesystem holding $1; walk up to the
  # nearest existing ancestor so not-yet-created roots still resolve.
  local path="$1" probe="$1"
  while [[ ! -e "$probe" ]]; do
    probe="$(dirname "$probe")"
  done
  df -Pk "$probe" | awk 'NR == 2 { printf "%.1f", $4 / 1048576 }'
}

disk_check() {
  local name="$1" path="$2" free
  free="$(free_gib_of "$path")"
  if awk -v a="$free" -v b="$MIN_FREE_GB" 'BEGIN { exit (a + 0 >= b + 0) ? 0 : 1 }'; then
    record PASS "$name" "${free} GiB free on the filesystem of $path (threshold ${MIN_FREE_GB} GiB)"
  else
    record WARN "$name" "only ${free} GiB free on the filesystem of $path, below the ${MIN_FREE_GB} GiB threshold (set PREFLIGHT_MIN_FREE_GB to change it); the smoke run stays far below this, the full stages do not"
  fi
}

disk_check "disk_runs_root" "$RUNS_ROOT"
disk_check "disk_external_root" "$EXTERNAL_ROOT"

# --------------------------------------------------------- UrQMD input state

urqmd_tree="$EXTERNAL_ROOT/urqmd-4.0"
urqmd_tarball="${URQMD_TARBALL:-$EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz}"
urqmd_receipt="$EXTERNAL_ROOT/urqmd-4.0.build-receipt.json"

# Reference mode accepts an existing UrQMD tree only through its
# build-source receipt (stage 02 enforces the same check).
urqmd_tree_receipt_verdict() {
  local label="$1" rc=0
  verify_build_receipt "$urqmd_tree" "$urqmd_receipt" || rc=$?
  case "$rc" in
    0)
      record PASS "urqmd_input" "UrQMD source tree present at $urqmd_tree ($label), verified against the build-source receipt $urqmd_receipt"
      ;;
    2)
      record FAIL "urqmd_input" "UrQMD source tree present at $urqmd_tree ($label) but no build-source receipt at $urqmd_receipt; stage 02 refuses it in reference mode -- supply the lock-verified tarball, or run stage 02 once with URQMD_ATTEST_TREE=1 to attest the tree"
      ;;
    4)
      record FAIL "urqmd_input" "UrQMD source tree at $urqmd_tree ($label) no longer matches the build-source receipt $urqmd_receipt; stage 02 refuses it in reference mode"
      ;;
    *)
      record FAIL "urqmd_input" "the UrQMD build-source receipt $urqmd_receipt is malformed (rc=$rc); stage 02 refuses the tree in reference mode"
      ;;
  esac
}

if [[ -d "$urqmd_tree" ]]; then
  if in_bootstrap_mode; then
    record PASS "urqmd_input" "UrQMD source tree present at $urqmd_tree"
  else
    urqmd_tree_receipt_verdict "existing tree"
  fi
elif [[ -n "${URQMD_SOURCE_DIR:-}" ]]; then
  if in_bootstrap_mode; then
    record PASS "urqmd_input" "UrQMD source tree supplied through URQMD_SOURCE_DIR=$URQMD_SOURCE_DIR (the tarball hash check is bypassed by design)"
  else
    record FAIL "urqmd_input" "URQMD_SOURCE_DIR=$URQMD_SOURCE_DIR supplies an unpacked tree without a verified source identity; reference mode refuses it -- run stage 02 once with URQMD_ATTEST_TREE=1 to attest the tree (it records the deterministic tree digest into a build-source receipt), or supply the lock-verified tarball"
  fi
elif [[ -s "$urqmd_tarball" ]] && is_sha256 "$manifest_urqmd_sha"; then
  record PASS "urqmd_input" "UrQMD tarball present at $urqmd_tarball; stage 02 verifies it against the recorded sha256"
elif [[ -s "$urqmd_tarball" ]]; then
  if in_bootstrap_mode; then
    record WARN "urqmd_input" "UrQMD tarball present at $urqmd_tarball but downloads.urqmd_source_tarball.sha256 still holds the marker ${manifest_urqmd_sha}; record the tarball sha256 in $(basename "$REPRO_LOCK_FILE")"
  else
    record FAIL "urqmd_input" "UrQMD tarball present at $urqmd_tarball but downloads.urqmd_source_tarball.sha256 still holds the marker ${manifest_urqmd_sha}; record the tarball sha256 in $(basename "$REPRO_LOCK_FILE") -- run_full stops before stage 02 and stage 02 refuses the unverified tarball in reference mode"
  fi
else
  record WARN "urqmd_input" "no UrQMD source tree at $urqmd_tree, no tarball at $urqmd_tarball, and URQMD_SOURCE_DIR is unset; this is the documented user-input blocker for the full reproduction (UrQMD is not redistributed by this repository; run_full.sh stops before stage 02)"
fi

# ------------------------------------------------ generator tree states (M3)
#
# Reference mode audits the always-built generators too: an existing HepMC2
# tree must carry a build-source receipt, and an existing CRMC checkout must
# carry .git metadata that verifies the pinned commit. Neither check runs in
# bootstrap mode, and neither affects the smoke path (which never runs this
# script).

if ! in_bootstrap_mode; then
  hepmc_tree="$EXTERNAL_ROOT/HepMC-2.06.11"
  hepmc_receipt="$EXTERNAL_ROOT/HepMC-2.06.11.build-receipt.json"
  if [[ -d "$hepmc_tree" ]]; then
    rc=0
    verify_build_receipt "$hepmc_tree" "$hepmc_receipt" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
      record PASS "hepmc2_tree" "HepMC2 source tree present at $hepmc_tree, verified against the build-source receipt $hepmc_receipt"
    elif [[ "$rc" -eq 2 ]]; then
      record FAIL "hepmc2_tree" "HepMC2 source tree present at $hepmc_tree but no build-source receipt at $hepmc_receipt; stage 02 refuses it in reference mode -- remove the tree to re-fetch the lock-verified archive, or run stage 02 with HEPMC_ATTEST_TREE=1 to attest it"
    elif [[ "$rc" -eq 4 ]]; then
      record FAIL "hepmc2_tree" "HepMC2 source tree at $hepmc_tree no longer matches the build-source receipt $hepmc_receipt; stage 02 refuses it in reference mode"
    else
      record FAIL "hepmc2_tree" "the HepMC2 build-source receipt $hepmc_receipt is malformed (rc=$rc); stage 02 refuses the tree in reference mode"
    fi
  else
    record PASS "hepmc2_tree" "no HepMC2 source tree yet; stage 02 fetches the lock-verified archive"
  fi

  crmc_tree="$EXTERNAL_ROOT/crmc"
  if [[ -d "$crmc_tree" ]]; then
    if [[ -d "$crmc_tree/.git" ]]; then
      if is_git_commit "$manifest_crmc_commit"; then
        crmc_actual="$(git -C "$crmc_tree" rev-parse HEAD 2>/dev/null || true)"
        if [[ -n "$crmc_actual" && "$crmc_actual" == "$manifest_crmc_commit" ]]; then
          record PASS "crmc_checkout" "CRMC checkout present at $crmc_tree on the pinned commit $crmc_actual"
        else
          record FAIL "crmc_checkout" "CRMC checkout at $crmc_tree is on commit ${crmc_actual:-<unresolvable>}, not the pinned commit $manifest_crmc_commit; stage 02 refuses it in reference mode -- remove the checkout to re-fetch the pinned commit"
        fi
      else
        record FAIL "crmc_checkout" "CRMC checkout present at $crmc_tree but the manifest pin is unrecorded (${manifest_crmc_commit}); stage 02 cannot verify the tree in reference mode"
      fi
    else
      record FAIL "crmc_checkout" "CRMC source tree present at $crmc_tree without .git metadata, so its commit identity cannot be verified; stage 02 refuses it in reference mode -- remove the tree to re-fetch the pinned commit"
    fi
  else
    record PASS "crmc_checkout" "no CRMC checkout yet; stage 02 fetches the pinned commit"
  fi
fi

# ------------------------------------------------ manifest TODO markers (M2)

if [[ "$manifest_ok" == "1" ]]; then
  if is_sha256 "$manifest_hepmc_sha"; then
    record PASS "hepmc2_sha256" "HepMC2 archive sha256 is recorded in the manifest"
  elif in_bootstrap_mode; then
    record WARN "hepmc2_sha256" "downloads.hepmc2_source_tarball.sha256 holds the marker ${manifest_hepmc_sha}; the first run of 02-build_external_generators downloads the archive once, prints its sha256, and stops without installing; record the printed value in $(basename "$REPRO_LOCK_FILE")"
  else
    record FAIL "hepmc2_sha256" "downloads.hepmc2_source_tarball.sha256 still holds the marker ${manifest_hepmc_sha}; reference mode requires the recorded archive hash -- record it in $(basename "$REPRO_LOCK_FILE"), or run with BOOTSTRAP_REPRO_LOCK=1 to keep the discovery behavior"
  fi
  if [[ "$manifest_crmc_commit" =~ ^[0-9a-f]{40}$ ]]; then
    record PASS "crmc_pinned_commit" "CRMC pinned commit is recorded in the manifest"
  elif in_bootstrap_mode; then
    record WARN "crmc_pinned_commit" "generators.crmc.pinned_commit holds the marker ${manifest_crmc_commit}; the first run of 02-build_external_generators clones tag ${manifest_crmc_tag}, prints the resolved commit, and proceeds without pin verification; record the printed commit in $(basename "$REPRO_LOCK_FILE")"
  else
    record FAIL "crmc_pinned_commit" "generators.crmc.pinned_commit still holds the marker ${manifest_crmc_commit}; reference mode requires the recorded commit -- record it in $(basename "$REPRO_LOCK_FILE"), or run with BOOTSTRAP_REPRO_LOCK=1 to keep the discovery behavior"
  fi
else
  record WARN "hepmc2_sha256" "skipped: the reproducibility manifest does not parse"
  record WARN "crmc_pinned_commit" "skipped: the reproducibility manifest does not parse"
fi

# ------------------------------------------------------------------- Python

system_python_version="$(python3 -c 'import platform; print(platform.python_version())')"
if [[ "$manifest_ok" == "1" && "$system_python_version" == "$manifest_python_version" ]]; then
  record PASS "python_version" "python3 is ${system_python_version}, matching the manifest"
elif [[ "$manifest_ok" == "1" ]]; then
  record WARN "python_version" "python3 is ${system_python_version}, the manifest records ${manifest_python_version}; stage 01 builds the venv from this interpreter"
else
  record WARN "python_version" "python3 is ${system_python_version}; manifest comparison skipped because the manifest does not parse"
fi

venv_ok=0
if [[ -x "$PYTHON_BIN" ]]; then
  record PASS "venv" "reference venv present at $VENV_DIR"
  venv_ok=1
else
  record WARN "venv" "no venv at $VENV_DIR; run_full.sh stage 01 creates it, and the smoke path falls back to system python3"
fi

if [[ "$venv_ok" == "1" ]]; then
  venv_python_version="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
  if [[ "$manifest_ok" == "1" && "$venv_python_version" == "$manifest_python_version" ]]; then
    record PASS "venv_python_version" "venv python is ${venv_python_version}, matching the manifest"
  elif [[ "$manifest_ok" == "1" ]]; then
    record WARN "venv_python_version" "venv python is ${venv_python_version}, the manifest records ${manifest_python_version}"
  else
    record WARN "venv_python_version" "venv python is ${venv_python_version}; manifest comparison skipped because the manifest does not parse"
  fi

  if [[ ! -f "$REQUIREMENTS_LOCK_FILE" ]]; then
    record FAIL "venv_lock_packages" "missing pinned requirements file: $REQUIREMENTS_LOCK_FILE"
  else
    # Compare the pinned versions against the installed environment through
    # importlib.metadata, which reads dist-info directories only; no heavy
    # package is imported.
    lock_report="$("$PYTHON_BIN" - "$REQUIREMENTS_LOCK_FILE" <<'PY'
import sys
from importlib import metadata

pins = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, wanted = line.partition("==")
        pins.append((name.strip(), wanted.strip()))

missing = []
mismatched = []
for name, wanted in pins:
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
        continue
    if got != wanted:
        mismatched.append("%s (lock %s, installed %s)" % (name, wanted, got))

if missing:
    print("MISSING %s" % ", ".join(missing))
if mismatched:
    print("MISMATCH %s" % "; ".join(mismatched))
print("OK %d of %d pins match" % (len(pins) - len(missing) - len(mismatched), len(pins)))
PY
)"
    lock_missing="$(grep '^MISSING' <<< "$lock_report" || true)"
    lock_mismatch="$(grep '^MISMATCH' <<< "$lock_report" || true)"
    lock_ok="$(grep '^OK' <<< "$lock_report" || true)"
    if [[ -n "$lock_missing" ]]; then
      record FAIL "venv_lock_packages" "pinned packages absent from the venv: ${lock_missing#MISSING } (this breaks the smoke run too; rerun 01-create_python_env/run.sh)"
    elif [[ -n "$lock_mismatch" ]]; then
      record WARN "venv_lock_packages" "installed versions differ from requirements-paper.lock.txt: ${lock_mismatch#MISMATCH }"
    else
      record PASS "venv_lock_packages" "${lock_ok#OK } against requirements-paper.lock.txt"
    fi
  fi
else
  record WARN "venv_python_version" "skipped: no venv at $VENV_DIR"
  record WARN "venv_lock_packages" "skipped: no venv at $VENV_DIR"
fi

# ------------------------------------------------------------------ verdict

if [[ "$n_fail" -gt 0 ]]; then
  verdict="FAIL"
elif [[ "$n_warn" -gt 0 ]]; then
  verdict="PASS WITH WARNINGS"
else
  verdict="PASS"
fi

log "preflight summary: $n_pass passed, $n_warn warnings, $n_fail failures"
log "preflight verdict: $verdict"

if [[ "$verdict" == "FAIL" ]]; then
  exit 1
fi
exit 0
