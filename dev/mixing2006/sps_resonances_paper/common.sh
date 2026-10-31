#!/usr/bin/env bash

set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/00runs}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$REPO_ROOT/external}"
EXTERNAL_DIST_DIR="${EXTERNAL_DIST_DIR:-$EXTERNAL_ROOT/dist}"
DEFAULT_PAPER_ROOT="$WORKSPACE_ROOT/latex/paper"
SPS_ROOT="${SPS_ROOT:-$REPO_ROOT/spshadrons.web.cern.ch}"
MIXING2006_DIR="${MIXING2006_DIR:-$REPO_ROOT}"

# Pinned reference environment for the paper pipeline: the manifest records
# URLs and sha256 values for every downloaded input, and the requirements
# file pins the exact Python package versions. The two paths are overridable
# so tests can point the pipeline at fixture manifests.
REPRO_LOCK_FILE="${REPRO_LOCK_FILE:-$PIPELINE_ROOT/reproducibility.lock.json}"
REQUIREMENTS_LOCK_FILE="${REQUIREMENTS_LOCK_FILE:-$PIPELINE_ROOT/requirements-paper.lock.txt}"

# BOOTSTRAP_REPRO_LOCK separates first-run discovery from reference
# reproduction. The default (0 or unset) is reference mode: every external
# generator must be fetched from, or verified against, a source identity
# recorded in the reproducibility manifest or in a build-source receipt, and
# unresolved pins are fatal. BOOTSTRAP_REPRO_LOCK=1 keeps the discovery
# behavior: unrecorded pins print the value to record and stop, a CRMC tag
# may be cloned without pin verification, and pre-existing source trees may
# be reused without a receipt.
: "${BOOTSTRAP_REPRO_LOCK:=0}"

in_bootstrap_mode() {
  [[ "$BOOTSTRAP_REPRO_LOCK" == "1" ]]
}

cpu_count() {
  local n
  n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  if [[ -z "$n" ]]; then
    n=4
  fi
  printf '%s\n' "$n"
}

: "${PIPELINE_STAMP:=paper_local_repro}"
: "${VENV_DIR:=$REPO_ROOT/.venv-sps-resonances-paper}"
: "${PAPER_ASSET_ROOT:=$RUNS_ROOT/paper_assets_${PIPELINE_STAMP}}"
: "${CHUNK_COUNT:=32}"
: "${EVENTS_PER_CHUNK:=5000}"
: "${PARALLEL_CHUNKS:=1}"
: "${MAKE_JOBS:=$(cpu_count)}"
: "${RES_EVENTS_PER_RES:=120000}"
: "${RES_MAX_DECAY_PASSES:=16}"
: "${RES_WORKERS:=$(cpu_count)}"
: "${SKIP_EXISTING:=1}"
: "${BOOTSTRAP_REPLICATES:=2500}"

: "${BEAM_MOMENTUM:=158.0}"
: "${TARGET_MOMENTUM:=0.0}"
: "${URQMD_SEED_BASE:=410000}"
: "${EPOS_SEED_BASE:=510000}"
: "${PYTHIA_SEED_BASE:=610000}"

: "${ABSYIELD_SCAN_LAMBDAS:=0.1 0.3 1.0}"
: "${ABSYIELD_SCAN_SIGMAS:=0.02 0.05 0.1}"
: "${TABLE9_FRACTION_LAMBDAS:=0 1 10 100 1000 10000 1000000}"
: "${NSTAR_TOWER_MAX_VALUES:=1.9,2.1,2.3,2.5,2.7,2.9,3.1}"
: "${NSTAR_LEVEL_COUNTS:=3,4,5,7,10,12,15,20}"
: "${NSTAR_SHAPES:=linear,regge_m2,hagedorn:0.12,hagedorn:0.16,hagedorn:0.25}"
: "${NSTAR_SMOOTH_LAMBDAS:=0,0.1,0.3,1,3,10,30,100,300,1000}"

if [[ -n "${PAPER_ROOT:-}" ]]; then
  PAPER_ROOT="$(cd "$PAPER_ROOT" && pwd)"
elif [[ -f "$DEFAULT_PAPER_ROOT/main.tex" ]]; then
  PAPER_ROOT="$DEFAULT_PAPER_ROOT"
else
  PAPER_ROOT=""
fi

PAPER_FIG_DIR="$PAPER_ASSET_ROOT/fig"
PAPER_GENERATED_DIR="$PAPER_ASSET_ROOT/generated"

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

TABLE9_PRIOR_JSON="$REPO_ROOT/helpers/benchmarking/config/fischer2022_table9_pi_minus.json"
PDG_PRIOR_JSON="$REPO_ROOT/helpers/benchmarking/config/fischer2022_pdg_pi_minus_composition.json"
CONSTRAINED_CONFIG_JSON="$REPO_ROOT/helpers/benchmarking/config/constrained_refit_table9_strange_absyield_defaults.json"

EXTERNAL_RUN_NAME="${EXTERNAL_RUN_NAME:-local_multispecies_allmodels_singlepass_pythia_${PIPELINE_STAMP}}"
EXTERNAL_RUN_DIR="$RUNS_ROOT/$EXTERNAL_RUN_NAME"
SHARED_RES_CACHE_DIR="${SHARED_RES_CACHE_DIR:-$RUNS_ROOT/res_component_cache_shared_${PIPELINE_STAMP}}"
BALANCED_RUN_NAME="${BALANCED_RUN_NAME:-constrained_refit_pdg_absyield_scan_l0300_s0050_${PIPELINE_STAMP}}"
BALANCED_RUN_DIR="$RUNS_ROOT/$BALANCED_RUN_NAME"
BALANCED_ASSET_DIR="$BALANCED_RUN_DIR/paper_assets"
TABLE9_CHECK_DIR="${TABLE9_CHECK_DIR:-$RUNS_ROOT/table9_check_refresh_${PIPELINE_STAMP}}"
TABLE9_CHECK_JSON="$TABLE9_CHECK_DIR/fischer2022_table9_pi_minus_yield_check.json"

ABSYIELD_SCAN_STAMP="${ABSYIELD_SCAN_STAMP:-$PIPELINE_STAMP}"
ABSYIELD_SUMMARY_DIR="$RUNS_ROOT/absyield_prior_stiffness_${ABSYIELD_SCAN_STAMP}"
TABLE9_FRACTION_STAMP="${TABLE9_FRACTION_STAMP:-$PIPELINE_STAMP}"
TABLE9_FRACTION_SUMMARY_DIR="$RUNS_ROOT/table9_fraction_scan_${TABLE9_FRACTION_STAMP}"

NSTAR_UNCONSTRAINED_RUN_NAME="${NSTAR_UNCONSTRAINED_RUN_NAME:-local_nstar_ladder_10x_extended_levels_${PIPELINE_STAMP}_unconstrained}"
NSTAR_TABLE9_RUN_NAME="${NSTAR_TABLE9_RUN_NAME:-local_nstar_ladder_10x_extended_levels_${PIPELINE_STAMP}_table9}"
NSTAR_PDG_RUN_NAME="${NSTAR_PDG_RUN_NAME:-local_nstar_ladder_10x_extended_levels_${PIPELINE_STAMP}_pdg}"
NSTAR_SUMMARY_DIR="${NSTAR_SUMMARY_DIR:-$RUNS_ROOT/nstar_ladder_extended_levels_${PIPELINE_STAMP}}"
NSTAR_SMOOTH_RUN_DIR="${NSTAR_SMOOTH_RUN_DIR:-$RUNS_ROOT/nstar_smooth_prior_scan_${PIPELINE_STAMP}}"
BOOTSTRAP_RUN_DIR="${BOOTSTRAP_RUN_DIR:-$RUNS_ROOT/bootstrap_robustness_balanced_${PIPELINE_STAMP}}"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

require_repro_lock() {
  if [[ ! -f "$REPRO_LOCK_FILE" ]]; then
    printf 'missing reproducibility manifest: %s\n' "$REPRO_LOCK_FILE" >&2
    exit 1
  fi
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

# A git commit hash is 40 hex characters, not the 64 hex of a sha256 digest;
# use this helper for generator commit pins such as
# generators.crmc.pinned_commit.
is_git_commit() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]]
}

# Emit "path<TAB>url<TAB>sha256" for every SPS web artifact in the manifest.
lock_sps_artifacts() {
  python3 - "$REPRO_LOCK_FILE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    lock = json.load(fh)
for artifact in lock["downloads"]["sps_web"]["artifacts"]:
    print("%s\t%s\t%s" % (artifact["path"], artifact["url"], artifact["sha256"]))
PY
}

# Print one field of downloads.<key> in the manifest.
lock_download_field() {
  python3 - "$REPRO_LOCK_FILE" "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    lock = json.load(fh)
print(lock["downloads"][sys.argv[2]][sys.argv[3]])
PY
}

# Print one field of generators.<name> in the manifest.
lock_generator_field() {
  python3 - "$REPRO_LOCK_FILE" "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    lock = json.load(fh)
print(lock["generators"][sys.argv[2]][sys.argv[3]])
PY
}

# Print the pip version pinned in requirements-paper.lock.txt.
lock_pip_pin() {
  awk -F'==' '$1 == "pip" {print $2; exit}' "$REQUIREMENTS_LOCK_FILE"
}

# check_file_sha256 <file> <expected-sha256> <label>
# Print one per-file PASS/FAIL line; return 1 on mismatch or absence.
check_file_sha256() {
  local file="$1" expected="$2" label="$3" actual
  if [[ ! -e "$file" ]]; then
    log "check FAIL $label (expected sha256 $expected, file absent)"
    return 1
  fi
  actual="$(sha256_of "$file")"
  if [[ "$actual" == "$expected" ]]; then
    log "check PASS $label"
    return 0
  fi
  log "check FAIL $label (expected sha256 $expected, found $actual)"
  return 1
}

# verify_existing_sha256 <file> <expected-sha256> <label>
# Verify an input file that the pipeline did not download itself.
verify_existing_sha256() {
  local file="$1" expected="$2" label="$3" actual
  actual="$(sha256_of "$file")"
  if [[ "$actual" == "$expected" ]]; then
    log "sha256 verified for $label"
    return 0
  fi
  printf 'sha256 mismatch for %s\n' "$label" >&2
  printf '  file:     %s\n  expected: %s\n  actual:   %s\n' "$file" "$expected" "$actual" >&2
  printf 'check where this input came from; update the manifest only if the reference input legitimately changed\n' >&2
  exit 1
}

# download_verified <dest> <url> <expected-sha256> <label>
# Download to a temporary file next to <dest>, verify its sha256 against the
# manifest value, and rename it into place only after the check passes.
download_verified() {
  local dest="$1" url="$2" expected="$3" label="$4"
  local tmp actual
  ensure_dir "$(dirname "$dest")"
  tmp="$(mktemp "$dest.tmp.XXXXXX")"
  log "downloading $label"
  if ! curl -fsSL --retry 3 -o "$tmp" "$url"; then
    rm -f "$tmp"
    printf 'download failed: %s\n' "$url" >&2
    exit 1
  fi
  actual="$(sha256_of "$tmp")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'sha256 mismatch for downloaded %s\n' "$label" >&2
    printf '  url:      %s\n  expected: %s\n  actual:   %s\n' "$url" "$expected" "$actual" >&2
    printf 'the downloaded file was discarded and nothing was changed on disk\n' >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$dest"
  log "installed $label (sha256 verified)"
}

# fetch_verified <dest> <url> <expected-sha256> <label>
# Keep an existing file whose sha256 matches the manifest, refuse an existing
# file whose sha256 does not match, and download-plus-verify a missing file.
# FORCE_DOWNLOAD=1 replaces a mismatching file, but only after the fresh
# copy passes the recorded sha256 check.
fetch_verified() {
  local dest="$1" url="$2" expected="$3" label="$4"
  local actual
  if ! is_sha256 "$expected"; then
    printf 'no verified sha256 recorded for %s in %s (manifest value: %s)\n' "$label" "$(basename "$REPRO_LOCK_FILE")" "$expected" >&2
    exit 1
  fi
  if [[ -e "$dest" ]]; then
    actual="$(sha256_of "$dest")"
    if [[ "$actual" == "$expected" ]]; then
      log "keeping $label (sha256 verified)"
      return 0
    fi
    if [[ "${FORCE_DOWNLOAD:-0}" == "1" ]]; then
      log "replacing $label after a verified re-download (FORCE_DOWNLOAD=1; old sha256 $actual, expected $expected)"
      download_verified "$dest" "$url" "$expected" "$label"
      return 0
    fi
    printf 'sha256 mismatch for existing %s\n' "$label" >&2
    printf '  file:     %s\n  expected: %s\n  actual:   %s\n' "$dest" "$expected" "$actual" >&2
    printf 'delete the file explicitly and rerun this step; nothing was re-downloaded over it\n' >&2
    exit 1
  fi
  download_verified "$dest" "$url" "$expected" "$label"
}

# first_verified_download <dest> <url> <label> <manifest-key>
# For an input whose sha256 is not recorded in the manifest yet: compute and
# print the sha256 (downloading to a temporary file when the file is absent)
# and stop without installing anything, so the value can be recorded first.
first_verified_download() {
  local dest="$1" url="$2" label="$3" manifest_key="$4"
  local actual tmp
  if [[ -e "$dest" ]]; then
    actual="$(sha256_of "$dest")"
    log "sha256 of the existing $label: $actual"
    log "record it in $REPRO_LOCK_FILE under downloads.$manifest_key.sha256 and rerun this step"
    exit 1
  fi
  require_cmd curl
  ensure_dir "$(dirname "$dest")"
  tmp="$(mktemp "$dest.tmp.XXXXXX")"
  log "downloading $label once to record its sha256 (nothing is installed yet)"
  if ! curl -fsSL --retry 3 -o "$tmp" "$url"; then
    rm -f "$tmp"
    printf 'download failed: %s\n' "$url" >&2
    exit 1
  fi
  actual="$(sha256_of "$tmp")"
  rm -f "$tmp"
  log "sha256 of the $label: $actual"
  log "record it in $REPRO_LOCK_FILE under downloads.$manifest_key.sha256 and rerun this step to install it verified"
  exit 1
}

# tree_sha256 <dir>
# Deterministic content digest of a source tree: one entry per regular file
# ("F<TAB>relpath<TAB>file-sha256") and one entry per symlink
# ("L<TAB>relpath<TAB>target"), sorted in the C locale; the digest is the
# sha256 of the concatenated entries. File metadata (mtimes, permissions)
# is deliberately not part of the digest, so a tree copied verbatim keeps
# its digest.
tree_sha256() {
  python3 - "$1" <<'PY'
import hashlib
import os
import sys

root = os.path.abspath(sys.argv[1])
entries = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    # os.walk lists symlinks to directories among dirnames without
    # following them, so record them here instead of in the file loop.
    for name in dirnames:
        path = os.path.join(dirpath, name)
        if os.path.islink(path):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            entries.append(("L", rel, os.readlink(path)))
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if os.path.islink(path):
            entries.append(("L", rel, os.readlink(path)))
        elif os.path.isfile(path):
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            entries.append(("F", rel, digest.hexdigest()))
final = hashlib.sha256()
for kind, rel, value in sorted(entries):
    final.update(("%s\t%s\t%s\n" % (kind, rel, value)).encode("utf-8"))
print(final.hexdigest())
PY
}

# write_build_receipt <receipt-path> <source-tree-name> <kind> <archive-sha256-or--> <tree-sha256>
# Write a build-source receipt: the verified source identity of an unpacked
# generator tree. kind is "archive_sha256" (built from a lock-verified
# tarball; archive_sha256 is the recorded tarball hash) or "tree_digest"
# (a user-attested tree; the tree_sha256 digest is the identity).
write_build_receipt() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import datetime
import json
import sys

receipt_path, source_tree, kind, archive_sha, tree_sha = sys.argv[1:6]
identity = {"kind": kind, "tree_sha256": tree_sha}
if kind == "archive_sha256":
    identity["archive_sha256"] = archive_sha
receipt = {
    "$schema_version": 1,
    "producer": "sps_resonances_paper/02-build_external_generators/run.sh",
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "source_tree": source_tree,
    "identity": identity,
}
with open(receipt_path, "w", encoding="utf-8") as fh:
    json.dump(receipt, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

# verify_build_receipt <tree-dir> <receipt-path>
# Check that the receipt exists and that the tree still matches it.
# Return codes: 0 receipt present and the tree matches; 2 receipt missing;
# 3 receipt malformed (unknown schema or missing identity fields);
# 4 receipt present but the tree no longer matches its recorded digest.
verify_build_receipt() {
  local tree_sha
  tree_sha="$(tree_sha256 "$1")"
  python3 - "$2" "$tree_sha" <<'PY'
import json
import sys

receipt_path, tree_sha = sys.argv[1], sys.argv[2]
try:
    with open(receipt_path, encoding="utf-8") as fh:
        receipt = json.load(fh)
except FileNotFoundError:
    raise SystemExit(2)
except Exception:
    raise SystemExit(3)
identity = receipt.get("identity") if isinstance(receipt, dict) else None
if not isinstance(identity, dict) or identity.get("kind") not in (
    "archive_sha256",
    "tree_digest",
):
    raise SystemExit(3)
if not isinstance(identity.get("tree_sha256"), str) or len(identity["tree_sha256"]) != 64:
    raise SystemExit(3)
if identity["kind"] == "archive_sha256":
    archive = str(identity.get("archive_sha256", ""))
    if len(archive) != 64 or any(c not in "0123456789abcdef" for c in archive):
        raise SystemExit(3)
if identity["tree_sha256"] != tree_sha:
    raise SystemExit(4)
raise SystemExit(0)
PY
}

activate_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'missing Python virtualenv: %s\n' "$VENV_DIR" >&2
    printf 'run 01-create_python_env/run.sh first\n' >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

ensure_dir() {
  mkdir -p "$1"
}

run_python() {
  "$PYTHON_BIN" "$@"
}

species_tag() {
  case "$1" in
    "pi-") printf 'piminus\n' ;;
    "pi+") printf 'piplus\n' ;;
    "k+") printf 'kaplus\n' ;;
    "k-") printf 'kaminus\n' ;;
    "p") printf 'proton\n' ;;
    "pbar") printf 'antiproton\n' ;;
    *) printf 'unsupported species: %s\n' "$1" >&2; exit 1 ;;
  esac
}

scan_tag_from_float() {
  local value="$1"
  local scaled
  scaled="$(python3 - <<'PY' "$value"
import sys
x=float(sys.argv[1])
print(int(round(x*1000.0)))
PY
)"
  printf '%04d\n' "$scaled"
}

table9_fraction_tag() {
  local value="$1"
  python3 - <<'PY' "$value"
import sys
x=float(sys.argv[1])
if x == 0.0:
    print('f0')
elif x >= 1.0 and float(int(round(x))) == x:
    print(f'f{int(round(x))}')
else:
    txt = f'{x:.6g}'.rstrip('0').rstrip('.')
    print('f' + txt.replace('.', 'p'))
PY
}

skip_if_existing() {
  local path="$1"
  if [[ "$SKIP_EXISTING" == "1" && -e "$path" ]]; then
    return 0
  fi
  return 1
}
