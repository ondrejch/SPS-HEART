#!/usr/bin/env bash
#
# Bootstrap and build the external event generators. Downloads and the CRMC
# checkout are verified against sps_resonances_paper/reproducibility.lock.json.
#
# CRMC_COMMIT=<sha> overrides the commit recorded in the manifest; this
# explicitly breaks reproducibility and is refused outside bootstrap mode.
# FORCE_BUILD=1 rebuilds the binaries.
#
# Reference mode is the default (BOOTSTRAP_REPRO_LOCK unset or not 1):
# unresolved generator pins are fatal, and an existing unpacked source tree
# is only accepted when a build-source receipt records the verified source
# identity that produced it (write_build_receipt/verify_build_receipt from
# common.sh). BOOTSTRAP_REPRO_LOCK=1 keeps the first-run discovery behavior
# (print-and-stop for unrecorded pins, tag clones without pin verification,
# reuse of unverified trees).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"

require_cmd make
require_cmd cmake
require_cmd tar
require_cmd git
require_cmd python3
require_cmd sha256sum
require_repro_lock

force="${FORCE_BUILD:-0}"

urqmd_dir="$EXTERNAL_ROOT/urqmd-4.0"
urqmd_tarball="${URQMD_TARBALL:-$EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz}"
urqmd_source_dir="${URQMD_SOURCE_DIR:-}"
hepmc_src="$EXTERNAL_ROOT/HepMC-2.06.11"
hepmc_archive="${HEPMC2_ARCHIVE:-$EXTERNAL_DIST_DIR/hepmc2.06.11.tgz}"
hepmc_build="$hepmc_src/build"
hepmc_install="$EXTERNAL_ROOT/hepmc2-install"
crmc_src="$EXTERNAL_ROOT/crmc"
crmc_build="$crmc_src/build"
crmc_install="$crmc_src/install"

# Build-source receipts: in reference mode an unpacked generator tree is only
# accepted when a receipt next to it records the verified source identity
# (archive sha256, or a tree digest for user-supplied trees) that produced
# it. Written by this stage after a verified build; checked by this stage
# and by preflight before reuse. The CRMC checkout is identified by its git
# commit instead (see verify_crmc_checkout).
urqmd_receipt="$EXTERNAL_ROOT/urqmd-4.0.build-receipt.json"
hepmc_receipt="$EXTERNAL_ROOT/HepMC-2.06.11.build-receipt.json"

# URLs and hashes come from the reproducibility manifest. The environment
# overrides below change where inputs come from; every fetched file still
# has to match its recorded sha256.
urqmd_tarball_sha256="$(lock_download_field urqmd_source_tarball sha256)"
hepmc_url="$(lock_download_field hepmc2_source_tarball url)"
hepmc_archive_sha256="$(lock_download_field hepmc2_source_tarball sha256)"
crmc_git_url="$(lock_generator_field crmc git_url)"
crmc_git_ref="$(lock_generator_field crmc tag_reference)"
crmc_pinned_commit="$(lock_generator_field crmc pinned_commit)"

hepmc_url="${HEPMC2_URL:-$hepmc_url}"
crmc_git_url="${CRMC_GIT_URL:-$crmc_git_url}"
crmc_git_ref="${CRMC_GIT_REF:-$crmc_git_ref}"

crmc_commit="$crmc_pinned_commit"
if [[ -n "${CRMC_COMMIT:-}" ]]; then
  if in_bootstrap_mode; then
    log "WARNING: CRMC_COMMIT=$CRMC_COMMIT overrides the commit recorded in the manifest; this explicitly breaks reproducibility"
    crmc_commit="$CRMC_COMMIT"
  else
    printf 'CRMC_COMMIT=%s overrides the pinned commit recorded in %s; reference mode (BOOTSTRAP_REPRO_LOCK is not 1) refuses to build against an override\n' "$CRMC_COMMIT" "$(basename "$REPRO_LOCK_FILE")" >&2
    printf 'unset CRMC_COMMIT to build against the recorded pin, or run with BOOTSTRAP_REPRO_LOCK=1 if the override is intentional\n' >&2
    exit 1
  fi
fi

# Gate the manifest-recorded source pins of the always-built generators
# (HepMC2 and CRMC) before any tree is fetched or reused. Reference mode
# makes unresolved pins fatal; bootstrap mode keeps the first-run discovery
# flow. The UrQMD pin is checked in prepare_urqmd_source, where the input
# path (tarball versus supplied tree) is known.
repro_pin_gate() {
  if in_bootstrap_mode; then
    log "bootstrap mode (BOOTSTRAP_REPRO_LOCK=1): first-run discovery stays available"
    return 0
  fi
  local problems=()
  if ! is_sha256 "$hepmc_archive_sha256"; then
    problems+=("downloads.hepmc2_source_tarball.sha256 (value: $hepmc_archive_sha256)")
  fi
  if ! is_git_commit "$crmc_pinned_commit"; then
    problems+=("generators.crmc.pinned_commit (value: $crmc_pinned_commit)")
  fi
  if [[ ${#problems[@]} -gt 0 ]]; then
    printf 'reference reproduction mode: the reproducibility manifest still holds unresolved external-generator pins\n' >&2
    for problem in "${problems[@]}"; do
      printf '  unrecorded: %s\n' "$problem" >&2
    done
    printf 'resolve each pin in %s (a bootstrap run prints the values to record), or run this stage with BOOTSTRAP_REPRO_LOCK=1 to keep the discovery behavior\n' "$REPRO_LOCK_FILE" >&2
    return 1
  fi
  return 0
}

extract_tarball_dir() {
  local archive="$1"
  local dest="$2"
  local tmp_root src_root
  tmp_root="$(mktemp -d "$EXTERNAL_ROOT/.extract.XXXXXX")"
  tar -xf "$archive" -C "$tmp_root"
  src_root="$(find "$tmp_root" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -z "$src_root" ]]; then
    printf 'failed to find extracted directory inside %s\n' "$archive" >&2
    exit 1
  fi
  mv "$src_root" "$dest"
  rm -rf "$tmp_root"
}

# Reject the unverified-tree path in reference mode. Bootstrap mode keeps the
# documented warning-and-continue behavior.
urqmd_unverified_tree() {
  local how="$1"
  printf 'the UrQMD source tree (%s) has no verified source identity in reference mode\n' "$how" >&2
  printf 'UrQMD is not redistributed by this repo, so its identity must be a lock-recorded tarball hash or an explicitly attested tree digest\n' >&2
  printf 'options:\n'
  printf '  1. supply the tarball at %s and record its sha256 in %s under downloads.urqmd_source_tarball.sha256 (recommended), or\n' "$urqmd_tarball" "$(basename "$REPRO_LOCK_FILE")" >&2
  printf '  2. attest this tree as the reference input: rerun this stage with URQMD_ATTEST_TREE=1, which records the deterministic tree digest into %s, or\n' "$urqmd_receipt" >&2
  printf '  3. run with BOOTSTRAP_REPRO_LOCK=1 to keep the bootstrap discovery behavior\n' >&2
  exit 1
}

# Accept a tree that now exists at $urqmd_dir but was not built this run from
# a lock-verified tarball: reference mode requires a matching build-source
# receipt (or an explicit attestation that writes one); bootstrap mode
# reuses it with a warning.
urqmd_adopt_tree() {
  local how="$1" rc=0
  if in_bootstrap_mode; then
    log "WARNING: reusing UrQMD source tree without verification ($how; bootstrap mode)"
    return 0
  fi
  verify_build_receipt "$urqmd_dir" "$urqmd_receipt" || rc=$?
  case "$rc" in
    0)
      log "UrQMD source tree ($how) verified against the build-source receipt $urqmd_receipt"
      return 0
      ;;
    2)
      if [[ "${URQMD_ATTEST_TREE:-0}" == "1" ]]; then
        write_build_receipt "$urqmd_receipt" "urqmd-4.0" "tree_digest" "-" "$(tree_sha256 "$urqmd_dir")"
        log "attested the UrQMD source tree as the reference input (URQMD_ATTEST_TREE=1); receipt: $urqmd_receipt"
        return 0
      fi
      urqmd_unverified_tree "$how"
      ;;
    4)
      printf 'the UrQMD source tree (%s) no longer matches the build-source receipt %s\n' "$how" "$urqmd_receipt" >&2
      printf 'restore the reference tree, or rerun with URQMD_ATTEST_TREE=1 to attest the current tree\n' >&2
      exit 1
      ;;
    *)
      printf 'the UrQMD build-source receipt %s is malformed (rc=%s)\n' "$urqmd_receipt" "$rc" >&2
      exit 1
      ;;
  esac
}

prepare_urqmd_source() {
  if [[ -d "$urqmd_dir" ]]; then
    urqmd_adopt_tree "existing tree at $urqmd_dir"
    return
  fi
  ensure_dir "$EXTERNAL_ROOT"
  ensure_dir "$EXTERNAL_DIST_DIR"
  if [[ -n "$urqmd_source_dir" ]]; then
    log "copying UrQMD source tree from $urqmd_source_dir"
    ensure_dir "$urqmd_dir"
    cp -a "$urqmd_source_dir"/. "$urqmd_dir"
    urqmd_adopt_tree "tree copied from URQMD_SOURCE_DIR=$urqmd_source_dir"
    return
  fi
  if [[ -s "$urqmd_tarball" ]]; then
    if is_sha256 "$urqmd_tarball_sha256"; then
      verify_existing_sha256 "$urqmd_tarball" "$urqmd_tarball_sha256" "UrQMD tarball $urqmd_tarball"
      log "extracting UrQMD source tree from $urqmd_tarball"
      extract_tarball_dir "$urqmd_tarball" "$urqmd_dir"
      if in_bootstrap_mode; then
        return 0
      fi
      write_build_receipt "$urqmd_receipt" "urqmd-4.0" "archive_sha256" "$urqmd_tarball_sha256" "$(tree_sha256 "$urqmd_dir")"
      log "build-source receipt recorded: $urqmd_receipt"
      return 0
    fi
    if in_bootstrap_mode; then
      log "WARNING: no sha256 recorded for the UrQMD tarball; proceeding without verification"
      log "record its sha256 in $(basename "$REPRO_LOCK_FILE") under downloads.urqmd_source_tarball.sha256"
      log "extracting UrQMD source tree from $urqmd_tarball"
      extract_tarball_dir "$urqmd_tarball" "$urqmd_dir"
      return 0
    fi
    printf 'no sha256 recorded for the UrQMD tarball in reference mode (manifest value: %s)\n' "$urqmd_tarball_sha256" >&2
    printf 'record the tarball sha256 in %s under downloads.urqmd_source_tarball.sha256 and rerun, supply an unpacked tree with URQMD_SOURCE_DIR, or run with BOOTSTRAP_REPRO_LOCK=1\n' "$REPRO_LOCK_FILE" >&2
    exit 1
  fi
  cat >&2 <<EOF
missing UrQMD source tree

Provide one of:
  URQMD_SOURCE_DIR=/absolute/path/to/urqmd-4.0
  URQMD_TARBALL=$EXTERNAL_DIST_DIR/urqmd-4.0.tar.gz

UrQMD is not redistributed by this repo, so step 02 cannot fetch it automatically.
EOF
  exit 1
}

prepare_hepmc_source() {
  local rc=0
  if [[ -d "$hepmc_src" ]]; then
    if in_bootstrap_mode; then
      log "WARNING: reusing HepMC2 source tree without verification (bootstrap mode)"
      return 0
    fi
    verify_build_receipt "$hepmc_src" "$hepmc_receipt" || rc=$?
    case "$rc" in
      0)
        log "HepMC2 source tree verified against the build-source receipt $hepmc_receipt"
        return 0
        ;;
      2)
        if [[ "${HEPMC_ATTEST_TREE:-0}" == "1" ]]; then
          write_build_receipt "$hepmc_receipt" "HepMC-2.06.11" "tree_digest" "-" "$(tree_sha256 "$hepmc_src")"
          log "attested the HepMC2 source tree as the reference input (HEPMC_ATTEST_TREE=1); receipt: $hepmc_receipt"
          return 0
        fi
        printf 'the HepMC2 source tree at %s has no build-source receipt in reference mode\n' "$hepmc_src" >&2
        printf 'remove %s to re-fetch the lock-verified archive, rerun with HEPMC_ATTEST_TREE=1 to attest the existing tree, or run with BOOTSTRAP_REPRO_LOCK=1\n' "$hepmc_src" >&2
        exit 1
        ;;
      4)
        printf 'the HepMC2 source tree at %s no longer matches the build-source receipt %s\n' "$hepmc_src" "$hepmc_receipt" >&2
        printf 'restore the reference tree, or rerun with HEPMC_ATTEST_TREE=1 to attest the current tree\n' >&2
        exit 1
        ;;
      *)
        printf 'the HepMC2 build-source receipt %s is malformed (rc=%s)\n' "$hepmc_receipt" "$rc" >&2
        exit 1
        ;;
    esac
  fi
  require_cmd curl
  ensure_dir "$EXTERNAL_ROOT"
  ensure_dir "$EXTERNAL_DIST_DIR"
  if is_sha256 "$hepmc_archive_sha256"; then
    fetch_verified "$hepmc_archive" "$hepmc_url" "$hepmc_archive_sha256" "HepMC2 source archive"
  else
    first_verified_download "$hepmc_archive" "$hepmc_url" "HepMC2 source archive" "hepmc2_source_tarball"
  fi
  log "extracting HepMC2 source tree"
  extract_tarball_dir "$hepmc_archive" "$hepmc_src"
  if in_bootstrap_mode; then
    return 0
  fi
  write_build_receipt "$hepmc_receipt" "HepMC-2.06.11" "archive_sha256" "$hepmc_archive_sha256" "$(tree_sha256 "$hepmc_src")"
  log "build-source receipt recorded: $hepmc_receipt"
}

# Gate for the CRMC commit pin, shared by verify_crmc_checkout and
# prepare_crmc_source. Returns 0 when a 40-hex commit is recorded, returns 1
# while the manifest still holds the TODO marker, and refuses a 64-hex
# sha256, which can never be a git commit.
crmc_commit_gate() {
  if is_git_commit "$crmc_commit"; then
    return 0
  fi
  if is_sha256 "$crmc_commit"; then
    printf 'generators.crmc.pinned_commit holds a 64-hex sha256 (%s)\n' "$crmc_commit" >&2
    printf 'a git commit is 40 hex characters; record the resolved commit in %s under generators.crmc.pinned_commit\n' "$REPRO_LOCK_FILE" >&2
    exit 1
  fi
  return 1
}

verify_crmc_checkout() {
  if ! crmc_commit_gate; then
    log "reusing CRMC source tree (no pinned commit recorded)"
    return
  fi
  local actual
  actual="$(git -C "$crmc_src" rev-parse HEAD)"
  if [[ "$actual" != "$crmc_commit" ]]; then
    printf 'CRMC checkout does not match the pinned commit\n' >&2
    printf '  expected: %s\n  actual:   %s\n' "$crmc_commit" "$actual" >&2
    printf 'remove %s to re-fetch it, or set CRMC_COMMIT explicitly (this breaks reproducibility)\n' "$crmc_src" >&2
    exit 1
  fi
  log "CRMC commit verified: $actual"
}

prepare_crmc_source() {
  if [[ -d "$crmc_src" ]]; then
    if [[ -d "$crmc_src/.git" ]]; then
      verify_crmc_checkout
    elif in_bootstrap_mode; then
      log "WARNING: reusing CRMC source tree without verification (no .git metadata; bootstrap mode)"
    else
      printf 'the CRMC source tree at %s has no .git metadata, so its commit identity cannot be verified in reference mode\n' "$crmc_src" >&2
      printf 'remove %s to re-fetch the pinned commit %s, or run with BOOTSTRAP_REPRO_LOCK=1\n' "$crmc_src" "$crmc_commit" >&2
      exit 1
    fi
    return
  fi
  ensure_dir "$EXTERNAL_ROOT"
  if crmc_commit_gate; then
    log "fetching CRMC source at pinned commit $crmc_commit"
    git init -q "$crmc_src"
    git -C "$crmc_src" remote add origin "$crmc_git_url"
    if git -C "$crmc_src" fetch --depth 1 origin "$crmc_commit"; then
      git -C "$crmc_src" -c advice.detachedHead=false checkout -q --force --detach FETCH_HEAD
    else
      log "the remote refused a shallow fetch of the pinned commit; falling back to a full clone"
      rm -rf "$crmc_src"
      git clone -q "$crmc_git_url" "$crmc_src"
      git -C "$crmc_src" -c advice.detachedHead=false checkout -q --detach "$crmc_commit"
    fi
    verify_crmc_checkout
  else
    log "WARNING: no CRMC commit recorded in $(basename "$REPRO_LOCK_FILE"); cloning tag $crmc_git_ref without commit verification"
    git clone --branch "$crmc_git_ref" --depth 1 "$crmc_git_url" "$crmc_src"
    crmc_resolved="$(git -C "$crmc_src" rev-parse HEAD)"
    log "resolved commit for tag $crmc_git_ref: $crmc_resolved"
    log "record it in $(basename "$REPRO_LOCK_FILE") under generators.crmc.pinned_commit to pin future builds"
  fi
}

repro_pin_gate
prepare_urqmd_source
prepare_hepmc_source
prepare_crmc_source

if [[ "$force" == "1" || ! -x "$urqmd_dir/urqmd.x86_64" ]]; then
  log "building UrQMD"
  make -C "$urqmd_dir" -j"$MAKE_JOBS"
else
  log "reusing UrQMD build"
fi

ensure_dir "$hepmc_build"
if [[ "$force" == "1" || ! -d "$hepmc_install/lib" ]]; then
  log "configuring HepMC2"
  cmake -S "$hepmc_src" -B "$hepmc_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$hepmc_install"
  log "building HepMC2"
  cmake --build "$hepmc_build" --parallel "$MAKE_JOBS"
  cmake --install "$hepmc_build"
else
  log "reusing HepMC2 install"
fi

ensure_dir "$crmc_build"
if [[ "$force" == "1" || ! -x "$crmc_install/bin/crmc" ]]; then
  log "configuring CRMC"
  cmake -S "$crmc_src" -B "$crmc_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$crmc_install" \
    -DHEPMC_ROOT_DIR="$hepmc_install" \
    -D__SIBYLL__=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Root=TRUE
  log "building CRMC"
  cmake --build "$crmc_build" --parallel "$MAKE_JOBS"
  cmake --install "$crmc_build"
else
  log "reusing CRMC install"
fi

log "external generators ready"
