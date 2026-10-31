"""Regression coverage for the stage-02 reproducibility-lock gate and the
build-source receipt machinery (review 8, Priority 1).

Reference mode (the default; ``BOOTSTRAP_REPRO_LOCK`` unset or not 1) makes
unresolved external-generator pins fatal and accepts an unpacked generator
tree only through a build-source receipt:

1. ``repro_pin_gate`` in ``02-build_external_generators/run.sh`` refuses to
   run when the manifest still holds a TODO marker for the HepMC2 archive
   sha256 or the CRMC pinned commit, and passes when both are recorded.
   Bootstrap mode (``BOOTSTRAP_REPRO_LOCK=1``) keeps the first-run
   discovery flow and returns 0 either way.
2. ``tree_sha256`` / ``write_build_receipt`` / ``verify_build_receipt``
   from ``common.sh`` form the receipt contract: a deterministic tree
   digest, a small JSON receipt carrying the verified source identity, and
   a verification that returns 0 (match), 2 (receipt missing), 3
   (malformed receipt), or 4 (tree no longer matches the receipt).
3. ``prepare_urqmd_source`` refuses an existing UrQMD tree without a valid
   receipt in reference mode, accepts the tree once ``URQMD_ATTEST_TREE=1``
   has written the attesting receipt, refuses a tampered tree, and keeps
   the warning-and-continue behavior in bootstrap mode.
4. ``preflight.sh`` records FAIL (not WARN) for unresolved HepMC2/CRMC pins
   and for a UrQMD tree without a valid receipt in reference mode, while
   ``BOOTSTRAP_REPRO_LOCK=1`` restores the discovery WARN verdicts.

The bash side is driven through subprocesses: ``common.sh`` is sourced, the
function-definition block of ``02-build_external_generators/run.sh`` is
extracted and sourced into the probe shell (so the top-of-script
``require_cmd`` calls and the build body never execute), and ``preflight.sh``
is run as a whole script against fixture manifests through the
``REPRO_LOCK_FILE`` override. The module skips cleanly when bash or the
pipeline scripts are absent (sdist checkouts). All scratch output lives
under ``dev/00runs/tmp/repro_lock_gate/``.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
PIPELINE_DIR = PKG_ROOT / "sps_resonances_paper"
RUN_SH = PIPELINE_DIR / "02-build_external_generators" / "run.sh"
COMMON_SH = PIPELINE_DIR / "common.sh"
PREFLIGHT_SH = PIPELINE_DIR / "preflight.sh"

# Scratch root: dev/00runs/tmp/repro_lock_gate/ next to dev/mixing2006/.
SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "00runs" / "tmp" / "repro_lock_gate"

# Manifest values under test: the committed reference pins and the markers
# the bootstrap flow used to carry.
TODO_HEPMC_SHA = "TODO_fill_on_first_verified_download"
TODO_CRMC_COMMIT = "TODO_resolve_commit"
TODO_URQMD_SHA = "TODO_user_supplied_tarball"
HEPMC_SHA = "86b66ea0278f803cde5774de8bd187dd42c870367f1cbf6cdaec8dc7cf6afc10"
CRMC_COMMIT = "4a41dd6ebdffe66e7ce46c1e2724470d5b4bb5b9"
WRONG_SHA256_64 = "ab" * 32

if shutil.which("bash") is None:
    pytest.skip("bash is not available; the stage-02 shell gate cannot be exercised", allow_module_level=True)
if not (COMMON_SH.exists() and RUN_SH.exists() and PREFLIGHT_SH.exists()):
    pytest.skip("sps_resonances_paper scripts not present (sdist checkout)", allow_module_level=True)

_FUNC_DEF = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{$")
_FIRST_TOPLEVEL_CALL = "prepare_urqmd_source"


def _extract_function_block(run_sh_text: str) -> str:
    """Return every function definition of run.sh and nothing else.

    The definitions run from the first ``name() {`` line up to the first
    top-level call (a bare, unindented line that names a function defined in
    the file), so sourcing the block defines the helpers without running
    ``require_cmd``/``require_repro_lock`` or the build body. Unindented
    heredoc text inside the function bodies is not a function name and is
    therefore not mistaken for the top-level section.
    """
    lines = run_sh_text.splitlines()
    defined = {line[: line.index("(")] for line in lines if _FUNC_DEF.match(line)}
    start = next((i for i, line in enumerate(lines) if _FUNC_DEF.match(line)), None)
    assert start is not None, "no function definitions found in run.sh"
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i] and not lines[i][0].isspace() and lines[i] in defined
        ),
        len(lines),
    )
    block = "\n".join(lines[start:end]) + "\n"
    assert "repro_pin_gate()" in block, "extraction missed repro_pin_gate"
    assert "prepare_urqmd_source()" in block, "extraction missed prepare_urqmd_source"
    assert "prepare_hepmc_source()" in block, "extraction missed prepare_hepmc_source"
    assert _FIRST_TOPLEVEL_CALL not in block.split(), "extraction leaked a top-level call"
    return block


@functools.lru_cache(maxsize=None)
def _function_block() -> str:
    return _extract_function_block(RUN_SH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def scratch_root():
    """Fresh scratch area under dev/00runs/tmp/repro_lock_gate/."""
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    SCRATCH_ROOT.mkdir(parents=True)
    return SCRATCH_ROOT


def _probe_env(case_dir: Path, **extra: str) -> dict:
    env = os.environ.copy()
    env.pop("PAPER_ROOT", None)  # keep common.sh's PAPER_ROOT resolution inert
    env.pop("BOOTSTRAP_REPRO_LOCK", None)
    env.update(
        {
            "PROBE_COMMON": str(COMMON_SH),
            "PROBE_FUNCTIONS": str(case_dir / "functions.sh"),
            "REPRO_LOCK_FILE": str(case_dir / "fixture_reproducibility.lock.json"),
        }
    )
    env.update(extra)
    return env


def _run_probe(template: str, case_dir: Path, **env_extra: str) -> subprocess.CompletedProcess:
    (case_dir / "functions.sh").write_text(_function_block(), encoding="utf-8")
    (case_dir / "probe.sh").write_text(template, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(case_dir / "probe.sh")],
        env=_probe_env(case_dir, **env_extra),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc


def _new_case(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=SCRATCH_ROOT))


def _tail(proc: subprocess.CompletedProcess) -> str:
    return f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}"


# ---------------------------------------------------------------------------
# Case 1: repro_pin_gate routing
# ---------------------------------------------------------------------------

_PIN_GATE_TEMPLATE = """\
set -euo pipefail
# shellcheck disable=SC1090
source "$PROBE_COMMON"
# shellcheck disable=SC1090
source "$PROBE_FUNCTIONS"

hepmc_archive_sha256="$PROBE_HEPMC_SHA"
crmc_pinned_commit="$PROBE_CRMC_COMMIT"
rc=0
repro_pin_gate || rc=$?
printf 'gate_rc=%s\\n' "$rc"
printf 'PROBE_DONE\\n'
"""


def _pin_gate(hhepmc_sha: str, crmc_commit: str, bootstrap: bool) -> subprocess.CompletedProcess:
    case_dir = _new_case("pingate")
    extra = {"PROBE_HEPMC_SHA": hhepmc_sha, "PROBE_CRMC_COMMIT": crmc_commit}
    if bootstrap:
        extra["BOOTSTRAP_REPRO_LOCK"] = "1"
    return _run_probe(_PIN_GATE_TEMPLATE, case_dir, **extra)


def _printed_gate_rc(proc: subprocess.CompletedProcess) -> int:
    for line in proc.stdout.splitlines():
        if line.startswith("gate_rc="):
            return int(line.removeprefix("gate_rc="))
    raise AssertionError(f"the gate never returned (refused by exit?); stdout tail: {proc.stdout[-400:]}")


def test_pin_gate_passes_with_recorded_pins():
    """Reference mode accepts the manifest once both pins are recorded."""
    proc = _pin_gate(HEPMC_SHA, CRMC_COMMIT, bootstrap=False)
    assert proc.returncode == 0, _tail(proc)
    assert _printed_gate_rc(proc) == 0, proc.stdout


@pytest.mark.parametrize("which", ["hepmc", "crmc", "both"])
def test_pin_gate_refuses_unresolved_pins_in_reference_mode(which):
    """Each TODO marker, individually and together, is fatal in reference mode."""
    hepmc = TODO_HEPMC_SHA if which in ("hepmc", "both") else HEPMC_SHA
    crmc = TODO_CRMC_COMMIT if which in ("crmc", "both") else CRMC_COMMIT
    proc = _pin_gate(hepmc, crmc, bootstrap=False)
    assert proc.returncode == 0, _tail(proc)
    assert _printed_gate_rc(proc) == 1, _tail(proc)
    if which in ("hepmc", "both"):
        assert "downloads.hepmc2_source_tarball.sha256" in proc.stderr
    if which in ("crmc", "both"):
        assert "generators.crmc.pinned_commit" in proc.stderr
    assert "BOOTSTRAP_REPRO_LOCK=1" in proc.stderr


def test_pin_gate_refuses_64_hex_crmc_pin():
    """A sha256 pasted into the commit field is an unresolved pin, not a pin."""
    proc = _pin_gate(HEPMC_SHA, WRONG_SHA256_64, bootstrap=False)
    assert _printed_gate_rc(proc) == 1, _tail(proc)
    assert "generators.crmc.pinned_commit" in proc.stderr


def test_pin_gate_bootstrap_mode_keeps_discovery():
    """Bootstrap mode returns 0 even with both markers still recorded."""
    proc = _pin_gate(TODO_HEPMC_SHA, TODO_CRMC_COMMIT, bootstrap=True)
    assert proc.returncode == 0, _tail(proc)
    assert "bootstrap mode" in proc.stdout
    assert _printed_gate_rc(proc) == 0, proc.stdout


# ---------------------------------------------------------------------------
# Case 2: the tree digest and the receipt contract
# ---------------------------------------------------------------------------

_HELPER_TEMPLATE = """\
set -euo pipefail
# shellcheck disable=SC1090
source "$PROBE_COMMON"

case "$PROBE_MODE" in
  digest)
    printf 'digest=%s\\n' "$(tree_sha256 "$PROBE_TREE")"
    ;;
  write)
    write_build_receipt "$PROBE_RECEIPT" "urqmd-4.0" "$PROBE_KIND" "$PROBE_ARCHIVE_SHA" "$(tree_sha256 "$PROBE_TREE")"
    ;;
  verify)
    rc=0
    verify_build_receipt "$PROBE_TREE" "$PROBE_RECEIPT" || rc=$?
    printf 'verify_rc=%s\\n' "$rc"
    ;;
  *)
    printf 'unknown PROBE_MODE: %s\\n' "$PROBE_MODE" >&2
    exit 2
    ;;
esac
printf 'PROBE_DONE\\n'
"""


def _make_tree(case_dir: Path, name: str, *, extra_file: str = "") -> Path:
    tree = case_dir / name
    tree.mkdir()
    (tree / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tree / "sub").mkdir()
    (tree / "sub" / "b.txt").write_text("beta\n", encoding="utf-8")
    if extra_file:
        (tree / extra_file).write_text("gamma\n", encoding="utf-8")
    return tree


def _digest(case_dir: Path, tree: Path) -> str:
    proc = _run_probe(_HELPER_TEMPLATE, case_dir, PROBE_MODE="digest", PROBE_TREE=str(tree))
    assert proc.returncode == 0, _tail(proc)
    line = next(line for line in proc.stdout.splitlines() if line.startswith("digest="))
    return line.removeprefix("digest=")


def _write_receipt(case_dir: Path, tree: Path, receipt: Path, *, kind: str, archive_sha: str) -> None:
    proc = _run_probe(
        _HELPER_TEMPLATE,
        case_dir,
        PROBE_MODE="write",
        PROBE_TREE=str(tree),
        PROBE_RECEIPT=str(receipt),
        PROBE_KIND=kind,
        PROBE_ARCHIVE_SHA=archive_sha,
    )
    assert proc.returncode == 0, _tail(proc)
    assert receipt.is_file()


def _verify(case_dir: Path, tree: Path, receipt: Path) -> int:
    proc = _run_probe(
        _HELPER_TEMPLATE,
        case_dir,
        PROBE_MODE="verify",
        PROBE_TREE=str(tree),
        PROBE_RECEIPT=str(receipt),
    )
    assert proc.returncode == 0, _tail(proc)
    line = next(line for line in proc.stdout.splitlines() if line.startswith("verify_rc="))
    return int(line.removeprefix("verify_rc="))


def test_tree_digest_is_deterministic_and_content_sensitive():
    case_dir = _new_case("digest")
    tree_a = _make_tree(case_dir, "tree_a")
    tree_b = _make_tree(case_dir, "tree_b")
    assert _digest(case_dir, tree_a) == _digest(case_dir, tree_b), "identical trees must share a digest"
    (tree_b / "c.txt").write_text("changed\n", encoding="utf-8")
    assert _digest(case_dir, tree_a) != _digest(case_dir, tree_b), "a content change must move the digest"


def test_tree_digest_covers_symlinks():
    case_dir = _new_case("digestlink")
    tree_a = _make_tree(case_dir, "tree_a")
    tree_b = _make_tree(case_dir, "tree_b")
    (tree_a / "link").symlink_to("a.txt")
    (tree_b / "link").symlink_to("sub/b.txt")
    assert _digest(case_dir, tree_a) != _digest(case_dir, tree_b), "symlink targets are part of the digest"


def test_receipt_roundtrip_matches_and_detects_change():
    case_dir = _new_case("receipt")
    tree = _make_tree(case_dir, "tree")
    receipt = case_dir / "receipt.json"
    _write_receipt(case_dir, tree, receipt, kind="archive_sha256", archive_sha=HEPMC_SHA)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["source_tree"] == "urqmd-4.0"
    assert payload["identity"]["kind"] == "archive_sha256"
    assert payload["identity"]["archive_sha256"] == HEPMC_SHA
    assert _verify(case_dir, tree, receipt) == 0
    (tree / "a.txt").write_text("tampered\n", encoding="utf-8")
    assert _verify(case_dir, tree, receipt) == 4, "a changed tree must no longer match its receipt"


def test_receipt_verify_reports_missing_and_malformed_receipts():
    case_dir = _new_case("receiptbad")
    tree = _make_tree(case_dir, "tree")
    assert _verify(case_dir, tree, case_dir / "absent.json") == 2, "a missing receipt is rc 2"
    malformed = case_dir / "malformed.json"
    malformed.write_text(json.dumps({"identity": {"kind": "unknown"}}), encoding="utf-8")
    assert _verify(case_dir, tree, malformed) == 3, "an unknown identity kind is rc 3"
    no_archive = case_dir / "no_archive.json"
    no_archive.write_text(
        json.dumps({"identity": {"kind": "archive_sha256", "tree_sha256": "c" * 64}}),
        encoding="utf-8",
    )
    assert _verify(case_dir, tree, no_archive) == 3, "an archive-kind receipt without the archive sha256 is rc 3"


def test_tree_digest_receipt_kind_verifies():
    case_dir = _new_case("receipttree")
    tree = _make_tree(case_dir, "tree")
    receipt = case_dir / "receipt.json"
    _write_receipt(case_dir, tree, receipt, kind="tree_digest", archive_sha="-")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["identity"]["kind"] == "tree_digest"
    assert "archive_sha256" not in payload["identity"]
    assert _verify(case_dir, tree, receipt) == 0


# ---------------------------------------------------------------------------
# Case 3: prepare_urqmd_source in reference and bootstrap mode
# ---------------------------------------------------------------------------

_URQMD_TEMPLATE = """\
set -euo pipefail
# shellcheck disable=SC1090
source "$PROBE_COMMON"
# shellcheck disable=SC1090
source "$PROBE_FUNCTIONS"

urqmd_dir="$PROBE_URQMD_DIR"
urqmd_receipt="$PROBE_URQMD_RECEIPT"
urqmd_tarball="$PROBE_URQMD_DIR/unused.tar.gz"
urqmd_source_dir=""
urqmd_tarball_sha256="TODO_user_supplied_tarball"
case "$PROBE_MODE" in
  existing)
    prepare_urqmd_source
    ;;
  missing)
    prepare_urqmd_source
    ;;
  *)
    printf 'unknown PROBE_MODE: %s\\n' "$PROBE_MODE" >&2
    exit 2
    ;;
esac
printf 'PROBE_DONE\\n'
"""


def _urqmd_probe(mode: str, case_dir: Path, bootstrap: bool = False) -> subprocess.CompletedProcess:
    extra = {
        "PROBE_MODE": mode,
        "PROBE_URQMD_DIR": str(case_dir / "urqmd-4.0"),
        "PROBE_URQMD_RECEIPT": str(case_dir / "urqmd-4.0.build-receipt.json"),
    }
    if bootstrap:
        extra["BOOTSTRAP_REPRO_LOCK"] = "1"
    return _run_probe(_URQMD_TEMPLATE, case_dir, **extra)


def _seed_urqmd_tree(case_dir: Path) -> Path:
    tree = case_dir / "urqmd-4.0"
    tree.mkdir(parents=True)
    (tree / "urqmd.x86_64").write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    return tree


def test_urqmd_existing_tree_without_receipt_is_refused():
    case_dir = _new_case("urqmdrefuse")
    _seed_urqmd_tree(case_dir)
    proc = _urqmd_probe("existing", case_dir, bootstrap=False)
    assert proc.returncode != 0, _tail(proc)
    assert "PROBE_DONE" not in proc.stdout
    assert "no verified source identity in reference mode" in proc.stderr
    assert "URQMD_ATTEST_TREE=1" in proc.stderr


def test_urqmd_attestation_writes_receipt_then_the_tree_verifies():
    case_dir = _new_case("urqmdattest")
    tree = _seed_urqmd_tree(case_dir)
    receipt = case_dir / "urqmd-4.0.build-receipt.json"
    proc = _urqmd_probe("existing", case_dir, bootstrap=False)
    # First pass without the flag refuses; the attested pass writes the receipt.
    assert proc.returncode != 0
    env = _probe_env(case_dir, PROBE_MODE="existing",
                     PROBE_URQMD_DIR=str(tree),
                     PROBE_URQMD_RECEIPT=str(receipt),
                     URQMD_ATTEST_TREE="1")
    (case_dir / "probe2.sh").write_text(_URQMD_TEMPLATE, encoding="utf-8")
    proc = subprocess.run(["bash", str(case_dir / "probe2.sh")], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, _tail(proc)
    assert "attested the UrQMD source tree" in proc.stdout
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["identity"]["kind"] == "tree_digest"
    # The now-attested tree is accepted on the next run.
    proc = _urqmd_probe("existing", case_dir, bootstrap=False)
    assert proc.returncode == 0, _tail(proc)
    assert "verified against the build-source receipt" in proc.stdout


def test_urqmd_tampered_tree_after_receipt_is_refused():
    case_dir = _new_case("urqmdtamper")
    tree = _seed_urqmd_tree(case_dir)
    receipt = case_dir / "urqmd-4.0.build-receipt.json"
    env = _probe_env(case_dir, PROBE_MODE="existing",
                     PROBE_URQMD_DIR=str(tree),
                     PROBE_URQMD_RECEIPT=str(receipt),
                     URQMD_ATTEST_TREE="1")
    (case_dir / "functions.sh").write_text(_function_block(), encoding="utf-8")
    (case_dir / "probe.sh").write_text(_URQMD_TEMPLATE, encoding="utf-8")
    proc = subprocess.run(["bash", str(case_dir / "probe.sh")], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, _tail(proc)
    (tree / "urqmd.x86_64").write_text("#!/bin/sh\necho tampered\n", encoding="utf-8")
    proc = _urqmd_probe("existing", case_dir, bootstrap=False)
    assert proc.returncode != 0, _tail(proc)
    assert "no longer matches the build-source receipt" in proc.stderr


def test_urqmd_existing_tree_without_receipt_is_warned_in_bootstrap_mode():
    case_dir = _new_case("urqmdboot")
    _seed_urqmd_tree(case_dir)
    proc = _urqmd_probe("existing", case_dir, bootstrap=True)
    assert proc.returncode == 0, _tail(proc)
    assert "WARNING: reusing UrQMD source tree without verification" in proc.stdout


def test_urqmd_missing_input_still_stops_with_instructions():
    case_dir = _new_case("urqmdmissing")
    proc = _urqmd_probe("missing", case_dir, bootstrap=False)
    assert proc.returncode != 0, _tail(proc)
    assert "missing UrQMD source tree" in proc.stderr
    assert "URQMD_SOURCE_DIR=" in proc.stderr


# ---------------------------------------------------------------------------
# Case 4: preflight verdicts follow the reference/bootstrap mode
# ---------------------------------------------------------------------------

LOCK_TEMPLATE = {
    "$schema_version": 1,
    "python": {"version": "3.12.3"},
    "downloads": {
        "sps_web": {
            "base_url": "https://example.invalid/spshadrons",
            "artifacts": [
                {
                    "path": "data/sps/na49/NA49_pp_158A_pi-.dat",
                    "url": "data/sps/na49/NA49_pp_158A_pi-.dat",
                    "sha256": "3f0b2d4f1c1a8f9d6e5c4b3a291807f6e5d4c3b2a1908f7e6d5c4b3a291807f6",
                },
                {
                    "path": "data/sps/na49/NA49_pp_276A_pi-.dat",
                    "url": "data/sps/na49/NA49_pp_276A_pi-.dat",
                    "sha256": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809",
                },
            ],
        },
        "hepmc2_source_tarball": {
            "url": "https://hepmc-testing.example.invalid/hepmc2.06.11.tgz",
            "sha256": HEPMC_SHA,
        },
        "urqmd_source_tarball": {"sha256": TODO_URQMD_SHA},
    },
    "generators": {
        "crmc": {"pinned_commit": CRMC_COMMIT, "tag_reference": "v1.7.0-alice1"}
    },
}


def _write_fixture_lock(case_dir: Path, *, hepmc_sha: str, crmc_commit: str) -> Path:
    lock = json.loads(json.dumps(LOCK_TEMPLATE))
    lock["downloads"]["hepmc2_source_tarball"]["sha256"] = hepmc_sha
    lock["generators"]["crmc"]["pinned_commit"] = crmc_commit
    path = case_dir / "reproducibility.lock.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return path


def _run_preflight(case_dir: Path, *, hepmc_sha: str, crmc_commit: str,
                   bootstrap: bool = False, urqmd_tree_src: Path | None = None,
                   urqmd_receipt_src: Path | None = None) -> subprocess.CompletedProcess:
    """Run preflight.sh against a fixture manifest; optionally plant a UrQMD
    tree (copied from urqmd_tree_src) and its receipt into EXTERNAL_ROOT so
    the receipt's recorded digest matches the planted tree exactly."""
    lock_path = _write_fixture_lock(case_dir, hepmc_sha=hepmc_sha, crmc_commit=crmc_commit)
    external = case_dir / "external"
    external.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PAPER_ROOT", None)
    env.pop("URQMD_SOURCE_DIR", None)
    env.pop("URQMD_TARBALL", None)
    env.pop("BOOTSTRAP_REPRO_LOCK", None)
    env.update(
        {
            "REPRO_LOCK_FILE": str(lock_path),
            "REQUIREMENTS_LOCK_FILE": str(case_dir / "requirements-paper.lock.txt"),
            "RUNS_ROOT": str(case_dir / "runs"),
            "EXTERNAL_ROOT": str(external),
            "SPS_ROOT": str(case_dir / "sps"),
            "VENV_DIR": str(case_dir / "venv-absent"),
            "PREFLIGHT_MIN_FREE_GB": "0",
        }
    )
    if bootstrap:
        env["BOOTSTRAP_REPRO_LOCK"] = "1"
    if urqmd_tree_src is not None:
        shutil.copytree(urqmd_tree_src, external / "urqmd-4.0", symlinks=True)
    if urqmd_receipt_src is not None:
        shutil.copyfile(urqmd_receipt_src, external / "urqmd-4.0.build-receipt.json")
    return subprocess.run(
        ["bash", str(PREFLIGHT_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PIPELINE_DIR),
    )


def test_preflight_reference_mode_fails_on_unresolved_pins():
    case_dir = _new_case("preffail")
    proc = _run_preflight(case_dir, hepmc_sha=TODO_HEPMC_SHA, crmc_commit=TODO_CRMC_COMMIT)
    assert proc.returncode == 1, _tail(proc)
    assert "check FAIL hepmc2_sha256" in proc.stdout
    assert "check FAIL crmc_pinned_commit" in proc.stdout
    assert "preflight verdict: FAIL" in proc.stdout


def test_preflight_bootstrap_mode_warns_on_unresolved_pins():
    case_dir = _new_case("prefwarn")
    proc = _run_preflight(case_dir, hepmc_sha=TODO_HEPMC_SHA, crmc_commit=TODO_CRMC_COMMIT,
                          bootstrap=True)
    assert proc.returncode == 0, _tail(proc)
    assert "check WARN hepmc2_sha256" in proc.stdout
    assert "check WARN crmc_pinned_commit" in proc.stdout
    assert "preflight verdict: PASS WITH WARNINGS" in proc.stdout


def test_preflight_reference_mode_fails_on_unreceipted_urqmd_tree():
    case_dir = _new_case("prefurqmd")
    tree_src = _new_case("prefurqmd-tree")
    tree = _make_tree(tree_src, "urqmd-4.0")
    proc = _run_preflight(case_dir, hepmc_sha=HEPMC_SHA, crmc_commit=CRMC_COMMIT,
                          urqmd_tree_src=tree)
    assert proc.returncode == 1, _tail(proc)
    assert "check FAIL urqmd_input" in proc.stdout
    assert "no build-source receipt" in proc.stdout


def test_preflight_reference_mode_passes_urqmd_tree_with_receipt():
    case_dir = _new_case("prefurqmdok")
    tree_src = _new_case("prefurqmdok-tree")
    tree = _make_tree(tree_src, "urqmd-4.0")
    receipt = tree_src / "urqmd-4.0.build-receipt.json"
    _write_receipt(tree_src, tree, receipt, kind="tree_digest", archive_sha="-")
    proc = _run_preflight(case_dir, hepmc_sha=HEPMC_SHA, crmc_commit=CRMC_COMMIT,
                          urqmd_tree_src=tree, urqmd_receipt_src=receipt)
    assert proc.returncode == 0, _tail(proc)
    assert "check PASS urqmd_input" in proc.stdout
    assert "verified against the build-source receipt" in proc.stdout
    assert "preflight verdict: PASS WITH WARNINGS" in proc.stdout
