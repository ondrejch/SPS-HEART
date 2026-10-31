"""Regression coverage for the stage-02 CRMC commit-pin gate (REV-20260830-01).

The stage-02 script ``sps_resonances_paper/02-build_external_generators/run.sh``
routes every CRMC pin decision through the shared ``crmc_commit_gate()``
predicate: a 40-hex git commit (``is_git_commit`` from ``common.sh``) selects
the pinned-commit fetch/verify flow, the ``TODO_resolve_commit`` manifest
marker selects the legacy tag-clone flow, and a 64-hex sha256 is refused
outright because a sha256 digest can never be a git commit.  Before nested
commit ``38a9bc9`` both gates tested ``is_sha256`` (64 hex) instead, so a
real 40-hex commit failed the sha256 check and silently fell back to the
unverified tag-clone flow, leaving the post-checkout ``git rev-parse HEAD``
verification unreachable.

The gate lives in bash, so this module drives it through subprocesses:

1. ``sps_resonances_paper/common.sh`` is sourced (sourcing it only defines
   variables and helpers),
2. the function-definition block is extracted from
   ``02-build_external_generators/run.sh`` and sourced into the probe shell,
   so the top-of-script ``require_cmd`` calls and the main build body never
   execute,
3. ``git`` is shadowed by a stub function that records every invocation and
   fakes ``rev-parse HEAD``, so the routing is observed without any network
   access or real repository.

Set ``CRMC_COMMIT_GATE_PIPELINE_DIR`` to point this module at a different
copy of the two scripts; the mutation check for this regression runs it
against the pre-fix scripts of nested commit ``b2bbb5b`` that way.  The
module skips cleanly when bash or the pipeline scripts are absent (sdist
checkouts).  All scratch output lives under ``dev/00runs/tmp/crmc_commit_gate/``.
"""
from __future__ import annotations

import functools
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

# Scratch root: dev/00runs/tmp/crmc_commit_gate/ next to dev/mixing2006/.
SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "00runs" / "tmp" / "crmc_commit_gate"

# Test-author hook: run the same module against another copy of the two
# scripts (e.g. the pre-fix b2bbb5b versions staged under dev/00runs/tmp/).
_OVERRIDE = os.environ.get("CRMC_COMMIT_GATE_PIPELINE_DIR")

# Values under test.  GIT_COMMIT and SHA256_64 differ only in length, which
# is the whole point of the regression: the gate must read the length, not
# guess that any hex blob is a digest.
GIT_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
OTHER_COMMIT = "1111222233334444555566667777888899990000"
SHA256_64 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
TODO_MARKER = "TODO_resolve_commit"
TAG_REF = "v1.7.0-test-tag"

if shutil.which("bash") is None:
    pytest.skip("bash is not available; the stage-02 shell gate cannot be exercised", allow_module_level=True)
if not (PIPELINE_DIR / "common.sh").exists() or not RUN_SH.exists():
    pytest.skip("sps_resonances_paper stage-02 scripts not present (sdist checkout)", allow_module_level=True)

_FUNC_DEF = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{$")
_FIRST_TOPLEVEL_CALL = "prepare_urqmd_source"


def _pipeline_dir() -> Path:
    directory = Path(_OVERRIDE).resolve() if _OVERRIDE else PIPELINE_DIR
    assert (directory / "common.sh").exists() and (directory / "02-build_external_generators" / "run.sh").exists(), (
        f"CRMC_COMMIT_GATE_PIPELINE_DIR does not hold common.sh and 02-build_external_generators/run.sh: {directory}"
    )
    return directory


def _run_sh() -> Path:
    return _pipeline_dir() / "02-build_external_generators" / "run.sh"


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
    assert "verify_crmc_checkout()" in block, "extraction missed verify_crmc_checkout"
    assert "prepare_crmc_source()" in block, "extraction missed prepare_crmc_source"
    assert _FIRST_TOPLEVEL_CALL not in block.split(), "extraction leaked a top-level call"
    return block


@functools.lru_cache(maxsize=None)
def _function_block() -> str:
    """Function definitions of the run.sh copy currently under test.

    The pipeline directory (and with it the extracted block) honors the
    CRMC_COMMIT_GATE_PIPELINE_DIR override, so the module can be pointed at
    the pre-fix b2bbb5b scripts for the mutation check.
    """
    return _extract_function_block(_run_sh().read_text(encoding="utf-8"))


_PROBE_TEMPLATE = """\
set -euo pipefail
# shellcheck disable=SC1090
source "$PROBE_COMMON"
# shellcheck disable=SC1090
source "$PROBE_FUNCTIONS"

crmc_commit="$PROBE_COMMIT"
crmc_src="$PROBE_EXTERNAL/crmc"
crmc_git_url="https://git-testing.example.invalid/crmc.git"
crmc_git_ref="v1.7.0-test-tag"

git() {
  # Stub: trace every invocation and fake the repository.  `rev-parse HEAD`
  # reports $PROBE_FAKE_HEAD so the post-checkout verification can be driven
  # to either outcome; no real git command and no network is ever touched.
  printf 'git %s\\n' "$*" >> "$PROBE_GIT_LOG"
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "rev-parse" ]]; then
      printf '%s\\n' "$PROBE_FAKE_HEAD"
      return 0
    fi
  done
  if [[ "$1" == "clone" || "$1" == "init" ]]; then
    mkdir -p "${@: -1}"
  fi
  return 0
}

case "$PROBE_MODE" in
  gate_rc)
    rc=0
    crmc_commit_gate || rc=$?
    printf 'gate_rc=%s\\n' "$rc"
    ;;
  prepare)
    prepare_crmc_source
    ;;
  verify)
    mkdir -p "$crmc_src"
    verify_crmc_checkout
    ;;
  predicates)
    while IFS= read -r value; do
      g=0
      s=0
      is_git_commit "$value" || g=$?
      is_sha256 "$value" || s=$?
      printf '%s\\tgit=%s\\tsha256=%s\\n' "$value" "$g" "$s"
    done < "$PROBE_VALUES"
    ;;
  *)
    printf 'unknown PROBE_MODE: %s\\n' "$PROBE_MODE" >&2
    exit 2
    ;;
esac
printf 'PROBE_DONE\\n'
"""


@pytest.fixture(scope="session", autouse=True)
def scratch_root():
    """Fresh scratch area under dev/00runs/tmp/crmc_commit_gate/."""
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    SCRATCH_ROOT.mkdir(parents=True)
    return SCRATCH_ROOT


def _run_probe(mode: str, *, commit: str, fake_head: str, values_file: Path | None = None):
    """Run one probe bash subprocess; return (completed process, git-call log path)."""
    case_dir = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=SCRATCH_ROOT))
    functions_sh = case_dir / "functions.sh"
    functions_sh.write_text(_function_block(), encoding="utf-8")
    probe_sh = case_dir / "probe.sh"
    probe_sh.write_text(_PROBE_TEMPLATE, encoding="utf-8")
    git_log = case_dir / "git_calls.log"
    env = os.environ.copy()
    env.pop("PAPER_ROOT", None)  # keep common.sh's PAPER_ROOT resolution inert
    env.update(
        {
            "PROBE_COMMON": str(_pipeline_dir() / "common.sh"),
            "PROBE_FUNCTIONS": str(functions_sh),
            "PROBE_MODE": mode,
            "PROBE_COMMIT": commit,
            "PROBE_FAKE_HEAD": fake_head,
            "PROBE_EXTERNAL": str(case_dir / "external"),
            "PROBE_GIT_LOG": str(git_log),
            "EXTERNAL_ROOT": str(case_dir / "external"),
            "PROBE_VALUES": str(values_file if values_file is not None else case_dir / "values.txt"),
        }
    )
    proc = subprocess.run(
        ["bash", str(probe_sh)], env=env, capture_output=True, text=True, timeout=120
    )
    return proc, git_log


def _git_calls(git_log: Path) -> list[str]:
    if not git_log.exists():
        return []
    return [line for line in git_log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tail(proc: subprocess.CompletedProcess) -> str:
    return f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}"


def _printed_gate_rc(proc: subprocess.CompletedProcess) -> int:
    """Parse the ``gate_rc=<n>`` line the probe prints when the gate returns."""
    for line in proc.stdout.splitlines():
        if line.startswith("gate_rc="):
            return int(line.removeprefix("gate_rc="))
    raise AssertionError(f"the gate never returned (refused by exit?); stdout tail: {proc.stdout[-400:]}")


# ---------------------------------------------------------------------------
# Case 4: the predicates themselves
# ---------------------------------------------------------------------------

PREDICATE_CASES = [
    pytest.param(GIT_COMMIT, 0, 1, id="40-hex-is-a-git-commit-not-a-sha256"),
    pytest.param(SHA256_64, 1, 0, id="64-hex-is-a-sha256-not-a-git-commit"),
    pytest.param("abcdef0", 1, 1, id="7-hex-short-hash-rejected"),
    pytest.param("abcdef0123456789abcdef0123456789abcdef0g", 1, 1, id="non-hex-40-char-rejected"),
    pytest.param(GIT_COMMIT + "a", 1, 1, id="41-hex-rejected"),
    pytest.param(TODO_MARKER, 1, 1, id="manifest-todo-marker-rejected"),
]


@pytest.fixture(scope="session")
def predicate_results():
    values_file = SCRATCH_ROOT / "predicate_values.txt"
    values_file.write_text(
        "\n".join(str(case.values[0]) for case in PREDICATE_CASES) + "\n",
        encoding="utf-8",
    )
    proc, _ = _run_probe("predicates", commit=TODO_MARKER, fake_head=OTHER_COMMIT,
                         values_file=values_file)
    assert proc.returncode == 0, f"predicate probe failed: {proc.stderr[-800:]}"
    assert "PROBE_DONE" in proc.stdout, proc.stdout[-800:]
    results: dict[str, dict[str, int]] = {}
    for line in proc.stdout.splitlines():
        if "\tgit=" not in line:
            continue
        value, git_field, sha_field = line.split("\t")
        results[value] = {
            "git": int(git_field.removeprefix("git=")),
            "sha256": int(sha_field.removeprefix("sha256=")),
        }
    return results


@pytest.mark.parametrize(("value", "git_rc", "sha_rc"), PREDICATE_CASES)
def test_commit_and_sha256_predicates(predicate_results, value, git_rc, sha_rc):
    """is_git_commit accepts exactly 40 hex; is_sha256 still serves 64 hex."""
    got = predicate_results[value]
    assert got["git"] == git_rc, (
        f"is_git_commit {value!r}: rc={got['git']}, want {git_rc}"
    )
    assert got["sha256"] == sha_rc, (
        f"is_sha256 {value!r}: rc={got['sha256']}, want {sha_rc}"
    )


# ---------------------------------------------------------------------------
# Case 1-3: crmc_commit_gate routing
# ---------------------------------------------------------------------------

def test_gate_takes_pinned_commit_path_on_40_hex(scratch_root):
    """A 40-hex pin makes the gate signal the pinned-commit flow (rc 0)."""
    proc, _ = _run_probe("gate_rc", commit=GIT_COMMIT, fake_head=OTHER_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert "PROBE_DONE" in proc.stdout
    assert _printed_gate_rc(proc) == 0, f"stdout tail: {proc.stdout[-800:]}"


def test_gate_takes_tag_clone_path_on_todo_marker(scratch_root):
    """The manifest marker keeps the legacy-flow semantics (gate returns 1)."""
    proc, _ = _run_probe("gate_rc", commit=TODO_MARKER, fake_head=OTHER_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert "PROBE_DONE" in proc.stdout
    assert _printed_gate_rc(proc) == 1, f"stdout tail: {proc.stdout[-800:]}"


def test_gate_refuses_64_hex_sha256_pin(scratch_root):
    """A 64-hex pin is refused by exit, naming the 40-hex format and the manifest field."""
    proc, _ = _run_probe("gate_rc", commit=SHA256_64, fake_head=OTHER_COMMIT)
    assert proc.returncode != 0, f"stdout tail: {proc.stdout[-800:]}"
    assert "gate_rc=" not in proc.stdout, "the gate returned instead of refusing"
    assert f"holds a 64-hex sha256 ({SHA256_64})" in proc.stderr, f"stderr tail: {proc.stderr[-800:]}"
    assert "40 hex" in proc.stderr
    assert "generators.crmc.pinned_commit" in proc.stderr


# ---------------------------------------------------------------------------
# Case 1-3 continued: the callers route on the gate
# ---------------------------------------------------------------------------

def test_prepare_fetches_pinned_commit_on_40_hex(scratch_root):
    """prepare_crmc_source fetches and verifies the 40-hex pin; it never clones."""
    proc, git_log = _run_probe("prepare", commit=GIT_COMMIT, fake_head=GIT_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert "PROBE_DONE" in proc.stdout
    assert f"fetching CRMC source at pinned commit {GIT_COMMIT}" in proc.stdout
    assert f"CRMC commit verified: {GIT_COMMIT}" in proc.stdout
    calls = _git_calls(git_log)
    assert any(f"fetch --depth 1 origin {GIT_COMMIT}" in call for call in calls), calls
    assert any("rev-parse HEAD" in call for call in calls), calls
    assert not any("clone" in call for call in calls), (
        f"pinned flow must not fall back to the tag clone: {calls}"
    )


def test_prepare_clones_tag_on_todo_marker(scratch_root):
    """prepare_crmc_source keeps the tag-clone flow while the marker is recorded."""
    proc, git_log = _run_probe("prepare", commit=TODO_MARKER, fake_head=OTHER_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert "PROBE_DONE" in proc.stdout
    assert "WARNING: no CRMC commit recorded" in proc.stdout
    assert f"resolved commit for tag {TAG_REF}: {OTHER_COMMIT}" in proc.stdout
    calls = _git_calls(git_log)
    assert any(f"clone --branch {TAG_REF}" in call for call in calls), calls
    assert not any("fetch" in call for call in calls), (
        f"marker flow must not fetch a commit: {calls}"
    )


def test_prepare_refuses_64_hex_pin_before_any_git_call(scratch_root):
    """The 64-hex refusal fires before any git command and nothing is fetched."""
    proc, git_log = _run_probe("prepare", commit=SHA256_64, fake_head=GIT_COMMIT)
    assert proc.returncode != 0, f"stdout tail: {proc.stdout[-800:]}"
    assert "PROBE_DONE" not in proc.stdout
    assert f"holds a 64-hex sha256 ({SHA256_64})" in proc.stderr, f"stderr tail: {proc.stderr[-800:]}"
    assert "40 hex" in proc.stderr
    assert "generators.crmc.pinned_commit" in proc.stderr
    assert _git_calls(git_log) == [], "the refusal must precede every git invocation"


# ---------------------------------------------------------------------------
# The post-checkout verification is reachable for a 40-hex pin
# ---------------------------------------------------------------------------

def test_verify_checks_head_against_40_hex_pin(scratch_root):
    """verify_crmc_checkout compares rev-parse HEAD with the 40-hex pin and passes."""
    proc, git_log = _run_probe("verify", commit=GIT_COMMIT, fake_head=GIT_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert f"CRMC commit verified: {GIT_COMMIT}" in proc.stdout
    assert any("rev-parse HEAD" in call for call in _git_calls(git_log))


def test_verify_refuses_head_mismatch_for_40_hex_pin(scratch_root):
    """A checkout whose HEAD differs from the 40-hex pin is refused with exit 1."""
    proc, _ = _run_probe("verify", commit=GIT_COMMIT, fake_head=OTHER_COMMIT)
    assert proc.returncode != 0, f"stdout tail: {proc.stdout[-800:]}"
    assert "CRMC checkout does not match the pinned commit" in proc.stderr, (
        f"stderr tail: {proc.stderr[-800:]}"
    )
    assert f"expected: {GIT_COMMIT}" in proc.stderr
    assert f"actual:   {OTHER_COMMIT}" in proc.stderr  # aligned columns in the message


def test_verify_reuses_tree_for_todo_marker(scratch_root):
    """The marker keeps the documented reuse branch; no verification is attempted."""
    proc, git_log = _run_probe("verify", commit=TODO_MARKER, fake_head=OTHER_COMMIT)
    assert proc.returncode == 0, _tail(proc)
    assert "reusing CRMC source tree (no pinned commit recorded)" in proc.stdout
    assert _git_calls(git_log) == [], "the marker branch must not run git"
