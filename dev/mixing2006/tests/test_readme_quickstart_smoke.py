"""Archive-safe Quick Start smoke coverage for README.md and examples/.

Phase P2 contract: every command documented under the README heading
``### Archive-Safe Quick Start`` must run on a clean extraction that carries
only bundled package material, so this module

1. parses those commands out of ``README.md`` and asserts they match a pinned
   expectation here (documentation cannot drift silently),
2. rebuilds a minimal sandbox under ``dev/00runs/tmp/review3_p2/`` that holds
   byte-copies of ``na49_mixing/`` and ``examples/`` and nothing else, i.e.
   neither ``.legacy/``, nor ``external/``, nor ``spshadrons.web.cern.ch/``,
   nor ``00runs/`` at its root, and executes the documented commands inside it,
3. asserts static hygiene (no ``/home/`` absolute paths in README.md or
   examples/*.py),
4. asserts the ``--na49-data`` argument of ``examples/benchmark_na49_models.py``
   is required and points users at the SPS HADRONS ``xsect_piminus.txt`` table,

There are no skip conditions by construction: if a documented command cannot
run hermetically, the documentation command has to change instead.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PKG_ROOT / "README.md"
EXAMPLES_DIR = PKG_ROOT / "examples"

# All fixture output lives under dev/00runs/tmp/review3_p2/.
SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "00runs" / "tmp" / "review3_p2"
SANDBOX_ROOT = SCRATCH_ROOT / "smoke_sandbox"

QUICKSTART_TITLE = "Archive-Safe Quick Start"
FULLCHECKOUT_TITLE = "Full Development Checkout"

TOTAL_BUDGET_S = 60.0


def _read(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8")


def _extract_heading_section(text: str, title: str) -> list[str]:
    """Return body lines of the ``<level> <title>`` section up to the next
    heading of the same or higher level."""
    lines = text.splitlines()
    start = None
    level = None
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if match and match.group(2) == title:
            start = idx + 1
            level = len(match.group(1))
            break
    assert start is not None, f"heading {title!r} not found"
    assert level is not None
    end = len(lines)
    for idx in range(start, len(lines)):
        match = re.match(r"^(#{1,4})\s+", lines[idx])
        if match and len(match.group(1)) <= level:
            end = idx
            break
    assert end is not None
    return lines[start:end]


def _bash_block_lines(section_lines: list[str]) -> list[str]:
    """Concatenate inner lines of every fenced ```bash block."""
    collected: list[str] = []
    inside = False
    for line in section_lines:
        stripped = line.strip()
        if stripped.startswith("```bash"):
            inside = True
            continue
        if inside and stripped.startswith("```"):
            inside = False
            continue
        if inside:
            collected.append(line)
    assert not inside, "unterminated bash fence"
    return collected


def _normalize_command(line: str) -> str:
    """Collapse whitespace after joining backslash continuations."""
    return " ".join(line.split())


def _logical_commands(lines: list[str]) -> list[str]:
    """Join trailing-backslash continuations into normalized one-line commands.

    Comment lines and blank lines are dropped. Each returned string is ready
    for shlex.split().
    """
    commands: list[str] = []
    current: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continuation = stripped.endswith("\\")
        if continuation:
            stripped = stripped[:-1].rstrip()
        current.append(stripped)
        if not continuation:
            commands.append(_normalize_command(" ".join(current)))
            current = []
    assert not current, "dangling continuation line"
    return commands


def _archive_safe_commands() -> list[str]:
    return _logical_commands(
        _bash_block_lines(_extract_heading_section(_read(README_PATH), QUICKSTART_TITLE))
    )


# ---------------------------------------------------------------------------
# Pinned expectations
# ---------------------------------------------------------------------------

GEN_CARD = "na49_mixing/generators/michal/deltas-to-pion/delta++.gen"

EXPECTED_ARCHIVE_SAFE_COMMANDS = [
    "mkdir -p 00runs/tmp/quickstart",
    (
        "na49-generate "
        f"--genfile {GEN_CARD} "
        "--outfile 00runs/tmp/quickstart/delta_pp_tracks.npz "
        "--nevents 300 --seed 20260404 --n-workers 1 "
        "--use-soft-decay-factory"
    ),
    (
        f"python examples/generate_deltas.py delta++ --genfile {GEN_CARD} "
        "--use-soft-decay-factory --nevents 500 --seed 424242 "
        "--outfile 00runs/tmp/quickstart/delta_pp.npz"
    ),
    (
        "python examples/plot_spectra.py "
        "--files 00runs/tmp/quickstart/delta_pp.npz "
        "--labels Delta++ --out 00runs/tmp/quickstart/delta_pp.pdf"
    ),
    (
        "python examples/generate_rho_omega.py --nevents 200 --seed 12345 "
        "--outfile-rho 00runs/tmp/quickstart/rho770.npz "
        "--outfile-omega 00runs/tmp/quickstart/omega782.npz"
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sandbox() -> Path:
    """Rebuild the clean-extraction mirror under dev/00runs/tmp/review3_p2."""
    shutil.rmtree(SANDBOX_ROOT, ignore_errors=True)
    SANDBOX_ROOT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PKG_ROOT / "na49_mixing", SANDBOX_ROOT / "na49_mixing",
                    ignore=ignore)
    shutil.copytree(EXAMPLES_DIR, SANDBOX_ROOT / "examples", ignore=ignore)

    # Prove the mirror really is an archive-condition extraction: none of the
    # development-only trees may be reachable from the sandbox root.
    for forbidden in ("00runs", ".legacy", "external", "spshadrons.web.cern.ch",
                      "tests", "helpers", "sps_resonances_paper"):
        assert not (SANDBOX_ROOT / forbidden).exists(), (
            f"sandbox unexpectedly exposes {forbidden!r}"
        )

    # Mirror the packaged console script defined in pyproject.toml
    # ([project.scripts] na49-generate = "na49_mixing._cli:generate_main")
    # without requiring an installed distribution, so the documented
    # command line and argv surface stay identical in the subprocesses.
    bin_dir = SANDBOX_ROOT / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "na49-generate"
    launcher.write_text(
        "#!/usr/bin/env python\n"
        "import sys\n\n"
        "from na49_mixing._cli import generate_main\n\n"
        "sys.exit(generate_main())\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return SANDBOX_ROOT


@pytest.fixture(scope="module")
def sandbox_root() -> Path:
    return _make_sandbox()


def _sandbox_env() -> dict:
    env = os.environ.copy()
    # Imports resolve to the sandbox copies; bytecode stays unrequested.
    env["PYTHONPATH"] = str(SANDBOX_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Matplotlib config/cache confined to the scratch area.
    mpl_cfg = SANDBOX_ROOT / "mplconfig"
    mpl_cfg.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cfg)
    env["PATH"] = f"{SANDBOX_ROOT / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        shlex.split(command),
        cwd=SANDBOX_ROOT,
        env=_sandbox_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Static contract: documentation shape and hygiene
# ---------------------------------------------------------------------------

def test_readme_quickstart_section_matches_pinned_commands():
    extracted = _archive_safe_commands()
    assert extracted == EXPECTED_ARCHIVE_SAFE_COMMANDS, (
        "README Archive-Safe Quick Start drifted from pinned commands:\n"
        f"extracted={extracted!r}"
    )


def test_archive_safe_section_references_only_bundled_material():
    section = "\n".join(
        _extract_heading_section(_read(README_PATH), QUICKSTART_TITLE)
    )
    for token in (".legacy/", "spshadrons", "external/", "/home/"):
        assert token not in section, (
            f"archive-safe section references {token!r}, which review "
            "archives do not carry"
        )


def test_full_checkout_section_names_excluded_trees():
    full = "\n".join(
        _extract_heading_section(_read(README_PATH), FULLCHECKOUT_TITLE)
    )
    readme = _read(README_PATH)
    for token in (".legacy/resonance-picture/",
                  "spshadrons.web.cern.ch/data/xsect_piminus.txt",
                  "review archives exclude"):
        assert token in full, f"full-checkout section misses {token!r}"
    # The exclusion statement travels wherever `.legacy/` presence matters.
    legacy_mentions = [line for line in readme.splitlines() if ".legacy/" in line]
    assert legacy_mentions, ".legacy/ not mentioned anywhere"
    joined_readme = readme
    assert joined_readme.count("excluded") >= 3, (
        "the excluded-from-review-archives statement is missing beside the "
        "dev-only trees"
    )


def test_no_absolute_home_paths_in_readme_or_examples():
    offenders: list[str] = []
    texts = [(_read(README_PATH), str(README_PATH))]
    for script in sorted(EXAMPLES_DIR.glob("*.py")):
        texts.append((script.read_text(encoding="utf-8"), str(script)))
    for content, origin in texts:
        if "/home/" in content:
            offenders.append(origin)
    assert offenders == [], f"/home/ absolute paths found in: {offenders}"


# ---------------------------------------------------------------------------
# CLI contract: benchmark_na49_models requires --na49-data
# ---------------------------------------------------------------------------

def test_benchmark_help_advertises_required_na49_data(sandbox_root):
    env = _sandbox_env()
    help_run = subprocess.run(
        [sys.executable, "examples/benchmark_na49_models.py", "--help"],
        cwd=sandbox_root, env=env, capture_output=True, text=True, timeout=120,
    )
    assert help_run.returncode == 0, help_run.stderr[-1000:]
    assert "--na49-data" in help_run.stdout
    assert "xsect_piminus.txt" in help_run.stdout

    missing_run = subprocess.run(
        [sys.executable, "examples/benchmark_na49_models.py"],
        cwd=sandbox_root, env=env, capture_output=True, text=True, timeout=120,
    )
    assert missing_run.returncode != 0
    combined = missing_run.stdout + missing_run.stderr
    assert "--na49-data" in combined and "required" in combined


# ---------------------------------------------------------------------------
# Execution: the documented pipeline runs hermetically
# ---------------------------------------------------------------------------

def test_quickstart_commands_execute_in_clean_extraction(sandbox_root):
    commands = _archive_safe_commands()

    t0 = time.monotonic()
    outputs: list[subprocess.CompletedProcess] = []
    for command in commands:
        proc = _run(command)
        outputs.append(proc)
        assert proc.returncode == 0, (
            f"documented command failed ({command!r})\n"
            f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}"
        )
    elapsed = time.monotonic() - t0
    assert elapsed < TOTAL_BUDGET_S, (
        f"quick-start execution took {elapsed:.1f}s, target is <~60s"
    )

    qdir = sandbox_root / "00runs" / "tmp" / "quickstart"

    # 1. Track-level output from na49-generate.
    with np.load(qdir / "delta_pp_tracks.npz", allow_pickle=False) as data:
        files = set(data.files)
        assert {"pdg", "xf", "pt", "source_side", "meta_json"} <= files
        pdg = data["pdg"]
        meta = json.loads(str(np.asarray(data["meta_json"])))
    assert pdg.shape[0] >= 1
    assert meta["n_events"] == 300
    assert meta["seed"] == 20260404
    assert meta["n_workers"] == 1
    assert meta["n_tracks"] == int(pdg.shape[0])

    # 2. Delta histogram sample written by generate_deltas.py.
    with np.load(qdir / "delta_pp.npz", allow_pickle=False) as data:
        keys = [str(x) for x in data["__hist_keys"]]
        fills = {}
        for key in keys:
            fills[key] = float(np.sum(data[key + "__contents"]))
    for needed in ("mp", "xfp", "xfdp", "xfdz"):
        assert needed in keys, f"missing histogram {needed!r}: {keys}"
    assert fills["xfp"] > 0.0, "parent x_F histogram received no entries"
    assert fills["xfdp"] > 0.0, "daughter pi+ x_F histogram received no entries"

    # 3. Plot PDF produced by plot_spectra.py.
    pdf_bytes = (qdir / "delta_pp.pdf").read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    # 4. Rho/omega histogram samples.
    for name in ("rho770.npz", "omega782.npz"):
        with np.load(qdir / name, allow_pickle=False) as data:
            sample_keys = [str(x) for x in data["__hist_keys"]]
        assert sample_keys, f"{name} carries no histograms"


def test_seeded_generation_is_reproducible(sandbox_root):
    """Identical seeds produce identical histogram contents in the sandbox."""
    env = _sandbox_env()
    base_cmd = [
        sys.executable, "examples/generate_deltas.py", "delta++",
        "--genfile", GEN_CARD, "--use-soft-decay-factory", "--nevents", "150",
    ]
    results = {}
    for tag, seed in (("first", "777"), ("second", "777")):
        outfile = sandbox_root / "00runs" / "tmp" / "repro" / f"delta_{tag}.npz"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            base_cmd + ["--seed", seed, "--outfile", str(outfile)],
            cwd=sandbox_root, env=env, capture_output=True, text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        with np.load(outfile, allow_pickle=False) as data:
            results[tag] = {
                key: data[key + "__contents"].copy() for key in ("mp", "xfp", "xfdp")
            }
    for key in ("mp", "xfp", "xfdp"):
        np.testing.assert_array_equal(results["first"][key], results["second"][key])
