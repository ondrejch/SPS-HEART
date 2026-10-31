"""Round 3 card 7: the R30 spectral-normalization driver smoke test.

Task ``TASK-20260917-r3-run-r30-spectral-normalization`` (plan section 9
run R30, line 638; the Round 3 card set, card 7).  This module is the
hermetic smoke test of the new dual-tree driver
``helpers/benchmarking/staged_r30_spectral_normalization.py``.  It
covers the three card items:

- **determinism of the report**: the driver is executed once through the
  full mode chain (--pre-registration, --integrate, --verify, --gate) in
  a scratch run dir from a clean interpreter (``-I``), and once
  in-process into a second scratch dir; ``family_integrals.json``,
  ``threshold_checks.json``, and ``gate3_r30.json`` must be reproduced
  byte for byte.  The three reports carry no wall-clock fields: their
  bytes are a pure function of the committed inputs.
- **the layout sync**: the nested and monorepo driver copies are
  byte-identical, and the layout checker
  (``helpers/benchmarking/check_helpers_layout_sync.py``) reports
  ``logic_drift=0`` over the two helper trees.
- **the no-generation contract** (round 3 index item 6): the driver
  source carries no random-number surface (no ``np.random``, no
  ``default_rng``, no ``SeedSequence``, no mass-draw or decay-routing
  surface, no ``RandomState``, no ``import random``); the
  pre-registration records the hermetic seeds block (master seed
  20260917 with ``draws: 0``), and every report carries the
  ``monte_carlo`` block with ``draws: 0`` and ``no_monte_carlo: true``.

The gate evidence structure is pinned alongside (the card's
``gate3_r30.json`` template: the three blocks -- family normalization
over all 7 families, the 5 named threshold checks, and the
double-counting identity over the three activated QN sets -- each with
its tolerance and a passing verdict, plus ``no_fitting: true`` and
``sealed: true``).

Hermetic: the scratch run dirs live under the pytest tmp tree (nothing
is written into the repository); the committed config, catalog, and
package are the only physics inputs; no event generation, no fitting.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_NESTED = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r30_spectral_normalization.py"
)
DRIVER_MONO = MONO_ROOT / "helpers" / "benchmarking" / (
    "staged_r30_spectral_normalization.py"
)
LAYOUT_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)

#: The random-number surfaces the driver must never reference (the
#: no-generation contract, round 3 index item 6: R30 draws no random
#: numbers, generates no events).
FORBIDDEN_DRAW_TOKENS = (
    "np.random",
    "default_rng",
    "SeedSequence",
    "family_draw",
    "route_source_decay",
    "RandomState",
    "import random",
)

REPORT_NAMES = (
    "family_integrals.json",
    "threshold_checks.json",
    "gate3_r30.json",
)

#: The pre-registered hermetic seeds block (the card).
MASTER_SEED = 20260917
DRAWS = 0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_driver(run_dir: Path) -> None:
    """The full R30 mode chain of the driver, one step per invocation,
    from a clean interpreter (``-I``) into the scratch run dir."""
    for mode in ("--pre-registration", "--integrate", "--verify", "--gate"):
        probe = subprocess.run(
            [sys.executable, "-I", str(DRIVER_NESTED), mode,
             "--run-dir", str(run_dir)],
            capture_output=True,
            text=True,
            cwd=str(MONO_ROOT),
        )
        assert probe.returncode == 0, (
            f"driver mode {mode} failed (return code {probe.returncode}):\n"
            f"{probe.stdout}\n{probe.stderr}"
        )


@pytest.fixture(scope="module")
def smoke_dir_a(tmp_path_factory) -> Path:
    """Scratch run dir A: the driver's full mode chain from a clean
    interpreter."""
    run_dir = tmp_path_factory.mktemp("r30_smoke") / "run_a"
    _run_driver(run_dir)
    return run_dir


def _load_driver_module():
    spec = importlib.util.spec_from_file_location(
        "staged_r30_spectral_normalization_under_test", DRIVER_NESTED
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# The layout sync
# ----------------------------------------------------------------------
def test_driver_copies_byte_identical():
    """The dual-tree contract: the nested and monorepo driver copies are
    byte-identical (the established ``_locate_roots`` pattern)."""
    nested = DRIVER_NESTED.read_bytes()
    mono = DRIVER_MONO.read_bytes()
    assert nested == mono
    assert _sha256_bytes(nested) == _sha256_bytes(mono)


def test_layout_sync_logic_drift_zero():
    """The layout checker reports ``logic_drift=0`` over the two helper
    trees (the new driver is byte-identical on both sides; the layout
    normalization allows only path-default differences elsewhere)."""
    probe = subprocess.run(
        [sys.executable, str(LAYOUT_CHECKER)],
        capture_output=True,
        text=True,
        cwd=str(MONO_ROOT),
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# Determinism of the report
# ----------------------------------------------------------------------
def test_reports_deterministic_across_invocations(smoke_dir_a, tmp_path):
    """The driver reproduces ``family_integrals.json``,
    ``threshold_checks.json``, and ``gate3_r30.json`` byte for byte from
    a second, independent invocation (in-process, fresh scratch dir);
    the reports carry no wall-clock fields."""
    run_dir_b = tmp_path / "run_b"
    run_dir_b.mkdir()
    r30 = _load_driver_module()
    r30.run_pre_registration(run_dir_b)
    r30.run_integrate(run_dir_b)
    r30.run_verify(run_dir_b)
    r30.run_gate(run_dir_b)
    for name in REPORT_NAMES:
        bytes_a = (smoke_dir_a / name).read_bytes()
        bytes_b = (run_dir_b / name).read_bytes()
        assert bytes_a == bytes_b, f"{name} is not reproducible byte for byte"
        # No wall-clock fields in the report bytes (the pure-function
        # contract the clean-environment validation re-hashes).
        text = bytes_a.decode("utf-8")
        assert "_utc" not in text
        assert "written_utc" not in text
        assert "verified_utc" not in text
        assert "sealed_at_utc" not in text


def test_gate3_blocks_structure(smoke_dir_a):
    """The gate evidence carries the card's three blocks with their
    tolerances, passing verdicts, ``no_fitting: true``, and
    ``sealed: true``."""
    gate = json.loads((smoke_dir_a / "gate3_r30.json").read_text("utf-8"))
    assert gate["gate"] == "gate 3 (plan section 9, line 650), " \
        "family-side evidence"
    assert gate["run"] == "R30"
    blocks = gate["blocks"]
    assert set(blocks) == {
        "family_normalization",
        "threshold_behavior",
        "double_counting_identity",
    }
    family_block = blocks["family_normalization"]
    assert family_block["families_checked"] == 7
    assert family_block["tolerance"] == 1e-6
    assert family_block["pass"] is True
    assert family_block["max_abs_deviation_from_unity"] <= 1e-6
    threshold_block = blocks["threshold_behavior"]
    assert threshold_block["checks"] == [
        "zero below m_min",
        "C1 at m_min",
        "C1 at m_min+W",
        "rho_total continuous at B",
        "rho_total continuous at B+W",
    ]
    assert threshold_block["pass"] is True
    identity_block = blocks["double_counting_identity"]
    assert identity_block["qn_sets"] == [[1, 1, 0], [1, 0, -1], [0, 0, 0]]
    assert identity_block["tolerance"] == 1e-6
    assert identity_block["pass"] is True
    assert identity_block["max_abs_deviation_from_unity"] <= 1e-6
    assert gate["no_fitting"] is True
    assert gate["sealed"] is True


# ----------------------------------------------------------------------
# The no-generation contract
# ----------------------------------------------------------------------
def test_driver_source_has_no_random_surface():
    """The driver source references no random-number surface (round 3
    index item 6: no Monte Carlo, no seeds, no event generation)."""
    for path in (DRIVER_NESTED, DRIVER_MONO):
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_DRAW_TOKENS:
            assert token not in source, (
                f"{path.name} references the forbidden draw surface "
                f"{token!r}"
            )


def test_pre_registration_seeds_block(smoke_dir_a):
    """The pre-registration records the hermetic seeds block: policy
    none, master seed 20260917, ``draws: 0``; no fitting; the seven
    committed families with the three activated and four declared
    inactive QN sets."""
    pre = json.loads(
        (smoke_dir_a / "pre_registration.json").read_text("utf-8")
    )
    assert pre["run"] == "R30"
    assert pre["card"] == "TASK-20260917-r3-run-r30-spectral-normalization"
    assert pre["written_before_evaluation"] is True
    assert pre["no_fitting"] is True
    seeds = pre["seeds"]
    assert seeds["master_seed"] == MASTER_SEED
    assert seeds["draws"] == DRAWS
    assert seeds["policy"].startswith("none")
    families = pre["families"]
    assert families["count"] == 7
    assert families["activated_qn_sets"] == [
        [1, 1, 0], [1, 0, -1], [0, 0, 0],
    ]
    assert families["declared_inactive_qn_sets"] == [
        [0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1],
    ]
    quadrature = pre["quadrature"]
    assert quadrature["rule"] == "trapezoid"
    assert quadrature["nodes"] == 10_001
    assert quadrature["spectral_quadrature_nodes"] == 10_001
    assert pre["normalization_tolerance"] == 1e-6
    assert pre["double_counting_identity"]["tolerance"] == 1e-6
    assert pre["threshold_behavior"]["c1_junctions"]["tolerance"] == 1e-12


def test_reports_record_zero_draws(smoke_dir_a):
    """Every report carries the zero-draw ``monte_carlo`` block and the
    integration/threshold verdicts are recorded as passing."""
    integration = json.loads(
        (smoke_dir_a / "family_integrals.json").read_text("utf-8")
    )
    threshold = json.loads(
        (smoke_dir_a / "threshold_checks.json").read_text("utf-8")
    )
    for report in (integration, threshold):
        assert report["monte_carlo"] == {
            "draws": DRAWS,
            "no_monte_carlo": True,
            "master_seed_recorded": MASTER_SEED,
        }
        assert report["no_fitting"] is True
    assert integration["family_normalization"]["pass"] is True
    assert integration["double_counting_identity"]["pass"] is True
    assert threshold["pass"] is True
    assert len(integration["families"]) == 7
    assert len(integration["activated_qn_sets"]) == 3
    assert set(threshold["checks"]) == {
        "zero below m_min",
        "C1 at m_min",
        "C1 at m_min+W",
        "rho_total continuous at B",
        "rho_total continuous at B+W",
    }
    for name, check in threshold["checks"].items():
        assert check["pass"] is True, name
