"""Coverage pins for the committed R10 data pipeline closure driver.

Companion to ``tests/test_data_pipeline_closure.py`` (the five card-named
hermetic checks of card
``TASK-20260911-r1-run-r10-data-pipeline-closure``); this module pins the
committed behavior around them.  All checks are hermetic and small-N: the
only event generation is the committed toy sampler at at most 5,000 toy
events (the 1,000,000-event sample of the run is never a test); the
gate-assembly and inventory-completeness checks run in-process on
hand-built payloads with no generation and no file round-trip of the real
tables; the CLI and seal checks use driver subprocesses whose
refusal paths fire before any table is read.  The read-only cross-checks
that load the bundled SPS tables or the ``pa_data`` p+Pb tables skip with
a recorded reason when the reference mirror is absent from the checkout
(the same convention as the committed five-test file and the R01/R02
harness tests).

CLI surface (subprocess; the refusals fire before the ``--out`` directory
is created or a single table is read):

- ``test_cli_missing_required_args_refused``: no arguments, and
  ``--mode`` without ``--out``, are argparse refusals (exit 2).
- ``test_cli_unknown_arg_refused``: an unknown flag is an argparse
  refusal (exit 2) and the ``--out`` directory is not created (the
  refusal stops the run before any work).
- ``test_cli_mode_outside_triple_refused``: a ``--mode`` value outside
  ``inventory|closure|all`` is refused with the "invalid choice" text
  (exit 2).
- ``test_cli_mc_events_nonpositive_refused_before_work``: 0 and negative
  ``--mc-events`` are refused with the exit-2 refusal text before the
  ``--out`` directory is created.
- ``test_cli_seal_requires_mode_all``: ``--seal`` with ``--mode
  inventory`` or ``--mode closure`` is refused with the committed
  refusal text (exit 2, before any work).
- ``test_cli_preregistration_mismatch_refused``: a pre-existing
  ``pre_registration.json`` that differs from the R10 pre-registration
  is refused with the exit-2 "pre-registration mismatch" text before
  the inventory is written (the registered payload is pinned, a
  different registration files its own card).
- ``test_cli_out_created_fresh_inventory_mode``: ``--mode inventory``
  on a fresh ``--out`` writes exactly ``pre_registration.json`` and
  ``data_inventory.json``; the written pre-registration equals the
  driver's default payload; no closure or seal artifacts.
- ``test_cli_mode_all_order_and_artifacts``: ``--mode all`` runs the
  inventory before the closure (the order is observable in the stdout
  lines and the artifact timestamps) and writes all three artifacts
  with no seal files (the no-``--seal`` rerun notice is printed).

Inventory completeness mechanics (in-process; the check function is
importable, so no file round-trip):

- ``test_inventory_records_and_splits``: ``build_inventory`` yields
  exactly 18 records, each carrying exactly the 16 §7.1 fields
  non-null; the six SPS tables carry ``split = "training"`` and the
  twelve p+Pb tables ``split = "held_out"``; the completeness check
  reports ``(True, 18, 18)``.
- ``test_record_complete_flags_a_single_null_field``: a hand-built
  record with all 16 fields non-null is complete; setting exactly one
  field (looped over all 16) to null flags the record; degenerate
  ``bin_edges`` (not a dict, empty, or missing one axis) are flagged;
  ``inventory_completeness`` on a hand-built 18-record payload with
  exactly one null field reports ``(False, 18, 17)``.

Closed-form toy integrals, independent of the driver's implementation:

- ``test_power_law_integral_independent_quadrature``: the xF integral in
  closed form times a 200-point trapezoid of the pT exponential
  (Richardson-corrected with the 100-point trapezoid, because the plain
  200-point trapezoid is O(h^2) ~ 1e-7..1e-8, above the 1e-9 tolerance)
  agrees with the driver's closed form to < 1e-9 on the 3x3 bin grid.
- ``test_gaussian_integral_independent_erfc_identity``: the bin
  integral recomputed with the complementary error function identity
  ``erf(a) - erf(b) = erfc(b) - erfc(a)`` (scalar stdlib ``math.erfc``
  vs the driver's vectorized ``scipy.special.erf``) agrees to < 1e-9
  on the 3x3 grid.  The per-bin uniform exactness (< 1e-15) is already
  pinned by the committed file and is not duplicated; instead
- ``test_uniform_multibin_sum_exact``: the sum of the uniform bin
  integrals over the whole 3x3 window equals ``A * xF_span * pT_span``
  to < 1e-12 (the telescoping multi-bin identity).

Analytic closure boundary and gate1 assembly (hand-built per-toy
metrics, no generation):

- ``test_gate1_verdict_logic_handbuilt``: the verdict logic pinned
  in-process by driving ``assemble_gate1`` on hand-built run
  directories.  18/18 complete tables plus every toy below the 1e-6
  analytic tolerance and at or above the 0.95 MC pass fraction
  assembles ``"pass"`` and the gate records the per-toy
  ``max_rel_deviation_analytic`` verbatim (the metric is a per-toy max
  of per-bin deviations, not a mean).  A toy at 5e-6 or 1e-5 analytic
  deviation, a toy exactly at the 1e-6 boundary, a toy at
  ``mc_sanity_pass_fraction`` 0.94, and a 17/18 complete inventory
  each flip the verdict to ``"fail"``; a toy exactly at the 0.95
  boundary still passes (the MC bound is inclusive).
- ``test_analytic_closure_metric_is_per_bin_max``: on the real
  1017-bin observable set, the reported
  ``max_rel_deviation_analytic`` equals the max of the per-bin
  ``rel_deviation_analytic`` records exactly (the metric is a max, not
  a mean), and every per-bin record reproduces
  ``|pipeline - model| / |model|``.

MC tolerance rule (pre-registered rule and its in-driver application):

- ``test_mc_tolerance_rule_and_preregistration_pins``: the committed
  constants pin the pre-registration (seed 20303000, the registered
  1,000,000 events, the tolerance rule string, the 1e-6 analytic
  tolerance, the 1017-bin observable set, the 0.95 MC pass floor), and
  the rule's two arithmetic branches are pinned: ``N_bin = 100`` gives
  ``3/sqrt(N_bin) = 0.3`` (the statistical branch) and
  ``N_bin = 10**8`` gives the ``1e-3`` floor (``3/sqrt(N_bin) = 3e-4``
  is below the floor).
- ``test_mc_closure_empty_bins_excluded_and_pass_fraction``: a small-N
  (1,000-event) in-process MC run at the committed seed: bins with
  ``N_bin == 0`` are excluded from the pass-fraction denominator and
  counted under ``empty_bins`` (``n_non_empty_bins + empty_bins ==
  1017``, the empty-bin records carry all-null tolerance fields), and
  the pass fraction equals ``mc_passes / n_non_empty_bins`` recomputed
  from the entries, with every non-empty entry at the pre-registered
  ``max(3/sqrt(N_bin), 1e-3)`` tolerance.

Seed/pipeline pinning (in-process, at most 1,000 toy events per run):

- ``test_mc_sampler_seed_reproducible_and_consumed``: the toy sampler
  with the pre-registered seed 20303000 is bit-reproducible across two
  in-process runs (same track count and same per-bin counts), a
  different seed changes the sample (the seed is actually consumed),
  and a different ``toy_index`` spawns a different stream (the
  pre-registered ``SeedSequence(mc_seed, spawn_key=(toy_index,))``
  rule).

Seal (subprocess, 100 toy events; the card's ``--seal`` path):

- ``test_cli_seal_small_run``: ``--mode all --seal --mc-events 100`` on
  a fresh run directory exercises the committed small-sample behavior.
  The pre-registered seed at 100 toy events is deterministic: the
  power-law and 2-D Gaussian MC sanity pass fractions fall below the
  0.95 floor (the sample is too small), so the gate verdict is
  ``"fail"`` and the driver exits 1.  A failure flags the sample, not
  the pipeline: the run directory is nevertheless sealed and never
  cleaned (the card forbids cleanup before the Gate 1 decision) --
  ``SHA256SUMS.txt`` is in ``sha256sum -c`` format over exactly the
  three payload files and re-hashes cleanly (a post-seal tamper of one
  payload file breaks the recorded digest), ``seal_manifest.json``
  excludes itself and records every other file with a matching digest
  and byte size, ``provenance.json`` validates, carries the seed and
  the registered versus executed event counts, records the code
  fingerprint (the git heads) and the 21 consumed inputs with matching
  digests, and the gate record carries the registered (not executed)
  MC event count.
- ``test_sealed_run_readonly_cross_checks``: the sealed 1,000,000-event
  run directory (the registered sample, read-only; skip with a
  recorded reason when the artifact store is absent) carries the
  ``"pass"`` verdict (18/18 tables, every toy analytic < 1e-6 and at
  or above the 0.95 MC floor), the immutable-archive triple re-hashes
  exactly as the small-sample run's, the pre-registration equals the
  card's payload, and the provenance record names the seed, the
  registered and executed event counts, and the code fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import staged_data_closure as r10
from helpers.benchmarking import run_provenance

NESTED_ROOT = Path(r10.NESTED_ROOT)
MONO_ROOT = Path(r10.MONO_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_data_closure.py"
SUBPROCESS_TIMEOUT = 600

# The 1017-bin grid and the p+Pb tables are read from the bundled SPS
# mirror and the pa_data files; the checks that need them skip with a
# recorded reason when the reference mirror is absent from the checkout.
SPS_DATA = NESTED_ROOT / "spshadrons.web.cern.ch" / "data"
_SPS_TABLE_NAMES = (
    "xsect_piminus.txt",
    "xsect_piplus.txt",
    "xsect_kaplus_pp.txt",
    "xsect_kaminus_pp.txt",
    "xsect_proton_pp.txt",
    "xsect_antiproton_pp.txt",
)
_SPS_PRESENT = all((SPS_DATA / name).is_file() for name in _SPS_TABLE_NAMES)
_PAPB_PRESENT = (r10.pa_data_dir() / "pPb_dens_minbias_piminus.txt").is_file()

requires_tables = pytest.mark.skipif(
    not (_SPS_PRESENT and _PAPB_PRESENT),
    reason=(
        "bundled SPS/pa_data tables absent from the checkout (source-only "
        "archive); the R10 checks that load the 1017-bin grid or the "
        "inventory need them"
    ),
)

# The committed toy sampler at small N (well under the 1,000,000-event
# registered sample, and at or under the 10,000-sample ceiling of this
# module).
SMALL_N_EVENTS = 1_000
SEAL_N_EVENTS = 100

# The same 3x3 bin grid as the committed five-test file (xF edges
# [0.0, 0.1, 0.2, 0.3], pT edges [0.05, 0.3, 0.6, 1.0]).
_X_EDGES = (0.0, 0.1, 0.2, 0.3)
_P_EDGES = (0.05, 0.3, 0.6, 1.0)

_TOY_POWER_LAW = {"name": "separable_power_law", "A": 1.0, "a": 2.0, "b": 1.0}
_TOY_GAUSSIAN = {
    "name": "gaussian_2d",
    "A": 1.0,
    "mu_xF": 0.2,
    "sig_xF": 0.1,
    "mu_pT": 0.5,
    "sig_pT": 0.3,
}
_TOY_UNIFORM = {"name": "uniform", "A": 1.0}

_SHA256SUMS_LINE = re.compile(
    r"\A([0-9a-f]{64})  (pre_registration\.json|data_inventory\.json|"
    r"closure_report\.json)\Z"
)


def _git_resolves_head(repo: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _run_r10(args, out_dir: Path):
    """One driver subprocess invocation from the nested tree root."""
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _trapz(f, lo: float, hi: float, n: int) -> float:
    """The plain trapezoid of ``f`` over ``[lo, hi]`` with ``n`` intervals."""
    xs = np.linspace(lo, hi, n + 1)
    ys = np.asarray([f(float(x)) for x in xs], dtype=float)
    return float(np.sum(ys) - 0.5 * (ys[0] + ys[-1])) * ((hi - lo) / n)


@pytest.fixture(scope="module")
def grids() -> list:
    return r10.closure_grids()


def _model_by_species(toy: dict, grids: list) -> dict:
    """The model side per species, as the committed five-test file builds it."""
    sqrt_s = r10.pipeline_sqrt_s()
    analytic = r10.analytic_closure(toy, grids, sqrt_s, n_ref=SMALL_N_EVENTS)
    model_by_species = {}
    offset = 0
    for grid in grids:
        model_by_species[grid.species_display] = np.asarray(
            [
                analytic["bins"][offset + i]["model"]
                for i in range(grid.n_bins)
            ],
            dtype=float,
        )
        offset += grid.n_bins
    return model_by_species


def _complete_record(split: str, table: str) -> dict:
    """A hand-built §7.1 table record with every field non-null."""
    return {
        "table": table,
        "experiment": "SPS",
        "paper": "hand-built citation",
        "beam_momentum_gev_c": 158.0,
        "sqrt_s_gev": 17.270203,
        "event_class": "inelastic",
        "trigger_definition": "double-arm",
        "normalization_convention": "f / 31.78 mb",
        "species": "pi-",
        "feeddown_convention": "as_published_inclusive",
        "bin_edges": {"xF": [0.0, 0.1, 0.2], "pT": [0.05, 0.1, 0.2]},
        "units": "1/GeV^2",
        "uncertainty_treatment": "per_bin_statistical + correlated_norm_nuisance",
        "covariance": "none_published; diagonal + norm_nuisance",
        "acceptance_correction": "trigger_bias_included",
        "split": split,
    }


def _write_gate_inputs(
    out_dir: Path, toy_metrics: list[dict], records: list[dict]
) -> None:
    """A hand-built run directory for ``assemble_gate1`` (no generation)."""
    _write_json(out_dir / r10.PREREGISTRATION_NAME, r10.default_pre_registration())
    _write_json(out_dir / r10.INVENTORY_NAME, {"tables": records})
    _write_json(
        out_dir / r10.CLOSURE_REPORT_NAME, {"toys": toy_metrics}
    )


# ---------------------------------------------------------------------------
# CLI surface (subprocess; the refusals fire before any work)
# ---------------------------------------------------------------------------


def test_cli_missing_required_args_refused(tmp_path):
    """No arguments, and --mode without --out, are argparse refusals."""
    empty = _run_r10([], tmp_path / "empty")
    assert empty.returncode == 2
    assert "the following arguments are required" in empty.stderr
    assert not (tmp_path / "empty").exists()

    no_out = _run_r10(["--mode", "inventory"], tmp_path / "no_out")
    assert no_out.returncode == 2
    assert "the following arguments are required" in no_out.stderr
    assert not (tmp_path / "no_out").exists()


def test_cli_unknown_arg_refused(tmp_path):
    """An unknown flag is refused before the --out directory is created."""
    out = tmp_path / "run"
    result = _run_r10(
        ["--mode", "inventory", "--out", str(out), "--bogus"], out
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert not out.exists()


def test_cli_mode_outside_triple_refused(tmp_path):
    """A --mode value outside inventory|closure|all is refused (exit 2)."""
    out = tmp_path / "run"
    result = _run_r10(["--mode", "bogus", "--out", str(out)], out)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not out.exists()


def test_cli_mc_events_nonpositive_refused_before_work(tmp_path):
    """--mc-events 0 and negative are refused before the --out is created."""
    for count in ("0", "-5"):
        out = tmp_path / f"mc_events_{count.lstrip('-')}"
        result = _run_r10(
            ["--mode", "inventory", "--out", str(out), "--mc-events", count],
            out,
        )
        assert result.returncode == 2
        assert "must be a positive integer" in result.stderr
        assert not out.exists()


def test_cli_seal_requires_mode_all(tmp_path):
    """--seal is refused unless --mode all (the gate needs both stages)."""
    for mode in ("inventory", "closure"):
        out = tmp_path / f"seal_{mode}"
        result = _run_r10(["--mode", mode, "--out", str(out), "--seal"], out)
        assert result.returncode == 2
        assert "--seal requires --mode all" in result.stderr
        assert not out.exists()


def test_cli_preregistration_mismatch_refused(tmp_path):
    """A pre-existing pre-registration that differs is refused before work."""
    out = tmp_path / "run"
    out.mkdir()
    mutated = dict(r10.default_pre_registration())
    mutated["mc_seed"] = 1
    _write_json(out / r10.PREREGISTRATION_NAME, mutated)
    result = _run_r10(["--mode", "inventory", "--out", str(out)], out)
    assert result.returncode == 2
    assert "pre-registration mismatch" in result.stderr
    assert not (out / r10.INVENTORY_NAME).is_file()


@requires_tables
def test_cli_out_created_fresh_inventory_mode(tmp_path):
    """--mode inventory on a fresh --out writes exactly the two artifacts."""
    out = tmp_path / "run"
    result = _run_r10(["--mode", "inventory", "--out", str(out)], out)
    assert result.returncode == 0, result.stderr
    written = sorted(p.name for p in out.iterdir())
    assert written == [r10.INVENTORY_NAME, r10.PREREGISTRATION_NAME]
    assert _load(out / r10.PREREGISTRATION_NAME) == r10.default_pre_registration()
    assert not (out / r10.CLOSURE_REPORT_NAME).is_file()


@requires_tables
def test_cli_mode_all_order_and_artifacts(tmp_path):
    """--mode all runs the inventory before the closure; no seal artifacts."""
    out = tmp_path / "run"
    result = _run_r10(
        ["--mode", "all", "--out", str(out), "--mc-events", str(SEAL_N_EVENTS)],
        out,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        r10.PREREGISTRATION_NAME,
        r10.INVENTORY_NAME,
        r10.CLOSURE_REPORT_NAME,
    ):
        assert (out / name).is_file()
    # No seal files without --seal.
    for name in (
        r10.PROVENANCE_NAME,
        r10.SHA256SUMS_NAME,
        r10.SEAL_NAME,
        r10.GATE_RECORD_NAME,
    ):
        assert not (out / name).is_file()
    # The order is observable in the stdout lines (the driver prints the
    # inventory summary before the closure summary, both flushed).
    inventory_idx = result.stdout.find("[r10] inventory:")
    closure_idx = result.stdout.find("[r10] closure:")
    assert inventory_idx >= 0
    assert closure_idx >= 0
    assert inventory_idx < closure_idx
    # ... and in the artifact timestamps (second resolution; the
    # inventory is written first).
    inventory = _load(out / r10.INVENTORY_NAME)
    closure = _load(out / r10.CLOSURE_REPORT_NAME)
    assert inventory["generated_utc"] <= closure["generated_utc"]
    assert closure["mc_events_used"] == SEAL_N_EVENTS
    assert (
        "[r10] artifacts written; rerun with --mode all --seal" in result.stdout
    )


# ---------------------------------------------------------------------------
# Inventory completeness mechanics (in-process)
# ---------------------------------------------------------------------------


@requires_tables
def test_inventory_records_and_splits():
    """18 records, exactly the 16 §7.1 fields, training vs held_out."""
    inventory = r10.build_inventory()
    assert inventory["field_list"] == list(r10.INVENTORY_FIELDS)
    assert len(r10.INVENTORY_FIELDS) == 16
    complete, n_tables, n_complete = r10.inventory_completeness(inventory)
    assert (complete, n_tables, n_complete) == (True, 18, 18)
    records = inventory["tables"]
    assert len(records) == 18
    for record in records:
        assert set(record.keys()) == set(r10.INVENTORY_FIELDS)

    splits = [record["split"] for record in records]
    assert splits.count("training") == 6
    assert splits.count("held_out") == 12
    sps_tables = [r for r in records if r["split"] == "training"]
    papb_tables = [r for r in records if r["split"] == "held_out"]
    for record in sps_tables:
        assert record["table"].startswith("spshadrons.web.cern.ch/data/")
        assert record["beam_momentum_gev_c"] == 158.0
        assert record["sqrt_s_gev"] == 17.270203
        assert record["normalization_convention"] == "f / 31.78 mb"
    assert {r["species"] for r in sps_tables} == {
        "pi-", "pi+", "K+", "K-", "p", "pbar",
    }
    for record in papb_tables:
        assert record["table"].startswith(
            "dev/mixing2006/helpers/benchmarking/pa_data/"
        )
        assert record["species"] in ("pi-", "pi+")


def test_record_complete_flags_a_single_null_field():
    """Exactly one null field (any of the 16) flags the record."""
    base = _complete_record("training", "hand-built/x.txt")
    assert r10.record_complete(base) is True

    for field in r10.INVENTORY_FIELDS:
        record = dict(base)
        record[field] = None
        assert r10.record_complete(record) is False, field

    # Degenerate bin_edges are flagged too.
    for bad_edges in (
        {"xF": [0.0, 0.1], "pT": []},  # empty pT edges
        {"xF": [0.0, 0.1]},  # missing pT
        {"pT": [0.05, 0.1]},  # missing xF
        [0.0, 0.1],  # not a dict
    ):
        record = dict(base)
        record["bin_edges"] = bad_edges
        assert r10.record_complete(record) is False, repr(bad_edges)

    # The payload-level check: 18 records, exactly one with a null field.
    records = [
        _complete_record("training", f"hand-built/t{i:02d}.txt")
        for i in range(18)
    ]
    records[7] = dict(base)
    records[7]["paper"] = None
    complete, n_tables, n_complete = r10.inventory_completeness(
        {"tables": records}
    )
    assert (complete, n_tables, n_complete) == (False, 18, 17)
    # An empty payload is incomplete (no table can have all fields).
    assert r10.inventory_completeness({"tables": []}) == (False, 0, 0)


# ---------------------------------------------------------------------------
# Closed-form toy integrals, independent of the driver's implementation
# ---------------------------------------------------------------------------


def test_power_law_integral_independent_quadrature():
    """xF closed form x 200-point pT trapezoid (Richardson) < 1e-9.

    The plain 200-point trapezoid of the pT exponential is O(h^2)
    (~1e-7..1e-8 absolute on this grid), above the 1e-9 tolerance, so
    the 100-point trapezoid enters as the standard Richardson
    correction ``(4*T_200 - T_100)/3`` (O(h^4), ~1e-13).  The xF axis
    stays in closed form per the card's suggestion; the committed
    five-test file already anchors the full 2-D antiderivative
    independently to < 1e-12, so the two routes cross-check the
    driver.
    """
    toy = dict(_TOY_POWER_LAW)
    b = float(toy["b"])
    worst = 0.0
    for ix in range(3):
        for ip in range(3):
            xlo, xhi = _X_EDGES[ix], _X_EDGES[ix + 1]
            plo, phi = _P_EDGES[ip], _P_EDGES[ip + 1]
            x_factor = ((1.0 - xlo) ** 3.0 - (1.0 - xhi) ** 3.0) / 3.0
            t200 = _trapz(lambda p: math.exp(-b * p), plo, phi, 200)
            t100 = _trapz(lambda p: math.exp(-b * p), plo, phi, 100)
            p_factor = (4.0 * t200 - t100) / 3.0
            independent = x_factor * p_factor
            got = float(r10.toy_bin_integral(toy, xlo, xhi, plo, phi))
            worst = max(worst, abs(got - independent))
    assert worst < 1e-9, f"worst disagreement {worst:.3e}"


def test_gaussian_integral_independent_erfc_identity():
    """The bin integral via erf(a)-erf(b) = erfc(b)-erfc(a), < 1e-9.

    The scalar stdlib ``math.erfc`` (complementary error function
    identity) replaces the driver's vectorized ``scipy.special.erf``
    differences; the 2-D prefactor ``A sig_xF sig_pT (pi/2)`` is the
    committed factorization of the two 1-D CDF differences.
    """
    toy = dict(_TOY_GAUSSIAN)
    mx, sx = float(toy["mu_xF"]), float(toy["sig_xF"])
    mp, sp = float(toy["mu_pT"]), float(toy["sig_pT"])

    def de_erfc(lo: float, hi: float, mu: float, sig: float) -> float:
        u_lo = (lo - mu) / (sig * math.sqrt(2.0))
        u_hi = (hi - mu) / (sig * math.sqrt(2.0))
        return math.erfc(u_lo) - math.erfc(u_hi)

    worst = 0.0
    for ix in range(3):
        for ip in range(3):
            xlo, xhi = _X_EDGES[ix], _X_EDGES[ix + 1]
            plo, phi = _P_EDGES[ip], _P_EDGES[ip + 1]
            independent = (
                float(toy["A"])
                * sx
                * sp
                * (math.pi / 2.0)
                * de_erfc(xlo, xhi, mx, sx)
                * de_erfc(plo, phi, mp, sp)
            )
            got = float(r10.toy_bin_integral(toy, xlo, xhi, plo, phi))
            worst = max(worst, abs(got - independent))
    assert worst < 1e-9, f"worst disagreement {worst:.3e}"


def test_uniform_multibin_sum_exact():
    """The sum of the uniform bin integrals over the window is exact.

    The per-bin exactness (< 1e-15) is already pinned by the committed
    five-test file; this pins the multi-bin telescoping identity: the
    sum over the whole 3x3 window equals ``A * xF_span * pT_span``.
    """
    total = 0.0
    for ix in range(3):
        for ip in range(3):
            total += float(
                r10.toy_bin_integral(
                    _TOY_UNIFORM,
                    _X_EDGES[ix],
                    _X_EDGES[ix + 1],
                    _P_EDGES[ip],
                    _P_EDGES[ip + 1],
                )
            )
    want = float(_TOY_UNIFORM["A"]) * (
        _X_EDGES[-1] - _X_EDGES[0]
    ) * (_P_EDGES[-1] - _P_EDGES[0])
    assert abs(total - want) < 1e-12


# ---------------------------------------------------------------------------
# Analytic closure boundary and gate1 assembly (hand-built, no generation)
# ---------------------------------------------------------------------------


def _toy_metrics(analytic: list[float], mc: list[float]) -> list[dict]:
    return [
        {
            "name": r10.TOY_SPECTRA[i]["name"],
            "max_rel_deviation_analytic": float(a),
            "mc_sanity_pass_fraction": float(m),
        }
        for i, (a, m) in enumerate(zip(analytic, mc))
    ]


def _full_records(n: int = 18) -> list[dict]:
    return [
        _complete_record(
            "training" if i < 6 else "held_out",
            f"hand-built/table_{i:02d}.txt",
        )
        for i in range(n)
    ]


def test_gate1_verdict_logic_handbuilt(tmp_path):
    """The verdict logic, driven on hand-built per-toy metrics."""
    def assemble(root: Path, toy_metrics: list[dict], records: list[dict]) -> dict:
        out = root / "run"
        out.mkdir(parents=True, exist_ok=True)
        _write_gate_inputs(out, toy_metrics, records)
        return r10.assemble_gate1(out)

    base_toy = _toy_metrics([1e-7, 1e-8, 0.0], [0.99, 0.99, 0.98])
    base_records = _full_records(18)

    # 18/18 tables + every toy below 1e-6 and at or above 0.95: pass.
    gate = assemble(tmp_path / "pass", base_toy, base_records)
    assert gate["verdict"] == "pass"
    assert gate["gate"] == "gate1"
    assert gate["clause"] == r10.GATE_CLAUSE
    assert gate["data_inventory_complete"] is True
    assert (gate["n_tables"], gate["n_tables_with_all_fields_non_null"]) == (18, 18)
    # The per-toy metric passes through verbatim: the perturbed bin's
    # 1e-7 deviation (all other bins exactly reproduced, deviation 0) is
    # what the gate records.  The metric is a max, not a mean.
    recorded = gate["closure"]["toy_spectra"]
    assert [t["name"] for t in recorded] == [t["name"] for t in base_toy]
    assert recorded[0]["max_rel_deviation_analytic"] == 1e-7
    assert recorded[0]["mc_sanity_pass_fraction"] == 0.99
    # The gate carries the registered (not executed) MC identity.
    assert gate["closure"]["mc_events"] == r10.DEFAULT_MC_EVENTS
    assert gate["closure"]["mc_seed"] == r10.MC_SEED
    assert gate["closure"]["analytic_tolerance"] == r10.ANALYTIC_TOLERANCE
    # The acceptance record is written into the run directory.
    assert _load(tmp_path / "pass" / "run" / r10.GATE_RECORD_NAME) == gate

    # A toy at 5e-6 (or 1e-5) analytic deviation flips the verdict.
    for deviation in (5e-6, 1e-5):
        toys = _toy_metrics([deviation, 1e-8, 0.0], [0.99, 0.99, 0.98])
        gate = assemble(tmp_path / f"analytic_{deviation}", toys, base_records)
        assert gate["verdict"] == "fail"
        assert gate["data_inventory_complete"] is True

    # The boundary itself fails: the criterion is a strict < 1e-6.
    toys = _toy_metrics([1e-6, 1e-8, 0.0], [0.99, 0.99, 0.98])
    gate = assemble(tmp_path / "analytic_boundary", toys, base_records)
    assert gate["verdict"] == "fail"

    # A toy at mc_sanity_pass_fraction 0.94 flips the verdict.
    toys = _toy_metrics([1e-7, 1e-8, 0.0], [0.94, 0.99, 0.98])
    gate = assemble(tmp_path / "mc_fail", toys, base_records)
    assert gate["verdict"] == "fail"

    # The 0.95 boundary passes: the MC bound is inclusive (>=).
    toys = _toy_metrics([1e-7, 1e-8, 0.0], [0.95, 0.99, 0.98])
    gate = assemble(tmp_path / "mc_boundary", toys, base_records)
    assert gate["verdict"] == "pass"

    # 17/18 tables complete flips the verdict.
    records = _full_records(18)
    records[3]["paper"] = None
    gate = assemble(tmp_path / "inventory_incomplete", base_toy, records)
    assert gate["verdict"] == "fail"
    assert gate["data_inventory_complete"] is False
    assert (gate["n_tables"], gate["n_tables_with_all_fields_non_null"]) == (18, 17)


@requires_tables
def test_analytic_closure_metric_is_per_bin_max(grids):
    """The reported max equals the per-bin max exactly (a max, not a mean)."""
    sqrt_s = r10.pipeline_sqrt_s()
    analytic = r10.analytic_closure(
        dict(_TOY_UNIFORM), grids, sqrt_s, n_ref=SMALL_N_EVENTS
    )
    assert analytic["n_bins"] == 1017
    per_bin = [b["rel_deviation_analytic"] for b in analytic["bins"]]
    assert analytic["max_rel_deviation_analytic"] == max(per_bin)
    # Every per-bin record reproduces |pipeline - model| / |model|.
    for record in analytic["bins"]:
        rel = abs(record["analytic_pipeline"] - record["model"]) / abs(
            record["model"]
        )
        assert rel == record["rel_deviation_analytic"]
    assert analytic["max_rel_deviation_analytic"] < r10.ANALYTIC_TOLERANCE


# ---------------------------------------------------------------------------
# MC tolerance rule (pre-registered rule and its in-driver application)
# ---------------------------------------------------------------------------


def test_mc_tolerance_rule_and_preregistration_pins():
    """The pre-registered identity and the rule's two arithmetic branches."""
    # The committed constants pin the pre-registration (card schema).
    assert r10.MC_SEED == 20303000
    assert r10.DEFAULT_MC_EVENTS == 1_000_000
    assert r10.ANALYTIC_TOLERANCE == 1e-6
    assert r10.MC_PASS_FRACTION_MIN == 0.95
    assert r10.MC_TOLERANCE_RULE == "max(3 / sqrt(N_bin), 1e-3) per bin"
    assert r10.SIGMA_INEL_MB == 31.78
    assert r10.TOTAL_BINS == 1017
    assert r10.PAPER_WINDOW == {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    assert sum(r10.SPECIES_BIN_COUNTS.values()) == 1017

    prereg = r10.default_pre_registration()
    assert prereg["mc_seed"] == 20303000
    assert prereg["mc_events"] == 1_000_000
    assert prereg["mc_tolerance_rule"] == r10.MC_TOLERANCE_RULE
    assert prereg["analytic_tolerance"] == 1e-6
    assert prereg["observable_set"] == {
        "window": {"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
        "total_bins": 1017,
    }
    assert prereg["toy_spectra"] == [dict(t) for t in r10.TOY_SPECTRA]
    assert [t["name"] for t in prereg["toy_spectra"]] == [
        "separable_power_law",
        "gaussian_2d",
        "uniform",
    ]

    # The rule's two arithmetic branches: the statistical branch
    # (3/sqrt(N_bin)) and the 1e-3 floor.
    assert max(3.0 / math.sqrt(100), 1e-3) == 0.3
    assert max(3.0 / math.sqrt(10**8), 1e-3) == 1e-3
    assert 3.0 / math.sqrt(100) > 1e-3  # the statistical branch governs
    assert 3.0 / math.sqrt(10**8) < 1e-3  # the floor governs


@requires_tables
def test_mc_closure_empty_bins_excluded_and_pass_fraction(grids):
    """Small-N MC run: N_bin == 0 excluded, counted, fraction identity."""
    toy = dict(_TOY_POWER_LAW)
    model_by_species = _model_by_species(toy, grids)
    sqrt_s = r10.pipeline_sqrt_s()
    mc = r10.mc_closure(
        toy,
        0,
        grids,
        model_by_species,
        sqrt_s,
        n_events=SMALL_N_EVENTS,
        mc_seed=r10.MC_SEED,
    )
    assert mc["mc_events_used"] == SMALL_N_EVENTS
    assert mc["mc_seed"] == r10.MC_SEED
    assert mc["mc_tracks"] > 0
    entries = mc["bins"]
    assert len(entries) == 1017
    # N_bin == 0 bins are excluded from the denominator and counted.
    assert mc["n_non_empty_bins"] + mc["empty_bins"] == 1017
    assert mc["empty_bins"] == sum(1 for e in entries if e["n_bin"] == 0)
    assert mc["empty_bins"] == len(mc["empty_bin_records"])
    for entry in entries:
        if entry["n_bin"] == 0:
            assert entry["mc"] == 0.0
            assert entry["rel_deviation_mc"] is None
            assert entry["tolerance"] is None
            assert entry["mc_pass"] is None
        else:
            tol = max(3.0 / math.sqrt(entry["n_bin"]), 1e-3)
            assert entry["tolerance"] == tol
            assert entry["mc_pass"] == (entry["rel_deviation_mc"] <= tol)
    # The pass fraction is n_pass / n_non_empty (the empty bins do not
    # enter the denominator).
    n_pass = sum(1 for e in entries if e["mc_pass"] is True)
    assert n_pass == mc["mc_passes"]
    if mc["n_non_empty_bins"] > 0:
        assert (
            mc["mc_sanity_pass_fraction"]
            == float(mc["mc_passes"]) / float(mc["n_non_empty_bins"])
        )
    # The empty-bin records name the species and the per-grid bin index.
    for record in mc["empty_bin_records"]:
        assert record["species"] in r10.SPECIES_DISPLAY.values()
        assert isinstance(record["bin_index"], int)


# ---------------------------------------------------------------------------
# Seed/pipeline pinning (in-process, at most 1,000 toy events per run)
# ---------------------------------------------------------------------------


@requires_tables
def test_mc_sampler_seed_reproducible_and_consumed(grids):
    """Seed 20303000 is bit-reproducible; seed and toy_index are consumed."""
    toy = dict(_TOY_POWER_LAW)
    model_by_species = _model_by_species(toy, grids)
    sqrt_s = r10.pipeline_sqrt_s()

    def signature(result: dict) -> tuple:
        return (
            result["mc_tracks"],
            tuple(
                (e["n_bin"], e["mc"]) for e in result["bins"]
            ),
        )

    run_a = r10.mc_closure(
        toy, 0, grids, model_by_species, sqrt_s,
        n_events=SMALL_N_EVENTS, mc_seed=r10.MC_SEED,
    )
    run_b = r10.mc_closure(
        toy, 0, grids, model_by_species, sqrt_s,
        n_events=SMALL_N_EVENTS, mc_seed=r10.MC_SEED,
    )
    # Bit-reproducible across two in-process runs.
    assert run_a["mc_tracks"] == run_b["mc_tracks"]
    assert run_a["lambda_tracks_per_event"] == run_b["lambda_tracks_per_event"]
    assert signature(run_a) == signature(run_b)
    assert run_a["sampler_proposals"] == run_b["sampler_proposals"]

    # A different seed changes the sample (the seed is actually consumed).
    run_other_seed = r10.mc_closure(
        toy, 0, grids, model_by_species, sqrt_s,
        n_events=SMALL_N_EVENTS, mc_seed=r10.MC_SEED + 1,
    )
    assert signature(run_other_seed) != signature(run_a)

    # A different toy_index spawns a different stream (the pre-registered
    # SeedSequence(mc_seed, spawn_key=(toy_index,)) rule).
    run_other_index = r10.mc_closure(
        toy, 1, grids, model_by_species, sqrt_s,
        n_events=SMALL_N_EVENTS, mc_seed=r10.MC_SEED,
    )
    assert signature(run_other_index) != signature(run_a)


# ---------------------------------------------------------------------------
# Seal (subprocess, 100 toy events; the card's --seal path) and the
# sealed 1,000,000-event run directory (read-only)
# ---------------------------------------------------------------------------

# The sealed R10 run directory (gitignored artifact store, read-only).
SEALED_R10_DIR = MONO_ROOT / "dev" / "00runs" / "r10_data_pipeline_closure_2026-09-11"

#: Consumed inputs whose file was updated after the R10 seal, keyed by path
#: suffix, with the digest the sealed provenance records. The pa_data README
#: gained the p+Pb table documentation and the corrected centrality
#: definition (grey-proton bins) in nested 5292d45 (2026-09-29); the sealed
#: run consumed the README of nested cdc379d. The 20 data tables it consumed
#: are byte-unchanged and stay checked against disk.
R10_INPUTS_UPDATED_AFTER_SEAL = {
    "helpers/benchmarking/pa_data/README.md":
        "905f0a061819731b04e2d60b70cad9c34e9e73d94039c6cdd9c38ee6f537de5e",
}
_SEALED_R10_PRESENT = (SEALED_R10_DIR / r10.GATE_RECORD_NAME).is_file() and (
    SEALED_R10_DIR / r10.PROVENANCE_NAME
).is_file() and (SEALED_R10_DIR / r10.SHA256SUMS_NAME).is_file() and (
    SEALED_R10_DIR / r10.SEAL_NAME
).is_file()

requires_sealed_r10 = pytest.mark.skipif(
    not _SEALED_R10_PRESENT,
    reason=(
        "sealed R10 run directory not present on this checkout "
        f"({SEALED_R10_DIR.name}); read-only cross-check skipped"
    ),
)


def _verify_seal_artifacts(out: Path, tamper: bool = True) -> None:
    """The immutable-archive triple of a sealed R10 run directory.

    The mechanism is identical for the small-sample subprocess run and
    the sealed 1,000,000-event run, so both drive the same checks:
    ``SHA256SUMS.txt`` in ``sha256sum -c`` format over exactly the three
    payload files (re-hashing cleanly, and breaking under a post-seal
    tamper of one payload file when ``tamper``), ``seal_manifest.json``
    excluding itself and matching every other file's digest and byte
    size, and ``provenance.json`` validating with the seed, the
    registered versus executed event counts, the code fingerprint, and
    the 21 consumed inputs with matching digests.  ``tamper`` must stay
    off for the sealed artifact store: the run directory is read-only.
    """
    expected_files = {
        r10.PREREGISTRATION_NAME,
        r10.INVENTORY_NAME,
        r10.CLOSURE_REPORT_NAME,
        r10.PROVENANCE_NAME,
        r10.SHA256SUMS_NAME,
        r10.SEAL_NAME,
        r10.GATE_RECORD_NAME,
    }
    assert {p.name for p in out.iterdir()} == expected_files

    # SHA256SUMS.txt: sha256sum -c format over exactly the three payload
    # files (the seal/provenance/acceptance files cannot carry their own
    # digests), each re-hashing cleanly.
    lines = (out / r10.SHA256SUMS_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    covered = set()
    sums_digests = {}
    for line in lines:
        match = _SHA256SUMS_LINE.fullmatch(line)
        assert match is not None, f"malformed sums line: {line!r}"
        digest, name = match.groups()
        covered.add(name)
        sums_digests[name] = digest
        assert _sha256_of(out / name) == digest, name
    assert covered == {
        r10.PREREGISTRATION_NAME, r10.INVENTORY_NAME, r10.CLOSURE_REPORT_NAME
    }

    # seal_manifest.json: every other file with a matching digest and
    # byte size; it cannot contain its own digest.
    seal = _load(out / r10.SEAL_NAME)
    sealed_paths = {entry["path"] for entry in seal["files"]}
    assert r10.SEAL_NAME not in sealed_paths
    assert sealed_paths == expected_files - {r10.SEAL_NAME}
    for entry in seal["files"]:
        path = out / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha256_of(path) == entry["sha256"]
    assert seal["card"] == r10.CARD

    # provenance.json: validates, carries the seed and the registered
    # versus executed event counts, the code fingerprint, and the 21
    # consumed inputs (the two configs, the pa_data README, the six SPS
    # tables, and the twelve p+Pb tables) with matching digests.
    prov = _load(out / r10.PROVENANCE_NAME)
    assert run_provenance.validate_run_provenance(out / r10.PROVENANCE_NAME) == []
    assert prov["producer"] == "staged_data_closure.py"
    assert prov["rng"]["seeds"]["mc_seed"] == r10.MC_SEED
    assert prov["rng"]["seeds"]["toy_indices"] == [0, 1, 2]
    assert prov["mc_seed"] == r10.MC_SEED
    assert prov["mc_events_registered"] == r10.DEFAULT_MC_EVENTS
    assert prov["gate_clause"] == r10.GATE_CLAUSE
    # A git checkout records both heads; a source-only tree (the gateway
    # stage, a release archive) records the SHA256SUMS.txt inventory.
    source = prov["source"]
    if _git_resolves_head(NESTED_ROOT) and _git_resolves_head(MONO_ROOT):
        assert source["recorded_from"] == "git"
        assert run_provenance.is_git_commit(source["nested_head"])
        assert run_provenance.is_git_commit(source["outer_head"])
    else:
        assert source["recorded_from"] == "source_inventory"
        assert run_provenance.is_sha256(source["source_inventory_sha256"])
    assert prov["sigma_inel_mb"] == r10.SIGMA_INEL_MB
    assert prov["use_data_covariance"] is False
    assert len(prov["inputs"]) == 3 + 6 + 12
    for entry in prov["inputs"]:
        assert run_provenance.is_sha256(entry["sha256"])
        if any(entry["path"].endswith(suffix) and entry["sha256"] == digest
               for suffix, digest in R10_INPUTS_UPDATED_AFTER_SEAL.items()):
            continue  # the sealed record of a documented post-seal update
        assert _sha256_of(Path(entry["path"])) == entry["sha256"], entry["path"]

    if not tamper:
        return

    # A post-seal tamper of one payload file breaks the digest recorded
    # by both seal mechanisms.  Runs last: the checks above must see the
    # pristine sealed files.
    seal_digest = next(
        entry["sha256"]
        for entry in seal["files"]
        if entry["path"] == r10.INVENTORY_NAME
    )
    assert seal_digest == sums_digests[r10.INVENTORY_NAME]
    with (out / r10.INVENTORY_NAME).open("ab") as handle:
        handle.write(b"tampered\n")
    assert _sha256_of(out / r10.INVENTORY_NAME) != seal_digest


@requires_tables
def test_cli_seal_small_run(tmp_path):
    """The --seal path on a small run directory (100 toy events).

    The pre-registered seed at 100 toy events is deterministic: the
    power-law and 2-D Gaussian MC sanity pass fractions fall below the
    0.95 floor (the sample is too small), so the verdict is ``"fail"``
    and the driver exits 1.  A failure flags the sample, not the
    pipeline: the run directory is nevertheless sealed and never
    cleaned (the card forbids cleanup before the Gate 1 decision).
    """
    out = tmp_path / "run"
    result = _run_r10(
        [
            "--mode", "all",
            "--out", str(out),
            "--mc-events", str(SEAL_N_EVENTS),
            "--seal",
        ],
        out,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[r10] run directory sealed" in result.stdout
    assert "gate1_r10.json verdict fail" in result.stdout

    # The acceptance record: fail (a sample, not a pipeline, problem),
    # the complete 18-table inventory, and the registered (not executed)
    # MC identity.
    gate = _load(out / r10.GATE_RECORD_NAME)
    assert gate["verdict"] == "fail"
    assert gate["data_inventory_complete"] is True
    assert (gate["n_tables"], gate["n_tables_with_all_fields_non_null"]) == (18, 18)
    assert gate["closure"]["mc_events"] == r10.DEFAULT_MC_EVENTS
    assert gate["closure"]["mc_seed"] == r10.MC_SEED
    toy_records = gate["closure"]["toy_spectra"]
    assert [t["name"] for t in toy_records] == [
        t["name"] for t in r10.TOY_SPECTRA
    ]
    for toy_record in toy_records:
        assert toy_record["max_rel_deviation_analytic"] < r10.ANALYTIC_TOLERANCE
    # The deterministic small sample: the two peaked toys' non-empty
    # bins all miss the tolerance at this size (the recorded pass
    # fractions are exactly zero), and the uniform fraction stays below
    # the 0.95 floor that flips the verdict.
    by_name = {t["name"]: t for t in toy_records}
    assert by_name["separable_power_law"]["mc_sanity_pass_fraction"] < 0.95
    assert by_name["gaussian_2d"]["mc_sanity_pass_fraction"] < 0.95
    assert by_name["uniform"]["mc_sanity_pass_fraction"] < 0.95

    # The closure report records the executed (small) sample and the
    # deterministic empty-bin counts of the 100-event sample.
    closure = _load(out / r10.CLOSURE_REPORT_NAME)
    assert closure["mc_events_used"] == SEAL_N_EVENTS
    assert closure["mc_seed"] == r10.MC_SEED
    assert closure["mc_sanity_pass_fraction_min"] < r10.MC_PASS_FRACTION_MIN
    for toy_report in closure["toys"]:
        assert toy_report["n_non_empty_bins"] + toy_report["empty_bins"] == 1017
        assert toy_report["mc_tracks"] > 0

    # The run directory is sealed and never cleaned before the Gate 1
    # decision: the immutable-archive triple is present and consistent.
    _verify_seal_artifacts(out)
    prov = _load(out / r10.PROVENANCE_NAME)
    assert prov["mc_events_executed"] == SEAL_N_EVENTS


@requires_sealed_r10
def test_sealed_run_readonly_cross_checks():
    """The sealed 1,000,000-event run directory (read-only)."""
    out = SEALED_R10_DIR

    # The acceptance record: the registered sample passes Gate 1.
    gate = _load(out / r10.GATE_RECORD_NAME)
    assert gate["gate"] == "gate1"
    assert gate["clause"] == r10.GATE_CLAUSE
    assert gate["verdict"] == "pass"
    assert gate["data_inventory_complete"] is True
    assert (gate["n_tables"], gate["n_tables_with_all_fields_non_null"]) == (18, 18)
    assert gate["closure"]["mc_events"] == r10.DEFAULT_MC_EVENTS
    assert gate["closure"]["mc_seed"] == r10.MC_SEED
    assert gate["closure"]["analytic_tolerance"] == r10.ANALYTIC_TOLERANCE
    toy_records = gate["closure"]["toy_spectra"]
    assert [t["name"] for t in toy_records] == [
        t["name"] for t in r10.TOY_SPECTRA
    ]
    for toy_record in toy_records:
        assert toy_record["max_rel_deviation_analytic"] < r10.ANALYTIC_TOLERANCE
        assert toy_record["mc_sanity_pass_fraction"] >= r10.MC_PASS_FRACTION_MIN

    # The closure report: the registered sample was executed, and the
    # per-toy summary metrics clear the gate criteria.
    closure = _load(out / r10.CLOSURE_REPORT_NAME)
    assert closure["mc_events_used"] == r10.DEFAULT_MC_EVENTS
    assert closure["pre_registered_mc_events"] == r10.DEFAULT_MC_EVENTS
    assert closure["mc_seed"] == r10.MC_SEED
    assert closure["max_rel_deviation_analytic_all"] < r10.ANALYTIC_TOLERANCE
    assert closure["mc_sanity_pass_fraction_min"] >= r10.MC_PASS_FRACTION_MIN

    # The pre-registration equals the card's payload (pinned, unmutated).
    assert _load(out / r10.PREREGISTRATION_NAME) == r10.default_pre_registration()

    # The immutable-archive triple re-hashes exactly as the small-
    # sample run's (the seal mechanism is the same).  Read-only: no
    # tamper of the sealed artifact store.
    _verify_seal_artifacts(out, tamper=False)
    prov = _load(out / r10.PROVENANCE_NAME)
    assert prov["mc_events_executed"] == r10.DEFAULT_MC_EVENTS
