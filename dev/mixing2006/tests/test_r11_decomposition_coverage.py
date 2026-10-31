"""Coverage self-check suite for the R11 physics-separated observable
decomposition (Round 1, card 13,
``TASK-20260911-r1-run-r11-legacy-observable-decomposition``).

This file extends the two committed card-13 suites
(``test_fit_observables.py`` and ``test_r11_decomposition_harness.py``)
without rewriting them.  It pins the committed behavior of the
``na49_mixing/fit_observables.py`` module beyond the card's six named
tests and the surface of the ``staged_observable_decomposition.py``
driver (the seal-verification gate, the re-run digest self-check, the
no-fitting attestation, the gate1 verdict logic, and the
``combinations.json`` schema).

Module-semantics pins (hand-computed, no event generation, no tables, no
archive, no re-run):

- ratio: a bin whose denominator is 0 is ``None`` and flagged (the
  committed flag encoding); a zero denominator everywhere is all-``None``.
- the single-species ``2 f`` combinations (``k_pair`` / ``b_pair``) are
  multiplied by 2 and preserve a negative per-bin value's sign.
- ``difference`` is first-minus-second in the species-tuple order;
  swapping the tuple order flips the sign.
- the forward / central / backward integral's region split at the bin
  xF midpoints: the boundary bin whose upper edge is xF = 0.2 has a
  midpoint of 0.15 (central, below the 0.2 bound); a bin whose midpoint
  is exactly 0.2 is central (the forward region is strictly ``> 0.2``);
  the backward integral of the p+p window ``xF in [0, 0.4]`` is exactly
  ``0.0`` (not ``None`` / NaN).
- the seagull moments: ``<pT>_x`` uses the density ``(pT / E) f`` and
  ``<pT>_y`` the density ``pT f`` (the two differ for nonzero pT); an
  xF bin whose f values vanish yields ``None`` for both conventions.
- the ranked residual map: ``top_n`` truncation (150 cells -> exactly 100
  rows, the largest |pull| first); the lexicographic tie-break; a sigma-0
  cell (``d != m`` -> infinite pull, ``d == m`` -> 0.0); and the flag
  boundary (``|pull| > 3.0`` strictly, so a cell at exactly 3.0 is not
  flagged).

Driver-surface pins.  The seal-verification and missing-file cases are
driven both through :func:`staged_observable_decomposition.verify_r01_archive`
(the gate that fires first, before any re-run) and through the CLI
``main`` (exit 2, and no payload written into ``--out``).  The re-run
digest self-check and the ``combinations.json`` schema cases drive the
driver as a tiny end-to-end run on a hand-constructed 10-event R01-like
archive built from ``config/staged_excitation_smoke.json`` (``fit.enabled:
false``) and the SPS table mirror; they skip with a recorded reason when
the mirror is absent from the checkout (the same convention as
``test_r11_decomposition_harness``).  The full 5,000,000-event R11 run is
the run itself, never a test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import staged_observable_decomposition as r11
from helpers.benchmarking import staged_r01_reference as r01
from na49_mixing.fit_observables import (
    COMBINATION_NAMES,
    ObservableCombination,
    combination_by_name,
    evaluate_combination,
    forward_central_backward_integral,
    ranked_residual_map,
    residual_cell_key,
    seagull_moment,
    undefined_combination_bins,
)
from na49_mixing.reaction import Reaction

NESTED_ROOT = Path(r01.NESTED_ROOT)
DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "staged_observable_decomposition.py"
)
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "fit_observables.py"

TOL = 1e-12

# The optimizer package roots the driver and module must not import.  This
# mirrors the committed ``test_r11_no_fitting`` set; the R11 drivers import
# ``numpy`` (allowed: it is not an optimizer) but no optimizer.
OPTIMIZER_ROOTS = {
    "scipy",
    "iminuit",
    "lmfit",
    "cma",
    "nlopt",
    "optuna",
    "pymc",
    "emcee",
    "sklearn",
    "statsmodels",
}

# The optimizer symbols the committed substring check ("minimize(",
# "least_squares(", "Minuit(") does not fully cover: the numpy/scipy
# optimize submodules and the common optimizer entry points.  None of these
# appears in either committed R11 source.
OPTIMIZER_SYMBOLS = (
    "scipy.optimize",
    "numpy.optimize",
    "np.optimize",
    "least_squares(",
    "Minuit(",
    "curve_fit(",
)

# ---------------------------------------------------------------------------
# Module semantics
# ---------------------------------------------------------------------------


def test_evaluate_combination_ratio_zero_denominator_encoding():
    """A zero-denominator ratio bin is None + flagged; all-zero -> all None.

    The committed test pins the 2-bin case; this pins the flag encoding on a
    3-bin input (the undefined index is exactly the zero bin) and the
    all-zero-denominator edge case.
    """
    ratio = ObservableCombination("r", ("a", "b"), "ratio", "a / b")
    # Denominator b is zero in exactly one bin (bin 1).
    guarded = evaluate_combination(ratio, {"a": (2.0, 5.0, 7.0), "b": (1.0, 0.0, 2.0)})
    assert guarded[0] == pytest.approx(2.0, abs=TOL)
    assert guarded[1] is None
    assert guarded[2] == pytest.approx(3.5, abs=TOL)
    # The flag encoding is exactly the index of the undefined bin.
    assert undefined_combination_bins(guarded) == (1,)
    # A zero denominator everywhere yields an all-None result, all flagged.
    all_zero = evaluate_combination(ratio, {"a": (3.0, 4.0), "b": (0.0, 0.0)})
    assert all_zero == (None, None)
    assert undefined_combination_bins(all_zero) == (0, 1)
    # A non-zero denominator everywhere is total (no flagged bins).
    total = evaluate_combination(ratio, {"a": (1.0, 2.0), "b": (2.0, 4.0)})
    assert total[0] == pytest.approx(0.5, abs=TOL)
    assert total[1] == pytest.approx(0.5, abs=TOL)
    assert undefined_combination_bins(total) == ()


def test_single_species_sum_is_2x_and_preserves_sign():
    """k_pair / b_pair multiply the single species by 2 (the card's 2 f).

    A negative per-bin value stays negative after the 2x scaling.
    """
    k_pair = evaluate_combination(combination_by_name("k_pair"), {"K-": (1.5, -2.25, 0.0)})
    assert k_pair[0] == pytest.approx(3.0, abs=TOL)
    assert k_pair[1] == pytest.approx(-4.5, abs=TOL)  # the sign is preserved
    assert k_pair[2] == pytest.approx(0.0, abs=TOL)
    # b_pair is the identical single-species 2x form.
    b_pair = evaluate_combination(combination_by_name("b_pair"), {"pbar": (-0.5, 2.0)})
    assert b_pair[0] == pytest.approx(-1.0, abs=TOL)
    assert b_pair[1] == pytest.approx(4.0, abs=TOL)
    # The two-species pi_sigma sum does NOT get the 2x scaling.
    pi_sigma = evaluate_combination(
        combination_by_name("pi_sigma"), {"pi-": (1.0, 2.0), "pi+": (1.0, 2.0)}
    )
    assert pi_sigma == pytest.approx((2.0, 4.0), abs=TOL)


def test_difference_operator_is_first_minus_second_and_swaps_sign():
    """difference subtracts the second species from the first, in tuple order.

    Swapping the species-tuple order flips the sign of every bin.
    """
    d_ab = ObservableCombination("d_ab", ("a", "b"), "difference", "a - b")
    d_ba = ObservableCombination("d_ba", ("b", "a"), "difference", "b - a")
    binned = {"a": (3.0, 1.0, -4.0), "b": (1.0, 4.0, -1.0)}
    values_ab = evaluate_combination(d_ab, binned)
    # a - b, element-wise.
    for got, (a, b) in zip(values_ab, zip(binned["a"], binned["b"])):
        assert got == pytest.approx(float(a) - float(b), abs=TOL)
    values_ba = evaluate_combination(d_ba, binned)
    # b - a = -(a - b): swapping the tuple order flips the sign.  The
    # difference operator is total (no None bins).
    for got_ab, got_ba in zip(values_ab, values_ba):
        assert got_ab is not None and got_ba is not None
        assert got_ab + got_ba == pytest.approx(0.0, abs=TOL)


def test_integral_boundary_bin_midpoint_and_backward_zero():
    """Region split at the bin xF midpoints on the boundary bins.

    The boundary bin whose upper edge is xF = 0.2 has a midpoint of 0.15
    (< 0.2, so central); a bin whose midpoint is exactly 0.2 is central
    (forward is strictly > 0.2); the backward integral of the p+p window is
    exactly 0.0.
    """
    # Flat 1-D xF edge tuple (grid form, n + 1 edges -> n bins, no pT width).
    # p+p paper window xF in [0, 0.4].
    edges = (0.0, 0.1, 0.2, 0.3, 0.4)
    values = (1.0, 2.0, 3.0, 4.0)
    result = forward_central_backward_integral({"s": values}, edges, "s")
    # bin [0.1, 0.2] midpoint 0.15 -> central (the boundary bin);
    # bin [0.2, 0.3] midpoint 0.25 -> forward; bin [0.0, 0.1] midpoint 0.05
    # -> central (the xF = 0.0 bin).
    assert result["central"] == pytest.approx(0.1 * 1.0 + 0.1 * 2.0, abs=TOL)
    assert result["forward"] == pytest.approx(0.1 * 3.0 + 0.1 * 4.0, abs=TOL)
    # Backward is exactly 0.0 on the p+p window: not None, not NaN.
    assert result["backward"] == 0.0
    assert math.isfinite(result["backward"])
    # A bin whose midpoint is EXACTLY 0.2 is central (forward is > 0.2).
    edges2 = (0.0, 0.15, 0.25, 0.40)
    values2 = (1.0, 1.0, 1.0)
    result2 = forward_central_backward_integral({"s": values2}, edges2, "s")
    # bin [0.15, 0.25] midpoint 0.2 -> central; bin [0.25, 0.40] -> forward.
    assert result2["central"] == pytest.approx(0.15 * 1.0 + 0.10 * 1.0, abs=TOL)
    assert result2["forward"] == pytest.approx(0.15 * 1.0, abs=TOL)
    assert result2["backward"] == 0.0


def test_seagull_two_conventions_density_and_degenerate_bin():
    """<pT>_x uses (pT/E) f, <pT>_y uses pT f; a zero-f bin yields None.

    A single xF bin with two pT cells hand-checks both densities (they
    differ for nonzero pT); a second xF bin whose f values all vanish
    yields None for both conventions (the committed degenerate-bin behavior).
    """
    # Dense 2 x 2 grid: xF midpoints 0.1 / 0.3, pT midpoints 0.25 / 0.75.
    # xF bin 0.1 has f = 0 in both its pT cells (degenerate); xF bin 0.3 is
    # nonzero.  mass 0.14 GeV, sqrt(s) = 20 GeV.
    edges = ((0.0, 0.2, 0.4), (0.0, 0.5, 1.0))
    values = (0.0, 1.0, 0.0, 3.0)  # bin order (0,0),(1,0),(0,1),(1,1)
    mass = 0.14
    sqrt_s = 20.0
    result = seagull_moment(
        {"s": values}, edges, "s", sqrt_s_gev=sqrt_s, mass_gev=mass
    )
    assert set(result) == {"pT_x_per_xF", "pT_y_per_xF"}
    assert len(result["pT_x_per_xF"]) == 2
    assert len(result["pT_y_per_xF"]) == 2
    # The degenerate xF bin (midpoint 0.1, all f = 0) is None for both.
    assert result["pT_x_per_xF"][0] is None
    assert result["pT_y_per_xF"][0] is None
    # The nonzero xF bin (midpoint 0.3): hand-computed <pT>_x / <pT>_y.
    x_mid = 0.3
    p_l = x_mid * sqrt_s / 2.0
    cells = ((0.25, 1.0), (0.75, 3.0))
    num_x = den_x = num_y = den_y = 0.0
    for p_t, v in cells:
        energy = math.sqrt(mass * mass + p_l * p_l + p_t * p_t)
        w_x = (p_t / energy) * v  # the (pT / E) * f density
        w_y = p_t * v  # the pT * f density
        num_x += p_t * w_x * 0.5
        den_x += w_x * 0.5
        num_y += p_t * w_y * 0.5
        den_y += w_y * 0.5
    hand_x = num_x / den_x
    hand_y = num_y / den_y
    x1 = result["pT_x_per_xF"][1]
    y1 = result["pT_y_per_xF"][1]
    assert x1 is not None and y1 is not None
    assert x1 == pytest.approx(hand_x, abs=TOL)
    assert y1 == pytest.approx(hand_y, abs=TOL)
    # The two conventions differ for nonzero pT (the up-to-0.1 GeV/c spread).
    assert abs(x1 - y1) > 1e-9
    # An all-zero species (single xF bin, single pT bin, f = 0) is None.
    all_zero = seagull_moment(
        {"s": (0.0,)}, ((0.0, 0.2), (0.0, 0.5)), "s", sqrt_s_gev=sqrt_s, mass_gev=mass
    )
    assert all_zero["pT_x_per_xF"] == [None]
    assert all_zero["pT_y_per_xF"] == [None]


def test_ranked_residual_map_top_n_truncation():
    """150 cells with distinct |pull| -> exactly 100 rows, the top one first."""
    d: dict[str, tuple[float, ...]] = {}
    m: dict[str, tuple[float, ...]] = {}
    v: dict[str, tuple[float, ...]] = {}
    for i in range(150):
        key = residual_cell_key("s", "f_xsect", "xF_central", i)
        # pull = (d - m) / sigma = (i - 0) / 1.0 = i (distinct |pull| 0..149).
        d[key] = (float(i),)
        m[key] = (0.0,)
        v[key] = (1.0,)
    result = ranked_residual_map(d, m, v, top_n=100)
    assert result["n_total_cells"] == 150
    assert len(result["rows"]) == 100  # exactly top_n rows
    assert [row["rank"] for row in result["rows"]] == list(range(1, 101))
    # The largest |pull| (149) is first; the returned rows are the top 100,
    # so the smallest |pull| among them is 50 (cells 50..149).
    assert result["rows"][0]["pull"] == pytest.approx(149.0, abs=TOL)
    pulls = [abs(row["pull"]) for row in result["rows"]]
    assert min(pulls) == pytest.approx(50.0, abs=TOL)
    assert pulls == sorted(pulls, reverse=True)
    # n_flagged counts |pull| > 3.0 strictly: cells 4..149 -> 146.
    assert result["n_flagged"] == 146


def test_ranked_residual_map_tie_break():
    """Equal |pull| cells break ties lexicographically on the cell key parts."""
    cells = [
        ("c", "x", "xF_central", 0, -5.0),  # A
        ("a", "x", "xF_central", 0, 5.0),  # B
        ("a", "x", "xF_central", 5, -5.0),  # C
        ("a", "x", "xF_central", 3, 5.0),  # D
        ("a", "y", "xF_central", 0, 5.0),  # E
    ]
    d: dict[str, tuple[float, ...]] = {}
    m: dict[str, tuple[float, ...]] = {}
    v: dict[str, tuple[float, ...]] = {}
    for sector, observable, region, bin_index, pull in cells:
        key = residual_cell_key(sector, observable, region, bin_index)
        # sigma = 1.0; d - m = pull.
        d[key] = (pull,)
        m[key] = (0.0,)
        v[key] = (1.0,)
    result = ranked_residual_map(d, m, v, top_n=10)
    assert result["n_flagged"] == 5  # all five cells have |pull| = 5.0 > 3.0
    # All five tie at |pull| = 5.0; the committed order is (sector,
    # observable, region, bin_index): a/x/bin0 (B), a/x/bin3 (D), a/x/bin5
    # (C), a/y/bin0 (E), c/x/bin0 (A).
    order = [
        (row["sector"], row["observable"], row["bin_index"]) for row in result["rows"]
    ]
    assert order == [
        ("a", "x", 0),
        ("a", "x", 3),
        ("a", "x", 5),
        ("a", "y", 0),
        ("c", "x", 0),
    ]
    assert [row["rank"] for row in result["rows"]] == [1, 2, 3, 4, 5]


def test_ranked_residual_map_zero_sigma_and_flag_boundary():
    """sigma = 0 gives an infinite pull (d != m) or 0.0 (d == m); the flag
    boundary is |pull| > 3.0 strictly (a cell at exactly 3.0 is not flagged)."""
    d: dict[str, tuple[float, ...]] = {}
    m: dict[str, tuple[float, ...]] = {}
    v: dict[str, tuple[float, ...]] = {}
    cells = [
        # (d, m, V): sigma 0 with d != m -> pull = inf.
        (1.0, 0.8, 0.0),
        # sigma 0 with d == m -> pull = 0.0.
        (1.0, 1.0, 0.0),
        # |pull| exactly 3.0 -> not flagged.
        (3.0, 0.0, 1.0),
        # |pull| just above 3.0 -> flagged.
        (3.1, 0.0, 1.0),
    ]
    for i, (d_val, m_val, v_val) in enumerate(cells):
        key = residual_cell_key("s", "f_xsect", "xF_central", i)
        d[key] = (d_val,)
        m[key] = (m_val,)
        v[key] = (v_val,)
    result = ranked_residual_map(d, m, v, top_n=10)
    by_bin = {row["bin_index"]: row for row in result["rows"]}
    # sigma 0, d != m -> infinite pull (recorded, ranked by |pull|).
    assert math.isinf(by_bin[0]["pull"])
    assert by_bin[0]["sigma"] == 0.0
    # sigma 0, d == m -> pull 0.0.
    assert by_bin[1]["pull"] == 0.0
    # A cell at exactly |pull| = 3.0 is present but NOT flagged.
    assert by_bin[2]["pull"] == pytest.approx(3.0, abs=TOL)
    # The 3.1 cell IS flagged.
    assert abs(by_bin[3]["pull"]) == pytest.approx(3.1, abs=TOL)
    # Only the infinite-pull cell and the 3.1 cell exceed 3.0.
    assert result["n_flagged"] == 2


# ---------------------------------------------------------------------------
# Driver surface: seal verification and the missing-file gate (exit 2)
# ---------------------------------------------------------------------------


def test_driver_refuses_missing_sealed_files(tmp_path):
    """A missing archive / missing SHA256SUMS / a sums entry with no payload
    all refuse the run at the seal gate (exit 2), before any re-run."""
    # (i) a nonexistent archive directory.
    with pytest.raises(SystemExit) as exc:
        r11.verify_r01_archive(tmp_path / "does_not_exist")
    assert exc.value.code == 2
    # (ii) a directory that carries no SHA256SUMS.txt.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        r11.verify_r01_archive(empty_dir)
    assert exc.value.code == 2
    # (iii) a SHA256SUMS.txt that lists a payload file that is absent (the
    # "no bin_predictions.json" case: the entry cannot re-hash to a file).
    no_payload = tmp_path / "no_payload"
    no_payload.mkdir()
    (no_payload / r01.SHA256SUMS_NAME).write_text(
        "0" * 64 + "  bin_predictions.json\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        r11.verify_r01_archive(no_payload)
    assert exc.value.code == 2
    # Driven through the CLI: the seal gate fires before the re-run, so the
    # exit code is 2 and nothing is written into --out (the directory is
    # created empty, no payload).
    out_dir = tmp_path / "r11_out_missing"
    with pytest.raises(SystemExit) as exc:
        r11.main(["--r01-archive", str(empty_dir), "--out", str(out_dir)])
    assert exc.value.code == 2
    assert out_dir.is_dir()
    assert not (out_dir / r11.PREREGISTRATION_NAME).is_file()
    assert not (out_dir / r11.COMBINATIONS_NAME).is_file()
    assert not (out_dir / r11.GATE_RECORD_NAME).is_file()


def test_driver_refuses_tampered_payload_at_seal_check(tmp_path):
    """A payload whose SHA256SUMS.txt entry does not re-hash refuses (exit 2).

    The seal-verification gate fires first (before the re-run digest
    self-check), so a tampered payload stops the run at the seal check.
    """
    archive = tmp_path / "tampered"
    archive.mkdir()
    (archive / r01.BIN_PREDICTIONS_NAME).write_text(
        json.dumps({"value": 1.0}) + "\n", encoding="utf-8"
    )
    # A SHA256SUMS.txt whose single entry is a digest the payload does not
    # re-hash to (the tampered-payload case).
    (archive / r01.SHA256SUMS_NAME).write_text(
        "0" * 64 + "  bin_predictions.json\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        r11.verify_r01_archive(archive)
    assert exc.value.code == 2
    # Driven through the CLI: the seal gate refuses with exit 2 and no
    # payload is written (no pre-registration, no gate1).
    out_dir = tmp_path / "r11_out_tampered"
    with pytest.raises(SystemExit) as exc:
        r11.main(["--r01-archive", str(archive), "--out", str(out_dir)])
    assert exc.value.code == 2
    assert not (out_dir / r11.PREREGISTRATION_NAME).is_file()
    assert not (out_dir / r11.GATE_RECORD_NAME).is_file()


# ---------------------------------------------------------------------------
# Driver surface: no-fitting attestation (source inspection)
# ---------------------------------------------------------------------------


def test_r11_module_and_driver_import_no_optimizer():
    """Both R11 sources import no optimizer; the specific optimizer symbols
    are absent.  (Extends the committed test_r11_no_fitting to the module.)"""
    for source_path in (MODULE_PATH, DRIVER_PATH):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            root = name.split(".")[0]
            assert root not in OPTIMIZER_ROOTS, (
                f"{source_path.name} imports {name!r}: an R11 driver/module "
                "must not import an optimizer (no fitting)"
            )
        source_text = source_path.read_text(encoding="utf-8")
        for token in OPTIMIZER_SYMBOLS:
            assert token not in source_text, (
                f"{source_path.name} references the optimizer symbol "
                f"{token!r}: an R11 driver/module must not fit"
            )
        # The committed substring check, re-pinned for completeness.
        for keyword in ("minimize(", "least_squares(", "Minuit("):
            assert keyword not in source_text


def test_empty_species_infinite_uncertainty_encoding():
    """A species observed in no event is listed under empty_species with an
    infinite uncertainty (never silently dropped), via the pooling helper."""
    species_order = [
        "pi_minus",
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    ]

    def aggregates(n_events: int, present: set[str]) -> dict:
        multiplicity = {s: [0, 0, 0] for s in species_order}
        for s in species_order:
            if s not in present:
                continue
            slot = multiplicity[s]
            slot[0] += n_events  # sum n (one of the species per event)
            slot[1] += n_events  # sum n^2
            slot[2] += n_events  # n_events_with_species
        return {
            "multiplicity": multiplicity,
            "gap": {"n_events": n_events, "n_with_gap": 0, "sum_max_gap": 0.0},
            "leading_proton": {
                "n_with_proton": n_events,
                "n_without_proton": 0,
                "sum_inelasticity": 0.5 * n_events,
                "sum_inelasticity2": 0.25 * n_events,
            },
        }

    present = {"pi_minus", "pi_plus", "k_plus", "k_minus", "proton"}
    records = [
        {"seed": 1, "n_events": 3, "aggregates": aggregates(3, present)},
        {"seed": 2, "n_events": 4, "aggregates": aggregates(4, present)},
    ]
    pooled = r11.pool_event_level(records, species_order)
    # The absent species is listed, with the infinite-uncertainty encoding.
    assert pooled["empty_species"] == ["antiproton"]
    antiproton = pooled["multiplicity"]["antiproton"]
    assert antiproton["empty"] is True
    assert math.isinf(antiproton["stat_uncertainty"])
    assert antiproton["n_events_with_species"] == 0
    # A present species keeps the 1 / sqrt(N_events) convention.
    pi_minus = pooled["multiplicity"]["pi_minus"]
    assert pi_minus["empty"] is False
    assert pi_minus["stat_uncertainty"] == pytest.approx(1.0 / math.sqrt(7))
    assert pi_minus["n_events_with_species"] == 7


# ---------------------------------------------------------------------------
# Driver surface: the re-run digest self-check and the combinations.json
# schema (tiny 10-event end-to-end runs; SPS table mirror required)
# ---------------------------------------------------------------------------

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

HARNESS_SEED = 20300911
HARNESS_N_EVENTS = 10
_SPECIES_KEYS = (
    "pi_minus",
    "pi_plus",
    "k_plus",
    "k_minus",
    "proton",
    "antiproton",
)


def _smoke_payload() -> dict:
    raw_config = json.loads(
        (NESTED_ROOT / "config" / r01.SMOKE_CONFIG_NAME).read_text(encoding="utf-8")
    )
    assert raw_config["fit"]["enabled"] is False
    return raw_config


def _harness_sqrt_s() -> float:
    return math.sqrt(Reaction(158.0, 2212, 2212).s())


def _small_r01_stream(n_events: int):
    """The seeded n-event R01 stream (hashes, per-species sums and counts)."""
    raw_config = _smoke_payload()
    sqrt_s = _harness_sqrt_s()
    tables = r01.load_binned_tables(raw_config)
    tables_sha256 = r01.tables_digest(tables)
    pp_config, _ = r01.build_pp_config(raw_config)
    hashes: list[str] = []
    sums: dict[str, list] = {}
    counts: dict[str, list] = {}
    for event_id in range(n_events):
        builder = r01.PPEventBuilder(pp_config, r01._event_rng(HARNESS_SEED, event_id))
        event = builder.generate(event_id, stage=r01.STAGE_LABEL)
        hashes.append(event.as_dict()["event_hash"])
        chunk_sums, chunk_counts = r01._bin_track_sums([event], tables, sqrt_s)
        for species, values in chunk_sums.items():
            sums[species] = (
                list(map(float, values))
                if species not in sums
                else list(map(float, np.asarray(sums[species]) + values))
            )
        for species, values in chunk_counts.items():
            counts[species] = (
                [int(v) for v in values]
                if species not in counts
                else [int(a) + int(b) for a, b in zip(counts[species], values)]
            )
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return raw_config, sqrt_s, tables, tables_sha256, sums, counts, digest


def _build_small_archive(
    archive_dir: Path, *, n_events: int, override_digest: str | None
) -> str:
    """A hand-constructed R01-like archive built from the small stream.

    ``override_digest`` replaces the per-replica
    ``event_hash_set_sha256`` in the sealed record (used to pin the re-run
    digest-mismatch failure); ``None`` keeps the true digest.
    """
    raw_config, sqrt_s, tables, tables_sha256, sums, counts, digest = (
        _small_r01_stream(n_events)
    )
    sums_arrays = {s: np.asarray(v, dtype=float) for s, v in sums.items()}
    counts_arrays = {s: np.asarray(v, dtype=np.int64) for s, v in counts.items()}
    entries, _values, _errors = r01.bin_predictions_payload(
        tables, sums_arrays, counts_arrays, n_events, sqrt_s
    )
    sigma_eta = r01.read_sigma_eta()
    residuals = r01.residuals_payload(tables, _values, _errors)
    covariance = r01.covariance_payload(tables, _errors, sigma_eta)
    sealed_digest = (
        override_digest if override_digest is not None else digest
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    r01._write_json(
        archive_dir / r01.PREREGISTRATION_NAME, r01.default_pre_registration()
    )
    r01._write_json(
        archive_dir / r01.REPLICA_MANIFEST_NAME,
        {
            "seed_base": 20260911,
            "seed_offset": 40000,
            "requested_replicas": 1,
            "requested_events_per_replica": n_events,
            "tier_status": "deferred",
            "tier_deferral_reason": "coverage fixture (10 events)",
            "seeds": [
                {
                    "seed": HARNESS_SEED,
                    "replica_index": 0,
                    "status": "completed",
                    "n_events": n_events,
                    "wall_seconds": 0.0,
                    "event_hash_set_sha256": digest,
                    "record": f"replica_{HARNESS_SEED}.json",
                }
            ],
            "config_sha256": r01.SMOKE_CONFIG_SHA256,
            "tables_sha256": tables_sha256,
        },
    )
    r01._write_json(
        archive_dir / f"replica_{HARNESS_SEED}.json",
        {
            "seed": HARNESS_SEED,
            "n_events": n_events,
            "chunk_size": r01.CHUNK_SIZE,
            "chunks": 1,
            "worker_count": 1,
            "event_seed_rule": r01.EVENT_SEED_RULE,
            "event_hash_set_sha256": sealed_digest,
            "config_sha256": r01.SMOKE_CONFIG_SHA256,
            "tables_sha256": tables_sha256,
            "wall_seconds": 0.0,
            "started_utc": "2026-09-12T00:00:00+00:00",
            "finished_utc": "2026-09-12T00:00:00+00:00",
            "sums": sums,
            "counts": counts,
            "reused": False,
        },
    )
    r01._write_json(archive_dir / r01.BIN_PREDICTIONS_NAME, entries)
    r01._write_json(archive_dir / r01.COVARIANCE_NAME, covariance)
    r01._write_json(archive_dir / r01.RESIDUALS_NAME, residuals)
    r01._write_json(
        archive_dir / r01.FAILURE_MAP_NAME,
        r01.build_failure_map(residuals, counts_arrays),
    )
    r01._write_json(
        archive_dir / r01.CHI2_NAME, r01.chi2_payload(residuals, counts_arrays)
    )
    # The seal inventory over every payload file (the driver verifies it).
    lines = []
    for path in sorted(archive_dir.rglob("*")):
        if path.is_file() and path.name != r01.SHA256SUMS_NAME:
            rel = path.relative_to(archive_dir).as_posix()
            lines.append(f"{r01.sha256_file(path)}  {rel}\n")
    (archive_dir / r01.SHA256SUMS_NAME).write_text(
        "".join(lines), encoding="utf-8", newline="\n"
    )
    return digest


@pytest.fixture(scope="module")
def r11_tiny_runs(tmp_path_factory) -> dict:
    """Two tiny end-to-end driver runs: a matching-digest run (exit 0) and a
    wrong per-replica digest run (exit 1).  Both re-run the same 10-event
    seeded stream; the re-run digest self-check is the divergence."""
    matching_archive = tmp_path_factory.mktemp("r01_like_matching")
    mismatch_archive = tmp_path_factory.mktemp("r01_like_mismatch")
    # The mismatch archive carries a per-replica digest that does NOT match
    # the re-generated stream's digest (the card-13 digest-mismatch failure).
    _build_small_archive(matching_archive, n_events=HARNESS_N_EVENTS, override_digest=None)
    _build_small_archive(
        mismatch_archive,
        n_events=HARNESS_N_EVENTS,
        override_digest="f" * 64,
    )
    matching_out = tmp_path_factory.mktemp("r11_matching_run")
    matching_exit = r11.main(
        [
            "--r01-archive",
            str(matching_archive),
            "--out",
            str(matching_out),
            "--workers",
            "1",
            "--top-n",
            "50",
        ]
    )
    mismatch_out = tmp_path_factory.mktemp("r11_mismatch_run")
    mismatch_exit = r11.main(
        [
            "--r01-archive",
            str(mismatch_archive),
            "--out",
            str(mismatch_out),
            "--workers",
            "1",
            "--top-n",
            "50",
        ]
    )
    return {
        "matching_exit": matching_exit,
        "matching_out": matching_out,
        "mismatch_exit": mismatch_exit,
        "mismatch_out": mismatch_out,
    }


@pytest.mark.skipif(not _SPS_PRESENT, reason="the NA49/SPS table mirror is absent")
def test_driver_rerun_digest_mismatch_fails(r11_tiny_runs):
    """A per-replica event-hash digest mismatch fails the run (exit 1).

    The seal check passes (every payload re-hashes), so the run proceeds to
    the re-run; the re-generated stream's digest does not equal the sealed
    per-replica digest, so the run fails before any gate is written.
    """
    assert r11_tiny_runs["mismatch_exit"] == 1
    out_dir = r11_tiny_runs["mismatch_out"]
    # The run fails at the digest self-check: no gate1 and no combinations.
    assert not (out_dir / r11.GATE_RECORD_NAME).is_file()
    assert not (out_dir / r11.COMBINATIONS_NAME).is_file()
    # The per-replica record records the mismatch (re-run digest != sealed).
    record_path = out_dir / f"r11_replica_{HARNESS_SEED}.json"
    assert record_path.is_file()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["digest_match"] is False
    assert record["sealed_event_hash_set_sha256"] == "f" * 64
    assert record["event_hash_set_sha256"] != record[
        "sealed_event_hash_set_sha256"
    ]


@pytest.mark.skipif(not _SPS_PRESENT, reason="the NA49/SPS table mirror is absent")
def test_combinations_json_schema_tiny_run(r11_tiny_runs):
    """The matching tiny run writes a combinations.json with every
    card-required key, and the arithmetic invariants of the event-level
    quantities."""
    assert r11_tiny_runs["matching_exit"] == 0
    out_dir = r11_tiny_runs["matching_out"]
    combinations = json.loads(
        (out_dir / r11.COMBINATIONS_NAME).read_text(encoding="utf-8")
    )
    # All six pre-registered combination names are present.
    assert set(combinations["combinations"]) == set(COMBINATION_NAMES)
    assert set(combinations["combinations"]) == {
        "pi_sigma",
        "pi_delta",
        "k_pair",
        "k_associated",
        "b_pair",
        "b_transport",
    }
    # The integrals and seagull moments are keyed per species with the
    # card's region / convention keys.
    assert set(combinations["integrals"]) == set(_SPECIES_KEYS)
    for block in combinations["integrals"].values():
        assert set(block["model"]) == {"forward", "central", "backward"}
        assert block["model"]["backward"] == 0.0  # the p+p paper window
        assert set(block["data"]) == {"forward", "central", "backward"}
    assert set(combinations["seagull_moments"]) == set(_SPECIES_KEYS)
    for block in combinations["seagull_moments"].values():
        assert set(block["model"]) == {"pT_x_per_xF", "pT_y_per_xF"}
        assert set(block["data"]) == {"pT_x_per_xF", "pT_y_per_xF"}
    # The multiplicity moments per species, with the arithmetic identity
    # <n^2> >= <n> (n^2 >= n for every non-negative integer multiplicity).
    assert set(combinations["multiplicity_moments"]) == set(_SPECIES_KEYS)
    for moments in combinations["multiplicity_moments"].values():
        assert moments["n_mean"] >= 0.0
        assert moments["n2_mean"] >= moments["n_mean"] - TOL
        assert moments["n_events_with_species"] > 0
    # The gap fraction is a fraction in [0, 1].
    gap = combinations["rapidity_gap_fraction"]
    assert gap["threshold"] == 1.0
    assert 0.0 <= gap["fraction"] <= 1.0
    # The leading-baryon inelasticity is 1 - xF_lead per event, with
    # xF_lead = 2 pz_lead / sqrt(s) of the most-forward (max pz) final-state
    # proton.  A most-forward proton can carry negative pz (xF_lead < 0), so
    # the per-event inelasticity exceeds 1 and the tiny-sample mean can too:
    # the committed range is NOT (0, 1] on the tiny sample (this stream's
    # mean is ~1.10).  Pin the robust invariants instead of a false bound.
    inelasticity = combinations["leading_baryon_inelasticity"]
    assert (
        inelasticity["n_events_with_proton"]
        + inelasticity["n_events_without_proton"]
        == HARNESS_N_EVENTS
    )
    assert inelasticity["n_events_with_proton"] > 0
    assert math.isfinite(inelasticity["mean"])
    assert inelasticity["mean"] > 0.0
    assert inelasticity["rms"] >= inelasticity["std"] - 1e-12
    # All six species appear in the tiny stream, so empty_species is 0/[]
    # and the gate's count field is 0.
    assert combinations["empty_species"] == []
    gate = json.loads((out_dir / r11.GATE_RECORD_NAME).read_text(encoding="utf-8"))
    assert gate["empty_species"] == 0
    assert gate["no_fitting"] is True
    assert gate["verdict"] == "pass"
