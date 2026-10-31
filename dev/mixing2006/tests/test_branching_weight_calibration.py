"""Round 3, card 1 branching-weight calibration: committed-behavior pins.

Task ``TASK-20260917-r3-gate2-followup-branching-weight-calibration``
(the Gate 2 closure obligation, follow-up 1 of the Round 2 closure
record; plan sections 24 and 8.2).  This module pins the committed
driver ``helpers/benchmarking/calibrate_branching_weights.py`` (nested
and outer trees byte-identical) and the calibration run evidence under
``dev/00runs/r3_branching_weight_calibration_2026-09-17/``.

The driver's contract: recompute the per-state, per-channel
open-weight expectations from the committed catalog (the card-4
Blatt-Weisskopf barrier with all L >= 0, the card-3 width law, the
card-5/6 class convention, on the R20 100-point grid; the committed
10,001-node reference quadrature normalizes the line shape and cancels
in the normalized open weight), compare them against the sealed R20
sampled channel shares at the pre-registered 3.0-sigma per-cell
tolerance, and diagnose the residual pattern across the four
pre-named convention dimensions plus the multiplicity of the per-cell
max criterion.  Read-only over the sealed R20 archive; writes only
into its own run dir; no random draws (master seed 20260917, draws 0).

Pins:

- ``test_dual_tree_byte_identical``: the nested and outer driver
  copies are byte-identical.
- ``test_expectation_recomputation_reproducible_named_state``: the
  expectation recomputation on the named state (pdg 122112,
  N(1650)0) is reproducible: two independent recomputations (fresh
  catalog loads) agree byte-for-byte and to 1e-12 on every recorded
  expected share, and reproduce the on-disk run expectation file
  byte-for-byte when the run dir is present.
- ``test_comparison_byte_stable_on_fixture``: the per-state sigma
  entry computed on a synthetic fixture of sampled shares (fabricated
  per-point channel counts against the committed catalog record's
  open-weight structure) is byte-stable across calls, and the sigma
  of a hand-perturbed cell matches the closed-form binomial value.
- ``test_no_catalog_change``: the committed catalog bytes are
  unchanged by the driver's recompute step (and equal the sealed R20
  provenance catalog digest), and the sealed R20 archive passes its
  own seal verification read-only.
- ``test_cli_modes_and_flags``: the CLI exposes exactly the staged
  modes ``--pre-registration / --recompute / --compare / --report /
  --seal / --clean-validate`` (a required mutually exclusive group)
  plus ``--state`` (only with ``--recompute``), ``--tag`` (only
  ``before``/``after``, with ``--compare``), ``--run-dir``, and
  ``--r20-archive``; a bare invocation is rejected.
- Sealed-run artifact pins (skipped with a clear message when the
  gitignored run dir is absent): the before- and after-tables are
  complete for all 80 R20 states, reproduce the sealed R20 gate-2
  per-state maxima exactly, record the failing/passing split (39 and
  41), agree with each other on every per-state entry (no fix was
  identified, so the tables differ only in their tag), and the
  disposition names every residual failing state with its diagnosis
  and ``fix_applied = false``.

Hermetic: reads the committed catalog, the sealed R20 archive, and
the sealed run dir (all gitignored artifacts skip cleanly); no event
generation, no external tables.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking"
    / "calibrate_branching_weights.py"
)
MIRROR_PATH = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "calibrate_branching_weights.py"
)
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
RUN_NAME = "r3_branching_weight_calibration_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
R20_RUN_DIR = (
    MONO_ROOT / "dev" / "00runs"
    / "r20_single_resonance_unit_grid_2026-09-13"
)

RUN_ID = "R3-BRANCHING-WEIGHT-CALIBRATION"
CARD_ID = "TASK-20260917-r3-gate2-followup-branching-weight-calibration"
MASTER_SEED = 20260917
DRAWS = 0
N_SIGMA = 3.0
N_STATES = 80
N_FAILING = 39
N_PASSING = 41

#: The named state of the reproducibility pin: N(1650)0, one of the
#: failing multi-channel states (7 channels, worst R20 residual state
#: family).
NAMED_STATE_PDG = 122112

#: The sealed R20 provenance catalog digest (the committed catalog must
#: equal it: no catalog change anywhere in the card).
SEALED_R20_CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3"
)


@lru_cache(maxsize=1)
def _driver():
    """The committed driver module, loaded from its nested-tree path."""
    spec = importlib.util.spec_from_file_location(
        "calibrate_branching_weights_pins", str(DRIVER_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Driver contract (no sealed artifact needed)
# ---------------------------------------------------------------------------
def test_dual_tree_byte_identical():
    """The nested and outer driver copies are byte-identical."""
    assert _file_sha256(DRIVER_PATH) == _file_sha256(MIRROR_PATH)


def test_cli_modes_and_flags():
    """The CLI exposes exactly the staged modes (a required mutually
    exclusive group) plus --state (only with --recompute), --tag
    (before/after), --run-dir, and --r20-archive; a bare invocation is
    rejected and a non-before/after --tag is rejected by argparse."""
    driver = _driver()
    parser = driver._build_parser()
    for mode in (
        ["--pre-registration"],
        ["--recompute"],
        ["--recompute", "--state", "122112"],
        ["--compare"],
        ["--compare", "--tag", "after"],
        ["--report"],
        ["--seal"],
        ["--clean-validate"],
    ):
        args = parser.parse_args(mode)
        flags = [
            value for name, value in vars(args).items()
            if name in (
                "pre_registration", "recompute", "compare", "report",
                "seal", "clean_validate",
            )
        ]
        assert sum(bool(flag) for flag in flags) == 1, (
            f"{mode}: the mutually exclusive group must select exactly "
            "one mode"
        )
    args = parser.parse_args(["--compare", "--tag", "after"])
    assert args.compare is True and args.tag == "after"
    assert parser.parse_args(["--compare"]).tag == "before"
    assert parser.parse_args(["--recompute"]).tag == "before"
    # --state is only valid together with --recompute
    with pytest.raises(SystemExit):
        driver.main(["--state", "122112"])
    # a bare invocation (no staged mode) is rejected
    with pytest.raises(SystemExit):
        driver.main([])
    # a tag outside {before, after} is rejected by argparse
    with pytest.raises(SystemExit):
        parser.parse_args(["--compare", "--tag", "mid"])
    # the run contract constants
    assert driver.RUN_ID == RUN_ID
    assert driver.CARD_ID == CARD_ID
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.DRAWS == DRAWS
    assert driver.N_SIGMA == N_SIGMA
    assert driver.RUN_DIR_REL == f"dev/00runs/{RUN_NAME}"
    assert driver.TABLE_TAGS == ("before", "after")
    assert driver.DIAGNOSIS_DIMENSIONS == (
        "barrier evaluation point",
        "quadrature grid",
        "channel summation order",
        "degenerate-weight handling",
    )


def _named_state_row(driver):
    catalog = driver.R20._load_catalog()
    rows = driver.R20._state_rows(catalog)
    row = next(
        r for r in rows
        if int(r["record"].pdg_id) == NAMED_STATE_PDG
    )
    return catalog, rows, row


def test_expectation_recomputation_reproducible_named_state(tmp_path):
    """The expectation recomputation on the named state (pdg 122112,
    N(1650)0) is reproducible: two independent recomputations with
    fresh catalog loads agree byte-for-byte and to 1e-12 on every
    recorded expected share, and reproduce the on-disk run expectation
    file byte-for-byte when the run dir is present."""
    driver = _driver()
    r20_pre = _load_json(R20_RUN_DIR / "pre_registration.json")
    pre_by_pdg = {
        int(entry["pdg_id"]): entry
        for entry in r20_pre["state_coverage"]["states"]
    }
    pre_entry = pre_by_pdg[NAMED_STATE_PDG]

    catalog_a, _rows_a, row_a = _named_state_row(driver)
    payload_a = driver._recompute_state_expectation(
        catalog_a, row_a, pre_entry
    )
    catalog_b = driver.R20._load_catalog()  # fresh load
    _rows_b, row_b = driver.R20._state_rows(catalog_b), None
    row_b = next(
        r for r in _rows_b
        if int(r["record"].pdg_id) == NAMED_STATE_PDG
    )
    payload_b = driver._recompute_state_expectation(
        catalog_b, row_b, pre_entry
    )

    bytes_a = _canonical_json_bytes(payload_a)
    bytes_b = _canonical_json_bytes(payload_b)
    assert bytes_a == bytes_b, (
        "the expectation recomputation is not byte-reproducible on the "
        "named state"
    )

    # the explicit 1e-12 numeric agreement on every recorded share and
    # weight, open point by open point
    for j, point in payload_a["points"].items():
        if point["degenerate"]:
            continue
        other = payload_b["points"][j]
        for key, share in point["expected_shares"].items():
            assert abs(share - other["expected_shares"][key]) <= 1e-12
        for key, weight in point["channel_weights"].items():
            assert abs(weight - other["channel_weights"][key]) <= 1e-12
        total = sum(point["channel_weights"].values())
        assert total > 0.0
        # the shares are the normalized open weights to 1e-12
        for key, weight in point["channel_weights"].items():
            assert abs(point["expected_shares"][key]
                       - weight / total) <= 1e-12

    # the payload reproduces the sealed R20 pre-registration contract
    assert payload_a["state"]["pdg_id"] == NAMED_STATE_PDG
    assert payload_a["state"]["channels"] == pre_entry["channels"]
    assert payload_a["grid"]["points"] == 100
    assert payload_a["pole_point"] is not None
    degenerate = payload_a["degenerate_points"]
    assert [int(e["j"]) for e in degenerate] == list(
        pre_entry["expected_skip_points"]
    )

    # the on-disk run expectation file is byte-identical when present
    run_file = RUN_DIR / "expectations" / f"state_{NAMED_STATE_PDG}.json"
    if run_file.is_file():
        assert run_file.read_bytes() == bytes_a, (
            "the sealed run expectation file does not reproduce the "
            "committed recomputation on the named state"
        )


def test_comparison_byte_stable_on_fixture():
    """The per-state sigma entry on a synthetic fixture of sampled
    shares is byte-stable across calls, and the sigma of a
    hand-perturbed cell matches the closed-form binomial value."""
    driver = _driver()
    catalog, _rows, row = _named_state_row(driver)
    ctx = driver.R20._StateContext(catalog, row["record"])
    record = row["record"]
    pdg = int(record.pdg_id)
    open_js = [j for j in range(driver.R20.GRID_POINTS)
               if j not in ctx.degenerate]
    n = row["decays_per_point"]

    # Build the fixture expectations from the committed convention, then
    # fabricate per-point channel counts that sit exactly on the
    # expected shares except for one hand-perturbed cell of the first
    # open point (the perturbation target is the channel with the
    # largest expected share at that point).
    weights_by_point = {}
    shares_by_point = {}
    classes_by_point = {}
    for j in open_js:
        weights, class_shares = driver.R20._expected_point(
            ctx, ctx.grid[j]
        )
        total = math.fsum(weights.values())
        weights_by_point[j] = weights
        shares_by_point[j] = {
            key: w / total for key, w in weights.items()
        }
        classes_by_point[j] = class_shares
    first = open_js[0]
    target_key = max(
        shares_by_point[first], key=lambda k: shares_by_point[first][k]
    )
    target_share = shares_by_point[first][target_key]
    assert 0.0 < target_share < 1.0
    perturbation = 150  # decays moved into the target cell
    counts_by_point = {}
    for j in open_js:
        counts = {
            key: int(round(n * share))
            for key, share in shares_by_point[j].items()
        }
        if j == first:
            counts[target_key] += perturbation
        counts_by_point[j] = counts

    expectation = {
        "state": {"pdg_id": pdg, "name": str(record.name)},
        "grid": {"masses_gev": list(ctx.grid)},
        "pole_point": driver.R20._pole_point(ctx),
        "degenerate_points": [
            {"j": int(j)} for j in sorted(ctx.degenerate)
        ],
        "points": {
            str(j): {
                "j": j,
                "mass_gev": ctx.grid[j],
                "degenerate": False,
                "expected_shares": shares_by_point[j],
                "class_shares": classes_by_point[j],
            }
            for j in open_js
        },
    }
    # the cos(theta) histograms sit exactly on the uniform 10-bin
    # expectation (n_two divisible by 10: the two-body vertex count is
    # the total count here, fabricated divisible)
    state_payload = {
        "degenerate_points": [
            {"j": int(j)} for j in sorted(ctx.degenerate)
        ],
        "points": {
            str(j): {
                "channel_counts": {
                    key: counts_by_point[j][key]
                    for key in counts_by_point[j]
                },
                "cos_theta_hist": [0] * driver.R20.ANGULAR_BINS,
                "n_two_body_vertices": 0,
            }
            for j in open_js
        },
    }
    # the fixture's own recomputed maxima (the gate echo must match the
    # fixture: the rounded counts leave tiny non-zero cells)
    fixture_branch = 0.0
    fixture_charge = 0.0
    for j in open_js:
        for key, share in shares_by_point[j].items():
            frac = counts_by_point[j].get(key, 0) / n
            fixture_branch = max(
                fixture_branch, driver.R20._zscore(frac, share, n)
            )
        for _cls, shares in classes_by_point[j].items():
            n_cls = sum(counts_by_point[j].get(k, 0) for k in shares)
            if n_cls <= 0:
                continue
            for key, (_pair_key, share) in shares.items():
                z = driver.R20._zscore(
                    counts_by_point[j].get(key, 0) / n_cls, share, n_cls
                )
                fixture_charge = max(fixture_charge, z)

    def _entry_once():
        return driver._state_sigma_entry(
            catalog, row, expectation, state_payload,
            {
                "branching_max_abs_sigma": fixture_branch,
                "charge_max_abs_sigma": fixture_charge,
                "angular_max_abs_sigma": 0.0,
            },
        )

    entry_a = _entry_once()
    entry_b = _entry_once()
    assert _canonical_json_bytes(entry_a) == _canonical_json_bytes(entry_b)

    # the closed-form binomial sigma of the perturbed cell: the cell
    # count is round(n p) + perturbation, so the residual against the
    # expected share is (k - n p) at n trials
    worst = entry_a["diagnostics"]["worst_cell"]
    assert worst["point_j"] == first
    assert worst["channel"] == target_key
    k_target = counts_by_point[first][target_key]
    expected_sigma = (k_target - n * target_share) / math.sqrt(
        n * target_share * (1.0 - target_share)
    )
    assert worst["abs_sigma"] == pytest.approx(expected_sigma, rel=1e-12)
    assert expected_sigma > 3.0
    # the charge and angular maxima come from the count rounding of the
    # fixture (the gate echo matches them by construction)
    assert entry_a["gate2_pass"] is False
    assert entry_a["matches_sealed_gate2"] == {
        "branching_max_abs_sigma": True,
        "charge_max_abs_sigma": True,
        "angular_max_abs_sigma": True,
    }


def test_no_catalog_change(tmp_path):
    """The committed catalog bytes are unchanged by the driver's
    recompute step (and equal the sealed R20 provenance digest), and
    the sealed R20 archive passes its own seal verification
    read-only."""
    driver = _driver()
    sha_before = _file_sha256(CATALOG_PATH)
    assert sha_before == SEALED_R20_CATALOG_SHA256
    # a read-only seal verification of the sealed R20 archive
    sums_sha = driver._verify_r20_seal(R20_RUN_DIR)
    assert sums_sha == _file_sha256(R20_RUN_DIR / "SHA256SUMS.txt")
    # the recompute step for one state into a scratch run dir
    pre_path = tmp_path / "pre_registration.json"
    payload = driver._build_pre_registration(R20_RUN_DIR)
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path.write_bytes(_canonical_json_bytes(payload))
    driver._run_recompute(tmp_path, NAMED_STATE_PDG)
    assert (tmp_path / "expectations"
            / f"state_{NAMED_STATE_PDG}.json").is_file()
    assert _file_sha256(CATALOG_PATH) == sha_before, (
        "the recompute step changed the committed catalog"
    )
    # the recomputed expectation is immutable once written
    with pytest.raises(RuntimeError, match="immutable once written"):
        driver._run_recompute(tmp_path, NAMED_STATE_PDG)


# ---------------------------------------------------------------------------
# Sealed-run artifact pins (skip cleanly on a fresh checkout)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run_dir():
    required = [
        "pre_registration.json",
        "provenance.json",
        "sigma_table_before.json",
        "sigma_table_after.json",
        "calibration_report.json",
        "disposition.json",
        "seal_manifest.json",
        "SHA256SUMS.txt",
    ]
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the run dir {RUN_DIR} is absent or incomplete (gitignored "
            "artifact); restore the sealed run to execute the artifact "
            "pins"
        )
    missing = [name for name in required if not (RUN_DIR / name).is_file()]
    if missing:
        pytest.skip(
            f"the run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing) + ")"
        )
    return RUN_DIR


def test_run_pre_registration_contract(run_dir):
    """The run pre-registration carries the committed contract: the
    deterministic T0-equivalent statistics (master seed 20260917,
    draws 0), the 80 sealed R20 states, the 3.0-sigma R20 per-cell
    comparison convention, and the card's scope clauses."""
    pre = _load_json(run_dir / "pre_registration.json")
    assert pre["schema"].startswith(
        "na49_mixing.calibrate_branching_weights."
    )
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_computation"] is True
    assert pre["statistics"]["master_seed"] == MASTER_SEED
    assert pre["statistics"]["draws"] == DRAWS
    assert pre["statistics"]["tier"] == (
        "T0-equivalent deterministic diagnostic"
    )
    assert pre["state_coverage"]["n_states"] == N_STATES
    assert len(pre["state_coverage"]["pdg_ids"]) == N_STATES
    assert pre["comparison"]["n_sigma"] == N_SIGMA
    assert set(pre["scope"]) == {
        "recompute", "compare", "diagnose", "fix_if_identified",
        "re_verify", "report",
    }
    assert pre["inputs"]["catalog"]["sha256"] == _file_sha256(
        CATALOG_PATH
    )
    prov = _load_json(run_dir / "provenance.json")
    assert prov["catalog"]["sha256"] == SEALED_R20_CATALOG_SHA256
    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS


def _sigma_table(run_dir, tag):
    return _load_json(run_dir / f"sigma_table_{tag}.json")


def test_sigma_tables_reproduce_sealed_gate2(run_dir):
    """Both sigma tables are complete for all 80 R20 states, reproduce
    the sealed R20 gate-2 per-state maxima exactly (the expectation
    mirror shares the committed convention that produced them), and
    record the 39/41 failing/passing split with the worst residual
    4.913 sigma on the Nbar(1650)0 (pdg -122112)."""
    gate2 = _load_json(R20_RUN_DIR / "gate2_r20.json")
    gate_per_state = gate2["clauses"]["branching_test"]["per_state"]
    for tag in ("before", "after"):
        table = _sigma_table(run_dir, tag)
        assert table["run"] == RUN_ID
        assert table["card"] == CARD_ID
        assert table["tag"] == tag
        assert table["n_sigma"] == N_SIGMA
        assert len(table["per_state"]) == N_STATES
        totals = table["totals"]
        assert totals["n_states_fail"] == N_FAILING
        assert totals["n_states_pass"] == N_PASSING
        assert totals["worst_state_pdg"] == -122112
        assert totals["worst_branching_abs_sigma"] == pytest.approx(
            4.913494961192409, rel=0.0, abs=1e-12
        )
        for pdg, entry in table["per_state"].items():
            gate_entry = gate_per_state[pdg]
            assert entry["branching_max_abs_sigma"] == (
                gate_entry["branching_max_abs_sigma"]
            ), f"pdg {pdg}: branching sigma drifted from the sealed gate"
            assert entry["charge_max_abs_sigma"] == (
                gate_entry["charge_max_abs_sigma"]
            ), f"pdg {pdg}: charge sigma drifted from the sealed gate"
            assert entry["angular_max_abs_sigma"] == (
                gate_entry["angular_max_abs_sigma"]
            ), f"pdg {pdg}: angular sigma drifted from the sealed gate"
            assert entry["gate2_pass"] == gate_entry["pass"]
            assert all(entry["matches_sealed_gate2"].values()), (
                f"pdg {pdg}: the recomputed maxima do not reproduce the "
                "sealed R20 gate table"
            )


def test_tables_identical_no_fix_and_disposition(run_dir):
    """With no convention fix identified (scope clause d does not
    trigger), the before- and after-tables agree on every per-state
    entry and differ only in their tag, and the disposition names all
    39 residual failing states with a diagnosis and fix_applied =
    false."""
    before = _sigma_table(run_dir, "before")
    after = _sigma_table(run_dir, "after")
    assert before["per_state"].keys() == after["per_state"].keys()
    for pdg, entry in before["per_state"].items():
        other = after["per_state"][pdg]
        for field in (
            "branching_pole_max_abs_sigma",
            "branching_open_max_abs_sigma",
            "charge_max_abs_sigma",
            "angular_max_abs_sigma",
            "gate2_pass",
        ):
            assert entry[field] == other[field], (
                f"pdg {pdg}: the {field} differs between the before "
                "and after tables although no convention fix was "
                "applied"
            )
    assert before["totals"] == after["totals"]

    disposition = _load_json(run_dir / "disposition.json")
    assert disposition["fix_applied"] is False
    assert disposition["n_failing_before"] == N_FAILING
    assert disposition["n_failing_after"] == N_FAILING
    assert disposition["repaired_states"] == []
    assert len(disposition["residual_states"]) == N_FAILING
    failing = {
        pdg for pdg, entry in before["per_state"].items()
        if not entry["gate2_pass"]
    }
    assert set(disposition["residual_states"]) == failing
    for pdg, entry in disposition["per_state"].items():
        assert entry["fix_applied"] is False
        assert entry["disposition"] == "residual recorded as a finding"
        assert "no open-weight convention defect" in entry["diagnosis"]
        # the per-state worst cell is recorded with its expected count
        assert entry["worst_cell"]["abs_sigma"] == (
            before["per_state"][pdg]["branching_max_abs_sigma"]
        )

    report = _load_json(run_dir / "calibration_report.json")
    assert report["n_failing_before"] == N_FAILING
    assert report["n_failing_after"] == N_FAILING
    assert report["constraint_record"]["no_branching_fit"] is True
    assert report["constraint_record"]["catalog_byte_unchanged"] is True
    assert report["constraint_record"]["r20_archive_read_only"] is True
    assert set(report["diagnosis_dimensions"]) == {
        "barrier evaluation point",
        "quadrature grid",
        "channel summation order",
        "degenerate-weight handling",
    }


def test_expectation_files_match_recomputation(run_dir):
    """Every on-disk expectation file is byte-identical to the
    committed recomputation from the catalog (the expectations are a
    pure function of the committed card-3/4/5/6 convention)."""
    driver = _driver()
    r20_pre = _load_json(R20_RUN_DIR / "pre_registration.json")
    pre_by_pdg = {
        int(entry["pdg_id"]): entry
        for entry in r20_pre["state_coverage"]["states"]
    }
    catalog = driver.R20._load_catalog()
    rows = driver.R20._state_rows(catalog)
    checked = 0
    for row in rows:
        pdg = int(row["record"].pdg_id)
        payload = driver._recompute_state_expectation(
            catalog, row, pre_by_pdg[pdg]
        )
        on_disk = run_dir / "expectations" / f"state_{pdg}.json"
        assert on_disk.read_bytes() == _canonical_json_bytes(payload), (
            f"pdg {pdg}: the on-disk expectation does not reproduce the "
            "committed recomputation"
        )
        checked += 1
    assert checked == N_STATES


def test_seal_manifest_and_sha256_sums(run_dir):
    """seal_manifest.json and SHA256SUMS.txt follow the Round 1
    sealing discipline: the sums file is in sha256sum -c format,
    verifies against every sealed file on disk, and excludes itself
    and the seal manifest."""
    manifest = _load_json(run_dir / "seal_manifest.json")
    assert manifest["schema"].startswith(
        "na49_mixing.calibrate_branching_weights."
    )
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt",
    ]
    sums_path = run_dir / "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == _file_sha256(sums_path)
    text = sums_path.read_text(encoding="utf-8")
    assert text.endswith("\n") and "\r" not in text
    entries = [line.split("  ", 1) for line in text.splitlines()]
    assert len(entries) == manifest["file_count"]
    expected_rels = (
        {f"expectations/state_{pdg}.json" for pdg in
         _load_json(run_dir / "pre_registration.json")
         ["state_coverage"]["pdg_ids"]}
        | {"pre_registration.json", "provenance.json",
           "sigma_table_before.json", "sigma_table_after.json",
           "calibration_report.json", "disposition.json"}
    )
    assert {rel for _digest, rel in entries} == expected_rels
    for digest, rel in entries:
        assert _file_sha256(run_dir / rel) == digest, (
            f"the sealed digest of {rel} does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    assert manifest["total_bytes"] == sum(
        (run_dir / rel).stat().st_size for _digest, rel in entries
    )
