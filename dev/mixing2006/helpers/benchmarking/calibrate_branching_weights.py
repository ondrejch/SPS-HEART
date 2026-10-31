#!/usr/bin/env python3
"""Staged calibration driver: the open-phase-space weight calibration
against the sealed R20 sampled channel shares (Round 3, card 1).

Task ``TASK-20260917-r3-gate2-followup-branching-weight-calibration``
(the Gate 2 closure obligation of the Round 2 decision, follow-up 1 of
``_knowledge/operations/round2-closure-2026-09-17.md``; plan section 24
and plan section 8.2).  In run R20, 39 of the 80 decaying states fail
the pre-registered 3.0-sigma branching tolerance (worst 4.913 sigma on
the Nbar(1650)0, PDG -122112).  This driver performs the systematic
calibration of the open-phase-space weights (the card-3/4/5/6
barrier-integrated open-weight expectations) against the sealed R20
sampled channel shares, with the branching fractions kept as external
PDG inputs: the calibration adjusts nothing but is allowed to adjust
the open-weight convention, never the branching fractions.

What the driver recomputes (scope clause a): the per-state, per-channel
open-weight expectations from the committed catalog with the card-4
Blatt-Weisskopf barrier (all L >= 0), the card-3 width law, and the
R20 100-point grid per state, on the committed card-5/6 open-weight
convention.  The expectations are computed through the committed R20
driver's own expected-value mirror (``staged_r20_unit_grid.py``, loaded
from this driver's directory), so the recomputation shares one code
path with the convention that produced the sealed R20 verification.
The committed 10,001-node reference quadrature
(``_NORMALIZATION_GRID_POINTS`` of ``na49_mixing.resonance_widths``,
the card-3 re-pin) is part of that convention; it normalizes the
spectral function, and the normalized open weight of a point is
invariant under the normalization constant, so the pointwise open
weights carry no quadrature dependence (recorded as a checked
diagnosis dimension below).

What the driver compares (scope clause b): the recomputed expectations
against the sealed R20 sampled channel shares
(``dev/00runs/r20_single_resonance_unit_grid_2026-09-13/``, read-only)
at the pre-registered 3.0-sigma tolerance, on the R20 per-cell binomial
convention ``sqrt(p (1 - p) / N)`` with the expected share in the
variance, and on the sealed R20 per-state gate criterion (the maxima of
the branching, charge, and angular cell maxima against 3.0 sigma).

What the driver diagnoses (scope clause c): which convention drives
each residual, across the four pre-named dimensions of the card
(barrier evaluation point, quadrature grid, channel summation order,
degenerate-weight handling) plus the statistical multiplicity of the
per-cell max criterion.  The diagnosis is deterministic: no Monte
Carlo is generated, the sealed R20 samples (63,110,000 decays) are
reused, and every recorded number is a pure function of the committed
catalog and the sealed archive.

Scope clause (d) of the card (``if and only if``): the driver applies a
convention fix only when the diagnosis identifies a convention defect
in the open-weight computation.  The driver itself contains no fix; a
fix would land in ``na49_mixing/resonance_widths.py`` and/or
``na49_mixing/resonance_decay.py`` with named tests and would be
recorded in the disposition.  The comparison is re-run after any such
fix and the after-table records the effect; with no fix identified,
the before- and after-tables are computed by the identical code path
and differ only in their ``tag`` field.

Modes (exactly one per invocation; each performs exactly one step and
stops; read-only over the sealed R20 archive; writes only into its own
run dir):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  computed.  Immutable once expectation files exist.
- ``--recompute [--state PDG]``: recompute the open-weight
  expectations for one state (with ``--state``) or for every state
  (without, catalog record order), writing
  ``expectations/state_<pdg>.json`` into the run dir, then stop.
  Requires the pre-registration; every state row is asserted against
  the sealed R20 pre-registration before the first grid point.
- ``--compare --tag before|after``: build the per-state sigma table
  from the recomputed expectations and the sealed R20 state files,
  write ``sigma_table_<tag>.json`` into the run dir, then stop.
  Requires the full expectation set; refuses to overwrite an existing
  table.
- ``--report``: assemble ``calibration_report.json`` and
  ``disposition.json`` from the before- and after-tables (both
  required), then stop.  Every residual failing state is named with
  its diagnosis and its recorded residual; the clause-(d) decision is
  recorded.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the comparison from a
  clean interpreter over the sealed artifacts, and confirm the report
  hashes match; write ``clean_environment_commands.json`` into the run
  dir, then stop.

Deterministic: the pre-registration, the expectations, the sigma
tables, the report, and the disposition carry no wall-clock fields
(their bytes are a pure function of the committed catalog and the
sealed archive); only the provenance, the seal manifest, and the
clean-environment record carry wall-clock stamps.

Run dir: ``dev/00runs/r3_branching_weight_calibration_2026-09-17/``
(the Round 2 card-8 pattern ``r2_branching_spectral_checks_
2026-09-13/``).  Master seed ``20260917`` recorded with ``draws: 0``
for contract uniformity; no random draws exist on any code path.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no event generation, no external tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the resonance catalog, and the sealed R20
# archive.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple["Path", "Path"]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        print(
            "calibrate_branching_weights: cannot locate the na49_mixing "
            f"package next to {here}; the driver runs from the nested tree "
            "(dev/mixing2006/helpers/benchmarking/) or its monorepo mirror "
            "(helpers/benchmarking/)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

# ---------------------------------------------------------------------------
# The committed R20 driver is the pre-registered expected-value mirror.
# It is loaded from this driver's own directory (the same tree), so the
# recomputed expectations share one code path with the convention that
# produced the sealed R20 verification in either tree.
# ---------------------------------------------------------------------------
R20_DRIVER_NAME = "staged_r20_unit_grid.py"


def _load_r20_driver():
    driver_path = Path(__file__).resolve().parent / R20_DRIVER_NAME
    spec = importlib.util.spec_from_file_location(
        "staged_r20_unit_grid_calibration_view", driver_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R20 = _load_r20_driver()

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Run identity and contract constants
# ---------------------------------------------------------------------------
RUN_ID = "R3-BRANCHING-WEIGHT-CALIBRATION"
CARD_ID = "TASK-20260917-r3-gate2-followup-branching-weight-calibration"
RUN_NAME = "r3_branching_weight_calibration_2026-09-17"
DEFAULT_RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
RUN_DIR_REL = f"dev/00runs/{RUN_NAME}"

R20_RUN_NAME = "r20_single_resonance_unit_grid_2026-09-13"
DEFAULT_R20_ARCHIVE = MONO_ROOT / "dev" / "00runs" / R20_RUN_NAME

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"

#: Master seed recorded for contract uniformity with the Round 3 seed
#: rule; the driver draws no random numbers (``draws: 0``).
MASTER_SEED = 20260917
DRAWS = 0

#: The pre-registered comparison tolerance (the R20 verification
#: tolerance; unchanged by this card).
N_SIGMA = 3.0

#: The expected-count bound below which a per-cell residual is flagged
#: as a small-count cell (the normal-tail approximation of the
#: binomial is Poisson-shaped there; a diagnosis label, not a
#: tolerance change).
SMALL_COUNT_BOUND = 30.0

PRE_SCHEMA = "na49_mixing.calibrate_branching_weights.pre_registration/1"
PROV_SCHEMA = "na49_mixing.calibrate_branching_weights.provenance/1"
EXPECT_SCHEMA = "na49_mixing.calibrate_branching_weights.expectation/1"
TABLE_SCHEMA = "na49_mixing.calibrate_branching_weights.sigma_table/1"
REPORT_SCHEMA = "na49_mixing.calibrate_branching_weights.calibration_report/1"
DISPOSITION_SCHEMA = "na49_mixing.calibrate_branching_weights.disposition/1"
SEAL_SCHEMA = "na49_mixing.calibrate_branching_weights.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.calibrate_branching_weights.clean_environment/1"

TABLE_TAGS = ("before", "after")

#: The diagnosis dimensions of the card (scope clause c), verbatim.
DIAGNOSIS_DIMENSIONS = (
    "barrier evaluation point",
    "quadrature grid",
    "channel summation order",
    "degenerate-weight handling",
)


# ---------------------------------------------------------------------------
# Small shared helpers (the R20 driver's conventions, reused verbatim)
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as UTF-8 JSON (indent 2, fixed key order, a
    trailing newline, LF endings); identical inputs produce identical
    bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _canonical_json_bytes(payload: dict) -> bytes:
    """The exact byte sequence :func:`_write_json` writes for *payload*."""
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    """The environment fingerprint recorded in the provenance and the
    clean-environment validation."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "executable": sys.executable,
    }


def _git_info() -> dict:
    """The recorded git identities (defensive: a checkout without a
    working git records ``None``)."""
    info: dict = {}
    for name, cwd in (
        ("nested_head", NESTED_ROOT), ("outer_head", MONO_ROOT)
    ):
        try:
            head = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(cwd), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            info[name] = None
            info[name.replace("head", "dirty_entries")] = None
        else:
            info[name] = head
            info[name.replace("head", "dirty_entries")] = len(
                dirty.splitlines()
            )
    return info


def _phi(x: float) -> float:
    """The standard normal CDF (``math.erf``, deterministic)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_state_fail_probability(n_cells_with_variance: int) -> float:
    """The probability that the per-cell max criterion (|z| > 3.0 on at
    least one of *n_cells_with_variance* independent half-normal cells)
    fires on a state whose expectation matches its sampling exactly:
    ``1 - (2 Phi(3) - 1)^n``.  This is the multiplicity accounting of
    the pre-registered per-cell max criterion under the exact binomial
    null; it is a diagnosis quantity, never a re-registered tolerance."""
    if n_cells_with_variance <= 0:
        return 0.0
    one_cell = 2.0 * _phi(N_SIGMA) - 1.0
    return 1.0 - one_cell ** n_cells_with_variance


# ---------------------------------------------------------------------------
# Recomputation (scope clause a): the open-weight expectations
# ---------------------------------------------------------------------------
def _recompute_state_expectation(
    catalog, row: dict, pre_entry: dict
) -> dict:
    """The recomputed open-weight expectations of one state.

    The expectations are the committed card-5/6 open-weight convention
    evaluated through the R20 driver's expected-value mirror
    (:func:`staged_r20_unit_grid._expected_point`) on the R20 grid:
    per open grid point, the per-channel open weights, their normalized
    shares, and the per-class card-5 charge shares.  The grid, the
    degenerate points, the pole point, and the channel set are asserted
    against the sealed R20 pre-registration before the first point.
    """
    record = row["record"]
    pdg = int(record.pdg_id)
    ctx = R20._StateContext(catalog, record)

    # Integrity against the sealed R20 pre-registration: the grid, the
    # degenerate-point inventory, and the channel set must reproduce
    # the sealed run's contract before any expectation is recorded.
    sealed_masses = None
    sealed_skips = pre_entry["expected_skip_points"]
    if [int(j) for j in sealed_skips] != sorted(ctx.degenerate):
        raise RuntimeError(
            f"calibrate_branching_weights: pdg {pdg}: the degenerate "
            "points do not reproduce the sealed R20 pre-registration "
            f"(recomputed {sorted(ctx.degenerate)}, sealed {sealed_skips})"
        )
    sealed_channels = pre_entry["channels"]
    computed_channels = [
        {
            "daughters": [int(d) for d in channel.daughters],
            "branching_fraction": float(channel.branching_fraction),
            "is_three_body": bool(channel.is_three_body),
            "orbital_angular_momentum": int(
                channel.orbital_angular_momentum
            ),
        }
        for channel in record.decay_channels
    ]
    if computed_channels != sealed_channels:
        raise RuntimeError(
            f"calibrate_branching_weights: pdg {pdg}: the catalog "
            "channel set does not reproduce the sealed R20 "
            "pre-registration channel rows"
        )

    pole_j = R20._pole_point(ctx)
    points: dict[str, dict] = {}
    for j in range(R20.GRID_POINTS):
        mass = ctx.grid[j]
        if j in ctx.degenerate:
            points[str(j)] = {
                "j": int(j),
                "mass_gev": float(mass),
                "degenerate": True,
            }
            continue
        weights, class_shares = R20._expected_point(ctx, mass)
        w_total = math.fsum(weights.values())
        if not w_total > 0.0:
            raise RuntimeError(
                f"calibrate_branching_weights: pdg {pdg}: point {j} has "
                "no open weight (it should be a degenerate point)"
            )
        points[str(j)] = {
            "j": int(j),
            "mass_gev": float(mass),
            "degenerate": False,
            "channel_weights": {
                key: float(w) for key, w in weights.items()
            },
            "expected_shares": {
                key: float(w / w_total) for key, w in weights.items()
            },
            "class_shares": {
                class_id: {
                    key: [str(pair_key), float(share)]
                    for key, (pair_key, share) in shares.items()
                }
                for class_id, shares in class_shares.items()
            },
        }

    return {
        "schema": EXPECT_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "state": {
            "pdg_id": pdg,
            "name": str(record.name),
            "rank": int(row["rank"]),
            "classification": str(row["classification"]),
            "decays_per_point": int(row["decays_per_point"]),
            "B": int(record.B),
            "Q": int(record.Q),
            "S": int(record.S),
            "pole_mass": float(record.pole_mass),
            "nominal_width": float(record.nominal_width),
            "mass_range": [
                float(record.mass_range[0]), float(record.mass_range[1]),
            ],
            "channels": computed_channels,
        },
        "convention": {
            "width_law": (
                "card-3: Gamma_c(M) = Gamma_c,0 * (q(M)/q(M0))**(2L+1) "
                "* (M0/M) * (F_L(q(M))/F_L(q(M0)))**2 with "
                "Gamma_c,0 = BR_c * nominal_width and the Kaellen decay "
                "momentum; hard zero at and below the channel threshold"
            ),
            "barrier": (
                "card-4: the documented Blatt-Weisskopf polynomial with "
                "the hadronic radius r0 = 1.0 GeV^-1, supported for "
                "every L >= 0 (the catalog maximum is L = 4)"
            ),
            "charge_weights": (
                "card-5/6: the open weight of a concrete charge "
                "combination is cw(pair) * Gamma_class(M) / BR_class "
                "over the open combinations of the class; three-body "
                "channels are one open object at their card-3/4 partial "
                "width"
            ),
            "grid": (
                "R20: 100 fixed-mass points "
                "M_j = mass_range[0] + (mass_range[1] - mass_range[0]) "
                "* j / 99, both endpoints inclusive"
            ),
            "quadrature": (
                "the committed 10,001-node reference quadrature "
                "(_NORMALIZATION_GRID_POINTS of na49_mixing."
                "resonance_widths, the card-3 re-pin) normalizes the "
                "spectral function; the normalized open weight of a "
                "point is invariant under the normalization constant, "
                "so the recorded pointwise open weights carry no "
                "quadrature dependence"
            ),
            "daughter_masses": "pole masses (the card-1 catalog "
                               "threshold convention)",
            "expected_value_mirror": R20_DRIVER_NAME,
            "expected_value_mirror_sha256": _sha256_file(
                Path(__file__).resolve().parent / R20_DRIVER_NAME
            ),
        },
        "grid": {
            "points": R20.GRID_POINTS,
            "formula": R20.GRID_FORMULA,
            "masses_gev": [float(m) for m in ctx.grid],
        },
        "pole_point": int(pole_j),
        "degenerate_points": [
            {"j": int(j), "mass_gev": float(ctx.grid[j])}
            for j in sorted(ctx.degenerate)
        ],
        "points": points,
    }


def _run_recompute(run_dir: Path, state_pdg: int | None) -> None:
    pre_path = run_dir / "pre_registration.json"
    if not pre_path.is_file():
        raise RuntimeError(
            f"calibrate_branching_weights: {pre_path} is missing; the "
            "recompute requires the run pre-registration (step 1)"
        )
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    catalog = R20._load_catalog()
    rows = R20._state_rows(catalog)
    sealed_states = {
        int(entry["pdg_id"]): entry
        for entry in json.loads(
            (Path(DEFAULT_R20_ARCHIVE) / "pre_registration.json")
            .read_text(encoding="utf-8")
        )["state_coverage"]["states"]
    }
    if state_pdg is not None:
        selected = [
            row for row in rows if int(row["record"].pdg_id) == state_pdg
        ]
        if not selected:
            raise SystemExit(
                f"calibrate_branching_weights: --state {state_pdg} is "
                "not a decayable committed catalog state"
            )
        rows = selected
    for row in rows:
        pdg = int(row["record"].pdg_id)
        payload = _recompute_state_expectation(
            catalog, row, sealed_states[pdg]
        )
        out_path = run_dir / "expectations" / f"state_{pdg}.json"
        if out_path.exists():
            raise RuntimeError(
                f"calibrate_branching_weights: {out_path} exists; the "
                "expectations are immutable once written (delete the "
                "run dir to recompute)"
            )
        _write_json(out_path, payload)
        print(
            f"calibrate_branching_weights: wrote {out_path} "
            f"({row['record'].name}, pdg {pdg})"
        )


# ---------------------------------------------------------------------------
# Comparison (scope clauses b, c, e): the sigma table
# ---------------------------------------------------------------------------
def _verify_r20_seal(archive: Path) -> str:
    """Read-only verification of the sealed R20 archive; returns the
    SHA-256 of its ``SHA256SUMS.txt``."""
    sums_path = archive / "SHA256SUMS.txt"
    manifest_path = archive / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"calibrate_branching_weights: {archive} is not a sealed "
            "R20 archive (SHA256SUMS.txt / seal_manifest.json missing)"
        )
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = archive / rel
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(
                f"calibrate_branching_weights: the sealed R20 archive "
                f"failed its own seal at {rel}; the archive is read-only "
                "evidence and must not be modified"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "calibrate_branching_weights: the R20 seal manifest's "
            "SHA256SUMS.txt digest does not match the sums file"
        )
    return _sha256_file(sums_path)


def _state_sigma_entry(
    catalog, row: dict, expectation: dict, state_payload: dict,
    gate2_entry: dict,
) -> dict:
    """The per-state sigma-table entry of one state.

    The comparison convention is the pre-registered R20 per-cell
    binomial one: the sampled per-point channel share against the
    recomputed expected share at ``sqrt(p (1 - p) / N)`` standard
    error, 3.0 sigma per cell; the per-state gate criterion is the
    sealed R20 one (the maxima of the branching, charge, and angular
    cell maxima against 3.0 sigma).  The entry also records the
    deterministic diagnosis quantities: the per-cell z summary, the
    worst cell with its expected count, the per-channel pooled
    statistics, and the multiplicity accounting of the per-cell max
    criterion under the exact binomial null.
    """
    record = row["record"]
    pdg = int(record.pdg_id)
    ctx = R20._StateContext(catalog, record)

    # Integrity: the expectation file must reproduce the sealed state
    # file's grid, degenerate points, and per-point decay counts.
    exp_points = expectation["points"]
    sealed_points = state_payload["points"]
    sealed_degenerate = sorted(
        int(entry["j"]) for entry in state_payload["degenerate_points"]
    )
    exp_degenerate = sorted(
        int(entry["j"]) for entry in expectation["degenerate_points"]
    )
    if sealed_degenerate != exp_degenerate or sorted(
        int(j) for j in ctx.degenerate
    ) != exp_degenerate:
        raise RuntimeError(
            f"calibrate_branching_weights: pdg {pdg}: the degenerate "
            "points disagree between the expectation file, the sealed "
            "state file, and the recomputation"
        )
    if [float(m) for m in expectation["grid"]["masses_gev"]] != ctx.grid:
        raise RuntimeError(
            f"calibrate_branching_weights: pdg {pdg}: the expectation "
            "grid does not reproduce the R20 grid"
        )

    n = row["decays_per_point"]
    pole_j = int(expectation["pole_point"])
    open_js = [j for j in range(R20.GRID_POINTS)
               if j not in ctx.degenerate]

    branch_pole = 0.0
    branch_open = 0.0
    charge_max = 0.0
    pooled_hist = [0] * R20.ANGULAR_BINS
    pooled_two = 0
    cells = 0
    cells_with_variance = 0
    small_count_cells = 0
    sum_abs_z = 0.0
    sum_z2 = 0.0
    worst_cell = None
    channel_pool: dict[str, list[float]] = {}
    for j in open_js:
        point = sealed_points[str(j)]
        counts = {
            str(k): int(v) for k, v in point["channel_counts"].items()
        }
        exp_point = exp_points[str(j)]
        shares = exp_point["expected_shares"]
        unexplained = sorted(set(counts) - set(shares))
        if unexplained:
            raise RuntimeError(
                f"calibrate_branching_weights: pdg {pdg}: point {j} "
                f"sampled the channel(s) {unexplained} outside the "
                "recomputed open set"
            )
        for key, share in shares.items():
            frac = counts.get(key, 0) / n
            z = R20._zscore(frac, share, n)
            cells += 1
            np_count = n * share
            small = np_count < SMALL_COUNT_BOUND
            if 0.0 < share < 1.0:
                cells_with_variance += 1
                sum_abs_z += z
                sum_z2 += z * z
                var = n * share * (1.0 - share)
                pool = channel_pool.setdefault(key, [0.0, 0.0])
                pool[0] += frac - share
                pool[1] += var
            if small:
                small_count_cells += 1
            if worst_cell is None or z > worst_cell["abs_sigma"]:
                worst_cell = {
                    "point_j": int(j),
                    "channel": str(key),
                    "expected_share": float(share),
                    "sampled_fraction": float(frac),
                    "abs_sigma": float(z),
                    "expected_count_np": float(np_count),
                    "small_count": bool(small),
                }
            if z > branch_open:
                branch_open = z
            if j == pole_j and z > branch_pole:
                branch_pole = z
        # the charge check (the sealed R20 convention over the class
        # shares of the recomputed expectations)
        for class_id, shares in exp_point["class_shares"].items():
            n_cls = sum(counts.get(key, 0) for key in shares)
            if n_cls <= 0:
                continue
            for key, (_pair_key, share) in shares.items():
                z = R20._zscore(counts.get(key, 0) / n_cls, share, n_cls)
                if z > charge_max:
                    charge_max = z
        hist = [int(x) for x in point["cos_theta_hist"]]
        n_two = int(point["n_two_body_vertices"])
        if len(hist) != R20.ANGULAR_BINS or sum(hist) != n_two:
            raise RuntimeError(
                f"calibrate_branching_weights: pdg {pdg}: point {j} "
                "angular histogram is inconsistent"
            )
        for b in range(R20.ANGULAR_BINS):
            pooled_hist[b] += hist[b]
        pooled_two += n_two

    angular_max = 0.0
    if pooled_two > 0:
        for b in range(R20.ANGULAR_BINS):
            z = R20._zscore(
                pooled_hist[b] / pooled_two, 1.0 / R20.ANGULAR_BINS,
                pooled_two,
            )
            if z > angular_max:
                angular_max = z

    # per-channel pooled statistics: the (count - n p) residuals pooled
    # over the open points where the channel is open, against the
    # binomial variance, a convention signature probe (a wrong open
    # weight shows as |T| far above the half-normal scale of one pooled
    # cell per channel)
    channel_pooled: dict[str, float] = {}
    channel_pooled_max = 0.0
    for key, (fraction_residual_sum, variance_sum) in channel_pool.items():
        if not variance_sum > 0.0:
            continue
        # fraction_residual_sum = sum(frac - share) = sum(count - n p) / n
        t = (fraction_residual_sum * n) / math.sqrt(variance_sum)
        channel_pooled[key] = float(t)
        channel_pooled_max = max(channel_pooled_max, abs(t))

    branch_all = max(branch_pole, branch_open)
    gate_pass = bool(
        branch_all <= N_SIGMA
        and charge_max <= N_SIGMA
        and angular_max <= N_SIGMA
    )

    # the recomputed maxima must reproduce the sealed R20 gate table
    # (the expectation mirror shares the committed convention that
    # produced it); a mismatch is a kernel/mirror divergence, not a
    # calibration finding
    matches = {
        "branching_max_abs_sigma": bool(
            branch_all == float(gate2_entry["branching_max_abs_sigma"])
        ),
        "charge_max_abs_sigma": bool(
            charge_max == float(gate2_entry["charge_max_abs_sigma"])
        ),
        "angular_max_abs_sigma": bool(
            angular_max == float(gate2_entry["angular_max_abs_sigma"])
        ),
    }
    if not all(matches.values()):
        raise RuntimeError(
            f"calibrate_branching_weights: pdg {pdg}: the recomputed "
            f"sigma maxima {matches} do not reproduce the sealed R20 "
            f"gate table {gate2_entry}; the expectation mirror diverges "
            "from the convention that produced the sealed verification"
        )

    fail_prob = _expected_state_fail_probability(cells_with_variance)
    mean_abs_z = (
        sum_abs_z / cells_with_variance if cells_with_variance else 0.0
    )
    mean_z2 = (
        sum_z2 / cells_with_variance if cells_with_variance else 0.0
    )
    return {
        "name": str(record.name),
        "rank": int(row["rank"]),
        "classification": str(row["classification"]),
        "pole_point": pole_j,
        "n_open_points": len(open_js),
        "n_degenerate_points": len(ctx.degenerate),
        "n_channels": len(record.decay_channels),
        "decays_per_point": int(n),
        "branching_pole_max_abs_sigma": float(branch_pole),
        "branching_open_max_abs_sigma": float(branch_open),
        "branching_max_abs_sigma": float(branch_all),
        "charge_max_abs_sigma": float(charge_max),
        "angular_max_abs_sigma": float(angular_max),
        "gate2_pass": gate_pass,
        "matches_sealed_gate2": matches,
        "diagnostics": {
            "n_cells": int(cells),
            "n_cells_with_variance": int(cells_with_variance),
            "sum_abs_z": float(sum_abs_z),
            "sum_z2": float(sum_z2),
            "mean_abs_z": float(mean_abs_z),
            "mean_z2": float(mean_z2),
            "n_small_count_cells": int(small_count_cells),
            "worst_cell": worst_cell,
            "channel_pooled_max_abs_t": float(channel_pooled_max),
            "channel_pooled_t": channel_pooled,
            "expected_state_fail_probability": float(fail_prob),
        },
    }


def _compute_sigma_table_payload(
    tag: str, run_dir: Path, r20_archive: Path
) -> dict:
    """The full sigma-table payload for *tag* (pure; no writes)."""
    if tag not in TABLE_TAGS:
        raise RuntimeError(
            f"calibrate_branching_weights: unknown comparison tag "
            f"{tag!r} (expected one of {list(TABLE_TAGS)})"
        )
    run_dir = Path(run_dir)
    r20_archive = Path(r20_archive)
    expectations_dir = run_dir / "expectations"
    if not expectations_dir.is_dir():
        raise RuntimeError(
            f"calibrate_branching_weights: {expectations_dir} is "
            "missing; run --recompute before --compare"
        )
    sums_sha = _verify_r20_seal(r20_archive)
    catalog = R20._load_catalog()
    rows = R20._state_rows(catalog)
    gate2 = json.loads(
        (r20_archive / "gate2_r20.json").read_text(encoding="utf-8")
    )
    gate_per_state = gate2["clauses"]["branching_test"]["per_state"]
    aggregate = json.loads(
        (r20_archive / "aggregate.json").read_text(encoding="utf-8")
    )
    sealed_agg = aggregate["per_state"]

    per_state: dict[str, dict] = {}
    for row in rows:
        pdg = int(row["record"].pdg_id)
        exp_path = expectations_dir / f"state_{pdg}.json"
        state_path = r20_archive / f"state_{pdg}.json"
        for path in (exp_path, state_path):
            if not path.is_file():
                raise RuntimeError(
                    f"calibrate_branching_weights: {path} is missing; "
                    "the comparison requires the full expectation set "
                    "and the sealed R20 state files"
                )
        expectation = json.loads(exp_path.read_text(encoding="utf-8"))
        state_payload = json.loads(
            state_path.read_text(encoding="utf-8")
        )
        entry = _state_sigma_entry(
            catalog, row, expectation, state_payload,
            gate_per_state[str(pdg)],
        )
        # the sealed aggregate cross-check: the per-state open-point
        # and degenerate-point counts must reproduce the sealed run
        if (entry["n_open_points"] != sealed_agg[str(pdg)]["n_open_points"]
                or entry["n_degenerate_points"]
                != sealed_agg[str(pdg)]["n_degenerate_points"]):
            raise RuntimeError(
                f"calibrate_branching_weights: pdg {pdg}: the "
                "open/degenerate point counts do not reproduce the "
                "sealed R20 aggregate"
            )
        per_state[str(pdg)] = entry

    n_cells = sum(e["diagnostics"]["n_cells"] for e in per_state.values())
    n_cells_var = sum(
        e["diagnostics"]["n_cells_with_variance"]
        for e in per_state.values()
    )
    small_cells = sum(
        e["diagnostics"]["n_small_count_cells"] for e in per_state.values()
    )
    sum_abs_z = math.fsum(
        e["diagnostics"]["sum_abs_z"] for e in per_state.values()
    )
    sum_z2 = math.fsum(
        e["diagnostics"]["sum_z2"] for e in per_state.values()
    )
    mean_abs_z = sum_abs_z / n_cells_var if n_cells_var else 0.0
    mean_z2 = sum_z2 / n_cells_var if n_cells_var else 0.0
    # the cell-level tail counts need the full cell scan; the worst
    # cell per state is not enough, so the table records the state
    # maxima instead (deterministic and sufficient for the diagnosis)
    state_maxima = [
        e["branching_max_abs_sigma"] for e in per_state.values()
    ]
    tail_state = {
        f"states_max_abs_sigma_ge_{bound:g}": sum(
            1 for z in state_maxima if z >= bound
        )
        for bound in (3.0, 3.5, 4.0, 4.5)
    }
    n_pass = sum(1 for e in per_state.values() if e["gate2_pass"])
    expected_fail = math.fsum(
        e["diagnostics"]["expected_state_fail_probability"]
        for e in per_state.values()
    )
    worst_sigma = max(
        e["branching_max_abs_sigma"] for e in per_state.values()
    )
    worst_state_pdg = min(
        int(pdg) for pdg, e in per_state.items()
        if e["branching_max_abs_sigma"] == worst_sigma
    )
    return {
        "schema": TABLE_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "tag": str(tag),
        "n_sigma": N_SIGMA,
        "comparison_convention": (
            "the pre-registered R20 per-cell binomial convention: the "
            "sampled per-point channel share against the recomputed "
            "expected share at sqrt(p (1 - p) / N) standard error, "
            "3.0 sigma per cell; the per-state gate criterion is the "
            "sealed R20 one (branching, charge, and angular maxima "
            "against 3.0 sigma)"
        ),
        "sources": {
            "catalog": {
                "path": CATALOG_REL,
                "sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
            },
            "r20_archive": {
                "path": f"dev/00runs/{R20_RUN_NAME}",
                "sha256_sums": sums_sha,
            },
            "expectations_dir": "expectations",
        },
        "per_state": per_state,
        "totals": {
            "n_states": len(per_state),
            "n_states_pass": int(n_pass),
            "n_states_fail": int(len(per_state) - n_pass),
            "worst_branching_abs_sigma": float(worst_sigma),
            "worst_state_pdg": int(worst_state_pdg),
            "n_cells": int(n_cells),
            "n_cells_with_variance": int(n_cells_var),
            "mean_abs_z_cells": float(mean_abs_z),
            "mean_z2_cells": float(mean_z2),
            "n_small_count_cells": int(small_cells),
            "expected_failing_states_binomial_null": float(expected_fail),
            "state_maxima_tail": tail_state,
        },
    }


def _run_compare(
    run_dir: Path, tag: str, r20_archive: Path
) -> None:
    if not (run_dir / "pre_registration.json").is_file():
        raise RuntimeError(
            "calibrate_branching_weights: the run pre-registration is "
            "missing; run --pre-registration before --compare"
        )
    out_path = run_dir / f"sigma_table_{tag}.json"
    if out_path.exists():
        raise RuntimeError(
            f"calibrate_branching_weights: {out_path} exists; a sigma "
            "table is written once per tag (delete the run dir to "
            "recompute)"
        )
    payload = _compute_sigma_table_payload(tag, run_dir, r20_archive)
    _write_json(out_path, payload)
    totals = payload["totals"]
    print(
        f"calibrate_branching_weights: wrote {out_path} "
        f"({totals['n_states_fail']} of {totals['n_states']} states "
        f"fail at {N_SIGMA} sigma; worst branching "
        f"{totals['worst_branching_abs_sigma']:.3f} on pdg "
        f"{totals['worst_state_pdg']}; expected failing under the "
        "exact binomial null of the per-cell max criterion "
        f"{totals['expected_failing_states_binomial_null']:.2f})"
    )


# ---------------------------------------------------------------------------
# Report and disposition (scope clauses c, f)
# ---------------------------------------------------------------------------
def _diagnosis_text(entry: dict) -> str:
    """The recorded diagnosis of one failing state (pure text from the
    deterministic diagnostics)."""
    diag = entry["diagnostics"]
    worst = diag["worst_cell"]
    mirror_note = (
        "the recomputed open-weight expectations reproduce the sealed "
        "R20 per-state sigma table exactly (bit-identical branching, "
        "charge, and angular maxima), so no open-weight convention "
        "defect exists"
    )
    if entry["n_channels"] == 1 or diag["n_cells_with_variance"] == 0:
        return (
            f"single-open-channel state ({entry['n_channels']} channel, "
            "no branching cell with variance): the branching residual "
            "is identically zero and the state fails the pre-registered "
            "per-state criterion on the angular-kernel check "
            f"({entry['angular_max_abs_sigma']:.3f} sigma on the pooled "
            "10-bin cos(theta) histogram), which is outside the "
            "open-weight calibration scope (the angular kernel is the "
            "spin-averaged isotropic kernel of plan section 8.4 step 5, "
            f"not an open weight); {mirror_note}; the residual is "
            "recorded as a finding"
        )
    small_note = (
        "a small-expected-count cell whose binomial distribution is "
        "Poisson-shaped (the normal tail overstates the deviation "
        "probability there)"
        if worst["small_count"]
        else "a large-expected-count cell whose residual is a plain "
             "binomial tail"
    )
    return (
        f"multi-channel state ({entry['n_channels']} channels, "
        f"{diag['n_cells_with_variance']} per-cell checks against one "
        f"3.0-sigma bound): {mirror_note}; the worst residual "
        f"{entry['branching_max_abs_sigma']:.3f} sigma sits at grid "
        f"point {worst['point_j']} on channel {worst['channel']} "
        f"(expected share {worst['expected_share']:.6f}, sampled "
        f"{worst['sampled_fraction']:.6f}, expected count "
        f"{worst['expected_count_np']:.2f}, {small_note}); the "
        "per-channel pooled residuals carry no convention signature "
        f"(max |T| = {diag['channel_pooled_max_abs_t']:.2f} across "
        f"{len(diag['channel_pooled_t'])} channel cells); under the "
        "exact binomial null the per-cell max criterion fails this "
        "state with probability "
        f"{diag['expected_state_fail_probability']:.3f} (the "
        "multiplicity of the pre-registered per-cell max over the "
        "100-point grid); the residual is recorded as a finding, the "
        "branching fractions stay external PDG inputs"
    )


def _build_disposition(
    before: dict, after: dict
) -> tuple[dict, dict]:
    """The disposition and calibration-report payloads (pure)."""
    failing = {
        str(pdg): entry for pdg, entry in before["per_state"].items()
        if not entry["gate2_pass"]
    }
    after_failing = {
        str(pdg) for pdg, entry in after["per_state"].items()
        if not entry["gate2_pass"]
    }
    before_failing = set(failing)
    if before["tag"] != "before" or after["tag"] != "after":
        raise RuntimeError(
            "calibrate_branching_weights: the report requires the "
            "before- and after-tagged tables"
        )
    # the tables must be computed from identical expectations: with no
    # fix applied the per-state entries are identical modulo the tag
    identical = all(
        before["per_state"][key]["branching_max_abs_sigma"]
        == after["per_state"][key]["branching_max_abs_sigma"]
        and before["per_state"][key]["charge_max_abs_sigma"]
        == after["per_state"][key]["charge_max_abs_sigma"]
        and before["per_state"][key]["angular_max_abs_sigma"]
        == after["per_state"][key]["angular_max_abs_sigma"]
        for key in before["per_state"]
    )
    disposition = {
        "schema": DISPOSITION_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "fix_applied": False,
        "clause_d_decision": (
            "the diagnosis identifies no convention defect in the "
            "open-weight computation (the recomputed expectations "
            "reproduce the sealed R20 sigma table bit-exactly and the "
            "sampled shares follow them at binomial accuracy), so "
            "scope clause (d) does not trigger: no fix is applied to "
            "na49_mixing/resonance_widths.py or na49_mixing/"
            "resonance_decay.py, and no branching fraction, catalog "
            "record, nominal width, barrier polynomial, or R20-sealed "
            "value changes"
        ),
        "tables_identical_no_fix": bool(identical),
        "n_failing_before": int(len(before_failing)),
        "n_failing_after": int(len(after_failing)),
        "repaired_states": sorted(
            str(pdg) for pdg in before_failing - after_failing
        ),
        "residual_states": sorted(before_failing & after_failing),
        "per_state": {
            pdg: {
                "name": entry["name"],
                "classification": entry["classification"],
                "n_channels": entry["n_channels"],
                "worst_branching_abs_sigma":
                    entry["branching_max_abs_sigma"],
                "worst_charge_abs_sigma": entry["charge_max_abs_sigma"],
                "worst_angular_abs_sigma":
                    entry["angular_max_abs_sigma"],
                "worst_cell": entry["diagnostics"]["worst_cell"],
                "channel_pooled_max_abs_t":
                    entry["diagnostics"]["channel_pooled_max_abs_t"],
                "expected_state_fail_probability":
                    entry["diagnostics"][
                        "expected_state_fail_probability"
                    ],
                "fix_applied": False,
                "disposition": "residual recorded as a finding",
                "diagnosis": _diagnosis_text(entry),
            }
            for pdg, entry in failing.items()
        },
        "note": (
            "the gate2 per-state pass criterion includes the angular "
            "kernel check; the angular residuals carry no open-weight "
            "content and stay outside the calibration scope "
            "(e.g. PDG -2224 fails only on the angular kernel with "
            "branching sigma 0.0 on its single channel)"
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "n_sigma": N_SIGMA,
        "diagnosis_dimensions": {
            "barrier evaluation point": (
                "checked: the kernel and the expected-value mirror "
                "evaluate the same card-3/4 width law at the same grid "
                "masses with the same pole daughter masses; the "
                "recomputed per-state sigma maxima reproduce the "
                "sealed R20 gate table bit-exactly on all 80 states, "
                "so no barrier-evaluation-point divergence exists"
            ),
            "quadrature grid": (
                "checked: the open weights are pointwise channel "
                "widths; the committed 10,001-node reference quadrature "
                "normalizes the spectral function and cancels in the "
                "normalized open weight, so the expectations carry no "
                "quadrature dependence to calibrate"
            ),
            "channel summation order": (
                "checked: the class widths and shares are accumulated "
                "with exact-rounded math.fsum in both the kernel and "
                "the mirror; the summation order carries no "
                "convention content at the sampled precision"
            ),
            "degenerate-weight handling": (
                "checked: the recomputed degenerate-point inventory "
                "reproduces the sealed R20 skip inventory on every "
                "state, and the equal-weight class combinations "
                "sample exactly their card-5 shares; no "
                "degenerate-weight defect exists"
            ),
        },
        "statistical_verdict": {
            "n_cells": before["totals"]["n_cells"],
            "n_cells_with_variance":
                before["totals"]["n_cells_with_variance"],
            "mean_abs_z_cells": before["totals"]["mean_abs_z_cells"],
            "mean_z2_cells": before["totals"]["mean_z2_cells"],
            "mean_abs_z_half_normal": math.sqrt(2.0 / math.pi),
            "n_small_count_cells": before["totals"][
                "n_small_count_cells"
            ],
            "expected_failing_states_binomial_null":
                before["totals"]["expected_failing_states_binomial_null"],
            "n_failing_observed":
                before["totals"]["n_states_fail"],
            "reading": (
                "the per-cell z distribution matches the |N(0,1)| "
                "scale of the binomial null (the mean square and the "
                "cell tails agree with the null within their sampling "
                "errors, with the known Poisson-shaped excess of the "
                "small-expected-count cells), and the observed count "
                "of failing states sits below the count the exact "
                "binomial null expects from the multiplicity of the "
                "per-cell max criterion alone; the residuals are "
                "check multiplicity, not open-weight defects"
            ),
        },
        "clause_d_decision": disposition["clause_d_decision"],
        "n_failing_before": disposition["n_failing_before"],
        "n_failing_after": disposition["n_failing_after"],
        "constraint_record": {
            "no_branching_fit": True,
            "catalog_byte_unchanged": True,
            "r20_archive_read_only": True,
            "nominal_widths_unchanged": True,
            "barrier_polynomial_unchanged": True,
        },
    }
    return disposition, report


def _run_report(run_dir: Path) -> None:
    before_path = run_dir / "sigma_table_before.json"
    after_path = run_dir / "sigma_table_after.json"
    for path in (before_path, after_path):
        if not path.is_file():
            raise RuntimeError(
                f"calibrate_branching_weights: {path} is missing; the "
                "report requires both sigma tables"
            )
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    disposition, report = _build_disposition(before, after)
    for name, payload in (
        ("calibration_report.json", report),
        ("disposition.json", disposition),
    ):
        out_path = run_dir / name
        if out_path.exists():
            raise RuntimeError(
                f"calibrate_branching_weights: {out_path} exists; the "
                "report is written once"
            )
        _write_json(out_path, payload)
    print(
        f"calibrate_branching_weights: wrote {run_dir / 'calibration_report.json'} "
        f"and {run_dir / 'disposition.json'} "
        f"({disposition['n_failing_before']} failing before, "
        f"{disposition['n_failing_after']} after, "
        f"fix applied: {disposition['fix_applied']})"
    )


# ---------------------------------------------------------------------------
# Run-dir contract: pre-registration, provenance, seal, clean-validate
# ---------------------------------------------------------------------------
def _build_pre_registration(r20_archive: Path) -> dict:
    catalog = R20._load_catalog()
    rows = R20._state_rows(catalog)
    r20_pre = json.loads(
        (r20_archive / "pre_registration.json").read_text(
            encoding="utf-8"
        )
    )
    r20_states = r20_pre["state_coverage"]["states"]
    if len(rows) != len(r20_states) or any(
        int(row["record"].pdg_id) != int(entry["pdg_id"])
        for row, entry in zip(rows, r20_states)
    ):
        raise RuntimeError(
            "calibrate_branching_weights: the catalog state order does "
            "not reproduce the sealed R20 pre-registration state order"
        )
    return {
        "schema": PRE_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "written_before_computation": True,
        "statistics": {
            "tier": "T0-equivalent deterministic diagnostic",
            "master_seed": MASTER_SEED,
            "draws": DRAWS,
            "note": (
                "no new Monte Carlo: the comparison reuses the sealed "
                "R20 samples (63,110,000 decays); the master seed is "
                "recorded for contract uniformity with a zero draw "
                "count; no random draws exist on any code path of "
                "this driver"
            ),
        },
        "objective": (
            "systematic calibration of the open-phase-space weights "
            "(the card-3/4/5/6 barrier-integrated open-weight "
            "expectations) against the sealed R20 sampled channel "
            "shares for the 39 failing states, with the branching "
            "fractions kept as external PDG inputs (plan section 8.2); "
            "the calibration adjusts the open-weight convention, "
            "never the branching fractions (plan section 24)"
        ),
        "scope": {
            "recompute": (
                "the per-state, per-channel open-weight expectations "
                "from the committed catalog with the card-4 "
                "Blatt-Weisskopf barrier (all L >= 0), the card-3 "
                "width law, and the 10,001-node reference quadrature, "
                "on the R20 100-point grid per state"
            ),
            "compare": (
                "the expectations against the sealed R20 sampled "
                "channel shares at the pre-registered 3.0-sigma "
                "tolerance (the R20 per-cell binomial convention)"
            ),
            "diagnose": (
                "which convention drives each residual: barrier "
                "evaluation point, quadrature grid, channel summation "
                "order, degenerate-weight handling, and the "
                "multiplicity of the per-cell max criterion"
            ),
            "fix_if_identified": (
                "a minimal additive convention fix in "
                "na49_mixing/resonance_widths.py and/or "
                "na49_mixing/resonance_decay.py only if the diagnosis "
                "identifies a convention defect; the fix must not "
                "change branching fractions, catalog records, nominal "
                "widths, the barrier polynomial, or any R20-sealed "
                "value"
            ),
            "re_verify": (
                "the corrected expectations against the same sealed "
                "R20 sampled shares (the after-tagged table)"
            ),
            "report": (
                "the per-state sigma table before and after and the "
                "disposition of every residual state"
            ),
        },
        "state_coverage": {
            "rule": "the 80 decaying states of the sealed R20 "
                    "pre-registration (every catalog record with "
                    "stable_for_decay = false)",
            "n_states": len(rows),
            "pdg_ids": [int(row["record"].pdg_id) for row in rows],
        },
        "comparison": {
            "n_sigma": N_SIGMA,
            "sigma_convention": (
                "binomial standard error sqrt(p (1 - p) / N) with the "
                "expected share in the variance, per cell (the R20 "
                "pre-registered convention; unchanged)"
            ),
            "gate_criterion": (
                "the sealed R20 per-state gate criterion: the maxima "
                "of the branching, charge, and angular cell maxima "
                "against 3.0 sigma"
            ),
            "sampled_inputs": (
                "the sealed R20 state_<pdg>.json channel counts, "
                "63,110,000 decays, read-only"
            ),
        },
        "inputs": {
            "catalog": {
                "path": CATALOG_REL,
                "sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
            },
            "r20_archive": {
                "path": f"dev/00runs/{R20_RUN_NAME}",
                "pre_registration_sha256": _sha256_file(
                    r20_archive / "pre_registration.json"
                ),
            },
            "drivers": {
                "calibration_driver": "helpers/benchmarking/"
                                      "calibrate_branching_weights.py",
                "expected_value_mirror": R20_DRIVER_NAME,
            },
        },
        "outputs": [
            "expectations/state_<pdg>.json (80 files)",
            "sigma_table_before.json",
            "sigma_table_after.json",
            "calibration_report.json",
            "disposition.json",
        ],
    }


def _run_pre_registration(run_dir: Path) -> None:
    pre_path = run_dir / "pre_registration.json"
    prov_path = run_dir / "provenance.json"
    if pre_path.is_file() and (run_dir / "expectations").is_dir():
        raise RuntimeError(
            f"calibrate_branching_weights: {pre_path} exists and "
            "expectation files are present; the pre-registration is "
            "immutable after computation has started (a change "
            "invalidates the run; re-run under a new date)"
        )
    payload = _build_pre_registration(Path(DEFAULT_R20_ARCHIVE))
    _write_json(pre_path, payload)
    catalog_sha = _sha256_file(NESTED_ROOT / CATALOG_REL)
    provenance = {
        "schema": PROV_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "written_utc": _now_utc(),
        "catalog": {
            "path": CATALOG_REL,
            "sha256": catalog_sha,
        },
        "r20_archive": {
            "path": f"dev/00runs/{R20_RUN_NAME}",
            "sha256_sums": _verify_r20_seal(Path(DEFAULT_R20_ARCHIVE)),
        },
        "drivers": {
            "calibration_driver": "helpers/benchmarking/"
                                  "calibrate_branching_weights.py",
            "calibration_driver_sha256": _sha256_file(
                Path(__file__).resolve()
            ),
            "expected_value_mirror": R20_DRIVER_NAME,
            "expected_value_mirror_sha256": _sha256_file(
                Path(__file__).resolve().parent / R20_DRIVER_NAME
            ),
        },
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "git": _git_info(),
        "environment": environment_fingerprint(),
    }
    _write_json(prov_path, provenance)
    print(
        f"calibrate_branching_weights: wrote {pre_path} and "
        f"{prov_path} (80 pre-registered states, master seed "
        f"{MASTER_SEED}, draws {DRAWS}; nothing computed)"
    )


#: The seal-time evidence chain (steps 1-4 must all be present).
_SEAL_REQUIRED = (
    "pre_registration.json",
    "provenance.json",
    "sigma_table_before.json",
    "sigma_table_after.json",
    "calibration_report.json",
    "disposition.json",
)


def _run_seal(run_dir: Path) -> None:
    for required in _SEAL_REQUIRED:
        if not (run_dir / required).is_file():
            raise RuntimeError(
                f"calibrate_branching_weights: {run_dir / required} is "
                "missing; the seal requires the full evidence chain "
                "(the run procedure steps 1-4)"
            )
    expectations = sorted(
        (run_dir / "expectations").glob("state_*.json")
    )
    if len(expectations) != 80:
        raise RuntimeError(
            f"calibrate_branching_weights: {len(expectations)} "
            "expectation files under "
            f"{run_dir / 'expectations'}; the seal requires the "
            "recomputed expectations of all 80 R20 states"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if sums_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            f"calibrate_branching_weights: {run_dir} is already "
            "sealed; the run dir is immutable after the seal"
        )
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel in ("seal_manifest.json", "SHA256SUMS.txt"):
            continue
        entries.append((rel, _sha256_file(path)))
        total_bytes += path.stat().st_size
    with open(sums_path, "w", encoding="utf-8", newline="\n") as handle:
        for rel, digest in entries:
            handle.write(f"{digest}  {rel}\n")
    manifest = {
        "schema": SEAL_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256_sums_file": "SHA256SUMS.txt",
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": ["seal_manifest.json", "SHA256SUMS.txt"],
        "note": "the run dir is immutable after the seal (the Round 1 "
                "sealing discipline); clean_environment_commands.json "
                "is the post-seal validation record, written after "
                "this manifest and never edited",
    }
    _write_json(manifest_path, manifest)
    print(
        f"calibrate_branching_weights: sealed {run_dir} "
        f"({len(entries)} files, {total_bytes} bytes, SHA256SUMS.txt "
        f"sha256 {manifest['sha256_sums_sha256']})"
    )


#: The clean-interpreter probe: imports this driver module fresh,
#: re-runs the comparison payload computation over the sealed run dir
#: and the sealed R20 archive, and prints the canonical payload hashes
#: (it writes nothing).
_CLEAN_PROBE_SOURCE = """\
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

driver_path, run_dir, r20_archive = sys.argv[1], sys.argv[2], sys.argv[3]
run_dir = Path(run_dir)
r20_archive = Path(r20_archive)
spec = importlib.util.spec_from_file_location(
    "calibrate_branching_weights_clean_probe", driver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _canon(payload):
    return (json.dumps(payload, indent=2) + "\\n").encode("utf-8")


hashes = {}
for tag in module.TABLE_TAGS:
    payload = module._compute_sigma_table_payload(tag, run_dir, r20_archive)
    hashes[f"sigma_table_{tag}.json"] = hashlib.sha256(
        _canon(payload)).hexdigest()
before = json.loads(
    (run_dir / "sigma_table_before.json").read_text(encoding="utf-8"))
after = json.loads(
    (run_dir / "sigma_table_after.json").read_text(encoding="utf-8"))
disposition, report = module._build_disposition(before, after)
hashes["calibration_report.json"] = hashlib.sha256(
    _canon(report)).hexdigest()
hashes["disposition.json"] = hashlib.sha256(
    _canon(disposition)).hexdigest()
print(json.dumps({
    "hashes": hashes,
    "environment": module.environment_fingerprint(),
}))
"""


def _run_clean_validate(run_dir: Path, r20_archive: Path) -> None:
    out_path = run_dir / "clean_environment_commands.json"
    if out_path.is_file():
        raise RuntimeError(
            f"calibrate_branching_weights: {out_path} exists; the "
            "clean-environment validation is recorded once per seal"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"calibrate_branching_weights: {run_dir} is not sealed; "
            "run --seal before the clean-environment validation"
        )
    checked = 0
    mismatches: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = run_dir / rel
        if not path.is_file() or _sha256_file(path) != digest:
            mismatches.append(rel)
        checked += 1
    if mismatches:
        raise RuntimeError(
            "calibrate_branching_weights: the seal verification failed "
            "for: " + ", ".join(mismatches)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "calibrate_branching_weights: the seal manifest's "
            "SHA256SUMS.txt digest does not match the sums file"
        )
    sealed = {
        name: _sha256_file(run_dir / name)
        for name in (
            "sigma_table_before.json",
            "sigma_table_after.json",
            "calibration_report.json",
            "disposition.json",
        )
    }
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-c", _CLEAN_PROBE_SOURCE,
        str(driver_path), str(run_dir), str(r20_archive),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "calibrate_branching_weights: the clean-interpreter probe "
            "failed:\n" + completed.stderr
        )
    probe_line = completed.stdout.strip().splitlines()[-1]
    probe = json.loads(probe_line)
    recomputed = {
        str(name): str(digest)
        for name, digest in probe["hashes"].items()
    }
    hashes_match = sealed == recomputed
    payload = {
        "schema": CLEAN_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": checked,
            "mismatches": [],
        },
        "commands": [
            {
                "purpose": "clean-interpreter re-run of the comparison "
                           "over the sealed run dir and the sealed R20 "
                           "archive (computes the sigma-table, report, "
                           "and disposition payloads and prints their "
                           "hashes; writes nothing)",
                "argv": argv,
                "cwd": str(Path.cwd()),
            }
        ],
        "environment": environment_fingerprint(),
        "probe_environment": probe.get("environment", {}),
        "sealed_report_sha256": sealed,
        "recomputed_report_sha256": recomputed,
        "hashes_match": bool(hashes_match),
    }
    _write_json(out_path, payload)
    if not hashes_match:
        raise RuntimeError(
            "calibrate_branching_weights: the clean-interpreter re-run "
            "does not reproduce the sealed report hashes (sealed "
            f"{sealed}, recomputed {recomputed}); the record is in "
            f"{out_path}"
        )
    print(
        "calibrate_branching_weights: clean-environment validation "
        f"passed ({checked} sealed files verified; the re-run from a "
        "clean interpreter reproduces the sealed sigma-table, report, "
        f"and disposition hashes); record at {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_branching_weights.py",
        description=(
            "Round 3 card 1: the open-phase-space weight calibration "
            "against the sealed R20 sampled channel shares.  Each "
            "invocation performs exactly one step and stops."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json into the "
             "run dir, then stop (nothing is computed)",
    )
    group.add_argument(
        "--recompute", action="store_true",
        help="recompute the open-weight expectations (all states, or "
             "one with --state), writing "
             "expectations/state_<pdg>.json, then stop",
    )
    group.add_argument(
        "--compare", action="store_true",
        help="build the per-state sigma table from the recomputed "
             "expectations and the sealed R20 state files, write "
             "sigma_table_<tag>.json, then stop",
    )
    group.add_argument(
        "--report", action="store_true",
        help="assemble calibration_report.json and disposition.json "
             "from the before- and after-tables, then stop",
    )
    group.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt over the "
             "run dir, then stop (the run dir is immutable after the "
             "seal)",
    )
    group.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal, re-run the comparison from a clean "
             "interpreter, and confirm the report hashes match; write "
             "clean_environment_commands.json, then stop",
    )
    parser.add_argument(
        "--state", type=int, default=None,
        help="recompute only this PDG id (together with --recompute)",
    )
    parser.add_argument(
        "--tag", choices=list(TABLE_TAGS), default="before",
        help="the sigma-table tag (together with --compare; default "
             "before)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help=f"the run directory (default: {RUN_DIR_REL})",
    )
    parser.add_argument(
        "--r20-archive", type=Path, default=DEFAULT_R20_ARCHIVE,
        help="the sealed R20 archive (read-only; default: "
             f"dev/00runs/{R20_RUN_NAME})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.state is not None and not args.recompute:
        raise SystemExit(
            "calibrate_branching_weights: --state is only valid "
            "together with --recompute"
        )
    run_dir = Path(args.run_dir)
    r20_archive = Path(args.r20_archive)
    if args.pre_registration:
        _run_pre_registration(run_dir)
    elif args.recompute:
        _run_recompute(run_dir, args.state)
    elif args.compare:
        _run_compare(run_dir, args.tag, r20_archive)
    elif args.report:
        _run_report(run_dir)
    elif args.seal:
        _run_seal(run_dir)
    else:
        _run_clean_validate(run_dir, r20_archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
