#!/usr/bin/env python3
"""Held-out source-local correlation helper (card TASK-20260905-physics-review, M9 phase A).

Executes the two held-out observables of spec section 9 (``dev/mixing2006/
docs/color_neutral_regge_excitation_model_spec.md``) at the M8 combined-error
calibration parameter set:

1. K-Lambda source-local association: the fraction of forward K+ (``xF >= 0.2``)
   whose ``source_id`` is shared with at least one final strange baryon
   (Lambda 3122, Sigma0 3212, Sigma- 3112), against a within-event
   source-shuffled baseline that breaks the co-source link.
2. p-pbar source-local traceability: the parent-source mass of every final
   pbar against the p-pbar pair threshold 1.876544 GeV, the source-mass
   distribution conditional on the event carrying a pbar against the
   unconditional distribution, and the same-source p-pbar invariant-mass
   structure against the same shuffle baseline.

The falsifier bands are pre-registered in
``dev/00runs/staged_heldout_2026-09-07/pre_registration.json``. This harness
reads that file, cross-checks every pre-registered constant against the
module constants below, and refuses to run on any mismatch: the bands, the
seeds, the cut, the estimator definitions, and the shuffle construction are
fixed before the events exist and are never tuned after the run (spec
section 7 falsifier policy). A recorded fail is a legitimate result.

Bookkeeping: the observables use only the existing
``FinalParticle.source_id`` / ``parent_id`` fields and the ``SourceRecord.mass``
property (``na49_mixing/continuum_channels.py``, ``SourceDecayer.decay``);
no new bookkeeping. The harness asserts ``parent_id == source_id`` on every
final particle and ``source_id`` in range of the event's source list, and
stops with a reported gap if either fails.

Shuffle baseline: within each event, the multiset of ``source_id`` values
over the event's final particles is permuted uniformly at random (strictly
within-event; particles never move between events). Broken: every co-source
relation between two different particles, in particular the
kaon-strange-baryon link of the associated-strangeness families and the
proton-antiproton link of the baryon-pair family. Kept: each particle's
species and four-momentum, the per-event particle list, the per-source
particle multiplicities (the ``source_id`` multiset is preserved exactly),
and the source records themselves, so the marginal source-mass distribution
is preserved exactly. The null isolates the source-link correlation.

Layout: this file is byte-identical in the monorepo helper tree
(``helpers/benchmarking/``) and the vendored tree
(``dev/mixing2006/helpers/benchmarking/``); the tree is detected from
``__file__``, so the mixing2006 package and the tree-local run-artifact root
resolve correctly on both sides (monorepo: ``dev/00runs``; vendored:
``00runs``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Layout (byte-identical in both helper trees) ---------------------------

_BENCH_DIR = Path(__file__).resolve().parent
if (_BENCH_DIR.parents[1] / "na49_mixing").is_dir():
    # vendored tree: dev/mixing2006/helpers/benchmarking/this_file.py
    MIXING2006_DIR = _BENCH_DIR.parents[1]
    RUNS_ROOT = MIXING2006_DIR / "00runs"
else:
    # monorepo tree: helpers/benchmarking/this_file.py
    REPO_ROOT = _BENCH_DIR.parents[1]
    MIXING2006_DIR = REPO_ROOT / "dev" / "mixing2006"
    RUNS_ROOT = REPO_ROOT / "dev" / "00runs"

# --- Pre-registered protocol constants (mirrored in the pre-registration;
#     the run cross-checks the file against these and refuses a mismatch) ---

ARTIFACT_SCHEMA_VERSION = 1
CARD = "TASK-20260905-physics-review"
MILESTONE = "M9"
PHASE = "A"
RUN_DESCRIPTION = "staged_heldout_2026-09-07"
ARTIFACT_NAME = "heldout_correlations_report.json"
STAGE_LABEL = "heldout_correlations"
PRODUCTION_MODE = "regge"
BEAM_MOMENTUM_GEV_C = 158.0

CONFIG_SEED = 20260831
PRODUCTION_SEED = 20310831  # config seed + 50000 (M9); the M8 replica block ends at 20300838
SHUFFLE_SEED_BASE = 20310832  # production seed + 1; realization r uses base + r, r = 0..15
SHUFFLE_REALIZATIONS = 16
BOOTSTRAP_SEED = 20312831  # production seed + 2000; one generator, sequential consumption
BOOTSTRAP_REPLICATES = 200
PILOT_SEED = 20309831  # N-rationale pilot only (config seed + 49000)

FORWARD_XF_MIN = 0.2
K_PLUS_PDG = 321
PROTON_PDG = 2212
ANTIPROTON_PDG = -2212
STRANGE_BARYON_IDS = (3122, 3212, 3112)  # Lambda, Sigma0, Sigma-
PAIR_THRESHOLD_GEV = 1.876544  # 2 x 0.938272
WILSON_Z = 1.959963984540054
SQRT_S_GEV = 17.270205201372406  # sqrt(s) of Reaction(158.0, 2212, 2212)

M8_PARAMETER_SET = {
    "regge.alpha_exc0": 1.1418864566363858,
    "regge.effective_transfer_slope": 8.504140762083612,
    "regge.edge_power": 2.3769493784466396,
    "continuum.match_mass": 2.680194110727528,
    "continuum.pion_multiplicity_slope": 1.5226475471033067,
    "continuum.strange_lambda_strength": 0.30184606498191385,
    "continuum.strange_sigma_strength": 0.3777282734799746,
    "continuum.strange_pair_strength": 0.19187855210738686,
    "continuum.pair_strength": 0.08411858717003319,
    "continuum.pair_threshold_power": 1.7610429093564994,
}

PRE_REGISTERED_BANDS = {
    "O1_klambda_association": {
        "statistic": "D1 = F1 - F1_shuffle_mean",
        "pass": {"min_D1": 0.2, "min_Z": 3.0},
        "fail": {"max_D1": 0.05},
        "inconclusive": "0.05 < D1 < 0.20, or D1 >= 0.20 with Z1 < 3.0",
    },
    "O2a_traceability": {
        "statistic": "F_tr = 1 - n_below / N_pbar (violations = pbars with parent-source mass < 1.876544 GeV)",
        "pass": {"max_violations": 0, "min_n_pbar": 30},
        "fail": {"min_violations": 1},
        "inconclusive": "zero violations but N_pbar < 30",
    },
    "O2a_parent_mass_shift": {
        "statistic": "dM_parent = mean(parent masses) - mean(unconditional source masses)",
        "pass": {"sign": "positive", "min_Z": 3.0},
        "fail": {"sign": "negative", "min_Z": 3.0},
        "inconclusive": "Z < 3.0 in either direction",
    },
    "O2a_event_conditioned_mass_shift": {
        "statistic": "dM_event = mean(source masses of pbar-carrying events) - mean(unconditional source masses)",
        "pass": {"sign": "positive", "min_Z": 3.0},
        "fail": {"sign": "negative", "min_Z": 3.0},
        "inconclusive": "Z < 3.0 in either direction",
    },
    "O2b_pair_structure": {
        "statistic": "D2 = mean(shuffled same-source p-pbar invariant masses) - mean(real same-source p-pbar invariant masses)",
        "pass": {"sign": "positive", "min_Z": 3.0},
        "fail": {"sign": "negative", "min_Z": 3.0},
        "inconclusive": "Z < 3.0 in either direction",
    },
    "O2_verdict_rule": "O2 passes only if all four components (traceability, parent-source mass shift, event-conditioned mass shift, pair structure) pass; any failing component makes O2 fail; otherwise O2 is inconclusive",
}

_B1 = PRE_REGISTERED_BANDS["O1_klambda_association"]
_B_TR = PRE_REGISTERED_BANDS["O2a_traceability"]
_B_PS = PRE_REGISTERED_BANDS["O2a_parent_mass_shift"]
_B_ES = PRE_REGISTERED_BANDS["O2a_event_conditioned_mass_shift"]
_B_PP = PRE_REGISTERED_BANDS["O2b_pair_structure"]

# Histogram edges for the recorded distributions (report only, no bands).
SOURCE_MASS_EDGES = np.arange(0.75, 15.01, 0.5)
PAIR_MASS_EDGES = np.arange(1.8, 12.01, 0.1)


# --- Small utilities ---------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        val = out.stdout.strip()
        return val if out.returncode == 0 and val else None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_jsonable(obj):
    """Recursively convert numpy scalars and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial fraction."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = n + z * z
    center = (k + z * z / 2.0) / denom
    half = z * math.sqrt(p * (1.0 - p) * n + z * z / 4.0) / denom
    return (center - half, center + half)


def wald_se(f: float, n: int) -> float:
    """Binomial Wald standard error at the observed fraction."""
    if n <= 0:
        return float("nan")
    return math.sqrt(f * (1.0 - f) / n)


def _pair_invariant_mass(ma: np.ndarray, mb: np.ndarray) -> float:
    e = ma[0] + mb[0]
    px = ma[1] + mb[1]
    py = ma[2] + mb[2]
    pz = ma[3] + mb[3]
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


# --- Per-event compact record ------------------------------------------------


@dataclass
class HeldoutEvent:
    """Compact per-event record carrying the source bookkeeping.

    ``source_masses[i]`` is the mass (GeV) of the source with ``source_id``
    ``i`` (``SourceRecord.mass`` = invariant mass of the source four-vector).
    ``src[j]`` is the ``FinalParticle.source_id`` of particle ``j`` and
    ``mom[j]`` its four-vector ``(e, px, py, pz)``. The index arrays are
    particle-position selections and are therefore invariant under the
    source shuffle (which permutes ``src`` only).
    """

    source_masses: np.ndarray
    source_roles: tuple
    pdg: np.ndarray
    src: np.ndarray
    mom: np.ndarray
    idx_forward_kplus: np.ndarray
    idx_strange_baryon: np.ndarray
    idx_pbar: np.ndarray
    idx_proton: np.ndarray

    def with_shuffled_sources(self, rng: np.random.Generator) -> "HeldoutEvent":
        """Within-event uniform permutation of the source_id multiset."""
        perm = rng.permutation(self.src.size)
        return HeldoutEvent(
            self.source_masses,
            self.source_roles,
            self.pdg,
            self.src[perm],
            self.mom,
            self.idx_forward_kplus,
            self.idx_strange_baryon,
            self.idx_pbar,
            self.idx_proton,
        )


def heldout_event_from_arrays(
    source_masses,
    source_roles,
    pdg,
    src,
    mom,
    sqrt_s: float = SQRT_S_GEV,
) -> HeldoutEvent:
    """Build a HeldoutEvent from raw arrays (the hermetic construction path).

    ``mom`` rows are ``(e, px, py, pz)``; ``src[j]`` must index
    ``source_masses``. The forward-K+ selection uses ``xF = 2 pz / sqrt_s``.
    """
    pdg = np.asarray(pdg, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    mom = np.asarray(mom, dtype=np.float64).reshape(-1, 4)
    source_masses = np.asarray(source_masses, dtype=np.float64)
    if src.size and (int(src.min()) < 0 or int(src.max()) >= source_masses.size):
        raise ValueError(
            f"source_id out of range: min {int(src.min())}, max {int(src.max())}, "
            f"n_sources {source_masses.size}"
        )
    xf = 2.0 * mom[:, 3] / float(sqrt_s)
    idx_fwd = np.nonzero((pdg == K_PLUS_PDG) & (xf >= FORWARD_XF_MIN))[0]
    idx_strange = np.nonzero(np.isin(pdg, np.asarray(STRANGE_BARYON_IDS, dtype=np.int64)))[0]
    idx_pbar = np.nonzero(pdg == ANTIPROTON_PDG)[0]
    idx_proton = np.nonzero(pdg == PROTON_PDG)[0]
    return HeldoutEvent(
        source_masses,
        tuple(source_roles),
        pdg,
        src,
        mom,
        idx_fwd,
        idx_strange,
        idx_pbar,
        idx_proton,
    )


def heldout_events_from_excitation(events, sqrt_s: float = SQRT_S_GEV):
    """Convert ExcitationEvent records, asserting the bookkeeping contract.

    Every final particle must carry ``parent_id == source_id`` (the continuum
    decay sets both to the decaying source's id) and a ``source_id`` within
    the event's source list. Any violation stops the run with a reported gap.
    """
    out = []
    for event in events:
        parts = event.final_particles
        for p in parts:
            if p.parent_id != p.source_id:
                raise ValueError(
                    f"bookkeeping gap: event {event.event_id} particle "
                    f"{p.particle_id} has parent_id {p.parent_id!r} != "
                    f"source_id {p.source_id!r}; the held-out observables "
                    "require the continuum parent_id == source_id contract; "
                    "stopping instead of adding bookkeeping"
                )
        pdg = np.array([p.pdg_id for p in parts], dtype=np.int64)
        src = np.array([int(p.source_id) for p in parts], dtype=np.int64)
        mom = np.empty((len(parts), 4), dtype=np.float64)
        for i, p in enumerate(parts):
            mom[i, 0] = p.p4.e
            mom[i, 1] = p.p4.px
            mom[i, 2] = p.p4.py
            mom[i, 3] = p.p4.pz
        source_masses = np.array([float(s.mass) for s in event.sources], dtype=np.float64)
        source_roles = tuple(str(s.role) for s in event.sources)
        out.append(heldout_event_from_arrays(source_masses, source_roles, pdg, src, mom, sqrt_s))
    return out


# --- Estimators ---------------------------------------------------------------


def association_fraction(
    events,
    sqrt_s: float = SQRT_S_GEV,
    xf_min: float = FORWARD_XF_MIN,
):
    """Forward-K+ source-local association fraction (pre-registered O1).

    Denominator: final K+ (pdg 321) with ``xF = 2 pz / sqrt_s >= xf_min``.
    Numerator: those sharing their ``source_id`` with at least one final
    strange baryon (3122, 3212, 3112) of the same event.
    Returns ``(n_numerator, n_denominator, co_counts)``.
    """
    n_num = 0
    n_den = 0
    co_counts: list[int] = []
    for ev in events:
        idx_fwd = ev.idx_forward_kplus.tolist()
        if not idx_fwd:
            continue
        src = ev.src.tolist()
        strange = ev.idx_strange_baryon.tolist()
        strange_src = [src[j] for j in strange]
        for i in idx_fwd:
            n_den += 1
            s = src[i]
            c = sum(1 for t in strange_src if t == s)
            co_counts.append(c)
            if c:
                n_num += 1
    return n_num, n_den, co_counts


def reverse_association_fraction(events):
    """Diagnostic: strange baryons sharing a source with a forward K+."""
    n_with = 0
    n_baryon = 0
    for ev in events:
        if not ev.idx_strange_baryon.size:
            continue
        src = ev.src.tolist()
        fwd_src = {src[i] for i in ev.idx_forward_kplus.tolist()}
        for j in ev.idx_strange_baryon.tolist():
            n_baryon += 1
            if src[j] in fwd_src:
                n_with += 1
    return n_with, n_baryon


def association_by_source_role(events):
    """Diagnostic: associated forward K+ split by the shared source's role."""
    counts: dict[str, int] = {}
    total = 0
    for ev in events:
        idx_fwd = ev.idx_forward_kplus.tolist()
        if not idx_fwd:
            continue
        src = ev.src.tolist()
        strange_src = {src[j] for j in ev.idx_strange_baryon.tolist()}
        for i in idx_fwd:
            if src[i] in strange_src:
                role = ev.source_roles[src[i]]
                counts[role] = counts.get(role, 0) + 1
                total += 1
    return counts, total


def pbar_parent_source_records(events, pair_threshold_gev: float = PAIR_THRESHOLD_GEV):
    """Pre-registered O2a statistics on the real assignment.

    Returns a dict with the per-event aggregates used by the bootstrap
    (``n_pbar``, ``sum_parent``, ``has_pbar``, ``n_src``, ``sum_src``) plus
    the parent-mass list and the below-threshold violation count.
    """
    n_pbar_e: list[float] = []
    sum_parent_e: list[float] = []
    has_pbar_e: list[float] = []
    n_src_e: list[float] = []
    sum_src_e: list[float] = []
    parent_masses: list[float] = []
    n_below = 0
    for ev in events:
        src = ev.src.tolist()
        pbar_idx = ev.idx_pbar.tolist()
        n_pb = len(pbar_idx)
        p_sum = 0.0
        for i in pbar_idx:
            pm = float(ev.source_masses[src[i]])
            parent_masses.append(pm)
            p_sum += pm
            if pm < pair_threshold_gev:
                n_below += 1
        n_pbar_e.append(float(n_pb))
        sum_parent_e.append(p_sum)
        has_pbar_e.append(1.0 if n_pb else 0.0)
        n_src_e.append(float(ev.source_masses.size))
        sum_src_e.append(float(ev.source_masses.sum()))
    return {
        "n_pbar_e": np.asarray(n_pbar_e, dtype=np.float64),
        "sum_parent_e": np.asarray(sum_parent_e, dtype=np.float64),
        "has_pbar_e": np.asarray(has_pbar_e, dtype=np.float64),
        "n_src_e": np.asarray(n_src_e, dtype=np.float64),
        "sum_src_e": np.asarray(sum_src_e, dtype=np.float64),
        "parent_masses": np.asarray(parent_masses, dtype=np.float64),
        "n_below": int(n_below),
    }


def same_source_ppbar_invariant_masses(events):
    """Same-source (p, pbar) invariant masses (pre-registered O2b, real side).

    One pair per (final pbar, final proton) with equal ``source_id`` in the
    same event; returns the per-event arrays of invariant masses.
    """
    out: list[np.ndarray] = []
    for ev in events:
        if not ev.idx_pbar.size:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        src = ev.src.tolist()
        vals: list[float] = []
        for i in ev.idx_pbar.tolist():
            si = src[i]
            mi = ev.mom[i]
            for j in ev.idx_proton.tolist():
                if src[j] == si:
                    vals.append(_pair_invariant_mass(mi, ev.mom[j]))
        out.append(np.asarray(vals, dtype=np.float64))
    return out


def all_source_masses(events) -> np.ndarray:
    """Unconditional source-mass sample (all sources of all events)."""
    return np.concatenate([ev.source_masses for ev in events]) if events else np.zeros(0)


def pbar_event_source_masses(events) -> np.ndarray:
    """Source masses of pbar-carrying events (the O2a conditioned sample)."""
    return np.concatenate(
        [ev.source_masses for ev in events if ev.idx_pbar.size]
    ) if events else np.zeros(0)


# --- Bootstrap (pre-registered method for the distribution statistics) --------


def _bootstrap_mean(sum_arr: np.ndarray, cnt_arr: np.ndarray, replicates: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Event-level bootstrap of sum/count over resampled events."""
    n = cnt_arr.size
    out = np.empty(replicates, dtype=np.float64)
    invalid = 0
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        c = cnt_arr[idx].sum()
        if c > 0:
            out[b] = sum_arr[idx].sum() / c
        else:
            out[b] = np.nan
            invalid += 1
    return out, invalid


def _bootstrap_mean_shift(sum_a: np.ndarray, n_a: np.ndarray,
                          sum_b: np.ndarray, n_b: np.ndarray,
                          replicates: int, rng: np.random.Generator):
    """Event-level bootstrap of (mean_a - mean_b) on the same resamples."""
    n = sum_a.size
    out = np.empty(replicates, dtype=np.float64)
    invalid = 0
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        ca = n_a[idx].sum()
        cb = n_b[idx].sum()
        if ca > 0 and cb > 0:
            out[b] = sum_a[idx].sum() / ca - sum_b[idx].sum() / cb
        else:
            out[b] = np.nan
            invalid += 1
    return out, invalid


def _stat(values: np.ndarray) -> dict:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"n": 0, "mean": None, "std_ddof1": None,
                "p025": None, "p975": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "p025": float(np.percentile(vals, 2.5)),
        "p975": float(np.percentile(vals, 97.5)),
    }


# --- Verdict machinery (implements the pre-registered decision rule) ----------


def verdict_o1(delta: float, z: float) -> str:
    if delta >= _B1["pass"]["min_D1"] and z >= _B1["pass"]["min_Z"]:
        return "pass"
    if delta <= _B1["fail"]["max_D1"]:
        return "fail"
    return "inconclusive"


def verdict_traceability(n_below: int, n_pbar: int) -> str:
    if n_below >= _B_TR["fail"]["min_violations"]:
        return "fail"
    if n_pbar < _B_TR["pass"]["min_n_pbar"]:
        return "inconclusive"
    return "pass"


def verdict_signed_shift(effect: float, z: float) -> str:
    if effect > 0.0 and z >= _B_PS["pass"]["min_Z"]:
        return "pass"
    if effect < 0.0 and z >= _B_PS["fail"]["min_Z"]:
        return "fail"
    return "inconclusive"


def verdict_o2(components: dict[str, str]) -> str:
    if any(v == "fail" for v in components.values()):
        return "fail"
    if all(v == "pass" for v in components.values()):
        return "pass"
    return "inconclusive"


def _z_ratio(effect: float, se: float) -> float:
    if not math.isfinite(se) or se <= 0.0:
        if effect > 0.0:
            return float("inf")
        if effect < 0.0:
            return float("-inf")
        return 0.0
    return effect / se


# --- Pre-registration cross-check ---------------------------------------------


def cross_check_preregistration(prereg: dict, n_events: int, seed: int,
                                shuffle_realizations: int,
                                bootstrap_replicates: int) -> None:
    """Refuse to run when the file and the harness constants disagree."""
    problems: list[str] = []

    def check(name: str, expected, actual) -> None:
        if expected != actual:
            problems.append(f"{name}: pre-registration {expected!r} != harness {actual!r}")

    decision = prereg["calibration_set_decision_of_record"]
    check("m8_fitted_parameters",
          decision["m8_fitted_parameters"], M8_PARAMETER_SET)
    seeds = prereg["seed_protocol"]
    check("production_seed", seeds["m9_production_seed"], seed)
    check("shuffle_seed_base", seeds["shuffle_seed_base"], SHUFFLE_SEED_BASE)
    check("bootstrap_seed", seeds["bootstrap_seed"], BOOTSTRAP_SEED)
    check("stage_label", seeds["stage_label"], STAGE_LABEL)
    check("n_events", prereg["event_count"]["n_events"], n_events)
    check("shuffle_realizations",
          prereg["uncertainty_methods"]["shuffle_realizations"], shuffle_realizations)
    check("bootstrap_replicates",
          prereg["uncertainty_methods"]["bootstrap_replicates"], bootstrap_replicates)
    check("forward_xf_min", prereg["cuts"]["forward_xf_min"], FORWARD_XF_MIN)
    check("strange_baryon_pdg_ids",
          prereg["species"]["strange_baryon_pdg_ids"], list(STRANGE_BARYON_IDS))
    check("pair_threshold_gev", prereg["definitions"]["pair_threshold_gev"],
          PAIR_THRESHOLD_GEV)
    check("pre_registered_bands", prereg["pre_registered_bands"], PRE_REGISTERED_BANDS)
    if problems:
        raise SystemExit(
            "pre-registration mismatch; refusing to run (the pre-registered "
            "bands and protocol are authoritative):\n  " + "\n  ".join(problems)
        )


# --- Parameter set ------------------------------------------------------------


def build_parameter_set(config: dict):
    """Bundled-config parameter dataclasses with the ten M8 values applied."""
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.exchange_model import ReggeExcitationParameters, SimpleExcitationParameters
    from dataclasses import replace

    simple = _dataclass_from_dict(SimpleExcitationParameters, config.get("simple"))
    regge = _dataclass_from_dict(ReggeExcitationParameters, config.get("regge"))
    continuum = _dataclass_from_dict(ContinuumParameters, config.get("continuum"))
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = replace(regge, **{field_name: float(value)})
        else:
            continuum = replace(continuum, **{field_name: float(value)})
    applied = {
        "regge": {k: float(v) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if applied[group][field_name] != float(value):
            raise SystemExit(f"parameter application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).strip().upper() != "H3":
        raise SystemExit("the H3 decay kernel must apply as the dataclass default")
    return simple, regge, continuum, applied


def _dataclass_from_dict(cls, values: dict | None):
    return cls(**(values or {}))


# --- Generation (the M7/M8 path) ----------------------------------------------


def generate_events(n_events: int, seed: int, config_path: Path):
    """Generate the p+p sample via _generate_pp (regge mode, M8 parameters)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.staged_evaluation import _generate_pp

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    simple, regge, continuum, applied = build_parameter_set(config)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    t0 = time.perf_counter()
    events, summary = _generate_pp(
        n_events, seed, simple, regge, continuum, PRODUCTION_MODE, beam, STAGE_LABEL
    )
    wall = time.perf_counter() - t0
    max_res = 0.0
    for event in events:
        event.validate()  # re-assert conservation at 1e-10 (M7 convention)
        max_res = max(max_res, float(event.conservation_residuals()["relative_max"]))
    return events, summary, wall, applied, beam, config


# --- Fused statistics pass ------------------------------------------------------


def _one_pass(events, sqrt_s: float, pair_threshold_gev: float) -> dict:
    """Every observable statistic for one source assignment, in one sweep."""
    n_assoc = 0
    n_kfwd = 0
    co_counts: list[int] = []
    pair_sum = 0.0
    pair_count = 0
    parent_sum = 0.0
    parent_count = 0
    parent_below = 0
    parent_masses: list[float] = []
    pair_masses: list[float] = []
    per_event = {
        "n_pbar": [], "sum_parent": [], "has_pbar": [],
        "n_src": [], "sum_src": [], "n_pairs": [], "sum_pairs": [],
    }
    for ev in events:
        src = ev.src.tolist()
        # O1
        fwd = ev.idx_forward_kplus.tolist()
        if fwd:
            strange = ev.idx_strange_baryon.tolist()
            strange_src = [src[j] for j in strange]
            for i in fwd:
                n_kfwd += 1
                s = src[i]
                c = sum(1 for t in strange_src if t == s)
                co_counts.append(c)
                if c:
                    n_assoc += 1
        # O2b pairs and O2a parent masses
        p_here = 0
        ps_here = 0.0
        par_here = 0.0
        n_pb = len(ev.idx_pbar)
        if n_pb:
            pbar_idx = ev.idx_pbar.tolist()
            proton_idx = ev.idx_proton.tolist()
            for i in pbar_idx:
                si = src[i]
                mi = ev.mom[i]
                pm = float(ev.source_masses[si])
                parent_masses.append(pm)
                par_here += pm
                if pm < pair_threshold_gev:
                    parent_below += 1
                for j in proton_idx:
                    if src[j] == si:
                        m_pair = _pair_invariant_mass(mi, ev.mom[j])
                        pair_masses.append(m_pair)
                        ps_here += m_pair
                        p_here += 1
        pair_sum += ps_here
        pair_count += p_here
        parent_sum += par_here
        parent_count += n_pb
        per_event["n_pbar"].append(float(n_pb))
        per_event["sum_parent"].append(par_here)
        per_event["has_pbar"].append(1.0 if n_pb else 0.0)
        per_event["n_src"].append(float(ev.source_masses.size))
        per_event["sum_src"].append(float(ev.source_masses.sum()))
        per_event["n_pairs"].append(float(p_here))
        per_event["sum_pairs"].append(ps_here)
    return {
        "n_assoc": n_assoc,
        "n_kfwd": n_kfwd,
        "co_counts": co_counts,
        "pair_sum": pair_sum,
        "pair_count": pair_count,
        "parent_sum": parent_sum,
        "parent_count": parent_count,
        "parent_below": parent_below,
        "parent_masses": parent_masses,
        "pair_masses": pair_masses,
        "per_event": {k: np.asarray(v, dtype=np.float64) for k, v in per_event.items()},
    }


# --- Main -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100000,
                        help="pre-registered production sample size (100000)")
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED,
                        help="pre-registered production seed (20310831)")
    parser.add_argument("--shuffle-realizations", type=int, default=SHUFFLE_REALIZATIONS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--out-dir", default=str(RUNS_ROOT / RUN_DESCRIPTION))
    parser.add_argument("--config", default="",
                        help="staged excitation JSON config "
                             "(default: <mixing2006>/config/staged_excitation.json)")
    parser.add_argument("--preregistration", default="",
                        help="pre-registration JSON (default: <out-dir>/pre_registration.json)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else MIXING2006_DIR / "config" / "staged_excitation.json"
    prereg_path = Path(args.preregistration) if args.preregistration else out_dir / "pre_registration.json"
    if not prereg_path.exists():
        raise SystemExit(
            f"pre-registration not found at {prereg_path}; the bands must be "
            "pre-registered before the events exist"
        )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    cross_check_preregistration(
        prereg, args.events, args.seed, args.shuffle_realizations, args.bootstrap_replicates
    )
    prereg_sha = _sha256_file(prereg_path)

    t_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    configuration_hash = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
    m8_record = RUNS_ROOT / "staged_binned_fit_mc_2026-09-06" / "staged_evaluation_report.json"

    print(f"[prereg] {prereg_path} (sha256 {prereg_sha[:12]}...): constants verified", flush=True)

    events, summary, gen_wall, applied, beam, _cfg = generate_events(
        args.events, args.seed, config_path
    )
    print(
        f"[gen] {len(events)} events in {gen_wall:.1f}s "
        f"({1000.0 * gen_wall / max(len(events), 1):.2f} ms/event); "
        f"max residual {summary['max_conservation_residual']:.2e}",
        flush=True,
    )

    t0 = time.perf_counter()
    heldout = heldout_events_from_excitation(events, SQRT_S_GEV)
    conv_wall = time.perf_counter() - t0
    n_events = len(heldout)

    # Family census (provenance; collected during conversion of the sample).
    family_census: dict[str, int] = {}
    fwd_kplus_family: dict[str, int] = {}
    for event in events:
        for p in event.final_particles:
            fam = p.channel_family
            family_census[fam] = family_census.get(fam, 0) + 1
            if p.pdg_id == K_PLUS_PDG:
                xf = 2.0 * p.p4.pz / SQRT_S_GEV
                if xf >= FORWARD_XF_MIN:
                    fwd_kplus_family[fam] = fwd_kplus_family.get(fam, 0) + 1

    # --- Real sample statistics -------------------------------------------
    t0 = time.perf_counter()
    real = _one_pass(heldout, SQRT_S_GEV, PAIR_THRESHOLD_GEV)
    real_wall = time.perf_counter() - t0

    n_kfwd = real["n_kfwd"]
    n_assoc = real["n_assoc"]
    f1 = n_assoc / n_kfwd if n_kfwd else float("nan")
    n_pbar = int(real["parent_count"])
    n_below = real["parent_below"]
    f_tr = 1.0 - n_below / n_pbar if n_pbar else float("nan")
    real_pair_mean = real["pair_sum"] / real["pair_count"] if real["pair_count"] else float("nan")
    all_masses = all_source_masses(heldout)
    cond_masses = pbar_event_source_masses(heldout)
    mean_all_mass = float(all_masses.mean()) if all_masses.size else float("nan")
    mean_cond_mass = float(cond_masses.mean()) if cond_masses.size else float("nan")
    mean_parent_mass = real["parent_sum"] / n_pbar if n_pbar else float("nan")

    # --- Shuffle realizations ----------------------------------------------
    t0 = time.perf_counter()
    base_f1: list[float] = []
    base_pair_sum = 0.0
    base_pair_count = 0
    base_parent_sum = 0.0
    base_parent_count = 0
    per_real = []
    for r in range(args.shuffle_realizations):
        rng = np.random.default_rng(SHUFFLE_SEED_BASE + r)
        shuffled = [ev.with_shuffled_sources(rng) for ev in heldout]
        stat = _one_pass(shuffled, SQRT_S_GEV, PAIR_THRESHOLD_GEV)
        f_r = stat["n_assoc"] / stat["n_kfwd"] if stat["n_kfwd"] else float("nan")
        base_f1.append(f_r)
        base_pair_sum += stat["pair_sum"]
        base_pair_count += stat["pair_count"]
        base_parent_sum += stat["parent_sum"]
        base_parent_count += stat["parent_count"]
        per_real.append({
            "realization": r,
            "seed": SHUFFLE_SEED_BASE + r,
            "n_kfwd": stat["n_kfwd"],
            "n_assoc": stat["n_assoc"],
            "f1": f_r,
            "n_pairs": stat["pair_count"],
            "pair_mean": stat["pair_sum"] / stat["pair_count"] if stat["pair_count"] else None,
            "parent_mean": stat["parent_sum"] / stat["parent_count"] if stat["parent_count"] else None,
        })
    shuffle_wall = time.perf_counter() - t0
    f1_base = float(np.nanmean(base_f1))
    se_f1_base = float(np.nanstd(base_f1, ddof=1)) if len(base_f1) > 1 else 0.0
    pair_base_mean = base_pair_sum / base_pair_count if base_pair_count else float("nan")
    parent_base_mean = base_parent_sum / base_parent_count if base_parent_count else float("nan")

    # --- Bootstrap (one generator, pre-registered consumption order) --------
    t0 = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pe = real["per_event"]
    pair_mean_boot, invalid_pairs = _bootstrap_mean(
        pe["sum_pairs"], pe["n_pairs"], args.bootstrap_replicates, rng
    )
    dm_parent_boot, invalid_parent = _bootstrap_mean_shift(
        pe["sum_parent"], pe["n_pbar"], pe["sum_src"], pe["n_src"],
        args.bootstrap_replicates, rng,
    )
    cond_sum = pe["sum_src"] * pe["has_pbar"]
    cond_cnt = pe["n_src"] * pe["has_pbar"]
    dm_event_boot, invalid_event = _bootstrap_mean_shift(
        cond_sum, cond_cnt, pe["sum_src"], pe["n_src"],
        args.bootstrap_replicates, rng,
    )
    boot_wall = time.perf_counter() - t0

    se_pair_boot = _stat(pair_mean_boot)["std_ddof1"]
    se_dm_parent = _stat(dm_parent_boot)["std_ddof1"]
    se_dm_event = _stat(dm_event_boot)["std_ddof1"]
    bootstrap_invalid = {
        "pair_mean_replicates": invalid_pairs,
        "parent_mass_shift_replicates": invalid_parent,
        "event_conditioned_mass_shift_replicates": invalid_event,
    }

    # --- Effect sizes, significances, verdicts ------------------------------
    d1 = f1 - f1_base
    se1 = wald_se(f1, n_kfwd)
    z1 = _z_ratio(d1, math.sqrt(se1 * se1 + se_f1_base * se_f1_base))
    v1 = verdict_o1(d1, z1)

    dm_parent = mean_parent_mass - mean_all_mass
    z_dm_parent = _z_ratio(dm_parent, se_dm_parent)
    v_dm_parent = verdict_signed_shift(dm_parent, z_dm_parent)

    dm_event = mean_cond_mass - mean_all_mass
    z_dm_event = _z_ratio(dm_event, se_dm_event)
    v_dm_event = verdict_signed_shift(dm_event, z_dm_event)

    d2 = pair_base_mean - real_pair_mean
    se_pair_base = _stat(np.asarray(
        [r["pair_mean"] if r["pair_mean"] is not None else np.nan for r in per_real],
        dtype=float,
    ))["std_ddof1"] if len(per_real) > 1 else 0.0
    z2 = _z_ratio(d2, math.sqrt(se_pair_boot * se_pair_boot + se_pair_base * se_pair_base))
    v_pair = verdict_signed_shift(d2, z2)

    v_trace = verdict_traceability(n_below, n_pbar)
    o2_components = {
        "traceability": v_trace,
        "parent_mass_shift": v_dm_parent,
        "event_conditioned_mass_shift": v_dm_event,
        "pair_structure": v_pair,
    }
    v2 = verdict_o2(o2_components)

    wil1 = wilson_interval(n_assoc, n_kfwd)
    wil_tr = wilson_interval(n_pbar - n_below, n_pbar)
    rev_with, rev_total = reverse_association_fraction(heldout)
    role_counts, _role_total = association_by_source_role(heldout)

    report = {
        "schema": f"staged_heldout_correlations/v{ARTIFACT_SCHEMA_VERSION}",
        "card": CARD,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_description": RUN_DESCRIPTION,
        "date_utc": _utc_now_iso(),
        "pre_registration": {
            "path": str(prereg_path),
            "sha256": prereg_sha,
            "cross_check": "all pre-registered constants verified against the harness before generation",
        },
        "calibration_set_decision_of_record": {
            "verbatim": prereg["calibration_set_decision_of_record"]["verbatim"],
            "m8_fitted_parameters": dict(M8_PARAMETER_SET),
            "source": prereg["calibration_set_decision_of_record"]["source"],
        },
        "provenance": {
            "outer_head": _git_head(MIXING2006_DIR.parents[1]),
            "nested_head": _git_head(MIXING2006_DIR),
            "config_path": str(config_path),
            "configuration_hash": configuration_hash,
            "m8_record": str(m8_record),
            "m8_record_sha256": _sha256_file(m8_record) if m8_record.exists() else None,
        },
        "parameter_set": {
            "m8_fitted": dict(M8_PARAMETER_SET),
            "effective": applied,
            "beam_momentum_gev_c": beam,
            "sqrt_s_gev": SQRT_S_GEV,
        },
        "seed_protocol": {
            "production_seed": int(args.seed),
            "shuffle_seed_base": SHUFFLE_SEED_BASE,
            "shuffle_realization_seeds": [SHUFFLE_SEED_BASE + r for r in range(args.shuffle_realizations)],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "stage_label": STAGE_LABEL,
        },
        "generation": {
            "n_events_requested": int(args.events),
            "n_events": n_events,
            "wall_s": gen_wall,
            "ms_per_event": 1000.0 * gen_wall / max(n_events, 1),
            "conversion_wall_s": conv_wall,
            "summary": summary,
            "conservation": {
                "n_events_validated": n_events,
                "tolerance": 1e-10,
                "max_relative_residual": summary["max_conservation_residual"],
            },
            "family_census": family_census,
            "forward_kplus_family_census": fwd_kplus_family,
        },
        "pre_registered_bands": _to_jsonable(PRE_REGISTERED_BANDS),
        "O1_klambda_association": {
            "n_numerator": n_assoc,
            "n_denominator": n_kfwd,
            "fraction": f1,
            "wilson_95": list(wil1),
            "wald_se": se1,
            "baseline_mean": f1_base,
            "baseline_per_realization": base_f1,
            "baseline_se": se_f1_base,
            "delta": d1,
            "z": z1,
            "verdict": v1,
            "diagnostics": {
                "reverse_association": {
                    "n_with_forward_kplus": rev_with,
                    "n_strange_baryons": rev_total,
                },
                "association_by_source_role": role_counts,
                "mean_co_strange_baryons_per_forward_kplus": (
                    float(np.mean(real["co_counts"])) if real["co_counts"] else None
                ),
            },
        },
        "O2_ppbar_traceability": {
            "traceability": {
                "n_pbar": n_pbar,
                "n_below_threshold": n_below,
                "fraction": f_tr,
                "wilson_95": list(wil_tr),
                "min_parent_mass": float(min(real["parent_masses"])) if n_pbar else None,
                "max_parent_mass": float(max(real["parent_masses"])) if n_pbar else None,
                "verdict": v_trace,
            },
            "parent_mass_shift": {
                "mean_parent_mass": mean_parent_mass,
                "mean_unconditional_mass": mean_all_mass,
                "delta": dm_parent,
                "se_bootstrap": se_dm_parent,
                "bootstrap": _stat(dm_parent_boot),
                "z": z_dm_parent,
                "verdict": v_dm_parent,
                "shuffle_baseline_parent_mean": parent_base_mean,
            },
            "event_conditioned_mass_shift": {
                "mean_conditioned_mass": mean_cond_mass,
                "mean_unconditional_mass": mean_all_mass,
                "delta": dm_event,
                "se_bootstrap": se_dm_event,
                "bootstrap": _stat(dm_event_boot),
                "z": z_dm_event,
                "verdict": v_dm_event,
            },
            "pair_structure": {
                "n_pairs_real": real["pair_count"],
                "mean_real": real_pair_mean,
                "n_pairs_baseline_mean": base_pair_count / max(len(per_real), 1),
                "mean_baseline": pair_base_mean,
                "baseline_se": se_pair_base,
                "se_bootstrap_real": se_pair_boot,
                "delta": d2,
                "z": z2,
                "verdict": v_pair,
            },
            "verdict": v2,
            "components": o2_components,
        },
        "distributions": {
            "unconditional_source_mass": _histogram(all_masses, SOURCE_MASS_EDGES),
            "pbar_parent_source_mass": _histogram(
                np.asarray(real["parent_masses"], dtype=np.float64), SOURCE_MASS_EDGES
            ),
            "pbar_event_source_mass": _histogram(cond_masses, SOURCE_MASS_EDGES),
            "same_source_ppbar_invariant_mass": _histogram(
                np.asarray(real["pair_masses"], dtype=np.float64), PAIR_MASS_EDGES
            ),
        },
        "wall_time": {
            "generation_s": gen_wall,
            "conversion_s": conv_wall,
            "real_pass_s": real_wall,
            "shuffle_realizations_s": shuffle_wall,
            "bootstrap_s": boot_wall,
            "total_s": time.perf_counter() - t_start,
        },
        "bootstrap_invalid_replicates": bootstrap_invalid,
        "falsifier_policy": (
            "the pre-registered bands decide pass/fail/inconclusive; a fail is a "
            "legitimate recorded result, not a milestone failure; the bands were "
            "not tuned after seeing the run"
        ),
        "deviations": [],
        "finished_utc": _utc_now_iso(),
    }

    artifact = out_dir / ARTIFACT_NAME
    _write_json(artifact, report)
    print(f"[o1] F1={f1:.4f} ({n_assoc}/{n_kfwd}) baseline={f1_base:.4f}+-{se_f1_base:.4f} "
          f"D1={d1:.4f} Z1={z1:.2f} -> {v1}", flush=True)
    print(f"[o2] traceability {n_below}/{n_pbar} below threshold (F_tr={f_tr:.4f}) -> {v_trace}; "
          f"dM_parent={dm_parent:.4f}+-{se_dm_parent:.4f} GeV; "
          f"dM_event={dm_event:.4f}+-{se_dm_event:.4f} GeV; "
          f"pair mean real={real_pair_mean:.4f} baseline={pair_base_mean:.4f} "
          f"D2={d2:.4f} Z2={z2:.2f} -> {v_pair}; O2 verdict {v2}", flush=True)
    print(f"[done] artifact: {artifact} (total wall {report['wall_time']['total_s'] / 60.0:.1f} min)", flush=True)
    return 0


def _histogram(values: np.ndarray, edges: np.ndarray) -> dict:
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist(),
            "n": int(values.size), "mean": float(values.mean()) if values.size else None}


if __name__ == "__main__":
    raise SystemExit(main())
