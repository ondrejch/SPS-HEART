"""The shared run contract of the staged benchmark runs (review record
``.collab/reviews/fable-speedup-2026-09-28.md``, code fix C2).

The Round 3 to 5 run drivers each carried their own copy of the run
mechanics: root location, refusal, the identity and provenance blocks,
the frozen-substrate verification, JSON and digest utilities, the
path-sorted checksum inventory, the seal manifest, and the replica
generation loop.  This module holds those mechanics once so that a run
driver states only its physics: the pre-registration constants, the
per-event record, the aggregates, and the gate predicates.  The formats
reproduce the sealed drivers' conventions (``pre_registration.json``,
``provenance.json``, ``REPORT.md``, ``SHA256SUMS.txt`` in ``sha256sum -c``
form, ``seal_manifest.json``, ``clean_environment_commands.json``).

Layout: the module runs from the nested tree
(``dev/mixing2006/helpers/benchmarking/``) or its monorepo mirror
(``helpers/benchmarking/``); the two copies are byte-identical and locate
the trees from their own path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, NoReturn, Sequence

import numpy as np

__all__ = [
    "RunContract",
    "locate_roots",
    "refuse",
    "now_utc",
    "write_json",
    "read_json",
    "canonical_digest",
    "sha256_file",
    "git_info",
    "environment_fingerprint",
    "verify_frozen_substrate",
    "seed_ledger",
    "check_seed_disjointness",
    "run_replicas",
    "write_sha256sums",
    "verify_sha256sums",
    "histogram",
    "stats",
    "pooled_mean_and_se",
]

PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
REPORT_NAME = "report.json"
REPORT_MD_NAME = "REPORT.md"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SEAL_NAME = "seal_manifest.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEED_LEDGER_REL = Path("config") / "v2_singlet" / "seed_ledger.json"


# ----------------------------------------------------------------------
# Roots, refusal, time, JSON, digests
# ----------------------------------------------------------------------
def locate_roots(anchor: Path | None = None) -> tuple[Path, Path]:
    """``(nested_root, mono_root)`` from the location of *anchor* (a
    driver file; default this module): the nested tree holds the
    ``na49_mixing`` package, the monorepo root is two levels above a
    ``dev/mixing2006`` nested tree, else the nested tree itself."""
    here = Path(anchor or __file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        refuse(f"cannot locate the na49_mixing package next to {here}")
    if not (nested / "config").is_dir() and (nested / "na49_mixing" / "_data" / "config").is_dir():
        # an installed wheel: the configuration is bundled inside the package,
        # and the install directory holds both na49_mixing and helpers
        return (nested / "na49_mixing" / "_data").resolve(), nested
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


def refuse(message: str, *, prefix: str = "run_contract") -> NoReturn:
    """Print a refusal to stderr and exit with code 2 before the step."""
    print(f"{prefix}: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    """Write *payload* as JSON atomically: a temporary file in the same
    directory, then a rename, so a concurrent reader (the gateway
    orchestrator's remote ``cat``) never sees a partial record
    (2026-10-02, external review q1 M7).  The bytes are those of the
    former direct write."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_digest(payload: Any) -> str:
    """The sha256 of the canonical JSON (sorted keys, compact separators)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(root: Path) -> dict:
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=30
        ).stdout
        return {
            "head": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(run("status", "--porcelain").strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "branch": None, "dirty": None}


def environment_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


# ----------------------------------------------------------------------
# Frozen substrate and seeds
# ----------------------------------------------------------------------
SEED_LEDGER_MONO_REL = "dev/mixing2006/config/v2_singlet/seed_ledger.json"


def verify_frozen_substrate(mono_root: Path, frozen: dict, *, prefix: str, skip: Iterable[str] = ()) -> dict:
    """Verify ``{key: (relative path, sha256)}`` against the working tree;
    refuse on any drift (the run never touches a drifted surface).  Paths in
    *skip* are reported as skipped instead of verified: the seed ledger is
    append-only bookkeeping whose later reservations must not invalidate an
    earlier run, so a run verifies its own ledger entry through
    :func:`check_seed_disjointness` instead of the whole file."""
    status: dict = {}
    skip = set(skip)
    for key, (rel, expected) in frozen.items():
        if rel in skip or key in skip:
            status[key] = {"sha256": expected, "verified": False, "skipped": "append-only bookkeeping; own entry checked by check_seed_disjointness"}
            continue
        path = Path(mono_root) / rel
        if not path.is_file():
            refuse(f"frozen substrate file missing: {path}", prefix=prefix)
        actual = sha256_file(path)
        if actual != expected:
            refuse(
                f"frozen substrate digest drift for {key} ({rel}): expected "
                f"{expected}, actual {actual}",
                prefix=prefix,
            )
        status[key] = {"sha256": actual, "verified": True}
    return status


#: The project source roots whose loaded modules a run freezes (relative to
#: the monorepo root).
PROJECT_SOURCE_ROOTS = ("dev/mixing2006/na49_mixing/", "helpers/benchmarking/")


def loaded_module_rels(mono_root: Path, roots: Sequence[str] = PROJECT_SOURCE_ROOTS) -> list[str]:
    """The monorepo-relative paths of every loaded module under the project
    source roots (2026-10-02, external review q1 H1: the hand-written
    FROZEN_RELS lists of R51i to R69 omit 33 of the 53 modules a driver
    loads, among them ``rng_substreams``).  A driver freezes
    ``FROZEN_RELS`` together with this list at pre-registration, so a
    change to any module the generation path imports refuses the run."""
    mono_root = Path(mono_root).resolve()
    rels = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            rel = Path(path).resolve().relative_to(mono_root).as_posix()
        except ValueError:
            continue
        if rel.endswith(".py") and any(rel.startswith(root) for root in roots):
            rels.add(rel)
    return sorted(rels)


def bounded_hessian(value: Callable[[np.ndarray], float], p0: Sequence[float], steps: Sequence[float],
                    lower: Sequence[float], upper: Sequence[float], names: Sequence[str]) -> dict:
    """The finite-difference Hessian of *value* at *p0* inside a box
    (2026-10-02, external review q1 M3: a central probe past a bound
    returned inf and was reported as a curvature failure).

    A parameter whose central probe would leave ``[lower, upper]`` takes a
    one-sided difference away from the bound (``bound_limited`` names it);
    a parameter with no room on either side shrinks its step to half the
    larger distance to a bound.  The record carries the matrix, its
    eigenvalues, ``positive_definite``, the errors ``sqrt(diag(2 H^-1))``
    when positive definite, and ``non_finite`` when a probe returned a
    non-finite value."""
    p0 = np.asarray(p0, dtype=float)
    h = np.asarray(steps, dtype=float).copy()
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    n = p0.size
    sign = np.zeros(n)  # 0 central, +1 forward, -1 backward
    for i in range(n):
        down, up = p0[i] - lo[i], hi[i] - p0[i]
        if down >= h[i] and up >= h[i]:
            continue
        if up >= 2.0 * h[i]:
            sign[i] = 1.0
        elif down >= 2.0 * h[i]:
            sign[i] = -1.0
        else:
            sign[i] = 1.0 if up >= down else -1.0
            h[i] = 0.5 * max(up, down)
    cache: dict = {}

    def f(offsets):
        key = tuple(np.round(offsets, 15))
        if key not in cache:
            cache[key] = float(value(p0 + np.asarray(offsets)))
        return cache[key]

    def e(i, k):
        v = np.zeros(n); v[i] = k * h[i]
        return v

    zero = np.zeros(n)
    f0 = f(zero)
    H = np.zeros((n, n))
    for i in range(n):
        if sign[i] == 0.0:
            H[i, i] = (f(e(i, 1)) - 2.0 * f0 + f(e(i, -1))) / h[i] ** 2
        else:
            s = sign[i]
            H[i, i] = (f(e(i, 2 * s)) - 2.0 * f(e(i, s)) + f0) / h[i] ** 2
        for j in range(i + 1, n):
            if sign[i] == 0.0 and sign[j] == 0.0:
                H[i, j] = (f(e(i, 1) + e(j, 1)) - f(e(i, 1) + e(j, -1)) - f(e(i, -1) + e(j, 1)) + f(e(i, -1) + e(j, -1))) / (4.0 * h[i] * h[j])
            else:
                si = sign[i] if sign[i] != 0.0 else 1.0
                sj = sign[j] if sign[j] != 0.0 else 1.0
                H[i, j] = (f(e(i, si) + e(j, sj)) - f(e(i, si)) - f(e(j, sj)) + f0) / (si * sj * h[i] * h[j])
            H[j, i] = H[i, j]
    finite = bool(np.all(np.isfinite(H)))
    eig = np.linalg.eigvalsh(H) if finite else np.full(n, np.nan)
    pd = bool(finite and np.all(eig > 0.0))
    errors = None
    if pd:
        cov = 2.0 * np.linalg.inv(H)
        errors = {name: float(math.sqrt(max(cov[k, k], 0.0))) for k, name in enumerate(names)}
    return {"names": list(names), "steps": [float(v) for v in h],
            "directions": ["central" if s == 0.0 else ("forward" if s > 0 else "backward") for s in sign],
            "bound_limited": [name for name, s in zip(names, sign) if s != 0.0],
            "matrix": H.tolist() if finite else None, "eigenvalues": [float(v) for v in eig] if finite else None,
            "positive_definite": pd, "quoted": pd, "errors": errors, "non_finite": not finite}


def seed_ledger(nested_root: Path) -> dict:
    """The committed ledger of consumed raw seed blocks and event-index
    partitions (``config/v2_singlet/seed_ledger.json``)."""
    return read_json(Path(nested_root) / SEED_LEDGER_REL)


def check_seed_disjointness(ledger: dict, run_id: str, raw_block: tuple[int, int],
                            partition_base: int) -> dict:
    """The run's raw block ``(first, last)`` and event-index partition must
    be disjoint from every block of every other run in the ledger; the
    run's own reservation (if listed) must match exactly."""
    first, last = int(raw_block[0]), int(raw_block[1])
    overlaps = []
    for entry in ledger["blocks"]:
        if entry["run"] == run_id:
            if (int(entry["raw_first"]), int(entry["raw_last"])) != (first, last) or (
                entry.get("partition_base") is not None
                and int(entry["partition_base"]) != int(partition_base)
            ):
                refuse(
                    f"the ledger reservation of {run_id} differs from the driver "
                    f"constants: {entry} vs {(first, last, partition_base)}"
                )
            continue
        if not (last < int(entry["raw_first"]) or first > int(entry["raw_last"])):
            overlaps.append(entry["run"])
        if entry.get("partition_base") is not None and int(entry["partition_base"]) == int(partition_base):
            overlaps.append(entry["run"] + " (partition)")
    if overlaps:
        refuse(f"the seed block {raw_block} of {run_id} overlaps {overlaps}")
    return {"raw_block": [first, last], "partition_base": int(partition_base),
            "checked_against": len(ledger["blocks"]), "disjoint": True}


# ----------------------------------------------------------------------
# Replica generation (one process per replica)
# ----------------------------------------------------------------------
def run_replicas(worker: Callable, jobs: Sequence[tuple], *, max_workers: int | None = None) -> list:
    """Run ``worker(*job)`` for every job in a process pool, returning the
    results in job order.  The worker must be a module-level function of
    picklable arguments; each replica is deterministic in its own seed."""
    n = max_workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    if n == 1 or len(jobs) == 1:
        return [worker(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n) as pool:
        return list(pool.map(worker, *zip(*jobs)))


# ----------------------------------------------------------------------
# Checksums and seal
# ----------------------------------------------------------------------
def write_sha256sums(run_dir: Path) -> int:
    """The strict path-sorted checksum inventory in ``sha256sum -c`` form
    (run-dir-relative POSIX paths, LF endings)."""
    run_dir = Path(run_dir)
    names = sorted(
        str(p.relative_to(run_dir).as_posix())
        for p in run_dir.rglob("*") if p.is_file() and p.name != SHA256SUMS_NAME
    )
    lines = [f"{sha256_file(run_dir / name)}  {name}" for name in names]
    (run_dir / SHA256SUMS_NAME).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return len(names)


def verify_sha256sums(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    entries = 0
    mismatches = []
    for line in (run_dir / SHA256SUMS_NAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        entries += 1
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            mismatches.append(name)
    return {"entries": entries, "mismatches": mismatches}


# ----------------------------------------------------------------------
# Aggregates
# ----------------------------------------------------------------------
def histogram(values: Iterable[float], lo: float, hi: float, n_bins: int) -> dict:
    arr = np.asarray(list(values), dtype=float)
    counts, edges = np.histogram(arr, bins=int(n_bins), range=(float(lo), float(hi)))
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "underflow": int(np.sum(arr < lo)),
        "overflow": int(np.sum(arr >= hi)),
        "n": int(arr.size),
    }


def stats(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pooled_mean_and_se(replica_means: Sequence[float]) -> dict:
    """The pooled mean of replica means and its standard error from the
    replica-to-replica scatter (the house pooled-SE convention)."""
    arr = np.asarray(list(replica_means), dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"replicas": n, "mean": mean, "se": se}


# ----------------------------------------------------------------------
# The run contract object
# ----------------------------------------------------------------------
class RunContract:
    """One run's identity, directory layout, and step bookkeeping.

    ``run_id`` names the run directory under ``dev/00runs``; ``card`` is
    the task-card identifier; ``prefix`` labels refusals.  The step
    methods refuse out-of-order execution: pre-registration first, then
    generation, evaluation, gate, one clean validation, and the seal, each
    performed by exactly one invocation.
    """

    def __init__(self, run_id: str, card: str, *, driver_file: Path, schema_prefix: str,
                 run_dir: Path | None = None):
        self.run_id = run_id
        self.card = card
        self.schema_prefix = schema_prefix
        self.nested_root, self.mono_root = locate_roots(driver_file)
        self.driver_file = Path(driver_file).resolve()
        self.run_dir = Path(run_dir) if run_dir else (self.mono_root / "dev" / "00runs" / run_id)
        self.prefix = Path(driver_file).stem

    # -- paths ---------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.run_dir / name

    def refuse(self, message: str) -> NoReturn:
        refuse(message, prefix=self.prefix)

    def require(self, *names: str) -> None:
        for name in names:
            if not self.path(name).is_file():
                self.refuse(f"the run dir is missing {name}: {self.run_dir}")

    def forbid(self, *names: str) -> None:
        for name in names:
            if self.path(name).is_file():
                self.refuse(f"{name} already exists in {self.run_dir}; each step runs once")

    # -- identity ------------------------------------------------------
    def identity(self) -> dict:
        return {
            "run": self.run_id,
            "card": self.card,
            "driver": str(self.driver_file.relative_to(self.mono_root)),
            "driver_sha256": sha256_file(self.driver_file),
            "run_dir": str(self.run_dir),
        }

    def provenance(self, frozen_status: dict, extra: dict | None = None) -> dict:
        payload = {
            "schema": f"{self.schema_prefix}.provenance/1",
            **self.identity(),
            "git": {
                "outer": git_info(self.mono_root),
                "nested": git_info(self.nested_root),
            },
            "environment": environment_fingerprint(),
            "frozen_substrate": frozen_status,
            "written_utc": now_utc(),
        }
        if extra:
            payload.update(extra)
        return payload

    # -- steps ---------------------------------------------------------
    def write_pre_registration(self, payload: dict, frozen_status: dict, extra_provenance: dict | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.forbid(PREREGISTRATION_NAME, PROVENANCE_NAME)
        payload = {"schema": f"{self.schema_prefix}.pre_registration/1", **self.identity(), **payload,
                   "written_utc": now_utc()}
        write_json(self.path(PREREGISTRATION_NAME), payload)
        write_json(self.path(PROVENANCE_NAME), self.provenance(frozen_status, extra_provenance))
        print(f"[{self.run_id}] pre-registration written: {self.run_dir}", flush=True)

    def load_pre_registration(self) -> dict:
        self.require(PREREGISTRATION_NAME)
        return read_json(self.path(PREREGISTRATION_NAME))

    def write_report(self, report: dict, markdown: str) -> dict:
        body = {k: v for k, v in report.items() if k != "digest"}
        body["schema"] = f"{self.schema_prefix}.report/1"
        body.update(self.identity())
        body["digest"] = canonical_digest(body)
        write_json(self.path(REPORT_NAME), body)
        self.path(REPORT_MD_NAME).write_text(markdown, encoding="utf-8")
        return body

    def report_digest_of(self, report: dict) -> str:
        body = {k: v for k, v in report.items() if k != "digest"}
        return canonical_digest(body)

    def write_clean_validation(self, payload: dict) -> None:
        self.forbid(CLEAN_ENV_NAME)
        payload = {"schema": f"{self.schema_prefix}.clean_env/1", **self.identity(), **payload,
                   "verified_utc": now_utc()}
        write_json(self.path(CLEAN_ENV_NAME), payload)

    def seal(self, required: Sequence[str], record_prefix: str) -> dict:
        self.forbid(SEAL_NAME)
        self.require(PREREGISTRATION_NAME, PROVENANCE_NAME, REPORT_NAME, REPORT_MD_NAME, CLEAN_ENV_NAME, *required)
        n_entries = write_sha256sums(self.run_dir)
        files = sorted(
            str(p.relative_to(self.run_dir).as_posix())
            for p in self.run_dir.rglob("*")
            if p.is_file() and p.name not in (SEAL_NAME, SHA256SUMS_NAME)
        )
        manifest = {
            "schema": f"{self.schema_prefix}.seal/1",
            **self.identity(),
            "files": files,
            "sha256sums_entries": n_entries,
            "record_count": sum(1 for name in files if name.startswith(record_prefix)),
            "sealed_utc": now_utc(),
            "git_state": git_info(self.mono_root),
            "environment": environment_fingerprint(),
            "clean_validation_record": CLEAN_ENV_NAME,
        }
        write_json(self.path(SEAL_NAME), manifest)
        # the seal manifest joins the inventory
        n_entries = write_sha256sums(self.run_dir)
        manifest["sha256sums_entries"] = n_entries
        write_json(self.path(SEAL_NAME), manifest)
        write_sha256sums(self.run_dir)
        print(f"[{self.run_id}] sealed {self.run_dir} ({n_entries} entries)", flush=True)
        return manifest
