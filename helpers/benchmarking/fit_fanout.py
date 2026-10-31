"""Fan-out of independent fit jobs over gateway jobs (2026-10-02).

The staged joint-fit drivers run their optimizer starts, profile points
and variants inside one 32-task gateway job, so eight of the nine workers
idle while a fit runs (R69: 3.5 of 4.3 hours on one worker).  This module
lets a driver step hand each independent job of a batch to its own gateway
job, with the orchestrator in between:

1. The step calls :meth:`FanOut.map` with a registered worker name and a
   list of JSON-serializable job argument lists.  In ``local`` mode the
   jobs run in this process's pool (``run_contract.run_replicas``), as
   before.  In ``gateway`` mode the batch is content-addressed: when every
   result file exists the results are read back; otherwise the request is
   written and :class:`FanOutPending` is raised.
2. The step entry point catches :class:`FanOutPending`, writes the pending
   record and exits; the orchestrator submits one job per missing result
   (``--fanout-task NAMESPACE BATCH INDEX``), waits, and runs the step
   again.  A re-entered step finds the finished batches on disk and
   proceeds to the next batch.

Both modes pass every result through the same JSON codec, so the values a
step sees are identical in the two modes (floats round-trip exactly).  A
namespace keeps re-validation honest: the clean re-evaluation uses its
own namespace, so it recomputes every fit instead of reading the
originals.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import run_contract as RC  # noqa: E402

SCHEMA = "na49_mixing.fit_fanout"
MODES = ("local", "gateway")
REQUEST_FMT = "fanout_{ns}_{batch}_request.json"
RESULT_FMT = "fanout_{ns}_{batch}_{index}.json"
PENDING_FMT = "fanout_pending_{key}.json"


class FanOutPending(Exception):
    """Raised by a step whose batches have missing results; carries
    ``[(namespace, batch, [missing indices]), ...]``."""

    def __init__(self, batches: Iterable[tuple]):
        self.batches = [(str(ns), str(b), [int(i) for i in idx]) for ns, b, idx in batches]
        super().__init__(f"{sum(len(i) for _, _, i in self.batches)} fan-out jobs pending in {len(self.batches)} batches")


# ----------------------------------------------------------------------
# The JSON codec
# ----------------------------------------------------------------------
def encode(value):
    """A JSON-safe copy of *value*: arrays carry their dtype and shape,
    numpy scalars become Python scalars, tuples become lists."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, float) or isinstance(value, (int, str, bool)) or value is None:
        return value
    raise TypeError(f"fit_fanout cannot encode {type(value).__name__}")


def decode(value):
    """The inverse of :func:`encode`."""
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "dtype", "shape"}:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"]).reshape(value["shape"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def canonical(value) -> str:
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"))


def batch_id(worker: str, jobs: list) -> str:
    return hashlib.sha256(canonical({"worker": worker, "jobs": jobs}).encode("utf-8")).hexdigest()[:20]


def job_digest(worker: str, job) -> str:
    return hashlib.sha256(canonical({"worker": worker, "job": job}).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------
class FanOut:
    """The batch dispatcher of one driver step.

    *registry* maps worker names to module-level callables; *mode* is
    ``local`` or ``gateway``; *namespace* separates independent sets of
    batches in one run dir (``main`` for the run, ``clean`` for the
    clean re-evaluation)."""

    def __init__(self, run_dir, registry: Mapping[str, Callable], *, mode: str = "local", namespace: str = "main"):
        if mode not in MODES:
            raise ValueError(f"fan-out mode {mode!r} is not one of {MODES}")
        if not namespace.isidentifier():
            raise ValueError(f"fan-out namespace {namespace!r} must be an identifier")
        self.run_dir = Path(run_dir)
        self.registry = dict(registry)
        self.mode = mode
        self.namespace = namespace

    def _path(self, fmt: str, **kw) -> Path:
        return self.run_dir / fmt.format(ns=self.namespace, **kw)

    def map(self, worker: str, jobs: list, *, local_workers: int | None = None) -> list:
        """Run *worker* on every job (an argument list) and return the
        decoded results in job order, or raise :class:`FanOutPending`."""
        if worker not in self.registry:
            raise KeyError(f"fan-out worker {worker!r} is not registered")
        jobs = decode(encode(list(jobs)))
        if self.mode == "local":
            raw = RC.run_replicas(self.registry[worker], [tuple(j) for j in jobs], max_workers=local_workers or len(jobs))
            return [decode(encode(r)) for r in raw]
        bid = batch_id(worker, jobs)
        results, missing = [], []
        for i, job in enumerate(jobs):
            path = self._path(RESULT_FMT, batch=bid, index=i)
            if path.is_file():
                rec = RC.read_json(path)
                if rec.get("job_digest") != job_digest(worker, job):
                    raise RuntimeError(f"{path.name}: the result belongs to a different job")
                results.append(decode(rec["result"]))
            else:
                missing.append(i)
                results.append(None)
        if not missing:
            return results
        request = self._path(REQUEST_FMT, batch=bid)
        if not request.is_file():
            RC.write_json(request, {"schema": f"{SCHEMA}.request/1", "namespace": self.namespace, "batch": bid,
                                    "worker": worker, "jobs": encode(jobs), "n_jobs": len(jobs)})
        raise FanOutPending([(self.namespace, bid, missing)])

    def gather(self, thunks: list[Callable[[], object]]) -> list:
        """Call every thunk; when any raises :class:`FanOutPending`, keep
        going and raise one combined pending record at the end, so
        independent batches are requested together."""
        out, pending = [], []
        for thunk in thunks:
            try:
                out.append(thunk())
            except FanOutPending as exc:
                pending.extend(exc.batches)
                out.append(None)
        if pending:
            raise FanOutPending(pending)
        return out


def run_task(run_dir, registry: Mapping[str, Callable], namespace: str, batch: str, index: int) -> Path:
    """Run one job of a requested batch and write its result atomically
    (a finished result is left as it is)."""
    run_dir = Path(run_dir)
    out = run_dir / RESULT_FMT.format(ns=namespace, batch=batch, index=int(index))
    if out.is_file():
        return out
    req = RC.read_json(run_dir / REQUEST_FMT.format(ns=namespace, batch=batch))
    jobs = decode(req["jobs"])
    if batch_id(req["worker"], jobs) != batch:
        raise RuntimeError(f"the fan-out request {batch} does not reproduce its batch id")
    job = jobs[int(index)]
    result = registry[req["worker"]](*job)
    RC.write_json(out, {"schema": f"{SCHEMA}.result/1", "namespace": namespace, "batch": batch, "index": int(index),
                        "worker": req["worker"], "job_digest": job_digest(req["worker"], job), "result": encode(result)})
    return out


def write_pending(run_dir, key: str, exc: FanOutPending) -> Path:
    """The pending record the orchestrator reads after a step exits."""
    path = Path(run_dir) / PENDING_FMT.format(key=key)
    RC.write_json(path, {"schema": f"{SCHEMA}.pending/1", "key": key,
                         "batches": [{"namespace": ns, "batch": b, "missing": idx} for ns, b, idx in exc.batches]})
    return path


def add_arguments(parser) -> None:
    """The driver CLI of the fan-out: the step mode, the pending key, and
    the task entry."""
    parser.add_argument("--fanout", choices=MODES, default="local",
                        help="run the fit batches in this job (local) or as separate gateway jobs (gateway)")
    parser.add_argument("--fanout-key", default=None, help="the pending-record key of this step round (gateway mode)")
    parser.add_argument("--fanout-task", nargs=3, metavar=("NAMESPACE", "BATCH", "INDEX"), default=None,
                        help="run one job of a requested fan-out batch")
