"""The fit fan-out (2026-10-02): the JSON codec, the content-addressed
batches, the identity of the local and the gateway mode, the combined
pending record of independent batches, and the task runner."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import fit_fanout as FO  # noqa: E402
import run_contract as RC  # noqa: E402


def toy_fit(x0, scale, label=None):
    x = np.asarray(x0, dtype=float) * float(scale)
    return {"x": x, "fun": float(np.sum(x * x)) / 3.0, "label": label, "n": np.int64(len(x)), "starts": [{"x0": list(x0)}]}


REGISTRY = {"toy": toy_fit}


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    # the toy worker lives in a test module; run the local pool serially
    monkeypatch.setattr(RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])


def test_codec_round_trips_exactly():
    value = {"a": np.array([[0.1, 1.0 / 3.0], [2.0 ** -40, -7.25]]), "b": (1, 2.5, None, "s", True), "c": np.float64(math.pi),
             "d": np.arange(4, dtype=np.int16), "e": [np.array([], dtype=float)]}
    back = FO.decode(json.loads(json.dumps(FO.encode(value))))
    assert back["a"].dtype == np.float64 and np.array_equal(back["a"], value["a"])
    assert back["b"] == [1, 2.5, None, "s", True] and back["c"] == math.pi
    assert back["d"].dtype == np.int16 and np.array_equal(back["d"], value["d"]) and back["e"][0].shape == (0,)
    with pytest.raises(TypeError):
        FO.encode({"bad": object()})


def test_batch_ids_are_content_addressed():
    jobs = [[[0.1, 0.2], 2.0], [[0.3, 0.4], 2.0]]
    assert FO.batch_id("toy", jobs) == FO.batch_id("toy", [list(j) for j in jobs])
    assert FO.batch_id("toy", jobs) != FO.batch_id("toy", jobs[::-1]) != FO.batch_id("other", jobs)


def _drain(run_dir, exc):
    for ns, batch, missing in exc.batches:
        for i in missing:
            FO.run_task(run_dir, REGISTRY, ns, batch, i)


def test_gateway_mode_reproduces_local_mode(tmp_path):
    jobs = [([0.1, 1.0 / 7.0], 3.0, "a"), ([2.0, -0.5], 0.1, "b")]
    local = FO.FanOut(tmp_path, REGISTRY, mode="local").map("toy", jobs)
    gw = FO.FanOut(tmp_path, REGISTRY, mode="gateway")
    with pytest.raises(FO.FanOutPending) as exc:
        gw.map("toy", jobs)
    (ns, batch, missing), = exc.value.batches
    assert ns == "main" and missing == [0, 1]
    assert (tmp_path / FO.REQUEST_FMT.format(ns=ns, batch=batch)).is_file()
    FO.run_task(tmp_path, REGISTRY, ns, batch, 1)
    with pytest.raises(FO.FanOutPending) as exc:
        gw.map("toy", jobs)
    assert exc.value.batches == [(ns, batch, [0])]
    _drain(tmp_path, exc.value)
    remote = gw.map("toy", jobs)
    for a, b in zip(local, remote):
        assert np.array_equal(a["x"], b["x"]) and a["fun"] == b["fun"] and a["label"] == b["label"] and a["n"] == b["n"]
        assert type(a["x"]) is type(b["x"]) and a["starts"] == b["starts"]
    # a namespace recomputes: the clean re-evaluation reads none of the originals
    with pytest.raises(FO.FanOutPending):
        FO.FanOut(tmp_path, REGISTRY, mode="gateway", namespace="clean").map("toy", jobs)


def test_gather_requests_independent_batches_together(tmp_path):
    gw = FO.FanOut(tmp_path, REGISTRY, mode="gateway")
    thunks = [lambda: gw.map("toy", [([1.0], 2.0)]), lambda: 7, lambda: gw.map("toy", [([3.0], 2.0), ([4.0], 1.0)])]
    with pytest.raises(FO.FanOutPending) as exc:
        gw.gather(thunks)
    assert [len(m) for _, _, m in exc.value.batches] == [1, 2]
    path = FO.write_pending(tmp_path, "profile_r0", exc.value)
    record = json.loads(path.read_text())
    assert record["key"] == "profile_r0" and [b["missing"] for b in record["batches"]] == [[0], [0, 1]]
    _drain(tmp_path, exc.value)
    first, seven, third = gw.gather(thunks)
    assert seven == 7 and first[0]["fun"] == 4.0 / 3.0 and third[1]["fun"] == 16.0 / 3.0


def test_run_task_is_idempotent_and_checks_the_request(tmp_path):
    gw = FO.FanOut(tmp_path, REGISTRY, mode="gateway")
    with pytest.raises(FO.FanOutPending) as exc:
        gw.map("toy", [([1.0, 2.0], 1.0)])
    (ns, batch, _), = exc.value.batches
    out = FO.run_task(tmp_path, REGISTRY, ns, batch, 0)
    stamp = out.read_bytes()
    assert FO.run_task(tmp_path, REGISTRY, ns, batch, 0) == out and out.read_bytes() == stamp
    # a result that belongs to another job is refused
    rec = json.loads(out.read_text())
    rec["job_digest"] = "0" * 64
    out.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="different job"):
        gw.map("toy", [([1.0, 2.0], 1.0)])
    # a tampered request does not reproduce its batch id
    req = tmp_path / FO.REQUEST_FMT.format(ns=ns, batch=batch)
    data = json.loads(req.read_text())
    data["jobs"][0][1] = 9.0
    req.write_text(json.dumps(data))
    out.unlink()
    with pytest.raises(RuntimeError, match="batch id"):
        FO.run_task(tmp_path, REGISTRY, ns, batch, 0)


def test_modes_and_namespaces_are_checked(tmp_path):
    with pytest.raises(ValueError):
        FO.FanOut(tmp_path, REGISTRY, mode="cluster")
    with pytest.raises(ValueError):
        FO.FanOut(tmp_path, REGISTRY, namespace="a-b")
    with pytest.raises(KeyError):
        FO.FanOut(tmp_path, REGISTRY).map("missing", [()])
