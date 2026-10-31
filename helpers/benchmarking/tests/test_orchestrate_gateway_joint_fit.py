"""The gateway orchestrator's local machinery (2026-10-02, external reviews
g1, m1 and q1): the FROZEN_RELS literal read by ``ast``, the persistent
ledger path with its one-time migration from the legacy session path, the
atomic ledger write and the refusal of a corrupt ledger, and the retried
remote JSON read.  No gateway is contacted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
MONO_ROOT = HERE.parents[3]
if not (MONO_ROOT / "helpers" / "na49_gateway").is_dir():
    pytest.skip("the monorepo gateway package is not present", allow_module_level=True)
sys.path.insert(0, str(MONO_ROOT))
import helpers.benchmarking.orchestrate_gateway_joint_fit as O  # noqa: E402


def test_frozen_rels_read_by_ast(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text('import os\nX = 1\nFROZEN_RELS = (  # the frozen substrate (see notes)\n'
                      '    "a/b.py",\n    "c/d.json",  # a comment with ) parens (\n)  # trailing\n'
                      'def f():\n    FROZEN_RELS = ("not", "this")\n')
    assert O.frozen_rels_of(driver) == ("a/b.py", "c/d.json")
    (tmp_path / "none.py").write_text("Y = 2\n")
    with pytest.raises(RuntimeError):
        O.frozen_rels_of(tmp_path / "none.py")


def test_every_registered_driver_has_a_readable_literal():
    for spec in O.RUNS.values():
        rels = O.frozen_rels_of(MONO_ROOT / spec["driver"], spec.get("spec", ""))
        assert rels and all(isinstance(r, str) for r in rels)


def test_ledger_path_migrates_the_legacy_ledger_once(tmp_path, monkeypatch):
    monkeypatch.setenv("NA49_GW_LEDGER_DIR", str(tmp_path / "state"))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setattr(O, "LEGACY_LEDGER_DIR", legacy)
    (legacy / "rX_gateway_jobs.json").write_text(json.dumps({"k": {"job_id": "J1"}}))
    path = O.ledger_path("rX")
    assert path == tmp_path / "state" / "rX_gateway_jobs.json"
    assert O.read_ledger(path) == {"k": {"job_id": "J1"}}
    # a later legacy change is not copied again: the persistent ledger rules
    (legacy / "rX_gateway_jobs.json").write_text(json.dumps({"k": {"job_id": "OLD"}}))
    O.write_ledger(path, {"k": {"job_id": "J2"}})
    assert O.read_ledger(O.ledger_path("rX")) == {"k": {"job_id": "J2"}}
    assert O.ledger_path("rY").parent == tmp_path / "state" and O.read_ledger(O.ledger_path("rY")) == {}


def test_ledger_write_is_atomic_and_a_corrupt_ledger_is_refused(tmp_path):
    path = tmp_path / "led" / "r_gateway_jobs.json"
    O.write_ledger(path, {"a": 1})
    assert sorted(p.name for p in path.parent.iterdir()) == ["r_gateway_jobs.json"]
    path.write_text('{"a": 1, "b":')
    with pytest.raises(RuntimeError, match="does not parse"):
        O.read_ledger(path)


def test_remote_json_retries_a_partial_read(monkeypatch):
    replies = iter(['{"points": [', '{"points": [{"generation": 0}]}'])
    gw = O.Gateway.__new__(O.Gateway)
    gw.run_dir = "/remote/run"
    gw.ssh = lambda command, **kw: next(replies)
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    assert gw.remote_json("generation_points_f0.json") == {"points": [{"generation": 0}]}
    gw.ssh = lambda command, **kw: ""
    assert gw.remote_json("absent.json") is None
    gw.ssh = lambda command, **kw: "{broken"
    with pytest.raises(json.JSONDecodeError):
        gw.remote_json("broken.json")


class _FakeGateway:
    """A remote run dir in memory: the steps and fan-out tasks run at
    submission (the step needs two fan-out rounds before its output)."""

    def __init__(self):
        self.files = {}
        self.submitted = []

    def remote_files(self):
        return set(self.files)

    def remote_json(self, name):
        return json.loads(self.files[name]) if name in self.files else None

    def execute(self, key, args):
        self.submitted.append((key, args))
        if args.startswith("--fanout-task"):
            _, ns, bid, i = args.split()
            self.files[f"fanout_{ns}_{bid}_{i}.json"] = "{}"
            return
        rkey = args.split("--fanout-key ")[1] if "--fanout-key" in args else None
        need = [("B1", 2), ("B2", 3)]
        for bid, n in need:
            missing = [i for i in range(n) if f"fanout_main_{bid}_{i}.json" not in self.files]
            if missing:
                assert rkey is not None, "a step without fan-out cannot pend"
                self.files[f"fanout_pending_{rkey}.json"] = json.dumps(
                    {"batches": [{"namespace": "main", "batch": bid, "missing": missing}]})
                return
        self.files["fit_report_f0_g0.json"] = "{}"


class _FakeJobs:
    def __init__(self, gw):
        self.gw = gw
        self.state = {}

    def ensure(self, key, args, tasks, name):
        if key in self.state:
            return
        self.state[key] = {"args": args, "tasks": tasks}
        self.gw.execute(key, args)

    def wait_for(self, expected, label):
        missing = {k: f for k, f in expected.items() if not O.present(f, self.gw.remote_files())}
        assert not missing, (label, missing)


def test_run_steps_fans_out_in_restart_safe_rounds(monkeypatch):
    monkeypatch.setattr(O, "FANOUT", True)
    monkeypatch.setattr(O, "FANOUT_TASKS", 32)
    gw = _FakeGateway()
    jobs = _FakeJobs(gw)
    step = [("eval_f0_g0", "--evaluate --floor-index 0", 32, "fit_report_f0_g0.json")]
    O.run_steps(gw, jobs, step, "evaluation")
    assert "fit_report_f0_g0.json" in gw.files
    keys = [k for k, _ in gw.submitted]
    assert keys == ["eval_f0_g0_r0", "fan_main_B1_0", "fan_main_B1_1", "eval_f0_g0_r1",
                    "fan_main_B2_0", "fan_main_B2_1", "fan_main_B2_2", "eval_f0_g0_r2"]
    assert gw.submitted[0][1] == "--evaluate --floor-index 0 --fanout gateway --fanout-key eval_f0_g0_r0"
    assert all(jobs.state[k]["tasks"] == 32 for k in keys)
    # a restarted orchestrator (empty ledger) replays the finished rounds and submits nothing
    gw.submitted.clear()
    O.run_steps(gw, _FakeJobs(gw), step, "evaluation")
    assert gw.submitted == []
    # a restart in the middle replays round 0 from its record and resumes at round 1
    del gw.files["fit_report_f0_g0.json"]
    for i in range(3):
        del gw.files[f"fanout_main_B2_{i}.json"]
    del gw.files["fanout_pending_eval_f0_g0_r1.json"]
    gw.submitted.clear()
    O.run_steps(gw, _FakeJobs(gw), step, "evaluation")
    assert [k for k, _ in gw.submitted] == ["eval_f0_g0_r1", "fan_main_B2_0", "fan_main_B2_1", "fan_main_B2_2", "eval_f0_g0_r2"]


def test_run_steps_without_fanout_is_one_job_per_step(monkeypatch):
    monkeypatch.setattr(O, "FANOUT", False)
    gw = _FakeGateway()
    for bid, n in (("B1", 2), ("B2", 3)):
        for i in range(n):
            gw.files[f"fanout_main_{bid}_{i}.json"] = "{}"
    jobs = _FakeJobs(gw)
    O.run_steps(gw, jobs, [("eval_f0_g0", "--evaluate --floor-index 0", 32, "fit_report_f0_g0.json")], "evaluation")
    assert gw.submitted == [("eval_f0_g0", "--evaluate --floor-index 0")]
