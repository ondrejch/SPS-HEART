"""The paper record's two checks (helpers/publishing/make_paper_record.py; the review of 2026-11-01, item R1): the
objective by segment plus the scale penalty must add to the reported total, and the launch commits are read from
the orchestrator's verified sync line."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("make_paper_record", ROOT / "helpers" / "publishing" / "make_paper_record.py")
MPR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MPR)


def _report(total: float) -> dict:
    return {"objective_at_optimum": {"chi2_by_segment": {"pi_minus": 800.0, "proton": 1000.0},
                                     "nuisances": {"penalty": 15.0}, "value": total},
            "isovector_test": {"ndf": 1253, "scale_factor_squared": 1.44}, "errors_scaled_by_s": {"s": 1.2}}


def test_the_segments_and_the_penalty_add_to_the_total():
    obj = MPR.objective_check(_report(1815.0))
    assert obj["sum_check"] == pytest.approx(1815.0) and obj["ndf"] == 1253 and obj["s"] == 1.2


def test_a_total_the_segments_do_not_carry_is_refused():
    with pytest.raises(SystemExit, match="differ from the reported total"):
        MPR.objective_check(_report(1816.0))


def test_the_launch_commits_come_from_the_verified_sync(tmp_path):
    log = tmp_path / "orch.log"
    log.write_text("[10:08:20] sync: dev/mixing2006/\n"
                   "[10:08:22] sync verified: 43 digests match (outer 2054c97, nested 91a496b)\n")
    launch = MPR.launch_commits(log)
    assert launch["outer"] == "2054c97" and launch["nested"] == "91a496b" and launch["digests_verified"] == 43


def test_a_log_without_a_verified_sync_is_refused(tmp_path):
    log = tmp_path / "orch.log"
    log.write_text("[10:08:20] sync: dev/mixing2006/\n")
    with pytest.raises(SystemExit, match="no 'sync verified' line"):
        MPR.launch_commits(log)
