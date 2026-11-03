"""The stale-number scan (helpers/publishing/stale_prose_numbers.py): a number of the previous fit that the new fit does
not reproduce is listed, a number both fits share is not, and comments are skipped."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("stale_prose_numbers", ROOT / "helpers" / "publishing" / "stale_prose_numbers.py")
SP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SP)


def _report(eps_nd, s_pi, chi2_km):
    return {"fitted": {"epsilon_nd": eps_nd, "sd_ope_share": s_pi, "sd_tail_share": 0.19, "f_sd_total": 0.39},
            "errors_scaled_by_s": {"errors": {"epsilon_nd": 0.0075}, "s": 1.58},
            "objective_at_optimum": {"value": 3150.0, "chi2_by_segment": {"k_minus": chi2_km},
                                     "nuisances": {"penalty": 15.0, "scale_neutron": {"pull": -1.3, "value": -0.13}}},
            "isovector_test": {"scale_factor_squared": 2.5}}


def test_an_old_number_is_listed_and_a_shared_one_is_not(tmp_path, monkeypatch, capsys):
    for run, rep in (("old", _report(0.253, 0.341, 346.5)), ("new", _report(0.2608, 0.316, 371.7))):
        (tmp_path / run).mkdir()
        (tmp_path / run / "report.json").write_text(json.dumps(rep))
    tex = tmp_path / "main.tex"
    tex.write_text("eps = 0.253 and s = 0.316\n% a comment with 0.341\nK- chi2 per point 2.25, S = 1.58\n")
    monkeypatch.setattr(SP, "RUNS", tmp_path)
    monkeypatch.setattr("sys.argv", ["x", "old", "new", str(tex)])
    SP.main()
    import re
    matched = set(re.findall(r"main\.tex:\d+: ([\d.]+) \(", capsys.readouterr().out))
    assert {"0.253", "2.25"} <= matched                   # the old epsilon_nd and the old chi2 per point
    assert not matched & {"0.316", "0.341", "1.58"}       # new, commented, shared
