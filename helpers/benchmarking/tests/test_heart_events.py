"""The HEART event writer (``helpers/benchmarking/heart_events.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import heart_events as H  # noqa: E402

SPEC = ROOT / "dev/mixing2006/config/runs/r141.json"
REPORT = ROOT / "dev/00runs/r141_aligned_lines_2026-11-05/report.json"


def _fitted(**over):
    f = {"epsilon": 0.08, "b_sd": 5.7, "epsilon_nd": 0.2, "x1": 0.3, "a": 1.1, "b": 0.9, "f_isovector": 0.01,
         "class_fractions": {"nd0": 0.6, "sd_projectile": 0.17, "sd_target": 0.17, "dd": 0.06, "nd": 0.0},
         "remnant_weights": {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.25, "high_mass_baryonic_continuum": 0.75}}
    f.update(over)
    return f


def test_the_report_point_is_read_in_the_engine_coordinates():
    engine = SimpleNamespace(EXTRA_SHAPE=("x1",), FLAVOR_ORDER=("a", "b"))
    p = H.fitted_point(engine, _fitted())
    assert p["shape"] == (0.08, 5.7, 0.2, 0.3)
    assert p["flavor"] == (1.1, 0.9)
    assert set(p["fractions"]) == {"nd0", "sd_projectile", "sd_target", "dd"}  # the empty class is dropped
    assert p["remnant_weights"]["high_mass_baryonic_continuum"] == 0.75


def test_class_fractions_that_do_not_sum_to_one_are_refused():
    engine = SimpleNamespace(EXTRA_SHAPE=(), FLAVOR_ORDER=())
    with pytest.raises(ValueError, match="sum to"):
        H.fitted_point(engine, _fitted(class_fractions={"nd0": 0.5, "dd": 0.1}))


def test_the_class_draw_is_fixed_by_seed_and_index_and_follows_the_fractions():
    classes = ["sd_projectile", "sd_target", "dd", "nd0"]
    fr = np.array([0.17, 0.17, 0.06, 0.60])
    cum = np.cumsum(fr)
    draws = [H._class_of(7, i, classes, cum) for i in range(20000)]
    assert draws[:50] == [H._class_of(7, i, classes, cum) for i in range(50)]
    counts = np.array([draws.count(c) for c in classes]) / len(draws)
    assert np.all(np.abs(counts - fr) < 4.0 * np.sqrt(fr * (1.0 - fr) / len(draws)))


@pytest.mark.skipif(not REPORT.is_file(), reason="the R141 fit report is not in this checkout")
def test_two_events_at_the_r141_point_conserve_charge_and_baryon_number(tmp_path):
    out = tmp_path / "ev.jsonl"
    assert H.main(["--spec", str(SPEC), "--report", str(REPORT), "--events", "2", "--seed", "3", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    assert head["format"] == H.FORMAT and head["events"] == 2 and head["particle_columns"] == list(H.PARTICLE_COLUMNS)
    events = [json.loads(line) for line in lines[1:]]
    assert [e["event"] for e in events] == [0, 1]
    col = {name: k for k, name in enumerate(H.PARTICLE_COLUMNS)}
    for e in events:
        rows = e["particles"]
        assert rows and all(len(r) == len(H.PARTICLE_COLUMNS) for r in rows)
        assert sum(r[col["Q"]] for r in rows) == 2 and sum(r[col["B"]] for r in rows) == 2
        m_p = 0.938272
        # lab frame: the beam proton at 158 GeV/c on a proton at rest
        assert sum(r[col["E"]] for r in rows) == pytest.approx(np.hypot(158.0, m_p) + m_p, abs=1e-3)
        assert sum(r[col["pz"]] for r in rows) == pytest.approx(158.0, abs=1e-3)
        assert abs(sum(r[col["px"]] for r in rows)) < 1e-3 and abs(sum(r[col["py"]] for r in rows)) < 1e-3
