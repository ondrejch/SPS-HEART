"""What the single-diffraction anchor counts (2026-11-01; the review of 2026-11-01, item R6).

The measured sigma_SD(M^2/s < 0.05) is the inclusive p + p -> p + X cross section below the edge, every intact proton.
The committed reading ("coherence") anchors the coherence component alone; "intact_protons" anchors the coherence
component together with the neutral pion exchange (p + p -> p + X+), which leaves the proton intact as well."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"


def _engine(spec: dict, tmp_path: Path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    s = importlib.util.spec_from_file_location("run_engine_window_counts", ENGINE)
    e = importlib.util.module_from_spec(s)
    s.loader.exec_module(e)
    e.configure(path)
    return e


def _x_at(e, values: dict):
    """The free shape coordinates at *values*, on each parameter's transform."""
    import math
    out = []
    for n in e.free_shape_names():
        v = float(values[n]) if n in values else float(e.PARAM_INDEX[n].gen0)
        t = e.PARAM_INDEX[n].transform
        out.append(math.log(v) if t == "log" else (math.log(v / (1.0 - v)) if t == "logit" else v))
    return out


def test_the_default_is_the_coherence_reading(tmp_path):
    e = _engine(json.loads((RUNS / "r133.json").read_text()), tmp_path)
    assert e.SD_WINDOW_INTACT is False and e.NEUTRAL_OPE_WINDOW == 0.0


def test_intact_protons_counts_the_neutral_exchange(tmp_path):
    spec = json.loads((RUNS / "r133.json").read_text())
    committed = _engine(spec, tmp_path)
    spec["single_diffraction_window_counts"] = "intact_protons"
    intact = _engine(spec, tmp_path)
    n0 = intact.NEUTRAL_OPE_WINDOW
    assert n0 == pytest.approx(0.0137, abs=0.0005)
    values = {"b_sd": 5.54, "intact_vertex_factor": 1.31, "sd_tail_share": 0.19, "sd_ope_share": 0.32}
    a0 = committed.physical_from_x(_x_at(committed, values), 0.0, committed.anchors())["f_sd_total"]
    a1 = intact.physical_from_x(_x_at(intact, values), 0.0, intact.anchors())["f_sd_total"]
    w_share = float(intact.anchors()[intact.WINDOW_NAME]["value"])
    frac = intact.sd_window_fraction(0.0808, intact.SD_WINDOW, 5.54, 1.31)
    s_rest = 1.0 - 0.19 - 0.32
    # the committed reading: the coherence part alone carries the window share
    assert a0 * s_rest * frac == pytest.approx(w_share, rel=1e-10)
    # the intact reading: the coherence part and the neutral exchange's in-window part together carry it
    assert a1 * (s_rest * frac + 0.32 * n0) == pytest.approx(w_share, rel=1e-10)
    assert 0.98 < a1 / a0 < 0.995


def test_the_refusals(tmp_path):
    spec = json.loads((RUNS / "r133.json").read_text())
    spec["single_diffraction_window_counts"] = "every_proton"
    with pytest.raises(ValueError, match="single_diffraction_window_counts"):
        _engine(spec, tmp_path)
    spec["single_diffraction_window_counts"] = "intact_protons"
    spec.pop("single_diffraction_window")
    with pytest.raises(ValueError, match="intact_protons"):
        _engine(spec, tmp_path)
