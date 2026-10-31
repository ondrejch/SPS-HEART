"""The run spec's split of the single-diffraction non-excited line
(2026-10-09, from R87): ``diffractive_split`` sets the remnant weights of
the two single-diffraction classes (an intact proton or a low-mass baryonic
resonance, summing to one); a spec without it keeps the nominal R50 split,
and the non-diffractive remnant weights stay the fitted ones."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
R86 = NESTED_ROOT / "config" / "runs" / "r86.json"
R87 = NESTED_ROOT / "config" / "runs" / "r87.json"
INTACT = {"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_spec_without_the_split_keeps_the_nominal_split():
    e = _load("run_engine_split_nominal")
    e.configure(R86)
    assert e.FROZEN_REMNANT_WEIGHTS == e.NOMINAL_DIFFRACTIVE_SPLIT == {
        "intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}
    for cls in ("sd_projectile", "sd_target"):
        assert e._params_at(cls, e.GEN0_SHAPE).remnant_weights == e.NOMINAL_DIFFRACTIVE_SPLIT


def test_the_r87_spec_takes_the_intact_line():
    e = _load("run_engine_split_r87")
    e.configure(R87)
    assert e.FROZEN_REMNANT_WEIGHTS == INTACT
    for cls in ("sd_projectile", "sd_target"):
        assert e._params_at(cls, e.GEN0_SHAPE).remnant_weights == INTACT
    for cls in e.ND_CLASSES:
        assert e._params_at(cls, e.GEN0_SHAPE).remnant_weights == e.ND_GENERATION_REMNANT_WEIGHTS
    # the nominal constant stays the R50 split
    assert e.NOMINAL_DIFFRACTIVE_SPLIT["intact_proton"] == 0.5
    # R87 is R86 with the intact line, the seeds and the texts
    r86, r87 = json.loads(R86.read_text()), json.loads(R87.read_text())
    assert r87["kernel"] == r86["kernel"] and r87["topology"] == r86["topology"] and r87["fit"] == r86["fit"]
    assert r87["diffractive_split"] == INTACT and "diffractive_split" not in r86


@pytest.mark.parametrize("split, message", [
    ({"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0}, "takes the weights"),
    ({"intact_proton": True, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0}, "weight in"),
    ({"intact_proton": 1.5, "low_mass_baryonic_resonance": -0.5, "high_mass_baryonic_continuum": 0.0}, "weight in"),
    ({"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.5}, "low-mass resonance"),
    ({"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.4, "high_mass_baryonic_continuum": 0.0}, "sum to 1"),
    ([1.0, 0.0, 0.0], "takes the weights"),
])
def test_the_split_validates(tmp_path, split, message):
    bad = copy.deepcopy(json.loads(R86.read_text()))
    bad["diffractive_split"] = split
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match=message):
        _load("run_engine_split_bad").configure(path)
